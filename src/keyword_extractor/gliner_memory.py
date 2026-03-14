"""
GLiNER + 双重记忆机制实体提取器 (V2.0 架构)

核心升级：
1. GLiNER Zero-shot 实体识别 - 替代传统分词+NER
2. 双流跨度匹配 - 无需训练即可识别新实体
3. 保留双重记忆机制 (ESM + GTM) - 跨段落全局融合
"""
import re
import math
import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from loguru import logger

try:
    from gliner import GLiNER
    GLINER_AVAILABLE = True
except ImportError:
    GLINER_AVAILABLE = False
    logger.warning("GLiNER 未安装，请运行: pip install gliner")

from .models import KeywordItem, ExtractionResult


@dataclass
class GLiNEREntityState:
    """GLiNER 实体状态记忆 (ESM)"""
    text: str                    # 实体文本
    label: str                   # 实体类型标签
    embedding: torch.Tensor      # 实体特征向量（GLiNER 输出）
    confidence: float            # GLiNER 置信度
    freq: int = 0                # 出现频次
    first_pos: float = -1.0      # 首次出现相对位置
    last_pos: float = -1.0       # 最后出现相对位置
    chunks: Set[int] = field(default_factory=set)  # 出现的 chunk 索引
    
    def update(self, confidence: float, embedding: torch.Tensor, relative_pos: float, chunk_idx: int):
        """更新实体状态（跨窗口追踪）"""
        # 置信度加权平均
        total_weight = self.freq
        self.confidence = (self.confidence * total_weight + confidence) / (total_weight + 1)
        # 向量平均池化
        self.embedding = (self.embedding * total_weight + embedding) / (total_weight + 1)
        
        self.freq += 1
        self.last_pos = relative_pos
        self.chunks.add(chunk_idx)


class GLiNERMemoryTracker:
    """GLiNER + 双重记忆追踪器"""
    
    def __init__(self, alpha: float = 0.2, dynamic_alpha: bool = True):
        """
        Args:
            alpha: GTM 移动平均衰减率
            dynamic_alpha: 是否使用动态 alpha
        """
        self.alpha_base = alpha
        self.dynamic_alpha = dynamic_alpha
        
        # 全局主题记忆 (GTM) - 使用 [CLS] 向量
        self.global_theme_vector: Optional[torch.Tensor] = None
        
        # 实体状态记忆 (ESM) - 按实体文本聚合
        self.entity_memory: Dict[str, GLiNEREntityState] = {}
        
        self.total_chunks = 0
        self.current_chunk_idx = 0
    
    def get_alpha(self, relative_pos: float) -> float:
        """动态 alpha - 首尾段落权重更高"""
        if not self.dynamic_alpha:
            return self.alpha_base
        
        # 首尾 10% 使用更大 alpha
        if relative_pos < 0.1 or relative_pos > 0.9:
            return min(0.4, self.alpha_base * 2)
        return self.alpha_base * 0.75
    
    def update_global_theme(self, cls_vector: torch.Tensor, relative_pos: float):
        """更新 GTM - 使用 Sentence Transformer 或 GLiNER 的 [CLS]"""
        alpha = self.get_alpha(relative_pos)
        
        if self.global_theme_vector is None:
            self.global_theme_vector = cls_vector.clone()
        else:
            self.global_theme_vector = (
                (1 - alpha) * self.global_theme_vector + alpha * cls_vector
            )
    
    def update_entity(self, entity_text: str, label: str, embedding: torch.Tensor,
                     confidence: float, relative_pos: float, chunk_idx: int):
        """更新 ESM - 接收 GLiNER 输出的实体"""
        key = f"{entity_text}_{label}"  # 区分不同标签的同名实体
        
        if key not in self.entity_memory:
            self.entity_memory[key] = GLiNEREntityState(
                text=entity_text,
                label=label,
                embedding=embedding.clone(),
                confidence=confidence,
                freq=1,
                first_pos=relative_pos,
                last_pos=relative_pos,
                chunks={chunk_idx}
            )
        else:
            self.entity_memory[key].update(confidence, embedding, relative_pos, chunk_idx)
    
    def compute_final_scores(
        self,
        lambda_conf: float = 0.4,       # GLiNER 置信度权重
        lambda_sim: float = 0.35,       # 全局主题相似度权重
        lambda_freq: float = 0.15,      # 频次权重
        lambda_idf: float = 0.1         # IDF 惩罚权重
    ) -> List[Tuple[str, str, float, Dict]]:
        """
        全局融合打分
        
        S_final = λ1*C_gliner + λ2*sim(E, V_global) + λ3*f(Freq, IDF)
        """
        if self.global_theme_vector is None or not self.entity_memory:
            return []
        
        results = []
        
        # 计算 IDF 相关统计
        total_entities = len(self.entity_memory)
        label_doc_freq = defaultdict(int)  # 标签文档频率
        for state in self.entity_memory.values():
            label_doc_freq[state.label] += 1
        
        for key, state in self.entity_memory.items():
            # 1. GLiNER 置信度 (归一化到 0-1)
            conf_score = min(state.confidence, 1.0)
            
            # 2. 与全局主题的相似度
            sim_score = F.cosine_similarity(
                state.embedding.unsqueeze(0),
                self.global_theme_vector.unsqueeze(0)
            ).item()
            
            # 3. 频次得分 (对数平滑)
            freq_score = math.log(1 + state.freq)
            
            # 4. IDF 惩罚 - 过滤通用高频词
            # 计算标签的 IDF: log(N / df)
            label_df = label_doc_freq.get(state.label, 1)
            idf_score = math.log(total_entities / label_df)
            
            # 综合打分
            final_score = (
                lambda_conf * conf_score +
                lambda_sim * sim_score +
                lambda_freq * freq_score +
                lambda_idf * idf_score
            )
            
            metadata = {
                "label": state.label,
                "freq": state.freq,
                "confidence": round(state.confidence, 4),
                "first_pos": round(state.first_pos, 3),
                "span_chunks": len(state.chunks),
                "component_scores": {
                    "confidence": round(conf_score, 4),
                    "similarity": round(sim_score, 4),
                    "frequency": round(freq_score, 4),
                    "idf": round(idf_score, 4)
                }
            }
            
            results.append((state.text, state.label, final_score, metadata))
        
        # 按得分降序
        results.sort(key=lambda x: x[2], reverse=True)
        return results


class GLiNEREntityExtractor:
    """GLiNER + 双重记忆实体提取器 (V2.0)"""
    
    # 预设模型
    PRESET_MODELS = {
        "gliner-chinese": "xianyun/gliner_chinese_large",
        "gliner-multi": "urchade/gliner_multi-v2.1",
    }
    
    # 默认标签定义
    DEFAULT_LABELS = [
        "科技公司",      # 英伟达、OpenAI、字节跳动
        "软件产品",      # ChatGPT、Claude、豆包
        "人工智能模型",  # GPT-4、Llama 3、DeepSeek-V2
        "核心技术",      # Transformer、MoE、Diffusion
        "硬件设备",      # H100、A100、iPhone
        "学术会议",      # NeurIPS、ICML、CVPR
        "人名",         # 马斯克、卡帕西、胡渊鸣
    ]
    
    def __init__(
        self,
        model_name: str = "xianyun/gliner_chinese_large",
        labels: Optional[List[str]] = None,
        device: Optional[str] = None,
        chunk_size: int = 800,          # GLiNER 通常可以处理更长文本
        chunk_overlap: int = 100,
        alpha: float = 0.2,
        threshold: float = 0.3          # GLiNER 置信度阈值
    ):
        """
        Args:
            model_name: GLiNER 模型名称
            labels: 实体类型标签列表
            device: 运行设备
            chunk_size: 滑动窗口大小
            chunk_overlap: 窗口重叠
            alpha: GTM 更新系数
            threshold: GLiNER 置信度阈值
        """
        if not GLINER_AVAILABLE:
            raise RuntimeError("GLiNER 未安装，请运行: pip install gliner")
        
        if model_name in self.PRESET_MODELS:
            model_name = self.PRESET_MODELS[model_name]
        
        self.model_name = model_name
        self.labels = labels or self.DEFAULT_LABELS
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.alpha = alpha
        self.threshold = threshold
        
        # 加载 GLiNER 模型
        logger.info(f"加载 GLiNER 模型: {model_name}")
        self.model = GLiNER.from_pretrained(model_name)
        self.model.eval()
        self.model.to(self.device)
        logger.info(f"✅ GLiNER 模型加载完成")
    
    def _split_into_chunks(self, text: str) -> List[Tuple[int, str]]:
        """滑动窗口切分，返回 (chunk_idx, chunk_text)"""
        # 按句子切分
        sentences = re.split(r'([。！？；\n]+)', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sent in sentences:
            sent_len = len(sent)
            
            if current_length + sent_len > self.chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                # 保留重叠
                overlap_text = ''.join(current_chunk)[-self.chunk_overlap:]
                current_chunk = [overlap_text, sent]
                current_length = len(overlap_text) + sent_len
            else:
                current_chunk.append(sent)
                current_length += sent_len
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return list(enumerate(chunks))
    
    def _get_chunk_embedding(self, text: str) -> torch.Tensor:
        """获取 chunk 的向量表示（用于 GTM）"""
        # 使用 GLiNER 的文本编码器获取 [CLS] 向量
        with torch.no_grad():
            # GLiNER 内部编码
            inputs = self.model.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(self.device)
            
            outputs = self.model.model.bert(**inputs)
            cls_vector = outputs.last_hidden_state[:, 0, :]  # [CLS]
            
        return cls_vector.squeeze(0).cpu()
    
    def extract(
        self,
        text: str,
        top_k: int = 10,
        return_metadata: bool = False
    ) -> ExtractionResult:
        """
        提取实体
        
        Args:
            text: 输入文本
            top_k: 返回实体数量
            return_metadata: 是否返回详细元数据
        
        Returns:
            提取结果
        """
        import time
        start_time = time.time()
        
        if not text or not text.strip():
            return ExtractionResult(
                text="",
                keywords=[],
                method="gliner-memory",
                elapsed_time=0.0,
                model=self.model_name
            )
        
        # 1. 滑动窗口切分
        chunks = self._split_into_chunks(text)
        self.total_chunks = len(chunks)
        
        logger.debug(f"文本分块: {self.total_chunks} 个 chunks")
        
        # 2. 初始化记忆追踪器
        tracker = GLiNERMemoryTracker(alpha=self.alpha)
        tracker.total_chunks = self.total_chunks
        
        # 3. 逐块处理
        for chunk_idx, chunk_text in chunks:
            relative_pos = chunk_idx / max(self.total_chunks - 1, 1)
            
            # 3.1 GLiNER 实体识别
            entities = self.model.predict_entities(
                chunk_text,
                self.labels,
                threshold=self.threshold
            )
            
            # 3.2 获取 chunk 向量（用于 GTM）
            chunk_embedding = self._get_chunk_embedding(chunk_text)
            tracker.update_global_theme(chunk_embedding, relative_pos)
            
            # 3.3 更新实体记忆
            for entity in entities:
                entity_text = entity["text"]
                label = entity["label"]
                confidence = entity["score"]
                
                # 获取实体向量（简化：使用文本编码）
                entity_emb = self._get_chunk_embedding(entity_text)
                
                tracker.update_entity(
                    entity_text, label, entity_emb,
                    confidence, relative_pos, chunk_idx
                )
        
        # 4. 全局融合打分
        scored_entities = tracker.compute_final_scores()
        
        # 5. 组装结果
        keywords = []
        for entity_text, label, score, metadata in scored_entities[:top_k]:
            kw_item = KeywordItem(
                keyword=f"{entity_text} ({label})",
                score=score,
                method="gliner-memory",
                metadata=metadata if return_metadata else None
            )
            keywords.append(kw_item)
        
        elapsed = time.time() - start_time
        
        return ExtractionResult(
            text=text[:200] + "..." if len(text) > 200 else text,
            keywords=keywords,
            method="gliner-memory",
            elapsed_time=elapsed,
            model=self.model_name
        )
