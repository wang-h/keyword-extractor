"""
BERT + 双重记忆机制关键词提取器

核心设计：
1. 全局主题记忆 (GTM) - 维护文档整体语义中心
2. 实体状态记忆 (ESM) - 跟踪每个实体的多出现位置语义
3. 滑动窗口编码 - 处理长文本
4. 全局特征聚合 - 综合打分输出
"""
import re
import math
import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict
from dataclasses import dataclass, field
from transformers import AutoModel, AutoTokenizer
import jieba
import jieba.posseg as pseg
from loguru import logger

from .models import KeywordItem, ExtractionResult, ExtractorConfig


@dataclass
class EntityState:
    """实体状态记忆 (ESM)"""
    embedding: torch.Tensor  # 聚合后的实体向量
    freq: int = 0            # 出现频次
    first_pos: float = -1.0  # 首次出现相对位置 (0-1)
    last_pos: float = -1.0   # 最后出现相对位置
    positions: List[int] = field(default_factory=list)  # 所有出现位置
    chunks: Set[int] = field(default_factory=set)       # 出现的 chunk 索引


class DocumentMemoryTracker:
    """文档记忆追踪器 - 管理 GTM 和 ESM"""
    
    def __init__(self, alpha: float = 0.2, dynamic_alpha: bool = True):
        """
        Args:
            alpha: GTM 移动平均衰减率
            dynamic_alpha: 是否使用动态 alpha（首尾段落权重更高）
        """
        self.alpha_base = alpha
        self.dynamic_alpha = dynamic_alpha
        
        # 全局主题记忆 (GTM)
        self.global_theme_vector: Optional[torch.Tensor] = None
        
        # 实体状态记忆 (ESM)
        self.entity_memory: Dict[str, EntityState] = {}
        
        self.total_chunks = 0
        self.current_chunk_idx = 0
    
    def get_alpha(self, relative_pos: float) -> float:
        """获取动态 alpha - 首尾段落权重更高"""
        if not self.dynamic_alpha:
            return self.alpha_base
        
        # 文章前 10% 和后 10% 使用更大的 alpha
        if relative_pos < 0.1 or relative_pos > 0.9:
            return min(0.4, self.alpha_base * 2)
        # 中间段落使用较小的 alpha
        return self.alpha_base * 0.75
    
    def update_global_theme(self, cls_vector: torch.Tensor, relative_pos: float):
        """更新全局主题向量"""
        alpha = self.get_alpha(relative_pos)
        
        if self.global_theme_vector is None:
            self.global_theme_vector = cls_vector.clone()
        else:
            # 移动平均融合
            self.global_theme_vector = (
                (1 - alpha) * self.global_theme_vector + alpha * cls_vector
            )
    
    def update_entity_state(
        self, 
        entity_text: str, 
        entity_vector: torch.Tensor, 
        relative_pos: float,
        chunk_idx: int
    ):
        """更新实体状态记忆"""
        if entity_text not in self.entity_memory:
            # 首次出现
            self.entity_memory[entity_text] = EntityState(
                embedding=entity_vector.clone(),
                freq=1,
                first_pos=relative_pos,
                last_pos=relative_pos,
                positions=[chunk_idx],
                chunks={chunk_idx}
            )
        else:
            state = self.entity_memory[entity_text]
            # 平均池化更新向量
            total_weight = state.freq
            state.embedding = (
                state.embedding * total_weight + entity_vector
            ) / (total_weight + 1)
            
            state.freq += 1
            state.last_pos = relative_pos
            state.positions.append(chunk_idx)
            state.chunks.add(chunk_idx)
    
    def compute_final_scores(
        self,
        lambda_sim: float = 0.6,      # 语义相似度权重
        lambda_freq: float = 0.25,    # 频次权重
        lambda_pos: float = 0.1,      # 位置权重
        lambda_span: float = 0.05     # 跨段落权重
    ) -> List[Tuple[str, float, Dict]]:
        """
        计算最终实体得分
        
        Returns:
            [(entity_text, score, metadata), ...]
        """
        if self.global_theme_vector is None:
            return []
        
        results = []
        
        for entity_text, state in self.entity_memory.items():
            # 1. 语义相关度得分 (与全局主题的余弦相似度)
            sim_score = F.cosine_similarity(
                state.embedding.unsqueeze(0),
                self.global_theme_vector.unsqueeze(0)
            ).item()
            
            # 2. 频次强化得分 (对数平滑)
            freq_score = math.log(1 + state.freq)
            
            # 3. 位置先验得分
            # 首尾出现奖励
            pos_score = 0.0
            if state.first_pos < 0.15:
                pos_score += 0.5
            if state.last_pos > 0.85:
                pos_score += 0.5
            if 0.15 <= state.first_pos <= 0.85:
                pos_score = 0.3
            
            # 4. 跨段落分布得分 (分布越广越重要)
            span_ratio = len(state.chunks) / max(self.total_chunks, 1)
            span_score = math.sqrt(span_ratio)
            
            # 综合打分
            final_score = (
                lambda_sim * sim_score +
                lambda_freq * freq_score +
                lambda_pos * pos_score +
                lambda_span * span_score
            )
            
            metadata = {
                "freq": state.freq,
                "first_pos": round(state.first_pos, 3),
                "last_pos": round(state.last_pos, 3),
                "span_chunks": len(state.chunks),
                "sim_score": round(sim_score, 4),
                "component_scores": {
                    "semantic": round(sim_score, 4),
                    "frequency": round(freq_score, 4),
                    "position": round(pos_score, 4),
                    "span": round(span_score, 4)
                }
            }
            
            results.append((entity_text, final_score, metadata))
        
        # 按得分降序排序
        results.sort(key=lambda x: x[1], reverse=True)
        return results


class BertMemoryExtractor:
    """BERT + 双重记忆提取器"""
    
    # 预设的中文 BERT 模型
    PRESET_MODELS = {
        "roberta-wwm": "hfl/chinese-roberta-wwm-ext",
        "roberta-wwm-large": "hfl/chinese-roberta-wwm-ext-large",
        "macbert": "hfl/chinese-macbert-base",
        "macbert-large": "hfl/chinese-macbert-large",
        "tinybert": "uer/tinybert-chinese",
    }
    
    def __init__(
        self,
        model_name: str = "hfl/chinese-roberta-wwm-ext",
        device: Optional[str] = None,
        chunk_size: int = 300,
        chunk_overlap: int = 50,
        alpha: float = 0.2,
        dynamic_alpha: bool = True
    ):
        """
        Args:
            model_name: BERT 模型名称或预设 key
            device: 运行设备
            chunk_size: 每个 chunk 的 token 数
            chunk_overlap: chunk 之间的重叠 token 数
            alpha: GTM 更新系数
            dynamic_alpha: 是否使用动态 alpha
        """
        # 解析模型名称
        if model_name in self.PRESET_MODELS:
            model_name = self.PRESET_MODELS[model_name]
        
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # 初始化模型
        logger.info(f"加载 BERT 模型: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        # 隐藏层维度
        self.hidden_size = self.model.config.hidden_size
        
        # 记忆追踪器参数
        self.alpha = alpha
        self.dynamic_alpha = dynamic_alpha
        
        logger.info(f"BERT 模型加载完成，hidden_size={self.hidden_size}")
    
    def _split_into_chunks(self, text: str) -> List[str]:
        """将长文本切分为带重叠的 chunks"""
        # 先按句子切分
        sentences = re.split(r'([。！？；\n]+)', text)
        # 还原句子（包含标点）
        sentences = [sentences[i] + (sentences[i+1] if i+1 < len(sentences) else '') 
                    for i in range(0, len(sentences), 2)]
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return [text]
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sent in sentences:
            tokens = self.tokenizer.tokenize(sent)
            sent_len = len(tokens)
            
            if current_length + sent_len > self.chunk_size and current_chunk:
                # 保存当前 chunk
                chunks.append(''.join(current_chunk))
                # 保留部分句子作为重叠
                overlap_tokens = 0
                overlap_sents = []
                for s in reversed(current_chunk):
                    tlen = len(self.tokenizer.tokenize(s))
                    if overlap_tokens + tlen > self.chunk_overlap:
                        break
                    overlap_sents.insert(0, s)
                    overlap_tokens += tlen
                current_chunk = overlap_sents
                current_length = overlap_tokens
            
            current_chunk.append(sent)
            current_length += sent_len
        
        # 保存最后一个 chunk
        if current_chunk:
            chunks.append(''.join(current_chunk))
        
        return chunks
    
    def _extract_candidates(self, text: str) -> Dict[str, List[Tuple[int, int]]]:
        """
        提取候选实体
        
        Returns:
            {entity_text: [(start_char, end_char), ...]}
        """
        candidates = defaultdict(list)
        
        # Jieba 词性标注
        words = list(pseg.cut(text))
        
        char_pos = 0
        for word, flag in words:
            word_len = len(word)
            # 提取名词、专有名词、机构名、人名
            if flag in ['n', 'nz', 'nt', 'nr', 'ns', 'nw']:
                # 长度过滤
                if 2 <= len(word) <= 12:
                    candidates[word].append((char_pos, char_pos + word_len))
            
            # 额外提取：英文大写开头的词（公司/产品名）
            if re.match(r'^[A-Z][a-zA-Z0-9]*(?:\-[A-Z]?[a-zA-Z0-9]*)*$', word):
                if 2 <= len(word) <= 20:
                    candidates[word].append((char_pos, char_pos + word_len))
            
            char_pos += word_len
        
        return dict(candidates)
    
    def _get_entity_token_indices(
        self,
        text: str,
        char_spans: List[Tuple[int, int]],
        offset_mapping: List[Tuple[int, int]]
    ) -> List[int]:
        """获取实体在 token 序列中的索引"""
        token_indices = []
        
        for start_char, end_char in char_spans:
            for idx, (tok_start, tok_end) in enumerate(offset_mapping):
                # 跳过特殊 token ([CLS], [SEP], [PAD])
                if tok_start == tok_end == 0:
                    continue
                # 如果 token 与实体有重叠
                if tok_start < end_char and tok_end > start_char:
                    token_indices.append(idx)
        
        return sorted(set(token_indices))
    
    @torch.no_grad()
    def extract(
        self,
        text: str,
        top_k: int = 10,
        return_metadata: bool = False
    ) -> ExtractionResult:
        """
        提取关键词
        
        Args:
            text: 输入文本
            top_k: 返回关键词数量
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
                method="bert-memory",
                elapsed_time=0.0,
                model=self.model_name
            )
        
        # 1. 分块
        chunks = self._split_into_chunks(text)
        self.total_chunks = len(chunks)
        
        logger.debug(f"文本分块: {self.total_chunks} 个 chunks")
        
        # 2. 初始化记忆追踪器
        tracker = DocumentMemoryTracker(
            alpha=self.alpha,
            dynamic_alpha=self.dynamic_alpha
        )
        tracker.total_chunks = self.total_chunks
        
        # 3. 逐块编码
        for chunk_idx, chunk_text in enumerate(chunks):
            relative_pos = chunk_idx / max(self.total_chunks - 1, 1)
            
            # 提取候选实体
            candidates = self._extract_candidates(chunk_text)
            
            if not candidates:
                continue
            
            # BERT 编码
            inputs = self.tokenizer(
                chunk_text,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                return_offsets_mapping=True
            ).to(self.device)
            
            outputs = self.model(**inputs)
            last_hidden_state = outputs.last_hidden_state.squeeze(0)  # [seq_len, hidden]
            offset_mapping = inputs.offset_mapping.squeeze(0).tolist()
            
            # 更新 GTM
            cls_vector = last_hidden_state[0]  # [CLS]
            tracker.update_global_theme(cls_vector, relative_pos)
            
            # 更新 ESM
            for entity_text, char_spans in candidates.items():
                token_indices = self._get_entity_token_indices(
                    chunk_text, char_spans, offset_mapping
                )
                
                if token_indices:
                    # 提取实体 token 向量并平均池化
                    entity_vectors = last_hidden_state[token_indices]
                    entity_vector = torch.mean(entity_vectors, dim=0)
                    
                    tracker.update_entity_state(
                        entity_text, entity_vector, relative_pos, chunk_idx
                    )
        
        # 4. 全局打分
        scored_entities = tracker.compute_final_scores()
        
        # 5. 组装结果
        keywords = []
        for entity_text, score, metadata in scored_entities[:top_k]:
            kw_item = KeywordItem(
                keyword=entity_text,
                score=score,
                method="bert-memory"
            )
            if return_metadata:
                kw_item.metadata = metadata
            keywords.append(kw_item)
        
        elapsed = time.time() - start_time
        
        return ExtractionResult(
            text=text[:200] + "..." if len(text) > 200 else text,
            keywords=keywords,
            method="bert-memory",
            elapsed_time=elapsed,
            model=self.model_name
        )
