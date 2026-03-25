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
from .html_cleaner import clean_wechat_article
from .noise_gate import apply_noise_gate_if_enabled
from .labels import GLINER_TRAINING_LABELS, DEFAULT_LABEL_THRESHOLDS
from .gliner_config import GLINER_BASE_MODEL_ID


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
        norm_key = f"{entity_text.strip().lower()}_{label}"

        if norm_key not in self.entity_memory:
            self.entity_memory[norm_key] = GLiNEREntityState(
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
            st = self.entity_memory[norm_key]
            if len(entity_text) > len(st.text):
                st.text = entity_text
            st.update(confidence, embedding, relative_pos, chunk_idx)
    
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
    
    # 与训练数据、评估脚本对齐的多类型标签
    DEFAULT_LABELS = list(GLINER_TRAINING_LABELS)
    
    def __init__(
        self,
        model_name: str = GLINER_BASE_MODEL_ID,
        labels: Optional[List[str]] = None,
        device: Optional[str] = None,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
        alpha: float = 0.2,
        threshold: float = 0.3,
        use_noise_gate: bool = True,
        use_semantic_chunks: bool = True,
        label_thresholds: Optional[Dict[str, float]] = None,
        use_label_embedding_rerank: bool = True,
        use_topk_gate: bool = False,
        topk_keep_k: int = 1500,
        topk_compress_after_layer: int = 2,
    ):
        """
        Args:
            model_name: GLiNER 模型名称或本地微调目录（须与 SFT 所用基座同架构；
                默认与 keyword_extractor.gliner_config.GLINER_BASE_MODEL_ID 一致）
            labels: 实体类型标签列表
            device: 运行设备
            chunk_size: 滑动窗口大小
            chunk_overlap: 窗口重叠
            alpha: GTM 更新系数
            threshold: 基础置信度阈值（各标签可取更高下限）
            use_noise_gate: 是否在编码前做段落级噪声门控
            use_semantic_chunks: True=按段落/句群分块；False=原句子滑窗
            label_thresholds: 覆盖默认的按标签阈值表
            use_label_embedding_rerank: 是否用 BGE 标签向量对置信度做轻微重加权
            use_topk_gate: 是否启用 Top-K 门控物理压缩（GatedGLiNER V3）
            topk_keep_k: Top-K 保留 token 数（仅当 use_topk_gate=True 时生效）
            topk_compress_after_layer: Top-K 插入 DeBERTa 层位置（0-based）
        """
        if not GLINER_AVAILABLE:
            raise RuntimeError("GLiNER 未安装，请运行: pip install gliner")
        
        if model_name in self.PRESET_MODELS:
            model_name = self.PRESET_MODELS[model_name]
        
        self.model_name = model_name
        self.labels = labels or self.DEFAULT_LABELS
        
        # 设备选择 - 优先 MPS (Mac GPU)，其次 CUDA，最后 CPU
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
                logger.info("✅ 使用 Mac GPU (MPS) 加速")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
                logger.info("✅ 使用 CUDA GPU 加速")
            else:
                self.device = torch.device("cpu")
                logger.info("⚠️  使用 CPU 推理")
        else:
            self.device = torch.device(device)
        
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.alpha = alpha
        self.threshold = threshold
        self.use_noise_gate = use_noise_gate
        self.use_semantic_chunks = use_semantic_chunks
        self.label_thresholds = dict(DEFAULT_LABEL_THRESHOLDS)
        if label_thresholds:
            self.label_thresholds.update(label_thresholds)
        self.use_label_embedding_rerank = use_label_embedding_rerank
        self._st_model = None
        self._label_emb_tensor: Optional[torch.Tensor] = None
        self._label_to_idx: Dict[str, int] = {}

        # 加载 GLiNER 模型
        logger.info(f"加载 GLiNER 模型: {model_name}")
        self.model = GLiNER.from_pretrained(model_name)

        # 可选：注入 Top-K 门控物理压缩 Encoder
        if use_topk_gate:
            try:
                from .gated_gliner import attach_topk_gate
                attach_topk_gate(
                    self.model,
                    compress_after_layer=topk_compress_after_layer,
                    keep_k=topk_keep_k,
                )
                logger.info(
                    f"✅ Top-K Gate 已注入: compress_after_layer={topk_compress_after_layer}, keep_k={topk_keep_k}"
                )
                # 启用 Top-K 后可加大 chunk_size（内部会压缩）
                if self.chunk_size <= 800:
                    self.chunk_size = min(topk_keep_k, 2000)
                    logger.info(f"   chunk_size 自动调整为 {self.chunk_size}")
            except ImportError as e:
                logger.warning(f"⚠️  Top-K Gate 注入失败（缺少依赖）: {e}")

        self.model.eval()
        self.model.to(self.device)
        logger.info(f"✅ GLiNER 模型加载完成")

    def _iter_semantic_units(self, text: str) -> List[str]:
        """先按空行分段，过长段再按句切分。"""
        paras = [p.strip() for p in re.split(r"\n\s*\n+", text.strip()) if p.strip()]
        units: List[str] = []
        for p in paras:
            if len(p) <= self.chunk_size:
                units.append(p)
                continue
            parts = re.split(r"(?<=[。！？；\n])", p)
            buf = ""
            for part in parts:
                part = part.strip()
                if not part:
                    continue
                if len(buf) + len(part) <= self.chunk_size:
                    buf = f"{buf}{part}" if buf else part
                else:
                    if buf:
                        units.append(buf)
                    buf = part
            if buf:
                units.append(buf)
        return units if units else ([text.strip()] if text.strip() else [])

    def _split_into_chunks(self, text: str) -> List[Tuple[int, str]]:
        """滑动窗口切分（原策略），返回 (chunk_idx, chunk_text)"""
        sentences = re.split(r"([。！？；\n]+)", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks: List[str] = []
        current_chunk: List[str] = []
        current_length = 0

        for sent in sentences:
            sent_len = len(sent)

            if current_length + sent_len > self.chunk_size and current_chunk:
                chunks.append("\n".join(current_chunk))
                overlap_text = "".join(current_chunk)[-self.chunk_overlap :]
                current_chunk = [overlap_text, sent] if overlap_text.strip() else [sent]
                current_length = sum(len(x) for x in current_chunk)
            else:
                current_chunk.append(sent)
                current_length += sent_len

        if current_chunk:
            chunks.append("\n".join(current_chunk))

        return list(enumerate(chunks))

    def _smart_chunks(self, text: str) -> List[Tuple[int, str]]:
        """语义单元贪心装箱 + 重叠。"""
        units = self._iter_semantic_units(text)
        if not units:
            return []
        chunks: List[str] = []
        cur: List[str] = []
        cur_len = 0
        sep_len = 1

        for u in units:
            add_len = len(u) + (sep_len if cur else 0)
            if cur_len + add_len > self.chunk_size and cur:
                piece = "\n".join(cur)
                chunks.append(piece)
                tail = piece[-self.chunk_overlap :] if len(piece) > self.chunk_overlap else piece
                cur = [tail, u] if tail.strip() else [u]
                cur_len = sum(len(x) + sep_len for x in cur) - sep_len
            else:
                cur.append(u)
                cur_len += add_len
        if cur:
            chunks.append("\n".join(cur))
        return list(enumerate(chunks))
    
    def _get_sentence_transformer(self):
        if self._st_model is None:
            logger.debug("Loading BGE-small-zh for GTM / label precompute...")
            from sentence_transformers import SentenceTransformer

            self._st_model = SentenceTransformer(
                "BAAI/bge-small-zh-v1.5",
                device=str(self.device),
            )
            self._st_model.eval()
        return self._st_model

    def _ensure_label_embeddings(self) -> None:
        if self._label_emb_tensor is not None:
            return
        try:
            model = self._get_sentence_transformer()
            with torch.no_grad():
                te = model.encode(
                    self.labels,
                    convert_to_tensor=True,
                    show_progress_bar=False,
                )
            self._label_emb_tensor = te.cpu()
            self._label_to_idx = {lb: i for i, lb in enumerate(self.labels)}
        except Exception as e:
            logger.debug(f"Label embedding skip: {e}")
            self._label_emb_tensor = None
            self._label_to_idx = {}

    def _label_alignment_factor(self, entity_text: str, label: str) -> float:
        """实体文本与标签提示的 BGE 余弦相似度，映射到缩放因子。"""
        if not self.use_label_embedding_rerank or not self._label_emb_tensor:
            return 1.0
        idx = self._label_to_idx.get(label)
        if idx is None:
            return 1.0
        try:
            model = self._get_sentence_transformer()
            with torch.no_grad():
                ev = model.encode(
                    entity_text,
                    convert_to_tensor=True,
                    show_progress_bar=False,
                )
            lev = self._label_emb_tensor[idx].to(ev.device)
            sim = F.cosine_similarity(ev.unsqueeze(0), lev.unsqueeze(0)).clamp(0, 1).item()
            return 0.55 + 0.45 * sim
        except Exception:
            return 1.0

    def _get_chunk_embedding(self, text: str) -> torch.Tensor:
        """chunk 向量（GTM）。"""
        try:
            model = self._get_sentence_transformer()
            with torch.no_grad():
                embedding = model.encode(
                    text,
                    convert_to_tensor=True,
                    show_progress_bar=False,
                )
            return embedding.cpu()
        except Exception as e:
            logger.debug(f"Embedding error: {e}, fallback to hash")
            return self._hash_embedding(text)
    
    def _hash_embedding(self, text: str) -> torch.Tensor:
        """哈希向量备选方案"""
        words = text.split()[:50]
        if not words:
            return torch.randn(384)  # MiniLM 是 384 维
        
        vec = torch.zeros(384)
        for i, word in enumerate(words):
            hash_val = hash(word) % 10000
            vec[i % 384] += hash_val / 10000.0
        
        vec = vec / len(words)
        return vec
    
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
        
        # 0. HTML 清洗（如果是 HTML 内容）
        if "<" in text and ">" in text:
            text = clean_wechat_article(text, method="auto")
            logger.debug(f"HTML 清洗后长度: {len(text)}")

        if self.use_noise_gate:
            text = apply_noise_gate_if_enabled(text, enabled=True)
            logger.debug(f"噪声门控后长度: {len(text)}")

        self._ensure_label_embeddings()

        # 1. 语义分块或滑窗
        if self.use_semantic_chunks:
            chunks = self._smart_chunks(text)
        else:
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
            # 先低阈值召回，再用按标签阈值 + BGE 对齐过滤
            low_th = max(0.05, min(0.14, self.threshold * 0.45))
            entities = self.model.predict_entities(
                chunk_text,
                self.labels,
                threshold=low_th,
            )

            # 3.2 获取 chunk 向量（用于 GTM）
            chunk_embedding = self._get_chunk_embedding(chunk_text)
            tracker.update_global_theme(chunk_embedding, relative_pos)

            # 3.3 更新实体记忆（按标签动态阈值 + 可选 BGE 对齐）
            for entity in entities:
                entity_text = entity["text"]
                label = entity["label"]
                confidence = float(entity["score"])
                min_l = self.label_thresholds.get(label, self.threshold)
                if confidence < min_l:
                    continue
                confidence *= self._label_alignment_factor(entity_text, label)
                if confidence < min_l:
                    continue
                
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
