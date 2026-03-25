"""
Keyword Extractor — 基于 Top-K Gated GLiNER V3 的中文关键词提取
"""

__version__ = "0.3.0"
__author__ = "Hao"

# ── 核心推理 ─────────────────────────────────────────────────────────────────
from .gliner_memory import GLiNEREntityExtractor, GLiNERMemoryTracker
from .gliner_config import GLINER_BASE_MODEL_ID, GLINER_DEFAULT_SFT_OUTPUT_DIR

# ── Top-K Gated GLiNER V3（核心架构）────────────────────────────────────────
from .topk_gated_dropping import TopKGatedDroppingLayer, map_span_to_original
from .topk_compressed_encoder import (
    TopKCompressedDebertaV2Encoder,
    attach_topk_compressed_encoder,
)
from .gated_gliner import GatedGLiNER, attach_topk_gate

# ── 预处理工具 ───────────────────────────────────────────────────────────────
from .html_cleaner import clean_wechat_article, extract_text
from .noise_gate import filter_text_by_noise_gate, apply_noise_gate_if_enabled

# ── 配置 & 标签 ──────────────────────────────────────────────────────────────
from .labels import GLINER_TRAINING_LABELS, DEFAULT_LABEL_THRESHOLDS, map_tag_type_to_label
from .models import KeywordItem, ExtractionResult

__all__ = [
    # 推理
    "GLiNEREntityExtractor",
    "GLiNERMemoryTracker",
    "GLINER_BASE_MODEL_ID",
    "GLINER_DEFAULT_SFT_OUTPUT_DIR",
    # Top-K 架构
    "TopKGatedDroppingLayer",
    "map_span_to_original",
    "TopKCompressedDebertaV2Encoder",
    "attach_topk_compressed_encoder",
    "GatedGLiNER",
    "attach_topk_gate",
    # 预处理
    "clean_wechat_article",
    "extract_text",
    "filter_text_by_noise_gate",
    "apply_noise_gate_if_enabled",
    # 配置 & 标签
    "GLINER_TRAINING_LABELS",
    "DEFAULT_LABEL_THRESHOLDS",
    "map_tag_type_to_label",
    "KeywordItem",
    "ExtractionResult",
]
