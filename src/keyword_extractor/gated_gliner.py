"""
GatedGLiNER：将 Top-K 物理压缩 Encoder 无缝集成到 GLiNER
==========================================================

**设计决策：Scatter-Expand 兼容策略**

1. ``TopKCompressedDebertaV2Encoder`` 在 DeBERTa 内部把序列压缩到 ``[B, K, D]``，
   节省 O(L² → K²) 的注意力计算。

2. ``GatedGLiNEREncoder.encode_text`` 拿到 ``[B, K, D]`` 之后，用 ``topk_indices``
   把它 scatter 回 ``[B, L, D]``，被丢弃的 token 位置填 0。

3. 此策略的好处：
   - GLiNER 的 ``extract_word_embeddings`` / ``words_mask`` / ``span_idx`` 等
     下游代码 **一行不改**，坐标始终在原文空间。
   - 被丢弃的 token 的 word embedding = 0 → span 分数低 → NER loss 高
     → 梯度回传到 router MLP，router 学会「不要丢有效词」。
   - 与 ``topk_gated_dropping.py`` 里的 ``* scores`` 乘法共同保证梯度不断裂。

4. ``predict_entities_with_mapping`` 包装 GLiNER 标准预测，输出的 span 坐标
   已经在原文 word 空间，无需额外映射。

**训练信号路径**::

  NER focal_loss
      → words_embedding[dropped_positions] = 0
      → scatter(expanded, topk_idx, compressed_states)   ← 可导
      → compressed_states = gathered × topk_scores       ← 可导
      → topk_scores → router MLP                         ← 可导
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Union
from pathlib import Path

import torch
import torch.nn as nn

try:
    from gliner import GLiNER
    from gliner.modeling.encoder import Encoder
    _GLINER_AVAILABLE = True
except ImportError:
    _GLINER_AVAILABLE = False
    GLiNER = None
    Encoder = nn.Module

from .topk_compressed_encoder import attach_topk_compressed_encoder
from .topk_gated_dropping import map_span_to_original


# ---------------------------------------------------------------------------
# GatedGLiNEREncoder
# ---------------------------------------------------------------------------

class GatedGLiNEREncoder(Encoder):
    """
    继承 GLiNER ``Encoder``，在 DeBERTa 内部插入 Top-K 压缩层，
    并在 ``encode_text`` 返回前将压缩序列 scatter 回原始长度，
    保证下游 GLiNER 代码（words_mask / span_idx 等）无需修改。

    额外公开属性
    ------------
    _last_topk_indices : Optional[Tensor]  [B, keep_k]
        最近一次前向的 top-k 保留索引（原文 token 位置），供可视化/分析使用。
    """

    def __init__(
        self,
        config: Any,
        from_pretrained: bool = False,
        compress_after_layer: int = 2,
        keep_k: int = 1500,
        n_prompt_tokens: int = 0,
        cache_dir: Optional[Union[str, Path]] = None,
    ) -> None:
        if not _GLINER_AVAILABLE:
            raise ImportError("请先安装 gliner：pip install gliner")
        super().__init__(config, from_pretrained, cache_dir=cache_dir)
        # 替换 DeBERTa 内部 encoder
        attach_topk_compressed_encoder(
            self.bert_layer.model,
            compress_after_layer=compress_after_layer,
            keep_k=keep_k,
            n_prompt_tokens=n_prompt_tokens,
            copy_weights=False,  # 权重已经在 Encoder.__init__ 里加载好了
        )
        self._last_topk_indices: Optional[torch.Tensor] = None

    def encode_text(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *args: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        """
        运行 DeBERTa（内部物理压缩 L→K），再 scatter 回 [B, L, D]。

        Dropped token 位置填 0，不影响下游 words_mask 的索引逻辑。
        """
        # --- 父类调用 (内部走压缩 encoder) ---
        # 结果 shape: [B, K, D]（K ≤ L，K = keep_k 或 L 若序列本身短）
        compressed_embeds = super().encode_text(
            input_ids, attention_mask, *args, **kwargs
        )

        # --- 取 topk_indices ---
        topk_indices: Optional[torch.Tensor] = getattr(
            self.bert_layer.model.encoder, "_last_topk_indices", None
        )
        self._last_topk_indices = topk_indices

        if topk_indices is None:
            # 没有发生压缩（序列太短 / 压缩层未触发）
            return compressed_embeds

        B, K, D = compressed_embeds.shape
        L = int(input_ids.size(1))

        if K == L:
            # 未发生实质压缩，直接返回
            return compressed_embeds

        # --- Scatter 回原始长度 ---
        # topk_indices: [B, K]  → expand 到 [B, K, D] 作为 index
        idx = topk_indices.long().unsqueeze(-1).expand(-1, -1, D)
        expanded = compressed_embeds.new_zeros(B, L, D)
        # scatter 是可导操作：grad_src[b,k] = grad_output[b, topk_indices[b,k]]
        expanded = expanded.scatter(1, idx, compressed_embeds)
        return expanded


# ---------------------------------------------------------------------------
# 工厂函数：attach_topk_gate
# ---------------------------------------------------------------------------

def attach_topk_gate(
    gliner_model: "GLiNER",
    compress_after_layer: int = 2,
    keep_k: int = 1500,
    n_prompt_tokens: int = 0,
) -> "GLiNER":
    """
    就地修改 ``GLiNER`` 实例：将其 ``model.token_rep_layer`` (Encoder)
    内部的 DeBERTa encoder 替换为 ``TopKCompressedDebertaV2Encoder``，
    并升级为 ``GatedGLiNEREncoder``（保留原有权重）。

    Args:
        gliner_model: 已加载的 ``GLiNER`` 实例（from_pretrained 或自定义）。
        compress_after_layer: 第 k 层（0-based）输出后做 Top-K。
        keep_k: 保留 token 数量。
        n_prompt_tokens: 强制保留的前置 prompt token 数（0=全部参与竞争）。

    Returns:
        同一 ``gliner_model`` 实例（就地修改）。
    """
    if not _GLINER_AVAILABLE:
        raise ImportError("请先安装 gliner：pip install gliner")

    inner_model = gliner_model.model        # UniEncoderSpanModel / 同类
    encoder: Encoder = inner_model.token_rep_layer

    # 在现有 Encoder 的 DeBERTa model 上插入 TopK 压缩层
    # 注意：GatedGLiNEREncoder.__init__ 会调 Encoder.__init__ 重新加载权重；
    # 为避免重新下载，我们直接在原 encoder 实例上做就地替换，不重建整个 Encoder。
    attach_topk_compressed_encoder(
        encoder.bert_layer.model,
        compress_after_layer=compress_after_layer,
        keep_k=keep_k,
        n_prompt_tokens=n_prompt_tokens,
        copy_weights=True,
    )

    # 给原 Encoder 打补丁：注入 _last_topk_indices 属性 + 重写 encode_text
    encoder._last_topk_indices = None

    _original_encode_text = encoder.encode_text.__func__

    def _gated_encode_text(self: Encoder, input_ids, attention_mask, *args, **kwargs):
        compressed_embeds = _original_encode_text(self, input_ids, attention_mask, *args, **kwargs)
        topk_indices = getattr(
            self.bert_layer.model.encoder, "_last_topk_indices", None
        )
        self._last_topk_indices = topk_indices
        if topk_indices is None:
            return compressed_embeds
        B, K, D = compressed_embeds.shape
        L = int(input_ids.size(1))
        if K == L:
            return compressed_embeds
        idx = topk_indices.long().unsqueeze(-1).expand(-1, -1, D)
        expanded = compressed_embeds.new_zeros(B, L, D)
        expanded = expanded.scatter(1, idx, compressed_embeds)
        return expanded

    import types
    encoder.encode_text = types.MethodType(_gated_encode_text, encoder)

    return gliner_model


# ---------------------------------------------------------------------------
# GatedGLiNER：高级封装
# ---------------------------------------------------------------------------

class GatedGLiNER:
    """
    对 ``GLiNER`` 的轻量封装，集成 Top-K Encoder，提供与原 ``GLiNER`` 相同的
    预测接口，同时暴露压缩元数据（``last_topk_indices``）供分析。

    用法::

        model = GatedGLiNER.from_pretrained(
            "urchade/gliner_multi-v2.1",
            compress_after_layer=2,
            keep_k=1500,
        )
        entities = model.predict_entities(text, labels, threshold=0.3)
    """

    def __init__(self, gliner_model: "GLiNER") -> None:
        if not _GLINER_AVAILABLE:
            raise ImportError("请先安装 gliner：pip install gliner")
        self.model = gliner_model

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        compress_after_layer: int = 2,
        keep_k: int = 1500,
        n_prompt_tokens: int = 0,
        **gliner_kwargs: Any,
    ) -> "GatedGLiNER":
        """
        加载标准 ``GLiNER`` 预训练模型，然后就地注入 Top-K 压缩 Encoder。

        Args:
            model_name: HuggingFace Hub 模型 ID 或本地路径。
            compress_after_layer: Top-K 插入层（0-based）。
            keep_k: 压缩后保留 token 数。
            n_prompt_tokens: 强制保留的前置 prompt token 数。
            **gliner_kwargs: 透传给 ``GLiNER.from_pretrained``。

        Returns:
            ``GatedGLiNER`` 实例。
        """
        base = GLiNER.from_pretrained(model_name, **gliner_kwargs)
        attach_topk_gate(
            base,
            compress_after_layer=compress_after_layer,
            keep_k=keep_k,
            n_prompt_tokens=n_prompt_tokens,
        )
        return cls(base)

    # ------------------------------------------------------------------
    # 代理接口（透传到底层 GLiNER）
    # ------------------------------------------------------------------

    def predict_entities(
        self,
        text: str,
        labels: Sequence[str],
        threshold: float = 0.3,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """与 ``GLiNER.predict_entities`` 完全相同的接口。"""
        return self.model.predict_entities(text, labels, threshold=threshold, **kwargs)

    def batch_predict_entities(
        self,
        texts: List[str],
        labels: Sequence[str],
        threshold: float = 0.3,
        **kwargs: Any,
    ) -> List[List[Dict[str, Any]]]:
        """与 ``GLiNER.batch_predict_entities`` 完全相同的接口。"""
        return self.model.batch_predict_entities(texts, labels, threshold=threshold, **kwargs)

    def train(self) -> "GatedGLiNER":
        self.model.model.train()
        return self

    def eval(self) -> "GatedGLiNER":
        self.model.model.eval()
        return self

    def to(self, device: Any) -> "GatedGLiNER":
        self.model.model.to(device)
        return self

    def parameters(self):
        return self.model.model.parameters()

    def named_parameters(self):
        return self.model.model.named_parameters()

    @property
    def last_topk_indices(self) -> Optional[torch.Tensor]:
        """最近一次前向的 [B, keep_k] 压缩索引，供分析使用。"""
        enc = self.model.model.token_rep_layer
        return getattr(enc, "_last_topk_indices", None)

    # ------------------------------------------------------------------
    # 辅助：映射 span 坐标（当 scatter-expand 未使用时的备用接口）
    # ------------------------------------------------------------------

    def map_word_spans(
        self,
        spans: List[Dict[str, Any]],
        words_mask: torch.Tensor,
    ) -> List[Dict[str, Any]]:
        """
        如果使用了压缩（``last_topk_indices`` 非 None），将 word 级 span
        坐标映射回原文 token 空间（通常不需要，因为 scatter-expand 已保持原坐标）。

        目前仅作为预留接口，直接返回原 spans。
        """
        return spans

    def router_gate_parameters(self) -> List[nn.Parameter]:
        """返回 router (门控 MLP) 的所有参数，用于阶段一单独优化。"""
        params: List[nn.Parameter] = []
        enc = self.model.model.token_rep_layer
        topk_enc = getattr(
            getattr(getattr(enc, "bert_layer", None), "model", None), "encoder", None
        )
        if topk_enc is not None and hasattr(topk_enc, "gate_layer"):
            params.extend(topk_enc.gate_layer.router.parameters())
        return params

    def freeze_non_gate(self) -> None:
        """阶段一训练：冻结除 router 以外的全部参数。"""
        gate_params = set(id(p) for p in self.router_gate_parameters())
        for p in self.model.model.parameters():
            p.requires_grad_(id(p) in gate_params)

    def unfreeze_all(self) -> None:
        """阶段二训练：解冻全部参数。"""
        for p in self.model.model.parameters():
            p.requires_grad_(True)


__all__ = [
    "GatedGLiNEREncoder",
    "GatedGLiNER",
    "attach_topk_gate",
]
