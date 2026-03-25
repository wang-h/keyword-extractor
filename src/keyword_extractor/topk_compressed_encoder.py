"""
Top-K 门控物理压缩 DeBERTa-v2 Encoder
======================================

**架构创新点（相对于现有工作）**

- vs CoLT5：CoLT5 在每层内部分轻重双分支（FFN/Attention），序列长度不变；
  本模块在「前 k 层后」做 **物理 gather 截断**，后续层输入长度真正从 L 降到 keep_k，
  注意力复杂度 O(L²) -> O(keep_k²)。

- vs 外部分块（gliner_memory.py）：外部切块是模型外的串行推理策略；本模块在
  **DeBERTa Encoder 内部、端到端可训**，Router 梯度可以回传。

- vs 软屏蔽（soft_masked_deberta_v2）：软屏蔽全长不变，仅屏蔽 key，后续层仍是
  O(L²)；本模块真降序列长度。

- **Prompt-aware 保留**：GLiNER 把 [label_prompts | text_tokens] 拼在一起输入
  DeBERTa；本模块可选地在 TopK 前强制保留前 ``n_prompt_tokens`` 个位置不被截断，
  确保实体类型提示不会被随机丢弃。

**梯度可导路径**

  compressed = gather(hidden, topk_idx) × topk_scores (unsqueeze)

  Loss -> compressed -> topk_scores -> scores (Sigmoid 输出) -> router MLP

**DeBERTa-v2 适配要点**

  压缩后 relative_pos 设为 None，每层 DisentangledSelfAttention 内部调
  build_relative_position(query_layer, key_layer) 自动根据新序列长度 keep_k
  重建相对位置表——无需手动干预。

  attention_mask 从原始 [B, L] gather 出 [B, keep_k]，再经 get_attention_mask()
  扩展为 4D，传入后续层。
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.deberta_v2.modeling_deberta_v2 import (
    DebertaV2Encoder,
    DebertaV2Model,
)

from .topk_gated_dropping import TopKGatedDroppingLayer


class TopKCompressedDebertaV2Encoder(DebertaV2Encoder):
    """
    继承 HF ``DebertaV2Encoder``，在第 ``compress_after_layer`` 层输出后插入
    ``TopKGatedDroppingLayer``，把序列长度物理压缩到 ``keep_k``，后续各层在压缩
    序列上运行。

    额外属性
    ---------
    compress_after_layer : int
        在哪一层输出后做 Top-K（0-based）。
    keep_k : int
        保留 token 数（压缩后序列长度）。
    n_prompt_tokens : int
        强制保留的前置 prompt token 数量（GLiNER 把标签 prompt 放在序列最前）。
        设为 0 表示不保留任何固定 token，全部参与 Top-K 竞争。
    gate_layer : TopKGatedDroppingLayer
        打分与压缩层。
    _last_topk_indices : Optional[Tensor]
        最近一次 forward 的 [B, keep_k] 原序列位置索引，供 span 坐标回传。
    """

    def __init__(
        self,
        config,
        compress_after_layer: int = 2,
        keep_k: int = 1500,
        n_prompt_tokens: int = 0,
    ) -> None:
        super().__init__(config)
        self.compress_after_layer = int(compress_after_layer)
        self.keep_k = int(keep_k)
        self.n_prompt_tokens = int(n_prompt_tokens)
        self.gate_layer = TopKGatedDroppingLayer(config.hidden_size, keep_k=keep_k)
        self._last_topk_indices: Optional[torch.Tensor] = None

        nn.init.normal_(self.gate_layer.router[0].weight, std=0.02)
        nn.init.zeros_(self.gate_layer.router[0].bias)

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    def _compress_attention_mask(
        self,
        attention_mask_1d: torch.Tensor,
        topk_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        从原始 1D attention_mask [B, L] 按 topk_indices [B, K] gather 出
        [B, K]，再扩展为 DeBERTa 兼容的 4D mask（调用父类 get_attention_mask）。
        """
        gather_idx = topk_indices.long()
        compressed_1d = torch.gather(attention_mask_1d, 1, gather_idx)
        return self.get_attention_mask(compressed_1d)

    def _topk_with_prompt_protect(
        self,
        hidden_states: torch.Tensor,
        attention_mask_1d: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        执行 Top-K 压缩，可选强制保留前 n_prompt_tokens 个 token。

        Returns:
            compressed_states: [B, keep_k, D]
            topk_indices: [B, keep_k] 原序列绝对位置
        """
        B, L, D = hidden_states.shape
        n_p = min(self.n_prompt_tokens, L)

        if n_p == 0 or L <= self.keep_k:
            # 不需要 prompt 保护，或序列本身不需压缩
            compressed, indices = self.gate_layer(hidden_states)
            return compressed, indices

        # 把前 n_p token（prompt）单独取出，剩余 text token 参与竞争
        k_text = max(1, self.keep_k - n_p)
        text_states = hidden_states[:, n_p:, :]  # [B, L-n_p, D]

        if text_states.shape[1] <= k_text:
            # text 部分本身就短，直接拼回
            text_idx = torch.arange(n_p, L, device=hidden_states.device, dtype=torch.long)
            text_idx = text_idx.unsqueeze(0).expand(B, -1)
            prompt_idx = torch.arange(n_p, device=hidden_states.device, dtype=torch.long)
            prompt_idx = prompt_idx.unsqueeze(0).expand(B, -1)
            all_idx = torch.cat([prompt_idx, text_idx], dim=1)  # [B, n_p + (L-n_p)]
            return hidden_states, all_idx

        # text 部分做 Top-K
        # 临时把 gate_layer 的 keep_k 改为 k_text
        old_k = self.gate_layer.keep_k
        self.gate_layer.keep_k = k_text
        text_compressed, text_rel_idx = self.gate_layer(text_states)
        self.gate_layer.keep_k = old_k

        # text_rel_idx 是相对 text_states 的下标，需要 + n_p 得到原序列绝对下标
        text_abs_idx = text_rel_idx + n_p

        # 拼接 prompt（绝对下标 0..n_p-1）
        prompt_abs_idx = torch.arange(n_p, device=hidden_states.device, dtype=torch.long)
        prompt_abs_idx = prompt_abs_idx.unsqueeze(0).expand(B, -1)

        all_idx = torch.cat([prompt_abs_idx, text_abs_idx], dim=1)  # [B, keep_k]

        # 再做一次 gather 拿完整 hidden（prompt 部分没有乘 score，保持原值）
        prompt_states = hidden_states[:, :n_p, :]  # [B, n_p, D]
        compressed_states = torch.cat([prompt_states, text_compressed], dim=1)

        return compressed_states, all_idx

    # ------------------------------------------------------------------
    # forward 主逻辑
    # ------------------------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        output_hidden_states: bool = True,
        output_attentions: bool = False,
        query_states: Optional[torch.Tensor] = None,
        relative_pos: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[BaseModelOutput, Tuple]:
        """
        前向流程：

        1. 第 0 .. compress_after_layer 层：全长 hidden（原始 L） + 原 4D mask + rel_pos
        2. TopKGatedDroppingLayer：[B, L, D] -> [B, keep_k, D] + 索引
        3. 压缩 attention_mask 为 [B, keep_k] -> 4D
        4. relative_pos = None（让 DisentangledSelfAttention 自动按 keep_k 重算）
        5. 第 compress_after_layer+1 .. end 层：在 [B, keep_k, D] 上运行
        6. 保存 _last_topk_indices
        """
        # 原始 1D mask 用于后面 gather
        if attention_mask.dim() <= 2:
            attention_mask_1d = attention_mask.float()
            input_mask = attention_mask_1d
        else:
            input_mask = attention_mask.sum(-2) > 0
            attention_mask_1d = input_mask.float()

        base_4d = self.get_attention_mask(attention_mask)
        rel_pos_full = self.get_rel_pos(hidden_states, query_states, relative_pos)
        rel_embeddings = self.get_rel_embedding()

        all_hidden_states: Optional[Tuple[torch.Tensor, ...]] = (
            (hidden_states,) if output_hidden_states else None
        )
        all_attentions = () if output_attentions else None

        next_kv = hidden_states
        compressed = False
        compressed_4d: Optional[torch.Tensor] = None
        self._last_topk_indices = None

        for i, layer_module in enumerate(self.layer):
            # 选择当前 attention_mask 和 rel_pos
            if not compressed:
                cur_mask = base_4d
                cur_rel = rel_pos_full
            else:
                cur_mask = compressed_4d
                cur_rel = None  # 压缩后让 DeBERTa 自动重算

            output_states, attn_w = layer_module(
                next_kv,
                cur_mask,
                query_states=query_states,
                relative_pos=cur_rel,
                rel_embeddings=rel_embeddings,
                output_attentions=output_attentions,
            )

            if output_attentions:
                all_attentions = all_attentions + (attn_w,)

            # 第 0 层之后可能有 ConvLayer
            if i == 0 and self.conv is not None:
                output_states = self.conv(hidden_states, output_states, input_mask)

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (output_states,)

            # 在 compress_after_layer 层后执行 Top-K 压缩
            if i == self.compress_after_layer and not compressed:
                compressed_states, topk_idx = self._topk_with_prompt_protect(
                    output_states, attention_mask_1d
                )
                self._last_topk_indices = topk_idx
                compressed = True
                # 压缩后的 4D attention_mask
                compressed_4d = self._compress_attention_mask(attention_mask_1d, topk_idx)
                output_states = compressed_states

                # 更新 input_mask 为压缩长度
                input_mask = torch.gather(attention_mask_1d, 1, topk_idx.long())

            # 更新下一层输入
            if query_states is not None:
                query_states = output_states
                if isinstance(hidden_states, Sequence):
                    next_kv = hidden_states[i + 1] if i + 1 < len(self.layer) else None
            else:
                next_kv = output_states

        if not return_dict:
            return tuple(v for v in [output_states, all_hidden_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=output_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
        )


# ------------------------------------------------------------------
# 工厂函数
# ------------------------------------------------------------------

def attach_topk_compressed_encoder(
    model: DebertaV2Model,
    compress_after_layer: int = 2,
    keep_k: int = 1500,
    n_prompt_tokens: int = 0,
    copy_weights: bool = True,
) -> DebertaV2Model:
    """
    将 ``model.encoder`` 替换为 :class:`TopKCompressedDebertaV2Encoder`，
    并可选拷贝原 encoder 层权重（``strict=False``，新增 ``gate_layer`` 随机初始化）。

    Args:
        model: ``DebertaV2Model`` 实例（如从 ``from_pretrained`` 加载）。
        compress_after_layer: 在哪一层（0-based）输出后做 Top-K。
        keep_k: 保留 token 数。
        n_prompt_tokens: 强制保留的前置 prompt token 数。
        copy_weights: 是否拷贝原 encoder 权重。

    Returns:
        同一 model 实例（就地修改）。
    """
    cfg = model.config
    new_enc = TopKCompressedDebertaV2Encoder(
        cfg,
        compress_after_layer=compress_after_layer,
        keep_k=keep_k,
        n_prompt_tokens=n_prompt_tokens,
    )
    if copy_weights:
        new_enc.load_state_dict(model.encoder.state_dict(), strict=False)
    model.encoder = new_enc
    return model


__all__ = [
    "TopKCompressedDebertaV2Encoder",
    "attach_topk_compressed_encoder",
]
