"""
动态门控 Token 截断层（Top-K 物理压缩 + 可导路由）
================================================

与 ``soft_masked_deberta_v2``（全长序列 + attention mask 软屏蔽）不同：本模块在指定层之后
**物理缩短** 序列长度 ``L -> keep_k``，后续层注意力复杂度从 ``O(L^2)`` 降为 ``O(keep_k^2)``。

**数学要点（梯度不断裂）**
  - ``torch.gather`` 对索引不可导，Router 若只产生离散 top-k，路由分支在索引处断图。
  - 对 gather 后的 ``compressed_states`` 乘以 **对应 token 的 router 分数**（与选中位置对齐），
    Loss 对压缩表示的反传仍会流经乘法节点回到 ``scores``，再回传到 Router MLP。

**工程大坑（坐标）**
  Span 在压缩空间 ``[i, j]`` 必须经 ``topk_indices[i], topk_indices[j]`` 映回原文序列下标；
  若下游用 **字符偏移** 或 **子词 tokenizer**，还需再经 offset_mapping 换到字符域。

集成 GLiNER / DeBERTa：需在 **第 k 层之后** 插入本层，并把 **第 k+1..L 层** 改为在压缩序列上跑；
当前仓库提供独立模块 + 映射工具，不直接 patch 第三方 ``gliner`` 包。

**接 DeBERTa-v2 的额外坑**：原模型相对位置 / attention mask 按 **原长 L** 构图；截断到 ``keep_k`` 后，
后续层须在 **压缩序列长度** 上重算 ``relative_pos`` 与 mask（不能沿用原 8000 的矩阵），否则与预训练
几何不一致；通常需 fork ``modeling_deberta_v2`` 或自定义 Encoder 循环。
"""
from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class TopKGatedDroppingLayer(nn.Module):
    """
    极轻量 Router：逐 token 打分 → Top-K → 按 **原文位置** 排序 → gather → × score 保持可导。

    输入 ``hidden_states``: ``[B, L, D]``
    输出 ``compressed_states``: ``[B, keep_k, D]``（当 ``L > keep_k``），以及 ``topk_indices`` ``[B, keep_k]``
    存 **原序列中的绝对位置**（0 .. L-1）。
    """

    def __init__(self, hidden_dim: int, keep_k: int = 1500):
        super().__init__()
        self.keep_k = int(keep_k)
        self.router = nn.Sequential(
            nn.Linear(hidden_dim, max(hidden_dim // 2, 1)),
            nn.GELU(),
            nn.Linear(max(hidden_dim // 2, 1), 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            compressed_states: ``[B, min(L, keep_k), D]`` — 短序列时不截断，长度为 L
            topk_indices: ``[B, min(L, keep_k)]`` — 每个位置在 **原长 L** 下的列索引；短序列时为 ``0..L-1``
        """
        B, L, D = hidden_states.shape

        # 短序列：不压缩，索引为恒等映射（方便下游统一用 map_span_to_original）
        if L <= self.keep_k:
            idx = torch.arange(L, device=hidden_states.device, dtype=torch.long).unsqueeze(0).expand(B, -1)
            return hidden_states, idx

        # [B, L, D] -> [B, L, 1] -> [B, L] 每个 token 保留强度
        scores = self.router(hidden_states).squeeze(-1)

        # Top-K：每行在序列维上取最大的 keep_k 个下标与分数
        # topk_values: [B, keep_k], topk_indices: [B, keep_k]
        topk_values, topk_indices = torch.topk(scores, self.keep_k, dim=1)

        # 必须按 **原绝对位置** 升序，保证子序列仍保持阅读顺序
        topk_indices_sorted, sort_idx = torch.sort(topk_indices, dim=1)
        topk_values_sorted = torch.gather(topk_values, 1, sort_idx)

        # gather: 索引 [B, keep_k] -> [B, keep_k, D]
        gather_idx = topk_indices_sorted.unsqueeze(-1).expand(-1, -1, D)
        compressed = torch.gather(hidden_states, dim=1, index=gather_idx)

        # 软路由乘法：梯度经 compressed 回传到 topk_values_sorted -> scores -> router
        compressed = compressed * topk_values_sorted.unsqueeze(-1)

        return compressed, topk_indices_sorted


def map_span_to_original(
    span_start_compressed: Union[int, torch.Tensor],
    span_end_compressed: Union[int, torch.Tensor],
    topk_indices: torch.Tensor,
    *,
    end_inclusive: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    将压缩空间下的 token 下标映回 **原长序列** 下标。

    Args:
        span_start_compressed: 压缩序列中实体起点下标（0-based），标量或 ``[B]`` 整型张量
        span_end_compressed: 实体终点下标；若 ``end_inclusive=True`` 为闭区间端点，与 GLiNER 常用 span 一致
        topk_indices: ``[B, K]``，由 :class:`TopKGatedDroppingLayer` 返回
        end_inclusive: 若为 True，``end`` 与 ``start`` 同为闭区间索引参与查表

    Returns:
        (orig_start, orig_end)，形状与广播后的 batch 一致，为 ``torch.long``
    """
    device = topk_indices.device
    B, K = topk_indices.shape

    if isinstance(span_start_compressed, int):
        s = torch.full((B,), span_start_compressed, device=device, dtype=torch.long)
    else:
        s = span_start_compressed.to(device=device, dtype=torch.long).view(-1)
        if s.numel() == 1 and B > 1:
            s = s.expand(B)
    if isinstance(span_end_compressed, int):
        e = torch.full((B,), span_end_compressed, device=device, dtype=torch.long)
    else:
        e = span_end_compressed.to(device=device, dtype=torch.long).view(-1)
        if e.numel() == 1 and B > 1:
            e = e.expand(B)

    if s.shape[0] != B or e.shape[0] != B:
        raise ValueError(f"span 批维 {s.shape[0]}/{e.shape[0]} 与 topk_indices 的 B={B} 不一致")

    batch_idx = torch.arange(B, device=device, dtype=torch.long)
    orig_s = topk_indices[batch_idx, s.clamp(0, K - 1)]
    orig_e = topk_indices[batch_idx, e.clamp(0, K - 1)]
    if not end_inclusive:
        # 若 end 为开区间，此处不自动 +1，由调用方约定
        pass
    return orig_s, orig_e


def _mock_test() -> None:
    """随机张量检查形状 + 梯度是否回到 router。"""
    torch.manual_seed(0)
    B, L, D, K = 2, 8000, 768, 1500
    device = torch.device("cpu")
    layer = TopKGatedDroppingLayer(D, keep_k=K).to(device)
    pre = torch.randn(B, L, D, device=device, requires_grad=True)

    comp, idx = layer(pre)
    assert comp.shape == (B, K, D), f"compressed 形状期望 {(B, K, D)}，得到 {tuple(comp.shape)}"
    assert idx.shape == (B, K), f"idx 形状期望 {(B, K)}，得到 {tuple(idx.shape)}"
    # 顺序：每行应严格递增（按原文位置）
    assert (idx[:, 1:] >= idx[:, :-1]).all(), "topk_indices 必须按原位置升序"

    loss = comp.sum()
    loss.backward()
    assert layer.router[0].weight.grad is not None, "Router 第一层应有梯度"
    assert layer.router[0].weight.grad.abs().sum() > 0, "Router 梯度不应全 0"
    print("[mock] TopKGatedDroppingLayer OK: 形状与 router 梯度连通")

    # 短序列分支
    layer2 = TopKGatedDroppingLayer(D, keep_k=5000).to(device)
    xs = torch.randn(B, 100, D, device=device, requires_grad=True)
    c2, i2 = layer2(xs)
    assert c2.shape == (B, 100, D) and i2.shape == (B, 100)
    (c2.sum()).backward()
    print("[mock] 短序列 L<=keep_k 直通 OK")

    # 映射
    os_, oe_ = map_span_to_original(10, 15, idx)
    assert os_.shape == (B,) and oe_.shape == (B,)
    for b in range(B):
        assert os_[b] == idx[b, 10] and oe_[b] == idx[b, 15]
    print("[mock] map_span_to_original OK")


__all__ = [
    "TopKGatedDroppingLayer",
    "map_span_to_original",
]


if __name__ == "__main__":
    _mock_test()
