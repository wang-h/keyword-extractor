"""
软屏蔽门控弱监督：用 ``noise_gate`` 段落分数生成 token 级「噪声」伪标签（0/1）。
不追求完美，仅用于 ``gate_head`` 冷启动。
"""
from __future__ import annotations

from typing import List, Tuple

import torch

from keyword_extractor.noise_gate import score_segment, segment_paragraphs


def char_level_noise_targets(
    text: str,
    *,
    noise_if_segment_score_below: float = 0.22,
) -> List[float]:
    """
    每个 **字符** 一个目标：该字符若落在「低分段落」内则为 1.0（视为噪声源），否则 0.0。
    """
    n = len(text)
    if n == 0:
        return []
    char_t = [0.0] * n
    pos = 0
    for seg in segment_paragraphs(text):
        if not seg.strip():
            continue
        sc = score_segment(seg)
        noisy = 1.0 if sc < noise_if_segment_score_below else 0.0
        j = text.find(seg, pos)
        if j < 0:
            j = text.find(seg)
        if j < 0:
            continue
        for k in range(j, min(j + len(seg), n)):
            char_t[k] = max(char_t[k], noisy)
        pos = max(pos, j + len(seg))
    return char_t


def token_noise_targets_from_text(
    text: str,
    offset_mapping: List[Tuple[int, int]],
    *,
    noise_if_segment_score_below: float = 0.22,
    aggregate: str = "max",
) -> torch.Tensor:
    """
    根据 tokenizer 的 offset_mapping（单条样本）生成 ``(seq_len,)`` float 目标。

    Args:
        text: 原始字符串（与编码时一致）
        offset_mapping: ``Encoding.offset_mapping`` 去掉 special token 后或与 batch 对齐
        aggregate: ``max`` 或 ``mean``，将字符目标聚合到 token
    """
    char_t = char_level_noise_targets(text, noise_if_segment_score_below=noise_if_segment_score_below)
    n = len(char_t)
    out: List[float] = []
    for s, e in offset_mapping:
        if s is None or e is None or (s == 0 and e == 0):
            out.append(0.0)
            continue
        s = max(0, min(s, n))
        e = max(0, min(e, n))
        if e <= s:
            out.append(0.0)
            continue
        span = char_t[s:e]
        if not span:
            out.append(0.0)
        elif aggregate == "mean":
            out.append(sum(span) / len(span))
        else:
            out.append(float(max(span)))
    return torch.tensor(out, dtype=torch.float32)


__all__ = [
    "char_level_noise_targets",
    "token_noise_targets_from_text",
]
