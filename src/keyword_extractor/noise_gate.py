"""
预处理噪声门控（ND-GLiNER 信息熵门控的轻量替代）

在送入 GLiNER 之前，按段落/行打分，丢弃低信息密度片段（CSS 残渣、导航语、纯符号等），
降低有效序列长度与注意力噪声，无需修改 Transformer 内部结构。

若需「不删 token、在注意力里软屏蔽噪声」的架构参考（SDPA 加性掩码），见
``keyword_extractor.soft_masking_reference``；接入 GLiNER 本体需 fork 并改 backbone forward。
"""
from __future__ import annotations

import math
import re
from typing import List, Optional, Tuple

# 明显非正文的短语（微信公众号常见）
_BOILERPLATE_PATTERNS = [
    re.compile(r"点击.{0,8}阅读原文", re.I),
    re.compile(r"关注.{0,6}公众号", re.I),
    re.compile(r"长按.{0,6}识别", re.I),
    re.compile(r"赞赏|在看|转发|点赞", re.I),
    re.compile(r"font-family|padding\s*:|margin\s*:", re.I),
    re.compile(r"rgb\s*\(", re.I),
    re.compile(r"#[0-9a-fA-F]{3,8}\b"),
]

# 行内若英文小写 slug 占比过高且中文极少 → 疑似样式残渣
_SLUG_TOKEN = re.compile(r"\b[a-z]{2,}(?:-[a-z]{2,}){2,}\b")


def _char_entropy(s: str) -> float:
    """字节/字符级香农熵（归一化到 0~1 左右）。"""
    if not s:
        return 0.0
    freq: dict[str, int] = {}
    for c in s:
        freq[c] = freq.get(c, 0) + 1
    n = len(s)
    h = 0.0
    for c in freq.values():
        p = c / n
        h -= p * math.log2(p)
    # 中文常用字熵上限约 10+；缩放到粗略 0~1
    return min(h / 8.0, 1.0)


def _cjk_ratio(s: str) -> float:
    if not s:
        return 0.0
    cjk = sum(1 for c in s if "\u4e00" <= c <= "\u9fff")
    return cjk / max(len(s), 1)


def _alpha_ratio(s: str) -> float:
    if not s:
        return 0.0
    a = sum(1 for c in s if ("a" <= c <= "z") or ("A" <= c <= "Z"))
    return a / max(len(s), 1)


def segment_paragraphs(text: str) -> List[str]:
    """按空行 / 单换行切分为候选段落。"""
    if not text.strip():
        return []
    blocks = re.split(r"\n\s*\n+", text.strip())
    out: List[str] = []
    for b in blocks:
        b = b.strip()
        if not b:
            continue
        if len(b) < 500:
            out.append(b)
        else:
            for line in b.split("\n"):
                line = line.strip()
                if line:
                    out.append(line)
    return out if out else [text.strip()]


def score_segment(seg: str) -> float:
    """
    分数越高越可能保留。启发式：中文比例、熵、反 boilerplate、反 slug。
    """
    s = seg.strip()
    if len(s) < 8:
        return 0.0
    for pat in _BOILERPLATE_PATTERNS:
        if pat.search(s):
            return 0.05

    cjk = _cjk_ratio(s)
    ent = _char_entropy(s)
    slug_hits = len(_SLUG_TOKEN.findall(s))
    slug_pen = min(0.4, slug_hits * 0.12)

    # 几乎无中文且充满连字符英文 → 强惩罚（技术文可保留：提高 cjk=0 时仍给基分）
    ar = _alpha_ratio(s)
    base = 0.25 * cjk + 0.35 * ent + 0.15 * min(ar, 0.9)
    if cjk < 0.08 and slug_hits >= 2:
        base *= 0.3
    return max(0.0, min(1.0, base - slug_pen + 0.15))


def filter_text_by_noise_gate(
    text: str,
    threshold: float = 0.18,
    max_segments: Optional[int] = None,
) -> str:
    """
    按段落打分，丢弃低于阈值的片段，拼接剩余文本。

    Args:
        text: 已 HTML 清洗后的正文
        threshold: 保留段落的最低分数
        max_segments: 最多保留段数（按分数排序后取 top），None 表示不限制
    """
    segs = segment_paragraphs(text)
    scored: List[Tuple[float, str]] = [(score_segment(s), s) for s in segs]
    scored.sort(key=lambda x: -x[0])
    if max_segments is not None:
        scored = scored[: max(1, max_segments)]
    kept = [s for sc, s in scored if sc >= threshold]
    if not kept:
        # 避免删光：至少保留得分最高的一段
        best = max(scored, key=lambda x: x[0], default=(0.0, text[:2000]))
        kept = [best[1]]
    # 恢复大致阅读顺序：按在原 text 中出现顺序
    kept_set = set(kept)
    ordered = [s for s in segs if s in kept_set]
    return "\n\n".join(ordered)


def apply_noise_gate_if_enabled(
    text: str,
    enabled: bool = True,
    **kwargs,
) -> str:
    if not enabled or not text:
        return text
    return filter_text_by_noise_gate(text, **kwargs)
