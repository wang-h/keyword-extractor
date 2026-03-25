#!/usr/bin/env python3
"""
评估 GLiNER 微调 checkpoint：全量测试集、文本级匹配 + 部分匹配 F1。
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import torch
from gliner import GLiNER

from keyword_extractor.labels import GLINER_TRAINING_LABELS  # noqa: E402


def load_test_data(test_file: str) -> List[Dict]:
    data = []
    with open(test_file, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def _normalize_mention(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[\s\-_]+", "", s)
    return s


def _core_text(s: str) -> str:
    """去空白用于包含判断。"""
    return re.sub(r"\s+", "", s.strip().lower())


def mention_match(pred: str, gold: str) -> bool:
    """
    预测与金标是否算匹配：
    - 归一化后相等，或
    - 一方是另一方的子串（处理「香港中文大学与香港大学…」vs「香港中文大学」）
    """
    p, g = pred.strip(), gold.strip()
    if not p or not g:
        return False
    if p.lower() == g.lower():
        return True
    np, ng = _normalize_mention(p), _normalize_mention(g)
    if np and ng and (np == ng):
        return True
    cp, cg = _core_text(p), _core_text(g)
    if len(cp) >= 2 and len(cg) >= 2:
        if cp in cg or cg in cp:
            return True
    return False


def gt_mentions(item: Dict) -> Set[str]:
    text = item.get("tokenized_text") or item.get("text", "")
    mentions: Set[str] = set()
    for ent in item.get("ner", []):
        if len(ent) >= 3:
            s, e = int(ent[0]), int(ent[1])
            if 0 <= s < e <= len(text):
                mentions.add(text[s:e])
    for e in item.get("entities", []):
        if isinstance(e, dict) and e.get("text"):
            mentions.add(e["text"])
    return mentions


def pairwise_match(preds: Set[str], golds: Set[str]) -> Tuple[int, int, int]:
    """返回 (tp, len(preds), len(golds))，一对一贪心匹配。"""
    gold_left = list(golds)
    tp = 0
    used_gold: Set[int] = set()
    for p in preds:
        for gi, g in enumerate(gold_left):
            if gi in used_gold:
                continue
            if mention_match(p, g):
                tp += 1
                used_gold.add(gi)
                break
    return tp, len(preds), len(golds)


def evaluate_model(
    model_path: str,
    test_data: List[Dict],
    device: str = "cuda",
    labels: List[str] | None = None,
    threshold: float = 0.35,
) -> Dict[str, float]:
    labels = labels or GLINER_TRAINING_LABELS
    print(f"加载模型: {model_path}")
    model = GLiNER.from_pretrained(model_path)
    model.to(device)

    total_tp = total_pred = total_gt = 0
    per_doc = []

    print(f"\n评估 {len(test_data)} 条（全量）, labels={len(labels)}, threshold={threshold}")

    for i, item in enumerate(test_data):
        text = item.get("tokenized_text") or item.get("text", "")
        golds = gt_mentions(item)

        try:
            raw = model.predict_entities(text, labels=labels, threshold=threshold)
            preds = {p["text"].strip() for p in raw if p.get("text", "").strip()}
        except Exception as e:
            print(f"样本 {i} 预测失败: {e}")
            continue

        tp, npred, ngt = pairwise_match(preds, golds)
        total_tp += tp
        total_pred += npred
        total_gt += ngt

        p = tp / npred if npred else 0.0
        r = tp / ngt if ngt else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        per_doc.append({"p": p, "r": r, "f1": f1, "tp": tp, "pred": npred, "gt": ngt})

        if i < 3:
            print(f"\n--- 样例 {i + 1} ---")
            print(f"text: {text[:80]}...")
            print(f"gold: {golds}")
            print(f"pred: {preds}")

    precision = total_tp / total_pred if total_pred else 0.0
    recall = total_tp / total_gt if total_gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    micro = {"precision": precision, "recall": recall, "f1": f1}
    if per_doc:
        micro["macro_f1"] = sum(d["f1"] for d in per_doc) / len(per_doc)
    else:
        micro["macro_f1"] = 0.0

    print(f"\n{'=' * 50}")
    print("微平均（所有预测/金标）:")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall:    {recall:.3f}")
    print(f"  F1:        {f1:.3f}")
    print(f"宏平均 F1:   {micro['macro_f1']:.3f}")
    print(f"{'=' * 50}")

    return micro


if __name__ == "__main__":
    model_path = sys.argv[1] if len(sys.argv) > 1 else "./models/gliner_sft/checkpoint-250"
    test_file = ROOT / "data" / "gliner_test.jsonl"
    test_data = load_test_data(str(test_file))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    evaluate_model(model_path, test_data, device=dev)
