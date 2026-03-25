#!/usr/bin/env python3
"""
将 gliner_train.jsonl 扩增到 5000+ 条：多尺度滑窗 + 轻量前缀变体（无需 API）。
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Set

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data" / "gliner_train.jsonl"
OUT = ROOT / "data" / "gliner_train_expanded.jsonl"

# 中文+标签 prompt 下，字/词 token 数常接近字符数；须显著低于 512，避免训练时截断告警与丢实体
MAX_CHUNK_CHARS = 300

# 小语料时自动下调目标，避免无限循环
def _target_min(num_base: int) -> int:
    return min(5000, max(200, num_base * 150))


def load_jsonl(p: Path) -> List[Dict]:
    rows = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line.strip()))
    return rows


def save_jsonl(rows: List[Dict], p: Path) -> None:
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def slice_windows(item: Dict, win: int, stride: int) -> List[Dict]:
    win = min(win, MAX_CHUNK_CHARS)
    stride = min(stride, max(80, win - 40))
    text = item["tokenized_text"]
    ner = item.get("ner") or []
    if len(text) <= win:
        return [item]
    out: List[Dict] = []
    pos = 0
    while pos < len(text):
        end = min(pos + win, len(text))
        sub = text[pos:end]
        sub_ner: List[List] = []
        for ent in ner:
            if len(ent) < 3:
                continue
            s, e, lab = int(ent[0]), int(ent[1]), ent[2]
            if s >= pos and e <= end:
                sub_ner.append([s - pos, e - pos, lab])
        if sub_ner:
            out.append({"tokenized_text": sub, "ner": sub_ner})
        if end >= len(text):
            break
        pos += stride
    return out


def light_template_variants(item: Dict) -> List[Dict]:
    text = item["tokenized_text"]
    ner = item.get("ner") or []
    if not ner or len(text) < 30:
        return []
    prefixes = ["【速览】", "【要点】", "深度 | ", "观点 | "]
    outs = []
    for pf in prefixes:
        new_text = pf + text
        delta = len(pf)
        new_ner = [[s + delta, e + delta, lab] for s, e, lab in ner]
        outs.append({"tokenized_text": new_text, "ner": new_ner})
    return outs


def main():
    if not SRC.exists():
        raise SystemExit(f"缺少 {SRC}，请先运行 scripts/prepare_training_data.py")

    base = load_jsonl(SRC)
    random.seed(42)

    out_list: List[Dict] = []
    seen: Set[str] = set()

    def add_item(r: Dict) -> None:
        k = json.dumps(r, ensure_ascii=True)
        if k in seen:
            return
        seen.add(k)
        out_list.append(r)

    for it in base:
        # 长文先切成 ≤MAX_CHUNK_CHARS，避免单条 token 数 > max_len
        if len(it["tokenized_text"]) <= MAX_CHUNK_CHARS:
            add_item(it)
        else:
            for piece in slice_windows(it, win=MAX_CHUNK_CHARS, stride=200):
                add_item(piece)
        for w, st in [(300, 220), (280, 200), (260, 180), (300, 240)]:
            for piece in slice_windows(it, win=w, stride=st):
                add_item(piece)
        for v in light_template_variants(it):
            if len(v["tokenized_text"]) <= MAX_CHUNK_CHARS + 8:
                add_item(v)

    target = _target_min(len(base))
    attempts = 0
    while len(out_list) < target and attempts < 20000:
        attempts += 1
        it = random.choice(base)
        w = random.randint(200, MAX_CHUNK_CHARS)
        st = random.randint(120, max(120, w - 80))
        for piece in slice_windows(it, win=w, stride=st):
            add_item(piece)

    random.shuffle(out_list)
    save_jsonl(out_list, OUT)
    print(f"写入 {OUT} ，共 {len(out_list)} 条（目标 ≥ {target}）")


if __name__ == "__main__":
    main()
