#!/usr/bin/env python3
"""
可选：用 OpenAI 兼容 API 从正文生成 GLiNER 格式 JSONL（实体 + 多类型标签）。

环境变量：
  OPENAI_API_KEY   必填（除非仅运行 --dry-run）
  OPENAI_BASE_URL  可选，默认 https://api.openai.com/v1

无 API Key 时：打印提示并退出码 0，便于 CI 跳过。
"""
from __future__ import annotations

import csv
import json
import os
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from keyword_extractor.html_cleaner import clean_wechat_article  # noqa: E402

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore


SYSTEM_PROMPT = textwrap.dedent(
    """\
    你是中文科技编辑。从给定正文中抽取「可打标签的关键词/实体」。
    输出唯一合法 JSON 数组，每项: {"text": "原文子串", "type": "以下之一"}
    type 取值: company | product | ai_model | technology | hardware | person
    只输出正文中连续出现的子串，不要编造；最多 12 条；不要解释。
    """
)


def _articles_rows(limit: int = 80) -> List[Dict[str, str]]:
    p = ROOT / "data" / "articles_full.csv"
    rows = []
    with open(p, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for i, row in enumerate(r):
            if i >= limit:
                break
            rows.append(row)
    return rows


def _map_type(t: str) -> str:
    from keyword_extractor.labels import map_tag_type_to_label

    return map_tag_type_to_label(t)


def call_llm(body: str, model: str) -> str:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    if not api_key:
        raise RuntimeError("未设置 OPENAI_API_KEY")
    if httpx is None:
        raise RuntimeError("请 pip install httpx")

    url = f"{base}/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"正文：\n{body[:6000]}"},
        ],
        "temperature": 0.2,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    with httpx.Client(timeout=120.0) as client:
        r = client.post(url, json=payload, headers=headers)
        r.raise_for_status()
        data = r.json()
    return data["choices"][0]["message"]["content"]


def parse_json_array(raw: str) -> List[Dict[str, Any]]:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:].lstrip()
    return json.loads(raw)


def row_to_gliner_record(row: Dict[str, str], llm_out: List[Dict[str, Any]]) -> Dict:
    title = (row.get("title") or "").strip()
    content = row.get("content") or ""
    if "<" in content and ">" in content:
        content = clean_wechat_article(content)
    text = f"{title}\n\n{content}" if title else content
    text = text[:12000]

    ner = []
    for item in llm_out:
        ent = (item.get("text") or "").strip()
        typ = (item.get("type") or "technology").lower()
        if not ent:
            continue
        label = _map_type(typ)
        lo = text.lower()
        idx = lo.find(ent.lower())
        if idx < 0:
            continue
        ner.append([idx, idx + len(ent), label])

    return {"tokenized_text": text, "ner": ner}


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "data" / "gliner_train_llm.jsonl"))
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.dry_run or not os.environ.get("OPENAI_API_KEY"):
        print("跳过 LLM 生成：设置 OPENAI_API_KEY 后运行本脚本。")
        print("数据扩增可使用: python scripts/expand_training_data.py")
        return

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with open(out_path, "w", encoding="utf-8") as fout:
        for row in _articles_rows(limit=args.limit):
            try:
                raw = call_llm(row.get("content", "") or row.get("title", ""), args.model)
                arr = parse_json_array(raw)
                rec = row_to_gliner_record(row, arr)
                if rec["ner"]:
                    fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n += 1
            except Exception as e:
                print(f"skip article: {e}")

    print(f"写入 {n} 条 -> {out_path}")


if __name__ == "__main__":
    main()
