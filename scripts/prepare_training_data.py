"""
训练数据生成器 - 从标注数据创建 GLiNER 微调训练集

流程：先对 HTML 正文做 clean_wechat_article，再在**纯文本**上标注实体偏移，
避免在 HTML 源码上 find 导致 span 错误。

输出 GLiNER 标准格式（每行 JSON）：
{
  "tokenized_text": "...",
  "ner": [[start, end, "科技公司全称"], ...]
}
"""
from __future__ import annotations

import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from keyword_extractor.html_cleaner import clean_wechat_article  # noqa: E402
from keyword_extractor.labels import map_tag_type_to_label  # noqa: E402

# 会与 articles_full.* 按 id 合并（列表中靠后的文件覆盖同 id）
EXTRA_ARTICLES_JSONL = "articles_export_extra.jsonl"
EXTRA_ARTICLES_CSV = "articles_export_extra.csv"


def _tags_from_article_json_obj(
    o: Dict,
    tag_types: Dict[str, str],
) -> List[Dict]:
    """从 WeRSS 导出的一行 JSON 里解析标签。"""
    out: List[Dict] = []
    seen: Set[str] = set()
    for t in o.get("tags") or []:
        if isinstance(t, dict) and (t.get("name") or "").strip():
            nm = t["name"].strip()
            if nm not in seen:
                seen.add(nm)
                out.append({"name": nm, "type": tag_types.get(nm, "unknown")})
    for n in o.get("tag_names") or []:
        if isinstance(n, str) and (s := n.strip()) and s not in seen:
            seen.add(s)
            out.append({"name": s, "type": tag_types.get(s, "unknown")})
    return out


def _load_tag_types(data_dir: Path) -> Dict[str, str]:
    tag_types: Dict[str, str] = {}
    tj = data_dir / "tags.jsonl"
    tc = data_dir / "tags.csv"
    if tj.exists():
        with open(tj, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                name = (row.get("name") or "").strip()
                if name:
                    tag_types[name] = row.get("type", "unknown")
    elif tc.exists():
        with open(tc, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                tag_types[row["name"]] = row.get("type", "unknown")
    return tag_types


def _load_article_tags_map(data_dir: Path, tag_types: Dict[str, str]) -> Dict[str, List[Dict]]:
    article_tags: Dict[str, List[Dict]] = defaultdict(list)
    aj = data_dir / "article_tags.jsonl"
    ac = data_dir / "article_tags.csv"
    if aj.exists():
        with open(aj, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                article_id = str(row.get("article_id") or "")
                tag_name = (row.get("tag_name") or "").strip()
                if article_id and tag_name:
                    article_tags[article_id].append(
                        {"name": tag_name, "type": tag_types.get(tag_name, "unknown")}
                    )
    elif ac.exists():
        with open(ac, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                article_id = row["article_id"]
                tag_name = row["tag_name"]
                article_tags[article_id].append(
                    {"name": tag_name, "type": tag_types.get(tag_name, "unknown")}
                )
    return article_tags


def _load_articles_from_path(
    path: Path,
    articles: Dict[str, Dict],
    tag_types: Dict[str, str],
) -> None:
    if not path.exists():
        return
    suf = path.suffix.lower()
    if suf in (".jsonl", ".json"):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    o = json.loads(line)
                except json.JSONDecodeError:
                    continue
                aid = str(o.get("id") or "")
                if not aid:
                    continue
                raw = o.get("content") or ""
                limit = 12000 if len(raw) > 12000 else len(raw)
                emb = _tags_from_article_json_obj(o, tag_types)
                rec: Dict = {
                    "title": (o.get("title") or "").strip(),
                    "content": raw[:limit],
                }
                if emb:
                    rec["tags"] = emb
                articles[aid] = rec
        return

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("id"):
                continue
            raw = row.get("content") or ""
            limit = 12000 if len(raw) > 12000 else len(raw)
            articles[row["id"]] = {
                "title": (row.get("title") or "").strip(),
                "content": raw[:limit],
            }


def _print_data_inventory(
    full_ids: Set[str],
    tagged_ids: Set[str],
    used_count: int,
) -> None:
    """说明为何训练集偏少：正文文件篇数 vs 标签关系篇数。"""
    overlap = full_ids & tagged_ids
    missing_body = tagged_ids - full_ids
    print("\n📊 数据量对照（排查「我记得有一千多篇」）")
    print(f"  • 正文体文件（jsonl/csv 合并后）中有正文的篇数: {len(full_ids)}")
    print(f"  • article_tags（jsonl/csv）里出现过的 article_id 数: {len(tagged_ids)}")
    print(f"  • 两者 id 能对上、可参与生成训练样本的: {len(overlap)}")
    if missing_body:
        print(f"  ⚠️  有标签但正文文件里没有该 id: {len(missing_body)} 篇")
        print("     → 导出 articles_full.jsonl（或 csv）并保证 id 与 article_tags 一致。")
    extra_j = ROOT / "data" / EXTRA_ARTICLES_JSONL
    extra_c = ROOT / "data" / EXTRA_ARTICLES_CSV
    if not extra_j.exists() and not extra_c.exists() and len(full_ids) < len(tagged_ids):
        print(f"  💡 可选：将大批量正文放到 data/{EXTRA_ARTICLES_JSONL} 或 {EXTRA_ARTICLES_CSV}，")
        print("     会先读 extra 再读 articles_full（后者覆盖同 id）。")
    print(f"  • 本次实际写出 GLiNER 样本（正文中能匹配到实体）: {used_count} 条\n")


def load_annotated_data():
    """
    加载标注数据。

    正文来源（按顺序加载，后者覆盖同 id）：
      data/articles_export_extra.jsonl / .csv
      data/articles_full.jsonl / .csv
    优先使用 .jsonl（适合大 HTML）。

    标签：tags.jsonl 或 tags.csv；article_tags.jsonl 或 article_tags.csv。
    若文章 JSON 行内自带 tags/tag_names，也会作为标签（可被 article_tags 表覆盖）。
    """
    data_dir = ROOT / "data"

    tag_types = _load_tag_types(data_dir)
    article_tags = _load_article_tags_map(data_dir, tag_types)

    articles: Dict[str, Dict] = {}
    for name in (
        EXTRA_ARTICLES_JSONL,
        EXTRA_ARTICLES_CSV,
        "articles_full.jsonl",
        "articles_full.csv",
    ):
        _load_articles_from_path(data_dir / name, articles, tag_types)

    for aid, tags in article_tags.items():
        if aid in articles:
            articles[aid]["tags"] = tags

    full_ids = set(articles.keys())
    tagged_ids = set(article_tags.keys()) | {k for k, v in articles.items() if v.get("tags")}
    merged = {k: v for k, v in articles.items() if "tags" in v}
    return merged, full_ids, tagged_ids


def _norm_span_to_text_span(text: str, needle: str) -> Optional[Tuple[int, int]]:
    """在 text 上按「去空白与 -_」归一化后查找 needle 的字符级 span。"""
    skips = set(" \n\t\r\-_")
    text_indices: List[int] = []
    norm_chars: List[str] = []
    for i, c in enumerate(text):
        if c in skips:
            continue
        norm_chars.append(c.lower())
        text_indices.append(i)
    nn = "".join(c.lower() for c in needle if c not in skips)
    if not nn:
        return None
    blob = "".join(norm_chars)
    s = blob.find(nn)
    if s < 0:
        return None
    e = s + len(nn) - 1
    return (text_indices[s], text_indices[e] + 1)


def _find_spans_literal(text: str, entity_name: str) -> List[Tuple[int, int]]:
    pat = re.compile(re.escape(entity_name), re.IGNORECASE)
    return [(m.start(), m.end()) for m in pat.finditer(text)]


def _find_spans_flexible_ws(text: str, entity_name: str) -> List[Tuple[int, int]]:
    parts = [p for p in entity_name.split() if p]
    if len(parts) < 2:
        return []
    pat_str = r"\s*".join(re.escape(p) for p in parts)
    pat = re.compile(pat_str, re.IGNORECASE)
    return [(m.start(), m.end()) for m in pat.finditer(text)]


def find_entity_positions(text: str, entities: List[Dict]) -> List[Dict]:
    """
    在纯文本中查找实体，返回带 start/end/label/text 的列表（去重 span）。
    """
    found: List[Dict] = []
    seen: Set[Tuple[int, int, str]] = set()

    for ent in entities:
        name = (ent.get("name") or "").strip()
        if not name:
            continue
        label = map_tag_type_to_label(ent["type"])

        spans: List[Tuple[int, int]] = []
        spans.extend(_find_spans_literal(text, name))
        if not spans:
            spans.extend(_find_spans_flexible_ws(text, name))
        if not spans:
            ns = _norm_span_to_text_span(text, name)
            if ns:
                spans.append(ns)

        # 每个后台标签只对应一处金标（首次出现），避免「OpenClaw」在全文匹配几十次
        if spans:
            start, end = spans[0]
            key = (start, end, label)
            if key not in seen:
                seen.add(key)
                surface = text[start:end]
                found.append(
                    {
                        "start": start,
                        "end": end,
                        "label": label,
                        "text": surface,
                    }
                )

    # 去掉被其它更长 span 完全包含的重复（同起点取最长）
    found.sort(key=lambda x: (x["start"], -(x["end"] - x["start"])))
    merged: List[Dict] = []
    for item in found:
        s, e = item["start"], item["end"]
        dominated = False
        for other in merged:
            if other["label"] != item["label"]:
                continue
            if other["start"] <= s and e <= other["end"] and (other["start"], other["end"]) != (s, e):
                dominated = True
                break
        if not dominated:
            merged.append(item)
    return merged


def _clean_article_field(raw: str, extra_preserve: Optional[Set[str]] = None) -> str:
    if not raw:
        return ""
    if "<" in raw and ">" in raw:
        return clean_wechat_article(raw, method="auto", preserve_keywords=extra_preserve)
    return raw.strip()


def example_to_gliner_record(text: str, entities: List[Dict]) -> Dict:
    ner = [[e["start"], e["end"], e["label"]] for e in entities]
    return {"tokenized_text": text, "ner": ner}


def create_gliner_training_data():
    print("=" * 70)
    print("生成 GLiNER 训练数据（HTML 先清洗 + 多类型标签）")
    print("=" * 70)

    articles, full_ids, tagged_ids = load_annotated_data()
    print(f"可训练（有正文 + 有标签）: {len(articles)} 篇")

    training_examples: List[Dict] = []

    for aid, article in articles.items():
        title_raw = article["title"]
        content_raw = article["content"]
        preserve = {t["name"] for t in article.get("tags", []) if t.get("name")}

        title = _clean_article_field(title_raw, preserve)
        body = _clean_article_field(content_raw, preserve)
        text = f"{title}\n\n{body}" if title else body

        entities = find_entity_positions(text, article["tags"])

        if entities:
            training_examples.append(example_to_gliner_record(text, entities))
            print(f"\n{article['title'][:42]}...")
            print(f"  实体 {len(entities)}/{len(article['tags'])}")
            for e in entities[:3]:
                print(f"  - {e['text']!r} ({e['label']})")

    split_idx = int(len(training_examples) * 0.8)
    train_data = training_examples[:split_idx]
    test_data = training_examples[split_idx:]

    data_dir = ROOT / "data"
    with open(data_dir / "gliner_train.jsonl", "w", encoding="utf-8") as f:
        for ex in train_data:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    with open(data_dir / "gliner_test.jsonl", "w", encoding="utf-8") as f:
        for ex in test_data:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print("\n" + "=" * 70)
    print("统计")
    print("=" * 70)
    print(f"  训练: {len(train_data)} 条 | 测试: {len(test_data)} 条")

    label_counts: Dict[str, int] = defaultdict(int)
    for ex in training_examples:
        for ent in ex["ner"]:
            label_counts[ent[2]] += 1

    print("  标签分布:")
    for lb, c in sorted(label_counts.items(), key=lambda x: -x[1]):
        print(f"    {lb}: {c}")

    print("\n已写入 data/gliner_train.jsonl , data/gliner_test.jsonl")

    _print_data_inventory(full_ids, tagged_ids, len(training_examples))


if __name__ == "__main__":
    create_gliner_training_data()
