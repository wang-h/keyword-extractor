#!/usr/bin/env python3
"""
从 WeRSS 服务拉取全部公众号文章（含完整 HTML 正文）。

默认导出 **JSONL**（每行一个 JSON，长 HTML 不用 CSV 转义，更稳）。
可选 `--format csv` 或 `--format both` 兼容旧流程。

API 文档: https://werss.deepling.tech/api/docs

说明：部分环境下「列表 + has_content=true + offset 翻页」第二页会为空，导致只落到 100 条。
默认策略改为：
  1) has_content=false 分页拉齐全部列表（轻量，翻页可靠）
  2) 再对每篇 GET /api/v1/wx/articles/{id}?content=true 拉正文

可选 --fast：仅一步列表且 has_content=true（快，但可能不全）。

同时导出大模型/系统打的标签：
  • 文章行内增加 tags_json（API 的 tags 对象列表 JSON）
  • 默认另存 data/article_tags.csv：article_id, tag_id, tag_name（供 prepare_training_data 使用）
  • 默认请求 GET /api/v1/wx/tags 分页，另存 data/tags.csv：id, name, type, …（与训练管线兼容）

认证: Authorization: Bearer <API Key>
环境变量: WEWSS_API_KEY
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

DEFAULT_BASE = "https://werss.deepling.tech"
LIST_PATH = "/api/v1/wx/articles"
TAGS_PATH = "/api/v1/wx/tags"
DETAIL_TMPL = "/api/v1/wx/articles/{article_id}"
MAX_LIMIT = 100


def _request_json(
    method: str,
    url: str,
    headers: Dict[str, str],
    *,
    body: Optional[bytes] = None,
    timeout: float = 120.0,
) -> Dict[str, Any]:
    try:
        import httpx

        r = httpx.request(method, url, headers=headers, content=body, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except ImportError:
        req = Request(url, headers=headers, method=method, data=body)
        with urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
        return json.loads(raw)


def _http_get_json(url: str, headers: Dict[str, str], timeout: float = 120.0) -> Dict[str, Any]:
    return _request_json("GET", url, headers, timeout=timeout)


def _ts_to_iso(ts: Any) -> str:
    if ts is None:
        return ""
    try:
        t = int(ts)
        if t > 1_000_000_000_000:
            t //= 1000
        return datetime.fromtimestamp(t, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    except (TypeError, ValueError, OSError):
        return str(ts)


def _list_url(
    base: str,
    offset: int,
    *,
    has_content: bool,
    mp_id: Optional[str],
    status: Optional[str],
    search: Optional[str],
) -> str:
    q: Dict[str, Any] = {
        "limit": MAX_LIMIT,
        "offset": offset,
        "has_content": "true" if has_content else "false",
    }
    if mp_id:
        q["mp_id"] = mp_id
    if status:
        q["status"] = status
    if search:
        q["search"] = search
    qs = urlencode(q)
    return f"{base.rstrip('/')}{LIST_PATH}?{qs}"


def fetch_article_list_page(
    base_url: str,
    api_key: str,
    offset: int,
    *,
    has_content: bool,
    mp_id: Optional[str],
    status: Optional[str],
    search: Optional[str],
) -> List[Dict[str, Any]]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }
    url = _list_url(base_url, offset, has_content=has_content, mp_id=mp_id, status=status, search=search)
    data = _http_get_json(url, headers)
    if data.get("code") != 0:
        raise RuntimeError(f"API 业务错误: {data}")
    return (data.get("data") or {}).get("list") or []


def fetch_all_list_metadata(
    base_url: str,
    api_key: str,
    *,
    mp_id: Optional[str],
    status: Optional[str],
    search: Optional[str],
    delay_s: float,
) -> List[Dict[str, Any]]:
    """has_content=false 分页，直到某一页为空。"""
    out: List[Dict[str, Any]] = []
    offset = 0
    while True:
        chunk = fetch_article_list_page(
            base_url,
            api_key,
            offset,
            has_content=False,
            mp_id=mp_id,
            status=status,
            search=search,
        )
        if not chunk:
            break
        out.extend(chunk)
        print(f"  列表 offset={offset} 本页 {len(chunk)} 篇，累计 {len(out)}", flush=True)
        offset += len(chunk)
        if len(chunk) < MAX_LIMIT:
            break
        if delay_s > 0:
            time.sleep(delay_s)
    return out


def fetch_article_detail(
    base_url: str,
    api_key: str,
    article_id: str,
) -> Dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }
    aid = quote(article_id, safe="")
    url = f"{base_url.rstrip('/')}{DETAIL_TMPL.format(article_id=aid)}?content=true"
    data = _http_get_json(url, headers, timeout=180.0)
    if data.get("code") != 0:
        raise RuntimeError(f"详情 API 错误 id={article_id}: {data}")
    return data.get("data") or {}


def merge_list_with_detail(
    summary: Dict[str, Any],
    detail: Dict[str, Any],
) -> Dict[str, Any]:
    """详情覆盖/补全 content；标签以详情为准，若详情无 tags 则保留列表里的标签。"""
    merged = dict(summary)
    keep_tags = summary.get("tags") or []
    keep_names = summary.get("tag_names") or []
    merged.update(detail)
    if not merged.get("tags"):
        merged["tags"] = keep_tags
    if not merged.get("tag_names"):
        merged["tag_names"] = keep_names
    return merged


def _normalize_tag_obj(t: Any) -> Optional[Dict[str, str]]:
    if isinstance(t, dict):
        tid = str(t.get("id") or "")
        name = (t.get("name") or t.get("tag_name") or "").strip()
        if name:
            return {"id": tid, "name": name}
    return None


def tags_from_article(a: Dict[str, Any]) -> List[Dict[str, str]]:
    raw = a.get("tags") or []
    out: List[Dict[str, str]] = []
    if isinstance(raw, list):
        for t in raw:
            o = _normalize_tag_obj(t)
            if o:
                out.append(o)
    if out:
        return out
    names = a.get("tag_names") or []
    if isinstance(names, list):
        for n in names:
            if isinstance(n, str) and n.strip():
                out.append({"id": "", "name": n.strip()})
    return out


def fetch_tags_master_page(
    base_url: str,
    api_key: str,
    offset: int,
    limit: int = MAX_LIMIT,
) -> List[Dict[str, Any]]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }
    qs = urlencode({"limit": limit, "offset": offset})
    url = f"{base_url.rstrip('/')}{TAGS_PATH}?{qs}"
    data = _http_get_json(url, headers)
    if data.get("code") != 0:
        raise RuntimeError(f"标签列表 API 错误: {data}")
    return (data.get("data") or {}).get("list") or []


def fetch_all_tag_definitions(
    base_url: str,
    api_key: str,
    delay_s: float,
) -> List[Dict[str, Any]]:
    """分页拉取系统内全部标签定义（含 type 等，便于写 tags.csv）。"""
    out: List[Dict[str, Any]] = []
    offset = 0
    while True:
        chunk = fetch_tags_master_page(base_url, api_key, offset, MAX_LIMIT)
        if not chunk:
            break
        out.extend(chunk)
        offset += len(chunk)
        if len(chunk) < MAX_LIMIT:
            break
        if delay_s > 0:
            time.sleep(delay_s)
    return out


def fetch_all_with_content_in_list(
    base_url: str,
    api_key: str,
    *,
    mp_id: Optional[str],
    status: Optional[str],
    search: Optional[str],
    delay_s: float,
) -> List[Dict[str, Any]]:
    """旧逻辑：列表直接带正文（部分部署翻页不全）。"""
    out: List[Dict[str, Any]] = []
    offset = 0
    while True:
        chunk = fetch_article_list_page(
            base_url,
            api_key,
            offset,
            has_content=True,
            mp_id=mp_id,
            status=status,
            search=search,
        )
        if not chunk:
            break
        out.extend(chunk)
        print(f"  offset={offset} +{len(chunk)} 累计 {len(out)}", flush=True)
        offset += len(chunk)
        if len(chunk) < MAX_LIMIT:
            break
        if delay_s > 0:
            time.sleep(delay_s)
    return out


def article_to_record(a: Dict[str, Any]) -> Dict[str, Any]:
    """JSONL 用：原生 list/dict，不再把 tags 压成字符串。"""
    tag_objs = tags_from_article(a)
    return {
        "id": str(a.get("id") or ""),
        "title": (a.get("title") or "").replace("\r\n", "\n").replace("\r", "\n"),
        "description": (a.get("description") or "").replace("\r\n", "\n").replace("\r", "\n"),
        "content": a.get("content") or "",
        "publish_time": _ts_to_iso(a.get("publish_time")),
        "tags": tag_objs,
        "tag_names": [t["name"] for t in tag_objs],
        "mp_id": a.get("mp_id"),
        "url": a.get("url"),
    }


def row_from_article_csv(a: Dict[str, Any]) -> Dict[str, str]:
    """CSV 用：tags 仍序列化为字符串。"""
    rec = article_to_record(a)
    tags = rec.pop("tags")
    rec.pop("tag_names", None)
    rec.pop("mp_id", None)
    rec.pop("url", None)
    names = [t["name"] for t in tags]
    rec["tag_names"] = json.dumps(names, ensure_ascii=False)
    rec["tags_json"] = json.dumps(tags, ensure_ascii=False)
    # flatten for csv row typing
    return {
        "id": str(rec["id"]),
        "title": str(rec["title"]),
        "description": str(rec["description"]),
        "content": str(rec["content"]),
        "publish_time": str(rec["publish_time"]),
        "tag_names": rec["tag_names"],
        "tags_json": rec["tags_json"],
    }


def write_articles_jsonl(records: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def write_csv(rows: List[Dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id",
        "title",
        "description",
        "content",
        "publish_time",
        "tag_names",
        "tags_json",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def write_article_tags_csv(articles: List[Dict[str, Any]], path: Path) -> int:
    """article_id, tag_id, tag_name — 与 prepare_training_data 期望的 tag_name 一致。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["article_id", "tag_id", "tag_name"],
            quoting=csv.QUOTE_MINIMAL,
        )
        w.writeheader()
        for a in articles:
            aid = str(a.get("id") or "")
            if not aid:
                continue
            for t in tags_from_article(a):
                w.writerow(
                    {
                        "article_id": aid,
                        "tag_id": t.get("id") or "",
                        "tag_name": t["name"],
                    }
                )
                n += 1
    return n


def write_article_tags_jsonl(articles: List[Dict[str, Any]], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for a in articles:
            aid = str(a.get("id") or "")
            if not aid:
                continue
            for t in tags_from_article(a):
                f.write(
                    json.dumps(
                        {
                            "article_id": aid,
                            "tag_id": t.get("id") or "",
                            "tag_name": t["name"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                n += 1
    return n


def write_tags_master_csv(tag_rows: List[Dict[str, Any]], path: Path) -> None:
    """
    与 data/tags.csv 兼容：prepare_training_data 使用 name -> type。
    API 字段名不一致时做常见映射。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["id", "name", "type", "is_custom", "status", "mps_id", "created_at"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        w.writeheader()
        for r in tag_rows:
            tid = str(r.get("id") or "")
            name = (r.get("name") or r.get("tag_name") or "").strip()
            if not name:
                continue
            typ = (
                r.get("type")
                or r.get("tag_type")
                or r.get("category")
                or "unknown"
            )
            if isinstance(typ, str):
                typ_s = typ.strip() or "unknown"
            else:
                typ_s = str(typ)
            w.writerow(
                {
                    "id": tid,
                    "name": name,
                    "type": typ_s,
                    "is_custom": r.get("is_custom", "f"),
                    "status": str(r.get("status", "1")),
                    "mps_id": (
                        ""
                        if isinstance(r.get("mps_id"), list)
                        else str(r.get("mps_id", "") or "")
                    ),
                    "created_at": str(r.get("created_at", r.get("updated_at", ""))),
                }
            )


def _tag_row_to_jsonl_obj(r: Dict[str, Any]) -> Dict[str, Any]:
    tid = str(r.get("id") or "")
    name = (r.get("name") or r.get("tag_name") or "").strip()
    if not name:
        return {}
    typ = r.get("type") or r.get("tag_type") or r.get("category") or "unknown"
    typ_s = typ.strip() if isinstance(typ, str) else str(typ)
    mp = r.get("mps_id", "")
    if isinstance(mp, list):
        mp = ""
    return {
        "id": tid,
        "name": name,
        "type": typ_s or "unknown",
        "is_custom": r.get("is_custom", "f"),
        "status": str(r.get("status", "1")),
        "mps_id": str(mp or ""),
        "created_at": str(r.get("created_at", r.get("updated_at", ""))),
    }


def write_tags_master_jsonl(tag_rows: List[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in tag_rows:
            o = _tag_row_to_jsonl_obj(r)
            if o:
                f.write(json.dumps(o, ensure_ascii=False) + "\n")


def _fetch_one_detail(
    args: Tuple[str, str, str, Dict[str, Any]],
) -> Tuple[str, Dict[str, Any], Optional[str]]:
    base, key, aid, summary = args
    try:
        d = fetch_article_detail(base, key, aid)
        return aid, merge_list_with_detail(summary, d), None
    except Exception as e:
        return aid, summary, str(e)


def main() -> None:
    ap = argparse.ArgumentParser(description="从 WeRSS API 下载全部文章（默认 JSONL）")
    ap.add_argument(
        "-o",
        "--output",
        default="data/articles_full.jsonl",
        help="文章主文件路径（默认 data/articles_full.jsonl；CSV 模式请改 .csv）",
    )
    ap.add_argument(
        "--format",
        choices=("jsonl", "csv", "both"),
        default="jsonl",
        help="jsonl=仅 JSONL；csv=仅 CSV；both=同 stem 各写一份",
    )
    ap.add_argument("--base-url", default=os.environ.get("WEWSS_BASE_URL", DEFAULT_BASE))
    ap.add_argument("--mp-id", default=os.environ.get("WEWSS_MP_ID"))
    ap.add_argument("--status")
    ap.add_argument("--search")
    ap.add_argument("--delay", type=float, default=0.12, help="列表分页间隔（秒）")
    ap.add_argument(
        "--workers",
        type=int,
        default=4,
        help="拉取正文时的并发数（默认 4，过大可能触发限流）",
    )
    ap.add_argument(
        "--fast",
        action="store_true",
        help="仅使用 has_content=true 列表（快，若只得到 ~100 条请去掉此参数）",
    )
    ap.add_argument(
        "--api-key",
        default=os.environ.get("WEWSS_API_KEY") or os.environ.get("WERSS_API_KEY"),
    )
    ap.add_argument(
        "--article-tags-out",
        default="",
        help="文章-标签关系输出路径；留空则按格式自动为 .jsonl / .csv",
    )
    ap.add_argument(
        "--tags-out",
        default="",
        help="标签主表输出路径；留空则自动为 data/tags.jsonl 或 .csv",
    )
    ap.add_argument(
        "--skip-tags-master",
        action="store_true",
        help="不请求全局标签列表，仅根据文章内嵌 tags 写 article_tags（不写 tags.csv）",
    )
    args = ap.parse_args()

    if not args.api_key:
        print("请设置 WEWSS_API_KEY 或使用 --api-key", file=sys.stderr)
        sys.exit(1)

    key = args.api_key.strip()
    base = args.base_url.rstrip("/")

    if args.fast:
        print(f"模式: fast（列表含正文） limit={MAX_LIMIT}")
        articles = fetch_all_with_content_in_list(
            base, key, mp_id=args.mp_id, status=args.status, search=args.search, delay_s=args.delay
        )
    else:
        print("模式: 全量（先列表元数据，再逐篇拉 content）")
        meta_list = fetch_all_list_metadata(
            base, key, mp_id=args.mp_id, status=args.status, search=args.search, delay_s=args.delay
        )
        if not meta_list:
            print("列表为空，退出", file=sys.stderr)
            sys.exit(1)

        by_id = {str(m.get("id")): m for m in meta_list if m.get("id")}
        ids = list(by_id.keys())
        print(f"共 {len(ids)} 篇，开始拉正文（workers={args.workers}）…")

        errors: List[str] = []
        work = [(base, key, aid, by_id[aid]) for aid in ids]
        id_to_merged: Dict[str, Dict[str, Any]] = {}

        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
            futs = {ex.submit(_fetch_one_detail, w): w[2] for w in work}
            done = 0
            for fut in as_completed(futs):
                aid, merged, err = fut.result()
                done += 1
                id_to_merged[aid] = merged
                if err:
                    errors.append(f"{aid}: {err}")
                if done % 50 == 0 or done == len(ids):
                    print(f"  正文 {done}/{len(ids)}", flush=True)

        articles = [id_to_merged[i] for i in ids]

        if errors:
            print(f"警告: {len(errors)} 篇详情失败（已保留列表字段，content 可能为空）", file=sys.stderr)
            for e in errors[:15]:
                print(f"  {e}", file=sys.stderr)
            if len(errors) > 15:
                print(f"  … 另有 {len(errors) - 15} 条", file=sys.stderr)

    print(f"完成，共 {len(articles)} 篇")
    out = Path(args.output)
    fmt = args.format
    stem = out.with_suffix("")
    suffix = out.suffix.lower()

    if fmt == "csv" and suffix not in (".csv",):
        out = stem.with_suffix(".csv")
    if fmt == "jsonl" and suffix not in (".jsonl", ".json"):
        out = stem.with_suffix(".jsonl")

    records = [article_to_record(a) for a in articles]

    if fmt in ("jsonl", "both"):
        jsonl_path = out if fmt == "jsonl" else stem.with_suffix(".jsonl")
        write_articles_jsonl(records, jsonl_path)
        print(f"已写入 JSONL: {jsonl_path.resolve()}")

    if fmt in ("csv", "both"):
        csv_path = out if fmt == "csv" else stem.with_suffix(".csv")
        write_csv([row_from_article_csv(a) for a in articles], csv_path)
        print(f"已写入 CSV: {csv_path.resolve()}")

    use_jsonl_sidecars = fmt in ("jsonl", "both")
    at_default = (
        Path("data/article_tags.jsonl")
        if use_jsonl_sidecars
        else Path("data/article_tags.csv")
    )
    at_path = Path(args.article_tags_out) if args.article_tags_out else at_default
    if use_jsonl_sidecars:
        n_links = write_article_tags_jsonl(articles, at_path)
    else:
        n_links = write_article_tags_csv(articles, at_path)
    print(f"已写入文章-标签关系: {at_path.resolve()} （{n_links} 行）")

    if not args.skip_tags_master:
        try:
            print("拉取全局标签定义 /api/v1/wx/tags …")
            tag_defs = fetch_all_tag_definitions(base, key, delay_s=args.delay)
            tags_default = (
                Path("data/tags.jsonl") if use_jsonl_sidecars else Path("data/tags.csv")
            )
            tags_path = Path(args.tags_out) if args.tags_out else tags_default
            if tags_path.suffix.lower() == ".jsonl":
                write_tags_master_jsonl(tag_defs, tags_path)
            else:
                write_tags_master_csv(tag_defs, tags_path)
            print(f"已写入标签主表: {tags_path.resolve()} （{len(tag_defs)} 条）")
        except Exception as e:
            print(
                f"警告: 无法拉取或写入全局标签表（可改用 --skip-tags-master）: {e}",
                file=sys.stderr,
            )


if __name__ == "__main__":
    try:
        main()
    except HTTPError as e:
        print(f"HTTP 错误: {e}", file=sys.stderr)
        sys.exit(1)
    except URLError as e:
        print(f"网络错误: {e}", file=sys.stderr)
        sys.exit(1)
