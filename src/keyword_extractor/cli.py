"""Command-line interface for keyword-extractor."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from . import __version__
from .gliner_memory import GLiNEREntityExtractor

app = typer.Typer(help="Top-K Gated GLiNER V3 关键词提取 CLI")


def _read_text(text: Optional[str], file_path: Optional[Path]) -> str:
    if text and file_path:
        raise typer.BadParameter("请二选一：传入 text 或 --file")
    if file_path:
        return file_path.read_text(encoding="utf-8")
    if text:
        return text
    raise typer.BadParameter("请提供文本参数，或使用 --file 指定输入文件")


@app.command("extract")
def extract_cmd(
    text: Optional[str] = typer.Argument(None, help="待提取文本"),
    file: Optional[Path] = typer.Option(None, "--file", "-f", help="输入文件路径"),
    model: str = typer.Option("urchade/gliner_multi-v2.1", "--model", "-m", help="GLiNER 模型名或本地路径"),
    top_k: int = typer.Option(10, "--top-k", "-k", min=1, max=100, help="返回关键词数量"),
    threshold: float = typer.Option(0.3, "--threshold", min=0.0, max=1.0, help="基础阈值"),
    no_noise_gate: bool = typer.Option(False, "--no-noise-gate", help="关闭噪声门控"),
    topk_gate: bool = typer.Option(False, "--topk-gate", help="启用 Top-K 物理压缩"),
    topk_keep_k: int = typer.Option(1500, "--topk-keep-k", min=128, help="Top-K 保留 token 数"),
    json_output: bool = typer.Option(False, "--json", help="输出 JSON"),
) -> None:
    content = _read_text(text, file)

    extractor = GLiNEREntityExtractor(
        model_name=model,
        threshold=threshold,
        use_noise_gate=not no_noise_gate,
        use_topk_gate=topk_gate,
        topk_keep_k=topk_keep_k,
    )
    result = extractor.extract(content, top_k=top_k, return_metadata=True)

    if json_output:
        typer.echo(result.model_dump_json(indent=2, exclude_none=True))
        return

    typer.echo(f"method: {result.method}")
    typer.echo(f"model: {result.model}")
    typer.echo(f"elapsed: {result.elapsed_time:.3f}s")
    typer.echo("keywords:")
    for idx, kw in enumerate(result.keywords, start=1):
        typer.echo(f"{idx}. {kw.keyword} | score={kw.score:.4f}")


@app.command("version")
def version_cmd() -> None:
    typer.echo(__version__)


@app.command("health")
def health_cmd(json_output: bool = typer.Option(False, "--json")) -> None:
    payload = {
        "package": "keyword-extractor",
        "version": __version__,
        "ready": True,
    }
    if json_output:
        typer.echo(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        typer.echo(f"keyword-extractor {__version__} ready")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
