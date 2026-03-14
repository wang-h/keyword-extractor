"""命令行工具"""
import sys
import json
from pathlib import Path
from typing import Optional, List

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from .extractor import KeywordExtractor
from .models import ExtractorConfig, PRESET_MODELS


app = typer.Typer(
    name="kwextract",
    help="中文关键词提取工具",
    no_args_is_help=True,
)
console = Console()


def version_callback(value: bool):
    if value:
        from . import __version__
        console.print(f"[bold blue]keyword-extractor[/] version [green]{__version__}[/]")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False, "--version", "-v", callback=version_callback, is_eager=True
    ),
):
    """中文关键词提取工具 - 基于 KeyBERT"""
    pass


@app.command(name="extract")
def extract_command(
    text: Optional[str] = typer.Argument(None, help="要提取关键词的文本"),
    file: Optional[Path] = typer.Option(
        None, "--file", "-f", help="从文件读取文本"
    ),
    top_k: int = typer.Option(5, "--top-k", "-k", help="提取关键词数量"),
    model: str = typer.Option(
        "text2vec", "--model", "-m", 
        help="模型名称",
        autocompletion=lambda: list(PRESET_MODELS.keys())
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="输出文件路径"
    ),
    json_output: bool = typer.Option(
        False, "--json", help="以 JSON 格式输出"
    ),
):
    """从文本中提取关键词"""
    
    # 获取输入文本
    if file:
        if not file.exists():
            console.print(f"[red]错误: 文件不存在 {file}[/]")
            raise typer.Exit(1)
        text = file.read_text(encoding="utf-8")
    elif not text:
        # 从 stdin 读取
        if sys.stdin.isatty():
            console.print("[yellow]请输入文本 (Ctrl+D 结束):[/]")
        text = sys.stdin.read()
    
    if not text or not text.strip():
        console.print("[red]错误: 文本不能为空[/]")
        raise typer.Exit(1)
    
    # 初始化提取器
    config = ExtractorConfig(
        model_name=model,
        top_k=top_k
    )
    extractor = KeywordExtractor(config)
    
    # 提取关键词
    with console.status("[bold green]正在提取关键词..."):
        result = extractor.extract(text, top_k=top_k)
    
    # 输出结果
    if json_output:
        output_data = {
            "text": result.text,
            "method": result.method,
            "elapsed_time": result.elapsed_time,
            "model": result.model,
            "keywords": [
                {"keyword": k.keyword, "score": k.score}
                for k in result.keywords
            ]
        }
        output_str = json.dumps(output_data, ensure_ascii=False, indent=2)
    else:
        # 表格输出
        table = Table(
            title=f"提取结果 (耗时: {result.elapsed_time:.3f}s)",
            box=box.ROUNDED
        )
        table.add_column("排名", style="cyan", justify="center")
        table.add_column("关键词", style="green")
        table.add_column("分数", style="yellow", justify="right")
        
        for i, kw in enumerate(result.keywords, 1):
            table.add_row(str(i), kw.keyword, f"{kw.score:.4f}")
        
        output_str = table
    
    if output:
        if json_output:
            output.write_text(output_str, encoding="utf-8")
        else:
            # 保存为文本格式
            lines = [f"{i+1}. {kw.keyword} (score: {kw.score:.4f})" 
                    for i, kw in enumerate(result.keywords)]
            output.write_text("\n".join(lines), encoding="utf-8")
        console.print(f"[green]结果已保存到: {output}[/]")
    else:
        console.print(output_str)


@app.command(name="models")
def list_models_command():
    """列出可用的模型"""
    table = Table(title="可用模型列表", box=box.ROUNDED)
    table.add_column("名称", style="cyan")
    table.add_column("描述", style="green")
    table.add_column("大小", style="yellow")
    table.add_column("中文优化", justify="center")
    table.add_column("推荐", justify="center")
    
    for key, info in PRESET_MODELS.items():
        table.add_row(
            key,
            info.description,
            info.size,
            "✓" if info.chinese_optimized else "",
            "★" if info.recommended else ""
        )
    
    console.print(table)
    console.print("\n[dim]使用示例: kwextract extract '文本' -m bge-m3[/]")


@app.command(name="compare")
def compare_command(
    text: str = typer.Argument(..., help="测试文本"),
    models: Optional[List[str]] = typer.Option(
        None, "--model", help="要对比的模型（可多次指定）"
    ),
    top_k: int = typer.Option(5, "--top-k", "-k", help="提取数量"),
):
    """对比不同模型的提取效果"""
    
    models = models or [k for k, v in PRESET_MODELS.items() if v.recommended]
    
    config = ExtractorConfig()
    extractor = KeywordExtractor(config)
    
    console.print(f"[bold]测试文本:[/] {text[:100]}...\n")
    
    with console.status("[bold green]正在对比模型..."):
        results = extractor.compare_models(text, models, top_k)
    
    # 显示结果
    for model_key, result in results.items():
        if result is None:
            console.print(Panel(
                f"[red]加载失败[/]",
                title=f"模型: {model_key}",
                border_style="red"
            ))
            continue
        
        keywords_str = "\n".join([
            f"  {i+1}. {kw.keyword} ({kw.score:.4f})"
            for i, kw in enumerate(result.keywords)
        ])
        
        console.print(Panel(
            f"[dim]耗时:[/] {result.elapsed_time:.3f}s\n"
            f"[dim]模型:[/] {result.model}\n"
            f"{keywords_str}",
            title=f"模型: {model_key}",
            border_style="green" if result.keywords else "yellow"
        ))


@app.command(name="interactive")
def interactive_command(
    model: str = typer.Option("text2vec", "--model", "-m", help="默认模型"),
):
    """交互式提取模式"""
    config = ExtractorConfig(model_name=model)
    extractor = KeywordExtractor(config)
    
    console.print(Panel.fit(
        "[bold blue]中文关键词提取工具 - 交互模式[/]\n"
        "[dim]输入文本提取关键词，输入 'quit' 退出，输入 'model <名称>' 切换模型[/]"
    ))
    
    while True:
        try:
            user_input = console.input("\n[bold green]> [/]")
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                break
            
            if user_input.lower().startswith('model '):
                new_model = user_input.split(maxsplit=1)[1]
                config.model_name = new_model
                extractor = KeywordExtractor(config)
                console.print(f"[green]已切换到模型: {new_model}[/]")
                continue
            
            if not user_input.strip():
                continue
            
            result = extractor.extract(user_input)
            
            table = Table(box=box.SIMPLE)
            table.add_column("关键词", style="green")
            table.add_column("分数", style="yellow", justify="right")
            
            for kw in result.keywords:
                table.add_row(kw.keyword, f"{kw.score:.4f}")
            
            console.print(table)
            console.print(f"[dim]耗时: {result.elapsed_time:.3f}s | 模型: {result.model}[/]")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            console.print(f"[red]错误: {e}[/]")
    
    console.print("\n[dim]再见![/]")


def main_entry():
    """入口函数"""
    app()
