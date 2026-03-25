"""
BERT-Memory vs 标注数据对比测试

评估指标：
- Precision (精确率): 提取的正确标签 / 提取的总标签
- Recall (召回率): 提取的正确标签 / 标注的总标签
- F1 Score: 2 * (Precision * Recall) / (Precision + Recall)
"""
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Set, Tuple
import time

# 添加项目路径
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from keyword_extractor import BertMemoryExtractor


def load_annotated_data(data_dir: Path) -> Tuple[Dict, Dict]:
    """
    加载标注数据
    
    Returns:
        articles: {article_id: {title, description, content, tags}}
        tags: {tag_id: tag_name}
    """
    # 加载标签
    tags = {}
    with open(data_dir / "tags.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tags[row["id"]] = row["name"]
    
    # 加载文章
    articles = {}
    with open(data_dir / "articles_annotated.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            articles[row["id"]] = {
                "title": row["title"],
                "description": row["description"],
                "content": row["content_preview"],
                "tags": set()  # 将在后面填充
            }
    
    # 加载文章-标签关联
    with open(data_dir / "article_tags.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            article_id = row["article_id"]
            tag_name = row["tag_name"]
            if article_id in articles:
                articles[article_id]["tags"].add(tag_name)
    
    # 过滤掉没有标签的文章
    articles = {k: v for k, v in articles.items() if v["tags"]}
    
    return articles, tags


def normalize_tag(tag: str) -> str:
    """标签归一化（用于匹配对比）"""
    return tag.lower().replace(" ", "").replace("-", "").replace("_", "")


def calculate_metrics(
    predicted: List[str],
    ground_truth: Set[str]
) -> Dict[str, float]:
    """
    计算评估指标
    
    Args:
        predicted: 模型预测的标签列表
        ground_truth: 人工标注的标签集合
    
    Returns:
        {precision, recall, f1, correct, predicted_count, ground_truth_count}
    """
    # 归一化
    pred_normalized = {normalize_tag(p) for p in predicted}
    gt_normalized = {normalize_tag(g) for g in ground_truth}
    
    # 计算正确数
    correct = len(pred_normalized & gt_normalized)
    
    precision = correct / len(predicted) if predicted else 0.0
    recall = correct / len(ground_truth) if ground_truth else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "correct": correct,
        "predicted_count": len(predicted),
        "ground_truth_count": len(ground_truth)
    }


def test_bert_memory(articles: Dict, sample_size: int = 20) -> Dict:
    """
    测试 BERT-Memory 提取效果
    
    Args:
        articles: 文章数据
        sample_size: 测试样本数
    
    Returns:
        评估结果
    """
    print(f"🧪 测试 BERT-Memory (样本数: {sample_size})")
    print("=" * 60)
    
    # 初始化提取器
    print("🔄 加载 MacBERT 模型...")
    extractor = BertMemoryExtractor(
        model_name="macbert",
        chunk_size=400,
        chunk_overlap=80,
        alpha=0.2,
        dynamic_alpha=True
    )
    
    # 采样测试
    article_ids = list(articles.keys())[:sample_size]
    
    results = []
    total_time = 0
    
    for i, article_id in enumerate(article_ids, 1):
        article = articles[article_id]
        title = article["title"]
        content = article["content"][:3000]  # 限制长度
        ground_truth = article["tags"]
        
        # 清洗 HTML
        from bs4 import BeautifulSoup
        try:
            soup = BeautifulSoup(content, 'html.parser')
            # 移除 script 和 style
            for script in soup(["script", "style"]):
                script.decompose()
            clean_content = soup.get_text(separator=' ', strip=True)
            # 清理多余空白
            import re
            clean_content = re.sub(r'\s+', ' ', clean_content)
        except:
            clean_content = content
        
        # 合并文本
        text = f"{title}\n\n{clean_content}"
        
        # 提取
        start = time.time()
        result = extractor.extract(text, top_k=10)
        elapsed = time.time() - start
        total_time += elapsed
        
        predicted = [kw.keyword for kw in result.keywords]
        
        # 计算指标
        metrics = calculate_metrics(predicted, ground_truth)
        metrics["article_id"] = article_id
        metrics["title"] = title[:50]
        metrics["predicted"] = predicted
        metrics["ground_truth"] = list(ground_truth)
        metrics["time"] = elapsed
        
        results.append(metrics)
        
        # 打印单条结果
        print(f"\n📄 [{i}/{sample_size}] {title[:40]}...")
        print(f"   标注标签: {', '.join(list(ground_truth)[:5])}")
        print(f"   预测标签: {', '.join(predicted[:5])}")
        print(f"   匹配: {metrics['correct']}/{len(ground_truth)}, P={metrics['precision']:.2f}, R={metrics['recall']:.2f}, F1={metrics['f1']:.2f}")
    
    # 汇总统计
    avg_precision = sum(r["precision"] for r in results) / len(results)
    avg_recall = sum(r["recall"] for r in results) / len(results)
    avg_f1 = sum(r["f1"] for r in results) / len(results)
    avg_time = total_time / len(results)
    
    summary = {
        "sample_size": sample_size,
        "avg_precision": avg_precision,
        "avg_recall": avg_recall,
        "avg_f1": avg_f1,
        "avg_time": avg_time,
        "total_time": total_time,
        "details": results
    }
    
    print("\n" + "=" * 60)
    print("📊 汇总结果")
    print(f"   样本数: {sample_size}")
    print(f"   平均精确率 (Precision): {avg_precision:.3f}")
    print(f"   平均召回率 (Recall): {avg_recall:.3f}")
    print(f"   平均 F1 Score: {avg_f1:.3f}")
    print(f"   平均处理时间: {avg_time:.2f}s")
    print(f"   总处理时间: {total_time:.2f}s")
    
    return summary


def save_results(results: Dict, output_path: Path):
    """保存测试结果"""
    # 保存详细结果
    with open(output_path / "bert_memory_test_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # 保存 CSV 摘要
    with open(output_path / "bert_memory_test_summary.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["article_id", "title", "precision", "recall", "f1", "correct", "predicted", "ground_truth", "time"])
        for r in results["details"]:
            writer.writerow([
                r["article_id"],
                r["title"],
                r["precision"],
                r["recall"],
                r["f1"],
                r["correct"],
                r["predicted_count"],
                r["ground_truth_count"],
                r["time"]
            ])
    
    print(f"\n💾 结果已保存到: {output_path}")


if __name__ == "__main__":
    # 数据路径
    data_dir = Path(__file__).parent.parent / "data"
    output_dir = Path(__file__).parent.parent / "test_results"
    output_dir.mkdir(exist_ok=True)
    
    print("📂 加载标注数据...")
    articles, tags = load_annotated_data(data_dir)
    print(f"✅ 加载完成: {len(articles)} 篇带标注文章, {len(tags)} 个标签")
    
    # 运行测试
    results = test_bert_memory(articles, sample_size=20)
    
    # 保存结果
    save_results(results, output_dir)
