"""
Hybrid 提取器测试 - Gazetteer + GLiNER 融合
"""
import sys
import csv
import time
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from keyword_extractor import HybridEntityExtractor


def load_data():
    """加载完整文章数据"""
    articles = {}
    with open(Path(__file__).parent.parent / "data" / "articles_full.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            articles[row["id"]] = {
                "title": row["title"],
                "content": row["content"][:4000] if len(row["content"]) > 4000 else row["content"],
            }
    
    # 加载标签
    with open(Path(__file__).parent.parent / "data" / "article_tags.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            article_id = row["article_id"]
            if article_id in articles:
                if "tags" not in articles[article_id]:
                    articles[article_id]["tags"] = set()
                articles[article_id]["tags"].add(row["tag_name"])
    
    return {k: v for k, v in articles.items() if "tags" in v}


def normalize(text):
    return text.lower().replace(" ", "").replace("-", "").replace("_", "").replace("(", "").replace(")", "")


def calc_metrics(predicted, ground_truth):
    pred_norm = {normalize(p) for p in predicted}
    gt_norm = {normalize(g) for g in ground_truth}
    
    correct = len(pred_norm & gt_norm)
    precision = correct / len(predicted) if predicted else 0.0
    recall = correct / len(ground_truth) if ground_truth else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {"precision": precision, "recall": recall, "f1": f1, "correct": correct}


def test_hybrid():
    """测试 Hybrid 提取器"""
    print("=" * 70)
    print("🧪 Hybrid 提取器测试 (Gazetteer + GLiNER + 双重记忆)")
    print("=" * 70)
    
    articles = load_data()
    print(f"✅ 加载 {len(articles)} 篇带标签文章")
    
    # 初始化 Hybrid 提取器
    extractor = HybridEntityExtractor(
        gazetteer_path=str(Path(__file__).parent.parent / "data" / "tags.csv"),
        gliner_model="urchade/gliner_multi-v2.1",
        gazetteer_weight=0.6,
        gliner_weight=0.4,
        threshold=0.25
    )
    
    # 测试20篇
    article_ids = list(articles.keys())[:20]
    results = []
    
    for i, aid in enumerate(article_ids, 1):
        article = articles[aid]
        title = article["title"]
        content = article["content"]
        
        start = time.time()
        result = extractor.extract(content, title=title, top_k=10)
        elapsed = time.time() - start
        
        predicted = [kw.keyword for kw in result.keywords]
        gt = article["tags"]
        metrics = calc_metrics(predicted, gt)
        
        status = "✅" if metrics["f1"] > 0.3 else ("⚠️" if metrics["f1"] > 0.1 else "❌")
        print(f"\n{status} [{i}] {title[:45]}...")
        print(f"   标注: {', '.join(list(gt)[:5])}")
        print(f"   预测: {', '.join(predicted[:5])}")
        print(f"   匹配: {metrics['correct']}/{len(gt)} | P={metrics['precision']:.2f} R={metrics['recall']:.2f} F1={metrics['f1']:.2f} | ⏱️ {elapsed:.2f}s")
        
        results.append({"title": title[:50], **metrics, "time": elapsed})
    
    # 汇总
    avg_p = sum(r["precision"] for r in results) / len(results)
    avg_r = sum(r["recall"] for r in results) / len(results)
    avg_f1 = sum(r["f1"] for r in results) / len(results)
    avg_t = sum(r["time"] for r in results) / len(results)
    
    print("\n" + "=" * 70)
    print("📊 Hybrid 汇总结果")
    print("=" * 70)
    print(f"   平均 Precision: {avg_p:.3f}")
    print(f"   平均 Recall:    {avg_r:.3f}")
    print(f"   平均 F1 Score:  {avg_f1:.3f}")
    print(f"   平均耗时:       {avg_t:.3f}s")
    print(f"\n   对比 GLiNER-only:")
    print(f"     F1: 0.182 → {avg_f1:.3f} ({((avg_f1-0.182)/0.182)*100:+.1f}%)")
    
    return avg_f1


if __name__ == "__main__":
    test_hybrid()
