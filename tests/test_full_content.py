"""
使用完整文章数据重新测试 GLiNER
"""
import sys
import csv
import json
import time
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from keyword_extractor import GLiNEREntityExtractor


def load_full_data():
    """加载完整文章数据"""
    articles = {}
    with open(Path(__file__).parent.parent / "data" / "articles_full.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            articles[row["id"]] = {
                "title": row["title"],
                "content": row["content"],  # 完整内容
                "description": row["description"],
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
    
    # 过滤无标签的
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


def test_with_full_content():
    """使用完整内容测试"""
    print("=" * 70)
    print("🧪 完整文章内容测试 (GLiNER + HTML清洗)")
    print("=" * 70)
    
    articles = load_full_data()
    print(f"✅ 加载 {len(articles)} 篇带标签文章")
    
    # 使用优化后的 labels
    extractor = GLiNEREntityExtractor(
        model_name="urchade/gliner_multi-v2.1",
        labels=[
            "科技公司",       # 英伟达、OpenAI
            "AI软件产品",     # ChatGPT、Claude
            "大语言模型",     # GPT-4、Llama
            "人名",          # 马斯克、卡帕西
            "技术术语"        # Transformer、MoE
        ],
        chunk_size=1000,  # 增大chunk
        threshold=0.25    # 降低阈值提高召回
    )
    
    # 测试20篇
    article_ids = list(articles.keys())[:20]
    results = []
    
    for i, aid in enumerate(article_ids, 1):
        article = articles[aid]
        title = article["title"]
        # 使用标题 + 描述 + 内容前3000字
        content = article["content"][:3000] if len(article["content"]) > 3000 else article["content"]
        text = f"{title}\n\n{content}"
        
        start = time.time()
        result = extractor.extract(text, top_k=10)
        elapsed = time.time() - start
        
        # 解析预测（去掉标签类型）
        predicted = []
        for kw in result.keywords:
            entity = kw.keyword.split(" (")[0]
            predicted.append(entity)
        
        gt = article["tags"]
        metrics = calc_metrics(predicted, gt)
        
        print(f"\n📄 [{i}] {title[:45]}...")
        print(f"   标注: {', '.join(list(gt)[:5])}")
        print(f"   预测: {', '.join(predicted[:5])}")
        print(f"   ✅ 匹配: {metrics['correct']}/{len(gt)} | P={metrics['precision']:.2f} R={metrics['recall']:.2f} F1={metrics['f1']:.2f} | ⏱️ {elapsed:.2f}s")
        
        results.append({"title": title[:50], **metrics, "time": elapsed})
    
    # 汇总
    avg_p = sum(r["precision"] for r in results) / len(results)
    avg_r = sum(r["recall"] for r in results) / len(results)
    avg_f1 = sum(r["f1"] for r in results) / len(results)
    avg_t = sum(r["time"] for r in results) / len(results)
    
    print("\n" + "=" * 70)
    print("📊 汇总结果")
    print("=" * 70)
    print(f"   平均 Precision: {avg_p:.3f}")
    print(f"   平均 Recall:    {avg_r:.3f}")
    print(f"   平均 F1 Score:  {avg_f1:.3f}")
    print(f"   平均耗时:       {avg_t:.3f}s")
    print(f"\n   较上次提升: F1 {0.182:.3f} → {avg_f1:.3f} (+{((avg_f1-0.182)/0.182)*100:.1f}%)")


if __name__ == "__main__":
    test_with_full_content()
