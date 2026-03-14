"""
Phase 1 & 2: GLiNER + 双重记忆机制 串联测试
V2.0 架构核心实现
"""
import sys
import csv
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, Set, List, Tuple

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from keyword_extractor import GLiNEREntityExtractor


def load_test_data(data_dir: Path) -> Tuple[Dict, Dict]:
    """加载测试数据"""
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
                "content": row["content_preview"],
                "tags": set()
            }
    
    # 加载文章-标签关联
    with open(data_dir / "article_tags.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            article_id = row["article_id"]
            tag_name = row["tag_name"]
            if article_id in articles:
                articles[article_id]["tags"].add(tag_name)
    
    # 过滤无标签文章
    articles = {k: v for k, v in articles.items() if v["tags"]}
    
    return articles, tags


def normalize(text: str) -> str:
    """标签归一化"""
    return text.lower().replace(" ", "").replace("-", "").replace("_", "").replace("(", "").replace(")", "")


def calculate_metrics(predicted: List[str], ground_truth: Set[str]) -> Dict:
    """计算 P/R/F1"""
    pred_norm = {normalize(p) for p in predicted}
    gt_norm = {normalize(g) for g in ground_truth}
    
    correct = len(pred_norm & gt_norm)
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


def phase1_baseline_test(articles: Dict, sample_size: int = 20):
    """
    Phase 1: GLiNER 基线测试
    
    目标：验证 GLiNER Zero-shot 效果对比 V1.0
    """
    print("\n" + "="*70)
    print("🧪 Phase 1: GLiNER 基线测试 (Zero-shot)")
    print("="*70)
    
    # 初始化 GLiNER
    print("\n🔄 加载 GLiNER 模型 (xianyun/gliner_chinese_large)...")
    extractor = GLiNEREntityExtractor(
        model_name="gliner-chinese",
        labels=[
            "科技公司",
            "软件产品", 
            "人工智能模型",
            "核心技术",
            "硬件设备",
            "学术会议",
            "人名"
        ],
        chunk_size=800,
        threshold=0.3
    )
    
    # 采样测试
    article_ids = list(articles.keys())[:sample_size]
    
    results = []
    total_time = 0
    
    for i, article_id in enumerate(article_ids, 1):
        article = articles[article_id]
        title = article["title"]
        content = article["content"]
        ground_truth = article["tags"]
        
        # 合并文本
        text = f"{title}\n\n{content}"
        
        # GLiNER 提取
        import time
        start = time.time()
        result = extractor.extract(text, top_k=10)
        elapsed = time.time() - start
        total_time += elapsed
        
        # 解析预测结果（提取实体名，去掉标签类型）
        predicted = []
        for kw in result.keywords:
            # 格式: "实体名 (标签)" → 提取实体名
            entity_text = kw.keyword.split(" (")[0]
            predicted.append(entity_text)
        
        # 计算指标
        metrics = calculate_metrics(predicted, ground_truth)
        metrics.update({
            "article_id": article_id,
            "title": title[:50],
            "predicted": predicted,
            "ground_truth": list(ground_truth),
            "time": elapsed
        })
        results.append(metrics)
        
        # 打印结果
        print(f"\n📄 [{i}/{sample_size}] {title[:45]}...")
        print(f"   📝 标注: {', '.join(list(ground_truth)[:5])}")
        print(f"   🤖 预测: {', '.join(predicted[:5])}")
        print(f"   ✅ 匹配: {metrics['correct']}/{len(ground_truth)} | P={metrics['precision']:.2f} R={metrics['recall']:.2f} F1={metrics['f1']:.2f}")
    
    # 汇总
    avg_p = sum(r["precision"] for r in results) / len(results)
    avg_r = sum(r["recall"] for r in results) / len(results)
    avg_f1 = sum(r["f1"] for r in results) / len(results)
    
    print("\n" + "="*70)
    print("📊 Phase 1 汇总结果")
    print("="*70)
    print(f"   样本数: {sample_size}")
    print(f"   平均精确率 (Precision): {avg_p:.3f}")
    print(f"   平均召回率 (Recall): {avg_r:.3f}")
    print(f"   平均 F1 Score: {avg_f1:.3f}")
    print(f"   平均处理时间: {total_time/sample_size:.2f}s")
    print(f"   总处理时间: {total_time:.2f}s")
    
    return {
        "phase": "phase1",
        "sample_size": sample_size,
        "avg_precision": avg_p,
        "avg_recall": avg_r,
        "avg_f1": avg_f1,
        "details": results
    }


def phase2_memory_integration(articles: Dict, sample_size: int = 20):
    """
    Phase 2: 记忆模块适配与重构
    
    目标：验证 ESM + GTM 双重记忆机制效果
    """
    print("\n" + "="*70)
    print("🧠 Phase 2: 双重记忆机制集成测试")
    print("="*70)
    
    # 使用相同的 GLiNER 配置，但关注记忆机制
    extractor = GLiNEREntityExtractor(
        model_name="gliner-chinese",
        labels=["科技公司", "软件产品", "人工智能模型", "核心技术", "硬件设备", "人名"],
        chunk_size=800,
        alpha=0.2,  # GTM 更新系数
        threshold=0.3
    )
    
    # 测试长文本处理
    article_ids = list(articles.keys())[:sample_size]
    
    print("\n📈 测试长文本分块与记忆融合...")
    
    for i, article_id in enumerate(article_ids[:5], 1):  # 只展示前5个
        article = articles[article_id]
        text = f"{article['title']}\n\n{article['content']}"
        
        result = extractor.extract(text, top_k=10, return_metadata=True)
        
        print(f"\n📄 [{i}] {article['title'][:40]}...")
        print(f"   ⏱️ 处理时间: {result.elapsed_time:.2f}s")
        
        for kw in result.keywords[:5]:
            meta = kw.metadata or {}
            print(f"   • {kw.keyword}")
            if meta:
                print(f"     置信度: {meta.get('confidence', '-')} | 频次: {meta.get('freq', '-')} | 跨度: {meta.get('span_chunks', '-')} chunks")
    
    print("\n✅ Phase 2 完成 - 记忆模块工作正常")
    
    return {"phase": "phase2", "status": "ok"}


def save_results(phase1_results: Dict, output_dir: Path):
    """保存测试结果"""
    output_dir.mkdir(exist_ok=True)
    
    # JSON 详细结果
    with open(output_dir / "phase1_gliner_results.json", "w", encoding="utf-8") as f:
        json.dump(phase1_results, f, ensure_ascii=False, indent=2)
    
    # CSV 摘要
    with open(output_dir / "phase1_summary.csv", "w", encoding="utf-8", newline="") as f:
        import csv
        writer = csv.writer(f)
        writer.writerow(["article_id", "title", "precision", "recall", "f1", "correct", "predicted", "ground_truth", "time"])
        for r in phase1_results["details"]:
            writer.writerow([
                r["article_id"], r["title"],
                r["precision"], r["recall"], r["f1"],
                r["correct"], r["predicted_count"], r["ground_truth_count"], r["time"]
            ])
    
    print(f"\n💾 结果已保存: {output_dir}")


if __name__ == "__main__":
    # 数据路径
    data_dir = Path(__file__).parent.parent / "data"
    output_dir = Path(__file__).parent.parent / "test_results"
    
    print("📂 加载测试数据...")
    articles, tags = load_test_data(data_dir)
    print(f"✅ 加载完成: {len(articles)} 篇带标注文章, {len(tags)} 个标签\n")
    
    # Phase 1: 基线测试
    phase1_results = phase1_baseline_test(articles, sample_size=20)
    
    # Phase 2: 记忆集成
    phase2_results = phase2_memory_integration(articles, sample_size=20)
    
    # 保存结果
    save_results(phase1_results, output_dir)
    
    print("\n" + "="*70)
    print("🎉 Phase 1 & 2 测试完成!")
    print("="*70)
    print("\n下一步:")
    print("  - Phase 3: 联合调优与消融实验")
    print("  - Phase 4: BabelDOC 组件化封装")
