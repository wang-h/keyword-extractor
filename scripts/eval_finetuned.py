"""
评估微调后的 GLiNER 模型
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import json
from keyword_extractor import GLiNEREntityExtractor


def normalize(text):
    return text.lower().replace(" ", "").replace("-", "")


def calc_metrics(predicted, ground_truth):
    pred_norm = {normalize(p) for p in predicted}
    gt_norm = {normalize(g) for g in ground_truth}
    
    correct = len(pred_norm & gt_norm)
    precision = correct / len(predicted) if predicted else 0.0
    recall = correct / len(ground_truth) if ground_truth else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {"precision": precision, "recall": recall, "f1": f1, "correct": correct}


def evaluate_finetuned():
    print("=" * 70)
    print("🧪 微调模型评估")
    print("=" * 70)
    
    # 加载微调模型
    extractor = GLiNEREntityExtractor(
        model_name='models/gliner_finetuned',
        labels=['科技公司', 'AI模型', '人名', '核心技术'],
        device='mps'
    )
    
    # 加载测试数据
    with open(Path(__file__).parent.parent / "data" / "gliner_test.jsonl", 'r') as f:
        test_data = [json.loads(line) for line in f]
    
    print(f"✅ 加载 {len(test_data)} 条测试数据")
    
    results = []
    for i, ex in enumerate(test_data[:5], 1):
        text = ex["text"][:500]  # 取前500字符
        gt_entities = [e["text"] for e in ex["entities"]]
        
        # 预测
        result = extractor.extract(text, top_k=10)
        predicted = [kw.keyword for kw in result.keywords]
        
        metrics = calc_metrics(predicted, gt_entities)
        results.append(metrics)
        
        status = "✅" if metrics["f1"] > 0.3 else ("⚠️" if metrics["f1"] > 0.1 else "❌")
        print(f"\n{status} [{i}] {text[:50]}...")
        print(f"   标注: {gt_entities}")
        print(f"   预测: {predicted}")
        print(f"   F1={metrics['f1']:.2f}")
    
    # 汇总
    avg_f1 = sum(r["f1"] for r in results) / len(results)
    print(f"\n📊 平均 F1: {avg_f1:.3f}")
    print(f"\n对比:")
    print(f"   微调前: 0.236")
    print(f"   微调后: {avg_f1:.3f}")


if __name__ == "__main__":
    evaluate_finetuned()
