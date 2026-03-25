#!/usr/bin/env python3
"""
模型评估脚本
对比 Zero-shot / SFT / RLHF 三种模型的性能
"""
import json
import sys
from pathlib import Path
from typing import List, Dict, Set, Tuple
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from keyword_extractor import GLiNEREntityExtractor


def load_test_data(test_path: str) -> List[Dict]:
    """加载测试数据"""
    data = []
    with open(test_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def normalize_tag(tag: str) -> str:
    """标签归一化"""
    return tag.lower().replace(" ", "").replace("-", "").replace("_", "")


def calculate_metrics(predicted: List[str], ground_truth: List[str]) -> Dict:
    """计算评估指标"""
    pred_set = {normalize_tag(p) for p in predicted}
    gt_set = {normalize_tag(g) for g in ground_truth}
    
    correct = len(pred_set & gt_set)
    
    precision = correct / len(pred_set) if pred_set else 0.0
    recall = correct / len(gt_set) if gt_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'correct': correct,
        'predicted_count': len(pred_set),
        'ground_truth_count': len(gt_set)
    }


def evaluate_model(model_path: str, test_data: List[Dict], model_name: str) -> Dict:
    """评估单个模型"""
    
    print(f"\n评估模型: {model_name}")
    print("-" * 40)
    
    # 初始化提取器
    try:
        extractor = GLiNEREntityExtractor(
            model_name=model_path,
            labels=["科技公司", "AI模型", "核心技术", "硬件设备", "人名"],
            threshold=0.3
        )
    except Exception as e:
        print(f"加载模型失败: {e}")
        return None
    
    # 评估
    all_metrics = []
    
    for i, item in enumerate(test_data):
        text = item['text']
        ground_truth = [e['text'] for e in item.get('entities', [])]
        
        # 预测
        result = extractor.extract(text, top_k=10)
        predicted = [kw.keyword for kw in result.keywords]
        
        # 计算指标
        metrics = calculate_metrics(predicted, ground_truth)
        all_metrics.append(metrics)
        
        # 打印部分结果
        if i < 3:  # 只打印前3条详情
            print(f"\n样本 {i+1}:")
            print(f"  原文: {text[:80]}...")
            print(f"  标注: {ground_truth}")
            print(f"  预测: {predicted}")
            print(f"  P={metrics['precision']:.2f}, R={metrics['recall']:.2f}, F1={metrics['f1']:.2f}")
    
    # 汇总
    avg_precision = sum(m['precision'] for m in all_metrics) / len(all_metrics)
    avg_recall = sum(m['recall'] for m in all_metrics) / len(all_metrics)
    avg_f1 = sum(m['f1'] for m in all_metrics) / len(all_metrics)
    
    print(f"\n汇总:")
    print(f"  Precision: {avg_precision:.3f}")
    print(f"  Recall: {avg_recall:.3f}")
    print(f"  F1 Score: {avg_f1:.3f}")
    
    return {
        'model': model_name,
        'precision': avg_precision,
        'recall': avg_recall,
        'f1': avg_f1
    }


def main():
    """主函数"""
    
    print("=" * 60)
    print("模型评估对比")
    print("=" * 60)
    
    # 加载测试数据
    test_data = load_test_data('./data/gliner_test.jsonl')
    print(f"\n测试数据: {len(test_data)} 条")
    
    # 评估三个模型
    results = []
    
    # 1. Zero-shot (基础模型)
    result = evaluate_model(
        'urchade/gliner_multi-v2.1',
        test_data,
        'Zero-shot (gliner_multi-v2.1)'
    )
    if result:
        results.append(result)
    
    # 2. SFT 模型
    sft_path = './models/gliner_sft'
    if Path(sft_path).exists():
        result = evaluate_model(sft_path, test_data, 'SFT (监督微调)')
        if result:
            results.append(result)
    else:
        print(f"\nSFT 模型不存在: {sft_path}")
    
    # 3. RLHF 模型
    rl_path = './models/gliner_rlhf/best'
    if Path(rl_path).exists():
        result = evaluate_model(rl_path, test_data, 'RLHF (强化学习)')
        if result:
            results.append(result)
    else:
        print(f"\nRLHF 模型不存在: {rl_path}")
    
    # 汇总对比
    print("\n" + "=" * 60)
    print("模型对比汇总")
    print("=" * 60)
    print(f"{'模型':<30} {'Precision':<10} {'Recall':<10} {'F1':<10}")
    print("-" * 60)
    
    for r in results:
        print(f"{r['model']:<30} {r['precision']:<10.3f} {r['recall']:<10.3f} {r['f1']:<10.3f}")
    
    # 保存结果
    output_path = Path('./test_results/evaluation_summary.json')
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n结果已保存: {output_path}")


if __name__ == "__main__":
    main()
