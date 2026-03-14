"""
GLiNER 微调训练脚本

基于生成的训练数据微调 GLiNER 模型
预期效果：F1 从 0.236 提升到 0.4+

使用方法：
    python scripts/train_gliner.py --epochs 3 --batch-size 4

输出：
    - models/gliner_finetuned/ (微调后的模型)
    - models/training_logs.json (训练日志)
"""
import json
import argparse
from pathlib import Path
from datetime import datetime

import torch
from torch.utils.data import Dataset
from loguru import logger

# GLiNER 导入
try:
    from gliner import GLiNER, GLiNERConfig
    from gliner.training import Trainer, TrainingArguments
    GLINER_AVAILABLE = True
except ImportError:
    GLINER_AVAILABLE = False
    logger.error("GLiNER 未安装，请运行: pip install gliner")


class GLiNERDataset(Dataset):
    """GLiNER 数据集包装器"""
    
    def __init__(self, data_path: str):
        self.examples = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                self.examples.append(json.loads(line))
        logger.info(f"加载 {len(self.examples)} 条训练数据")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return self.examples[idx]


def train_gliner(
    train_path: str,
    test_path: str,
    output_dir: str,
    base_model: str = "urchade/gliner_multi-v2.1",
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 5e-5,
    max_length: int = 512,
):
    """
    微调 GLiNER 模型
    
    Args:
        train_path: 训练数据路径
        test_path: 测试数据路径
        output_dir: 模型保存路径
        base_model: 基础模型
        epochs: 训练轮数
        batch_size: 批次大小
        learning_rate: 学习率
        max_length: 最大序列长度
    """
    if not GLINER_AVAILABLE:
        raise ImportError("GLiNER 未安装")
    
    logger.info("=" * 70)
    logger.info("🚀 GLiNER 微调训练")
    logger.info("=" * 70)
    
    # 1. 加载数据
    logger.info("📚 加载数据集...")
    train_dataset = GLiNERDataset(train_path)
    test_dataset = GLiNERDataset(test_path)
    
    # 2. 加载基础模型
    logger.info(f"🤖 加载基础模型: {base_model}")
    model = GLiNER.from_pretrained(base_model)
    
    # 3. 配置训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_dir=f"{output_dir}/logs",
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )
    
    # 4. 创建 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )
    
    # 5. 开始训练
    logger.info("🏃 开始训练...")
    trainer.train()
    
    # 6. 保存模型
    logger.info(f"💾 保存模型到: {output_dir}")
    trainer.save_model(output_dir)
    model.save_pretrained(output_dir)
    
    # 7. 保存训练信息
    info = {
        "base_model": base_model,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "max_length": max_length,
        "train_samples": len(train_dataset),
        "test_samples": len(test_dataset),
        "training_time": datetime.now().isoformat(),
    }
    
    with open(f"{output_dir}/training_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)
    
    logger.info("✅ 训练完成!")
    
    return output_dir


def evaluate_model(model_path: str, test_path: str):
    """评估微调后的模型"""
    logger.info("=" * 70)
    logger.info("🧪 模型评估")
    logger.info("=" * 70)
    
    # 加载模型
    model = GLiNER.from_pretrained(model_path)
    
    # 加载测试数据
    test_examples = []
    with open(test_path, 'r', encoding='utf-8') as f:
        for line in f:
            test_examples.append(json.loads(line))
    
    # 评估
    correct = 0
    total_pred = 0
    total_gt = 0
    
    for ex in test_examples:
        text = ex["text"]
        gt_entities = ex["entities"]
        
        # 预测
        predictions = model.predict_entities(text, [e["label"] for e in gt_entities])
        
        # 统计
        pred_set = set((p["start"], p["end"]) for p in predictions)
        gt_set = set((e["start"], e["end"]) for e in gt_entities)
        
        correct += len(pred_set & gt_set)
        total_pred += len(pred_set)
        total_gt += len(gt_set)
    
    precision = correct / total_pred if total_pred > 0 else 0
    recall = correct / total_gt if total_gt > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    logger.info(f"📊 评估结果:")
    logger.info(f"   Precision: {precision:.3f}")
    logger.info(f"   Recall:    {recall:.3f}")
    logger.info(f"   F1 Score:  {f1:.3f}")
    
    return {"precision": precision, "recall": recall, "f1": f1}


def main():
    parser = argparse.ArgumentParser(description="GLiNER 微调训练")
    parser.add_argument("--train", default="data/gliner_train.jsonl", help="训练数据路径")
    parser.add_argument("--test", default="data/gliner_test.jsonl", help="测试数据路径")
    parser.add_argument("--output", default="models/gliner_finetuned", help="模型保存路径")
    parser.add_argument("--base-model", default="urchade/gliner_multi-v2.1", help="基础模型")
    parser.add_argument("--epochs", type=int, default=3, help="训练轮数")
    parser.add_argument("--batch-size", type=int, default=4, help="批次大小")
    parser.add_argument("--lr", type=float, default=5e-5, help="学习率")
    parser.add_argument("--max-length", type=int, default=512, help="最大序列长度")
    parser.add_argument("--eval-only", action="store_true", help="仅评估，不训练")
    
    args = parser.parse_args()
    
    # 创建输出目录
    Path(args.output).mkdir(parents=True, exist_ok=True)
    
    if not args.eval_only:
        # 训练
        train_gliner(
            train_path=args.train,
            test_path=args.test,
            output_dir=args.output,
            base_model=args.base_model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            max_length=args.max_length,
        )
    
    # 评估
    results = evaluate_model(args.output, args.test)
    
    # 保存结果
    with open(f"{args.output}/eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info("\n🎉 全部完成!")


if __name__ == "__main__":
    main()
