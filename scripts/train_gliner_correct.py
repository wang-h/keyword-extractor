"""
GLiNER 微调训练脚本 - 正确版本

使用 GLiNER 官方 train_model API

运行:
    python scripts/train_gliner_correct.py

预计时间: 
    - CPU: ~10-20分钟 (Mac Mini)
    - GPU: ~3-5分钟
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from loguru import logger
from gliner import GLiNER
from gliner.training import TrainingArguments


def load_gliner_data(data_path: str):
    """加载 GLiNER 格式数据并转换为训练格式"""
    samples = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            ex = json.loads(line)
            # 转换为 GLiNER 训练格式
            ner = []
            for ent in ex["entities"]:
                ner.append([
                    ent["start"],
                    ent["end"],
                    ent["label"],
                    ent["text"]
                ])
            samples.append({
                "text": ex["text"],
                "ner": ner
            })
    return samples


def main():
    logger.info("=" * 70)
    logger.info("🚀 GLiNER 微调训练")
    logger.info("=" * 70)
    
    # 路径
    data_dir = Path(__file__).parent.parent / "data"
    model_dir = Path(__file__).parent.parent / "models"
    model_dir.mkdir(exist_ok=True)
    output_dir = model_dir / "gliner_finetuned"
    
    # 1. 加载数据
    logger.info("📚 加载训练数据...")
    train_data = load_gliner_data(data_dir / "gliner_train.jsonl")
    test_data = load_gliner_data(data_dir / "gliner_test.jsonl")
    
    logger.info(f"   训练集: {len(train_data)} 条")
    logger.info(f"   测试集: {len(test_data)} 条")
    
    # 2. 加载基础模型
    base_model = "urchade/gliner_multi-v2.1"
    logger.info(f"🤖 加载基础模型: {base_model}")
    model = GLiNER.from_pretrained(base_model)
    
    # 3. 配置训练参数
    logger.info("⚙️  配置训练参数...")
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=5e-5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=3,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_steps=5,
        do_train=True,
        do_eval=True,
    )
    
    # 4. 开始训练
    logger.info("🏃 开始训练 (3 epochs, batch_size=2)...")
    logger.info("   预计时间: CPU 10-20分钟")
    
    try:
        trainer = model.train_model(
            train_dataset=train_data,
            eval_dataset=test_data,
            training_args=training_args,
            output_dir=str(output_dir)
        )
        logger.info("✅ 训练完成!")
        
        # 5. 保存最终模型
        logger.info(f"💾 保存模型到: {output_dir}")
        model.save_pretrained(output_dir)
        
        # 6. 快速测试
        logger.info("🧪 快速测试...")
        test_text = "OpenAI发布了GPT-5，马斯克惊叹。"
        entities = model.predict_entities(
            test_text, 
            ["科技公司", "AI模型", "人名"]
        )
        logger.info(f"   输入: {test_text}")
        logger.info(f"   输出: {entities}")
        
        logger.info("\n🎉 全部完成!")
        logger.info(f"微调模型保存在: {output_dir}")
        logger.info(f"\n使用方式:")
        logger.info(f"  extractor = GLiNEREntityExtractor(")
        logger.info(f"      model_name='{output_dir}'")
        logger.info(f"  )")
        
    except Exception as e:
        logger.error(f"❌ 训练失败: {e}")
        logger.error("可能原因: 内存不足，尝试减小 batch_size 到 1")
        raise


if __name__ == "__main__":
    main()
