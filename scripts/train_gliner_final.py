"""
GLiNER 微调训练脚本 - 最终正确版本

使用 GLiNER 官方 train_model API，正确处理数据格式

运行:
    python scripts/train_gliner_final.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from loguru import logger
from gliner import GLiNER
from gliner.data_processing import WordsSplitter
from gliner.training import TrainingArguments


def prepare_training_data(data_path: str, splitter: WordsSplitter):
    """
    准备 GLiNER 训练数据格式
    
    需要的格式:
    {
        'tokenized_text': ['token1', 'token2', ...],  # 字符串列表
        'ner': [[start, end, label, text], ...]
    }
    """
    samples = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            ex = json.loads(line)
            text = ex["text"]
            
            # 使用 WordsSplitter 分词，提取字符串部分
            tokens_with_pos = list(splitter(text))
            tokenized_text = [t[0] for t in tokens_with_pos]  # 提取字符串
            
            # 转换 NER 格式 (只需要 start, end, label 三个元素)
            ner = []
            for ent in ex["entities"]:
                ner.append([
                    ent["start"],
                    ent["end"],
                    ent["label"]
                ])
            
            samples.append({
                'tokenized_text': tokenized_text,
                'ner': ner
            })
    
    return samples


def main():
    logger.info("=" * 70)
    logger.info("🚀 GLiNER 微调训练 (最终版)")
    logger.info("=" * 70)
    
    # 路径
    data_dir = Path(__file__).parent.parent / "data"
    model_dir = Path(__file__).parent.parent / "models"
    model_dir.mkdir(exist_ok=True)
    output_dir = model_dir / "gliner_finetuned"
    
    # 1. 加载基础模型
    base_model = "urchade/gliner_multi-v2.1"
    logger.info(f"🤖 加载基础模型: {base_model}")
    model = GLiNER.from_pretrained(base_model)
    
    # 获取分词器
    splitter = WordsSplitter()
    
    # 2. 准备数据
    logger.info("📚 准备训练数据...")
    train_data = prepare_training_data(data_dir / "gliner_train.jsonl", splitter)
    test_data = prepare_training_data(data_dir / "gliner_test.jsonl", splitter)
    
    logger.info(f"   训练集: {len(train_data)} 条")
    logger.info(f"   测试集: {len(test_data)} 条")
    
    # 3. 配置训练参数
    logger.info("⚙️  配置训练参数...")
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=5e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=5,  # 增加 epochs
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_steps=5,
        do_train=True,
        do_eval=True,
        warmup_ratio=0.1,
    )
    
    # 4. 开始训练
    logger.info("🏃 开始训练 (5 epochs)...")
    logger.info("   预计时间: Mac MPS ~5-10分钟")
    
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
        
    except Exception as e:
        logger.error(f"❌ 训练失败: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
