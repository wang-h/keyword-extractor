"""
GLiNER 微调训练脚本 - 简化版

使用 GLiNER 官方推荐的方式微调

依赖安装:
    pip install gliner

运行:
    python scripts/train_gliner_simple.py

预计训练时间:
    - CPU: ~30分钟 (3 epochs, 33 samples)
    - GPU: ~5分钟
"""
import json
import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from loguru import logger


def load_data(data_path: str):
    """加载 GLiNER 格式数据"""
    examples = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            examples.append(json.loads(line))
    return examples


def train_simple():
    """简化版训练流程"""
    logger.info("=" * 70)
    logger.info("🚀 GLiNER 微调训练 (简化版)")
    logger.info("=" * 70)
    
    try:
        from gliner import GLiNER
    except ImportError:
        logger.error("GLiNER 未安装，运行: pip install gliner")
        return
    
    # 路径
    data_dir = Path(__file__).parent.parent / "data"
    model_dir = Path(__file__).parent.parent / "models"
    model_dir.mkdir(exist_ok=True)
    
    # 1. 加载数据
    logger.info("📚 加载训练数据...")
    train_data = load_data(data_dir / "gliner_train.jsonl")
    test_data = load_data(data_dir / "gliner_test.jsonl")
    
    logger.info(f"   训练集: {len(train_data)} 条")
    logger.info(f"   测试集: {len(test_data)} 条")
    
    # 2. 加载基础模型
    logger.info("🤖 加载基础模型: urchade/gliner_multi-v2.1")
    model = GLiNER.from_pretrained("urchade/gliner_multi-v2.1")
    
    # 3. 准备数据格式
    # GLiNER 训练需要特定格式
    train_samples = []
    for ex in train_data:
        text = ex["text"]
        labels = []
        for ent in ex["entities"]:
            labels.append([
                ent["start"],
                ent["end"],
                ent["label"],
                ent["text"]
            ])
        train_samples.append({"text": text, "labels": labels})
    
    logger.info(f"✅ 准备 {len(train_samples)} 个训练样本")
    
    # 4. 训练参数
    logger.info("🏃 开始训练...")
    logger.info("   Epochs: 3")
    logger.info("   Batch size: 2")
    logger.info("   Learning rate: 5e-5")
    
    # 使用 GLiNER 的 fit 方法训练
    try:
        # 新版本的 GLiNER API
        model.fit(
            train_samples,
            num_epochs=3,
            batch_size=2,
            learning_rate=5e-5,
            verbose=True
        )
    except AttributeError:
        # 旧版本可能需要不同方式
        logger.warning("当前 GLiNER 版本可能不支持直接 fit，尝试替代方案...")
        # 保存数据供后续使用
        with open(model_dir / "train_data_prepared.json", "w", encoding="utf-8") as f:
            json.dump(train_samples, f, ensure_ascii=False, indent=2)
        logger.info("训练数据已准备好，请查看 GLiNER 官方文档进行微调")
        return
    
    # 5. 保存模型
    output_path = model_dir / "gliner_finetuned"
    logger.info(f"💾 保存模型到: {output_path}")
    model.save_pretrained(output_path)
    
    # 6. 测试
    logger.info("🧪 测试微调效果...")
    test_text = "OpenAI发布了GPT-5，Anthropic推出了Claude Opus。"
    entities = model.predict_entities(test_text, ["科技公司", "AI模型"])
    logger.info(f"   测试: {test_text}")
    logger.info(f"   预测: {entities}")
    
    logger.info("\n✅ 训练完成!")
    logger.info(f"模型保存在: {output_path}")


if __name__ == "__main__":
    train_simple()
