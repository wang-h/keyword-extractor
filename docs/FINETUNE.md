# GLiNER 微调指南

## 📊 当前状态

| 模型 | F1 Score | 速度 | 说明 |
|------|----------|------|------|
| GLiNER (zero-shot) | 0.182 | 0.09s | 基础模型 |
| **Hybrid** | **0.236** | 0.43s | Gazetteer + GLiNER |
| **目标 (微调后)** | **0.4+** | ~0.1s | 预期提升 |

## 🎯 微调目标

1. **提升 Precision**: 学习识别真正的实体，减少 CSS/HTML 噪音
2. **提升 Recall**: 学习标注数据中的实体模式
3. **领域适配**: 针对中文科技文章优化

## 📁 训练数据

```
data/
├── gliner_train.jsonl    # 训练集 (33篇, 102实体)
└── gliner_test.jsonl     # 测试集 (9篇)
```

### 数据格式
```json
{
  "text": "文章标题\n\n文章内容...",
  "entities": [
    {"start": 0, "end": 6, "label": "科技公司", "text": "OpenAI"},
    {"start": 8, "end": 13, "label": "AI模型", "text": "GPT-4"}
  ]
}
```

## 🚀 训练方法

### 方法 1: 简化训练脚本
```bash
cd ~/projects/keyword-extractor
.venv/bin/python scripts/train_gliner_simple.py
```

### 方法 2: 完整训练脚本
```bash
.venv/bin/python scripts/train_gliner.py \
    --epochs 5 \
    --batch-size 4 \
    --lr 5e-5 \
    --output models/gliner_finetuned
```

### 方法 3: 手动训练 (推荐)
```python
from gliner import GLiNER

# 1. 加载基础模型
model = GLiNER.from_pretrained("urchade/gliner_multi-v2.1")

# 2. 加载数据
import json
train_data = []
with open("data/gliner_train.jsonl") as f:
    for line in f:
        ex = json.loads(line)
        train_data.append({
            "text": ex["text"],
            "labels": [[e["start"], e["end"], e["label"], e["text"]] 
                      for e in ex["entities"]]
        })

# 3. 训练
model.fit(train_data, num_epochs=3, batch_size=2)

# 4. 保存
model.save_pretrained("models/gliner_finetuned")
```

## 🧪 评估

```python
from keyword_extractor import GLiNEREntityExtractor

# 加载微调模型
extractor = GLiNEREntityExtractor(
    model_name="models/gliner_finetuned",
    labels=["科技公司", "AI模型", "核心技术"]
)

# 测试
text = "OpenAI发布GPT-5，马斯克惊叹。"
result = extractor.extract(text, top_k=5)
print(result.keywords)
```

## 📈 预期效果

### 成功案例 (训练后应能正确识别)

| 文本 | 当前错误 | 预期正确 |
|------|---------|---------|
| "OpenAI发布GPT-5" | "RL", "TOM" | "OpenAI", "GPT-5" |
| "Claude考场觉醒" | "Microsoft" | "Claude", "Anthropic" |
| "马斯克赛博果蝇" | "RL", "STEM" | "马斯克", "Eon Systems" |

## ⚠️ 注意事项

1. **数据量小**: 只有 33 篇训练数据，可能过拟合
   - 建议: 数据增强、正则化

2. **标签不平衡**: 所有标签都是 "技术实体"
   - 建议: 细分标签类型

3. **实体位置**: 部分实体在标注中位置不准确
   - 建议: 人工校验

## 🔧 优化建议

### 1. 数据增强
```python
# 同义词替换
"OpenAI" -> "Open AI", "openai"
"GPT-4" -> "GPT4", "gpt-4"
```

### 2. 负样本采样
添加明确不是实体的片段作为负样本

### 3. 标签细化
```python
labels = [
    "科技公司全称",      # OpenAI, Google
    "AI模型及版本号",    # GPT-5, Claude-3.5
    "软件产品",         # OpenClaw, VSCode
    "核心技术术语",      # Transformer, MoE
    "知名人名",          # 马斯克, 卡帕西
]
```

## 📚 参考

- GLiNER GitHub: https://github.com/urchade/GLiNER
- 论文: https://arxiv.org/abs/2311.08541
