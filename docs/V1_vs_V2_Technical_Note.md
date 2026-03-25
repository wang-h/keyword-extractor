# V1.0 vs V2.0 技术路线对比说明

## 一、核心任务定义

**目标**：从科技类微信公众号文章（1000-5000字）中精准提取三类实体：
- **公司名**：OpenAI、DeepSeek、字节跳动、英伟达
- **产品/模型名**：GPT-4、Claude Opus 4.6、豆包、Kimi
- **核心技术名**：Transformer、MoE、Diffusion、RLHF

**关键难点**：这些实体**每天都在新增**（如昨天刚发布的"GPT-5.4"），传统词典永远无法覆盖。

---

## 二、V1.0 方案复盘（BERT-Memory）

### 技术路线
```
Jieba 分词 → BERT 编码 → 余弦相似度打分 → 双重记忆融合
```

### 实测结果（20篇标注文章）
| 指标 | 数值 | 说明 |
|------|------|------|
| **F1 Score** | **0.047** | 近乎失效 |
| Precision | 0.040 | 预测10个对0.4个 |
| Recall | 0.060 | 标注的实体找不回6% |

### 失败案例分析

**案例1**："Claude考场突然「觉醒」，自行写代码偷答案！"
- **人工标注**：`Anthropic`, `Claude Opus 4.6`, `XOR加密`
- **V1.0 提取**：`编辑器`, `UI`, `Helvetica`, `YaHei`, `Microsoft`
- **问题**：提取的是HTML/CSS样式关键词，不是文章实体

**案例2**："GPT-5.4 到底变强了多少？"
- **人工标注**：`GPT-5.4`, `Claude Opus 4.6`, `Codex`
- **V1.0 提取**：`GPT`, `电脑`, `核心`, `Claude`
- **问题**：
  - 无法识别带版本号的实体（"GPT-5.4"被切成"GPT"）
  - 无法区分实体类型（把普通词"电脑"也当关键词）

### 根本原因

| 问题 | 具体表现 | 技术根源 |
|------|---------|---------|
| **词典覆盖不足** | "Claude Opus 4.6" 无法识别 | Jieba 词表没有 |
| **版本号断裂** | "GPT-5.4" → "GPT" + "5.4" | 基于空格/标点分词 |
| **无实体类型知识** | 分不清公司/产品/技术 | BERT 只有语义，无类别 |
| **HTML噪音干扰** | 提取CSS字体名 | 无文本清洗机制 |

**结论**：V1.0 提取的是"文章主题词"，不是你要的"专有实体"。**此路不通。**

---

## 三、V2.0 方案设计（GLiNER + Dual-Memory）

### 技术路线
```
滑动窗口切分 → GLiNER Zero-shot实体识别 → ESM/GTM双重记忆 → 全局融合打分
```

### 核心升级

#### 1. GLiNER：Zero-shot 实体识别

**原理**：双流跨度匹配（Bipartite Span Matching）

```python
# 定义你想提取的实体类型（无需训练）
labels = ["科技公司", "软件产品", "AI模型", "核心技术", "人名"]

# GLiNER 直接输出实体
entities = model.predict_entities(text, labels)
# 输出: [
#   {"text": "GPT-5.4", "label": "AI模型", "score": 0.95},
#   {"text": "OpenAI", "label": "科技公司", "score": 0.98}
# ]
```

**优势**：
- ✅ **Zero-shot**：无需训练，直接识别新实体
- ✅ **版本号保留**："GPT-5.4" 作为一个整体识别
- ✅ **类型标注**：自动区分公司/产品/技术
- ✅ **置信度打分**：过滤低质量识别

#### 2. 双重记忆机制（Dual-Memory）

**ESM（实体状态记忆）**：
```python
Memory = {
    "GPT-5.4": {
        "embedding": [768-dim向量],  # GLiNER输出的特征
        "confidence": 0.95,           # 平均置信度
        "freq": 3,                    # 出现3次
        "chunks": {0, 2, 5}          # 出现在第0/2/5个chunk
    }
}
```

**GTM（全局主题记忆）**：
```
V_global = (1-α)·V_prev + α·V_cls
```
维护整篇文章的语义中心，用于过滤偏离主题的实体。

#### 3. 全局融合打分

```
S_final = λ₁·C_gliner + λ₂·sim(E, V_global) + λ₃·log(freq) + λ₄·IDF
```

| 成分 | 作用 |
|------|------|
| C_gliner | GLiNER置信度（局部确信度） |
| sim(E, V_global) | 与文章主题的契合度 |
| log(freq) | 出现频次（跨段落一致性） |
| IDF | 逆文档频率（过滤通用词） |

---

## 四、两种方案对比总结

| 维度 | V1.0 (BERT-Memory) | V2.0 (GLiNER-Memory) |
|------|-------------------|---------------------|
| **实体识别** | ❌ Jieba分词，词典受限 | ✅ GLiNER，Zero-shot |
| **版本号处理** | ❌ "GPT-5.4"断裂 | ✅ 完整保留 |
| **类型标注** | ❌ 无 | ✅ 公司/产品/技术 |
| **新词发现** | ❌ 需更新词典 | ✅ 自动识别 |
| **HTML清洗** | ❌ 提取CSS噪音 | ✅ 预处理过滤 |
| **跨段落融合** | ✅ ESM+GTM | ✅ ESM+GTM |
| **速度** | 🐢 2-3s/篇 | ⚡ <1s/篇 |
| **F1 Score** | **0.047** | 待测试（预期>0.6） |

---

## 五、学生开发指南

### 已完成工作
- ✅ GLiNER 模型接入（`gliner_memory.py`）
- ✅ 双重记忆机制适配（`GLiNERMemoryTracker`）
- ✅ Phase 1 & 2 串联代码（`phase1_phase2_pipeline.py`）
- ✅ 20篇标注测试集准备（`data/`）

### 待完成任务

**Phase 3（Week 3）：联合调优**
1. 修复 GTM embedding 接口（当前使用 sentence-transformer 作为 fallback）
2. 调试 λ₁/λ₂/λ₃/λ₄ 权重
3. 消融实验：对比有无 GTM、有无 IDF 的效果

**Phase 4（Week 4）：组件化封装**
1. 封装为标准 Python API
2. 性能 Profiling（目标：<1s/篇）
3. BabelDOC 集成测试

### 关键代码入口

```python
# src/keyword_extractor/gliner_memory.py
class GLiNEREntityExtractor:
    def extract(self, text, top_k=10):
        # 1. 滑动窗口切分
        # 2. GLiNER 实体识别
        # 3. ESM/GTM 更新
        # 4. 全局融合打分
        pass

# tests/phase1_phase2_pipeline.py
def phase1_baseline_test(articles):
    # 对比 V1.0 和 V2.0 效果
    pass
```

---

## 六、一句话结论

> **V1.0 失败不是因为记忆机制不好，而是因为候选召回阶段（BERT+Jieba）从根本上无法解决「新实体识别」问题。GLiNER 的 Zero-shot 能力才是破局关键，双重记忆机制在此基础上做「跨段落融合」才是正确架构。**

---

*文档版本：v1.0*  
*作者：白虎 (OpenClaw)*  
*日期：2026-03-14*
