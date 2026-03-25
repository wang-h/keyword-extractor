# Top-K Gated GLiNER V3 —— 模型设计文档

> 本文梳理当前关键词提取系统的整体架构、各模块职责、训练流程与设计决策，作为开发参考。

---

## 一、问题定义

**目标**：从微信公众号文章（含大量 HTML/CSS 残渣）中准确提取科技领域关键词。

**核心挑战**：

| 挑战 | 具体表现 |
|------|----------|
| **长文本** | 文章 token 数常超 2000，DeBERTa-v2 默认上限 512，暴力截断丢失尾部内容 |
| **噪声污染** | HTML 标签、CSS 样式、微信导航语等占 50~80% 篇幅，干扰 attention |
| **计算开销** | 注意力复杂度 O(L²)，序列长时训练/推理极慢 |
| **实体多样** | 公司名、产品名、AI 模型版本、技术术语等均需识别 |

---

## 二、整体架构

```
原始文章 HTML
    │
    ▼ html_cleaner.py
清洁文本（去 HTML 标签）
    │
    ▼ noise_gate.py（推理阶段可选）
低噪声段落（丢弃 CSS 残渣、广告语等低信息密度段）
    │
    ▼
GLiNER Tokenizer
    │  input_ids: [label_prompt_tokens | text_tokens]  shape [B, L]
    ▼
────────────────── GatedGLiNER（核心模型）─────────────────────
│                                                           │
│   DeBERTa-v2 Encoder（12 层）                              │
│   ┌─────────────────────────────────────┐                 │
│   │  Layer 0                            │                 │
│   │  Layer 1                            │                 │
│   │  Layer 2  ←── compress_after_layer  │                 │
│   │    │                                │                 │
│   │    ▼  TopKGatedDroppingLayer        │                 │
│   │  打分 router(h) → scores [B,L]       │                 │
│   │  TopK → gather → ×scores            │                 │
│   │  [B,L,D] ──── 压缩 ──→ [B,K,D]       │                 │
│   │    │                                │                 │
│   │  Layer 3  (在 [B,K,D] 上运行)        │                 │
│   │  ...                                │                 │
│   │  Layer 11                           │                 │
│   └─────────────────────────────────────┘                 │
│                                                           │
│   Scatter-Expand: [B,K,D] ──→ [B,L,D]                     │
│   （dropped token 位置填 0）                                │
│                                                           │
│   extract_word_embeddings (words_mask)                    │
│   → words_embedding [B,W,D]                               │
│   → span_rep_layer → einsum(span, prompt) → scores        │
│   → focal_loss / predict                                   │
──────────────────────────────────────────────────────────────
    │
    ▼
实体列表（keyword, label, score）
    │
    ▼ gliner_memory.py（双重记忆 ESM + GTM）
跨段落融合、置信度加权、去重
    │
    ▼
最终关键词
```

---

## 三、创新点：与现有工作的对比

| 方案 | 序列长度 | 计算复杂度 | 可训练 | 坐标处理 |
|------|---------|-----------|--------|---------|
| 外部分块（chunking） | 分块各 512 | O(chunk²) × N 块 | ✗ 模型外 | 需手动拼接 |
| CoLT5 条件路由 | 不变（L） | O(L²) 但系数小 | ✓ | 无需处理 |
| 软屏蔽（attention mask） | 不变（L） | O(L²)，flops 不变 | ✓ | 无需处理 |
| **Top-K Gated（本方案）** | **L → K（K≪L）** | **O(K²)** | **✓** | **Scatter-Expand** |

**本方案独特点**：

1. **物理截断**：后续层真正在 `[B, K, D]` 上运行，attention 计算量从 O(L²) 降到 O(K²)，L=2000、K=512 时约 **15× 加速**。

2. **端到端可训**：Router MLP 的梯度通过 `compressed × scores` 乘法节点不断裂地回传，NER loss 直接监督哪些 token 值得保留。

3. **Scatter-Expand 兼容**：压缩发生在 DeBERTa 内部，输出时 scatter 回原长 L，GLiNER 下游代码（`words_mask`、`span_idx`、坐标系）**零修改**。

4. **Prompt-aware 保留**：GLiNER 把标签 prompt 和正文拼接输入，可配置 `n_prompt_tokens` 强制保留前置 prompt token 不参与竞争，防止标签语义被截断。

---

## 四、各模块详解

### 4.1 预处理层

#### `html_cleaner.py`
- 清除 HTML 标签、提取纯文本
- 处理微信公众号特有的嵌套 HTML 结构

#### `noise_gate.py`
- **推理阶段**的轻量段落过滤（训练阶段不使用，避免遮蔽有标注位置）
- 打分维度：字符熵、CJK 比例、套话模式、CSS slug 特征
- 得分 < 阈值（默认 0.22）的段落直接丢弃，不进入 tokenizer

#### `soft_mask_weak_labels.py`
- **训练阶段**工具：把 `noise_gate` 的段落分数转换为 **token 级 0/1 弱标签**
- 通过 tokenizer 的 `offset_mapping` 把字符级标签对齐到子词 token
- 专供门控预热（阶段一）的 BCE 监督

---

### 4.2 核心模型层

#### `topk_gated_dropping.py` — 门控截断层

```
TopKGatedDroppingLayer(hidden_dim, keep_k)

forward(hidden_states [B, L, D]):
  scores = Linear → GELU → Linear → Sigmoid   # [B, L]
  topk_values, topk_indices = topk(scores, K)  # [B, K]
  topk_indices_sorted = sort(topk_indices)      # 保持阅读顺序
  compressed = gather(hidden, sorted_idx)        # [B, K, D]
  compressed = compressed × topk_values         # ← 梯度不断裂的关键
  return compressed, topk_indices_sorted
```

**梯度路径**：`NER loss → compressed → topk_values_sorted → router`

`map_span_to_original(comp_start, comp_end, topk_indices)`：将压缩空间 span 坐标映射回原文下标（备用接口，Scatter-Expand 策略下通常不需要）。

---

#### `topk_compressed_encoder.py` — 压缩 DeBERTa Encoder

继承 HuggingFace `DebertaV2Encoder`，重写 `forward`：

```
第 0..compress_after_layer 层：全长 [B,L,D] + 原 4D mask + rel_pos
          ↓
TopKGatedDroppingLayer：[B,L,D] → [B,K,D]，保存 _last_topk_indices
          ↓
压缩 attention_mask：gather(mask_1d, topk_idx) → 新 4D mask
relative_pos = None（DeBERTa 内部自动按新长度 K 重建）
          ↓
第 compress_after_layer+1..11 层：在 [B,K,D] 上运行
```

**DeBERTa-v2 适配细节**：
- `relative_pos = None` → 每层 `DisentangledSelfAttention.build_relative_position()` 用 `query.size(-2)` 自动重建，无需手动截断相对位置矩阵
- `ConvLayer`（第 0 层后的卷积增强）在压缩前执行，不受影响

---

#### `gated_gliner.py` — GLiNER 集成层

**`attach_topk_gate(gliner_model, ...)`**：就地 monkey-patch 现有 `Encoder`：
1. 替换 `encoder.bert_layer.model.encoder` 为 `TopKCompressedDebertaV2Encoder`
2. 重写 `encoder.encode_text` 方法，在调用父类后执行 **Scatter-Expand**

**Scatter-Expand 逻辑**：
```python
compressed_embeds = super().encode_text(...)   # [B, K, D]
topk_indices = encoder._last_topk_indices      # [B, K]
idx = topk_indices.unsqueeze(-1).expand(-1,-1,D)
expanded = zeros(B, L, D).scatter(1, idx, compressed_embeds)  # [B, L, D]
```

**`GatedGLiNER`**：高级封装类，提供与 `GLiNER` 完全相同的预测接口，额外暴露：
- `router_gate_parameters()` — 仅获取 router 参数，用于阶段一单独优化
- `freeze_non_gate()` / `unfreeze_all()` — 两阶段训练开关
- `last_topk_indices` — 最近一次压缩的 token 保留位置

---

### 4.3 推理层

#### `gliner_memory.py` — 双重记忆推理

**GLiNERMemoryTracker**（双重记忆机制）：
- **ESM（实体状态记忆）**：跨段落追踪实体，置信度加权平均，向量池化融合
- **GTM（全局主题记忆）**：维护全局语义向量，跨段落语义一致性约束

**GLiNEREntityExtractor**：
- 自动分块（语义分块优先，超长段再按句切分）
- 每块独立调 GLiNER 预测，结果经 ESM 融合去重
- `use_topk_gate=True` 时调用 `attach_topk_gate` 注入压缩 encoder，同时自动放大 `chunk_size`

---

## 五、训练流程

### 数据来源

```
WeRSS API
    │
    ▼ download_werss_articles.py
articles_full.jsonl（文章 + 标签）
    │
    ▼ prepare_training_data.py
gliner_train.jsonl / gliner_test.jsonl
格式：{ "text": "...", "ner": [{"start":0,"end":5,"label":"AI模型及版本号"},...] }
    │
    ▼（可选）expand_training_data.py / data_prep.py
增强后的训练集
```

### 阶段一：门控预热（Gate Warmup）

**目标**：让 router 学会区分噪声 token 和有效 token，不截断序列（防止没有 NER 监督时 router 随机删有用词）。

```
compress_after_layer = 999（不可达，不截断）
冻结 DeBERTa 全部层 + GLiNER NER 头
只训 gate_layer.router（约 2 层 MLP）

Loss = BCE(router_scores, token_noise_labels)
token_noise_labels 来自 soft_mask_weak_labels.py：
    noise_gate 段落分数 → 字符级 0/1 → offset_mapping → token 级 0/1
```

**收敛信号**：router 对 CSS/广告语段落打低分，对正文 token 打高分。

### 阶段二：端到端微调（E2E Fine-tune）

**目标**：在真实截断条件下，让 NER loss 直接监督 router 保留实体相关 token。

```
compress_after_layer = 2（真实截断）
解冻全部参数
keep_k 退火：3000 → 1500（若数据量允许）

Loss = GLiNER focal_loss
梯度路径：
  focal_loss
    → words_embedding[dropped] = 0（被截断词无贡献）
    → scatter(expanded, topk_idx, compressed)  ← ∂/∂compressed 正常
    → compressed = gathered × topk_scores      ← ∂/∂topk_scores 正常
    → topk_scores → router MLP                 ← router 得到梯度
```

**训练命令**：
```bash
# 完整两阶段
python scripts/train_gated_gliner.py --keep-k 1500 --compress-layer 2

# 仅预热
python scripts/train_gated_gliner.py --phase1-only --warmup-epochs 3

# 从预热检查点继续
python scripts/train_gated_gliner.py --phase2-only --warmup-ckpt ./models/gated_gliner_warmup
```

---

## 六、关键超参数

| 参数 | 默认值 | 含义 |
|------|-------|------|
| `keep_k` | 1500 | 压缩后保留 token 数，影响计算量与召回率 |
| `compress_after_layer` | 2 | 在第几层后截断，越早压缩越省计算，越晚压缩表示越丰富 |
| `n_prompt_tokens` | 0 | 强制保留的前置 prompt token 数（GLiNER 标签语义保护） |
| `noise_threshold` | 0.22 | 弱标签生成阈值，段落得分低于此视为噪声 |
| `warmup_epochs` | 3 | 阶段一轮数 |
| `finetune_epochs` | 10 | 阶段二轮数 |
| `keep_k_start` | -1 | 阶段二 keep_k 退火起点（-1=不退火） |

---

## 七、文件结构

```
src/keyword_extractor/
├── 核心架构
│   ├── topk_gated_dropping.py     # 门控截断层 + span 坐标映射
│   ├── topk_compressed_encoder.py # 压缩 DeBERTa Encoder
│   └── gated_gliner.py            # GLiNER 集成 + GatedGLiNER 封装
│
├── 预处理
│   ├── html_cleaner.py            # HTML 清洗
│   ├── noise_gate.py              # 段落级噪声过滤（推理用）
│   └── soft_mask_weak_labels.py   # token 级弱标签生成（训练用）
│
├── 推理
│   ├── gliner_memory.py           # 双重记忆推理管线（ESM + GTM）
│   └── gliner_config.py           # 模型路径全局配置
│
├── 配置
│   ├── labels.py                  # 实体类型标签 + 阈值
│   └── models.py                  # KeywordItem / ExtractionResult
│
└── __init__.py                    # 公共 API

scripts/
├── 训练
│   └── train_gated_gliner.py      # 两阶段训练（阶段一预热 + 阶段二 E2E）
│
├── 数据准备
│   ├── download_werss_articles.py # WeRSS API 下载
│   ├── prepare_training_data.py   # 构建 GLiNER 训练格式
│   ├── expand_training_data.py    # 数据扩充
│   ├── data_prep.py               # 数据增强
│   └── llm_generate_training_data.py # LLM 辅助生成标注
│
├── 评估
│   ├── evaluate.py                # 标准 NER 评估
│   ├── eval_finetuned.py          # 微调后模型评估
│   └── quick_eval.py              # 快速评估
│
└── smoke_gated_gliner.py          # 端到端冒烟测试（7 项）
```

---

## 八、已知局限与后续方向

### 当前局限

1. **弱标签质量**：阶段一的 `noise_gate` 弱标签基于规则，对领域特定噪声（如行情表格、英文技术文档）覆盖不全，router 冷启动效果依赖噪声阈值调参。

2. **keep_k 固定**：压缩比在推理时静态，长文与短文用同一 K 值，短文有冗余压缩损耗。可改为按输入长度动态计算 `K = min(L, max_k)`。

3. **Scatter-Expand 内存**：`encode_text` 输出仍是 `[B, L, D]`，节省的只是 DeBERTa 后续层的 FLOPs，不节省最终 embedding 的内存。如果 L 很大（> 4096），此处仍有内存压力。

4. **GLiNER 训练数据量**：`gliner_train.jsonl` 来自 WeRSS 文章+标签的弱监督对齐，标注噪声较高，需持续扩充高质量标注。

### 潜在改进方向

- **动态 K**：按 attention entropy 或段落得分动态决定每个样本的 keep_k
- **分层压缩**：在多个层设置不同的 keep_k（前层保留更多、后层更激进），渐进式压缩
- **Router 蒸馏**：用教师模型（完整 DeBERTa）的 attention 权重蒸馏 router，比弱标签 BCE 更直接
- **实体感知 router**：在 router 特征中加入标签 prompt 的交叉注意力信号，让 router 知道「正在找什么实体」
