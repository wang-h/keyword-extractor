# keyword-extractor

基于 **GLiNER + Top-K Gated Compression** 的中文关键词提取项目。  
当前主线是 `v0.3.x` 的 **Top-K Gated GLiNER V3**。

## 项目状态

- 推理主入口：`GLiNEREntityExtractor`
- Top-K 门控主入口：`attach_topk_gate` / `GatedGLiNER`
- 命令行工具：`kwextract`
- 训练脚本：`scripts/train_gated_gliner.py`（两阶段：warmup + e2e）

## 核心能力

- 面向中文长文本的关键词提取
- 可选噪声门控（过滤低信息密度片段）
- 可选 Top-K 物理压缩（在 encoder 中将序列长度从 `L -> K`）
- 兼容直接加载 Hub 模型或本地微调目录

## 安装

```bash
# 开发模式
pip install -e .

# 或普通安装
pip install .
```

## 快速开始

### CLI

```bash
# 版本
kwextract version

# 健康检查
kwextract health

# 直接提取
kwextract extract "苹果发布新一代 M 系列芯片，强调端侧 AI 推理能力" -k 8

# 文件输入 + JSON 输出
kwextract extract --file article.txt --json
```

### CLI 常用参数

- `--model, -m`：模型名或本地目录，默认 `urchade/gliner_multi-v2.1`
- `--top-k, -k`：返回关键词数量（1-100）
- `--threshold`：提取阈值（0.0-1.0）
- `--no-noise-gate`：关闭噪声门控
- `--topk-gate`：启用 Top-K 物理压缩
- `--topk-keep-k`：Top-K 保留 token 数（默认 1500）
- `--json`：输出 JSON

示例：

```bash
kwextract extract "长文本内容" --topk-gate --topk-keep-k 1500 --json
```

### Python API

```python
from keyword_extractor import GLiNEREntityExtractor

extractor = GLiNEREntityExtractor(
    model_name="urchade/gliner_multi-v2.1",
    threshold=0.3,
    use_noise_gate=True,
    use_topk_gate=False,
    topk_keep_k=1500,
)

result = extractor.extract(
    "苹果发布新一代 M 系列芯片，强调端侧 AI 推理能力。",
    top_k=10,
    return_metadata=True,
)

print(result.method, result.model, result.elapsed_time)
for kw in result.keywords:
    print(kw.keyword, kw.score)
```

## 训练（Top-K Gated GLiNER V3）

训练脚本：`scripts/train_gated_gliner.py`

### 数据文件

- 训练集：`data/gliner_train.jsonl`
- 测试集：`data/gliner_test.jsonl`

如数据缺失，可先执行：

```bash
PYTHONPATH=src ./.venv/bin/python scripts/prepare_training_data.py
```

### 两阶段训练（默认）

```bash
PYTHONPATH=src ./.venv/bin/python scripts/train_gated_gliner.py \
  --device cuda \
  --keep-k 1500 \
  --compress-layer 2 \
  --max-len 512 \
  --batch-size 2
```

### 仅阶段一（Gate Warmup）

```bash
PYTHONPATH=src ./.venv/bin/python scripts/train_gated_gliner.py \
  --phase1-only \
  --warmup-epochs 3 \
  --device cuda
```

### 仅阶段二（从已有 warmup/checkpoint 继续）

```bash
PYTHONPATH=src ./.venv/bin/python scripts/train_gated_gliner.py \
  --phase2-only \
  --warmup-ckpt models/gated_gliner_warmup \
  --finetune-epochs 6 \
  --output models/gated_gliner_gpu_run_e6 \
  --device cuda \
  --keep-k 1500 \
  --compress-layer 2 \
  --max-len 512 \
  --batch-size 2
```

## 自检与测试

```bash
# 快速冒烟（随机权重，不依赖下载）
PYTHONPATH=src ./.venv/bin/python scripts/smoke_gated_gliner.py --quick

# 完整冒烟（可能下载模型，耗时更长）
PYTHONPATH=src ./.venv/bin/python scripts/smoke_gated_gliner.py --full

# 测试
pytest
```

## 目录说明

- `src/keyword_extractor/gliner_memory.py`：生产提取器与记忆策略
- `src/keyword_extractor/gated_gliner.py`：GLiNER 级 Top-K 门控注入
- `src/keyword_extractor/topk_compressed_encoder.py`：DeBERTa encoder 压缩实现
- `src/keyword_extractor/topk_gated_dropping.py`：Top-K 打分与索引映射
- `scripts/train_gated_gliner.py`：两阶段训练
- `scripts/smoke_gated_gliner.py`：端到端冒烟测试

## 已知事项

- `max_len` 会在 GLiNER data processor 处硬截断，长文本会出现截断 warning
- 训练会产生较大 checkpoint（含 `optimizer.pt` / `scheduler.pt`），请预留磁盘空间
- 若只做推理，可仅保留最终模型目录，删除中间 `epoch_xx/checkpoint-*`
