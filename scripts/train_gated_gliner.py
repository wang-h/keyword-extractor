#!/usr/bin/env python3
"""
两阶段训练：Top-K Gated GLiNER V3
====================================

阶段一：门控预热（Gate Warmup）
  - 冻结 DeBERTa + GLiNER NER 头，只训 router MLP
  - compress_after_layer = 999（不真正截断），让 router 学会打分
  - Loss = BCE(router_scores, noise_weak_labels)
  - 弱标签来源：noise_gate 段落分数 → token 级 0/1（via soft_mask_weak_labels）

阶段二：端到端微调（E2E Fine-tune）
  - 解冻全部参数
  - compress_after_layer 生效（真截断），keep_k 从大往小退火
  - Loss = GLiNER 原始 focal_loss（在压缩后再 scatter-expand 回来的序列上算）
  - Router 的 * scores 乘法保证梯度贯穿

用法示例::

    # 完整两阶段训练
    python scripts/train_gated_gliner.py

    # 只做门控预热（调试 router 收敛）
    python scripts/train_gated_gliner.py --phase1-only --warmup-epochs 3

    # 从已预热的检查点直接跑阶段二
    python scripts/train_gated_gliner.py --phase2-only \\
        --warmup-ckpt ./models/gated_gliner_warmup

    # 自定义参数
    python scripts/train_gated_gliner.py \\
        --keep-k 1024 \\
        --compress-layer 3 \\
        --warmup-epochs 2 \\
        --finetune-epochs 8
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from keyword_extractor.gliner_config import (
    GLINER_BASE_MODEL_ID,
    GLINER_DEFAULT_SFT_OUTPUT_DIR,
)
from keyword_extractor.gated_gliner import GatedGLiNER, attach_topk_gate
from keyword_extractor.soft_mask_weak_labels import token_noise_targets_from_text

try:
    from gliner import GLiNER
except ImportError:
    print("ERROR: gliner 未安装，请运行: pip install gliner", file=sys.stderr)
    sys.exit(1)

# --------------------------------------------------------------------------
# 默认路径
# --------------------------------------------------------------------------
DATA_DIR = ROOT / "data"
TRAIN_JSONL = DATA_DIR / "gliner_train.jsonl"
TEST_JSONL = DATA_DIR / "gliner_test.jsonl"
WARMUP_OUTPUT = ROOT / "models" / "gated_gliner_warmup"
FINETUNE_OUTPUT = ROOT / "models" / "gated_gliner_v3"


# --------------------------------------------------------------------------
# 配置
# --------------------------------------------------------------------------
@dataclass
class GatedGLiNERConfig:
    model_name: str = GLINER_BASE_MODEL_ID

    # Top-K 参数
    keep_k: int = 1500
    compress_after_layer: int = 2
    n_prompt_tokens: int = 0
    max_len: int = 512

    # 训练数据
    train_file: str = str(TRAIN_JSONL)
    test_file: str = str(TEST_JSONL)

    # 阶段一：门控预热
    warmup_epochs: int = 3
    warmup_lr: float = 3e-4
    warmup_batch_size: int = 4
    warmup_output_dir: str = str(WARMUP_OUTPUT)

    # 阶段二：端到端微调
    finetune_epochs: int = 10
    finetune_lr: float = 2e-5
    finetune_batch_size: int = 2
    finetune_output_dir: str = str(FINETUNE_OUTPUT)
    weight_decay: float = 0.01

    # keep_k 退火（阶段二从 keep_k_start 线性降到 keep_k）
    keep_k_start: int = -1  # -1 表示不退火，直接用 keep_k

    # 通用
    device: str = "cpu"
    seed: int = 42
    noise_threshold: float = 0.22
    lambda_gate: float = 0.1  # 阶段二联合 gate BCE loss 权重（0=只用 NER loss）


# --------------------------------------------------------------------------
# 数据加载
# --------------------------------------------------------------------------
def load_jsonl(path: str) -> List[Dict]:
    data: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


# --------------------------------------------------------------------------
# 阶段一：门控预热
# --------------------------------------------------------------------------

def _get_tokenizer(gliner_model: "GLiNER"):
    """从 GLiNER 模型取 tokenizer（适配多版本）。"""
    for attr in ("data_processor", "_tokenizer", "tokenizer"):
        obj = getattr(gliner_model, attr, None)
        if obj is not None:
            tok = getattr(obj, "transformer_tokenizer", None) or getattr(obj, "tokenizer", None)
            if tok is not None:
                return tok
    # 降级：直接从 bert_layer.model 取
    try:
        from transformers import AutoTokenizer
        cfg = gliner_model.model.token_rep_layer.bert_layer.model.config
        return AutoTokenizer.from_pretrained(cfg.name_or_path)
    except Exception:
        return None


def warmup_gate(
    base_model: "GLiNER",
    config: GatedGLiNERConfig,
    train_data: List[Dict],
) -> "GLiNER":
    """
    阶段一：只训 router MLP，用噪声弱标签做 BCE 监督。

    compress_after_layer=999 → TopK 层永不触发（序列不足 999 层），
    gate_layer.router 正常打分但不截断，梯度只通过 BCE 传回 router。
    """
    print("\n" + "=" * 60)
    print("阶段一：门控预热（Gate Warmup）")
    print(f"  epochs={config.warmup_epochs}, lr={config.warmup_lr}")
    print("=" * 60)

    # 注入 router，但不启用截断（layer=999 不可达）
    attach_topk_gate(
        base_model,
        compress_after_layer=999,
        keep_k=config.keep_k,
        n_prompt_tokens=config.n_prompt_tokens,
    )

    device = torch.device(config.device)
    base_model.model.to(device)

    # 冻结除 router 外的所有参数
    encoder = base_model.model.token_rep_layer
    topk_enc = encoder.bert_layer.model.encoder
    gate_params = list(topk_enc.gate_layer.router.parameters())
    gate_param_ids = {id(p) for p in gate_params}

    for p in base_model.model.parameters():
        p.requires_grad_(id(p) in gate_param_ids)

    trainable = sum(p.numel() for p in gate_params)
    total = sum(p.numel() for p in base_model.model.parameters())
    print(f"  可训参数: {trainable:,} / {total:,} ({trainable/total*100:.2f}%)")

    optimizer = optim.AdamW(gate_params, lr=config.warmup_lr)
    loss_fn = nn.BCEWithLogitsLoss()

    tokenizer = _get_tokenizer(base_model)
    if tokenizer is None:
        print("  ⚠️  无法获取 tokenizer，跳过门控预热（将直接进入阶段二）")
        return base_model

    base_model.model.train()
    topk_enc.gate_layer.train()

    for epoch in range(config.warmup_epochs):
        total_loss = 0.0
        n_batches = 0

        for sample in train_data:
            text: str = sample.get("text", "") or sample.get("content", "")
            if not text:
                continue

            enc = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=config.max_len,
                return_offsets_mapping=True,
            )
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)
            offsets = enc["offset_mapping"][0].tolist()

            # 弱标签：噪声 token = 1
            weak_labels = token_noise_targets_from_text(
                text,
                offsets,
                noise_if_segment_score_below=config.noise_threshold,
            ).to(device)  # [L]

            # forward（不截断，但经过 router 打分）
            embedding_output = topk_enc.embeddings(input_ids=input_ids) if hasattr(topk_enc, "embeddings") else None
            if embedding_output is None:
                embedding_output = base_model.model.token_rep_layer.bert_layer.model.embeddings(input_ids=input_ids)

            # 直接调 router 在 embeddings 上打分（不走完整 encoder，节省显存）
            # shape: [1, L, hidden]
            scores = topk_enc.gate_layer.router(embedding_output).squeeze(-1)  # [1, L]

            # 对齐标签
            L = weak_labels.size(0)
            scores_trunc = scores[0, :L]  # [L]

            loss = loss_fn(scores_trunc, weak_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(1, n_batches)
        print(f"  Epoch {epoch+1}/{config.warmup_epochs}  loss={avg_loss:.4f}")

    # 解冻全部，便于阶段二
    for p in base_model.model.parameters():
        p.requires_grad_(True)

    # 保存
    warmup_dir = Path(config.warmup_output_dir)
    warmup_dir.mkdir(parents=True, exist_ok=True)
    base_model.save_pretrained(str(warmup_dir))
    torch.save(
        topk_enc.gate_layer.state_dict(),
        warmup_dir / "gate_layer.pt",
    )
    print(f"\n  阶段一完成，模型保存到: {warmup_dir}")
    return base_model


# --------------------------------------------------------------------------
# 阶段二：端到端微调
# --------------------------------------------------------------------------

def finetune_e2e(
    gliner_model: "GLiNER",
    config: GatedGLiNERConfig,
    train_data: List[Dict],
    test_data: List[Dict],
):
    """
    阶段二：激活真正的 Top-K 截断，解冻全部参数，用 GLiNER focal loss 端到端训练。
    """
    print("\n" + "=" * 60)
    print("阶段二：端到端微调（E2E Fine-tune）")
    print(f"  epochs={config.finetune_epochs}, lr={config.finetune_lr}, keep_k={config.keep_k}")
    print("=" * 60)

    # 更新 compress_after_layer 为真实值（阶段一用了 999）
    enc = gliner_model.model.token_rep_layer.bert_layer.model.encoder
    if hasattr(enc, "compress_after_layer"):
        enc.compress_after_layer = config.compress_after_layer
        print(f"  已激活 compress_after_layer={config.compress_after_layer}")

    # keep_k 退火设置
    k_start = config.keep_k_start if config.keep_k_start > 0 else config.keep_k
    k_end = config.keep_k
    k_values: List[int] = []
    if k_start != k_end and config.finetune_epochs > 1:
        for ep in range(config.finetune_epochs):
            frac = ep / max(1, config.finetune_epochs - 1)
            k_val = int(k_start - (k_start - k_end) * frac)
            k_values.append(max(k_val, k_end))
    else:
        k_values = [k_end] * config.finetune_epochs

    total = sum(p.numel() for p in gliner_model.model.parameters())
    print(f"  总参数量: {total:,}")
    if k_start != k_end:
        print(f"  keep_k 退火: {k_start} → {k_end}")

    output_dir = Path(config.finetune_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 用 GLiNER 官方 train_model 进行端到端训练
    # 注意：每 epoch 开始时更新 keep_k
    steps_per_epoch = max(1, math.ceil(len(train_data) / max(1, config.finetune_batch_size)))

    for ep_idx, keep_k_ep in enumerate(k_values):
        # 更新本 epoch 的 keep_k
        enc_obj = gliner_model.model.token_rep_layer.bert_layer.model.encoder
        if hasattr(enc_obj, "keep_k"):
            enc_obj.keep_k = keep_k_ep
            enc_obj.gate_layer.keep_k = keep_k_ep
        ep_dir = str(output_dir / f"epoch_{ep_idx+1:02d}")

        print(f"\n  Epoch {ep_idx+1}/{config.finetune_epochs}  keep_k={keep_k_ep}")

        gliner_model.train_model(
            train_dataset=train_data,
            eval_dataset=test_data if ep_idx == config.finetune_epochs - 1 else None,
            output_dir=ep_dir,
            num_train_epochs=1,
            max_steps=-1,
            per_device_train_batch_size=config.finetune_batch_size,
            per_device_eval_batch_size=config.finetune_batch_size,
            learning_rate=config.finetune_lr,
            weight_decay=config.weight_decay,
            logging_steps=20,
            save_strategy="epoch",
            warmup_ratio=0.05,
            lr_scheduler_type="cosine",
        )

    # 最终保存
    gliner_model.save_pretrained(str(output_dir))
    print(f"\n  阶段二完成，模型保存到: {output_dir}")
    return gliner_model


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="两阶段训练：Top-K Gated GLiNER V3",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", default=GLINER_BASE_MODEL_ID, help="基座模型 ID 或本地路径")
    p.add_argument("--train-file", default=str(TRAIN_JSONL), help="训练数据 JSONL")
    p.add_argument("--test-file", default=str(TEST_JSONL), help="测试数据 JSONL")
    p.add_argument("--keep-k", type=int, default=1500, help="Top-K 保留 token 数")
    p.add_argument("--max-len", type=int, default=512, help="GLiNER 训练最大序列长度（超出会截断）")
    p.add_argument("--keep-k-start", type=int, default=-1, help="阶段二退火起始 keep_k (-1=不退火)")
    p.add_argument("--compress-layer", type=int, default=2, help="TopK 插入层（0-based）")
    p.add_argument("--n-prompt-tokens", type=int, default=0, help="强制保留的 prompt token 数")
    p.add_argument("--warmup-epochs", type=int, default=3, help="阶段一训练轮数")
    p.add_argument("--warmup-lr", type=float, default=3e-4, help="阶段一学习率")
    p.add_argument("--finetune-epochs", type=int, default=10, help="阶段二训练轮数")
    p.add_argument("--finetune-lr", type=float, default=2e-5, help="阶段二学习率")
    p.add_argument("--batch-size", type=int, default=2, help="阶段二 batch size")
    p.add_argument("--warmup-output", default=str(WARMUP_OUTPUT), help="阶段一输出目录")
    p.add_argument("--output", default=str(FINETUNE_OUTPUT), help="阶段二输出目录")
    p.add_argument("--device", default="cpu", help="训练设备 (cuda/mps/cpu)")
    p.add_argument("--noise-threshold", type=float, default=0.22, help="噪声门控阈值")
    p.add_argument("--phase1-only", action="store_true", help="仅运行阶段一（门控预热）")
    p.add_argument("--phase2-only", action="store_true", help="仅运行阶段二（E2E 微调）")
    p.add_argument("--warmup-ckpt", default="", help="阶段二起点：已预热的模型目录（--phase2-only 时用）")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    cfg = GatedGLiNERConfig(
        model_name=args.model,
        keep_k=args.keep_k,
        max_len=args.max_len,
        keep_k_start=args.keep_k_start,
        compress_after_layer=args.compress_layer,
        n_prompt_tokens=args.n_prompt_tokens,
        train_file=args.train_file,
        test_file=args.test_file,
        warmup_epochs=args.warmup_epochs,
        warmup_lr=args.warmup_lr,
        warmup_output_dir=args.warmup_output,
        finetune_epochs=args.finetune_epochs,
        finetune_lr=args.finetune_lr,
        finetune_batch_size=args.batch_size,
        finetune_output_dir=args.output,
        device=args.device,
        noise_threshold=args.noise_threshold,
        seed=args.seed,
    )

    # ---------- 检查数据文件 ----------
    if not Path(cfg.train_file).exists():
        print(f"ERROR: 训练文件不存在: {cfg.train_file}", file=sys.stderr)
        print("请先运行: python scripts/prepare_training_data.py", file=sys.stderr)
        sys.exit(1)
    if not Path(cfg.test_file).exists():
        print(f"WARNING: 测试文件不存在: {cfg.test_file}，阶段二不评估", file=sys.stderr)

    train_data = load_jsonl(cfg.train_file)
    test_data = load_jsonl(cfg.test_file) if Path(cfg.test_file).exists() else []
    print(f"训练样本: {len(train_data)}  测试样本: {len(test_data)}")

    # ---------- 加载模型 ----------
    if args.phase2_only:
        src = args.warmup_ckpt or cfg.warmup_output_dir
        if not Path(src).exists():
            print(f"ERROR: 找不到预热模型: {src}", file=sys.stderr)
            sys.exit(1)
        print(f"从预热检查点加载: {src}")
        gliner_model = GLiNER.from_pretrained(src)
        # 阶段一已保存的 gate_layer 权重
        gate_ckpt = Path(src) / "gate_layer.pt"
        if gate_ckpt.exists():
            attach_topk_gate(
                gliner_model,
                compress_after_layer=cfg.compress_after_layer,
                keep_k=cfg.keep_k,
                n_prompt_tokens=cfg.n_prompt_tokens,
            )
            enc_obj = gliner_model.model.token_rep_layer.bert_layer.model.encoder
            enc_obj.gate_layer.load_state_dict(torch.load(gate_ckpt, map_location="cpu"))
            print(f"  已加载 gate_layer 权重: {gate_ckpt}")
    else:
        print(f"加载基座模型: {cfg.model_name}")
        gliner_model = GLiNER.from_pretrained(cfg.model_name)

    # 关键：GLiNER 的 data processor 会按 config.max_len 硬截断（日志里 384 就来自这里）
    # Top-K 门控发生在 Encoder 内部，若这里先截断，则后续层永远看不到被截掉的 token。
    if hasattr(gliner_model, "config"):
        gliner_model.config.max_len = int(cfg.max_len)
    if hasattr(gliner_model, "data_processor") and hasattr(gliner_model.data_processor, "config"):
        gliner_model.data_processor.config.max_len = int(cfg.max_len)
    print(f"已设置 GLiNER max_len={cfg.max_len}（注意：超过该长度仍会被截断）")

    # ---------- 阶段一 ----------
    if not args.phase2_only:
        gliner_model = warmup_gate(gliner_model, cfg, train_data)

    if args.phase1_only:
        print("\n✅ 门控预热完成（--phase1-only 模式，跳过阶段二）")
        return

    # ---------- 阶段二 ----------
    if not args.phase2_only:
        # 阶段一后 compress_after_layer=999，需要重新激活真实截断
        attach_topk_gate(
            gliner_model,
            compress_after_layer=cfg.compress_after_layer,
            keep_k=cfg.keep_k,
            n_prompt_tokens=cfg.n_prompt_tokens,
        )

        # 恢复 gate_layer 权重（attach_topk_gate 会重新 attach，需重载）
        gate_ckpt = Path(cfg.warmup_output_dir) / "gate_layer.pt"
        if gate_ckpt.exists():
            enc_obj = gliner_model.model.token_rep_layer.bert_layer.model.encoder
            enc_obj.gate_layer.load_state_dict(torch.load(gate_ckpt, map_location="cpu"))
            print(f"  已恢复 gate_layer 权重: {gate_ckpt}")

    finetune_e2e(gliner_model, cfg, train_data, test_data)
    print("\n✅ 两阶段训练全部完成！")
    print(f"  最终模型: {cfg.finetune_output_dir}")


if __name__ == "__main__":
    main()
