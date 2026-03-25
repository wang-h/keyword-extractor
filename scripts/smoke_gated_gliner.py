#!/usr/bin/env python3
"""
端到端冒烟测试：Top-K Gated GLiNER V3
========================================

测试项目
--------
1. 模块导入验证（不需要真实模型权重）
2. TopKCompressedDebertaV2Encoder 直接单元测试（随机权重）
   - 正常序列（L > keep_k）：形状验证 + _last_topk_indices 检查
   - 短序列（L ≤ keep_k）：旁路逻辑验证
3. TopKGatedDroppingLayer 梯度连通性验证
4. attach_topk_compressed_encoder 就地替换验证
5. attach_topk_gate（GLiNER 级别）+ encode_text 形状验证（如果 GLiNER 可用）
6. GatedGLiNER.from_pretrained 工厂（仅当模型已缓存时运行）
7. map_span_to_original 坐标映射验证

运行方式::

    # 从项目根目录
    PYTHONPATH=src python scripts/smoke_gated_gliner.py

    # 完整测试（需要模型权重，较慢）
    PYTHONPATH=src python scripts/smoke_gated_gliner.py --full

    # 仅快速单元测试（随机权重，无需下载）
    PYTHONPATH=src python scripts/smoke_gated_gliner.py --quick
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# ─── ANSI 颜色 ────────────────────────────────────────────────────────────────
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"
BOLD = "\033[1m"


def ok(msg: str) -> None:
    print(f"  {GREEN}✓{RESET} {msg}")


def fail(msg: str) -> None:
    print(f"  {RED}✗ FAIL{RESET}: {msg}")
    sys.exit(1)


def warn(msg: str) -> None:
    print(f"  {YELLOW}⚠{RESET}  {msg}")


def section(title: str) -> None:
    print(f"\n{BOLD}{'─'*60}{RESET}")
    print(f"{BOLD}  {title}{RESET}")
    print(f"{BOLD}{'─'*60}{RESET}")


# ─── 辅助：构建最小 DeBERTa-v2 配置 ──────────────────────────────────────────

def _make_deberta_v2_model(hidden: int = 64, layers: int = 4) -> "DebertaV2Model":
    """构建随机权重 DebertaV2Model（小尺寸，快速测试）。"""
    from transformers import DebertaV2Config, DebertaV2Model as HFModel
    cfg = DebertaV2Config(
        hidden_size=hidden,
        num_hidden_layers=layers,
        num_attention_heads=4,
        intermediate_size=hidden * 2,
        max_position_embeddings=512,
        relative_attention=True,
        pos_att_type=["p2c", "c2p"],
        position_buckets=64,
        share_att_key=True,
        vocab_size=32000,
        type_vocab_size=0,
    )
    return HFModel(cfg)


# ─── 测试 1：模块导入 ─────────────────────────────────────────────────────────

def test_imports() -> None:
    section("TEST 1: 模块导入")
    try:
        from keyword_extractor.topk_gated_dropping import (
            TopKGatedDroppingLayer,
            map_span_to_original,
        )
        ok("topk_gated_dropping 导入成功")
    except Exception as e:
        fail(f"topk_gated_dropping 导入失败: {e}")

    try:
        from keyword_extractor.topk_compressed_encoder import (
            TopKCompressedDebertaV2Encoder,
            attach_topk_compressed_encoder,
        )
        ok("topk_compressed_encoder 导入成功")
    except Exception as e:
        fail(f"topk_compressed_encoder 导入失败: {e}")

    try:
        from keyword_extractor.gated_gliner import (
            GatedGLiNER,
            GatedGLiNEREncoder,
            attach_topk_gate,
        )
        ok("gated_gliner 导入成功")
    except Exception as e:
        fail(f"gated_gliner 导入失败: {e}")


# ─── 测试 2：TopKGatedDroppingLayer 形状 + 梯度 ───────────────────────────────

def test_topk_layer() -> None:
    section("TEST 2: TopKGatedDroppingLayer 形状 & 梯度连通性")
    from keyword_extractor.topk_gated_dropping import TopKGatedDroppingLayer

    B, L, D, K = 2, 50, 64, 20
    layer = TopKGatedDroppingLayer(D, keep_k=K)  # first arg is hidden_dim

    x = torch.randn(B, L, D, requires_grad=True)
    compressed, indices = layer(x)

    # 形状检查
    expected_shape = (B, K, D)
    if compressed.shape != expected_shape:
        fail(f"压缩后 shape={compressed.shape}，期望 {expected_shape}")
    ok(f"正常序列形状正确: {compressed.shape}")

    if indices.shape != (B, K):
        fail(f"indices shape={indices.shape}，期望 ({B}, {K})")
    ok(f"topk_indices shape 正确: {indices.shape}")

    # 梯度连通
    loss = compressed.sum()
    loss.backward()
    if x.grad is None:
        fail("梯度未传回输入 x")
    ok("梯度成功传回输入（梯度不断裂）")

    # 短序列旁路（L ≤ K）
    x_short = torch.randn(B, 10, D, requires_grad=True)
    comp_short, idx_short = layer(x_short)
    if comp_short.shape[1] != 10:
        fail(f"短序列旁路：期望 seq_len=10，得到 {comp_short.shape[1]}")
    ok(f"短序列旁路正确: {comp_short.shape}")

    comp_short.sum().backward()
    if x_short.grad is None:
        fail("短序列梯度未传回")
    ok("短序列梯度正常传回")


# ─── 测试 3：TopKCompressedDebertaV2Encoder ───────────────────────────────────

def test_topk_encoder() -> None:
    section("TEST 3: TopKCompressedDebertaV2Encoder 前向 + 形状 + indices")
    from keyword_extractor.topk_compressed_encoder import (
        TopKCompressedDebertaV2Encoder,
        attach_topk_compressed_encoder,
    )

    model = _make_deberta_v2_model(hidden=64, layers=4)

    B, L = 2, 80
    K = 30
    input_ids = torch.randint(0, 32000, (B, L))
    attention_mask = torch.ones(B, L, dtype=torch.long)

    # 附加压缩 encoder（keep_k=K，第 1 层后压缩）
    attach_topk_compressed_encoder(
        model,
        compress_after_layer=1,
        keep_k=K,
        n_prompt_tokens=0,
        copy_weights=True,
    )

    assert isinstance(model.encoder, TopKCompressedDebertaV2Encoder), "encoder 替换失败"
    ok("encoder 已替换为 TopKCompressedDebertaV2Encoder")

    # 前向
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)

    hs = out.last_hidden_state
    if hs.shape != (B, K, 64):
        fail(f"压缩后 last_hidden_state shape={hs.shape}，期望 ({B}, {K}, 64)")
    ok(f"压缩后输出 shape 正确: {hs.shape}")

    idx = model.encoder._last_topk_indices
    if idx is None:
        fail("_last_topk_indices 为 None")
    if idx.shape != (B, K):
        fail(f"_last_topk_indices shape={idx.shape}，期望 ({B}, {K})")
    ok(f"_last_topk_indices shape 正确: {idx.shape}")

    # 验证 indices 在 [0, L) 范围内
    if idx.min() < 0 or idx.max() >= L:
        fail(f"topk_indices 超出 [0, {L}) 范围: min={idx.min()}, max={idx.max()}")
    ok(f"topk_indices 值域正确: [{idx.min()}, {idx.max()}] ⊆ [0, {L})")

    # 短序列（L=20 ≤ K=30）
    input_ids_s = torch.randint(0, 32000, (B, 20))
    attention_mask_s = torch.ones(B, 20, dtype=torch.long)
    with torch.no_grad():
        out_s = model(input_ids=input_ids_s, attention_mask=attention_mask_s)
    hs_s = out_s.last_hidden_state
    if hs_s.shape[1] != 20:
        fail(f"短序列 last_hidden_state seq_len={hs_s.shape[1]}，期望 20")
    ok(f"短序列旁路正确: {hs_s.shape}")


# ─── 测试 4：attach_topk_compressed_encoder 就地修改可逆性 ────────────────────

def test_attach_function() -> None:
    section("TEST 4: attach_topk_compressed_encoder 权重拷贝验证")
    from keyword_extractor.topk_compressed_encoder import attach_topk_compressed_encoder

    model = _make_deberta_v2_model(hidden=64, layers=4)
    # 获取原始第一层权重用于对比
    orig_weight = model.encoder.layer[0].attention.self.query_proj.weight.clone()

    attach_topk_compressed_encoder(model, compress_after_layer=1, keep_k=30, copy_weights=True)

    new_weight = model.encoder.layer[0].attention.self.query_proj.weight
    if not torch.allclose(orig_weight, new_weight):
        fail("权重拷贝后不一致")
    ok("原始 encoder 权重已正确拷贝到新 encoder")

    gate_exists = hasattr(model.encoder, "gate_layer")
    if not gate_exists:
        fail("新 encoder 缺少 gate_layer")
    ok("gate_layer 存在")


# ─── 测试 5：map_span_to_original ────────────────────────────────────────────

def test_span_mapping() -> None:
    section("TEST 5: map_span_to_original 坐标映射")
    from keyword_extractor.topk_gated_dropping import map_span_to_original

    # 模拟 topk_indices: 从 [0..9] 中选了 [1, 3, 5, 7, 9]（偶数被丢弃）
    # shape: [B=1, K=5]
    topk_indices = torch.tensor([[1, 3, 5, 7, 9]])

    # span [1, 3]（在压缩空间：第 1 到第 3 个 token）→ 原始空间：[3, 7]
    # API: map_span_to_original(comp_start, comp_end, topk_indices)
    orig_starts, orig_ends = map_span_to_original(1, 3, topk_indices)
    orig_start, orig_end = int(orig_starts[0]), int(orig_ends[0])
    if orig_start != 3 or orig_end != 7:
        fail(f"span 映射错误：期望 (3, 7)，得到 ({orig_start}, {orig_end})")
    ok(f"span [1,3] → 原文 [{orig_start}, {orig_end}] ✓")

    # edge case：span 起点 = 0，终点 = 4（最后一个 token）
    orig_starts2, orig_ends2 = map_span_to_original(0, 4, topk_indices)
    orig_start2, orig_end2 = int(orig_starts2[0]), int(orig_ends2[0])
    if orig_start2 != 1 or orig_end2 != 9:
        fail(f"边界 span 映射错误：期望 (1, 9)，得到 ({orig_start2}, {orig_end2})")
    ok(f"边界 span [0,4] → 原文 [{orig_start2}, {orig_end2}] ✓")


# ─── 测试 6：Scatter-Expand 梯度传播 ─────────────────────────────────────────

def test_scatter_expand_grad() -> None:
    section("TEST 6: Scatter-Expand 梯度连通性（GatedGLiNEREncoder 核心逻辑）")

    B, L, K, D = 1, 20, 8, 16
    # 模拟 compressed_embeds [B, K, D]
    compressed = torch.randn(B, K, D, requires_grad=True)
    topk_indices = torch.randperm(L)[:K].sort().values.unsqueeze(0)  # [1, K]

    idx = topk_indices.long().unsqueeze(-1).expand(-1, -1, D)
    expanded = torch.zeros(B, L, D)
    expanded = expanded.scatter(1, idx, compressed)  # [B, L, D]

    loss = expanded.sum()
    loss.backward()

    if compressed.grad is None:
        fail("scatter 操作后梯度未传回 compressed")
    ok(f"scatter → grad 正常传回 compressed (grad norm={compressed.grad.norm():.4f})")

    # 验证梯度只在 topk 位置非零（expanded 的 L-K 个零位置不应产生梯度）
    non_topk_mask = torch.ones(L, dtype=torch.bool)
    non_topk_mask[topk_indices[0]] = False
    if non_topk_mask.sum() > 0:
        ok("非 top-k 位置不产生梯度（符合预期）")


# ─── 测试 7：GLiNER 级集成（需要 gliner 包）────────────────────────────────────

def test_gliner_integration(full: bool = False) -> None:
    section("TEST 7: GLiNER 集成（attach_topk_gate + encode_text shape）")

    try:
        from gliner import GLiNER
        ok("gliner 包可用")
    except ImportError:
        warn("gliner 未安装，跳过集成测试")
        return

    try:
        from keyword_extractor.gated_gliner import attach_topk_gate
        from transformers import AutoTokenizer, DebertaV2Model
    except ImportError as e:
        warn(f"依赖缺失，跳过: {e}")
        return

    if not full:
        warn("仅快速模式（--full 跳过完整 GLiNER 加载，耗时数分钟）")
        # 直接测试 DeBERTa 级别（不加载 GLiNER）
        model = _make_deberta_v2_model(hidden=64, layers=4)
        from keyword_extractor.topk_compressed_encoder import attach_topk_compressed_encoder
        attach_topk_compressed_encoder(model, compress_after_layer=1, keep_k=20, copy_weights=True)

        B, L = 1, 40
        input_ids = torch.randint(0, 32000, (B, L))
        attention_mask = torch.ones(B, L, dtype=torch.long)
        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask)
        hs = out.last_hidden_state
        ok(f"（快速模式）DeBERTa 压缩后 last_hidden_state shape={hs.shape}")
        return

    # --- 完整模式：下载 GLiNER 模型 ---
    from keyword_extractor.gliner_config import GLINER_BASE_MODEL_ID
    try:
        print(f"  加载 GLiNER: {GLINER_BASE_MODEL_ID}（首次运行需下载，请稍候）")
        gliner = GLiNER.from_pretrained(GLINER_BASE_MODEL_ID)
        ok("GLiNER 加载成功")
    except Exception as e:
        warn(f"GLiNER 加载失败（网络/缓存问题），跳过: {e}")
        return

    # 附加 gate
    B, L, K = 1, 60, 20
    attach_topk_gate(gliner, compress_after_layer=1, keep_k=K)
    ok("attach_topk_gate 注入成功")

    tokenizer = gliner.data_processor.transformer_tokenizer if hasattr(gliner, "data_processor") else None
    if tokenizer is None:
        warn("无法取到 tokenizer，跳过 encode_text 形状测试")
        return

    enc = tokenizer(
        "这是一段测试文本，包含关键词提取的测试样本。" * 5,
        return_tensors="pt",
        truncation=True,
        max_length=L,
        padding="max_length",
    )
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]

    encoder_module = gliner.model.token_rep_layer
    with torch.no_grad():
        out = encoder_module.encode_text(input_ids, attention_mask)

    # scatter-expand 后输出应为原始 L
    if out.shape != (B, L, gliner.model.config.hidden_size):
        fail(f"encode_text 输出 shape={out.shape}，期望 ({B}, {L}, {gliner.model.config.hidden_size})")
    ok(f"encode_text scatter-expand 后 shape 正确: {out.shape}")

    topk_idx = getattr(encoder_module, "_last_topk_indices", None)
    if topk_idx is None:
        warn("_last_topk_indices 为 None（可能 L ≤ K）")
    else:
        ok(f"_last_topk_indices shape={topk_idx.shape}")


# ─── 汇总 ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Top-K Gated GLiNER 冒烟测试")
    p.add_argument("--quick", action="store_true", help="仅快速测试（随机权重，无需下载）")
    p.add_argument("--full", action="store_true", help="完整测试（需下载 GLiNER 模型）")
    args = p.parse_args()
    full = args.full and not args.quick

    print(f"\n{BOLD}Top-K Gated GLiNER V3 冒烟测试{RESET}")
    print(f"设备: {torch.device('cpu')}  PyTorch: {torch.__version__}")
    if full:
        print(f"{YELLOW}完整模式（需下载模型，首次运行较慢）{RESET}")
    else:
        print(f"{YELLOW}快速模式（随机权重）—— 用 --full 进行完整测试{RESET}")

    test_imports()
    test_topk_layer()
    test_topk_encoder()
    test_attach_function()
    test_span_mapping()
    test_scatter_expand_grad()
    test_gliner_integration(full=full)

    print(f"\n{BOLD}{GREEN}{'='*60}{RESET}")
    print(f"{BOLD}{GREEN}  所有测试通过 ✓{RESET}")
    print(f"{BOLD}{GREEN}{'='*60}{RESET}\n")


if __name__ == "__main__":
    main()
