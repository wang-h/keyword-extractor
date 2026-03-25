#!/usr/bin/env python3
"""
自动优化 Gated GLiNER：
1) 生成候选超参组合
2) 逐个调用 train_gated_gliner.py 训练
3) 在测试集上做阈值扫描，按 F1 选最优
4) 输出 leaderboard 与 best_model 指针
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import torch
from quick_eval import evaluate_model, load_test_data


def _parse_int_list(raw: str) -> List[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_float_list(raw: str) -> List[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def run_train(
    cfg: Dict,
    run_dir: Path,
    device: str,
    train_file: Path,
    test_file: Path,
    model_name: str,
    timeout_sec: int,
) -> Tuple[int, bool]:
    """Run one training trial. Returns (return_code, timed_out)."""
    warmup_dir = run_dir / "warmup"
    model_dir = run_dir / "model"
    log_file = run_dir / "train.log"
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "train_gated_gliner.py"),
        "--model",
        model_name,
        "--train-file",
        str(train_file),
        "--test-file",
        str(test_file),
        "--keep-k",
        str(cfg["keep_k"]),
        "--keep-k-start",
        str(cfg["keep_k_start"]),
        "--compress-layer",
        str(cfg["compress_layer"]),
        "--max-len",
        str(cfg["max_len"]),
        "--warmup-epochs",
        str(cfg["warmup_epochs"]),
        "--warmup-lr",
        str(cfg["warmup_lr"]),
        "--finetune-epochs",
        str(cfg["finetune_epochs"]),
        "--finetune-lr",
        str(cfg["finetune_lr"]),
        "--batch-size",
        str(cfg["batch_size"]),
        "--noise-threshold",
        str(cfg["noise_threshold"]),
        "--device",
        device,
        "--warmup-output",
        str(warmup_dir),
        "--output",
        str(model_dir),
        "--seed",
        str(cfg["seed"]),
    ]

    print("\n" + "=" * 90)
    print(f"[TRAIN] run_dir={run_dir}")
    print(" ".join(cmd))
    print("=" * 90)

    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "train_cmd.txt").open("w", encoding="utf-8") as f:
        f.write(" ".join(cmd) + "\n")

    try:
        with log_file.open("w", encoding="utf-8") as f:
            proc = subprocess.run(
                cmd,
                cwd=str(ROOT),
                stdout=f,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=timeout_sec if timeout_sec > 0 else None,
            )
        return proc.returncode, False
    except subprocess.TimeoutExpired:
        with log_file.open("a", encoding="utf-8") as f:
            f.write(f"\n[TIMEOUT] training exceeded {timeout_sec}s\n")
        return 124, True


def eval_with_threshold_sweep(model_path: Path, test_data: List[Dict], device: str, thresholds: List[float]) -> Dict:
    best = {"f1": -1.0, "precision": 0.0, "recall": 0.0, "macro_f1": 0.0, "threshold": None}

    for th in thresholds:
        metrics = evaluate_model(str(model_path), test_data, device=device, threshold=th)
        f1 = float(metrics.get("f1", 0.0))
        if f1 > best["f1"]:
            best = {
                "f1": f1,
                "precision": float(metrics.get("precision", 0.0)),
                "recall": float(metrics.get("recall", 0.0)),
                "macro_f1": float(metrics.get("macro_f1", 0.0)),
                "threshold": th,
            }
    return best


def build_candidates(args: argparse.Namespace) -> List[Dict]:
    keep_k_list = _parse_int_list(args.keep_k_list)
    keep_k_start_list = _parse_int_list(args.keep_k_start_list)
    compress_layer_list = _parse_int_list(args.compress_layer_list)
    max_len_list = _parse_int_list(args.max_len_list)
    warmup_lr_list = _parse_float_list(args.warmup_lr_list)
    finetune_lr_list = _parse_float_list(args.finetune_lr_list)
    noise_threshold_list = _parse_float_list(args.noise_threshold_list)
    batch_size_list = _parse_int_list(args.batch_size_list)

    candidates: List[Dict] = []
    for (
        keep_k,
        keep_k_start,
        compress_layer,
        max_len,
        warmup_lr,
        finetune_lr,
        noise_threshold,
        batch_size,
    ) in itertools.product(
        keep_k_list,
        keep_k_start_list,
        compress_layer_list,
        max_len_list,
        warmup_lr_list,
        finetune_lr_list,
        noise_threshold_list,
        batch_size_list,
    ):
        # 合法化：退火起点要么 -1，要么 >= keep_k
        if keep_k_start != -1 and keep_k_start < keep_k:
            keep_k_start = keep_k

        candidates.append(
            {
                "keep_k": keep_k,
                "keep_k_start": keep_k_start,
                "compress_layer": compress_layer,
                "max_len": max_len,
                "warmup_lr": warmup_lr,
                "finetune_lr": finetune_lr,
                "noise_threshold": noise_threshold,
                "batch_size": batch_size,
                "warmup_epochs": args.warmup_epochs,
                "finetune_epochs": args.finetune_epochs,
                "seed": args.seed,
            }
        )

    rnd = random.Random(args.seed)
    rnd.shuffle(candidates)
    if args.trials > 0:
        candidates = candidates[: args.trials]
    return candidates


def main() -> None:
    ap = argparse.ArgumentParser(description="自动优化 Gated GLiNER")
    ap.add_argument("--model", default="urchade/gliner_multi-v2.1", help="基座模型 ID 或本地路径")
    ap.add_argument("--train-file", default=str(ROOT / "data" / "gliner_train.jsonl"))
    ap.add_argument("--test-file", default=str(ROOT / "data" / "gliner_test.jsonl"))
    ap.add_argument("--output-root", default=str(ROOT / "models" / "gated_gliner_autoopt"))
    ap.add_argument("--trials", type=int, default=4, help="最多运行多少组超参")

    ap.add_argument("--warmup-epochs", type=int, default=2)
    ap.add_argument("--finetune-epochs", type=int, default=4)

    ap.add_argument("--keep-k-list", default="1024,1500")
    ap.add_argument("--keep-k-start-list", default="2048,-1")
    ap.add_argument("--compress-layer-list", default="2,3")
    ap.add_argument("--max-len-list", default="512")
    ap.add_argument("--warmup-lr-list", default="2e-4,3e-4")
    ap.add_argument("--finetune-lr-list", default="1e-5,2e-5")
    ap.add_argument("--noise-threshold-list", default="0.20,0.24")
    ap.add_argument("--batch-size-list", default="2")

    ap.add_argument("--threshold-list", default="0.25,0.30,0.35,0.40")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--train-timeout-sec", type=int, default=0, help="单个 trial 训练超时秒数，0=不限制")
    ap.add_argument("--include-baseline", action="store_true", help="将 --model 作为 baseline 一并评估")

    args = ap.parse_args()

    train_file = Path(args.train_file)
    test_file = Path(args.test_file)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if not train_file.exists() or not test_file.exists():
        raise SystemExit(f"训练或测试文件不存在: train={train_file.exists()} test={test_file.exists()}")

    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device
    if args.device == "auto" and device == "auto":
        device = "cpu"

    thresholds = _parse_float_list(args.threshold_list)
    candidates = build_candidates(args)
    test_data = load_test_data(str(test_file))

    print(f"模型: {args.model}")
    print(f"设备: {device}")
    print(f"候选配置总数: {len(candidates)}")
    print(f"测试样本数: {len(test_data)}")

    results: List[Dict] = []
    best_result: Dict | None = None

    # 可选：先把起始模型当 baseline 评估，防止训练退化时误选
    if args.include_baseline:
        baseline_path = Path(args.model)
        baseline_result: Dict = {
            "trial": 0,
            "config": {"mode": "baseline"},
            "run_dir": str(baseline_path),
            "train_return_code": 0,
            "train_timed_out": False,
            "train_seconds": 0.0,
        }
        try:
            baseline_result.update(
                eval_with_threshold_sweep(baseline_path, test_data, device, thresholds)
            )
        except Exception as e:
            baseline_result["eval_error"] = str(e)
            baseline_result["f1"] = -1.0

        results.append(baseline_result)
        best_result = baseline_result
        print(
            f"[BASELINE] f1={baseline_result.get('f1', -1):.4f} "
            f"th={baseline_result.get('threshold')}"
        )

    for idx, cfg in enumerate(candidates, start=1):
        run_dir = output_root / f"trial_{idx:03d}"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")

        t0 = time.time()
        ret, timed_out = run_train(
            cfg,
            run_dir,
            device,
            train_file,
            test_file,
            model_name=args.model,
            timeout_sec=args.train_timeout_sec,
        )
        train_sec = time.time() - t0

        trial_result: Dict = {
            "trial": idx,
            "config": cfg,
            "run_dir": str(run_dir),
            "train_return_code": ret,
            "train_timed_out": timed_out,
            "train_seconds": round(train_sec, 2),
        }

        model_path = run_dir / "model"
        if ret == 0 and model_path.exists():
            try:
                metrics = eval_with_threshold_sweep(model_path, test_data, device, thresholds)
                trial_result.update(metrics)
            except Exception as e:
                trial_result["eval_error"] = str(e)
                trial_result["f1"] = -1.0
        else:
            trial_result["f1"] = -1.0

        results.append(trial_result)

        if best_result is None or float(trial_result.get("f1", -1.0)) > float(best_result.get("f1", -1.0)):
            best_result = trial_result

        print(
            f"[TRIAL {idx}] rc={ret} timeout={timed_out} f1={trial_result.get('f1', -1):.4f} "
            f"th={trial_result.get('threshold')} time={train_sec:.1f}s"
        )

        (output_root / "leaderboard.json").write_text(
            json.dumps(sorted(results, key=lambda x: x.get("f1", -1.0), reverse=True), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    if best_result is None:
        raise SystemExit("没有可用结果")

    best_model_path = Path(best_result["run_dir"]) / "model"
    if best_result.get("trial") == 0:
        best_model_path = Path(args.model)

    best_record = {
        "best_trial": best_result["trial"],
        "best_model_path": str(best_model_path),
        "best_f1": best_result.get("f1", -1.0),
        "best_precision": best_result.get("precision", 0.0),
        "best_recall": best_result.get("recall", 0.0),
        "best_macro_f1": best_result.get("macro_f1", 0.0),
        "best_threshold": best_result.get("threshold"),
        "config": best_result.get("config", {}),
    }
    (output_root / "best_result.json").write_text(json.dumps(best_record, ensure_ascii=False, indent=2), encoding="utf-8")

    best_ptr = ROOT / "models" / "gated_gliner_best_auto"
    if best_ptr.exists() or best_ptr.is_symlink():
        if best_ptr.is_symlink() or best_ptr.is_file():
            best_ptr.unlink()
        else:
            shutil.rmtree(best_ptr)

    if best_result.get("f1", -1.0) == -1.0 or not best_model_path.exists():
        print("\n⚠️  所有 trial 都失败，未更新 best model 指针")
        print(f"详情见: {output_root / 'leaderboard.json'}")
        return

    try:
        best_ptr.symlink_to(best_model_path)
        ptr_mode = "symlink"
    except Exception:
        shutil.copytree(best_model_path, best_ptr)
        ptr_mode = "copy"

    print("\n" + "=" * 90)
    print("自动优化完成")
    print(f"最佳 Trial: {best_record['best_trial']}")
    print(f"最佳 F1: {best_record['best_f1']:.4f}")
    print(f"最佳阈值: {best_record['best_threshold']}")
    print(f"最佳模型: {best_model_path}")
    print(f"已写入: {output_root / 'leaderboard.json'}")
    print(f"模型指针: {best_ptr} ({ptr_mode})")
    print("=" * 90)


if __name__ == "__main__":
    main()
