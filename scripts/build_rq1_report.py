#!/usr/bin/env python3
"""Aggregate RQ1 numbers (FP32 / PTQ / QAT) into one report for slides."""

from __future__ import annotations

import argparse
import json
import sys
import time
from argparse import Namespace
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quantize.ptq import apply_ptq, calibrate_activation_scales, wrap_ptq
from scripts.train_baseline import build_loaders, evaluate, pick_device, set_seed
from scripts.quantize_eval import load_fp32_model


def pct(x: float) -> str:
    return f"{x * 100:.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "runs/baseline_chen_soc/seed_12321/best.pt",
    )
    parser.add_argument(
        "--qat-checkpoint",
        type=Path,
        default=ROOT / "runs/qat_chen_weight_only/best.pt",
    )
    parser.add_argument(
        "--full-int8-json",
        type=Path,
        default=ROOT / "runs/quant_chen/full_int8_calib.json",
    )
    parser.add_argument("--out-dir", type=Path, default=ROOT / "runs/rq1")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=12321)
    args = parser.parse_args()

    set_seed(args.seed)
    device = pick_device(args.device)
    data_dir = ROOT / "processed/aconvnet_patches_chen"
    ns = Namespace(
        data_dir=data_dir,
        dataset="auto",
        batch_size=100,
        workers=4,
        per_channel_scale=False,
    )
    train_loader, test_loader, _, _ = build_loaders(ns, args.seed, device, per_channel_scale=False)

    fp32, ckpt = load_fp32_model(args.checkpoint, device)
    fp32_acc = evaluate(fp32, test_loader, device)

    t0 = time.time()
    act = calibrate_activation_scales(fp32, train_loader, device, max_batches=64)
    st_pc = apply_ptq(fp32, act, per_channel_weights=True)
    ptq_pc = wrap_ptq(fp32, st_pc, quantize_activations=False).to(device)
    ptq_pc_acc = evaluate(ptq_pc, test_loader, device)

    fp32_t, _ = load_fp32_model(args.checkpoint, device)
    st_pt = apply_ptq(fp32_t, act, per_channel_weights=False)
    ptq_pt = wrap_ptq(fp32_t, st_pt, quantize_activations=False).to(device)
    ptq_pt_acc = evaluate(ptq_pt, test_loader, device)

    qat_acc = None
    qat_note = "missing"
    if args.qat_checkpoint.is_file():
        qckpt = torch.load(args.qat_checkpoint, map_location=device, weights_only=False)
        qat_acc = float(qckpt.get("best_test_acc", 0))
        qat_note = str(args.qat_checkpoint)

    full_best = None
    if args.full_int8_json.is_file():
        full = json.loads(args.full_int8_json.read_text(encoding="utf-8"))
        rows = [r for r in full.get("results", []) if r.get("name") != "fp32"]
        if rows:
            full_best = max(rows, key=lambda r: r["test_acc"])

    verify_ref = ROOT / "runs/quant_chen/int_ref_verify/verify_report.json"
    verify_cpp = ROOT / "runs/quant_chen/cpp_verify/cpp_verify_report.json"
    correctness = {}
    if verify_ref.is_file():
        correctness["numpy_ref_weight_only"] = json.loads(verify_ref.read_text())
    if verify_cpp.is_file():
        correctness["cpp_vs_numpy"] = json.loads(verify_cpp.read_text())

    table = [
        {
            "method": "FP32",
            "weight_granularity": "—",
            "activations": "FP32",
            "test_acc": fp32_acc,
            "drop_pp": 0.0,
        },
        {
            "method": "PTQ",
            "weight_granularity": "per_channel",
            "activations": "FP32 (weight-only)",
            "test_acc": ptq_pc_acc,
            "drop_pp": (fp32_acc - ptq_pc_acc) * 100,
        },
        {
            "method": "PTQ",
            "weight_granularity": "per_tensor",
            "activations": "FP32 (weight-only)",
            "test_acc": ptq_pt_acc,
            "drop_pp": (fp32_acc - ptq_pt_acc) * 100,
        },
    ]
    if qat_acc is not None:
        table.append(
            {
                "method": "QAT",
                "weight_granularity": "per_channel",
                "activations": "FP32 (weight-only fake-quant)",
                "test_acc": qat_acc,
                "drop_pp": (fp32_acc - qat_acc) * 100,
            }
        )
    if full_best:
        table.append(
            {
                "method": "PTQ",
                "weight_granularity": "per_channel",
                "activations": f"INT8 ({full_best['name']})",
                "test_acc": full_best["test_acc"],
                "drop_pp": (fp32_acc - full_best["test_acc"]) * 100,
            }
        )

    md_lines = [
        "# RQ1 — Accuracy (Chen SOC, seed 12321)",
        "",
        "| Method | Weights | Activations | Test acc | Δ vs FP32 |",
        "|--------|---------|-------------|----------|-----------|",
    ]
    for row in table:
        md_lines.append(
            f"| {row['method']} | {row['weight_granularity']} | {row['activations']} | "
            f"{pct(row['test_acc'])} | {-row['drop_pp']:+.2f}pp |"
        )
    md_lines += [
        "",
        "## RQ2 handoff (correctness)",
        "",
        f"- Weight-only ref verify: `{verify_ref}`",
        f"- C++ vs NumPy: `{verify_cpp}`",
        "",
        "## Export",
        "",
        f"- Canonical INT8 package: `runs/quant_chen/int8_package/`",
        f"- QAT backbone for export: `{qat_note}`",
        f"- Rounding contract: `docs/ROUNDING.md`",
    ]

    summary = {
        "rq1_table": table,
        "fp32_test_acc": fp32_acc,
        "checkpoint": str(args.checkpoint),
        "qat_checkpoint": qat_note,
        "full_int8_best": full_best,
        "correctness": correctness,
        "generated_seconds": time.time() - t0,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "rq1_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.out_dir / "RQ1_TABLE.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print("\n".join(md_lines))


if __name__ == "__main__":
    main()