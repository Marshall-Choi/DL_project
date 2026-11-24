#!/usr/bin/env python3
"""Sweep activation calibration methods for full INT8 PTQ (weight + act fake-quant)."""

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

from models.aconvnet import AConvNet
from quantize.ptq import (
    apply_ptq,
    calibrate_activation_scales,
    calibrate_activation_scales_ptq_weights,
    wrap_ptq,
)
from scripts.train_baseline import build_loaders, evaluate, pick_device, set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "runs/baseline_chen_soc/seed_12321/best.pt",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "processed/aconvnet_patches_chen",
    )
    parser.add_argument("--out", type=Path, default=ROOT / "runs/quant_chen/full_int8_calib.json")
    parser.add_argument("--calib-batches", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--seed", type=int, default=12321)
    args = parser.parse_args()

    set_seed(args.seed)
    device = pick_device(args.device)
    ns = Namespace(
        data_dir=args.data_dir,
        dataset="auto",
        batch_size=args.batch_size,
        workers=4,
        per_channel_scale=False,
    )
    train_loader, test_loader, _, _ = build_loaders(ns, args.seed, device, per_channel_scale=False)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = AConvNet(
        num_classes=len(ckpt.get("class_names") or []) or 10,
        in_channels=int(ckpt.get("in_channels", 2)),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    fp32_acc = evaluate(model, test_loader, device)

    sweeps: list[tuple[str, dict]] = [
        ("fp32", {"calib_fn": "fp32", "method": "max"}),
        ("act_max_fp32_weights", {"calib_fn": "fp32", "method": "max"}),
        ("act_p999_fp32_weights", {"calib_fn": "fp32", "method": "p999"}),
        ("act_max_ptq_weights", {"calib_fn": "ptq_weights", "method": "max"}),
        ("act_p999_ptq_weights", {"calib_fn": "ptq_weights", "method": "p999"}),
    ]

    results: list[dict] = [{"name": "fp32", "test_acc": fp32_acc}]
    for name, cfg in sweeps:
        if name == "fp32":
            continue
        t0 = time.time()
        if cfg["calib_fn"] == "fp32":
            act = calibrate_activation_scales(
                model, train_loader, device, max_batches=args.calib_batches, method=cfg["method"]
            )
        else:
            act = calibrate_activation_scales_ptq_weights(
                model,
                train_loader,
                device,
                max_batches=args.calib_batches,
                method=cfg["method"],
            )
        state = apply_ptq(model, act, per_channel_weights=True)
        ptq_full = wrap_ptq(model, state, quantize_activations=True).to(device)
        acc = evaluate(ptq_full, test_loader, device)
        row = {
            "name": name,
            "test_acc": acc,
            "acc_drop_vs_fp32": fp32_acc - acc,
            "calib_fn": cfg["calib_fn"],
            "method": cfg["method"],
            "act_scales": {k: float(v.cpu()) for k, v in act.items()},
            "seconds": time.time() - t0,
        }
        results.append(row)
        print(f"{name}: {acc * 100:.2f}% ({time.time() - t0:.1f}s)")

    out = {
        "fp32_test_acc": fp32_acc,
        "calib_batches": args.calib_batches,
        "checkpoint": str(args.checkpoint),
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    best = max(
        (r for r in results if r["name"] != "fp32"),
        key=lambda r: r["test_acc"],
        default=None,
    )
    if best:
        print("best full_int8:", best["name"], f"{best['test_acc'] * 100:.2f}%")


if __name__ == "__main__":
    main()