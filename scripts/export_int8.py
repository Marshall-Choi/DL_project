#!/usr/bin/env python3
"""Export INT8 weights + scales for NumPy reference and C++ engine."""

from __future__ import annotations

import argparse
import sys
from argparse import Namespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quantize.export import package_from_checkpoint, save_package
from scripts.train_baseline import build_loaders, pick_device


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
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "runs/quant_chen/int8_package",
    )
    parser.add_argument("--per-channel-weights", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--calib-batches", type=int, default=64)
    parser.add_argument(
        "--act-calib",
        choices=("ptq_weights", "fp32_graph"),
        default="ptq_weights",
        help="Activation scale calibration graph",
    )
    parser.add_argument(
        "--act-method",
        choices=("max", "p999", "p99"),
        default="max",
    )
    parser.add_argument(
        "--deploy-path",
        default="weight_only_fp_act",
        help="Label: weight_only_fp_act | full_int8",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()

    device = pick_device(args.device)
    ns = Namespace(
        data_dir=args.data_dir,
        dataset="auto",
        batch_size=args.batch_size,
        workers=args.workers,
        per_channel_scale=False,
    )
    train_loader, _, _, _ = build_loaders(ns, 12321, device, per_channel_scale=False)

    pkg = package_from_checkpoint(
        args.checkpoint,
        per_channel_weights=args.per_channel_weights,
        calib_loader=train_loader,
        device=device,
        calib_batches=args.calib_batches,
        act_calib=args.act_calib,
        act_method=args.act_method,
        deploy_path=args.deploy_path,
    )
    save_package(pkg, args.out_dir)
    print(f"Wrote INT8 package to {args.out_dir}")


if __name__ == "__main__":
    main()