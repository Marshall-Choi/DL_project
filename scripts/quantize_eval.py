#!/usr/bin/env python3
"""PTQ and QAT for Chen A-ConvNet; compare INT8 fake-quant vs FP32 checkpoint."""

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
from quantize.ptq import apply_ptq, calibrate_activation_scales, wrap_ptq
from quantize.qat_model import QATAConvNet
from scripts.train_qat import train_qat as run_qat_training
from scripts.train_baseline import build_loaders, evaluate, pick_device, set_seed


def load_fp32_model(ckpt_path: Path, device: torch.device) -> tuple[AConvNet, dict]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    class_names = ckpt.get("class_names") or []
    num_classes = len(class_names) if class_names else 10
    in_channels = int(ckpt.get("in_channels", 2))
    model = AConvNet(num_classes=num_classes, in_channels=in_channels).to(device)
    model.load_state_dict(ckpt["model_state"])
    return model, ckpt


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
    parser.add_argument("--out-dir", type=Path, default=ROOT / "runs/quant_chen")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--calib-batches", type=int, default=64)
    parser.add_argument("--per-channel-weights", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--ptq-activations",
        action="store_true",
        help="Also fake-quant activations (default: weight-only PTQ)",
    )
    parser.add_argument("--skip-ptq", action="store_true")
    parser.add_argument("--skip-qat", action="store_true")
    parser.add_argument(
        "--qat-mode",
        choices=("weight_only", "full", "full_fixed_act"),
        default="weight_only",
        help="QAT fake-quant path (weight-only matches INT8 deploy)",
    )
    parser.add_argument("--qat-epochs", type=int, default=15)
    parser.add_argument("--qat-lr", type=float, default=1e-4)
    parser.add_argument(
        "--qat-freeze-backbone",
        action="store_true",
        help="Train only activation scales (default: fine-tune full network)",
    )
    parser.add_argument("--seed", type=int, default=12321)
    args = parser.parse_args()

    set_seed(args.seed)
    device = pick_device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    ns = Namespace(
        data_dir=args.data_dir,
        dataset="auto",
        batch_size=args.batch_size,
        workers=args.workers,
        per_channel_scale=False,
    )
    train_loader, test_loader, _, _ = build_loaders(ns, args.seed, device, per_channel_scale=False)

    fp32, ckpt = load_fp32_model(args.checkpoint, device)
    fp32_acc = evaluate(fp32, test_loader, device)
    print(f"FP32 test acc: {fp32_acc * 100:.4f}%")

    report: dict = {
        "checkpoint": str(args.checkpoint),
        "data_dir": str(args.data_dir),
        "fp32_test_acc": fp32_acc,
        "per_channel_weights": args.per_channel_weights,
        "calib_batches": args.calib_batches,
        "fp32_ckpt_test_acc": ckpt.get("test_acc"),
    }

    if not args.skip_ptq:
        t0 = time.time()
        act_scales = calibrate_activation_scales(
            fp32, train_loader, device, max_batches=args.calib_batches
        )
        state = apply_ptq(fp32, act_scales, per_channel_weights=args.per_channel_weights)
        ptq_model = wrap_ptq(
            fp32, state, quantize_activations=args.ptq_activations
        ).to(device)
        ptq_acc = evaluate(ptq_model, test_loader, device)
        mode = "weights+acts" if args.ptq_activations else "weights_only"
        print(
            f"PTQ ({state.weight_granularity}, {mode}) test acc: "
            f"{ptq_acc * 100:.4f}% ({time.time() - t0:.1f}s)"
        )
        report["ptq"] = {
            "test_acc": ptq_acc,
            "weight_granularity": state.weight_granularity,
            "quantize_activations": args.ptq_activations,
            "acc_drop_vs_fp32": fp32_acc - ptq_acc,
        }
        torch.save(
            {
                "ptq_state": {
                    "weight_granularity": state.weight_granularity,
                    "weight_scales": {k: v.cpu() for k, v in state.weight_scales.items()},
                    "act_scales": {k: v.cpu() for k, v in state.act_scales.items()},
                },
                "ptq_test_acc": ptq_acc,
            },
            args.out_dir / "ptq_state.pt",
        )

    if not args.skip_qat:
        in_ch = int(ckpt.get("in_channels", 2))
        num_classes = len(ckpt.get("class_names") or []) or 10
        act_init = None
        if args.qat_mode != "weight_only":
            raw = calibrate_activation_scales(
                fp32, train_loader, device, max_batches=args.calib_batches
            )
            act_init = {k: float(raw[k].cpu()) for k in ("in", "conv1", "conv2", "conv3", "conv4")}
        qat = QATAConvNet(
            num_classes=num_classes,
            in_channels=in_ch,
            per_channel_weights=args.per_channel_weights,
            mode=args.qat_mode,
            act_scale_init=act_init,
        )
        qat.load_fp32_state(ckpt["model_state"])
        best_qat, _, _ = run_qat_training(
            qat,
            train_loader,
            test_loader,
            device,
            epochs=args.qat_epochs,
            lr=args.qat_lr,
            lr_step=max(args.qat_epochs // 2, 1),
            lr_decay=0.1,
            momentum=0.9,
            weight_decay=0.0,
            optimizer="adam",
            freeze_backbone=args.qat_freeze_backbone,
        )
        report["qat"] = {
            "mode": args.qat_mode,
            "best_test_acc": best_qat,
            "epochs": args.qat_epochs,
            "lr": args.qat_lr,
            "acc_drop_vs_fp32": fp32_acc - best_qat,
        }
        torch.save(
            {"model_state": qat.backbone.state_dict(), "qat_test_acc": best_qat},
            args.out_dir / "qat_best.pt",
        )

    (args.out_dir / "quantize_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
