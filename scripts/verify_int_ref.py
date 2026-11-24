#!/usr/bin/env python3
"""Compare PyTorch PTQ vs NumPy integer reference; optional layer dumps."""

from __future__ import annotations

import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.aconvnet import AConvNet
from quantize.export import load_package, package_from_checkpoint
from quantize.int_ref import AConvNetIntReference
from quantize.ptq import apply_ptq, calibrate_activation_scales, wrap_ptq
from scripts.train_baseline import build_loaders, pick_device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "runs/baseline_chen_soc/seed_12321/best.pt",
    )
    parser.add_argument(
        "--int8-dir",
        type=Path,
        default=ROOT / "runs/quant_chen/int8_package",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "processed/aconvnet_patches_chen",
    )
    parser.add_argument("--out-dir", type=Path, default=ROOT / "runs/quant_chen/int_ref_verify")
    parser.add_argument("--max-samples", type=int, default=64)
    parser.add_argument("--full-int8", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    device = pick_device(args.device)
    ns = Namespace(
        data_dir=args.data_dir,
        dataset="auto",
        batch_size=args.batch_size,
        workers=0,
        per_channel_scale=False,
    )
    _, test_loader, _, _ = build_loaders(ns, 12321, device, per_channel_scale=False)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = AConvNet(
        num_classes=len(ckpt.get("class_names") or []) or 10,
        in_channels=int(ckpt.get("in_channels", 2)),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])

    if args.int8_dir.is_dir():
        pkg = load_package(args.int8_dir)
    else:
        train_loader, _, _, _ = build_loaders(ns, 12321, device, per_channel_scale=False)
        pkg = package_from_checkpoint(
            args.checkpoint, calib_loader=train_loader, device=device
        )

    ref = AConvNetIntReference(pkg)
    act_scales = calibrate_activation_scales(model, test_loader, device, max_batches=64)
    state = apply_ptq(model, act_scales, per_channel_weights=pkg.weight_granularity == "per_channel")
    ptq = wrap_ptq(model, state, quantize_activations=args.full_int8).to(device)
    ptq.eval()

    seen = 0
    exact_logit = 0
    near_exact = 0
    same_class = 0
    max_abs_diff: list[float] = []
    args.out_dir.mkdir(parents=True, exist_ok=True)
    first_dump: dict | None = None

    for images, labels in test_loader:
        if seen >= args.max_samples:
            break
        take = min(images.size(0), args.max_samples - seen)
        images = images[:take].to(device)
        labels = labels[:take]

        with torch.no_grad():
            logits_pt = ptq(images).cpu().numpy()

        x_np = images.cpu().numpy()
        dump_this: dict | None = {} if seen == 0 and args.max_samples >= 1 else None
        if args.full_int8:
            logits_np = ref.forward_full_int8(x_np, dump=dump_this)
        else:
            logits_np = ref.forward_weight_only(x_np, dump=dump_this)
        if dump_this:
            first_dump = dump_this

        for i in range(take):
            a, b = logits_pt[i], logits_np[i]
            if np.allclose(a, b, rtol=0, atol=0, equal_nan=True):
                exact_logit += 1
            if np.allclose(a, b, rtol=0, atol=1e-4):
                near_exact += 1
            if int(a.argmax()) == int(b.argmax()):
                same_class += 1
            max_abs_diff.append(float(np.max(np.abs(a - b))))
        seen += take

    report = {
        "mode": "full_int8" if args.full_int8 else "weight_only_fp_act",
        "num_samples": seen,
        "logit_bit_exact_count": exact_logit,
        "logit_bit_exact_fraction": exact_logit / max(seen, 1),
        "logit_near_exact_atol_1e-4_fraction": near_exact / max(seen, 1),
        "same_argmax_fraction": same_class / max(seen, 1),
        "max_abs_diff_mean": float(np.mean(max_abs_diff)),
        "max_abs_diff_p95": float(np.percentile(max_abs_diff, 95)),
    }
    report_name = "verify_report_full_int8.json" if args.full_int8 else "verify_report.json"
    (args.out_dir / report_name).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

    if first_dump is not None:
        dump_dir = args.out_dir / "layer_dump_sample0"
        dump_dir.mkdir(exist_ok=True)
        for k, v in first_dump.items():
            np.save(dump_dir / f"{k}.npy", v)


if __name__ == "__main__":
    main()