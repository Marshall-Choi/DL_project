#!/usr/bin/env python3
"""Layer tensor dumps for C++ bit-exact checks (weight-only + full INT8)."""

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
from scripts.quantize_eval import load_fp32_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--int8-dir", type=Path, default=ROOT / "runs/quant_chen/int8_package")
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
    parser.add_argument("--out-dir", type=Path, default=ROOT / "runs/quant_chen/layer_dumps")
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--modes", nargs="+", default=("weight_only", "full_int8"))
    args = parser.parse_args()

    device = pick_device("cpu")
    ns = Namespace(
        data_dir=args.data_dir,
        dataset="auto",
        batch_size=args.num_samples,
        workers=0,
        per_channel_scale=False,
    )
    train_loader, test_loader, _, _ = build_loaders(ns, 12321, device, per_channel_scale=False)

    if args.int8_dir.is_dir():
        pkg = load_package(args.int8_dir)
    else:
        pkg = package_from_checkpoint(
            args.checkpoint, calib_loader=train_loader, device=device
        )

    ckpt_path = Path(pkg.checkpoint_source) if pkg.checkpoint_source else args.checkpoint
    if not ckpt_path.is_file():
        ckpt_path = args.checkpoint

    ref = AConvNetIntReference(pkg)
    fp32, _ = load_fp32_model(ckpt_path, device)
    act = calibrate_activation_scales(fp32, train_loader, device, max_batches=64)
    state = apply_ptq(fp32, act, per_channel_weights=True)
    ptq_full = wrap_ptq(fp32, state, quantize_activations=True).to(device)
    ptq_wo = wrap_ptq(fp32, state, quantize_activations=False).to(device)

    images, labels = next(iter(test_loader))
    images = images[: args.num_samples]
    labels = labels[: args.num_samples]
    x_np = images.numpy()

    manifest: dict = {"num_samples": args.num_samples, "modes": {}}
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for mode in args.modes:
        mode_dir = args.out_dir / mode
        mode_dir.mkdir(parents=True, exist_ok=True)
        np.save(mode_dir / "input.npy", x_np)
        np.save(mode_dir / "labels.npy", labels.numpy())

        dump: dict = {}
        if mode == "full_int8":
            logits_ref = ref.forward_full_int8(x_np, dump=dump)
            with torch.no_grad():
                logits_pt = ptq_full(images).cpu().numpy()
        else:
            logits_ref = ref.forward_weight_only(x_np, dump=dump)
            with torch.no_grad():
                logits_pt = ptq_wo(images).cpu().numpy()

        np.save(mode_dir / "logits_ref.npy", logits_ref)
        np.save(mode_dir / "logits_ptq.npy", logits_pt)
        for name, arr in dump.items():
            np.save(mode_dir / f"{name}.npy", arr)

        manifest["modes"][mode] = {
            "keys": sorted(dump.keys()) + ["logits_ref", "logits_ptq"],
            "ptq_ref_max_diff": float(np.max(np.abs(logits_pt - logits_ref))),
            "same_argmax": bool(np.all(logits_pt.argmax(1) == logits_ref.argmax(1))),
        }

    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()