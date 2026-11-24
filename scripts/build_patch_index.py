#!/usr/bin/env python3
"""Materialize 49 train patches per chip (94x94 on 100 center) for faster SOC training."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mstar.dataset import _scale_channels
from mstar.preprocess import center_crop


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=ROOT / "processed" / "aconvnet_128")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "processed" / "aconvnet_patches")
    parser.add_argument("--mid-size", type=int, default=100)
    parser.add_argument("--patch-size", type=int, default=94)
    parser.add_argument(
        "--per-channel-scale",
        action="store_true",
        help="Min-max each channel (legacy); Chen AConvNet-PyTorch SOC uses raw floats",
    )
    args = parser.parse_args()

    x = np.load(args.data_dir / "train_x.npy", mmap_mode="r")
    y = np.load(args.data_dir / "train_y.npy")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    grid = args.mid_size - args.patch_size + 1
    patches_per = grid * grid
    out_x = np.lib.format.open_memmap(
        args.out_dir / "train_x.npy",
        mode="w+",
        dtype=np.float32,
        shape=(len(y) * patches_per, 2, args.patch_size, args.patch_size),
    )
    out_y = np.repeat(y, patches_per)

    idx = 0
    for i in range(len(y)):
        chip = np.array(x[i], dtype=np.float32)
        chip = center_crop(chip, 128)
        chip = center_crop(chip, args.mid_size)
        for pid in range(patches_per):
            top = pid // grid
            left = pid % grid
            patch = chip[:, top : top + args.patch_size, left : left + args.patch_size]
            out_x[idx] = _scale_channels(patch) if args.per_channel_scale else patch
            idx += 1
        if (i + 1) % 200 == 0:
            print(f"packed {i + 1}/{len(y)}")

    np.save(args.out_dir / "train_y.npy", out_y)
    test_x = np.load(args.data_dir / "test_x.npy")
    test_y = np.load(args.data_dir / "test_y.npy")
    np.save(args.out_dir / "test_x.npy", test_x)
    np.save(args.out_dir / "test_y.npy", test_y)

    cfg = json.loads((args.data_dir / "config.json").read_text(encoding="utf-8"))
    cfg.update(
        {
            "format": "aconvnet_patches",
            "patches_per_chip": patches_per,
            "patch_size": args.patch_size,
            "input_size": 88,
            "train_count": int(len(out_y)),
            "per_channel_scale": args.per_channel_scale,
            "preprocess_ref": "jangsoopark/AConvNet-pytorch SOC (chip 100, patch 94, 49 stride-1)",
        }
    )
    (args.out_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print("train_x", out_x.shape, "train_y", out_y.shape)


if __name__ == "__main__":
    main()
