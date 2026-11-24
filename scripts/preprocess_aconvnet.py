#!/usr/bin/env python3
"""Preprocess SOC chips for Chen A-ConvNet (2ch mag+phase, 128 center crop)."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mstar.io import read_mstar_pair
from mstar.preprocess import center_crop
from mstar.soc import CLASS_NAMES, collect_soc_samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "processed" / "aconvnet_128")
    parser.add_argument("--chip-size", type=int, default=128)
    args = parser.parse_args()

    samples = collect_soc_samples(args.data_root)
    by_split: dict[str, list] = {"train": [], "test": []}
    for s in samples:
        by_split[s.split].append(s)

    class_to_idx = {c: i for i, c in enumerate(CLASS_NAMES)}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    failed: list[str] = []
    rows_meta: list[dict] = []

    def build(split: str) -> tuple[np.ndarray, np.ndarray]:
        xs: list[np.ndarray] = []
        ys: list[int] = []
        for rec in sorted(by_split[split], key=lambda r: (r.class_name, str(r.path))):
            try:
                chip = read_mstar_pair(rec.path)
                x = center_crop(chip.tensor, args.chip_size)
            except Exception as exc:  # noqa: BLE001
                failed.append(f"{rec.path}: {exc}")
                continue
            xs.append(x)
            ys.append(class_to_idx[rec.class_name])
            rows_meta.append(
                {
                    "split": split,
                    "class": rec.class_name,
                    "label": class_to_idx[rec.class_name],
                    "path": str(rec.path),
                    "target_az": chip.target_az,
                }
            )
        return np.stack(xs, axis=0).astype(np.float32), np.array(ys, dtype=np.int64)

    train_x, train_y = build("train")
    test_x, test_y = build("test")

    np.save(args.out_dir / "train_x.npy", train_x)
    np.save(args.out_dir / "train_y.npy", train_y)
    np.save(args.out_dir / "test_x.npy", test_x)
    np.save(args.out_dir / "test_y.npy", test_y)

    config = {
        "format": "aconvnet_mag_phase",
        "chip_size": args.chip_size,
        "input_size": 88,
        "in_channels": 2,
        "class_names": list(CLASS_NAMES),
        "class_to_idx": class_to_idx,
        "train_count": int(len(train_y)),
        "test_count": int(len(test_y)),
    }
    (args.out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    with (args.out_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["split", "class", "label", "path", "target_az"]
        )
        writer.writeheader()
        writer.writerows(rows_meta)

    print("train", train_x.shape, "test", test_x.shape)
    for split, y in [("train", train_y), ("test", test_y)]:
        c = Counter(int(v) for v in y)
        print(split, len(y), {CLASS_NAMES[i]: c.get(i, 0) for i in range(10)})
    if failed:
        (args.out_dir / "failed.txt").write_text("\n".join(failed), encoding="utf-8")
        print("failed", len(failed))


if __name__ == "__main__":
    main()
