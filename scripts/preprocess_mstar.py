#!/usr/bin/env python3
"""Build SOC preprocessed MSTAR tensors for training."""

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

from mstar.io import read_mstar_chip
from mstar.preprocess import apply_global_norm, compute_train_stats, preprocess_magnitude
from mstar.soc import CLASS_NAMES, collect_soc_samples


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess MSTAR SOC dataset")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=ROOT / "data",
        help="Directory containing extracted MSTAR folders",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "processed" / "soc_88",
        help="Output directory for tensors and manifest",
    )
    parser.add_argument("--crop-size", type=int, default=88)
    parser.add_argument("--no-log1p", action="store_true")
    args = parser.parse_args()

    samples = collect_soc_samples(args.data_root)
    if not samples:
        raise SystemExit(f"No SOC samples under {args.data_root}")

    by_split: dict[str, list] = {"train": [], "test": []}
    for s in samples:
        by_split[s.split].append(s)

    class_to_idx = {c: i for i, c in enumerate(CLASS_NAMES)}
    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = args.out_dir / "manifest.csv"
    rows_meta: list[dict] = []
    failed: list[str] = []

    def load_split(split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        xs: list[np.ndarray] = []
        ys: list[int] = []
        azs: list[float] = []
        for rec in sorted(by_split[split], key=lambda r: (r.class_name, str(r.path))):
            try:
                chip = read_mstar_chip(rec.path)
                x = preprocess_magnitude(
                    chip.magnitude,
                    crop_size=args.crop_size,
                    log1p=not args.no_log1p,
                )
            except Exception as exc:  # noqa: BLE001 — collect all parse errors
                failed.append(f"{rec.path}: {exc}")
                continue
            xs.append(x)
            ys.append(class_to_idx[rec.class_name])
            azs.append(chip.target_az)
            rows_meta.append(
                {
                    "split": split,
                    "class": rec.class_name,
                    "label": class_to_idx[rec.class_name],
                    "path": str(rec.path),
                    "target_az": chip.target_az,
                    "measured_depression": chip.measured_depression,
                    "orig_rows": chip.rows,
                    "orig_cols": chip.cols,
                }
            )
        if not xs:
            raise SystemExit(f"No chips loaded for split={split}")
        return (
            np.stack(xs, axis=0),
            np.array(ys, dtype=np.int64),
            np.array(azs, dtype=np.float32),
        )

    print(f"Loading {len(by_split['train'])} train / {len(by_split['test'])} test paths...")
    x_train, y_train, az_train = load_split("train")
    x_test, y_test, az_test = load_split("test")

    mean, std = compute_train_stats(x_train)
    x_train_n = apply_global_norm(x_train, mean, std)
    x_test_n = apply_global_norm(x_test, mean, std)

    np.save(args.out_dir / "train_x.npy", x_train_n)
    np.save(args.out_dir / "train_y.npy", y_train)
    np.save(args.out_dir / "train_az.npy", az_train)
    np.save(args.out_dir / "test_x.npy", x_test_n)
    np.save(args.out_dir / "test_y.npy", y_test)
    np.save(args.out_dir / "test_az.npy", az_test)

    config = {
        "crop_size": args.crop_size,
        "log1p": not args.no_log1p,
        "global_mean": mean,
        "global_std": std,
        "class_names": list(CLASS_NAMES),
        "class_to_idx": class_to_idx,
        "train_count": int(len(y_train)),
        "test_count": int(len(y_test)),
        "data_root": str(args.data_root.resolve()),
    }
    (args.out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split",
                "class",
                "label",
                "path",
                "target_az",
                "measured_depression",
                "orig_rows",
                "orig_cols",
            ],
        )
        writer.writeheader()
        writer.writerows(rows_meta)

    print("\n=== Counts ===")
    for split, y in [("train", y_train), ("test", y_test)]:
        c = Counter(int(v) for v in y)
        print(f"{split}: {len(y)} total")
        for idx, name in enumerate(CLASS_NAMES):
            print(f"  {name:8} {c.get(idx, 0)}")

    fail_log = args.out_dir / "failed.txt"
    if failed:
        fail_log.write_text("\n".join(failed), encoding="utf-8")
        print(f"\nWarning: {len(failed)} files failed (see {fail_log})")
    elif fail_log.exists():
        fail_log.unlink()

    print(f"\nSaved to {args.out_dir}")
    print(f"  tensors: train_x {x_train_n.shape}, test_x {x_test_n.shape}, dtype {x_train_n.dtype}")
    print(f"  norm: mean={mean:.6f}, std={std:.6f}")


if __name__ == "__main__":
    main()
