#!/usr/bin/env python3
"""Build C++ engine (if needed) and compare logits vs NumPy integer reference."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quantize.export import load_package
from quantize.int_ref import AConvNetIntReference


def build(engine_dir: Path) -> Path:
    script = engine_dir / "build.sh"
    subprocess.check_call(["bash", str(script)], cwd=ROOT)
    exe = engine_dir / "build" / "aconvnet_int8"
    if not exe.is_file():
        raise FileNotFoundError(exe)
    return exe


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--int8-dir", type=Path, default=ROOT / "runs/quant_chen/int8_package"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "runs/quant_chen/int_ref_verify/layer_dump_sample0/input.npy",
    )
    parser.add_argument("--mode", choices=("weight_only", "full_int8"), default="weight_only")
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--out", type=Path, default=ROOT / "runs/quant_chen/cpp_verify")
    args = parser.parse_args()

    exe = ROOT / "engine/build/aconvnet_int8"
    if args.rebuild or not exe.is_file():
        exe = build(ROOT / "engine")

    if not args.input.is_file():
        raise FileNotFoundError(
            f"missing {args.input}; run scripts/verify_int_ref.py first for layer dumps"
        )

    x = np.load(args.input).astype(np.float32)
    if args.max_samples > 0:
        x = x[: args.max_samples]

    dump_np = args.out / "cpp_logits.npy"
    args.out.mkdir(parents=True, exist_ok=True)
    tmp_in = args.out / "_batch_in.npy"
    np.save(tmp_in, x)

    subprocess.check_call(
        [
            str(exe),
            "--package",
            str(args.int8_dir),
            "--input",
            str(tmp_in),
            "--output",
            str(dump_np),
            "--mode",
            args.mode,
        ],
        cwd=ROOT,
    )

    pkg = load_package(args.int8_dir)
    ref = AConvNetIntReference(pkg)
    if args.mode == "full_int8":
        ref_logits = ref.forward_full_int8(x)
    else:
        ref_logits = ref.forward_weight_only(x)

    cpp_logits = np.load(dump_np)
    ref_logits = np.atleast_2d(ref_logits)
    cpp_logits = np.atleast_2d(cpp_logits)
    per_exact = np.all(ref_logits == cpp_logits, axis=1)
    per_near = np.max(np.abs(ref_logits - cpp_logits), axis=1) <= 1e-5
    exact = int(np.sum(per_exact))
    near = int(np.sum(per_near))
    same = int(np.sum(ref_logits.argmax(axis=1) == cpp_logits.argmax(axis=1)))
    report = {
        "mode": args.mode,
        "num_samples": int(x.shape[0]),
        "ref_vs_cpp_bit_exact_samples": exact,
        "ref_vs_cpp_near_atol_1e-5_samples": near,
        "same_argmax_fraction": same / max(x.shape[0], 1),
        "max_abs_diff": float(np.max(np.abs(ref_logits - cpp_logits))),
        "mean_abs_diff": float(np.mean(np.abs(ref_logits - cpp_logits))),
    }
    (args.out / "cpp_verify_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()