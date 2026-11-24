#!/usr/bin/env python3
"""Measure FP32 baseline resource use (size, latency) for embedded deployment comparison."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.aconvnet import AConvNet


def fp32_weight_bytes(state_dict: dict) -> int:
    total = 0
    for v in state_dict.values():
        if torch.is_tensor(v):
            total += v.numel() * v.element_size()
    return total


def int8_weight_bytes_estimate(state_dict: dict, scale_bytes_per_tensor: int = 4) -> int:
    """Weights INT8 + one FP32 scale per weight tensor (deployment-oriented lower bound)."""
    n = 0
    tensors = 0
    for v in state_dict.values():
        if torch.is_tensor(v) and v.dtype.is_floating_point:
            n += v.numel()
            tensors += 1
    return n + tensors * scale_bytes_per_tensor


@torch.no_grad()
def latency_batch1(
    model: torch.nn.Module,
    device: torch.device,
    in_channels: int,
    input_size: int,
    *,
    warmup: int,
    repeats: int,
) -> dict[str, float]:
    model.eval()
    x = torch.randn(1, in_channels, input_size, input_size, device=device)
    for _ in range(warmup):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    times_ms: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times_ms.append((time.perf_counter() - t0) * 1000.0)
    arr = np.array(times_ms, dtype=np.float64)
    return {
        "latency_ms_mean": float(arr.mean()),
        "latency_ms_std": float(arr.std()),
        "latency_ms_p95": float(np.percentile(arr, 95)),
        "latency_ms_max": float(arr.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark FP32 A-ConvNet checkpoint")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "runs" / "baseline_aconvnet" / "seed_12321" / "best.pt",
    )
    parser.add_argument("--input-size", type=int, default=88)
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--out", type=Path, default=None, help="Write JSON next to checkpoint if omitted")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    class_names = ckpt.get("class_names") or []
    in_channels = int(ckpt.get("in_channels", 2))
    num_classes = len(class_names) if class_names else 10

    model = AConvNet(num_classes=num_classes, in_channels=in_channels)
    model.load_state_dict(ckpt["model_state"])
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    model.to(device)

    state = ckpt["model_state"]
    fp32_b = fp32_weight_bytes(state)
    int8_b = int8_weight_bytes_estimate(state)
    ckpt_bytes = args.checkpoint.stat().st_size
    num_params = sum(p.numel() for p in model.parameters())

    lat = latency_batch1(
        model,
        device,
        in_channels,
        args.input_size,
        warmup=args.warmup,
        repeats=args.repeats,
    )

    report = {
        "checkpoint": str(args.checkpoint),
        "test_acc_at_save": ckpt.get("test_acc"),
        "epoch_at_save": ckpt.get("epoch"),
        "num_classes": num_classes,
        "in_channels": in_channels,
        "input_size": args.input_size,
        "num_params": num_params,
        "checkpoint_bytes": ckpt_bytes,
        "fp32_weight_bytes": fp32_b,
        "int8_weight_bytes_estimate": int8_b,
        "compression_ratio_weights": round(fp32_b / max(int8_b, 1), 3),
        "device": str(device),
        **lat,
    }
    out_path = args.out or (args.checkpoint.parent / "resource_baseline.json")
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
