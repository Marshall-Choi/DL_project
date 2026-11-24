#!/usr/bin/env python3
"""Recover full INT8 accuracy: per-channel percentile PTQ, then short QAT."""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from argparse import Namespace
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.aconvnet import AConvNet
from quantize.act_calib import collect_activation_samples, scales_from_collected
from quantize.ptq import apply_ptq, wrap_ptq
from quantize.qat_model import QATAConvNet
from scripts.train_baseline import build_loaders, evaluate, pick_device, set_seed


def load_model(ckpt_path: Path, device: torch.device) -> tuple[AConvNet, dict]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = AConvNet(
        num_classes=len(ckpt.get("class_names") or []) or 10,
        in_channels=int(ckpt.get("in_channels", 2)),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def train_full(
    qat: QATAConvNet,
    train_loader,
    test_loader,
    device: torch.device,
    *,
    epochs: int,
    backbone_lr: float,
    scale_lr: float,
) -> tuple[float, int]:
    qat.to(device)
    qat.backbone.dropout.p = 0.0
    opt = torch.optim.Adam(
        [
            {"params": list(qat.backbone.parameters()), "lr": backbone_lr},
            {"params": qat.act_scale_parameters(), "lr": scale_lr},
        ]
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    best_acc = -1.0
    best_epoch = -1
    best_state = copy.deepcopy(qat.state_dict())
    for epoch in range(1, epochs + 1):
        qat.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            opt.zero_grad(set_to_none=True)
            loss = criterion(qat(images), labels)
            loss.backward()
            opt.step()
        sched.step()
        acc = evaluate(qat, test_loader, device)
        print(f"QAT epoch={epoch:02d} test_acc={acc * 100:.2f}%")
        if acc > best_acc:
            best_acc = acc
            best_epoch = epoch
            best_state = copy.deepcopy(qat.state_dict())
    qat.load_state_dict(best_state)
    return best_acc, best_epoch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=ROOT / "runs/baseline_chen_soc/seed_12321/best.pt",
    )
    parser.add_argument(
        "--qat-init",
        type=Path,
        default=ROOT / "runs/qat_chen_weight_only/best.pt",
    )
    parser.add_argument("--out-dir", type=Path, default=ROOT / "runs/full_int8_recovered")
    parser.add_argument("--calib-batches", type=int, default=80)
    parser.add_argument("--qat-epochs", type=int, default=30)
    parser.add_argument("--backbone-lr", type=float, default=2e-5)
    parser.add_argument("--scale-lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=12321)
    args = parser.parse_args()

    set_seed(args.seed)
    device = pick_device(args.device)
    ns = Namespace(
        data_dir=ROOT / "processed/aconvnet_patches_chen",
        dataset="auto",
        batch_size=100,
        workers=4,
        per_channel_scale=False,
    )
    train_loader, test_loader, _, _ = build_loaders(ns, args.seed, device, per_channel_scale=False)
    model, ckpt = load_model(args.checkpoint, device)
    fp32_acc = evaluate(model, test_loader, device)
    print(f"FP32 test acc: {fp32_acc * 100:.2f}%")

    t0 = time.time()
    samples = collect_activation_samples(
        model, train_loader, device, max_batches=args.calib_batches, quantize_weights=True
    )
    print(f"collected activation samples in {time.time() - t0:.1f}s")

    configs = []
    for method in ("max", "p99", "p995", "p999"):
        for per_channel in (False, True):
            configs.append((method, per_channel, False))
        configs.append((method, False, True))  # input channel only

    rows = []
    best = None
    for method, per_channel, input_only in configs:
        name = f"{method}_ch" if per_channel else f"{method}_tensor"
        if input_only:
            name = f"{method}_inputch"
        scales = scales_from_collected(
            samples,
            method,
            per_channel=per_channel,
            per_channel_input_only=input_only,
        )
        state = apply_ptq(model, scales, per_channel_weights=True)
        ptq = wrap_ptq(model, state, quantize_activations=True).to(device)
        acc = evaluate(ptq, test_loader, device)
        row = {"name": name, "test_acc": acc, "drop_pp": (fp32_acc - acc) * 100}
        rows.append(row)
        print(f"PTQ {name}: {acc * 100:.2f}%")
        if best is None or acc > best["test_acc"]:
            best = {"name": name, "test_acc": acc, "scales": scales, "method": method,
                    "per_channel": per_channel, "input_only": input_only}

    print(f"best PTQ {best['name']}: {best['test_acc'] * 100:.2f}%")

    qat_acc = None
    qat_epoch = None
    qat_path = None
    if best["test_acc"] < fp32_acc - 0.002:
        qat_model, qckpt = load_model(args.qat_init, device)
        qat = QATAConvNet(
            num_classes=len(qckpt.get("class_names") or []) or 10,
            in_channels=int(qckpt.get("in_channels", 2)),
            mode="full",
            per_channel_weights=True,
            act_scale_init=best["scales"],
        )
        qat.load_fp32_state(qat_model.state_dict())
        qat_acc, qat_epoch = train_full(
            qat,
            train_loader,
            test_loader,
            device,
            epochs=args.qat_epochs,
            backbone_lr=args.backbone_lr,
            scale_lr=args.scale_lr,
        )
        print(f"QAT best epoch {qat_epoch}: {qat_acc * 100:.2f}%")
        args.out_dir.mkdir(parents=True, exist_ok=True)
        qat_path = args.out_dir / "qat_full_best.pt"
        torch.save(
            {
                "model_state": qat.backbone.state_dict(),
                "act_scales": {k: v.detach().cpu() for k, v in best["scales"].items()},
                "learned_act_scales": {
                    name: mod.log_scale.detach().exp().cpu()
                    for name, mod in (
                        ("in", qat.act_in),
                        ("conv1", qat.act1),
                        ("conv2", qat.act2),
                        ("conv3", qat.act3),
                        ("conv4", qat.act4),
                    )
                },
                "best_test_acc": qat_acc,
                "best_epoch": qat_epoch,
                "class_names": qckpt.get("class_names"),
                "in_channels": qckpt.get("in_channels", 2),
                "init_ptq": best["name"],
            },
            qat_path,
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "fp32_test_acc": fp32_acc,
        "ptq_rows": rows,
        "best_ptq": {k: best[k] for k in ("name", "test_acc", "method", "per_channel", "input_only")},
        "qat_best_test_acc": qat_acc,
        "qat_best_epoch": qat_epoch,
        "qat_checkpoint": str(qat_path) if qat_path else None,
        "scale_shapes": {k: list(v.shape) for k, v in best["scales"].items()},
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    torch.save({k: v.cpu() for k, v in best["scales"].items()}, args.out_dir / "best_ptq_scales.pt")
    print(json.dumps({k: summary[k] for k in summary if k != "ptq_rows"}, indent=2))


if __name__ == "__main__":
    main()