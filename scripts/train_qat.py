#!/usr/bin/env python3
"""QAT fine-tuning for A-ConvNet (weight-only or full fake-quant)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from argparse import Namespace
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.aconvnet import AConvNet
from quantize.ptq import calibrate_activation_scales
from quantize.qat_model import QATAConvNet
from scripts.train_baseline import build_loaders, evaluate, pick_device, set_seed


def load_fp32(ckpt_path: Path, device: torch.device) -> tuple[dict, AConvNet]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    nclass = len(ckpt.get("class_names") or []) or 10
    in_ch = int(ckpt.get("in_channels", 2))
    model = AConvNet(num_classes=nclass, in_channels=in_ch).to(device)
    model.load_state_dict(ckpt["model_state"])
    return ckpt, model


def train_qat(
    qat: QATAConvNet,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    *,
    epochs: int,
    lr: float,
    lr_step: int,
    lr_decay: float,
    momentum: float,
    weight_decay: float,
    optimizer: str,
    freeze_backbone: bool,
) -> tuple[float, int, list[dict]]:
    qat.to(device)
    # QAT fine-tune: no dropout noise (inference-like).
    qat.backbone.dropout.p = 0.0
    qat.train()

    if freeze_backbone:
        for p in qat.backbone.parameters():
            p.requires_grad = False
        params = qat.act_scale_parameters()
        if not params:
            raise ValueError("freeze_backbone requires mode=full")
    else:
        params = list(qat.parameters())

    if optimizer == "sgd":
        opt = torch.optim.SGD(
            params, lr=lr, momentum=momentum, weight_decay=weight_decay
        )
    else:
        opt = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

    scheduler = None
    if lr_step > 0:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            opt, milestones=[lr_step], gamma=lr_decay
        )

    criterion = nn.CrossEntropyLoss()
    best_acc = -1.0
    best_epoch = -1
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        qat.train()
        t0 = time.time()
        running = 0.0
        seen = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            opt.zero_grad(set_to_none=True)
            loss = criterion(qat(images), labels)
            loss.backward()
            opt.step()
            running += loss.item() * labels.size(0)
            seen += labels.size(0)

        if scheduler:
            scheduler.step()

        test_acc = evaluate(qat, test_loader, device)
        lr_now = opt.param_groups[0]["lr"]
        history.append(
            {
                "epoch": epoch,
                "train_loss": running / max(seen, 1),
                "test_acc": test_acc,
                "lr": lr_now,
                "seconds": time.time() - t0,
            }
        )
        print(
            f"QAT epoch={epoch:3d} lr={lr_now:.1e} "
            f"loss={history[-1]['train_loss']:.4f} test_acc={test_acc * 100:.2f}%"
        )
        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch

    return best_acc, best_epoch, history


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
    parser.add_argument("--out-dir", type=Path, default=ROOT / "runs/qat_chen")
    parser.add_argument(
        "--mode",
        choices=("weight_only", "full", "full_fixed_act"),
        default="weight_only",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-step", type=int, default=20)
    parser.add_argument("--lr-decay", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--optimizer", choices=("adam", "sgd"), default="adam")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--per-channel-weights", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--calib-batches", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=12321)
    args = parser.parse_args()

    set_seed(args.seed)
    device = pick_device(args.device)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    ckpt, fp32 = load_fp32(args.checkpoint, device)
    ns = Namespace(
        data_dir=args.data_dir,
        dataset="auto",
        batch_size=args.batch_size,
        workers=args.workers,
        per_channel_scale=False,
    )
    train_loader, test_loader, _, _ = build_loaders(ns, args.seed, device, per_channel_scale=False)

    fp32_acc = evaluate(fp32, test_loader, device)
    print(f"FP32 test acc: {fp32_acc * 100:.4f}%")

    act_init: dict[str, float] | None = None
    if args.mode in ("full", "full_fixed_act"):
        scales = calibrate_activation_scales(
            fp32, train_loader, device, max_batches=args.calib_batches
        )
        act_init = {
            "in": float(scales["in"].cpu()),
            "conv1": float(scales["conv1"].cpu()),
            "conv2": float(scales["conv2"].cpu()),
            "conv3": float(scales["conv3"].cpu()),
            "conv4": float(scales["conv4"].cpu()),
        }
        print("Act scale init:", act_init)

    qat = QATAConvNet(
        num_classes=len(ckpt.get("class_names") or []) or 10,
        in_channels=int(ckpt.get("in_channels", 2)),
        per_channel_weights=args.per_channel_weights,
        mode=args.mode,
        act_scale_init=act_init,
    )
    qat.load_fp32_state(ckpt["model_state"])

    best_acc, best_epoch, history = train_qat(
        qat,
        train_loader,
        test_loader,
        device,
        epochs=args.epochs,
        lr=args.lr,
        lr_step=args.lr_step,
        lr_decay=args.lr_decay,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        optimizer=args.optimizer,
        freeze_backbone=args.freeze_backbone,
    )

    summary = {
        "mode": args.mode,
        "fp32_test_acc": fp32_acc,
        "best_test_acc": best_acc,
        "best_epoch": best_epoch,
        "acc_drop_vs_fp32": fp32_acc - best_acc,
        "per_channel_weights": args.per_channel_weights,
        "freeze_backbone": args.freeze_backbone,
        "act_scale_init": act_init,
        "hyperparameters": {
            "epochs": args.epochs,
            "lr": args.lr,
            "lr_step": args.lr_step,
            "optimizer": args.optimizer,
        },
        "history": history,
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    torch.save(
        {
            "model_state": qat.backbone.state_dict(),
            "qat_mode": args.mode,
            "best_test_acc": best_acc,
            "class_names": ckpt.get("class_names"),
            "in_channels": ckpt.get("in_channels", 2),
            "act_log_scales": (
                {
                    "in": float(qat.act_in.scale if hasattr(qat.act_in, "scale") else qat.act_in.log_scale.exp()),
                    "conv1": float(qat.act1.scale if hasattr(qat.act1, "scale") else qat.act1.log_scale.exp()),
                    "conv2": float(qat.act2.scale if hasattr(qat.act2, "scale") else qat.act2.log_scale.exp()),
                    "conv3": float(qat.act3.scale if hasattr(qat.act3, "scale") else qat.act3.log_scale.exp()),
                    "conv4": float(qat.act4.scale if hasattr(qat.act4, "scale") else qat.act4.log_scale.exp()),
                }
                if args.mode != "weight_only"
                else {}
            ),
        },
        args.out_dir / "best.pt",
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "history"}, indent=2))


if __name__ == "__main__":
    main()