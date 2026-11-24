#!/usr/bin/env python3
"""Train FP32 A-ConvNet baseline on processed MSTAR SOC tensors."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mstar.dataset import AConvNetDataset, AConvNetPatchDataset, MstarNpyDataset
from models.aconvnet import AConvNet


def pick_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        preds = model(images).argmax(dim=1)
        correct += int((preds == labels).sum().item())
        total += labels.size(0)
    return correct / max(total, 1)


@torch.no_grad()
def confusion_matrix(
    model: nn.Module, loader: DataLoader, device: torch.device, num_classes: int
) -> np.ndarray:
    model.eval()
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        preds = model(images).argmax(dim=1)
        for t, p in zip(labels.view(-1), preds.view(-1)):
            cm[int(t.item()), int(p.item())] += 1
    return cm


def build_loaders(
    args: argparse.Namespace, seed: int, device: torch.device, *, per_channel_scale: bool | None
):
    config_path = args.data_dir / "config.json"
    fmt = ""
    if config_path.exists():
        fmt = json.loads(config_path.read_text(encoding="utf-8")).get("format", "")

    if fmt == "aconvnet_patches":
        train_ds = AConvNetPatchDataset(
            args.data_dir, "train", seed=seed, scale=per_channel_scale
        )
        test_ds = AConvNetPatchDataset(args.data_dir, "test", seed=seed, scale=per_channel_scale)
        in_channels = 2
    elif fmt == "aconvnet_mag_phase" or args.dataset == "aconvnet":
        scale = False if per_channel_scale is None else per_channel_scale
        train_ds = AConvNetDataset(args.data_dir, "train", seed=seed, scale=scale)
        test_ds = AConvNetDataset(args.data_dir, "test", seed=seed, scale=scale)
        in_channels = 2
    else:
        train_ds = MstarNpyDataset(args.data_dir, "train")
        test_ds = MstarNpyDataset(args.data_dir, "test")
        in_channels = 1

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    return train_loader, test_loader, train_ds, in_channels


def train_one_seed(args: argparse.Namespace) -> dict:
    set_seed(args.seed)
    device = pick_device(args.device)
    train_loader, test_loader, train_ds, in_channels = build_loaders(
        args, args.seed, device, per_channel_scale=args.per_channel_scale
    )

    model = AConvNet(
        num_classes=len(train_ds.class_names),
        in_channels=in_channels,
        dropout=args.dropout,
    ).to(device)

    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

    scheduler = None
    if args.lr_step:
        milestones = [int(x) for x in args.lr_step.split(",") if x.strip()]
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=args.lr_decay
        )

    criterion = nn.CrossEntropyLoss()
    run_dir = args.out_dir / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    best_acc = -1.0
    best_epoch = -1
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        seen = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
            seen += labels.size(0)

        if scheduler is not None:
            scheduler.step()

        train_loss = running_loss / max(seen, 1)
        test_acc = evaluate(model, test_loader, device)
        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "test_acc": test_acc,
                "lr": lr_now,
                "seconds": elapsed,
            }
        )
        print(
            f"seed={args.seed} epoch={epoch:3d} lr={lr_now:.1e} "
            f"loss={train_loss:.4f} test_acc={test_acc * 100:.2f}% ({elapsed:.1f}s)"
        )

        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "test_acc": test_acc,
                    "class_names": train_ds.class_names,
                    "in_channels": in_channels,
                    "data_dir": str(args.data_dir),
                    "architecture": "AConvNet-Chen2016",
                },
                run_dir / "best.pt",
            )

    ckpt = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    cm = confusion_matrix(model, test_loader, device, num_classes=len(train_ds.class_names))

    best_pt = run_dir / "best.pt"
    num_params = sum(p.numel() for p in model.parameters())
    fp32_weight_bytes = sum(
        v.numel() * v.element_size()
        for v in ckpt["model_state"].values()
        if torch.is_tensor(v)
    )
    summary = {
        "seed": args.seed,
        "best_epoch": best_epoch,
        "best_test_acc": best_acc,
        "num_params": num_params,
        "fp32_weight_bytes": fp32_weight_bytes,
        "int8_weight_bytes_estimate": num_params + sum(
            1 for v in ckpt["model_state"].values() if torch.is_tensor(v) and v.dtype.is_floating_point
        )
        * 4,
        "checkpoint_bytes": best_pt.stat().st_size if best_pt.is_file() else 0,
        "device": str(device),
        "data_dir": str(args.data_dir),
        "in_channels": in_channels,
        "hyperparameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "lr_step": args.lr_step,
            "lr_decay": args.lr_decay,
            "optimizer": args.optimizer,
            "weight_decay": args.weight_decay,
            "dropout": args.dropout,
            "per_channel_scale": args.per_channel_scale,
        },
        "baseline_ref": args.baseline_ref,
        "history": history,
    }
    (run_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.save(run_dir / "confusion_matrix.npy", cm)
    return summary


def apply_chen_repro_defaults(args: argparse.Namespace) -> None:
    """Align with jangsoopark/AConvNet-pytorch experiments/config/AConvNet-SOC.json."""
    args.data_dir = ROOT / "processed" / "aconvnet_patches_chen"
    args.out_dir = ROOT / "runs" / "baseline_chen_soc"
    args.epochs = 100
    args.batch_size = 100
    args.lr = 1e-3
    args.lr_step = "50"
    args.lr_decay = 0.1
    args.momentum = 0.9
    args.weight_decay = 4e-3
    args.dropout = 0.5
    args.optimizer = "sgd"
    args.per_channel_scale = False
    args.baseline_ref = "Chen TGRS 2016 / AConvNet-PyTorch SOC"
    if not args.seeds and args.seed == 12321:
        pass
    elif not args.seeds:
        args.seed = 12321


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--chen-repro",
        action="store_true",
        help="Hyperparameters + raw mag/phase tensors per AConvNet-PyTorch SOC",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "processed" / "aconvnet_patches",
    )
    parser.add_argument(
        "--dataset",
        choices=("auto", "aconvnet", "legacy"),
        default="auto",
    )
    parser.add_argument("--out-dir", type=Path, default=ROOT / "runs" / "baseline_aconvnet")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--lr-step",
        type=str,
        default="50",
        help="MultiStepLR milestones (comma-separated)",
    )
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr-decay", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=4e-3)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--optimizer", choices=("sgd", "adam"), default="sgd")
    parser.add_argument("--seed", type=int, default=12321)
    parser.add_argument("--seeds", type=str, default="")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument(
        "--per-channel-scale",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Min-max norm per channel (default: read from config, else False)",
    )
    parser.add_argument("--baseline-ref", type=str, default="")
    args = parser.parse_args()

    if args.chen_repro:
        apply_chen_repro_defaults(args)
    if args.per_channel_scale is None and (args.data_dir / "config.json").exists():
        cfg = json.loads((args.data_dir / "config.json").read_text(encoding="utf-8"))
        args.per_channel_scale = bool(cfg.get("per_channel_scale", False))
    elif args.per_channel_scale is None:
        args.per_channel_scale = False

    if args.dataset == "auto" and not (args.data_dir / "config.json").exists():
        args.data_dir = ROOT / "processed" / "aconvnet_128"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()] or [args.seed]

    summaries = []
    for seed in seeds:
        args.seed = seed
        summaries.append(train_one_seed(args))

    if len(summaries) > 1:
        accs = [s["best_test_acc"] for s in summaries]
        agg = {
            "seeds": seeds,
            "best_test_acc_mean": float(np.mean(accs)),
            "best_test_acc_std": float(np.std(accs)),
            "per_seed": summaries,
        }
        (args.out_dir / "aggregate.json").write_text(json.dumps(agg, indent=2), encoding="utf-8")
        print(
            f"\nAggregate test acc: {agg['best_test_acc_mean'] * 100:.2f}% "
            f"+/- {agg['best_test_acc_std'] * 100:.2f}%"
        )


if __name__ == "__main__":
    main()
