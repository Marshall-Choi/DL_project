"""Build INT8 numpy package from FP32 checkpoint + PTQ scales."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from models.aconvnet import AConvNet
from quantize.fake_quant import weight_scale_symmetric
from quantize.int_ref import AConvNetIntPackage, Int8LayerParams
from quantize.ptq import calibrate_activation_scales, calibrate_activation_scales_ptq_weights
from quantize.rounding import quantize_symmetric


def _bias_numpy(layer: torch.nn.Conv2d) -> np.ndarray:
    if layer.bias is None:
        return np.zeros(layer.out_channels, dtype=np.float32)
    return layer.bias.detach().cpu().numpy().astype(np.float32)


def build_int8_layers(
    model: AConvNet, *, per_channel: bool
) -> dict[str, Int8LayerParams]:
    layers: dict[str, Int8LayerParams] = {}
    for name in ("conv1", "conv2", "conv3", "conv4", "conv5"):
        conv = getattr(model, name)
        w = conv.weight.detach().cpu().numpy()
        scale = weight_scale_symmetric(conv.weight.data, per_channel=per_channel)
        scale_np = scale.cpu().numpy()
        if per_channel:
            sw = scale_np.reshape(-1)
            w_q = quantize_symmetric(w, scale_np)
        else:
            sw = np.array([float(scale_np)], dtype=np.float32)
            w_q = quantize_symmetric(w, float(scale_np))
        layers[name] = Int8LayerParams(
            weight_q=w_q.astype(np.int8),
            scale_w=sw.astype(np.float32),
            bias_fp=_bias_numpy(conv),
        )
    return layers


def package_from_checkpoint(
    checkpoint: Path,
    *,
    per_channel_weights: bool = True,
    calib_loader: torch.utils.data.DataLoader | None = None,
    device: torch.device | None = None,
    calib_batches: int = 64,
    act_calib: str = "ptq_weights",
    act_method: str = "max",
    deploy_path: str = "weight_only_fp_act",
) -> AConvNetIntPackage:
    device = device or torch.device("cpu")
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    in_ch = int(ckpt.get("in_channels", 2))
    nclass = len(ckpt.get("class_names") or []) or 10
    model = AConvNet(num_classes=nclass, in_channels=in_ch).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    if calib_loader is not None:
        if act_calib == "fp32_graph":
            act_scales_t = calibrate_activation_scales(
                model, calib_loader, device, max_batches=calib_batches, method=act_method
            )
        else:
            act_scales_t = calibrate_activation_scales_ptq_weights(
                model,
                calib_loader,
                device,
                max_batches=calib_batches,
                method=act_method,
                per_channel_weights=per_channel_weights,
            )
        act_scales = {k: float(v.cpu()) for k, v in act_scales_t.items()}
    else:
        act_scales = {k: 1.0 for k in ("in", "conv1", "conv2", "conv3", "conv4")}

    layers = build_int8_layers(model, per_channel=per_channel_weights)
    gran = "per_channel" if per_channel_weights else "per_tensor"
    quantize_acts = deploy_path.startswith("full_int8")
    pkg = AConvNetIntPackage(
        weight_granularity=gran,
        layers=layers,
        act_scales=act_scales,
        num_classes=nclass,
        in_channels=in_ch,
        quantize_activations=quantize_acts,
    )
    pkg.checkpoint_source = str(checkpoint)
    pkg.deploy_path = deploy_path
    pkg.act_calib = act_calib
    pkg.act_method = act_method
    return pkg


def save_package(pkg: AConvNetIntPackage, out_dir: Path) -> None:
    import json

    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "weight_granularity": pkg.weight_granularity,
        "quantize_activations": pkg.quantize_activations,
        "deploy_path": getattr(pkg, "deploy_path", "weight_only_fp_act"),
        "checkpoint_source": getattr(pkg, "checkpoint_source", ""),
        "act_calib": getattr(pkg, "act_calib", "ptq_weights"),
        "act_method": getattr(pkg, "act_method", "max"),
        "num_classes": pkg.num_classes,
        "in_channels": pkg.in_channels,
        "act_scales": pkg.act_scales,
        "rounding": "numpy.round (half to even), clip [-128,127]",
    }
    (out_dir / "manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    for name, layer in pkg.layers.items():
        ld = out_dir / name
        ld.mkdir(exist_ok=True)
        np.save(ld / "weight_q.npy", layer.weight_q)
        np.save(ld / "scale_w.npy", layer.scale_w)
        np.save(ld / "bias_fp.npy", layer.bias_fp)


def load_package(out_dir: Path) -> AConvNetIntPackage:
    import json

    meta = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    layers: dict[str, Int8LayerParams] = {}
    for name in ("conv1", "conv2", "conv3", "conv4", "conv5"):
        ld = out_dir / name
        layers[name] = Int8LayerParams(
            weight_q=np.load(ld / "weight_q.npy"),
            scale_w=np.load(ld / "scale_w.npy"),
            bias_fp=np.load(ld / "bias_fp.npy"),
        )
    return AConvNetIntPackage(
        weight_granularity=meta["weight_granularity"],
        layers=layers,
        act_scales=meta["act_scales"],
        num_classes=meta["num_classes"],
        in_channels=meta["in_channels"],
        quantize_activations=bool(meta.get("quantize_activations", False)),
        deploy_path=meta.get("deploy_path", "weight_only_fp_act"),
        checkpoint_source=meta.get("checkpoint_source", ""),
        act_calib=meta.get("act_calib", "ptq_weights"),
        act_method=meta.get("act_method", "max"),
    )