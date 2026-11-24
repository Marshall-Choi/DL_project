"""Per-tensor and per-channel activation scales from sampled values.

Magnitude SAR chips are heavy-tailed: a single max-abs scale is set by speckle
spikes and rounds the rest of the channel to zero. Percentile clip levels keep
the bulk of each channel inside the INT8 grid.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.aconvnet import AConvNet
from quantize.fake_quant import fake_quantize, weight_scale_symmetric

ACT_NAMES = ("in", "conv1", "conv2", "conv3", "conv4")


def _quantile_level(method: str) -> float | None:
    if method == "max":
        return None
    if method.startswith("p") and method[1:].isdigit():
        return float("0." + method[1:])
    raise ValueError(f"unknown activation scale method: {method}")


def scale_from_samples(
    samples: torch.Tensor, method: str, *, per_channel: bool
) -> torch.Tensor:
    """samples (C, S) -> scale scalar or (1, C, 1, 1)."""
    values = samples if per_channel else samples.reshape(1, -1)
    abs_v = values.abs()
    level = _quantile_level(method)
    if level is None:
        clip = abs_v.amax(dim=1)
    else:
        n = abs_v.shape[1]
        k = max(1, min(n, int(round(level * n))))
        clip = abs_v.kthvalue(k, dim=1).values
    scale = clip.clamp(min=1e-8) / 127.0
    if per_channel:
        return scale.view(1, -1, 1, 1)
    return scale.reshape(())


def _subsample(t: torch.Tensor, k: int) -> torch.Tensor:
    """(N,C,H,W) -> (C, k) random spatial samples, or all values if fewer."""
    flat = t.detach().reshape(t.shape[0], t.shape[1], -1).permute(1, 0, 2).reshape(t.shape[1], -1)
    if flat.shape[1] <= k:
        return flat
    idx = torch.randint(0, flat.shape[1], (k,), device=flat.device)
    return flat[:, idx]


def _quant_weight(conv: nn.Conv2d, *, per_channel: bool) -> torch.Tensor:
    scale = weight_scale_symmetric(conv.weight, per_channel=per_channel)
    return fake_quantize(conv.weight, scale)


@torch.no_grad()
def collect_activation_samples(
    model: AConvNet,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    *,
    max_batches: int = 80,
    per_batch: int = 8192,
    quantize_weights: bool = True,
    per_channel_weights: bool = True,
) -> dict[str, torch.Tensor]:
    """Return {name: (C, S)} samples at the tensors that get activation-quantized."""
    model.eval()
    buckets: dict[str, list[torch.Tensor]] = {name: [] for name in ACT_NAMES}

    for bi, (images, _) in enumerate(loader):
        if bi >= max_batches:
            break
        x = images.to(device)
        buckets["in"].append(_subsample(x, per_batch))
        w1 = _quant_weight(model.conv1, per_channel=per_channel_weights) if quantize_weights else model.conv1.weight
        x = model.pool(torch.relu(nn.functional.conv2d(x, w1, model.conv1.bias)))
        buckets["conv1"].append(_subsample(x, per_batch))
        w2 = _quant_weight(model.conv2, per_channel=per_channel_weights) if quantize_weights else model.conv2.weight
        x = model.pool(torch.relu(nn.functional.conv2d(x, w2, model.conv2.bias)))
        buckets["conv2"].append(_subsample(x, per_batch))
        w3 = _quant_weight(model.conv3, per_channel=per_channel_weights) if quantize_weights else model.conv3.weight
        x = model.pool(torch.relu(nn.functional.conv2d(x, w3, model.conv3.bias)))
        buckets["conv3"].append(_subsample(x, per_batch))
        w4 = _quant_weight(model.conv4, per_channel=per_channel_weights) if quantize_weights else model.conv4.weight
        x = torch.relu(nn.functional.conv2d(x, w4, model.conv4.bias))
        buckets["conv4"].append(_subsample(x, per_batch))

    return {name: torch.cat(parts, dim=1) for name, parts in buckets.items()}


def scales_from_collected(
    samples: dict[str, torch.Tensor],
    method: str,
    *,
    per_channel: bool,
    per_channel_input_only: bool = False,
) -> dict[str, torch.Tensor]:
    scales: dict[str, torch.Tensor] = {}
    for name, values in samples.items():
        channelwise = per_channel or (per_channel_input_only and name == "in")
        scales[name] = scale_from_samples(values, method, per_channel=channelwise)
    return scales
