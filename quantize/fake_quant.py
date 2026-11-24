"""Symmetric INT8 fake-quantization with straight-through estimator (Jacob et al.)."""

from __future__ import annotations

import torch


class FakeQuantSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        qmin, qmax = -128.0, 127.0
        q = torch.round(x / scale).clamp(qmin, qmax)
        return q * scale

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return grad_output, None


def fake_quantize(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Quantize-dequantize; scale broadcasts over x (per-tensor or per-channel)."""
    while scale.ndim < x.ndim:
        scale = scale.unsqueeze(-1)
    scale = scale.to(dtype=x.dtype, device=x.device)
    return FakeQuantSTE.apply(x, scale)


def weight_scale_symmetric(
    weight: torch.Tensor, *, per_channel: bool
) -> torch.Tensor:
    """FP32 conv weight -> scale for symmetric INT8 (÷127)."""
    if per_channel:
        amax = weight.abs().amax(dim=(1, 2, 3), keepdim=True).clamp(min=1e-8)
    else:
        amax = weight.abs().max().clamp(min=1e-8)
    return amax / 127.0


def activation_scale_symmetric(
    activations: list[torch.Tensor], *, method: str = "max"
) -> torch.Tensor:
    """Symmetric INT8 scale = amax(method) / 127."""
    if method == "max":
        amax = torch.stack([a.detach().abs().amax() for a in activations]).max()
        return amax.clamp(min=1e-8) / 127.0

    # Percentile: pool per-batch amax values (memory-safe vs full flatten).
    batch_amax = torch.stack([a.detach().abs().amax() for a in activations])
    q = 0.999 if method in ("p999", "percentile_999") else 0.99
    if method not in ("p999", "percentile_999", "p99", "percentile_99"):
        raise ValueError(f"unknown activation scale method: {method}")
    amax = torch.quantile(batch_amax.float(), q)
    return amax.clamp(min=1e-8) / 127.0
