"""Post-training quantization (weight INT8 + calibrated activation scales)."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from models.aconvnet import AConvNet
from quantize.fake_quant import activation_scale_symmetric, fake_quantize, weight_scale_symmetric


@dataclass
class PTQState:
    weight_granularity: str  # per_channel | per_tensor
    weight_scales: dict[str, torch.Tensor]
    act_scales: dict[str, torch.Tensor]


class PTQAConvNet(nn.Module):
    """FP32 A-ConvNet with frozen PTQ fake-quant in forward (inference simulation)."""

    def __init__(
        self,
        base: AConvNet,
        state: PTQState,
        *,
        quantize_activations: bool = False,
    ) -> None:
        super().__init__()
        self.base = base
        self.state = state
        self.quantize_activations = quantize_activations
        for p in self.base.parameters():
            p.requires_grad = False

    def _qweight(self, name: str, w: torch.Tensor) -> torch.Tensor:
        scale = self.state.weight_scales[name].to(w.device, w.dtype)
        return fake_quantize(w, scale)

    def _qact(self, name: str, x: torch.Tensor) -> torch.Tensor:
        if not self.quantize_activations:
            return x
        scale = self.state.act_scales[name].to(x.device, x.dtype)
        return fake_quantize(x, scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = self.base
        x = self._qact("in", x)
        x = self._qact(
            "conv1",
            m.pool(torch.relu(nn.functional.conv2d(x, self._qweight("conv1", m.conv1.weight), m.conv1.bias))),
        )
        x = self._qact(
            "conv2",
            m.pool(torch.relu(nn.functional.conv2d(x, self._qweight("conv2", m.conv2.weight), m.conv2.bias))),
        )
        x = self._qact(
            "conv3",
            m.pool(torch.relu(nn.functional.conv2d(x, self._qweight("conv3", m.conv3.weight), m.conv3.bias))),
        )
        x = self._qact(
            "conv4",
            torch.relu(nn.functional.conv2d(x, self._qweight("conv4", m.conv4.weight), m.conv4.bias)),
        )
        x = m.dropout(x)
        x = nn.functional.conv2d(x, self._qweight("conv5", m.conv5.weight), m.conv5.bias)
        return x.flatten(1)


@torch.no_grad()
def calibrate_activation_scales(
    model: AConvNet,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    *,
    max_batches: int = 64,
    method: str = "max",
) -> dict[str, torch.Tensor]:
    """Activation symmetric scales from calibration set (method: max | p999 | p99)."""
    model.eval()
    buckets: dict[str, list[torch.Tensor]] = {
        "in": [],
        "conv1": [],
        "conv2": [],
        "conv3": [],
        "conv4": [],
    }

    def capture(name: str, t: torch.Tensor) -> torch.Tensor:
        buckets[name].append(t.detach())
        return t

    for bi, (images, _) in enumerate(loader):
        if bi >= max_batches:
            break
        x = images.to(device)
        capture("in", x)
        x = model.pool(torch.relu(model.conv1(x)))
        capture("conv1", x)
        x = model.pool(torch.relu(model.conv2(x)))
        capture("conv2", x)
        x = model.pool(torch.relu(model.conv3(x)))
        capture("conv3", x)
        x = torch.relu(model.conv4(x))
        capture("conv4", x)

    return {k: activation_scale_symmetric(v, method=method) for k, v in buckets.items()}


@torch.no_grad()
def calibrate_activation_scales_ptq_weights(
    model: AConvNet,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    *,
    max_batches: int = 64,
    method: str = "max",
    per_channel_weights: bool = True,
) -> dict[str, torch.Tensor]:
    """Calibrate acts on the graph with quantized conv weights (FP32 activations)."""
    model.eval()
    act_scales = calibrate_activation_scales(
        model, loader, device, max_batches=max_batches, method=method
    )
    state = apply_ptq(model, act_scales, per_channel_weights=per_channel_weights)
    ptq = wrap_ptq(model, state, quantize_activations=False).to(device)
    ptq.eval()

    buckets: dict[str, list[torch.Tensor]] = {
        "in": [],
        "conv1": [],
        "conv2": [],
        "conv3": [],
        "conv4": [],
    }

    def capture(name: str, t: torch.Tensor) -> torch.Tensor:
        buckets[name].append(t.detach())
        return t

    m = ptq.base
    for bi, (images, _) in enumerate(loader):
        if bi >= max_batches:
            break
        x = images.to(device)
        capture("in", x)
        x = m.pool(
            torch.relu(
                nn.functional.conv2d(
                    x, ptq._qweight("conv1", m.conv1.weight), m.conv1.bias
                )
            )
        )
        capture("conv1", x)
        x = m.pool(
            torch.relu(
                nn.functional.conv2d(
                    x, ptq._qweight("conv2", m.conv2.weight), m.conv2.bias
                )
            )
        )
        capture("conv2", x)
        x = m.pool(
            torch.relu(
                nn.functional.conv2d(
                    x, ptq._qweight("conv3", m.conv3.weight), m.conv3.bias
                )
            )
        )
        capture("conv3", x)
        x = torch.relu(
            nn.functional.conv2d(
                x, ptq._qweight("conv4", m.conv4.weight), m.conv4.bias
            )
        )
        capture("conv4", x)

    return {k: activation_scale_symmetric(v, method=method) for k, v in buckets.items()}


def apply_ptq(
    model: AConvNet,
    act_scales: dict[str, torch.Tensor],
    *,
    per_channel_weights: bool = True,
) -> PTQState:
    per_ch = per_channel_weights
    names = ("conv1", "conv2", "conv3", "conv4", "conv5")
    weight_scales = {
        name: weight_scale_symmetric(getattr(model, name).weight.data, per_channel=per_ch)
        for name in names
    }
    gran = "per_channel" if per_ch else "per_tensor"
    return PTQState(
        weight_granularity=gran,
        weight_scales=weight_scales,
        act_scales=act_scales,
    )


def wrap_ptq(
    model: AConvNet, state: PTQState, *, quantize_activations: bool = False
) -> PTQAConvNet:
    return PTQAConvNet(model, state, quantize_activations=quantize_activations)
