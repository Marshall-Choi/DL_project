"""QAT wrapper: fake-quant weights (STE) + optional fake-quant activations."""

from __future__ import annotations

import torch
import torch.nn as nn

from models.aconvnet import AConvNet
from quantize.fake_quant import fake_quantize, weight_scale_symmetric


class _ActScale(nn.Module):
    def __init__(self, init_scale: float | torch.Tensor, *, trainable: bool = True) -> None:
        super().__init__()
        if not isinstance(init_scale, torch.Tensor):
            init_scale = torch.tensor(float(init_scale), dtype=torch.float32)
        else:
            init_scale = init_scale.detach().float()
        init_scale = init_scale.clamp(min=1e-8)
        if trainable:
            self.log_scale = nn.Parameter(init_scale.log())
        else:
            self.register_buffer("scale", init_scale.clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, "log_scale"):
            scale = self.log_scale.exp().clamp(min=1e-8)
        else:
            scale = self.scale.clamp(min=1e-8)
        return fake_quantize(x, scale)


class _IdentityQuant(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class QATAConvNet(nn.Module):
    """
    Modes:
      weight_only — fake-quant conv weights only (matches PTQ deploy path, adapts weights).
      full        — weight + activation fake-quant (init scales from PTQ calibration).
    """

    def __init__(
        self,
        num_classes: int = 10,
        in_channels: int = 2,
        dropout: float = 0.5,
        *,
        per_channel_weights: bool = True,
        mode: str = "full",
        act_scale_init: dict[str, float] | None = None,
    ) -> None:
        super().__init__()
        if mode not in ("weight_only", "full", "full_fixed_act"):
            raise ValueError("mode must be weight_only, full, or full_fixed_act")
        self.mode = mode
        self.per_channel_weights = per_channel_weights
        self.backbone = AConvNet(num_classes=num_classes, in_channels=in_channels, dropout=dropout)
        defaults: dict[str, float | torch.Tensor] = {
            "in": 0.05,
            "conv1": 0.05,
            "conv2": 0.05,
            "conv3": 0.05,
            "conv4": 0.05,
        }
        if act_scale_init:
            for k, v in act_scale_init.items():
                defaults[k] = v.detach().float() if torch.is_tensor(v) else float(v)

        train_act = mode == "full"
        if mode in ("full", "full_fixed_act"):
            self.act_in = _ActScale(defaults["in"], trainable=train_act)
            self.act1 = _ActScale(defaults["conv1"], trainable=train_act)
            self.act2 = _ActScale(defaults["conv2"], trainable=train_act)
            self.act3 = _ActScale(defaults["conv3"], trainable=train_act)
            self.act4 = _ActScale(defaults["conv4"], trainable=train_act)
        else:
            self.act_in = _IdentityQuant()
            self.act1 = _IdentityQuant()
            self.act2 = _IdentityQuant()
            self.act3 = _IdentityQuant()
            self.act4 = _IdentityQuant()

    def load_fp32_state(self, state_dict: dict) -> None:
        self.backbone.load_state_dict(state_dict)

    def _qw(self, layer: nn.Conv2d) -> torch.Tensor:
        scale = weight_scale_symmetric(layer.weight, per_channel=self.per_channel_weights)
        return fake_quantize(layer.weight, scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        m = self.backbone
        x = self.act_in(x)
        x = m.pool(torch.relu(nn.functional.conv2d(x, self._qw(m.conv1), m.conv1.bias)))
        x = self.act1(x)
        x = m.pool(torch.relu(nn.functional.conv2d(x, self._qw(m.conv2), m.conv2.bias)))
        x = self.act2(x)
        x = m.pool(torch.relu(nn.functional.conv2d(x, self._qw(m.conv3), m.conv3.bias)))
        x = self.act3(x)
        x = torch.relu(nn.functional.conv2d(x, self._qw(m.conv4), m.conv4.bias))
        x = self.act4(x)
        x = m.dropout(x)
        x = nn.functional.conv2d(x, self._qw(m.conv5), m.conv5.bias)
        return x.flatten(1)

    def act_scale_parameters(self) -> list[nn.Parameter]:
        if self.mode != "full":
            return []
        return [m.log_scale for m in (self.act_in, self.act1, self.act2, self.act3, self.act4)]