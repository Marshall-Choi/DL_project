"""NumPy integer reference for A-ConvNet (weight INT8 + optional activation INT8)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from quantize.rounding import dequantize, quantize_symmetric


def max_pool2d_nchw(x: np.ndarray, k: int = 2, s: int = 2) -> np.ndarray:
    n, c, h, w = x.shape
    h_out = (h - k) // s + 1
    w_out = (w - k) // s + 1
    out = np.empty((n, c, h_out, w_out), dtype=x.dtype)
    for i in range(h_out):
        for j in range(w_out):
            patch = x[:, :, i * s : i * s + k, j * s : j * s + k]
            out[:, :, i, j] = patch.max(axis=(2, 3))
    return out


def _im2col_nchw(x: np.ndarray, kh: int, kw: int) -> np.ndarray:
    """(N,C,H,W) -> (N, C*kh*kw, h_out*w_out)."""
    n, c, h, w = x.shape
    h_out, w_out = h - kh + 1, w - kw + 1
    cols = np.empty((n, c * kh * kw, h_out * w_out), dtype=np.float32)
    idx = 0
    for ic in range(c):
        for dy in range(kh):
            for dx in range(kw):
                cols[:, idx] = x[:, ic, dy : dy + h_out, dx : dx + w_out].reshape(n, -1)
                idx += 1
    return cols


def conv2d_nchw(
    x: np.ndarray,
    weight: np.ndarray,
    bias: np.ndarray | None,
) -> np.ndarray:
    """Valid conv, stride 1, groups=1. x (N,C,H,W), weight (O,C,kH,kW)."""
    n, _, h, w = x.shape
    out_c, _, kh, kw = weight.shape
    h_out, w_out = h - kh + 1, w - kw + 1
    cols = _im2col_nchw(x, kh, kw)
    w_col = weight.reshape(out_c, -1)
    out = np.empty((n, out_c, h_out, w_out), dtype=np.float32)
    for b in range(n):
        out[b] = (w_col @ cols[b]).reshape(out_c, h_out, w_out)
    if bias is not None:
        out = out + bias.reshape(1, out_c, 1, 1).astype(np.float32)
    return out


def conv2d_int32(
    x_q: np.ndarray,
    w_q: np.ndarray,
    bias_fp: np.ndarray | None,
    *,
    scale_in: float,
    scale_w: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    INT8 conv with INT32 accumulation; returns (acc_int32, dequant_fp).
    acc_raw = sum(x_q * w_q); dequant = acc_raw * (scale_in * scale_w[oc]) + bias.
    """
    n, _, h, w = x_q.shape
    out_c, _, kh, kw = w_q.shape
    h_out, w_out = h - kh + 1, w - kw + 1
    cols = _im2col_nchw(x_q.astype(np.float32), kh, kw).astype(np.int32)
    w_col = w_q.reshape(out_c, -1).astype(np.int32)
    acc = np.empty((n, out_c, h_out, w_out), dtype=np.int32)
    for b in range(n):
        acc[b] = (w_col @ cols[b]).reshape(out_c, h_out, w_out)
    dequant = np.empty(acc.shape, dtype=np.float32)
    sw = np.asarray(scale_w, dtype=np.float32).reshape(-1)
    for oc in range(out_c):
        mult = scale_in * float(sw[oc] if sw.size > 1 else sw[0])
        dequant[:, oc] = acc[:, oc].astype(np.float32) * mult
        if bias_fp is not None:
            dequant[:, oc] += float(bias_fp[oc])
    return acc, dequant


@dataclass
class Int8LayerParams:
    weight_q: np.ndarray  # int8 (O,C,kH,kW)
    scale_w: np.ndarray  # (O,) per-channel or scalar
    bias_fp: np.ndarray  # float32 (O,)


@dataclass
class AConvNetIntPackage:
    """Exported INT8 weights + scales for reference / C++."""
    weight_granularity: str
    layers: dict[str, Int8LayerParams]
    act_scales: dict[str, float]
    num_classes: int
    in_channels: int
    quantize_activations: bool
    deploy_path: str = "weight_only_fp_act"
    checkpoint_source: str = ""
    act_calib: str = "ptq_weights"
    act_method: str = "max"

    def to_dict(self) -> dict[str, Any]:
        return {
            "weight_granularity": self.weight_granularity,
            "quantize_activations": self.quantize_activations,
            "num_classes": self.num_classes,
            "in_channels": self.in_channels,
            "act_scales": self.act_scales,
            "layers": {
                name: {
                    "weight_q": p.weight_q.tolist(),
                    "scale_w": np.asarray(p.scale_w).reshape(-1).tolist(),
                    "bias_fp": p.bias_fp.reshape(-1).tolist(),
                }
                for name, p in self.layers.items()
            },
        }


class AConvNetIntReference:
    """
    Inference reference.

    * weight_only_fp_act: fake-quant weights, FP32 activations (matches current PTQ).
    * full_int8: INT8 activations with calibrated scales (experimental).
    """

    LAYER_NAMES = ("conv1", "conv2", "conv3", "conv4", "conv5")

    def __init__(self, package: AConvNetIntPackage) -> None:
        self.pkg = package

    def _dequant_weight(self, name: str) -> np.ndarray:
        p = self.pkg.layers[name]
        sw = np.asarray(p.scale_w, dtype=np.float32).reshape(-1, 1, 1, 1)
        return dequantize(p.weight_q, sw)

    def forward_weight_only(
        self, x: np.ndarray, *, dump: dict[str, np.ndarray] | None = None
    ) -> np.ndarray:
        """x float32 (N,C,88,88) -> logits (N, num_classes)."""
        x = x.astype(np.float32, copy=False)
        if dump is not None:
            dump["input"] = x

        x = max_pool2d_nchw(np.maximum(conv2d_nchw(x, self._dequant_weight("conv1"), self._bias("conv1")), 0))
        if dump is not None:
            dump["after_conv1"] = x
        x = max_pool2d_nchw(np.maximum(conv2d_nchw(x, self._dequant_weight("conv2"), self._bias("conv2")), 0))
        if dump is not None:
            dump["after_conv2"] = x
        x = max_pool2d_nchw(np.maximum(conv2d_nchw(x, self._dequant_weight("conv3"), self._bias("conv3")), 0))
        if dump is not None:
            dump["after_conv3"] = x
        x = np.maximum(conv2d_nchw(x, self._dequant_weight("conv4"), self._bias("conv4")), 0)
        if dump is not None:
            dump["after_conv4"] = x
        x = conv2d_nchw(x, self._dequant_weight("conv5"), self._bias("conv5"))
        if dump is not None:
            dump["logits"] = x.reshape(x.shape[0], -1)
        return x.reshape(x.shape[0], -1)

    def forward_full_int8(
        self, x: np.ndarray, *, dump: dict[str, np.ndarray] | None = None
    ) -> np.ndarray:
        """INT8 activations + INT8 weights (uses calibrated act_scales)."""
        s_in = float(self.pkg.act_scales["in"])
        x_q = quantize_symmetric(x, s_in)
        s_act = s_in
        if dump is not None:
            dump["input_q"] = x_q

        for lname, pool in (("conv1", True), ("conv2", True), ("conv3", True)):
            p = self.pkg.layers[lname]
            _, y_fp = conv2d_int32(
                x_q, p.weight_q, p.bias_fp, scale_in=s_act, scale_w=p.scale_w
            )
            y_fp = np.maximum(y_fp, 0)
            if pool:
                y_fp = max_pool2d_nchw(y_fp)
            s_act = float(self.pkg.act_scales[lname])
            x_q = quantize_symmetric(y_fp, s_act)
            if dump is not None:
                dump[f"after_{lname}_q"] = x_q

        p4 = self.pkg.layers["conv4"]
        _, y_fp = conv2d_int32(
            x_q, p4.weight_q, p4.bias_fp, scale_in=s_act, scale_w=p4.scale_w
        )
        y_fp = np.maximum(y_fp, 0)
        s_act = float(self.pkg.act_scales["conv4"])
        x_q = quantize_symmetric(y_fp, s_act)
        if dump is not None:
            dump["after_conv4_q"] = x_q

        p5 = self.pkg.layers["conv5"]
        _, logits = conv2d_int32(
            x_q, p5.weight_q, p5.bias_fp, scale_in=s_act, scale_w=p5.scale_w
        )
        logits = logits.reshape(logits.shape[0], -1)
        if dump is not None:
            dump["logits"] = logits
        return logits

    def _bias(self, name: str) -> np.ndarray:
        return self.pkg.layers[name].bias_fp