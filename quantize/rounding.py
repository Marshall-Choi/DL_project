"""Single rounding rule for INT8 reference and C++ engine (proposal contract)."""

from __future__ import annotations

import numpy as np

# Matches PyTorch fake_quant: torch.round (half to even), then clip [-128, 127].
QMIN, QMAX = -128, 127


def round_to_int(x: np.ndarray) -> np.ndarray:
    return np.round(x).astype(np.int32)


def quantize_symmetric(x: np.ndarray, scale: float | np.ndarray) -> np.ndarray:
    """FP32 -> INT8 symmetric."""
    scale = np.asarray(scale, dtype=np.float64)
    q = round_to_int(x.astype(np.float64) / scale)
    return np.clip(q, QMIN, QMAX).astype(np.int8)


def dequantize(q: np.ndarray, scale: float | np.ndarray) -> np.ndarray:
    return q.astype(np.float32) * np.asarray(scale, dtype=np.float32)