"""Spatial crops for MSTAR chips."""

from __future__ import annotations

import numpy as np


def center_crop(arr: np.ndarray, size: int) -> np.ndarray:
    """Crop square patch; arr is (H,W) or (C,H,W)."""
    if arr.ndim == 2:
        h, w = arr.shape
        if h < size or w < size:
            raise ValueError(f"Crop {size} exceeds shape {arr.shape}")
        top = (h - size) // 2
        left = (w - size) // 2
        return arr[top : top + size, left : left + size]
    if arr.ndim == 3:
        c, h, w = arr.shape
        if h < size or w < size:
            raise ValueError(f"Crop {size} exceeds shape {arr.shape}")
        top = (h - size) // 2
        left = (w - size) // 2
        return arr[:, top : top + size, left : left + size]
    raise ValueError(f"Unsupported ndim {arr.ndim}")


def random_crop(arr: np.ndarray, size: int, rng: np.random.Generator) -> np.ndarray:
    if arr.ndim == 2:
        h, w = arr.shape
    elif arr.ndim == 3:
        _, h, w = arr.shape
    else:
        raise ValueError(f"Unsupported ndim {arr.ndim}")
    if h < size or w < size:
        raise ValueError(f"Crop {size} exceeds shape {arr.shape}")
    top = int(rng.integers(0, h - size + 1))
    left = int(rng.integers(0, w - size + 1))
    if arr.ndim == 2:
        return arr[top : top + size, left : left + size]
    return arr[:, top : top + size, left : left + size]
