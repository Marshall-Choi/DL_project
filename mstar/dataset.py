"""PyTorch datasets for MSTAR."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from mstar.preprocess import center_crop, random_crop


class MstarNpyDataset(Dataset):
    """Legacy single-channel tensors (e.g. processed/soc_88)."""

    def __init__(self, data_dir: str | Path, split: str = "train") -> None:
        data_dir = Path(data_dir)
        if split not in ("train", "test"):
            raise ValueError("split must be train or test")
        self.x = np.load(data_dir / f"{split}_x.npy", mmap_mode="r")
        self.y = np.load(data_dir / f"{split}_y.npy")
        config = json.loads((data_dir / "config.json").read_text(encoding="utf-8"))
        self.class_names: list[str] = config["class_names"]

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        image = np.array(self.x[idx], dtype=np.float32, copy=True)
        image = torch.from_numpy(image).unsqueeze(0)
        label = torch.tensor(int(self.y[idx]), dtype=torch.long)
        return image, label


def _scale_channels(chip: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    out = chip.astype(np.float32, copy=True)
    for c in range(out.shape[0]):
        mn = float(out[c].min())
        mx = float(out[c].max())
        if mx > mn:
            out[c] = (out[c] - mn) / (mx - mn + eps)
    return out


class AConvNetPatchDataset(Dataset):
    """Precomputed 94x94 train patches; random 88 crop at train time."""

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        *,
        input_size: int = 88,
        seed: int = 0,
        scale: bool | None = None,
    ) -> None:
        data_dir = Path(data_dir)
        self.split = split
        self.input_size = input_size
        self.x = np.load(data_dir / f"{split}_x.npy", mmap_mode="r")
        self.y = np.load(data_dir / f"{split}_y.npy")
        config = json.loads((data_dir / "config.json").read_text(encoding="utf-8"))
        self.class_names: list[str] = config["class_names"]
        if scale is None:
            scale = bool(config.get("per_channel_scale", False))
        self.scale = scale
        self.rng = np.random.default_rng(seed if split == "train" else seed + 10_000)

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        chip = np.array(self.x[idx], dtype=np.float32, copy=True)
        if self.split == "train":
            chip = random_crop(chip, self.input_size, self.rng)
        else:
            chip = center_crop(chip, self.input_size)
        if self.scale:
            chip = _scale_channels(chip)
        return torch.from_numpy(chip), torch.tensor(int(self.y[idx]), dtype=torch.long)


class AConvNetDataset(Dataset):
    """
    Mag+phase (N, 2, 128, 128).
    Train: center 100 -> random 94 patch -> random 88 crop (A-ConvNet-PyTorch SOC).
    Test: center 88 from 128.
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        *,
        input_size: int = 88,
        chip_size: int = 128,
        mid_size: int = 100,
        patch_size: int = 94,
        seed: int = 0,
        scale: bool = False,
    ) -> None:
        data_dir = Path(data_dir)
        if split not in ("train", "test"):
            raise ValueError("split must be train or test")
        self.split = split
        self.input_size = input_size
        self.chip_size = chip_size
        self.mid_size = mid_size
        self.patch_size = patch_size
        self.scale = scale
        self.x = np.load(data_dir / f"{split}_x.npy", mmap_mode="r")
        self.y = np.load(data_dir / f"{split}_y.npy")
        config = json.loads((data_dir / "config.json").read_text(encoding="utf-8"))
        self.class_names: list[str] = config["class_names"]
        self.rng = np.random.default_rng(seed if split == "train" else seed + 10_000)

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def _train_crop(self, chip: np.ndarray) -> np.ndarray:
        chip = center_crop(chip, self.chip_size)
        chip = center_crop(chip, self.mid_size)
        grid = self.mid_size - self.patch_size + 1
        top = int(self.rng.integers(0, grid))
        left = int(self.rng.integers(0, grid))
        patch = chip[:, top : top + self.patch_size, left : left + self.patch_size]
        return random_crop(patch, self.input_size, self.rng)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        chip = np.array(self.x[idx], dtype=np.float32, copy=True)
        if self.split == "train":
            chip = self._train_crop(chip)
        else:
            chip = center_crop(chip, self.chip_size)
            chip = center_crop(chip, self.input_size)
        if self.scale:
            chip = _scale_channels(chip)
        label = torch.tensor(int(self.y[idx]), dtype=torch.long)
        return torch.from_numpy(chip), label
