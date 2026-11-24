"""Standard Operating Condition (SOC) file index for 10-class MSTAR."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from mstar.io import is_chip_file

CLASS_NAMES = (
    "2S1",
    "BMP2",
    "BRDM2",
    "BTR60",
    "BTR70",
    "D7",
    "T62",
    "T72",
    "ZIL131",
    "ZSU234",
)

_FOLDER_TO_CLASS = {
    "2S1": "2S1",
    "BRDM_2": "BRDM2",
    "BRDM-2": "BRDM2",
    "BTR_60": "BTR60",
    "BTR-60": "BTR60",
    "D7": "D7",
    "T62": "T62",
    "ZIL131": "ZIL131",
    "ZSU_23_4": "ZSU234",
    "ZSU-23/4": "ZSU234",
    "BMP2": "BMP2",
    "BTR70": "BTR70",
    "T72": "T72",
}

SOC_SERIAL = {
    "BMP2": {"SN_9563"},
    "BTR70": {"SN_C71"},
    "T72": {"SN_132"},
}


@dataclass(frozen=True)
class SampleRecord:
    path: Path
    split: str  # "train" | "test"
    class_name: str
    depression: str  # "17" | "15"


def _walk_mixed(root: Path, angle_dir: str, split: str) -> list[SampleRecord]:
    out: list[SampleRecord] = []
    base = root / angle_dir
    if not base.is_dir():
        return out
    for dirpath, _, files in os.walk(base):
        folder = Path(dirpath).name
        if folder in ("15_DEG", "17_DEG") or folder.startswith("COL") or folder.startswith(
            "SCENE"
        ):
            continue
        if folder == "SLICY":
            continue
        cls = _FOLDER_TO_CLASS.get(folder)
        if cls is None:
            continue
        chips = [f for f in files if is_chip_file(f)]
        if not chips:
            continue
        for f in chips:
            out.append(
                SampleRecord(
                    path=Path(dirpath) / f,
                    split=split,
                    class_name=cls,
                    depression="17" if split == "train" else "15",
                )
            )
    return out


def _walk_targets(root: Path, split: str, angle_dir: str) -> list[SampleRecord]:
    out: list[SampleRecord] = []
    base = root / "TARGETS" / ("TRAIN" if split == "train" else "TEST") / angle_dir
    if not base.is_dir():
        return out
    for veh in base.iterdir():
        if not veh.is_dir() or veh.name not in ("BMP2", "BTR70", "T72"):
            continue
        cls = veh.name
        for sn_dir in veh.iterdir():
            if not sn_dir.is_dir():
                continue
            allowed = SOC_SERIAL.get(cls)
            if allowed and sn_dir.name not in allowed:
                continue
            for f in sn_dir.iterdir():
                if f.is_file() and is_chip_file(f.name):
                    out.append(
                        SampleRecord(
                            path=f,
                            split=split,
                            class_name=cls,
                            depression="17" if split == "train" else "15",
                        )
                    )
    return out


def collect_soc_samples(data_root: str | Path) -> list[SampleRecord]:
    data_root = Path(data_root)
    mixed_cd2 = data_root / "MSTAR_PUBLIC_MIXED_TARGETS_CD2"
    mixed_cd1 = data_root / "MSTAR_PUBLIC_MIXED_TARGETS_CD1"
    targets = data_root / "MSTAR_PUBLIC_TARGETS_CHIPS_T72_BMP2_BTR70_SLICY"

    samples: list[SampleRecord] = []
    samples.extend(_walk_mixed(mixed_cd2, "17_DEG", "train"))
    samples.extend(_walk_mixed(mixed_cd1, "15_DEG", "test"))
    samples.extend(_walk_targets(targets, "train", "17_DEG"))
    samples.extend(_walk_targets(targets, "test", "15_DEG"))
    return samples
