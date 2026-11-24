"""Read MSTAR Phoenix-header SAR chips (magnitude only)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_HEADER_END = b"[EndofPhoenixHeader]"


@dataclass(frozen=True)
class MstarChip:
    magnitude: np.ndarray  # float32, shape (H, W)
    target_az: float
    measured_depression: float
    target_type: str
    rows: int
    cols: int
    source_path: str


@dataclass(frozen=True)
class MstarPairChip:
    """Magnitude and phase, shape (2, H, W)."""

    tensor: np.ndarray
    target_az: float
    measured_depression: float
    target_type: str
    rows: int
    cols: int
    source_path: str


def _parse_header(text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for line in text.splitlines():
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        fields[key.strip()] = val.strip()
    return fields


def _pick_decode_offset(raw: bytes, base_start: int, rows: int, cols: int) -> tuple[int, str]:
    count = rows * cols
    best_skip, best_endian = 0, ">"
    best_score = -1.0
    for skip in (0, 2, 4):
        offset = base_start + skip
        if offset + 4 * count > len(raw):
            continue
        for endian in (">", "<"):
            mag = np.frombuffer(raw, dtype=f"{endian}f4", count=count, offset=offset)
            finite = np.isfinite(mag)
            if finite.mean() < 0.99:
                continue
            values = mag[finite]
            in_range = ((values >= 0) & (values <= 100)).mean()
            if in_range < 0.99:
                continue
            score = float(in_range) - skip * 1e-4
            if score > best_score:
                best_score = score
                best_skip, best_endian = skip, endian
    if best_score < 0:
        raise ValueError("Could not decode magnitude (endian / corrupt file)")
    return best_skip, best_endian


def _decode_magnitude(raw: bytes, base_start: int, rows: int, cols: int) -> np.ndarray:
    skip, endian = _pick_decode_offset(raw, base_start, rows, cols)
    count = rows * cols
    offset = base_start + skip
    mag = np.frombuffer(raw, dtype=f"{endian}f4", count=count, offset=offset)
    return mag.reshape(rows, cols).astype(np.float32, copy=True)


def _decode_mag_phase(raw: bytes, base_start: int, rows: int, cols: int) -> np.ndarray:
    skip, endian = _pick_decode_offset(raw, base_start, rows, cols)
    count = rows * cols
    offset = base_start + skip
    mag = np.frombuffer(raw, dtype=f"{endian}f4", count=count, offset=offset)
    phase_offset = offset + 4 * count
    if phase_offset + 4 * count <= len(raw):
        phase = np.frombuffer(raw, dtype=f"{endian}f4", count=count, offset=phase_offset)
    else:
        phase = np.zeros(count, dtype=np.float32)
    stacked = np.stack(
        [
            mag.reshape(rows, cols),
            phase.reshape(rows, cols),
        ],
        axis=0,
    ).astype(np.float32, copy=False)
    stacked = np.nan_to_num(stacked, nan=0.0, posinf=0.0, neginf=0.0)
    return stacked


def read_mstar_chip(path: str | Path, *, endian: str | None = None) -> MstarChip:
    """Load one chip; magnitude block endianness is auto-detected when omitted."""
    path = Path(path)
    raw = path.read_bytes()
    end_idx = raw.find(_HEADER_END)
    if end_idx < 0:
        raise ValueError(f"Phoenix header not found: {path}")

    hdr_text = raw[:end_idx].decode("latin-1", errors="replace")
    fields = _parse_header(hdr_text)
    rows = int(fields["NumberOfRows"])
    cols = int(fields["NumberOfColumns"])
    target_az = float(fields.get("TargetAz", "0"))
    measured_depression = float(fields.get("MeasuredDepression", "0"))
    target_type = fields.get("TargetType", "")

    data_start = end_idx + len(_HEADER_END)
    while data_start < len(raw) and raw[data_start : data_start + 1] in (
        b"\n",
        b"\r",
        b"=",
        b" ",
    ):
        data_start += 1

    if endian is None:
        magnitude = _decode_magnitude(raw, data_start, rows, cols)
    else:
        count = rows * cols
        mag = np.frombuffer(raw, dtype=f"{endian}f4", count=count, offset=data_start)
        if mag.size != count:
            raise ValueError(f"Truncated magnitude block: {path}")
        magnitude = mag.reshape(rows, cols).astype(np.float32, copy=True)
    magnitude = np.nan_to_num(magnitude, nan=0.0, posinf=0.0, neginf=0.0)
    magnitude = np.maximum(magnitude, 0.0, out=magnitude)

    return MstarChip(
        magnitude=magnitude,
        target_az=target_az,
        measured_depression=measured_depression,
        target_type=target_type,
        rows=rows,
        cols=cols,
        source_path=str(path.resolve()),
    )


def read_mstar_pair(path: str | Path) -> MstarPairChip:
    """Load magnitude + phase as (2, H, W), matching A-ConvNet reference readers."""
    path = Path(path)
    raw = path.read_bytes()
    end_idx = raw.find(_HEADER_END)
    if end_idx < 0:
        raise ValueError(f"Phoenix header not found: {path}")

    hdr_text = raw[:end_idx].decode("latin-1", errors="replace")
    fields = _parse_header(hdr_text)
    rows = int(fields["NumberOfRows"])
    cols = int(fields["NumberOfColumns"])
    target_az = float(fields.get("TargetAz", "0"))
    measured_depression = float(fields.get("MeasuredDepression", "0"))
    target_type = fields.get("TargetType", "")

    data_start = end_idx + len(_HEADER_END)
    while data_start < len(raw) and raw[data_start : data_start + 1] in (
        b"\n",
        b"\r",
        b"=",
        b" ",
    ):
        data_start += 1

    tensor = _decode_mag_phase(raw, data_start, rows, cols)
    return MstarPairChip(
        tensor=tensor,
        target_az=target_az,
        measured_depression=measured_depression,
        target_type=target_type,
        rows=rows,
        cols=cols,
        source_path=str(path.resolve()),
    )


def is_chip_file(name: str) -> bool:
    if not name.startswith("HB") or name.upper().endswith(".JPG"):
        return False
    return bool(re.match(r"HB.+\.\d+$", name))
