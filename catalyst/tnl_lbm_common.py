from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np


@dataclass
class MPIContext:
    """Lightweight MPI wrapper that works with and without mpi4py."""

    comm: Optional[object]
    rank: int
    size: int

    @property
    def is_root(self) -> bool:
        return self.rank == 0


def setup_mpi(disable_mpi: bool) -> MPIContext:
    """Initialise MPI and return a context usable across the scripts."""

    if disable_mpi:
        return MPIContext(comm=None, rank=0, size=1)

    try:
        from mpi4py import MPI  # type: ignore
    except ImportError:
        logging.getLogger(__name__).warning(
            "mpi4py not found; continuing in single-rank mode. Use --nompi to silence this message."
        )
        return MPIContext(comm=None, rank=0, size=1)

    comm = MPI.COMM_WORLD
    return MPIContext(comm=comm, rank=comm.Get_rank(), size=comm.Get_size())


@dataclass
class PlaneSelection:
    start: List[int]
    count: List[int]
    fullshape: Tuple[int, int]
    index: int


PLANE_TO_AXIS = {"xy": 0, "xz": 1, "yz": 2}


def parse_figsize(spec: str) -> Tuple[float, float]:
    try:
        width_str, height_str = spec.lower().split("x")
        return float(width_str), float(height_str)
    except Exception as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid figsize specification '{spec}'. Use WIDTHxHEIGHT, e.g. 8x6."
        ) from exc


def configure_logging(level: str) -> None:
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise argparse.ArgumentTypeError(f"Invalid log level: {level}")
    logging.basicConfig(level=numeric_level, format="%(asctime)s | %(levelname)s | %(message)s")


def expand_planes(choice: str) -> List[str]:
    return ["xy", "xz", "yz"] if choice == "all" else [choice]


def clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, value))


def interpret_index(value: str, max_index: int) -> int:
    token = value.strip().lower()
    if token in {"min", "start"}:
        return 0
    if token in {"max", "end"}:
        return max_index
    if token in {"mid", "center", "centre", "median"}:
        return max_index // 2
    try:
        if token.endswith("%"):
            frac = float(token[:-1]) / 100.0
            return int(round(frac * max_index))
        parsed = float(token)
        if 0.0 <= parsed <= 1.0 and not token.isdigit():
            return int(round(parsed * max_index))
        return int(round(parsed))
    except ValueError as exc:
        raise ValueError(f"Cannot interpret plane index token '{value}'.") from exc


def parse_plane_overrides(entries: Sequence[str], logger: logging.Logger) -> Dict[str, str]:
    overrides: Dict[str, str] = {}
    for item in entries:
        if "=" not in item:
            logger.warning("Ignoring malformed plane override '%s' (expected format plane=index).", item)
            continue
        plane, value = item.split("=", 1)
        plane = plane.strip().lower()
        if plane not in PLANE_TO_AXIS:
            logger.warning("Unknown plane '%s' in override '%s'.", plane, item)
            continue
        overrides[plane] = value.strip()
    return overrides


def compute_plane_selection(plane: str, index: int, shape: Tuple[int, int, int]) -> PlaneSelection:
    nz, ny, nx = shape
    if plane == "xy":
        start = [clamp(index, 0, nz - 1), 0, 0]
        count = [1, ny, nx]
        fullshape = (ny, nx)
    elif plane == "xz":
        start = [0, clamp(index, 0, ny - 1), 0]
        count = [nz, 1, nx]
        fullshape = (nz, nx)
    elif plane == "yz":
        start = [0, 0, clamp(index, 0, nx - 1)]
        count = [nz, ny, 1]
        fullshape = (nz, ny)
    else:
        raise ValueError(f"Unsupported plane '{plane}'.")
    return PlaneSelection(start=start, count=count, fullshape=fullshape, index=index)


def reshape_plane(array: np.ndarray, count: Sequence[int]) -> np.ndarray:
    reshaped = np.asarray(array).reshape(count)
    squeezed = np.squeeze(reshaped)
    if squeezed.ndim != 2:
        raise RuntimeError(f"Unexpected plane dimensionality {squeezed.ndim}, expected 2.")
    return squeezed


def read_scalar(fr_step, var_name: str, selection: PlaneSelection) -> np.ndarray:
    data = fr_step.read(var_name, selection.start, selection.count)
    return reshape_plane(data, selection.count)


def read_velocity(
    fr_step,
    base_name: str,
    component: str,
    selection: PlaneSelection,
    vars_info: Mapping[str, Mapping[str, str]],
) -> np.ndarray:
    suffixes = {"x": "X", "y": "Y", "z": "Z"}
    comp = component.lower()
    if comp in suffixes:
        target = f"{base_name}{suffixes[comp]}"
        if target not in vars_info:
            raise KeyError(f"Variable '{target}' not present in stream.")
        return read_scalar(fr_step, target, selection)

    required = [f"{base_name}X", f"{base_name}Y", f"{base_name}Z"]
    missing = [name for name in required if name not in vars_info]
    if missing:
        raise KeyError(f"Missing components for '{base_name}': {', '.join(missing)}")
    comps = [read_scalar(fr_step, name, selection) for name in required]
    return np.sqrt(sum(np.square(comp.astype(np.float64)) for comp in comps))


__all__ = [
    "MPIContext",
    "PlaneSelection",
    "PLANE_TO_AXIS",
    "clamp",
    "compute_plane_selection",
    "configure_logging",
    "expand_planes",
    "interpret_index",
    "parse_figsize",
    "parse_plane_overrides",
    "read_scalar",
    "read_velocity",
    "reshape_plane",
    "setup_mpi",
]
