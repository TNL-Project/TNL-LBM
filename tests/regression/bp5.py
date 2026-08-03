"""ADIOS2 BP5 readers shared by the regression test modules."""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

import adios2
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterable


def read_last_step(
    bp_path: pathlib.Path, var_names: Iterable[str]
) -> dict[str, np.ndarray]:
    """Read the last step of the given variables from a BP5 file."""
    names = list(var_names)
    if not bp_path.exists():
        msg = f"Output file does not exist: {bp_path}"
        raise FileNotFoundError(msg)

    data: dict[str, np.ndarray] = {}
    with adios2.FileReader(str(bp_path)) as reader:
        num_steps = reader.num_steps()
        if num_steps == 0:
            msg = f"No steps written to {bp_path}"
            raise RuntimeError(msg)

        available = reader.available_variables()
        missing = [v for v in names if v not in available]
        if missing:
            msg = f"Missing variables in {bp_path}: {missing}"
            raise RuntimeError(msg)

        for name in names:
            var = reader.inquire_variable(name)
            arr = reader.read(var, step_selection=[num_steps - 1, 1])
            data[name] = np.asarray(arr)

    return data


def squeeze_2d(arr: np.ndarray) -> np.ndarray:
    """Remove the leading Z=1 dimension from a 3D 2D-cut array."""
    if arr.ndim == 3 and arr.shape[0] == 1:
        return arr[0]
    return arr
