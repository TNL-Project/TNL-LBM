"""Helper functions shared by the TNL-LBM test modules."""

from __future__ import annotations

import os
import pathlib
import subprocess
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from collections.abc import Sequence

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
BUILD_DIR = PROJECT_ROOT / "build"

FieldData = dict[str, np.ndarray]

# Simulations are relatively heavy; guard against hangs in CI.
DEFAULT_TIMEOUT = 900.0


def _aa_pattern_enabled() -> bool:
    """Detect TNL_LBM_AA_PATTERN=ON in BUILD_DIR's CMakeCache."""
    cache = BUILD_DIR / "CMakeCache.txt"
    try:
        for line in cache.read_text().splitlines():
            if line.strip() == "TNL_LBM_AA_PATTERN:BOOL=ON":
                return True
    except OSError:
        pass
    return False


# True when the sim binaries use the A-A streaming pattern (some boundary
# conditions are known to not be faithful under it; see AGENTS.md).
AA_PATTERN = _aa_pattern_enabled()


def run_sim(
    cmd: Sequence[str | pathlib.Path],
    *,
    workdir: pathlib.Path,
    timeout: float = DEFAULT_TIMEOUT,
    env: dict[str, str] | None = None,
) -> str:
    """Run a simulation executable in workdir; fail on error/timeout; return stdout."""
    command = [str(c) for c in cmd]
    run_env = os.environ.copy()
    # On nodes where both ROCm and CUDA are visible to Open MPI, select CUDA.
    run_env.setdefault("OMPI_MCA_accelerator", "cuda")
    if env is not None:
        run_env.update(env)
    print(f"Running simulation: {' '.join(command)}")
    try:
        proc = subprocess.run(
            command,
            cwd=workdir,
            env=run_env,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(
            f"simulation timed out after {timeout:.0f}s: {' '.join(command)}",
            pytrace=False,
        )
    except OSError as exc:
        pytest.fail(
            f"cannot launch simulation: {command[0]} ({exc}) — build the project first",
            pytrace=False,
        )
    if proc.returncode != 0:
        pytest.fail(
            f"simulation failed with exit code {proc.returncode}: {' '.join(command)}\n"
            f"--- stdout ---\n{proc.stdout}\n"
            f"--- stderr ---\n{proc.stderr}",
            pytrace=False,
        )
    return proc.stdout


def assert_all_finite(data: FieldData) -> None:
    """Assert every field contains only finite values."""
    for name, arr in data.items():
        assert np.all(np.isfinite(arr)), (
            f"{name} has {np.sum(~np.isfinite(arr))} non-finite values"
        )


def assert_mass_conserved(rho: np.ndarray, tolerance: float) -> None:
    """Assert the mean density deviates from 1 by less than tolerance."""
    mean_rho = float(np.mean(rho))
    assert abs(mean_rho - 1.0) < tolerance, (
        f"mean(rho)={mean_rho:.8e} deviates from 1 by more than {tolerance:.0e}"
    )
