"""Helper functions shared by the TNL-LBM test modules."""

from __future__ import annotations

import os
import pathlib
import subprocess
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import pytest

if TYPE_CHECKING:
    from collections.abc import Sequence

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
# overridable so different configurations can be tested without moving build directories
BUILD_DIR = pathlib.Path(
    os.environ.get("TNL_LBM_BUILD_DIR", PROJECT_ROOT / "build")
).resolve()

ADIOS_CONFIG = PROJECT_ROOT / "adios2.xml"
ADIOS_CONFIG_SST = PROJECT_ROOT / "adios2_sst.xml"

FieldData = dict[str, np.ndarray]

# Simulations are relatively heavy; guard against hangs in CI.
# All test simulations are expected to finish well below this limit.
DEFAULT_TIMEOUT = 200.0


@dataclass
class SimRun:
    """Record of a single simulation execution for the end-of-session summary."""

    command: list[str]
    np_ranks: int
    elapsed: float
    status: Literal["ok", "failed", "timeout", "launch-error"]
    exit_code: int | None
    timeout: float


# Populated by run_sim(); printed by the pytest_terminal_summary hook in conftest.py.
_SIM_RUNS: list[SimRun] = []


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
    np_ranks: int = 1,
) -> str:
    """Run a simulation executable in workdir; fail on error/timeout; return stdout.

    With ``np_ranks`` greater than 1 the executable is launched under
    ``mpirun`` so the distributed-domain code paths are exercised.
    """
    command = [str(c) for c in cmd]
    if np_ranks > 1:
        command = ["mpirun", "-np", str(np_ranks), *command]
    run_env = os.environ.copy()
    # On nodes where both ROCm and CUDA are visible to Open MPI, select CUDA.
    run_env.setdefault("OMPI_MCA_accelerator", "cuda")
    if env is not None:
        run_env.update(env)
    print(f"Running simulation: {' '.join(command)}")
    t0 = time.perf_counter()
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
    except subprocess.TimeoutExpired as exc:
        elapsed = time.perf_counter() - t0
        _SIM_RUNS.append(SimRun(command, np_ranks, elapsed, "timeout", None, timeout))
        # Simulations can print long logs; keep only the tail of stdout (where
        # the stall is usually visible) but show stderr in full.
        # The exception may carry partial output as bytes or None depending on timing.
        exc_stdout = exc.stdout or ""
        exc_stderr = exc.stderr or ""
        if isinstance(exc_stdout, bytes):
            exc_stdout = exc_stdout.decode(errors="replace")
        if isinstance(exc_stderr, bytes):
            exc_stderr = exc_stderr.decode(errors="replace")
        stdout_tail = "\n".join(exc_stdout.splitlines()[-50:])
        pytest.fail(
            f"simulation timed out after {timeout:.0f}s: {' '.join(command)}\n"
            f"--- stdout (last 50 lines) ---\n{stdout_tail}\n"
            f"--- stderr ---\n{exc_stderr}",
            pytrace=False,
        )
    except OSError as exc:
        elapsed = time.perf_counter() - t0
        _SIM_RUNS.append(
            SimRun(command, np_ranks, elapsed, "launch-error", None, timeout)
        )
        pytest.fail(
            f"cannot launch simulation: {command[0]} ({exc}) — build the project first",
            pytrace=False,
        )
    elapsed = time.perf_counter() - t0
    if proc.returncode != 0:
        _SIM_RUNS.append(
            SimRun(command, np_ranks, elapsed, "failed", proc.returncode, timeout)
        )
        # Simulations can print long logs; keep only the tail of stdout (where
        # the failure is usually visible) but show stderr in full.
        stdout_tail = "\n".join(proc.stdout.splitlines()[-50:])
        pytest.fail(
            f"simulation failed with exit code {proc.returncode}: {' '.join(command)}\n"
            f"--- stdout (last 50 lines) ---\n{stdout_tail}\n"
            f"--- stderr ---\n{proc.stderr}",
            pytrace=False,
        )
    _SIM_RUNS.append(SimRun(command, np_ranks, elapsed, "ok", proc.returncode, timeout))
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
