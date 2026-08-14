"""Shared pytest fixtures and hooks for the TNL-LBM test suite."""

from __future__ import annotations

import pathlib

import cuda.bindings.driver as cuda_driver
import pytest

from tests.lbmtest import _SIM_RUNS, SimRun


def _gpu_available() -> bool:
    try:
        if cuda_driver.cuInit(0)[0] != cuda_driver.CUresult.CUDA_SUCCESS:
            return False
        err, count = cuda_driver.cuDeviceGetCount()
        return err == cuda_driver.CUresult.CUDA_SUCCESS and count > 0
    except Exception:
        return False


GPU_AVAILABLE = _gpu_available()


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Skip the whole suite without a CUDA-capable GPU; all builds use CUDA."""
    if GPU_AVAILABLE:
        return
    skip_gpu = pytest.mark.skip(reason="no CUDA-capable GPU available")
    for item in items:
        item.add_marker(skip_gpu)


def pytest_terminal_summary(
    terminalreporter: pytest.TerminalReporter,
    exitstatus: pytest.ExitCode,
    config: pytest.Config,
) -> None:
    """Append a summary of simulation runs with their elapsed wall time."""
    if not _SIM_RUNS:
        return
    tr = terminalreporter
    tr.write_sep("=", "simulation runs", bold=True)

    def _format(cmd: list[str]) -> str:
        # Strip the mpirun prefix; rank count is shown in the ranks column.
        args = (
            cmd[3:] if len(cmd) >= 3 and cmd[0] == "mpirun" and cmd[1] == "-np" else cmd
        )
        parts = [
            pathlib.Path(a).name if "/" in a and not a.startswith("-") else a
            for a in args
        ]
        return " ".join(parts)

    def _status(run: SimRun) -> str:
        if run.status == "failed":
            return f"FAILED({run.exit_code})"
        return run.status

    for run in _SIM_RUNS:
        elapsed = f"{run.elapsed:>9.1f}s"
        status = f"{_status(run):<14}"
        cmd = _format(run.command)
        tr.write_line(f"  {run.np_ranks:>2} ranks  {elapsed}  {status}  {cmd}")
    total = sum(r.elapsed for r in _SIM_RUNS)
    tr.write_line(f"  {'':>7}   {total:>9.1f}s  {'total':<14}  ({len(_SIM_RUNS)} runs)")


@pytest.fixture(scope="session")
def workspace(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    """Per-invocation scratch directory where simulations write their outputs.

    Lifespan follows pytest's tmp_path_retention_policy.
    """
    return tmp_path_factory.mktemp("workspace")


@pytest.fixture()
def test_dir(workspace: pathlib.Path, request: pytest.FixtureRequest) -> pathlib.Path:
    """Per-test subdirectory of the workspace (avoids colliding output names)."""
    name = "".join(c if c.isalnum() or c in "_-" else "_" for c in request.node.name)
    directory = workspace / name
    directory.mkdir()
    return directory
