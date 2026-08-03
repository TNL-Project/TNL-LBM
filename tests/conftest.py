"""Shared pytest fixtures and hooks for the TNL-LBM test suite."""

from __future__ import annotations

import pathlib

import cuda.bindings.driver as cuda_driver
import pytest


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
