"""Shared setup for the python bindings unit tests."""

from __future__ import annotations

import os
import pathlib
import sys

import pytest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]
BINDINGS_DIR = PROJECT_ROOT / "build" / "pytnl_lbm"
if str(BINDINGS_DIR) not in sys.path:
    sys.path.insert(0, str(BINDINGS_DIR))

# When PyTNL comes from FetchContent (CI, no system install), the `pytnl`
# Python package only exists under the build directory.
PYTNL_BUILD_SRC = PROJECT_ROOT / "build" / "_deps" / "pytnl-build" / "src"
if PYTNL_BUILD_SRC.is_dir() and str(PYTNL_BUILD_SRC) not in sys.path:
    sys.path.insert(0, str(PYTNL_BUILD_SRC))

# OMPI_MCA_accelerator=null avoids MPI_Init aborts on builders where Open MPI
# detects both ROCm and CUDA.
os.environ.setdefault("OMPI_MCA_accelerator", "null")

try:
    import pytnl_lbm  # noqa: F401
except ImportError:
    if list(BINDINGS_DIR.glob("pytnl_lbm*.so")):
        # .so exists but import fails — a real build error, not a missing build
        pytest.fail(
            f"pytnl_lbm shared object exists in {BINDINGS_DIR} but cannot be imported",
            pytrace=False,
        )
    pytest.skip(
        f"python bindings not built ({BINDINGS_DIR / 'pytnl_lbm.*.so'} missing)",
        allow_module_level=True,
    )
