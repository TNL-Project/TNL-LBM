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

# OMPI_MCA_accelerator=null avoids MPI_Init aborts on builders where Open MPI
# detects both ROCm and CUDA.
os.environ.setdefault("OMPI_MCA_accelerator", "null")

pytest.importorskip(
    "pytnl_lbm",
    reason="python bindings not built (build/pytnl_lbm/pytnl_lbm.*.so missing)",
)
