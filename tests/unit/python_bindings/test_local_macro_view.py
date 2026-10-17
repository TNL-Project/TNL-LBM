"""Test of the ``pytnl_lbm.getLocalMacroView`` binding.

``getLocalMacroView`` returns a strictly-local 4D view (no overlap cells) of
the distributed ``hmacro`` array that can be imported into NumPy via DLPack.
Its anchor-offset contract (the data pointer must point at the first owned
element, not the leading ghost cells) only manifests with non-zero overlaps,
so the actual check runs as an ``mpirun -np 2`` payload; this module launches
it and asserts the exit code.
"""

from __future__ import annotations

import pathlib
import shutil
import subprocess
import sys

import pytest

PAYLOAD = pathlib.Path(__file__).parent / "local_macro_view_check.py"


def test_get_local_macro_view_strictly_local_np2(tmp_path: pathlib.Path) -> None:
    if shutil.which("mpirun") is None:
        pytest.skip("mpirun not available")
    proc = subprocess.run(
        ["mpirun", "-np", "2", sys.executable, str(PAYLOAD)],
        capture_output=True,
        check=False,
        text=True,
        timeout=300.0,
        cwd=tmp_path,
    )
    assert proc.returncode == 0, f"payload failed:\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
