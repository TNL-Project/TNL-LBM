"""Compile-and-run exactness locks for the F2C_SCHONHERR opt-in (T14).

Drives the two ``test_amr_f2c_schonherr_{ab,aa}`` executables — one binary
per streaming pattern, compiled from ``tests/test_amr_f2c_schonherr.cu``
with ``F2C_SCHONHERR`` hardcoded (Schönherr ch7 conversion, commit 13 /
plan row 14).

The define is a per-TU compile-time switch selecting the thesis §7.2
σ-form compact-moment transfer (σ = 2) inside ``cudaAMR_FineToCoarse`` and
cannot share a binary with the default Lagrava-filter build, so it locks
as standalone per-pattern binaries — the same pytest-side,
build-variants-in-the-default-build idiom as the ``test_amr_c2f_smoke_*``
debug-define binaries.  The default build
(``tests/test_amr_coupling.cu`` + ``tests/run-amr-tests.sh``) pins the
Lagrava path separately, so the two batteries are green under both
strategies.

Each binary runs the F2C transfer on (i) a uniform field, (ii) a
CE-consistent linear field, and (iii) a CE-consistent quadratic-velocity
field, and asserts the T14 exactness classes: constant exact; linear
velocity exact; quadratic-velocity + linear-density exact at t = (0,0,0);
CE-consistent strain round-trip at σ = 2; and Σf = d0 exactly at the
destination (see the source header for the derivation and tolerance
documentation).  A missing binary is a hard failure with a build hint,
never a silent skip.
"""

from __future__ import annotations

import pathlib

import pytest

from tests.lbmtest import BUILD_DIR, run_sim

PATTERNS = ["ab", "aa"]


def _binary_path(pattern: str) -> pathlib.Path:
    return BUILD_DIR / "tests" / f"test_amr_f2c_schonherr_{pattern}"


@pytest.mark.parametrize("pattern", PATTERNS)
def test_f2c_schonherr_exactness(pattern: str, test_dir: pathlib.Path) -> None:
    binary = _binary_path(pattern)
    if not binary.is_file():
        pytest.fail(
            f"cannot find {binary} — build the smoke targets first: "
            f"cmake --build {BUILD_DIR} --target test_amr_f2c_schonherr_{pattern}",
            pytrace=False,
        )
    stdout = run_sim([str(binary)], workdir=test_dir, timeout=120.0)
    assert "RESULT: all AMR F2C Schönherr exactness locks passed" in stdout
