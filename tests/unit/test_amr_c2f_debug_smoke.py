"""Compile-and-run smoke locks for the C2F_*_ONLY debug defines (T10g).

Drives the eight ``test_amr_c2f_smoke_{eq,dev,norm,shear}_{ab,aa}``
executables — one binary per seam-investigation debug define
(``C2F_EQ_ONLY`` / ``C2F_DEV_ONLY`` / ``C2F_NORM_ONLY`` /
``C2F_SHEAR_ONLY``) per streaming pattern, compiled from
``tests/test_amr_c2f_debug_smoke.cu`` (Schönherr ch7 conversion, commit 10
/ plan row 11 T10g).

The defines are per-TU compile-time switches inside the default
compact-moment branch of ``cudaAMR_CoarseToFine`` and cannot share a binary
with the default build, so they smoke as standalone per-define binaries —
the same pytest-side, build-variants-in-the-default-build idiom as the
``_ab``/``_aa`` AMR test binaries (the defines themselves are the
compile-time switches the seam investigation rebuilds ``sim_AMR`` with).

Each binary runs the nominal CM fill on a CE-consistent linear field and
asserts (S1) the reconstructed macros against the analytic field at the
Tests-8/9 tolerance class and (S2) the recovered non-equilibrium pressure
tensor against the define-filtered analytic strain targets (see the source
header for the derivation and tolerance documentation).  A missing binary
is a hard failure with a build hint, never a silent skip.
"""

from __future__ import annotations

import pathlib

import pytest

from tests.lbmtest import BUILD_DIR, run_sim

SMOKE_DEFINES = ["eq", "dev", "norm", "shear"]
PATTERNS = ["ab", "aa"]

BINARY_CASES = [(define, pattern) for define in SMOKE_DEFINES for pattern in PATTERNS]


def _binary_path(define: str, pattern: str) -> pathlib.Path:
    return BUILD_DIR / "tests" / f"test_amr_c2f_smoke_{define}_{pattern}"


@pytest.mark.parametrize(
    "define,pattern",
    [
        pytest.param(define, pattern, id=f"{define}-{pattern}")
        for define, pattern in BINARY_CASES
    ],
)
def test_c2f_debug_define_smoke(
    define: str, pattern: str, test_dir: pathlib.Path
) -> None:
    binary = _binary_path(define, pattern)
    if not binary.is_file():
        pytest.fail(
            f"cannot find {binary} — build the smoke targets first: "
            f"cmake --build {BUILD_DIR} --target test_amr_c2f_smoke_{define}_{pattern}",
            pytrace=False,
        )
    stdout = run_sim([str(binary)], workdir=test_dir, timeout=120.0)
    assert "RESULT: all AMR C2F debug-define smoke checks passed" in stdout
