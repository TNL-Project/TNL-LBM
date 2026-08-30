"""AMR gate doctest suites: the four consolidated per-pattern binary runs.

Drives the two ``test_amr_units_{ab,aa}`` executables — the doctest
consolidation of the coupling, subcycling, vtkhdf-writer and nesting test
suites, one binary per streaming pattern (the per-pattern defines select
the A-B/A-A kernels throughout) — once per TEST_SUITE via doctest's
``--test-suite=`` filter. These 2×4 runs are the gate half of the AMR
battery (the ParaView end-to-end arms live in
``tests/integration/test_amr_paraview.py``).

The suites are doctests: every legacy PASS/FAIL site maps one-to-one to a
``CHECK_MESSAGE`` (positive-coverage counting preserved; see the
amr-doctest-port plan), so pytest asserts the doctest all-pass banner
beyond the bare exit code. A missing binary is a hard failure with a
build hint, never a silent skip.
"""

from __future__ import annotations

import pathlib
import re
import shutil

import pytest

from tests.lbmtest import ADIOS_CONFIG, BUILD_DIR, run_sim

PATTERNS = ["ab", "aa"]
GATE_SUITES = ["amr_coupling", "amr_subcycling", "amr_vtkhdf_writer", "amr_nesting"]

GATE_CASES = [(suite, pattern) for pattern in PATTERNS for suite in GATE_SUITES]


@pytest.mark.parametrize(
    "suite,pattern",
    [
        pytest.param(suite, pattern, id=f"{suite}-{pattern}")
        for suite, pattern in GATE_CASES
    ],
)
def test_amr_gate_suite(suite: str, pattern: str, test_dir: pathlib.Path) -> None:
    binary = BUILD_DIR / "tests" / f"test_amr_units_{pattern}"
    if not binary.is_file():
        pytest.fail(
            f"cannot find {binary} — build the gate target first: "
            f"cmake --build {BUILD_DIR} --target test_amr_units_{pattern}",
            pytrace=False,
        )
    # the suite State ctors read adios2.xml from their cwd
    shutil.copy(ADIOS_CONFIG, test_dir / "adios2.xml")
    stdout = run_sim(
        [str(binary), f"--test-suite={suite}", "--no-colors", "--no-duration"],
        workdir=test_dir,
        timeout=600.0,
    )
    # doctest all-pass banner: every registered case passed and no
    # assertion failed (the exit code alone only proves the runner finished)
    cases_banner = re.search(
        r"\[doctest\] test cases: +(\d+) \| +\d+ passed \| +0 failed", stdout
    )
    assert cases_banner is not None and int(cases_banner.group(1)) > 0
    assert re.search(
        r"\[doctest\] assertions: +\d+ \| +\d+ passed \| +0 failed", stdout
    )
