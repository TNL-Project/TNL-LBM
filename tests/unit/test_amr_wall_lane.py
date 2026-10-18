"""AMR wall-refinement lane: frozen-cell C2F read audit under each streaming pattern.

Drives the dedicated ``test_amr_wall_lane_{ab,aa,eso_twist,eso_pull,eso_push}``
doctest binaries (built from ``doctest_main_amr.cu`` +
``tests/unit/test_amr_wall_lane.cu`` with the per-pattern pin; standalone
binaries because the bit-identity manifest pins the consolidated gate
binaries' doctest summary line byte-for-byte, and a co-resident suite
would shift their test-case totals).

The suite registers a wall-adjacent mL1 coupling (GEO_INFLOW slab plane
one coarse cell outside the footprint's min-x face, steering the Sec-7.3
wall guard onto the {c=0 ring, c=1 frozen skin} source pair), drives the
F2C(N-1) -> coarse step N -> C2F fill N closed loop in the sanctioned
reorder (the cycle-end skin F2C deferred to after the fill's read point),
and classifies every audited frozen-skin slot read against the tracked
F2C-authored vintages. The gate is the SP-1 / Defect-1 failing-first
in-situ proof of results_drag_crisis/out_of_scope_survey.md group (3):
A-B pull must classify 100 % fresh/stale (direction-correct at every
parity, the documented 2-cycle staleness); the single-array patterns must
too (the parity-aware frozen-read fix).

Like the other gate arms, a missing binary is a hard failure with a build
hint, never a silent skip; the ``ab`` arm skips on single-pattern AA
trees for the same reason as test_amr_units.py.
"""

from __future__ import annotations

import pathlib
import re
import shutil

import pytest

from tests.lbmtest import ADIOS_CONFIG, BUILD_DIR, run_sim

PATTERNS = ["ab", "aa", "eso_twist", "eso_pull", "eso_push"]


@pytest.mark.parametrize("pattern", PATTERNS)
def test_amr_wall_lane_gate(pattern: str, test_dir: pathlib.Path) -> None:
    binary = BUILD_DIR / "tests" / f"test_amr_wall_lane_{pattern}"
    if not binary.is_file():
        if (
            pattern == "ab"
            and (BUILD_DIR / "tests" / "test_amr_wall_lane_aa").is_file()
        ):
            pytest.skip(
                "single-pattern AA tree: the AB-pinned gate binary is intentionally not built (AMR_TEST_PATTERNS)"
            )
        pytest.fail(
            f"cannot find {binary} — build the gate target first: "
            f"cmake --build {BUILD_DIR} --target test_amr_wall_lane_{pattern}",
            pytrace=False,
        )
    # the suite State ctors read adios2.xml from their cwd
    shutil.copy(ADIOS_CONFIG, test_dir / "adios2.xml")
    stdout = run_sim(
        [str(binary), "--test-suite=amr_wall_lane", "--no-colors", "--no-duration"],
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
