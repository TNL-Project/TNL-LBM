"""AMR wall-corner C2F destination-partition gate: the symmetric corner-column contract.

Drives the dedicated ``test_amr_wall_corners_{ab,aa,eso_twist,eso_pull,eso_push}``
doctest binaries (built from ``doctest_main_amr.cu`` +
``tests/unit/test_amr_wall_corners.cu`` with the per-pattern pin; standalone
binaries because the bit-identity manifest pins the consolidated gate
binaries' doctest summary line byte-for-byte, and a co-resident suite
would shift their test-case totals).

The suite runs eight wall-attachment topologies of the 16^3 mL1 fixture
(the sim_AMR_channel wall-attachment class rotated onto every axis
combination): single wall-attached faces (y_min/y_max -- the 3d963c40
surface -- and x_min, its symmetric complement), the two-wall corners
(y_z, x_z, x_y), the three-wall corner (x_y_z) and the all-max mirror
(right_top_front). Each scenario pins the fine block's wall mask, the
six-face C2F destination patch table (the partition spec with the
symmetric corner-column widening), the disjointness/coverage census of
the corner columns, and the fill-activity witness over driven coarse
cycles (a partition hole leaves the corner columns at the initial
condition forever).

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
def test_amr_wall_corners_gate(pattern: str, test_dir: pathlib.Path) -> None:
    binary = BUILD_DIR / "tests" / f"test_amr_wall_corners_{pattern}"
    if not binary.is_file():
        if pattern == "ab" and (BUILD_DIR / "tests" / "test_amr_wall_corners_aa").is_file():
            pytest.skip("single-pattern AA tree: the AB-pinned gate binary is intentionally not built (AMR_TEST_PATTERNS)")
        pytest.fail(
            f"cannot find {binary} — build the gate target first: "
            f"cmake --build {BUILD_DIR} --target test_amr_wall_corners_{pattern}",
            pytrace=False,
        )
    # the suite State ctors read adios2.xml from their cwd
    shutil.copy(ADIOS_CONFIG, test_dir / "adios2.xml")
    stdout = run_sim(
        [str(binary), "--test-suite=amr_wall_corners", "--no-colors", "--no-duration"],
        workdir=test_dir,
        timeout=600.0,
    )
    # doctest all-pass banner: every registered case passed and no
    # assertion failed (the exit code alone only proves the runner finished)
    cases_banner = re.search(r"\[doctest\] test cases: +(\d+) \| +\d+ passed \| +0 failed", stdout)
    assert cases_banner is not None and int(cases_banner.group(1)) > 0
    assert re.search(r"\[doctest\] assertions: +\d+ \| +\d+ passed \| +0 failed", stdout)
