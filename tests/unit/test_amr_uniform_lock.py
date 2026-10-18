"""Cross-pattern bitwise lock of the UNIFORM AMR fallback (mL0 surface).

Drives the ``test_amr_uniform_lock`` executable — one binary per tree that
co-instantiates the A-B pull, A-A and the three esoteric in-place streaming
patterns through the explicit ``D3Q27_STREAMING_*`` types (five parallel
runner translation units ``test_amr_uniform_lock_*.cu`` plus the kernel-free
comparator ``test_amr_uniform_lock.cu``), runs the same uniform
(``max_level == 0``, single block, the driver falls back to the base
``State<NSE>::SimUpdate`` byte-identically) ``State_AMR`` scenario of the
sim_AMR_ball channel geometry under each pattern for 10 coarse cycles, and
asserts per-cycle BITWISE equality of the GEO map and the macroscopic
channels against the A-B pull reference — raw DF frames are not compared
because each pattern stores them in its own parity/layout encoding.

This arms the SP-6 surface of ``results_drag_crisis/out_of_scope_survey.md``
(uniform ``max_level == 0`` under the single-array patterns; the historical
abort of ``results_drag_crisis/aa_seed_report.md`` §7): the comparator
additionally asserts the GEO_NOTHING frame cleanliness of the uniform map on
every pattern (the ghost-layer idiom the single-array patterns' unclamped
streaming depends on). The sim-side guard that repairs the stray-stamp seed
is exercised live by the ``sim_AMR_ball --max-level 0`` reproducer (see
``results_drag_crisis/sp6/``); this lock pins the repaired uniform fallback
against regression.

The companion of the mL1 and nested locks (``test_amr_bitwise_lock``,
``test_amr_nested_lock``). A missing binary is a hard failure with a build
hint, never a silent skip: the binary is built unconditionally in every
MPI-configured tree (and in every ``TNL_LBM_STREAMING_PATTERN`` tree —
pattern diversity lives in the runner TUs, not in per-pattern defines).
"""

from __future__ import annotations

import pathlib
import re
import shutil

import pytest

from tests.lbmtest import ADIOS_CONFIG, BUILD_DIR, run_sim


def test_amr_uniform_lock(test_dir: pathlib.Path) -> None:
    binary = BUILD_DIR / "tests" / "test_amr_uniform_lock"
    if not binary.is_file():
        pytest.fail(
            f"cannot find {binary} — build the lock target first: "
            f"cmake --build {BUILD_DIR} --target test_amr_uniform_lock",
            pytrace=False,
        )
    # the scenario State ctors read adios2.xml from their cwd
    shutil.copy(ADIOS_CONFIG, test_dir / "adios2.xml")
    stdout = run_sim(
        [str(binary), "--no-colors", "--no-duration"],
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
