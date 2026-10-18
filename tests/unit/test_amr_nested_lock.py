"""Cross-pattern bitwise lock of the NESTED AMR machinery (mL2 surface).

Drives the ``test_amr_nested_lock`` executable — one binary per tree that
co-instantiates the A-B pull, A-A and the three esoteric in-place streaming
patterns through the explicit ``D3Q27_STREAMING_*`` types (five parallel
runner translation units ``test_amr_nested_lock_*.cu`` plus the kernel-free
comparator ``test_amr_nested_lock.cu``), runs the same three-level
(max-level 2, the L0→L1 link plus one nested L1→L2 link) periodic
``State_AMR`` scenario under each pattern for 10 coarse cycles with the
mL1 lock's traveling sinusoidal density gradient crossing both links, and
asserts per-cycle, per-level BITWISE equality of the GEO map and the
macroscopic channels (rho/vx/vy/vz over the whole stored box, the C2F-fed
ghost rows included) against the A-B pull reference — raw DF frames are
not compared because each pattern stores them in its own parity/layout
encoding.

The companion of the mL1 lock (``test_amr_bitwise_lock``), this arms the
nested-link family the mL1 lock deliberately excludes (the SP-4 surface of
``results_drag_crisis/out_of_scope_survey.md``: the mid-sync C2F fill inside
``advancePair`` that exists only at nesting depth ≥ 2, sourced from the
parent's live post-substep-A state, plus the doubled per-pair F2C cadence
and the level-ascending cycle-end cascade — the seed anatomy of
``results_drag_crisis/aa_seed_report.md`` §4 stage (iv)).

A missing binary is a hard failure with a build hint, never a silent skip:
the binary is built unconditionally in every MPI-configured tree (and in
every ``TNL_LBM_STREAMING_PATTERN`` tree — pattern diversity lives in the
runner TUs, not in per-pattern defines).
"""

from __future__ import annotations

import pathlib
import re
import shutil

import pytest

from tests.lbmtest import ADIOS_CONFIG, BUILD_DIR, run_sim


def test_amr_nested_lock(test_dir: pathlib.Path) -> None:
    binary = BUILD_DIR / "tests" / "test_amr_nested_lock"
    if not binary.is_file():
        pytest.fail(
            f"cannot find {binary} — build the lock target first: "
            f"cmake --build {BUILD_DIR} --target test_amr_nested_lock",
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
