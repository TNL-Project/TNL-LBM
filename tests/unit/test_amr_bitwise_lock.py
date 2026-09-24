"""Cross-pattern bitwise lock of the AMR machinery (mL1 surface).

Drives the ``test_amr_bitwise_lock`` executable — one binary per tree that
co-instantiates the A-B pull, A-A and the three esoteric in-place streaming
patterns through the explicit ``D3Q27_STREAMING_*`` types (five parallel
runner translation units ``test_amr_bitwise_lock_*.cu`` plus the kernel-free
comparator ``test_amr_bitwise_lock.cu``), runs the same two-level
(max-level 1, single L0→L1 link) periodic ``State_AMR`` scenario
under each pattern for 10 coarse cycles with a traveling sinusoidal density
gradient crossing the link, and asserts per-cycle, per-level BITWISE
equality of the GEO map and the macroscopic channels (rho/vx/vy/vz over the
whole stored box, the C2F-fed ghost rows included) against the A-B pull
reference — raw DF frames are not compared because each pattern stores them
in its own parity/layout encoding.

The lock surface is deliberately narrow — only what is proven/provable
exact: A-A ≡ A-B pull on the mL1 class is proven by the production evidence
harness (``results_drag_crisis/aa_seed_report.md`` §2: bit-exact for 18
cycles through front-crossing cascade fills and per-cycle F2C feedback; §8
item 2 recommends exactly this in-binary harness), and the esoteric
patterns are locked after empirical bitwise verification on this host. The
KNOWN DEFECTS — nested ≥2-level links (mid-sync C2F vintage authoring,
report §4), the R = 1 inflow-adjacent C2F wall-guard path (§6), and mL0
under A-A (§7) — are NOT locked; the binary carries a skipped documentation
arm naming them with the report pointer.

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


def test_amr_bitwise_lock(test_dir: pathlib.Path) -> None:
    binary = BUILD_DIR / "tests" / "test_amr_bitwise_lock"
    if not binary.is_file():
        pytest.fail(
            f"cannot find {binary} — build the lock target first: "
            f"cmake --build {BUILD_DIR} --target test_amr_bitwise_lock",
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
