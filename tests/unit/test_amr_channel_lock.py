"""Cross-pattern channel-geometry lock of the AMR machinery (the SP-5 surface).

Drives the ``test_amr_channel_lock`` executable — one binary per tree that
co-instantiates the A-B pull, A-A and the three esoteric in-place streaming
patterns through the explicit ``D3Q27_STREAMING_*`` types (five parallel
runner translation units ``test_amr_channel_lock_*.cu`` plus the kernel-free
comparator ``test_amr_channel_lock.cu``), runs the same developing-channel
mL1 scenario (sim_AMR_channel's ``--resolution 1`` boundary/IC plumbing with
TWO footprints: one GEO_INFLOW_MOMENT-adjacent — the R = 1 Sec. 7.3 guarded
inflow class of ``results_drag_crisis/aa_seed_report.md`` §6 — and the
channel's own wall-attached slab) under each pattern for 10 coarse cycles,
and compares the init-time GEO map, the macroscopic channels and the decoded
canonical DF frames (each pattern's pre-collision population per site and
direction, a convention-free physical surface that resolves differences the
macroscopic channels are blind to) cycle by cycle against the A-B pull
reference, with the divergence anatomy classified by map-tag region.

The comparator prints the per-cycle evidence census (first-divergent cycle,
per-level/per-surface cell counts with region histograms); the armed
assertions are the pattern-independent init-time topology (map tagging must
be bitwise identical across patterns) and the classification contract of the
survey's SP-5 partition verdict.

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


def test_amr_channel_lock(test_dir: pathlib.Path) -> None:
    binary = BUILD_DIR / "tests" / "test_amr_channel_lock"
    if not binary.is_file():
        pytest.fail(
            f"cannot find {binary} — build the lock target first: "
            f"cmake --build {BUILD_DIR} --target test_amr_channel_lock",
            pytrace=False,
        )
    # the scenario State ctors read adios2.xml from their cwd
    shutil.copy(ADIOS_CONFIG, test_dir / "adios2.xml")
    stdout = run_sim(
        [str(binary), "--test-suite=amr_channel_lock", "--no-colors", "--no-duration"],
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
