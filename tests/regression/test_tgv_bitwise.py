"""Bitwise equivalence of streaming patterns on the Taylor-Green vortex.

Runs the same TGV case under the build selected by ``--build-dir`` and under
a reference build directory (default: ``<project>/build``, overridable via
the ``TGV_REFERENCE_BUILD_DIR`` environment variable), and requires the final
macroscopic fields to be BITWISE identical. A streaming pattern changes where
populations live in memory; it must never change the arithmetic.

Covered patterns: AA, AB_PULL, AB_PUSH, ESO_TWIST, ESO_PULL, ESO_PUSH.
"""

from __future__ import annotations

import os
import pathlib
from functools import lru_cache

import numpy as np
import pytest

from tests.lbmtest import ADIOS_CONFIG, BUILD_DIR, PROJECT_ROOT, run_sim
from tests.regression.bp5 import read_last_step, squeeze_2d

REFERENCE_BUILD_DIR = pathlib.Path(
    os.environ.get("TGV_REFERENCE_BUILD_DIR", PROJECT_ROOT / "build")
).resolve()


@lru_cache(maxsize=None)
def _pattern_of(build_dir: pathlib.Path) -> str:
    """Streaming pattern recorded in a build tree's CMakeCache (AB_PULL default)."""
    try:
        for line in (build_dir / "CMakeCache.txt").read_text().splitlines():
            key, sep, value = line.partition("=")
            if sep and key.strip() == "TNL_LBM_STREAMING_PATTERN:STRING":
                return value.strip()
    except OSError:
        pass
    return "AB_PULL"


CURRENT_PATTERN = _pattern_of(BUILD_DIR)
REFERENCE_PATTERN = _pattern_of(REFERENCE_BUILD_DIR)

_skip_reason = None
if not (REFERENCE_BUILD_DIR / "CMakeCache.txt").exists():
    _skip_reason = f"reference build tree {REFERENCE_BUILD_DIR} does not exist"
elif CURRENT_PATTERN == REFERENCE_PATTERN:
    _skip_reason = f"reference build tree uses the same pattern ({CURRENT_PATTERN})"

if _skip_reason:
    pytestmark = pytest.mark.skip(reason=_skip_reason)


def _run_sim_to_results(
    exe: pathlib.Path,
    args: list[str | pathlib.Path],
    workdir: pathlib.Path,
    results_glob: str,
    np_ranks: int = 1,
) -> pathlib.Path:
    workdir.mkdir(parents=True, exist_ok=True)
    run_sim([exe, *args], workdir=workdir, np_ranks=np_ranks)
    candidates = sorted(workdir.glob(results_glob))
    assert candidates, f"{exe.name} produced no results directory"
    return candidates[0]


def _read_2d_tgv_fields(results_dir: pathlib.Path) -> dict[str, np.ndarray]:
    data = read_last_step(
        results_dir / "output_2D_.bp", ["lbm_density", "velocity_x", "velocity_y"]
    )
    return {name: squeeze_2d(arr) for name, arr in data.items()}


def _assert_bitwise_equal(
    candidate: dict[str, np.ndarray],
    reference: dict[str, np.ndarray],
    candidate_label: str,
) -> None:
    for name, ref_arr in reference.items():
        cand_arr = candidate[name]
        if np.array_equal(cand_arr, ref_arr):
            continue
        diff = np.abs(cand_arr.astype(np.float64) - ref_arr.astype(np.float64))
        pytest.fail(
            f"{candidate_label}: {name} is not bitwise identical "
            f"({np.count_nonzero(diff)} cells, max|diff| = {diff.max():.6e})"
        )


def test_tgv2d_bitwise_identical(tmp_path: pathlib.Path) -> None:
    """2D Taylor-Green vortex: final fields bitwise identical to the reference build."""
    results_dir = _run_sim_to_results(
        BUILD_DIR / "sim_2D" / "sim2d_Taylor_Green",
        ["--resolution", "1", "--adios-config", ADIOS_CONFIG],
        tmp_path / "current",
        "results_sim2d_Taylor_Green_*",
    )
    ref_dir = _run_sim_to_results(
        REFERENCE_BUILD_DIR / "sim_2D" / "sim2d_Taylor_Green",
        ["--resolution", "1", "--adios-config", ADIOS_CONFIG],
        tmp_path / "reference",
        "results_sim2d_Taylor_Green_*",
    )
    _assert_bitwise_equal(
        _read_2d_tgv_fields(results_dir), _read_2d_tgv_fields(ref_dir), CURRENT_PATTERN
    )


def test_tgv2d_mpi_bitwise_identical(tmp_path: pathlib.Path) -> None:
    """2D TGV under mpirun -np 2: final fields bitwise identical to the reference.

    The decomposed run exercises the DF halo exchange (per-slot, per-axis,
    parity-dependent descriptors; diagonal ghost-corner coverage) and the
    natural-layout exchange of the initial field before the esoteric
    permutation - all MPI-only code paths the single-rank gate cannot reach.
    """
    results_dir = _run_sim_to_results(
        BUILD_DIR / "sim_2D" / "sim2d_Taylor_Green",
        ["--resolution", "1", "--adios-config", ADIOS_CONFIG],
        tmp_path / "current",
        "results_sim2d_Taylor_Green_*",
        np_ranks=2,
    )
    ref_dir = _run_sim_to_results(
        REFERENCE_BUILD_DIR / "sim_2D" / "sim2d_Taylor_Green",
        ["--resolution", "1", "--adios-config", ADIOS_CONFIG],
        tmp_path / "reference",
        "results_sim2d_Taylor_Green_*",
        np_ranks=2,
    )
    _assert_bitwise_equal(
        _read_2d_tgv_fields(results_dir), _read_2d_tgv_fields(ref_dir), CURRENT_PATTERN
    )


def test_tgv2d_mpi4_bitwise_identical(tmp_path: pathlib.Path) -> None:
    """2D TGV under mpirun -np 4 (2x2 decomposition): final fields bitwise identical.

    A 1D decomposition only ever allocates face buffers in the distributed
    axis - the synchronizer skips all corner buffers when a single axis
    carries the overlap. The 2x2 decomposition (the optimal split of the
    square TGV domain into 4 blocks) distributes both lattice axes, so the
    diagonal ghost-corner exchanges fire for the first time: per-slot corner
    buffers with mixed-pass face axes under EsoTwist's staged passes, and
    the combined-mask corner geometry of the esoteric pull/push schemes.
    """
    results_dir = _run_sim_to_results(
        BUILD_DIR / "sim_2D" / "sim2d_Taylor_Green",
        ["--resolution", "1", "--adios-config", ADIOS_CONFIG],
        tmp_path / "current",
        "results_sim2d_Taylor_Green_*",
        np_ranks=4,
    )
    ref_dir = _run_sim_to_results(
        REFERENCE_BUILD_DIR / "sim_2D" / "sim2d_Taylor_Green",
        ["--resolution", "1", "--adios-config", ADIOS_CONFIG],
        tmp_path / "reference",
        "results_sim2d_Taylor_Green_*",
        np_ranks=4,
    )
    _assert_bitwise_equal(
        _read_2d_tgv_fields(results_dir), _read_2d_tgv_fields(ref_dir), CURRENT_PATTERN
    )


def test_tgv3d_bitwise_identical(tmp_path: pathlib.Path) -> None:
    """3D Taylor-Green vortex: final fields bitwise identical to the reference build."""
    results_dir = _run_sim_to_results(
        BUILD_DIR / "sim_NSE" / "sim_4",
        ["--resolution", "1", "--adios-config", ADIOS_CONFIG],
        tmp_path / "current",
        "results_sim_4_*",
    )
    ref_dir = _run_sim_to_results(
        REFERENCE_BUILD_DIR / "sim_NSE" / "sim_4",
        ["--resolution", "1", "--adios-config", ADIOS_CONFIG],
        tmp_path / "reference",
        "results_sim_4_*",
    )

    def _read_3d_tgv_fields(rd: pathlib.Path) -> dict[str, np.ndarray]:
        sim_4_nest = sorted(rd.glob("res=*/output_3D.bp"))[0]
        return read_last_step(
            sim_4_nest,
            ["lbm_density", "velocity_x", "velocity_y", "velocity_z", "wall"],
        )

    _assert_bitwise_equal(
        _read_3d_tgv_fields(results_dir), _read_3d_tgv_fields(ref_dir), CURRENT_PATTERN
    )
