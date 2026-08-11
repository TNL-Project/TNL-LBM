"""Multi-rank MPI regression tests.

Every other regression module executes simulations single-rank. This module
runs a subset under multiple MPI ranks (domain decomposition along X, rank
synchronization, subfile-aggregated BP5 output) and checks the same physical
properties with the same tolerances as the single-rank modules.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from tests.lbmtest import (
    BUILD_DIR,
    PROJECT_ROOT,
    FieldData,
    assert_all_finite,
    assert_mass_conserved,
    run_sim,
)
from tests.regression.bp5 import read_last_step, squeeze_2d

if TYPE_CHECKING:
    import pathlib

# GEO enum values (must match include/lbm3d/d2q9/bc.h)
GEO2D_WALL = 1
GEO2D_INFLOW_LEFT = 3
GEO2D_NOTHING = 8

# GEO enum values (must match include/lbm3d/d3q27/bc.h)
GEO3D_WALL = 1
GEO3D_INFLOW_LEFT = 3
GEO3D_NOTHING = 10

ADIOS_CONFIG = str(PROJECT_ROOT / "adios2.xml")
NP_RANKS = 3


@pytest.fixture(scope="module")
def mpi_workspace(workspace: pathlib.Path) -> pathlib.Path:
    """Dedicated run directory so MPI results cannot shadow the np=1 ones."""
    directory = workspace / "mpi"
    directory.mkdir()
    return directory


@pytest.fixture(scope="module")
def poiseuille_dir(mpi_workspace: pathlib.Path) -> pathlib.Path:
    """Run sim2d_2 (Hagen-Poiseuille) under multiple ranks."""
    run_sim(
        [
            BUILD_DIR / "sim_2D" / "sim2d_2",
            "--resolution",
            "1",
            "--adios-config",
            ADIOS_CONFIG,
        ],
        workdir=mpi_workspace,
        np_ranks=NP_RANKS,
    )
    candidates = sorted(mpi_workspace.glob("results_sim2d_2_*"))
    assert candidates, "sim2d_2 produced no results directory"
    return candidates[0]


@pytest.fixture(scope="module")
def channel_dir(mpi_workspace: pathlib.Path) -> pathlib.Path:
    """Run sim_1 (3D channel with baffle hole) under multiple ranks."""
    run_sim(
        [
            BUILD_DIR / "sim_NSE" / "sim_1",
            "--resolution",
            "1",
            "--adios-config",
            ADIOS_CONFIG,
        ],
        workdir=mpi_workspace,
        np_ranks=NP_RANKS,
    )
    candidates = sorted(mpi_workspace.glob("results_sim_1_*"))
    assert candidates, "sim_1 produced no results directory"
    return candidates[0]


class TestPoiseuilleMPI:
    """Multi-rank sim2d_2: checks mirror the single-rank TestSim2d2."""

    @pytest.fixture(scope="class")
    def data(self, poiseuille_dir: pathlib.Path) -> FieldData:
        data = read_last_step(
            poiseuille_dir / "output_2D_.bp",
            [
                "lbm_density",
                "velocity_x",
                "velocity_y",
                "error_vx",
                "lbm_error_vx",
                "wall",
            ],
        )
        return {name: squeeze_2d(arr) for name, arr in data.items()}

    def test_finiteness(self, data: FieldData) -> None:
        assert_all_finite(data)

    def test_mass_conservation(self, data: FieldData) -> None:
        assert_mass_conserved(data["lbm_density"], tolerance=1e-3)

    def test_analytical_error_phys(self, data: FieldData) -> None:
        mask = data["wall"] != GEO2D_NOTHING
        max_err = float(np.max(data["error_vx"][mask]))
        assert max_err < 5e-3, f"max|error_vx|(phys)={max_err:.2e} (tol=5e-3)"

    def test_analytical_error_lbm(self, data: FieldData) -> None:
        mask = data["wall"] != GEO2D_NOTHING
        max_err = float(np.max(data["lbm_error_vx"][mask]))
        assert max_err < 3e-3, f"max|lbm_error_vx|(lattice)={max_err:.2e} (tol=3e-3)"

    def test_wall_no_slip(self, data: FieldData) -> None:
        solid_mask = (data["wall"] == GEO2D_WALL) | (data["wall"] == GEO2D_NOTHING)
        assert solid_mask.any(), "no wall/nothing cells found"
        max_v = max(
            float(np.max(np.abs(data["velocity_x"][solid_mask]))),
            float(np.max(np.abs(data["velocity_y"][solid_mask]))),
        )
        assert max_v < 1e-10, f"max|v| in wall/nothing cells={max_v:.2e}"


class TestChannelMPI:
    """Multi-rank sim_1: checks mirror the single-rank TestSim1."""

    @pytest.fixture(scope="class")
    def data(self, channel_dir: pathlib.Path) -> FieldData:
        return read_last_step(
            channel_dir / "output_3D.bp",
            ["lbm_density", "velocity_x", "velocity_y", "velocity_z", "wall"],
        )

    def test_finiteness(self, data: FieldData) -> None:
        assert_all_finite(data)

    def test_mass_conservation(self, data: FieldData) -> None:
        assert_mass_conserved(data["lbm_density"], tolerance=5e-3)

    def test_symmetry_y(self, data: FieldData) -> None:
        # The hole is centered in Y and Z, so a Galilean-invariant collision
        # operator keeps the jet perfectly centered. Exclude the outermost
        # layers (GEO_NOTHING) before comparing.
        vx = data["velocity_x"]
        nz, ny = vx.shape[0], vx.shape[1]
        vx_inner = vx[1 : nz - 1, 1 : ny - 1, :]
        max_diff = float(np.max(np.abs(vx_inner - vx_inner[:, ::-1, :])))
        peak = float(np.max(np.abs(vx)))
        rel = max_diff / peak * 100
        assert max_diff < 1e-6, (
            f"max|vx(y)-vx(Y-1-y)|={max_diff:.2e} ({rel:.4f}% of peak)"
        )

    def test_symmetry_z(self, data: FieldData) -> None:
        vx = data["velocity_x"]
        nz, ny = vx.shape[0], vx.shape[1]
        vx_inner = vx[1 : nz - 1, 1 : ny - 1, :]
        max_diff = float(np.max(np.abs(vx_inner - vx_inner[::-1, :, :])))
        peak = float(np.max(np.abs(vx)))
        rel = max_diff / peak * 100
        assert max_diff < 1e-6, (
            f"max|vx(z)-vx(Z-1-z)|={max_diff:.2e} ({rel:.4f}% of peak)"
        )

    def test_inflow_uniform(self, data: FieldData) -> None:
        vx, wall = data["velocity_x"], data["wall"]
        inflow_mask = wall[:, :, 1] == GEO3D_INFLOW_LEFT
        assert inflow_mask.any(), "no inflow cells found at x=1"
        inflow_vx = vx[:, :, 1][inflow_mask]
        spread = float(np.max(inflow_vx) - np.min(inflow_vx))
        assert spread < 1e-6, (
            f"inflow vx spread={spread:.2e} (vx={float(np.mean(inflow_vx)):.6f})"
        )

    def test_wall_no_slip(self, data: FieldData) -> None:
        solid_mask = (data["wall"] == GEO3D_WALL) | (data["wall"] == GEO3D_NOTHING)
        assert solid_mask.any(), "no wall/nothing cells found"
        max_v = max(
            float(np.max(np.abs(data["velocity_x"][solid_mask]))),
            float(np.max(np.abs(data["velocity_y"][solid_mask]))),
            float(np.max(np.abs(data["velocity_z"][solid_mask]))),
        )
        assert max_v < 1e-10, f"max|v| in wall/nothing cells={max_v:.2e}"
