"""D3Q27 adjoint LBM workflow regression tests.

The workflow has two stages, executed once per test session (module-scoped
fixture):

1. ``sim_pseudomeasure`` — generates synthetic "measured" data;
2. ``sim_adjoint`` — runs primary + adjoint for a few epochs, iterates to
   minimize the loss.

Validated properties:

- Pseudomeasure: 3D BP5 fields finite, mass conservation, wall no-slip,
  inflow present, measured macro BP5 file readable.
- Adjoint: loss function finite/positive and non-increasing across epochs,
  velocity profiles (X, Y, Z) finite, primary and adjoint 3D fields finite.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from tests.lbmtest import (
    BUILD_DIR,
    PROJECT_ROOT,
    assert_all_finite,
    assert_mass_conserved,
    run_sim,
)
from tests.regression.bp5 import read_last_step

if TYPE_CHECKING:
    import pathlib

# D3Q27 GEO enum (subset relevant to adjoint sims)
GEO_WALL = 1
GEO_INFLOW_BB_LEFT = 4

RESOLUTION = 1
EPOCHS = 5
# The default --eps (1e-3) overshoots the time-accumulated sensitivity at this
# resolution and the first epoch lands strictly worse; 1e-5 descends.
EPS = "1e-5"
# Each epoch runs a primary (physFinalTime=4) and an adjoint (16) simulation.
WORKFLOW_TIMEOUT = 3600.0

RESULTS_PSEUDOMEASURE = "results_sim_pseudomeasure_*"
RESULTS_ADJOINT_DATA = "results_adjoint"


@pytest.fixture(scope="module")
def adjoint_workflow(workspace: pathlib.Path) -> dict[str, pathlib.Path]:
    """Run the pseudomeasure + adjoint workflow; return key directories."""
    adios_config = str(PROJECT_ROOT / "adios2.xml")
    run_sim(
        [
            BUILD_DIR / "sim_adjoint" / "sim_pseudomeasure",
            "--resolution",
            str(RESOLUTION),
            "--adios-config",
            adios_config,
        ],
        workdir=workspace,
    )
    run_sim(
        [
            BUILD_DIR / "sim_adjoint" / "sim_adjoint",
            "--resolution",
            str(RESOLUTION),
            "--epochs",
            str(EPOCHS),
            "--eps",
            EPS,
            "--adios-config",
            adios_config,
        ],
        workdir=workspace,
        timeout=WORKFLOW_TIMEOUT,
    )

    pseudomeasure = sorted(workspace.glob(RESULTS_PSEUDOMEASURE))
    assert pseudomeasure, "sim_pseudomeasure produced no results directory"
    adjoint_data = sorted(workspace.glob(f"{RESULTS_ADJOINT_DATA}/adjoint_data_*"))
    assert adjoint_data, "sim_adjoint produced no adjoint_data directory"
    return {
        "workspace": workspace,
        "pseudomeasure": pseudomeasure[0],
        "adjoint_data": adjoint_data[0],
    }


class TestPseudomeasure:
    """Synthetic measurement generation stage."""

    @pytest.fixture(scope="class")
    def data(self, adjoint_workflow: dict[str, pathlib.Path]) -> dict[str, np.ndarray]:
        return read_last_step(
            adjoint_workflow["pseudomeasure"] / "output_3D.bp",
            [
                "lbm_density",
                "lbm_velocity_x",
                "lbm_velocity_y",
                "lbm_velocity_z",
                "wall",
            ],
        )

    def test_finiteness(self, data: dict[str, np.ndarray]) -> None:
        assert_all_finite(data)

    def test_mass_conservation(self, data: dict[str, np.ndarray]) -> None:
        assert_mass_conserved(data["lbm_density"], tolerance=1e-3)

    def test_wall_no_slip(self, data: dict[str, np.ndarray]) -> None:
        solid_mask = data["wall"] == GEO_WALL
        assert solid_mask.any(), "no wall cells found"
        max_v = max(
            float(np.max(np.abs(data["lbm_velocity_x"][solid_mask]))),
            float(np.max(np.abs(data["lbm_velocity_y"][solid_mask]))),
            float(np.max(np.abs(data["lbm_velocity_z"][solid_mask]))),
        )
        assert max_v < 1e-10, f"max|v| in wall cells={max_v:.2e}"

    def test_inflow_present(self, data: dict[str, np.ndarray]) -> None:
        count = int((data["wall"] == GEO_INFLOW_BB_LEFT).sum())
        assert count > 0, "no GEO_INFLOW_BB_LEFT cells found"

    def test_measured_file(self, adjoint_workflow: dict[str, pathlib.Path]) -> None:
        measured = adjoint_workflow["adjoint_data"] / "macro_measured.bp.bp"
        assert measured.exists(), f"measured macro file not found: {measured}"


class TestAdjoint:
    """Primary + adjoint optimization stage."""

    @pytest.fixture(scope="class")
    def loss_values(self, adjoint_workflow: dict[str, pathlib.Path]) -> list[float]:
        loss_file = adjoint_workflow["adjoint_data"] / "lossFunction.txt"
        assert loss_file.exists(), f"loss function file not found: {loss_file}"
        return [float(line) for line in loss_file.read_text().strip().splitlines()]

    def test_loss_finite_positive(self, loss_values: list[float]) -> None:
        assert loss_values, "loss function file is empty"
        assert all(np.isfinite(v) and v > 0 for v in loss_values), (
            f"loss values: {loss_values}"
        )

    def test_loss_monotonic(self, loss_values: list[float]) -> None:
        # Fewer than 2 recorded epochs means the optimizer made no progress
        # (a rejected step is never reverted, so it deadlocks — a regression).
        assert len(loss_values) >= 2, (
            f"optimizer recorded only {len(loss_values)} epoch(s): {loss_values}"
        )
        increases = [
            (i, loss_values[i - 1], loss_values[i])
            for i in range(1, len(loss_values))
            if loss_values[i] > loss_values[i - 1]
        ]
        assert not increases, f"loss increased across epochs: {increases}"

    @pytest.mark.parametrize("axis", ["X", "Y", "Z"])
    def test_velocity_profile(
        self, adjoint_workflow: dict[str, pathlib.Path], axis: str
    ) -> None:
        profile_file = adjoint_workflow["adjoint_data"] / f"velocityProfile{axis}.txt"
        assert profile_file.exists(), f"velocity profile not found: {profile_file}"
        profile = np.loadtxt(profile_file)
        n_bad = int(np.sum(~np.isfinite(profile)))
        assert n_bad == 0, f"velocityProfile{axis}.txt has {n_bad} non-finite values"

    def test_primary_3d_finiteness(
        self, adjoint_workflow: dict[str, pathlib.Path]
    ) -> None:
        primary_dirs = sorted(
            adjoint_workflow["workspace"].glob("results_sim_primary_*")
        )
        assert primary_dirs, "no results_sim_primary_* directory found"
        data = read_last_step(
            primary_dirs[0] / "output_3D.bp",
            ["lbm_density", "lbm_velocity_x", "lbm_velocity_y", "lbm_velocity_z"],
        )
        assert_all_finite(data)

    def test_primary_mass(self, adjoint_workflow: dict[str, pathlib.Path]) -> None:
        primary_dirs = sorted(
            adjoint_workflow["workspace"].glob("results_sim_primary_*")
        )
        assert primary_dirs, "no results_sim_primary_* directory found"
        data = read_last_step(primary_dirs[0] / "output_3D.bp", ["lbm_density"])
        assert_mass_conserved(data["lbm_density"], tolerance=5e-3)

    def test_adjoint_3d_finiteness(
        self, adjoint_workflow: dict[str, pathlib.Path]
    ) -> None:
        adjoint_dirs = sorted(
            adjoint_workflow["workspace"].glob("results_sim_adjoint_*")
        )
        assert adjoint_dirs, "no results_sim_adjoint_* directory found"
        data = read_last_step(
            adjoint_dirs[0] / "output_3D.bp",
            ["lbm_density", "lbm_velocity_x", "lbm_velocity_y", "lbm_velocity_z"],
        )
        assert_all_finite(data)
