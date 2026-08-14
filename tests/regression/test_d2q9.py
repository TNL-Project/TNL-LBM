"""D2Q9 regression tests.

Each simulation is executed once per test session (module-scoped fixture) and
its final ADIOS2 BP5 output step is validated property by property. The
checked properties are known to hold for the correct implementation
(Geier 2017 CLBM + fixed boundary conditions):

- ``sim2d_1`` (channel with baffle hole): centerline symmetry (Galilean
  invariance), inflow velocity uniformity, wall no-slip.
- ``sim2d_2`` (Hagen-Poiseuille): analytical accuracy, mass conservation,
  wall no-slip.
- ``sim2d_hills``: SYM_TOP smoothness (no frozen-population spike), mass
  conservation, inflow uniformity.
- ``sim2d_Taylor_Green``: analytical decay accuracy, mass conservation,
  velocity symmetry.

Thresholds are tuned from confirmed-correct behaviour at resolution 1.
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
GEO_WALL = 1
GEO_INFLOW_LEFT = 3
GEO_NOTHING = 7
GEO_SYM_TOP = 8

SIMS = ["sim2d_1", "sim2d_2", "sim2d_hills", "sim2d_Taylor_Green"]


@pytest.fixture(scope="module")
def d2q9_results(workspace: pathlib.Path) -> dict[str, pathlib.Path]:
    """Run all D2Q9 example simulations and return their results directories."""
    outputs: dict[str, pathlib.Path] = {}
    adios_config = str(PROJECT_ROOT / "adios2.xml")
    for sim in SIMS:
        run_sim(
            [
                BUILD_DIR / "sim_2D" / sim,
                "--resolution",
                "1",
                "--adios-config",
                adios_config,
            ],
            workdir=workspace,
        )
        candidates = sorted(workspace.glob(f"results_{sim}_*"))
        assert candidates, f"{sim} produced no results directory"
        bp = candidates[0] / "output_2D_.bp"
        assert bp.exists(), f"{sim} did not produce {bp}"
        outputs[sim] = candidates[0]
    return outputs


def read_2d_fields(results_dir: pathlib.Path, var_names: list[str]) -> FieldData:
    """Read last-step fields squeezed to 2D."""
    data = read_last_step(results_dir / "output_2D_.bp", var_names)
    return {name: squeeze_2d(arr) for name, arr in data.items()}


def wall_velocity_max(vx: np.ndarray, vy: np.ndarray, wall: np.ndarray) -> float:
    """Max velocity magnitude in wall and nothing cells."""
    solid_mask = (wall == GEO_WALL) | (wall == GEO_NOTHING)
    if not solid_mask.any():
        return 0.0
    return max(
        float(np.max(np.abs(vx[solid_mask]))) if vx[solid_mask].size else 0.0,
        float(np.max(np.abs(vy[solid_mask]))) if vy[solid_mask].size else 0.0,
    )


class TestSim2d1:
    """sim2d_1: channel with a baffle hole centered in Y."""

    @pytest.fixture(scope="class")
    def data(self, d2q9_results: dict[str, pathlib.Path]) -> FieldData:
        return read_2d_fields(
            d2q9_results["sim2d_1"], ["lbm_density", "velocity_x", "velocity_y", "wall"]
        )

    def test_finiteness(self, data: FieldData) -> None:
        assert_all_finite(data)

    def test_centerline_symmetry(self, data: FieldData) -> None:
        # The baffle hole is centered (40-60% of Y), so the geometry is
        # mirror-symmetric about the channel centre. A Galilean-invariant
        # collision operator (Geier 2017 CLBM) keeps the jet perfectly centred;
        # the anisotropic Straka 2016 operator deflects it.
        vx = data["velocity_x"]
        ny = vx.shape[0]
        # Exclude the outermost rows (GEO_NOTHING) before comparing.
        vx_inner = vx[1 : ny - 1, :]
        max_diff = float(np.max(np.abs(vx_inner - vx_inner[::-1, :])))
        peak = float(np.max(np.abs(vx)))
        rel = max_diff / peak * 100
        assert max_diff < 1e-6, (
            f"max|vx(y)-vx(Y-1-y)|={max_diff:.2e} ({rel:.4f}% of peak)"
        )

    def test_inflow_uniform(self, data: FieldData) -> None:
        vx, wall = data["velocity_x"], data["wall"]
        inflow_mask = wall[:, 1] == GEO_INFLOW_LEFT
        assert inflow_mask.any(), "no inflow cells found at x=1"
        inflow_vx = vx[inflow_mask, 1]
        spread = float(np.max(inflow_vx) - np.min(inflow_vx))
        assert spread < 1e-6, (
            f"inflow vx spread={spread:.2e} (vx={float(np.mean(inflow_vx)):.6f})"
        )

    def test_wall_no_slip(self, data: FieldData) -> None:
        max_v = wall_velocity_max(data["velocity_x"], data["velocity_y"], data["wall"])
        assert max_v < 1e-10, f"max|v| in wall/nothing cells={max_v:.2e}"


class TestSim2d2:
    """sim2d_2: Hagen-Poiseuille channel flow with analytical reference."""

    @pytest.fixture(scope="class")
    def data(self, d2q9_results: dict[str, pathlib.Path]) -> FieldData:
        return read_2d_fields(
            d2q9_results["sim2d_2"],
            [
                "lbm_density",
                "velocity_x",
                "velocity_y",
                "error_vx",
                "lbm_error_vx",
                "wall",
            ],
        )

    def test_finiteness(self, data: FieldData) -> None:
        assert_all_finite(data)

    def test_mass_conservation(self, data: FieldData) -> None:
        assert_mass_conserved(data["lbm_density"], tolerance=1e-3)

    def test_analytical_error_phys(self, data: FieldData) -> None:
        # GEO_NOTHING ghost cells hold v=0, so their error equals the full
        # analytical profile; the tolerance was calibrated on interior cells.
        mask = data["wall"] != GEO_NOTHING
        max_err = float(np.max(data["error_vx"][mask]))
        assert max_err < 5e-3, f"max|error_vx|(phys)={max_err:.2e} (tol=5e-3)"

    def test_analytical_error_lbm(self, data: FieldData) -> None:
        mask = data["wall"] != GEO_NOTHING
        max_err = float(np.max(data["lbm_error_vx"][mask]))
        assert max_err < 3e-3, f"max|lbm_error_vx|(lattice)={max_err:.2e} (tol=3e-3)"

    def test_wall_no_slip(self, data: FieldData) -> None:
        max_v = wall_velocity_max(data["velocity_x"], data["velocity_y"], data["wall"])
        assert max_v < 1e-10, f"max|v| in wall/nothing cells={max_v:.2e}"


class TestSim2dHills:
    """sim2d_hills: channel flow over three hill-like bumps."""

    @pytest.fixture(scope="class")
    def data(self, d2q9_results: dict[str, pathlib.Path]) -> FieldData:
        return read_2d_fields(
            d2q9_results["sim2d_hills"],
            ["lbm_density", "velocity_x", "velocity_y", "wall"],
        )

    def test_finiteness(self, data: FieldData) -> None:
        assert_all_finite(data)

    def test_mass_conservation(self, data: FieldData) -> None:
        # Open outflow -> slightly looser than periodic.
        assert_mass_conserved(data["lbm_density"], tolerance=5e-3)

    def test_sym_top_smooth(self, data: FieldData) -> None:
        # Without collision on SYM_TOP cells, the tangential distributions
        # freeze and vx spikes or drops. With the fix, vx at the SYM_TOP row
        # should match the fluid row below it.
        vx, wall = data["velocity_x"], data["wall"]
        sym_rows = np.where((wall == GEO_SYM_TOP).any(axis=1))[0]
        assert sym_rows.size > 0, "no SYM_TOP cells found"
        y_sym, nx = int(sym_rows[0]), vx.shape[1]
        y_fluid = y_sym - 1
        # Compare vx magnitudes in the interior (skip inflow/outflow columns).
        x_start, x_end = (15 * nx) // 100, (85 * nx) // 100
        ratio = float(
            np.max(np.abs(vx[y_sym, x_start:x_end]))
            / max(np.max(np.abs(vx[y_fluid, x_start:x_end])), 1e-30)
        )
        assert ratio < 1.05, (
            f"|vx_max(SYM_TOP)|/|vx_max(fluid)|={ratio:.6f} (tol < 1.05)"
        )

    def test_sym_top_continuity(self, data: FieldData) -> None:
        vx, wall = data["velocity_x"], data["wall"]
        y_sym = int(np.where((wall == GEO_SYM_TOP).any(axis=1))[0][0])
        # Interior columns only: the bump wake reaches the slip row near the
        # inflow/outflow ends and would dominate the maximum there.
        nx = vx.shape[1]
        x_start, x_end = (15 * nx) // 100, (85 * nx) // 100
        row_diff = float(
            np.max(np.abs(vx[y_sym, x_start:x_end] - vx[y_sym - 1, x_start:x_end]))
        )
        peak = float(np.max(np.abs(vx)))
        assert peak > 0
        rel = row_diff / peak * 100
        assert row_diff / peak < 0.05, (
            f"max|vx(SYM_TOP)-vx(fluid)|={row_diff:.2e} ({rel:.2f}% of peak)"
        )

    def test_inflow_uniform(self, data: FieldData) -> None:
        vx, wall = data["velocity_x"], data["wall"]
        inflow_mask = wall[:, 1] == GEO_INFLOW_LEFT
        assert inflow_mask.any(), "no inflow cells found at x=1"
        inflow_vx = vx[inflow_mask, 1]
        spread = float(np.max(inflow_vx) - np.min(inflow_vx))
        assert spread < 1e-6, (
            f"inflow vx spread={spread:.2e} (vx={float(np.mean(inflow_vx)):.6f})"
        )

    def test_wall_no_slip(self, data: FieldData) -> None:
        max_v = wall_velocity_max(data["velocity_x"], data["velocity_y"], data["wall"])
        assert max_v < 1e-10, f"max|v| in wall/nothing cells={max_v:.2e}"


class TestSim2dTaylorGreen:
    """sim2d_Taylor_Green: decaying vortex on a periodic domain."""

    @pytest.fixture(scope="class")
    def data(self, d2q9_results: dict[str, pathlib.Path]) -> FieldData:
        return read_2d_fields(
            d2q9_results["sim2d_Taylor_Green"],
            [
                "lbm_density",
                "velocity_x",
                "velocity_y",
                "error_vx",
                "error_vy",
                "lbm_error_vx",
                "lbm_error_vy",
                "wall",
            ],
        )

    def test_finiteness(self, data: FieldData) -> None:
        assert_all_finite(data)

    def test_mass_conservation(self, data: FieldData) -> None:
        assert_mass_conserved(data["lbm_density"], tolerance=1e-3)

    def test_analytical_error_phys(self, data: FieldData) -> None:
        max_err = max(float(np.max(data["error_vx"])), float(np.max(data["error_vy"])))
        assert max_err < 1e-5, f"max|error_v|={max_err:.2e} (tol=1e-5)"

    def test_analytical_error_lbm(self, data: FieldData) -> None:
        max_err = max(
            float(np.max(data["lbm_error_vx"])), float(np.max(data["lbm_error_vy"]))
        )
        assert max_err < 1e-4, f"max|lbm_error_v|={max_err:.2e} (tol=1e-4)"

    def test_velocity_symmetry(self, data: FieldData) -> None:
        # The analytical Taylor-Green solution has u symmetric and v
        # antisymmetric in y. On a periodic domain with an isotropic operator
        # this holds up to truncation error; an anisotropic operator
        # (Straka2016) shows violations of O(1).
        vx, vy = data["velocity_x"], data["velocity_y"]
        max_sym = max(
            float(np.max(np.abs(vx - vx[:, ::-1]))),
            float(np.max(np.abs(vy + vy[:, ::-1]))),
        )
        signal = max(float(np.max(np.abs(vx))), float(np.max(np.abs(vy))))
        assert max_sym < 0.1, (
            f"max|sym_violation|={max_sym:.2e} (signal peak={signal:.2e})"
        )
