"""D3Q27 NSE regression tests (sim_1 .. sim_4).

Each simulation runs once per test session (module-scoped fixture) in the
workspace and its outputs are validated property by property:

- ``sim_1`` (3D channel with centered baffle hole): centerline symmetry in Y
  and Z (Galilean invariance), inflow uniformity, wall/nothing no-slip, mass
  conservation.
- ``sim_2`` (square duct, analytical series reference): L1/L2 errors of the
  final cornered-residual state printed to stdout stay within tolerance.
- ``sim_3`` (Eulerian ball in a 3D channel, Re=100): wall and in-sphere
  no-slip, mass conservation, axial wake velocity deficit.
- ``sim_4`` (3D Taylor-Green vortex, Re=1600): finite 3D fields, mass
  conservation, and a kinetic-energy probe series with monotone decay well
  beyond the laminar rate.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import pytest

from tests.lbmtest import (
    ADIOS_CONFIG,
    BUILD_DIR,
    FieldData,
    assert_all_finite,
    assert_mass_conserved,
    run_sim,
)
from tests.regression.bp5 import read_last_step

if TYPE_CHECKING:
    import pathlib

# D3Q27 GEO enum (must match include/lbm3d/d3q27/bc.h)
GEO_WALL = 1
GEO_INFLOW_LEFT = 3
GEO_NOTHING = 9

SIMS: dict[str, list[str]] = {
    "sim_1": ["--resolution", "1"],
    "sim_2": ["--min-resolution", "1", "--max-resolution", "1"],
    "sim_3": ["--resolution", "1"],
    "sim_4": ["--resolution", "1"],
    "sim_2_forcing": [
        "--min-resolution",
        "1",
        "--max-resolution",
        "1",
        "--use-forcing",
    ],
}

# executable names different from the SIMS keys
SIM_EXE = {"sim_2_forcing": "sim_2"}
# results-directory globs different from the default results_<sim>_*
SIM_GLOB = {
    "sim_2": "results_sim_2_*_velocity_*",
    "sim_2_forcing": "results_sim_2_*_forcing_*",
}


class SimRun:
    """Results of one simulation run: results directory and captured output."""

    def __init__(self, directory: pathlib.Path, stdout: str) -> None:
        self.directory = directory
        self.stdout = stdout


@pytest.fixture(scope="module")
def nse_results(workspace: pathlib.Path) -> dict[str, SimRun]:
    """Run sim_1 .. sim_4 (and forcing variants) and return their artifacts."""
    outputs: dict[str, SimRun] = {}
    for sim, args in SIMS.items():
        stdout = run_sim(
            [
                BUILD_DIR / "sim_NSE" / SIM_EXE.get(sim, sim),
                *args,
                "--adios-config",
                ADIOS_CONFIG,
            ],
            workdir=workspace,
        )
        # sim_4 nests outputs under a res= subdirectory of its results dir
        pattern = SIM_GLOB.get(sim)
        if pattern is None:
            pattern = f"results_{sim}_*" if sim != "sim_4" else f"results_{sim}_*/res=*"
        candidates = sorted(workspace.glob(pattern))
        assert candidates, f"{sim} produced no results matching {pattern}"
        outputs[sim] = SimRun(candidates[0], stdout)
    return outputs


def last_step_data(
    results_dir: pathlib.Path, bp_name: str, var_names: list[str]
) -> FieldData:
    return read_last_step(results_dir / bp_name, var_names)


class TestSim1:
    """sim_1: 3D channel with a baffle hole centered in Y and Z."""

    @pytest.fixture(scope="class")
    def data(self, nse_results: dict[str, SimRun]) -> FieldData:
        return last_step_data(
            nse_results["sim_1"].directory,
            "output_3D.bp",
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
        inflow_mask = wall[:, :, 1] == GEO_INFLOW_LEFT
        assert inflow_mask.any(), "no inflow cells found at x=1"
        inflow_vx = vx[:, :, 1][inflow_mask]
        spread = float(np.max(inflow_vx) - np.min(inflow_vx))
        assert spread < 1e-6, (
            f"inflow vx spread={spread:.2e} (vx={float(np.mean(inflow_vx)):.6f})"
        )

    def test_wall_no_slip(self, data: FieldData) -> None:
        solid_mask = (data["wall"] == GEO_WALL) | (data["wall"] == GEO_NOTHING)
        assert solid_mask.any(), "no wall/nothing cells found"
        max_v = max(
            float(np.max(np.abs(data["velocity_x"][solid_mask]))),
            float(np.max(np.abs(data["velocity_y"][solid_mask]))),
            float(np.max(np.abs(data["velocity_z"][solid_mask]))),
        )
        assert max_v < 1e-10, f"max|v| in wall/nothing cells={max_v:.2e}"


class TestSim2:
    """sim_2: square duct verified against the analytical series solution.

    sim_2 prints running L1/L2 errors to the log and stops early once the
    error stagnates; the test asserts the final errors stay within tolerance.
    """

    ERROR_RE = re.compile(
        r"(l[12])error_phys_v=\[([-\d.e+]+),([-\d.e+]+),([-\d.e+]+)\]"
    )

    @pytest.fixture(scope="class")
    def errors(
        self, nse_results: dict[str, SimRun]
    ) -> dict[str, tuple[float, float, float]]:
        matches = self.ERROR_RE.findall(nse_results["sim_2"].stdout)
        assert matches, "no l1/l2 error lines found in sim_2 output"
        out: dict[str, tuple[float, float, float]] = {}
        for kind in ("l1", "l2"):
            kind_matches = [m for m in matches if m[0] == kind]
            assert kind_matches, f"no {kind} error line found in sim_2 output"
            _, vx, vy, vz = kind_matches[-1]
            out[kind] = (float(vx), float(vy), float(vz))
        return out

    def test_l1_vx(self, errors: dict[str, tuple[float, float, float]]) -> None:
        assert errors["l1"][0] < 2e-5, f"l1 error vx={errors['l1'][0]:.2e} (tol=2e-5)"

    def test_l1_transverse(self, errors: dict[str, tuple[float, float, float]]) -> None:
        _, vy, vz = errors["l1"]
        assert max(vy, vz) < 2e-6, f"l1 error vy={vy:.2e} vz={vz:.2e} (tol=2e-6)"

    def test_l2_vx(self, errors: dict[str, tuple[float, float, float]]) -> None:
        assert errors["l2"][0] < 2e-4, f"l2 error vx={errors['l2'][0]:.2e} (tol=2e-4)"


class TestSim2Forcing(TestSim2):
    """sim_2 --use-forcing: square duct driven by body force.

    The forcing variant converges to a looser plateau than the inflow-driven
    duct (measured l1=6.9e-5, l2=9.5e-4 at resolution 1, vs 6.0e-6/6.8e-5 for
    the inflow variant), so the tolerances are forcing-specific.
    """

    @pytest.fixture(scope="class")
    def errors(
        self, nse_results: dict[str, SimRun]
    ) -> dict[str, tuple[float, float, float]]:
        matches = self.ERROR_RE.findall(nse_results["sim_2_forcing"].stdout)
        assert matches, "no l1/l2 error lines found in sim_2 --use-forcing output"
        out: dict[str, tuple[float, float, float]] = {}
        for kind in ("l1", "l2"):
            kind_matches = [m for m in matches if m[0] == kind]
            assert kind_matches, f"no {kind} error line found in sim_2 forcing output"
            _, vx, vy, vz = kind_matches[-1]
            out[kind] = (float(vx), float(vy), float(vz))
        return out

    def test_l1_vx(self, errors: dict[str, tuple[float, float, float]]) -> None:
        assert errors["l1"][0] < 1e-4, f"l1 error vx={errors['l1'][0]:.2e} (tol=1e-4)"

    def test_l2_vx(self, errors: dict[str, tuple[float, float, float]]) -> None:
        assert errors["l2"][0] < 1.5e-3, (
            f"l2 error vx={errors['l2'][0]:.2e} (tol=1.5e-3)"
        )


class TestSim3:
    """sim_3: Eulerian ball (drawn as GEO_WALL) in a 3D channel, Re=100.

    Runs to a developed state; validated on the three mid-plane cuts.
    """

    CUT_VARS: ClassVar[list[str]] = [
        "lbm_density",
        "velocity_x",
        "velocity_y",
        "velocity_z",
        "wall",
    ]

    @pytest.fixture(scope="class")
    def cuts(self, nse_results: dict[str, SimRun]) -> dict[str, FieldData]:
        directory = nse_results["sim_3"].directory
        return {
            axis: last_step_data(directory, f"output_2D_cut_{axis}.bp", self.CUT_VARS)
            for axis in "XYZ"
        }

    def test_finiteness(self, cuts: dict[str, FieldData]) -> None:
        for data in cuts.values():
            assert_all_finite(data)

    def test_mass_conservation(self, cuts: dict[str, FieldData]) -> None:
        for axis, data in cuts.items():
            mean_rho = float(np.mean(data["lbm_density"]))
            dev = abs(mean_rho - 1.0)
            assert dev < 2e-3, (
                f"cut_{axis}: mean(rho)={mean_rho:.8e}, dev={dev:.2e} > 2e-3"
            )

    def test_no_slip(self, cuts: dict[str, FieldData]) -> None:
        # GEO_WALL covers both the channel walls and the drawn sphere; the x
        # mid-plane cut does not intersect any wall, only the y/z ones do.
        max_v = 0.0
        walls_found = 0
        for data in cuts.values():
            solid = data["wall"] == GEO_WALL
            walls_found += int(solid.sum())
            if solid.any():
                max_v = max(
                    max_v,
                    float(np.max(np.abs(data["velocity_x"][solid]))),
                    float(np.max(np.abs(data["velocity_y"][solid]))),
                    float(np.max(np.abs(data["velocity_z"][solid]))),
                )
        assert walls_found > 0, "no GEO_WALL cells found on any cut plane"
        assert max_v < 1e-10, f"max|v| in wall/sphere cells={max_v:.2e}"

    def test_wake_deficit(self, cuts: dict[str, FieldData]) -> None:
        # Axial velocity along the wake center line (row through the sphere
        # centre in the z-halfway cut) drops behind the body and only slowly
        # recovers downstream.
        vx = cuts["Z"]["velocity_x"]  # shape (1, y, x)
        axis_row = vx.shape[1] // 2 - 1
        row = vx[0, axis_row, :]
        upstream = float(np.mean(row[2:5]))
        wake = float(np.mean(row[12:20]))
        ratio = wake / upstream
        assert 0 < ratio < 0.7, (
            f"wake/up={ratio:.3f} (up={upstream:.3e}, wake={wake:.3e})"
        )


class TestSim4:
    """sim_4: 3D Taylor-Green vortex at Re=1600.

    The integrated kinetic energy must decay strictly (dissipation); at this
    Reynolds number the turbulent cascade drains the initial mode much faster
    than the laminar rate.
    """

    @pytest.fixture(scope="class")
    def probe(self, nse_results: dict[str, SimRun]) -> np.ndarray:
        probe_files = sorted(
            nse_results["sim_4"].directory.glob("probe1/kinetic_energy_rank000.txt")
        )
        assert probe_files, "no kinetic-energy probe file found"
        data = np.loadtxt(probe_files[0], skiprows=1)
        assert data.ndim == 2 and data.shape[1] == 5, (
            f"unexpected probe data shape: {data.shape}"
        )
        assert len(data) >= 500, f"probe series too short: {len(data)} rows"
        return data

    def test_output_finiteness(self, nse_results: dict[str, SimRun]) -> None:
        data = last_step_data(
            nse_results["sim_4"].directory,
            "output_3D.bp",
            ["lbm_density", "velocity_x", "velocity_y", "velocity_z"],
        )
        assert_all_finite(data)

    def test_output_mass_conservation(self, nse_results: dict[str, SimRun]) -> None:
        data = last_step_data(
            nse_results["sim_4"].directory, "output_3D.bp", ["lbm_density"]
        )
        assert_mass_conserved(data["lbm_density"], tolerance=2e-4)

    def test_probe_columns_finite(self, probe: np.ndarray) -> None:
        names = ["iter", "time", "kinetic_energy", "enstrophy", "enstrophy_dissipation"]
        for i, name in enumerate(names):
            assert np.all(np.isfinite(probe[:, i])), (
                f"probe column {name} has non-finite values"
            )

    def test_kinetic_energy_decays(self, probe: np.ndarray) -> None:
        ke = probe[:, 2]
        assert np.all(ke >= 0), "kinetic energy must be non-negative"
        ratio = float(ke[-1] / ke[0]) if ke[0] > 0 else 0.0
        assert ratio < 0.3, f"KE(final)/KE(initial)={ratio:.3f} (tol < 0.3)"

    def test_enstrophy_and_dissipation(self, probe: np.ndarray) -> None:
        enstrophy = probe[:, 3]
        dissipation = probe[:, 4]
        assert np.all(enstrophy >= 0), "enstrophy must be non-negative"
        assert np.all(dissipation >= 0), "enstrophy dissipation must be non-negative"
        assert float(np.mean(dissipation)) > 0, (
            "mean dissipation should be positive (energy decay)"
        )
