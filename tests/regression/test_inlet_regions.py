"""Inflow-inlet regions regression tests (authored inlet regions in real simulations).

End-to-end coverage of the authored inlet-region stack (addInletRegion in
setupBoundaries -> discovery/complement gating -> scale normalization -> the
kernel weight path) in both streaming patterns; the module is pattern-agnostic
and must pass against both the A-B (build) and A-A (build-aa) trees:

- ``sim2d_inlet_regions --mode parabolic``: one PARABOLIC inlet region spanning the
  fluid rows of the interior plane x=1, amplitude flux-matched to the discrete
  Poiseuille inflow profile of ``sim2d_2``. The developed Poiseuille error is
  compared against the error printed by plain ``sim2d_2`` at the same final
  time (both report the l2error_phys_v line of the shared error idiom).
- ``sim2d_inlet_regions --mode dual``: two UNIFORM inlet regions on the same plane,
  lower half imposing v and upper half 2v (v = the channel-mean velocity of
  the parabolic reference profile), proving per-inlet region amplitudes. The moment
  BC reproduces the authored amplitudes at the inlet region cells exactly.
- ``sim_inlet_regions``: one PARABOLIC inlet region on the interior plane x=1 of a
  square D3Q27 channel; the axial velocity integrated over the inlet region cells
  at final time recovers the authored flux.
- Legacy-quiet contract: plain ``sim2d_2`` prints no inlet region lines (discovery
  stays silent when nothing is authored, keeping legacy runs bit-quiet).

Calibrated at resolution 1 (D2Q9) and 4 (D3Q27); error bands measured on
sm_120 in both patterns are cited per check.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import pytest

from tests.lbmtest import ADIOS_CONFIG, BUILD_DIR, run_sim

if TYPE_CHECKING:
    import pathlib

# final times of the regression runs; the parabolic tolerance degrades with
# longer runs (the inlet regions' cell-centered paraboloid relaxes to the channel's
# node-centered steady profile near the inlet and their L2 plateaus drift
# apart: ratio 2.13x at t=60), so the comparison stays at the short t=20 where
# both sims sit in the same transient phase
FINAL_TIME_2D = "20"
FINAL_TIME_DUAL = "20"
FINAL_TIME_3D = "0.3"

L2_RE = re.compile(r"l2error_phys_v=\[([-\d.e+]+),([-\d.e+]+)\]")
MEAN_RE = re.compile(r"inlet_region_mean_vx\[(\d+)\]\s*=\s*([-\d.e+]+)")
TARGET_RE = re.compile(r"inlet_region_target_vx\[(\d+)\]\s*=\s*([-\d.e+]+)")
FLUX_BALANCE_RE = re.compile(r"flux_balance\s*=\s*([-\d.e+]+)")
FLUX_RE = re.compile(r"inlet_region_flux\s*=\s*([-\d.e+]+)")
FLUX_REL_ERROR_RE = re.compile(r"flux_rel_error\s*=\s*([-\d.e+]+)")


class SimRun:
    """Results of one simulation run: results directory and captured output."""

    def __init__(self, directory: pathlib.Path, stdout: str) -> None:
        self.directory = directory
        self.stdout = stdout


def last_l2_vx(stdout: str) -> float:
    """L2 vx error from the last printed error line."""
    matches = L2_RE.findall(stdout)
    assert matches, "no l2error_phys_v lines found in simulation output"
    return float(matches[-1][0])


@pytest.fixture(scope="module")
def inlet_regions_results(workspace: pathlib.Path) -> dict[str, SimRun]:
    """Run the inlet regions sims (and the sim2d_2 reference) once per session."""
    runs: dict[str, tuple[list[str], str]] = {
        # flux-matched parabolic inlet region vs plain sim2d_2 at the same time
        "parabolic": (
            [
                str(BUILD_DIR / "sim_2D" / "sim2d_inlet_regions"),
                "--mode",
                "parabolic",
                "--resolution",
                "1",
                "--final-time",
                FINAL_TIME_2D,
            ],
            "results_sim2d_inlet_regions_*parabolic*",
        ),
        "reference": (
            [
                str(BUILD_DIR / "sim_2D" / "sim2d_2"),
                "--resolution",
                "1",
                "--final-time",
                FINAL_TIME_2D,
            ],
            "results_sim2d_2_*",
        ),
        # two uniform inlet regions with amplitudes v and 2v
        "dual": (
            [
                str(BUILD_DIR / "sim_2D" / "sim2d_inlet_regions"),
                "--mode",
                "dual",
                "--resolution",
                "1",
                "--final-time",
                FINAL_TIME_DUAL,
            ],
            "results_sim2d_inlet_regions_*dual*",
        ),
        # D3Q27 square channel with a single parabolic inlet region
        "d3q27": (
            [
                str(BUILD_DIR / "sim_NSE" / "sim_inlet_regions"),
                "--resolution",
                "4",
                "--final-time",
                FINAL_TIME_3D,
            ],
            "results_sim_inlet_regions_*",
        ),
        # legacy regression gate: no inlet regions authored, no inlet region output
        "legacy": (
            [
                str(BUILD_DIR / "sim_2D" / "sim2d_2"),
                "--resolution",
                "1",
                "--final-time",
                "5",
            ],
            "results_sim2d_2_*",
        ),
    }
    outputs: dict[str, SimRun] = {}
    for name, (cmd, pattern) in runs.items():
        # unique workdir per run: the state ids of "reference" and "legacy"
        # coincide, and a finished results dir makes a rerun a no-op
        workdir = workspace / name
        workdir.mkdir()
        stdout = run_sim([*cmd, "--adios-config", ADIOS_CONFIG], workdir=workdir)
        candidates = sorted(workdir.glob(pattern))
        assert candidates, f"{name} produced no results matching {pattern}"
        outputs[name] = SimRun(candidates[0], stdout)
    return outputs


class TestParabolicPoiseuilleD2Q9:
    """sim2d_inlet_regions --mode parabolic recovers the sim2d_2 Poiseuille error."""

    def test_poiseuille_error(self, inlet_regions_results: dict[str, SimRun]) -> None:
        l2_inlet_regions = last_l2_vx(inlet_regions_results["parabolic"].stdout)
        l2_reference = last_l2_vx(inlet_regions_results["reference"].stdout)
        # measured at resolution 1, final-time 20 (identical in AA and AB):
        # inlet regions 3.98e-4, sim2d_2 reference 2.62e-4 (ratio 1.52); locked at
        # 2x the in-module reference plus an absolute cap ~4x the measured band
        assert l2_inlet_regions <= 2 * l2_reference, (
            f"l2error_phys_vx={l2_inlet_regions:.2e} exceeds 2x the sim2d_2 reference {l2_reference:.2e}"
        )
        assert l2_inlet_regions <= 1.5e-3, (
            f"l2error_phys_vx={l2_inlet_regions:.2e} (absolute cap 1.5e-3, measured band 3.98e-4 in both patterns)"
        )


class TestDualInletRegionAmplitudesD2Q9:
    """sim2d_inlet_regions --mode dual imposes the per-inlet region amplitudes v and 2v."""

    def test_inlet_region_amplitudes(self, inlet_regions_results: dict[str, SimRun]) -> None:
        stdout = inlet_regions_results["dual"].stdout
        means = {int(k): float(v) for k, v in MEAN_RE.findall(stdout)}
        targets = {int(k): float(v) for k, v in TARGET_RE.findall(stdout)}
        assert means.keys() == targets.keys() == {0, 1}, (
            f"expected inlet_region_mean_vx/target_vx lines for inlet regions 0 and 1, got {means} / {targets}"
        )
        # measured at resolution 1, final-time 20: mean == target identical at
        # 8-digit print in both patterns (rel error < 1e-9); design target 1e-3
        for k in (0, 1):
            rel = abs(means[k] - targets[k]) / abs(targets[k])
            assert rel <= 1e-3, (
                f"inlet_region_mean_vx[{k}]={means[k]:.6e} deviates from target "
                f"{targets[k]:.6e} by {rel:.2e} (tol=1e-3, measured <1e-9)"
            )

    def test_flux_balance(self, inlet_regions_results: dict[str, SimRun]) -> None:
        flux_balance = float(FLUX_BALANCE_RE.findall(inlet_regions_results["dual"].stdout)[-1])
        # outlet flux vs summed inlet amplitudes; measured 9.0e-3 at
        # final-time 20 (identical in AA and AB) -- the two-speed jet evolves
        # unsteady shear structures later on, so the check stays at the short
        # time with a wide smoke bound
        assert flux_balance < 1e-1, f"flux_balance={flux_balance:.2e} (tol < 1e-1)"


class TestParabolicFluxD3Q27:
    """sim_inlet_regions: integrated inlet region-plane flux recovers the authored flux."""

    def test_inlet_region_flux(self, inlet_regions_results: dict[str, SimRun]) -> None:
        stdout = inlet_regions_results["d3q27"].stdout
        flux = float(FLUX_RE.findall(stdout)[-1])
        rel_error = float(FLUX_REL_ERROR_RE.findall(stdout)[-1])
        assert flux > 0, f"inlet_region_flux={flux:.6e} (expected positive flux)"
        # measured at resolution 4, final-time 0.3: rel error 8.5e-9
        # (identical in AA and AB); design cap 5%
        assert rel_error <= 5e-2, f"flux_rel_error={rel_error:.2e} (tol=5e-2)"


class TestLegacyNoInletRegionsSilence:
    """Plain sim2d_2 stays silent about inlet regions (legacy-quiet contract)."""

    def test_no_inlet_region_lines(self, inlet_regions_results: dict[str, SimRun]) -> None:
        stdout = inlet_regions_results["legacy"].stdout
        assert "inlet region" not in stdout.lower(), (
            "legacy sim2d_2 printed inlet region-related lines it must stay silent about"
        )
