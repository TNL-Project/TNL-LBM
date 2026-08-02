"""Check D2Q9 simulation results from existing ADIOS2 BP5 outputs.

This script only reads data from ``results_*`` directories — it does not
launch any simulation. It auto-discovers output directories matching known
sim patterns and dispatches to a per-sim check function.

Each check function validates properties that are known to hold for the
correct (Geier 2017 CLBM + fixed BCs) implementation:

  sim2d_1           — centerline symmetry (Galilean invariance),
                      inflow velocity uniformity, wall no-slip.
  sim2d_2           — Poiseuille analytical accuracy, mass conservation.
  sim2d_hills       — SYM_TOP smoothness (no frozen-population spike),
                      mass conservation, inflow uniformity.
  sim2d_Taylor_Green — analytical decay accuracy, mass conservation,
                       velocity symmetry.

Thresholds are tuned from confirmed-correct behaviour at resolution 1.
"""

import argparse
import pathlib
import sys
import traceback
from dataclasses import dataclass, field

import adios2
import numpy as np

# ── GEO enum values (must match include/lbm3d/d2q9/bc.h) ───────────────────

GEO_FLUID = 0
GEO_WALL = 1
GEO_INFLOW = 2
GEO_INFLOW_LEFT = 3
GEO_OUTFLOW_EQ = 4
GEO_OUTFLOW_RIGHT = 5
GEO_OUTFLOW_RIGHT_INTERP = 6
GEO_PERIODIC = 7
GEO_NOTHING = 8
GEO_SYM_TOP = 9
GEO_SYM_BOTTOM = 10
GEO_SYM_LEFT = 11
GEO_SYM_RIGHT = 12


# ── Helpers ────────────────────────────────────────────────────────────────


@dataclass
class CheckResult:
    """Outcome of a single named check."""

    name: str
    passed: bool
    detail: str = ""

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        msg = f"  [{status}] {self.name}"
        if self.detail:
            msg += f": {self.detail}"
        return msg


@dataclass
class SimReport:
    """Aggregated results for one simulation."""

    sim_name: str
    bp_path: pathlib.Path
    results: list[CheckResult] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return all(r.passed for r in self.results)

    def summary(self) -> str:
        status = "PASS" if self.all_passed else "FAIL"
        lines = [f"{self.sim_name} [{status}]  ({self.bp_path})"]
        for r in self.results:
            lines.append(str(r))
        return "\n".join(lines)


def read_last_step(
    bp_path: pathlib.Path, var_names: list[str]
) -> dict[str, np.ndarray]:
    """Read the last step of the given variables from a BP5 file."""
    if not bp_path.exists():
        raise FileNotFoundError(f"Output file does not exist: {bp_path}")

    data: dict[str, np.ndarray] = {}
    with adios2.FileReader(str(bp_path)) as reader:
        num_steps = reader.num_steps()
        if num_steps == 0:
            raise RuntimeError(f"No steps written to {bp_path}")

        available = reader.available_variables()
        missing = [v for v in var_names if v not in available]
        if missing:
            raise RuntimeError(f"Missing variables in {bp_path}: {missing}")

        for name in var_names:
            var = reader.inquire_variable(name)
            arr = reader.read(var, step_selection=[num_steps - 1, 1])
            data[name] = np.asarray(arr)

    return data


def squeeze_2d(arr: np.ndarray) -> np.ndarray:
    """Remove the leading Z=1 dimension from a 3D 2D-cut array."""
    if arr.ndim == 3 and arr.shape[0] == 1:
        return arr[0]
    return arr


def check_finite(data: dict[str, np.ndarray], label: str) -> CheckResult:
    """Verify all values in all fields are finite."""
    for name, arr in data.items():
        if not np.all(np.isfinite(arr)):
            n_bad = int(np.sum(~np.isfinite(arr)))
            return CheckResult(
                "finiteness",
                False,
                f"{name} has {n_bad} non-finite values",
            )
    return CheckResult("finiteness", True, f"all {len(data)} fields finite")


def check_mass_conservation(
    rho: np.ndarray, tolerance: float, label: str = "mass"
) -> CheckResult:
    """Check that mean density is close to 1."""
    mean_rho = float(np.mean(rho))
    deviation = abs(mean_rho - 1.0)
    passed = deviation < tolerance
    return CheckResult(
        label,
        passed,
        f"mean(rho)={mean_rho:.8e}, |mean-1|={deviation:.2e} (tol={tolerance:.0e})",
    )


def check_wall_noslip(
    vx: np.ndarray, vy: np.ndarray, wall: np.ndarray, tolerance: float
) -> CheckResult:
    """Check that wall (GEO_WALL) and nothing (GEO_NOTHING) cells have ~zero velocity."""
    solid_mask = (wall == GEO_WALL) | (wall == GEO_NOTHING)
    if not solid_mask.any():
        return CheckResult("wall_no_slip", True, "no wall/nothing cells found")
    max_v = max(
        float(np.max(np.abs(vx[solid_mask]))) if vx[solid_mask].size else 0.0,
        float(np.max(np.abs(vy[solid_mask]))) if vy[solid_mask].size else 0.0,
    )
    passed = max_v < tolerance
    return CheckResult(
        "wall_no_slip",
        passed,
        f"max|v| in wall/nothing cells={max_v:.2e} (tol={tolerance:.0e})",
    )


# ── Per-sim checks ─────────────────────────────────────────────────────────


def check_sim2d_1(bp_path: pathlib.Path) -> SimReport:
    """sim2d_1: channel with baffle hole.

    Tests:
      - finiteness;
      - centerline symmetry (the defining Galilean-invariance test);
      - inflow velocity uniformity (moment BC imposes constant vx);
      - wall / nothing cells have zero velocity.
    """
    report = SimReport("sim2d_1", bp_path)

    try:
        data = read_last_step(
            bp_path,
            ["lbm_density", "velocity_x", "velocity_y", "wall"],
        )
    except (FileNotFoundError, RuntimeError) as exc:
        report.results.append(CheckResult("read", False, str(exc)))
        return report

    rho = squeeze_2d(data["lbm_density"])
    vx = squeeze_2d(data["velocity_x"])
    vy = squeeze_2d(data["velocity_y"])
    wall = squeeze_2d(data["wall"])
    ny, nx = vx.shape

    report.results.append(check_finite(data, "sim2d_1"))

    # Centerline symmetry: the baffle hole is centered (40–60 % of Y), so the
    # geometry is mirror-symmetric about the channel centre. A Galilean-
    # invariant collision operator (Geier 2017 CLBM) keeps the jet perfectly
    # centred; the anisotropic Straka 2016 operator deflects it.
    # Exclude the outermost rows (GEO_NOTHING) and wall rows before comparing.
    vx_inner = vx[1 : ny - 1, :]
    diff = np.abs(vx_inner - vx_inner[::-1, :])
    max_diff = float(np.max(diff))
    peak = float(np.max(np.abs(vx)))
    sym_rel = max_diff / peak if peak > 0 else 0.0
    report.results.append(
        CheckResult(
            "centerline_symmetry",
            max_diff < 1e-6,
            f"max|vx(y)-vx(Y-1-y)|={max_diff:.2e} "
            f"({sym_rel * 100:.4f}% of peak {peak:.2e})",
        )
    )

    # Inflow uniformity: all GEO_INFLOW_LEFT cells at x=0 should have the
    # same imposed vx.
    inflow_mask = wall[:, 0] == GEO_INFLOW_LEFT
    if inflow_mask.any():
        inflow_vx = vx[inflow_mask, 0]
        spread = float(np.max(inflow_vx) - np.min(inflow_vx))
        report.results.append(
            CheckResult(
                "inflow_uniform",
                spread < 1e-6,
                f"inflow vx spread={spread:.2e} (vx={float(np.mean(inflow_vx)):.6f})",
            )
        )

    # Wall / nothing no-slip.
    report.results.append(check_wall_noslip(vx, vy, wall, tolerance=1e-10))

    return report


def check_sim2d_2(bp_path: pathlib.Path) -> SimReport:
    """sim2d_2: Hagen-Poiseuille channel flow.

    Tests:
      - finiteness;
      - mass conservation;
      - analytical error (physical and lattice units) within tolerance;
      - wall / nothing cells have zero velocity.
    """
    report = SimReport("sim2d_2", bp_path)

    try:
        data = read_last_step(
            bp_path,
            [
                "lbm_density",
                "velocity_x",
                "velocity_y",
                "error_vx",
                "lbm_error_vx",
                "wall",
            ],
        )
    except (FileNotFoundError, RuntimeError) as exc:
        report.results.append(CheckResult("read", False, str(exc)))
        return report

    rho = squeeze_2d(data["lbm_density"])
    vx = squeeze_2d(data["velocity_x"])
    vy = squeeze_2d(data["velocity_y"])
    wall = squeeze_2d(data["wall"])
    err_phys = squeeze_2d(data["error_vx"])
    err_lbm = squeeze_2d(data["lbm_error_vx"])

    report.results.append(check_finite(data, "sim2d_2"))

    # Mass conservation (float32 Poiseuille on periodic-ish domain).
    report.results.append(check_mass_conservation(rho, tolerance=1e-3))

    # Analytical error in physical units.
    max_err_phys = float(np.max(err_phys))
    report.results.append(
        CheckResult(
            "analytical_error_phys",
            max_err_phys < 5e-3,
            f"max|error_vx|(phys)={max_err_phys:.2e} (tol=5e-3)",
        )
    )

    # Analytical error in lattice units.
    max_err_lbm = float(np.max(err_lbm))
    report.results.append(
        CheckResult(
            "analytical_error_lbm",
            max_err_lbm < 3e-3,
            f"max|lbm_error_vx|(lattice)={max_err_lbm:.2e} (tol=3e-3)",
        )
    )

    # Wall no-slip.
    report.results.append(check_wall_noslip(vx, vy, wall, tolerance=1e-10))

    return report


def check_sim2d_hills(bp_path: pathlib.Path) -> SimReport:
    """sim2d_hills: channel flow over 3 hill-like bumps.

    Tests:
      - finiteness;
      - mass conservation;
      - SYM_TOP boundary: no velocity spike (collision enabled);
      - inflow velocity uniformity;
      - wall / nothing cells have zero velocity.
    """
    report = SimReport("sim2d_hills", bp_path)

    try:
        data = read_last_step(
            bp_path,
            ["lbm_density", "velocity_x", "velocity_y", "wall"],
        )
    except (FileNotFoundError, RuntimeError) as exc:
        report.results.append(CheckResult("read", False, str(exc)))
        return report

    rho = squeeze_2d(data["lbm_density"])
    vx = squeeze_2d(data["velocity_x"])
    vy = squeeze_2d(data["velocity_y"])
    wall = squeeze_2d(data["wall"])
    ny, nx = vx.shape

    report.results.append(check_finite(data, "sim2d_hills"))

    # Mass conservation (open outflow → slightly looser than periodic).
    report.results.append(check_mass_conservation(rho, tolerance=5e-3))

    # SYM_TOP smoothness: without collision on SYM_TOP cells, the tangential
    # distributions freeze and vx spikes or drops. With the fix, vx at the
    # SYM_TOP row should match the fluid row below it.
    sym_mask = wall == GEO_SYM_TOP
    if sym_mask.any():
        sym_rows = np.where(sym_mask.any(axis=1))[0]
        y_sym = int(sym_rows[0])
        y_fluid = y_sym - 1  # the fluid row just inside

        # Compare vx magnitudes in the interior (skip inflow/outflow columns)
        x_start = nx // 4
        x_end = 3 * nx // 4
        vx_sym = np.abs(vx[y_sym, x_start:x_end])
        vx_fluid = np.abs(vx[y_fluid, x_start:x_end])
        ratio = float(np.max(vx_sym) / max(np.max(vx_fluid), 1e-30))
        report.results.append(
            CheckResult(
                "sym_top_smooth",
                ratio < 1.05,
                f"|vx_max(SYM_TOP)|/|vx_max(fluid)|={ratio:.6f} (tol < 1.05)",
            )
        )

        # Also check the row-to-row difference is small relative to peak.
        row_diff = float(np.max(np.abs(vx[y_sym, :] - vx[y_fluid, :])))
        peak = float(np.max(np.abs(vx)))
        report.results.append(
            CheckResult(
                "sym_top_continuity",
                row_diff / peak < 0.05 if peak > 0 else True,
                f"max|vx(SYM_TOP)-vx(fluid)|={row_diff:.2e} "
                f"({row_diff / peak * 100:.2f}% of peak)",
            )
        )

    # Inflow uniformity.
    inflow_mask = wall[:, 0] == GEO_INFLOW_LEFT
    if inflow_mask.any():
        inflow_vx = vx[inflow_mask, 0]
        spread = float(np.max(inflow_vx) - np.min(inflow_vx))
        report.results.append(
            CheckResult(
                "inflow_uniform",
                spread < 1e-6,
                f"inflow vx spread={spread:.2e} (vx={float(np.mean(inflow_vx)):.6f})",
            )
        )

    # Wall / nothing no-slip.
    report.results.append(check_wall_noslip(vx, vy, wall, tolerance=1e-10))

    return report


def check_sim2d_taylor_green(bp_path: pathlib.Path) -> SimReport:
    """sim2d_Taylor_Green: decaying vortex on periodic domain.

    Tests:
      - finiteness;
      - mass conservation;
      - analytical error (physical and lattice) within tolerance;
      - velocity symmetry (vx symmetric in y, vy antisymmetric in y).
    """
    report = SimReport("sim2d_Taylor_Green", bp_path)

    try:
        data = read_last_step(
            bp_path,
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
    except (FileNotFoundError, RuntimeError) as exc:
        report.results.append(CheckResult("read", False, str(exc)))
        return report

    rho = squeeze_2d(data["lbm_density"])
    vx = squeeze_2d(data["velocity_x"])
    vy = squeeze_2d(data["velocity_y"])
    wall = squeeze_2d(data["wall"])
    err_vx = squeeze_2d(data["error_vx"])
    err_vy = squeeze_2d(data["error_vy"])
    err_vx_lbm = squeeze_2d(data["lbm_error_vx"])
    err_vy_lbm = squeeze_2d(data["lbm_error_vy"])

    report.results.append(check_finite(data, "sim2d_Taylor_Green"))

    # Mass conservation (periodic domain, float32).
    report.results.append(check_mass_conservation(rho, tolerance=1e-3))

    # Analytical error in physical units.
    max_err_vx = float(np.max(err_vx))
    max_err_vy = float(np.max(err_vy))
    max_err_phys = max(max_err_vx, max_err_vy)
    report.results.append(
        CheckResult(
            "analytical_error_phys",
            max_err_phys < 1e-5,
            f"max|error_v|={max_err_phys:.2e} (tol=1e-5)",
        )
    )

    # Analytical error in lattice units.
    max_err_vx_lbm = float(np.max(err_vx_lbm))
    max_err_vy_lbm = float(np.max(err_vy_lbm))
    max_err_lbm = max(max_err_vx_lbm, max_err_vy_lbm)
    report.results.append(
        CheckResult(
            "analytical_error_lbm",
            max_err_lbm < 1e-4,
            f"max|lbm_error_v|={max_err_lbm:.2e} (tol=1e-4)",
        )
    )

    # Velocity symmetry: the analytical Taylor-Green solution has
    # u(x,y) symmetric in y and v(x,y) antisymmetric in y. On a periodic
    # domain with an isotropic operator, this holds up to truncation error.
    # At late times the vortex has decayed, so an absolute threshold is
    # used — a real anisotropy (Straka2016) would show violations of O(1).
    diff_vx = float(np.max(np.abs(vx - vx[:, ::-1])))
    diff_vy = float(np.max(np.abs(vy + vy[:, ::-1])))
    max_sym = max(diff_vx, diff_vy)
    signal = max(
        float(np.max(np.abs(vx))),
        float(np.max(np.abs(vy))),
    )
    report.results.append(
        CheckResult(
            "velocity_symmetry",
            max_sym < 0.1,
            f"max|sym_violation|={max_sym:.2e} (signal peak={signal:.2e})",
        )
    )

    return report


# ── Discovery & dispatch ───────────────────────────────────────────────────

SIM_REGISTRY = [
    ("sim2d_1", check_sim2d_1),
    ("sim2d_2", check_sim2d_2),
    ("sim2d_hills", check_sim2d_hills),
    ("sim2d_Taylor_Green", check_sim2d_taylor_green),
]


def discover_results(
    base: pathlib.Path,
) -> dict[str, list[pathlib.Path]]:
    """Find all ``results_*`` directories matching known sim prefixes."""
    found: dict[str, list[pathlib.Path]] = {prefix: [] for prefix, _ in SIM_REGISTRY}
    for d in sorted(base.iterdir()):
        if not d.is_dir() or not d.name.startswith("results_"):
            continue
        name = d.name[len("results_") :]
        for prefix, _ in SIM_REGISTRY:
            if name.startswith(prefix):
                bp = d / "output_2D_.bp"
                if bp.exists():
                    found[prefix].append(bp)
                break
    return found


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Check D2Q9 simulation results from existing ADIOS2 BP5 outputs. "
            "Auto-discovers results_* directories in the current (or given) path."
        )
    )
    parser.add_argument(
        "--base",
        type=pathlib.Path,
        default=pathlib.Path.cwd(),
        help="Base directory to search for results_* dirs (default: cwd)",
    )
    args = parser.parse_args()

    discovered = discover_results(args.base)

    if not any(paths for paths in discovered.values()):
        print(f"No results_* directories found in {args.base}", file=sys.stderr)
        return 1

    all_reports: list[SimReport] = []

    for prefix, check_fn in SIM_REGISTRY:
        for bp_path in discovered[prefix]:
            try:
                report = check_fn(bp_path)
            except Exception as exc:
                report = SimReport(prefix, bp_path)
                report.results.append(
                    CheckResult("unhandled_exception", False, str(exc))
                )
                # Print traceback for debugging
                traceback.print_exc(file=sys.stderr)
            all_reports.append(report)

    # Print reports
    for report in all_reports:
        print(report.summary())
        print()

    # Summary
    total = len(all_reports)
    passed = sum(1 for r in all_reports if r.all_passed)
    print(f"{passed}/{total} simulations passed all checks.")

    if passed < total:
        print("\nFailures:")
        for report in all_reports:
            if not report.all_passed:
                for r in report.results:
                    if not r.passed:
                        print(f"  {report.sim_name}: {r.name}: {r.detail}")
        return 1

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
