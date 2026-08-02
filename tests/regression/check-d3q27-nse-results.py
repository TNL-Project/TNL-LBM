"""Check D3Q27 NSE simulation results from existing ADIOS2 BP5 outputs.

Reads 3D (or 2D-cut) output from sim_NSE simulations. Does not launch
any simulation — only post-processes existing ``results_sim_*`` dirs.

Each check function validates properties known to hold for the correct
(CuLBM + fixed BCs) implementation:

  sim_1  — 3D channel with baffle hole: centerline symmetry in Y and Z
            (Galilean invariance), inflow uniformity, wall/nothing no-slip,
            mass conservation.
"""

import argparse
import pathlib
import sys
import traceback
from dataclasses import dataclass, field

import adios2
import numpy as np

# D3Q27 GEO enum (must match include/lbm3d/d3q27/bc.h)
GEO_FLUID = 0
GEO_WALL = 1
GEO_INFLOW = 2
GEO_INFLOW_LEFT = 3
GEO_INFLOW_BB_LEFT = 4
GEO_INFLOW_EQ_LEFT = 5
GEO_OUTFLOW_EQ = 6
GEO_OUTFLOW_RIGHT = 7
GEO_OUTFLOW_RIGHT_INTERP = 8
GEO_PERIODIC = 9
GEO_NOTHING = 10
GEO_SYM_TOP = 11
GEO_SYM_BOTTOM = 12
GEO_SYM_LEFT = 13
GEO_SYM_RIGHT = 14
GEO_SYM_BACK = 15
GEO_SYM_FRONT = 16


@dataclass
class CheckResult:
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


def check_finite(data: dict[str, np.ndarray]) -> CheckResult:
    for name, arr in data.items():
        if not np.all(np.isfinite(arr)):
            n_bad = int(np.sum(~np.isfinite(arr)))
            return CheckResult(
                "finiteness", False, f"{name} has {n_bad} non-finite values"
            )
    return CheckResult("finiteness", True, f"all {len(data)} fields finite")


def check_mass_conservation(rho: np.ndarray, tolerance: float) -> CheckResult:
    mean_rho = float(np.mean(rho))
    deviation = abs(mean_rho - 1.0)
    return CheckResult(
        "mass",
        deviation < tolerance,
        f"mean(rho)={mean_rho:.8e}, |mean-1|={deviation:.2e} (tol={tolerance:.0e})",
    )


def check_wall_noslip(
    vx: np.ndarray, vy: np.ndarray, vz: np.ndarray, wall: np.ndarray, tolerance: float
) -> CheckResult:
    solid_mask = (wall == GEO_WALL) | (wall == GEO_NOTHING)
    if not solid_mask.any():
        return CheckResult("wall_no_slip", True, "no wall/nothing cells")
    max_v = max(
        float(np.max(np.abs(vx[solid_mask]))) if vx[solid_mask].size else 0.0,
        float(np.max(np.abs(vy[solid_mask]))) if vy[solid_mask].size else 0.0,
        float(np.max(np.abs(vz[solid_mask]))) if vz[solid_mask].size else 0.0,
    )
    return CheckResult(
        "wall_no_slip",
        max_v < tolerance,
        f"max|v| in wall/nothing cells={max_v:.2e} (tol={tolerance:.0e})",
    )


def check_sim1(bp_path: pathlib.Path) -> SimReport:
    """sim_1: 3D channel with baffle hole.

    The hole is centered in Y and Z, so the geometry is mirror-symmetric
    about both the Y and Z centerlines. A Galilean-invariant collision
    operator keeps the jet perfectly centered.
    """
    report = SimReport("sim_1", bp_path)

    try:
        data = read_last_step(
            bp_path,
            ["lbm_density", "velocity_x", "velocity_y", "velocity_z", "wall"],
        )
    except (FileNotFoundError, RuntimeError) as exc:
        report.results.append(CheckResult("read", False, str(exc)))
        return report

    rho = data["lbm_density"]
    vx = data["velocity_x"]
    vy = data["velocity_y"]
    vz = data["velocity_z"]
    wall = data["wall"]

    report.results.append(check_finite(data))

    report.results.append(check_mass_conservation(rho, tolerance=5e-3))

    # Centerline symmetry in Y (axis 1) and Z (axis 0).
    # Exclude the outermost layers (GEO_NOTHING) before comparing.
    nz, ny, nx = vx.shape
    vx_inner = vx[1 : nz - 1, 1 : ny - 1, :]
    diff_y = np.abs(vx_inner - vx_inner[:, ::-1, :])
    diff_z = np.abs(vx_inner - vx_inner[::-1, :, :])
    max_diff_y = float(np.max(diff_y)) if diff_y.size else 0.0
    max_diff_z = float(np.max(diff_z)) if diff_z.size else 0.0
    peak = float(np.max(np.abs(vx)))
    report.results.append(
        CheckResult(
            "symmetry_y",
            max_diff_y < 1e-6,
            f"max|vx(y)-vx(Y-1-y)|={max_diff_y:.2e} ({max_diff_y / peak * 100:.4f}% of peak)",
        )
    )
    report.results.append(
        CheckResult(
            "symmetry_z",
            max_diff_z < 1e-6,
            f"max|vx(z)-vx(Z-1-z)|={max_diff_z:.2e} ({max_diff_z / peak * 100:.4f}% of peak)",
        )
    )

    # Inflow uniformity: all GEO_INFLOW_LEFT cells at x=0 have same vx.
    inflow_mask = wall[:, :, 0] == GEO_INFLOW_LEFT
    if inflow_mask.any():
        inflow_vx = vx[:, :, 0][inflow_mask]
        spread = float(np.max(inflow_vx) - np.min(inflow_vx))
        report.results.append(
            CheckResult(
                "inflow_uniform",
                spread < 1e-6,
                f"inflow vx spread={spread:.2e} (vx={float(np.mean(inflow_vx)):.6f})",
            )
        )

    report.results.append(check_wall_noslip(vx, vy, vz, wall, tolerance=1e-10))

    return report


# (prefix, check_function, bp_filename)
SIM_REGISTRY = [
    ("sim_1", check_sim1, "output_3D.bp"),
]


def discover_results(base: pathlib.Path) -> dict[str, list[pathlib.Path]]:
    found: dict[str, list[pathlib.Path]] = {prefix: [] for prefix, _, _ in SIM_REGISTRY}
    for d in sorted(base.iterdir()):
        if not d.is_dir() or not d.name.startswith("results_"):
            continue
        name = d.name[len("results_") :]
        for prefix, _, bp_name in SIM_REGISTRY:
            if name.startswith(prefix):
                bp = d / bp_name
                if bp.exists():
                    found[prefix].append(bp)
                break
    return found


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check D3Q27 NSE simulation results from existing ADIOS2 BP5 outputs."
    )
    parser.add_argument("--base", type=pathlib.Path, default=pathlib.Path.cwd())
    args = parser.parse_args()

    discovered = discover_results(args.base)

    if not any(paths for paths in discovered.values()):
        print(f"No results_* directories found in {args.base}", file=sys.stderr)
        return 1

    all_reports: list[SimReport] = []

    for prefix, check_fn, _ in SIM_REGISTRY:
        for bp_path in discovered[prefix]:
            try:
                report = check_fn(bp_path)
            except Exception as exc:
                report = SimReport(prefix, bp_path)
                report.results.append(
                    CheckResult("unhandled_exception", False, str(exc))
                )
                traceback.print_exc(file=sys.stderr)
            all_reports.append(report)

    for report in all_reports:
        print(report.summary())
        print()

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
