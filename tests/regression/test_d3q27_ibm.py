"""D3Q27 IBM regression tests.

Matrix regression: ``sim_IBM2`` is run for every (method, dirac) combination
with matrix output enabled, once on CPU and once on GPU. The generated
``ibm_{CPU|GPU}_matrix-{A|M}_method-{modified|original}_dirac-{1-4}.mtx``
files are compared against baselines in ``baseline_ibm_matrices/``:

- identical dimensions (rows, columns, nnz);
- identical sparsity pattern (row, col indices match);
- values within tolerance (max absolute difference < margin).

Flow-field regression: ``sim_IBM2`` default single-sphere channel at Re=100,
run for 5 physical time units, validated on the mid-plane cuts — finite
velocity/density/IBM-force fields, mass conservation, full-face inflow/outflow
coverage over the surrounding symmetry planes, a recirculating wake
behind the sphere, and the drag coefficient converged to the expected value.
"""

from __future__ import annotations

import pathlib
import re
from typing import TypedDict

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

METHODS = ["modified", "original"]
DIRACS = [1, 2, 3, 4]
MATRICES = ["A", "M"]
COMPUTE_MODES = ["CPU", "GPU"]

BASELINE_DIR = pathlib.Path(__file__).parent / "baseline_ibm_matrices"
VALUE_TOLERANCE = 1e-5


def mtx_name(compute: str, matrix: str, method: str, dirac: int) -> str:
    return f"ibm_{compute}_matrix-{matrix}_method-{method}_dirac-{dirac}.mtx"


@pytest.fixture(scope="module", params=COMPUTE_MODES)
def ibm_matrices(
    request: pytest.FixtureRequest,
    workspace: pathlib.Path,
) -> tuple[str, dict[tuple[str, str, int], pathlib.Path]]:
    """Generate all IBM matrices for one compute mode; returns (compute, paths)."""
    compute: str = request.param
    for method in METHODS:
        for dirac in DIRACS:
            run_sim(
                [
                    BUILD_DIR / "sim_NSE" / "sim_IBM2",
                    "--compute",
                    compute,
                    "--method",
                    method,
                    "--dirac",
                    str(dirac),
                    "--discretization-ratio",
                    "0.5",
                    "--resolution",
                    "1",
                    "--spheres",
                    "1",
                    "--final-time",
                    "0.0",
                    "--mtx-output",
                    "--adios-config",
                    str(PROJECT_ROOT / "adios2.xml"),
                ],
                workdir=workspace,
            )
    paths: dict[tuple[str, str, int], pathlib.Path] = {}
    for method in METHODS:
        for matrix in MATRICES:
            for dirac in DIRACS:
                path = workspace / mtx_name(compute, matrix, method, dirac)
                assert path.exists(), f"sim_IBM2 did not produce {path.name}"
                paths[(matrix, method, dirac)] = path
    return compute, paths


FLOW_VAR_NAMES = [
    "lbm_density",
    "velocity_x",
    "velocity_y",
    "velocity_z",
    "force_x",
    "force_y",
    "force_z",
    "wall",
]

# D3Q27 GEO enum (must match include/lbm3d/d3q27/bc.h)
GEO_INFLOW_MOMENT = 3
GEO_OUTFLOW_RIGHT_INTERP = 8
GEO_NOTHING = 9

# C_D≈1.19 (10-probe average) at t=5s in the Re=100 single-sphere channel flow.
# The flow does not reach steady state due to pressure wave oscillations; the
# last probe fluctuates, so the test averages the last 10 C_D values.
# Matrix runs use nas=0.5 so the flow run (default 0.25) gets its own state id.
FLOW_FINAL_TIME = "5.0"
DRAG_LOW, DRAG_HIGH = 1.0, 1.25


class IbmFlowResult(TypedDict):
    directory: pathlib.Path
    stdout: str
    cut_y: dict[str, np.ndarray]
    cut_z: dict[str, np.ndarray]


@pytest.fixture(scope="module")
def ibm_flow(workspace: pathlib.Path) -> IbmFlowResult:
    """Run the single-sphere IBM flow for FLOW_FINAL_TIME; return results."""
    stdout = run_sim(
        [
            BUILD_DIR / "sim_NSE" / "sim_IBM2",
            "--resolution",
            "1",
            "--spheres",
            "1",
            "--final-time",
            FLOW_FINAL_TIME,
            "--adios-config",
            str(PROJECT_ROOT / "adios2.xml"),
        ],
        workdir=workspace,
    )
    candidates = sorted(workspace.glob("results_sim_IBM2_*nas_0.2500*spheres1"))
    assert candidates, "sim_IBM2 flow run produced no results directory"
    data_y = read_last_step(candidates[0] / "output_2D_cut_Y.bp", FLOW_VAR_NAMES)
    data_z = read_last_step(candidates[0] / "output_2D_cut_Z.bp", FLOW_VAR_NAMES)
    return {
        "directory": candidates[0],
        "stdout": stdout,
        "cut_y": data_y,
        "cut_z": data_z,
    }


class TestIbmFlow:
    """Single-sphere IBM channel flow at Re=100 after 5 physical time units."""

    def test_finiteness(self, ibm_flow: IbmFlowResult) -> None:
        for plane in ("cut_y", "cut_z"):
            assert_all_finite(ibm_flow[plane])

    def test_mass_conservation(self, ibm_flow: IbmFlowResult) -> None:
        for plane in ("cut_y", "cut_z"):
            assert_mass_conserved(ibm_flow[plane]["lbm_density"], tolerance=6e-3)

    def test_boundary_map_coverage(self, ibm_flow: IbmFlowResult) -> None:
        # The four symmetry channel planes surround the inflow/outflow faces; on
        # the y/z mid-plane cuts the face columns must be tagged INFLOW/OUTFLOW
        # on every interior cell — the symmetry planes must not overwrite them.
        for plane, get_col in {
            "cut_y": (lambda w: w[:, 0, 1]),
            "cut_z": (lambda w: w[0, :, 1]),
        }.items():
            inflow_col = get_col(ibm_flow[plane]["wall"])
            assert inflow_col[0] == GEO_NOTHING
            assert inflow_col[-1] == GEO_NOTHING
            assert np.all(inflow_col[1:-1] == GEO_INFLOW_MOMENT), (
                f"{plane}: inflow edges overwritten by symmetry "
                f"(tags: {np.unique(inflow_col[1:-1])})"
            )
        for plane, get_col in {
            "cut_y": (lambda w: w[:, 0, -2]),
            "cut_z": (lambda w: w[0, :, -2]),
        }.items():
            outflow_col = get_col(ibm_flow[plane]["wall"])
            assert outflow_col[0] == GEO_NOTHING
            assert outflow_col[-1] == GEO_NOTHING
            assert np.all(outflow_col[1:-1] == GEO_OUTFLOW_RIGHT_INTERP), (
                f"{plane}: outflow edges overwritten by symmetry "
                f"(tags: {np.unique(outflow_col[1:-1])})"
            )

    def test_wake_deficit(self, ibm_flow: IbmFlowResult) -> None:
        # Axial velocity on the cut_Y plane (z, 1, x) along the row through
        # the sphere centre: positive inflow, recirculation behind the body,
        # partial recovery downstream.
        vx = ibm_flow["cut_y"]["velocity_x"]
        z_center = vx.shape[0] // 2
        row = vx[z_center, 0, :]
        upstream = float(row[2])
        min_behind = float(np.min(row[5:10]))
        wake = float(np.mean(row[12:20]))
        ratio = wake / upstream
        assert upstream > 0.05, f"upstream vx={upstream:.3e} too small"
        assert min_behind < -1e-3, (
            f"no recirculation behind the sphere (min vx={min_behind:.3e})"
        )
        assert -0.7 < ratio < 0.5, (
            f"wake/up={ratio:.3f} (up={upstream:.3e}, wake={wake:.3e})"
        )

    def test_sphere_drag(self, ibm_flow: IbmFlowResult) -> None:
        drags = re.findall(r"C_D=\s*([-\d.e+]+)", ibm_flow["stdout"])
        assert drags, "no C_D values found in sim_IBM2 output"
        drag = float(np.mean([float(d) for d in drags[-10:]]))
        assert DRAG_LOW < drag < DRAG_HIGH, (
            f"C_D={drag:.4f} outside [{DRAG_LOW}, {DRAG_HIGH}]"
        )


def parse_mtx(
    path: pathlib.Path,
) -> tuple[tuple[int, int, int], np.ndarray, np.ndarray, np.ndarray]:
    """Parse a Matrix Market coordinate file.

    Returns (shape, rows, cols, vals) where shape = (nrows, ncols, nnz).
    """
    with path.open("r") as f:
        header = f.readline()
        if not header.startswith("%%MatrixMarket"):
            msg = f"Not a Matrix Market file: {path}"
            raise ValueError(msg)

        # Skip comment lines (start with %)
        line = f.readline()
        while line.startswith("%"):
            line = f.readline()

        # Size line: rows cols nnz
        parts = line.split()
        nrows, ncols, nnz = int(parts[0]), int(parts[1]), int(parts[2])

        rows = np.empty(nnz, dtype=np.int64)
        cols = np.empty(nnz, dtype=np.int64)
        vals = np.empty(nnz, dtype=np.float64)

        for i in range(nnz):
            parts = f.readline().split()
            rows[i] = int(parts[0])
            cols[i] = int(parts[1])
            vals[i] = float(parts[2])

    return (nrows, ncols, nnz), rows, cols, vals


SparseMatrix = tuple[tuple[int, int, int], np.ndarray, np.ndarray, np.ndarray]


def sorted_coo(
    matrix: SparseMatrix,
) -> tuple[tuple[int, int, int], np.ndarray, np.ndarray, np.ndarray]:
    """Return the matrix reordered by (row, col) so patterns can be compared."""
    shape, rows, cols, vals = matrix
    order = np.lexsort((cols, rows))
    return shape, rows[order], cols[order], vals[order]


def baseline_path(matrix: str, method: str, dirac: int) -> pathlib.Path:
    return BASELINE_DIR / f"matrix-{matrix}_method-{method}_dirac-{dirac}.mtx"


@pytest.mark.parametrize("dirac", DIRACS)
@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("matrix", MATRICES)
class TestIbmMatrices:
    """Compare each generated IBM matrix against its baseline."""

    @pytest.fixture()
    def pair(
        self,
        ibm_matrices: tuple[str, dict[tuple[str, str, int], pathlib.Path]],
        matrix: str,
        method: str,
        dirac: int,
    ) -> tuple[str, SparseMatrix, SparseMatrix]:
        compute, paths = ibm_matrices
        generated = parse_mtx(paths[(matrix, method, dirac)])
        baseline = parse_mtx(baseline_path(matrix, method, dirac))
        return compute, sorted_coo(generated), sorted_coo(baseline)

    def test_dims(self, pair: tuple[str, SparseMatrix, SparseMatrix]) -> None:
        _, generated, baseline = pair
        assert generated[0] == baseline[0], (
            f"dims: gen={generated[0]} != base={baseline[0]}"
        )

    def test_sparsity_pattern(
        self, pair: tuple[str, SparseMatrix, SparseMatrix]
    ) -> None:
        _, (_, gen_rows, gen_cols, _), (_, base_rows, base_cols, _) = pair
        assert gen_rows.shape == base_rows.shape, "nnz differs"
        assert np.array_equal(gen_rows, base_rows), (
            f"row indices differ in {np.sum(gen_rows != base_rows)} entries"
        )
        assert np.array_equal(gen_cols, base_cols), (
            f"col indices differ in {np.sum(gen_cols != base_cols)} entries"
        )

    def test_values(self, pair: tuple[str, SparseMatrix, SparseMatrix]) -> None:
        _, (_, gen_rows, _, gen_vals), (_, base_rows, _, base_vals) = pair
        if not (
            gen_rows.shape == base_rows.shape and np.array_equal(gen_rows, base_rows)
        ):
            pytest.fail("sparsity pattern differs, value comparison not meaningful")
        max_diff = float(np.max(np.abs(gen_vals - base_vals))) if gen_vals.size else 0.0
        assert max_diff < VALUE_TOLERANCE, (
            f"max|diff|={max_diff:.2e} (tol={VALUE_TOLERANCE:.0e})"
        )
