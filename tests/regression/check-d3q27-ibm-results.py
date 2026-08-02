"""Check D3Q27 IBM matrix regression results.

Compares generated ``ibm_{CPU|GPU}_matrix-{A|M}_method-{modified|original}_dirac-{1-4}.mtx``
files against baselines in ``tests/regression/baseline_ibm_matrices/``.

Does not launch any simulation — only reads existing ``.mtx`` files.
Each matrix pair is checked for:
  - identical dimensions (rows, columns, nnz);
  - identical sparsity pattern (row, col indices match);
  - values within tolerance (max absolute difference < margin).
"""

import argparse
import pathlib
import sys
from dataclasses import dataclass, field

import numpy as np

# Methods and Dirac types exercised by tests/compare-IBM-matrices.sh
METHODS = ["modified", "original"]
DIRACS = [1, 2, 3, 4]
MATRICES = ["A", "M"]


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
class MatrixReport:
    label: str
    generated_path: pathlib.Path
    baseline_path: pathlib.Path
    results: list[CheckResult] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return all(r.passed for r in self.results)

    def summary(self) -> str:
        status = "PASS" if self.all_passed else "FAIL"
        lines = [f"{self.label} [{status}]"]
        lines.append(f"  generated: {self.generated_path}")
        lines.append(f"  baseline:  {self.baseline_path}")
        for r in self.results:
            lines.append(str(r))
        return "\n".join(lines)


def parse_mtx(
    path: pathlib.Path,
) -> tuple[tuple[int, int, int], np.ndarray, np.ndarray, np.ndarray]:
    """Parse a Matrix Market coordinate file.

    Returns (shape, rows, cols, vals) where shape = (nrows, ncols, nnz).
    """
    with open(path, "r") as f:
        # Skip header line
        header = f.readline()
        if not header.startswith("%%MatrixMarket"):
            raise ValueError(f"Not a Matrix Market file: {path}")

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
            line = f.readline()
            parts = line.split()
            rows[i] = int(parts[0])
            cols[i] = int(parts[1])
            vals[i] = float(parts[2])

    return (nrows, ncols, nnz), rows, cols, vals


def check_matrix(
    generated: pathlib.Path,
    baseline: pathlib.Path,
    margin: float = 1e-5,
) -> MatrixReport:
    report = MatrixReport(generated.stem, generated, baseline)

    if not generated.exists():
        report.results.append(
            CheckResult("file_exists", False, f"generated file not found: {generated}")
        )
        return report
    if not baseline.exists():
        report.results.append(
            CheckResult("file_exists", False, f"baseline file not found: {baseline}")
        )
        return report
    report.results.append(CheckResult("file_exists", True, "both files present"))

    try:
        gen_shape, gen_rows, gen_cols, gen_vals = parse_mtx(generated)
        base_shape, base_rows, base_cols, base_vals = parse_mtx(baseline)
    except Exception as exc:
        report.results.append(CheckResult("parse", False, str(exc)))
        return report
    report.results.append(CheckResult("parse", True, "both files parsed"))

    # Dimensions match
    dims_match = gen_shape == base_shape
    report.results.append(
        CheckResult(
            "dims",
            dims_match,
            f"gen=({gen_shape[0]}, {gen_shape[1]}, {gen_shape[2]}) "
            f"base=({base_shape[0]}, {base_shape[1]}, {base_shape[2]})",
        )
    )

    if not dims_match:
        return report

    # Sort both by (row, col) so sparsity patterns can be compared
    gen_order = np.lexsort((gen_cols, gen_rows))
    base_order = np.lexsort((base_cols, base_rows))

    gen_rows_sorted = gen_rows[gen_order]
    gen_cols_sorted = gen_cols[gen_order]
    gen_vals_sorted = gen_vals[gen_order]
    base_rows_sorted = base_rows[base_order]
    base_cols_sorted = base_cols[base_order]
    base_vals_sorted = base_vals[base_order]

    # Sparsity pattern match
    pattern_match = np.array_equal(
        gen_rows_sorted, base_rows_sorted
    ) and np.array_equal(gen_cols_sorted, base_cols_sorted)
    report.results.append(
        CheckResult(
            "sparsity_pattern",
            pattern_match,
            f"{'match' if pattern_match else 'MISMATCH'} ({gen_shape[2]} entries)",
        )
    )

    if not pattern_match:
        # Count mismatches
        row_mismatch = np.sum(gen_rows_sorted != base_rows_sorted)
        col_mismatch = np.sum(gen_cols_sorted != base_cols_sorted)
        report.results[
            -1
        ].detail += f" (row mismatches={row_mismatch}, col mismatches={col_mismatch})"
        return report

    # Values within tolerance
    max_diff = float(np.max(np.abs(gen_vals_sorted - base_vals_sorted)))
    report.results.append(
        CheckResult(
            "values",
            max_diff < margin,
            f"max|diff|={max_diff:.2e} (tol={margin:.0e})",
        )
    )

    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check D3Q27 IBM matrix regression: compare generated .mtx files against baselines."
    )
    parser.add_argument(
        "--base",
        type=pathlib.Path,
        default=pathlib.Path.cwd(),
        help="Base directory containing generated ibm_*.mtx files (default: cwd)",
    )
    parser.add_argument(
        "--baseline-dir",
        type=pathlib.Path,
        default=pathlib.Path(__file__).parent / "baseline_ibm_matrices",
        help="Directory containing baseline .mtx files",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=1e-5,
        help="Tolerance for value comparison (default: 1e-5)",
    )
    args = parser.parse_args()

    all_reports: list[MatrixReport] = []

    for method in METHODS:
        for dirac in DIRACS:
            for matrix in MATRICES:
                # Try both CPU and GPU generated files
                for compute in ["CPU", "GPU"]:
                    gen_name = f"ibm_{compute}_matrix-{matrix}_method-{method}_dirac-{dirac}.mtx"
                    gen_path = args.base / gen_name
                    if not gen_path.exists():
                        continue

                    base_name = f"matrix-{matrix}_method-{method}_dirac-{dirac}.mtx"
                    base_path = args.baseline_dir / base_name

                    label = f"{compute}/{matrix}/{method}/dirac{dirac}"
                    report = check_matrix(gen_path, base_path, margin=args.margin)
                    report.label = label
                    all_reports.append(report)

    if not all_reports:
        print(
            "No ibm_*.mtx files found. Run tests/compare-IBM-matrices.sh first.",
            file=sys.stderr,
        )
        return 1

    for report in all_reports:
        print(report.summary())
        print()

    total = len(all_reports)
    passed = sum(1 for r in all_reports if r.all_passed)
    print(f"{passed}/{total} matrix comparisons passed.")

    if passed < total:
        print("\nFailures:")
        for report in all_reports:
            if not report.all_passed:
                for r in report.results:
                    if not r.passed:
                        print(f"  {report.label}: {r.name}: {r.detail}")
        return 1

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (RuntimeError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
