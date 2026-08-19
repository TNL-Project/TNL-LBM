#!/usr/bin/env python3
"""
Interface seam metric for TNL-LBM AMR verification.

Face-mean signed delta of rho/f00/f05/f06 between a fine-level row and the
paired coarse-level row at the x-min face of the AMR interface (the "seam"),
evaluated per output cycle from a series of per-iteration VTKHDF
OverlappingAMR frames.

The reader idiom follows tests/between_metric.py; cell data are raveled
z*ny*nx + y*nx + x (x fastest) and reshaped to (z, y, x) cubes, so the
x-normal seam face is indexed on the last axis. All analysis is float64
(the OverlappingAMR writer stores the fields as f64 datasets).

Default row pairing is the pre-reanchor ("old") seam pairing of the
Schönherr-ch7 conversion plan (§1.4): the fine standard row local 0
(fine-global index 32, cell center 16.25 coarse dx) vs the coarse ring cell
c=-1 (domain index 15, center 15.5). The defaults are derived from the
first frame's Level1 AMRBox, not hardcoded:

  fine_row    = 0                     (fine-local x of the footprint surface)
  coarse_row  = lo_x // 2 - 1         (the coarse ring row c=-1)
  fine_face   = [0, fine_local)       (full fine tangent extent)
  coarse_face = [lo_x // 2, hi_x // 2 + 1)

The post-reanchor pairing of plan §1.4 (fine-global 33 vs coarse ring-2 c=0)
is reachable without code edits:

  today (old band geometry): --fine-row 1 --coarse-row 16
  after the re-anchor lands: --fine-row 0 --coarse-row 16

(fine destination rows are ghost/overlap rows and are not part of the
current VTKHDF extent; the "dest rows" pairing of §1.4 becomes readable
once the overlap rows are emitted.)

Usage:
  python3 interface_seam_metric.py [--dir results_sim_AMR_res01_np001]
      [--fine-row N] [--coarse-row N] [--fine-face LO:HI]
      [--coarse-face LO:HI] [--max-cycle N] [--baseline FILE] [--atol TOL]

Output: a CSV block `cycle,rho,f00,f05,f06` with one row per frame at %.10e
precision (11 significant digits, enough for ulp-level comparison), plus a
`post3to{N}` row with the post-formation mean over cycles 3..N (the bias
reduction pinned by the Schönherr-ch7 plan). With --baseline, the series is
diffed against a previous machine output of this script (same CSV block)
and the exit status is nonzero when any shared row deviates by more than
--atol.
"""

import argparse
import os
import sys

import h5py
import numpy as np

FIELDS = ("rho", "f00", "f05", "f06")
CSV_HEADER = "cycle,rho,f00,f05,f06"


def read_vtkhdf_levels(path: str) -> dict[int, dict[str, np.ndarray]]:
    """Read the seam fields and AMRBox from a VTKHDF OverlappingAMR file.

    v1 layout: one block per level; AMRBox row is
    {lo_x, hi_x, lo_y, hi_y, lo_z, hi_z} in the level's own lattice.
    """
    f = h5py.File(path, "r")
    vtk = f["VTKHDF"]
    levels: dict[int, dict[str, np.ndarray]] = {}
    for key in vtk.keys():
        if key.startswith("Level"):
            lv = int(key[5:])
            grp = vtk[key]
            cd = grp["CellData"]
            levels[lv] = {}
            box = grp["AMRBox"][0]
            levels[lv]["amrbox_lo"] = np.array([box[0], box[2], box[4]])
            levels[lv]["amrbox_hi"] = np.array([box[1], box[3], box[5]])
            for fld in FIELDS:
                if fld not in cd:
                    f.close()
                    raise KeyError(
                        f"{path}: dataset '{fld}' missing from {key}/CellData"
                    )
                arr = cd[fld][:]
                n = arr.size
                s = round(n ** (1 / 3))
                if s * s * s == n:
                    levels[lv][fld] = arr.reshape(s, s, s)
                else:
                    levels[lv][fld] = arr
    f.close()
    return levels


def parse_face(spec: str) -> tuple[int, int]:
    """Parse a 'LO:HI' tangent window (python slice semantics, HI exclusive)."""
    lo_s, hi_s = spec.split(":")
    return int(lo_s), int(hi_s)


def face_mean(arr: np.ndarray, row: int, face: tuple[int, int]) -> float:
    """Float64 face-mean over the x-normal plane `row` restricted to `face`."""
    lo, hi = face
    return float(np.mean(arr[lo:hi, lo:hi, row], dtype=np.float64))


def seam_delta(
    levels: dict[int, dict[str, np.ndarray]],
    fine_row: int,
    coarse_row: int,
    fine_face: tuple[int, int],
    coarse_face: tuple[int, int],
) -> dict[str, float]:
    """Face-mean signed delta (fine row minus coarse row) per field for one cycle."""
    return {
        fld: face_mean(levels[1][fld], fine_row, fine_face)
        - face_mean(levels[0][fld], coarse_row, coarse_face)
        for fld in FIELDS
    }


def read_series_file(path: str) -> dict[str, list[float]]:
    """Parse the CSV block of a previous interface_seam_metric output."""
    series: dict[str, list[float]] = {}
    with open(path) as f:
        in_csv = False
        for raw_line in f:
            stripped = raw_line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped == CSV_HEADER:
                in_csv = True
                continue
            if not in_csv:
                continue
            key, *vals = stripped.split(",")
            series[key] = [float(v) for v in vals]
    return series


def diff_baseline(
    series: dict[str, list[float]], baseline_path: str, atol: float
) -> int:
    """Diff the series against a baseline capture; return 0 on PASS, 1 on FAIL.

    The current series is quantized to the printed representation first, so
    a self-baseline diffs exactly to zero and `atol` measures deviation
    beyond the documented %.10e exchange precision.
    """
    baseline = read_series_file(baseline_path)
    current = {k: [float(f"{v:+.10e}") for v in row] for k, row in series.items()}
    shared = [k for k in baseline if k in current]
    if not shared:
        print(f"ERROR: no shared rows with baseline {baseline_path}")
        return 1
    worst = 0.0
    max_dev = dict.fromkeys(FIELDS, 0.0)
    for k in shared:
        for i, fld in enumerate(FIELDS):
            dev = abs(current[k][i] - baseline[k][i])
            max_dev[fld] = max(max_dev[fld], dev)
            worst = max(worst, dev)
    for k in baseline:
        if k not in series:
            print(f"# baseline row '{k}' not present in the current series")
    dev_str = " ".join(f"{fld}: {max_dev[fld]:.3e}" for fld in FIELDS)
    status = "PASS" if worst <= atol else "FAIL"
    print(
        f"BASELINE DIFF ({baseline_path}): rows={len(shared)} "
        f"max|delta| per field [{dev_str}] atol={atol:.3e} -> {status}"
    )
    return 1 if worst > atol else 0


def main() -> int:
    """Run the seam metric over the per-cycle frame series."""
    parser = argparse.ArgumentParser(
        description="Interface seam metric for AMR LBM (face-mean signed delta)"
    )
    parser.add_argument(
        "--dir",
        default="results_sim_AMR_res01_np001",
        help="results directory with output_amr_*.vtkhdf frames",
    )
    parser.add_argument(
        "--fine-row",
        type=int,
        default=None,
        help="fine-level (Level1) local x row (default: 0)",
    )
    parser.add_argument(
        "--coarse-row",
        type=int,
        default=None,
        help="coarse-level (Level0) domain x row (default: lo_x//2 - 1)",
    )
    parser.add_argument(
        "--fine-face",
        default=None,
        help="fine tangent window 'LO:HI' in Level1 local indices (default: full face)",
    )
    parser.add_argument(
        "--coarse-face",
        default=None,
        help="coarse tangent window 'LO:HI' in Level0 domain "
        "indices (default: footprint span)",
    )
    parser.add_argument(
        "--max-cycle",
        type=int,
        default=10,
        help="highest cycle index to attempt (reading stops at the first gap)",
    )
    parser.add_argument(
        "--baseline",
        default=None,
        help="previous machine output of this script to diff against (ulp-level)",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=0.0,
        help="absolute tolerance for the baseline diff (default: 0 = exact)",
    )
    args = parser.parse_args()

    # read the per-cycle frame series (stop at the first gap, between_metric idiom)
    cycles: list[int] = []
    levels_by_cycle: list[dict[int, dict[str, np.ndarray]]] = []
    for cyc in range(args.max_cycle + 1):
        path = os.path.join(args.dir, f"output_amr_{cyc:04d}.vtkhdf")
        if not os.path.exists(path):
            break
        try:
            levels_by_cycle.append(read_vtkhdf_levels(path))
        except KeyError as exc:
            print(f"ERROR: {exc} (the f-fields require --write-dfs simulation output)")
            return 1
        cycles.append(cyc)

    if not cycles:
        print(f"ERROR: no output_amr_*.vtkhdf frames found in {args.dir}")
        return 1
    if 0 not in levels_by_cycle[0] or 1 not in levels_by_cycle[0]:
        print(f"ERROR: expected Level0 and Level1 groups in the frames of {args.dir}")
        return 1

    # pairing: CLI options override the AMRBox-derived defaults (see module docstring)
    lo = levels_by_cycle[0][1]["amrbox_lo"]
    hi = levels_by_cycle[0][1]["amrbox_hi"]
    fine_local = levels_by_cycle[0][1][FIELDS[0]].shape[2]
    fine_row = args.fine_row if args.fine_row is not None else 0
    coarse_row = args.coarse_row if args.coarse_row is not None else int(lo[0]) // 2 - 1
    fine_face = (
        parse_face(args.fine_face) if args.fine_face is not None else (0, fine_local)
    )
    coarse_face = (
        parse_face(args.coarse_face)
        if args.coarse_face is not None
        else (int(lo[0]) // 2, int(hi[0]) // 2 + 1)
    )

    print("# interface_seam_metric.py (face-mean signed delta: fine row - coarse row)")
    print(f"# dir: {args.dir}")
    print(f"# fields: {' '.join(FIELDS)}")
    print(
        f"# pairing: fine_row={fine_row} (Level1 local x) vs "
        f"coarse_row={coarse_row} (Level0 domain x); "
        f"fine_face=[{fine_face[0]}:{fine_face[1]}) "
        f"coarse_face=[{coarse_face[0]}:{coarse_face[1]})"
    )
    print(CSV_HEADER)

    series: dict[str, list[float]] = {}
    deltas: list[dict[str, float]] = []
    for cyc, levels in zip(cycles, levels_by_cycle, strict=True):
        d = seam_delta(levels, fine_row, coarse_row, fine_face, coarse_face)
        deltas.append(d)
        row = [d[fld] for fld in FIELDS]
        series[str(cyc)] = row
        print(f"{cyc}," + ",".join(f"{v:+.10e}" for v in row))

    # post-formation bias reduction (Schönherr-ch7 plan: the offset forms
    # within 2 cycles, so cycles 3..end are the post-formation plateau)
    if len(deltas) > 3:
        post = deltas[3:]
        post_row = []
        for fld in FIELDS:
            vals = np.array([d[fld] for d in post], dtype=np.float64)
            post_row.append(float(np.mean(vals)))
            ratio = float(np.mean(vals) / np.mean(np.abs(vals)))
            print(f"# {fld}: signed/abs mean ratio 3..{cycles[-1]} = {ratio:+.4f}")
        series[f"post3to{cycles[-1]}"] = post_row
        print(f"post3to{cycles[-1]}," + ",".join(f"{v:+.10e}" for v in post_row))

    if args.baseline is not None:
        return diff_baseline(series, args.baseline, args.atol)
    return 0


if __name__ == "__main__":
    sys.exit(main())
