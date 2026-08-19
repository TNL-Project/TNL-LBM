#!/usr/bin/env python3
"""
Between-property metric for TNL-LBM AMR verification.

Compares the AMR composite against a bracket formed by independent uniform-coarse
and uniform-fine reference runs, evaluated in physical coordinates so that the
error is comparable across different resolutions.

Two metrics are reported per cycle:
  1. Violation count — number of cells outside the bracket [min(C,F) - tol, max(C,F) + tol].
  2. Excess norm — L1, L2, and Linf norms of the excess (how far outside the bracket),
     integrated in physical space (each fine cell carries volume physDl_f^3).

The physical-coordinate sampling ensures that a coarse cell (volume physDl_c^3) and
the 8 fine subcells it covers (total volume 8 * physDl_f^3 = physDl_c^3) contribute
equally to the norm, making the metric resolution-independent.

Usage:
  python3 between_metric.py [--amr-dir results_sim_AMR_res01_np001]
                             [--ref-coarse results_sim_NSE_ref_coarse]
                             [--ref-fine results_sim_NSE_ref_fine]
                             [--gamma 3.0] [--eps-rho 1e-5] [--eps-vel 2e-5]
                             [--selftest]
"""

import argparse
import os
import sys

import h5py
import numpy as np

FIELDS = ["rho", "vx", "vy", "vz"]


def read_vtkhdf_levels(path):
    """Read all levels from a VTKHDF OverlappingAMR file."""
    f = h5py.File(path, "r")
    vtk = f["VTKHDF"]
    levels = {}
    for key in vtk.keys():
        if key.startswith("Level"):
            lv = int(key[5:])
            cd = vtk[key]["CellData"]
            levels[lv] = {}
            for fld in FIELDS:
                arr = cd[fld][:]
                n = arr.size
                s = round(n ** (1 / 3))
                if s * s * s == n:
                    levels[lv][fld] = arr.reshape(s, s, s)
                else:
                    levels[lv][fld] = arr
            if "vtkGhostType" in cd:
                gt = cd["vtkGhostType"][:]
                s = round(gt.size ** (1 / 3))
                levels[lv]["ghost"] = (
                    gt.reshape(s, s, s) if s * s * s == gt.size else gt
                )
            else:
                levels[lv]["ghost"] = None
    f.close()
    return levels


def read_ref_cycles(ref_dir, max_cycle=10):
    """Read reference simulation output cycles."""
    cycles = []
    for cyc in range(max_cycle + 1):
        path = os.path.join(ref_dir, f"output_amr_{cyc:04d}.vtkhdf")
        if not os.path.exists(path):
            break
        levels = read_vtkhdf_levels(path)
        if 0 in levels:
            cycles.append(levels[0])
        else:
            cycles.append(list(levels.values())[0])
    return cycles


def _upsample_axis_2x(arr, axis, periodic):
    """Trilinear 2x upsampling along one axis.

    Each coarse cell i produces two fine subcells:
      sub0 at -0.25*dx (toward neighbor i-1): weight 3/4 on i, 1/4 on i-1
      sub1 at +0.25*dx (toward neighbor i+1): weight 3/4 on i, 1/4 on i+1
    """
    n = arr.shape[axis]
    if periodic:
        left = np.roll(arr, 1, axis=axis)
        right = np.roll(arr, -1, axis=axis)
    else:
        left = np.roll(arr, 1, axis=axis)
        right = np.roll(arr, -1, axis=axis)
        sl0 = [slice(None)] * arr.ndim
        sl0[axis] = 0
        left[tuple(sl0)] = arr[tuple(sl0)]
        sl1 = [slice(None)] * arr.ndim
        sl1[axis] = -1
        right[tuple(sl1)] = arr[tuple(sl1)]

    sub0 = 0.75 * arr + 0.25 * left
    sub1 = 0.75 * arr + 0.25 * right

    out_shape = list(arr.shape)
    out_shape[axis] = 2 * n
    result = np.empty(out_shape, dtype=arr.dtype)
    sl_even = [slice(None)] * arr.ndim
    sl_even[axis] = slice(0, None, 2)
    sl_odd = [slice(None)] * arr.ndim
    sl_odd[axis] = slice(1, None, 2)
    result[tuple(sl_even)] = sub0
    result[tuple(sl_odd)] = sub1
    return result


def upsample2x(arr, periodic=True):
    """Trilinear 2x upsampling for cell-centered data (separable, 3/4:1/4 weights)."""
    result = _upsample_axis_2x(arr, 0, periodic)
    result = _upsample_axis_2x(result, 1, periodic)
    result = _upsample_axis_2x(result, 2, periodic)
    return result


def amr_composite(
    amr_levels,
    fld,
    footprint_origin=(32, 32, 32),
    footprint_size=(64, 64, 64),
    periodic=True,
):
    """Build a composite field at fine resolution from AMR levels."""
    if 1 in amr_levels:
        fine = amr_levels[1][fld]
        if 0 in amr_levels:
            coarse_up = upsample2x(amr_levels[0][fld], periodic)
            result = coarse_up.copy()
            oz, oy, ox = footprint_origin
            sz, sy, sx = footprint_size
            result[oz : oz + sz, oy : oy + sy, ox : ox + sx] = fine
            return result
        return fine
    return amr_levels[0][fld]


def compute_excess(A, C, F, gamma, eps):
    """Compute the excess: how far A is outside the bracket [min(C,F)-tol, max(C,F)+tol].

    Returns an array of the same shape; zero where A is inside the bracket.
    """
    lo = np.minimum(C, F)
    hi = np.maximum(C, F)
    tol = gamma * np.abs(F - C) + eps
    excess = np.zeros_like(A)
    below = A < (lo - tol)
    above = A > (hi + tol)
    excess[below] = (lo - tol - A)[below]
    excess[above] = (A - hi - tol)[above]
    return excess


def norms(arr, cell_volume):
    """Compute volume-weighted L1, L2, and Linf norms.

    L1  = sum |arr| * cell_volume
    L2  = sqrt(sum |arr|^2 * cell_volume)
    Linf = max |arr|
    """
    abs_arr = np.abs(arr)
    l1 = float(np.sum(abs_arr) * cell_volume)
    l2 = float(np.sqrt(np.sum(abs_arr**2) * cell_volume))
    linf = float(abs_arr.max()) if abs_arr.size > 0 else 0.0
    return l1, l2, linf


def main():
    parser = argparse.ArgumentParser(description="Between-property metric for AMR LBM")
    parser.add_argument("--amr-dir", default="results_sim_AMR_res01_np001")
    parser.add_argument("--ref-coarse", default="results_sim_NSE_ref_coarse")
    parser.add_argument("--ref-fine", default="results_sim_NSE_ref_fine")
    parser.add_argument("--gamma", type=float, default=3.0)
    parser.add_argument("--eps-rho", type=float, default=1e-5)
    parser.add_argument("--eps-vel", type=float, default=2e-5)
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--max-cycle", type=int, default=10)
    args = parser.parse_args()

    eps = {
        "rho": args.eps_rho,
        "vx": args.eps_vel,
        "vy": args.eps_vel,
        "vz": args.eps_vel,
    }

    print(
        f"metric: gamma={args.gamma} eps_rho={args.eps_rho} eps_vel={args.eps_vel} selftest={args.selftest}"
    )

    # Read reference cycles
    refC = read_ref_cycles(args.ref_coarse, args.max_cycle)
    refF = read_ref_cycles(args.ref_fine, args.max_cycle)

    if not refC or not refF:
        print(
            f"ERROR: reference data missing (coarse: {len(refC)} cycles, fine: {len(refF)} cycles)"
        )
        print(f"  ref-coarse dir: {args.ref_coarse}")
        print(f"  ref-fine dir: {args.ref_fine}")
        sys.exit(1)

    ncycles = min(len(refC), len(refF), args.max_cycle + 1)
    print(f"Loaded {ncycles} reference cycles (coarse {len(refC)}, fine {len(refF)})")

    # Physical parameters (from sim_AMR: PHYS_HEIGHT=0.41, N_coarse=64)
    PHYS_HEIGHT = 0.41
    N_COARSE = 64
    physDl_c = PHYS_HEIGHT / N_COARSE
    physDl_f = physDl_c / 2
    cell_volume_f = physDl_f**3  # fine cell volume (the common resolution)

    # Fine footprint origin/size in fine coords, re-anchored (Schönherr-ch7
    # band registration): offset' = 2*origin_coarse + 1 = 33, interior
    # local' = 2*K - 2 = 62 -- the composite splices the fine-authoritative
    # interior only (the coarse-authoritative ring rows and the covered F2C
    # destination rows come from the coarse level)
    footprint_origin = (33, 33, 33)
    footprint_size = (62, 62, 62)

    # Domain volume for normalization
    domain_volume = PHYS_HEIGHT**3

    print(
        f"physical: physDl_c={physDl_c:.8e} physDl_f={physDl_f:.8e} "
        f"cell_volume_f={cell_volume_f:.8e} domain_volume={domain_volume:.8e}"
    )
    print()

    all_pass = True
    for cyc in range(ncycles):
        amr_path = os.path.join(args.amr_dir, f"output_amr_{cyc:04d}.vtkhdf")
        if not os.path.exists(amr_path):
            print(f"cycle {cyc}: AMR output missing, skipping")
            continue

        amr_levels = read_vtkhdf_levels(amr_path)

        total_viol = 0
        field_strs = []
        total_l1 = 0.0
        total_l2_sq = 0.0
        total_linf = 0.0

        for fld in FIELDS:
            C = upsample2x(refC[cyc][fld], periodic=True)
            F = refF[cyc][fld]
            if args.selftest:
                A = F.copy()
            else:
                A = amr_composite(amr_levels, fld, footprint_origin, footprint_size)

            excess = compute_excess(A, C, F, args.gamma, eps[fld])

            n_viol = int(np.count_nonzero(excess))
            l1, l2, linf = norms(excess, cell_volume_f)

            total_viol += n_viol
            total_l1 += l1
            total_l2_sq += l2**2
            total_linf = max(total_linf, linf)

            field_strs.append(
                f"{fld}: viol={n_viol} L1={l1:.4e} L2={l2:.4e} Linf={linf:.4e}"
            )

        total_l2 = float(np.sqrt(total_l2_sq))

        # Normalize by domain volume for resolution-independent comparison
        total_l1_norm = total_l1 / domain_volume
        total_l2_norm = total_l2 / float(np.sqrt(domain_volume))

        status = "pass" if total_viol == 0 else "FAIL"
        if total_viol > 0:
            all_pass = False

        print(f"cycle {cyc:2d}: {status} viol={total_viol}")
        for s in field_strs:
            print(f"         {s}")
        print(
            f"         TOTAL: L1={total_l1:.4e} (norm={total_l1_norm:.4e}) "
            f"L2={total_l2:.4e} (norm={total_l2_norm:.4e}) Linf={total_linf:.4e}"
        )

    print()
    if all_pass:
        print("RESULT: PASS - AMR within reference envelope")
    else:
        print("RESULT: FAIL - between-property violated")


if __name__ == "__main__":
    main()
