#!/usr/bin/env python3
"""
Between-property metric for TNL-LBM AMR verification.

Compares every cell of the AMR composite against a bracket formed by
independent uniform-coarse and uniform-fine reference runs.

Usage:
  python3 between_metric.py [--amr-dir results_sim_AMR_res01_np001]
                             [--ref-coarse results_sim_NSE_ref_coarse]
                             [--ref-fine results_sim_NSE_ref_fine]
                             [--gamma 3.0] [--eps-rho 1e-5] [--eps-vel 2e-5]
                             [--selftest]
"""

import argparse
import csv
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
                # VTKHDF stores flat 1D arrays; reshape to 3D
                n = arr.size
                s = round(n ** (1 / 3))
                if s * s * s == n:
                    levels[lv][fld] = arr.reshape(s, s, s)
                else:
                    levels[lv][fld] = arr  # leave as-is if not cubic
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
    """Read reference simulation output cycles (output_3D.bp via h5py or vtkhdf)."""
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


def broadcast2x(arr):
    """Upsample a 3D array by factor 2 (nearest-neighbor broadcast)."""
    return np.repeat(np.repeat(np.repeat(arr, 2, axis=0), 2, axis=1), 2, axis=2)


def amr_composite(
    amr_levels, fld, footprint_origin=(32, 32, 32), footprint_size=(64, 64, 64)
):
    """Build a composite field at fine resolution from AMR levels.
    Level 0 (coarse) is broadcast 2x; level 1 (fine) overrides where present."""
    if 1 in amr_levels:
        fine = amr_levels[1][fld]
        if 0 in amr_levels:
            coarse_bc = broadcast2x(amr_levels[0][fld])
            result = coarse_bc.copy()
            oz, oy, ox = footprint_origin
            sz, sy, sx = footprint_size
            result[oz : oz + sz, oy : oy + sy, ox : ox + sx] = fine
            return result
        return fine
    return amr_levels[0][fld]


def signed_iface_dist(shape, origin, size):
    """Compute signed distance from the fine footprint boundary.
    Negative = inside fine, positive = outside (coarse-only region)."""
    nz, ny, nx = shape
    oz, oy, ox = origin
    sz, sy, sx = size
    dz = np.maximum(np.maximum(oz - np.arange(nz), np.arange(nz) - (oz + sz - 1)), -1)
    dy = np.maximum(np.maximum(oy - np.arange(ny), np.arange(ny) - (oy + sy - 1)), -1)
    dx = np.maximum(np.maximum(ox - np.arange(nx), np.arange(nx) - (ox + sx - 1)), -1)
    return dz[:, None, None], dy[None, :, None], dx[None, None, :]


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

    # Fine footprint origin/size in fine coords (for face classification)
    # sim_AMR: 64^3 coarse, footprint [16,48)^3 = fine [32,96)^3
    footprint_origin = (32, 32, 32)
    footprint_size = (64, 64, 64)

    all_pass = True
    for cyc in range(ncycles):
        amr_path = os.path.join(args.amr_dir, f"output_amr_{cyc:04d}.vtkhdf")
        if not os.path.exists(amr_path):
            print(f"cycle {cyc}: AMR output missing, skipping")
            continue

        amr_levels = read_vtkhdf_levels(amr_path)
        t = cyc * 0.05472  # approximate; real t from log if needed

        total_viol = 0
        field_strs = []
        for fld in FIELDS:
            C = broadcast2x(refC[cyc][fld])
            F = refF[cyc][fld]
            if args.selftest:
                A = F.copy()
            else:
                A = amr_composite(amr_levels, fld, footprint_origin, footprint_size)

            lo = np.minimum(C, F)
            hi = np.maximum(C, F)
            tol = args.gamma * np.abs(F - C) + eps[fld]

            viol = (A < lo - tol) | (A > hi + tol)
            n_viol = int(viol.sum())
            if n_viol > 0:
                diff = np.abs(A - np.clip(A, lo, hi))
                max_amp = float(diff[viol].max())
            else:
                max_amp = 0.0

            total_viol += n_viol
            field_strs.append(f"{fld}:{n_viol} (max {max_amp:.6e})")

        status = "pass" if total_viol == 0 else "FAIL"
        if total_viol > 0:
            all_pass = False
        print(
            f"cycle {cyc:2d} t={t:.5f}: {status} total_viol={total_viol} | {' | '.join(field_strs)}"
        )

    print()
    if all_pass:
        print("RESULT: PASS - AMR within reference envelope")
    else:
        print("RESULT: FAIL - between-property violated")


if __name__ == "__main__":
    main()
