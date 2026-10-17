# pyright: reportAttributeAccessIssue=false
# The Lattice.global_ attribute and the getLocalMacroView function are missing
# in the generated .pyi stub: nanobind stubgen cannot express their C++ dependent
# types (see sim_NSE/sim_1.py for the same workaround).
"""MPI payload for test_local_macro_view.py -- run via ``mpirun -np 2``.

Exercises the ``pytnl_lbm.getLocalMacroView`` binding (a strictly-local 4D view
of the distributed ``hmacro`` array, exported to NumPy via DLPack). The
anchor-offset contract only manifests with non-zero overlaps, so the check
needs at least two MPI ranks and therefore runs as a subprocess payload driven
by pytest.

Falsification design: an analytic pattern is written into the strictly-local
box *through* the exported NumPy view, and every owned cell is then read back
through the independent global-indexed ``__getitem__`` path of the distributed
array. A wrong ghost-offset anchor would land the writes in shifted cells and
fail the readback; the ghost cells themselves must stay untouched.
"""

from __future__ import annotations

import os
import pathlib
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]
BINDINGS_DIR = PROJECT_ROOT / "build" / "pytnl_lbm"
if str(BINDINGS_DIR) not in sys.path:
    sys.path.insert(0, str(BINDINGS_DIR))
PYTNL_BUILD_SRC = PROJECT_ROOT / "build" / "_deps" / "pytnl-build" / "src"
if PYTNL_BUILD_SRC.is_dir() and str(PYTNL_BUILD_SRC) not in sys.path:
    sys.path.insert(0, str(PYTNL_BUILD_SRC))

# avoids MPI_Init aborts on machines where Open MPI detects both ROCm and CUDA
os.environ.setdefault("OMPI_MCA_accelerator", "null")

import numpy as np  # noqa: E402
from mpi4py import MPI  # noqa: E402

import pytnl_lbm  # noqa: E402


def cell_value(field: int, x: int, y: int, z: int) -> float:
    # unique per (field, global cell); every value stays below 2**23, so it is
    # exactly representable in the float32 macro storage
    return float(field * 1_000_000 + x * 10_000 + y * 100 + z)


def build_state(comm: MPI.Intracomm) -> pytnl_lbm.State_SP_D3Q27_CUM_ConstInflow:
    lat_t = pytnl_lbm.Lattice_3_double_long
    lat = lat_t()
    lat.global_ = lat_t.CoordinatesType((32, 16, 16))
    lat.physDl = 1.0
    lat.physDt = 1.0
    lat.physViscosity = 1.0

    state = pytnl_lbm.State_SP_D3Q27_CUM_ConstInflow("local_macro_view_check", comm, lat, str(PROJECT_ROOT / "adios2.xml"))
    assert state.canCompute(), "state.canCompute() failed"

    block = state.nse.blocks[0]
    if block.hmacro.getLocalStorageSize() == 0:
        block.allocateHostData()
    assert block.hmacro.getLocalStorageSize() > 0, "hmacro not allocated"
    return state


def write_pattern(view: np.ndarray, begins: tuple[int, ...], ends: tuple[int, ...]) -> None:
    # analytic pattern of the OWNED global coordinates, written through the view
    ii = np.arange(begins[1], ends[1])
    jj = np.arange(begins[2], ends[2])
    kk = np.arange(begins[3], ends[3])
    ff = np.arange(view.shape[0])
    pattern = (
        ff[:, None, None, None] * 1_000_000
        + ii[None, :, None, None] * 10_000
        + jj[None, None, :, None] * 100
        + kk[None, None, None, :]
    )
    view[:] = pattern.astype(view.dtype)


def count_owned_mismatches(
    h: pytnl_lbm.dist_hmacro_array_SP_D3Q27_CUM_ConstInflow,
    begins: tuple[int, ...],
    local: tuple[int, ...],
    fields: int,
    rank: int,
) -> int:
    # readback through the independent global-indexed __getitem__ path
    errors = 0
    for field in range(fields):
        for i in range(local[0]):
            for j in range(local[1]):
                for k in range(local[2]):
                    x, y, z = begins[1] + i, begins[2] + j, begins[3] + k
                    got = h[field, x, y, z]
                    if got != cell_value(field, x, y, z):
                        errors += 1
                        if errors <= 5:
                            print(f"[rank {rank}] MISMATCH f={field} ({i},{j},{k}): got {got}", file=sys.stderr)
    return errors


def count_written_ghosts(
    h: pytnl_lbm.dist_hmacro_array_SP_D3Q27_CUM_ConstInflow,
    sizes: tuple[int, ...],
    begins: tuple[int, ...],
    ends: tuple[int, ...],
    overlaps: tuple[int, ...],
    fields: int,
    rank: int,
) -> int:
    # ghost cells must stay untouched by writing through the view
    errors = 0
    for field in range(fields):
        for x in range(max(begins[1] - overlaps[1], 0), min(ends[1] + overlaps[1], sizes[1])):
            for y in range(max(begins[2] - overlaps[2], 0), min(ends[2] + overlaps[2], sizes[2])):
                for z in range(max(begins[3] - overlaps[3], 0), min(ends[3] + overlaps[3], sizes[3])):
                    owned = begins[1] <= x < ends[1] and begins[2] <= y < ends[2] and begins[3] <= z < ends[3]
                    if not owned and h[field, x, y, z] != 0.0:
                        errors += 1
                        if errors <= 5:
                            print(f"[rank {rank}] GHOST WRITTEN at ({field},{x},{y},{z})", file=sys.stderr)
    return errors


def main() -> int:
    comm = MPI.COMM_WORLD
    if comm.size < 2:
        print("payload must run under mpirun with at least 2 ranks", file=sys.stderr)
        return 1

    state = build_state(comm)
    block = state.nse.blocks[0]
    h = block.hmacro

    sizes = h.getSizes()
    begins = h.getLocalBegins()
    ends = h.getLocalEnds()
    overlaps = h.getOverlaps()
    if not any(overlaps[1:]):
        print(f"[rank {comm.rank}] expected non-zero overlaps in a distributed dimension", file=sys.stderr)
        return 1

    # the exported strictly-local view: shape (fields, local.x, local.y, local.z)
    view = np.from_dlpack(pytnl_lbm.getLocalMacroView(h))
    local = (ends[1] - begins[1], ends[2] - begins[2], ends[3] - begins[3])
    fields = sizes[0]
    if view.shape != (fields, *local):
        print(f"[rank {comm.rank}] shape mismatch: {view.shape} != {(fields, *local)}", file=sys.stderr)
        return 1

    write_pattern(view, begins, ends)

    errors = count_owned_mismatches(h, begins, local, fields, comm.rank)
    if errors:
        print(f"[rank {comm.rank}] {errors} mismatched owned cells", file=sys.stderr)
        return 1

    ghost_errors = count_written_ghosts(h, sizes, begins, ends, overlaps, fields, comm.rank)
    if ghost_errors:
        print(f"[rank {comm.rank}] {ghost_errors} ghost cells were modified by the view", file=sys.stderr)
        return 1

    print(f"[rank {comm.rank}] OK: strictly-local view anchored correctly, ghosts untouched", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
