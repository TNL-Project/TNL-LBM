# TNL-LBM PROJECT KNOWLEDGE BASE

**Updated:** 2026-09-11
**Branch:** main

## OVERVIEW

TNL-LBM is a C++/CUDA header-only Lattice Boltzmann Method (LBM) framework built on top of the Template Numerical Library (TNL).
It provides pluggable collision operators, streaming patterns, boundary conditions, and macroscopic quantities for 2D and 3D direct numerical simulations,
with optional Python bindings via nanobind and distributed execution through CUDA-aware MPI.

## STRUCTURE

```
.
├── include/
│   ├── lbm3d/           # Core LBM framework (2D and 3D)
│   │   ├── d3q27/       # D3Q27 lattice model kernels
│   │   ├── d3q7/        # D3Q7 lattice model kernels
│   │   ├── d2q9/        # D2Q9 lattice model kernels
│   │   └── py_*.h       # nanobind Python binding wrappers
│   ├── lbm_common/      # Shared utilities (logging, PNG, file I/O)
│   └── lbm2d/           # 2D placeholder (only .gitkeep, unused)
├── sim_NSE/             # 3D Navier-Stokes example simulations
├── sim_NSE_ADE/         # 3D NSE + advection-diffusion examples
├── sim_adjoint/         # 3D Adjoint-based sensitivity examples
├── sim_2D/              # 2D example simulations
├── pytnl_lbm/           # Python extension module
├── tests/               # pytest unit, regression & integration suites + subproject test
│   ├── unit/            # pytest unit tests: python_bindings/ + C++ unit tests (.cu → doctest)
│   ├── regression/      # pytest result checks (ibm, nse, d2q9, mpi, adjoint) + IBM matrix baselines
│   ├── integration/     # end-to-end output-data pipeline test (pytest + CUDA driver sim)
│   └── subproject/      # external consumption test via CMake FetchContent
├── CMakeLists.txt       # Root build configuration
├── pyproject.toml       # Python tooling (ruff, mypy, pyright)
└── .gitlab-ci.yml       # CUDA/HIP CI pipeline
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Add a 3D collision operator | `include/lbm3d/d3q27/col_*.h` | Inherit from `D3Q27_COMMON` or `D3Q27_COMMON_WELL` |
| Add a 2D collision operator | `include/lbm3d/d2q9/col_*.h` | Inherit from `D2Q9_COMMON` |
| Add a 3D boundary condition | `include/lbm3d/d3q27/bc.h` | Extend `D3Q27_BC_All::GEO` enum and handlers |
| Add a 2D boundary condition | `include/lbm3d/d2q9/bc.h` | Extend `D2Q9_BC_All::GEO` enum and handlers |
| Change 3D streaming pattern | `include/lbm3d/d3q27/streaming_*.h` | Select `D3Q27_STREAMING_{AA,AB_PULL,AB_PUSH,ESO_TWIST,ESO_PULL,ESO_PUSH}` in the sim's CONFIG; the `D3Q27_STREAMING` alias in `streaming.h` (A-B pull default) follows the `TNL_LBM_STREAMING_PATTERN` CMake selection |
| Change 2D streaming pattern | `include/lbm3d/d2q9/streaming_*.h` | Select `D2Q9_STREAMING_{AA,AB_PULL,AB_PUSH,ESO_TWIST,ESO_PULL,ESO_PUSH}` in the sim's CONFIG; the `D2Q9_STREAMING` alias in `streaming.h` (A-B pull default) follows the `TNL_LBM_STREAMING_PATTERN` CMake selection |
| Simulation driver loop | `include/lbm3d/core.h` | `execute<STATE>(state)` orchestrates init/update/finalize |
| Python binding surface | `pytnl_lbm/pytnl_lbm.cpp` | Exports one concrete `SP_D3Q27_CUM_ConstInflow` instantiation |
| 3D example simulations | `sim_NSE/*.cu`, `sim_NSE_ADE/*.cu`, `sim_adjoint/*.cu` | Each `int main()` is a standalone CMake executable |
| 2D example simulations | `sim_2D/*.cu` | sim2d_1 (channel+hole), sim2d_2 (Poiseuille), sim2d_Taylor_Green, sim2d_hills |
| Unit-test C++ binary | `tests/unit/*.cu` | doctest cases compiled into one binary |
| Regression tests | `tests/regression/` | pytest suites: IBM matrices vs `baseline_ibm_matrices/` + IBM flow-field checks, D3Q27 NSE (sim_1..sim_4 + forcing variants) checks, D2Q9 verification checks + forcing variant, MPI multi-rank checks (test_mpi.py) |
| Output-data pipeline test | `tests/integration/` | pytest suite driving `test_outputdata` (BP5, SST, Catalyst inline/plugin engines) |
| External consumption test | `tests/subproject/` | Verifies TNL-LBM works via CMake `FetchContent` |

## CODE MAP

| Symbol | Type | Location | Role |
|--------|------|----------|------|
| `LBM<CONFIG>` | struct | `include/lbm3d/lbm.h` | Top-level distributed lattice manager; owns `Lattice`, `LBM_BLOCK`s, MPI state |
| `LBM_BLOCK<CONFIG>` | struct | `include/lbm3d/lbm_block.h/.hpp` | Local subdomain data and compute parameters |
| `State<NSE>` | struct | `include/lbm3d/state.h/.hpp` | Simulation state orchestrator with counters, probes, checkpoints, IBM |
| `execute<STATE>` | function | `include/lbm3d/core.h` | Main simulation loop: init → update → I/O → wall/final-time checks |
| `LBM_CONFIG<TRAITS,MACRO,COLL,STREAMING,DATA>` | struct | `include/lbm3d/defs.h` | Compile-time policy bundle selecting lattice model components |
| `D3Q27_CUM` / `D3Q27_CLBM` / `D3Q27_KBC_*` | struct | `include/lbm3d/d3q27/col_*.h` | 3D collision operators (cumulant, cascaded LBM, KBC) |
| `D2Q9_CLBM` / `D2Q9_CLBM_Straka2016` | struct | `include/lbm3d/d2q9/col_clbm.h` | 2D CLBM: Geier 2017 (Galilean invariant) and Straka 2016 (anisotropic, legacy) |
| `D2Q9_SRT` | struct | `include/lbm3d/d2q9/col_srt.h` | 2D single-relaxation-time BGK |
| `D3Q27_STREAMING_{AA,AB_PULL,AB_PUSH,ESO_TWIST,ESO_PULL,ESO_PUSH}` | struct | `include/lbm3d/d3q27/streaming_*.h` | 3D streaming implementations (all co-includable); `D3Q27_STREAMING` in `streaming.h` is a macro-selected legacy alias |
| `D2Q9_STREAMING_{AA,AB_PULL,AB_PUSH,ESO_TWIST,ESO_PULL,ESO_PUSH}` | struct | `include/lbm3d/d2q9/streaming_*.h` | 2D streaming implementations (all co-includable); `D2Q9_STREAMING` in `streaming.h` is a macro-selected legacy alias |
| `D3Q27_BC_All` | struct | `include/lbm3d/d3q27/bc.h` | 3D boundary condition dispatch for all GEO tags |
| `D2Q9_BC_All` | struct | `include/lbm3d/d2q9/bc.h` | 2D boundary condition dispatch for all GEO tags |
| `D3Q27_MACRO_Default` | struct | `include/lbm3d/d3q27/macro.h` | 3D default macroscopic output: density + velocity |
| `D2Q9_MACRO_Default` | struct | `include/lbm3d/d2q9/macro.h` | 2D default macroscopic output: density + velocity |
| `dir9` | struct | `include/lbm3d/defs.h` | D2Q9 direction enum (zz, pz, mz, zp, zm, pp, mm, pm, mp) |
| `D2Q9_KernelStruct` | struct | `include/lbm3d/defs.h` | D2Q9 kernel data carrier (9 DFs, rho, vx, vy, fx, fy) |
| `Lagrange3D<LBM>` | struct | `include/lbm3d/lagrange_3D.h/.hpp` | Immersed Boundary Method (IBM) point cloud manager |
| `DataManager` | struct | `include/lbm3d/DataManager.h` | Output variable registration and ADIOS2 I/O coordination |
| `CheckpointManager` | struct | `include/lbm3d/checkpoint.h` | Save/load simulation state and iteration counters |
| `pytnl_lbm` | module | `pytnl_lbm/pytnl_lbm.cpp` | nanobind module exposing `Lattice`, `LBM`, `State`, `execute` |

## CONVENTIONS

- **Commit messages**: Conventional Commits (`type(scope): description`), e.g. `fix(examples): guard against zero dimensions`.
  Types: `fix`, `feat`, `refactor`, `perf`, `docs`, `test`, `ci`, `chore`, `build`, `style`.
  Body formatting: ≤72 chars per line, use markdown where appropriate.
- **Assisted-by trailer**: When AI tools contribute, add `Assisted-by: AGENT_NAME:MODEL_VERSION` (Linux kernel convention from `Documentation/process/coding-assistants.rst`).
  Example: `Assisted-by: Opencode:glm-5.1`.
  Optional tool names may follow: `Assisted-by: Claude:claude-3-opus coccinelle sparse`.
  Do NOT use `Signed-off-by` for AI — only humans certify DCO.
- **Header-only library**: `TNL_LBM` is a CMake `INTERFACE` target; executables carry all compilation cost.
- **C++ source suffixes**: Headers use `.h` (not `.hpp`); `.hpp` files are template implementations included from `.h`.
- **Formatting**: Tabs for C++/CUDA, 2 spaces for YAML/config;
  `.clang-format` disables `SortIncludes` due to cyclic includes.
- **Comments**: descriptive comments documenting non-obvious functionality.
- **Column limit**: 150 (with a `TODO` to lower to 128).
- **C++17 required**, compiler extensions off (`CMAKE_CXX_EXTENSIONS OFF`).
- **Dependencies**: Fetched via `FetchContent` (fmt, spdlog, nlohmann_json, argparse, magic_enum, TNL, nanobind, PyTNL);
  system packages required: ADIOS2, PNG, MPI, OpenMP.
- **CUDA architecture**: Defaults to `"native"`; CI falls back to `75` on GPU-less runners.
- **HIP debug builds**: Use `-O1 -g`, not `-O0`, to avoid ROCm memory-access faults.
- **Python**: `pyproject.toml` targets Python 3.12; bindings are optional via `TNL_LBM_BUILD_PYTHON`.
- **No CTest**: Tests are shell scripts invoked post-build, not registered with CMake.
- **doctest**: C++ unit tests (tests/unit/*.cu) use doctest `TEST_SUITE_BEGIN`/`TEST_SUITE_END`; one binary per module (`test_cpp_units`).

## ANTI-PATTERNS (THIS PROJECT)

- **Variable-length arrays**: `-Werror=vla` makes them a compile error.
- **Including headers in the wrong order**: `SortIncludes: Never` is intentional; reordering can break compilation.
- **Copying core objects**: `LBM`, `LBM_BLOCK`, `State`, `Lagrange3D`, and writers have deleted copy constructors.
- **Assuming all Lagrangian points share one GPU**: IBM code assumes points reside on the first GPU.
- **Using `-O0` for HIP debug**: Causes memory-access faults; use `-O1`.
- **Ignoring `isDDNonZero` / `is3DiracNonZero`**: Dirac-delta callers must check non-zero support explicitly.
- **Unrestricted viscosity**: `LBM_VISCOSITY` must stay below `1/6` for stability in some setups.
- **Wrong `setBoundary*` call order**: `setBoundaryX/Y/Z` stamp whole planes and overwrite each other at shared edges/corners
  (last call wins, see `lbm_block.hpp`).
  Set `GEO_SYMMETRY` planes first, inflow/outflow next, then walls, and the `GEO_NOTHING` ghost layer always last
  — otherwise symmetry tags capture the inflow/outflow face edges.
- **Fenced comments**: Do not add decorative comments with "fences", e.g. `# -----------------` or `// -----------------`.
- **Inlining many heavy per-face BC bodies into the fused A-A kernel**: ptxas
  trades registers for local-memory spills (proven by ncu on sim_2 AA: 96->80
  regs, ~1.2 GB/launch spill traffic, -6.8% GLUPS). The symmetric failure is
  a runtime-face BC body in D2Q9's small fused kernels: defeats constant
  folding (hills AA -5%, and drifts values off the legacy FP contractions
  until the mass-conservation regression fails at final time). Neither
  extreme works: `__noinline__` outlines collapse D2Q9 AA 26.2 -> 11.0 GLUPS
  (ABI overhead); one runtime-generic body for all models regresses D3Q27 AB
  -3.5%. The tuned dispatch lives in `bc.h`: D3Q27 carries one
  runtime-parameterized body called directly in both patterns; D2Q9 carries
  a `template <int AXIS, int SIGN>` body (constexpr slot arithmetic)
  instantiated per face in the preCollision switch.
- **Verifying FP-bitwise contracts in test kernels only**: `lbm_fma_rn` pins
  tuned against fp-contract fusion spots in a small test kernel do not
  guarantee the same contractions in the larger fused production kernel; the
  production regression suites are the gate.

## UNIQUE STYLES

- **Simulation-centric layout**: Example executables live in domain-named directories (`sim_NSE`, `sim_NSE_ADE`, `sim_adjoint`, `sim_2D`)
  rather than a single `apps/` folder.
- **Lattice-model subpackages**: `d3q27/`, `d3q7/`, and `d2q9/` mirror each other with `col_*`, `eq_*`, `streaming_*`, `bc.h`, `macro.h`, `common*.h`.
- **Streaming pattern as a template policy**: every `*_STREAMING_*` struct carries `DFMAX` (number of DF arrays) and `output_df`; pattern predicates are variable templates specialized next to each struct — `is_AA_v` / `is_AB_PULL_v` / `is_AB_PUSH_v` (identity), `twisted_layout_v` (twisted initial DF storage), `requires_ghost_layer_v` (cross-site streaming requires the ghost-layer idiom) — and all pattern-dependent branches are `if constexpr` on them. The esoteric patterns add `is_ESO_TWIST_v` / `is_ESO_PULL_v` / `is_ESO_PUSH_v` (identity) and `is_esoteric_in_place_v` (all three in-place pairs schemes) with an `is_pair_head` helper (odd-numbered slot of each opposite direction pair). `LBM_CONFIG` instantiates `DATA = _DATA<TRAITS, STREAMING::DFMAX>` so the kernel-argument `dfs[]` array is pattern-sized. The legacy `TNL_LBM_STREAMING_PATTERN_*` macros are consulted only by the three `streaming.h` umbrella aliases.
- **Traits-driven arrays**: Type aliases encode host/device and content (`__hmap_array_t`, `__dlat_array_t`, `__hmacro_array_t`).
- **nanobind exports**: All export functions follow `export_<Thing>(m, "Name")`;
  the module exposes one fully-instantiated D3Q27 cumulant configuration.
- **Unified error calculation**: 2D verification sims (sim2d_2, sim2d_Taylor_Green, sim_2) share the same error calculation pattern: `_vx`/`_vy`/`_vz` naming, `to_phys` lambda, structured bindings, `hmap`+`isFluid||isPeriodic` guard.

## COMMANDS

```bash
# Configure and build with default CUDA auto-detection
cmake -B build -S . -G Ninja
cmake --build build

# Run a 3D example simulation
./build/sim_NSE/sim_1 4
mpirun -np 2 ./build/sim_NSE/sim_1 4

# Convenience build-and-run scripts
./sim_NSE/run sim_1 4
./sim_NSE_ADE/run sim_T1 4

# Run 2D verification simulations
./build/sim_2D/sim2d_1 --resolution 1
./build/sim_2D/sim2d_2 --resolution 1
./build/sim_2D/sim2d_Taylor_Green --resolution 1
./build/sim_2D/sim2d_hills --resolution 1 --Re 1000

# Run all tests (unit + regression + integration; default pytest collection)
# All tests need a CUDA GPU (all executables are CUDA builds; suite skips itself without one).
pytest
pytest tests/unit tests/integration  # skip the heavier regression suite
pytest tests/regression  # simulation result checks only
# Test the A-B reference build without moving directories:
pytest --build-dir build-ab

# Python bindings (after build)
PYTHONPATH=build/pytnl_lbm python -c "import pytnl_lbm"

# Spell-check (CI lint job)
typos --color always --sort
```

## STREAMING PATTERN SELECTION (TNL_LBM_STREAMING_PATTERN)

`-DTNL_LBM_STREAMING_PATTERN=<AA|AB_PULL|AB_PUSH|ESO_TWIST|ESO_PULL|ESO_PUSH>`
(root CMakeLists.txt) selects which streaming pattern all simulations and the
Python bindings are compiled with; default `AB_PULL` (e.g. configure the A-A
tree with `cmake -B build-aa -S . -G Ninja -DTNL_LBM_STREAMING_PATTERN=AA`).
The variable only defines the
`TNL_LBM_STREAMING_PATTERN_{AA,AB_PULL,AB_PUSH,ESO_TWIST,ESO_PULL,ESO_PUSH}`
macro via the `TNL_LBM` interface target, which the `streaming.h` umbrellas
read to pick the `D{3Q27,3Q7,2Q9}_STREAMING` aliases — library headers
themselves are pattern-agnostic, and all six patterns can be instantiated in
one translation unit via the explicit
`*_STREAMING_{AA,AB_PULL,AB_PUSH,ESO_TWIST,ESO_PULL,ESO_PUSH}` types.

**Esoteric in-place patterns (ESO_TWIST, ESO_PULL, ESO_PUSH):**

- Three single-array, in-place patterns after Lehmann 2022 (esoteric pull and
  push) and Geier & Schönherr 2017 (esoteric twist), available for D2Q9, D3Q7
  and D3Q27 (`streaming_ESO_{PULL,PUSH,TWIST}.h` next to the other patterns in
  each lattice-model directory, `DFMAX = 1`, `output_df = df_cur`,
  `requires_ghost_layer_v = true`). Each alternates two parity phases on one
  DF array on the same `even_iter = (iterations % 2) == 1` schedule as A-A;
  pairs of opposite slots (heads = odd-numbered directions) cover memory the
  way the A-A twist does, but tensor-ordered pairwise exchange instead of a
  direction swap.
- Bitwise-identical physics to A-B pull by construction: the initialization
  runs as a virtual "-1 → 0" iteration in
  `LBM_BLOCK::setInitialCondition(ic)` (called from `State::reset()` via the
  scalar-constant `setEquilibrium(rho, vx, vy, vz)` convenience or a
  site-wise IC functor `(KS&, gx, gy, gz)`): collision is replaced by
  equilibrium evaluation and the pattern's own `postCollisionStreaming`
  writes the parity-0 placement at the layout-authoring parity
  (`even_iter = true`), so launches read identical populations from
  launch 0. Kernel A first stamps the analytic-IC equilibrium at the own
  site, all Q slots in the natural layout, over the halo-padded range (the
  boundary fallback: never-transported cells keep it) and writes the
  initial macro from the analytic condition directly (exact for every
  pattern and decomposition); kernel B then replays the streaming write on
  the owned range (A-A: halo-padded, its twisted own-site mapping covers
  the ghost cells it reads) temporarily binding `df_out` to the live array
  for the two-array patterns. `ESO_TWIST` is the exception: its p(c) =
  max(c, 0) componentwise placement cannot be assembled by the pairwise
  exchange (which moves each pair by the FULL head velocity - mixed-sign
  diagonals disagree), so it stages the natural field and gathers the
  placement directly. Under MPI, `State::reset()` then finalizes the ghost
  planes with the regular per-slot DF+macro exchange at `even_iter = true`
  (there is no separate natural-layout init exchange). The two-pass
  outflow scheme and `GEO_WALL` swap are pattern-agnostic; the outflow
  pass uses per-pattern `streamingOutflow{,Interp}` gathers in each
  esoteric struct.
- Verified: final TGV fields are bitwise identical to A-B pull for all three
  esoteric patterns (and A-B push) in 2D (sim2d_Taylor_Green, 64², 25 000
  iterations) and 3D (sim_4, 32³), single-rank, under `mpirun -np 2` (1D
  decomposition), under `mpirun -np 4` with a forced 2×2 decomposition, and
  in 3D under `mpirun -np 8` with a forced 2×2×2 decomposition;
  `tests/regression/test_tgv_bitwise.py` (2D single-rank + 2D np=2 MPI +
  2D np=4 multi-dim + 3D single-rank + 3D np=8 multi-dim gates, frame 0 and
  final fields) enforces it against a reference build
  (`TGV_REFERENCE_BUILD_DIR`, default `build`). The multi-dim gates need a
  truly multi-dimensional split, but the interface-optimal decomposer always
  1D-splits a square domain, so the gates force 2×2 / 2×2×2 via the
  `TNL_LBM_FORCE_DECOMPOSITION="nx,ny,nz"` env hook in
  `lattice_decomposition.h` — only multi-dimensional decompositions allocate
  corner exchange buffers at all. Scope of the bitwise claim: it is these TGV
  gates only — the residual AA-vs-AB D3Q27 codegen divergence below is a
  separate, known issue that does not involve the esoteric patterns.
- DF halo exchange uses per-slot, per-axis, parity-dependent
  `STREAMING::dfSyncDirection(dir, axis, even)`/`dfSyncOffset` descriptors
  (parity of the launch that authored the layout): PULL and PUSH use one
  combined-mask synchronization per slot with a slot-uniform buffer offset;
  TWIST needs a different offset per axis, which one combined mask cannot
  express, so `start4DArraySynchronization` runs two staged combined-mask
  passes per slot — the offset-1 axes first (their receive planes are interior
  sites that the offset-0 pass of a neighbor may source), the offset-0 axes
  second (post-phase-A: offset-1 = axes with c<0; post-phase-B mirrored).
  The combined masks carry the sync pattern's diagonal buffers, so ghost
  edges and corners are exchanged too (a per-axis chain could not reach the
  diagonal ghost cells at all). The external stage_2/3/4 loop then sees the
  parked None mask and does nothing. All patterns apply their buffer
  offsets AFTER stage_0: on the synchronizer's first use its allocateHelper
  resets the offsets to the default, and only stages 1/3 consume them, so
  the re-set keeps them effective also on the first call.
- Multi-dimensional decompositions exposed two further synchronizer defect
  classes, both invisible under 1D splits (which never allocate corner
  buffers) and both guarded by the np=4 gate:
  - Buffer over-activation by mask bitmasking: the TNL synchronizer
    activates an exchange buffer whenever the runtime mask shares any face
    bit with the buffer's direction, so a slot's combined multi-face mask
    also fired partially-overlapping corner buffers whose shifted send
    regions carried stale diagonal-halo values into the neighbor's owned
    cells.     `LBM_BLOCK::setLatticeDecomposition` restricts the per-slot
    `df_sync[i]` patterns to buffers fully contained in one of the slot's
    masks (plus the opposite closure the synchronizer's opposite-buffer
    lookups need).
  - Corner/face unpack race under shift-1 exchanges: the synchronizer
    unpacks each buffer on its own CUDA stream, and a face buffer's shifted
    receive region covers the corner cell owned by a corner buffer's
    receive region, so which message landed last in the shared cell was
    nondeterministic — diagonal-junction owned corner cells (each block's
    local (0,0) slot pp) kept their previous-phase values, an init-time
    seed that decays over iterations and broke the np=4 bitwise gate.
    All buffers of the esoteric per-iteration synchronizers now share one
    sequencing stream (`df_seq_stream` in `LBM_BLOCK`): kernel execution
    then follows the std::map (enum) order, which places subset directions
    before their supersets — faces first, then edges, then corners — so the
    corner's overwrite is deterministic. The compute boundary kernels are
    already host-synchronized before the DF exchange starts (SimUpdate), so
    collapsing the buffer streams cannot race the pack side.
  - Contiguous-recv direct placement vs kernel unpacks (3-cut-axis
    decompositions, np=8 gate): the TNL synchronizer's `copyHelper` binds
    contiguous receive views directly into the array, so MPI delivered
    fresh edge/corner payloads (x/z-thin y-full edge lines, single-cell
    corners — regions that only exist when all three axes are cut) during
    stage_2, and stage_3's copy-kernel unpacks of colliding non-contiguous
    faces then overwrote them with stale face-routed content — the
    subset-before-superset stream ordering cannot order direct MPI writes.
    Fixed by a TNL patch (recv side always staged + copy kernel; contiguous
    direct-bind kept for sends only), applied to the fetched
    `build*/_deps/tnl-src/src/TNL/Containers/DistributedNDArraySynchronizer.h`
    copies. IMPORTANT: a fresh clone/CI refetches unpatched TNL — the np=8
    gate fails there until the patch is merged upstream into TNL and the
    FetchContent tag is bumped (patch copy: `patches/tnl-recv-staging.patch`).
  - ESO_TWIST spurious third exchange: after the two staged TWIST passes,
    `start4DArraySynchronization` must park the None mask and skip the
    shared stage_0/stage_1 tail (`continue`) — re-arming with the canonical
    slot direction ran a duplicate exchange with the last pass' buffer
    offsets, whose shift-1 receivers hit owned planes and overwrote freshly
    authored values (deterministic seam corruption, NaN by ~500 iterations
    under `mpirun -np 2`).
- A-B push (`streaming_AB_PUSH.h`): the pattern's identity `streaming()` read
  expects the post-stream layout (slot `(i, s)` = population that arrived at
  `s` from `s - c_i`, the same field the A-B pull launch-0 gather reads),
  which the init-time `postCollisionStreaming` replay of
  `setInitialCondition` produces directly. The per-slot DF halo exchange
  keeps the canonical slot
  direction with `setBufferOffsets(1)` — push arrivals live in the rank's
  ghost planes and must land in the neighbor's OWNED planes — and restricts
  each slot's sync pattern to the buffers fully contained in the canonical
  slot direction or its opposite (a partially-overlapping buffer of a
  shift-1 exchange ships un-authored ghost values; the opposite buffer must
  stay in the pattern or stage_2's receive has no tags and is silently
  skipped; the containment also keeps the face/edge sub-buffers of a
  diagonal canonical direction, without which the push-arrival region
  spanning the ghost column, row and corner could not be exchanged),
  sequenced on the shared `df_seq_stream`. `tests/regression/test_tgv_bitwise.py`
  asserts A-B push bitwise-equal to A-B pull like every other pattern.

**Considerations for boundary conditions under A-A:**

- Boundary conditions must NOT sit on the outermost array layer: AA neighbor
  indices are unclamped (`kernels.h`), so an edge BC wrap-writes into the
  opposite column/row. Apply the ghost-layer idiom: outermost plane
  `GEO_NOTHING`, BC on `1`/`N-2`.
- `GEO_INFLOW_MOMENT` BC planes must not intersect with another inflow plain in a corner or edge:
  the corner sites have no interior-side neighbor for the runtime face detection
  and are rejected by `validateFaceDetectedBC`.
- `GEO_OUTFLOW_RIGHT` and `GEO_OUTFLOW_RIGHT_INTERP` run through a
  deterministic two-pass scheme in *both* A-A and A-B streaming patterns
  (it replaced the legacy fused kernel path, which raced with same-launch
  `postCollisionStreaming` writers in the A-A single array;
  A-B never had the race but shares the scheme so there is one outflow code path).
- `GEO_OUTFLOW_RIGHT_INTERP` blend arithmetic is pinned to a canonical rounding:
  all 36 blend sites (AA 6+18, AB 3+9) use the canonical `lbm_fma_rn(cs,A,(1-cs)*B)` form
  from `include/lbm_common/rounding.h`,
  making D2Q9 bitwise-identical AA vs AB.
  Originally, NVVM contracted the `cs*A + (1-cs)*B` blend in mixed operand orders per statement
  on sm_75/sm_86 Release (`fma(A,cs,wB)` for most, `fma(B,w,csA)` for the mp blend),
  giving 1-ulp-different values between mirrored direction pairs (mm/mp) at the outflow column;
  the chaotic wake amplified this to 1e-3-class mirror-symmetry breakage.
  Architecture codegen issue, not hardware: compute_86 PTX reproduced the failure bit-for-bit on sm_120.

**Known limitations under A-A:**

- NSE_ADE (`sim_T1`, `sim_T2`) is NOT covered by the two-pass scheme
  (state_NSE_ADE.h launches no outflow kernel and its BC placement ignores
  the ghost-layer idiom) — do not run these under AA or the esoteric
  in-place patterns (they share the single-array + ghost-layer constraints);
  enforced by a `static_assert` in `State_NSE_ADE` (only the two-array
  A-B patterns compile).
- `sim_adjoint` requires the A-B pull pattern and is EXCLUDED from non-`AB_PULL` builds (CMake-level; its pytest module skips when `STREAMING_PATTERN != "AB_PULL"`).
  Findings for a future AA-native adjoint design:
  `streamingAdjoint` even-phase two-step reads escape the 1-cell ghost layer (CUDA 700 at boundary-adjacent sites);
  the reversed gather races with same-launch `postCollisionStreaming` writers in the single array (nondeterministic garbage profiles);
  the `GEO_ADJOINT_INFLOW_BB_LEFT` refill in d3q27/bc.h must be parity-aware
  (m-family from the site's own slot after an even/twisted write, matching p-slot one hop downstream after an odd push).
- A-A fails the 2D TGV bitwise gate under multi-dimensional MPI
  decompositions (np=4 forced 2×2 via `TNL_LBM_FORCE_DECOMPOSITION`,
  max|diff| on the order of 1.4e-4 across the field; the 3D np=8 2×2×2 gate
  fails with ~7.1e-4): pre-existing defect,
  strictly out of scope for the esoteric work — do not run A-A with
  multi-dimensional decompositions when bitwise identity to A-B pull is
  required. The esoteric in-place patterns pass the same gates.
- Residual AA-vs-AB divergence in D3Q27
  (open; root cause known on both arch classes studied; fix decision *deferred*):
  after the blend pin both patterns are individually mirror-perfect,
  but AA and AB still drift apart through wake-amplified ulp seeds whose seeding site is arch-dependent:
  - sm_75/86-class codegen (compute_86-virtual JIT'd on sm_120 reproduces the failures bit-for-bit):
    a single ≤2-ulp flip authored inside `outflowPass` at step ~261 at the x=126 column
    (from input state bitwise-identical between patterns),
    then wake-amplified to max|d| ≈ 2.75e-4 (vx), ~1.55e-4 (vy/vz), 7.15e-7 (density) by final time;
    seed rate ≈ 1 flip per (261 steps × 784 pass cells).
    `D3Q27_CUM::collision` is provably bit-identical between builds;
    the divergence lives in the *non*-blend part of the pass chain
    — AA compiles it as 4 outlined `.func` calls vs fully-inlined under AB,
    with different FMA/regrouping choices of the same source expressions
    Forensics: `docs/aa-ab-outflow-divergence/`.
  - native sm_120: the outflow pass is already bit-identical between patterns;
    the divergence seeds in the *main* kernel — predominantly the `D3Q27_CUM` collision core (`col_cum.h`),
    where NVVM makes per-expression FMA-contraction/CSE choices that differ between the AA and AB builds,
    secondarily the `GEO_INFLOW_MOMENT` moment BC;
    macro helpers and all init kernels are bit-identical
    and both streamings carry zero FP ops.
    First field diff at frame ~1 (≈step 40) in the inflow/baffle region x=1..33,
    ~72% of cells carry ulp diffs by mid-run, final max|d| ≈ 3.57e-4 (vx).
    Codegen attribution: `docs/aa-ab-divergence-sm120-codegen/`.
  Two candidate fixes `fix-outflow-unify-codegen` (`a164865`) and `fix-outflow-pin-arithmetic` (`aaaac43`).

## NOTES

- `include/lbm2d/` is a placeholder (unused); all 2D code lives under `include/lbm3d/d2q9/` — the `lbm3d` namespace is shared by 2D and 3D code.
- `CUDA` is always defined for `lbm3d` (`-DUSE_CUDA`), even when compiling with HIP.
- When both CUDA and HIP compilers are detected, CMake enables CUDA and disables HIP (mirrors TNL's own handling); HIP is only enabled when no CUDA compiler is found.
- Python bindings (`pytnl_lbm`) are built only for CUDA builds; HIP builds skip them entirely.
- The CI matrix exercises CUDA Release/Debug (all six streaming patterns), HIP Release/Debug, non-MPI, and subproject consumption.
