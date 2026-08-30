# TNL-LBM PROJECT KNOWLEDGE BASE

**Updated:** 2026-08-28
**Branch:** feat/amr-schonherr-ch7

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
├── sim_AMR/             # AMR example simulations (Taylor-Green + developing channel, nested 2-level..5-level modes)
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
| Change 3D streaming pattern | `include/lbm3d/d3q27/streaming_AA.h` / `streaming_AB.h` | AB is default; define `AA_PATTERN` or `AB_PATTERN` before including core LBM headers |
| Change 2D streaming pattern | `include/lbm3d/d2q9/streaming_AA.h` / `streaming_AB.h` | AB is default; define `AA_PATTERN` or `AB_PATTERN` before including core LBM headers |
| Simulation driver loop | `include/lbm3d/core.h` | `execute<STATE>(state)` orchestrates init/update/finalize |
| Python binding surface | `pytnl_lbm/pytnl_lbm.cpp` | Exports one concrete `SP_D3Q27_CUM_ConstInflow` instantiation |
| 3D example simulations | `sim_NSE/*.cu`, `sim_NSE_ADE/*.cu`, `sim_adjoint/*.cu` | Each `int main()` is a standalone CMake executable |
| 2D example simulations | `sim_2D/*.cu` | sim2d_1 (channel+hole), sim2d_2 (Poiseuille), sim2d_Taylor_Green, sim2d_hills |
| Unit-test doctest binaries | `tests/unit/*.cu` | doctest cases (runner `doctest_main.cu`): `test_cpp_units` carries `test_outflowcover.cu`, `test_decomposition.cu` and the AMR Schönherr registration/exactness suites; the AMR gate suites live in the per-pattern `test_amr_units_{ab,aa}` (suites `amr_coupling`/`amr_subcycling`/`amr_vtkhdf_writer`/`amr_nesting`), the per-define exactness drivers in `test_amr_f2c_schonherr_{ab,aa}` and `test_amr_c2f_smoke_{eq,dev,norm,shear}_{ab,aa}` |
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
| `D3Q27_STREAMING` | struct | `include/lbm3d/d3q27/streaming_*.h` | 3D AA or AB streaming implementation |
| `D2Q9_STREAMING` | struct | `include/lbm3d/d2q9/streaming_*.h` | 2D AA or AB streaming implementation |
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

## UNIQUE STYLES

- **Simulation-centric layout**: Example executables live in domain-named directories (`sim_NSE`, `sim_NSE_ADE`, `sim_adjoint`, `sim_2D`)
  rather than a single `apps/` folder.
- **Lattice-model subpackages**: `d3q27/`, `d3q7/`, and `d2q9/` mirror each other with `col_*`, `eq_*`, `streaming_*`, `bc.h`, `macro.h`, `common*.h`.
- **Streaming pattern compile-time switch**: `AA_PATTERN` or `AB_PATTERN` must be defined before `core.h` is included.
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
TNL_LBM_BUILD_DIR=build-ab pytest

# Python bindings (after build)
PYTHONPATH=build/pytnl_lbm python -c "import pytnl_lbm"

# AMR gate: build the doctest AMR binaries + run the 10 counted targets
# (4 gate suites × {ab,aa} via doctest --test-suite + 2 ParaView E2E; needs a CUDA GPU)
./tests/run-amr-tests.sh
# 2-level AMR example simulations (Taylor-Green; --convective-times 20 for the long decision-table run)
./build/sim_AMR/sim_AMR --resolution 1
./build/sim_AMR/sim_AMR_channel --resolution 1
# Nested wall-attached channel with the windbreak rod array (5 lattice levels):
./build/sim_AMR/sim_AMR_channel --resolution 1 --max-level 4

# Spell-check (CI lint job)
typos --color always --sort
```

## A-A STREAMING PATTERN (TNL_LBM_AA_PATTERN)

`-DTNL_LBM_AA_PATTERN=ON` (root CMakeLists.txt) compiles all simulations
(except `sim_adjoint`, see below) and the Python bindings with the
single-array A-A pattern; default OFF keeps A-B. The pattern must be selected
via CMake — per-file `#define AB_PATTERN` was removed; `include/lbm3d/defs.h`
provides the AB default when neither is set.

**Considerations for boundary conditions under A-A:**

- Boundary conditions must NOT sit on the outermost array layer: AA neighbor
  indices are unclamped (`kernels.h`), so an edge BC wrap-writes into the
  opposite column/row. Apply the ghost-layer idiom: outermost plane
  `GEO_NOTHING`, BC on `1`/`N-2`.
- Lateral `GEO_INFLOW_LEFT` moment BCs diverge under AA on ghost-adjacent planes.
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
  the ghost-layer idiom) — do not run these under AA.
- `sim_adjoint` requires the A-B pattern and is EXCLUDED from AA builds (CMake-level; its pytest module skips via `AA_PATTERN`).
  Findings for a future AA-native adjoint design:
  `streamingAdjoint` even-phase two-step reads escape the 1-cell ghost layer (CUDA 700 at boundary-adjacent sites);
  the reversed gather races with same-launch `postCollisionStreaming` writers in the single array (nondeterministic garbage profiles);
  the `GEO_ADJOINT_INFLOW_BB_LEFT` refill in d3q27/bc.h must be parity-aware
  (m-family from the site's own slot after an even/twisted write, matching p-slot one hop downstream after an odd push).
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
    secondarily the `GEO_INFLOW_LEFT` moment BC;
    macro helpers and all init kernels are bit-identical
    and both streamings carry zero FP ops.
    First field diff at frame ~1 (≈step 40) in the inflow/baffle region x=1..33,
    ~72% of cells carry ulp diffs by mid-run, final max|d| ≈ 3.57e-4 (vx).
    Codegen attribution: `docs/aa-ab-divergence-sm120-codegen/`.
  Two candidate fixes `fix-outflow-unify-codegen` (`a164865`) and `fix-outflow-pin-arithmetic` (`aaaac43`).

## AMR (STATIC 2:1 REFINEMENT — SCHÖNHERR-CH7 BAND, N-LEVEL NESTING)

Static, cell-centered, volumetric AMR — nested 2:1 refinement levels
(`max_level` ≤ 4, i.e. five lattice levels on the realized windbreak target),
single MPI rank, single GPU, D3Q27, CUDA-only coupling kernels. The coupling is
the Schönherr-2015 ch.7 target-band conversion landed on this branch (16
commits), generalized from one fine level to N-level nesting by the
amr-nlevel-nesting arc (commits A–G): parent-frame `global_offset`, the V1–V10
creation suite (+ V9 advisory), the `advancePair` Berger–Colella recursion, and
the parent-keyed wall chain with R4 wall-pedestal prisms (internals doc
`docs/AMR-for-LBM-implementation.md`, multi-level chapter §13; normative
band/cycle contract `docs/AMR-schonherr-ch7-target-contract.md`, per-pair
nesting addendum §11).

- **Simulations**: `sim_AMR/sim_AMR.cu` (Taylor-Green 2-level AMR,
  `--convective-times N` long runs), `sim_AMR/sim_AMR_channel.cu` (Dirichlet
  developing-channel diagnostic, the B.7 artifact; `--max-level 2..4` opts into
  the nested wall-attached chain, with the windbreak rod array stamped on the
  finest level by default — `--no-windbreak` and the
  `--windbreak-{diameter,pitch,height,row-spacing}` knobs steer it, and
  `--max-level 2..3` need `--no-windbreak` or tuned knobs with the default rod
  geometry). Probe CLI on both: `--out3d-iter-period N` (per-iteration frame
  cadence).
- **Surfaces**: `include/lbm3d/amr_decomposition.h` (`createAMRBlocks` —
  footprint re-anchored one fine cell inward per face, gs ≥ 3 minimum, V1–V10
  nesting validation, parent-frame `global_offset` normalization;
  `markAMRInterface` — ring {halo c=−1 + reactivated surface shell c=0} tagged
  `GEO_AMR_INTERFACE`, footprint-depth ≥ 1 cells frozen `GEO_NOTHING`),
  `include/lbm3d/amr_state.h` (`State_AMR` driver: `SimUpdate` = the
  `advancePair` pair recursion with cumulative per-level substep counters,
  `buildCouplings` vertex-straddling patches + R4 wall-pedestal prisms,
  `buildFineWallMasks` wall chain, SimInit map-pattern assertion),
  `include/lbm3d/d3q27/amr_coupling.h` (`cudaAMR_CoarseToFine`,
  `cudaAMR_FineToCoarse`), `include/lbm3d/viz/OverlappingAMRWriter.{h,hpp}`,
  `sim_AMR/amr_chain_solver.h` (nested footprint derivation),
  `sim_AMR/amr_windbreak.h` (windbreak rod layout/stamping).
- **Schönherr cycle with simulated band** (per adjacent level pair; the
  `max_level == 1` reduction is byte-frozen by the bit-identity harness): fine
  substep 1
  (**widened extent [−1, local+1)** — the inner ghost rows are INTEGRATED,
  collide+stream like interior fluid, sourcing the outer ghost row) → fine substep 2
  (interior-only; its boundary data is substep 1's updated inner rows in the
  other AB frame) → coarse step → F2C once (depth-1 skin, reads rotation-1
  frame) → C2F single fill of the substep-0 frame covering **both** ghost rows
  (SimInit does the same single-frame fill for cycle 0; the former frame-1 fill
  is removed as dead traffic). Converted 2026-08-23 per the contract's fork row
  (c) trigger (T16 null verdict); the conversion-era six-step passive band is
  superseded. H9 and the BVP refill are hard-removed; F2C and C2F touch
  disjoint sets. Nesting: pairs recurse (level L runs 2^L substeps per coarse
  step), F2C once per parent substep, C2F once per pair plus the cycle-end
  level-ascending cascade. Checkpoint restart does not carry across the band
  registration.
- **Strategy surfaces** (`sim_AMR/CMakeLists.txt`): C2F default is the σ-form
  compact-moment (σ = 1/2; `TNL_LBM_C2F_STRATEGY=C2F_LAGRANGE` opts back to the
  3rd-order Lagrange). The carve pre-pass was hard-removed on 2026-08-23 —
  the ch7 band map-pattern assertion rejects covered windows at SimInit, so
  the pre-pass could never fire; `C2F_CARVE`/`C2F_NO_CARVE` warn at configure
  and gate no code. F2C default
  is `TNL_LBM_F2C_STRATEGY=F2C_SCHONHERR` (the §7.2 σ = 2 compact-moment
  transfer, default since commit 15); `=F2C_LAGRAVA` opts out to the 4×4×4
  Lagrava filter — a named no-op define: the kernel splits on
  `#ifdef F2C_SCHONHERR` only, and `F2C_BOX_AVERAGE` selects the 1/8 average
  inside that else-branch; nested wall sharing hard-errors under F2C_LAGRAVA
  at SimInit (the R4 pedestal depth 3 covers only the Schönherr own-8 window).
  Debug channel defines:
  `C2F_EQ_ONLY/DEV_ONLY/NORM_ONLY/SHEAR_ONLY`. Pre-flip build caches keep the
  old empty strategy — re-default with `cmake -B build -S . -UTNL_LBM_F2C_STRATEGY`.
- **AMR gate**: `tests/run-amr-tests.sh` builds the two consolidated per-pattern
  doctest binaries `test_amr_units_{ab,aa}` (TEST_SUITEs
  `amr_coupling`/`amr_subcycling`/`amr_vtkhdf_writer`/`amr_nesting`, all suites
  ported to doctest together with the `test_amr_f2c_schonherr_{ab,aa}` /
  `test_amr_c2f_smoke_*` drivers) and runs the 10 AMR targets
  (the 4 gate suites × {ab,aa} via doctest `--test-suite=` + ParaView E2E +
  the E2E nesting arm); 10/10 at HEAD. Bit-identity evidence harness:
  `tests/regression/test_amr_bitidentity.py` — verify mode compares every
  `max_level == 1` artifact against the committed
  `tests/regression/amr_ref/manifest.json` (11/11 at HEAD; re-record ONLY from
  a trusted pre-change tree; its mock-suite artifacts drive the consolidated
  binaries per-suite).
  pytest sides: `tests/unit/test_cpp_units.py` (AMR doctest suites),
  `tests/unit/test_amr_f2c_schonherr.py`, `tests/unit/test_amr_c2f_debug_smoke.py`.
- **Measured verdict (recorded, not repaired)**: the conversion was an
  experiment (contract §1). On the T16 20-tc decision table the −23 % mass
  leak is closed ~5 orders on every HEAD arm (era effect of the band
  registration + six-step cycle, not the F2C branch), the seam bias amplifies
  ×1.166 within the pre-registered ×1.2 bound, and the vortex does NOT survive
  at 20 tc on any arm (the control-era survival was interface-pump-fed) —
  honest negative/null result; the full table is quoted in commit `1bd158c`'s
  body and at `docs/AMR-for-LBM-implementation.md` ¶ "Interface density bias".
  Probe tools: `tests/interface_seam_metric.py` (`--fine-row 0 --coarse-row 16`
  = the re-paired pairing of contract §5), `tests/between_metric.py`
  (footprint window re-pinned 33/62).
- **Multi-level status (shipped, windbreak target achieved)**: the R = 1 chain
  realizes 5 wall-chained lattice levels 0..4 (L4 spans 86×22×43 parent cells;
  the level-4 y fine span is 44, not 48 — the telescoping budget deviation,
  hard floors pass) with the rod array on the finest map (3 rods, 2+1
  half-pitch stagger, 1440 cells); pre-registered mass/KE tables for the
  no-rod chain and the rod run are recorded in the commit `6ae4a61`/`5214b01`
  bodies and in doc §13.5.

## NOTES

- `include/lbm2d/` is a placeholder (unused); all 2D code lives under `include/lbm3d/d2q9/` — the `lbm3d` namespace is shared by 2D and 3D code.
- `CUDA` is always defined for `lbm3d` (`-DUSE_CUDA`), even when compiling with HIP.
- When both CUDA and HIP compilers are detected, CMake enables CUDA and disables HIP (mirrors TNL's own handling); HIP is only enabled when no CUDA compiler is found.
- Python bindings (`pytnl_lbm`) are built only for CUDA builds; HIP builds skip them entirely.
- The CI matrix exercises CUDA Release/Debug, HIP Release/Debug, non-MPI, and subproject consumption.
