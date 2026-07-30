# TNL-LBM PROJECT KNOWLEDGE BASE

**Generated:** 2026-07-07T10:36:19Z
**Commit:** 7949240
**Branch:** agents

## OVERVIEW

TNL-LBM is a C++/CUDA header-only Lattice Boltzmann Method (LBM) framework built on top of the Template Numerical Library (TNL).
It provides pluggable collision operators, streaming patterns, boundary conditions, and macroscopic quantities for 3D direct numerical simulations,
with optional Python bindings via nanobind and distributed execution through CUDA-aware MPI.

## STRUCTURE

```
.
├── include/
│   ├── lbm3d/           # Core 3D LBM framework
│   │   ├── d3q27/       # D3Q27 lattice model kernels
│   │   ├── d3q7/        # D3Q7 lattice model kernels
│   │   └── py_*.h       # nanobind Python binding wrappers
│   ├── lbm_common/      # Shared utilities (logging, PNG, file I/O)
│   └── lbm2d/           # 2D placeholder (only .gitkeep)
├── sim_NSE/             # Navier-Stokes example simulations
├── sim_NSE_ADE/         # NSE + advection-diffusion examples
├── sim_adjoint/         # Adjoint-based sensitivity examples
├── pytnl_lbm/           # Python extension module
├── tests/               # IBM matrix regression tests + subproject test
├── CMakeLists.txt       # Root build configuration
├── pyproject.toml       # Python tooling (ruff, mypy, pyright)
└── .gitlab-ci.yml       # CUDA/HIP CI pipeline
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Add a collision operator | `include/lbm3d/d3q27/col_*.h` | Inherit from `D3Q27_COMMON` or `D3Q27_COMMON_WELL` |
| Add a boundary condition | `include/lbm3d/d3q27/bc.h` | Extend `D3Q27_BC_All::GEO` enum and handlers |
| Change streaming pattern | `include/lbm3d/d3q27/streaming_AA.h` / `streaming_AB.h` | Define `AA_PATTERN` or `AB_PATTERN` before including `core.h` |
| Simulation driver loop | `include/lbm3d/core.h` | `execute<STATE>(state)` orchestrates init/update/finalize |
| Python binding surface | `pytnl_lbm/pytnl_lbm.cpp` | Exports one concrete `SP_D3Q27_CUM_ConstInflow` instantiation |
| Example simulations | `sim_NSE/*.cu`, `sim_NSE_ADE/*.cu`, `sim_adjoint/*.cu` | Each `int main()` is a standalone CMake executable |
| Regression tests | `tests/compare-IBM-matrices.sh` | Compares generated IBM matrices against `baseline_ibm_matrices/` |
| External consumption test | `tests/subproject/` | Verifies TNL-LBM works via CMake `FetchContent` |

## CODE MAP

| Symbol | Type | Location | Role |
|--------|------|----------|------|
| `LBM<CONFIG>` | struct | `include/lbm3d/lbm.h` | Top-level distributed lattice manager; owns `Lattice`, `LBM_BLOCK`s, MPI state |
| `LBM_BLOCK<CONFIG>` | struct | `include/lbm3d/lbm_block.h/.hpp` | Local subdomain data and compute parameters |
| `State<NSE>` | struct | `include/lbm3d/state.h/.hpp` | Simulation state orchestrator with counters, probes, checkpoints, IBM |
| `execute<STATE>` | function | `include/lbm3d/core.h` | Main simulation loop: init → update → I/O → wall/final-time checks |
| `LBM_CONFIG<TRAITS,MACRO,COLL,STREAMING,DATA>` | struct | `include/lbm3d/defs.h` | Compile-time policy bundle selecting lattice model components |
| `D3Q27_CUM` / `D3Q27_CLBM` / `D3Q27_KBC_*` | struct | `include/lbm3d/d3q27/col_*.h` | Collision operators (cumulant, cascaded LBM, KBC) |
| `D3Q27_STREAMING` | struct | `include/lbm3d/d3q27/streaming_*.h` | AA or AB streaming implementation |
| `D3Q27_BC_All` | struct | `include/lbm3d/d3q27/bc.h` | Boundary condition dispatch for all GEO tags |
| `D3Q27_MACRO_Default` | struct | `include/lbm3d/d3q27/macro.h` | Default macroscopic output: density + velocity |
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
- **Column limit**: 150 (with a `TODO` to lower to 128).
- **C++17 required**, compiler extensions off (`CMAKE_CXX_EXTENSIONS OFF`).
- **Dependencies**: Fetched via `FetchContent` (fmt, spdlog, nlohmann_json, argparse, magic_enum, TNL, nanobind, PyTNL);
  system packages required: ADIOS2, PNG, MPI, OpenMP.
- **CUDA architecture**: Defaults to `"native"`; CI falls back to `75` on GPU-less runners.
- **HIP debug builds**: Use `-O1 -g`, not `-O0`, to avoid ROCm memory-access faults.
- **Python**: `pyproject.toml` targets Python 3.12; bindings are optional via `TNL_LBM_BUILD_PYTHON`.
- **No CTest**: Tests are shell scripts invoked post-build, not registered with CMake.

## ANTI-PATTERNS (THIS PROJECT)

- **Variable-length arrays**: `-Werror=vla` makes them a compile error.
- **Including headers in the wrong order**: `SortIncludes: Never` is intentional; reordering can break compilation.
- **Copying core objects**: `LBM`, `LBM_BLOCK`, `State`, `Lagrange3D`, and writers have deleted copy constructors.
- **Assuming all Lagrangian points share one GPU**: IBM code assumes points reside on the first GPU.
- **Using `-O0` for HIP debug**: Causes memory-access faults; use `-O1`.
- **Ignoring `isDDNonZero` / `is3DiracNonZero`**: Dirac-delta callers must check non-zero support explicitly.
- **Unrestricted viscosity**: `LBM_VISCOSITY` must stay below `1/6` for stability in some setups.

## UNIQUE STYLES

- **Simulation-centric layout**: Example executables live in domain-named directories (`sim_NSE`, `sim_NSE_ADE`, `sim_adjoint`)
  rather than a single `apps/` folder.
- **Lattice-model subpackages**: `d3q27/` and `d3q7/` mirror each other with `col_*`, `eq_*`, `streaming_*`, `bc.h`, `macro.h`, `common*.h`.
- **Streaming pattern compile-time switch**: `AA_PATTERN` or `AB_PATTERN` must be defined before `core.h` is included.
- **Traits-driven arrays**: Type aliases encode host/device and content (`__hmap_array_t`, `__dlat_array_t`, `__hmacro_array_t`).
- **nanobind exports**: All export functions follow `export_<Thing>(m, "Name")`;
  the module exposes one fully-instantiated D3Q27 cumulant configuration.

## COMMANDS

```bash
# Configure and build with default CUDA auto-detection
cmake -B build -S . -G Ninja
cmake --build build

# Run an example simulation
./build/sim_NSE/sim_1 4
mpirun -np 2 ./build/sim_NSE/sim_1 4

# Convenience build-and-run scripts
./sim_NSE/run sim_1 4
./sim_NSE_ADE/run sim_T1 4

# Run IBM regression tests (CPU or GPU)
./tests/compare-IBM-matrices.sh CPU
./tests/compare-IBM-matrices.sh GPU

# Python bindings (after build)
PYTHONPATH=build/pytnl_lbm python -c "import pytnl_lbm"

# Spell-check (CI lint job)
typos --color always --sort
```

## NOTES

- `include/lbm2d/` is a placeholder; all active code is 3D.
- `CUDA` is always defined for `lbm3d` (`-DUSE_CUDA`), even when compiling with HIP.
- The CI matrix exercises CUDA Release/Debug, HIP Release/Debug, non-MPI, and subproject consumption.
