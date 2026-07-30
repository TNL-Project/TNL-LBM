# include/lbm3d DIRECTORY KNOWLEDGE BASE

**Generated:** 2026-07-07T10:36:19Z  
**Commit:** 7949240  
**Branch:** agents

## OVERVIEW

Header-only core of the 3D LBM framework:
simulation orchestration,
distributed lattice blocks,
lattice-model subpackages,
immersed-boundary kernels,
and nanobind wrappers.

## STRUCTURE

```
include/lbm3d/
├── d3q27/                   # D3Q27 lattice model: collision, streaming, BCs, macros
├── d3q7/                    # D3Q7 lattice model: smaller mirrored subpackage
├── py_*.h                   # nanobind export helpers (lattice, LBM, State, BCs, macros)
├── core.h                   # Default simulation entry point (pulls in d3q27)
├── state.h/.hpp             # Simulation state, counters, probes, checkpoints, IBM
├── lbm.h/.hpp               # Distributed lattice manager and MPI state
├── lbm_block.h/.hpp         # Local subdomain data and device arrays
├── lagrange_3D.h/.hpp       # IBM Lagrangian point cloud and sparse matrices
├── DataManager.h            # ADIOS2 IO/engine registry
├── checkpoint.h             # Save/load state and iteration counters
├── kernels.h                # Main collide-and-stream kernel
├── ibm_kernels.h            # IBM interpolation/spreading CUDA kernels
├── dirac.h                  # Dirac-delta support checks and kernels
├── lattice.h                # Physical/lattice unit conversions
├── lattice_decomposition.h  # Domain partitioning helpers
├── nonNewtonian.h           # Non-Newtonian fluid kernels
├── obstacles_lbm.h          # Eulerian obstacle setup (cube, sphere, cylinder)
├── obstacles_ibm.h          # Lagrangian obstacle setup (rectangle, cylinder)
└── defs.h                   # Traits, policy config, streaming pattern defaults
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Customize the driver loop | `core.h` + `state.h` | Override `State::SimInit`, `SimUpdate`, `AfterSimUpdate` |
| Add a full lattice model | `d3q27/` or `d3q7/` | Provide `col_*.h`, `eq_*.h`, `streaming_*.h`, `bc.h`, `macro.h`, `common*.h` |
| Change policies (collision, streaming, BC, data) | `defs.h` | Specialize `LBM_CONFIG<TRAITS,MACRO,COLL,STREAMING,DATA>` |
| Decompose the lattice across MPI ranks | `lattice_decomposition.h` | D1Q3 strip or optimal block decomposition |
| Write simulation output | `DataManager.h` | IO/engine registry for ADIOS2 variables |
| Save/resume a run | `checkpoint.h` | Attribute and array save/load via `DataManager` |
| Add immersed-boundary geometry | `obstacles_ibm.h` | Rectangle and cylinder point-cloud builders |
| Add Eulerian wall geometry | `obstacles_lbm.h` | Cube, sphere, cylinder, bounding-box helpers |
| Expose types to Python | `py_*.h` | One `export_<Thing>(m, "Name")` per wrapper |
| Implement non-Newtonian models | `nonNewtonian.h` | Viscosity update and map-check kernels |

## CONVENTIONS

- **`core.h` is the umbrella include** for the default D3Q27 model;
  include it after defining `AA_PATTERN` or `AB_PATTERN`.
- **`.hpp` files are private implementations** included from their `.h` counterpart;
  never include them directly.
- **New lattice models live in mirrored subpackages** (`d3q27/` / `d3q7/`)
  with matching `col_`, `eq_`, `streaming_`, `bc`, `macro`, and `common` files.
- **`LBM_CONFIG`** is the compile-time policy bundle;
  concrete simulations specialize it to pick collision, streaming, macroscopic output, and data layout.
- **`LBM_Data`** is a base carrier for kernel data;
  concrete models extend it with Q-specific arrays.
- **Python export headers** are named `py_<topic>.h` and expose one templated `export_<Thing>` function per class.

## ANTI-PATTERNS

- **Including `d3q27/*.h` directly** instead of `core.h` unless you are building a custom lattice model.
- **Treating `LBM_Data` as a concrete model**;
  it is a base type that needs a model-specific subclass.
- **Hard-coding `DFMAX` or distribution-function indices**;
  use `df_cur`, `df_out`, and `df_prev`.
- **Mixing physical and lattice units outside `Lattice`**;
  use `phys2lbmPoint`, `lbm2physPoint`, and the viscosity helpers.
