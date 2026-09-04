# D3Q27 LATTICE MODEL KNOWLEDGE BASE

**Updated:** 2026-09-04
**Branch:** fix/well-conditioning

## OVERVIEW

D3Q27 lattice model kernels:
collision operators,
equilibrium functions,
streaming patterns,
boundary conditions,
and macroscopic output for the 3D Navier-Stokes solver.

## STRUCTURE

```
.
├── col_*.h              # Collision operators (CUM, BGK, CLBM, MRT, SRT, KBC, adjoint variants)
├── eq_*.h               # Equilibrium distributions (standard, well, entropic, adjoint, inverse-cumulant)
├── streaming_AA.h       # A-A pattern streaming (single lattice, lower memory)
├── streaming_AB.h       # A-B pattern streaming (two-lattice swap)
├── bc.h                 # Boundary condition dispatch over GEO enum
├── macro.h              # Macroscopic quantity output and forcing hooks
├── common.h             # Base mixin: density/velocity/equilibrium; WELL_CONDITIONED param shifts density by +1
└── common_adjoint.h     # Base mixin for adjoint sensitivity problems
```

## WHERE TO LOOK

| Task | Location | Notes |
|------|----------|-------|
| Add a collision operator | `col_*.h` | Inherit from `D3Q27_COMMON<TRAITS, EQ>` or `D3Q27_COMMON_WELL<TRAITS, EQ>`; set a static `id` string |
| Add a boundary condition | `bc.h` | Extend `D3Q27_BC_All::GEO` and add a handler in `preCollision` / `postCollision` |
| Switch streaming pattern | `streaming_AA.h` / `streaming_AB.h` | Define `AA_PATTERN` or `AB_PATTERN` before including `lbm3d/core.h` |
| Customize macro output | `macro.h` | Inherit from `D3Q27_MACRO_Base`; `outputMacro` writes to `SD.macro` |
| Change equilibrium | `eq_*.h` | Each file provides `eq_<dir>(rho, vx, vy, vz, ...)` for all 27 directions |

## CONVENTIONS

- **`col_*`** = collision operators;
  well-conditioned variants use the `D3Q27_COMMON_WELL` alias (`D3Q27_COMMON<TRAITS, EQ, true>`) and shift density by one.
- **`D3Q27_COMMON_WELL`** = alias of `D3Q27_COMMON` with `WELL_CONDITIONED = true`;
  stored DFs are `f - w` with `sum(w) = 1` and `fromStorage`/`toStorage` convert by `+/-w_i`.
- **`eq_*`** = equilibrium distribution functions;
  called direction-by-direction by `setEquilibrium`.
- **`streaming_*`** = streaming implementations;
  `AA_PATTERN` stores to same site/opposite direction on even iterations.
- **`common*.h`** = base mixins providing `computeDensityAndVelocity`, `setEquilibrium`, and `setEquilibriumLat`.
- Inheritance:
  derive standard operators from `D3Q27_COMMON<TRAITS, EQ>`,
  well-conditioned formulations from the `D3Q27_COMMON_WELL` alias,
  and adjoint problems from `D3Q27_COMMON_ADJOINT`.
- Collision operators expose a `static constexpr const char* id` used for logging and configuration.
- Direction naming follows `p`/`z`/`m` for +1/0/-1 along x/y/z (e.g., `ppp`, `zmz`, `mmm`).

## ANTI-PATTERNS

- Mixing `*_well` collision operators with a non-well equilibrium:
  density shift must stay consistent.
- Calling `EQ::eq_*` with wrong argument count:
  adjoint equilibria take both forward and adjoint macroscopic variables.
- Hard-coding 27 directions instead of using the `ppp`/`zzz`/`mmm` constants from `lbm_common/ciselnik.h`.
