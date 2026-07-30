# AMR for LBM — Implementation Notes

**Status:** documents the TNL-LBM AMR implementation as of commit `5237b2f` (branch `agents`).
**Scope:** static, cell-centered, volumetric coupling (Rohde 2006 / Chen 1998 / Guzik 2014), 2:1 refinement ratio, single-GPU single-rank, D3Q27 lattice.

---

## 1. Approach

Cell-centered volumetric AMR for LBM: distribution functions (DFs) are point densities at cell centers; the 1/8 Lagrava per-direction averaging in the fine-to-coarse direction IS the volumetric conversion (no separate volume factor appears anywhere). Refinement is static (regions fixed at SimInit, no dynamic adaptation). The coarse lattice covers the entire domain; fine blocks overlay subregions at double resolution.

Physical scaling is convective (dt ∝ dx): lattice velocity u_lb is identical on all levels; lattice viscosity doubles per level (ν_lb,f = 2 ν_lb,c); relaxation time τ = 3 ν_lb + 0.5 differs between levels (τ_f ≠ τ_c).

---

## 2. Data architecture

### 2.1 Per-block storage (`include/lbm3d/lbm_block.h`, `.hpp`)

`LBM_BLOCK<CONFIG>` holds:
- DistributedNDArray storage for DFs (27 directions × DFMAX arrays), map (`dmap`/`hmap`), macros (`dmacro`/`hmacro`).
- `global`, `local`, `offset` extents; `level` (0 = coarsest).
- `lat_local` per-level lattice, initialized by `initLevelLattice(base_lat, level)`:
  - `physDl = base_physDl / 2^level`, `physDt = base_physDt / 2^level`, `physViscosity` level-independent.
  - `lbmViscosity = physViscosity · physDt / physDl²` — doubles per level.
- `storage_overlap` member: defaults to `overlap_width` (1 with MPI, 0 without); set to 2 for `level > 0` in `initLevelLattice` (commit `089e47a`). All per-cell arrays (DF, map, macro) share one `data.indexer` (derived from `dmap`), so they must use the same overlap width. The allocation only materializes overlap on axes where `local != global` (fine blocks always satisfy this since their footprint is strictly smaller than the refined global extent). **Note:** all overlap `setSize` + `allocate()` calls are inside `#ifdef HAVE_MPI`; non-MPI builds use plain NDArrays with no overlap support. AMR v1 effectively requires an MPI-enabled single-rank (nproc=1) build.

### 2.2 Multi-level LBM (`include/lbm3d/lbm.h`, `.hpp`)

`LBM<CONFIG>` holds a flat `blocks` vector with per-level bookkeeping (`level_block_counts`, `max_level`). `getBlocksAtLevel(L)` returns pointers to blocks at level L. `updateKernelDataForLevel(L, substep)` toggles the fine level's parity/rotation for the given substep (AA: `even_iter = (substep % 2) == 1`; AB: DF pointer rotation) and restores the per-level `lbmViscosity` that the global `updateKernelData()` clobbers from the level-0 lattice. `updateKernelData()` (global, driven by `iterations` counter) sets parity/rotation for **all** blocks; fine levels are subsequently overridden by `updateKernelDataForLevel(L, substep)` before each kernel launch.

### 2.3 Block creation (`include/lbm3d/amr_decomposition.h`)

`createAMRBlocks(lbm, regions)` takes a pre-parsed `std::vector<AMR_Region<CONFIG>>` (parsing is done by `parseAMRConfig` separately), validates each region, constructs fine `LBM_BLOCK` instances with the level-aware constructor, sets `global_offset` in parent-level coordinates, allocates host/device data, resets map to `GEO_FLUID`, and initializes DFs to equilibrium.

`markAMRInterface(lbm)` tags coarse cells adjacent to each fine footprint (the 1-cell halo ring, 34³−32³ cells for a 32³ footprint) as `GEO_AMR_INTERFACE`. Also allocates `dinterface_dir` bitmask storage for blocks owning interface cells (vestigial in v1 — `getInterfaceDir` has no callers; reserved for future directional coupling).

---

## 3. Coupling structure

### 3.1 Patch construction (`State_AMR::buildCouplings`, `amr_state.h`)

For each fine block at level L and each of the six faces of its footprint's 1-cell halo box (disjoint partition: x-faces span full y/z halo, y-faces interior x-range, z-faces interior x/y range), `buildCouplings` clips the face rectangle against every level-(L−1) block's range. A patch is appended iff it contains at least one `GEO_AMR_INTERFACE` cell not shadowed by another same-level block.

Each `AMR_InterfacePatch` stores:
- `coarse_origin` / `coarse_size` — the parent-level halo rectangle (1 cell thick in face normal).
- `fine_origin` / `fine_size` — the matching fine-level ghost rectangle (2 fine cells per coarse cell, 2 cells thick: outer layer feeds the F2C filter, inner layer is the ghost layer read by fine-level streaming).
- `face` — the interface normal direction.

### 3.2 Launch helpers (`amr_state.h`)

`launchCoarseToFineTransfers(L)` iterates patches for level L, clips the fine rectangle to the fine block's allocated overlap (`df_overlap_X/Y/Z()` per-axis from the indexer), and launches `cudaAMR_CoarseToFine` per patch.

`launchFineToCoarseTransfers(L)` iterates the same patches, launches `cudaAMR_FineToCoarse` over the FULL patch coarse rectangle (non-storable cells are skipped per-cell inside the kernel). The storability guard uses per-axis allocated overlap (`idx3d ov`) passed as a kernel parameter.

---

## 4. Coupling kernels (`include/lbm3d/d3q27/amr_coupling.h`)

### 4.1 Coarse-to-fine (`cudaAMR_CoarseToFine`)

Fills fine ghost cells from coarse data:

1. **Trilinear macro interpolation**: for each fine ghost cell at global fine coordinate `fg`, compute the home coarse cell `home = floor_div(fg, 2)` (true floor division via `fdiv2`). Bracketing corners: `{home−1+(fg&1), home+(fg&1)}` per axis. Weights: 3/4 on the home side, 1/4 on the far side (exact for linear fields; binary fractions so constant fields are preserved exactly).
2. **Equilibrium re-evaluation**: `COLL::setEquilibrium(KS_F)` at the interpolated (rho, u) — one `EQ::eq_*` call per direction, never shared.
3. **Non-equilibrium interpolation + rescale**: per direction, trilinearly interpolate `f_neq = f_corner − eq_corner(rho_c, u_c)` from the 8 coarse corners, rescale by `τ_f / τ_c` (the non-equilibrium rescaling factor; with 2:1 refinement τ_f ≠ τ_c since ν_lb doubles per level). Result: `f_fine = eq_f + (τ_f/τ_c) · f_neq_interp`.
4. **Write** to `fine_SD.df(df_cur, ...)` in the storage-parity expected by the upcoming fine substep (AB: natural orientation; AA: twisted — direction q stored in `opposite_direction(q)` slot because the spatial/odd substep pulls from the opposite slot).
5. **Macro write guard**: if `fine_SD.map(x,y,z) == GEO_AMR_INTERFACE`, write interpolated macros to `dmacro` (no-op in v1 — fine ghosts are never tagged).

### 4.2 Fine-to-coarse (`cudaAMR_FineToCoarse`)

Projects filtered fine data onto coarse interface ring cells:

1. **Storability guard** (per cell, per axis): all 8 fine subcells must be within the fine block's overlap-extended storage `[-ov_i, fine_local_i + ov_i)`. Cells failing the guard are skipped individually (commit `26e36db` replaced a launch-extent clip that evaluated storability in the wrong origin-aligned frame). With `storage_overlap = 2` on fine blocks (commit `089e47a`), all ring cells are storable.
2. **Lagrava filter** (mandatory): per-direction arithmetic 1/8 average of the 8 fine subcells covered by the coarse ring cell. The 1/8 factor IS the volumetric fine-to-coarse conversion — suppresses unresolved high-frequency fine modes before projection (Lagrava et al. 2012).
3. **Macro recompute + equilibrium**: `COLL::computeDensityAndVelocity(KS)` on the averaged DFs; `COLL::setEquilibrium(KS_EQ)` at those macros.
4. **Non-equilibrium rescale**: `f_coarse = eq_c + (τ_c/τ_f) · (f_avg − eq_c)` — the reciprocal of the C2F factor.
5. **Write** to the coarse DF state the NEXT coarse substep will consume (AB: logical `df_out`; next `updateKernelData()` rotation makes it `df_cur`; AA: natural if next substep is even/reflect, twisted if odd/spatial).
6. **Macro write**: for `GEO_AMR_INTERFACE` cells, write filtered macros to `dmacro` (authoritative coupling value for output; the main kernel also recomputes macros for these collision-active cells every step, but the transfer's write is the output-relevant one until the next coarse step).

### 4.3 Global-frame indexing

All coupling kernel coordinate arithmetic is computed in the GLOBAL frame (per axis): fine global `fg = x + fine_off`; home coarse `home = fdiv2(fg)`; brackets converted back to the coarse block's indexer frame via `− coarse_off`. `fine_off` and `coarse_off` are the blocks' indexer origins in global coordinates of their level (`LBM_BLOCK::offset`). True floor division (`fdiv2`) handles negative fine ghost indices correctly (commit `26e36db`).

### 4.4 Streaming-pattern handling

The kernels receive parity parameters (`coarse_even_iter`, `fine_even_iter`) from the caller:

- **AB pattern**: `read_coarse_df` reads `df_out` (post-collision, natural orientation); `read_fine_df` reads `df_out`; F2C writes to logical `df_out`. The next global `updateKernelData()` rotates frames, so the physical array written becomes `df_cur` for the next consuming kernel.
- **AA pattern**: twisted storage — post-collision state stores direction q in slot `opposite_direction(q)` (even/reflect substep); post-stream state is natural (odd/spatial substep). `coarse_even_iter == true` → read `df_cur[opposite(q)]` (twisted); `false` → read `df_cur[q]` (natural). The F2C write orientation is chosen for the NEXT consuming coarse substep parity: `next_coarse_even_iter = (level==0) ? ((iterations % 2) == 1) : false`.

`updateKernelDataForLevel(L, substep)` is called BEFORE each fill so the fine block's `data.dfs` pointers / `even_iter` are set for the upcoming substep's consumption frame.

---

## 5. Time stepping: Berger-Colella schedule (`State_AMR::SimUpdate`)

Per global iteration (= one coarse step, physDt advance = physDt_coarse):

1. `computeBeforeLBMKernel()` hook; `nse.iterations++`.
2. **Coarse step**: `launchLBMKernelForLevel(0, compute_macro)` — all level-0 blocks stream + collide (including `GEO_AMR_INTERFACE` ring cells, collision-active since commit `5237b2f`).
3. **Per finer level L** (1 to `max_level`):
    1. `updateKernelDataForLevel(L, 0)` — set fine parity for substep 0.
    2. `launchCoarseToFineTransfers(L)` — fill fine ghost ring from coarse post-step state.
    3. `launchLBMKernelForLevel(L, compute_macro)` — fine substep 1 (fine advances physDt_coarse/2).
    4. `updateKernelDataForLevel(L, 1)` — set fine parity for substep 1.
    5. `launchCoarseToFineTransfers(L)` — BVP re-fill (ghost ring consumed by substep 1's streaming).
    6. `launchLBMKernelForLevel(L, compute_macro)` — fine substep 2 (fine advances to physDt_coarse).
    7. `launchFineToCoarseTransfers(L)` — project Lagrava-filtered fine state onto coarse ring cells.

Physical time consistency: coarse advances physDt_coarse per cycle; fine advances 2 × physDt_coarse/2 = physDt_coarse. Verified exact in `test_amr_subcycling` Test 2.

**Known temporal gap (intentionally unfixed)**: both C2F fills use the coarse post-step state (t_{n+1}); the first fine substep (advancing t_n → t_{n+1/2}) receives boundary data from a full coarse step ahead. Time interpolation between coarse states (the "H9" variant) was tested and measured to make violations **+13.4% worse** (variant v4 in the falsification matrix) — it is intentionally NOT included.

---

## 6. Interface ring handling (commit `5237b2f`)

`GEO_AMR_INTERFACE` cells are **collision-active**:
- `D3Q27_BC_All::preCollision` — no early return; ring cells proceed through normal streaming.
- `doCollision` — returns `true`; ring cells collide like fluid.
- `postCollision` — no early return; `postCollisionStreaming` writes DFs back.

The F2C kernel still overwrites their DFs and macros at the end of each coarse step (competitive/coupled overwrite). This replaced the earlier collision-inactive design where `preCollision` wrote a `rho=1, v=0` placeholder and the kernel skipped streaming+collision — which made the ring a stiff boundary layer imposing an O(Ma²) pressure offset with no viscous decay channel.

---

## 7. Output

### 7.1 VTKHDF OverlappingAMR writer (`include/lbm3d/viz/OverlappingAMRWriter.{h,hpp}`)

Flat per-level layout: Level0 = whole coarse lattice, Level1.. = fine footprints. Global `Origin` attribute + per-level `Spacing`. `vtkGhostType` marks under-footprint coarse cells as `REFINEDCELL` (bit 0x4) so ParaView blanks them under the fine data. `CellData` contains `rho`, `vx`, `vy`, `vz`, `vtkGhostType` per level.

### 7.2 ParaView end-to-end test (`tests/test_amr_paraview_e2e.py`)

Runs under pvpython: opens the VTKHDF file, asserts OverlappingAMR structure (level counts, field availability, ghost-type distributions), checks field statistics (rho ≈ 1.0, |vx| < V_max), renders a z-midplane slice to PNG, verifies the screenshot has ≥64 unique colors and ≥30 KB.

---

## 8. Verification infrastructure

### 8.1 Unit tests (`tests/run-amr-tests.sh`)

| Target | Tests |
|---|---|
| `test_amr_coupling_{ab,aa}` | 6 mock tests: uniform-field C2F (2 parities), uniform-field F2C (4 parity combos), linear-gradient C2F exactness, mass-conservation F2C (quadratic field), mass-conservation C2F (linear field), nested-geometry geography regression with halo/Lagrava/storability sub-checks (commit `30139d4`) |
| `test_amr_subcycling_{ab,aa}` | 4 tests: substep counting (1 coarse + 2 fine kernel preparations per cycle), time synchronization (t_coarse == t_fine), max_level==0 bitwise-identical-to-base-driver, interface-ring freshness (Test 4: no ring cell holds placeholder after one cycle) |
| `test_amr_vtkhdf_writer_{ab,aa}` | VTKHDF file structure, ghost-type tagging, field values |
| `test_amr_paraview_e2e` | ParaView >= 6.0 fetch + field stats + slice render (6.1 in the test environment; skipped with exit 77 if pvpython absent) |

Current status: **7/7 targets pass** on both AA and AB streaming patterns.

### 8.2 Between-property metric (`/tmp/opencode/amr-debug/between_metric.py`)

Compares every cell of the AMR composite against a bracket formed by independent uniform-coarse (64³) and uniform-fine (128³) reference runs (γ=3, eps_rho=1e-5, eps_vel=2e-5). Reports per-cycle total/max violations, per-face slab breakdowns, and top-100 violator coordinates. Self-test mode (substitutes AMR = fine reference) passes trivially, confirming harness consistency.

### 8.3 Conservation monitoring

`State_AMR::AfterSimUpdate` logs `"AMR conservation: mass = {total_mass}"` at the PRINT interval. Mass is computed as a volume-weighted host-side reduction over all blocks. Current measurement: mass = 2.949119e+05 invariant across all 11 outputs (exact conservation).

---

## 9. Measured residual and falsification matrix

### 9.1 Current acceptance state

- **Unit suite**: 7/7 green (both patterns).
- **sim_AMR Taylor-Green** (64³ coarse + 32³-coarse-footprint fine, Re=100, 10 coarse cycles):
  - between-metric cycle 10: **685,713 violations** (rho 463,276 max 2.66e-4; vx 19,898 max 2.52e-4; vy 19,824 max 2.56e-4; vz 182,715 max 2.74e-4).
  - 0 frozen placeholder cells; mass exactly conserved.
  - Residual character: smooth bulk drift (~3e-4 rho, ~2.5e-4 velocity) distributed across the domain, concentrated in the fine-boundary band (2–4 fine cells inside the interface); no localized spikes or frozen cells.

### 9.2 Root cause of residual: one-way clamp

The F2C kernel reads fine **ghost** cells, which were C2F-filled from coarse interpolation. The ring values are therefore a `(1/8, 3/4, 1/8)³` binomial smoothing of coarse neighborhood data — no fine-interior information ever reaches the coarse lattice. The fine patch is a one-way clamp on a coarser boundary stencil. Velocity errors diffuse away (momentum has a viscous decay channel); rho offsets do not (pressure offsets have no decay channel in the one-way scheme).

### 9.3 Variant falsification matrix (bit-exact control)

Full out-of-tree builds with the variant applied; same sim_AMR run; same between-metric harness. Control reproduces baseline bit-exactly.

| Variant | Description | Cycle-10 total violations | Delta | Verdict |
|---|---|---|---|---|
| baseline | current code (post-`089e47a`) | 1,265,680 | — | — |
| v1 | τ-rescale halved (`τ_f/τ_c` → `(τ_f−0.5)/(τ_c−0.5)·(dt_f/dt_c)`) | 1,261,505 | −0.3% | **falsified** (noise) |
| v2 | zero-out non-equilibrium rescale | 1,309,989 | +3.5% | worse |
| v3 | ring-DF freshness parity alternation | 1,503,302 | +18.9% | worse |
| v4 | time-centered first fill (H9 midpoint interpolation) | 1,435,570 | +13.4% | **falsified** |
| v6 | naive inside-shell two-way (F2C from footprint-underneath, no stabilization) | unstable (rho max 0.96 @ cyc1, NaN @ cyc2) | — | **unstable** |
| **v7** | **collision-active ring** (bc.h: `doCollision` → true, remove placeholder skips) | **685,713** | **−45.8%** | **committed (`5237b2f`)** |
| v8 | deeper F2C reach (into footprint cells) | 2,139,397 | +69% | worse |

### 9.4 Pre-fix trajectory

| Milestone | Commit | Cycle-10 violations | Key change |
|---|---|---|---|
| Geography corruption | `26e36db` | ~844k (pre-overlap) | Coupling kernel frame alignment, fill ordering, AA twisted writes |
| Frozen placeholder ring | `089e47a` | 1,265,680 | `storage_overlap=2` on fine blocks; F2C covers all ring cells |
| Collision-active ring | `5237b2f` | **685,713** | Ring streams+collides like fluid; F2C overwrites at end of step |

---

## 10. Known limitations of v1

- **One-way information flow**: no fine-interior data feeds back into the coarse lattice (the residual root cause; see §9.2).
- **Static refinement only**: regions fixed at SimInit; no dynamic adaptation.
- **Single-rank, single-GPU**: fine blocks have no same-level MPI neighbors; coupling kernels are CUDA-only.
- **Multi-level (level > 1)**: `createAMRBlocks` hard-rejects `level > 1` during validation (throws at `amr_decomposition.h:187-188`); deeper nesting is impossible by construction in v1, not merely untested.
- **Non-Newtonian viscosity with AMR**: excluded.
- **IBM (immersed boundary)**: excluded.

---

## 11. Code references

| File | Role |
|---|---|
| `include/lbm3d/d3q27/amr_coupling.h` | Coupling kernels: `cudaAMR_CoarseToFine`, `cudaAMR_FineToCoarse` |
| `include/lbm3d/amr_state.h` | `State_AMR` driver: `SimInit`, `SimUpdate`, `buildCouplings`, `launchCoarseToFineTransfers`, `launchFineToCoarseTransfers`, `launchLBMKernelForLevel` |
| `include/lbm3d/amr_decomposition.h` | `createAMRBlocks`, `markAMRInterface`, `allocateInterfaceDirArray` |
| `include/lbm3d/d3q27/bc.h` | `D3Q27_BC_All`: `GEO_AMR_INTERFACE` tag, `preCollision`/`doCollision`/`postCollision` collision-active handling |
| `include/lbm3d/lbm_block.h` / `.hpp` | `LBM_BLOCK`: `storage_overlap`, `initLevelLattice`, `allocateHostData`/`allocateDeviceData` |
| `include/lbm3d/lbm.h` / `.hpp` | `LBM`: multi-level block management, `updateKernelDataForLevel` |
| `include/lbm3d/viz/OverlappingAMRWriter.h` / `.hpp` | VTKHDF OverlappingAMR output |
| `include/lbm3d/viz/OverlappingAMRWriter.hpp` | Writer implementation: per-level AMRBox, ghost tagging, CellData layout |
| `sim_AMR/sim_AMR.cu` | Taylor-Green benchmark with 2-level AMR |
| `tests/test_amr_coupling.cu` | Mock-block coupling kernel unit tests |
| `tests/test_amr_subcycling.cu` | State_AMR end-to-end subcycling tests (incl. Test 4 ring freshness) |
| `tests/test_amr_vtkhdf_writer.cu` | VTKHDF writer unit test |
| `tests/test_amr_paraview_e2e.py` / `.sh` | ParaView end-to-end visualization test |
| `tests/run-amr-tests.sh` | Test runner (builds + runs all 7 targets) |

### Commits in this debugging arc

| Commit | Type | Description |
|---|---|---|
| `30139d4` | `test` | Coupling geography regression test (fails on baseline) |
| `26e36db` | `fix` | Coupling frame alignment, fill ordering, AA twisted writes |
| `089e47a` | `fix` | 2-cell ghost ring on fine blocks so F2C covers the interface ring |
| `bd227d1` | `docs` | Regenerated Taylor-Green slice render |
| `5237b2f` | `fix` | Collision-active GEO_AMR_INTERFACE ring (−45.8% violations) |
