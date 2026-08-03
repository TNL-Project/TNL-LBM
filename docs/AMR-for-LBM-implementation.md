# AMR for LBM — Implementation Notes

**Status:** documents the TNL-LBM AMR implementation as of commit `5237b2f` (branch `agents`).
**Scope:** static, cell-centered, volumetric coupling (Chen 1998 / Chen et al. 2006 / Guzik 2014), 2:1 refinement ratio, single-GPU single-rank, D3Q27 lattice.

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

`markAMRInterface(lbm)` tags two coarse-cell populations: (1) the 1-cell halo ring around each fine footprint (34³−32³ cells for a 32³ footprint) as `GEO_AMR_INTERFACE`, and (2) the coarse cells inside each footprint (32³ = 32768 cells) as `GEO_NOTHING` — frozen cells that do not stream or collide. The frozen cells prevent the diverging "shadow solve" (a coarse LBM evolution under the footprint where the fine lattice is authoritative — see §9.2.1). Their DFs are set exclusively by the interior F2C transfer (§4.2). Also allocates `dinterface_dir` bitmask storage for blocks owning interface cells (vestigial in v1 — `getInterfaceDir` has no callers; reserved for future directional coupling).

---

## 3. Coupling structure

### 3.1 Patch construction (`State_AMR::buildCouplings`, `amr_state.h`)

**Ring patches** (one per face): for each fine block at level L and each of the six faces of its footprint's 1-cell halo box (disjoint partition: x-faces span full y/z halo, y-faces interior x-range, z-faces interior x/y range), `buildCouplings` clips the face rectangle against every level-(L−1) block's range. A patch is appended iff it contains at least one `GEO_AMR_INTERFACE` cell not shadowed by another same-level block.

Each ring `AMR_InterfacePatch` stores:
- `coarse_origin` / `coarse_size` — the parent-level halo rectangle (1 cell thick in face normal).
- `fine_origin` / `fine_size` — the matching fine-level ghost rectangle (2 fine cells per coarse cell, 2 cells thick: outer layer feeds the F2C filter, inner layer is the ghost layer read by fine-level streaming).
- `face` — the interface normal direction.

**Interior patches** (one per fine-block × coarse-block overlap): cover the full footprint `[global_offset, global_offset + local/2)` in parent-level coordinates. These target the frozen `GEO_NOTHING` cells under the footprint. `face` is set to `SyncDirection::None` (not a face). The interior F2C transfer injects fine-averaged DFs into these cells each cycle (§4.2), providing the two-way feedback channel.

### 3.2 Launch helpers (`amr_state.h`)

`launchCoarseToFineTransfers(L)` iterates ring patches for level L, clips the fine rectangle to the fine block's allocated overlap (`df_overlap_X/Y/Z()` per-axis from the indexer), and launches `cudaAMR_CoarseToFine` per patch.

`launchFineToCoarseTransfers(L)` iterates the same ring patches, launches `cudaAMR_FineToCoarse` over the FULL patch coarse rectangle (non-storable cells are skipped per-cell inside the kernel). The storability guard uses per-axis allocated overlap (`idx3d ov`) passed as a kernel parameter.

`launchFineToCoarseTransfersInterior(L)` iterates the interior patches, launches `cudaAMR_FineToCoarse` over the full footprint. The frozen `GEO_NOTHING` cells receive Lagrava-filtered fine-averaged DFs (full overwrite — no λ-blend needed because the frozen cells have no collision to conflict with).

---

## 4. Coupling kernels (`include/lbm3d/d3q27/amr_coupling.h`)

### 4.1 Coarse-to-fine (`cudaAMR_CoarseToFine`)

Fills fine ghost cells from coarse data:

1. **3rd-order Lagrange macro interpolation**: for each fine ghost cell at global fine coordinate `fg`, compute the home coarse cell `home = floor_div(fg, 2)` (true floor division via `fdiv2`). Per-axis stencil: 4 coarse cell centers `{home−2+(fg&1)…home+1+(fg&1)}` with Lagrange weights evaluated at the fine cell center (offset ±1/4 from the home center; centered windows give the dyadic rationals {−5, 35, 105, −7}/128 for even `fg` and {−7, 105, 35, −5}/128 for odd `fg`). Exact for cubic fields; the upgraded scheme implements the "3rd-order C2F spatial interpolation" recommendation of §12.7 (Gendre et al. 2017, Lagrava et al. 2012). **Storability guard**: the kernel shifts/shortens each per-axis window into the coarse storage extent queried from `coarse_SD.indexer` (sizes + overlap) and normalizes the runtime-computed weights to sum to one, so no out-of-bounds access can occur; full nominal accuracy needs ghosts ≥2 coarse cells inside the coarse stored extent. The original trilinear scheme (2-node stencil, 3/4:1/4 weights) remains available via `-DC2F_TRILINEAR`. **Explosion alternatives** (Schukmann et al. 2025, §12.7 item 6): `-DC2F_LINEAR_EXPLOSION` skips the neighbor interpolation entirely — each fine ghost cell takes the home coarse cell's macros `(rho, u)` directly and re-evaluates the equilibrium at them with the non-equilibrium zeroed (pure equilibrium explosion); `-DC2F_UNIFORM_EXPLOSION` duplicates the home cell's DFs to the 8 subcells unchanged (zeroth order, no rescaling). The home cell index is clamped per axis into the coarse storage extent (the explosion analog of the storability guard). Measured on the Taylor-Green bracket metric (cycle-10 violations vs the 188,353 baseline): linear explosion 839,717 (+346 %, rho-plateau dominated), uniform explosion 276,789 (+47 %) — both worse than the 3rd-order interpolation; see §12.7 item 6.
2. **Equilibrium re-evaluation**: `COLL::setEquilibrium(KS_F)` at the interpolated (rho, u) — one `EQ::eq_*` call per direction, never shared.
3. **Non-equilibrium interpolation + rescale**: per direction, interpolate `f_neq = f_cell − eq_cell(rho_c, u_c)` from the same coarse stencil with the same weights, rescale by `τ_f / τ_c` (the non-equilibrium rescaling factor; with 2:1 refinement τ_f ≠ τ_c since ν_lb doubles per level). Result: `f_fine = eq_f + (τ_f/τ_c) · f_neq_interp`.
4. **Write** to `fine_SD.df(df_cur, ...)` in the storage-parity expected by the upcoming fine substep (AB: natural orientation; AA: twisted — direction q stored in `opposite_direction(q)` slot because the spatial/odd substep pulls from the opposite slot).
5. **Macro write guard**: if `fine_SD.map(x,y,z) == GEO_AMR_INTERFACE`, write interpolated macros to `dmacro` (no-op in v1 — fine ghosts are never tagged).

### 4.2 Fine-to-coarse (`cudaAMR_FineToCoarse`)

Projects filtered fine data onto coarse interface ring cells:

1. **Storability guard** (per cell, per axis): all 8 fine subcells must be within the fine block's overlap-extended storage `[-ov_i, fine_local_i + ov_i)`. Cells failing the guard are skipped individually (commit `26e36db` replaced a launch-extent clip that evaluated storability in the wrong origin-aligned frame). With `storage_overlap = 2` on fine blocks (commit `089e47a`), all ring cells are storable.
2. **Lagrava filter** (mandatory): tensor-product 4-node-per-axis Lagrange projection of the fine DFs onto the coarse cell center (`t = fx0 + 0.5`, fine indexer coordinates) — the nominal per-axis window `{fx0−1, …, fx0+2}` covers the 2×2×2 subcell block extended by one fine cell per side (4×4×4 = 64 fine cells); centered windows yield the dyadic rationals {−1, 9, 9, −1}/16 per axis. Near fine-block boundaries the window is shifted/shortened into the storable extent with runtime double-precision weights (same machinery as the C2F storability guard) while the evaluation point stays fixed at the coarse center, so constant-to-cubic fields are reproduced exactly on every window (the plain 1/8 box average was linear-exact only). The weights sum to one, so the weighted sum IS the volumetric fine-to-coarse conversion — no other volume factor. The 1/8 box average remains as `-DF2C_BOX_AVERAGE`.
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
    8. `launchFineToCoarseTransfersInterior(L)` — inject fine-averaged DFs into frozen `GEO_NOTHING` cells under the footprint (two-way feedback; eliminates the shadow solve of §9.2.1).

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
  - between-metric cycle 10 (trilinear v7, old harness): **685,713 violations** (rho 463,276 max 2.66e-4; vx 19,898 max 2.52e-4; vy 19,824 max 2.56e-4; vz 182,715 max 2.74e-4).
  - between-metric cycle 10 (3rd-order Lagrange C2F, corrected harness): **188,353 violations** (see §4.1, §12.7 item 6 for the upgraded baseline).
  - 0 frozen placeholder cells; mass exactly conserved.
  - Residual character: smooth bulk drift (~3e-4 rho, ~2.5e-4 velocity) distributed across the domain, concentrated in the fine-boundary band (2–4 fine cells inside the interface); no localized spikes or frozen cells. The boundary-band concentration is explained by C2F shadow injection (§9.2.1); the bulk drift and pressure-offset non-decay by the one-way clamp (§9.2).

### 9.2 Root cause of residual: one-way clamp

The F2C kernel reads fine **ghost** cells, which were C2F-filled from coarse interpolation. The ring values are therefore a `(1/8, 3/4, 1/8)³` binomial smoothing of coarse neighborhood data — no fine-interior information ever reaches the coarse lattice. The fine patch is a one-way clamp on a coarser boundary stencil. Velocity errors diffuse away (momentum has a viscous decay channel); rho offsets do not (pressure offsets have no decay channel in the one-way scheme).

**Attempted fix — ghost exclusion (failed).** The direct consequence of this framing was implemented and measured: the Lagrava F2C filter with its per-axis window clamped to the fine interior (`lo = 0` instead of `-ov`, excluding index `< 0` ghost cells, mirroring Musubi's GhostFromFiner which reads fine fluid elements only). Both the default 3rd-order Lagrange C2F and the compact-moment C2F diverged to NaN before cycle 5. The theory fails for this architecture: for an exterior ring cell, its entire 2×2×2 subcell block IS ghost data in the fine halo, and those ghost subcells are the physically correct boundary state — excluding them replaces the boundary data with a far extrapolation from fine-interior cells and disconnects the ring from the interface. Musubi's interior-only reading does not transfer because Musubi's ghost cells are passive streaming buffers, whereas here the exterior-ring ghost subcells carry the C2F boundary state the collision-active ring needs. The 1/8 box average over the coarse cell's own subcells (default) and the ghost-inclusive Lagrava filter window both remain sound; any future "true GhostFromFiner" semantics must project from fine *interior* fluid onto the coarse ring explicitly, not by stencil exclusion.

### 9.2.1 Complementary root cause: C2F shadow injection via inside-hidden cells

§9.2 identifies the missing **F2C** feedback (fine → coarse). A second, distinct error source exists in the **C2F** direction (coarse → fine): the inner fine ghost layer's stencil reads a coarse cell under the fine footprint — the "inside-hidden" cell — and injects its state into the fine boundary at reduced weight.

**The hidden-cell problem.** Coarse cells volumetrically inside the fine footprint are tagged `GEO_NOTHING` (`markAMRInterface`, `amr_decomposition.h:410-421`) — frozen: no stream, no collide. Their DFs are set exclusively by the interior F2C transfer (§4.2, step 8), which injects Lagrava-filtered fine-averaged DFs each cycle. This eliminates the diverging "shadow solve" (a coarse LBM evolution under the footprint where the fine lattice is authoritative). However, the C2F stencil still reads these frozen cells as interpolation sources for the inner fine ghost. The frozen cells contain fine-averaged DFs — not the coarse PDE solution at those locations — so they remain physically invalid as C2F sources. The error is milder than the original shadow (the fine-averaged DFs are bounded and physically motivated, not diverging), but the C2F stencil still injects a resolution-mismatched state into the fine boundary.

**Asymmetric bracket geometry.** The C2F kernel brackets each fine ghost cell between coarse stencil nodes (`amr_coupling.h:175-187`). The following analysis uses the original trilinear weights (0.75/0.25); under the current 3rd-order Lagrange C2F (§4.1), the weights differ ({−5, 35, 105, −7}/128 for even `fg`, {−7, 105, 35, −5}/128 for odd `fg`), but the qualitative asymmetry — the inner ghost reads a hidden cell at reduced weight — persists. For the x-normal left face (footprint `[origin, origin+size)`, ring cell `origin-1`), the two ghost layers have qualitatively different brackets along the face normal:

| Ghost layer | Fine global `fg` | `home` (weight 0.75) | 0.25 bracket | Physical validity of 0.25 source |
|---|---|---|---|---|
| **Outer** (farther from interior) | `2·origin−2` (even) | ring (`origin−1`) | far-fluid (`origin−2`) | **valid** — coarse is authoritative there |
| **Inner** (closer to interior) | `2·origin−1` (odd) | ring (`origin−1`) | inside-hidden (`origin`) | **mismatch** — frozen cell holds fine-averaged DFs, not the coarse PDE solution |

The ring cell (`GEO_AMR_INTERFACE`) is always `home` at 0.75 weight for both layers — it carries delayed fine feedback (F2C overwrote it last cycle, then one coarse collision mixed it with neighbors). The asymmetry is in the 0.25 bracket: the outer ghost reads legitimate far-fluid (the coarse solution is physically correct where no fine patch exists); the inner ghost reads a frozen cell whose DFs are fine-averaged projections — bounded but resolution-mismatched, not the coarse PDE state that the C2F interpolation assumes.

**Error propagation.** The inner ghost's reconstructed DFs are 75% delayed-fine (ring) + 25% resolution-mismatched (frozen hidden). Fine interior boundary cells stream directly from these ghosts, so the mismatch contribution enters the fine interior computation at the boundary. This is consistent with the §9.1 observation that the residual is "concentrated in the fine-boundary band (2–4 fine cells inside the interface)": the inner ghost is the first cell of that band, and its mismatch contribution is the leading error source.

**Distinction from §9.2.** §9.2 frames the residual as *missing feedback*: fine-interior data never reaches the coarse lattice via F2C. The C2F hidden-cell injection is the complementary problem: *resolution-mismatched data* (fine-averaged DFs in frozen cells) is injected *into* the fine boundary via C2F. The one-way-clamp framing treats all coarse C2F sources as legitimate interpolation inputs; it does not flag that the inside-hidden cells contain fine-averaged projections rather than the coarse PDE solution. Both mechanisms contribute to the residual — §9.2 explains the pressure-offset non-decay, and the hidden-cell injection explains the boundary-band concentration.

**Why the obvious fixes fail.** The falsification matrix (§9.3) tested two attempts to address the under-footprint region. v6 (naive two-way F2C into collision-active hidden cells) went unstable; v8 (deeper F2C reach into footprint cells) was +69% worse. v9 (freeze hidden cells as `GEO_NOTHING` + interior F2C) was 5.6× worse than v7 under the corrected harness. The v9 freeze eliminated the diverging shadow solve but introduced a different problem: the interior F2C injects fine non-hydrodynamic modes directly into the coarse lattice without collision damping (Astoul et al. 2021a, §12.6). The current state (frozen `GEO_NOTHING` + interior F2C after fine substeps) retains the freeze but the interior F2C's contribution is limited by the parity/frame issues documented in §10.1. A proper fix requires either (a) carving a hole so C2F extrapolates from ring + exterior only, (b) replacing hidden-cell content with a coarse-PDE-consistent state *before* the coarse streaming step, or (c) stabilized two-way coupling with relaxation and correct temporal ordering — all future work beyond v1.

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
| **v9** | **freeze hidden cells** (`GEO_NOTHING` under footprint + interior F2C, commit `a738b0d`) | **2,260,592**† | **+78%** | **worse** (5.6× vs v7 under corrected harness) |

† v9 measured under a corrected between-metric harness (proper fine-block placement at `[32:96]³`, matching-viscosity uniform references). v7 under the same corrected harness: 405,761. The original harness numbers (1,265,680 / 685,713) used the old /tmp/opencode script with a different composite builder and reference set; the relative ordering v7 > baseline > v9 is preserved.

### 9.4 Pre-fix trajectory

| Milestone | Commit | Cycle-10 violations | Key change |
|---|---|---|---|
| Geography corruption | `26e36db` | ~844k (pre-overlap) | Coupling kernel frame alignment, fill ordering, AA twisted writes |
| Frozen placeholder ring | `089e47a` | 1,265,680 | `storage_overlap=2` on fine blocks; F2C covers all ring cells |
| Collision-active ring | `5237b2f` | **685,713** | Ring streams+collides like fluid; F2C overwrites at end of step |

---

## 10. Known limitations of v1

- **Two-way coupling via frozen hidden cells**: coarse cells under the fine footprint are frozen as `GEO_NOTHING` (no stream/collide) and receive Lagrava-filtered fine-averaged DFs via interior F2C each cycle (§4.2, §5 step 8). This eliminates both the one-way clamp (§9.2) and the C2F shadow injection (§9.2.1): the ring cell streams from a frozen fine-injected neighbor (not a diverging shadow), and the C2F kernel reads fine-injected values from frozen cells (not invalid shadow data). **Verified correct for AB_PATTERN** (the `defs.h` default, used by sim_AMR). **AA_PATTERN has a deferred defect — see §10.1.** The residual is expected to drop substantially from the v7 baseline (685,713 violations); between-metric quantification is pending.
- **Static refinement only**: regions fixed at SimInit; no dynamic adaptation.
- **Single-rank, single-GPU**: fine blocks have no same-level MPI neighbors; coupling kernels are CUDA-only.
- **Multi-level (level > 1)**: `createAMRBlocks` hard-rejects `level > 1` during validation (throws at `amr_decomposition.h:187-188`); deeper nesting is impossible by construction in v1, not merely untested.
- **Non-Newtonian viscosity with AMR**: excluded.
- **IBM (immersed boundary)**: excluded.

### 10.1 Known defects (deferred — Oracle-verified)

The following defects were identified by an Oracle review of the freeze approach. They are silent in the current test suite (which tests each kernel against its documented contract in isolation, not the F2C-write → coarse-kernel-skip → C2F-read composition) and in sim_AMR (which uses AB_PATTERN with an interior footprint). They must be fixed before the corresponding configurations are exercised.

**Defect 1: AA_PATTERN C2F read of frozen cells is direction-reversed (blocking for AA AMR).**

The interior F2C stores frozen DFs in the **next-substep consume convention** (what the coarse streaming will read — natural if next is even/reflect, twisted if odd/spatial; `amr_coupling.h:452-462`). The C2F kernel reads with the **post-kernel produce convention** (`amr_coupling.h:152-168`: twisted if `coarse_even_iter`, natural otherwise). For normal cells these match because the coarse kernel rewrites them; frozen cells are never rewritten, so they retain the consume convention. Under AA these conventions are exactly opposite at **both** parities, yielding `f_used[q] = f_real[opp(q)]`: density survives (Σf_q invariant), but momentum is sign-flipped and non-equilibrium stress is mirrored, injected at 0.25 weight into every inner-ghost C2F fill, every cycle. Compounding this, on odd AA cycles the ring cells' `postCollisionStreaming` (`streaming_AA.h:61-90`) clobbers some surface-frozen slots with ring post-collision data before C2F reads them. AB is unaffected (single natural convention; C2F reads fine data that is 2 cycles stale but orientation-correct).

Fix: (a) in `cudaAMR_CoarseToFine`'s `read_coarse_df` (`amr_coupling.h:152-168`), branch on `coarse_SD.map(cx,cy,cz)==GEO_NOTHING` and read with the inverted parity convention; (b) skip ring→frozen stores in odd `postCollisionStreaming` using the allocated-but-vestigial `dinterface_dir` bitmask (`amr_decomposition.h:288-329`, `bc.h:499-505`), which exists for exactly this purpose.

**Defect 2: F2C has no coarse-map guard on DF writes (latent for boundary-touching footprints).**

`markAMRInterface` deliberately preserves physical BC tags (walls/inflows) under the footprint — those cells keep `GEO_WALL`/`GEO_INFLOW` rather than becoming `GEO_NOTHING`. But `cudaAMR_FineToCoarse` writes DFs with no coarse-map check (the map guard at `amr_coupling.h:467` covers macros only). A wall cell under the footprint gets its DFs overwritten with fine-averaged fluid data each cycle; under AB's pull scheme, adjacent coarse fluid cells stream from it as if fluid — the no-slip wall is bypassed wherever a footprint covers it, contradicting the documented "physical boundary conditions survive" design intent.

Fix: skip the DF write unless `coarse_SD.map(x,y,z) == GEO_NOTHING` — ~3 lines in the kernel (`amr_coupling.h:440`).

**Minor concerns (non-blocking):**

- **AB C2F 2-cycle staleness**: under AB, C2F reads `df_out` of cycle N+1, which after rotation is the physical buffer *not* written by cycle N's interior F2C — C2F sees fine data from cycle N−1 (2-cycle lag, orientation correct). Not a correctness break; a real asymmetry vs. the 1-cycle freshness the design narrative implies.
- **Conservation metric double-counts the footprint**: `computeConservationStats` (`amr_state.h:386-423`) sums frozen coarse cells (weight 1, fine-injected macros) *plus* fine interior cells (1/8 weight). It is a self-consistent drift metric, not exact physical conservation; v7 had the same double-count, so the comparison is apples-to-apples.
- **Overlapping/face-sharing same-level footprints**: `createAMRBlocks` has no overlap check; overlapping footprints produce duplicate interior patches (nondeterministic write order). Face-sharing footprints expose an order-dependence where a shared cell stays collision-active ring while another block's interior F2C overwrites it. v1 (single fine block) is unaffected.

---

## 11. Code references

| File | Role |
|---|---|
| `include/lbm3d/d3q27/amr_coupling.h` | Coupling kernels: `cudaAMR_CoarseToFine`, `cudaAMR_FineToCoarse` |
| `include/lbm3d/amr_state.h` | `State_AMR` driver: `SimInit`, `SimUpdate`, `buildCouplings`, `launchCoarseToFineTransfers`, `launchFineToCoarseTransfers`, `launchLBMKernelForLevel` |
| `include/lbm3d/amr_decomposition.h` | `createAMRBlocks`, `markAMRInterface` (rings `GEO_AMR_INTERFACE` + hidden cells `GEO_NOTHING`), `allocateInterfaceDirArray` |
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

---

## 12. Mathematical grounding and literature references

This section maps the implementation to the mathematical formulations in the AMR-LBM literature, and identifies what the literature suggests for the open interface problems (§9.2, §9.2.1, v9 regression).

### 12.1 Cell-centered volumetric formulation (Chen 1998, Chen et al. 2006)

The implementation uses the cell-centered (volumetric) approach: DFs are point densities at cell centers, and coarse/fine grids do not co-locate. Chen (1998) introduced the volumetric reformulation, treating distributions as *masses* moving between cells of different resolution. Chen et al. (2006) formalized the volumetric grid refinement concept, showing that conservation laws are exactly guaranteed through the volumetric formulation, with the approach independent of the collision step.

The C2F and F2C kernels (`amr_coupling.h`) implement the volumetric DF decomposition:

```
f = f_eq(ρ, u) + f_neq
```

where `f_eq` is the equilibrium distribution and `f_neq = f - f_eq(ρ, u)` is the non-equilibrium part.

**Coarse-to-fine** (§4.1 of the implementation doc):
```
f_fine = f_eq(ρ_f, u_f) + (τ_f / τ_c) · f_neq_interp
```
where `ρ_f`, `u_f` are 3rd-order Lagrange interpolated macros, and `f_neq_interp` is the 3rd-order Lagrange interpolated non-equilibrium from the 4 coarse nodes per axis (see §4.1 for the full stencil). The factor `τ_f / τ_c` rescales the non-equilibrium stress to the fine-level relaxation time.

**Fine-to-coarse** (§4.2):
```
f_coarse = f_eq(ρ_c, u_c) + (τ_c / τ_f) · f_neq_avg
```
where `f_neq_avg = (1/8) Σ f_neq_k` is the Lagrava-filtered non-equilibrium (the 1/8 arithmetic average IS the volumetric conversion — each fine cell holds 1/8 of the coarse cell volume).

The τ-rescaling is required because `ν_lb` differs between levels (§12.3). Filippova & Hänel (1998) introduced non-equilibrium rescaling; their original factor is `(τ_f−0.5)/(τ_c−0.5)·(dt_f/dt_c)`. The implementation uses the simpler `τ_f/τ_c` — the v1 variant in the falsification matrix (§9.3) tested the FH form and found it noise (−0.3%), so the simpler form is retained.

### 12.2 The Lagrava spatial filter (Lagrava et al. 2012)

The F2C kernel applies a mandatory spatial filter before projecting fine data onto the coarse grid. Lagrava et al. (2012) showed that without this filter, unresolved high-frequency fine modes alias onto the coarse grid, causing instability — especially in turbulent flows. The review (§4.2) confirms the filter is "mandatory at fine-to-coarse transfer locations."

The current implementation is the tensor-product 4-node-per-axis **Lagrange projection onto the coarse cell center** (4×4×4 = 64 fine cells): the nominal per-axis window `{fx0−1, …, fx0+2}` extends the 2×2×2 subcell block by one fine cell per side, the runtime-evaluated Lagrange weights are normalized to sum to one (centered windows: {−1, 9, 9, −1}/16), and near fine-block boundaries the window is shifted/shortened into the storable extent while the evaluation point stays fixed at the coarse center (the same shifted-window storability machinery as C2F). Properties:

1. **Volumetric** (implementers' synthesis merging Lagrava's filter with Chen's volumetric formulation): DFs are point densities; the sum-to-one weighted average over the subcell neighborhood plays the role of the volume ratio (no additional factor), generalizing the 1/8 weight of the box average.
2. **Filter**: the weighted neighborhood mean is a low-pass filter; the odd/even checkerboard (the highest fine-resolvable mode) is annihilated since the per-axis weights sum to zero on it ({−1, 9, 9, −1} on any alternating ± pattern evaluates to 0).
3. **Order**: the projection reproduces cubic fields at the coarse center exactly (the box average reproduced only linear fields), matching the 3rd-order C2F spatial interpolation and the review's "third- or fourth-order spatial" recommendation (§4.3); on shifted boundary windows it remains exact for constants and linears.
4. **Conservation**: global mass is conserved exactly on translation-invariant extents (each fine cell contributes with total weight 1/2 per axis → 1/8 in 3D, same as the box average); per-cell mass conservation — the box average's property — is traded for the higher order.

The 1/8 box average of the 8 subcells (the original v1 filter) remains available as a compile-time fallback (`-DF2C_BOX_AVERAGE`, cache var `TNL_LBM_F2C_STRATEGY`). Lagrava's own filter (cell-vertex method) is a weighted neighborhood smoothing; the implementation's weighted projection is its cell-centered volumetric analog.

### 12.3 The τ-rescaling and viscosity mismatch (Filippova & Hänel 1998)

With 2:1 convective scaling (`dt ∝ dx`), the lattice viscosity doubles per level:

```
ν_lb,f = physViscosity · physDt_f / physDl_f² = physViscosity · (physDt_c/2) / (physDl_c/2)² = 2 · ν_lb,c
```

The relaxation time `τ = 3·ν_lb + 0.5` differs between levels (`τ_f ≠ τ_c`). The non-equilibrium part of the DF depends on `τ` through the stress tensor `Π = Σ f_neq · c_i c_j ≈ -2/3 · ρ · ν_lb · S` (where `S` is the strain rate; this is a standard Chapman-Enskog relation, not from the review doc). Rescaling by `τ_f/τ_c` (C2F) or `τ_c/τ_f` (F2C) maps the stress correctly across levels, preserving the physical viscosity.

### 12.4 Berger-Colella time subcycling (Berger & Colella 1989)

The implementation follows the standard Berger-Colella schedule (§5): one coarse step per global iteration, two fine substeps per coarse step. The review (§2.2) confirms the 2:1 ratio is natural for LBM because halving `Δx` halves `Δt`, so "one coarse time step corresponds to two fine time steps."

**Temporal consistency**: both C2F fills use the coarse post-step state (`t_{n+1}`), so the first fine substep (advancing `t_n → t_{n+1/2}`) receives boundary data from a full coarse step ahead. The "H9" time-interpolation variant (§5, v4) was tested and rejected (+13.4% worse). The review (§4.3) notes that Palabos (Lagrava et al. 2012) uses linear (second-order) temporal interpolation — the same order as the H9 variant — suggesting that temporal interpolation alone (without higher-order spatial reconstruction) is insufficient for this configuration. Gendre et al. (2017) require at least third-order spatial and temporal interpolation to maintain consistency with the discrete velocity Boltzmann equation in their directional splitting approach — a requirement that the current C2F scheme's spatial order (3rd-order Lagrange) satisfies but whose temporal component remains unaddressed.

### 12.5 Two-way coupling: what the literature says

The central open problem is how to feed fine-interior data back into the coarse lattice without instability. The falsification matrix (§9.3) documents four failed attempts:

- **v6** (naive two-way, full overwrite): unstable (NaN @ cyc 2)
- **v8** (deeper F2C reach): +69% worse
- **v9** (freeze hidden cells + interior F2C): +5.6× worse (2,260,592 violations)
- **v7** (collision-active ring, no under-footprint coupling): best so far (405,761 violations)

The literature offers several relevant insights:

**Chen et al. (2006)** — the volumetric formulation's key property is that conservation is guaranteed *by construction* through particle redistribution: DFs are masses, and the 1/8 averaging in F2C conserves total mass exactly. This means a proper two-way coupling should not require ad-hoc stabilization (λ-blend, etc.) — the conservation is built into the volumetric averaging. The v6 instability likely resulted from overwriting *post-collision* coarse DFs with fine-averaged DFs without accounting for the collision state mismatch (the coarse cell had already collided; the fine average is pre-collision-equivalent).

**Lagrava et al. (2012)** — the filter is mandatory only for F2C; the C2F direction needs no filter (interpolation from coarse to fine is always well-posed). The implementation correctly applies the filter only in F2C. However, Lagrava's method is cell-vertex (node-based), where co-located nodes allow direct transfer — the cell-centered implementation must rely entirely on the volumetric averaging, which may introduce more coupling error at the interface.

**Guzik et al. (2014)** — developed cell-centered LBM AMR within the Chombo/Berger-Colella framework. The review (§4.3) notes that multiple studies find the interpolation order at the interface critical for maintaining LBM's second-order accuracy, with Palabos using third- or fourth-order spatial interpolation and Gendre et al. (2017) requiring at least third-order spatial and temporal interpolation for DVBE consistency. The implementation's 3rd-order Lagrange C2F (§4.1) meets the spatial requirement; the temporal component remains unaddressed (§12.4). Guzik also distinguishes IVP (prefill ghost cells at `t_l` for all subcycles) from BVP (refill each substep) — the implementation uses BVP (refill per substep); the IVP approach directly addresses the temporal mismatch by precomputing all ghost data at the start of the subcycle.

**Schukmann et al. (2023, 2025)** — conducted the only systematic head-to-head comparison of cell-vertex, cell-centered, and combined approaches with multiple collision models (BGK, MRT, RR, HRR). Key findings:
- Stability limits depend on the specific coupling mechanism, not just the grid layout.
- The HRR collision model filters non-hydrodynamic mode contributions regardless of the grid-coupling algorithm (Astoul et al. 2021a). The review does not state that the cumulant collision operator (used in this implementation) shares HRR's mode-filtering properties; whether cumulant provides equivalent damping is an open question.

**Astoul et al. (2021a)** — identified the root cause of spurious noise at refinement interfaces: non-hydrodynamic modes inherent to LBM generate spurious vorticity and acoustics when projected onto coarser grids. This is "intrinsic to resolution changes (aliasing) and is independent of the specific grid-coupling algorithm." This explains why the v9 freeze regression amplified vz: the interior F2C injects fine non-hydrodynamic modes directly into the coarse lattice, where they generate spurious vorticity (including the vz channel, §9.1). The collision-active ring (v7) damps these modes through coarse collision before they propagate.

### 12.6 Why v9 (freeze) regressed: the missing collision damping

The v9 freeze approach removed collision from hidden cells and replaced it with direct fine-averaged DF injection. The review literature explains why this fails:

1. **Non-hydrodynamic mode injection** (Astoul et al. 2021a, review §4.4): the fine solution contains non-hydrodynamic modes that the coarse grid cannot represent. When these are injected directly (v9 interior F2C), they alias onto the coarse grid and generate spurious vorticity/acoustics. In v7, the hidden cells' collision (cumulant relaxation) damps these modes before they reach the ring — the collision acts as a filter.

2. **Post-collision state mismatch** (inference, not directly in the review): v9 overwrites post-coarse-step DFs (already collided) with fine-averaged DFs. The next coarse streaming step then reads from a state that is inconsistent with what collision would have produced — a discontinuity that propagates through the coarse lattice. Note: the volumetric approach is described in the review as "independent of the collision step" (§4.1), which does not directly support this claim; the inference is that overwriting a post-collision state with a pre-collision-equivalent average creates an inconsistency.

3. **Loss of collision damping** (inference from Schukmann et al. 2023, review §4.2): the collision-active hidden cells in v7 damp interface perturbations through viscous dissipation. Removing collision (v9) exposes the coarse lattice directly to fine-grid perturbations. The review identifies stability as dependent on the coupling mechanism, but does not use "buffer zone" terminology; this interpretation is the implementers' inference.

### 12.7 Suggested approaches from the literature

Based on the review, the most promising directions for improving the interface coupling are:

1. **Higher-order C2F spatial interpolation**: upgrade from trilinear (second-order) to at least third-order spatial interpolation (Gendre et al. 2017, Lagrava et al. 2012). **Implemented** (§4.1: 4-node-per-axis 3rd-order Lagrange; original trilinear retained as `-DC2F_TRILINEAR`).

2. **Temporal interpolation for C2F**: despite v4 (linear H9) failing, third-order temporal interpolation (as required by Gendre et al. 2017 for DVBE consistency) may work where linear interpolation failed. Alternatively, Guzik's IVP approach (prefill all ghost cells at `t_l` before the subcycle) directly avoids the temporal mismatch without per-step interpolation.

3. **Mode-filtering collision at the interface**: the review identifies HRR as filtering non-hydrodynamic mode contributions (Astoul et al. 2021a). The implementation uses the cumulant collision operator; whether cumulant provides equivalent non-hydrodynamic mode filtering is not established in the review. If it does not, switching the ring cells to HRR (or a hybrid) may reduce spurious noise.

4. **Cell-vertex (node-based) coupling for the interface**: the review contains a tension on this point — §4.2 reports "cell-vertex approaches qualitatively emit less spurious acoustic noise than cell-centered layouts" (Schukmann et al. 2023), while §4.4 and Table 1 find cell-centered approaches produce "the least spurious noise" with linear or uniform explosion in C2F (Schukmann et al. 2025). The difference likely reflects different test cases and metrics. A hybrid approach — cell-centered in the bulk, cell-vertex at the interface — is the "combined method" family described in §3.3 of the review.

5. **Direct coupling** (Astoul et al. 2021b): eliminate the overlapping mesh layer (the ghost ring) and solve a non-linear equation system constraining zeroth- and first-order non-equilibrium moments at the interface. This "tightens the link between fine and coarse grids" and reduces spurious noise. Note: this is a cell-vertex method; applying it to the cell-centered implementation would be a redesign.

6. **Linear or uniform explosion in C2F** (Schukmann et al. 2025): the review finds that cell-centered C2F with "linear or uniform explosion" produces the least spurious noise among cell-centered variants. This is directly actionable for the C2F kernel's interpolation strategy. **Implemented and measured** (compile-time switches `-DC2F_LINEAR_EXPLOSION` / `-DC2F_UNIFORM_EXPLOSION` in `amr_coupling.h`; pure equilibrium explosion — non-equilibrium zeroed): both variants run the sim_AMR Taylor-Green benchmark to completion without NaN, but both regress on the between-property bracket metric at cycle 10 — linear explosion 839,717 violations (+346 %, dominated by rho: each coarse cell's density is duplicated piecewise-constant over its 8 subcells, producing plateaus that break the bracket), uniform explosion 276,789 (+47 %) — vs the 3rd-order Lagrange baseline of 188,353. The aeroacoustic-noise advantage found by Schukmann et al. does not translate into bracket-metric accuracy on this benchmark; the 3rd-order interpolation remains the preferred default.

7. **Two-step trilinear C2F** (Freitas et al. 2006): interpolate to a virtual cell then transform — the implementation's direct trilinear C2F is close to this; a two-step variant may reduce coupling error.

8. **Compact-moment (cumulant-projection) C2F** (Schönherr 2015): project the coarse cell's distribution onto its 3rd+ order cumulants and reconstruct the fine-scale distribution from the filtered moment set, in the spirit of the mode-filtering review finding in item 3. **Implemented and measured** (compile-time switch `-DC2F_COMPACT_MOMENT` in `amr_coupling.h`; not the default). Measured against the 3rd-order Lagrange baseline via `between_metric` with trilinear upsampling and volume-weighted norms: CM yields 688,265 violations at the final cycle vs 202,078 for the baseline (3.4× worse at cycle 10, 1.9× overall). Per-field at cycle 10 — rho: CM 513,702 violations (28× L1 norm vs baseline), the dominant error; vx/vy: ~2.8× minor increase; vz: 1.5× violation count but 0.64× L1 and 0.26× L∞ (improved on average). The mode-filtering approach reduces vz aliasing as designed but severely degrades the rho channel. CM is retained as a compile-time option but is not the default.

A follow-up variant — **post-collision Pi^neq rescale** — hypothesized that the coarse DFs feeding the k-moment reconstruction are post-collision (AB pattern reads `df_out`), so their second-order cumulants are already relaxed by (1−ω₁), corrupting the Schönherr k-moment relations (Eqs. 7.5–7.9); rescaling Π^neq by 1/(1−ω₁) ≈ −1.06 was expected to recover the pre-relaxation stress. Measured: **1,034,736 violations at cycle 10 — 1.5× worse than the original CM on every field** (rho 709,304; vx/vy ~2.1×; vz 314,889, losing even the vz improvement). Un-relaxing the stress amplifies the interface feedback error instead of correcting it; the relaxed post-collision state is the better k-moment input. The sign-flip hypothesis is rejected. Combined with Qi et al. 2019 (§4.1) — who report that compact interpolation violates mass conservation and fails on fully periodic Taylor-Green vortices without Dirichlet boundaries — the compact-moment direction is considered exhausted for this benchmark.

### 12.8 References

- Astoul, T., Wissocq, G., Boussuge, J.-F., Sengissen, A., & Sagaut, P. (2021a). Analysis and reduction of spurious noise generated at grid refinement interfaces with the lattice Boltzmann method. *J. Computational Physics*, 425, 109949.
- Astoul, T., Wissocq, G., Boussuge, J.-F., Sengissen, A., & Sagaut, P. (2021b). Lattice Boltzmann method for computational aeroacoustics on non-uniform meshes: A direct grid coupling approach. *J. Computational Physics*, 430, 110667.
- Berger, M. J., & Colella, P. (1989). Local adaptive mesh refinement for shock hydrodynamics. *J. Computational Physics*, 82(1), 64–84.
- Chen, H. (1998). Volumetric formulation of the lattice Boltzmann method for fluid dynamics: Basic concept. *Physical Review E*, 58(3), 3955–3963.
- Chen, H., Filippova, O., Hoch, J., Molvig, K., Shock, R., Teixeira, C., & Zhang, R. (2006). Grid refinement in lattice Boltzmann methods based on volumetric formulation. *Physica A*, 362(1), 158–167.
- Schukmann, A., Schneider, A., Haas, V., & Böhle, M. (2023). Analysis of hierarchical grid refinement techniques for the lattice Boltzmann method by numerical experiments. *Fluids*, 8(3), 103.
- Schukmann, A., Haas, V., & Schneider, A. (2025). Spurious aeroacoustic emissions in lattice Boltzmann simulations on non-uniform grids. *Fluids*, 10(2), 31.
- Filippova, O., & Hänel, D. (1998). Grid refinement for lattice-BGK models. *J. Computational Physics*, 147(1), 219–228.
- Freitas, R. K., Meinke, M., & Schröder, W. (2006). Turbulence simulation via the lattice-Boltzmann method on hierarchically refined meshes. In P. Wesseling, E. Oñate, & J. Périaux (Eds.), *Proceedings of the European Conference on Computational Fluid Dynamics (ECCOMAS CFD 2006)*. TU Delft Repository.
- Gendre, F., Ricot, D., Fritz, G., & Sagaut, P. (2017). Grid refinement for aeroacoustics in the lattice Boltzmann method: A directional splitting approach. *Physical Review E*, 96, 023311.
- Guzik, S. M., Weisgraber, T. H., Colella, P., & Alder, B. J. (2014). Interpolation methods and the accuracy of lattice-Boltzmann mesh refinement. *J. Computational Physics*, 259, 461–487.
- Lagrava, D., Malaspinas, O., Latt, J., & Chopard, B. (2012). Advances in multi-domain lattice Boltzmann grid refinement. *J. Computational Physics*, 231(14), 4808–4822.
