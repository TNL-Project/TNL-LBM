# AMR for LBM — Implementation Notes

**Status:** Schönherr-ch7 conversion close-out (2026-08-22, branch `feat/amr-schonherr-ch7` at commit 16 of `.omo/plans/schonherr-ch7-conversion.md`), **plus the simulated-band flip (2026-08-23)**: after the T16 20-tc null verdict the contract's fork-row-(c) trigger fired and the passive band (both-frames fill, destinations never collided/streamed) was converted to Schönherr's simulated band — the widened substep-1 fine launch integrates the inner ghost rows, F2C runs once per coarse step, and C2F runs once per cycle filling both ghost rows of the substep-0 frame (§5, contract §3/§4). The same wave HARD-REMOVED the C2F carve pre-pass, its knobs (`C2F_CARVE`/`C2F_NO_CARVE`/`TNL_TEST_NO_CARVE`) and the mock carve tests (10–13/17) — the band registration statically rejects covered sources at SimInit (`checkCouplingMapPattern`), so the pre-pass could never fire; dated carve records below stay as the audit trail. §2.1 (`storage_overlap`), §§2.3/3/4/5/6, §§8.1–8.2, the §9.2.1 closing note, the §10 lead bullet, the §12.1 supersession note and the "Interface density bias" paragraph (¶ ~:271) describe the **converted** band registration, cycle and σ-form transfers; §§2.2/7/9/10.1/12 otherwise remain the dated v1 debugging-arc record (kept, date-stamped). The normative band/cycle contract is `docs/AMR-schonherr-ch7-target-contract.md`; the conversion's measured verdict (T16 20-tc decision table, honest negative/null) is quoted at the end of that ¶.
**Scope:** static, cell-centered, volumetric coupling (Chen 1998 / Chen et al. 2006 / Guzik 2014 / Schönherr 2015 ch.7), 2:1 refinement ratio, single-GPU single-rank, D3Q27 lattice.

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
- `storage_overlap` member: defaults to `overlap_width` (1 with MPI, 0 without); set to **2** for `level > 0` in `initLevelLattice` (`lbm_block.hpp:50-61`) — the overlap hosts the TWO coarse-to-fine destination rows of the converted band registration (§3, contract §2); the registration grows inward, never outward, so the allocated extent per axis stays `2·size + 2` under the re-anchored fine interior (16-local/1-deep and 14-local/2-deep blocks of one footprint store the same 18 rows per axis). (History: 2 at `089e47a`, 1 from the 2026-08-16 Phase-C removal, back to 2 at conversion commit 5.) All per-cell arrays (DF, map, macro) share one `data.indexer` (derived from `dmap`), so they must use the same overlap width. The allocation only materializes overlap on axes where `local != global` (fine blocks always satisfy this since their footprint is strictly smaller than the refined global extent). (2026-08-26: per-axis overrides raise the overlap to 3 on axes whose footprint attaches to a physical wall — the extra row is the GEO_NOTHING streaming buffer behind the fine-level bounce-back wall, §6.1; TNL overlaps are per-axis symmetric, so the far face of the same axis gains one allocated-but-unread row.) **Note:** all overlap `setSize` + `allocate()` calls are inside `#ifdef HAVE_MPI`; non-MPI builds use plain NDArrays with no overlap support. AMR v1 effectively requires an MPI-enabled single-rank (nproc=1) build.

### 2.2 Multi-level LBM (`include/lbm3d/lbm.h`, `.hpp`)

`LBM<CONFIG>` holds a flat `blocks` vector with per-level bookkeeping (`level_block_counts`, `max_level`). `getBlocksAtLevel(L)` returns pointers to blocks at level L. `updateKernelDataForLevel(L, substep)` toggles the fine level's parity/rotation for the given substep (AA: `even_iter = (substep % 2) == 1`; AB: DF pointer rotation) and restores the per-level `lbmViscosity` that the global `updateKernelData()` clobbers from the level-0 lattice. `updateKernelData()` (global, driven by `iterations` counter) sets parity/rotation for **all** blocks; fine levels are subsequently overridden by `updateKernelDataForLevel(L, substep)` before each kernel launch.

### 2.3 Block creation (`include/lbm3d/amr_decomposition.h`)

`createAMRBlocks(lbm, regions)` takes a pre-parsed `std::vector<AMR_Region<CONFIG>>` (parsing is done by `parseAMRConfig` separately), validates each region, constructs fine `LBM_BLOCK` instances with the level-aware constructor, sets `global_offset` in parent-level coordinates, allocates host/device data, resets map to `GEO_FLUID`, and initializes DFs to equilibrium. Fine-interior indexer (Schönherr-ch7 band registration ruling, 2026-08-19/20): the outermost requested-footprint row becomes a coarse-authoritative ring row (simulated), the covered F2C-refilled destination row starts one coarse row inside, and the fine-authoritative full coverage is the footprint inset 1 coarse row per face — `offset_fine = 2·(origin_coarse ≫ level−1) + 1` and `local_fine = 2·(size_coarse ≫ level−1) − 2` per axis (with depth-2 ghost overlap the allocated extent stays `2·(size ≫ level−1) + 2`; physical positions and every band row's storage parity are fine-global-coordinate properties, invariant under the shift — old local index = new local index + 1; the ≫(level−1) shifts are the amr-nlevel-nesting commit-A parent-frame normalization — the region file stays in level-0 coordinates for every level while `global_offset` reads parent-frame at any depth, and reduce to the level-1 formulas bit-for-bit at level 1). Footprint extents derived from `local` therefore read `(local + 2)/2` everywhere (`buildCouplings`, `markAMRInterface`, `isShadowedBySameLevelBlock`, the VTKHDF writer's `vtkGhostType`). Validation minimum (updated for the ruling): each refinement-region axis must span ≥ 3 PARENT-LEVEL cells — gs = 2 leaves a dual-role row (one face's c=0 ring row is the opposite face's c=1 destination row; the former ≥ 2 rationale was the F2C filter's 4-node window against the fine interior, which gs < 3 also violates) — and `createAMRBlocks` rejects such regions in its read-only validation phase before any block is created (`spdlog::error` + `std::runtime_error`), locked by Test 7 of `tests/unit/test_amr_subcycling.cu` and the gs ≥ 3 locks of the conversion plan. Since the amr-nlevel-nesting commit B the validation is the nesting V-suite (V1–V10: level bounds, positive size, parent-frame domain containment, alignment, ascending file order, unique containment, telescoping gaps with a hard floor of 2 and an advisory tier below 3, sibling separation, positive parent-frame origin, wall-shared-chain agreement), which superseded the v1 `level > 1` reject; arbitrarily deep telescoping chains are created level-ascending.

`markAMRInterface(lbm)` tags two coarse-cell populations by the distance-band rule of the contract band map (`amr_decomposition.h:479-491`, contract §2): (1) the **interface ring** — the 1-cell halo around each fine footprint (ring row c=−1) PLUS the footprint's surface shell (ring row c=0, the former GEO_NOTHING skin **reactivated** into the ring) — as `GEO_AMR_INTERFACE`; both rows are collision-active (§6) and serve as the C2F source pair `{c=−1, c=0}` (§3), and (2) the **hidden (frozen) cells** — every covered cell at footprint-depth ≥ 1 (the c=1 skin destination row and the c≥2 deep core) — as `GEO_NOTHING`, cells that do not stream or collide. The marking is per-pair (each finer level re-tags only its own parent's map, disjoint across pairs) and its cell-TAGGING RULE needed no change for multi-level nesting: the amr-nlevel-nesting commits verified it offset-correct at depth ≥ 1 under the parent-frame `global_offset` (the nested map census of `tests/unit/test_amr_nesting.cu`, incl. the wall-shared halo rows landing in the parent's ghost zone, and the SimInit `checkCouplingMapPattern` rails re-anchored to the same frame). The commit-B fix was confined to the dir-array indexing, which must convert the parent loop's global cell coordinates to the block-local storage frame (the bias is identically zero for level-0 parents). The frozen cells prevent the diverging "shadow solve" (a coarse LBM evolution under the footprint where the fine lattice is authoritative — see §9.2.1). The depth-1 skin cells' DFs are written exclusively by the interior F2C transfer once per cycle (§4.2, §5 step 4); the deep frozen core is never written after initialization (D.5, 2026-08-16). Census (contract §2.4): ring = halo + c=0 shell = **784** on the K=8 fixture (488 + 296; (K+2)³−(K−2)³), frozen = **216** (c=1 shell 152 + deep 64); TGV K=32 region: ring 12,304 (halo 6,536 + c=0 shell 5,768), frozen **27,000** (skin 5,048 + deep 21,952; conservation recount +2,791 counted fine cells — the constant-volume accounting artifact of contract §2.3). Also allocates `dinterface_dir` bitmask storage for blocks owning interface cells (vestigial in v1 — `getInterfaceDir` has no callers; reserved for future directional coupling).

---

## 3. Coupling structure

### 3.1 Patch construction (`State_AMR::buildCouplings`, `amr_state.h:560`)

**Ring patches** (one per face): for each fine block at level L and each of the six faces of its footprint's ring band (disjoint partition idiom on the ring's coarse cells: x-faces span the full y/z tangent, y-faces the interior x-range, z-faces the interior x/y range), `buildCouplings` clips the face rectangle against every level-(L−1) block's range. A patch is appended iff it contains at least one `GEO_AMR_INTERFACE` cell not shadowed by another same-level block.

Each ring `AMR_InterfacePatch` stores:
- `coarse_origin` / `coarse_size` — the parent-level ring rectangle, **TWO cells thick in the face normal**, spanning the C2F source pair `{c=−1, c=0}` on the min faces (mirrored `{c=gs−1, c=gs}` on the max faces — contract §2 band map).
- `fine_origin` / `fine_size` — the matching fine-level destination rectangle: the face's disjoint **vertex-straddling** partition of the overlap complement (fine-indexer coordinates under the re-anchored block offset — old fine index = new + 1; refinement blocks allocate a **depth-2 ghost overlap**, §2.1, so each face owns a nominal 4-row face-normal destination rectangle of which the launch clip realizes the middle 2).
- `face` — the interface normal direction.

Destination census on the K=8 fixture (contract §2.4): partition totals per face family {648, 648, 504, 504, 392, 392}, total **3,088** = 18³−14³ (corner rows 64, edge rows 672, face interiors 2,352), each written once. Both fine destination rows share the nearest coarse vertex −0.5 and the source pair `{c=−1, c=0}`, so the nominal per-axis windows straddle the shared vertex ⇒ **no C2F source at a valid face is a frozen covered cell** — a covered source at a valid face is an invalid registration outright (map-pattern assertion at SimInit; the former carve fallback pre-pass was hard-removed on 2026-08-23). Fine destination/standard/F2C-source rows all sit over SIMULATED coarse cells; only the F2C destination sits over frozen. The fine destination rows are Schönherr's SIMULATED destinations (contract §4 fork (c) amended 2026-08-23): the inner row is integrated by the widened substep-1 fine kernel, the outer row stays fill-only as that integration's streaming source.

**Interior patches** (per fine block: the footprint's 6 disjoint inset-face SKIN rectangles — the depth-1 shell one coarse row inside the reactivated c=0 ring row, partitioned at begin+1 / tangent insets+2; the gs=3 max-side slab dedupe-clamps empty — contract §2.2). These target the frozen `GEO_NOTHING` c=1 skin cells. `face` is set to `SyncDirection::None` (not a face). The interior F2C transfer injects the strategy-selected fine state into these cells once per cycle end (§4.2, §5 step 4), providing the sole two-way feedback channel; the deep frozen core is never written.

A host-side **map-pattern assertion** runs in `SimInit` after `buildCouplings` (`checkCouplingMapPattern`, amr_state.h:344, `#ifndef NDEBUG`): every C2F destination cell's nominal face-normal source pair ⊂ `GEO_AMR_INTERFACE`, and every interior-patch (F2C destination) cell ∈ `GEO_NOTHING` at exactly surface-depth-1 (`c=1`/`c=gs−2`) — the static 1-cell-registration canary of the stage-1 gate (Oracle review F2; `spdlog::error` + terminate on violation).

### 3.2 Launch helpers (`amr_state.h`)

`launchCoarseToFineTransfers(L)` iterates ring patches for level L and launches `cudaAMR_CoarseToFine` per patch, clipping each fine rectangle **face-aware per axis** to the fine block's allocated overlap storage: face-normal `[−ov, 0)` on min faces / `[local, local+ov)` on max faces, tangential `[−ov, local+ov)` (`df_overlap_X/Y/Z()` per-axis from the indexer). On the disjoint partition rectangles of §3.1 the clip realizes the middle two destination rows of the nominal 4-row rectangle; on global-extent-touching axes (`df_overlap == 0`) it empties that face's fill.

(The former ring launcher `launchFineToCoarseTransfers(L)` over these ring patches was deleted in D.1; its per-cell storability guard — non-storable cells skipped inside the kernel using the per-axis allocated overlap `idx3d ov` — survives unchanged in the skin launcher below: D.5, 2026-08-16.)

`launchFineToCoarseTransfersInterior(L)` iterates the interior patches (the 6 skin rectangles), launching `cudaAMR_FineToCoarse` over each — the ONLY fine-to-coarse channel since D.1 (D.5, 2026-08-16). The frozen `GEO_NOTHING` c=1 skin cells receive the strategy-selected fine state (the Schönherr σ=2 compact-moment transfer by default since commit 15; the Lagrava-filtered fine average under the `F2C_LAGRAVA` opt-out — §4.2), full overwrite — no λ-blend needed because the frozen cells have no collision to conflict with; the deep frozen core is never written.

---

## 4. Coupling kernels (`include/lbm3d/d3q27/amr_coupling.h`)

### 4.1 Coarse-to-fine (`cudaAMR_CoarseToFine`)

Fills the fine destination rows (the depth-2 ghost rows of the converted band, §3) from coarse data. The strategy surface is `-DTNL_LBM_C2F_STRATEGY=...` (`sim_AMR/CMakeLists.txt`); all branches share the launch and the write/guard plumbing (steps 4–5 below).

**Default since 2026-08-18 (user ruling): compact-moment (CM) σ-form** (the carve companion pre-pass was hard-removed on 2026-08-23). Per destination cell, the 8 source coarse cells of the nominal window — at valid faces the vertex-straddling `{c=−1, c=0}` pair per face-normal axis, §3.1 — contribute their macros and second-order k-moments evaluated at the SOURCE rate `ω_s = 1/τ_coarse` (thesis Eqs. 7.5–7.9); the density/velocity polynomial coefficient families `d_*/a_*/b_*/c_*` are fitted (Eqs. 7.10–7.33, with the averaged-moment corrections `avk` of Eqs. 7.29–7.33) and evaluated at the destination-relative positions |t_rel| = 0.25 per axis (Eqs. 7.34–7.37) — the band registration restores these centered evaluation points where the old geometry's carve extrapolated at |t_rel| = 0.75; the reconstructed DFs are assembled via the Schönherr second-order cumulants with `σ_{c→f} = 1/2` and the DESTINATION rate `ω_d = 1/τ_fine` (the shared `sigma`/`off_factor`/`diag_factor` machinery now at `amr_coupling.h:513-540`, invoked with `sig = n1o2`) and the cumulant back-transform of the collision operator (Geier 2015 Eqs. 81–96). **The σ-form carries NO Filippova–Hänel τ-rescale** — the three surviving τ-rescale sites of the tree are named at step 3 below (census: contract doc, Appendix "§7.2 equation audit" §A.2.2). The CM branch was audited against the printed thesis §7.2 equation-by-equation: **45 match / 12 convention-difference / 0 bug**; the only code↔print deviations are the Eq. 7.18/7.23/7.24 family, where the code implements the nodal-consistent, cyclically closed family and the print is internally inconsistent (suspected thesis-print errata, recorded in-place at `amr_coupling.h:676-701`; full worksheet: contract doc, Appendix). The former **carve** pre-pass (one-cell window shift per axis with re-evaluation up to |t_rel| = 0.75 wherever a nominal source is a covered `GEO_NOTHING` cell — the Eqs. 7.49–7.57 implementation for |offset| ≤ 1 per axis, degenerate collapse with a rate-limited warning) was HARD-REMOVED on 2026-08-23 together with its `C2F_NO_CARVE`/`TNL_TEST_NO_CARVE` knobs and the mock carve tests (10–13/17): the band registration makes the nominal windows at valid faces never touch covered cells, statically asserted by SimInit's `checkCouplingMapPattern`, so the pre-pass could never fire. Debug channel defines `C2F_EQ_ONLY` / `C2F_DEV_ONLY` / `C2F_NORM_ONLY` / `C2F_SHEAR_ONLY` zero respectively all / the (automatically cancelling) trace / the off-diagonal-shear / the diagonal-deviatoric non-equilibrium content of the CM fill — the attribution tooling of the "Interface density bias" investigation and of the T16 ladder (¶ ~:271; `C2F_DEV_ONLY` is an algebraic no-op on the CM branch, locked by `tests/unit/test_amr_c2f_debug_smoke.py`).

**Legacy opt-out pipeline (`C2F_LAGRANGE`, steps 1–3):**

1. **3rd-order Lagrange macro interpolation**: for each fine ghost cell at global fine coordinate `fg`, compute the home coarse cell `home = floor_div(fg, 2)` (true floor division via `fdiv2`). Per-axis stencil: 4 coarse cell centers `{home−2+(fg&1)…home+1+(fg&1)}` with Lagrange weights evaluated at the fine cell center (offset ±1/4 from the home center; centered windows give the dyadic rationals {−5, 35, 105, −7}/128 for even `fg` and {−7, 105, 35, −5}/128 for odd `fg`). Exact for cubic fields; the upgraded scheme implements the "3rd-order C2F spatial interpolation" recommendation of §12.7 (Gendre et al. 2017, Lagrava et al. 2012). **Storability guard**: the kernel shifts/shortens each per-axis window into the coarse storage extent queried from `coarse_SD.indexer` (sizes + overlap) and normalizes the runtime-computed weights to sum to one, so no out-of-bounds access can occur; full nominal accuracy needs ghosts ≥2 coarse cells inside the coarse stored extent. The original trilinear scheme (2-node stencil, 3/4:1/4 weights) remains available via `-DC2F_TRILINEAR`. **Explosion alternatives** (Schukmann et al. 2025, §12.7 item 6): `-DC2F_LINEAR_EXPLOSION` skips the neighbor interpolation entirely — each fine ghost cell takes the home coarse cell's macros `(rho, u)` directly and re-evaluates the equilibrium at them with the non-equilibrium zeroed (pure equilibrium explosion); `-DC2F_UNIFORM_EXPLOSION` duplicates the home cell's DFs to the 8 subcells unchanged (zeroth order, no rescaling). The home cell index is clamped per axis into the coarse storage extent (the explosion analog of the storability guard). Measured on the Taylor-Green bracket metric (cycle-10 violations vs the 188,353 baseline): linear explosion 839,717 (+346 %, rho-plateau dominated), uniform explosion 276,789 (+47 %) — both worse than the 3rd-order interpolation; see §12.7 item 6.
2. **Equilibrium re-evaluation**: `COLL::setEquilibrium(KS_F)` at the interpolated (rho, u) — one `EQ::eq_*` call per direction, never shared.
3. **Non-equilibrium interpolation + rescale**: per direction, interpolate `f_neq = f_cell − eq_cell(rho_c, u_c)` from the same coarse stencil with the same weights, rescale by `τ_f / τ_c` (the non-equilibrium rescaling factor; with 2:1 refinement τ_f ≠ τ_c since ν_lb doubles per level). Result: `f_fine = eq_f + (τ_f/τ_c) · f_neq_interp`. **τ-rescale census (3 surviving sites, audit §A.2.2)**: this legacy branch (`amr_coupling.h:1229`), the F2C Lagrava opt-out branch (`:1660`, §4.2), and the opt-in `C2F_LINEAR_EXPLOSION` debug branch (`:810`) are the ONLY Filippova–Hänel τ-ratio sites in the tree (`include/`, `sim_AMR/`) — the σ-form CM default above and the σ=2 F2C default of §4.2 carry none.
4. **Write** to `fine_SD.df(df_cur, ...)` in the storage-parity expected by the upcoming fine substep (AB: natural orientation; AA: twisted — direction q stored in `opposite_direction(q)` slot because the spatial/odd substep pulls from the opposite slot).
5. **Macro write guard**: if `fine_SD.map(x,y,z) == GEO_AMR_INTERFACE`, write interpolated macros to `dmacro` (no-op in v1 — fine ghosts are never tagged).

### 4.2 Fine-to-coarse (`cudaAMR_FineToCoarse`)

Projects the fine state onto frozen coarse coupling cells — in production the depth-1 `GEO_NOTHING` c=1 skin cells under each fine footprint (the ring-cell channel this kernel formerly also served was hard-deleted in D.1; D.5, 2026-08-16). The strategy surface is `-DTNL_LBM_F2C_STRATEGY=...` (`sim_AMR/CMakeLists.txt:28`): the kernel splits on `#ifdef F2C_SCHONHERR` only (`amr_coupling.h:1429`), and **`F2C_LAGRAVA` is a named no-op define** — every value other than `F2C_SCHONHERR` (including the empty string) lands on the Lagrava else-branch; `F2C_BOX_AVERAGE` keeps selecting the original 1/8 box average inside that branch.

**Default since commit 15 (T17, 2026-08-22): `F2C_SCHONHERR` — the Schönherr §7.2 σ=2 compact-moment F2C** (`amr_coupling.h:1429-1694`). Sources are the destination cell's own 8 subcells (contract fork (h), closed; the new-indexer fine locals {1,2}): per source, macros + second-order k-moments at `ω_s = 1/τ_fine`; the coefficient fits are evaluated at the destination t = (0,0,0), where only the `d_0`/`a_0`/`b_0`/`c_0` macros survive (all five A/B/C aggregates vanish identically and the avk gradient corrections cancel — audit §A.2.3, verified; the `a_0` family carries the R1 Eq.-7.18 code-family choice locked at commit 10); cumulants with `σ_{f→c} = 2` and the DESTINATION (coarse) rate `ω_d = 1/τ_coarse` (thesis Eqs. 7.38–7.48); the reconstruction reuses the shared compact-moment device helpers extracted at commit 12 (bitwise-gated) — same σ/prefactor machinery and back-transform as the C2F default of §4.1. **NO τ-rescale, no f_avg, no lo=0 clamp, no 4×4×4 window** (those all stay in the Lagrava branch). Exactness locks (`tests/unit/test_amr_schonherr_exactness.cu`, `tests/unit/test_amr_f2c_schonherr.cu` + its pytest driver): constant field exact; linear velocity exact; quadratic velocity + linear density exact at (0,0,0); CE-consistent strain round-trip at σ=2; Σf = d0 exactly at the destination. Measured comparison on the 20-tc table (§"Interface density bias" ¶): F2C_SCHONHERR conserves mass ~18× tighter than the opt-out (final drift +7.55e-07 vs −1.36e-05); its residual KE floor sits ~535× lower; its standing interface bias amplifies ×1.166 (within the pre-registered ×1.2 bound).

**Opt-out: `F2C_LAGRAVA` — the Lagrava 4×4×4 filter + FH τ-rescale** (the pre-conversion default, steps 2–4 below F2C content).

1. **Storability guard** (per cell, per axis; shared): all 8 fine subcells must be within the fine block's overlap-extended storage `[-ov_i, fine_local_i + ov_i)`. Cells failing the guard are skipped individually (commit `26e36db` replaced a launch-extent clip that evaluated storability in the wrong origin-aligned frame). (2026-08-22: the fine-block overlap depth is 2 under the converted band registration, §2.1; production launches cover only the under-footprint skin rectangles — the retired ring launches were the outer ghost layer's only reason to exist; commit `089e47a`'s first overlap-2 era is recorded in §9.4.)
2. **Lagrava filter** (mandatory in the opt-out branch): tensor-product 4-node-per-axis Lagrange projection of the fine DFs onto the coarse cell center (`t = fx0 + 0.5`, fine indexer coordinates) — the nominal per-axis window `{fx0−1, …, fx0+2}` covers the 2×2×2 subcell block extended by one fine cell per side (4×4×4 = 64 fine cells); centered windows yield the dyadic rationals {−1, 9, 9, −1}/16 per axis. Near fine-block boundaries the window is shifted/shortened into the storable extent with runtime double-precision weights (same machinery as the C2F storability guard) while the evaluation point stays fixed at the coarse center, so constant-to-cubic fields are reproduced exactly on every window (the plain 1/8 box average was linear-exact only). Additionally the window's lower bound clamps to the fine interior (`lo = 0` — unconditional in the shipping design: D.5, 2026-08-16), so min-side footprint-face windows evaluate on `{0,1,2,3}` and never read fine ghost cells. The weights sum to one, so the weighted sum IS the volumetric fine-to-coarse conversion — no other volume factor. The 1/8 box average remains as `-DF2C_BOX_AVERAGE`.
3. **Macro recompute + equilibrium** (opt-out branch): `COLL::computeDensityAndVelocity(KS)` on the averaged DFs; `COLL::setEquilibrium(KS_EQ)` at those macros.
4. **Non-equilibrium rescale** (opt-out branch): `f_coarse = eq_c + (τ_c/τ_f) · (f_avg − eq_c)` — the reciprocal of the C2F factor; one of the three surviving Filippova–Hänel τ-rescale sites (§4.1 census): the F2C Lagrava branch `:1660` (τc/τf), the legacy `C2F_LAGRANGE` branch `:1229` (τf/τc), the `C2F_LINEAR_EXPLOSION` opt-in `:810` (τf/τc).
5. **Write** to the coarse DF state the NEXT coarse substep will consume (AB: logical `df_out`; next `updateKernelData()` rotation makes it `df_cur`; AA: natural if next substep is even/reflect, twisted if odd/spatial).
6. **Macro write**: filtered macros go to `dmacro` for every coupling cell the launch covers (the DF/macro store guard admits `GEO_NOTHING` and `GEO_AMR_INTERFACE` cells; in production the launches cover only the `GEO_NOTHING` skin cells — ring cells' macros come from the main coarse kernel only: D.5, 2026-08-16).

### 4.3 Global-frame indexing

All coupling kernel coordinate arithmetic is computed in the GLOBAL frame (per axis): fine global `fg = x + fine_off`; home coarse `home = fdiv2(fg)`; brackets converted back to the coarse block's indexer frame via `− coarse_off`. `fine_off` and `coarse_off` are the blocks' indexer origins in global coordinates of their level (`LBM_BLOCK::offset`). True floor division (`fdiv2`) handles negative fine ghost indices correctly (commit `26e36db`).

### 4.4 Streaming-pattern handling

The kernels receive parity parameters (`coarse_even_iter`, `fine_even_iter`) from the caller:

- **AB pattern**: `read_coarse_df` reads `df_out` (post-collision, natural orientation); `read_fine_df` reads `df_out`; F2C writes to logical `df_out`. The next global `updateKernelData()` rotates frames, so the physical array written becomes `df_cur` for the next consuming kernel.
- **AA pattern**: twisted storage — post-collision state stores direction q in slot `opposite_direction(q)` (even/reflect substep); post-stream state is natural (odd/spatial substep). `coarse_even_iter == true` → read `df_cur[opposite(q)]` (twisted); `false` → read `df_cur[q]` (natural). The F2C write orientation is chosen for the NEXT consuming coarse substep parity: `next_coarse_even_iter = (level==0) ? ((iterations % 2) == 1) : false`.

`updateKernelDataForLevel(L, substep)` is called BEFORE each fill so the fine block's `data.dfs` pointers / `even_iter` are set for the upcoming substep's consumption frame.

---

## 5. Time stepping: the Schönherr cycle with simulated band (`State_AMR::SimUpdate`, `amr_state.h`)

One global iteration advances `physDt_coarse` (Berger-Colella 2:1 subcycling); per cycle the schedule is the Schönherr cycle with **simulated band** (the six-step form shipped by the conversion was flipped after the T16 20-tc null verdict — contract §4 fork row (c); cycle contract of `docs/AMR-schonherr-ch7-target-contract.md` §3). Since the amr-nlevel-nesting commit C `SimUpdate` is the `advancePair(level)` Berger–Colella PAIR recursion: the positional `updateKernelDataForLevel(L, 0)`/`(L, 1)` calls below are de-facto invocations with the level's cumulative substep counter (the steps below are the `max_level == 1` reduction, locked bit-identical; the mid-cycle fill of each finer band sources the parent's live post-substep-A state, and the cycle end runs a level-ascending re-arm + C2F cascade — all pinned by the schedule census of `tests/unit/test_amr_nesting.cu` at 2 and 3 fine levels and by the per-pair transfer census of `tests/unit/test_amr_coupling.cu`). `computeBeforeLBMKernel()` and `nse.iterations++` (which counts COARSE steps only) run once before step 1:

1. **Fine substep 1 of 2** at level L: `updateKernelDataForLevel(L, 0)` (mandatory per-substep parity/rotation selection — the global `updateKernelData()` is driven by the coarse clock and must not drive the fine substeps) + `launchLBMKernelForLevel(L, compute_macro, /*ghost_layers=*/1)` — the launch extent is **widened one overlap cell per face ([-1, local+1))**, so the **inner ghost rows are INTEGRATED** (collide + stream like interior fluid; they are `GEO_FLUID` by construction) with their streaming input pulled from the outer ghost row filled at the END of the previous cycle (step 5; cycle 0 reads the `SimInit` single-frame fill);
2. **Fine substep 2 of 2**: `updateKernelDataForLevel(L, 1)` + `launchLBMKernelForLevel(L, compute_macro, 0)` — interior-only extent; its boundary data is substep 1's **kernel-updated** inner ghost rows in the other AB frame, so the band advances synchronously with the fine clock and no fill is needed for this frame;
3. **Coarse step** (level 0, one LBM step): the collision-active ring (§6) is written only here — fine feedback reaches it through streaming from the F2C-refreshed c=1 skin written at step 4 of the previous cycle.
4. **Fine-to-coarse** (`launchFineToCoarseTransfersInterior(L)`, once per cycle per Schönherr): reads the fine level's rotation-1 frame (the post-substep-2 array) and injects the strategy-selected fine state into the frozen `GEO_NOTHING` c=1 skin cells (§4.2) — the sole two-way feedback channel since D.1.
5. **Coarse-to-fine (the single fill of the cycle)**: `updateKernelDataForLevel(L, 0)` + `launchCoarseToFineTransfers(L)` — fills **both** overlap rows of the destination band in the frame the next cycle's substep 1 consumes (the inner row is what the interior pulls and what substep 1 integrates; the outer row is that integration's streaming source). The other frame needs no fill: substep 2 consumes substep 1's updated inner rows from it, and its outer row is unreachable (substep 2 is interior-only).

Steps 4 (F2C) and 5 (C2F) may be reordered freely — their touched sets are disjoint (F2C writes coarse skin cells of the coarse post-step array; C2F writes fine ghost rows), declared per the cycle contract. `SimInit` performs the same single-frame fill after `buildCouplings`, so cycle 0 starts with a valid substep-0 frame — the fine substep 1 of cycle 0/1 therefore reads a t_0 fill (the declared startup transient of contract §3; the seam-metric probe compares cycle 1 separately from cycle ≥ 2 for exactly this reason).

Physical time consistency: coarse advances physDt_coarse per cycle; fine advances 2 × physDt_coarse/2 = physDt_coarse. Verified exact in `test_amr_subcycling`.

**Retired with the reorder** (commit 8): the H9 time-centered first fill and the legacy per-substep BVP refill (fill #2 was content-identical to fill #1 — the removal is lossless; the H9 variant had measured +13.4 % worse on the v4 falsification row of §9.3). Configuring with the old `C2F_H9` knob now emits a CMake retirement warning (`sim_AMR/CMakeLists.txt`).

**Declared incompatibilities (contract §3, forks (f)/(i))**: checkpoint restart does not carry across the band registration (array shapes change); MPI is nproc=1 only.

---

## 6. Interface ring handling (collision-active since commit `5237b2f`, two rows since the conversion)

The interface ring is a **two-row band** under the converted band registration (§2.3): the 1-cell halo around the footprint (ring row c=−1) and the reactivated footprint-surface shell (ring row c=0, the former GEO_NOTHING skin) — both tagged `GEO_AMR_INTERFACE`. **Ring cells are collision-active**:
- `D3Q27_BC_All::preCollision` — no early return; ring cells proceed through normal streaming.
- `doCollision` — returns `true`; ring cells collide like fluid.
- `postCollision` — no early return; `postCollisionStreaming` writes DFs back.

The main coarse kernel (§5 step 3) is the SOLE writer of ring DFs and macros: the retired ring-F2C overwrite was hard-deleted in D.1, and under the six-step cycle the fine feedback reaches the ring via streaming from the F2C-refreshed `GEO_NOTHING` c=1 skin on the next coarse step (§5 step 4). Both ring rows are SIMULATED cells — this is what makes the C2F source pair `{c=−1, c=0}` simulated on both legs at valid faces (§3.1 vertex rule), replacing the old geometry's covered-skin interior read that forced the former carve's |t_rel| = 0.75 extrapolation (the carve itself was hard-removed on 2026-08-23).

Context (2026-08-14): the collision-active ring replaced the earlier collision-inactive design where `preCollision` wrote a `rho=1, v=0` placeholder and the kernel skipped streaming+collision — which made the ring a stiff boundary layer imposing an O(Ma²) pressure offset with no viscous decay channel.

### 6.1 Fine-level wall BC placement: outer band slot (option A) vs inner ghost row (option B, rejected 2026-08-26)

Decision record for the wall-attached refinement extension (Schönherr §7.3, commit `9a80751` attach + in-tree uncommitted fine-level bounce-back in `sim_AMR/sim_AMR_channel.cu` / `amr_state.h`): when a fine footprint face is flush against a physical wall, the fine block imposes its **own** bounce-back at the same no-slip link plane as the coarse level; the face's C2F fill is dropped (the ghost band there is BC-managed) and the fine kernel processes the wall row in both substeps (level-0 bounce-every-substep semantics). Two placements were analyzed; **option A (outer band slot) shipped**.

**Option A (shipped).** The uniform band registration is preserved (`offset = 2·go + 1`, `local = 2·gs − 2` per axis, §2.3); the wall occupies the face's OUTER band slot — min face: `GEO_WALL` at local −2 (= fine global 2·go−1, the footprint-adjacent subcell inside the coarse wall cell at go−1), `GEO_NOTHING` streaming buffer at local −3; max face: wall at local+1 (= 2·(go+gs), subcell inside the coarse wall cell at go+gs), buffer at local+2. The cell-centered bounce-back link plane then lands exactly on the coarse wall/fluid cell boundary on any face (rotation-invariant). A per-axis storage override raises the overlap to 3 on walled axes (TNL overlaps are per-axis symmetric: the far side gains one allocated-but-never-read dead row, §2.1). Physically identical placement either way: the wall fine cell always sits inside the coarse wall cell, the buffer inside the wall's other subcell.

**Option B (rejected).** Wall at the INNER ghost row (min: local −1; max: local), buffer at the outer band slot, overlap stays 2 everywhere. Because the wall fine cell must remain at fine global 2·go−1 (min) resp. 2·(go+gs) (max) to keep the link plane fixed, B forces a face-dependent re-anchor: `offset = 2·go + (wall_min ? 0 : 1)`, `local = 2·gs − (wall_min?0:1) − (wall_max?0:1)`.

**B is a bookkeeping isomorphism of A, not a numerical change.** Relabeling the channel's z-axis `(offset, local, ov) = (5, 14, 3) → (4, 15, 2)` shows every physical quantity conserved:

| quantity | option A (z-min wall) | option B |
|---|---|---|
| wall fine cell | local −2 = global 3 | local −1 = global 3 |
| streaming buffer | local −3 = global 2 | local −2 = global 2 |
| substep-1 processed set | locals [−2,15) = globals 3…19 | locals [−1,16) = globals 3…19 |
| substep-2 processed set | locals [−2,14) = globals 3…18 | locals [−1,15) = globals 3…18 |
| z-max C2F destinations | globals 19, 20 | globals 19, 20 |
| streaming read rows | globals 2…20 | globals 2…20 |

Same global rows, same roles, same launches — the ONLY difference is A's allocated-but-dead far-side row per walled axis. B buys nothing numerically; it is a pure refactor of which local indices are labeled "interior" vs "ghost".

**Why A ships despite the extra allocation:** (1) **geometry stays wall-agnostic** — the coarse map carries wall info only after `reset()`→`setupBoundaries()` at SimInit, long after `createAMRBlocks` fixes axes and allocations; B requires wall faces declared at config time (dual source of truth, bridged by a SimInit map-consistency assertion). (2) **The `gs = (local+2)/2` footprint invariant survives** — `buildCouplings`, `checkCouplingMapPattern`'s host replica (the commit-`9a80751` rail), `markAMRInterface`, and the OverlappingAMR writer's `REFINEDCELL` pairing all derive the coarse footprint from `(global_offset, local)`; under B a min-face wall gives `local = 2·gs − 1`, so `(local+2)/2 = gs + 0.5` and every one of those derivations breaks (explicit footprint metadata on the block + T11 exactness re-validation, for zero numerical change). (3) A is validated in-tree (AMR gate 7/7, channel run clean, wall link planes coincident).

**Revisit triggers:** wall-attached blocks proliferating enough that the dead row / per-axis override API hurts, an AA-native wall lane requiring re-derivation regardless, or a careful bookkeeping refactor with rail-conversion tooling.

**Generalization rule (shipped with A).** Wall faces are discovered dynamically from the coarse map at SimInit into a per-block 6-bit mask (per face: count of interior cross-section columns whose matching coarse column on the face-adjacent plane — min: go_a−1, max: go_a+gs_a — is `GEO_WALL`); partial walls (count ∉ {0, cross-section}) and masked faces with `df_overlap_a < 3` are hard SimInit errors (a silent fallback would let the C2F patch overwrite tagged wall columns). Kernel extents and C2F patch construction key off the mask; mask-empty blocks keep the pre-wall launch geometry byte-identical.

---

## 7. Output

### 7.1 VTKHDF OverlappingAMR writer (`include/lbm3d/viz/OverlappingAMRWriter.{h,hpp}`)

Flat per-level layout: Level0 = whole coarse lattice, Level1.. = fine footprints. Global `Origin` attribute + per-level `Spacing`. `vtkGhostType` marks under-footprint coarse cells as `REFINEDCELL` (bit 0x4) so ParaView blanks them under the fine data. `CellData` contains `rho`, `vx`, `vy`, `vz`, `vtkGhostType` per level.

### 7.2 ParaView end-to-end test (`tests/integration/test_amr_paraview.py` driving `tests/integration/amr_paraview_e2e.py` under pvpython)

Runs under pvpython: opens the VTKHDF file, asserts OverlappingAMR structure (level counts, field availability, ghost-type distributions), checks field statistics (rho ≈ 1.0, |vx| < V_max), renders a z-midplane slice to PNG, verifies the screenshot has ≥64 unique colors and ≥30 KB.

---

## 8. Verification infrastructure

### 8.1 Unit tests (pytest: `tests/unit/test_amr_units.py` + `tests/integration/test_amr_paraview.py`)

The gate is fully pytest-native (the retired `tests/run-amr-tests.sh` shell and the two e2e `.sh` wrappers were the last shell launchers; the gate suites run as doctest TEST_SUITEs of the consolidated `test_amr_units_{ab,aa}` binaries):

| Pytest module | Tests |
|---|---|
| `tests/unit/test_amr_units.py` suite `amr_coupling` × {ab,aa} | mock coupling-kernel suite, Tests 1–18 (all active in the default build since the 2026-08-18 CM default flip; strategy-split since commit 14/T15 — the suite compiles an `f2c_strategy_name` token and asserts the branch-conditional expectations natively on BOTH strategies, so the default build runs the Schönherr-arm mean-density expectations and the `F2C_LAGRAVA` build the center-value ones): uniform-field C2F (2 parities), uniform-field F2C (4 parity combos), linear-gradient C2F exactness, mass-conservation F2C (quadratic field), mass-conservation C2F (linear field), nested-geometry geography regression with halo/Lagrava/storability sub-checks (commit `30139d4`), Defect-2 DF-store map guard (Test 7), compact-moment exactness (Tests 8/9; the carve arm Tests 10–13/17 was excised with the carve itself on 2026-08-23), skin-F2C exactness (Tests 14/16) and the `lo = 0` clamp sentinel locks whose authority lives on the Lagrava opt-out branch (Tests 15/18 — they print an explicit deferral line under the F2C_SCHONHERR default; retired nothing); plus the commit-D nesting locks, silent on success (the bit-identity manifest of §7.5 pins these suites' stdout): Test 19 per-pair 2-hop transfer census (absolute substep counters + parent/fine rotations at every call site over 3 cycles on the 3-level chain), Test 20 live-source ordering lock (the mid-cycle fill of the L2 band provably sources L1's live post-substep-A state), Test 21 two-hop kernel composition (C2F doubled interpolation exact; F2C own-8 mean-of-mean on the Schönherr arm) |
| `tests/unit/test_amr_units.py` suite `amr_subcycling` × {ab,aa} | 8 tests: six-step schedule census + parity-at-call-site (Test 1 test_subcycling_schedule — 2 fine + 1 coarse kernel launches + 3 fill launches per cycle per level), time synchronization (t_coarse == t_fine after the 2:1 subcycling), max_level==0 bitwise-identical-to-base-driver, interface-ring freshness (Test 4: every ring cell's macros bitwise-equal a kernels-only reference after one cycle — kernels-only equality is physically unextendable past cycle 1 because the ring streams from the F2C-refreshed skin), conservation hidden-cell exclusion (Test 5: the 216-cell frozen set of the recount excluded from the conservation sums), skin-partition geometry (Test 6: the depth-1 disjoint partition — pushed skin rectangles pairwise disjoint, union == exact depth-1 face shell, empties dropped — 8³/3³/{3,8,8}), footprint minimum (Test 7: per-axis gs ≥ 3 accepted, thinner rejected with the verbatim dual-role message — F3 F-1), generative fill-freshness model over 3 cycles (Test 8), and the 4-cycle parity-chain derivation of the seam's even/odd alternation (Test 9) — Tests 8/9 landed at commit 9/T8 |
| `tests/unit/test_amr_units.py` suite `amr_vtkhdf_writer` × {ab,aa} | VTKHDF file structure, ghost-type tagging, field values; plus the 3-level nesting census (commit D): per-level block grouping/spacing/AMRBox extents and the direct-pair REFINEDCELL census on a 3-level chain (silent-on-success rows like the coupling locks below) |
| `tests/unit/test_amr_units.py` suite `amr_nesting` × {ab,aa} | the amr-nlevel-nesting commits A–C suite: the V-suite reject corpus with verbatim messages + advisory-tier warnings, 3-level chain creation (exact parent-frame offsets/locals, per-level lattice scaling), the 3-level markAMRInterface map census, the 2/3-fine-level schedule censuses of the advancePair recursion (parity locks at every launch, 2^L substeps per level per cycle), the 3-level conservation smoke |
| `tests/integration/test_amr_paraview.py::test_amr_paraview_e2e` | ParaView >= 6.0 fetch + field stats + slice render (6.1 in the test environment; skipped if pvpython absent — the retired shells' exit-77 convention) |
| `tests/integration/test_amr_paraview.py::test_amr_paraview_e2e_nesting` | the 3-level nesting arm (commit D, target #10): the dedicated mock `test_amr_nesting_sim` writes a 4-level (0..3) nested-frame VTKHDF; ParaView asserts 4 levels × 1 block, per-level cell/REFINEDCELL censuses, no EXTERIORCELL-only blanking at any level, rho finite everywhere, fetch + slice render (skipped if pvpython absent) |

Current status: **all 10 gate targets pass** (4 suites × 2 patterns + 2 e2e arms) on both AA and AB streaming patterns.

### 8.2 Between-property metric (`tests/between_metric.py`)

Compares every cell of the AMR composite against a bracket formed by independent uniform-coarse (64³) and uniform-fine (128³) reference runs (γ=3, eps_rho=1e-5, eps_vel=2e-5). Reports per-cycle total/max violations, per-face slab breakdowns, and top-100 violator coordinates. Self-test mode (substitutes AMR = fine reference) passes trivially, confirming harness consistency. The acceptance footprint constants were **re-pinned 32/64 → 33/62** at conversion commit 4 alongside the fine-interior re-anchor (the excluded face-adjacent rows were the old C2F-fill-error band); on the converted HEAD the flipped-default bracket sits far below the old-code window for the first ~3 cycles and exits it upward from cycle ~5 — the script's RESULT: FAIL there is the era-consistent class, and per commit 4's body the bracket is a diagnostic instrument, not a physics verdict (commit 15's between/ evidence). §8.2-level companion: `tests/interface_seam_metric.py` (face-mean signed Δ of rho across a chosen interface row pair over per-iteration VTKHDF frames; the committed seam/bias instrument of the conversion probes, with `--fine-row 0 --coarse-row 16` encoding the re-paired pairing of contract §5).

### 8.3 Conservation monitoring

`State_AMR::AfterSimUpdate` logs `"AMR conservation: mass = {total_mass}"` at the PRINT interval. Mass is computed as a volume-weighted host-side reduction over all blocks. Current measurement (post-P0.2 corrected metric — D.5, 2026-08-16): mass = 2.621440e+05 invariant across all 11 outputs (|drift| ≤ 3.8e-7, one print ulp; the pre-fix metric printed 2.949119e+05 through the double-counted footprint — §9.1).

---

## 9. Measured residual and falsification matrix

### 9.1 Current acceptance state

- **Unit suite**: 7/7 green (both patterns).
- **sim_AMR Taylor-Green** (64³ coarse + 32³-coarse-footprint fine, Re=100, 10 coarse cycles):
  - between-metric cycle 10 (trilinear v7, old harness): **685,713 violations** (rho 463,276 max 2.66e-4; vx 19,898 max 2.52e-4; vy 19,824 max 2.56e-4; vz 182,715 max 2.74e-4).
  - between-metric cycle 10 (3rd-order Lagrange C2F, corrected harness): **188,353 violations** (see §4.1, §12.7 item 6 for the upgraded baseline).
  - 0 frozen placeholder cells; mass exactly conserved.
  - Residual character: smooth bulk drift (~3e-4 rho, ~2.5e-4 velocity) distributed across the domain, concentrated in the fine-boundary band (2–4 fine cells inside the interface); no localized spikes or frozen cells. The boundary-band concentration is explained by C2F shadow injection (§9.2.1); the bulk drift and pressure-offset non-decay by the one-way clamp (§9.2).
- **sim_AMR Taylor-Green corrected baseline (P0.3, 2026-08-15)** — measured on the merged Phase-0 tree (corrected metric with `GEO_NOTHING` hidden-cell exclusion, corrected F2C polarity with the Lagrava 4×4×4 projection as the live default, Defect-2 DF-store guard); default build (`TNL_LBM_C2F_STRATEGY`/`TNL_LBM_F2C_STRATEGY` unset → 3rd-order Lagrange C2F + Lagrava F2C, current ring + full-footprint feedback), AB streaming, single rank, float precision, two sequential runs of the same binary (`build-p0a/sim_AMR/sim_AMR`) on the acceptance configuration (64³ coarse + 32³-coarse-footprint fine, Re=100, 37 coarse iterations over t=0.506 s):
  - bracket cycle series (total violations; run1 = run2 exactly): c0 0; c1 62,715; c2 87,828; c3 129,040; c4 220,163; c5 **317,491**; c6 391,993; c7 487,375; c8 554,827; c9 583,334; c10 **605,583**.
  - per-channel violations (c5 / c10): rho 221,657 / 383,152; vx 649 / 2,535; vy 648 / 2,353; vz 94,537 / 217,543 (rho ≈ 63 % and vz ≈ 36 % of the totals).
  - conservation drift (per-iteration prints, definition `.omo/evidence/p03/drift-definition.md`): total mass 2.621440e+05 with |drift| ≤ 3.8e-7 (one print ulp) across all 38 samples; per-level KE (unweighted diagnostic) decay ratio vs analytic `exp(−4νk²t)` at t=0.506 s: level 0 0.9932, level 1 0.9925 (at t=0.260 s: 0.9926 / 0.9961).
  - noise band NB = **0** — the two runs are bitwise identical in every written field dataset (both levels, all 11 cycles); the full run-to-run deviation of every per-channel count at every checkpoint is 0, so gates A/B use a zero band (ties require exact count equality).
  - **this row supersedes the historical 188,353 @ cycle 10 for all later gate comparisons** (that number predates the corrected metric's hidden-cell double-count fix, the corrected F2C filter polarity, and the Defect-2 guard, and its original reference set no longer exists).
  - provenance: git `9ba1e947eff3bb7d9cac77e9c5723a34f902620b`; binary sha256 `7335cb833d51235b7cad32b6643661aa7118289b761a76a04f761170cdd94cbd`; run dirs `results_sim_AMR_res01_np001_p03_run{1,2}`; uniform references regenerated on the same tree as `results_sim_NSE_ref_{coarse,fine}` (validated: cycle-0 AMR Level0 vs coarse reference bitwise-identical, analytic TG initial-condition max deviation ≤ 2.1e-7); evidence `.omo/evidence/p03/`.
  - cadence note: `sim_AMR.cu` hardcodes `physFinalTime = 0.5` and `cnt[OUT3D].period = 0.05`, so output-frame indices cap at cycle 10 — of the plan's 5/10/15/20 checkpoint set, only cycles 5 and 10 exist for this binary (any arm built from it); extending the decay series beyond t=0.506 s is a sanctioned-change decision outside P0.3 scope.
- **Experiment A grid arms (A.3, 2026-08-16)** — same acceptance configuration, corrected metric, and reference sets (`results_sim_NSE_ref_{coarse,fine}`) as the corrected baseline above; default feedback (ring + full-footprint F2C), AB streaming, single rank, float precision, one sequential run per arm; A1 = compact-moment C2F uncarved (`build-a1/sim_AMR/sim_AMR`, `-DTNL_LBM_C2F_STRATEGY=C2F_COMPACT_MOMENT`), A2 = compact-moment C2F + carve (`build-a2/sim_AMR/sim_AMR`, additionally `-DC2F_CARVE=ON`); L-ctrl is the corrected baseline above (P0.3, NB = 0 → ties require exact count equality):
  - bracket cycle series (total violations), A1: c0 0; c1 91,961; c2 216,780; c3 301,786; c4 410,534; c5 **489,133**; c6 517,850; c7 526,674; c8 516,008; c9 542,512; c10 **613,183**.
  - bracket cycle series (total violations), A2: c0 0; c1 113,445; c2 265,699; c3 447,692; c4 755,286; c5 **1,123,286**; c6 1,431,564; c7 1,900,985; c8 2,418,124; c9 2,802,112; c10 **3,346,370**.
  - per-channel violations (c5 / c10), A1: rho 409,622 / 450,761; vx 623 / 1,972; vy 644 / 2,028; vz 78,244 / 158,422.
  - per-channel violations (c5 / c10), A2: rho 708,405 / 1,669,400; vx 25,904 / 259,322; vy 25,771 / 266,679; vz 363,206 / 1,150,969.
  - conservation drift (per-iteration prints, same definition as the corrected baseline above): all three arms mass-invariant (max |drift| ≤ 3.81e-7, one print ulp; least-squares fitted drift slopes print-precision-dominated, no resolved signal); per-level KE (unweighted diagnostic) decay ratio vs analytic `exp(−4νk²t)` at t=0.506 s: L-ctrl 0.9932 / 0.9925, A1 0.9865 / 0.9760, A2 1.7739 / 1.4999 (A2 grows energy instead of decaying); at t=0.260 s: 0.9926 / 0.9961, 0.9897 / 0.9860, 0.9935 / 1.0114.
  - gate note: decision recorded in §9.3 — CM + carve falsified (gate-A branch 3: carve did not repair the density fit); A1 is the attribution probe only, never a decision input on its own.
  - provenance: git `9ba1e947eff3bb7d9cac77e9c5723a34f902620b`; A1 binary sha256 `545f8c11daad476d5a1ef17d875963dbcf8612a2e964b511e8b5aed514ee64d6`; A2 binary sha256 `ec0d7af87af6c76390c8b832b8dab0e6188ee26600cb75de59e992f724c2ced3`; results dirs `results_sim_AMR_res01_np001_a3_{a1,a2}`; sanity: formal runs reproduced A.1's smoke runs exactly (full metric logs diff-clean); evidence `.omo/evidence/a3/`.
- **Experiment B feedback arm (B.6, 2026-08-16)** — same acceptance configuration, corrected metric, and reference sets (`results_sim_NSE_ref_{coarse,fine}`) as the corrected baseline above; AB streaming, single rank, float precision, one formal run; B-on = `F2C_SKIN_ONLY=ON` (`build-bon/sim_AMR/sim_AMR`: the 6 disjoint one-cell-deep skin rectangles replace the full-footprint interior F2C patches, the ring-F2C launch is skipped, and the F2C filter window clamps to the fine interior `lo = 0`; default 3rd-order Lagrange C2F); B-off is the corrected baseline above (P0.3) verbatim, NB = 0 ⇒ ties require exact count equality:
  - bracket cycle series (total violations), B-on: c0 0; c1 84,115; c2 171,091; c3 236,232; c4 298,672; c5 **327,256**; c6 320,684; c7 290,019; c8 251,265; c9 240,735; c10 **220,737** (B-off series repeats above: monotone growth to 605,583; B-on peaks at c5 then decays through c10, crossing the baseline between c5 and c6; c10 = −63.6 % vs B-off).
  - per-channel violations (c5 / c10), B-on: rho 250,313 / 105,902; vx 837 / 1,387; vy 667 / 1,289; vz 75,439 / 112,159.
  - conservation drift (per-iteration prints, same definition as the corrected baseline above): mass invariant (max |drift| ≤ 3.81e-7, one print ulp; least-squares fitted slope −9.52e-07 1/s ≡ B-off, print-precision dominated, no resolved signal); per-level KE (unweighted diagnostic) decay ratio vs analytic `exp(−4νk²t)` at t=0.506 s: B-on L0 0.99120 / L1 0.99845 (B-off 0.99316 / 0.99253 — no energy injection in either arm); at t=0.260 s: B-on 0.99191 / 0.99892 (B-off 0.99264 / 0.99612); fitted decay-rate deviations vs `α = 4νk²`: B-on +1.0789 (L0) / +0.1281 (L1), B-off +0.7848 / +1.0792; KE1 final (raw sum) 1.982707e+00.
  - Dirichlet channel diagnostic (informative, non-blocking): the B.7 developing-channel variant (`sim_AMR/sim_AMR_channel.cu`, refinement slab in the developing region) shows a quasi-steady transverse recirculation plus backflow bubble inside the slab and a ~60 % downstream-centerline deficit vs the (mutually grid-converged) uniform references in BOTH arms alike — arm-common, not a differential defect of the skin path; channel mass is BC-clamped and therefore arm-identical to 1.4e-4 relative (the Dirichlet Qi-drift masking works as designed); profiles and station tables at `.omo/evidence/b67/channel-diagnostic.md`.
  - gate note: changes 2+3 live per gate-B (user ruling A, 2026-08-16); the c1–c5 transient overshoot caveat (+3.1 % at c5) and its mechanism are recorded in §9.3.
  - provenance: git `9ba1e947eff3bb7d9cac77e9c5723a34f902620b` (+ uncommitted redesign stack); B-on binary sha256 `00d0a046e978dc9ecafaedc8a93f9ebaaf96ecb22a610851f017e1cc867763eb`; results dir `results_sim_AMR_res01_np001_b67_bon`; sanity: the formal metric log is diff-clean (exit 0) against all three B.1 activation smoke metric logs (`.omo/evidence/b13/metric-bon-run{1,2,3}.log`); evidence `.omo/evidence/b67/`.
- **Phase C storage_overlap=1 (C, 2026-08-16)** — same acceptance configuration, corrected metric, and reference sets (`results_sim_NSE_ref_{coarse,fine}`) as the corrected baseline above, with the gate-B winner's semantics (skin-only F2C, default 3rd-order Lagrange C2F, F2C window `lo = 0` clamp) and fine-block `storage_overlap` set explicitly to 1 (`initLevelLattice`: the outer ghost `−2` layer's only consumer was the ring-F2C filter window deleted by gate B); AB streaming, single rank, float precision, one run per side — `build-c1/sim_AMR/sim_AMR` (overlap 1) vs the sha-pinned copy of the overlap-2 B-on binary (`.omo/evidence/c/reference/sim_AMR-bon-ov2`):
  - bitwise gate (Phase-C precondition for proceeding: same kernels, same indices, only allocation depth changes): field data BITWISE identical between the overlap-1 and overlap-2 runs — 132 arrays = 11 cycles × 2 levels × 6 datasets (`rho`, `vx`, `vy`, `vz`, `vtkGhostType`, `AMRBox`), 0 bit-differences (`.omo/evidence/c/gate-c3-compare.log`); bracket counts equal the B.6 series exactly at all cycles (c1 84,115; c5 327,256; c10 **220,737** — `.omo/evidence/c/metric-c1-ov1.log` vs `.omo/evidence/b67/`).
  - memory/throughput (plan check "≈ 8.6 % block storage at 64³ fine, half the C2F fill work per launch"): fine-block storage −26,936 of 314,432 cells = **−8.566 % per scalar array** (68³ → 66³ allocated extent; ≈ 12.7 MB per fine block host+device over all shared-indexer arrays); C2F fill work 52,288 → 25,352 cells/round (**−48.5 %**; launch-shape fingerprints `[2,68,68]` → `[1,66,66]` etc. at unchanged fill cadence); walltime 13.7 → 12.6 s on the 37-iteration acceptance run; accounting at `.omo/evidence/c/memory-lups.md` (the block-size-optimizer launch extents are the runtime proof the allocation changed — the sim's `CPU/GPU RAM for DFs` budget print sums local volumes only and does not see overlap cells).
  - audit: 17 ghost-`−2` consumer candidates enumerated, 0 shipping consumers of the removed layer (the retired ring read was the only one); the deepest shipping F2C window reach is exactly the one retained ghost cell (max-side skin windows; `.omo/evidence/c/ghost-2-audit.md`); non-MPI smoke (C.4) outcome-equal pre/post (same compile-time MPI-redeclaration signature; the edit is invisible in non-MPI translation units).
  - provenance: git `9ba1e947eff3bb7d9cac77e9c5723a34f902620b` (+ uncommitted redesign stack); c1 binary sha256 `b7e5a3e44e6a8e1cbb4e6991bdacf9ebd40cc2986228960ca851dcbe363a61eb`; results dirs `results_sim_AMR_res01_np001_c1_{ref_ov2,ov1}`; evidence `.omo/evidence/c/`.
- **D.1 ring-F2C hard-delete (D.1, 2026-08-17)** — the ring fine-to-coarse channel already retired by gate B was hard-deleted after gates B and C passed: the ring launcher (`launchFineToCoarseTransfers`), SimUpdate step 7's `#ifndef F2C_SKIN_ONLY` ring launch, the full-footprint `#else` interior-patch branch, the ring-era test harnesses, and the ring-era prose — 29 items enumerated before cutting (`.omo/evidence/d1/delete-map.md`). The `F2C_SKIN_ONLY` experiment define was retired WITH the deletion: the skin became the sole surviving fine-to-coarse channel, so its semantics (skin interior patches, `lo = 0` filter-window clamp) are unconditional — **the zero-define default build IS the shipping configuration** (open decision 4 realized by deletion; no CMake flip was needed: per gate A the 3rd-order Lagrange C2F was already the CMake default, so the `TNL_LBM_C2F_STRATEGY` default stands).
  - post-deletion acceptance (zero-define `build-final`, the shipping default): three sequential runs bitwise reproduce the B.6/C state exactly — field data BITWISE identical to `results_sim_AMR_res01_np001_c1_ov1` (132 arrays = 11 cycles × 2 levels × 6 datasets, 0 bit-differences; `.omo/evidence/d1/gate-d1-compare.log`) and metric logs 3× md5-identical (`770609d9d89918c542d84097751d3c93`) with bracket counts == B.6 at all cycles (c1 84,115; c5 327,256; c10 220,737).
  - suite green incl. subcycling: 6/6 AMR mock binaries exit 0 (coupling/subcycling/vtkhdf × {ab,aa}), `test_cpp_units` 34/34 cases, 780/780 assertions, ParaView e2e all checks PASS (`.omo/evidence/d1/suite-final.log`, `cpp_units.log`, `e2e-run.log`); the subcycling ring-freshness lock (Test 4: every ring cell's macros bitwise-equal the kernels-only reference after one cycle, plus the placeholder probe) stays green.
  - provenance: evidence `.omo/evidence/d1/`.

### 9.2 Root cause of residual: one-way clamp

The F2C kernel reads fine **ghost** cells, which were C2F-filled from coarse interpolation. The ring values are therefore a `(1/8, 3/4, 1/8)³` binomial smoothing of coarse neighborhood data — no fine-interior information ever reaches the coarse lattice. The fine patch is a one-way clamp on a coarser boundary stencil. Velocity errors diffuse away (momentum has a viscous decay channel); rho offsets do not (pressure offsets have no decay channel in the one-way scheme).

**Attempted fix — ghost exclusion (failed).** The direct consequence of this framing was implemented and measured: the Lagrava F2C filter with its per-axis window clamped to the fine interior (`lo = 0` instead of `-ov`, excluding index `< 0` ghost cells, mirroring Musubi's GhostFromFiner which reads fine fluid elements only). Both the default 3rd-order Lagrange C2F and the compact-moment C2F diverged to NaN before cycle 5. The theory fails for this architecture: for an exterior ring cell, its entire 2×2×2 subcell block IS ghost data in the fine halo, and those ghost subcells are the physically correct boundary state — excluding them replaces the boundary data with a far extrapolation from fine-interior cells and disconnects the ring from the interface. Musubi's interior-only reading does not transfer because Musubi's ghost cells are passive streaming buffers, whereas here the exterior-ring ghost subcells carry the C2F boundary state the collision-active ring needs. The 1/8 box average over the coarse cell's own subcells (the `-DF2C_BOX_AVERAGE` opt-out since the P0.1 filter-polarity correction — D.5, 2026-08-16) and the ghost-inclusive Lagrava filter window both remain sound; any future "true GhostFromFiner" semantics must project from fine *interior* fluid onto the coarse ring explicitly, not by stencil exclusion.

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

**Conversion-era update (2026-08-22, branch `feat/amr-schonherr-ch7`).** The hidden-cell-injection channel this § describes is **closed at valid faces** under the converted band registration: both fine destination rows now share the SIMULATED `{c=−1, c=0}` source pair (§3.1 vertex rule — the reactivated c=0 ring row replaced exactly the covered-skin interior read that bracketed the frozen hidden cell), so the C2F stencil's nominal windows touch no frozen cell at any valid face, the |t_rel| = 0.75 carve extrapolation is demoted to wall/degenerate-only sites (§4.1), and option (a) above is effectively what the conversion built (by re-registration, not by extrapolating). The freeze + interior-F2C design itself survives unchanged below the ring (depth-1 c=1 skin refilled once per cycle end; deep core unwritten); the v9-era lesson above stands as the design rationale for keeping it. AA-pattern Defect-1 remnants at frozen cells are unaffected by the registration and remain deferred (§10.1).

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
| **CM+carve (A2)** | **compact-moment C2F with carve off covered cells** (Experiments 1+5, gate A, 2026-08-16) | **3,346,370**‡ | **+452.6%** | **FALSIFIED** |
| **changes 2+3 (F2C_SKIN_ONLY, B-on)** | **skin-only F2C interior + ring-F2C launch removal + F2C window `lo = 0` clamp** (Experiments 2+3, gate B, 2026-08-16) | **220,737**§ | **−63.6%** | **VERIFIED LIVE (user ruling A)** |

† v9 measured under a corrected between-metric harness (proper fine-block placement at `[32:96]³`, matching-viscosity uniform references). v7 under the same corrected harness: 405,761. The original harness numbers (1,265,680 / 685,713) used the old /tmp/opencode script with a different composite builder and reference set; the relative ordering v7 > baseline > v9 is preserved.

‡ CM+carve measured against the P0.3 corrected baseline (§9.1: 605,583 @ cycle 10, NB = 0) on the same acceptance configuration and reference sets. Gate-A branch-3 arithmetic: A2 rho violations ≥ A1 rho violations at every cycle 1..10 (c5 708,405 vs 409,622; c10 1,669,400 vs 450,761) → the carve did not repair the density fit; A2 totals exceed baseline + NB at both checkpoints (1,123,286 vs 317,491 @ c5; 3,346,370 vs 605,583 @ c10). Corroboration: A2 per-level KE decay ratios vs analytic `exp(−4νk²t)` exceed 1 (energy growth instead of decay): t=0.506 s level 0 1.7739, level 1 1.4999 (mass stays invariant at one print ulp in all arms, so the mass-drift criterion did not fire). The carve implementation itself is fp-exact — `tests/unit/test_amr_coupling.cu` Tests 8–13 (nominal, 1-axis carved, 3-axis corner-carved windows, and both degenerate fallbacks; classification "physics, all exact") — so the falsification is extrapolation-distance physics, not a code defect. The uncarved control A1 (compact-moment C2F without carve: 613,183 @ c10, within +1.3 % of baseline, but +54 % @ c5) is the attribution probe only, never a decision input on its own. Disposition per the gate-A decision map: C2F keeps the 3rd-order Lagrange default for the direct interface path; carve-on-Lagrange NOT applied (2-cell shift cost); CM + carve wiring retained only as non-default compile-time experiment options (`TNL_LBM_C2F_STRATEGY=C2F_COMPACT_MOMENT`, `C2F_CARVE=ON`).

§ changes 2+3 measured against the P0.3 corrected baseline (§9.1: 605,583 @ cycle 10, NB = 0 ⇒ ties require exact count equality) on the same acceptance configuration and reference sets; B-on = `build-bon` (`-DF2C_SKIN_ONLY=ON`, default 3rd-order Lagrange C2F). Gate-B arithmetic: the bracket leg is SPLIT — c5 327,256 exceeds baseline + NB by +9,765 (+3.1 %, a strict fail at that checkpoint under the leg's letter), while c10 220,737 vs 605,583 is −384,846 (−63.6 %); the series is non-monotone (peaks at c5, then decays, crossing the monotonically-growing baseline between c5 and c6); rho c5 250,313 (+28,656), c10 105,902 (−277,250). CAUTION: the c1–c5 transient overshoot is inherent to the change — the ring now receives skin-fed data one streaming step later instead of the retired instantaneous ring-F2C injection; steady state is the improvement regime. KE/drift legs: per-level fitted KE decay-rate deviations within 2× of L-ctrl's own deviation per level (L0 1.0789 ≤ 2× 0.7848 = 1.5696; L1 0.1281 ≤ 2× 1.0792 = 2.1584); KE ratios vs analytic at t=0.506 s L0 0.99120 / L1 0.99845 (no energy injection); mass-drift slopes identical to baseline (−9.52e-07 1/s, ulp-level). The fallback chain never fired: its trigger condition (a thin-channel interface defect distinguishing the arms) is absent — the B.7 Dirichlet channel diagnostic shows the downstream deficit and in-slab recirculation are arm-common (present in B-off and B-on alike; `.omo/evidence/b67/channel-diagnostic.md`). The split bracket leg maps to no branch of the gate-B decision map, so acceptance came by ruling instead of arithmetic: **gate B resolved by user ruling A (pass-in-substance), 2026-08-16** — changes 2+3 (skin-only F2C feedback + ring-F2C removal + `lo = 0` clamp) are the accepted design. Consequence: the ring-F2C path is deleted by outcome (hard-delete of the retired code path deferred to D.1), and Phase C is enabled (D.1/D.4 unblocked by this ruling).

‖ D.1's hard-delete record is appended at §9.1 above (29 items; zero-define default = shipping configuration; acceptance 3× md5 `770609d9d89918c542d84097751d3c93` == B.6 bitwise).

¶ **Post-gates D.4 — AA-pattern revisit under the accepted redesign** (analysis only, zero product changes; full record `.omo/evidence/d4/aa-revisit.md`, 2026-08-16): verdict **AA-likely-small-effort**. Every coupling parity site verified parity-aware in code (C2F coarse read + always-twisted ghost store; F2C fine read + oriented skin store; driver parity threading; the ring provably coarse-kernel-only; adjoint findings scoped out — `sim_adjoint` is CMake-excluded from AA builds). Probe: 3/3 mock binaries, 9/9 runs exit 0 on a global-AA build (PASS lines byte-identical to the pre-existing AA mock evidence). Exactly ONE live defect remains — the **AA Defect 1 remnant**: frozen coarse cells (the F2C-written skin and the twist-init deep core) do not track coarse-step parity, but the two per-cycle C2F fills read them with the just-run step's parity assumption: even coarse iterations see 27/27 direction-opposite slots at the skin; odd iterations see 18/27 at the skin (the 9-slot streamed family is fresh and correct) plus 27/27 at the twist-init deep core. The flipped frozen samples enter the fills at first-order weights (+35/128 or −7/128 per axis at skin nodes, −5/128 at the first core node) — a deterministic velocity-sign-flipped systematic alternating every cycle (no race; the redesign shrank the class vs B-off's full-footprint window). Candidate fixes (analysis only, none implemented): (a) always-twisted AA coarse store (delete the natural branch — its documented consumer is vacuous at frozen `GEO_NOTHING` cells), (b) a twist-normalization pre-pass over the 6 skin rectangles before fill #1, (c) a Lagrange-side frozen-cell carve in C2F — flagged as a **user/design decision** (the plan's A3 note marks the 2-cell shift as outside sanctioned stencil use). Gap list: (i) the missing gate — a failing-first two-cycle closed-loop marker mock (F2C store cycle N−1 → coarse step N → C2F fill N) to prove the class in-situ and then prove any fix; (ii) the `defs.h` MPI-corner AA race TODO (precondition for any multi-rank AMR-AA; inert in v1 single-rank); (iii) the hi-side ghost-sliver read asymmetry (twisted refill under AA vs stale `df_out` under AB) stands as a documented deterministic caveat, not a defect. AA stays off by default until the remnant is fixed and gated.

¶ **Post-gates D.3/D.2 — conservation fallback evaluation + Guzik caveat** (2026-08-16): D.3 evaluated **NOT TRIGGERED** — total mass is invariant at one print ulp (max |drift| ≤ 3.81e-7) in ALL four measured configurations (L-ctrl, A1, A2, B-on; least-squares slopes print-precision-dominated), and the B.7 Dirichlet channel finding is arm-common, so no constrained-least-squares conservation fallback is needed. D.2 caveat (standing flag, `.omo/evidence/d2/guzik-caveat.md` + `:coverage-audit.md`): Guzik's constrained-least-squares stencil derivations cover **D3Q19 only** (corner ‖e‖₁ = 3 velocity directions explicitly deferred), and NO least-squares machinery exists anywhere in this scheme — the C2F carve and the F2C skin clamp are shifted-window Lagrange evaluations at a fixed evaluation point (runtime-computed weights), not a `min ‖Aξ−b‖²` solve with conservation/corner constraints. A dedicated D3Q27 corner-pattern derivation (source geometry + per-corner-direction constraint bookkeeping) is required before any multi-face constrained-LS use on D3Q27. Mock-envelope caveat recorded: carve/skin one-sided behavior is fp-exact-verified at faces, edges, and corners (Tests 8–18) only within the mock envelope (single rectangular footprint, separable fields) — no D3Q27-validity claim beyond it.

¶ **Default flip — compact-moment + carve becomes the shipping C2F** (user ruling, 2026-08-18; evidence `.omo/evidence/cm_carve_post/`): the interpolation must never source covered (`GEO_NOTHING`) cells — the skin layer is F2C-write / streaming-read only. Post-redesign CM+carve recompute on the acceptance configuration: c5 101,655 / c10 72,697 viol counts — **−68.0 % / −88.0 %** vs the corrected baseline (NB = 0) and **−68.9 % / −67.1 %** vs the accepted B-on arm, inverting the pre-redesign gate-A falsification (+452.6 % @ c10); no c5 transient overshoot; mass invariant at print precision. Implementation (`amr_coupling.h`, mirroring the P0.1 polarity-flip pattern): no-define selects the CM branch with the carve pre-pass active; `C2F_LAGRANGE` opts back to the 3rd-order Lagrange scheme (its 4-node window can read covered cells), `C2F_NO_CARVE` disables the carve, `C2F_COMPACT_MOMENT`/`C2F_CARVE` remain accepted as no-op selectors with a CMake retirement warning for the latter (`TNL_TEST_NO_CARVE` is the test-side escape hatch). Test guards in `test_amr_coupling.cu` mirror production conditions (CM/carve exactness Tests 8–13+17 now run in the default build). Verification on the zero-define default build: all six AMR test binaries green (AB+AA), acceptance run bitwise-identical to the explicitly-defined arm (c10 72,697; full c1–c10 series count-identical). Open legs for a future formal gate revisit: per-level KE decay fit vs analytic (the leg that killed A2), drift-slope fit, arm-repeat bitwise check.

¶ **Interface density bias — seam investigation** (2026-08-19, user-driven debug with per-iteration `--write-dfs` frames; full record `.omo/evidence/interface_bias/seam-investigation.md`): the ParaView-visible f05/f06/rho "discontinuity" at the interface is a real **uniform DC density bias +1.3e-5 (fine > coarse ring), formed within 2 iterations and sustained**, not a startup transient and not stale macros (Σ f_q tracks the rho field to ≤2.4e-7 everywhere). Anatomy: ring LOSES ~0.9e-5 while the frozen-skin/fine-boundary pair GAINS ~0.5e-5 (taper ~4 coarse cells); skin mirrors the fine side to 6e-7. Probe ladder excluded in turn: the skin→ring lag (H9 halves only cycle 1), interpolation order (Lagrange AMPLIFIES the bias to +1.9e-5 — hence not the CM quadratic-density fit), and resolution spacing (signed/abs = 1.00). Verdict: the standing bias is carried by the fill's τ-rescaled **second-order strain-rate content** — `C2F_EQ_ONLY` (debug define) collapses it to a decaying alternating oscillation with no accumulation; CM's projection of non-hydrodynamic modes ≥3 removes only ~30 % (consistent with CM measuring ~30 % better than Lagrange), so the dominant share rides the hydrodynamic strain moments, the trace (τ-rescaled compressional part) being the prime density-pump suspect. Both discriminating probes are now closed: `C2F_DEV_ONLY` (trace-off) matches the control **bitwise** — the non-equilibrium pressure tensor is traceless at the TGV fill sites, killing the compressional-suspect hypothesis — and `C2F_NORM_ONLY` (diagonal deviatoric only) yields **135 % of the control bias (+1.76e-5 vs +1.30e-5)**: the pump is the normal-stress anisotropy channel (C₂₀₀−C₀₂₀ / C₂₀₀−C₀₀₂ cumulants with the (3/2)ω_s rescale), the off-diagonal shear partially cancelling it (~35 %). A post-investigation 20-tc re-verification with the current source (CM+carve+H9 build) reproduced the first run **bitwise** (mass −0.2296387, slope −1.0370e-04 1/s) — a deterministic property of the configuration, not run-to-run noise. The three probe defines (`C2F_EQ_ONLY`/`C2F_DEV_ONLY`/`C2F_NORM_ONLY`, all off by default, acceptance-neutral) are committed as debug tooling; the per-iteration debug cadence (`OUT3D = PHYS_DT`) used for the probes is intentionally NOT committed. Long-run channel attribution (20-tc runs, same configuration; harvest in `.omo/evidence/h9_retry/`): `C2F_EQ_ONLY` conserves mass (+3.1e-05 drift vs the control's −0.2296, slope −4.9e-10 vs −1.04e-04 1/s) but the vortex dies (ke0 6.5e-04 vs 0.415 at 20tc); `C2F_SHEAR_ONLY` (diagonal deviatoric zeroed) likewise conserves (+5.6e-04, positive) with dead physics (ke0 1.3e-03); `C2F_NORM_ONLY` (shear zeroed) keeps the dynamics alive but over-energized (ke0 0.706, ke1 2.196) and leaks 2.3× faster (−52.6 %, slope −2.43e-04 1/s). The diagonal normal-stress channel is simultaneously the leak carrier and load-bearing for the dynamics — channel selection is dead as a fix; the empirically forced direction is a conservation-constrained transfer (Guzik-class constrained moments at the interface).

**Measured verdict of the conversion (2026-08-22 — appended at the close-out; full 20-convective-time decision table in commit 15's body, `1bd158c`; evidence `.omo/evidence/schonherr_conversion/t16/`).** The conversion-era pinned constants above (control arm: final drift **−0.2296387 rel**, slope **−1.0370e-04 1/s**, max |drift| 0.2296; KE at 20 tc ke0/ke0_0 **0.415**, ke1/ke1_0 **0.927**; standing bias **+1.30e-05** under the OLD seam pairing — recorded N/A under the re-paired pairing of contract §5, plan Appendix ¶C) were re-measured on the converted HEAD across the five-arm ladder (TGV res 1, `--convective-times 20`, house defaults, SP, AB pattern, np=1; the pinned harvest `.omo/evidence/h9_retry/harvest_tc20.py` applied as-is; bias = post-formation mean of the re-paired seam-primary signed Δ over iterations 3–10, per-arm 10-iteration probes, `tests/interface_seam_metric.py --fine-row 0 --coarse-row 16`):

| arm | defines | final drift | slope [1/s] | max \|drift\| | ke0/ke0_0 @20tc | ke1/ke1_0 @20tc | seam bias (iters 3–10) |
|---|---|---|---|---|---|---|---|
| section-0 control pin (h9-era) | (old default) | −0.2296387 | −1.0370e-04 | 0.2296 | 0.415 | 0.927 | +1.30e-05 — N/A under re-paired pairing |
| (a) F2C_SCHONHERR default | F2C_SCHONHERR | **+7.549021e-07** | −1.0094e-10 | 2.264706e-06 | **6.580475e-06** | 6.824245e-06 | **−2.8358715121e-05** |
| (b) F2C_LAGRAVA opt-out | F2C_LAGRAVA | **−1.358824e-05** | −8.4657e-09 | 1.358824e-05 | **3.528245e-03** | 3.424864e-03 | **−2.4314957687e-05** |
| (c1) C2F_EQ_ONLY on (a) | +C2F_EQ_ONLY | +1.132353e-06 | −7.7681e-10 | 7.549021e-06 | 6.073860e-04 | 6.476043e-04 | −2.4123981728e-05 |
| (c2) C2F_NORM_ONLY on (a) | +C2F_NORM_ONLY | **+0.000000e+00** | −3.5310e-10 | 1.887255e-06 | 2.666106e-06 | 2.845816e-06 | −2.4122452914e-05 |
| (c3) C2F_SHEAR_ONLY on (a) | +C2F_SHEAR_ONLY | +1.132353e-06 | −6.3557e-10 | 6.794119e-06 | 6.057986e-04 | 6.520640e-04 | −2.8225972300e-05 |

(b) reproduces the T15b/stage-2 control bias digit-for-digit and (a) the T15b arm's; bias amplification |a|/|b| = **1.16630740** (PASS vs the pre-registered ×1.2 T15b abort bound; no across-iteration sign flip on any arm, signed/abs = −1.0000).

**Interpretation (quoted from the decision-table verdict):** (i) The ~23 % mass leak is **closed by roughly five orders of magnitude on every HEAD arm** (max |drift| 2.26e-06 / 1.36e-05 vs the pinned 0.2296) and the leak's channel carrier is dead at HEAD — NORM_ONLY drifts **exactly 0.000000e+00** where its control-era twin leaked −52.6 % of the mass and sustained the vortex. The closure is therefore an **era effect of the converted band registration + six-step cycle + σ-form C2F, NOT of the F2C branch choice**. (ii) The standing seam bias is NOT reduced: it **amplifies ×1.16630740** under the σ=2 F2C (a real but bounded effect within the pre-registered ×1.2 T15b bound), carried by the **shear channel** of the fill's second-order strain content (EQ/NORM-only arms sit at −2.412e-05, SHEAR-only reproduces −2.823e-05 of the full (a) arm's −2.836e-05) — the σ = 2 non-equilibrium injection is ~2× stronger than the old τ-rescaled one, so a bounded amplification of the standing bias is the expected signature. (iii) The vortex does **not** survive at 20 tc on ANY HEAD arm ((a) 6.6e-06/6.8e-06; (b) 3.5e-03/3.4e-03 — vs the pinned control's 0.415/0.927). The C2F arms on BOTH eras show the control-era survival was pump-fed by the interface error, not physics: the control's −23 % leak paired with KE 0.415/0.927, and only its NORM-channel arm could sustain the vortex (leaking −52.6 % of the mass) — the same non-conserving interface pump Qi et al. 2019 report for the sibling Musubi compact interpolation on this exact benchmark — while the control-era EQ/SHEAR arms died at 6.46e-04/1.26e-03 with the leak frozen, the same dead class as the HEAD EQ/SHEAR arms (6.1e-04): without a pumping interface the vortex dissipates to the SP solver's numerical floor (still ~1e8× above the analytic laminar decay at 20 tc on level 0). **Verdict per the pre-registered taxonomy (success = reduction in |drift| AND bias magnitudes WITH vortex survival): NOT met — an honest negative/null result**: negative on the vortex-survival clause, null on any F2C-specific conservation impact (both strategies deliver the closure), recorded per the experiment framing of contract §1 ("null/negative results are recorded, not repaired ad hoc"). The conversion's conservation question answers affirmatively (the interface pump is closed; mass is conserved to ≤ 1.4e-5 rel over 20 tc on every arm), and the F2C_SCHONHERR flip stands (commit 15) under the T15b auto-path — at HEAD it conserves mass ~18× tighter than the Lagrava opt-out (7.55e-07 vs 1.36e-05) with a residual KE floor ~535× lower.

**Attribution-matrix addendum:** the 20-tc fill-channel attribution matrix this investigation used as its prop (committed parallel to this ¶: `.omo/evidence/interface_bias/seam-investigation.md`) gains a conversion-era refresh — the ladder's C2F-arm rows above were compiled with exactly the same EQ/NORM/SHEAR debug-define protocol and are directly comparable with the control-era rows quoted in this ¶, re-harvested under the pinned metric at `.omo/evidence/schonherr_conversion/t16/harvest/harvest_reference_{control,eqonly,normonly,shearonly}_h9era.txt`.

### 9.4 Pre-fix trajectory

| Milestone | Commit | Cycle-10 violations | Key change |
|---|---|---|---|
| Geography corruption | `26e36db` | ~844k (pre-overlap) | Coupling kernel frame alignment, fill ordering, AA twisted writes |
| Frozen placeholder ring | `089e47a` | 1,265,680 | `storage_overlap=2` on fine blocks; F2C covers all ring cells |
| Collision-active ring | `5237b2f` | **685,713** | Ring streams+collides like fluid; F2C overwrites at end of step |

---

## 10. Known limitations of v1

- **Two-way coupling via frozen hidden cells**: the depth-1 c=1 skin cells under the fine footprint are frozen as `GEO_NOTHING` (no stream/collide) and receive the strategy-selected fine state via the interior F2C once per cycle end (§4.2, §5 step 4 — the Schönherr σ=2 transfer by default since commit 15, the Lagrava 4×4×4 filtered average under the `F2C_LAGRAVA` opt-out; the deep c≥2 core stays frozen-unwritten). This eliminates both the one-way clamp (§9.2) and — after the converted band registration — the C2F shadow injection at valid faces (§9.2.1 closing note): the ring cell streams from a frozen fine-injected neighbor (not a diverging shadow), and the C2F kernel's nominal windows read simulated cells only. **Verified correct for AB_PATTERN** (the `defs.h` default, used by sim_AMR). **AA_PATTERN has a deferred defect — see §10.1 and the D.4 revisit record in §9.3 (2026-08-16).** Milestone quantification by era: 220,737 bracket violations at cycle 10 vs the corrected baseline 605,583 (−63.6 %) under the 2026-08-16 acceptance rows (§9.1, D.5); on the converted HEAD (33/62 re-pinned bracket, 2026-08-22) the series sits far below the old-code window early and exits it upward from cycle ~5 — the era-consistent FAIL class of §8.2, not a physics verdict; the conversion-era conservation/KE verdict is the measured decision table at the "Interface density bias" ¶.
- **Static refinement only**: regions fixed at SimInit; no dynamic adaptation.
- **Single-rank, single-GPU**: fine blocks have no same-level MPI neighbors; coupling kernels are CUDA-only.
- **Multi-level nesting (target 5 levels)**: implemented by the amr-nlevel-nesting plan (commits A–G, 2026-08-27/28; the full multi-level chapter is §13) — the v1 `level > 1` reject is REPLACED by the nesting validation V-suite (V1–V10: ascending file order, unique containment, telescoping gaps, sibling separation, wall-shared chains) in `createAMRBlocks`; `block.global_offset` is normalized to the immediate parent frame (`amrParentFrameOrigin/amrFineOffset/amrFineLocal`, exact divisions); the schedule is the `advancePair` Berger–Colella pair recursion driven by per-level cumulative substep counters (`LBM::totalSubstepCount`), bit-identical at `max_level == 1` (the census table of §5 is the `max_level == 1` reduction); `markAMRInterface`'s cell-tagging rule needed no change (verified per-pair, incl. across the parent-frame offset conversion; the commit-B fix was confined to its dir-array indexing); the `checkCouplingMapPattern` rails were re-anchored to the same frame. Residual v1-scope pins: single MPI rank, single GPU, 2:1 ratio, adjacent-pair couplings only; wall-shared faces at level ≥ 2 are implemented at commit E (the wall chain keys on the immediate parent's map in the parent frame, the R4 wall-pedestal prisms author the frozen rows behind the parent's upward own-8 window, and nested wall sharing is hard-guarded against the `F2C_LAGRAVA` opt-out).
- **Non-Newtonian viscosity with AMR**: excluded.
- **IBM (immersed boundary)**: excluded.

### 10.1 Known defects (deferred — Oracle-verified)

The following defects were identified by an Oracle review of the freeze approach. They are silent in the current test suite (which tests each kernel against its documented contract in isolation, not the F2C-write → coarse-kernel-skip → C2F-read composition) and in sim_AMR (which uses AB_PATTERN with an interior footprint). They must be fixed before the corresponding configurations are exercised.

**Defect 1: AA_PATTERN C2F read of frozen cells is direction-reversed (blocking for AA AMR).**

The interior F2C stores frozen DFs in the **next-substep consume convention** (what the coarse streaming will read — natural if next is even/reflect, twisted if odd/spatial; `amr_coupling.h:452-462`). The C2F kernel reads with the **post-kernel produce convention** (`amr_coupling.h:152-168`: twisted if `coarse_even_iter`, natural otherwise). For normal cells these match because the coarse kernel rewrites them; frozen cells are never rewritten, so they retain the consume convention. Under AA these conventions are exactly opposite at **both** parities, yielding `f_used[q] = f_real[opp(q)]`: density survives (Σf_q invariant), but momentum is sign-flipped and non-equilibrium stress is mirrored, injected at 0.25 weight into every inner-ghost C2F fill, every cycle. Compounding this, on odd AA cycles the ring cells' `postCollisionStreaming` (`streaming_AA.h:61-90`) clobbers some surface-frozen slots with ring post-collision data before C2F reads them. AB is unaffected (single natural convention; C2F reads fine data that is 2 cycles stale but orientation-correct).

Fix: (a) in `cudaAMR_CoarseToFine`'s `read_coarse_df` (`amr_coupling.h:152-168`), branch on `coarse_SD.map(cx,cy,cz)==GEO_NOTHING` and read with the inverted parity convention; (b) skip ring→frozen stores in odd `postCollisionStreaming` using the allocated-but-vestigial `dinterface_dir` bitmask (`amr_decomposition.h:288-329`, `bc.h:499-505`), which exists for exactly this purpose.

(D.5, 2026-08-16 — status update, body kept as the original Oracle-review record: the D.4 revisit under the accepted redesign verified every coupling parity site parity-aware and localized the surviving remnant to the frozen skin/core cells' orientation-vs-parity mismatch at C2F reads, with different candidate fixes; see the D.4 record in §9.3.)

**Defect 2: F2C has no coarse-map guard on DF writes (latent for boundary-touching footprints).**

`markAMRInterface` deliberately preserves physical BC tags (walls/inflows) under the footprint — those cells keep `GEO_WALL`/`GEO_INFLOW` rather than becoming `GEO_NOTHING`. But `cudaAMR_FineToCoarse` writes DFs with no coarse-map check (the map guard at `amr_coupling.h:467` covers macros only). A wall cell under the footprint gets its DFs overwritten with fine-averaged fluid data each cycle; under AB's pull scheme, adjacent coarse fluid cells stream from it as if fluid — the no-slip wall is bypassed wherever a footprint covers it, contradicting the documented "physical boundary conditions survive" design intent.

Fix: skip the DF write unless `coarse_SD.map(x,y,z) == GEO_NOTHING` — ~3 lines in the kernel (`amr_coupling.h:440`).

(FIXED 2026-08-15, Phase-0 item 4 — DF and macro stores are skipped unless the target cell is `GEO_NOTHING`/`GEO_AMR_INTERFACE`, locked by `test_amr_coupling` Test 7; this entry stands as the original finding. D.5, 2026-08-16.)

**Minor concerns (non-blocking):**

- **AB C2F 2-cycle staleness**: under AB, C2F reads `df_out` of cycle N+1, which after rotation is the physical buffer *not* written by cycle N's interior F2C — C2F sees fine data from cycle N−1 (2-cycle lag, orientation correct). Not a correctness break; a real asymmetry vs. the 1-cycle freshness the design narrative implies.
- **Conservation metric double-counts the footprint**: `computeConservationStats` (`amr_state.h:386-423`) sums frozen coarse cells (weight 1, fine-injected macros) *plus* fine interior cells (1/8 weight). It is a self-consistent drift metric, not exact physical conservation; v7 had the same double-count, so the comparison is apples-to-apples.
- **Overlapping/face-sharing same-level footprints**: `createAMRBlocks` has no overlap check; overlapping footprints produce duplicate interior patches (nondeterministic write order). Face-sharing footprints expose an order-dependence where a shared cell stays collision-active ring while another block's interior F2C overwrites it. v1 (single fine block) is unaffected.

---

## 11. Code references

| File | Role |
|---|---|
| `include/lbm3d/d3q27/amr_coupling.h` | Coupling kernels: `cudaAMR_CoarseToFine`, `cudaAMR_FineToCoarse` |
| `include/lbm3d/amr_state.h` | `State_AMR` driver: `SimInit`, `SimUpdate`, `buildCouplings`, `launchCoarseToFineTransfers`, `launchFineToCoarseTransfersInterior` (the ring launcher `launchFineToCoarseTransfers` was deleted in D.1 — D.5, 2026-08-16), `launchLBMKernelForLevel` |
| `include/lbm3d/amr_decomposition.h` | `createAMRBlocks`, `markAMRInterface` (rings `GEO_AMR_INTERFACE` + hidden cells `GEO_NOTHING`), `allocateInterfaceDirArray` |
| `include/lbm3d/d3q27/bc.h` | `D3Q27_BC_All`: `GEO_AMR_INTERFACE` tag, `preCollision`/`doCollision`/`postCollision` collision-active handling |
| `include/lbm3d/lbm_block.h` / `.hpp` | `LBM_BLOCK`: `storage_overlap`, `initLevelLattice`, `allocateHostData`/`allocateDeviceData` |
| `include/lbm3d/lbm.h` / `.hpp` | `LBM`: multi-level block management, `updateKernelDataForLevel` |
| `include/lbm3d/viz/OverlappingAMRWriter.h` / `.hpp` | VTKHDF OverlappingAMR output |
| `include/lbm3d/viz/OverlappingAMRWriter.hpp` | Writer implementation: per-level AMRBox, ghost tagging, CellData layout |
| `sim_AMR/sim_AMR.cu` | Taylor-Green benchmark with 2-level AMR |
| `sim_AMR/sim_AMR_channel.cu` | Developing-channel (Dirichlet inflow/outflow) 2-level AMR diagnostic — the B.7 artifact; refinement slab in the developing region |
| `tests/unit/test_amr_coupling.cu` | Mock-block coupling kernel unit tests |
| `tests/unit/test_amr_subcycling.cu` | State_AMR end-to-end subcycling tests (incl. Test 4 ring freshness + Tests 6/7 as above) |
| `tests/unit/test_amr_vtkhdf_writer.cu` | VTKHDF writer unit test |
| `tests/integration/test_amr_paraview.py` | ParaView end-to-end visualization tests (2 arms; drives the pvpython check scripts below) |
| `tests/integration/amr_paraview_e2e.py` / `tests/integration/amr_paraview_e2e_nesting.py` | pvpython check scripts of the e2e arms (not pytest modules — ParaView python) |
| `tests/unit/test_amr_units.py` | AMR gate doctest launcher (the four suites of `test_amr_units_{ab,aa}` × patterns) |
| `tests/between_metric.py` / `tests/interface_seam_metric.py` | Bracket (uniform-coarse/fine reference) and interface-seam metrics used by the conversion probes (§8.2) |
| `tests/unit/test_amr_schonherr_registration.cu` / `test_amr_schonherr_exactness.cu` | doctest lock suites of the conversion (registration/census fingerprints; §7.2 exactness incl. the R1 code-family lock) wrapped by `tests/unit/test_cpp_units.py` |
| `tests/unit/test_amr_f2c_schonherr.cu` / `tests/unit/test_amr_f2c_schonherr.py` | F2C_SCHONHERR branch exactness binary + its pytest driver |
| `docs/AMR-schonherr-ch7-target-contract.md` | Normative band/cycle contract of the conversion (band map, fork table, re-pairing, gates, audit appendix) |

### Commits in this debugging arc

| Commit | Type | Description |
|---|---|---|
| `30139d4` | `test` | Coupling geography regression test (fails on baseline) |
| `26e36db` | `fix` | Coupling frame alignment, fill ordering, AA twisted writes |
| `089e47a` | `fix` | 2-cell ghost ring on fine blocks so F2C covers the interface ring |
| `bd227d1` | `docs` | Regenerated Taylor-Green slice render |
| `5237b2f` | `fix` | Collision-active GEO_AMR_INTERFACE ring (−45.8% violations) |

### Commits in the Schönherr-ch7 conversion arc (branch `feat/amr-schonherr-ch7`)

| Commit | Type | Description |
|---|---|---|
| `61c1503` | `docs` | Schönherr ch7 target-band contract — registration ruling + review fixes |
| `1420cd5` | `test` | Per-iteration frame cadence CLI (`--out3d-iter-period`) + interface seam metric + calibration table |
| `021ec1a` | `test` | Registration/parity/conservation census + geometry fingerprint locks (xfail→green at commit 7) |
| `fb4dcce` | `refactor` | Re-anchor fine interior one fine cell inward per footprint face (+ `(local+2)/2` idiom fixes, gs ≥ 3 validation, between-metric re-pin 33/62) |
| `0563a7d` | `feat` | Depth-2 ghost overlap on the re-anchored level blocks |
| `69d41a7` | `feat` | Reactivate footprint outer row into the ring; move F2C skin to depth 1 |
| `83b206f` | `feat` | Vertex-straddling C2F band patches + face-aware launcher + depth-1 skin partition + SimInit map-pattern assertion (stage-1 gate) |
| `4d15252` | `refactor` | Reorder AMR cycle to the Schönherr six-step schedule; retire H9 and the content-no-op BVP refill |
| `5f5514b` | `test` | Lock the six-step schedule; cycle-end fill freshness |
| `24b641b` | `test` | Schönherr §7.2 exactness locks |
| `b44235b` | `fix` | Align CM C2F reconstruction to the printed §7.2 equations (verified no-op; errata + carve docstrings) |
| `611c03b` | `refactor` | Extract shared compact-moment device helpers (bitwise-gated) |
| `cf8ab33` | `feat` | F2C_SCHONHERR — §7.2 σ=2 F2C (opt-in; Lagrava remains default) |
| `58213bd` | `test` | Re-scope F2C suite locks to the strategy split (T15b seam gate PASS at ×1.166 ≤ ×1.2) |
| `1bd158c` | `feat` | Make the Schönherr §7.2 F2C the default transfer (T16 20-tc decision table — the measured verdict quoted at the "Interface density bias" ¶) |

---

## 12. Mathematical grounding and literature references

This section maps the implementation to the mathematical formulations in the AMR-LBM literature, and identifies what the literature suggests for the open interface problems (§9.2, §9.2.1, v9 regression).

### 12.1 Cell-centered volumetric formulation (Chen 1998, Chen et al. 2006)

The implementation uses the cell-centered (volumetric) approach: DFs are point densities at cell centers, and coarse/fine grids do not co-locate. Chen (1998) introduced the volumetric reformulation, treating distributions as *masses* moving between cells of different resolution. Chen et al. (2006) formalized the volumetric grid refinement concept, showing that conservation laws are exactly guaranteed through the volumetric formulation, with the approach independent of the collision step.

(2026-08-22, supersession note: the τ-rescaled scheme this § presents implemented v1's transfers; the SHIPPING defaults are now the σ-form σ = 1/2 compact-moment C2F (since 2026-08-18) and the σ = 2 `F2C_SCHONHERR` F2C (since commit 15, `amr_coupling.h:1429`), both with NO τ-rescale — the equations below survive as the `C2F_LAGRANGE` / `F2C_LAGRAVA` opt-out branches; see §4 for the current internals. The "f_neq_avg" phrasing below predates the strategy split; the volumetric 1/8 conversion argument is unchanged under both strategies.)

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

---

## 13. Multi-level static nesting (the amr-nlevel-nesting arc, commits A–G)

Shipped state (the amr-nlevel-nesting arc commits A–G, 2026-08-27/28, HEAD `5214b01`, plan
`.omo/plans/amr-nlevel-nesting.md`): the v1 two-level design generalizes to **N statically nested 2:1 refinement levels**
(`max_level` ≤ 4, i.e. five lattice levels on the realized target), single MPI rank and single GPU retained from v1, every
band/cycle rule of the contract extending **per adjacent pair, unchanged** (couplings stay strictly adjacent-pair; nothing
crosses two levels). The arc ran under two discipline instruments, both green at HEAD: the **AMR gate 10/10**
(the shell gate of the arc — since migrated entirely to pytest: `tests/unit/test_amr_units.py` runs the six v1 mocks × {ab,aa} + the `test_amr_nesting_*` suite, `tests/integration/test_amr_paraview.py` runs both ParaView e2e arms:
the 2-level arm pinned, the 3-level nesting arm added) and the **bit-identity harness 11/11 verify mode**
(`tests/regression/test_amr_bitidentity.py` — every `max_level == 1` artifact of the mocks and both sims byte-compared
against the committed `tests/regression/amr_ref/manifest.json` after each commit, proving the nesting machinery changed no
v1 behavior).

### 13.1 Parent-frame `global_offset` and the three conversion helpers (commit A)

The region file keeps its v1 convention — **level-0 coordinates for every level** — while `block.global_offset` is
normalized to store the footprint origin in the **immediate parent's** lattice (in v1 this was latent: level 1's parent frame
*is* the level-0 frame). The conversion lives at the single write site (`createAMRBlocks` phase 2) as three pure-integer
helpers — parent-frame origin `go = origin >> (L−1)`, fine offset `offset = 2·(origin >> (L−1)) + 1`, fine interior
`local = 2·(size >> (L−1)) − 2` per axis — all exact by the pre-existing alignment check (origin/size multiples of 2^(L−1)),
and each reduces to the level-1 formula bit-for-bit (the shift is 0), which is the bit-identity-by-construction argument.
Every consumer (`markAMRInterface`, `buildCouplings`, `buildFineWallMasks`, `checkCouplingMapPattern`,
`isShadowedBySameLevelBlock`, the writer's REFINEDCELL pairing) was verified correct unmodified under parent-frame storage —
that was the point of normalizing at the write site.

### 13.2 The nesting V-suite (commit B `4712ebe`)

The v1 `level > 1` reject is removed and superseded by phase-1 validation in `createAMRBlocks` (read-only, before any block
is created; `spdlog::error` + `std::runtime_error` in the existing style). V1–V4 are the pre-existing checks (level in
[1, max_level]; per-axis footprint ≥ 3 parent cells; level-0 containment; 2^(L−1) alignment). The new checks:

- **V5 — ascending level order** (hard error): a level-L region's parent (level L−1) must appear earlier in the region file;
  establishes the level-ascending creation order the wall chain derives through.
- **V6 — parent existence & uniqueness** (hard error): exactly one level-(L−1) region fully contains the child rect (level-0
  coords, V4-exact conversions) — the "orphan" / "ambiguous parent" rejects.
- **V7 — telescoping gap** (hard error): the child sits strictly inside its parent; per-face inset ≥ **2** parent cells on
  every face, except a wall-candidate face which must sit at exactly **s = −1** (its halo row on the parent's wall machinery).
- **V8 — sibling separation** (hard error): same-level regions pairwise disjoint with Chebyshev separation ≥ 2 level-L cells.
- **V9 — gap advisory** (`spdlog::warn` only): 2 ≤ gap < 3 on a non-shared face warns — the parent's upward F2C windows then
  read coupling-authored ring/skin cells instead of plain fluid. The user-decided 2026-08-27 tier: floor 2 is the validity
  bound, warn below 3; the exact-coverage wall-to-wall corner stays valid.
- **V10 — wall-candidate agreement** (hard error): an s = −1 face of a child requires the parent's same axis/side face to
  itself be a wall-candidate, recursively down the chain; map-backed confirmation (a full `GEO_WALL` parent plane) is
  deferred to `buildFineWallMasks` at SimInit.

### 13.3 The `advancePair` Berger–Colella recursion (commit C `cef7ae5`)

`State_AMR::SimUpdate` is a literal recursive pair emission: each pair at level L covers two of L's substeps (widened substep
1 → optional recursion into level L+1's first pair + F2C mid-sync + the mid-cycle C2F fill → interior substep 2 → optional
second pair + F2C end-sync), so **level L performs exactly 2^L substeps per coarse step** (correct Berger–Colella). Per-level
**cumulative substep counters** (`LBM::totalSubstepCount`) drive `updateKernelDataForLevel(L, s_L)` — an absolute setter on
`substep % 2` (parity) and `substep % DFMAX` (DF rotation; DFMAX = 1 under AA, so the rotation is identically a no-op there)
— making the cumulative invocation equivalent to v1's positional 0/1, and the `max_level == 1` reduction **byte-identical**
to the pre-refactor flat schedule of §5 (census-locked at 1, 2 and 3 fine levels, ab + aa, with per-event parity locks). The
F2C write parity keys on the parent's **next** substep counter and collapses to v1's `(iterations % 2) == 1` at level 0.
Placement rules: the C2F fill runs once per pair before its substep 1, sourcing the parent's **live** post-substep-A state
mid-cycle; the cycle ends F2C(1→0)-first, then a **level-ascending** re-arm + C2F cascade — the AA in-place discipline (a
level's launch destroys its previous frame, so fills issue while their source frames still exist; F2C and C2F touch disjoint
sets).

### 13.4 The wall chain (commit E `6c25740`)

Wall-shared faces (s = −1) nest through the chain by keying `buildFineWallMasks` on the **immediate parent's** map (one
floor(fine/2) hop, exact under parent-frame storage), scanned against the parent's overlap-extended **storage extent** — the
backing `GEO_WALL` row of a nested wall-shared face sits on the parent's own fine wall row at parent-local −2, inside its
ghost zone. Blocks are visited level-ascending (V5), so a depth-K mask derives through K hops. The one genuinely new mechanism
is the **R4 wall-pedestal prism**: on a wall-shared face the parent's upward own-8 F2C window reads face-normal rows {2,3} —
deep-frozen `GEO_NOTHING` cells no standard transfer authors in a chain (the "shadow solve" the band registration exists to
kill) — so `buildCouplings` appends twice-inset tangent rectangles covering exactly those rows to the coupling's interior
patches (disjoint from the six depth-1 skins, empty at level 1, so the v1 census is untouched); `checkCouplingMapPattern`'s
rail (b) generalizes to the face-specific depth sets, and the rail's sibling shadowing was verified already parent-frame at
depth. Fail-fasts: partial parent wall and a **broken chain** (own `GEO_WALL` row with zero parent backing) throw at SimInit;
**`F2C_LAGRAVA` + nested wall sharing is a hard SimInit error** (the 4-node filter window underflows the 3-row pedestal;
`F2C_SCHONHERR` is unaffected).

### 13.5 The chain solver and the windbreak target (commits F `6ae4a61` + G `5214b01`)

`sim_AMR/amr_chain_solver.h` derives the level-2..max footprints from the 2-level channel's level-1 anchor in
createAMRBlocks' integer parent-cell frames (`rect_L = 2 · inset(rect_{L−1})`): insets of 3 parent cells per non-wall face
per hop (the V9 no-warning tier) and a gap-0 wall-chained z-min face per hop, holding every level's z-min face on the level-0
wall-candidate lane z = R+1 — the V10 wall chain 0..max_level. Nested mode is opt-in via `--max-level 2..4` on
`sim_AMR_channel` (the default `--max-level 1` path emits the byte-frozen 2-level config through the same `fmt::format` call
as before); the full V-suite remains the authoritative guard on the emitted spec. The derived **R = 1 chain** carries an L4
span of 86×22×43 parent cells; its one locked deviation is the **level-4 y fine span of 44 cells, not 48** — the 16-cell
anchor minus three y-hops of inset ≥ 3 exhausts the budget (all hard floors pass; the 44-cell span keeps a workable wake
margin).

`sim_AMR/amr_windbreak.h` stamps the rod array on the **finest level's map only** (every parent treats rod columns as plain
fluid — sub-grid geometry never appears upstream): axis-edge discs `(2dx+1)² + (2dy+1)² ≤ d²` extruded from the wall-chain row
under integer-exact guardrails. The realized default-knob geometry is **3 rods in a 2+1 stagger** (row 1 at x = 32 with y axes
{13, 29}; row 2 at x = 66 with y {21} — the half-pitch stagger), 12-cell discs (d = 4), height 40 rows, 480 cells per rod,
**1440 cells total**. The four locked knobs are CLI-tunable (`--windbreak-diameter/-pitch/-height/-row-spacing`, plus
`--no-windbreak`); **`--max-level 2..3` with the default rod geometry hard-errors at SimInit** (those y spans cannot host the
p = 16 staggered second row) — use `--no-windbreak` or tuned knobs there.

Pre-registered runs (640-step convective pass, R = 1, `--max-level 4`; both re-run digit-for-digit at commit time):

- **F — 5-level chain, no rods**: final mass 1.863994e+04 (−0.011 % vs the 2-level 1.866006e+04); KE per level L0→L4 =
  8.145986e+01 / 3.221264e+01 / 7.490310e+01 / 2.158624e+02 / 3.090272e+03; smooth saturating inflow-driven rise,
  max\|v\| 1.2e-1 class on every level, zero NaN.
- **G — same + windbreak rods**: mass 1.423224e+04 → 1.864126e+04 (+0.007 % vs F); KE L0→L4 = 8.179896e+01 / 3.220101e+01 /
  7.333960e+01 / 2.054079e+02 / 2.596176e+03 — the −16.0 % level-4 blockage signature (deficit deepening toward the finest
  level); rod cells hold u = 0, rho = 1 (a documented constant mass offset).
