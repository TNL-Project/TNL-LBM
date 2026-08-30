# AMR Interface — Proposed Redesign (1D Schematic)

**Status:** PROPOSAL, not implemented. Companion to
`AMR-interface-1d-diagram.md` (the current design); section numbers §x.y refer
to `AMR-for-LBM-implementation.md` where the documented problems live.
All diagrams are 1D simplifications (left interface; right mirrors) and assume
the **AB** (default) streaming pattern; AA is deferred with §10.1 Defect 1.

Grid convention (same as the companion doc): coarse cells are 10 characters
wide, fine cells are 5 — fine cells `2c, 2c+1` are the two subcell volumes of
coarse cell `c` (each fine cell spans 1/2 of it), the fine ghost cell `-1`
covers the right half of the ring cell `o-1`'s volume, and the fine block
boundary is the *face between the coarse volumes* `o-1 | o` — a cell-centered
(volumetric) scheme: no quantities live at that face itself.

## Design principle

Five changes relative to the current design:

1. **C2F uses the compact-moment scheme** (`C2F_COMPACT_MOMENT`, Schönherr
   2015 §7.2): 2-cell-per-axis window, second-order cumulant reconstruction,
   all third-order and higher central moments zeroed at the fine cell.
2. **F2C fills only the one-cell-thick hidden skin** under the footprint
   (frozen `GEO_NOTHING` cells adjacent to the footprint boundary), reading
   **fine-interior cells only**.
3. **The ring-F2C overwrite is deleted:** `GEO_AMR_INTERFACE` cells stream and
   collide like ordinary fluid and are never written by a coupling kernel.
   Fine-to-coarse feedback reaches the ring through *streaming* from the
   freshly filled hidden skin, on the next coarse step.
4. **The outer ghost layer `-2` is removed entirely** (`storage_overlap` set
   explicitly to 1 on refinement-level blocks): its only consumer was the ring-F2C filter
   window (commit `089e47a`), deleted by change 3. The BVP refill schedule is
   RETAINED for the remaining 1-deep ghost — substep 1 still drains and
   contaminates it, so the C2F fill still runs twice per coarse step.
5. **The C2F stencil is carved one-sided: no covered (skin/hidden) coarse
   cell is ever an interpolation source.** The CM window shifts one cell
   outward wherever it would touch a `GEO_NOTHING` cell and evaluates the
   polynomial off-center toward the fine side. Covered cells are never C2F
   sources in ANY published scheme — Guzik's covered regions are literally
   named "invalid" and his stencils have "no stencil cells" pointing into
   them (`25_Guzik_2014:401-406, :1000-1001, :1050-1051`); Chombo shifts its
   interpolation stencil to "use only values in Ω_valid" (`01_Adams_2015
   _Chombo:2717-2748`); Schönherr does not even allocate coarse nodes under
   the fine grid, and provides the offset/extrapolation mechanism for
   exactly this shift (`47_Schonherr_2015_Thesis:2922-2924, :3490-3551`,
   Eqs 7.49-7.57; same mechanism in Kutscher 2018). The skin is thereby
   demoted to a pure **streaming conductor**: it supplies ringward
   populations to the next coarse streaming step and is never read by an
   interpolation stencil (see §3a).

Rationale: the ring's fine subcells *are* the ghost halo, so a ring-F2C never
read real fine data — it recycled C2F output back into the coarse lattice at
>100 % recycled weight (the 20/16-of-Σw shifted-face window; see
`AMR-interface-1d-diagram.md` §3a and §9.2). The hidden skin, in contrast,
volumetrically *is* fine territory (Chen's formulation applies), and it is the
only coarse region the fine level is authoritative for.

---

## 1. Layout (roles)

```
                 o-3       o-2       o-1        o        o+1       o+2
              ┌─────────┬─────────┬─────────┬─────────┬─────────┬─────────┐
              │  fluid  │  fluid  │  RING   │ FROZEN  │ FROZEN  │ FROZEN  │
              │         │         │ evolve  │  SKIN   │  core   │  core   │
              └─────────┴─────────┴─────────┴─────────┴─────────┴─────────┘
                                          -1    0    1    2    3    4    5
                                        GHOST  interior fluid (GEO_FLUID)
                                          └─ fine block boundary = face between the coarse volumes (o-1 | o)
                                            (1-deep ghost — the BVP refill layer;
                                             no outer layer: nothing reads it)

 fluid / RING: pure coarse evolution (streaming + collision); RING also reads
               the hidden skin through streaming every step
 SKIN (o, o+S-1 per face): frozen GEO_NOTHING, filled exclusively by skin F2C
 core          : frozen GEO_NOTHING, never read, never written (dead weight)
 GHOST (-1)    : filled by C2F, refilled between substeps (BVP); read by fine
               streaming's 1-cell stencil; the only fine halo cell allocated
```

Memory effect of removing the second layer: the 2-deep overlap shell of a
64³ fine block costs `(68³ − 64³)` cells per array vs `(66³ − 64³)` at 1-deep
— 26,936 of 314,432 cells ≈ **8.6 % of fine-block storage** (≈ 6.4 MB per
64³ block in single precision counting both DF arrays + macro + map), plus the C2F fill
does half the work per launch.

## 2. C2F via compact-moment (`C2F_COMPACT_MOMENT`)

**Output:** the fine ghost cell's DFs, reconstructed from six second-order
cumulants (cumulant back-transformation of `col_cum.h`); 3rd-order and higher
central moments zeroed — non-hydrodynamic modes are projected out at the
interface instead of being interpolated per direction.

**Inputs:** the 2-cell-per-axis window, **carved one-sided onto uncovered
coarse cells** (change 5): the nominal CM window `{home−1+(fg&1), home+(fg&1)}`
is shifted one cell outward wherever it would touch a covered (`GEO_NOTHING`)
cell, and the polynomial is evaluated off-center toward the fine side. Per
source cell, the five independent second-order non-eq moments (strain rates)
are extracted and rescaled `ω_s → ω_d` (source/destination relaxation rates).

```
                 o-3       o-2       o-1        o        o+1       o+2
              ┌─────────┬─────────┬─────────┬─────────┬─────────┬─────────┐
              │  fluid  │  fluid  │  RING   │  SKIN   │  core   │  core   │
              │         │         │         │ (never  │  (never │         │
              │         │         │         │  read)  │  read)  │         │
              └─────────┴─────────┴─────────┴─────────┴─────────┴─────────┘
                        └──────────────────┘
                                     ▲  ghost -1 ◄── {o-2, o-1}  fluid + ring
                                         nominal window {o-1, o} would touch
                                         covered o → shifted one cell outward;
                                         polynomial evaluated at t_rel = +3/4
                                         (0.25-cell off-center extrapolation —
                                          Schönherr's offset mechanism,
                                          Eqs 7.49-7.57; Kutscher 2018 §3)
```

(The second ghost layer used to be filled from the `{o-3,o-2,o-1,o}` /
`{o-2,o-1,o,o+1}` windows; with it gone, the storability clip in
`launchCoarseToFineTransfers` simply shrinks the fill to the single layer —
no patch-geometry change.)

3D: the tensor-product window of a face-adjacent ghost is
`{o−2, o−1} × {fy} × {fz}` — all sources valid coarse cells: at `x=o−2`
outer fluid, at `x=o−1` ring cells (the face/edge halo columns are
collision-active and temporal-fresh from the coarse kernel that just ran).
**No covered cell appears anywhere in the stencil** — matching every
published CM implementation, whose C2F sources are 100 % on the coarse side
(Geier 2009; Schönherr 2011/2015; Kutscher 2018; Qi/Musubi 2019), and
Chombo's/Guzik's general rule that covered ("invalid") cells never serve as
interpolation sources. The earlier draft of this section had the window
reading ring + F2C-fresh skin — literature review showed no published scheme
reads covered cells for C2F at ANY freshness level, so the window is carved;
the skin's fresh fine state reaches this ghost *indirectly*, through the
ring's own streaming absorption.

Why CM over the 3rd-order Lagrange C2F for this architecture:

- **Stencil reach 1, carved**: the window is 2 cells per axis (one cell less
  per direction than the 3rd-order Lagrange), and the carve keeps every
  source on valid coarse fluid — the extrapolation machinery it needs for
  this is shipped with the scheme itself (Schönherr §7.3 offset transform).
- **Mode projection** (3rd+ central moments = 0): the exact mechanism
  recommended for interface noise (Astoul et al. 2021a; §12.7 item 3) is
  applied on the fine side of the loop, and the reciprocity of the τ
  rescale (`σ = 1/2` in the cumulant correction) round-trips the
  F2C `τ_c/τ_f` scaling correctly.
- Strain-rate (2nd-order moment) information crosses with ω-rescaling —
  unlike explosion variants, no viscous stress is lost at the interface.
- Production scheme of Musubi (Qi et al. 2019) — the cell-vertex cousin of
  this exact design.

## 3. F2C → the one hidden skin layer (proposed scheme)

**Output:** DFs + macros of the frozen skin cells (`o` and `o+S−1` per face;
1-cell-thick skin in 3D: 32³ footprint → 5,768 cells, the `32³−30³` shell).
The deep frozen core is never read by CM (stencil reach 1) and never written —
dead weight, blanked in visualization by `vtkGhostType` regardless.

**Required properties:** (i) fine-interior reads only — the ghost-exclusion
divergence happened because the *ring's* subcells are all-ghost, which does
not apply here: every skin cell's own subcells and every shifted window lie
inside fine fluid; (ii) mandatory low-pass filter (Lagrava) so fine-resolvable
non-hydrodynamic modes do not alias into coarse storage that no collision ever
damps; (iii) sum-to-one volumetric weights (no volume factor).

**Proposal — interior-shifted cubic Lagrava projection:** the existing
`F2C_FULL_LAGRAVA` machinery with the window clamp moved from the storage
bound (`lo = −ov`) to the **fine-interior bound (`lo = 0`)**:

```
                                          -1    0    1    2    3    4    5
                                        ghost   fine interior fluid
                                            └──────────────────┘
                                                 ▲  skin o ◄── {0,1,2,3}
                                                      nominal {2o−1 .. 2o+2} hits
                                                      ghost −1 → shifted interior-only,
                                                      evaluation still at the coarse
                                                      center (cubic-exact)
                                                      (centered windows {−1,9,9,−1}/16
                                                       apply to non-edge cells)
                                                      (o+1 … core: not filled — never
                                                       read by CM-C2F)
```

- Edge windows are off-center but remain exact for cubics at the fixed
  evaluation point; centered windows apply everywhere except the cells
  adjacent to each footprint face.
- Per-cell write: `f_skin = eq(ρ̄, ū) + (τ_c/τ_f)·(f_avg − eq)`, the current
  kernel's rescale, unchanged.
- Fallback for smoke-testing: plain 1/8 box average of the cell's own 8
  subcells (always all-interior, trivially per-cell conservative; weaker
  filter — second choice, first milestone).

**Conservation note:** on translation-invariant interiors the projection sums
each fine cell with total weight 1/2 per axis (1/8 in 3D), same as the box
average; shifted edge windows trade per-cell conservation for exactness.
The deviation shows in the existing `computeConservationStats` drift metric —
flag it in evaluation, don't hide it.

**Ring receives no F2C write.** Its DF and macro state come from the coarse
kernel only; the fine→coarse feedback path is now exclusively:

```
fine interior ──F2C(skin)──▶ hidden skin o ──next coarse streaming──▶ ring o−1 ──C2F──▶ fine ghost
     (real fine data)           (fresh,              (streaming                   (carved:
                                  filtered)             + collision)                fluid+ring)
```

### 3a. The skin as Guzik's stream register — relocated from receiver to source

Guzik's BVP handles the covered zone's ringward flux with **stream
registers** (Eqs 24-26, `25_Guzik_2014:1102-1131`): after the coarse stream,
the population that crossed from the covered zone into the ring is
*subtracted* (`δf = −f_covered`, Eq 24) and replaced by the exact
volume-weighted sum of fine-grid populations leaving the fine grid
(Eq 25-26). The net effect: the ring's population pointing from the covered
zone carries exactly the fine outgoing distributions, and the covered cell's
own dynamics are nullified.

Our skin-F2C write produces the **same net effect at the source instead of
the receiver**: the skin cells hold ONLY fine-injected content (frozen, no
own dynamics), so when the ring's streaming pulls ringward populations from
them, it receives exactly fine-sourced populations — Guzik's
`−coarse_stream + fine_sum` ≈ our `no coarse stream (frozen) + injected fine
state`. The design differences to keep on record (directionally analogous,
not identical — in particular the skin write is a non-conservative state pull,
not the register's exact conservative flux sum):

1. **Whole-state vs directional**: Guzik corrects only the ringward-pointing
   populations of ring cells (a flux correction); the skin write replaces
   the entire 27-population state of every skin cell (stronger than the
   stream needs, but makes the skin a coherent field for viz/logging).
2. **Exact flux sums vs filtered average**: Guzik's register is exactly
   conserving by construction (volume-weighted sums of the actual outgoing
   fine populations); the skin holds a Lagrava-filtered 1/8 average —
   conservative only on translation-invariant extents, slightly lossy on
   shifted edge windows (see the conservation note above).
3. **Advanced-then-discarded vs frozen**: Guzik collides+streams covered
   cells and discards the result (subtracting it, Eq 24, before injecting
   the fine sum); freezing the skin skips that work outright, which the
   literature study confirmed is the identical net effect — the covered
   cell's own evolution is functionally meaningless in his design too.

This closes the loop on the design: the skin's ONLY remaining roles are
(1) this streaming-conductor duty and (2) visualization/conservation
logging. It is never an interpolation source (change 5). Guzik's
receiver-side register formulation remains on record as the alternative
mechanism if the source-side write ever shows a defect — it corrects the
ring populations directly from the fine sums and the skin would then need
no state at all.

## 4. Cycle timeline (per coarse step)

1. Coarse kernel: all level-0 cells stream+collide (ring included; ring pulls
   from the skin filled last cycle — the one-step-delayed fine feedback).
2. CM-C2F fill (`updateKernelDataForLevel` → `cudaAMR_CoarseToFine`, CM path);
   fills the 1-deep ghost.
3. Fine substep 1.
4. **BVP** CM-C2F refill (substep 1 drained/contaminated the ghost; refill is
   required for a valid boundary field at substep 2 — unchanged mechanism,
   one layer instead of two).
5. Fine substep 2.
6. Interior-skin F2C (interior-shifted Lagrava, interior-only windows).
   **No ring-F2C launch.**

## 5. What this fixes (mapped to the documented issues)

| Issue (doc §) | Current design | Proposed |
|---|---|---|
| §9.2 one-way clamp | ring re-written from ghost-dominated window (20/16 recycled weight) | channel deleted; feedback via streaming from fresh skin |
| §9.2.1 hidden-cell injection | frozen cells hold stale/mismatched data, read back into C2F | read removed entirely: C2F stencil carved to uncovered cells only (Guzik/Chombo/Schönherr-offset) |
| skin as C2F source (new §) | hidden skin read post-consumption, temporally inconsistent + self-referential | never read: skin = pure streaming conductor (Guzik register at source); fill sources = fluid+ring only |
| ghost reads in F2C | ring window reads ghosts at 20/16 | eliminated: skin windows interior-only, shifted at the edge |
| stale outer ghost `-2` | read at 5/16 after never evolving | **not allocated at all** (storage_overlap = 1) |
| Astoul mode injection | interior F2C injects undamped modes (v9 autopsy, §12.6) | Lagrava filter on the write + CM zeroing 3rd+ modes on the read-back |
| inertia (wide-stencil recycling) | 4-cell C2F reaches o+1 deep frozen | CM reach 1; less recycled-state weight in every fill |

## 6. Risks and open questions

1. **Residual non-hydro escape into the coarse lattice.** v9's regression was
   attributed to direct injection without collision damping (§12.6); frozen
   cells still never collide here. Mitigation order: Lagrava filter (in) → CM
   projection on read-back (in) → λ-relaxed skin write
   `f ← f + λ(avg − f)`, λ ≈ 0.7 (circuit breaker, only if measurement rings).
2. **Untested as a combination.** Nearest falsified neighbors (v6: hidden
   collision-active, NaN; v9: freeze + interior F2C *with* ring-F2C kept,
   5.6× worse — itself confounded by the pre-`089e47a` placeholder bug, after
   whose fix the frozen design recovered to 188,353 violations) differ
   exactly in the removed channel — the falsification
   matrix has no cell for this design; expect measurement to decide.
3. **Temporal gap unchanged**: both CM fills use the coarse post-step state;
   hidden skin carries t_{n+1} fine state into a ring that streams at the next
   step's start. Also note the **outer ghost layer is gone for good**: a
   future IVP revival would need to re-grow it (with BVP kept, there is no
   consumer for a second layer).
4. **CM mass conservation**: compact interpolation provably violates global
   mass conservation — "the total density over the whole domain was
   oscillating and slowly increasing over time" and — the exact case we
   benchmark — "without Dirichlet boundary conditions, such as this
   Taylor–Green vortex, the simulation will therefore eventually fail"
   (`50_Qi_2019:688-691`); Geier's analysis: locally conservative,
   vulnerable at boundaries (`49_Geier_2009:391-394`). The conservation
   monitor (`computeConservationStats`) is a mandatory part of evaluation,
   and a conservation-enforcing variant (Guzik's constrained-least-squares
   ghost fill) is the fallback if drift ships.
5. **D3Q27 corner-velocity patterns**: Guzik's stencil derivations cover
   D3Q19 only; corner velocity directions (‖e‖₁ = 3) are explicitly deferred
   (`25_Guzik_2014:2386-2399`). The carve + constraint method transfers, but
   D3Q27 corner-direction stencils must be derived/validated separately —
   flag before any multi-face geometry use.
6. **Defect 2 carries over**: the skin F2C DF write needs a coarse-map
   guard (`GEO_NOTHING` only) for boundary-touching footprints — the same
   ~3-line guard as in §10.1. It does not exist today (only the macro write
   is guarded): the wording is "add", not "keep".

## 7. Implementation sketch

- Default-this-variant strategy defines: `C2F_COMPACT_MOMENT` already exists
  (`TNL_LBM_C2F_STRATEGY`); add `F2C_SKIN_ONLY` (new):
  - `lbm_block.hpp` (`initLevelLattice`): under the define, set
    `storage_overlap` explicitly to 1 for refinement-level blocks
    (`089e47a`'s overlap motivation, the ring-F2C filter window, is deleted
    by design). Do NOT rely on the library default: it is 1 only with MPI
    and 0 without, so a non-MPI build would silently allocate no ghost
    layer and the C2F fill would write nothing;
  - `amr_state.h`: skip `launchFineToCoarseTransfers` (ring) under the define;
    restrict `interior_patches` to the 6 skin rectangles (same disjoint face
    partition as the ring patches, one cell deep, inside the footprint) —
    this IS new patch geometry in `buildCouplings` (today there is a single
    full-footprint rectangle per block pair). The C2F patch fine rectangles
    stay 2 cells thick — the existing storability clip (`df_overlap_X/Y/Z`
    = 1) shrinks the fill to the single allocated layer, no change needed
    on the C2F side;
  - `amr_coupling.h` (`cudaAMR_FineToCoarse`): under the define, clamp window
    lower bound to `0` instead of `-ov` (fine interior only);
  - `amr_coupling.h` (`cudaAMR_CoarseToFine`, CM branch): the carve — the
    per-axis window predicate changes from "index in storage" to
    "`map != GEO_NOTHING`", shifting the 2-cell window one cell outward
    wherever it would touch a covered cell; the runtime weight/window
    machinery already present (storability guard) computes the shifted
    weights — this is exactly Schönherr's offset transform
    (Eqs 7.49-7.57) played by the shifted-stencil path;
  - DF-write map guard (`GEO_NOTHING`) added in the same pass (Defect 2).
- Tests: coupling mock tests that allocate `ov = 2` explicitly exercise the
  deleted ring path — those cases go with the channel; C2F fill tests are
  unaffected apart from the overlap allocation depth.
- No new kernels; all changes reuse the shifted-window storability machinery.

## 8. Evaluation protocol

Single variable vs the current design (ring-F2C kept), same benchmark:

1. Bracket metric cycle series (5/10/15/20), not a single snapshot —
   inertia shows as slow growth + larger asymptote;
2. kinetic-energy decay vs analytic TG `exp(−4νk²t)` per level —
   anomalous decay flags smoothing-masquerading-as-stability;
3. conservation drift (existing logging, MANDATORY per risk 4 — compact
   interpolation is provably non-conservative on periodic domains);
4. CM exactness unit test (part of change 5) + a carved-window case
   (ghost adjacent to the footprint face: verify the shifted window +
   off-center evaluation reproduces linear/quadratic fields exactly); full
   `run-amr-tests.sh` on AB (the gate shell; retired in favor of
   `pytest tests/unit/test_amr_units.py tests/integration/test_amr_paraview.py`);
5. memory/LUPS comparison: 1-deep vs 2-deep overlap (expect ≈8.5 % block
   storage reduction at 64³ fine, half the C2F fill work per launch).

---

## Appendix A — Literature grounding for the carve (change 5)

This proposal's stencil-validity rules come from two literature reviews of
`docs/references/AMR/` (full compact-moment layout study:
`docs/references/AMR/report_cm_grid_layout.md`):

| Claim | Source |
|---|---|
| Covered regions are "invalid"; C2F stencils never read them ("no stencil cells" pointing in) | `25_Guzik_2014:401-406, :1000-1001, :1050-1051` |
| Chombo shifts ghost-fill stencils to "use only values in Ω_valid" (QuadCFInterp) | `01_Adams_2015_Chombo:2717-2748` |
| Covered cells: F2C averaging target + stream-correction subject — never C2F source | `25_Guzik_2014:460-483, :1063-1131` |
| Stream registers: subtract covered stream, inject exact fine flux sums (Eqs 24-26) | `25_Guzik_2014:1102-1131` |
| Frozen-skin ≡ advance-then-discard (Eq 24 subtrahend is zero for frozen cells) | `25_Guzik_2014:494, :505, :521-525` (analysis) |
| One-sided/asymmetric stencils are standard at boundaries (non-centered 3rd order) | `33_Lagrava_2012:704-719` (Eq 39); `24_Gendre_2017:1087-1093` (Eq 41) |
| Schönherr's offset mechanism: shift source cell, evaluate off-center, "arbitrary offsets" | `47_Schonherr_2015_Thesis:3490-3551` (Eqs 7.49-7.57); Kutscher 2018 (`44:1429-1433`) |
| Coarse nodes under the fine grid are not allocated/used at all | `47_Schonherr_2015_Thesis:2922-2924`; `45_Schonherr_2011:953-954` |
| GhostFromFiner reads fine FLUID elements only (never fine ghosts) | `50_Qi_2019:458-460` |
| Compact interpolation violates mass conservation; worst in periodic domains | `50_Qi_2019:688-691`; `49_Geier_2009:391-394` |
| τ-rescaled injected state is a CE-legitimate CM source; a stale one is not | derivation analysis (report appendix); texts silent on staleness — carved away |
| Guzik corner-velocity stencils are D3Q19-only; corner directions deferred | `25_Guzik_2014:2386-2399` |
| BVP (our ghost strategy): 1 layer, refill per subcycle; time-interpolated; 2nd-order L∞ | `25_Guzik_2014:557-565, :1450-1457, :2248-2252` |

---

## Implementation outcome (D.5, 2026-08-16)

This document was the design-phase proposal; the "PROPOSAL, not implemented"
status and all "current design / proposed" language in the body are the
historical record. What the gates measured (all numbers on the §9.1
acceptance configuration of `AMR-for-LBM-implementation.md`, which carries
the full gate records):

- **Change 1 (compact-moment C2F) — IMPLEMENTED, then FALSIFIED by gate A
  (branch 3) for the direct interface path.** A1 (CM uncarved) converged
  near baseline (613,183 vs 605,583 @ cycle 10, +1.3 %, but +54 % @ cycle
  5); A2 (CM + change-5 carve) diverged (3,346,370 @ cycle 10, +452.6 %),
  with A2 rho ≥ A1 rho at every cycle 1..10 — the carve never repaired the
  density fit — and per-level KE ratios growing to 1.7739/1.4999
  (extrapolation-distance physics: the 0.25-cell off-center evaluation
  amplifies interface noise). The carve math itself is fp-exact (Tests
  8–13 + 17). Disposition: C2F keeps the 3rd-order Lagrange default; CM +
  `C2F_CARVE` remain non-default compile-time experiment options.
- **Change 5 (carve off covered cells) — mathematically sound, rejected as
  the direct-path scheme with the CM it serves.** Its §8-evaluation wording
  ("reproduces linear/quadratic fields exactly") stands with one recorded
  nuance (A.2): pure-quadratic exactness holds for the k-corrected
  11-coefficient VELOCITY fits only — the 8-coefficient density fit is the
  plain trilinear nodal fit (linear/constant-exact); the exactness tests
  deliberately keep density linear/constant.
- **Changes 2+3 (skin-only F2C + ring-F2C removal) — ACCEPTED (gate B,
  user ruling A: pass-in-substance).** Cycle-10 violations 220,737 vs the
  corrected baseline 605,583 (−63.6 %); series non-monotone with a +3.1 %
  c5 transient overshoot (the ring receives skin-fed data one streaming
  step later than the retired instantaneous ring-F2C injection); KE-decay
  and mass legs clean (ulp-invariant); the B.7 Dirichlet channel deficit is
  arm-common, so no thin-channel fallback fired. The ring F2C was
  hard-deleted in D.1 (29 items) and `F2C_SKIN_ONLY` retired — the skin
  semantics (6 inset-face rectangles, `lo = 0` window clamp) are the
  unconditional shipping configuration.
- **Change 4 (remove the outer ghost layer) — PASS, bitwise.** With
  `storage_overlap = 1` the fields are bit-identical to the overlap-2
  winner (132 arrays, 0 bit-differences); fine-block storage −8.566 % per
  scalar array, C2F fill work −48.5 %, walltime 13.7 → 12.6 s.
- **Risk 6 (Defect-2 guard) — implemented in Phase 0.4** (DF/macro stores
  skipped unless the target is `GEO_NOTHING`/`GEO_AMR_INTERFACE`; locked by
  Test 7) — the "add, not keep" note is resolved.
- **§3's `F2C_FULL_LAGRAVA` machinery mention (:169) — corrected by
  record, not by deletion (the historical text stands):** that token never
  wired into CMake and nothing defined it, so pre-P0.1 builds silently ran
  the 1/8 box average; the P0.1 polarity fix made the 4×4×4 Lagrava
  projection the unconditional default (`F2C_BOX_AVERAGE` opts out) — the
  "existing machinery" the proposal pointed at is exactly what shipped. No
  `F2C_FULL_LAGRAVA` remains anywhere in the code.
- **Open items this proposal raised, resolved elsewhere:** Guzik
  constrained-LS fallback NOT TRIGGERED (mass ulp-invariant in all four
  measured configs); Guzik CLS stencils are D3Q19-only — no least-squares
  machinery exists in the shipped scheme at all, and a D3Q27 corner-pattern
  derivation is required before any multi-face CLS use (standing caveat);
  the λ-relaxed skin write circuit breaker never needed to fire.
