# AMR Schönherr-2015 ch.7 Target-Band Contract

**Status:** binding engineering contract for branch `feat/amr-schonherr-ch7` (off `agents`). Deviation from any normative statement in this document requires a new user ruling — not a plan edit, not a commit-level judgment call (plan Appendix B, ruling 4).
**Provenance:** registration ruling 2026-08-19 (binding user rulings, plan Appendix B.1–B.4), plus two review dispositions: Oracle GO-WITH-CHANGES (3 conditions + F1–F5) and Momus [OKAY] with survivors; false evidence claims retracted (plan Appendix A).
**Full plan:** `.omo/plans/schonherr-ch7-conversion.md`. This contract carries its normative content forward; the plan doc remains the source of record for stage-by-stage work items, risk ranking, and commit order.

---

## 1. Motivation — experiment, not conservation-fix prediction

This conversion is an **experiment, not a conservation-fix prediction** (Oracle F1 framing, adopted).

- The C2F direction **already is** Schönherr's σ-form since 2026-08-18 (CM branch: `sigma = n1o2`, ω_d = 1/τ_fine — `amr_coupling.h:383-384, :794, :800, :808`). The measured −23 % / 20-tc mass leak was produced **with that branch** (arms are CM + carve).
- The remaining FH τ-rescale lives **only** in: (a) the `C2F_LAGRANGE` opt-out branch, (b) the F2C kernel (`neq_scale = tau_coarse/tau_fine`, `amr_coupling.h:1393`).
- Schönherr ch.7 **makes no conservation claim** anywhere; Qi et al. 2019 report the sibling Musubi compact interpolation **violates** mass conservation on a fully periodic TGV (`50_Qi_2019_Musubi_CompactInterp.txt:688-691`) — the exact benchmark run here.
- What actually changes in this conversion:
  1. **Band registration** — the new geometry restores centered `|t_rel| = 0.25` fill evaluation where today's carve extrapolates at `|t_rel| = 0.75`; every C2F source becomes a simulated cell with no carve reads at valid faces.
  2. **F2C σ-form** — `σ·ω_f/ω_c = 2·τ_c/τ_f`, factor-2 stronger neq injection than today (flagged risk F4).
  3. **Six-step cycle** — fills once per cycle at cycle end, retiring today's content-no-op BVP refill and the H9 band-aid.
- The probe ladder (per-iteration `--write-dfs` seam metric + 20-tc harvest + between-metric bracket) accepts or rejects the conversion **by measurement only**. Null/negative results are recorded, not repaired ad hoc.

**Pinned baseline constants** (TGV res 1, `--convective-times 20`, control arm):

| constant | value |
|---|---|
| mass drift | **−0.2296387 rel** |
| drift slope | **−1.0370e-04 1/s** |
| max \|drift\| | 0.2296 |
| standing interface bias | **+1.30e-05** (signed = abs; uniform-sign DC offset, formed within 2 cycles) |
| KE at 20 tc, coarse | **ke0/ke0_0 = 0.415** |
| KE at 20 tc, fine | **ke1/ke1_0 = 0.927** |

**Metric stability:** the harvest tool `.omo/evidence/h9_retry/harvest_tc20.py` (DT = 2.241333 s/sample, ν = 1.5e-5, k = 2π/0.41) is the pinned metric definition, held as-is. T2 adds a calibration leg that reproduces these constants on the pre-change build (Momus-survivor #1).

---

## 2. Binding band map

Face-normal axis; coarse cell centers at integers; ring row c=−1, today's skin c=0 (plan §1.1, verbatim).

| position | target role | today's tag → target tag | old fine indexer → new |
|---|---|---|---|
| c=−2 | bulk (today's carve-era outer source; never read nominally) | fluid (unchanged) | — |
| c=−1 | C2F source line 1 | GEO_AMR_INTERFACE (unchanged) | — |
| −0.75 (+¼ of c=−1) | C2F destination (ghost row 1) | ghost | old −1 → **new −2** |
| c=0 | C2F source line 2, SIMULATED | skin GEO_NOTHING → **GEO_AMR_INTERFACE** (ring row 2) | — |
| −0.25 (−¼ of c=0) | C2F destination (ghost row 2) | interior → ghost | old 0 → **new −1** |
| +0.25 (+¼ of c=0) | standard LBM | interior | old 1 → **new local 0** |
| +0.75, +1.25 (subcells of c=1) | F2C source rows | interior | old {2,3} → new {1,2} |
| c=1 | F2C destination (frozen, refilled) | 1st deep frozen → **GEO_NOTHING skin target** | — |
| c≥2 | outside the band | GEO_NOTHING (unchanged) | — |

### 2.1 Vertex rule

Both destination rows share nearest coarse vertex −0.5 and source pair {c=−1, c=0}; the nominal per-axis window straddles the shared vertex ⇒ **zero frozen reads at valid faces; carve demoted to wall/degenerate-only** (curated tests 12–13 keep it alive). Fine destinations/standard/F2C-source rows all sit over SIMULATED coarse cells; only the F2C destination sits over frozen.

### 2.2 Insets / offset arithmetic

- `offset_fine' = 2·origin + 1`, `local_fine' = 2K − 2`, ov=2 ⇒ stored rows/axis = 2K+2 **unchanged**; no exterior storage growth. Parity invariant by construction (fg-global property; every band row keeps (home, t)).
- Footprint extents: full idempotence of `gs = fine->local/2` breaks under re-anchor — replace with **`(fine->local+2)/2`** or region bookkeeping at **every site**: `amr_state.h:506` (buildCouplings gs), `amr_decomposition.h:411` (markAMRInterface size), `isShadowedBySameLevelBlock ~:653` (Oracle F3).
- Footprint minimum: **gs ≥ 3 per axis** (`createAMRBlocks`, `amr_decomposition.h:168-192`); gs=3 max-side slab dedupe-clamped empty; gs=2 invalid (dual-role row); planar `[2,8,8]` fixture now **rejected**, fixtures move to `[3,8,8]` (fine local 4×14×14).

### 2.3 Conservation recount

Tag-keyed exclusion, unchanged code path.

| census | today | target |
|---|---|---|
| fixture excluded (16³, K=8) | 512 | **216** (c=1 shell 152 + deep 64) |
| TGV K=32 region excluded | 32768 | **27000** (skin 5048, deep 21952) |
| counted fine volume | 32³ | 31³ = 29791 |

Net counted-volume shift **+2791** (corners ×1.125, edges ×1.25, faces ×1.5) — a **constant-volume accounting artifact inside the band; cancels in drift-slope terms; documented in the metric definition**, NOT an error and NOT patched.

### 2.4 Census reference (16³ fixture, K=8, local′=14, stored 18³)

| set | count |
|---|---|
| ring | **784** = halo 488 + c=0 shell 296 |
| GEO_NOTHING | **216** = c=1 shell 152 + deep 64 |
| skin (F2C target), K=8 | **152** = (K−2)³−(K−4)³ depth-1 shell |
| skin, K=32 | **5048** |
| patch destinations total | **3088** = 18³−14³ |

Patch destination census per face family: **{648, 648, 504, 504, 392, 392}**; corner 64, edges 672, face interiors 2352; disjoint partition idiom at begin+1/insets+2, written once each.

---

## 3. Cycle contract (T7)

Per cycle, per level:

`fine substep 1 → fine substep 2 → coarse step → F2C → C2F (frame 0) → C2F (frame 1)`

- **F2C** reads the rotation-1 post-substep-2 frame.
- **C2F runs twice with identical content** — once per AB frame so each substep's frame carries valid destinations; SimInit does the same both-frames fill for cycle 0.
- **F2C↔C2F order is irrelevant** — touched sets are disjoint (declared in docstring).
- **BVP refill = content no-op today** — today's fill #2 is content-identical to fill #1 (verified by review), so its removal is **lossless**. **H9 removal is lossless** on the same basis.
- **Probe-visible cycle-1 caveat:** the both-frames fill means cycle-1 fine1 reads a t_0 fill — the seam metric compares cycle-1 separately from cycle ≥ 2 (Oracle startup-transient note).
- **Checkpoint restart compatibility is declared INCOMPATIBLE** across the re-anchor (array shapes change). Declared, acceptable on this branch; intentional per fork (i).

---

## 4. Fork table (defaults; fallback arms pre-registered — Oracle F5)

| # | fork decision (default) | registered fallback arm |
|---|---|---|
| (a) | F2C destination row **frozen `GEO_NOTHING`** (Guzik-equivalence: frozen covered cell + adjusted stream semantics is verbatim-blessed — report_guzik_bvp_stencils.md §7) | **`F2C_DEST_ACTIVE`** collision-active-if-tagged variant — only budgeted/implemented if the T16 decision table is null on the frozen arm |
| (b) | deep footprint stays allocated frozen (storage-only superset; Schönherr's unallocated deep cells ≈ same physics — memory, not physics) | — |
| (c) | **declared deviation:** fine destination rows are **passive overlap rows** (never collide/stream), i.e. IVP-class semantics, NOT Schönherr's simulated destinations; ghost row −2 is un-read by v1 kernels — kept for band parity, declared, harmless | **simulated-band variant budgeted if T16 null** (`F2C_SIMULATED_BAND` arm; Guzik §1 steady-state-perturbation class argument stands against a passive band in general — our per-cycle fill frequency mitigates; measured verdict decides) |
| (d) | H9 + BVP refill: **hard removal** (`C2F_H9`, `h9_first_fill`, `c2f_time_centered`, CMake var + retirement warning idiom) | — |
| (e) | stage-3 reuses CM coefficient code (`sd/sa/sb/sc/sk` sums exist; extract shared helpers in T13 bitwise-gated) | — |
| (f) | MPI nproc=1 only; explicit non-goal; SimInit note | — |
| (g) | destination rows live in overlap storage only, never kernel-domain | — |
| (h) | F2C sources = the destination cell's own 8 subcells (**closed**; new locals {1,2}) | — |
| (i) | ring tagging = reactivation semantics (ring row 2 = reactivated skin; **no outward growth**, c=−2 stays fluid) — **checkpoint restart compatibility intentionally broken** | — |
| (j) | fix-forward audit deltas; **`F2C_SCHONHERR` default at T17** with **`F2C_LAGRAVA` opt-out kept** | — |

---

## 5. Seam-metric re-pairing table (Momus #2 — pinned)

| metric pair | old | new |
|---|---|---|
| seam primary | fine local 0 (center 16.25) vs coarse c=−1 (15.5) | **fine std row new local 0 (fg 33, center 16.75) vs coarse ring-2 c=0 (16.5)** |
| dest rows | (not separated) | fine dest-2 (fg 32, ctr 16.25) vs c=0 (16.5); dest-1 (fg 31, ctr 15.75) vs c=−1 (15.5) |
| microprofile columns | x=14 bulk, 15 ring, 16 skin/target, fine ghost, fine std | x=14 bulk, 15 ring-1, 16 ring-2 (reactivated), 17 skin/F2C-dest, dest rows, fine std |

Fixture reference: TGV-K32; positions are face-normal x-coordinates (fg = fine-global index).

---

## 6. Gates & abort policy

### 6.1 Stage gates (summary)

| gate | content |
|---|---|
| T1 (this document) | user sign-off |
| T2 probe gate | calibration leg reproduces the pinned constants (+1.30e-5 bias, −0.2296387 drift) ±1 print ulp (float64 analysis); deviations reported before proceeding |
| T5 stage-1 gate (commit 7, 6 items) | (i) full `tests/run-amr-tests.sh` green with rewritten censuses; (ii) deterministic double-run bitwise; (iii) light-cone aisle (Chebyshev ≥2 cells bitwise-equal to parent after ONE SimUpdate); (iv) 10-iter seam table recorded (re-paired columns) with **SimInit map-pattern assertion** (every C2F ghost cell's nominal source pair ∈ `GEO_AMR_INTERFACE` only; every F2C destination cell ∈ `GEO_NOTHING` at exactly surface-depth-1 rows c=1 / c=gs−2) and **expected-geometry fingerprint** test from the ruling formulas; (v) registration assertion lock (Oracle F2); (vi) unit tests far from patches bitwise |
| T12 stage-3 gate | suite green, T10 locks green, deterministic double-run, 10-iter recorded |
| T15b post-T14 seam gate | 10-iter seam metric on the `F2C_SCHONHERR` arm vs parent control; **standing bias amplification ≤ ×1.2 of parent \|bias\| AND no new sign-flip alternation**; determinism lock. **On violation: abort → keep `F2C_LAGRAVA` default in T16/T17** (conversion continues on layout+C2F only); the amplified table is recorded as the option arm |
| T16 stage-4 gate | 20-tc harvest on the `F2C_SCHONHERR` arm; decision table vs §1 pinned constants; table **includes the C2F diagnostic arms** (EQ/NORM/SHEAR defines recompiled against the new default, matching the attribution-matrix methodology) so a null F2C effect is disambiguated from a C2F-side null |
| T17 acceptance ladder | full ladder harvested (default, EQ/NORM/SHEAR arms, Lagrava opt-out A/B); between-metric bracket on the re-pinned 33/62 constants; verdict: reduction in \|drift\|+bias with vortex survival = success; null = recorded negative result; amplification = opt-out already applied at T15b |

Stage order rule: every stage lands its locks first (TDD); a stage may not start while a predecessor gate is red. Stages survive one red inter-stage window only inside stage 1 (commits 4–6 red, green from 7).

### 6.2 Abort/rollback policy (Momus survivor #3)

Gate failure after **2 fix attempts** at any stage: `git revert` to the predecessor stage commit, reopen the implicated fork item for user decision, and record the failed-attempt evidence in `.omo/evidence/schonherr_conversion/`. T15b violation auto-path (§6.1) is the exception (no revert; strategy stays opt-in). Rollback is never silent: each abort lands a `docs` note in the stage commit.

---

## 7. Environment policy matrix (Momus survivor #4)

| axis | policy |
|---|---|
| CI red window (commits 4–6) | Land on branch; `pytest` red is expected during the window; **no merge to agents until commit 7 green**; every red-window commit body declares RED-WINDOW in the subject-line trailer line 3 |
| non-MPI (`build_without_MPI`) | Compile-only expectations; overlap allocation paths are `HAVE_MPI`-gated — the re-anchor is semantically dead there (pre-existing degenerate AMR path); record a compile gate + one info log line, no runtime promise |
| AA pattern | Mocks (`test_amr_*_{ab,aa}`) in scope and must stay green; `sim_AMR` acceptance is AB-only (D.4 defect documented); the new cycle contract is written AB-first with AA store-lambda paths preserved |
| Determinism | Double-run bitwise gates are pinned to one GPU/driver + build; documented in each gate body (GPU model + nvcc + flags) |
| Anchor-rot | Cite symbols first (`markAMRInterface`, `buildCouplings`, `axis_window`); line numbers re-snapshotted at each stage gate and refreshed in that commit |
| Budgets | Probes: 10-iter runs ≈ 1.4 GB/arm (11 frames × 128³ × ~27 fields); 20-tc ≈ 216 MB per run (11 VTKHDF frames; the 6.9 GB item was the h9_t20 401-frame outlier — per-iteration `--write-dfs` drivers are capped at 11 frames). Prewarm cache: keep per-arm dirs; cleanup policy at T18 |

---

## 8. B.7 carry-forward — channel diagnostic semantics

The re-anchor changes the fine-row index mapping (§2 indexer column: a fine row's old indexer equals its new indexer plus one, old = new + 1), so the channel diagnostic built on the old rows must be re-read against the §5 re-paired seam/dest rows, not against the old row identities. The **arm-common deficit interpretation stands**; the refreshed table is produced at T18 with the channel diagnostic re-run (plan §2, T18 bullet). Probe semantics are otherwise unchanged: the §5 seam metric remains a face-mean Δ of rho/f00/f05/f06 over per-iteration `--write-dfs` frames.

---

## 9. Glossary

| term | meaning |
|---|---|
| avk | the a-coefficient family in the Schönherr compact-interpolation polynomial reconstruction (d0/a0/b0/c0 evaluation macros); avk gradient corrections are retained in `F2C_SCHONHERR` |
| σ | relaxation-time scaling factor in cumulant-space grid coupling; C2F σ-form `sigma = n1o2` (σ = 1/2), F2C σ-form `σ·ω_f/ω_c = 2·τ_c/τ_f` (hence "σ=2" shorthand) |
| k-moments | the per-source macroscopic moments evaluated at ω_s = 1/τ_source in the σ-form coupling |
| CM | compact-moment / central-moment interpolation branch of the current C2F coupling (the σ-form branch since 2026-08-18) |
| carve / demote | the extrapolation path used when a nominal C2F source is non-simulated (`|t_rel| = 0.75` evaluation); demoted = restricted to wall/degenerate-only sites, unreachable at valid faces |
| skin | the depth-1 cell shell at the footprint surface; the F2C destination/refill target (`(K−2)³−(K−4)³` cells) |
| ring | the `GEO_AMR_INTERFACE`-tagged cell band around the footprint, including the reactivated row at coarse position c=0 (ring row 2) |
| band | the full coupling band from c=−2 (bulk) through c=1 (F2C destination) per §2; the region the re-anchor re-registers |
| seam metric | face-mean Δ of rho/f00/f05/f06 between the re-paired rows of §5, computed per iteration from per-iteration `--write-dfs` frames |
| light-cone aisle | cells at Chebyshev distance ≥2 from every coupling patch, asserted bitwise-equal to the parent after ONE SimUpdate (stage-1 gate) |
| stored extent | the allocated per-axis row count, `2K+2 = local′ + 2·ov`, invariant across the re-anchor |
| indexer frames old/new | the fine-row index maps of plan §1.1 (old frame: destinations at −1, 0; new frame after the one-cell inward re-anchor: destinations at −2, −1; standard LBM starts at new local 0; F2C sources move from {2,3} to {1,2}) |

---

## 10. Evidence index (verified present 2026-08-19)

- `.omo/evidence/interface_bias/seam-investigation.md` — seam anatomy + probe ladder + 20-tc channel attribution matrix (committed parallel to docs ¶ ~271).
- `.omo/evidence/h9_retry/harvest_tc20.py` — 20-tc mass/KE harvester (pinned metric definition; DT = 2.241333 s/sample, ν = 1.5e-5, k = 2π/0.41).
- `.omo/evidence/literature/schonherr-ch7-close-read.md` — quote-level ch.7 close-read + figure atlas (Fig 7.1–7.10, PNG rasters in-situ).
- `docs/references/AMR/report_guzik_bvp_stencils.md` — Guzik 2014 BVP/IVP + constrained-LS + frozen-skin equivalence (§7).
- `docs/references/AMR/47_Schonherr_2015_Thesis.{pdf,txt}`, `50_Qi_2019_Musubi_CompactInterp.{pdf,txt}`, `25_Guzik_2014_Interpolation_AMR.{pdf,txt}`, `12_Chen_H_1998_Volumetric.*`.
- Committed docs: `docs/AMR-for-LBM-implementation.md` ¶ line ~271 (seam investigation record).

---

## Open questions

None. Every item required for implementation was normatively specified in the plan doc; anything not covered there is intentionally left to the stage commits and their gates rather than guessed here.
