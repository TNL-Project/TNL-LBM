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
| −0.75 (+¼ of c=−1) | C2F destination (ghost row 1) — filled every cycle; **simulated band: the streaming source of the inner row's substep-1 integration (never itself collided/streamed)** | ghost | old −1 → **new −2** |
| c=0 | C2F source line 2, SIMULATED | skin GEO_NOTHING → **GEO_AMR_INTERFACE** (ring row 2) | — |
| −0.25 (−¼ of c=0) | C2F destination (ghost row 2) — filled every cycle; **simulated band: INTEGRATED by the widened substep-1 fine kernel (collide+stream), then consumed by substep 2 as the boundary** | interior → ghost | old 0 → **new −1** |
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

## 3. Cycle contract (T7; **amended 2026-08-23 to the simulated band** — trigger of fork row (c) fired with the T16 null verdict)

Per cycle, per level:

`fine substep 1 (widened extent [−1, local+1): inner ghost rows INTEGRATED) → fine substep 2 (interior-only) → coarse step → F2C (once) → C2F (single fill of the substep-0 frame, BOTH ghost rows)`

- **Fine substep 1** runs on the widened extent: the inner ghost rows (ghost row 2 of the band map) collide + stream like interior fluid (GEO_FLUID by construction), sourcing the outer ghost row (ghost row 1); the interior sources the inner row's fill.
- **Fine substep 2** runs on the interior-only extent; its boundary data is substep 1's kernel-updated inner rows in the other AB frame — the band advances synchronously with the fine clock.
- **F2C** reads the rotation-1 post-substep-2 frame; runs **once per coarse step** (as it already did).
- **C2F runs once per cycle** — the single fill targets the substep-0 frame (consumed by the next cycle's substep 1) and covers **both** ghost rows of the band; SimInit does the same single-frame fill for cycle 0. The other frame needs no fill: substep 2 consumes substep 1's updated inner rows from it, and its outer row is unreachable (substep 2 is interior-only) — the former frame-1 fill was dead traffic under the simulated band and is **removed**.
- **F2C↔C2F order is irrelevant** — touched sets are disjoint (declared in docstring).
- **BVP refill = content no-op today** — the conversion-era fill #2 was content-identical to fill #1 (verified by review), so its removal was **lossless**. **H9 removal is lossless** on the same basis.
- **Probe-visible cycle-1 caveat:** the single-frame fill means cycle-1 fine1 reads a t_0 fill — the seam metric compares cycle-1 separately from cycle ≥ 2 (Oracle startup-transient note).
- **Checkpoint restart compatibility is declared INCOMPATIBLE** across the re-anchor (array shapes change). Declared, acceptable on this branch; intentional per fork (i).

---

## 4. Fork table (defaults; fallback arms pre-registered — Oracle F5)

| # | fork decision (default) | registered fallback arm |
|---|---|---|
| (a) | F2C destination row **frozen `GEO_NOTHING`** (Guzik-equivalence: frozen covered cell + adjusted stream semantics is verbatim-blessed — report_guzik_bvp_stencils.md §7) | **`F2C_DEST_ACTIVE`** collision-active-if-tagged variant — only budgeted/implemented if the T16 decision table is null on the frozen arm |
| (b) | deep footprint stays allocated frozen (storage-only superset; Schönherr's unallocated deep cells ≈ same physics — memory, not physics) | — |
| (c) | **CONVERTED (2026-08-23):** fine destination rows are **Schönherr's simulated destinations** — the inner ghost row is integrated by the widened substep-1 fine kernel (collide+stream like interior fluid); the outer ghost row stays fill-only as that integration's streaming source. The trigger of the registered fallback arm fired (T16 20-tc null verdict on every arm; measured verdict decided); the conversion-era passive band (IVP-class semantics, both-frames fill) is superseded — see §3. The arm landed as a **default behavior change** (no `F2C_SIMULATED_BAND` macro) | (superseded: the passive band of the conversion wave) |
| (d) | H9 + BVP refill: **hard removal** (`C2F_H9`, `h9_first_fill`, `c2f_time_centered`, CMake var + retirement warning idiom) | — |
| (e) | stage-3 reuses CM coefficient code (`sd/sa/sb/sc/sk` sums exist; extract shared helpers in T13 bitwise-gated) | — |
| (f) | MPI nproc=1 only; explicit non-goal; SimInit note | — |
| (g) | destination rows live in overlap storage; **amended with row (c) (2026-08-23):** the INNER destination row is additionally kernel-domain during fine substep 1 (widened launch extent [−1, local+1)); the outer destination row stays fill-only (never kernel-domain) | — |
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

### 8.1 Refreshed record (T18, 2026-08-22, at the conversion HEAD)

**Run.** Full developing-channel diagnostic at HEAD (the F2C_SCHONHERR default): `build/sim_AMR/sim_AMR_channel --resolution 1 --write-dfs --out3d-iter-period 1` — 641 coarse iterations + the cycle-0 frame (642 outputs, 144 s wall; env pin as the T16 ladder: RTX 5080 sm_120, CUDA 13.3, SP, AB, np=1). Census at HEAD on the channel footprint "1 24 4 4 16 8 8": 1,296 interface cells (halo 776 + reactivated c=0 shell 520), 504 frozen (c=1 skin 312 + deep 192); 6 interface + 6 interior patches; BC-clamped mass invariant at print precision (1.404375e+04, the Dirichlet Qi-drift masking working as designed).

**Index mapping.** The old pairing's FINE row (slab x-min face, old fine local 0 = fine-global 48, center 24.25, vs coarse ring c=−1 at domain 23) is **not re-readable**: old local 0 is now destination row −1 (depth-2 ghost, outside the VTKHDF Level-1 extent). The §5 re-paired pairing on the slab: fine standard row **new local 0** (fine-global 49, center 24.75) vs coarse ring row 2 **c=0** (domain x = 24, center 24.5); tangent windows fine [0:14), coarse [4:12). The committed seam metric's cubic reshape cannot read the channel's non-cubic Level extents (30×14×14 / 64×16×16), so the refresh used a one-off session probe with identical face-mean-Δ semantics (float64, %.10e; `channel_seam_probe.py` in the evidence dir).

**Refreshed seam table (re-paired rows).** Startup cycles 3–10: rho mean **−2.1667504874e-09** (uniform sign — the slab is untouched there: in 10 steps the front has advected ~1 coarse cell from inflow at U_lb = 0.1 and even the acoustic signal at c_s ≈ 0.577 has travelled < 6 cells, far short of the slab's x-min face at x = 24; cycle-0 rows bitwise 0 by the uniform IC). Front arrival before cycle 40 (rho −4.1458e-02). Quasi-steady (last 100 cycles):

| field | mean | signed/abs |
|---|---|---|
| rho | **−3.4941984034e-02** | −1.0000 |
| f00 | −9.5707324016e-03 | −1.0000 |
| f05 | −4.7143329692e-03 | −1.0000 |
| f06 | +1.1198423597e-03 | +1.0000 |

The same fine row vs the OLD coarse-row c=−1 column: at quasi-steady the two read within ~1.5e-3 of each other (cycle 640: −3.4436e-02 vs −3.4826e-02), transiently up to 2.8e-2 apart during the front's arrival (cycle 40: −4.1458e-02 vs −6.9014e-02) — the re-pairing changes the row identities, not the class.

**Interpretation.** The B.7-era arm-common deficit reading stands: the ~6e-2 discontinuity bracket the B.7 record tabulated at the slab's min face (cycle-28 coarse centerline 1.1507 / 1.1553 / 1.0966 at x = 23/24/25) re-reads on the converted band as a ~3.5e-02 standing seam jump on the re-paired rows at quasi-steady, uniform-sign. The conversion's band registration was never expected to move this class — the deficit was measured arm-common (B-off and B-on alike), not a differential defect of any transfer path; the refreshed state confirms it persists at the same order at HEAD.

**Evidence** (untracked): `.omo/evidence/schonherr_conversion/t18-b7-channel/` — `sim_channel_probe.log` (sim stdout), `channel_seam_probe.py` (the one-off probe), `channel_seam_probe.log` (the digest above), `channel_seam_series.csv` (all 642 per-iteration rows), `results_sim_AMR_channel_res01_np001/` (642 frames), `adios2.xml`.

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

---

## 11. Addendum: multi-level nesting (2026-08-28)

This contract, written for two levels (coarse + one fine), generalizes to **N statically nested 2:1 levels** (shipped as
commits A–G of `.omo/plans/amr-nlevel-nesting.md`, branch HEAD `5214b01`; implementation chapter
`docs/AMR-for-LBM-implementation.md` §13). The addendum records how the binding rules carry over; the contract body above is
unmodified.

- **Per-pair extension, unchanged.** The band map (§2), the simulated-band cycle contract (§3), and the map-pattern rails
  remain binding **per adjacent level pair**: couplings are strictly adjacent-pair (nothing couples L to L−2), each finer
  level re-tags only its own parent's map, and the band registration is identical at every pair.
- **Schedule recursion.** The single-level cycle of §3 is the `max_level == 1` reduction of the `advancePair(L)` recursion:
  level L runs exactly 2^L substeps per coarse step, F2C once per parent substep (frame-forced, not a physics choice), the
  C2F fill once per pair before its substep 1 (sourcing the parent's live post-substep-A state mid-cycle), and a
  level-ascending re-arm + C2F cascade at cycle end (the AA in-place discipline, per §3's ordering argument). The reduction
  is census-locked byte-identical at 1, 2 and 3 fine levels under both streaming patterns (`tests/test_amr_nesting_{ab,aa}`).
- **New validation V5–V10** (hard `createAMRBlocks` floors: ascending file order, unique containing parent, telescoping
  gap ≥ 2 with the wall-candidate s = −1 exception, sibling Chebyshev separation, wall-candidate chain agreement) plus the
  **V9 advisory warn tier** (gap 2 valid, warn below 3 — user-decided 2026-08-27); V1–V4 are this contract's pre-existing
  checks.
- **Wall chain.** Wall-shared faces (the s = −1 lane) chain through nested levels: masks key on the immediate parent's map
  over its storage extent, **R4 wall-pedestal prisms** author the frozen rows behind the parent's upward own-8 F2C window,
  and `F2C_LAGRAVA` + nested wall sharing is a hard SimInit error (the §7.2 σ = 2 default is unaffected).
- **Evidence discipline.** The pre-registered instrument for proving v1 unchanged is the bit-identity harness
  (`tests/regression/test_amr_bitidentity.py`, verify mode against the committed manifest), complementing the 10-target
  gate; the conversion-era §6 gates are unaffected.

---

## Appendix: §7.2 equation audit

**Provenance:** plan row 10 (T9) worksheet — the §7.2 equation-by-equation audit of 2026-08-19, extended (A.5-U1) and integrated into this contract doc at commit 11 (T11, 2026-08-21). Session draft: `.omo/evidence/schonherr_conversion/eq-audit-draft.md`.
**Audit target:** `include/lbm3d/d3q27/amr_coupling.h` — CM branch of `cudaAMR_CoarseToFine` (:340–1000) plus the named τ/σ sites and the carve (:408–565), the legacy Lagrange opt-out (:1002–1116), and the F2C kernel (`cudaAMR_FineToCoarse`, :1245–1461).
**Thesis source:** `docs/references/AMR/47_Schonherr_2015_Thesis.txt`, §7.2 (printed pp. 57–61) and §7.3 (printed pp. 61–63).
**Cross-check:** `47_Schonherr_2015_Thesis.pdf` rendered at 150 dpi (physical pp. 70–75; physical = printed + 13) — used to verify every quoted equation where the `.txt` linearization is ambiguous (all of §7.2's numbered equations, and decisively Eqs. 7.18, 7.22–7.24, 7.29–7.33, 7.38–7.57).
**Anchor policy:** symbols first; `:line` numbers are an advisory snapshot of `amr_coupling.h` as of commit 11 and are not re-maintained per commit (anchor-rot policy of §7).

### A.0 Method and reading conventions

- Per equation: thesis quote (as printed, PDF-verified) + code quote + verdict ∈ {match, convention-difference, bug} + one-line justification.
- The `.txt` flattens math; ambiguous lines were resolved against the PDF renders, not against expectation. Two print artifacts matter for the row numbering and the "print typo" findings:
  1. **Eqs. 7.49+7.50 are ONE formula** (â₀ split over two printed lines, both right-margin-tagged) → 57 tags, 56 formulas.
  2. The thesis print itself carries typos: 7.54 prints "+ _off·d_y" (dropped y-subscript), 7.56/7.57 misprint the LHS as d̂_x (intended d̂_y, d̂_z).
- Eqs. 7.5–7.9 print the ω_s factor in front of a bracketed fraction; the bracket placement is unambiguous only with the equilibrium-vanishing requirement (k ≡ 0 when Π^neq = 0). Read as −3ω_s·[Σijf/ρ − uv] etc., which is also the only dimensionally/equilibrium-consistent reading and the Musubi semantics; flagged in §A.5 (U2).
- All numbered-code fragments quoted verbatim from `include/lbm3d/d3q27/amr_coupling.h` at this readonly session; constants (`ciselnik.h`): `n1o2=1/2`, `n1o3=1/3`, `n1o4=1/4`, `n1o8=1/8`, `n1o32=1/32` (code :777 computes it as `n1o8*n1o4`), `n3o2=3/2`, `no1..no16` integers.
- The b/c velocity-coefficient families are **not printed** in the thesis ("only the formulas of the a coefficients are shown", txt :3221–3224); the code's b/c rows are audited as the cyclic closure of the a-family rows.
- F2C side: `cudaAMR_FineToCoarse` implements **no** §7.2 content today (Lagrava filter + FH τ-rescale; replacement is T14/commit 13). F2C-impact notes are recorded per row where relevant.

### A.1 Notation glossary (thesis ↔ code) — input for T13 helper extraction

| thesis symbol | code identifier + site | role |
|---|---|---|
| source-node local coords (±½)³ | `xn,yn,zn = static_cast<dreal>(ib*) - n1o2`, :577/:580/:583 | fit stencil positions |
| destination evaluation point (±¼)³ | `tx,ty,tz` from `axis_window`, :383–401, set :404–406 (carve may re-offset to \|t_rel\| = 0.75, :504–505) | C2F destination in window frame |
| ρ at source node | `rho_n` via `COLL::computeDensityAndVelocity(KS_C)`, :594 (+ `common.h:17–50`) | Eq. 7.1 |
| u,v,w at source node | `u,v,w` = `KS_C.vx/vy/vz`, :596–598 | Eqs. 7.2–7.4 |
| ω_s (source = coarse) | `omega_s = no1 / tau_coarse`, :385 | Eqs. 7.5–7.9 prefactor |
| ω_d (destination = fine) | `omega_d = no1 / tau_fine`, :386 | Eqs. 7.38–7.43 prefactor |
| Π_ab = Σq c_a c_b (f−f^eq) | `Pi_xx..Pi_yz`, :608–624 (vel tables `vel_cx/cy/cz`, :364–366; `COLL::setEquilibrium(KS_E)`, :607) | algebraic form of the Eq. 7.5–7.9 raw-moment differences |
| k_xy, k_yz, k_xz | `k_xy, k_yz, k_xz`, :665–667 (`-no3*om_rho*Pi_*`) | Eqs. 7.5–7.7 |
| k_xx−yy, k_xx−zz | `k_xx_yy, k_xx_zz`, :668–669 (`-n3o2*om_rho*(Pi_xx−Pi_yy/zz)`), `om_rho = omega_s/rho_n` :664 | Eqs. 7.8–7.9 |
| (k_yy−xx+k_yy−zz) etc. cyclic combos | `K_a = k_xx_yy + k_xx_zz`, `K_b = k_xx_zz − no2*k_xx_yy`, `K_c = k_xx_yy − no2*k_xx_zz`, :672–674 | diagonal-moment families of a₀/b₀/c₀ + a_xx/… (developed in Eq. 7.22 explicitly; K_b = k_yy−zz+k_yy−xx, K_c = k_zz−xx+k_zz−yy by trace identities) |
| d₀,d_x,d_y,d_z,d_xy,d_yz,d_xz,d_xyz | sums `sd0..sdxyz` :571, accumulated :704–711; coefficients `d_0..d_xyz` :764–771 | Eqs. 7.10–7.17 |
| a₀,a_x,…,a_xyz | sums `sa0..saxyz` :572, accumulated :714–724; coefficients `a_0..a_xyz` :778–788 (b: `sb*` :573/:727–737/:789–799; c: `sc*` :574/:740–750/:800–810) | Eqs. 7.18–7.28 (+ cyclic, unprinted) |
| 8-node k-sums (for avk) | `sk_xy, sk_yz, sk_xz, sk_xx_yy, sk_xx_zz` :575, accumulated :753–757 | numerators of Eqs. 7.29–7.33 |
| avk_xy … avk_xx−zz | `avg_k_xy … avg_k_xx_zz`, :820–824 | Eqs. 7.29–7.33 |
| destination macros ρ,u,v,w | `rho_f` :772; `vx_f/vy_f/vz_f` :811–816 | Eqs. 7.34–7.37 |
| σ (c→f) | `sigma = n1o2`, :828 | Eqs. 7.38–7.43 scaling |
| A_011, A_101, A_110 | `A011, A101, A110`, :831–833 | Eqs. 7.44–7.46 |
| B, C | `corr_B, corr_C`, :829–830 | Eqs. 7.47–7.48 |
| σρ/(3ω_d), 2σρ/(9ω_d) | `off_factor` :834, `diag_factor` :842 | cumulant prefactors |
| C_011…C_002 | `C011, C101, C110` :835–837; `C200, C020, C002` :844–846 via `X,Y` :840–841 and `diag_eq = rho_f*n1o3` :843 | Eqs. 7.38–7.43 |
| κ_000 = ρ, κ_100 = 0, κ_≥3 = 0 | `ks_000..ks_111` :854–870 | Step G (prose :3406–3412) |
| back-transformation (as §2.2.5) | Step H :877–1000 (copied from `col_cum.h` non-`USE_GEIER_CUM_2017` path, Geier 2015 Eqs. 81–96: `col_cum.h:344–366` sibling) | prose :3486–3488 |
| hat coefficients â/d̂ (offset) | **no literal hat coefficients**; analog = carve pre-pass `carve_window_off_covered_cells` :487–508 + degenerate collapse :545–565 | Eqs. 7.49–7.57 |
| FH τ-rescale (Filippova–Hänel) | `neq_scale` at :329 (explosion opt-in), :1112 (Lagrange opt-out), :1427 (F2C kernel) | NOT §7.2 — the pre-conversion scheme |

### A.2 Explicit confirmations (plan T9 items i–iii)

#### A.2.1 (i) σ = n1o2 already implemented at the C2F CM branch — CONFIRMED

The σ-form of Eqs. 7.38–7.43 with σ_c→f = 1/2 and ω_d = ω₁(destination) is the
code's compact-moment **default** branch, verbatim:

- :385–386 `const dreal omega_s = no1 / tau_coarse;` / `const dreal omega_d = no1 / tau_fine;`
- :828 `const dreal sigma = n1o2;`
- :834 `const dreal off_factor = sigma * rho_f / (no3 * omega_d);`
- :842 `const dreal diag_factor = no2 * sigma * rho_f / (no9 * omega_d);`
- docstring :826–827 "[…] (Eqs. 7.38-7.48); sigma_{c->f} = 1/2, omega_d is
  the destination (fine) grid rate".

Thesis anchor (txt :3456–3458, PDF-verified): "[…] For the coarse to fine
interpolation it is σ_c→f = 1/2, for the fine to coarse interpolation it is
σ_f→c = 2." — i.e. plan §0/contract §1 item "C2F already σ-form (F1)" holds;
not re-litigated here, per charter.

#### A.2.2 (ii) FH τ-rescale confined to the F2C kernel and the legacy Lagrange branch — CONFIRMED (3 sites; one more than plan §0 lists)

1. **F2C kernel** (:1426–1430): `// volumetric rescaling, f_coarse[q] = eq_q(rho_c,u_c) + (tau_c/tau_f)*f_neq[q]`
   `const dreal neq_scale = tau_coarse / tau_fine;` → `const dreal f_coarse = KS_EQ.f[q] + neq_scale * (f_avg[q] - KS_EQ.f[q]);`
   (kernel docstring :1195–1199 step 4).
2. **Legacy Lagrange C2F opt-out** (:1111–1114): `// volumetric rescaling, f_fine[q] = eq_q(rho_f,u_f) + (tau_f/tau_c)*f_neq[q]`
   `const dreal neq_scale = tau_fine / tau_coarse;` → `store_fine_df(q, KS_F.f[q] + neq_scale * f_neq[q]);`
3. **`C2F_LINEAR_EXPLOSION` debug/opt-in branch** (:329–331):
   `const dreal neq_scale = tau_fine / tau_coarse;` → `store_fine_df(q, KS_EQ.f[q] + neq_scale * (KS_H.f[q] - KS_EQ.f[q]));`
   (not in plan §0's two-site list; opt-in only, default-off — noted for
   completeness since commit 13's "no τ-rescale" scope statement should name
   all three).
A repo-wide grep shows **no other** τ-ratio/neq-injection sites (`include/`,
`sim_AMR/`).

#### A.2.3 (iii) F2C evaluation point t = (0,0,0): vanishing-coefficient census (input to T14's F2C_SCHONHERR minimal-macro set)

Thesis anchors: F2C destination "is at (0, 0, 0) in the local coordinate
system" (txt :3391–3394); "only the coefficients d₀, a₀, b₀ and c₀ are
required to set the density and the three velocities at the coarse destination
node" (txt :3406–3408). Evaluating the code's own CM definitions at
tx = ty = tz = 0:

**Vanish identically (every term carries ≥1 coordinate factor):**

- d_x,d_y,d_z,d_xy,d_yz,d_xz,d_xyz in `rho_f` (:772) → `rho_f = d_0`.
- a_x..a_xyz, b_*, c_* in `vx_f/vy_f/vz_f` (:811–816) → `vx_f = a_0`,
  `vy_f = b_0`, `vz_f = c_0`.
- All five aggregates `corr_B, corr_C, A011, A101, A110` (:829–833) ≡ 0.
  → T14's pinned "A/B/C ≡ 0" **verified**.

**Cancel algebraically (not zero individually, but drop out of the cumulants):**

- avk gradient terms (:820–824) vs the Step-F summands (:835–837), e.g.
  `C110 = -off_factor*(a_y + b_x + avg_k_xy + A110)` with
  `avg_k_xy = n1o8*sk_xy - (a_y + b_x)` ⇒ `C110 = -off_factor*(n1o8*sk_xy)`.
  Same cancellation in `C011`, `C101`; and in the diagonals:
  `X = n1o8*sk_xx_yy`, `Y = n1o8*sk_xx_zz` (:840–841 with corr_B=corr_C=0).
  T14 "avk corrections retained" is honored either literally (compute as
  written) or reduced — the two are value-identical at t = 0.

**Reduced F2C cumulants at (0,0,0)** (σ = 2, ω_d = 1/τ_coarse under T14):

- `C011 = −(σρ/3ω_d)·⅛·sk_yz`; `C101 = −(σρ/3ω_d)·⅛·sk_xz`; `C110 = −(σρ/3ω_d)·⅛·sk_xy`
- `C200 = ρ/3 − (2σρ/9ω_d)·⅛·(sk_xx_yy + sk_xx_zz)`
- `C020 = ρ/3 − (2σρ/9ω_d)·⅛·(−2·sk_xx_yy + sk_xx_zz)`
- `C002 = ρ/3 − (2σρ/9ω_d)·⅛·(sk_xx_yy − 2·sk_xx_zz)`

**Minimal live accumulator set for F2C_SCHONHERR (T13/T14):**
`sd0, sa0, sb0, sc0, sk_xy, sk_yz, sk_xz, sk_xx_yy, sk_xx_zz` (9 sums) —
note that `sa0/sb0/sc0` (:714/:727/:740) already fold in the
K_a/K_b/K_c + k_xy/k_xz/k_yz cross terms, so the per-source k-moment pipeline
(:607–674) stays live in full; the 37 other accumulators
(sdx..sdxyz; sax..saxyz; sb*; sc*) and the Step-C/D prefactor fits
(:776–824) are **dead at t = 0** (still live for C2F).

**⚠ T14-critical caveat:** `a_0` itself carries the k-corrected sum
`sa0 += −xn·K_a − 2·yn·k_xy − 2·zn·k_xz + 4u + 4·xn·yn·v + 4·xn·zn·w`
(:714) with `K_a = k_xx_yy + k_xx_zz` — the **Eq. 7.18 deviation family**
(§A.3.3). The F2C destination **velocity** therefore inherits the 7.18
verdict: print-7.18 would carry `−x·k_xx−yy` instead of `−x·(k_xx−yy +
k_xx−zz)` (and likewise in the b₀/c₀ analogs). The 7.18/7.23/7.24 decision
(A.4-R1) was locked at commit 10 and is reused at commit 13.

### A.3 Per-equation worksheet

Code quotes are from `include/lbm3d/d3q27/amr_coupling.h`; thesis quotes from
the PDF-checked print (txt lines cited for retrieval). Verdict labels:
**match** · **conv** = convention-difference (convention stated) · **bug**.

#### A.3.1 Moments (Eqs. 7.1–7.9)

| Eq | Thesis (print) | Code site + quote | Verdict — justification |
|---|---|---|---|
| 7.1 | ρ = Σ_ijk f_ijk (p.57; txt :3074–3076) | `COLL::computeDensityAndVelocity(KS_C)` :594 → `common.h:19–35` Kahan/grouped sum of `KS.f` | **match** — zeroth raw moment per node, identical definition |
| 7.2 | u = Σ i f_ijk/ρ (p.57; txt :3082–3088) | `KS_C.vx` via :594 → `common.h:42–45` (Σ c_x f + fx/2)/ρ, `fx=0` in the fresh KS (`defs.h:274`) | **match** — coupling KS has zero force terms, so it reduces to the thesis's force-free first moment (note: volume-forcing half-offset is absent by construction — thesis §7.2 defines no force either; observation not a defect, see §A.4-R3; one-clause note at the macro read :585–590) |
| 7.3 | v = Σ j f_ijk/ρ (p.57; txt :3089–3091) | `KS_C.vy` :596–597 → `common.h:46–49` | **match** — as 7.2 |
| 7.4 | w = Σ k f_ijk/ρ (p.57; txt :3092–3094) | `KS_C.vz` :598 → `common.h:38–41` | **match** — as 7.2 |
| 7.5 | k_xy = −3ω_s[Σijf/ρ − uv] (p.57; txt :3103–3117) | :664–665 `om_rho = omega_s/rho_n; k_xy = -no3 * om_rho * Pi_xy;` with `Pi_xy += cqx*cqy*f_neq` :615/:621 | **match** — `Pi_xy = Σij(f−f^eq)`; Σij f^eq = ρuv exactly for the used D3Q27 equilibrium (2nd-order Hermite constraint) ⇒ algebraically identical to print; ω_s = 1/τ_coarse :385 matches "ω₁ of the source grid" (txt :3151; ν = ⅓(1/ω₁−½) ⇔ τ = 3ν+½) |
| 7.6 | k_yz = −3ω_s[Σjkf/ρ − vw] (p.57; txt :3106–3121) | :666 `k_yz = -no3 * om_rho * Pi_yz;` | **match** — as 7.5 |
| 7.7 | k_xz = −3ω_s[Σikf/ρ − uw] (p.57; txt :3110–3123) | :667 `k_xz = -no3 * om_rho * Pi_xz;` | **match** — as 7.5 |
| 7.8 | k_xx−yy = −(3/2)ω_s[Σ(i²−j²)f/ρ − (u²−v²)] (p.57; txt :3125–3138) | :668 `k_xx_yy = -n3o2 * om_rho * (Pi_xx - Pi_yy);` | **match** — Pi_xx−Pi_yy = Σ(i²−j²)(f−f^eq); Σ(c_x²−c_y²)f^eq = ρ(u²−v²) exactly ⇒ identical; n3o2 = 3/2 ✓ |
| 7.9 | k_xx−zz = −(3/2)ω_s[Σ(i²−k²)f/ρ − (u²−w²)] (p.57; txt :3141–3149) | :669 `k_xx_zz = -n3o2 * om_rho * (Pi_xx - Pi_zz);` | **match** — as 7.8 |

#### A.3.2 Density coefficients (Eqs. 7.10–7.17)

| Eq | Thesis (print) | Code site + quote | Verdict — justification |
|---|---|---|---|
| 7.10 | d₀ = (1/8)Σρ (p.58; txt :3165–3171) | :704 `sd0 += rho_n;` → :764 `d_0 = n1o8 * sd0;` | **match** |
| 7.11 | d_x = (1/2)Σ xρ (p.58; txt :3173–3179) | :705 `sdx += xn * rho_n;` → :765 `d_x = n1o2 * sdx;` | **match** |
| 7.12 | d_y = (1/2)Σ yρ (p.58; txt :3181–3187) | :706 → :766 | **match** |
| 7.13 | d_z = (1/2)Σ zρ (p.58; txt :3189–3195) | :707 → :767 | **match** |
| 7.14 | d_xy = 2Σ xyρ (p.58; txt :3197–3199) | :708 `sdxy += xn*yn*rho_n;` → :768 `d_xy = no2 * sdxy;` | **match** |
| 7.15 | d_yz = 2Σ yzρ (p.58; txt :3201–3203) | :709 → :769 | **match** |
| 7.16 | d_xz = 2Σ xzρ (p.58; txt :3205–3207) | :710 → :770 | **match** |
| 7.17 | d_xyz = 8Σ xyzρ (p.58; txt :3209–3211) | :711 `sdxyz += xn*yn*zn*rho_n;` → :771 `d_xyz = no8 * sdxyz;` | **match** |

#### A.3.3 Velocity coefficients (Eqs. 7.18–7.28) — **contains the only code↔print deviations in §7.2**

| Eq | Thesis (print, PDF-verified) | Code site + quote | Verdict — justification |
|---|---|---|---|
| 7.18 | a₀ = (1/32)Σ[**−x k_xx−yy** − 2y k_xy − 2z k_xz + 4u + 4xy v + 4xz w] (p.59; txt :3230–3236) — PDF: single k_xx−yy, NO k_xx−zz | :714 `sa0 += -xn * K_a - no2*yn*k_xy - no2*zn*k_xz + no4*u + no4*xn*yn*v + no4*xn*zn*w;` with :672 `K_a = k_xx_yy + k_xx_zz;` prefactor :777–778 `n1o32` | **conv** (code implements the closed/consistent family: −x(k_xx−yy+k_xx−zz); print carries −x·k_xx−yy alone). Code family satisfies the nodal-consistency identity a₀ = ⅛Σu − (a_xx+a_yy+a_zz)/4 for **any** strain carrier (proof §A.3.3-note); the printed set satisfies it for none; thesis print carries typos elsewhere (7.54/7.56/7.57). Same pattern in the unprinted b₀/c₀ analogs (code :727/:740 exactly cyclic, K_b = k_yy−zz+k_yy−xx ⇔ k_xx_zz−2k_xx_yy, K_c = k_zz−xx+k_zz−yy ⇔ k_xx_yy−2k_xx_zz). Suspected thesis-print erratum; **do not align code → print** (would break quadratic exactness, §A.4-R1); the errata record is in the code at :676–701 |
| 7.19 | a_x = (1/2)Σ xu (p.59; txt :3238–3244) | :715 `sax += xn*u;` → :779 `a_x = n1o2*sax;` | **match** |
| 7.20 | a_y = (1/2)Σ yu (p.59; txt :3246–3252) | :716 → :780 | **match** |
| 7.21 | a_z = (1/2)Σ zu (p.59; txt :3254–3260) | :717 → :781 | **match** |
| 7.22 | a_xx = (1/8)Σ[**x(k_xx−yy + k_xx−zz)** + 4xy v + 4xz w] (p.59; txt :3262–3268) — PDF: BOTH diagonal moments, explicitly | :718 `saxx += xn*K_a + no4*xn*yn*v + no4*xn*zn*w;` → :782 `a_xx = n1o8*saxx;` | **match** — print and code both use the k-sum here; makes the 7.18 singleton internally asymmetric in the print (extra evidence for the 7.18 erratum hypothesis) |
| 7.23 | a_yy = (1/8)Σ[**y k_xy − 4xy v**] (p.59; txt :3270–3276) — PDF: coefficients 1 and 4 | :719 `sayy += no2*yn*k_xy - no8*xn*yn*v;` → :783 `a_yy = n1o8*sayy;` (effective (1/8)Σ[2y k_xy − 8xy v]) | **conv** — code carries exactly 2× the printed inner terms; the doubled form is the one that closes the nodal identity jointly with the code's 7.18/7.22/7.24 (print-consistent reading fails it; §A.3.3-note). Cyclic analogs: code `sbxx` :731, `scxx` :744 |
| 7.24 | a_zz = (1/8)Σ[**z k_xz − 4xz w**] (p.59; txt :3278–3284) — PDF: coefficients 1 and 4 | :720 `sazz += no2*zn*k_xz - no8*xn*zn*w;` → :784 | **conv** — as 7.23 (cyclic analogs `sbzz` :733, `scyy` :745) |
| 7.25 | a_xy = 2Σ xy u (p.59; txt :3286–3288) | :721 `saxy += xn*yn*u;` → :785 `a_xy = no2*saxy;` | **match** |
| 7.26 | a_yz = 2Σ yz u (p.59; txt :3290–3292) | :722 → :786 | **match** |
| 7.27 | a_xz = 2Σ xz u (p.59; txt :3294–3296) | :723 → :787 | **match** |
| 7.28 | a_xyz = 8Σ xyz u (p.59; txt :3298–3300) | :724 `saxyz += xn*yn*zn*u;` → :788 `a_xyz = no8*saxyz;` | **match** |

**§A.3.3-note (why the code family is the consistent one — nodal-identity
proof sketch).** With trilinear parts fixed by Eqs. 7.19–7.21/7.25–7.28, the
11-coefficient fit has a 3-dim ambiguity (polynomials vanishing at all 8
corners: {x²−¼, y²−¼, z²−¼}); resolving it so the polynomial still takes the
nodal values forces **a₀ = ⅛Σu − (a_xx+a_yy+a_zz)/4**.
Substituting the code's definitions cancels identically (all k and cross-terms
close: (1/8)·2 vs (1/32)·2 bookkeeping balances). Substituting the printed
7.18 (single k_xx−yy) + printed 7.23/7.24 (coefficients 1,4) leaves
mismatched strain terms for **every** choice of the strain↔k proportionality
— e.g. on the divergence-free quadratic pair (u = qy², v = w = 0) with
k := strain: print yields a_yy = q/2 (should be q) and a₀ = 0 only via
cancellation luck; on u = qx² it yields a₀ = q(2−c)/8 with no carrier value c
making a₀ = 0, while the code gives a₀ = 0, a_xx = cq, a_yy = a_zz = 0 — exact
for c = 1. Conclusion: the code family is the exactly-consistent, cyclically
closed variant; the print is internally inconsistent → thesis-print erratum is
the working explanation (flag U1). Verified: Tests 8/9
(`tests/test_amr_coupling.cu` `CMLinearField`/`CMQuadraticField`, :1541–1583)
assert exactly this exactness class against the code as-is, and the T10c lock
(`tests/unit/test_amr_schonherr_exactness.cu`) discriminates the two families
(code family green; the print-aligned family would sit ~1e-5 off).

#### A.3.4 Averaged moments avk (Eqs. 7.29–7.33)

| Eq | Thesis (print, PDF-verified on p.59) | Code site + quote | Verdict — justification |
|---|---|---|---|
| 7.29 | avk_xy = (1/8)Σk_xy − (a_y + b_x) (p.59–60; txt :3313–3337) | :753 `sk_xy += k_xy;` → :820 `avg_k_xy = n1o8*sk_xy - (a_y + b_x);` | **match** |
| 7.30 | avk_yz = (1/8)Σk_yz − (b_z + c_y) (txt :3339–3341) | :754 → :821 | **match** |
| 7.31 | avk_xz = (1/8)Σk_xz − (a_z + c_x) (txt :3343–3345) | :755 → :822 | **match** |
| 7.32 | avk_xx−yy = (1/8)Σk_xx−yy − (a_x − b_y) (txt :3363–3370) | :756 → :823 | **match** |
| 7.33 | avk_xx−zz = (1/8)Σk_xx−zz − (a_x − c_z) (txt :3372–3374) | :757 → :824 | **match** — note the subtractions involve only the plain nodal-gradient coefficients a_x,a_y,b_x,b_y,c_y,c_z (Eqs. 7.19–7.21 class), NOT the 7.18/7.23/7.24-deviant ones, so the avk rows are unaffected by R1 |

#### A.3.5 Evaluation polynomials (Eqs. 7.34–7.37)

| Eq | Thesis (print, PDF-verified) | Code site + quote | Verdict — justification |
|---|---|---|---|
| 7.34 | u = a₀ + x(a_x + x a_xx + y a_xy + z a_xz + yz a_xyz) + y a_y + z a_z + y²a_yy + z²a_zz + yz a_yz (p.60; txt :3399) | :811–812 `vx_f = a_0 + tx*(a_x + tx*a_xx + ty*a_xy + tz*a_xz + ty*tz*a_xyz) + ty*a_y + tz*a_z + ty*ty*a_yy + tz*tz*a_zz + ty*tz*a_yz;` | **match** — term-for-term, tx/ty/tz = thesis x/y/z at (±¼)³ (:404–406; anchors txt :3391–3394) |
| 7.35 | v = b₀ + x(b_x + …) + … (txt :3400) | :813–814 | **match** |
| 7.36 | w = c₀ + x(c_x + …) + … (txt :3401) | :815–816 | **match** |
| 7.37 | ρ = d₀ + x d_x + y d_y + z d_z + xy d_xy + xz d_xz + yz d_yz + xyz d_xyz (txt :3402–3404) | :772 `rho_f = d_0 + d_x*tx + d_y*ty + d_z*tz + d_xy*tx*ty + d_xz*tx*tz + d_yz*ty*tz + d_xyz*tx*ty*tz;` | **match** |

#### A.3.6 Second-order cumulants (Eqs. 7.38–7.43)

| Eq | Thesis (print, PDF-verified) | Code site + quote | Verdict — justification |
|---|---|---|---|
| 7.38 | C_011 = −(σρ/3ω_d)(b_z + c_y + avk_yz + A_011) (p.60–61; txt :3414–3440) | :834–835 `off_factor = sigma*rho_f/(no3*omega_d); C011 = -off_factor*(b_z + c_y + avg_k_yz + A011);` | **match** |
| 7.39 | C_101 = −(σρ/3ω_d)(a_z + c_x + avk_xz + A_101) (txt :3417–3442) | :836 `C101 = -off_factor*(a_z + c_x + avg_k_xz + A101);` | **match** |
| 7.40 | C_110 = −(σρ/3ω_d)(a_y + b_x + avk_xy + A_110) (txt :3421–3446) | :837 `C110 = -off_factor*(a_y + b_x + avg_k_xy + A110);` | **match** |
| 7.41 | C_200 = ρ/3 − (2σρ/9ω_d)[(a_x−b_y+avk_xx−yy+B) + (a_x−c_z+avk_xx−zz+C)] (txt :3425–3447) | :840–844 `X = a_x-b_y+avg_k_xx_yy+corr_B; Y = a_x-c_z+avg_k_xx_zz+corr_C; diag_factor = no2*sigma*rho_f/(no9*omega_d); diag_eq = rho_f*n1o3; C200 = diag_eq - diag_factor*(X+Y);` | **match** |
| 7.42 | C_020 = ρ/3 − (2σρ/9ω_d)[−2(X) + (Y)] (txt :3429–3448) | :845 `C020 = diag_eq - diag_factor*(-no2*X + Y);` | **match** |
| 7.43 | C_002 = ρ/3 − (2σρ/9ω_d)[(X) − 2(Y)] (txt :3433–3449) | :846 `C002 = diag_eq - diag_factor*(X - no2*Y);` | **match** |

Also matching (prose anchors): σ_c→f = 1/2 ⇒ `sigma = n1o2` :828; "ω_d is ω₁
on the destination grid" ⇒ `omega_d = no1/tau_fine` :386; "zeroth cumulant
replaced by κ_000 = ρ; first-order central moments zero; all ≥3rd-order
cumulants zero" ⇒ Step G `ks_000 = rho_f`, `ks_100/010/001 = 0`,
`ks_210..ks_111 = 0` (:854–870) — and the derived 4th-order central moments
`ks_211..ks_222` (:884–900) come from the 2nd-order cumulants via Geier 2015
Eqs. 81–84 exactly as in the collision operator (thesis: "transformed to
distributions as in the collision operator (see Section 2.2.5)", txt
:3486–3488); the back-transformation Eqs. 88–96 chain :877–1000 is verbatim the
`col_cum.h:344+` (non-`USE_GEIER_CUM_2017`) path with velocity from Eqs.
7.34–7.36 (:873–875; thesis txt :3487).

#### A.3.7 Aggregates (Eqs. 7.44–7.48)

| Eq | Thesis (print, PDF-verified) | Code site + quote | Verdict — justification |
|---|---|---|---|
| 7.44 | A_011 = b_xz x + c_xy x + b_yz y + 2c_yy y + b_xyz xy + 2b_zz z + c_yz z + c_xyz xz (p.61; txt :3466–3468) | :831 `A011 = b_xz*tx + c_xy*tx + b_yz*ty + no2*c_yy*ty + b_xyz*tx*ty + no2*b_zz*tz + c_yz*tz + c_xyz*tx*tz;` | **match** — 8/8 terms |
| 7.45 | A_101 = a_xz x + 2c_xx x + a_yz y + c_xy y + a_xyz xy + 2a_zz z + c_xz z + c_xyz yz (txt :3470–3472) | :832 `A101 = a_xz*tx + no2*c_xx*tx + a_yz*ty + c_xy*ty + a_xyz*tx*ty + no2*a_zz*tz + c_xz*tz + c_xyz*ty*tz;` | **match** — 8/8 terms |
| 7.46 | A_110 = a_xy x + 2b_xx x + 2a_yy y + b_xy y + a_yz z + b_xz z + a_xyz xz + b_xyz yz (txt :3474–3476) | :833 `A110 = a_xy*tx + no2*b_xx*tx + no2*a_yy*ty + b_xy*ty + a_yz*tz + b_xz*tz + a_xyz*tx*tz + b_xyz*ty*tz;` | **match** — 8/8 terms |
| 7.47 | B = 2a_xx x − b_xy x + a_xy y − 2b_yy y + a_xz z − b_yz z − b_xyz xz + a_xyz yz (txt :3478–3480) | :829 `corr_B = no2*a_xx*tx - b_xy*tx + a_xy*ty - no2*b_yy*ty + a_xz*tz - b_yz*tz - b_xyz*tx*tz + a_xyz*ty*tz;` | **match** — all 8 terms incl. signs |
| 7.48 | C = 2a_xx x − c_xz x + a_xy y − c_yz y − c_xyz xy + a_xz z − 2c_zz z + a_xyz yz (txt :3482–3484) | :830 `corr_C = no2*a_xx*tx - c_xz*tx + a_xy*ty - c_yz*ty - c_xyz*tx*ty + a_xz*tz - no2*c_zz*tz + a_xyz*ty*tz;` | **match** — all 8 terms incl. signs |

#### A.3.8 Wall offset / hat coefficients (Eqs. 7.49–7.57, §7.3)

Thesis mechanics: for a source cell incomplete at a wall, use the nearest
complete cell and store offset (x_off,y_off,z_off); re-expand the fitted
polynomial about the destination cell via the hat coefficients (7.49–7.50:
â₀, one formula over two tags; 7.51–7.53 â_x/â_y/â_z; 7.54 d̂₀; 7.55 d̂_x;
7.56/7.57 misprinted d̂_x = intended d̂_y/d̂_z; txt :3505–3543, PDF-verified).
The hat transform is the Taylor shift of the polynomial ⇒ evaluating the
source-cell polynomial at destination-frame positions x+offset.

Code mechanics: the **carve** pre-pass (:408–565): detect GEO_NOTHING cells in
the 8-candidate window (:457–461), shift the window one cell outward per axis
(:487–508 `carve_window_off_covered_cells`), and evaluate **off-center** at
the same relative destination position (`t_rel` moves with the window center;
|t_rel| = 0.75 for a one-cell shift, :504–505; comment :414–415 cites
"Schönherr 2015 thesis Eqs. 7.49-7.57"). Degenerate cases collapse to the
mirrored home cell with a rate-limited warning (:545–565). The demotion
docstring at :438–453 records the scope explicitly (see A.4-R2).

| Eq | Thesis (print) | Code site + quote | Verdict — justification |
|---|---|---|---|
| 7.49–7.50 | â₀ = a₀ + x_off a_x + y_off a_y + z_off a_z + x_off²a_xx + y_off²a_yy + z_off²a_zz + x_off y_off a_xy + x_off z_off a_xz + y_off z_off a_yz (p.62; txt :3511–3516) — ONE formula, two tags | :494–508 `start = taint_hi ? nodes[0]-1 : nodes[0]+1; … t_rel += static_cast<dreal>(nodes[0]-start); nodes[0]=start; nodes[1]=start+1;` | **conv** (implementation route) — code refits on the shifted window and evaluates at t_rel = ∓0.75 = the destination's source-frame position; mathematically the same polynomial evaluation as the hat shift for \|offset\| ≤ 1 per axis. Not implemented: multi-cell/arbitrary offsets (thesis: "the principle works with arbitrary offsets", txt :3507); code takes the degenerate home-cell collapse instead (:545–565) |
| 7.51 | â_x = a_x + 2x_off a_xx + y_off a_xy + z_off a_xz (txt :3518–3520) | as above | **conv** — folded into the same window-shift route (per-axis shift ⇒ derivatives transform identically to the hat rules for 1-cell offsets) |
| 7.52 | â_y = a_y + 2y_off a_yy + x_off a_xy + z_off a_yz (txt :3522–3524) | as above | **conv** — as 7.51 |
| 7.53 | â_z = a_z + 2z_off a_zz + x_off a_xz + y_off a_yz (txt :3526/3530) | as above | **conv** — as 7.51 |
| 7.54 | d̂₀ = d₀ + x_off d_x + **y_off** d_y + z_off d_z + x_off y_off d_xy + x_off z_off d_xz + y_off z_off d_yz (txt :3527; print drops the y-subscript — thesis typo, PDF-verified) | as above | **conv** — as 7.49–7.50 (density polynomial refit on shifted window; d-family has no quadratic terms so the shift is already exact at d̂₀). Print typo recorded for this appendix's errata list |
| 7.55 | d̂_x = d_x + y_off d_xy + z_off d_xz (txt :3528/3535) | as above | **conv** — as 7.51 |
| 7.56 | d̂_y = d_y + x_off d_xy + z_off d_yz (txt :3532/3536; LHS misprinted as d̂_x — thesis typo, PDF-verified) | as above | **conv** — as 7.51 |
| 7.57 | d̂_z = d_z + x_off d_xz + y_off d_yz (txt :3533/3537; LHS misprinted as d̂_x — thesis typo, PDF-verified) | as above | **conv** — as 7.51 |

Carve-vs-thesis scope note: thesis ties the offset to "non-fluid nodes that
are inappropriate as source nodes" (:3499 — wall nodes); the code gate is
`GEO_NOTHING` coverage (:461). Both fire only at covered/wall cells; the
conversion band map demotes carve to wall/degenerate-only
(contract §2.1), so this entire family is primarily T11-docstring material
(landed :438–453).

### A.4 Verdict summary and concern ranking

**Counts (57 equation tags = 56 formulas; 7.49+7.50 merged):**

- **match: 45** (7.1–7.9, 7.10–7.17, 7.19–7.22, 7.25–7.28, 7.29–7.33,
  7.34–7.37, 7.38–7.43, 7.44–7.48)
- **convention-difference: 12** (7.18, 7.23, 7.24 — corrected-family/print-
  erratum class; 7.49, 7.50, 7.51–7.57 — 9 hat-tags/8 formulas,
  implementation-route class)
- **bug: 0** (no code defect against the printed §7.2/§7.3 in the CM branch)

**Concern ranking (no bugs → ranked by decision consequence), as resolved at commits 10–11:**

- **R1 (RESOLVED):** Eqs. 7.18/7.23/7.24 (+ unprinted cyclic b/c analogs) —
  code ≠ print. The code implements the closed, nodal-consistent, cyclically
  complete family (§A.3.3-note proof); the printed set is internally
  inconsistent under every strain carrier ⇒ suspected thesis-print errata
  (same class as the verified 7.54/7.56/7.57 typos). **Commit 11 landed as
  the plan's verified-no-op + errata record arm**: no code change; the
  errata record documents the code's family in-place
  (`amr_coupling.h:676–701`); T10 encodes the code family (not the print) as
  the normative exactness reference (`test_amr_schonherr_exactness.cu`, case
  T10c discriminates the families — code family green, print-aligned would
  sit ~1e-5). A literal "align to printed §7.2" would degrade the scheme
  (would break the Tests-8/9 exactness class and inject O(strain·Δx²)
  errors at the destination cells). **T14 impact:** `a_0`/`b_0`/`c_0` carry
  the 7.18 correction into the F2C destination velocity — the same family
  choice propagates (§A.2.3 caveat); lock reused at commit 13.
- **R2 (RESOLVED):** Eqs. 7.49–7.57 — the carve implemented the 1-cell-shift
  case of the hat extrapolation by refit+off-center evaluation (verified
  equivalent for |offset| ≤ 1/axis, incl. multi-axis corner shifts);
  arbitrary offsets unsupported (degenerate collapse). After the carve
  demotion (valid faces never carve) this was wall-lane documentation, and
  on 2026-08-23 the pre-pass itself was HARD-REMOVED together with the
  `C2F_NO_CARVE`/`TNL_TEST_NO_CARVE` knobs and the mock carve tests
  (10–13/17): under the simulated-band map a covered window is an invalid
  registration, statically rejected by checkCouplingMapPattern at SimInit.
  This R-item stays as the audit trail of the removed code path.
- **R3 (noted, no action beyond the docstring clause):** (a) coupling macros
  use the force-free moment (`fx=0` fresh KS) — matches thesis §7.2 (no
  force); if volume forcing ever reaches the band, the coupling velocity
  excludes the Guo half-offset (thesis-consistent; the one-clause note now
  sits at the macro read, `amr_coupling.h:585–590`).
  (b) Thesis-print errata inventory recorded here: 7.54 missing
  y-subscript; 7.56/7.57 LHS d̂_x for d̂_y/d̂_z; 7.49/7.50 dual tags on one
  formula; (by verdict of this audit) the 7.18/7.23/7.24 family itself.

### A.5 Flagged uncertainties

- **U1 (investigated and closed at commit 11).** Musubi production-source
  agreement on the R1 family: **source read — cannot discriminate; the
  internal-consistency proof (§A.3.3-note) stands alone.** The shipping
  Musubi source (`apes-suite/musubi-source`, `main` @ `81f8c4f13772f6d4af31f335e1e3f99b02726e25`,
  read 2026-08-21; lineage of the prose-Qi's 2019 Musubi) implements **no**
  §7.2 coefficient family anywhere: the default C2F `quadratic`
  interpolation (`source/intp/mus_interpolate_quadratic_module.fpp`,
  `fillFinerGhostsFromMe_quad_feq_fneq`) is a generic least-squares
  quadratic fit over the octree-found source set
  (`mus_interpolate_quad3D_leastSq` with runtime-assembled
  `tem_intpMatrixLSF` matrices `((A^T)A)^{-1}A^T`, doc'd sizes (10,QQ) for
  D3Q19/D3Q27), applied to f_eq/f_neq separately, with the non-equilibrium
  part rescaled by `getNonEqFac_intp_coarse_to_fine(cω, fω)` — the
  Dazhi-Yu/FH-style τ-rescale, not the σ-form cumulant transfer of thesis
  ch.7. Exhaustive token search over `source/intp/` finds none of the
  §7.2 machinery (`k_xy`/`k_xx`/`saxx`/`cumul`/`avg_k`), and the config
  docs' advertised `'compact'` method name has **no implementation** in the
  current shipping source (no `compact` token anywhere in `source/`;
  `mus_interpolate_header_module.f90` :82–84 enumerates only
  `weighted_average`/`linear`/`quadratic`). Qi et al. 2019's own
  formulation (D3Q19, 4-source-element, 30-unknown LSQ) matches this same
  generic-LSQ machinery and likewise cannot discriminate. The R1
  singleton-vs-closed question never arises in any public Musubi code;
  literature absence of complaints also does not discriminate (the
  deviation sits inside the interpolation-error class). Outcome: U1 is
  closed as **"source read — no §7.2 implementation exists shipping-side;
  evidence stands on the nodal-consistency proof alone."**
- **U2.** Eqs. 7.5–7.9 bracket placement in the print (ω_s multiplying the
  whole bracket vs only the fraction) is typographically fragile in both
  the txt and the rendered PDF; the bracketed reading was adopted because
  any other makes k ≠ 0 at equilibrium. Code implements the bracketed
  reading. If a reviewer reads the glyph layout otherwise, rows 7.5–7.9
  would flip to "convention" — the physical content is unchanged.
- **U3.** Whether the 7.18-family deviation was a deliberate correction by
  the original implementer or an independent re-derivation of the closed
  family — git history of `amr_coupling.h` (pre-branch) could attribute
  it; not consulted in this lane (not needed for the verdict).
- **U4.** The K_a-route choice matters beyond incompressible/CE-consistent
  content: under compressible window content the code's a₀/a_xx (k-sum)
  and the print's (singleton) differ genuinely (even after carrier
  rescaling) — the code family's exactness is pinned to the
  divergence-free/CE-consistent class, which is the T10 lock class. Not a
  defect; a scope note for T10's test-field selection.

### A.6 Top-3 findings for commits 10–13

1. **The C2F CM branch matches the printed §7.2 at 54/56 formulas; the only
   code↔print deviations are Eqs. 7.18, 7.23, 7.24 (+ cyclic b/c analogs),
   and there the CODE is the consistent family.** Commit 11 landed as the
   plan's "(possibly verified-no-op + carve demotion docstring)" arm: no
   code change, the errata record is anchored in the code
   (`amr_coupling.h:676–701`) and recorded here; T10 encodes the code
   family (not the print) as the normative exactness reference. A literal
   "align to printed §7.2" would degrade the scheme.
2. **T14's minimal set is now exact and verified:** at t = (0,0,0) all five
   A/B/C aggregates vanish identically and the avk gradient terms cancel →
   F2C cumulants reduce to the five sk_* means; required machinery =
   {sd0, sa0, sb0, sc0, sk_* (5)} + per-source k-moments + cumulants +
   back-transform. **But** `sa0/sb0/sc0` retain the k-corrected sums, so
   the F2C destination velocity inherits the R1 (7.18) family choice —
   R1 was locked before commit 13 (T10c executable lock).
3. **FH τ-rescale site census: 3 sites, not 2** — F2C kernel :1427 (τc/τf),
   Lagrange C2F opt-out :1112 (τf/τc), **and** the `C2F_LINEAR_EXPLOSION`
   opt-in debug branch :329 (τf/τc). The σ-form is the default CM branch
   only. T14's "no τ-rescale" scope statement and T18 docs should name all
   three; the explosion branch needs no change (opt-in, default-off).
