# AMR Seam Checkerboard Investigation — Findings Record

**Dates:** 2026-09-03 → 2026-09-05 (+2026-09-19 post-refactor recheck, §2.6; +2026-09-21 operator-defect closure, §2.7)
**Branch:** feat/amr
**Status:** CLOSED on the canonical case — the artifact's two code-level authors are eliminated: the D1/D2 placement traffic by the streaming-slot refactor (6fbc6b7, §2.6) and the operator's antialiasing transcription defect by the col_cum.h fix (11cd893, §2.7). The pre-registered §6 Tier-1 bars pass with NO default flip (parity 1.84e-6 / seam 3.09e-5 on the corrected operator); §5's ranked candidates stand measured-not-adopted. Remaining open threads: the DFMAX == 1 parity-class mass pump at BC-adjacent interfaces (next campaign), the Tier-0 re-record ruling, and the unrun Tier-2/3 legs.

Normative companions: `docs/AMR-schonherr-ch7-target-contract.md` (cycle contract),
`docs/AMR-for-LBM-implementation.md` (internals; interface density bias paragraph),
`include/lbm3d/d3q27/amr_coupling.h` file docstring (per-arm dated probe records).

---

## 1. Artifact definition and measurement apparatus

**Artifact.** A decaying period-2 checkerboard in the velocity field, locked to the
fine-cell subcell parity at the coarse/fine seam of the static 2:1 AMR coupling
(Schönherr-ch7 simulated band, D3Q27 cumulant operator `USE_GEIER_CUM_2017`, AB
streaming). It is visually prominent at the seam band and decays over ~3 fine rows
into the interior.

**Canonical case.** `sim_AMR/sim_AMR_ball.cu` (ball in channel) at
`--resolution 2 --lattice-viscosity 0.001 --phys-final-time 1.0 --max-level 1`.

**Metrics.**

- *parity*: `|mean(vx : odd-fg) − mean(vx : even-fg)|` over the three interior rows
  1..3 of the x-max seam — i.e. the amplitude of a decaying `(−1)^n` ripple off the
  seam (rows {1,3} vs {2}).
- *seam mean|dvx|*: re-paired fine/coarse interface velocity defect
  (`tests/interface_seam_metric.py`, `--fine-row 0 --coarse-row 16`).
- Supporting: interior-only rho std (band rows excluded), development indicators
  (max|vx|, centerline min vx), double-run determinism.

**Verdict bars used throughout.** A genuine seed collapses parity **≥5×** (full
explanation **≥10×**) or drives seam mean|dvx| **≤ 4e-4** (Lagrange-arm class
2.6e-4). The 2026-09-03 baseline is parity 3.249e-3 / seam 1.526e-3; the 2026-09-05
round reproduced it exactly at 3.2487e-3 / 1.5256e-3.

---

## 2. Falsification history (measured rounds)

### 2.1 Round 1 (2026-09-03) — fill parity and ω3-persistence: FALSIFIED, signed

Hypotheses: the checkerboard is seeded by (a) the per-parity reconstruction windows
of the C2F compact-moment fill, or (b) an ω3-persistence amplifier — the seven
transferred Geier third-order cumulants forming a slow channel (~17-coarse-step
memory at the τ ≈ 0.503/0.506 level pair) that retains injected period-2 content
inside the band.

| arm | define | parity | seam mean\|dvx\| | verdict |
|---|---|---|---|---|
| baseline | — | 3.249e-3 | 1.526e-3 | — |
| 1a parity-averaged fill | `C2F_PARITY_AVERAGED` | 2.95e-3 (−9 %) | 1.29e-3 | FALSIFIED (~1.1× vs ≥10× bar) |
| 1b shared 27-node fit | `C2F_SHARED_FIT` | 3.04e-3 (−6 %) | 1.35e-3 | FALSIFIED (~1.07× null) |
| ω3 transfer discount g≈0.836 | `C2F_OMEGA3_DISCOUNT` | 3.486e-3 (+7.3 %) | 1.610e-3 (+5.5 %) | FALSIFIED (wrong direction) |
| ω3 discount, memory ratio g≈0.258 | `C2F_OMEGA3_DISCOUNT_MEMORY` | 4.370e-3 (+34.5 %) | 1.892e-3 (+24 %) | FALSIFIED (dose-response grows) |
| ω3 inverse discount g≈1.197 | `C2F_OMEGA3_DISCOUNT_INVERSE` | 2.975e-3 (−8.4 %) | 1.423e-3 (−6.7 %) | sign probe: more retained content mildly SUPPRESSES |
| band-local ω3 off (κ=1) | `AMR_BAND_OMEGA3=1` | 6.914e-3 (+112.8 %) | 2.915e-3 (+91 %) | FALSIFIED (amplifies up to ~2×) |
| band-local ω3 damped (κ=0.5) | `AMR_BAND_OMEGA3=0.5` | 4.959e-3 (+52.6 %) | — | dose-consistent |

Candidate 1b's shared home-centered 27-node tensor-quadratic fit is the
single-interpolation-cell instance of the Kutscher-Geier-Krafczyk 2018 compact
quadratic interpolation (their Eqs. 79–91, `44_...:1195–1317`) — i.e. round 1
already tested the literature's own production reconstruction basis and found it
inert for the artifact (see §4.2).

**Round-1 synthesis (signed).** Monotone negative dose-response across all
retention-scaling arms (over-retained −8 %, identity baseline, discounted +7 %,
strongly discounted +35 %, band-half-relaxed +53 %, band-off +113 %): the
seven-cumulant third-order channel is a partial **SUPPRESSOR** of the checkerboard,
not its amplifier. The two parity-removing reconstructions being inert shows ~85 %
of the seam bias is parity-INDEPENDENT at the fill. Surviving suspects handed to
round 2: the cycle timing of the substep-1 widened extent (T1) and the C2F/F2C
frame timing (T2).

### 2.2 Round 2 (2026-09-05) — timing suspects: T1 CONFIRMED, T2 falsified

All arms on the canonical case (baseline re-measured this session). Implementation:
frame-displacement arms as `#ifdef` branches in `amr_coupling.h` read/store lambdas;
cycle arms in `amr_state.h` (`advancePair` substep windows, mid-sync, pre-substep-2
refill, SimInit, cascade) with new cache options in `sim_AMR/CMakeLists.txt`.

| arm | define | parity | seam mean\|dvx\| | verdict |
|---|---|---|---|---|
| baseline | — | 3.2487e-3 | 1.5256e-3 | — |
| T2a C2F reads pre-step frame | `AMR_C2F_READ_DF_CUR` | 3.2502e-3 (+0.05 %) | 1.5258e-3 | FALSIFIED (metric-inert) |
| T2b F2C reads pair-mid frame | `AMR_F2C_READ_DF_CUR` | 2.3423e-3 (−28 %) | 5.38e-4 (−65 %) | FALSIFIED as seed — pair-end freshness is a CARRIER, not the seed |
| T2c C2F writes non-consumed frame | `AMR_C2F_WRITE_DF_OUT` | 4.5139e-3 (+39 %) | 6.317e-3 (+314 %) | FALSIFIED (development destroyed; opposite direction) |
| T1a passive band (witness) | `AMR_PASSIVE_BAND` | 9.85e-6 (−330×) | (stale band-macro artifact) | factorization witness — collapse confirms the SITE |
| **T1b substep-2 C2F refill** | `AMR_SUBSTEP2_C2F_REFILL` | **9.79e-6 (−332×)** | **2.70e-4 (−82.3 %)** | **CONFIRMED seed site + production-fix arm — both bars crossed** |
| T1a+T1b | both | 9.79e-6 | (T1a-class) | witness |

Interior-only rho std (real physics, band rows excluded): baseline 2.41e-4,
T1a 2.707e-4, T1b 2.710e-4, T2b 2.567e-4. The apparent seam/std explosion of T1a in
the canonical probe is a metric artifact of never-recomputed band rows, not a
blowup; the widened substep-1 launch itself is **load-bearing** for the seam's
transport (it carries the interior one fine step into the coarse domain, which is
what replaces time interpolation in the Schönherr cycle).

**T2 static trace (definitive).** AB streaming, DFMAX=2 (`defs.h:54-60`; rotation
in `lbm.hpp:469-505`): substep 1 (`amr_state.h:2357-2359`, even rotation, widened)
integrates the inner ghost row into df_out; substep 2 (`:2393-2395`) consumes it;
F2C (`amr_coupling.h:2786-2788`) reads fine df_out = post-substep-2 — time-aligned;
C2F (`:1331-1335`) reads coarse df_out = post-step, writes fine df_cur
(`:1362-1366`) = exactly the frame the next cycle's substep 1 streams from. No
off-by-one anywhere — hence all three displacement arms are inert or harmful.

### 2.3 Round 2b (2026-09-05) — operator-parametrization suspect: rates-to-one AMPLIFIES

Hypothesis: with ω3/4/5 = 1 GLOBALLY (the Kutscher-2018/Schönherr-school production
rates family `44_...:1185–1187`, whose figures report no seam artifact), the slow
third-order kinetic channel (~17-coarse-step memory at base ω3 ≈ 0.06 for
ω1 ≈ 1.988, τ = 0.503) no longer retains or transports the parity-locked (−1)ⁿ
mode the band collision injects — fork (a): parity collapses WITHOUT any refill
(the rate parametrization is a necessary author, the school's clean record
genuine); fork (b): parity persists at ~3e-3 (their production figures were
simply blind to it, e.g. turbulence masking). The GLOBAL arm is the opposite
regime of the round-1 BAND-LOCAL `AMR_BAND_OMEGA3` κ-arms (which amplified up to
+112.8 %); the two defines are compile-time mutually exclusive (`#error` in
`col_cum.h`). Implementation: `CUM_OMEGA345_ONE` pins omega3/omega4/omega5 and
the seven Sec.-7 limiter-adapted rates to `no1` in `col_cum.h` (ω2 and ω6..ω10
are already 1; the limiter is an exact no-op at base rate 1; lambda3/4/5 become
dead constants; arm exists only under `USE_GEIER_CUM_2017` — sim_AMR_ball is the
only AMR target in that mode; define state of the arm verbatim:
`#define USE_GEIER_CUM_2017`, `#define USE_GEIER_CUM_ANTIALIAS` at
`sim_AMR_ball.cu:1–2`, so the arm is exactly the K2018 production regime —
arm Y of the task plan is folded into arm X).

| arm | define | parity | seam mean\|dvx\| | verdict |
|---|---|---|---|---|
| baseline (reused, same session) | — | 3.2487e-3 | 1.5256e-3 | — |
| **2b rates-to-one GLOBAL** | `CUM_OMEGA345_ONE` | **8.2085e-3 (+152.7 %)** | **3.5778e-3 (+134.5 %)** | **fork (a) REJECTED — AMPLIFIED, largest worsening of the campaign; fork (b) strengthened** |

Supporting measurements (same probe, same session): seam max 5.5476e-3 (+134.1 %
vs 2.3697e-3); parity means odd/even 5.687e-3 / 1.3896e-2 (baseline ≈1.26e-2 /
1.58e-2 class); mid-run frame 0010 parity 8.2823e-3 / seam 3.5709e-3 — the
amplification is systematic through the run, not a final-frame fluke; interior
rho std (band rows excluded) 3.0475e-4 (+26.4 % vs 2.4115e-4 — just over the
tier-1 +25 % watch bar); rho std incl. band rows 3.2470e-4 (−36.0 % vs
5.0753e-4 — the rates-to-one regime smooths the fine box's density field);
development intact but costlier: max|vx| 2.6810e-2 (+4.2 % vs 2.5719e-2),
centerline min vx −1.8803e-3 (recirculation ×1.89 vs −9.963e-4), level-1
kinetic energy 1.0597e+1 (−18 % vs 1.2942e+1). The physics cost mirrors the
school's own rates-to-one control failing the drag crisis (Part II Fig. 2
asterisk): the regime is more dissipative where it matters and visibly worse at
the seam. Arm definition for reproduction: `cmake -B build -S . -DCUM_OMEGA345_ONE=ON`,
`cmake --build build --target sim_AMR_ball`, run
`sim_AMR_ball --resolution 2 --lattice-viscosity 0.001 --phys-final-time 1.0 --max-level 1 --adios-config adios2.xml`,
probe `/tmp/opencode/checkerboard_sweep/checkerboard_probe.py
results_sim_AMR_ball_res002_np001/output_amr_0020.vtkhdf` (+ the interior-row
rho filter); run dir `/tmp/opencode/band_probe/run_ratesone/`.

**Round-2b synthesis (signed).** Neither fork fired as predicted: parity did not
collapse (fork (a) rejected — ×2.53 growth against the ≥5×-collapse bar, far
beyond any noise; the baseline of the same session reproduced bit-for-bit) and
did not persist at ~3e-3 either (plain fork (b)) — it AMPLIFIED +152.7%, the
largest worsening of the campaign. The response is dose-consistent with the
round-1 retention family on a third, independent handle: over-retention −8.4 %,
identity baseline 0, transfer discount +7.3 %, memory discount +34.5 %, band
κ=0.5 +52.6 %, band κ=1 +112.8 %, GLOBAL rates-to-one +152.7 % — monotone in
retainedness, opposite sign to the amplifier hypothesis, through
transfer-side, band-local, and now global-parametrization probes. The school's
clean figure record is therefore NOT attributable to their ω2..ω10 = 1 rates:
in their own rate family the artifact amplifies on our case, so the fork-(b)
reading (their production figures were blind to the parity-class artifact —
turbulence masking, figure scales) stands, strengthened. The round-1 signed
conclusion extends unchanged: the seven-cumulant third-order channel is a
partial SUPPRESSOR of the checkerboard.

### 2.4 Round 3 (2026-09-05) — operator × streaming-pattern matrix: antialiasing is the carrier under AB; AA carries its own interface defect

Hypothesis (user-directed): grid refinement exchanges distribution functions
and should be independent of the streaming pattern that produces them. Measured
as the full 2×2×2 matrix — Geier-2017 A/B terms × antialiasing derivative
feedback × {A-B, A-A} — on the canonical case. The `USE_GEIER_CUM_2017`-without-
`USE_GEIER_CUM_ANTIALIAS` combination was not compilable before this round: the
2017 Eqs 46–48 A/B terms read the cross-derivatives `DxvDyu/DxwDzu/DywDzv` that
only the ANTIALIAS block declares (`col_cum.h`); the arm was completed by
zeroing them alongside `Dxu/Dyv/Dzw` in the `#else` (uniform removal of all
derivative feedback; temporary, source reverted after measurement — the tree is
clean at HEAD).

| A-B combo (default build) | parity 0020 / 0010 | seam mean | seam max | rho std | recirc min vx |
|---|---|---|---|---|---|
| 2017 + ANTIALIAS (committed default) | 3.2487e-3 / 3.3542e-3 | 1.5256e-3 | 2.3697e-3 | 5.0753e-4 | −9.96e-4 |
| 2017, no ANTIALIAS | **1.8126e-6 / 1.0467e-6** | 3.0887e-5 | 2.3971e-4 | 5.5266e-5 | −2.34e-3 |
| no 2017, ANTIALIAS | 1.2048e-4 / 1.2090e-4 | 4.9702e-5 | 3.8918e-4 | 9.4674e-5 | −2.70e-3 |
| neither | **2.6078e-6 / 1.2817e-6** | 4.2416e-5 | 3.6056e-4 | 6.2119e-5 | −2.72e-3 |

| A-A combo (`build-aa`, `-DTNL_LBM_AA_PATTERN=ON`) | parity 0020 / 0010 | seam mean | seam max | rho std | recirc min vx |
|---|---|---|---|---|---|
| 2017 + ANTIALIAS | 3.8598e-3 / 4.1819e-3 | 2.3073e-3 | 4.2259e-3 | 8.6530e-4 | −2.52e-3 |
| 2017, no ANTIALIAS | 8.9022e-4 / 7.0764e-4 | 1.3169e-3 | 2.7155e-3 | 6.0184e-4 | −2.32e-3 |
| no 2017, ANTIALIAS | 1.4297e-3 / 6.9861e-4 | 1.8188e-3 | 3.2105e-3 | 6.2141e-4 | −2.09e-3 |
| neither | 1.4967e-3 / 8.0410e-4 | 1.8314e-3 | 3.2267e-3 | 6.1658e-4 | −2.06e-3 |

Factor decomposition (final-frame parity):

| factor | A-B | A-A |
|---|---|---|
| drop ANTIALIAS, 2017 ON | **÷1793** | ÷4.3 |
| drop ANTIALIAS, 2017 OFF | ÷46 | ×1.05 (noise) |
| drop 2017, ANTIALIAS ON | ÷27 | ÷2.7 |
| drop 2017, ANTIALIAS OFF | ×1.4 (floor vs floor) | ×1.7 (floor vs floor) |

**Findings (signed).**

1. Under A-B the artifact requires ANTIALIAS to exist at all — both no-ANTIALIAS
   combos sit at the ~2e-6 floor at both probe frames. Full attribution:
   seed (T1, ~2e-6) × Eqs 33–35 second-order feedback (~46×, → 1.2e-4) ×
   2017 A/B cross-derivative channel (Eqs 46–48, ~27×, → 3.2e-3).
2. Under A-A nothing collapses: all four combos sit ≥ 7e-4 with an
   operator-insensitive floor (~1e-3) — the A-A pattern carries its own,
   operator-roughly-independent parity mechanism at the interface. The ÷46
   Eqs 33–35 contribution is invisible there (masked by the floor), which is
   exactly why the A-A no-2017 combos are indistinguishable; only the A/B
   channel (the one contribution reaching above the floor) still shows
   (÷4.3 / ÷2.7).
3. The A-A runs' wake is uniformly ~40 % more energetic (max|vx| 0.0363–0.0369
   vs 0.0257–0.0258 A-B, all combos) — pattern-driven, in line with the
   documented wake-amplified A-A/A-B divergence (AGENTS); within-pattern
   comparisons are unaffected.
4. §4.3 falsifier (1) is settled: the artifact persists under A-A in every
   operator combo — fork (b) strengthened — and the A-A floor exposes a NEW
   defect class: the interface is NOT streaming-pattern independent.

**Round-3 synthesis (signed).** The premise that refinement is
streaming-pattern independent is violated in both directions: A-A is clearly
broken at the interface (operator-insensitive ~1e-3 parity floor), and A-B is
affected too in the sense that its artifact lives entirely in the operator's
antialiasing response to what the interface feeds it (no-ANTIALIAS A-B is clean
at 2e-6 with the same seed present). Open audit question: which interface data
dependency authors the A-A floor — candidate classes are the frame/slot identity
of the C2F/F2C traffic (the A-A twisted even/odd phases vs the A-B ping-pong
against the odd 3-step AMR cycle: 2 fine substeps + 1 coarse step flips array
roles every cycle), and the substep-1 widened extent [−1, local+1) streaming
through the unclamped single-array indexing. Physics note: ANTIALIAS-off is a
diagnostic, not a fix — the derivative feedback is the Galilean-invariance
correction (Geier 2015; K2018 runs it in production), and the no-2017 runs
recirculate ×2.7 deeper (−2.7e-3 vs −1.0e-3).

Reproduction: source reverted after measurement (tree clean at HEAD); the two
defines are `sim_AMR_ball.cu:1–2` (comment per combo; the 2017-without-ANTIALIAS
combo additionally needs the `col_cum.h` `#else` cross-derivative zeroing), the
A-B build is `build`, the A-A build is `build-aa`
(`cmake -B build-aa -S . -G Ninja -DTNL_LBM_AA_PATTERN=ON`); run dirs
`/tmp/opencode/band_probe/{run_baseline,run_noantialias,ab_no2017_anti,ab_no2017_noanti}`
(A-B) and `/tmp/opencode/band_probe/aa_{2017_anti,2017_noanti,no2017_anti,no2017_noanti}`
(A-A); probe frames 0010+0020 with
`/tmp/opencode/checkerboard_sweep/checkerboard_probe.py`. /tmp hygiene: only
frames 0010/0020 and `sim.log` are retained per run dir (tmpfs — clear big
artifacts often to avoid OOM).

### 2.5 Round 3b (2026-09-05) — interface data-dependency audit: ROOT CAUSE of the A-A floor (deep agent, code-verified)

A dedicated line-by-line audit of every interface df read/write under both
patterns (all file:line claims re-verified on the tree after the audit) plus
in-situ measurements on the canonical case. Verdict: the **fine-side** band
machinery (C2F fill → substep A/B consumption → F2C read) is pattern-independent
by deliberate parity arming and cycle-invariant under both patterns — but the
**coarse-side feedback path is not**, and that is the A-A floor's author.

**The structural asymmetry.** The fine level runs exactly 2 substeps per cycle
(`totalSubstepCount` +2, `amr_state.h:2413,:2471`) → Φ_S (spatial substep A)
then Φ_T (reflect substep B) on *every* cycle — cycle-invariant. The coarse
level runs 1 step per cycle (`amr_state.h:2589`, `lbm.hpp:452-454`) → its A-A
phase **alternates every cycle** (odd iterations Φ_S, even Φ_T).

**D1 — A-A frozen-adjacency of the collision-active ring (parity author).**
The coarse ring row c=0 (`GEO_AMR_INTERFACE`, collision-active, `bc.h:563`)
streams from its frozen neighbor, the c=1 skin (`GEO_NOTHING` never streams,
`bc.h:221-227,:574-575`). Under A-A, foreign content enters a cell's slots only
via another cell's push or the cell's own pull: on Φ_S steps the ring's pull
correctly ingests the F2C-armed skin slots; on Φ_T steps the reflect phase reads
*only same-site slots* (`streaming_AA.h:97-100`), and the ring's footprint-facing
slots were last written by its own previous Φ_T twist-store — its own
post-collision populations in mirrored orientation, one coarse-step stale. Every
other coarse step the interface is effectively closed (bounce-back-like), and the
F2C natural-slot write for those cycles (`amr_coupling.h:3133`, armed via
`next_coarse_even_iter`, `amr_state.h:2287-2288`) is dead traffic.

**D2 — parity-armed C2F reads of the contaminated ring (seam author).** C2F
reads ring rows with the producing step's convention (`amr_coupling.h:1509-1515`):
post-Φ_S cycles the ring's footprint-facing slots are *not* what the Φ_S
collision ingested (they hold the stale mirrored Φ_T product); post-Φ_T cycles
the returned product itself collided on the D1 reflection. The fill therefore
deposits the reflection on the fine band rows, spatially locked to subcell
parity through the window shift (`axis_window`).

**Measured evidence (canonical case, both patterns).** (1) Baselines reproduce
the §2.4 matrix bit-for-bit. (2) In-situ period-2 signature (160 consecutive
coarse-step frames, `--out3d-iter-period 1`, no source edits): A-A ring rows
carry a period-2 vx square wave — per-parity mean offsets +5.73e-3 (c=0 row,
x-max), +7.50e-3 (x-min), −4.81e-3 (halo), flip fraction 0.95-0.98 — while
upstream reference/interior planes are exactly clean and all A-B planes sit at
the 1e-4 class with no period-2; the A-A interface also blocks the flow
(downstream halo mean vx 1.06e-3 vs 1.71e-2 under A-B). (3) Discriminating arm
(define-gated `AMR_AA_F2C_NEIGHBOR_REFRESH` probe, since reverted): poke each
skin cell's F2C DF f_q into the frozen-facing slot q of each live neighbor with
the same phase convention — full arm: parity ÷5.7 to 6.73e-4 and the square wave
collapses (halo vx recovers to 1.77e-2 ≈ A-B); TAU-only (fixes D1 ingestion):
parity ÷3.9 → D1 dominates the parity checkerboard; SPATIAL-only (freshens D2
fill reads): seam ÷3.4 to 6.73e-4 → D2 dominates the smooth seam bias. (4)
Full-arm double-run deterministic to ~1e-4 relative (the probe itself adds a
small freshness race — see T8 below). Refuted: unclamped-index wrap of the
widened extent (statically impossible — non-periodic, overlap 2) and A-B role
inversion across the odd 3-step cycle (role-named arrays + absolute per-substep
rotation).

**Does the same class seed the A-B antialias channel? No.** The A-B artifact is
the fine-side T1 band collision amplified by the operator (established in §2.4:
removing ANTIALIAS collapses A-B to 1.8e-6 with the same T1 seed present); the
D1/D2 class is structurally impossible under A-B's pull scheme, which reads the
armed frame every step. Under A-A the D1/D2 alternation additionally modulates
the C2F fill feeding substep 1's mixed shell, so the channels add there
(A-A 2017+anti 3.86e-3 ≈ A-B 3.25e-3 + the ~1e-3 pattern floor).

**Fix designs (not yet implemented).** D1 (priority 1): refresh the frozen-facing
slots of every live cell adjacent to a `GEO_NOTHING` cell once per cycle with
the phase-correct convention before the next coarse step, sequenced on one
stream before the C2F reads (kills the probe's added race); confirmation bar:
canonical A-A parity ≤ ~1e-3 at both frames AND ring square-wave offsets ≤1e-4
class. D2 (priority 2): sequence the D1 refresh before the C2F launches with a
barrier (or parity-gate the C2F coarse-source read); bar: A-A seam_mean ≤ ~1e-3.
Caveat for the production flip: the full arm's seam (2.06e-3) does not beat the
SPATIAL-only arm's (6.7e-4) — the TAU pokes change the ring state feeding F2C;
the ordered variant must be evaluated on the Tier-1/2 matrix first. Latent
hazard noted for both patterns (severe only under A-A): the F2C-writes-skin ∥
C2F-reads-skin stream overlap is a genuine read/write race whose loser value is
convention-mismatched under A-A; at HEAD the winner is deterministic-fresh
(floor reproduces bit-for-bit). Nested levels carry the same class on nested
rings (mid-sync fills read post-Φ_S parent state — same analysis per pair;
flagged, not measured). The A-A defect is orthogonal to the T1b refill question
and does not change §5's A-B fix ranking.

### 2.6 Round 4 (2026-09-19) — recheck after the streaming-slot refactor (commit 6fbc6b7)

Commit 6fbc6b7 (`refactor(amr): route coupling DF access through streaming
slots`) moved the coupling's pre/post-collision DF placement knowledge out of
`amr_coupling.h` into the streaming policy classes (pattern-owned
`preCollisionSlot`/`postCollisionSlot`; reads at `amr_coupling.h:1571,:3119`,
stores at `:1640-1642,:3304-3305`). Three placement classes are candidates of
interest for this investigation: AB_PULL is byte-identical by construction (the
pattern-owned calls reduce to the old `(q, x)` addressing, storage-affordability
store kept unconditional); AB_PUSH reads/stores now hit the downwind `+c_q`
slots (heals the sim_AMR_ball mass-freeze under push); under A-A the odd-phase
reconstruction reads the true post-collision slot (was the arrival layout) and
the even-next stores reach the consumer address (was the dead twist-at-ghost) —
exactly the D1/D2 traffic §2.5 root-caused as the A-A floor's author. Hypothesis
under test: the operator-insensitive ~1e-3 A-A parity floor moves if the
corrected placements were its carrier; the push arms should now reproduce the
pull arms.

**Probe calibration (signed, prerequisite).** The doc-era probe
(`/tmp/opencode/checkerboard_sweep/checkerboard_probe.py`) was deleted; the
Round-4 probe (`/tmp/opencode/checker_recheck/checkerboard_probe.py`)
re-implements the metrics from §1 + the probe-record anchors in
`amr_coupling.h :358-368` and the 2026-09-03 session's metric delegation. On the
morning-of-2026-09-19 canonical-case runs (`results_sim_AMR_ball_2x2/`, the
same arg set, exit 0) the calibrated conventions reproduce the §2.4 A-B
baselines **exactly to all recorded digits**: parity rows fg {88,89,90} with the
int1 tangent window (geier 3.2486973e-3 / 3.3541826e-3 @0020/@0010 vs recorded
3.2487e-3 / 3.3542e-3; neither 2.6077870e-6 / 1.2817192e-6 vs 2.6078e-6 /
1.2817e-6); seam defect = downsampled last fine coarse-column block (fg {90,91},
2×2×2 uniform mean) vs the seam-plane midpoint interpolation (c45+c46)/2
(1.525613e-3 / 2.369747e-3 vs 1.5256e-3 / 2.3697e-3; neither 4.241638e-5 /
3.605613e-04 vs 4.2416e-5 / 3.6056e-4); rho std over map==0 fluid cells
(5.075263e-4 vs 5.0753e-4; 6.2119e-5 both); max|vx| over the Level0 domain
(2.5719946e-2 vs the §2.3-recorded 2.5719e-2). **Convention gap (recorded
honestly):** the recirc line's exact doc-era row choice was not recovered — the
probe takes the fine centerline row nearest below the physical midplane (fg
y=z=62); measured −9.5262e-4 vs recorded −9.963e-4 (−4.4 %) and −2.6458e-3 vs
−2.72e-3 (−2.7 %) on the same byte-identical fields that reproduce every other
recorded cell exactly. All cross-round recirc comparisons carry that ≤ ~5 %
systematic offset; all within-Round-4 comparisons use one convention.

**A-B recheck (from disk, exit 0; probe rows above).**

| A-B combo | parity 0020 / 0010 (§2.4) | Round-4 | seam mean | seam max | rho std | recirc min vx |
|---|---|---|---|---|---|---|
| 2017+ANTIALIAS, **pull** | 3.2487e-3 / 3.3542e-3 | 3.2486973e-3 / 3.3541826e-3 (=) | 1.5256126e-3 (= 1.5256e-3) | 2.3697468e-3 (= 2.3697e-3) | 5.0752633e-4 (= 5.0753e-4) | −9.5262e-4 (conv) |
| neither, **pull** | 2.6078e-6 / 1.2817e-6 | 2.6077870e-6 / 1.2817192e-6 (=) | 4.2416381e-5 (= 4.2416e-5) | 3.6056130e-4 (= 3.6056e-4) | 6.2119318e-5 (= 6.2119e-5) | −2.6458e-3 (conv) |
| 2017+ANTIALIAS, **push** | — | 3.2486959e-3 / 3.3541846e-3 | 1.5256112e-3 | 2.3697838e-3 | 5.0752405e-4 | −9.5259e-4 |
| neither, **push** | — | 2.6099135e-6 / 1.2758509e-6 | 4.2419505e-5 | 3.6056968e-4 | 6.2116945e-5 | −2.6458e-3 |

**A-A recheck (fresh runs this session, exit 0, `build-aa`, canonical case).**

| A-A combo | parity 0020 / 0010 (§2.4) | Round-4 parity 0020 / 0010 | seam mean @0020 | seam max @0020 | rho std 0020 / 0010 | recirc min vx 0020 / 0010 |
|---|---|---|---|---|---|---|
| 2017+ANTIALIAS | 3.8598e-3 / 4.1819e-3 | **2.4785232e-3 (÷1.56) / 3.2657809e-3 (÷1.28)** | 1.0241138e-3 (÷2.25) | 1.7072503e-3 (÷2.47) | 5.4651e-4 / 5.1775e-4 (÷1.58 / ÷1.67) | −6.10e-4 / −2.77e-4 |
| neither | 1.4967e-3 / 8.0410e-4 | **2.1445663e-6 (÷698) / 1.3126526e-6 (÷613)** | 4.0149144e-5 (÷45.6) | 2.8695143e-4 (÷11.2) | 1.3549e-4 / 6.1719e-5 (÷4.6 / ÷10) | −2.6119e-3 / −1.7533e-3 |

A-A run caveat (signed): both A-A arms carry the known pre-existing
mass-injection onset at the run's end (mass print frozen 7.13–7.18e+05 through
t ≈ 0.99, then a single jump to 7.29–7.34e+05 with a × 8 KE print jump at
t = 1.000 — the pathology 6fbc6b7's message lists as pre-existing and
not-healed). Frame 0010 rows are pre-onset clean (neither-arm rho std
6.1719e-5, the A-B base 6.2e-5 class); the collapse signature (÷698 / ÷45.6)
dwarfs the onset perturbation, and the 0020 rows are quoted nonetheless for the
§2.4-comparable cadence.

**Verdict (signed).**

1. **A-B side: reproduced (pull) and healed (push), both bars signed.** The
   AB_PULL rows reproduce every calibrated §2.4 cell exactly — the refactor's
   byte-identical-by-construction claim is measured true on the canonical case.
   The AB_PUSH rows match AB_PULL to floating-point noise on the 2017+ANTIALIAS
   arm (parity rel. diff −4.3e-7 @0020, seam −9.2e-7, max|vx| +2.5e-6 — plus
   conservation/KE series identical at print precision) and to ~8e-4 relative
   on the neither arm (+0.082 % parity, +0.007 % seam) — the push mass-freeze is
   healed by the downwind `+c_q` placement, exactly per the commit's
   expectation.
2. **The A-A operator-insensitive ~1e-3 parity floor COLLAPSED — the corrected
   placements were its carrier.** Under the neither combo (no antialiasing
   channel), Round-4 A-A parity sits at 2.14e-6 / 1.31e-6 (@0020/@0010) — ÷698
   / ÷613 below §2.4's 1.4967e-3 / 8.0410e-4, i.e. AT the T1 seed level
   (A-B neither: 2.61e-6 / 1.28e-6). The §1 genuine-seed bars are crossed by
   orders of magnitude (parity ÷ ≥ 5–10×; neither seam mean|dvx| 4.0e-5 ≤ 4e-4,
   and the §2.5 fix bars: A-A parity ≤ ~1e-3 ✓ at both frames, A-A
   seam_mean ≤ ~1e-3 ✓). The A-A floor is gone with precisely the two
   correction classes that §2.5's root-cause audit named — the odd-writer
   reconstruction now reads the true post-collision slot and the even-next
   stores reach the consumer address (the D1/D2 traffic), which **confirms
   §2.5's root cause by its own pre-registered confirmation bar**. The
   §2.4-finding-3 signature (A-A wake ~40 % more energetic than A-B) also
   disappears (A-A max|vx| 2.5699e-2 / 2.5679e-2 vs A-B 2.5720e-2 / 2.5723e-2 —
   the +40 % was the D1 blockage artifact, not pattern physics).
3. **What survives is the T1-found operator channel, pattern-symmetric.** The
   A-A 2017+ANTIALIAS arm retains parity 2.48e-3 / 3.27e-3 and seam 1.0e-3 — the
   antialias channel amplifying the same T1 band seed as under A-B (amplifier
   factor vs the same-arm neither seed: 1156 @0020 / 2488 @0010 vs A-B's
   1246 / 2617 — the same operator response within pattern jitter). §2.4's
   signed conclusion stands, restricted to its carrier: the A-B artifact lives
   in the operator's antialiasing response to what the interface feeds it; the
   interface is NOW streaming-pattern independent (neither-class floors equal
   at the 2e-6 seed level, geier-class parities equal at the 2.5–3.4e-3
   operator-amplified level). §4.3 fork (b) stands strengthened with the
   pattern class closed.
4. §2.5's D1/D2 fix designs are superseded by the shipped correction; the
   flagged nested-level mid-sync class (same analysis per pair) is expected
   healed by the same slot ownership but is NOT measured here (single
   canonical case, max-level 1).

Reproduction: A-B rows probed on disk at
`results_sim_AMR_ball_2x2/{ab_pull_base,ab_push_base,ab_pull_geier,ab_push_geier}/results_sim_AMR_ball_res002_np001/`
(built from `build-ab` = AB_PULL, `build` = AB_PUSH); A-A rows run this session
at `build-aa` with the sim's macro lines toggled at `sim_AMR_ball.cu:1-2`
(committed bytes = 2017+ANTIALIAS ON; commented bytes = neither), binaries via
`cmake --build build-aa --target sim_AMR_ball`, run from a fresh CWD as
`build-aa/sim_AMR/sim_AMR_ball --resolution 2 --lattice-viscosity 0.001
--phys-final-time 1.0 --max-level 1 --adios-config <repo>/adios2.xml` (stdouts
kept at `/tmp/opencode/checker_recheck/runs/aa_{neither,geier}/stdout.txt`,
per-frame metrics JSONL at `/tmp/opencode/checker_recheck/metrics.jsonl`, run
dirs deleted for tmpfs hygiene); probe
`python3 /tmp/opencode/checker_recheck/checkerboard_probe.py <results_dir>
--frames 10,20`, conventions in the probe docstring (calibrated rows above);
working tree restored byte-exact (md5, `git status --short` shows only the
user's own `sim_AMR_ball.cu` macro comments), `build-aa` sim_AMR_ball relinked
against the restored source.


### 2.7 Round 5 (2026-09-21) — the operator root cause: a transcription defect in the antialiasing derivative estimator, eliminated at col_cum.h (`11cd893`)

§2.6 verdict 3 left the T1-found operator antialias channel as the surviving
carrier (~1250× pattern-symmetric amplification). A verify-only literature
audit of the antialiasing terms was commissioned before designing any flip;
its outcome relocates the carrier entirely and nulls the §5 candidate list on
the canonical case.

**The defect (100 % confidence, JCP-closed).** The `Dxu` estimator at
`col_cum.h:341` subtracted `(rho − 1)` from `(C200+C020+C002)` instead of the
full density. JCP 348 Part I Eq (27) prints `(C200+C020+C002 − κ000)`
(`docs/references/CuLBM/02_...:441–446,467,471`), with the paper defining its
variables as well-conditioned distributions **f̂ = f − w** and
`ρ = δρ + 1, δρ = m000` (`02_...:218–223`) — i.e. κ000 = δρ = ρ−1 *in the
paper's variables*. The faithful port into this operator's raw variables
subtracts the full density (`−rho`), which is the exact expression that has
sat **commented out one line above** since the code's early history
(`col_cum.h:338`). As coded, the bracket evaluates to **+1 at local
equilibrium** instead of 0: every near-equilibrated cell carries a persistent
`Dxu ≈ −ω2/(2ρ) ≈ −0.5` bias, injected per collision through the very
Eqs 33–35 channel this investigation measured as the artifact's only carrier.
Quantified (canonical-case τ): spurious deviatoric forcing
≈ 2.3–5.9e-6 *per step* — the T1-seed scale, but persistent and coherent —
plus a ~8e-4 trace inflation and an O(1e-3)-class leak into the 2017 Eq-45
A-term; the intended feedback signal (~1e-9–1e-8) is buried ~10³–10⁴ below
it. **Every "ANTIALIAS vs none" arm in §2.3/§2.4 therefore measured
*defective feedback vs none*, not paper feedback vs none.**
`col_cum_well.h:284` was initially flagged under the same verdict and
**flipped on print evidence**: its κ000 = δρ *and* its cumulants are
conditioning-shifted, so its line is verbatim Eq (27) in the paper's own
variables — never defective, and thereby the natural always-correct control
operator used below. Everything else audited is faithful verbatim (JCP
anchors): A/B cross-derivative terms + recombination, A/B constants,
ω3/ω4/ω5 parametrizations, limiter Eq (116), transforms; the AMR coupling's
compact-moment conversion chain is thesis-exact — no coupling-side mirror
defect class exists. Provenance: deviation introduced 2019-01-18 (misread
δρ as "rho^(2) = 1−rho"), sign-flipped 2021-03-18 (`37de8de`), never
paper-matched. **Fix commit `11cd893`** (restore the full-density bracket,
drop the ρ⁽²⁾ remark); collateral commit `bed30e3` (the 2017-without-
ANTIALIAS `#else` compile landmine; citation anchors; the mislabeled
`docs/references/AMR/23_Geier_2017_*` corpus renamed to `*_2015_*`).

**Acceptance on the canonical case — full 2×2×2 matrix** (corrected operator;
pattern × operator-class × macros, frame-0020 probe, calibrated pipeline;
`results_sim_AMR_ball_2x2x2/`):

| cell (pattern/operator/macros) | parity | seam mean | seam max | rho_std | final mass |
|---|---|---|---|---|---|
| pull std base | 2.6078e-6 | 4.2416e-5 | 3.6056e-4 | 6.2119e-5 | 5.029922e+05 |
| pull std geier (**fixed**) | **1.8429e-6** | **3.0923e-5** | 2.4003e-4 | 5.5268e-5 | 5.029546e+05 |
| pull well base | 2.6005e-6 | 4.2416e-5 | 3.6005e-4 | 6.2140e-5 | 5.029940e+05 |
| pull well geier | 2.0116e-6 | 2.9348e-5 | 2.0493e-4 | 5.5242e-5 | 5.029563e+05 |
| push std base | 2.6099e-6 | 4.2420e-5 | 3.6057e-4 | 6.2117e-5 | 5.029922e+05 |
| push std geier (**fixed**) | 1.8374e-6 | 3.0932e-5 | 2.4004e-4 | 5.5267e-5 | 5.029546e+05 |
| push well base | 2.5989e-6 | 4.2416e-5 | 3.6004e-4 | 6.2140e-5 | 5.029940e+05 |
| push well geier | 2.0108e-6 | 2.9348e-5 | 2.0494e-4 | 5.5242e-5 | 5.029563e+05 |

(defective-era reference: geier parity 3.2487e-3, seam 1.5256e-3, seam max
2.3697e-3, rho_std 5.0753e-4, recirc −9.96e-4-era; base rows unchanged at
print precision.)

**Visual seal** (same palette + shared range, local untracked):
`results_sim_AMR_ball_2x2x2/snapshots_vx_xz/ab_pull_geier_pre-correction.png`
vs `ab_pull_std_geier.png` — the defective frame shows the fine patch
completely filled with dense alternating red/white seam-band striping,
abruptly terminating at the right seam line; the corrected frame is perfectly
smooth with the patch footprint invisible.

**Verdict (signed).**

1. **The surviving §2.6 carrier is eliminated:** the corrected antialias arm
   lands *below* the macros-off floor (parity 1.84e-6 vs base 2.61e-6; seam
   3.09e-5 vs 4.24e-5) — the paper-faithful feedback is benign at this case,
   even mildly suppressive. The 1250× amplification chain was the response of
   a polluted estimator, not of the published terms.
2. **Cross-validation signed:** the never-defective CUM_WELL operator and the
   corrected standard operator agree on every metric within ~1–4 % (geier
   rows) and ~0.01 % (base rows), pull ≡ push at FP noise throughout; the
   7-digit conservation/KE agreement between patterns holds per cell.
3. **§6 Tier-1 bars pass WITHOUT any refill flip:** parity 1.84e-6 ≤ 1e-5,
   seam mean 3.09e-5 ≤ 4e-4, seam max 2.40e-4 ≤ 1e-3, rho_std improved vs
   baseline, development intact (max|vx| class preserved; recirc −2.29e-3
   between the two macro states — the genuine operator-physics difference the
   correction is supposed to make). **The default flip recorded in §5 is
   therefore not needed on the canonical case:** the T1 seed persists
   (~2e-6, honest baseline) but its amplification engine was never in the
   band — it sat in the operator text. §5's ranked candidates stand
   measured-not-adopted (reference arms for the residual campaign below).
4. **§4.3 resolution (signed):** the Part-II-class interface was seeded and
   seen. The artifact's two authors were code-level: the D1/D2 placement
   traffic (closed by the slot refactor, §2.6) and the operator's
   transcription defect (closed here). No intrinsic or unresolved third
   author remains on the canonical case; what remains is the null remnant
   plus the DFMAX == 1 class issue below.
5. **Standing open issues (moved to §7 register):** (a) the DFMAX == 1
   parity-class mass pump at BC-adjacent interfaces (AA + ESO trio; channel
   superlinear pump, ball runaway — pre-existing, byte-identical to stock
   `build-aa` with and without the new ESO code) — the next campaign's
   subject; (b) Tier-0 housekeeping: the canonical bitidentity gate sits at
   16/19 with three documented digest classes (two doctest line-reference
   shifts, two deliberate AA data moves) pending the owner's re-record
   ruling — current accepted reference behavior is established by this
   document, not by the recording; (c) unrun Tier-1 long-window legs
   (t = 5 s/10 s) and Tier-2/3 items; (d) the +0.52 AA pile audit; (e)
   Oracle LOW follow-ups (direction-table hoisting, CMake stickiness docs).

Reproduction: operator fix `11cd893`; macro/operator toggles at
`sim_AMR_ball.cu:1-2` (macros) and `:553-554` (`D3Q27_CUM` ↔ `D3Q27_CUM_WELL`
with `EQ_INV_CUM_WELL`), per-state rebuilds of both trees; runs at the §1
canonical args into `results_sim_AMR_ball_2x2x2/<pattern>_<std|well>_<base|geier>/`;
probe = the §2.6-calibrated `checkerboard_probe.py`; snapshot pipeline
`snapshots_vx_xz/snapshot_xz_vx.py` + the pre-correction one-off it spawned.

---

## 3. Confirmed mechanism

**Measured core (definitive).** The authoring site is substep 2's streaming
consumption of the inner ghost row's substep-1 collision product. Feeding substep 2
a fresh fill instead (T1b) collapses parity 332× and the seam 5.65×, with flow
development intact and deterministic re-run. The fill's window parity content is
inert, the third-order channel is suppressive, and the frame timing is clean.

**Mechanism read (inference, flagged — consistent with all arms of both rounds):**

1. **The mixed-shell collision is the only period-2-capable author.** The fill is
   smooth and parity-free at injection. At substep 1 the inner ghost row collides
   on a structurally inconsistent shell: outward populations are low-pass fill
   (coarse-projected, σ-mapped), inward populations are broadband live fine
   content. No smooth field has that pull state; the collision equilibrates about
   the mixture, and its outgoing populations toward the interior inherit a defect
   at the **fine-grid Nyquist** — the only wavenumber at which “fill vs live” can
   differ identically on every coarse cell.
2. **Why period-2 rather than smooth band error: both exist.** The smooth part is
   the recorded standing seam bias (−2.8e-5, shear-carried; implementation doc
   ¶316, 318–331). The defect's remaining component lives in the hydrodynamically
   invisible kinetic sector — a `(−1)^n` mode the strain-driven collision channels
   neither see nor damp (the Krüger/Rhie–Chow class), relaxed only via the slow k3
   sector. On a 2:1 cell-centered refinement the seam's only period-2 degree of
   freedom is the subcell pair, so the aliased mode presents locked to subcell
   parity. The parity metric is exactly the amplitude of the resulting decaying
   ripple.
3. **Sign coherence of the ω3 arms.** Band-local full relaxation (κ=1, +113 %)
   makes the live rows arrive at the shell more equilibrium-like, enlarging the
   live↔fill shell mismatch; over-retained fill content (−8.4 %) lets fine-scale
   texture through, shrinking the mismatch. Artifact magnitude tracks **shell
   mismatch** — which is why the third-order channel reads as suppressor.
4. **Why the refill erases it.** Fresh interpolation is strictly low-pass and
   time-consistent: it re-imposes both parity families' boundary data from the same
   underlying coarse field each substep, so no `(−1)^n` component is injected at
   consumption.

---

## 4. Literature correlation (Oracle consultation, 2026-09-05)

Grounded in the in-tree full texts (`docs/references/AMR/*.txt`; anchors below).
The previously mislabeled `04_Astoul_2021a_SpuriousNoise.txt` was replaced on
2026-09-05 with the true paper (arXiv:2004.11863v1 [physics.comp-ph] preprint, no
DOI) and re-read first-hand; Astoul 2021a line anchors below refer to that in-tree
preprint (section/figure numbering may differ from the published version).

### 4.1 Who integrates band nodes vs re-interpolates each fine substep

| Scheme | Band/ghost treatment between substeps | In-tree anchor |
|---|---|---|
| Filippova & Hänel 1998 (JCP 147, 219) | fine boundary nodes re-computed **every fine step** from coarse values, 2nd-order interpolation **in space and time**; never collided | `20_...:379–403` |
| Dupuis & Chopard 2003 (PRE 67, 066707) | (not in tree) cell-vertex family restated in Eitel-Amor 2023: missing interface populations provided **before every collision step**, coarse populations **linearly interpolated in time** during asynchronous iteration | `16_...:1424–1444`, Algorithm 1 |
| Lagrava et al. 2012 (JCP 231, 4808) | fine interface sites **completely reconstructed** at the mid-substep from coarse values **linearly interpolated to t+Δt_c/2**, cubic space interpolation + neq-filtered reconstruction; known cost: artificial interface viscosity / strong dissipation | `33_...:621–652`, `:556`, `16_...:2932`; secondary restatement `04_...:581–587`, `694–702`, `733–766` |
| Touil, Ricot, Lévêque 2014 (JCP 256, 220) | direction-weighted neq filtering in the reconstructive coupling | `16_...:1477–1489` |
| Gendre et al. 2017 (PRE 96, 023311; Wissocq school) | ghost layer of fine nodes filled **at each fine time step**, requiring **≥3rd order in space and time** | `24_...:1063–1075`, `1107–1112` |
| Eitel-Amor, Meinke, Schröder 2023 (Fluids 8, 103) | per-substep coupling with linear-in-time interpolation | `16_...:1424–1489` |
| Schornbaum & Rüde 2016 (waLBerla) | “**cells in ghost layers are not included in the collision**”; innermost ghost layers filled from coarse each fine-substep cycle, stream-only | `38_...:641–648` |
| Astoul et al. 2021b (direct coupling) | missing populations at transition nodes reconstructed per iteration with **temporal interpolation** of coarse populations | `05_...:821`, `:947` |
| Dorschner/Frapolli entropic multi-domain | twin + reconstruction overlap, per-iteration reconstruction; biased corner stencils “trigger spurious artifacts at the grid interface” | `48_...:13614–13624`, `13692–13696` |
| Chen et al. 2006 (volumetric/PowerFLOW) | different family: volumetric formulation, exact interface conservation by construction, no population shells | `13_...:811–820` |
| **Schönherr et al. 2011 (CMA 61, 3730)** | **the simulated band's provenance**: “Time interpolation is not required… the fine domain is enclosed in two rings of interpolated fine grid nodes. The outer ring becomes invalid in the asynchronous time step. Both rings are refilled immediately after the inner ring becomes invalid.” Time interpolation **explicitly considered and discarded** on GPU memory/efficiency grounds | `45_...:218–234` (non-reflectivity claim `:204`) |
| **Schönherr thesis 2015, ch.7** | same erosion protocol (Fig. 7.4–7.10); the chapter makes **no conservation claim and reports no artifact**; explicitly warns coincident-node schemes are “susceptible to **grid scale oscillations**” from unfiltered F2C transfer | `47_...:2961–2986`, `2898–2901`; close-read §8 |
| **Kutscher, Geier & Krafczyk 2018 (Computers & Fluids 165, 48)** | the same cycle in production (cumulant operator, staggered compact interpolation): “**Instead of interpolating in time we use a small overlap between the grids**”; two fine rows interpolated together at synchronization, the first fine line **erodes** in the asynchronous step (no coupling performed), all invalid nodes refilled once per cycle at the following synchronization | `44_...:1423–1438` |
| Musubi / Qi et al. 2019 | compact-interpolation production descendant; measured **mass-conservation violation** on periodic TGV | `50_...:688–691` |

### 4.2 Verdicts

- **Provenance.** The “simulated band” (integrate the inner ring at the async step,
  let invalidity erode, refill both rows once per cycle) is original to the
  Krafczyk school — Schönherr et al. 2011 is its first written statement, the 2015
  thesis ch.7 restates it, Kutscher 2018 carries it into production in their own
  words (“instead of interpolating in time we use a small overlap between the
  grids”, `44_...:1426–1427`), and Musubi is its production descendant. **Every
  other published subcycling scheme re-interpolates the fine boundary at every
  fine substep with linear-or-better interpolation in time, and never collides
  band cells.**
- **Kutscher 2018 is the direct parent of our C2F fill — and of candidate 1b's
  umbrella.** The production scheme of our compact-moment reconstruction is
  Kutscher, Geier & Krafczyk 2018 §3: quadratic velocity interpolation functions
  (their Eqs. 79–81, `44_...:1195–1248`) whose 33 coefficients are closed by nodal
  velocities plus pre-collision-cumulant derivative constraints (Eqs. 82–91,
  `44_...:1250–1317`), trilinear density (Eq. 92), and σ-scaled destination
  second-order cumulants from the analytic polynomial derivatives — their
  A011/B/C(x,y,z) correction terms (Eqs. 93–97, `44_...:1343–1413`) are the
  namesake of the A011/A101/A110/corr_B/corr_C variation terms in
  `amr_coupling.h`. Candidate 1b's shared 27-node quadratic fit is the
  single-interpolation-cell instance of that same umbrella. Two bearing notes:
  (i) the school attributes its interface robustness to the STAGGERED ALIGNMENT —
  an implicit low-pass filter “conforming with the Nyquist–Shannon theorem” —
  not to the cycle schedule (`44_...:1213–1232`); our T1 finding refines that
  picture: alignment guards the fine→coarse decimation, but the coarse→fine
  consumption schedule still authors the parity mode. (ii) Their production runs
  (refined 3D meshes beyond a billion nodes) used ω2..ω10 = 1 (“a very stable
  choice but more accurate choices exist”, `44_...:1185–1187`); the round-1
  `AMR_BAND_OMEGA3` κ=1 arm imposes exactly that rate family band-locally and
  AMPLIFIES the artifact (+112.8 %) — consistent with the artifact not being
  rate-authored. No parity-class artifact is reported anywhere in the paper.
- **Novelty of the finding.** No published record reports a defect of this parity
  class for the simulated band. Gendre 2017 states the gap outright for the compact
  cell-centered class that avoids time interpolation: “the presence or absence of
  discontinuities on the density field is not investigated” (`24_...:1081–1085`).
  The T1 confirmation is, as far as the published record shows, a **newly
  documented defect of that class**. The first-hand 2021a text does not change
  this: its coupling is Lagrava (per-substep re-interpolation, so the band-authored
  class cannot occur there), its parity diagnostics are time-domain only, and its
  findings are dynamic wave emissions, not a stationary parity-locked seam defect.
- **Parity pathology class.** The odd–even class is canonical: Rhie–Chow on
  collocated grids; Krüger book §2.1.1.2 (`31_...:3462–3489`); LBM-AMR “grid scale
  oscillations” at unfiltered interfaces (thesis intro; Fakhari 2016:
  “moment-based approach very sensitive to grid scale oscillations”, `43_...:392`).
  A period-2 mode **locked to refinement subcell parity** is not explicitly
  reported in the surveyed literature.
- **Astoul 2021a, first-hand content (re-read 2026-09-05).** The paper analyzes
  the cell-vertex Lagrava coupling (per-substep re-interpolation, no simulated
  band) and establishes: a six-mode eigenmode taxonomy — physical shear/Ac± plus
  the non-hydrodynamic SpuriousS (carries transverse velocity, wrong phase speed),
  SpuriousAc (carries acoustics, wrong celerity), SpuriousG (macroscopically
  invisible ghost) (`04_...:1002–1022`); a passage-matrix description of mode
  non-preservation across the resolution change — block-diagonal over
  acoustic/shear blocks and ghosts, and explicitly collision-operator-dependent
  (`04_...:1270–1305`, `1410–1416`, `1626–1638`); an interface amplifier in which
  parity-in-time modes (ωr≈π, amplitude inverting every iteration) sit in phase
  opposition across the asynchronously updated seam and defeat the mandatory
  temporal interpolation — up to 4×10⁵ amplification of incident SpuriousAc
  (`04_...:1966–1975`, `2560–2563`); three simulation-time non-hydrodynamic
  sensors (`04_...:1658–1806`); and a collision-side remedy — H-RR hybrid
  recursive regularization dissipating SpuriousAc at ≈418ν and cutting
  spurious-noise PSD by up to four orders of magnitude, low-pass filters
  explicitly dismissed as insufficient (`04_...:521–570`, `1173–1186`,
  `2826–2832`, `3773–3783`). Bearing on §3: our "hydrodynamically invisible
  kinetic mode" framing is the same checkerboard family — their period-2 is in
  TIME (ωr≈π amplitude inversion), ours in SPACE (fine Nyquist, subcell lock);
  their named amplifier (phase-opposed parity modes meeting at an asynchronously
  updated seam with interpolation across them) is the time-domain sibling of our
  mixed-shell collision authoring (low-pass fill vs broadband live populations
  equilibrated together).
- **Ghost-row non-equilibrium contamination.** Documented first-hand by Astoul
  2021a: non-hydrodynamic modes at refinement interfaces generate spurious
  vorticity and acoustics, addressed by the choice of collision model
  (`04_...:102–108`; sensors `04_...:1658–1806`). One directional nuance: their
  contamination modes arrive FROM the fluid core and are damped upstream of a
  well-interpolated interface — the remedy works because their algorithm
  re-authors the interface populations from coarse data before every collision
  (`04_...:770–829`). Our ghost row is authored AT the interface: after one
  integration it carries no valid non-equilibrium source term, because its own
  neighbor shell was interpolated, not integrated — exactly that mechanism,
  transposed from the acoustic to the hydrodynamic sector.
- **A counterpoint to record.** Astoul 2021a's headline claim — spurious wave
  generation is "intrinsically due to the change in grid resolution (aliasing)…
  independently of the grid transition algorithm" (`04_...:47–49`, `121–124`) —
  is in measured tension with T1b: at FIXED resolution change, refilling the
  consumed frame collapses parity 332× (§2.2). The resolution change is necessary
  (their passage-matrix projection), but in the simulated band the dominant
  author is the cycle schedule, not the projection.
- **Structural position of our coupling.** The cell-vertex schemes also collide
  their interface nodes, but they **re-author the interface populations before
  every collision** from time-interpolated coarse data. The unique property of the
  simulated band is that substep 2's boundary data is the previous substep's
  collision product on a half-fill/half-live shell — something no published scheme
  consumes. The T1b refill moves the scheme one notch toward the Lagrava/FH
  position (interpolation shell at the consumed substep) while keeping substep 1's
  integration — a hybrid with **no published equivalent**, which is precisely why
  its conservation properties must be re-measured.
- *(Pointer, not citation — no in-tree text)*: the uniform-LBM relatives of the
  period-2 lock are the staggered invariants of the lattice-gas era and the
  parity/even–odd decompositions in Ginzburg & d'Humières' TRT and boundary
  analyses (“magic parameter” cancellation of staggered content); also Bellotti's
  equivalent-equation sublattice view (`09_Bellotti_2023`, weak anchor).

### 4.3 Open problem (recorded 2026-09-05): is the Part-II interface clean by mechanism, or seeded and unseen?

**Resolved 2026-09-21 by §2.6/§2.7: seeded and seen.** The artifact's two
authors were code-level (the D1/D2 placement traffic; the operator's
antialiasing transcription defect) and are both closed; no intrinsic third
author remains on the canonical case.

A close read of the thesis stepwise description (`47_...:2947–3053`, Figs 7.4–7.10)
settles a reading that the terse 2011 passage (`45_...:218–234`) leaves open: the
rings are live stream participants, not passive receivers — invalidity arrives
via streaming (`47_...:2967–2969`). The inner ring is collided and streamed in both
fine substeps (“the lattice Boltzmann collision and streaming on the fine grid are
executed”, `:2961–2963`, and again at the synchronous fine step, `:2975–2977`);
the outer ring's collision participation is implementation-inferred (the uniform
fused GPGPU kernel, `45_...:224–225`) with no load-bearing consequence either way.
What distinguishes the rings is the erosion of validity from the outside inward,
one node layer per fine substep (`:2985–2987`), with the wholesale C2F repair at
the synchronization (step 4, `:3011–3024`). The outer ring's fill has a single
load-bearing consumer per cycle — the inner ring's asynchronous step, for which it
is the valid outer-side source (consumed directly under pull-gather, or via the
outer ring's own collision first under a fused kernel), so that the inner ring's
post-substep-1 state is a genuine fine state at t+Δt_f — the state substep 2's
interior boundary then consumes. The thesis states the design intent verbatim:
“This erosion of the interface combined with a finite overlap is the alternative
to an explicit interpolation in time” (`47_...:2979–2980`). The 2011 Fig.-1 caption
(“filled with distributions obtained by second order compact interpolation”,
`45_...:191–193`) describes only the synchronization fill, not the substep
behavior — the “passive fringe” reading it invites is wrong, and is logged here as
corrected so it is not re-made.

Consequence, element by element — the Part-II production interface (thesis §7.1 =
2011 = what Part II runs with “compact quadratic interpolation [25–27]”,
`03_...:104–106`) and our ch7 band are topologically isomorphic on the entire
load-bearing path:

| element | Part-II interface | our ch7 band | same |
|---|---|---|---|
| fringe | two rings, in coarse territory | two ghost rows, footprint re-anchored one fine cell inward | ✓ |
| fill cadence + time level | once per cycle at sync, coarse post-step state (step 4 after step 3) | once per cycle, C2F substep-0 frame, coarse df_out = post-step (§2.2 static trace) | ✓ |
| substep 1 | whole fine grid collides; inner ring sources outer-ring fill + live interior | widened extent; inner ghost row sources outer-row fill + live interior | ✓ |
| substep-2 boundary data | inner ring's substep-1 product (genuine fine state at t+Δt_f) | inner row's substep-1 product (other AB frame) | ✓ |
| outer row's substep-1 product | streamed into the inner ring at step 2 (the erosion read), consumed by no valid collision | written by the substep-1 stream (scatter destination), never read | ✓ functionally |
| fringe repair | wholesale C2F overwrite each cycle | wholesale C2F overwrite each cycle | ✓ |
| subcycling | Berger–Colella recursion, 2:1 per pair | `advancePair` recursion, 2^L substeps | ✓ |

The mixed-shell collision — T1's confirmed authoring site (§2.2) — therefore exists
in their interface in the same form: the substep-2 interior-boundary collision
mixes live interior populations with the inner ring's fill-derived substep-1
product, whose outer-side validity the interpolated outer-ring fill sustains —
the same consumption chain T1 located in our cycle. Residual differences and
their measured status in our cycle:

| difference | measured status |
|---|---|
| fill operator class (bubble-function quadratic interpolation vs σ-form compact-moment C2F) | INERT — round-1 candidate 1b swapped the K2018 Eqs. 79–91 basis (the direct descendant of their quarter-point scheme) into our cycle; ~85 % of seam bias parity-independent at the fill (§2.1) |
| rates (their production ω3/4/5 = 1) | ADVERSE — +152.7 % parity, the largest worsening of the campaign (§2.3) |
| C2F/F2C cycle order | FALSIFIED as a factor — T2a inert, T2b carrier, T2c destructive (§2.2) |
| coarse-side policy (their rim updates-then-repairs vs our frozen footprint + depth-1 skin) | non-author — the artifact is authored fine-side (§2.2) |
| streaming array layout (Esoteric Twist single-array in-place vs our AB two-array) | UNTESTED — our AA pattern is the nearest cousin; the parity metric has never been run under AA |
| fill-site geometry (quarter points inside coarse cells vs σ-form ghost-row placement) | UNTESTED as a package — round-1 tested the reconstruction basis inside our cycle only |
| case + validation regime | unmeasurable from here — seams at the sphere surface, wake deliberately unresolved (`03_...:107–108`), drag averaged over ~2·10⁶ steps, no seam-parity diagnostic in the lineage |

**The open problem.** Every mechanical difference we have measured went inert,
adverse, or falsified — the measured set cannot explain the school's clean
record; the untested named differences (Esoteric-Twist layout, quarter-point fill
geometry, wider F2C outset) remain candidates. Meanwhile no parity-resolved
diagnostic exists in the lineage — the school's interface diagnostics are
smooth-class (e.g. K2018's pressure-drop kink test, `44_...:1670–1672`, which
their compact fill passes) and Part II validates by drag over 1.8–2.88·10⁶ steps.
Either an unisolated mechanical difference breaks the authoring
chain in their code — candidates: the single-array Esoteric Twist streaming layout,
the staggered quarter-point fill geometry (their own attribution: the implicit
low-pass “conforming with the Nyquist–Shannon theorem”, `44_...:1213–1232`), the
wider fine-side F2C source stencil — or their interface carries the same (−1)ⁿ
seeding site and their validation modalities never see it: drag averages integrate
it out, instantaneous turbulent figures sit far above a 10⁻³-class zero-mean ripple,
and the seam sits where nothing turbulent crosses it. Our measurements cannot
decide this from inside our codebase. Falsifiers, in cost order: (1) the parity
metric under the AA build (nearest cousin of in-place streaming; cheap) —
**settled 2026-09-05 (§2.4): the artifact persists under A-A in all four
operator combos, on top of an operator-insensitive ~1e-3 floor (a separate
interface defect; the audit question posed in the round-3 synthesis)**;
(2) a fill-site-geometry-only arm — staggered quarter-point destination placement
grafted onto our σ-form fill inside our cycle, keeping basis and schedule fixed —
isolating the school's own attributional candidate (the stagger as implicit
low-pass, `44_...:1226–1232`); (3) a turbulent high-Re arm of the canonical case
(does our own metric resolve the mode under turbulence — the masking question
asked of ourselves); (4) the full Part-II package arm (quarter-point quadratic
fill + their erosion discipline as a unit, optionally including the wider
fine-side F2C outset). A measured collapse under (2) or (4) would isolate the
missing difference and vindicate clean-by-mechanism; persistence under (1)–(4)
would seal the seeded-and-unseen reading — fork (b) with a mechanism basis rather
than a masking conjecture.

---

## 5. Ranked production-fix candidates

**Superseding note (2026-09-21, §2.7):** on the canonical case these
candidates are measured-not-adopted — the Tier-1 bars pass with no flip
because the amplification engine was the operator's transcription defect, not
the substep-2 mixed shell. Keep them as reference arms for the residual
DFMAX == 1 campaign; the §6 matrix remains the ruling instrument if any
residual artifact class emerges there.

Constraints: keep the measured 332× suppression; keep the widened substep-1 launch
untouched (it is load-bearing for conservation/transport — it is the device that
closed the −23 % mass leak by ~5 orders in the ch7 conversion). The consumption
channel to fix is substep 2's streaming source at the inner ghost line, nothing
else.

| # | candidate | artifact hypothesis | conservation / trade | surface | falsifier |
|---|---|---|---|---|---|
| 1 | **row-targeted substep-2 refill** (narrowed T1b): refill only the consuming frame's inner ghost row between substeps (outer-row writes are measured dead traffic; further narrowing to inward-propagating populations possible) | measured 332× class — collapse should transfer verbatim (superset arm verified) | same demoted-terminal-shell question as T1b — the exact trade the Schönherr 2011 paper rejected on efficiency grounds (+1 fill set per pair; watch rho std) | small: new patch extent in `buildCouplings` + one `launchCoarseToFineTransfers` site; define-gated first | parity suppression materially below T1b's 332× on the canonical case (contradicts the T1 trace; not expected) |
| 2 | **time-harmonized mid-pair refill** (schedule swap: substep A → coarse step → refill from ½(pre+post) coarse frames = Lagrava/FH/DC03 linear-in-time at t+Δt_f → substep B) | should match or beat rank 1; **also removes the one-substep staleness** — the only candidate fully consistent with the entire literature | preserves widened launch; both coarse AB frames exist at that point (one-line source blend, no extra storage); but `advancePair` ordering + nested mid-sync F2C pairing + cycle-end cascade all re-derive | medium: schedule surgery, F2C frame audit under nesting, bit-identity re-record, contract §3 amendment | 20-tc table: must keep drift closure; this is the escalation if rank 1's table re-opens drift |
| 3 | **neq-only refill** (consumed populations := eq(ρ,u of integrated band product) + CM σ-form interpolated neq) | collapses if authoring is neq-channel (most likely) | keeps the collision's hydrodynamic moments — the leak-closing transport — verbatim; moment-consistent at ρ,u but discontinuous in strain flux | small: new store path in the C2F kernel, define-gated | **cheapest decisive attribution probe**: if parity does NOT collapse → the macro channel is the author and §3's mechanism is wrong; run first regardless of adoption |
| 4 | **stream-only band** (never collide the band rows; per-substep fill + stream pass-through — waLBerla's “no collision in ghost layers” on our band map) | collapses (removes the mixed-shell collision site) | no Dirichlet clamp, physical flux through the band, but a relaxation-less band of the Lagrava dissipation class — drift closure must be re-measured; band macros come from the fill | small–medium: GEO tag + collision predicate in `kernels.h` + SimInit map contract | parity survives with collision off → streaming of the mixed shell alone sustains the mode; run as a structural witness even unadopted |
| 5 | **T1b as measured** (both rows, parent pre-step frame, `AMR_SUBSTEP2_C2F_REFILL`) | 332× / 5.65× verified, deterministic | the demotion ruling; +12 % interior rho std; one full extra C2F launch set per pair | none (in tree, define-gated) | fallback default-flip arm |

**Answer on the diagnostics-only idea** (recompute band macros each substep): a
priori inert for the artifact — parity lives in the DFs, and T1a proved band-row
macros are outputs, not inputs. Useful as post-fix visualization hygiene only.

**Literature-suggested alternative not in the candidate list:** higher-order (≥3rd)
time interpolation à la Gendre — only relevant if aeroacoustic quietness becomes an
acceptance axis (Astoul framing). Overkill now. The first-hand 2021a read sharpens
the endorsement of ranks 1/2: their baseline collides interface nodes only after
re-authoring populations from coarse data at every asynchronous substep (rank 1's
structural position, `04_...:770–829`), and third-order time interpolation of the
coarse state is "mandatory" in their scheme (rank 2's prescription,
`04_...:733–745`; the phase-opposition amplifier it prevents, `04_...:1966–1975`).
Cumulant is explicitly named as not yet covered by their spectral program
(`04_...:4074–4078`) — our ω3 arms are the first (negative) evidence for the
cumulant band-relaxation class.

**Do not resurrect (measured):** parity-window fills 1a/1b (inert), ω3
band/transfer discount arms (wrong sign), frame-displacement arms (inert/harmful),
H9/BVP-class content no-ops, HRR-class band-local collision-side mode damping (the
Astoul remedy class — it presumes modes arriving from the fluid core at a
well-interpolated interface; our mode is authored AT the band by shell mismatch,
and the round-1 `AMR_BAND_OMEGA3` arms already swept the band-local relaxation
class the wrong way: κ=1 → +112.8 %, κ=0.5 → +52.6 %).

---

## 6. Validation matrix (pre-registered, gates for the default flip)

**Tier 0 — invariants (macros-off bit-identity):**

1. `pytest tests/regression/test_amr_bitidentity.py` — 11/11 on the macros-off
   build (protects the byte-frozen `max_level == 1` reduction).
2. AMR gate `pytest tests/unit/test_amr_units.py tests/integration/test_amr_paraview.py`
   — 10/10, both streaming patterns; deterministic double-run bitwise on the
   canonical case.

**Tier 1 — artifact (reuse the calibrated ball probe, same environment pin as the
2026-09-05 runs):**

3. Canonical case, t = 1 s: **PASS** = parity ≤ 1e-5 (≥100× suppression), seam
   mean|dvx| ≤ 4e-4, seam max ≤ 1e-3, development intact (max|vx| within baseline
   ± tolerance; centerline min vx no weaker than baseline), interior rho std
   reported (watch > +25 % vs baseline).
4. **New long-window leg:** same case at t = 5 s and t = 10 s; **PASS** = parity ≤
   5e-5 with no monotone re-growth (catches slow re-seeding the 1 s window can't
   see).
5. *(Instrumentation, ~30 lines)* full-row parity metric on the inner ghost row
   (fg −1) pre/post refill — directly measures the authoring-site content delta
   per arm.

**Tier 2 — conservation (contract §1/§6 instruments):**

6. TGV 20-tc decision table re-run (`sim_AMR --resolution 1 --convective-times 20`,
   5-arm ladder incl. EQ/NORM/SHEAR attribution arms + Lagrava opt-out): **PASS** =
   final |drift| ≤ 5e-5 rel (HEAD arms sit at 2.3e-6…1.4e-5; the −23 % closure must
   survive within one order), |slope| ≤ 1e-8, seam-bias amplification ≤ ×1.2 of the
   HEAD default (T15b bound), KE arms in the same class as HEAD (the vortex is dead
   at HEAD — the bar is “same dead class”, not survival), fill-channel attribution
   sign pattern stable (shear-carried).
7. Channel B.7 refresh per contract §8.1 (641-iteration `--write-dfs` run):
   **PASS** = quasi-steady re-paired seam jump stays in the ~3.5e-2 arm-common
   class (no new sign alternation); Dirichlet-clamped mass invariant at print
   precision.

**Tier 3 — generality:**

8. Windbreak chain (`sim_AMR_channel --resolution 1 --max-level 4` with rod array,
   + max-level 2,3 reductions): **PASS** = pre-registered mass/KE tables of
   `6ae4a61`/`5214b01` reproduced; parity ≤ bar at the level-1 seam; **record wall
   time** (the nested refill fires once per pair per level ⇒ cost scales with the
   2^L pair cadence).
9. Robustness: ball case at `--resolution 1` and `3` (τ/ω3 ladder moves the
   measured g-ratios): **PASS** = ≥100× suppression at every resolution; one
   corner-exercising variant (footprint shifted or ball off-axis): **PASS** =
   parity ≤ bar at corner-adjacent seams.

**Tier 4 — process:** the flip is a contract §1/§3 user ruling; pre-register the
Tier-1/2 bars *before* candidate runs; reproduce the pinned baseline constants on
the pre-change build (T2-style calibration leg); flip only if tiers 0–2 pass on
the *define-off* build as well (macros-off must keep compiling the HEAD text).

---

## 7. Action plan and current tree state

### 7.1 Recommended sequence

1. Land rank 3 (**neq-only refill**) as a define-gated *probe* first — the cheapest
   decisive attribution test remaining (macro vs neq authoring channel) and a
   plausible production candidate. **(Short)**
2. Land rank 1 (**narrowed T1b**) as the production-candidate arm; probe legs:
   canonical case + long-window legs (tier 1). **(Short)**
3. Run tier-2 legs (TGV 20-tc ladder, B.7) on rank 1 (and on T1b as fallback).
   **(Medium — compute-bound)**
4. If rank 1's drift closure holds: assemble the user ruling with the tier 0–2
   matrix. If it re-opens: implement rank 2 (schedule-swapped time-centered
   refill) and re-run the same matrix. **(Rank 2 is 1–2 d code)**
5. Tier-3 windbreak + robustness sweep on the adopted arm; document cost lines
   (extra C2F set per pair per level). **(Medium)**
6. Update the `amr_coupling.h` docstring + contract §3/§11 with the ruling; keep
   all arms define-gated with macros-off compiling HEAD text. **(Quick)**

### 7.2 Watch out for

- **The widened launch is doing two jobs** (conservation transport *and*
  substep-2 boundary supply). Every ranked fix touches only the second; any arm
  that touches the first (passive band, stream-only band) needs the full tier-2
  table, not the artifact metrics alone.
- **Nested-level refill semantics** (ranks 1/2 under `max_level > 1`): the
  between-substeps refill at level L fires once per pair, i.e. 2^L× more often at
  deep levels — check both correctness (mid-sync source frames) and launch
  overhead on the windbreak chain before the ruling.
- **Bit-identity harness is a guard, not a judge:** the adopted arm will change
  `max_level == 1` bytes by design; re-record the manifest only after tier 2
  passes and the ruling is logged, from a trusted tree, exactly as the harness
  documentation demands.

### 7.3 Tree state at the time of writing

Uncommitted probe tree, purely additive over HEAD `08c8829`: 7 files,
+1174/−34 (`amr_coupling.h`, `amr_state.h`, `col_cum.h`, `defs.h`, `kernels.h`,
`lbm_data.h`, `sim_AMR/CMakeLists.txt`). All arms define-gated; macros-off
compiles the verbatim HEAD text. Gates at handoff: bit-identity harness 11/11,
AMR doctest gate 8/8, full build green.

Round-2 cache options (wired in `sim_AMR/CMakeLists.txt`, empty by default):
`AMR_C2F_READ_DF_CUR`, `AMR_F2C_READ_DF_CUR`, `AMR_C2F_WRITE_DF_OUT`,
`AMR_PASSIVE_BAND`, `AMR_SUBSTEP2_C2F_REFILL`. Round-1 selectors are compile-time
defines with the precedence documented in the `amr_coupling.h` docstring:
`C2F_SHARED_FIT` > `C2F_PARITY_AVERAGED` > `C2F_PER_PARITY` > [default];
`C2F_OMEGA3_DISCOUNT{,_MEMORY,_INVERSE}`; `AMR_BAND_OMEGA3=<κ>`.

**Reproduce the winning arm:**

```bash
cmake -B build -S . -G Ninja -DAMR_SUBSTEP2_C2F_REFILL=ON
cmake --build build --target sim_AMR_ball -j 16
mkdir run_t1b && cd run_t1b
../build/sim_AMR/sim_AMR_ball --resolution 2 --lattice-viscosity 0.001 \
  --phys-final-time 1.0 --max-level 1 --adios-config <repo>/adios2.xml > sim.log 2>&1
python3 /tmp/opencode/checkerboard_sweep/checkerboard_probe.py \
  results_sim_AMR_ball_res002_np001/output_amr_0020.vtkhdf
```

Macros-off restore: `cmake -B build -S . -UAMR_*` (all five options).
Round-2 run directories preserved under `/tmp/opencode/band_probe/run_{baseline,t2a,t2b,t2c,t1a,t1b,t1ab}/`.

---

*Provenance: two measured falsification rounds by delegated deep-research agents
(2026-09-03 ω3/parity round; 2026-09-05 T1/T2 timing round), two literature
consultations (Oracle, 2026-09-05) grounded in `docs/references/AMR/` — the second
re-reading Astoul 2021a first-hand after the `04_...` txt replacement (the earlier
copy was a mislabeled unrelated paper; the reference now cites the arXiv preprint,
no DOI) — and a first-hand pass over Kutscher 2018 (`44_...`, verified correctly
labeled with its DOI). All numbers above are session-measured values quoted from
the arm records in the `amr_coupling.h` docstring and the round-2 task report.*
