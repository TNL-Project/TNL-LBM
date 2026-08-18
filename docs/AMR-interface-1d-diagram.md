# AMR Interface — 1D Schematic

**Scope:** 1D simplification of the coarse/fine interface as implemented in
`include/lbm3d/amr_decomposition.h` (`markAMRInterface`) and
`include/lbm3d/d3q27/amr_coupling.h` (`cudaAMR_CoarseToFine`,
`cudaAMR_FineToCoarse`). All diagrams show the **AB** (default) streaming
pattern only; the AA twisted-slot layer is the deferred Defect 1 of
`AMR-for-LBM-implementation.md` §10.1 and is intentionally omitted.
The right interface mirrors the left one; only the left is drawn.

Grid convention: coarse cells are 10 characters wide, fine cells are 5 —
fine cells `2c, 2c+1` are the two subcell volumes of coarse cell `c`
(each fine cell spans 1/2 of it), so the fine ghost ring `{-2, -1}` covers
the ring cell `o-1`'s volume, and the fine block boundary (between ghost
`-1` and interior `0`) is the *face between the coarse volumes* `o-1 | o` —
a cell-centered (volumetric) scheme: no quantities live at that face itself.

---

## 1. Layout — left interface

The fine footprint starts at coarse cell `o`. Fine cells are numbered in the
fine-block frame, so fine `0` = global `2o` and the ghost ring is `{-2, -1}`
(`storage_overlap = 2` on refinement-level blocks, see `LBM_BLOCK`).

```
                 o-3       o-2       o-1        o        o+1       o+2
              ┌─────────┬─────────┬─────────┬─────────┬─────────┬─────────┐
              │  fluid  │  fluid  │  RING   │ FROZEN  │ FROZEN  │ FROZEN  │
              │         │         │coll-act │ hidden  │ hidden  │ hidden  │
              └─────────┴─────────┴─────────┴─────────┴─────────┴─────────┘
                         -4   -3   -2   -1    0    1    2    3    4    5
                        (unstored)   GHOST      interior fluid (GEO_FLUID)
                                          └─ fine block boundary = face between the coarse volumes (o-1 | o)

 RING (o-1)   : GEO_AMR_INTERFACE — collision-active (streams + collides like
               fluid); its DFs are additionally overwritten by ring F2C at the
               end of each coarse step
 FROZEN (o..) : GEO_NOTHING — no stream/collide; DFs written EXCLUSIVELY by
               interior F2C each cycle
 GHOST (-2,-1): fine halo cells, 2 deep (storage overlap), filled by C2F;
               tiled by the ring coarse cell o-1 (they ARE its subcells)
```

---

## 2. Coarse→Fine (`cudaAMR_CoarseToFine`)

Runs twice per coarse step, before each fine substep (BVP re-fill).

**Output:** the two fine ghost cells' DFs,
`f = eq(ρ, u)_interp + (τ_f / τ_c) · f_neq_interp`, so fine interior cell `0`
has valid pull-sources for streaming.

**Inputs:** a 4-cell-per-axis 3rd-order Lagrange stencil per ghost cell
(shifted only near coarse block boundaries):

```
                 o-3       o-2       o-1        o        o+1       o+2
              ┌─────────┬─────────┬─────────┬─────────┬─────────┬─────────┐
              │  fluid  │  fluid  │  RING   │ FROZEN  │ FROZEN  │ FROZEN  │
              │         │         │coll-act │ hidden  │ hidden  │ hidden  │
              └─────────┴─────────┴─────────┴─────────┴─────────┴─────────┘
              └──────────────────────────────────────┘
                                     ▲  OUTER ghost -2 ◄── {o-3,o-2,o-1,o}
                                         w = {-5,35,105,-7}/128  (1 frozen cell: o @ -7/128)
                        └──────────────────────────────────────┘
                                          ▲  INNER ghost -1 ◄── {o-2,o-1,o,o+1}
                                              w = {-7,105,35,-5}/128  (2 frozen cells @ 35-5 = 30/128)
                                              ← §9.2.1 hidden-cell injection
```

Mirror at the right face: the inner ghost again reads frozen cells at 30/128.
(Under the trilinear fallback `-DC2F_TRILINEAR`, the frozen-cell weight of the
inner ghost is 1/4; the qualitative asymmetry is the same.)

---

## 3. Fine→Coarse (`cudaAMR_FineToCoarse`)

Runs once per coarse step, after fine substep 2. Two distinct outputs.

### (a) Ring F2C → ring cell `o-1` (overwrite of its post-collision state + macros)

**Inputs:** 4-node-per-axis Lagrava filter (nominal window `{f0-1 .. f0+2}`
evaluated at the coarse center `t = f0 + 0.5`), shifted right by one at this
face because `f = -3` is outside fine storage; weights recomputed for the
shifted window:

```
                 o-3       o-2       o-1        o        o+1       o+2
              ┌─────────┬─────────┬─────────┬─────────┬─────────┬─────────┐
              │  fluid  │  fluid  │  RING   │ FROZEN  │ FROZEN  │ FROZEN  │
              │         │         │coll-act │ hidden  │ hidden  │ hidden  │
              └─────────┴─────────┴─────────┴─────────┴─────────┴─────────┘
                         -4   -3   -2   -1    0    1    2    3    4    5
                                  ghost = C2F products (stale after fine cycle)
                                  └──────────────────┘
                                       ▲  ring o-1 ◄── {-2,-1,0,1}
                                           w = {5/16,15/16,-5/16,1/16}
                                           (2 of 4 inputs are C2F-produced ghost
                                            data; ghost weight 20/16 of Σw = 1)
                                           ← §9.2 one-way clamp, visible
```

### (b) Interior F2C → frozen cells under the footprint (exclusive DF writer + macros)

**Inputs:** centered (unshifted) window on fine fluid for non-edge cells; the
footprint-edge frozen cell `o` still dips one cell into the ghost:

```
                                       └──────────────────┘
                                                 ▲  frozen o ◄── {-1,0,1,2}
                                                      w = {-1,9,9,-1}/16 (centered)
                                                      … o+1 ◄── {0,1,2,3} (same weights)
```

The 1/8 box average over the subcells (1D analog of the 8 subcells in 3D)
remains available as `-DF2C_BOX_AVERAGE`.

---

## 4. Why the residual lives where it lives

- **The ring is doubly synthesized:** it is written by F2C from mostly-ghost
  inputs (3a), then consumed by C2F as the dominant "home" node (105/128) of
  the next fill (2) — fine-interior information crosses the ring only through
  the fluid cells at the window edge.
- **Frozen cells are both output and input:** interior F2C fills them with
  fine-averaged DFs (3b), and C2F immediately reads them back as interpolation
  sources at up to 30/128 weight (2) — the §9.2.1 hidden-cell injection.
  This is also exactly where the deferred AA Defect 1 lands: the AA read of
  these frozen cells is direction-reversed, injecting mirrored momentum into
  the C2F path every cycle.
- **The only channel carrying pure fine-interior data into the coarse lattice
  is interior F2C** (3b); the ring channel (3a) is structurally
  ghost-dominated — the §9.2 one-way clamp in one picture.

---

## References

| Mechanism | Doc section |
|---|---|
| One-way clamp | `AMR-for-LBM-implementation.md` §9.2 |
| Hidden-cell injection | §9.2.1 |
| Collision-active ring (v7) | §6, §9.3 |
| C2F/F2C kernel details | §4.1, §4.2 |
| AA deferred defect | §10.1 Defect 1 |

---

## Implementation outcome (D.5, 2026-08-16)

This document freezes the **pre-2026-08-16 design** (ring-F2C overwrite at
the end of each coarse step, 2-deep fine ghost `{-2,-1}` under
`storage_overlap = 2`, full-footprint interior F2C dipping one cell into
the ghost at footprint edges) — read its "as implemented" scope as the
record of that design and its §4 mechanism analysis (one-way clamp,
hidden-cell injection, the ring's 20/16 ghost recycling) as the accurate
autopsy that motivated the redesign. The shipping design recorded in
`AMR-for-LBM-implementation.md` §9.1/§9.3 differs as follows:

- **Ghost halo is 1 cell deep** (`{-1}` only): Phase C set
  `storage_overlap = 1` on refinement-level blocks — the second layer's
  only consumer was the ring-F2C filter window. Fields bit-identical to the
  overlap-2 winner (132 arrays, 0 bit-differences); fine-block storage
  −8.566 %; C2F fill work −48.5 %.
- **§3(a) ring F2C is deleted** (gate B user ruling A, then the D.1
  hard-delete): ring cells are written by the main coarse kernel only;
  fine feedback reaches them via streaming from the skin on the next
  coarse step. The §3(a) `{5,15,−5,1}/16` shifted-face constants numerically
  coincide with the LIVE min-side skin window's `{0,1,2,3}` weights — but the
  live scheme recomputes them at runtime at the fixed coarse-center
  evaluation point; nothing of the retired path is reused (D.2).
- **§3(b) interior F2C is restricted to the 1-cell-deep skin** (6 disjoint
  inset-face rectangles), with the filter window's lower bound clamped to
  the fine interior (`lo = 0`): the footprint-edge frozen cell `o` no
  longer dips into the ghost — its `{-1,0,1,2}` window shifts to
  `{0,1,2,3}`. The deep frozen core is never written.
- **Gate numbers** (§9.1 acceptance configuration, corrected baseline
  605,583 @ cycle 10): redesign B-on 220,737 @ cycle 10 (−63.6 %), series
  non-monotone (+3.1 % c5 transient, mechanism recorded); mass
  ulp-invariant; the attempted CM + carve C2F (a sibling experiment, not
  drawn here) was falsified by gate A (3,346,370 @ cycle 10).
- **The AA remark (§4, Defect 1)** was revisited under the accepted design
  (D.4): machinery parity-aware everywhere, "AA-likely-small-effort"; one
  precisely-localized remnant (frozen skin/core cells' direction
  orientation vs cycle parity at C2F reads, alternating every cycle at
  first-order weights) plus candidate fixes and a user-decision flag for a
  Lagrange-side carve — see the D.4 record in
  `AMR-for-LBM-implementation.md` §9.3.
