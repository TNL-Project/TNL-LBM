#pragma once

// Cross-pattern bitwise lock of the NESTED AMR schedule — shared
// declarations (the SP-4 surface of the out-of-scope survey: nested links,
// max_level >= 2).
//
// Companion harness of the mL1 bitwise lock (amr_bitwise_lock.h, same
// structure): one binary (test_amr_nested_lock) co-instantiates the
// streaming patterns through the explicit D3Q27_STREAMING_{AB_PULL,AA,
// ESO_TWIST,ESO_PULL,ESO_PUSH} types (one runner translation unit per
// pattern), runs the SAME three-level State_AMR scenario under each
// pattern, and asserts BITWISE equality of the pattern-agnostic physical
// state — the GEO map and the macroscopic channels over the whole stored
// box of each level, cycle by cycle — against the A-B pull reference. The
// comparison surface is the mL1 lock's (map + macro; raw DF frames are NOT
// compared because each pattern stores them in its own parity/layout
// encoding), reached through the same snapshot/comparator types, which are
// therefore reused by include rather than duplicated.
//
// Locked surface (the surface the mL1 harness deliberately excludes):
//
// - the mL2 class (max_level == 2: levels 0..2, the L0->L1 link plus ONE
//   nested L1->L2 link) driven 10 cycles with the mL1 harness's traveling
//   sinusoidal density gradient crossing BOTH links, so the per-cycle
//   mid-sync C2F fill inside advancePair (the band for each finer pair #2,
//   sourced from the parent's live post-substep-A state), the doubled
//   per-pair F2C cadence, and the level-ascending cycle-end cascade all
//   carry evolving content every cycle. This is exactly the nested-link
//   family of results_drag_crisis/aa_seed_report.md §4 stage (iv): the
//   mid-sync fill is the one Schönherr-cycle launch that exists only at
//   nesting depth >= 2 (the mL1 control class it reduces to is locked
//   bitwise-exact by the mL1 harness; §8 item 2's recommended in-binary
//   harness over a nested state driven ~10 cycles with a traveling
//   gradient).
//
// The scenario geometry is the mL1 fixture extended by one nested level:
// the same 16^3 periodic box and the same centered level-1 footprint
// [4,12)^3, with the level-2 footprint nested at parent-frame [12,20)^3
// (4 parent-level cells per axis, telescoping gap 4 parent cells on every
// face -- above the recommended 3, so no V-suite warn tier fires).
//
// This header is pattern-independent: the comparator TU includes only it
// (no kernel instantiation), while the five runner TUs include
// amr_nested_lock_runner.h with the full State_AMR machinery.

#include "amr_bitwise_lock.h"

// number of driven coarse cycles: mirrors the mL1 lock (AMR_LOCK_CYCLES);
// the mL1 surface births no divergence through 18 cycles and the seed
// report's production-geometry birth (frame 7) tracks the front's arrival
// at the nested link, which the fixture's fast acoustic crossing reaches
// within the first cycles
constexpr int AMR_NESTED_LOCK_CYCLES = 10;

// scenario runners, one per pinned streaming pattern
// (test_amr_nested_lock_*.cu)
AMRLockTrace runAMRNestedLockScenario_ab();
AMRLockTrace runAMRNestedLockScenario_aa();
AMRLockTrace runAMRNestedLockScenario_eso_twist();
AMRLockTrace runAMRNestedLockScenario_eso_pull();
AMRLockTrace runAMRNestedLockScenario_eso_push();
