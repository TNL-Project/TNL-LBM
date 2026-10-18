#pragma once

// Cross-pattern bitwise lock of the UNIFORM (max_level == 0) State_AMR
// fallback — shared declarations (the SP-6 surface of the out-of-scope
// survey: the uniform fallback under the single-array patterns).
//
// Companion harness of the mL1 and nested bitwise locks
// (amr_bitwise_lock.h / amr_nested_lock.h, same structure): one binary
// (test_amr_uniform_lock) co-instantiates the streaming patterns through
// the explicit D3Q27_STREAMING_{AB_PULL,AA,ESO_TWIST,ESO_PULL,ESO_PUSH}
// types (one runner translation unit per pattern), runs the SAME uniform
// channel State_AMR scenario under each pattern (max_level == 0: no
// refinement blocks, the driver falls back to the base State<NSE>::
// SimUpdate byte-identically), and asserts BITWISE equality of the GEO map
// and the macroscopic channels over the whole stored box, cycle by cycle,
// against the A-B pull reference.
//
// Locked surface (the SP-6 surface of results_drag_crisis/
// out_of_scope_survey.md, CLOSED 2026-09-30): the uniform fallback of the
// sim_AMR_ball channel geometry. The historical defect was NOT schedular:
// the sim-side stampBallOnFineBlocks() level filter admitted the level-0
// block at max_level == 0 and stamped the resolved ball through its
// zero-initialized lat_local (the decomposeLattice constructor leaves
// lat_local default-constructed -- only createAMRBlocks' per-level path
// initializes it), tagging a lone stray GEO_WALL at the (0,0,0) domain
// frame. A-B clamps its neighbor indices to the stored box and silently
// tolerated the frame-BC cell (the "A-B healthy" half of the seed
// report's §7); the single-array patterns' unclamped ghost-layer gather
// (requires_ghost_layer_v -- the ghost-layer idiom demands a clean
// GEO_NOTHING frame) read one cell below the zero-overlap storage on the
// very first odd-parity launch ('-overlap <= i' NDArray bounds + CUDA
// ERROR 719). The sim-side guard (max_level == 0 -> level 0 is
// lbmDrawSphere's job alone) removes the stray stamp; this lock pins the
// repaired uniform path: the frame planes stay GEO_NOTHING-clean on every
// pattern and the full 10-cycle trajectory is bitwise across the pattern
// family.
//
// The scenario mirrors sim_AMR_ball's --resolution 1 uniform reference
// (62 x 32 x 32 inflow/outflow channel, wall planes, GEO_NOTHING frame,
// staircased ball in the interior) minus the drag probe, with the bitwise
// locks' traveling sinusoidal density gradient as the initial condition
// so the evolution-sanity anti-triviality guard keeps teeth (the
// inflow-driven development from rest alone moves too few cells inside 10
// cycles).
//
// This header is pattern-independent: the comparator TU includes only it
// (no kernel instantiation), while the five runner TUs include
// amr_uniform_lock_runner.h with the full State_AMR machinery.

#include "amr_bitwise_lock.h"

// number of driven coarse cycles (mirrors AMR_LOCK_CYCLES of the mL1 lock):
// the trajectory must carry both parities of every single-array schedule
// (the historical abort fired on the first odd-parity launch of
// State::SimUpdate); 10 cycles sit well inside the proven
// uniform-exactness class
constexpr int AMR_UNIFORM_LOCK_CYCLES = 10;

// scenario runners, one per pinned streaming pattern
// (test_amr_uniform_lock_*.cu)
AMRLockTrace runAMRUniformLockScenario_ab();
AMRLockTrace runAMRUniformLockScenario_aa();
AMRLockTrace runAMRUniformLockScenario_eso_twist();
AMRLockTrace runAMRUniformLockScenario_eso_pull();
AMRLockTrace runAMRUniformLockScenario_eso_push();
