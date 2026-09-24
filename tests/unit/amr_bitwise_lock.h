#pragma once

// Cross-pattern bitwise lock of the AMR machinery — shared declarations.
//
// One binary (test_amr_bitwise_lock) co-instantiates the streaming patterns
// through the explicit D3Q27_STREAMING_{AB_PULL,AA,ESO_TWIST,ESO_PULL,
// ESO_PUSH} types (all six patterns are co-includable, AGENTS.md), runs the
// SAME two-level State_AMR scenario under each pattern (one runner
// translation unit per pattern, so the full State_AMR kernel instantiations
// compile in parallel), and asserts BITWISE equality of the pattern-agnostic
// physical state — the GEO map and the macroscopic channels over the whole
// stored box (interior + the C2F-fed ghost rows), level by level, cycle by
// cycle — against the A-B pull reference. Raw DF frames are NOT compared:
// each pattern stores them in its own parity/layout encoding, so only the
// macroscopic surface is pattern-agnostic (the same comparison surface the
// production evidence harness of results_drag_crisis/aa_seed_report.md §1
// uses on the VTKHDF frames).
//
// Locked surface (deliberately narrow — only what is PROVEN/provable exact):
//
// - the mL1 class (max_level == 1: levels 0..1, a single L0->L1 link) driven
//   10 cycles with a traveling sinusoidal density gradient crossing the link,
//   including the per-cycle cascade C2F fill and the F2C feedback. A-A vs
//   A-B pull is bitwise-exact on this class by direct production evidence
//   (aa_seed_report.md §2: "mL1 over identical physics/config is bit-exact
//   for 18 cycles, including cycles 5-18 during which the front crosses the
//   L0->L1 cascade fill"); §8.2 recommends exactly this harness ("in-binary
//   A-A-vs-A-B-pull bitwise harness over a two-level nested state driven
//   ~10 cycles with a traveling gradient, asserting per-cycle bitwise
//   equality level-by-level"). The esoteric in-place patterns are locked
//   against the same reference after being verified bitwise-exact on this
//   host empirically (RTX 5080, native arch).
//
// KNOWN DEFECTS — deliberately NOT locked (see the skipped documentation
// case in test_amr_bitwise_lock.cu):
//
// - nested links (max_level >= 2) diverge at the mid-sync C2F fill inside
//   advancePair under the single-array patterns (wrong frame/parity vintage
//   authoring — aa_seed_report.md §4, stage (iv));
// - the R = 1 inflow-adjacent C2F wall-guard path (a second, independent
//   seed; §6) — this harness is fully periodic and never exercises a
//   GEO_INFLOW_MOMENT-adjacent link;
// - mL0 (uniform max_level == 0) crashes under A-A at SimInit (§7).
//
// This header is pattern-independent: the comparator TU includes only it
// (no kernel instantiation), while the five runner TUs include
// amr_bitwise_lock_runner.h with the full State_AMR machinery.

#include <string>
#include <vector>

// number of driven coarse cycles (aa_seed_report.md §8.2: "~10 cycles"); the
// proven mL1 evidence horizon is 18 cycles, so 10 cycles of lock headroom
// still sit inside the proven-exact class
constexpr int AMR_LOCK_CYCLES = 10;

// one coarse cycle's snapshot of one block level: the GEO map (cycle 0 only —
// interface tagging is init-time) and every macroscopic channel over the
// whole stored box [origin, origin + size) including the overlap rows
struct AMRLockLevelSnap
{
	int level = -1;
	int origin[3] = {0, 0, 0};
	int size[3] = {0, 0, 0};
	int macro_channels = 0;
	std::vector<int> map;		// size.x * size.y * size.z entries (cycle 0 only)
	std::vector<double> macro;	// macro_channels * cells entries, widened float (exact)
};

struct AMRLockCycle
{
	std::vector<AMRLockLevelSnap> levels;  // ascending level order
};

struct AMRLockTrace
{
	std::string pattern;
	bool init_ok = false;	   // SimInit completed (canCompute, no terminate flag)
	int cycles_completed = 0;  // number of appended cycle snapshots (index 0 = post-SimInit)
	std::vector<AMRLockCycle> cycles;
};

// scenario runners, one per pinned streaming pattern (test_amr_bitwise_lock_*.cu)
AMRLockTrace runAMRLockScenario_ab();
AMRLockTrace runAMRLockScenario_aa();
AMRLockTrace runAMRLockScenario_eso_twist();
AMRLockTrace runAMRLockScenario_eso_pull();
AMRLockTrace runAMRLockScenario_eso_push();
