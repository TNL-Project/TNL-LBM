#pragma once

// Cross-pattern channel-geometry lock of the AMR machinery — shared
// declarations (the SP-5 surface of results_drag_crisis/out_of_scope_survey.md:
// the R = 1 inflow-adjacent C2F wall-guard class and the channel's
// wall-attached slab class, aa_seed_report.md §6).
//
// One binary (test_amr_channel_lock) co-instantiates the streaming patterns
// through the explicit D3Q27_STREAMING_{AB_PULL,AA,ESO_TWIST,ESO_PULL,
// ESO_PUSH} types, runs the SAME developing-channel mL1 scenario under each
// pattern (one runner translation unit per pattern), and compares the
// pattern-agnostic state cycle by cycle against the A-B pull reference —
// the same comparator discipline as the mL1 bitwise lock
// (amr_bitwise_lock.h), extended from macroscopic channels to DECODED
// canonical DF frames: for each stored site and direction the trace carries
// the value a kernel launch of the upcoming parity gathers there (the
// pattern's preCollisionSlot(q, site, W) read, a convention-free physical
// quantity), so the comparison resolves differences the macroscopic surface
// (rho/vx/vy/vz) is blind to (non-hydrodynamic population modes) while
// staying pattern-agnostic.
//
// Scenario (mirrors sim_AMR/sim_AMR_channel.cu at --resolution 1 --max-level 1):
// - 64x16x16 domain, planar channel along x (periodic tangent y), inflow
//   x = 1 GEO_INFLOW_MOMENT (uniform profile, U_lb = 0.1), outflow x = 62
//   GEO_OUTFLOW_RIGHT_INTERP, walls z = 1 and z = 14 GEO_WALL, GEO_NOTHING
//   on z in {0, 15} and x in {0, 63} edge layers (sim_AMR_channel's
//   boundary idiom verbatim, including the face-generic fine-level
//   bounce-back wall machinery and the storage_overlap_z = 3 buffer);
// - two level-1 footprints: "1 2 6 6 4 4 4" (x-min halo row lies ON the
//   GEO_INFLOW_MOMENT plane x = 1 — the Sec. 7.3 wall-guard inflow-adjacent
//   registration class of sim_AMR_ball at R = 1, cp. aa_seed_report.md §6)
//   and "1 24 4 2 16 8 8" (the channel's own wall-attached slab: the z-min
//   halo row lies ON the bottom wall plane z = 1, the sim's byte-frozen mL1
//   region spec), so the scenario exercises the guarded {c = 0 ring, c = 1
//   frozen skin} source pair on an inflow-adjacent face AND on the x-face
//   fills' tangent z windows over the wall-attached slab;
// - uniform-at-rest initial condition (rho = 1, u = 0 on every block) with
//   the inflow BC developing the flow from t = 0, exactly like the sim — so
//   the interface content becomes non-stationary cycle by cycle (the
//   transient regime the mL1 bitwise lock's periodic fixture cannot
//   produce at a guarded face).
//
// The traces compare the init-time GEO map, the macroscopic channels and the
// decoded canonical DF frames (see below) level by level, cycle by cycle.
// Snapshot storage is held in the comparator process only; each runner
// produces one trace.

#include <string>
#include <vector>

// number of driven coarse cycles: the R = 1 inflow-adjacent birth fires at
// cycle 3 in production evidence (aa_seed_report.md §6 and the SP-5 era
// re-measurements), so 10 cycles cover the seed and the first feedback
// amplification (the mL1 lock horizon)
constexpr int AMR_CHANNEL_LOCK_CYCLES = 10;

// sentinel for canonical-frame slots whose preCollisionSlot target lies
// outside the block's stored extent (never read by any kernel; both
// patterns encode physical content only over in-range slots). NaN bit
// pattern chosen once so bitwise comparison of masked entries stays exact.
double amrChannelFrameMask();
bool amrChannelIsFrameMask(double v);

// one coarse cycle's snapshot of one block level: the GEO map (cycle 0 only
// — interface tagging is init-time), every macroscopic channel over the
// whole stored box [origin, origin + size) including the overlap rows, and
// the decoded canonical DF frame (27 values per cell: direction q's
// population gathered at the site by the next consuming launch's parity,
// amrChannelFrameMask() where the slot is out of storage)
struct AMRChannelLevelSnap
{
	int level = -1;
	int origin[3] = {0, 0, 0};
	int size[3] = {0, 0, 0};
	int macro_channels = 0;
	std::vector<int> map;		// size.x * size.y * size.z entries (cycle 0 only)
	std::vector<double> macro;	// macro_channels * cells entries, widened float (exact)
	std::vector<double> frame;	// 27 * cells canonical DF entries, widened float (exact, masked out-of-range)
};

struct AMRChannelCycle
{
	std::vector<AMRChannelLevelSnap> levels;  // ascending level order
};

struct AMRChannelTrace
{
	std::string pattern;
	bool init_ok = false;
	int cycles_completed = 0;
	std::vector<AMRChannelCycle> cycles;
	// launch-level attribution: per-phase snapshots inside cycles 1..3
	// (label, block snapshots at that phase point) — the instrument the
	// SP-5 deconstruction is built from
	std::vector<std::pair<std::string, AMRChannelCycle>> phases;
};

// scenario runners, one per pinned streaming pattern (test_amr_channel_lock_*.cu)
AMRChannelTrace runAMRChannelScenario_ab();
AMRChannelTrace runAMRChannelScenario_aa();
AMRChannelTrace runAMRChannelScenario_eso_twist();
AMRChannelTrace runAMRChannelScenario_eso_pull();
AMRChannelTrace runAMRChannelScenario_eso_push();
