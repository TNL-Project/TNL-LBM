// Comparator translation unit of the cross-pattern AMR bitwise lock: runs
// the five per-pattern scenario runners (test_amr_bitwise_lock_*.cu,
// instantiated through the explicit D3Q27_STREAMING_* types) SEQUENTIALLY
// in one process — the State constructor registers a global spdlog logger
// per instance, so the pattern instances cannot coexist, exactly the
// sequential-snapshot idiom of the subcycling suite's Test 3 — and asserts
// BITWISE equality of the map and the macroscopic channels against the A-B
// pull reference trace, level by level, cycle by cycle.
//
// The lock surface (mL1 only) and its evidence base are documented in
// amr_bitwise_lock.h; the KNOWN DEFECT surfaces that must stay outside the
// lock are listed in the skipped documentation case below. This TU
// instantiates no CUDA kernel: it links the five runner objects and works
// on their host-side traces only.

#include <fmt/format.h>

#include <doctest/doctest.h>

#include "amr_bitwise_lock.h"

TEST_SUITE_BEGIN("amr_bitwise_lock");

TEST_CASE("mL1 two-level state: per-cycle bitwise lock of aa/eso_twist/eso_pull/eso_push against ab")
{
	// run the five pattern instances sequentially (they cannot coexist);
	// every scenario must complete SimInit plus AMR_LOCK_CYCLES cycles
	AMRLockTrace traces[5] = {
		runAMRLockScenario_ab(),
		runAMRLockScenario_aa(),
		runAMRLockScenario_eso_twist(),
		runAMRLockScenario_eso_pull(),
		runAMRLockScenario_eso_push(),
	};
	for (const auto& trace : traces) {
		CHECK_MESSAGE(trace.init_ok, fmt::format("AMR bitwise lock: scenario '{}' failed to initialize (canCompute/SimInit)", trace.pattern));
		CHECK_MESSAGE(
			trace.cycles_completed == AMR_LOCK_CYCLES + 1,
			fmt::format("AMR bitwise lock: scenario '{}' completed {} cycle snapshots, expected {}", trace.pattern, trace.cycles_completed, AMR_LOCK_CYCLES + 1)
		);
	}
	const AMRLockTrace& ref = traces[0];
	if (! ref.init_ok || ref.cycles_completed != AMR_LOCK_CYCLES + 1)
		return;

	// anti-triviality guard: the lock is vacuous if the scenario ever
	// regresses into a fixed point (a no-op SimUpdate would pass pattern
	// agreement silently), so the reference trace must actually evolve --
	// require non-bitwise change in at least half of the compared macro
	// entries between the SimInit anchor and the final cycle (measured:
	// 32563/39712 = 82% at cycle 10; every kernel-integrated cell of both
	// levels evolves, the static remainder is the fill-only outer ghost
	// rows of the fine block)
	{
		long changed = 0, total = 0;
		const AMRLockCycle& first = ref.cycles.front();
		const AMRLockCycle& last = ref.cycles.back();
		for (std::size_t l = 0; l < first.levels.size(); l++)
			for (std::size_t i = 0; i < first.levels[l].macro.size(); i++) {
				total++;
				if (! sameBits(first.levels[l].macro[i], last.levels[l].macro[i]))
					changed++;
			}
		fmt::print(
			"AMR bitwise lock: reference scenario evolution sanity: {}/{} macro entries changed between cycle 0 and cycle {}\n",
			changed,
			total,
			AMR_LOCK_CYCLES
		);
		CHECK_MESSAGE(
			changed * 2 >= total,
			fmt::format(
				"AMR bitwise lock: the reference scenario barely evolves ({}/{} macro entries changed over {} cycles) "
				"-- the lock would be vacuous, investigate the harness instead of trusting the agreement",
				changed,
				total,
				AMR_LOCK_CYCLES
			)
		);
	}

	bool all_exact = true;
	for (int p = 1; p < 5; p++) {
		const AMRLockTrace& test = traces[p];
		if (! test.init_ok || test.cycles_completed != AMR_LOCK_CYCLES + 1) {
			all_exact = false;
			continue;
		}
		std::string shape_error;
		const AMRLockDiff diff = compareLockTraces(ref, test, shape_error);
		if (! shape_error.empty()) {
			all_exact = false;
			CHECK_MESSAGE(false, fmt::format("AMR bitwise lock: {} vs ab: {}", test.pattern, shape_error));
			continue;
		}
		// unconditional summary line: the empirical per-pair evidence (the
		// lock assertion itself only proves the aggregate)
		fmt::print(
			"AMR bitwise lock: {} vs ab: {} differing cells over {} cycles, max |diff| = {:.3e}{}{}\n",
			test.pattern,
			diff.cells,
			AMR_LOCK_CYCLES + 1,
			diff.max_abs,
			diff.cells > 0 ? fmt::format(" (first at cycle {} level {} channel {})", diff.max_cycle, diff.max_level, diff.max_channel) : "",
			diff.cells > 0 ? "" : " -- bitwise identical"
		);
		std::string detail;
		for (const auto& site : diff.first_sites)
			detail += "\n    " + site;
		CHECK_MESSAGE(
			diff.cells == 0,
			fmt::format(
				"AMR bitwise lock: {} vs ab: {} differing cells, max |diff| = {:.3e} at cycle {} level {} channel {}{}",
				test.pattern,
				diff.cells,
				diff.max_abs,
				diff.max_cycle,
				diff.max_level,
				diff.max_channel,
				detail
			)
		);
		all_exact = all_exact && diff.cells == 0;
	}
	CHECK(all_exact);
}

// DOCUMENTATION ARM — never executed (decorated skip). The cross-pattern
// bitwise lock deliberately covers ONLY the mL1 surface (see
// amr_bitwise_lock.h); the following surfaces are documented known-defect
// or deferred families of the single-array schedule with direct production
// evidence in results_drag_crisis/aa_seed_report.md:
//
//  the R = 1 inflow-adjacent C2F wall-guard path: constant-density
//  offset on the L0 GEO_INFLOW_MOMENT plane plus 1-2 ulp band seeds
//  (seed report §6) — SP-5 closed 2026-09-29 as a DOCUMENTED PARTITION
//  VERDICT (structurally unattainable without regressing the pinned AB
//  reference), permanently gated by the test_amr_channel_lock
//  classification contract, NOT by this lock.
//
// The nested (max_level >= 2) surface is CLOSED (SP-4, 2026-09-29): the
// nested cycle-end cascade's C2F fill authors its band from a mid-sync
// stashed snapshot of the source level's post-substep-A DF box (re-running
// the unmodified fill kernel on the stashed vintage instead of re-reading
// the dead single-array frame — State_AMR::stashC2FSource /
// swapC2FSource / restoreC2FSource in amr_state.h), and is permanently
// LOCKED by the sibling nested lock binary test_amr_nested_lock (see
// tests/unit/test_amr_nested_lock.py) — bitwise on all five patterns in
// both trees.
//
// The mL0 uniform surface is CLOSED (SP-6, 2026-09-30): the seed was
// sim-side, not schedular — sim_AMR_ball's stampBallOnFineBlocks level
// filter admitted the level-0 block at max_level == 0 and stamped one
// stray GEO_WALL at the (0,0,0) frame through its zero-initialized
// lat_local, which A-B's clamped neighbor indexing silently tolerated and
// the single-array patterns' unclamped ghost-layer gather read out of
// storage ('-overlap <= i' NDArray bounds + CUDA ERROR 719; seed report
// §7). The sim-side guard (max_level == 0 skips the fine stamp) repairs
// it, and the repaired uniform fallback is permanently LOCKED by the
// sibling uniform lock binary test_amr_uniform_lock (see
// tests/unit/test_amr_uniform_lock.py): bitwise on all five patterns in
// both trees, with a GEO_NOTHING frame-cleanliness anatomy assertion.
TEST_CASE("cross-pattern AMR known defects (R=1 inflow-adjacent wall guard) — NOT locked by this binary, see results_drag_crisis/aa_seed_report.md" * doctest::skip(true))
{}

TEST_SUITE_END();
