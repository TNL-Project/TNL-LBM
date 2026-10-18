// Comparator translation unit of the NESTED cross-pattern AMR bitwise
// lock: runs the five per-pattern scenario runners
// (test_amr_nested_lock_*.cu, instantiated through the explicit
// D3Q27_STREAMING_* types) SEQUENTIALLY in one process — the State
// constructor registers a global spdlog logger per instance, so the
// pattern instances cannot coexist, exactly the sequential-snapshot idiom
// of the mL1 lock — and asserts BITWISE equality of the map and the
// macroscopic channels against the A-B pull reference trace, level by
// level, cycle by cycle.
//
// The locked surface (the mL2 class: one nested L1->L2 link on top of the
// mL1 fixture, exercising the mid-sync C2F fill inside advancePair that
// exists only at nesting depth >= 2) and its evidence base are documented
// in amr_nested_lock.h — this is the SP-4 surface of the out-of-scope
// survey, delinquently excluded from the mL1 lock until the parity/vintage
// defect at the nested-schedule fill is repaired. This TU instantiates no
// CUDA kernel: it links the five runner objects and works on their
// host-side traces only.

#include <fmt/format.h>

#include <doctest/doctest.h>

#include "amr_nested_lock.h"

TEST_SUITE_BEGIN("amr_nested_lock");

TEST_CASE("mL2 three-level state: per-cycle bitwise lock of aa/eso_twist/eso_pull/eso_push against ab")
{
	// run the five pattern instances sequentially (they cannot coexist);
	// every scenario must complete SimInit plus AMR_NESTED_LOCK_CYCLES cycles
	AMRLockTrace traces[5] = {
		runAMRNestedLockScenario_ab(),
		runAMRNestedLockScenario_aa(),
		runAMRNestedLockScenario_eso_twist(),
		runAMRNestedLockScenario_eso_pull(),
		runAMRNestedLockScenario_eso_push(),
	};
	for (const auto& trace : traces) {
		CHECK_MESSAGE(
			trace.init_ok, fmt::format("AMR nested lock: scenario '{}' failed to initialize (canCompute/SimInit)", trace.pattern)
		);
		CHECK_MESSAGE(
			trace.cycles_completed == AMR_NESTED_LOCK_CYCLES + 1,
			fmt::format(
				"AMR nested lock: scenario '{}' completed {} cycle snapshots, expected {}", trace.pattern, trace.cycles_completed,
				AMR_NESTED_LOCK_CYCLES + 1
			)
		);
	}
	const AMRLockTrace& ref = traces[0];
	if (! ref.init_ok || ref.cycles_completed != AMR_NESTED_LOCK_CYCLES + 1)
		return;

	// anti-triviality guard (mirrors the mL1 lock): the lock is vacuous if
	// the scenario ever regresses into a fixed point (a no-op SimUpdate
	// would pass pattern agreement silently), so the reference trace must
	// actually evolve -- require non-bitwise change in at least half of
	// the compared macro entries between the SimInit anchor and the final
	// cycle
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
			"AMR nested lock: reference scenario evolution sanity: {}/{} macro entries changed between cycle 0 and cycle {}\n",
			changed,
			total,
			AMR_NESTED_LOCK_CYCLES
		);
		CHECK_MESSAGE(
			changed * 2 >= total,
			fmt::format(
				"AMR nested lock: the reference scenario barely evolves ({}/{} macro entries changed over {} cycles) "
				"-- the lock would be vacuous, investigate the harness instead of trusting the agreement",
				changed,
				total,
				AMR_NESTED_LOCK_CYCLES
			)
		);
	}

	bool all_exact = true;
	for (int p = 1; p < 5; p++) {
		const AMRLockTrace& test = traces[p];
		if (! test.init_ok || test.cycles_completed != AMR_NESTED_LOCK_CYCLES + 1) {
			all_exact = false;
			continue;
		}
		std::string shape_error;
		const AMRLockDiff diff = compareLockTraces(ref, test, shape_error);
		if (! shape_error.empty()) {
			all_exact = false;
			CHECK_MESSAGE(false, fmt::format("AMR nested lock: {} vs ab: {}", test.pattern, shape_error));
			continue;
		}
		// unconditional summary line: the empirical per-pair evidence (the
		// lock assertion itself only proves the aggregate); the birth
		// anchor (first divergent cycle/level) is the documented SP-4
		// signature -- first divergence at the L1-frozen band under L2 and
		// the L2 band rows, ulp-class (aa_seed_report.md §3)
		fmt::print(
			"AMR nested lock: {} vs ab: {} differing cells over {} cycles, max |diff| = {:.3e}{}{}\n",
			test.pattern,
			diff.cells,
			AMR_NESTED_LOCK_CYCLES + 1,
			diff.max_abs,
			diff.cells > 0
				? fmt::format(
					  " (max at cycle {} level {} channel {}; first at cycle {} level {})",
					  diff.max_cycle,
					  diff.max_level,
					  diff.max_channel,
					  diff.first_cycle,
					  diff.first_level
				  )
				: "",
			diff.cells > 0 ? "" : " -- bitwise identical"
		);
		std::string detail;
		for (const auto& site : diff.first_sites)
			detail += "\n    " + site;
		CHECK_MESSAGE(
			diff.cells == 0,
			fmt::format(
				"AMR nested lock: {} vs ab: {} differing cells, max |diff| = {:.3e} at cycle {} level {} channel {}{}",
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

TEST_SUITE_END();
