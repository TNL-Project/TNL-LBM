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
// amr_bitwise_lock.h); the following surfaces are KNOWN DEFECTS of the
// single-array schedule with direct production evidence in
// results_drag_crisis/aa_seed_report.md and must NOT be locked until the
// underlying parity/vintage defect is repaired:
//
//  1. nested links (max_level >= 2): the mid-sync C2F fill launched inside
//     advancePair diverges under A-A (and, by the same schedule family, the
//     esoteric in-place patterns) — wrong frame/parity vintage authoring at
//     the nested-schedule fill (seed report §4 stage (iv), §8 item 1);
//  2. the R = 1 inflow-adjacent C2F wall-guard path: a second, independent
//     seed — constant-density offset on the L0 GEO_INFLOW_MOMENT plane plus
//     1-2 ulp band seeds (seed report §6);
//  3. mL0 under A-A: uniform max_level == 0 aborts at SimInit
//     ('-overlap <= i' NDArray bounds + CUDA ERROR 719; seed report §7).
TEST_CASE("cross-pattern AMR known defects (nested >= 2 links, R=1 inflow-adjacent wall guard, mL0 under A-A) — NOT locked, see results_drag_crisis/aa_seed_report.md" * doctest::skip(true))
{}

TEST_SUITE_END();
