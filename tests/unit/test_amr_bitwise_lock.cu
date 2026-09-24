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

#include <cmath>
#include <cstdint>
#include <cstring>

#include <fmt/format.h>

#include <doctest/doctest.h>

#include "amr_bitwise_lock.h"

TEST_SUITE_BEGIN("amr_bitwise_lock");

namespace
{

// per-pair comparison outcome: totals plus the first few differing sites
// (cycle, level, channel, coordinates, both values) for the failure report
struct AMRLockDiff
{
	long cells = 0;
	double max_abs = 0;
	int max_cycle = -1, max_level = -1, max_channel = -1;
	std::vector<std::string> first_sites;

	void record(double a, double b, const std::string& site, int cycle, int level, int channel)
	{
		cells++;
		const double d = std::abs(a - b);
		if (d > max_abs) {
			max_abs = d;
			max_cycle = cycle;
			max_level = level;
			max_channel = channel;
		}
		if (first_sites.size() < 5)
			first_sites.push_back(site);
	}
};

// bitwise compare of two double values that were widened from float (exact
// widening): bit equality of the doubles, with the float-ulp distance of the
// narrowed values for the report
bool sameBits(double a, double b)
{
	std::uint64_t ua, ub;
	std::memcpy(&ua, &a, 8);
	std::memcpy(&ub, &b, 8);
	return ua == ub;
}

long floatUlpDistance(double a, double b)
{
	const float fa = static_cast<float>(a);
	const float fb = static_cast<float>(b);
	std::int32_t ia, ib;
	std::memcpy(&ia, &fa, 4);
	std::memcpy(&ib, &fb, 4);
	return std::abs(static_cast<long>(ia) - static_cast<long>(ib));
}

AMRLockDiff compareTraces(const AMRLockTrace& ref, const AMRLockTrace& test, std::string& shape_error)
{
	AMRLockDiff diff;
	shape_error.clear();

	if (ref.cycles.size() != test.cycles.size()) {
		shape_error = fmt::format("cycle counts differ: {} (ref ab) vs {} ({})", ref.cycles.size(), test.cycles.size(), test.pattern);
		return diff;
	}
	for (std::size_t c = 0; c < ref.cycles.size(); c++) {
		const AMRLockCycle& rc = ref.cycles[c];
		const AMRLockCycle& tc = test.cycles[c];
		if (rc.levels.size() != tc.levels.size()) {
			shape_error = fmt::format("cycle {}: level counts differ: {} vs {}", c, rc.levels.size(), tc.levels.size());
			return diff;
		}
		for (std::size_t l = 0; l < rc.levels.size(); l++) {
			const AMRLockLevelSnap& rs = rc.levels[l];
			const AMRLockLevelSnap& ts = tc.levels[l];
			if (rs.level != ts.level || std::memcmp(rs.origin, ts.origin, sizeof rs.origin) != 0
				|| std::memcmp(rs.size, ts.size, sizeof rs.size) != 0 || rs.macro_channels != ts.macro_channels) {
				shape_error = fmt::format("cycle {}, level slot {}: block geometry differs (level {} vs {})", c, l, rs.level, ts.level);
				return diff;
			}
			const long cells_xyz = static_cast<long>(rs.size[0]) * rs.size[1] * rs.size[2];
			// init-time GEO map tagging (cycle 0 only: GEO_AMR_INTERFACE
			// ring, frozen footprint cells and the wall/NOTHING overlay)
			if (c == 0) {
				for (long i = 0; i < cells_xyz; i++)
					if (rs.map[i] != ts.map[i]) {
						const long x = i % rs.size[0] + rs.origin[0];
						const long y = (i / rs.size[0]) % rs.size[1] + rs.origin[1];
						const long z = (i / (rs.size[0] * rs.size[1])) + rs.origin[2];
						diff.record(
							rs.map[i],
							ts.map[i],
							fmt::format("map cycle 0 level {} cell ({} {} {}): ref {} vs {} {}", rs.level, x, y, z, rs.map[i], ts.map[i], test.pattern),
							0,
							rs.level,
							-1
						);
					}
			}
			for (int m = 0; m < rs.macro_channels; m++)
				for (long i = 0; i < cells_xyz; i++) {
					const double a = rs.macro[static_cast<std::size_t>(m) * cells_xyz + i];
					const double b = ts.macro[static_cast<std::size_t>(m) * cells_xyz + i];
					if (! sameBits(a, b)) {
						const long x = i % rs.size[0] + rs.origin[0];
						const long y = (i / rs.size[0]) % rs.size[1] + rs.origin[1];
						const long z = (i / (rs.size[0] * rs.size[1])) + rs.origin[2];
						diff.record(
							a,
							b,
							fmt::format(
								"macro cycle {} level {} channel {} cell ({} {} {}): ref {:.9e} vs {} {:.9e} ({} float ulps)",
								c,
								rs.level,
								m,
								x,
								y,
								z,
								a,
								test.pattern,
								b,
								floatUlpDistance(a, b)
							),
							static_cast<int>(c),
							rs.level,
							m
						);
					}
				}
		}
	}
	return diff;
}

}  // namespace

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
		const AMRLockDiff diff = compareTraces(ref, test, shape_error);
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
