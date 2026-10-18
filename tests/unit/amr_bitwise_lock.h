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

#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include <fmt/format.h>

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

// ---- shared comparator helpers (host-only, pattern-independent): the
// mL1/nested/uniform lock comparators all reduce AMRLockTrace pairs with
// the same bitwise diff collector and the same trace walk

// per-pair comparison outcome: totals plus the first few differing sites
// (cycle, level, channel, coordinates, both values) for the failure report;
// first_cycle/first_level anchor the earliest (and deepest-level) divergence
struct AMRLockDiff
{
	long cells = 0;
	double max_abs = 0;
	int max_cycle = -1, max_level = -1, max_channel = -1;
	int first_cycle = -1, first_level = -1;
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
		if (first_cycle < 0 || cycle < first_cycle || (cycle == first_cycle && level > first_level)) {
			first_cycle = cycle;
			first_level = level;
		}
		if (first_sites.size() < 5)
			first_sites.push_back(site);
	}
};

// bitwise compare of two double values that were widened from float (exact
// widening): bit equality of the doubles, with the float-ulp distance of
// the narrowed values for the report
inline bool sameBits(double a, double b)
{
	std::uint64_t ua, ub;
	std::memcpy(&ua, &a, 8);
	std::memcpy(&ub, &b, 8);
	return ua == ub;
}

inline long floatUlpDistance(double a, double b)
{
	const float fa = static_cast<float>(a);
	const float fb = static_cast<float>(b);
	std::int32_t ia, ib;
	std::memcpy(&ia, &fa, 4);
	std::memcpy(&ib, &fb, 4);
	return std::abs(static_cast<long>(ia) - static_cast<long>(ib));
}

// walk two traces of the same scenario (ref = the A-B pull reference) and
// record every bitwise difference of the init-time GEO map (cycle 0 only)
// and of all macroscopic channels over the whole stored box; shape
// mismatches (cycle/level counts, block geometry) abort with a
// shape_error instead of a diff
inline AMRLockDiff compareLockTraces(const AMRLockTrace& ref, const AMRLockTrace& test, std::string& shape_error)
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
			if (rs.level != ts.level || std::memcmp(rs.origin, ts.origin, sizeof rs.origin) != 0 || std::memcmp(rs.size, ts.size, sizeof rs.size) != 0
				|| rs.macro_channels != ts.macro_channels)
			{
				shape_error = fmt::format("cycle {}, level slot {}: block geometry differs (level {} vs {})", c, l, rs.level, ts.level);
				return diff;
			}
			const long cells_xyz = static_cast<long>(rs.size[0]) * rs.size[1] * rs.size[2];
			// init-time GEO map tagging (cycle 0 only: the scenario's
			// init-time marks -- interface ring, frozen cells, BC planes
			// and the GEO_NOTHING frame)
			if (c == 0) {
				for (long i = 0; i < cells_xyz; i++)
					if (rs.map[i] != ts.map[i]) {
						const long x = i % rs.size[0] + rs.origin[0];
						const long y = (i / rs.size[0]) % rs.size[1] + rs.origin[1];
						const long z = (i / (rs.size[0] * rs.size[1])) + rs.origin[2];
						diff.record(
							rs.map[i],
							ts.map[i],
							fmt::format(
								"map cycle 0 level {} cell ({} {} {}): ref {} vs {} {}", rs.level, x, y, z, rs.map[i], ts.map[i], test.pattern
							),
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
