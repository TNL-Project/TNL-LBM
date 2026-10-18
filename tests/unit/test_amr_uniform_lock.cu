// Comparator translation unit of the UNIFORM cross-pattern AMR bitwise
// lock: runs the five per-pattern scenario runners
// (test_amr_uniform_lock_*.cu, instantiated through the explicit
// D3Q27_STREAMING_* types) SEQUENTIALLY in one process — the State
// constructor registers a global spdlog logger per instance, so the
// pattern instances cannot coexist, exactly the sequential-snapshot idiom
// of the mL1 lock — asserts the GEO_NOTHING frame cleanliness of the
// sim_AMR_ball-uniform map on EVERY pattern (the SP-6 anatomy: a single
// stray GEO_WALL at the (0,0,0) frame silently survived A-B's clamped
// neighbor indexing but OOB-aborted the single-array patterns' unclamped
// ghost-layer gather), and asserts BITWISE equality of the map and the
// macroscopic channels against the A-B pull reference trace, cycle by
// cycle.
//
// The locked surface (the mL0 class: uniform max_level == 0, the driver
// falls back to the base State<NSE>::SimUpdate) and its evidence base are
// documented in amr_uniform_lock.h — this is the SP-6 surface of the
// out-of-scope survey, closed 2026-09-30 by the sim-side
// stampBallOnFineBlocks guard and permanently locked here. This TU
// instantiates no CUDA kernel: it links the five runner objects and works
// on their host-side traces only.

#include <fmt/format.h>

#include <doctest/doctest.h>

#include "amr_uniform_lock.h"

TEST_SUITE_BEGIN("amr_uniform_lock");

namespace
{

// value of D3Q27_BC_All::GEO_NOTHING as a plain int (the tag enum's
// numeric values are byte-frozen project-wide — bc.h appends new tags
// last to keep existing values stable, and the bit-identity manifest
// pins the map channels that carry them); the comparator TU includes no
// NSE instantiation, so the literal names the tag here
constexpr int GEO_NOTHING_TAG = 9;

}  // namespace

TEST_CASE("mL0 uniform single-block state: per-cycle bitwise lock of aa/eso_twist/eso_pull/eso_push against ab")
{
	// run the five pattern instances sequentially (they cannot coexist);
	// every scenario must complete SimInit plus AMR_UNIFORM_LOCK_CYCLES
	// cycles of the uniform fallback (the historical SP-6 abort fired on
	// the first odd-parity launch, so completion alone is the primary arm)
	AMRLockTrace traces[5] = {
		runAMRUniformLockScenario_ab(),
		runAMRUniformLockScenario_aa(),
		runAMRUniformLockScenario_eso_twist(),
		runAMRUniformLockScenario_eso_pull(),
		runAMRUniformLockScenario_eso_push(),
	};
	for (const auto& trace : traces) {
		CHECK_MESSAGE(
			trace.init_ok, fmt::format("AMR uniform lock: scenario '{}' failed to initialize (canCompute/SimInit)", trace.pattern)
		);
		CHECK_MESSAGE(
			trace.cycles_completed == AMR_UNIFORM_LOCK_CYCLES + 1,
			fmt::format(
				"AMR uniform lock: scenario '{}' completed {} cycle snapshots, expected {}",
				trace.pattern,
				trace.cycles_completed,
				AMR_UNIFORM_LOCK_CYCLES + 1
			)
		);
	}

	// SP-6 anatomy assertion: the six frame planes of the uniform map must
	// be GEO_NOTHING on EVERY pattern (the stray frame-adjacent GEO_WALL
	// cell of the seed report §7 is what the sim-side stamp guard removes;
	// the map is host-authored identically under all patterns, so one
	// check per trace documents the contract on each)
	for (const auto& trace : traces) {
		if (trace.cycles.empty() || trace.cycles[0].levels.empty())
			continue;
		const AMRLockLevelSnap& snap = trace.cycles[0].levels[0];
		const long cells_xyz = static_cast<long>(snap.size[0]) * snap.size[1] * snap.size[2];
		long frame_non_nothing = 0;
		for (long i = 0; i < cells_xyz; i++) {
			const long x = i % snap.size[0] + snap.origin[0];
			const long y = (i / snap.size[0]) % snap.size[1] + snap.origin[1];
			const long z = (i / (snap.size[0] * snap.size[1])) + snap.origin[2];
			const bool on_frame =
				x == 0 || x == snap.size[0] - 1 || y == 0 || y == snap.size[1] - 1 || z == 0 || z == snap.size[2] - 1;
			if (on_frame && snap.map[i] != GEO_NOTHING_TAG)
				frame_non_nothing++;
		}
		CHECK_MESSAGE(
			frame_non_nothing == 0,
			fmt::format(
				"AMR uniform lock: {}: {} frame-plane cells are not GEO_NOTHING (the ghost-layer idiom requires "
				"a clean GEO_NOTHING frame; the single-array patterns stream the frame unclamped)",
				trace.pattern,
				frame_non_nothing
			)
		);
	}

	const AMRLockTrace& ref = traces[0];
	if (! ref.init_ok || ref.cycles_completed != AMR_UNIFORM_LOCK_CYCLES + 1)
		return;

	// anti-triviality guard (mirrors the mL1/nested locks): the lock is
	// vacuous if the scenario ever regresses into a fixed point, so the
	// reference trace must actually evolve -- require non-bitwise change in
	// at least half of the compared macro entries between the SimInit
	// anchor and the final cycle
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
			"AMR uniform lock: reference scenario evolution sanity: {}/{} macro entries changed between cycle 0 and cycle {}\n",
			changed,
			total,
			AMR_UNIFORM_LOCK_CYCLES
		);
		CHECK_MESSAGE(
			changed * 2 >= total,
			fmt::format(
				"AMR uniform lock: the reference scenario barely evolves ({}/{} macro entries changed over {} cycles) "
				"-- the lock would be vacuous, investigate the harness instead of trusting the agreement",
				changed,
				total,
				AMR_UNIFORM_LOCK_CYCLES
			)
		);
	}

	bool all_exact = true;
	for (int p = 1; p < 5; p++) {
		const AMRLockTrace& test = traces[p];
		if (! test.init_ok || test.cycles_completed != AMR_UNIFORM_LOCK_CYCLES + 1) {
			all_exact = false;
			continue;
		}
		std::string shape_error;
		const AMRLockDiff diff = compareLockTraces(ref, test, shape_error);
		if (! shape_error.empty()) {
			all_exact = false;
			CHECK_MESSAGE(false, fmt::format("AMR uniform lock: {} vs ab: {}", test.pattern, shape_error));
			continue;
		}
		// unconditional summary line: the empirical per-pair evidence (the
		// lock assertion itself only proves the aggregate)
		fmt::print(
			"AMR uniform lock: {} vs ab: {} differing cells over {} cycles, max |diff| = {:.3e}{}{}\n",
			test.pattern,
			diff.cells,
			AMR_UNIFORM_LOCK_CYCLES + 1,
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
				"AMR uniform lock: {} vs ab: {} differing cells, max |diff| = {:.3e} at cycle {} level {} channel {}{}",
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
