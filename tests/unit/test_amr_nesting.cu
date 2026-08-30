// Unit tests for the multi-level nesting validation in createAMRBlocks
// (include/lbm3d/amr_decomposition.h, the amr-nlevel-nesting plan's V-suite
// replacing the level > 1 reject) and for block creation on 3-level nested
// region chains, and for the advancePair schedule recursion of State_AMR
// (include/lbm3d/amr_state.h) running SimUpdate on nested configurations.
//
// - test_vsuite_reject_corpus: one createAMRBlocks call per invalid class of
//   the V-suite (V5 ascending file order, V6 parent existence/uniqueness,
//   V7 telescoping gap, V8 sibling separation, V9 origin positivity,
//   V10 wall-shared chain), each constructed from regions alone, expecting a
//   std::runtime_error with the rule-specific message -- the FULL message
//   text (the createAMRBlocks reject envelope plus the reason) is locked
//   verbatim, mirroring test_footprint_min_size_validation's wording lock in
//   test_amr_subcycling.cu. Every rejection must happen in the read-only
//   phase (no partial block creation).
//   The corpus also covers the generalized frame math of the pre-existing
//   V2/V3 checks at level >= 2 (footprint gs >= 3 in PARENT-level cells,
//   footprint contained in the parent-level global lattice), which reduce
//   to the historical level-1 formulas at level 1.
// - test_vsuite_gap2_warning / test_vsuite_sep2_warning: the user-decided
//   advisory tier (plan sec. 11 item 2): gap exactly 2 parent-level cells is
//   VALID but warns (coupling-authored cells enter the transfer windows);
//   both cases must be ACCEPTED and still create the blocks.
// - test_three_level_creation: a valid telescoping 3-level chain (z-min
//   wall-shared chain, y span pinned to the parent's footprint faces)
//   validating the whole V-suite chain top-down: block census, per-level
//   lat_local (physDt halving, viscosity doubling), and the EXACT
//   parent-frame global_offset / fine offset / local conversion values of
//   every fine block, with exactly zero warnings emitted (the no-new-warnings
//   invariant of the regression contract).
// - test_three_level_mark_census: markAMRInterface on the same 3-level
//   chain: per-level GEO_AMR_INTERFACE / GEO_NOTHING censuses equal to an
//   independent host-side enumeration of the contract band tag rule,
//   full-map equality per parent block against the same reference, ring-free
//   tagging on the wall-shared faces (halo rows land in the parent block's
//   ghost zone and must be clipped away), and re-invocation idempotency.
// - test_two_level_schedule_census / test_three_level_schedule_census: the
//   advancePair schedule recursion (amr-nlevel-nesting commit C) observed
//   through the StateSchedule_AMR spy of the shared fixture on nested
//   mocks: the per-cycle event expansion of the plan's sec. 3.4 table at
//   max_level == 2 and its one-level-deeper expansion at max_level == 3,
//   with per-event rotation locks (AB df-pointer identity / AA even_iter),
//   the {true, false} next-substep parity alternation at the mid/end f2c
//   sites, and the cumulative per-level substep counters 2^L per cycle.
// - test_three_level_conservation_smoke: plotting-free 20-cycle run of the
//   3-level chain: conservation stats stay finite (no NaN) and internally
//   consistent with the GEO_NOTHING-excluding reference, with the global
//   mass stable.
// - test_wall_chain_masks / test_wall_pedestal_prisms /
//   test_wall_chain_failfast / test_wall_chain_lagrava_guard: the commit-E
//   wall chain (plan sec. 5 Tests 12--14 + the sec. 5.4 strategy guard): the
//   fine wall masks derive from the immediate parent's map at every level
//   (the 3-level all-z-min wall-shared stack), the launch windows deepen to
//   the wall row on every level, the R4 wall-pedestal prisms author the deep
//   frozen rows behind the parent's upward fine-to-coarse window, the three
//   silent-lane misconfigurations throw named errors, and the F2C_LAGRAVA
//   opt-out is hard-guarded against nested wall-sharing.
// - test_five_level_channel_chain_creation: the commit-F 5-level channel
//   chain (sim_AMR/amr_chain_solver.h's derivation at R = 1, locked here as
//   the five_level_channel_chain fixture constant): createAMRBlocks accepts
//   the derived chain on the 64 x 16 x 16 channel lattice, blocks 0..4 with
//   one block per level, exact parent-frame global_offset / fine offset /
//   fine local per level, zero advisory warnings.
// - test_windbreak_rod_census: the commit-G windbreak rod array
//   (sim_AMR/amr_windbreak.h's layout and stamping, the same helper
//   sim_AMR_channel rides) on the 5-level chain's level-4 block: the
//   staggered two-row census matches the analytic cross-section integral
//   (per-layer disc cells times the rod height), the array keeps its face
//   clearances, no face wall row is touched (the buildFineWallMasks census
//   is unchanged), no parent block sees a rod cell, and the layout
//   guardrails reject the forbidden classes with named errors.
//
// The streaming pattern is selected at compile time (AB_PATTERN/AA_PATTERN);
// this suite is compiled into the consolidated doctest binaries
// test_amr_units_{ab,aa} (tests/unit/CMakeLists.txt), which provide main().
// Everything is single-rank. Shared fixture machinery (lattice factory,
// report) comes from tests/unit/amr_test_fixture.h.

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include <spdlog/sinks/base_sink.h>
#include <spdlog/spdlog.h>

#include "amr_test_fixture.h"
#include "../../sim_AMR/amr_windbreak.h"

using BC = typename NSE_CONFIG::BC;

// captures the messages reaching the default spdlog logger during one
// createAMRBlocks call (installed AFTER the State construction: the State
// constructor registers its own global logger); createAMRBlocks logs
// advisory warnings (never destroys state) and reject errors (then throws)
struct LogCapture
{
	struct VecSink final : spdlog::sinks::base_sink<std::mutex>
	{
		std::vector<std::string> warnings;

		void sink_it_(const spdlog::details::log_msg& msg) override
		{
			if (msg.level != spdlog::level::warn)
				return;
			std::string line(msg.payload.data(), msg.payload.size());
			// keep only createAMRBlocks' V-suite advisory messages: block
			// creation and lattice setup emit their own pre-existing
			// heuristic warnings (the CUDA block-size domain checks), which
			// are not part of the V-suite channel
			if (line.rfind("createAMRBlocks:", 0) == 0)
				warnings.push_back(std::move(line));
		}
		void flush_() override {}
	};

	std::shared_ptr<VecSink> sink = std::make_shared<VecSink>();

	LogCapture()
	{
		spdlog::default_logger()->sinks().push_back(sink);
	}

	LogCapture(const LogCapture&) = delete;
	LogCapture& operator=(const LogCapture&) = delete;

	~LogCapture()
	{
		auto& sinks = spdlog::default_logger()->sinks();
		sinks.erase(std::remove(sinks.begin(), sinks.end(), sink), sinks.end());
	}

	bool hasWarning(const std::string& substring) const
	{
		for (const std::string& line : sink->warnings)
			if (line.find(substring) != std::string::npos)
				return true;
		return false;
	}
};

// one reject-corpus row: a name, a region file, and the FULL expected
// rejection message (envelope + reason) -- fmt::format-style placeholders of
// the envelope are pre-computed per row, the reason text is the wording lock
struct RejectCase
{
	const char* name;
	const char* config;
	std::string expected_message;
};

TEST_SUITE_BEGIN("amr_nesting");

void test_vsuite_reject_corpus()
{
	// envelope of createAMRBlocks' reject lambda (locked by Test 7's wording
	// lock in test_amr_subcycling.cu for the pre-existing V1-V4 checks); the
	// reason slots are the per-rule wording locks of the V-suite below
	const auto envelope = [](int i, int level, const char* origin, const char* size, const std::string& reason)
	{
		return fmt::format(
			"createAMRBlocks: invalid region #{} (level {}, origin [{}], size [{}]): {}", i, level, origin, size, reason
		);
	};

	const std::vector<RejectCase> cases = {
		// V9 (origin positivity): a level-1 region whose parent-frame origin
		// is 0 would put the block's interface halo row outside the fine
		// lattice storage under the re-anchored band registration
		{"v9_origin_x",
		 "1 0 4 4 8 8 8",
		 envelope(
			 0,
			 1,
			 "0,4,4",
			 "8,8,8",
			 "footprint origin resolves to 0 parent-level cells on axis X (must be at least 1): the interface halo row one parent "
			 "cell outside the footprint would lie outside the parent lattice storage"
		 )},
		// V9 on two further axes for face coverage of the loop
		{"v9_origin_y",
		 "1 4 0 4 8 8 8",
		 envelope(
			 0,
			 1,
			 "4,0,4",
			 "8,8,8",
			 "footprint origin resolves to 0 parent-level cells on axis Y (must be at least 1): the interface halo row one parent "
			 "cell outside the footprint would lie outside the parent lattice storage"
		 )},
		// V5 (ascending file order): the unique containing parent is listed
		// AFTER its level-2 child
		{"v5_order",
		 "2 38 4 16 16 120 16\n1 8 1 4 12 30 12",
		 envelope(
			 0,
			 2,
			 "38,4,16",
			 "16,120,16",
			 "the unique containing level-1 region #1 appears later in the config; a level-2 region's parent must be listed "
			 "earlier so that blocks are created level-ascending"
		 )},
		// V6 (orphan): no level-1 region contains the level-2 footprint
		{"v6_orphan",
		 "1 8 8 8 8 8 8\n2 68 68 68 8 8 8",
		 envelope(
			 1,
			 2,
			 "68,68,68",
			 "8,8,8",
			 "no level-1 region fully contains this footprint (nested refinement requires exactly one containing parent region)"
		 )},
		// V6 (ambiguous parent): two overlapping level-1 regions both contain
		// the level-2 footprint (the child is validated before the offenders'
		// own sibling violation at region #2)
		{"v6_ambiguous",
		 "1 4 4 4 12 12 12\n2 40 40 40 8 8 8\n1 6 6 6 12 12 12",
		 envelope(
			 1,
			 2,
			 "40,40,40",
			 "8,8,8",
			 "footprint is fully contained in 2 level-1 regions (#0, #2); nested refinement requires exactly one containing "
			 "parent region"
		 )},
		// V7 (telescoping gap): all six faces inset exactly 1 parent-level
		// cell from the parent footprint; x-min is the first face checked
		{"v7_gap",
		 "1 4 4 4 12 12 12\n2 18 18 18 8 8 8",
		 envelope(
			 1,
			 2,
			 "18,18,18",
			 "8,8,8",
			 "telescoping gap below the 2-parent-cell minimum on the x-min face (got 1 parent-level cells; a non-wall face must "
			 "sit at least 2 parent cells inside the parent footprint, a wall-shared face must align exactly with the parent's "
			 "footprint edge)"
		 )},
		// V8 (sibling separation): two level-1 footprints 1 parent-level cell
		// apart along x (one footprint's halo would reach into the other)
		{"v8_siblings",
		 "1 4 4 4 6 6 6\n1 11 4 4 6 6 6",
		 envelope(
			 1,
			 1,
			 "11,4,4",
			 "6,6,6",
			 "same-level footprints must be separated by at least 2 parent-level cells (Chebyshev separation to level-1 region "
			 "#0 is 1 parent-level cells; one footprint's interface halo must not reach into the other footprint)"
		 )},
		// V10 (wall-shared chain): the level-3 z-min face aligns with its
		// parent's footprint edge, but the level-2 parent's z-min face is
		// interior (inset 3 from the level-1 edge), so the gap-0 alignment is
		// not chain-backed
		{"v10_chain",
		 "1 4 4 4 12 12 12\n2 22 22 22 20 20 20\n3 100 100 88 12 12 12",
		 envelope(
			 2,
			 3,
			 "100,100,88",
			 "12,12,12",
			 "z-min face aligns with the parent's footprint edge (wall-shared candidate) but parent region #1's z-min face is "
			 "inset 3 parent-level cells from its own parent; gap-0 alignment is legal only down a chain of wall-shared faces "
			 "reaching level 1"
		 )},
		// V2 generalized: a level-2 footprint thinner than 3 PARENT-level
		// cells on x (file size 2 resolves to gs 1 in the parent frame) --
		// same reason wording as the level-1 minimum, parent-frame value
		{"v2_thin_l2",
		 "1 4 4 4 12 12 12\n2 20 20 20 2 8 8",
		 envelope(
			 1,
			 2,
			 "20,20,20",
			 "2,8,8",
			 "AMR footprint size below the 3-parent-cell minimum required by the interface band structure (distinct c=0 ring "
			 "and c=1 destination rows) on axis X (got 1)"
		 )},
		// V3 generalized: the level-2 footprint extends beyond the
		// parent-level global lattice
		{"v3_domain_l2",
		 "1 4 4 4 8 8 8\n2 124 4 4 16 8 8",
		 envelope(
			 1,
			 2,
			 "124,4,4",
			 "16,8,8",
			 "region extends beyond the global coarsest-level domain [32,32,32]"
		 )},
	};

	for (const RejectCase& kase : cases) {
		lat_t lat = makeLattice(32);
		const std::string id = fmt::format("test_amr_nesting_{}_{}", pattern_name, kase.name);
		StateLocal_AMR<NSE_CONFIG> state(id, MPI_COMM_WORLD, lat, "adios2.xml", /*periodic=*/TRAITS::bool3d{true, true, true}, /*max_level=*/3);
		if (! state.canCompute()) {
			CHECK_MESSAGE(false, fmt::format("V-suite corpus {} setup: state.canCompute()", kase.name));
			return;
		}

		LogCapture capture;
		std::string message;
		try {
			createAMRBlocks(state.nse, parseAMRConfig<NSE_CONFIG>(kase.config));
		}
		catch (const std::runtime_error& e) {
			message = e.what();
		}
		CHECK_MESSAGE(
			message == kase.expected_message,
			fmt::format(
				"V-suite corpus {}: rejected with the rule-specific message verbatim ({})",
				kase.name,
				message.empty() ? "no exception thrown" : fmt::format("threw: {}", message)
			)
		);
		CHECK_MESSAGE(
			state.nse.blocks.size() == 1,
			fmt::format(
				"V-suite corpus {}: rejection happened in the read-only validation phase ({} blocks, expected the level-0 block only)",
				kase.name,
				state.nse.blocks.size()
			)
		);
	}
}

// the valid 3-level telescoping chain: z-min wall-shared chain (every level's
// z-min footprint face aligns with its parent's edge, chaining to the
// level-1 face that keeps the sim-side wall contract), y span pinned
// gap-0 to the parent's footprint faces, x telescoping with gaps >= 3
// parent-level cells (no advisory warnings anywhere); level-0 domain 32^3
constexpr const char* three_level_chain = "1 8 1 4 12 30 12\n"
										  "2 38 4 16 16 120 16\n"
										  "3 164 16 64 16 480 16";

// conservation-smoke chain: telescoping with gaps >= 3 on EVERY face, so all
// six coarse-to-fine destination bands of every level receive a fill (the
// commit-B chain above shares wall-candidate faces whose destinations clip
// empty against the parent interior pre-wall-machinery -- those bands keep
// their all-zero initial state and sink rho to NaN in a few cycles, a
// geometry-bound effect unrelated to the nested schedule the smoke must pin)
constexpr const char* three_level_interior_chain = "1 8 8 8 12 12 12\n"
												   "2 40 40 40 16 16 16\n"
												   "3 176 176 176 32 32 32";

void test_vsuite_gap2_warning()
{
	lat_t lat = makeLattice(32);
	const std::string id = fmt::format("test_amr_nesting_{}_gap2warn", pattern_name);
	StateLocal_AMR<NSE_CONFIG> state(id, MPI_COMM_WORLD, lat, "adios2.xml", /*periodic=*/TRAITS::bool3d{true, true, true}, /*max_level=*/2);
	if (! state.canCompute()) {
		CHECK_MESSAGE(false, "V-suite gap-2 warning setup: state.canCompute()");
		return;
	}

	// every min face inset exactly 2 parent-level cells: VALID (the floor is
	// inclusive) but below the recommended 3 -- one advisory warning per
	// affected face, blocks still created
	LogCapture capture;
	std::string message;
	try {
		createAMRBlocks(state.nse, parseAMRConfig<NSE_CONFIG>("1 4 4 4 12 12 12\n2 20 20 20 8 8 8"));
	}
	catch (const std::runtime_error& e) {
		message = e.what();
	}
	CHECK_MESSAGE((
		message.empty() && state.nse.blocks.size() == 3),
		fmt::format(
			"V-suite gap-2 floor: the footprint inset exactly 2 parent-level cells is accepted and the block is created ({})",
			message.empty() ? "no exception" : fmt::format("threw: {}", message)
		)
	);
	CHECK_MESSAGE((
		capture.sink->warnings.size() == 3
			&& capture.hasWarning("telescoping gap of 2 parent-level cells on the x-min face is below the recommended 3")
			&& capture.hasWarning("the parent's fine-to-coarse transfer windows will read coupling-authored ring/skin cells")),
		fmt::format(
			"V-suite gap-2 floor: one advisory warning per 2-cell face emitted ({} warnings) naming the face and the",
			capture.sink->warnings.size()
		)
	);
}

void test_vsuite_sep2_warning()
{
	lat_t lat = makeLattice(32);
	const std::string id = fmt::format("test_amr_nesting_{}_sep2warn", pattern_name);
	StateLocal_AMR<NSE_CONFIG> state(id, MPI_COMM_WORLD, lat, "adios2.xml", /*periodic=*/TRAITS::bool3d{true, true, true}, /*max_level=*/1);
	if (! state.canCompute()) {
		CHECK_MESSAGE(false, "V-suite separation-2 warning setup: state.canCompute()");
		return;
	}

	// two sibling footprints exactly 2 parent-level cells apart along x:
	// VALID (inclusive floor) but below the recommended 3
	LogCapture capture;
	std::string message;
	try {
		createAMRBlocks(state.nse, parseAMRConfig<NSE_CONFIG>("1 4 4 4 6 6 6\n1 12 4 4 6 6 6"));
	}
	catch (const std::runtime_error& e) {
		message = e.what();
	}
	CHECK_MESSAGE((
		message.empty() && state.nse.blocks.size() == 3),
		fmt::format(
			"V-suite separation floor: siblings exactly 2 parent-level cells apart are accepted and both blocks are created ({})",
			message.empty() ? "no exception" : fmt::format("threw: {}", message)
		)
	);
	CHECK_MESSAGE((
		capture.sink->warnings.size() == 1
			&& capture.hasWarning("separated by exactly 2 parent-level cells")
			&& capture.hasWarning("below the recommended 3")),
		fmt::format(
			"V-suite separation floor: one advisory warning emitted for the 2-cell sibling separation ({})",
			capture.sink->warnings.size()
		)
	);
}

// hand-computed block geometry of the three_level_chain (the creation
// asserts): block id, level, parent-frame global_offset, fine offset, fine
// local -- amrParentFrameOrigin / amrFineOffset / amrFineLocal per component
struct ChainBlockExpectation
{
	int level;
	idx3d global_offset;
	idx3d offset;
	idx3d local;
};

void test_three_level_creation()
{
	lat_t lat = makeLattice(32);
	const std::string id = fmt::format("test_amr_nesting_{}_chain", pattern_name);
	StateLocal_AMR<NSE_CONFIG> state(id, MPI_COMM_WORLD, lat, "adios2.xml", /*periodic=*/TRAITS::bool3d{true, true, true}, /*max_level=*/3);
	if (! state.canCompute()) {
		CHECK_MESSAGE(false, "3-level creation setup: state.canCompute()");
		return;
	}

	LogCapture capture;
	std::string message;
	try {
		createAMRBlocks(state.nse, parseAMRConfig<NSE_CONFIG>(three_level_chain));
	}
	catch (const std::runtime_error& e) {
		message = e.what();
	}
	CHECK_MESSAGE(
		message.empty(),
		fmt::format(
			"3-level creation: the telescoping chain passes the whole V-suite (the level > 1 reject is superseded) ({})",
			message.empty() ? "no exception" : fmt::format("threw: {}", message)
		)
	);
	if (! message.empty())
		return;

	// level-0 blocks are created by the LBM constructor, not by
	// createAMRBlocks, and are intentionally not accounted in
	// level_block_counts[0] (the vector tracks fine levels only)
	const std::vector<int> counts = state.nse.level_block_counts;
	CHECK_MESSAGE((
		state.nse.blocks.size() == 4 && counts == std::vector<int>({0, 1, 1, 1})),
		fmt::format(
			"3-level creation: block census -- {} blocks (1 per level), level_block_counts = [{},{},{},{}]",
			state.nse.blocks.size(),
			counts.size() > 0 ? counts[0] : -1,
			counts.size() > 1 ? counts[1] : -1,
			counts.size() > 2 ? counts[2] : -1,
			counts.size() > 3 ? counts[3] : -1
		)
	);

	const std::vector<ChainBlockExpectation> expectations = {
		{1, {8, 1, 4}, {17, 3, 9}, {22, 58, 22}},
		{2, {19, 2, 8}, {39, 5, 17}, {14, 118, 14}},
		{3, {41, 4, 16}, {83, 9, 33}, {6, 238, 6}},
	};
	bool geometry_ok = true;
	for (const ChainBlockExpectation& expected : expectations) {
		const std::vector<BLOCK*> blocks = state.nse.getBlocksAtLevel(expected.level);
		if (blocks.size() != 1) {
			geometry_ok = false;
			CHECK_MESSAGE(false, fmt::format("3-level creation: expected exactly one level-{} block (got {})", expected.level, blocks.size()));
			continue;
		}
		const BLOCK* fine = blocks.front();
		geometry_ok = geometry_ok && fine->global_offset == expected.global_offset && fine->offset == expected.offset
				   && fine->local == expected.local;
	}
	CHECK_MESSAGE(
		geometry_ok,
		"3-level creation: every fine block carries the exact parent-frame global_offset, re-anchored fine offset and fine "
		"local of the region chain (L1 22x58x22@(17,3,9), L2 14x118x14@(39,5,17), L3 6x238x6@(83,9,33))"
	);

	// per-level lattice scaling (the initLevelLattice chain): physDt/physDl
	// halve per level, the lattice viscosity doubles per level (the same
	// binary-exact relations the level-1 fixtures pin)
	bool scaling_ok = true;
	const double base_dl = static_cast<double>(lat.physDl);
	const double base_dt = static_cast<double>(lat.physDt);
	const double base_nu = static_cast<double>(lat.lbmViscosity());
	for (const auto& fine : state.nse.blocks) {
		if (fine.level == 0)
			continue;
		const double ratio = static_cast<double>(1 << fine.level);
		scaling_ok = scaling_ok && std::abs(static_cast<double>(fine.lat_local.physDl) * ratio - base_dl) <= 1e-12 * base_dl;
		scaling_ok = scaling_ok && std::abs(static_cast<double>(fine.lat_local.physDt) * ratio - base_dt) <= 1e-12 * base_dt;
		scaling_ok = scaling_ok && std::abs(static_cast<double>(fine.lat_local.lbmViscosity()) - ratio * base_nu) <= 1e-12 * base_nu;
	}
	CHECK_MESSAGE(
		scaling_ok,
		fmt::format(
			"3-level creation: per-level physDt/physDl halve and lattice viscosity doubles per level (nu level1..3 = {:.3e}, "
			"{:.3e}, {:.3e})",
			static_cast<double>(state.nse.getBlocksAtLevel(1).front()->lat_local.lbmViscosity()),
			static_cast<double>(state.nse.getBlocksAtLevel(2).front()->lat_local.lbmViscosity()),
			static_cast<double>(state.nse.getBlocksAtLevel(3).front()->lat_local.lbmViscosity())
		)
	);

	CHECK_MESSAGE(
		capture.sink->warnings.empty(),
		fmt::format(
			"3-level creation: zero warnings emitted for a fully valid nested chain (the no-new-warnings invariant; got {})",
			capture.sink->warnings.size()
		)
	);
}

// independent host-side enumeration of the contract band tag rule on the
// same parent-frame projections markAMRInterface reads: the fine block's
// footprint rect [go, go + gs) on the parent lattice (go = global_offset,
// gs = (local + 2)/2); the iteration and clipping mirror markAMRInterface:
// the 1-parent-cell halo around the rect clipped to the parent block's
// interior. Cells INSIDE the rect at depth >= 1 are GEO_NOTHING; the rect's
// surface shell (depth 0) and every halo cell with a neighbor inside the
// rect are GEO_AMR_INTERFACE. Maps in this fixture start all-FLUID and no
// walls exist, so the expected final map is exactly this re-tagging.
struct TagReference
{
	long interface_count = 0;
	long nothing_count = 0;
	std::map<std::tuple<idx, idx, idx>, int> tags;
};

TagReference referenceTagging(const BLOCK& parent, const idx3d& go, const idx3d& gs)
{
	TagReference ref;
	const idx x_begin = std::max(parent.offset.x(), go.x() - 1);
	const idx x_end = std::min(parent.offset.x() + parent.local.x(), go.x() + gs.x() + 1);
	const idx y_begin = std::max(parent.offset.y(), go.y() - 1);
	const idx y_end = std::min(parent.offset.y() + parent.local.y(), go.y() + gs.y() + 1);
	const idx z_begin = std::max(parent.offset.z(), go.z() - 1);
	const idx z_end = std::min(parent.offset.z() + parent.local.z(), go.z() + gs.z() + 1);
	for (idx x = x_begin; x < x_end; x++)
		for (idx y = y_begin; y < y_end; y++)
			for (idx z = z_begin; z < z_end; z++) {
				const bool inside = x >= go.x() && x < go.x() + gs.x() && y >= go.y() && y < go.y() + gs.y() && z >= go.z()
								 && z < go.z() + gs.z();
				if (inside) {
					const bool surface = x == go.x() || x == go.x() + gs.x() - 1 || y == go.y() || y == go.y() + gs.y() - 1
									  || z == go.z() || z == go.z() + gs.z() - 1;
					if (! surface) {
						ref.tags[std::make_tuple(x, y, z)] = BC::GEO_NOTHING;
						ref.nothing_count++;
						continue;
					}
				}
				bool crosses = false;
				for (int q = 1; q < 27 && ! crosses; q++) {
					const idx nx = x + amr_d3q27_directions[q][0];
					const idx ny = y + amr_d3q27_directions[q][1];
					const idx nz = z + amr_d3q27_directions[q][2];
					crosses = nx >= go.x() && nx < go.x() + gs.x() && ny >= go.y() && ny < go.y() + gs.y() && nz >= go.z()
						   && nz < go.z() + gs.z();
				}
				if (! crosses)
					continue;
				ref.tags[std::make_tuple(x, y, z)] = BC::GEO_AMR_INTERFACE;
				ref.interface_count++;
			}
	return ref;
}

void test_three_level_mark_census()
{
	lat_t lat = makeLattice(32);
	const std::string id = fmt::format("test_amr_nesting_{}_chainmark", pattern_name);
	StateLocal_AMR<NSE_CONFIG> state(id, MPI_COMM_WORLD, lat, "adios2.xml", /*periodic=*/TRAITS::bool3d{true, true, true}, /*max_level=*/3);
	if (! state.canCompute()) {
		CHECK_MESSAGE(false, "3-level mark census setup: state.canCompute()");
		return;
	}
	state.nse.allocateHostData();
	state.nse.allocateDeviceData();
	std::string message;
	try {
		createAMRBlocks(state.nse, parseAMRConfig<NSE_CONFIG>(three_level_chain));
	}
	catch (const std::runtime_error& e) {
		message = e.what();
	}
	CHECK_MESSAGE(
		message.empty(),
		fmt::format("3-level mark census setup: the chain validates ({})", message.empty() ? "no exception" : fmt::format("threw: {}", message))
	);
	if (! message.empty())
		return;

	// all-FLUID interior maps on every block (the fixture's starting point of
	// the reference tagging; createAMRBlocks reset the fine blocks' maps, the
	// level-0 block's map is reset here explicitly)
	for (auto& block : state.nse.blocks)
		block.resetMap(BC::GEO_FLUID);

	markAMRInterface(state.nse);

	// per-level census vs the independent reference enumeration; the full
	// interior map of every parent block is compared cell by cell and the
	// per-level map mismatches accumulate into one summary counter (the
	// clip regions of the chain's levels are disjoint parent lattices)
	std::map<int, long> interface_census;
	std::map<int, long> nothing_census;
	long map_mismatches_total = 0;
	for (auto& fine : state.nse.blocks) {
		if (fine.level == 0)
			continue;
		BLOCK* parent = state.nse.getBlocksAtLevel(fine.level - 1).front();
		const idx3d go = fine.global_offset;
		const idx3d gs{(fine.local.x() + 2) / 2, (fine.local.y() + 2) / 2, (fine.local.z() + 2) / 2};
		const TagReference ref = referenceTagging(*parent, go, gs);

		long interface_count = 0;
		long nothing_count = 0;
		long mismatches = 0;
		for (idx x = parent->offset.x(); x < parent->offset.x() + parent->local.x(); x++)
			for (idx y = parent->offset.y(); y < parent->offset.y() + parent->local.y(); y++)
				for (idx z = parent->offset.z(); z < parent->offset.z() + parent->local.z(); z++) {
					const int actual = parent->hmap(x, y, z);
					const auto it = ref.tags.find(std::make_tuple(x, y, z));
					const int expected = (it == ref.tags.end()) ? BC::GEO_FLUID : it->second;
					if (actual == BC::GEO_AMR_INTERFACE)
						interface_count++;
					if (actual == BC::GEO_NOTHING)
						nothing_count++;
					if (actual != expected)
						mismatches++;
				}
		interface_census[fine.level] = interface_count;
		nothing_census[fine.level] = nothing_count;
		map_mismatches_total += mismatches;
		CHECK_MESSAGE((
			mismatches == 0 && interface_count == ref.interface_count && nothing_count == ref.nothing_count),
			fmt::format(
				"3-level mark census: level-{} map matches the contract band tag rule of the level-{} fine footprint "
				"(GEO_AMR_INTERFACE = {}, GEO_NOTHING = {}, map mismatches = {})",
				fine.level - 1,
				fine.level,
				interface_count,
				nothing_count,
				mismatches
			)
		);
	}

	// spot checks for readable failure messages on top of the full-map
	// reference comparison: deep footprint cells are frozen and the
	// in-iteration halo rows are tagged on every parent lattice (the
	// wall-shared z-min faces' halo/surface rows land in the parent blocks'
	// ghost zone below the interior z-min, outside the map storage, so the
	// tag rules there are covered by the reference comparison's clip)
	{
		BLOCK* level0 = state.nse.getBlocksAtLevel(0).front();
		CHECK_MESSAGE((
			level0->hmap(12, 16, 10) == BC::GEO_NOTHING && level0->hmap(8, 0, 8) == BC::GEO_AMR_INTERFACE),
			"3-level mark census: level-0 spot checks -- deep footprint cell frozen, halo cell tagged"
		);
	}
	{
		BLOCK* level1 = state.nse.getBlocksAtLevel(1).front();
		CHECK_MESSAGE((
			level1->hmap(20, 30, 10) == BC::GEO_NOTHING && level1->hmap(20, 30, 16) == BC::GEO_AMR_INTERFACE),
			"3-level mark census: level-1 spot checks -- deep footprint cell frozen, halo cell tagged"
		);
	}
	{
		BLOCK* level2 = state.nse.getBlocksAtLevel(2).front();
		CHECK_MESSAGE((
			level2->hmap(42, 10, 18) == BC::GEO_NOTHING && level2->hmap(42, 10, 20) == BC::GEO_AMR_INTERFACE),
			"3-level mark census: level-2 spot checks -- deep footprint cell frozen, halo cell tagged"
		);
	}

	// re-invocation idempotency: a second markAMRInterface pass must not
	// change the maps (the bitmask OR-accumulates the same bits, only
	// GEO_FLUID cells are re-tagged)
	std::map<int, std::vector<int>> maps_before;
	for (const auto& block : state.nse.blocks) {
		std::vector<int> tags;
		for (idx x = block.offset.x(); x < block.offset.x() + block.local.x(); x++)
			for (idx y = block.offset.y(); y < block.offset.y() + block.local.y(); y++)
				for (idx z = block.offset.z(); z < block.offset.z() + block.local.z(); z++)
					tags.push_back(block.hmap(x, y, z));
		maps_before[block.level] = std::move(tags);
	}
	markAMRInterface(state.nse);
	bool idempotent = true;
	for (const auto& block : state.nse.blocks) {
		std::size_t k = 0;
		for (idx x = block.offset.x(); x < block.offset.x() + block.local.x(); x++)
			for (idx y = block.offset.y(); y < block.offset.y() + block.local.y(); y++)
				for (idx z = block.offset.z(); z < block.offset.z() + block.local.z(); z++, k++)
					idempotent = idempotent && block.hmap(x, y, z) == maps_before.at(block.level)[k];
	}
	CHECK_MESSAGE(idempotent, "3-level mark census: markAMRInterface re-invocation is idempotent on the 3-level chain");

	CHECK_MESSAGE(
		map_mismatches_total == 0,
		fmt::format(
			"3-level mark census: per-level censuses -- L1->L0 interface {} / frozen {}, L2->L1 interface {} / frozen {}, "
			"L3->L2 interface {} / frozen {}",
			interface_census[1],
			nothing_census[1],
			interface_census[2],
			nothing_census[2],
			interface_census[3],
			nothing_census[3]
		)
	);
}

// Schedule census on nested mocks (the advancePair recursion's event
// expansion; plan sec. 3.2-3.4).

using St = StateSchedule_AMR<NSE_CONFIG>::Stage;

// one row of the per-cycle expected schedule: the launch kind and level, the
// kernel extent class (kernels only), the expected DF rotation of the
// event-level block and of its parent-level block at the call site (-1 = do
// not assert, CLOCK = the level-0 global clock, alternating per cycle), and
// for f2c events the expected write-side next-substep parity (CLOCK =
// (iterations % 2) == 1 with the post-incremented clock); f2c rows carry an
// additional ordinal slot (unused, see checkCycleEvents)
struct ExpectedEvent
{
	St stage;
	int level;
	int ghost_layers;
	int fine_rot;
	int parent_rot;
	int f2c_next_parity;
};

constexpr int EVT_NA = -1;	  // field not asserted
constexpr int EVT_CLOCK = -2; // level-0 global clock value at this cycle

// the 2-level per-cycle expansion (plan sec. 3.4, s_1 = s_2 = 0 at cycle 0;
// rotations repeat every cycle because each level's per-cycle substep count
// 2^L is even and updateKernelDataForLevel is an absolute mod-2 setter)
const ExpectedEvent table_max2[13] = {
	{St::kernel, 1, 1, 0, EVT_CLOCK},
	{St::kernel, 2, 1, 0, 0},
	{St::kernel, 2, 0, 1, 0},
	{St::f2c, 2, 0, 1, 0, 1},
	{St::c2f, 2, 0, 0, 0},
	{St::kernel, 1, 0, 1, EVT_CLOCK},
	{St::kernel, 2, 1, 0, 1},
	{St::kernel, 2, 0, 1, 1},
	{St::f2c, 2, 0, 1, 1, 0},
	{St::kernel, 0, 0, EVT_NA, EVT_NA},
	{St::f2c, 1, 0, 1, EVT_CLOCK, EVT_CLOCK},
	{St::c2f, 1, 0, 0, EVT_CLOCK},
	{St::c2f, 2, 0, 0, 0},
};

// the 3-level per-cycle expansion (advancePair(1) containing two
// advancePair(2) invocations, each containing two advancePair(3)
// invocations): 15 kernel launches (1+2+4+8 substeps per level), 7 f2c and
// 6 c2f launches per cycle
const ExpectedEvent table_max3[28] = {
	{St::kernel, 1, 1, 0, EVT_CLOCK},
	{St::kernel, 2, 1, 0, 0},
	{St::kernel, 3, 1, 0, 0},
	{St::kernel, 3, 0, 1, 0},
	{St::f2c, 3, 0, 1, 0, 1},
	{St::c2f, 3, 0, 0, 0},
	{St::kernel, 2, 0, 1, 0},
	{St::kernel, 3, 1, 0, 1},
	{St::kernel, 3, 0, 1, 1},
	{St::f2c, 3, 0, 1, 1, 0},
	{St::f2c, 2, 0, 1, 0, 1},
	{St::c2f, 2, 0, 0, 0},
	{St::kernel, 1, 0, 1, EVT_CLOCK},
	{St::kernel, 2, 1, 0, 1},
	{St::kernel, 3, 1, 0, 0},
	{St::kernel, 3, 0, 1, 0},
	{St::f2c, 3, 0, 1, 0, 1},
	{St::c2f, 3, 0, 0, 0},
	{St::kernel, 2, 0, 1, 1},
	{St::kernel, 3, 1, 0, 1},
	{St::kernel, 3, 0, 1, 1},
	{St::f2c, 3, 0, 1, 1, 0},
	{St::f2c, 2, 0, 1, 1, 0},
	{St::kernel, 0, 0, EVT_NA, EVT_NA},
	{St::f2c, 1, 0, 1, EVT_CLOCK, EVT_CLOCK},
	{St::c2f, 1, 0, 0, EVT_CLOCK},
	{St::c2f, 2, 0, 0, 0},
	{St::c2f, 3, 0, 0, 0},
};

const char* stageName(St stage)
{
	switch (stage) {
		case St::kernel:
			return "kernel";
		case St::c2f:
			return "c2f";
		case St::f2c:
			return "f2c";
	}
	return "?";
}

// rotation state of the parity evidence the spy captured at one call site,
// reduced to the 0/1 form the expected tables use (AB: which physical array
// the captured df_cur pointer aliases; AA: the captured even_iter flag)
int capturedRotation(const BLOCK& block, const void* captured_cur, bool captured_even)
{
#ifdef AB_PATTERN
	static_cast<void>(captured_even);
	if (captured_cur == block.dfs[0].getData())
		return 0;
	if (captured_cur == block.dfs[1].getData())
		return 1;
	return -1;
#elif defined(AA_PATTERN)
	static_cast<void>(block);
	static_cast<void>(captured_cur);
	return captured_even ? 1 : 0;
#endif
}

// assert one 1-based cycle of the census against the table: events between
// [base, base + table_size) of the spy log; clock_rot is the level-0 block's
// rotation for this cycle ((cycle - 1) % 2 under the updateKernelData +
// SimUpdate driver), and next_parity EVT_CLOCK resolves to (cycle % 2) == 1
// because SimUpdate post-increments `iterations` before the f2c launch
bool checkCycleEvents(
	StateSchedule_AMR<NSE_CONFIG>& state,
	int cycle,
	const ExpectedEvent* table,
	std::size_t table_size,
	std::size_t base,
	std::string& failure
)
{
	using Evt = StateSchedule_AMR<NSE_CONFIG>::Event;
	const int clock_rot = (cycle - 1) % 2;
	if (state.events.size() != base + table_size) {
		failure = fmt::format("expected {} events, recorded {}", table_size, state.events.size() - base);
		return false;
	}
	BLOCK* level0 = state.nse.getBlocksAtLevel(0).front();
	// 1-based ordinal of the f2c launches per coupling inside this cycle: the
	// j-th f2c at coupling (L -> L-1) fires when the parent's cumulative
	// counter reads 2^(L-1)*(cycle-1) + j (once per parent substep)
	int f2c_ordinal[8] = {0, 0, 0, 0, 0, 0, 0, 0};
	for (std::size_t i = 0; i < table_size; i++) {
		const ExpectedEvent& want = table[i];
		const Evt& evt = state.events[base + i];
		if (evt.stage != want.stage || evt.level != want.level) {
			failure = fmt::format(
				"event {}: expected {} L{}, recorded {} L{}", i + 1, stageName(want.stage), want.level, stageName(evt.stage), evt.level
			);
			return false;
		}
		if (want.stage == St::kernel && evt.ghost_layers != want.ghost_layers) {
			failure = fmt::format("event {} (kernel L{}): ghost_layers {} != {}", i + 1, want.level, evt.ghost_layers, want.ghost_layers);
			return false;
		}
		BLOCK* block = want.level > 0 ? state.nse.getBlocksAtLevel(want.level).front() : nullptr;
		BLOCK* parent = want.level > 0 ? state.nse.getBlocksAtLevel(want.level - 1).front() : nullptr;
		const void* fine_cur = nullptr;
		const void* parent_cur = nullptr;
		bool fine_even = false, parent_even = false;
#ifdef AB_PATTERN
		fine_cur = evt.fine_cur;
		parent_cur = evt.parent_cur;
#elif defined(AA_PATTERN)
		fine_even = evt.fine_even;
		parent_even = evt.parent_even;
#endif
		if (want.stage == St::kernel && want.level == 0) {
			// the level-0 step is driven by the global updateKernelData
			// clock (set before SimUpdate) and must not be re-armed by any
			// fine-level preparation
#ifdef AB_PATTERN
			if (evt.coarse_cur != (clock_rot == 0 ? level0->dfs[0].getData() : level0->dfs[1].getData())) {
				failure = fmt::format("event {} (kernel L0): level-0 rotation does not match the global clock of cycle {}", i + 1, cycle);
				return false;
			}
#elif defined(AA_PATTERN)
			if (evt.coarse_even != (clock_rot == 1)) {
				failure = fmt::format("event {} (kernel L0): level-0 even_iter does not match the global clock of cycle {}", i + 1, cycle);
				return false;
			}
#endif
		}
		if (block != nullptr && want.fine_rot != EVT_NA) {
			const int expected_rot = want.fine_rot == EVT_CLOCK ? clock_rot : want.fine_rot;
			const int recorded_rot = capturedRotation(*block, fine_cur, fine_even);
			if (recorded_rot != expected_rot) {
				failure = fmt::format(
					"event {} ({} L{}): level rotation mismatch (expected {}, recorded {})",
					i + 1,
					stageName(want.stage),
					want.level,
					expected_rot,
					recorded_rot
				);
				return false;
			}
		}
		if (parent != nullptr && want.parent_rot != EVT_NA) {
			const int expected_rot = want.parent_rot == EVT_CLOCK ? clock_rot : want.parent_rot;
			const int recorded_rot = capturedRotation(*parent, parent_cur, parent_even);
			if (recorded_rot != expected_rot) {
				failure = fmt::format(
					"event {} ({} L{}): parent (L{}) rotation mismatch (expected {}, recorded {})",
					i + 1,
					stageName(want.stage),
					want.level,
					want.level - 1,
					expected_rot,
					recorded_rot
				);
				return false;
			}
		}
		if (want.stage == St::f2c) {
			// EVT_CLOCK: the f2c(1 -> 0) write side is the post-incremented
			// global clock of THIS cycle (SimUpdate already incremented
			// `iterations` to `cycle`)
			const bool expected_parity = want.f2c_next_parity == EVT_CLOCK ? (cycle % 2) == 1 : want.f2c_next_parity == 1;
			const bool recorded_parity = (evt.next_parent_substep % 2) == 1;
			if (evt.next_parent_substep < 0 || recorded_parity != expected_parity) {
				failure = fmt::format(
					"event {} (f2c L{} -> L{}): write-side next-substep parity mismatch (recorded index {}, parity {})",
					i + 1,
					want.level,
					want.level - 1,
					evt.next_parent_substep,
					recorded_parity
				);
				return false;
			}
			f2c_ordinal[want.level]++;
			const int expected_abs =
				want.level > 1 ? (1 << (want.level - 1)) * (cycle - 1) + f2c_ordinal[want.level] : cycle;
			if (evt.next_parent_substep != expected_abs) {
				failure = fmt::format(
					"event {} (f2c L{} -> L{}): write-side next-substep index {} != {} (cumulative L{} counter)",
					i + 1,
					want.level,
					want.level - 1,
					evt.next_parent_substep,
					expected_abs,
					want.level - 1
				);
				return false;
			}
		}
	}
	return true;
}

// driver shared by the 2-level and 3-level census: SimInit's level-ascending
// initial-fill cascade, then 3 full cycles each asserted against the table
// and the cumulative per-level counters
void checkScheduleCensus(int max_level, const char* regions, const char* label, const ExpectedEvent* table, std::size_t table_size)
{
	using St = StateSchedule_AMR<NSE_CONFIG>::Stage;
	lat_t lat = makeLattice(32);
	const std::string id = fmt::format("test_amr_nesting_{}_census{}", pattern_name, max_level);
	StateSchedule_AMR<NSE_CONFIG> state(id, MPI_COMM_WORLD, lat, "adios2.xml", /*periodic=*/TRAITS::bool3d{true, true, true}, max_level);
	if (! state.canCompute()) {
		CHECK_MESSAGE(false, fmt::format("{} setup: state.canCompute()", label));
		return;
	}
	std::string message;
	try {
		createAMRBlocks(state.nse, parseAMRConfig<NSE_CONFIG>(regions));
		state.SimInit();
	}
	catch (const std::runtime_error& e) {
		message = e.what();
	}
	CHECK_MESSAGE((
		message.empty() && ! state.nse.terminate),
		fmt::format("{} setup: chain creation and SimInit ({})", label, message.empty() ? "no exception" : fmt::format("threw: {}", message))
	);
	if (! message.empty() || state.nse.terminate)
		return;

	// SimInit's initial fill: one c2f launch per level, level-ascending, each
	// launched at substep counter 0 (the cycle-0 anchor of the pair schedule;
	// the fill targets the substep-0 rotation exactly as at every cycle end)
	bool init_ok = state.events.size() == static_cast<std::size_t>(max_level);
	for (int L = 1; L <= max_level && init_ok; L++) {
		const auto& evt = state.events[L - 1];
		BLOCK* block = state.nse.getBlocksAtLevel(L).front();
		init_ok = evt.stage == St::c2f && evt.level == L;
#ifdef AB_PATTERN
		init_ok = init_ok && evt.fine_cur == block->dfs[0].getData();
#elif defined(AA_PATTERN)
		init_ok = init_ok && evt.fine_even == false;
#endif
	}
	CHECK_MESSAGE(init_ok, fmt::format("{} SimInit: level-ascending initial c2f cascade ({} events, all at rotation 0)", label, state.events.size()));
	state.events.clear();

	bool cycles_ok = true;
	std::string failure;
	for (int cycle = 1; cycle <= 3 && cycles_ok; cycle++) {
		const std::size_t base = state.events.size();
		state.updateKernelData();
		state.SimUpdate();
		if (state.nse.terminate) {
			failure = "SimUpdate set the terminate flag";
			cycles_ok = false;
			break;
		}
		if (! checkCycleEvents(state, cycle, table, table_size, base, failure))
			cycles_ok = false;
		// cumulative per-level substep counters: exactly 2^L substeps per
		// cycle at level L
		for (int L = 1; L <= max_level && cycles_ok; L++) {
			const int expected_count = cycle * (1 << L);
			if (state.nse.totalSubstepCount[L] != expected_count) {
				failure = fmt::format(
					"cycle {}: totalSubstepCount[{}] = {} != {} (2^L per cycle)", cycle, L, state.nse.totalSubstepCount[L], expected_count
				);
				cycles_ok = false;
			}
		}
	}
	CHECK_MESSAGE((
		cycles_ok && state.nse.iterations == 3),
		fmt::format(
			"{}: 3 cycles match the {}-event advancePair expansion per cycle (rotations, f2c next-parity/indices, counters){}",
			label,
			table_size,
			failure.empty() ? "" : fmt::format(" -- first failure: {}", failure)
		)
	);
	CHECK_MESSAGE(! state.nse.terminate, fmt::format("{}: no termination during the census run", label));
}

void test_two_level_schedule_census()
{
	// the top two levels of the validated chain (same geometry family as the
	// 3-level census, max_level 2)
	checkScheduleCensus(2, "1 8 1 4 12 30 12\n2 38 4 16 16 120 16", "2-level census", table_max2, 13);
}

void test_three_level_schedule_census()
{
	checkScheduleCensus(3, three_level_chain, "3-level census", table_max3, 28);
}

// 3-level conservation smoke (plotting-free): 20 coupled cycles on a
// telescoping 3-level chain whose faces are all coarse-to-fine-fillable
// (three_level_interior_chain); the conservation stats must stay finite (no
// NaN), internally consistent with the GEO_NOTHING-excluding host reference,
// and the global mass stable across cycles
void test_three_level_conservation_smoke()
{
	lat_t lat = makeLattice(32);
	const std::string id = fmt::format("test_amr_nesting_{}_smoke", pattern_name);
	StateSchedule_AMR<NSE_CONFIG> state(id, MPI_COMM_WORLD, lat, "adios2.xml", /*periodic=*/TRAITS::bool3d{true, true, true}, /*max_level=*/3);
	if (! state.canCompute()) {
		CHECK_MESSAGE(false, "3-level conservation smoke setup: state.canCompute()");
		return;
	}
	std::string message;
	try {
		createAMRBlocks(state.nse, parseAMRConfig<NSE_CONFIG>(three_level_interior_chain));
		state.SimInit();
	}
	catch (const std::runtime_error& e) {
		message = e.what();
	}
	CHECK_MESSAGE((
		message.empty() && ! state.nse.terminate),
		fmt::format(
			"3-level conservation smoke setup: chain creation and SimInit ({})",
			message.empty() ? "no exception" : fmt::format("threw: {}", message)
		)
	);
	if (! message.empty() || state.nse.terminate)
		return;

	// force macroscopic output inside every kernel so the per-level macros
	// of the cycle-end snapshot are fresh on all 4 levels (the subcycling
	// tests' OUT3DCUT idiom)
	state.cnt[OUT3DCUT].period = 1e-30;

	double drift_final = 0;
	double mass_earliest = -1;
	bool finite_ok = true;
	bool consistent_ok = true;
	bool stable_ok = true;
	std::string failure;
	for (int iter = 1; iter <= 20 && ! state.nse.terminate; iter++) {
		state.updateKernelData();
		state.SimUpdate();
		if (iter % 5 != 0)
			continue;

		const AMRConservationStats s = state.computeConservationStats();
		const RefStats ref = computeReferenceStats(state);
		const double mass = s.total_mass;
		if (mass_earliest < 0)
			mass_earliest = mass;

		if (! std::isfinite(mass) || ! std::isfinite(s.total_momentum_x) || ! std::isfinite(s.total_momentum_y)
			|| ! std::isfinite(s.total_momentum_z) || s.per_level_kinetic_energy.size() != 4) {
			finite_ok = false;
			failure = fmt::format("cycle {}: non-finite conservation entry (mass {:.6e})", iter, mass);
			break;
		}
		for (int L = 0; L <= 3; L++)
			if (! std::isfinite(s.per_level_kinetic_energy[L]) || s.per_level_kinetic_energy[L] < 0) {
				finite_ok = false;
				failure = fmt::format("cycle {}: non-finite/negative level-{} kinetic energy {:.6e}", iter, L, s.per_level_kinetic_energy[L]);
			}
		if (! finite_ok)
			break;

		// internal consistency with the host-side GEO_NOTHING-excluding
		// reference (the nesting metric's hidden-cell exclusion at 3 levels)
		if (! closeRel(mass, ref.mass) || ! closeRel(s.total_momentum_x, ref.mx) || ! closeRel(s.total_momentum_y, ref.my)
			|| ! closeRel(s.total_momentum_z, ref.mz)) {
			consistent_ok = false;
			failure = fmt::format("cycle {}: metric {:.6e} deviates from the reference {:.6e}", iter, mass, ref.mass);
			break;
		}
		// the drift bound: periodic box, no in- or outflow; every band of
		// the interior chain is coarse-to-fine-filled each cycle, so the
		// mass must stay stable -- the 5% bound over cycles 5 -> 20 is
		// pre-registered generous, the observed drift is printed below
		const double drift = std::abs(mass - mass_earliest) / mass_earliest;
		if (iter == 20) {
			drift_final = drift;
			if (drift > 5e-2) {
				stable_ok = false;
				failure = fmt::format("mass drift {:.6e} over cycles 5 -> 20 exceeds the 5e-2 bound", drift);
			}
		}
	}
	CHECK_MESSAGE(
		finite_ok,
		fmt::format(
			"3-level conservation smoke: all conservation entries finite (no NaN) over 20 cycles{}",
			failure.empty() && finite_ok ? "" : fmt::format(" -- {}", failure)
		)
	);
	CHECK_MESSAGE(
		consistent_ok,
		fmt::format(
			"3-level conservation smoke: nested metric matches the GEO_NOTHING-excluding reference at every checkpoint{}",
			failure.empty() && consistent_ok ? "" : fmt::format(" -- {}", failure)
		)
	);
	CHECK_MESSAGE((
		stable_ok && mass_earliest > 0),
		fmt::format("3-level conservation smoke: global mass stable over 20 coupled cycles (relative drift {:.6e} over cycles 5 -> 20)", drift_final)
	);
	CHECK_MESSAGE((state.nse.iterations == 20 && ! state.nse.terminate), "3-level conservation smoke: 20 coupled cycles, no termination");
}

// Commit E wall-chain tests (plan sec. 5 + sec. 5.5 Tests 12--14): the wall
// masks derive from the IMMEDIATE PARENT's map at every level (plan sec. 5.1),
// the wall-shared faces keep their empty coarse-to-fine destinations with the
// launch windows deepened to the wall row, the R4 wall-pedestal prisms author
// the deep frozen rows the parent's own-8 upward fine-to-coarse window reads
// (plan sec. 5.3), and the fail-fast rails reject every silent lane. The
// geometry is three_level_chain with all z-min faces wall-backed down to a
// level-0 plane (the StateWallChain_AMR fixture).

// shared wall-chain setup: the 3-level telescoping chain with the z-min wall
// chain tagged at every level (plus the level-0 backing plane) and the 3-deep
// z overlap on every fine block; SimInit must pass untouched
void setupWallChain(StateWallChain_AMR<NSE_CONFIG>& state, const char* label)
{
	state.tag_level0_wall = true;
	state.tag_fine_wall[1] = state.tag_fine_wall[2] = state.tag_fine_wall[3] = true;
	std::string message;
	try {
		createAMRBlocks(state.nse, parseAMRConfig<NSE_CONFIG>(three_level_chain));
		for (auto& block : state.nse.blocks)
			if (block.level > 0)
				block.storage_overlap_z = 3;
		state.SimInit();
	}
	catch (const std::runtime_error& e) {
		message = e.what();
	}
	CHECK_MESSAGE((
		message.empty() && ! state.nse.terminate),
		fmt::format("{} setup: chain, wall tagging and SimInit ({})", label, message.empty() ? "no exception" : fmt::format("threw: {}", message))
	);
}

// Test 12 (nested mask census): the fine_wall_masks bit 4 is set on the blocks
// at levels 1, 2 and 3, derived through BOTH parent hops; no z-min
// coarse-to-fine destination survives on any coupling; the z launch windows
// deepen to the wall row at local -2 on all three levels
void test_wall_chain_masks()
{
#if defined(F2C_LAGRAVA)
	CHECK_MESSAGE(
		true,
		"wall chain: masks census is a default-strategy (F2C_SCHONHERR) resident -- under F2C_LAGRAVA the wall chain is "
		"guard-rejected at SimInit (see the Lagrava guard test)"
	);
	return;
#endif
	using SyncDirection = TNL::Containers::SyncDirection;

	lat_t lat = makeLattice(32);
	const std::string id = fmt::format("test_amr_nesting_{}_wallchain", pattern_name);
	StateWallChain_AMR<NSE_CONFIG> state(id, MPI_COMM_WORLD, lat, "adios2.xml", /*periodic=*/TRAITS::bool3d{true, true, true}, /*max_level=*/3);
	if (! state.canCompute()) {
		CHECK_MESSAGE(false, "wall-chain masks setup: state.canCompute()");
		return;
	}
	setupWallChain(state, "wall-chain masks");
	if (state.nse.terminate)
		return;

	// (a) mask census: exactly the z-min bit at every fine level -- level 1
	// backed by the level-0 plane, levels 2 and 3 backed by the parent's own
	// wall row one coarse cell outside the projected footprint edge
	const int zmin_bit = State_AMR<NSE_CONFIG>::fineWallFaceBit(SyncDirection::Back);
	for (int L = 1; L <= 3; L++) {
		BLOCK* fine = state.nse.getBlocksAtLevel(L).front();
		CHECK_MESSAGE(
			state.fineWallMask(*fine) == (1 << zmin_bit),
			fmt::format(
				"wall chain: level-{} fine_wall_masks == {{z-min: 1, else: 0}} (mask = {})", L, static_cast<int>(state.fineWallMask(*fine))
			)
		);
	}

	// launch census: the wall-row deepening is observable on both substep
	// classes at every level (z begin -2, extents local+2 / local+3 while the
	// other axes keep the nominal simulated-band / interior windows)
	for (int L = 1; L <= 3; L++) {
		BLOCK* fine = state.nse.getBlocksAtLevel(L).front();
		const auto w1 = state.kernelLaunchWindow(*fine, 1);
		const auto w0 = state.kernelLaunchWindow(*fine, 0);
		CHECK_MESSAGE((
			w1.first.z() == -2 && w1.second.z() == fine->local.z() + 3 && w1.first.x() == -1 && w1.first.y() == -1 && w0.first.z() == -2
				&& w0.second.z() == fine->local.z() + 2 && w0.first.x() == 0 && w0.first.y() == 0),
			fmt::format(
				"wall chain: level-{} launch windows deepen to the GEO_WALL row at local z=-2 on both substeps "
				"(widened [{},{},{}] + [{},{},{}], interior [{},{},{}] + [{},{},{}])",
				L,
				w1.first.x(),
				w1.first.y(),
				w1.first.z(),
				w1.second.x(),
				w1.second.y(),
				w1.second.z(),
				w0.first.x(),
				w0.first.y(),
				w0.first.z(),
				w0.second.x(),
				w0.second.y(),
				w0.second.z()
			)
		);
	}

	// destination census: every z-min coarse-to-fine destination of every
	// coupling is dropped (masked faces are BC-managed end to end)
	bool zmin_destinations_empty = true;
	for (const auto& coupling : state.couplings)
		for (const auto& patch : coupling.patches)
			if (patch.face == SyncDirection::Back && patch.fine_size.z() != 0)
				zmin_destinations_empty = false;
	CHECK_MESSAGE(zmin_destinations_empty, "wall chain: no coupling carries a coarse-to-fine fill on any z-min (wall-shared) face");

	// nested channel smoke (plan sec. 8 row E): 3 coupled cycles over the
	// wall chain run cleanly -- the deepened launches process every level's
	// wall row without termination, and the conservation stats stay finite
	for (int cycle = 0; cycle < 3 && ! state.nse.terminate; cycle++) {
		state.updateKernelData();
		state.SimUpdate();
	}
	CHECK_MESSAGE((
		state.nse.iterations == 3 && ! state.nse.terminate),
		"wall chain: 3 coupled cycles over the wall chain, no termination (the deepened launches process every wall row)"
	);
	const AMRConservationStats stats = state.computeConservationStats();
	bool finite = std::isfinite(stats.total_mass) && std::isfinite(stats.total_momentum_x) && std::isfinite(stats.total_momentum_y)
			   && std::isfinite(stats.total_momentum_z) && stats.per_level_kinetic_energy.size() == 4;
	for (int L = 0; L <= 3 && finite; L++)
		finite = finite && std::isfinite(stats.per_level_kinetic_energy[L]);
	CHECK_MESSAGE(finite, "wall chain: conservation entries finite after 3 cycles over the wall chain");
}

// Test 14 (R4 census) + the census invariants of plan sec. 5: the (1,2)
// coupling carries the standard 6 depth-1 skins plus exactly 1 R4 prism on
// the z-min face at relative rows {1,2} beyond the skin; the (2,3) coupling
// carries no prism (the level-3 footprint's twice-inset tangents are empty on
// its gs = 4 axes -- it has no unreachable deep core at all); every interior
// destination cell is frozen GEO_NOTHING at a covered depth and no window
// touches the never-written deep core beyond the pedestal
void test_wall_pedestal_prisms()
{
#if defined(F2C_LAGRAVA)
	CHECK_MESSAGE(
		true,
		"R4 pedestal: census is a default-strategy (F2C_SCHONHERR) resident -- under F2C_LAGRAVA the wall chain is "
		"guard-rejected at SimInit (see the Lagrava guard test)"
	);
	return;
#endif
	lat_t lat = makeLattice(32);
	const std::string id = fmt::format("test_amr_nesting_{}_wallprism", pattern_name);
	StateWallChain_AMR<NSE_CONFIG> state(id, MPI_COMM_WORLD, lat, "adios2.xml", /*periodic=*/TRAITS::bool3d{true, true, true}, /*max_level=*/3);
	if (! state.canCompute()) {
		CHECK_MESSAGE(false, "wall-prism census setup: state.canCompute()");
		return;
	}
	setupWallChain(state, "wall-prism census");
	if (state.nse.terminate)
		return;

	BLOCK* l1 = state.nse.getBlocksAtLevel(1).front();
	BLOCK* l2 = state.nse.getBlocksAtLevel(2).front();
	const idx3d& go2 = l2->global_offset;
	const idx3d gs2{(l2->local.x() + 2) / 2, (l2->local.y() + 2) / 2, (l2->local.z() + 2) / 2};

	// hand-computed pedestal geometry of three_level_chain's level-2 block
	// (footprint {19,2,8} + {8,60,8} in level-1 cells): twice-inset
	// tangents [21, 25) x [4, 60), normal rows z in {10, 11} (the relative
	// rows {1,2} of the wall pedestal behind the standard skin at z = 9),
	// expressed in the parent's indexer frame
	const idx3d prism_begin{go2.x() + 2 - l1->offset.x(), go2.y() + 2 - l1->offset.y(), go2.z() + 2 - l1->offset.z()};
	const idx3d prism_size{gs2.x() - 4, gs2.y() - 4, 2};
	const auto& c12 = state.couplings[1];
	int prisms_found = 0;
	bool skin_zmin_present = false;
	for (const auto& patch : c12.interior_patches) {
		if (patch.coarse_origin == prism_begin && patch.coarse_size == prism_size)
			prisms_found++;
		if (patch.coarse_origin.z() == prism_begin.z() - 1 && patch.coarse_size.z() == 1)
			skin_zmin_present = true;
	}
	CHECK_MESSAGE((
		c12.interior_patches.size() == 7 && prisms_found == 1),
		fmt::format(
			"R4 pedestal: the (1,2) coupling carries the 6 depth-1 skins plus exactly 1 z-min prism at relative rows "
			"{{1,2}} ({} interior patches, {} prisms)",
			c12.interior_patches.size(),
			prisms_found
		)
	);
	CHECK_MESSAGE(skin_zmin_present, "R4 pedestal: the standard z-min depth-1 skin is untouched (disjoint from the prism)");

	// the (2,3) coupling: the level-3 footprint (gs = 4 on x and z) has no
	// twice-inset tangent and no deep core, so it needs and gets no prism
	const auto& c23 = state.couplings[2];
	CHECK_MESSAGE(
		c23.interior_patches.size() == 2,
		fmt::format(
			"R4 pedestal: the (2,3) coupling carries no prism on the thin level-3 footprint "
			"(gs x/z = 4 -- empty twice-inset tangents; {} interior patches)",
			c23.interior_patches.size()
		)
	);

	// census invariant (d): enumerate every interior-destination cell of
	// both nested couplings: frozen GEO_NOTHING on the parent map, at a
	// depth covered by the {1} skins or the {1,2,3} pedestal of the z-min
	// wall face, and nothing deeper (the never-written deep core stays
	// untouched)
	const auto depth_of = [](idx gx, idx gy, idx gz, const idx3d& go, const idx3d& gs) -> idx
	{
		return std::min(
			std::min(std::min(gx - go.x(), go.x() + gs.x() - 1 - gx), std::min(gy - go.y(), go.y() + gs.y() - 1 - gy)),
			std::min(gz - go.z(), go.z() + gs.z() - 1 - gz)
		);
	};
	bool tags_ok = true;
	bool depths_ok = true;
	bool pedestal_cover = true;
	for (std::size_t c = 1; c < state.couplings.size(); c++) {
		const auto& coupling = state.couplings[c];
		BLOCK* fine = state.nse.getBlocksAtLevel(static_cast<int>(c) + 1).front();
		BLOCK* parent = state.nse.getBlocksAtLevel(static_cast<int>(c)).front();
		const idx3d& go = fine->global_offset;
		const idx3d gs{(fine->local.x() + 2) / 2, (fine->local.y() + 2) / 2, (fine->local.z() + 2) / 2};
		for (const auto& patch : coupling.interior_patches)
			for (idx x = patch.coarse_origin.x(); x < patch.coarse_origin.x() + patch.coarse_size.x(); x++)
				for (idx y = patch.coarse_origin.y(); y < patch.coarse_origin.y() + patch.coarse_size.y(); y++)
					for (idx z = patch.coarse_origin.z(); z < patch.coarse_origin.z() + patch.coarse_size.z(); z++) {
						const idx gx = parent->offset.x() + x, gy = parent->offset.y() + y, gz = parent->offset.z() + z;
						if (parent->hmap(gx, gy, gz) != BC::GEO_NOTHING)
							tags_ok = false;
						const idx depth = depth_of(gx, gy, gz, go, gs);
						// depth 1 everywhere else, {1,2,3} in z on the
						// level-2 wall-shared z-min face
						const idx max_depth = c == 1 ? 3 : 1;
						if (depth < 1 || depth > max_depth)
							depths_ok = false;
					}
		if (c == 1) {
			// coverage: every frozen cell of the pedestal rows with
			// twice-inset x/y is an interior-destination cell
			for (idx gx = go.x() + 2; gx < go.x() + gs.x() - 2; gx++)
				for (idx gy = go.y() + 2; gy < go.y() + gs.y() - 2; gy++)
					for (idx gz = go.z() + 2; gz < go.z() + 4; gz++) {
						if (parent->hmap(gx, gy, gz) != BC::GEO_NOTHING) {
							pedestal_cover = false;
							continue;
						}
						bool authored = false;
						for (const auto& patch : coupling.interior_patches)
							if (gx - parent->offset.x() >= patch.coarse_origin.x()
								&& gx - parent->offset.x() < patch.coarse_origin.x() + patch.coarse_size.x()
								&& gy - parent->offset.y() >= patch.coarse_origin.y()
								&& gy - parent->offset.y() < patch.coarse_origin.y() + patch.coarse_size.y()
								&& gz - parent->offset.z() >= patch.coarse_origin.z()
								&& gz - parent->offset.z() < patch.coarse_origin.z() + patch.coarse_size.z())
								authored = true;
						if (! authored)
							pedestal_cover = false;
					}
		}
	}
	CHECK_MESSAGE(tags_ok, "R4 census (d): every interior destination cell of the nested couplings is frozen GEO_NOTHING");
	CHECK_MESSAGE(depths_ok, "R4 census (d): destinations sit at depth {1} everywhere except the z-min pedestal {1,2,3} (no deeper cell touched)");
	CHECK_MESSAGE(pedestal_cover, "R4 census (d): the deep frozen rows the parent's upward own-8 window reads are fully F2C-authored");
}

// Test 13 (fail-fast): the three silent lanes of the wall chain each die
// with a named throw -- a partial parent wall at level 2, a level-2 block
// missing the storage_overlap_z = 3 override, and a wall-shared face whose
// own wall row is tagged but the parent level holds no wall (the unbacked
// gap-0 lane)
void test_wall_chain_failfast()
{
#if defined(F2C_LAGRAVA)
	CHECK_MESSAGE(
		true,
		"wall fail-fast: the three wall-chain rails are default-strategy (F2C_SCHONHERR) residents -- under F2C_LAGRAVA the "
		"guard fires first at SimInit (see the Lagrava guard test)"
	);
	return;
#endif
	// (i) partial parent wall at level 2: the level-1 block's own wall row
	// is truncated in x, so the level-2 scan sees backed and unbacked
	// columns on the same face
	{
		lat_t lat = makeLattice(32);
		const std::string id = fmt::format("test_amr_nesting_{}_wallpartial", pattern_name);
		StateWallChain_AMR<NSE_CONFIG> state(id, MPI_COMM_WORLD, lat, "adios2.xml", /*periodic=*/TRAITS::bool3d{true, true, true}, /*max_level=*/3);
		if (! state.canCompute()) {
			CHECK_MESSAGE(false, "wall fail-fast (i) setup: state.canCompute()");
			return;
		}
		state.tag_level0_wall = true;
		state.tag_fine_wall[1] = state.tag_fine_wall[2] = state.tag_fine_wall[3] = true;
		state.tag_level1_wall_partial = true;
		std::string message;
		try {
			createAMRBlocks(state.nse, parseAMRConfig<NSE_CONFIG>(three_level_chain));
			for (auto& block : state.nse.blocks)
				if (block.level > 0)
					block.storage_overlap_z = 3;
			state.SimInit();
		}
		catch (const std::runtime_error& e) {
			message = e.what();
		}
		CHECK_MESSAGE((
			message.find("PARTIAL fine-level wall") != std::string::npos && message.find("z-min") != std::string::npos
				&& message.find("block 2") != std::string::npos),
			fmt::format(
				"wall fail-fast (i): partial parent wall at level 2 throws the named error (block, face, counts) -- {}",
				message.empty() ? "no exception thrown" : fmt::format("threw: {}", message)
			)
		);
	}

	// (ii) full wall chain but the level-2 block misses the
	// storage_overlap_z = 3 override (its GEO_NOTHING streaming buffer row
	// would lie outside the allocated storage)
	{
		lat_t lat = makeLattice(32);
		const std::string id = fmt::format("test_amr_nesting_{}_wallnooverlap", pattern_name);
		StateWallChain_AMR<NSE_CONFIG> state(id, MPI_COMM_WORLD, lat, "adios2.xml", /*periodic=*/TRAITS::bool3d{true, true, true}, /*max_level=*/3);
		if (! state.canCompute()) {
			CHECK_MESSAGE(false, "wall fail-fast (ii) setup: state.canCompute()");
			return;
		}
		state.tag_level0_wall = true;
		state.tag_fine_wall[1] = state.tag_fine_wall[2] = state.tag_fine_wall[3] = true;
		std::string message;
		try {
			createAMRBlocks(state.nse, parseAMRConfig<NSE_CONFIG>(three_level_chain));
			for (auto& block : state.nse.blocks)
				if (block.level == 1 || block.level == 3)
					block.storage_overlap_z = 3;
			state.SimInit();
		}
		catch (const std::runtime_error& e) {
			message = e.what();
		}
		CHECK_MESSAGE((
			message.find("the z-axis overlap is 2 (< 3)") != std::string::npos && message.find("storage_overlap_z") != std::string::npos
				&& message.find("z-min") != std::string::npos && message.find("block 2") != std::string::npos),
			fmt::format(
				"wall fail-fast (ii): missing storage_overlap_z = 3 at level 2 throws the named error -- {}",
				message.empty() ? "no exception thrown" : fmt::format("threw: {}", message)
			)
		);
	}

	// (iii) unbacked gap-0 face: only the level-2 block's own wall row is
	// tagged (no level-0 plane, no level-1 row); the parent hop then holds
	// no wall and the configuration must die instead of silently dropping
	// the wall chain
	{
		lat_t lat = makeLattice(32);
		const std::string id = fmt::format("test_amr_nesting_{}_wallunbacked", pattern_name);
		StateWallChain_AMR<NSE_CONFIG> state(id, MPI_COMM_WORLD, lat, "adios2.xml", /*periodic=*/TRAITS::bool3d{true, true, true}, /*max_level=*/3);
		if (! state.canCompute()) {
			CHECK_MESSAGE(false, "wall fail-fast (iii) setup: state.canCompute()");
			return;
		}
		state.tag_fine_wall[2] = true;
		std::string message;
		try {
			createAMRBlocks(state.nse, parseAMRConfig<NSE_CONFIG>(three_level_chain));
			for (auto& block : state.nse.blocks)
				if (block.level > 0)
					block.storage_overlap_z = 3;
			state.SimInit();
		}
		catch (const std::runtime_error& e) {
			message = e.what();
		}
		CHECK_MESSAGE((
			message.find("has GEO_WALL tags on its own z-min wall row but no wall backing on the parent level") != std::string::npos
				&& message.find("block 2") != std::string::npos),
			fmt::format(
				"wall fail-fast (iii): wall-shared face without parent wall backing throws the named error -- {}",
				message.empty() ? "no exception thrown" : fmt::format("threw: {}", message)
			)
		);
	}
}

// Test (c) (strategy coupling guard, plan sec. 5.4): nested wall-shared faces
// + the F2C_LAGRAVA fine-to-coarse strategy (whose 4-node window underflows
// the 3-row wall pedestal) must hard-error at SimInit naming the required
// strategy. The guard is compiled only under the F2C_LAGRAVA define (the
// binaries build green under either strategy, mirroring the strategy-split
// idiom of test_amr_coupling.cu); under the default F2C_SCHONHERR build the
// guard is inactive and the wall chain initializes cleanly (asserted by the
// mask census above)
void test_wall_chain_lagrava_guard()
{
#if defined(F2C_LAGRAVA)
	lat_t lat = makeLattice(32);
	const std::string id = fmt::format("test_amr_nesting_{}_wallguard", pattern_name);
	StateWallChain_AMR<NSE_CONFIG> state(id, MPI_COMM_WORLD, lat, "adios2.xml", /*periodic=*/TRAITS::bool3d{true, true, true}, /*max_level=*/3);
	if (! state.canCompute()) {
		CHECK_MESSAGE(false, "Lagrava guard setup: state.canCompute()");
		return;
	}
	state.tag_level0_wall = true;
	state.tag_fine_wall[1] = state.tag_fine_wall[2] = state.tag_fine_wall[3] = true;
	std::string message;
	try {
		createAMRBlocks(state.nse, parseAMRConfig<NSE_CONFIG>(three_level_chain));
		for (auto& block : state.nse.blocks)
			if (block.level > 0)
				block.storage_overlap_z = 3;
		state.SimInit();
	}
	catch (const std::runtime_error& e) {
		message = e.what();
	}
	CHECK_MESSAGE((
		message.find("wall-shared nesting requires F2C_SCHONHERR") != std::string::npos && message.find("F2C_LAGRAVA") != std::string::npos),
		fmt::format(
			"Lagrava guard: nested wall-shared faces under F2C_LAGRAVA hard-error at SimInit naming the strategy -- {}",
			message.empty() ? "no exception thrown" : fmt::format("threw: {}", message)
		)
	);
#else
	CHECK_MESSAGE(
		true,
		"Lagrava guard: default F2C_SCHONHERR build -- the guard is compiled out and the wall chain SimInit runs green "
		"(see the wall-chain mask census); the F2C_LAGRAVA arm is exercised by a strategy-flipped binary"
	);
#endif
}

// the commit-F 5-level channel chain at R = 1 (the regression lock of
// sim_AMR/amr_chain_solver.h's derivation: level 1 is the 2-level channel's
// anchor footprint, levels 2..4 telescope with insets >= 3 parent-level
// cells on every non-wall face and a gap-0 wall-shared z-min face per hop,
// holding every level's z-min face on the level-0 wall-candidate lane
// z = R+1; the level-0 domain is the channel's 64 x 16 x 16 lattice)
constexpr const char* five_level_channel_chain = "1 24 4 2 16 8 8\n"
												  "2 102 22 8 52 20 26\n"
												  "3 420 100 32 184 56 92\n"
												  "4 1704 424 128 688 176 344";

void test_five_level_channel_chain_creation()
{
	// the channel lattice at R = 1 (the chain is channel-specific: the
	// level-1 anchor footprint and every nested containment bound key on
	// this level-0 domain)
	lat_t lat;
	lat.global = typename lat_t::CoordinatesType(64, 16, 16);
	lat.physOrigin = point_t{0., 0., 0.};
	lat.physDl = 0.041 / 16;
	lat.physDt = 0.005 / 1.5e-5 * lat.physDl * lat.physDl;
	lat.physViscosity = 1.5e-5;

	const std::string id = fmt::format("test_amr_nesting_{}_channel5", pattern_name);
	StateLocal_AMR<NSE_CONFIG> state(id, MPI_COMM_WORLD, lat, "adios2.xml", /*periodic=*/TRAITS::bool3d{false, true, false}, /*max_level=*/4);
	if (! state.canCompute()) {
		CHECK_MESSAGE(false, "5-level channel chain setup: state.canCompute()");
		return;
	}

	LogCapture capture;
	std::string message;
	try {
		createAMRBlocks(state.nse, parseAMRConfig<NSE_CONFIG>(five_level_channel_chain));
	}
	catch (const std::runtime_error& e) {
		message = e.what();
	}
	CHECK_MESSAGE(
		message.empty(),
		fmt::format(
			"5-level channel chain: the derived chain (blocks 0..4, every z-min face wall-chained to level 1) passes the full "
			"V-suite ({})",
			message.empty() ? "no exception" : fmt::format("threw: {}", message)
		)
	);
	if (! message.empty())
		return;

	// level-0 blocks are created by the LBM constructor and are intentionally
	// not accounted in level_block_counts[0] (see test_three_level_creation)
	const std::vector<int> counts = state.nse.level_block_counts;
	CHECK_MESSAGE((
		state.nse.blocks.size() == 5 && counts == std::vector<int>({0, 1, 1, 1, 1})),
		fmt::format("5-level channel chain: block census -- {} blocks (one per level 0..4)", state.nse.blocks.size())
	);

	// hand-computed block geometry of five_level_channel_chain (amrParentFrameOrigin
	// / amrFineOffset / amrFineLocal per component)
	const std::vector<ChainBlockExpectation> expectations = {
		{1, {24, 4, 2}, {49, 9, 5}, {30, 14, 14}},
		{2, {51, 11, 4}, {103, 23, 9}, {50, 18, 24}},
		{3, {105, 25, 8}, {211, 51, 17}, {90, 26, 44}},
		{4, {213, 53, 16}, {427, 107, 33}, {170, 42, 84}},
	};
	bool geometry_ok = true;
	for (const ChainBlockExpectation& expected : expectations) {
		const std::vector<BLOCK*> blocks = state.nse.getBlocksAtLevel(expected.level);
		if (blocks.size() != 1) {
			geometry_ok = false;
			CHECK_MESSAGE(false, fmt::format("5-level channel chain: expected exactly one level-{} block (got {})", expected.level, blocks.size()));
			continue;
		}
		const BLOCK* fine = blocks.front();
		geometry_ok = geometry_ok && fine->global_offset == expected.global_offset && fine->offset == expected.offset
				   && fine->local == expected.local;
	}
	CHECK_MESSAGE(
		geometry_ok,
		"5-level channel chain: every fine block carries the chain solver's exact parent-frame global_offset, re-anchored "
		"fine offset and fine local"
	);

	CHECK_MESSAGE(
		capture.sink->warnings.empty(),
		fmt::format(
			"5-level channel chain: zero V-suite warnings emitted (every inset lands in the no-warning tier; got {})",
			capture.sink->warnings.size()
		)
	);
}

// the commit-G windbreak rod array (sim_AMR/amr_windbreak.h's layout and
// stamping, the same helper sim_AMR_channel rides) on the
// five_level_channel_chain level-4 block: the staggered two-row census
// matches the analytic cross-section integral (per-layer disc cells times
// the rod height, summed over the z rows the same way the tagging writes
// them), the array keeps its clearance to every footprint face except the
// wall plane it stands on, the 12 face wall rows stay fluid so the
// buildFineWallMasks census is unchanged (rods are interior obstacles, not
// face walls), and no parent block sees a rod cell (plan sec. 9 locked
// item 1). The layout guardrails reject the forbidden classes with named
// errors.
void test_windbreak_rod_census()
{
	// the channel lattice at R = 1 (the five_level_channel_chain fixture)
	lat_t lat;
	lat.global = typename lat_t::CoordinatesType(64, 16, 16);
	lat.physOrigin = point_t{0., 0., 0.};
	lat.physDl = 0.041 / 16;
	lat.physDt = 0.005 / 1.5e-5 * lat.physDl * lat.physDl;
	lat.physViscosity = 1.5e-5;

	const std::string id = fmt::format("test_amr_nesting_{}_windbreak", pattern_name);
	StateLocal_AMR<NSE_CONFIG> state(id, MPI_COMM_WORLD, lat, "adios2.xml", /*periodic=*/TRAITS::bool3d{false, true, false}, /*max_level=*/4);
	if (! state.canCompute()) {
		CHECK_MESSAGE(false, "windbreak rod census setup: state.canCompute()");
		return;
	}
	state.nse.allocateHostData();
	state.nse.allocateDeviceData();
	std::string message;
	try {
		createAMRBlocks(state.nse, parseAMRConfig<NSE_CONFIG>(five_level_channel_chain));
	}
	catch (const std::runtime_error& e) {
		message = e.what();
	}
	CHECK_MESSAGE(
		message.empty(),
		fmt::format(
			"windbreak rod census setup: the 5-level chain validates ({})",
			message.empty() ? "no exception" : fmt::format("threw: {}", message)
		)
	);
	if (! message.empty())
		return;

	// all-FLUID maps on every block (the marker's starting point; the
	// level-4 block is the layout's only target)
	for (auto& block : state.nse.blocks)
		block.resetMap(BC::GEO_FLUID);

	BLOCK* l4 = state.nse.getBlocksAtLevel(4).front();

	// the sim's defaults (d = 4, pitch = 16, height = 40, row spacing = 34)
	const WindbreakRodParams params;
	WindbreakLayout layout;
	try {
		layout = deriveWindbreakLayout(l4->local.x(), l4->local.y(), l4->local.z(), params);
	}
	catch (const std::runtime_error& e) {
		message = e.what();
	}
	CHECK_MESSAGE(
		message.empty(),
		fmt::format(
			"windbreak layout: derives on the level-4 footprint ({}x{}x{}) ({})",
			l4->local.x(),
			l4->local.y(),
			l4->local.z(),
			message.empty() ? "no exception" : fmt::format("threw: {}", message)
		)
	);
	if (! message.empty())
		return;

	// the locked layout expectation: row 1 at x = 32 with y axes {13, 29},
	// row 2 staggered to (66, 21); the 12-cell disc of d = 4; the analytic
	// census 3 rods * 12 cells/layer * 40 rows
	const bool layout_ok = layout.rods.size() == 3 && layout.disc.size() == 12 && layout.cells_per_rod == 480
						&& layout.cells_total == 1440 && layout.rods[0].axis_x == 32 && layout.rods[0].axis_y == 13
						&& layout.rods[0].row == 1 && layout.rods[1].axis_x == 32 && layout.rods[1].axis_y == 29
						&& layout.rods[1].row == 1 && layout.rods[2].axis_x == 66 && layout.rods[2].axis_y == 21
						&& layout.rods[2].row == 2;
	CHECK_MESSAGE(
		layout_ok,
		fmt::format(
			"windbreak layout: two staggered rows at the hand-computed axes (32,13) + (32,29) and (66,21), the 12-cell "
			"d = 4 disc, analytic census 3 * 12 * 40 = {} cells",
			layout.cells_total
		)
	);

	// stamp with the same helper the sim calls
	stampWindbreak(
		layout,
		[&](idx x, idx y, idx z)
		{
			l4->hmap(l4->offset.x() + x, l4->offset.y() + y, l4->offset.z() + z) = BC::GEO_WALL;
		}
	);

	// total census over the ring-inclusive window [-1, local+1)^3: the count
	// equals the analytic cross-section integral exactly (integer counts)
	long wall_total = 0;
	idx min_x = l4->local.x(), max_x = -1000, min_y = l4->local.y(), max_y = -1000, min_z = l4->local.z(), max_z = -1000;
	for (idx z = -1; z < l4->local.z() + 1; z++)
		for (idx y = -1; y < l4->local.y() + 1; y++)
			for (idx x = -1; x < l4->local.x() + 1; x++)
				if (l4->hmap(l4->offset.x() + x, l4->offset.y() + y, l4->offset.z() + z) == BC::GEO_WALL) {
					wall_total++;
					min_x = std::min(min_x, x);
					max_x = std::max(max_x, x);
					min_y = std::min(min_y, y);
					max_y = std::max(max_y, y);
					min_z = std::min(min_z, z);
					max_z = std::max(max_z, z);
				}
	CHECK_MESSAGE(
		wall_total == layout.cells_total,
		fmt::format(
			"windbreak census: {} tagged rod cells match the analytic cross-section integral {} (12-cell disc * 40 rows * "
			"3 rods)",
			wall_total,
			layout.cells_total
		)
	);

	// per-rod census inside each rod's disc window (the rods are disjoint by
	// pitch/row spacing >= diameter, so every window holds exactly its rod)
	bool per_rod_ok = true;
	for (const WindbreakRod& rod : layout.rods) {
		long count = 0;
		for (idx z = layout.z_first; z < layout.z_first + params.height; z++)
			for (idx y = rod.axis_y - params.diameter; y <= rod.axis_y + params.diameter; y++)
				for (idx x = rod.axis_x - params.diameter; x <= rod.axis_x + params.diameter; x++)
					if (l4->hmap(l4->offset.x() + x, l4->offset.y() + y, l4->offset.z() + z) == BC::GEO_WALL)
						count++;
		per_rod_ok = per_rod_ok && count == layout.cells_per_rod;
	}
	CHECK_MESSAGE(per_rod_ok, "windbreak census: every rod's window census matches the per-rod analytic count (12 * 40 = 480)");

	// the clearance rule on the observed tagged window: >= 8 cells off the x
	// faces, >= 4 off the y faces, the base sits on the wall row (z = -1)
	// and the top keeps >= 4 cells below the z-max face
	const bool clearance_ok = min_x >= 8 && max_x <= l4->local.x() - 1 - 8 && min_y >= params.clearance
						   && max_y <= l4->local.y() - 1 - params.clearance && min_z == layout.z_first
						   && max_z <= l4->local.z() - 1 - params.clearance;
	CHECK_MESSAGE(
		clearance_ok,
		fmt::format(
			"windbreak clearance: tagged window x [{}, {}], y [{}, {}], z [{}, {}] keeps the face margins (>= 8 on x, "
			">= 4 elsewhere; the wall plane is the only face the rods touch)",
			min_x,
			max_x,
			min_y,
			max_y,
			min_z,
			max_z
		)
	);

	// wall-chain masks unchanged: the 12 face wall rows (local -2 on a min
	// face, local+1 on a max face -- the planes State_AMR::buildFineWallMasks
	// keys on) carry no tagged cell after the stamping
	bool wall_rows_fluid = true;
	for (int a = 0; a < 3; a++) {
		const int b = (a + 1) % 3, c = (a + 2) % 3;
		for (const idx plane : {idx(-2), idx(l4->local[a] + 1)})
			for (idx ib = -2; ib < l4->local[b] + 2; ib++)
				for (idx ic = -2; ic < l4->local[c] + 2; ic++) {
					idx3d loc{0, 0, 0};
					loc[a] = plane;
					loc[b] = ib;
					loc[c] = ic;
					if (l4->hmap(l4->offset.x() + loc.x(), l4->offset.y() + loc.y(), l4->offset.z() + loc.z()) == BC::GEO_WALL)
						wall_rows_fluid = false;
				}
	}
	CHECK_MESSAGE(
		wall_rows_fluid,
		"windbreak wall rows: the stamping touched no face wall row (the local -2 / local+1 planes stay fluid, so "
		"buildFineWallMasks' census is unchanged -- rods are interior obstacles, not face walls)"
	);

	// no rod cell outside the finest level's map (locked item 1: the parents
	// treat the rod columns as plain fluid)
	bool parents_fluid = true;
	for (const auto& block : state.nse.blocks) {
		if (block.id == l4->id)
			continue;
		for (idx x = block.offset.x(); x < block.offset.x() + block.local.x(); x++)
			for (idx y = block.offset.y(); y < block.offset.y() + block.local.y(); y++)
				for (idx z = block.offset.z(); z < block.offset.z() + block.local.z(); z++)
					if (block.hmap(x, y, z) == BC::GEO_WALL)
						parents_fluid = false;
	}
	CHECK_MESSAGE(parents_fluid, "windbreak parents: not a single rod cell outside the finest level's map (locked item 1)");

	// guardrail rails: the forbidden classes die with named errors
	{
		WindbreakRodParams bad;
		bad.diameter = 2;
		message.clear();
		try {
			deriveWindbreakLayout(l4->local.x(), l4->local.y(), l4->local.z(), bad);
		}
		catch (const std::runtime_error& e) {
			message = e.what();
		}
		CHECK_MESSAGE(
			message.find("below the 3-cell minimum") != std::string::npos,
			fmt::format(
				"windbreak guardrail: diameter 2 is rejected ({})",
				message.empty() ? "no exception thrown" : fmt::format("threw: {}", message)
			)
		);
	}
	{
		// a footprint too narrow for the staggered second row
		message.clear();
		try {
			deriveWindbreakLayout(170, 18, 84, params);
		}
		catch (const std::runtime_error& e) {
			message = e.what();
		}
		CHECK_MESSAGE(
			message.find("stagger") != std::string::npos,
			fmt::format(
				"windbreak guardrail: a y span too narrow for the staggered second row is rejected ({})",
				message.empty() ? "no exception thrown" : fmt::format("threw: {}", message)
			)
		);
	}
	{
		// a height that would leave no clearance below the footprint top
		WindbreakRodParams tall;
		tall.height = 82;
		message.clear();
		try {
			deriveWindbreakLayout(l4->local.x(), l4->local.y(), l4->local.z(), tall);
		}
		catch (const std::runtime_error& e) {
			message = e.what();
		}
		CHECK_MESSAGE(
			message.find("partial-height") != std::string::npos,
			fmt::format(
				"windbreak guardrail: a non-partial height is rejected ({})",
				message.empty() ? "no exception thrown" : fmt::format("threw: {}", message)
			)
		);
	}
}

TEST_CASE("V-suite reject corpus") { test_vsuite_reject_corpus(); }
TEST_CASE("V-suite gap-2 warning") { test_vsuite_gap2_warning(); }
TEST_CASE("V-suite separation-2 warning") { test_vsuite_sep2_warning(); }
TEST_CASE("three-level creation") { test_three_level_creation(); }
TEST_CASE("three-level mark census") { test_three_level_mark_census(); }
TEST_CASE("two-level schedule census") { test_two_level_schedule_census(); }
TEST_CASE("three-level schedule census") { test_three_level_schedule_census(); }
TEST_CASE("three-level conservation smoke") { test_three_level_conservation_smoke(); }
TEST_CASE("wall-chain masks") { test_wall_chain_masks(); }
TEST_CASE("wall pedestal prisms") { test_wall_pedestal_prisms(); }
TEST_CASE("wall-chain fail-fast") { test_wall_chain_failfast(); }
TEST_CASE("Lagrava wall-chain guard") { test_wall_chain_lagrava_guard(); }
TEST_CASE("five-level chain creation") { test_five_level_channel_chain_creation(); }
TEST_CASE("windbreak rod census") { test_windbreak_rod_census(); }

TEST_SUITE_END();
