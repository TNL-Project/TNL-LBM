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
//
// The streaming pattern is selected at compile time (AB_PATTERN/AA_PATTERN
// from tests/CMakeLists.txt), everything is single-rank. Shared fixture
// machinery (lattice factory, report) comes from tests/amr_test_fixture.h.

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include <spdlog/sinks/base_sink.h>
#include <spdlog/spdlog.h>

#include "amr_test_fixture.h"

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
			report(false, fmt::format("V-suite corpus {} setup: state.canCompute()", kase.name));
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
		report(
			message == kase.expected_message,
			fmt::format(
				"V-suite corpus {}: rejected with the rule-specific message verbatim ({})",
				kase.name,
				message.empty() ? "no exception thrown" : fmt::format("threw: {}", message)
			)
		);
		report(
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
		report(false, "V-suite gap-2 warning setup: state.canCompute()");
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
	report(
		message.empty() && state.nse.blocks.size() == 3,
		fmt::format(
			"V-suite gap-2 floor: the footprint inset exactly 2 parent-level cells is accepted and the block is created ({})",
			message.empty() ? "no exception" : fmt::format("threw: {}", message)
		)
	);
	report(
		capture.sink->warnings.size() == 3
			&& capture.hasWarning("telescoping gap of 2 parent-level cells on the x-min face is below the recommended 3")
			&& capture.hasWarning("the parent's fine-to-coarse transfer windows will read coupling-authored ring/skin cells"),
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
		report(false, "V-suite separation-2 warning setup: state.canCompute()");
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
	report(
		message.empty() && state.nse.blocks.size() == 3,
		fmt::format(
			"V-suite separation floor: siblings exactly 2 parent-level cells apart are accepted and both blocks are created ({})",
			message.empty() ? "no exception" : fmt::format("threw: {}", message)
		)
	);
	report(
		capture.sink->warnings.size() == 1
			&& capture.hasWarning("separated by exactly 2 parent-level cells")
			&& capture.hasWarning("below the recommended 3"),
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
		report(false, "3-level creation setup: state.canCompute()");
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
	report(
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
	report(
		state.nse.blocks.size() == 4 && counts == std::vector<int>({0, 1, 1, 1}),
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
			report(false, fmt::format("3-level creation: expected exactly one level-{} block (got {})", expected.level, blocks.size()));
			continue;
		}
		const BLOCK* fine = blocks.front();
		geometry_ok = geometry_ok && fine->global_offset == expected.global_offset && fine->offset == expected.offset
				   && fine->local == expected.local;
	}
	report(
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
	report(
		scaling_ok,
		fmt::format(
			"3-level creation: per-level physDt/physDl halve and lattice viscosity doubles per level (nu level1..3 = {:.3e}, "
			"{:.3e}, {:.3e})",
			static_cast<double>(state.nse.getBlocksAtLevel(1).front()->lat_local.lbmViscosity()),
			static_cast<double>(state.nse.getBlocksAtLevel(2).front()->lat_local.lbmViscosity()),
			static_cast<double>(state.nse.getBlocksAtLevel(3).front()->lat_local.lbmViscosity())
		)
	);

	report(
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
		report(false, "3-level mark census setup: state.canCompute()");
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
	report(
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
		report(
			mismatches == 0 && interface_count == ref.interface_count && nothing_count == ref.nothing_count,
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
		report(
			level0->hmap(12, 16, 10) == BC::GEO_NOTHING && level0->hmap(8, 0, 8) == BC::GEO_AMR_INTERFACE,
			"3-level mark census: level-0 spot checks -- deep footprint cell frozen, halo cell tagged"
		);
	}
	{
		BLOCK* level1 = state.nse.getBlocksAtLevel(1).front();
		report(
			level1->hmap(20, 30, 10) == BC::GEO_NOTHING && level1->hmap(20, 30, 16) == BC::GEO_AMR_INTERFACE,
			"3-level mark census: level-1 spot checks -- deep footprint cell frozen, halo cell tagged"
		);
	}
	{
		BLOCK* level2 = state.nse.getBlocksAtLevel(2).front();
		report(
			level2->hmap(42, 10, 18) == BC::GEO_NOTHING && level2->hmap(42, 10, 20) == BC::GEO_AMR_INTERFACE,
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
	report(idempotent, "3-level mark census: markAMRInterface re-invocation is idempotent on the 3-level chain");

	report(
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
		report(false, fmt::format("{} setup: state.canCompute()", label));
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
	report(
		message.empty() && ! state.nse.terminate,
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
	report(init_ok, fmt::format("{} SimInit: level-ascending initial c2f cascade ({} events, all at rotation 0)", label, state.events.size()));
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
	report(
		cycles_ok && state.nse.iterations == 3,
		fmt::format(
			"{}: 3 cycles match the {}-event advancePair expansion per cycle (rotations, f2c next-parity/indices, counters){}",
			label,
			table_size,
			failure.empty() ? "" : fmt::format(" -- first failure: {}", failure)
		)
	);
	report(! state.nse.terminate, fmt::format("{}: no termination during the census run", label));
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
		report(false, "3-level conservation smoke setup: state.canCompute()");
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
	report(
		message.empty() && ! state.nse.terminate,
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
	report(
		finite_ok,
		fmt::format(
			"3-level conservation smoke: all conservation entries finite (no NaN) over 20 cycles{}",
			failure.empty() && finite_ok ? "" : fmt::format(" -- {}", failure)
		)
	);
	report(
		consistent_ok,
		fmt::format(
			"3-level conservation smoke: nested metric matches the GEO_NOTHING-excluding reference at every checkpoint{}",
			failure.empty() && consistent_ok ? "" : fmt::format(" -- {}", failure)
		)
	);
	report(
		stable_ok && mass_earliest > 0,
		fmt::format("3-level conservation smoke: global mass stable over 20 coupled cycles (relative drift {:.6e} over cycles 5 -> 20)", drift_final)
	);
	report(state.nse.iterations == 20 && ! state.nse.terminate, "3-level conservation smoke: 20 coupled cycles, no termination");
}

int main(int argc, char** argv)
{
	TNLMPI_INIT mpi(argc, argv);

	if (TNL::MPI::GetSize(MPI_COMM_WORLD) != 1) {
		fmt::println("RESULT: AMR nesting tests are single-rank only (nproc = {})", TNL::MPI::GetSize(MPI_COMM_WORLD));
		return 1;
	}

	fmt::println("AMR nesting validation tests (streaming pattern: {})", pattern_name);

	test_vsuite_reject_corpus();
	test_vsuite_gap2_warning();
	test_vsuite_sep2_warning();
	test_three_level_creation();
	test_three_level_mark_census();
	test_two_level_schedule_census();
	test_three_level_schedule_census();
	test_three_level_conservation_smoke();

	if (g_failures == 0) {
		fmt::println("RESULT: all AMR nesting tests passed");
		return 0;
	}
	fmt::println("RESULT: {} AMR nesting check(s) FAILED", g_failures);
	return 1;
}
