/*
 * Schönherr-2015 ch.7 AMR conversion -- registration/parity/conservation census
 * locks and buildCouplings geometry fingerprints (plan row 3 of
 * .omo/plans/schonherr-ch7-conversion.md; contract doc
 * docs/AMR-schonherr-ch7-target-contract.md sections 1.2 and 2).
 *
 * This file is the TDD land-first commit of the conversion: every expected
 * value below is computed from the RULING FORMULAS (cited next to each
 * expectation), never from the current runtime output. The new-geometry locks
 * are RED by design against the current band geometry and flip green when the
 * commits 4-7 geometry lands.
 *
 * Two doctest suites:
 * - "amr_schonherr_registration_formulas": pure formula-level sanity cases
 *   (no runtime geometry) that pin the census arithmetic itself. These stay
 *   GREEN at every commit.
 * - "amr_schonherr_registration_locks": runtime locks against real State_AMR
 *   fixtures (same mock-state idiom as tests/test_amr_subcycling.cu). The
 *   pytest wrapper (tests/unit/test_cpp_units.py) marks this suite's batch
 *   xfail(strict=True); the mark is removed at the commit-7 stage-1 gate.
 *
 * Fixture (shared by all runtime cases): 16^3 periodic domain with one
 * centered level-1 region, coarse footprint go = (4,4,4), gs = K = 8 cells
 * per axis, i.e. the "1 4 4 4 8 8 8" configuration from
 * tests/test_amr_subcycling.cu. Under the ruling (contract section 2.2) the
 * fine block becomes offset' = 2*go+1 = 9 and local' = 2K-2 = 14 with a
 * 2-deep overlap, keeping the stored extent 2K+2 = 18 rows per axis
 * unchanged. The census identities used throughout (contract section 2.4):
 *
 *   ring        = (K+2)^3 - (K-2)^3   = 784  (halo 488 + reactivated c=0 shell 296)
 *   GEO_NOTHING = (K-2)^3             = 216  (skin (K-2)^3-(K-4)^3 = 152 + deep (K-4)^3 = 64)
 *   C2F patches = 18^3 - 14^3         = 3088 fine destination cells
 *   F2C skin    = (K-2)^3 - (K-4)^3   = 152  (depth-1 shell; K=32 gives 5048)
 */

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <fmt/core.h>

#include <doctest/doctest.h>

#include "lbm3d/core.h"
#include "lbm3d/amr_state.h"

using TRAITS = TraitsSP;
using COLL = D3Q27_CUM<TRAITS, D3Q27_EQ_INV_CUM<TRAITS>>;
using NSE_CONFIG = LBM_CONFIG<
	TRAITS,
	D3Q27_KernelStruct,
	NSE_Data_ConstInflow<TRAITS>,
	COLL,
	typename COLL::EQ,
	D3Q27_STREAMING<TRAITS>,
	D3Q27_BC_All,
	D3Q27_MACRO_Default<TRAITS>>;

using idx = typename TRAITS::idx;
using idx3d = typename TRAITS::idx3d;
using lat_t = Lattice<3, typename TRAITS::real, idx>;
using BLOCK = LBM_BLOCK<NSE_CONFIG>;
using SyncDirection = TNL::Containers::SyncDirection;

namespace {

// fixture constants: footprint [go, go + K)^3 in parent-level coordinates
// (the "1 4 4 4 8 8 8" AMR configuration)
constexpr int K = 8;
constexpr idx GO = 4;

// census shell volumes from the ruling formulas (contract section 2.4):
// ring        = (K+2)^3 - (K-2)^3  (1-cell halo around the footprint PLUS the
//                                   reactivated footprint surface shell c=0)
// halo        = (K+2)^3 - K^3      (1-cell halo only, today's complete ring)
// surface     = K^3 - (K-2)^3      (footprint surface shell at depth 0, c=0)
// skin        = (K-2)^3 - (K-4)^3  (depth-1 shell, the F2C target c=1)
// geo_nothing = (K-2)^3            (covered cells kept frozen: skin + deep)
long ringCensus(long k)
{
	return (k + 2) * (k + 2) * (k + 2) - (k - 2) * (k - 2) * (k - 2);
}

long haloCensus(long k)
{
	return (k + 2) * (k + 2) * (k + 2) - k * k * k;
}

long surfaceCensus(long k)
{
	return k * k * k - (k - 2) * (k - 2) * (k - 2);
}

long skinCensus(long k)
{
	return (k - 2) * (k - 2) * (k - 2) - (k - 4) * (k - 4) * (k - 4);
}

long nothingCensus(long k)
{
	return (k - 2) * (k - 2) * (k - 2);
}

// the same 16^3 periodic box in physical units as tests/test_amr_subcycling.cu
// (nu_lb coarse 0.005, binary-halved on the fine level)
lat_t makeLattice()
{
	const int N = 16;
	const typename TRAITS::real LBM_VISCOSITY = 0.005;
	const typename TRAITS::real PHYS_HEIGHT = 0.41;
	const typename TRAITS::real PHYS_VISCOSITY = 1.5e-5;
	const typename TRAITS::real PHYS_DL = PHYS_HEIGHT / N;
	const typename TRAITS::real PHYS_DT = LBM_VISCOSITY / PHYS_VISCOSITY * PHYS_DL * PHYS_DL;

	lat_t lat;
	lat.global = typename lat_t::CoordinatesType(N, N, N);
	lat.physOrigin = typename TRAITS::point_t{0., 0., 0.};
	lat.physDl = PHYS_DL;
	lat.physDt = PHYS_DT;
	lat.physViscosity = PHYS_VISCOSITY;
	return lat;
}

// adios2.xml from the project root, anchored at this source file (the pytest
// wrapper runs the binary in a scratch CWD where the plain relative name
// "adios2.xml" does not exist)
std::string adiosConfigPath()
{
	const std::filesystem::path root = std::filesystem::path(__FILE__).parent_path().parent_path().parent_path();
	return (root / "adios2.xml").string();
}

// minimal State_AMR subclass (same idiom as tests/test_amr_subcycling.cu):
// pass-through constructor plus the pure-virtual output-name surface
template <typename NSE>
struct StateLock_AMR : State_AMR<NSE>
{
	template <typename... ARGS>
	StateLock_AMR(ARGS&&... args)
	: State_AMR<NSE>(std::forward<ARGS>(args)...)
	{}

	[[nodiscard]] std::vector<std::string> getOutputDataNames() const override
	{
		return {};
	}
};

// per-case fresh state with the shared fixture (each case scopes its own
// instance: the State constructor registers a global spdlog logger per
// instance, so two states must never be alive at the same time)
StateLock_AMR<NSE_CONFIG> makeState(const std::string& id)
{
	return StateLock_AMR<NSE_CONFIG>(
		id, MPI_COMM_WORLD, makeLattice(), adiosConfigPath(), /*periodic=*/TRAITS::bool3d{true, true, true}, /*max_level=*/1
	);
}

// create the shared level-1 fixture and run SimInit (allocation, boundary
// setup, markAMRInterface tagging, buildCouplings)
template <typename STATE>
void initFixture(STATE& state)
{
	REQUIRE(state.canCompute());
	createAMRBlocks(state.nse, parseAMRConfig<NSE_CONFIG>("1 4 4 4 8 8 8"));
	state.SimInit();
	REQUIRE(! state.nse.terminate);
	REQUIRE(state.couplings.size() == 1);
}

std::string str(const idx3d& v)
{
	return fmt::format("({},{},{})", v.x(), v.y(), v.z());
}

// lexicographic order on (origin, size) rectangles for set comparisons
struct Rect
{
	idx3d origin;
	idx3d size;

	bool operator<(const Rect& other) const
	{
		auto key = [](const Rect& r)
		{
			return std::make_tuple(r.origin.x(), r.origin.y(), r.origin.z(), r.size.x(), r.size.y(), r.size.z());
		};
		return key(*this) < key(other);
	}

	bool operator==(const Rect& other) const
	{
		return origin == other.origin && size == other.size;
	}
};

long rectVolume(const idx3d& size)
{
	return static_cast<long>(size.x()) * size.y() * size.z();
}

} // anonymous namespace

TEST_SUITE_BEGIN("amr_schonherr_registration_formulas");

// Pins the census arithmetic itself: every identity is a formula from contract
// section 2.4 / plan section 1.2 evaluated on the K=8 fixture and on the K=32
// TGV region. No runtime geometry involved -- these stay GREEN at all times.
TEST_CASE("census shell formulas (K=8 fixture, K=32 TGV region)")
{
	// ring = halo + reactivated c=0 shell (contract 2.4: 784 = 488 + 296)
	CHECK(ringCensus(K) == 784);
	CHECK(haloCensus(K) == 488);
	CHECK(surfaceCensus(K) == 296);
	CHECK(ringCensus(K) == haloCensus(K) + surfaceCensus(K));
	// GEO_NOTHING = c=1 skin shell + deep core (contract 2.4: 216 = 152 + 64)
	CHECK(nothingCensus(K) == 216);
	CHECK(skinCensus(K) == 152);
	CHECK(nothingCensus(K) == skinCensus(K) + 64 /* (K-4)^3 = 4^3 deep core */);
	// fixture excluded-volume recount (contract 2.3): 512 = K^3 today -> 216
	CHECK(K * K * K == 512);
	// C2F patch destination census (contract 2.4): fine rectangles of the
	// disjoint partition of stored(=2K+2=18 rows) minus interior(=local'=2K-2=14 rows)
	const long local2 = 2 * K - 2;	  // 14 (new local')
	const long stored = 2 * K + 2;	  // 18 (stored rows per axis)
	CHECK(2 * stored * stored == 648);			   // x-normal face: 2-thick normal, full stored tangent
	CHECK(local2 * 2 * stored == 504);			   // y-normal face: tangent x inset by one coarse row
	CHECK(local2 * 2 * local2 == 392);			   // z-normal face: tangent x/y inset
	CHECK(2 * (648 + 504 + 392) == 3088);		   // {648, 648, 504, 504, 392, 392}
	CHECK(stored * stored * stored - local2 * local2 * local2 == 3088);	 // 18^3 - 14^3
	// destination-region geometry split (contract 2.4): corner / edge / face interior
	CHECK(8 * 2 * 2 * 2 == 64);					   // 8 corner boxes of 2^3 cells
	CHECK(12 * local2 * 2 * 2 == 672);			   // 12 edges: (2K-2) cells long, 2x2 cross-section
	CHECK(6 * 2 * local2 * local2 == 2352);		   // 6 face interiors: 2-thick, (2K-2)^2
	CHECK(64 + 672 + 2352 == 3088);
	// F2C skin depth-1 shell (contract 2.4): K=8 -> 152, K=32 -> 5048
	CHECK(skinCensus(8) == 152);
	CHECK(skinCensus(32) == 5048);
	// TGV K=32 excluded-volume recount (contract 2.3): 32768 = K^3 -> 27000,
	// counted fine volume 32^3 -> 31^3 = 29791, net shift +2791
	CHECK(nothingCensus(32) == 27000);
	CHECK(nothingCensus(32) == skinCensus(32) + 21952 /* (K-4)^3 = 28^3 deep core */);
	CHECK(31 * 31 * 31 == 29791);
	CHECK(31 * 31 * 31 - nothingCensus(32) == 2791);
}

// The band-map parity invariant (plan section 2, T0'; contract section 2.2):
// under the new indexer old = new + 1, every band row keeps its fine-global
// coordinate fg, hence its (home, t) storage parity. Pure arithmetic, no
// runtime geometry -- stays GREEN.
TEST_CASE("parity row mapping old = new + 1")
{
	// band rows of plan section 1.1: (old local, new local) pairs
	// dest row 1: -1 -> -2; dest row 2: 0 -> -1; standard row 0: 1 -> 0;
	// F2C source rows: {2, 3} -> {1, 2}
	const std::pair<int, int> band_rows[] = {{-1, -2}, {0, -1}, {1, 0}, {2, 1}, {3, 2}};
	for (const auto& [old_local, new_local] : band_rows) {
		// old indexer: offset = 2*origin; new indexer: offset' = 2*origin + 1
		const long fg_old = 2L * GO + old_local;
		const long fg_new = (2L * GO + 1) + new_local;
		CHECK(fg_old == fg_new);
	}
	// stored-extent identity (contract 2.2): local' + 2*ov' = 14 + 4 = 18 = 2K+2
	CHECK(2 * K - 2 + 2 * 2 == 2 * K + 2);
	// first stored fine-global row identical under both indexers:
	// offset - ov = 2*go - 1 (old) = (2*go + 1) - 2 (new)
	CHECK(2L * GO - 1 == (2L * GO + 1) - 2);
}

TEST_SUITE_END();

TEST_SUITE_BEGIN("amr_schonherr_registration_locks");

// Census lock (contract 2.4). Expected ring = (K+2)^3 - (K-2)^3 = 784 (halo
// (K+2)^3 - K^3 = 488 plus the reactivated c=0 surface shell K^3 - (K-2)^3 =
// 296); expected GEO_NOTHING = (K-2)^3 = 216. The current geometry tags only
// the halo ring (488) and the whole footprint (512), so this lock is RED
// until the commit-6 retag.
TEST_CASE("ring + GEO_NOTHING census (K=8 fixture)")
{
	StateLock_AMR<NSE_CONFIG> state = makeState("schonherr_locks_census");
	initFixture(state);

	BLOCK* coarse = state.nse.getBlocksAtLevel(0).front();
	coarse->copyMapToHost();

	long ring_total = 0, ring_in_footprint = 0, ring_in_halo = 0;
	long nothing_total = 0, nothing_at_surface = 0, nothing_at_skin_depth = 0, nothing_deep = 0;
	for (idx z = 0; z < coarse->local.z(); z++)
		for (idx y = 0; y < coarse->local.y(); y++)
			for (idx x = 0; x < coarse->local.x(); x++) {
				const auto tag = coarse->hmap(x, y, z);
				// position classes against the footprint [GO, GO+K)^3 and its
				// c>=1 / c>=2 inset boxes (depth measured into the footprint)
				const bool in_fp = x >= GO && x < GO + K && y >= GO && y < GO + K && z >= GO && z < GO + K;
				const bool in_inset = x >= GO + 1 && x < GO + K - 1 && y >= GO + 1 && y < GO + K - 1 && z >= GO + 1 && z < GO + K - 1;
				const bool in_core = x >= GO + 2 && x < GO + K - 2 && y >= GO + 2 && y < GO + K - 2 && z >= GO + 2 && z < GO + K - 2;
				const bool surface = in_fp && ! in_inset;  // c=0 shell (the ring row 2 to be reactivated)
				const bool skin_pos = in_inset && ! in_core;  // c=1 depth-1 shell (the F2C skin target)
				const bool deep = in_core;					   // c>=2 deep frozen core
				if (tag == NSE_CONFIG::BC::GEO_AMR_INTERFACE) {
					ring_total++;
					if (in_fp)
						ring_in_footprint++;
					else
						ring_in_halo++;
				}
				if (tag == NSE_CONFIG::BC::GEO_NOTHING) {
					nothing_total++;
					if (surface)
						nothing_at_surface++;
					else if (skin_pos)
						nothing_at_skin_depth++;
					else if (deep)
						nothing_deep++;
				}
			}

	// ring decomposition (formulas: halo = (K+2)^3-K^3, c=0 shell = K^3-(K-2)^3)
	CHECK(ring_total == 784);			  // (K+2)^3 - (K-2)^3
	CHECK(ring_in_footprint == 296);	  // K^3 - (K-2)^3 (reactivated c=0 shell)
	CHECK(ring_in_halo == 488);			  // (K+2)^3 - K^3 (this part is already green)
	// GEO_NOTHING decomposition (formulas: skin = (K-2)^3-(K-4)^3, deep = (K-4)^3)
	CHECK(nothing_total == 216);		  // (K-2)^3
	CHECK(nothing_at_surface == 0);		  // the c=0 shell is reactivated to the ring
	CHECK(nothing_at_skin_depth == 152);  // (K-2)^3 - (K-4)^3
	CHECK(nothing_deep == 64);			  // (K-4)^3
}

// C2F patch destination census (contract 2.4). The fine-side rectangles of
// buildCouplings' halo patches must partition exactly the stored fine rows
// outside the simulated interior: total 3088 = 18^3 - 14^3 with the per-face
// family split {648, 648, 504, 504, 392, 392} and the corner/edge/face
// interior split 64/672/2352. Current code pushes the old 1-coarse-row halo
// (3904 fine cells nominal), so the lock is RED until commit 7.
TEST_CASE("C2F patch destination census + splits")
{
	StateLock_AMR<NSE_CONFIG> state = makeState("schonherr_locks_c2f_census");
	initFixture(state);

	const auto& coupling = state.couplings.front();
	const auto& patches = coupling.patches;
	CHECK(patches.size() == 6);

	// per-face census grouped by the interface normal (Left/Right = x family,
	// Bottom/Top = y, Back/Front = z)
	long total = 0;
	long family[3] = {0, 0, 0};
	// union of the fine rectangles for the corner/edge/face split and the
	// disjointness check (pure integer bookkeeping, no storage access)
	std::set<std::tuple<long, long, long>> union_cells;
	for (const auto& patch : patches) {
		const long vol = rectVolume(patch.fine_size);
		total += vol;
		const int fam = (patch.face == SyncDirection::Left || patch.face == SyncDirection::Right) ? 0
			: (patch.face == SyncDirection::Bottom || patch.face == SyncDirection::Top)			  ? 1
																									: 2;
		family[fam] += vol;
		for (long x = patch.fine_origin.x(); x < patch.fine_origin.x() + patch.fine_size.x(); x++)
			for (long y = patch.fine_origin.y(); y < patch.fine_origin.y() + patch.fine_size.y(); y++)
				for (long z = patch.fine_origin.z(); z < patch.fine_origin.z() + patch.fine_size.z(); z++)
					union_cells.emplace(x, y, z);
	}
	// family split (formulas: x face 2*(2K+2)^2 = 648, y face 2*(2K-2)*(2K+2)
	// = 504, z face 2*(2K-2)^2 = 392 -- per patch)
	if (patches.size() == 6) {
		CHECK(total == 3088);		// 18^3 - 14^3
		CHECK(family[0] == 2 * 648);
		CHECK(family[1] == 2 * 504);
		CHECK(family[2] == 2 * 392);
	}

	// disjoint partition: pushed cells equal the union size
	CHECK(total == static_cast<long>(union_cells.size()));

	// corner/edge/face interior split, anchored at the new interior frame
	// local' = 2K-2 = 14: an axis is "outside" when the row index is in the
	// 2-row overlap bands {..., -1} and {local', local'+1}
	const long local2 = 2 * K - 2;
	auto outside = [&](long c)
	{
		return c < 0 || c >= local2;
	};
	long by_class[4] = {0, 0, 0, 0};
	for (const auto& [x, y, z] : union_cells) {
		const int n_out = static_cast<int>(outside(x)) + static_cast<int>(outside(y)) + static_cast<int>(outside(z));
		by_class[n_out]++;
	}
	CHECK(by_class[3] == 64);	// 8 corner boxes of 2^3 cells
	CHECK(by_class[2] == 672);	// 12 edges: (2K-2) long, 2x2 cross-section
	CHECK(by_class[1] == 2352); // 6 face interiors: 2-thick, (2K-2)^2
	CHECK(by_class[0] == 0);	// no destination cell inside the simulated interior
}

// F2C skin census (contract 2.4, plan T4): the interior_patches pushed by
// buildCouplings must partition the depth-1 shell INSIDE the footprint,
// (K-2)^3 - (K-4)^3 = 152 cells, with the per-family rectangle split
// {36, 36, 24, 24, 16, 16}. Current code pushes the depth-0 footprint surface
// (8^3 - 6^3 = 296), so the lock is RED until commit 7.
TEST_CASE("F2C skin destination census (depth-1 shell)")
{
	StateLock_AMR<NSE_CONFIG> state = makeState("schonherr_locks_f2c_census");
	initFixture(state);

	const auto& coupling = state.couplings.front();
	const auto& rects = coupling.interior_patches;
	CHECK(rects.size() == 6);

	long total = 0;
	long family[3] = {0, 0, 0};  // grouped by the rectangle's 1-cell-thick axis
	std::set<std::tuple<long, long, long>> union_cells;
	for (const auto& patch : rects) {
		const long vol = rectVolume(patch.coarse_size);
		total += vol;
		int fam = -1;
		if (patch.coarse_size.x() == 1 && patch.coarse_size.y() != 1 && patch.coarse_size.z() != 1)
			fam = 0;
		else if (patch.coarse_size.y() == 1 && patch.coarse_size.x() != 1 && patch.coarse_size.z() != 1)
			fam = 1;
		else if (patch.coarse_size.z() == 1 && patch.coarse_size.x() != 1 && patch.coarse_size.y() != 1)
			fam = 2;
		REQUIRE(fam >= 0);
		family[fam] += vol;
		for (long x = patch.coarse_origin.x(); x < patch.coarse_origin.x() + patch.coarse_size.x(); x++)
			for (long y = patch.coarse_origin.y(); y < patch.coarse_origin.y() + patch.coarse_size.y(); y++)
				for (long z = patch.coarse_origin.z(); z < patch.coarse_origin.z() + patch.coarse_size.z(); z++)
					union_cells.emplace(x, y, z);
	}
	CHECK(total == 152);	  // (K-2)^3 - (K-4)^3 depth-1 shell
	CHECK(total == static_cast<long>(union_cells.size()));	// disjoint partition
	// per-family split (formulas from the begin+1 / tangent insets+2 idiom of
	// plan section 2-T4: x face (K-2)^2 = 36, y face (K-4)(K-2) = 24,
	// z face (K-4)^2 = 16 -- per rectangle)
	CHECK(family[0] == 2 * 36);
	CHECK(family[1] == 2 * 24);
	CHECK(family[2] == 2 * 16);
}

// Parity lock (contract 2.2, plan T0'). Under the re-anchored indexer the
// fine block of the fixture has offset' = 2*origin+1 = 9 and local' = 2K-2 =
// 14 per axis, while the stored extent 2K+2 = 18 and the first stored
// fine-global row 2*go-1 = 7 stay invariant. The current block has offset 8
// and local 16, so the identities are RED until commit 4.
TEST_CASE("fine indexer parity identities under the re-anchor")
{
	StateLock_AMR<NSE_CONFIG> state = makeState("schonherr_locks_parity");
	REQUIRE(state.canCompute());
	createAMRBlocks(state.nse, parseAMRConfig<NSE_CONFIG>("1 4 4 4 8 8 8"));

	const std::vector<BLOCK*> level1 = state.nse.getBlocksAtLevel(1);
	REQUIRE(level1.size() == 1);
	BLOCK* fine = level1.front();

	// re-anchored indexer (ruling: offset_fine' = 2*origin + 1, local_fine'
	// = 2K - 2) -- RED against the current (8, 8, 8) / (16, 16, 16) geometry
	CHECK(fine->offset == idx3d{2 * GO + 1, 2 * GO + 1, 2 * GO + 1});
	CHECK(fine->local == idx3d{2 * K - 2, 2 * K - 2, 2 * K - 2});

	// permanence guards (green under BOTH indexers): total stored extent and
	// the first stored fine-global row are invariant across the re-anchor
	CHECK(fine->local.x() + 2 * fine->df_overlap_X() == 2 * K + 2);
	CHECK(fine->local.y() + 2 * fine->df_overlap_Y() == 2 * K + 2);
	CHECK(fine->local.z() + 2 * fine->df_overlap_Z() == 2 * K + 2);
	CHECK(fine->offset.x() - fine->df_overlap_X() == 2 * GO - 1);
	CHECK(fine->offset.y() - fine->df_overlap_Y() == 2 * GO - 1);
	CHECK(fine->offset.z() - fine->df_overlap_Z() == 2 * GO - 1);

	// band-row fg invariance assembled from the runtime values: for every band
	// row of the old = new + 1 map, the fine-global coordinate must coincide
	// (the (home, t) storage parity travels with fg)
	const std::pair<int, int> band_rows[] = {{-1, -2}, {0, -1}, {1, 0}, {2, 1}, {3, 2}};
	bool fg_consistent = true;
	for (const auto& [old_local, new_local] : band_rows) {
		const long fg_old = 2L * GO + old_local;
		const long fg_new = static_cast<long>(fine->offset.x()) + new_local;
		if (fg_old != fg_new)
			fg_consistent = false;
	}
	CHECK(fg_consistent);
}

// Conservation recount (contract 2.3, plan T4). Two sub-locks:
// (i)  the excluded-volume census drops 512 (K^3, today) -> 216 ((K-2)^3):
//      the reactivated c=0 shell must be counted again;
// (ii) the positive-inclusion sentinel: injecting sentinel rho ONLY into the
//      296 reactivated ring cells must shift the total counted mass by
//      exactly 296 * sentinel (volume factor 1 on level 0). Under the current
//      geometry those cells are GEO_NOTHING and excluded, so the shift is 0
//      and both locks are RED until the commit-6 retag.
TEST_CASE("conservation recount: excluded census + sentinel positive inclusion")
{
	StateLock_AMR<NSE_CONFIG> state = makeState("schonherr_locks_conservation");
	initFixture(state);

	BLOCK* coarse = state.nse.getBlocksAtLevel(0).front();
	coarse->copyMapToHost();
	coarse->copyMacroToHost();

	// counted mass BEFORE the injection (the metric refreshes host mirrors
	// from the device itself; uniform rho = 1 initial state)
	const AMRConservationStats s0 = state.computeConservationStats();

	// reactivated ring cells = the c=0 footprint surface shell, identified by
	// POSITION (never by tag -- the formula K^3 - (K-2)^3 = 296)
	const double sentinel = 177.0;
	long injected = 0;
	for (idx z = 0; z < coarse->local.z(); z++)
		for (idx y = 0; y < coarse->local.y(); y++)
			for (idx x = 0; x < coarse->local.x(); x++) {
				const bool in_fp = x >= GO && x < GO + K && y >= GO && y < GO + K && z >= GO && z < GO + K;
				const bool in_inset = x >= GO + 1 && x < GO + K - 1 && y >= GO + 1 && y < GO + K - 1 && z >= GO + 1 && z < GO + K - 1;
				if (! (in_fp && ! in_inset))
					continue;
				coarse->hmacro(NSE_CONFIG::MACRO::e_rho, x, y, z) = static_cast<typename TRAITS::real>(sentinel);
				coarse->hmacro(NSE_CONFIG::MACRO::e_vx, x, y, z) = 0;
				coarse->hmacro(NSE_CONFIG::MACRO::e_vy, x, y, z) = 0;
				coarse->hmacro(NSE_CONFIG::MACRO::e_vz, x, y, z) = 0;
				injected++;
			}
	CHECK(injected == 296);	 // K^3 - (K-2)^3 footprint surface shell
	coarse->copyMacroToDevice();

	// excluded-volume census (tag-keyed, the unchanged code path of
	// computeConservationStats): 512 today -> (K-2)^3 = 216 after the retag
	long excluded = 0;
	for (idx z = 0; z < coarse->local.z(); z++)
		for (idx y = 0; y < coarse->local.y(); y++)
			for (idx x = 0; x < coarse->local.x(); x++)
				if (coarse->hmap(x, y, z) == NSE_CONFIG::BC::GEO_NOTHING)
					excluded++;
	CHECK(excluded == 216);	 // (K-2)^3 = skin 152 + deep 64

	// positive-inclusion sentinel: the mass shift equals exactly the ring
	// shell sum (the metric's OpenMP atomic summation reassociates, so the
	// comparison carries the house 1e-6 relative tolerance)
	const AMRConservationStats s1 = state.computeConservationStats();
	const double shift = s1.total_mass - s0.total_mass;
	const double expected_shift = sentinel * 296;  // (K^3 - (K-2)^3) * sentinel
	CHECK(std::abs(shift - expected_shift) <= 1e-6 * std::max(1.0, std::abs(expected_shift)));

	// permanence rail (green under both geometries): the metric equals the
	// direct tag-keyed reference sum that excludes exactly the GEO_NOTHING
	// cells of every block (the code path itself is unchanged by the ruling)
	double ref_mass = 0;
	for (const auto& block : state.nse.blocks) {
		const double volume_factor = std::pow(0.5, 3.0 * block.level);
		for (idx x = 0; x < block.local.x(); x++)
			for (idx z = 0; z < block.local.z(); z++)
				for (idx y = 0; y < block.local.y(); y++) {
					if (block.hmap(x, y, z) == NSE_CONFIG::BC::GEO_NOTHING)
						continue;
					ref_mass += static_cast<double>(block.hmacro(NSE_CONFIG::MACRO::e_rho, x, y, z)) * volume_factor;
				}
	}
	CHECK(std::abs(s1.total_mass - ref_mass) <= 1e-6 * std::max({1.0, std::abs(s1.total_mass), std::abs(ref_mass)}));
}

// Geometry fingerprint of the halo (C2F) patches (plan T5/T6, Oracle F2):
// the exact coarse and fine rectangles that commit 7's buildCouplings must
// push, derived from the ruling formulas -- NEVER from the runtime output.
// Coarse: 2-thick face-normal rectangles partitioning the ring (total 784);
// fine: the disjoint partition of contract 2.4 (total 3088), written once
// each. RED against the current 1-thick halo geometry until commit 7.
TEST_CASE("halo patch rectangle fingerprint (buildCouplings)")
{
	StateLock_AMR<NSE_CONFIG> state = makeState("schonherr_locks_halo_fingerprint");
	initFixture(state);

	const auto& coupling = state.couplings.front();
	const auto& patches = coupling.patches;
	CHECK(patches.size() == 6);

	// expected rectangles from the ruling formulas (footnotes):
	// min x face: coarse [go-1, go+1) x [go-1, go+K+1)^2, fine [-2, 0) x [-2, 2K+2)^2
	// min y face: coarse [go+1, go+K-1) x [go-1, go+1) x [go-1, go+K+1),
	//             fine [0, 2K-2) x [-2, 0) x [-2, 2K+2)
	// min z face: coarse [go+1, go+K-1)^2 x [go-1, go+1),
	//             fine [0, 2K-2)^2 x [-2, 0); max faces mirrored
	struct Expect
	{
		SyncDirection face;
		const char* name;
		idx3d coarse_origin, coarse_size, fine_origin, fine_size;
	};
	const idx S = 2 * K + 2;   // stored fine rows per axis (18)
	const idx L2 = 2 * K - 2;  // new local' (14)
	const std::vector<Expect> expected = {
		{SyncDirection::Left, "x-min", {GO - 1, GO - 1, GO - 1}, {2, K + 2, K + 2}, {-2, -2, -2}, {2, S, S}},
		{SyncDirection::Right, "x-max", {GO + K - 1, GO - 1, GO - 1}, {2, K + 2, K + 2}, {L2, -2, -2}, {2, S, S}},
		{SyncDirection::Bottom, "y-min", {GO + 1, GO - 1, GO - 1}, {K - 2, 2, K + 2}, {0, -2, -2}, {L2, 2, S}},
		{SyncDirection::Top, "y-max", {GO + 1, GO + K - 1, GO - 1}, {K - 2, 2, K + 2}, {0, L2, -2}, {L2, 2, S}},
		{SyncDirection::Back, "z-min", {GO + 1, GO + 1, GO - 1}, {K - 2, K - 2, 2}, {0, 0, -2}, {L2, L2, 2}},
		{SyncDirection::Front, "z-max", {GO + 1, GO + 1, GO + K - 1}, {K - 2, K - 2, 2}, {0, 0, L2}, {L2, L2, 2}},
	};

	// map the runtime patches by their face tag
	std::map<SyncDirection, std::size_t> by_face;
	for (std::size_t i = 0; i < patches.size(); i++)
		by_face[patches[i].face] = i;
	CHECK(by_face.size() == 6);

	for (const Expect& e : expected) {
		const auto it = by_face.find(e.face);
		if (it == by_face.end()) {
			FAIL_CHECK(fmt::format("halo fingerprint: no patch with face {}", e.name));
			continue;
		}
		const auto& patch = patches[it->second];
		INFO(fmt::format("face {} coarse origin: expected {}, got {}", e.name, str(e.coarse_origin), str(patch.coarse_origin)));
		CHECK(patch.coarse_origin == e.coarse_origin);
		INFO(fmt::format("face {} coarse size: expected {}, got {}", e.name, str(e.coarse_size), str(patch.coarse_size)));
		CHECK(patch.coarse_size == e.coarse_size);
		INFO(fmt::format("face {} fine origin: expected {}, got {}", e.name, str(e.fine_origin), str(patch.fine_origin)));
		CHECK(patch.fine_origin == e.fine_origin);
		INFO(fmt::format("face {} fine size: expected {}, got {}", e.name, str(e.fine_size), str(patch.fine_size)));
		CHECK(patch.fine_size == e.fine_size);
	}
}

// Geometry fingerprint of the F2C skin rectangles (plan T4/T6): the exact
// depth-1 shell partition inside the footprint (begin+1, tangent insets+2
// idiom; total 152 = (K-2)^3 - (K-4)^3). RED against the current depth-0
// surface partition (296) until commit 7.
TEST_CASE("skin rectangle fingerprint (buildCouplings)")
{
	StateLock_AMR<NSE_CONFIG> state = makeState("schonherr_locks_skin_fingerprint");
	initFixture(state);

	const auto& coupling = state.couplings.front();
	const auto& rects = coupling.interior_patches;
	CHECK(rects.size() == 6);

	// expected rectangles from the ruling formulas (plan section 2-T4):
	// min x face: [go+1, go+2) x [go+1, go+K-1)^2
	// min y face: [go+2, go+K-2) x [go+1, go+2) x [go+1, go+K-1)
	// min z face: [go+2, go+K-2)^2 x [go+1, go+2); max faces mirrored
	const std::vector<Rect> expected = {
		{{GO + 1, GO + 1, GO + 1}, {1, K - 2, K - 2}},
		{{GO + K - 2, GO + 1, GO + 1}, {1, K - 2, K - 2}},
		{{GO + 2, GO + 1, GO + 1}, {K - 4, 1, K - 2}},
		{{GO + 2, GO + K - 2, GO + 1}, {K - 4, 1, K - 2}},
		{{GO + 2, GO + 2, GO + 1}, {K - 4, K - 4, 1}},
		{{GO + 2, GO + 2, GO + K - 2}, {K - 4, K - 4, 1}},
	};

	std::vector<Rect> actual;
	for (const auto& patch : rects)
		actual.push_back({patch.coarse_origin, patch.coarse_size});
	std::sort(actual.begin(), actual.end());
	std::vector<Rect> want = expected;
	std::sort(want.begin(), want.end());
	if (actual != want) {
		std::string got;
		for (const Rect& r : actual)
			got += fmt::format(" origin {} size {};", str(r.origin), str(r.size));
		FAIL_CHECK(fmt::format("skin fingerprint mismatch:{}", got));
	}
	CHECK(actual == want);
}

TEST_SUITE_END();
