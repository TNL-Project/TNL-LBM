/*
 * Unit tests for `LBM_BLOCK<CONFIG>::updateOutflowPassRegion()` — the greedy
 * rectangle cover of outflow-pass sites in the host map.
 *
 * The cover enforces a minimum box extent: each box side is either
 * a single cell (the degenerate mask direction, e.g. the wall-normal of
 * a plane outlet) or at least `min_outflow_box_extent` (32) cells, clipped
 * to the local grid; padded cells early-out on the kernel's per-cell
 * `isOutflowPassBC` check. Boxes that overlap after padding are merged
 * into their bounding box until pairwise disjoint. A cover with more than
 * `max_outflow_boxes` (64) boxes is merge-reduced to the cap by repeatedly
 * merging the pair whose bounding box adds the least dead volume, which
 * keeps the total covered volume approximately minimal at the cost of
 * dropping sides below the min extent.
 *
 * ## Cases
 *
 * - `empty` — no outflow sites -> no boxes
 * - `full-plane` — one full plane >= 32x32 -> 1 box, no padding needed
 * - `two-windows` — two distant windows -> pads stay disjoint: 2 boxes
 * - `proximity-merge` — two windows closer than the padding radius ->
 *   padded boxes overlap -> merged into 1 bounding box
 * - `clip-slide` — window near the domain corner -> padding slides to fit
 *   the grid in both tangential dims
 * - `random-11`, `random-22`, `random-33` — seeded random masks on the
 *   (8,8,4) grid (small-grid padding), and `random-44`, `random-55`,
 *   `random-66` on the (4,96,48) grid (realistic padding) -> invariants:
 *   pairwise disjoint boxes, mask covered by the box union, at most
 *   `max_outflow_boxes` boxes, exclusive ends, begin <= end per dim, boxes
 *   inside the local domain, every side 1 or >= min extent
 * - `checkerboard` — >64 one-cell components (single-cell sides are
 *   exempt from padding) -> merged down to exactly 64 boxes covering
 *   strictly fewer cells than the legacy single bounding box
 * - `offset-block` — non-zero block offset -> mask read in GLOBAL
 *   indexing, padded boxes reported in LOCAL indexing
 * - `confined-checkerboard` — >64 components in a sub-region -> same
 *   merge-reduction as checkerboard (count 64, beats the legacy bbox)
 * - `z-split` — same rect at two distant z-ranges -> no merge across z,
 *   pads stay disjoint: 2 boxes
 * - `boundary-64` — exactly 64 one-cell components -> 64 boxes, NO
 *   merging (threshold is a strict >)
 * - `boundary-65` — 65 components -> merged down to exactly 64 boxes by
 *   the cheapest merge: 63 one-cell boxes + 1 three-cell box
 *
 * ## Running
 *
 * Each case is a doctest TEST_CASE registered under the same kebab-case
 * name within the "outflowcover" TEST_SUITE. Running the binary without
 * arguments executes all cases, `--test-suite=outflowcover` filters to the
 * suite. The pytest driver groups cases by suite and comma-joins selected
 * names into a single `--test-case=` flag (doctest's flag is last-flag-wins),
 * so each pytest item runs one suite in one subprocess. On a failed check
 * the mask and the box list are dumped to stderr (see dumpContext), which
 * pytest surfaces in full in the failure report.
 */

#include <array>
#include <cstdint>
#include <random>
#include <vector>

#include <fmt/core.h>

// the unit-test binary's main() lives in doctest_main.cu (MPI initialization)
#include <doctest/doctest.h>

#include "lbm3d/lbm_data.h"
#include "lbm3d/d2q9/bc.h"
#include "lbm3d/d2q9/col_srt.h"
#include "lbm3d/d2q9/macro.h"

// exactly one streaming header must be included
#ifdef AA_PATTERN
	#include "lbm3d/d2q9/streaming_AA.h"
#endif
#ifdef AB_PATTERN
	#include "lbm3d/d2q9/streaming_AB.h"
#endif

#include "lbm3d/lbm_block.h"

using TRAITS = Traits<float, double, int>;
using COLL = D2Q9_SRT<TRAITS>;
using CONFIG =
	LBM_CONFIG<TRAITS, D2Q9_KernelStruct, NSE_Data<TRAITS>, COLL, typename COLL::EQ, D2Q9_STREAMING<TRAITS>, D2Q9_BC_All, D2Q9_MACRO_Default<TRAITS>>;

using BLOCK = LBM_BLOCK<CONFIG>;
using idx = typename TRAITS::idx;
using idx3d = typename TRAITS::idx3d;
using BC = typename CONFIG::BC;

// a box {xb, yb, zb, xe, ye, ze} with EXCLUSIVE ends in LOCAL indices,
// mirroring OutflowBox's begin/end fields (TestBox[0..2] = begin, TestBox[3..5] = end)
using TestBox = std::array<idx, 6>;

// reference masks are always defined over the LOCAL grid; fillBlock writes
// them into the host map at GLOBAL indices offset + local indices
static BLOCK makeBlock(const idx3d& global, const idx3d& local, const idx3d& offset)
{
	return {MPI_COMM_WORLD, global, local, offset, 0};
}

static void fillBlock(BLOCK& block, const std::vector<std::uint8_t>& ref, const idx3d& local, const idx3d& offset)
{
	block.hmap.setValue(BC::GEO_FLUID);
	for (idx z = 0; z < local.z(); z++)
		for (idx y = 0; y < local.y(); y++)
			for (idx x = 0; x < local.x(); x++)
				if (ref[(z * local.y() + y) * local.x() + x] != 0) {
					if ((x + y + z) % 2 != 0)
						block.hmap(offset.x() + x, offset.y() + y, offset.z() + z) = BC::GEO_OUTFLOW_RIGHT;
					else
						block.hmap(offset.x() + x, offset.y() + y, offset.z() + z) = BC::GEO_OUTFLOW_RIGHT_INTERP;
				}
	block.updateOutflowPassRegion();
}

// dumps the diagnostic context of a failed check (reference mask + cover
// boxes) to stderr; pytest surfaces the full stderr in the failure report
static void dumpContext(const BLOCK& block, const std::vector<std::uint8_t>& ref, const idx3d& local)
{
	fmt::println(stderr, "mask (z-slices over the local grid, 1=outflow):");
	for (idx z = 0; z < local.z(); z++) {
		fmt::println(stderr, " z={}:", z);
		for (idx y = 0; y < local.y(); y++) {
			for (idx x = 0; x < local.x(); x++)
				fmt::print(stderr, " {}", ref[(z * local.y() + y) * local.x() + x]);
			fmt::println(stderr, "");
		}
	}
	fmt::println(stderr, "boxes ({}):", block.outflow_boxes.size());
	for (const auto& b : block.outflow_boxes)
		fmt::println(stderr, "  begin=({},{},{}) end=({},{},{})", b.begin.x(), b.begin.y(), b.begin.z(), b.end.x(), b.end.y(), b.end.z());
}

static void requireBoxes(const BLOCK& block, const std::vector<TestBox>& expect, const std::vector<std::uint8_t>& ref, const idx3d& local)
{
	if (block.outflow_boxes.size() != expect.size()) {
		dumpContext(block, ref, local);
		FAIL("unexpected box count");
	}
	for (std::size_t i = 0; i < expect.size(); i++) {
		const auto& b = block.outflow_boxes[i];
		const auto& e = expect[i];
		if (b.begin.x() != e[0] || b.begin.y() != e[1] || b.begin.z() != e[2] || b.end.x() != e[3] || b.end.y() != e[4] || b.end.z() != e[5]) {
			dumpContext(block, ref, local);
			FAIL("box mismatch (order or extents)");
		}
	}
}

// disjoint + mask covered by boxes + min-extent sides + exclusive ends +
// begin<=end + inside-local-domain invariants (box coordinates are LOCAL
// indices); padding makes the box union a superset of the mask, so coverage
// is checked one-way, and the min-extent side check is skipped at exactly
// max_outflow_boxes boxes, loose proxy for the merge-reduction having fired
// (merging exceeds max_outflow_boxes boxes with sequential pairwise fusion
// shrink sides below the extent)
static void checkInvariants(const BLOCK& block, const std::vector<std::uint8_t>& ref, const idx3d& local)
{
	const idx ncells = local.x() * local.y() * local.z();
	if ((idx) block.outflow_boxes.size() > BLOCK::max_outflow_boxes) {
		dumpContext(block, ref, local);
		FAIL("more than max_outflow_boxes boxes");
	}

	std::vector<int> cov(ncells, 0);
	for (const auto& b : block.outflow_boxes) {
		for (int d = 0; d < 3; d++) {
			if (b.begin[d] > b.end[d]) {
				dumpContext(block, ref, local);
				FAIL("begin > end in some dim");
			}
			const idx side = b.end[d] - b.begin[d];
			const idx target = TNL::min(BLOCK::min_outflow_box_extent, local[d]);
			if (block.outflow_boxes.size() != BLOCK::max_outflow_boxes && side > 1 && side < target) {
				dumpContext(block, ref, local);
				FAIL("box side below min_outflow_box_extent");
			}
		}
		if (b.begin.x() < 0 || b.end.x() > local.x() || b.begin.y() < 0 || b.end.y() > local.y() || b.begin.z() < 0 || b.end.z() > local.z()) {
			dumpContext(block, ref, local);
			FAIL("box outside local domain");
		}
		for (idx z = b.begin.z(); z < b.end.z(); z++)
			for (idx y = b.begin.y(); y < b.end.y(); y++)
				for (idx x = b.begin.x(); x < b.end.x(); x++) {
					if (cov[(z * local.y() + y) * local.x() + x] != 0) {
						dumpContext(block, ref, local);
						FAIL("boxes are not pairwise disjoint");
					}
					cov[(z * local.y() + y) * local.x() + x] = 1;
				}
	}
	for (idx i = 0; i < ncells; i++)
		if (ref[i] != 0 && cov[i] == 0) {
			dumpContext(block, ref, local);
			FAIL("mask cell not covered by any box");
		}
}

static TestBox maskBbox(const std::vector<std::uint8_t>& ref, const idx3d& local)
{
	idx x0 = local.x();
	idx y0 = local.y();
	idx z0 = local.z();
	idx x1 = 0;
	idx y1 = 0;
	idx z1 = 0;

	for (idx z = 0; z < local.z(); z++)
		for (idx y = 0; y < local.y(); y++)
			for (idx x = 0; x < local.x(); x++)
				if (ref[(z * local.y() + y) * local.x() + x] != 0) {
					x0 = x0 < x ? x0 : x;
					y0 = y0 < y ? y0 : y;
					z0 = z0 < z ? z0 : z;
					x1 = x1 > x + 1 ? x1 : x + 1;
					y1 = y1 > y + 1 ? y1 : y + 1;
					z1 = z1 > z + 1 ? z1 : z + 1;
				}

	return {x0, y0, z0, x1, y1, z1};
}

static long totalVolume(const BLOCK& block)
{
	long total = 0;
	for (const auto& b : block.outflow_boxes) {
		long v = 1;
		for (int d = 0; d < 3; d++)
			v *= b.end[d] - b.begin[d];
		total += v;
	}
	return total;
}

static long testBoxVolume(const TestBox& b)
{
	return (long) (b[3] - b[0]) * (b[4] - b[1]) * (b[5] - b[2]);
}

// mirrors the pad phase of updateOutflowPassRegion for deriving the expected
// fallback box
static TestBox padToMinExtent(TestBox box, const idx3d& local)
{
	for (int d = 0; d < 3; d++) {
		const idx extent = box[d + 3] - box[d];
		const idx target = TNL::min(BLOCK::min_outflow_box_extent, local[d]);
		if (extent <= 1 || extent >= target)
			continue;
		const idx extra = target - extent;
		box[d] = TNL::max(box[d] - extra / 2, 0);
		box[d + 3] = box[d] + target;
		if (box[d + 3] > local[d]) {
			box[d + 3] = local[d];
			box[d] = local[d] - target;
		}
	}
	return box;
}

TEST_SUITE_BEGIN("outflowcover");

TEST_CASE("empty")
{
	const idx3d local{8, 8, 4};
	const idx3d offset{0, 0, 0};
	const idx ncells = local.x() * local.y() * local.z();
	std::vector<std::uint8_t> ref(ncells, 0);
	BLOCK block = makeBlock(local, local, offset);
	block.allocateHostData();
	fillBlock(block, ref, local, offset);
	if (! block.outflow_boxes.empty()) {
		dumpContext(block, ref, local);
		FAIL("expected no boxes");
	}
}

// grid with both tangential dims larger than min_outflow_box_extent, so
// padding is exercisable without hitting the opposite boundary
static const idx3d PLANE{2, 96, 48};

// window mask on the x==1 wall of the PLANE grid: y-extent [yb, yb + 4),
// z-extent [zb, zb + 4)
static void setWindow(std::vector<std::uint8_t>& ref, idx yb, idx zb)
{
	for (idx z = zb; z < zb + 4; z++)
		for (idx y = yb; y < yb + 4; y++)
			ref[(z * PLANE.y() + y) * PLANE.x() + 1] = 1;
}

TEST_CASE("full-plane")
{
	const idx3d local = PLANE;
	const idx3d offset{0, 0, 0};
	std::vector<std::uint8_t> ref(PLANE.x() * PLANE.y() * PLANE.z(), 0);
	for (idx z = 0; z < local.z(); z++)
		for (idx y = 0; y < local.y(); y++)
			ref[(z * local.y() + y) * local.x() + 1] = 1;
	BLOCK block = makeBlock(local, local, offset);
	block.allocateHostData();
	fillBlock(block, ref, local, offset);
	// both tangential sides are already >= min_outflow_box_extent -> no pad
	requireBoxes(block, {{1, 0, 0, 2, 96, 48}}, ref, local);
}

TEST_CASE("two-windows")
{
	const idx3d local = PLANE;
	const idx3d offset{0, 0, 0};
	std::vector<std::uint8_t> ref(PLANE.x() * PLANE.y() * PLANE.z(), 0);
	setWindow(ref, 2, 2);
	setWindow(ref, 70, 2);
	BLOCK block = makeBlock(local, local, offset);
	block.allocateHostData();
	fillBlock(block, ref, local, offset);
	// each 4x4 window pads to 32x32; the 32-cell gap in y keeps the pads
	// disjoint: window 1 -> y [0,32), window 2 -> y [56,88), both z [0,32)
	requireBoxes(block, {{1, 0, 0, 2, 32, 32}, {1, 56, 0, 2, 88, 32}}, ref, local);
	checkInvariants(block, ref, local);
}

TEST_CASE("proximity-merge")
{
	const idx3d local = PLANE;
	const idx3d offset{0, 0, 0};
	std::vector<std::uint8_t> ref(PLANE.x() * PLANE.y() * PLANE.z(), 0);
	setWindow(ref, 2, 2);	// pads to y [0,32), z [0,32)
	setWindow(ref, 34, 2);	// pads to y [20,52), z [0,32)
	BLOCK block = makeBlock(local, local, offset);
	block.allocateHostData();
	fillBlock(block, ref, local, offset);
	// pads overlap in y on [20,32) -> merged into the bounding box
	requireBoxes(block, {{1, 0, 0, 2, 52, 32}}, ref, local);
	checkInvariants(block, ref, local);
}

TEST_CASE("clip-slide")
{
	const idx3d local = PLANE;
	const idx3d offset{0, 0, 0};
	std::vector<std::uint8_t> ref(PLANE.x() * PLANE.y() * PLANE.z(), 0);
	setWindow(ref, 92, 44);
	BLOCK block = makeBlock(local, local, offset);
	block.allocateHostData();
	fillBlock(block, ref, local, offset);
	// centered growth [78,110) / [30,62) exceeds the grid: both sides slide
	// to the far clip, keeping exactly the min extent
	requireBoxes(block, {{1, 64, 16, 2, 96, 48}}, ref, local);
	checkInvariants(block, ref, local);
}

static void check_random_seed(unsigned seed, const idx3d& local, double density)
{
	const idx3d offset{0, 0, 0};
	const idx ncells = local.x() * local.y() * local.z();
	std::mt19937 gen(seed);
	std::bernoulli_distribution coin(density);
	std::vector<std::uint8_t> ref(ncells, 0);
	for (idx i = 0; i < ncells; i++)
		ref[i] = coin(gen) ? 1 : 0;
	BLOCK block = makeBlock(local, local, offset);
	block.allocateHostData();
	fillBlock(block, ref, local, offset);
	checkInvariants(block, ref, local);
}

TEST_CASE("random-11")
{
	check_random_seed(11, {8, 8, 4}, 0.3);
}

TEST_CASE("random-22")
{
	check_random_seed(22, {8, 8, 4}, 0.3);
}

TEST_CASE("random-33")
{
	check_random_seed(33, {8, 8, 4}, 0.3);
}

TEST_CASE("random-44")
{
	check_random_seed(44, {4, 96, 48}, 0.05);
}

TEST_CASE("random-55")
{
	check_random_seed(55, {4, 96, 48}, 0.05);
}

TEST_CASE("random-66")
{
	check_random_seed(66, {4, 96, 48}, 0.05);
}

TEST_CASE("checkerboard")
{
	const idx3d local{8, 8, 4};
	const idx3d offset{0, 0, 0};
	const idx ncells = local.x() * local.y() * local.z();
	std::vector<std::uint8_t> ref(ncells, 0);
	for (idx z = 0; z < local.z(); z++)
		for (idx y = 0; y < local.y(); y++)
			for (idx x = 0; x < local.x(); x++)
				if ((x + y + z) % 2 == 0)
					ref[(z * local.y() + y) * local.x() + x] = 1;
	BLOCK block = makeBlock(local, local, offset);
	block.allocateHostData();
	fillBlock(block, ref, local, offset);
	// merge-reduced to exactly max_outflow_boxes boxes; the greedy merge must
	// cover strictly fewer cells than the legacy single bounding box (padded
	// to the whole domain), the exact layout is deliberately not pinned
	if (block.outflow_boxes.size() != BLOCK::max_outflow_boxes) {
		dumpContext(block, ref, local);
		FAIL("expected exactly max_outflow_boxes boxes");
	}
	const long legacy = testBoxVolume(padToMinExtent(maskBbox(ref, local), local));
	if (totalVolume(block) >= legacy) {
		dumpContext(block, ref, local);
		FAIL("merged boxes cover no fewer cells than the legacy bounding box");
	}
	checkInvariants(block, ref, local);
}

TEST_CASE("offset-block")
{
	// a block owning a corner of a larger global lattice: the reference mask
	// is defined on the LOCAL grid, so the x==1 wall maps to global x==3 and
	// the windows sit at offset-shifted global positions; the same padded
	// boxes as in two-windows must come out in LOCAL indices
	const idx3d global{4, 99, 49};
	const idx3d local = PLANE;
	const idx3d offset{2, 3, 1};
	std::vector<std::uint8_t> ref(PLANE.x() * PLANE.y() * PLANE.z(), 0);
	setWindow(ref, 2, 2);
	setWindow(ref, 70, 2);
	BLOCK block = makeBlock(global, local, offset);
	block.allocateHostData();
	fillBlock(block, ref, local, offset);
	requireBoxes(block, {{1, 0, 0, 2, 32, 32}, {1, 56, 0, 2, 88, 32}}, ref, local);
	checkInvariants(block, ref, local);
}

TEST_CASE("confined-checkerboard")
{
	// checkerboard confined to x in [1,7): 96 components -> merge-reduced to
	// exactly max_outflow_boxes boxes covering fewer cells than the legacy
	// single bounding box (padded mask bbox)
	const idx3d local{8, 8, 4};
	const idx3d offset{0, 0, 0};
	const idx ncells = local.x() * local.y() * local.z();
	std::vector<std::uint8_t> ref(ncells, 0);
	for (idx z = 0; z < local.z(); z++)
		for (idx y = 0; y < local.y(); y++)
			for (idx x = 1; x < 7; x++)
				if ((x + y + z) % 2 == 0)
					ref[(z * local.y() + y) * local.x() + x] = 1;
	BLOCK block = makeBlock(local, local, offset);
	block.allocateHostData();
	fillBlock(block, ref, local, offset);
	if (block.outflow_boxes.size() != BLOCK::max_outflow_boxes) {
		dumpContext(block, ref, local);
		FAIL("expected exactly max_outflow_boxes boxes");
	}
	const long legacy = testBoxVolume(padToMinExtent(maskBbox(ref, local), local));
	if (totalVolume(block) >= legacy) {
		dumpContext(block, ref, local);
		FAIL("merged boxes cover no fewer cells than the legacy padded bbox");
	}
	checkInvariants(block, ref, local);
}

TEST_CASE("z-split")
{
	// the same window at two distant z-ranges: the first rect's box closes on
	// the empty z-planes between them and a second box opens; both pads must
	// stay disjoint in z (no z-merge, no pad overlap)
	const idx3d local{2, 96, 96};
	const idx3d offset{0, 0, 0};
	std::vector<std::uint8_t> ref(local.x() * local.y() * local.z(), 0);
	setWindow(ref, 2, 2);	// pads to y [0,32), z [0,32)
	setWindow(ref, 2, 66);	// pads to y [0,32), z [52,84)
	BLOCK block = makeBlock(local, local, offset);
	block.allocateHostData();
	fillBlock(block, ref, local, offset);
	requireBoxes(block, {{1, 0, 0, 2, 32, 32}, {1, 0, 52, 2, 32, 84}}, ref, local);
	checkInvariants(block, ref, local);
}

TEST_CASE("boundary-64")
{
	// a checkerboard on y in [0,4) has exactly 64 one-cell components -> 64
	// one-cell boxes; exactly at max_outflow_boxes, so the merge-reduction
	// must NOT fire (the threshold is a strict >)
	const idx3d local{8, 8, 4};
	const idx3d offset{0, 0, 0};
	const idx ncells = local.x() * local.y() * local.z();
	std::vector<std::uint8_t> ref(ncells, 0);
	for (idx z = 0; z < local.z(); z++)
		for (idx y = 0; y < 4; y++)
			for (idx x = 0; x < local.x(); x++)
				if ((x + y + z) % 2 == 0)
					ref[(z * local.y() + y) * local.x() + x] = 1;
	BLOCK block = makeBlock(local, local, offset);
	block.allocateHostData();
	fillBlock(block, ref, local, offset);
	if (block.outflow_boxes.size() != (std::size_t) BLOCK::max_outflow_boxes) {
		dumpContext(block, ref, local);
		FAIL("expected exactly max_outflow_boxes boxes (merge-reduction must not fire)");
	}
	for (const auto& b : block.outflow_boxes)
		for (int d = 0; d < 3; d++)
			if (b.end[d] != b.begin[d] + 1) {
				dumpContext(block, ref, local);
				FAIL("expected one-cell boxes");
			}
	checkInvariants(block, ref, local);
}

TEST_CASE("boundary-65")
{
	// the 64-component checkerboard plus one isolated cell at y==7: one over
	// the cap -> merge-reduced to exactly max_outflow_boxes boxes by the
	// cheapest merge (dead volume 1: a distance-2 collinear pair with one
	// dead cell between) — 63 one-cell boxes plus exactly one 3-cell box
	const idx3d local{8, 8, 4};
	const idx3d offset{0, 0, 0};
	const idx ncells = local.x() * local.y() * local.z();
	std::vector<std::uint8_t> ref(ncells, 0);
	for (idx z = 0; z < local.z(); z++)
		for (idx y = 0; y < 4; y++)
			for (idx x = 0; x < local.x(); x++)
				if ((x + y + z) % 2 == 0)
					ref[(z * local.y() + y) * local.x() + x] = 1;
	ref[7 * local.x()] = 1;	 // (x,y,z) = (0,7,0)
	BLOCK block = makeBlock(local, local, offset);
	block.allocateHostData();
	fillBlock(block, ref, local, offset);
	if (block.outflow_boxes.size() != BLOCK::max_outflow_boxes) {
		dumpContext(block, ref, local);
		FAIL("expected exactly max_outflow_boxes boxes");
	}
	long ones = 0;
	long threes = 0;
	for (const auto& b : block.outflow_boxes) {
		long v = 1;
		for (int d = 0; d < 3; d++)
			v *= b.end[d] - b.begin[d];
		if (v == 1)
			ones++;
		else if (v == 3)
			threes++;
		else {
			dumpContext(block, ref, local);
			FAIL("unexpected merged-box volume");
		}
	}
	if (ones != 63 || threes != 1) {
		dumpContext(block, ref, local);
		FAIL("expected 63 one-cell boxes and 1 three-cell box");
	}
	checkInvariants(block, ref, local);
}

TEST_SUITE_END();
