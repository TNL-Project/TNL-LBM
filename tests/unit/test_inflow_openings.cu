/*
 * Unit tests for the pure host helpers behind inflow openings
 * (include/lbm3d/inflow_openings.h; plan .omo/plans/inflow-openings.md,
 * checkbox 12). No State construction, no MPI, no device launches.
 *
 * - labelComponents2D: 4-connected union-find labeling of candidate cells
 *   on a tangential plane (used by State::discoverInflowOpenings to cut
 *   discovered openings). The tests pin exact component counts, exact
 *   bounding rects, the id order (ascending row-major first cell), the
 *   diagonal-touch exclusion, labeling of the unclaimed remainder around
 *   an authored claim, degenerate inputs, and run-to-run determinism
 *   (all MPI ranks must label the reconstructed global picture
 *   identically).
 * - openingProfileWeight: the LOCKED cell-centered product paraboloid
 *   xi = (u - u0 + 0.5) / (u1 - u0 + 1) (likewise eta),
 *   w = (1-(2xi-1)^2) * (1-(2eta-1)^2); UNIFORM == 1 without touching the
 *   geometry arguments. Exact checks where the arithmetic is exact
 *   (0.5625, 7/16, 15/16, unit factors), doctest::Approx at 1e-15 near
 *   the non-representable thirds (5/9, 25/81).
 */

#include <doctest/doctest.h>

#include <algorithm>
#include <string>
#include <vector>

#include "lbm3d/defs.h"
#include "lbm3d/inflow_openings.h"

using TRAITS = Traits<double>;

// labeling test helpers: turns an ASCII picture (one row per string, '#' = candidate cell) into the
// isCandidate lambda of labelComponents2D; u = column, v = row. Rows may be
// shorter than the picture width (missing trailing cells read as
// non-candidate), so pictures never need trailing spaces
static auto pictureCandidate(const std::vector<std::string>& rows)
{
	return [rows](int u, int v)
	{
		return v >= 0 && v < static_cast<int>(rows.size()) && u >= 0 && u < static_cast<int>(rows[static_cast<std::size_t>(v)].size())
			&& rows[static_cast<std::size_t>(v)][static_cast<std::size_t>(u)] == '#';
	};
}

static int pictureWidth(const std::vector<std::string>& rows)
{
	int w = 0;
	for (const auto& r : rows)
		w = std::max(w, static_cast<int>(r.size()));
	return w;
}

static void checkRect(const ComponentRect2D& r, int lo_u, int lo_v, int hi_u, int hi_v)
{
	CHECK(r.lo_u == lo_u);
	CHECK(r.lo_v == lo_v);
	CHECK(r.hi_u == hi_u);
	CHECK(r.hi_v == hi_v);
}

// labelComponents2D
TEST_SUITE_BEGIN("inflow_openings_labeling");

TEST_CASE("single-rect")
{
	// one 3x2 block at cols 2..4, rows 1..2 of a 5x4 picture
	const std::vector<std::string> rows = {
		"",
		"  ###",
		"  ###",
		"",
	};
	const auto [ids, rects] = labelComponents2D(pictureWidth(rows), static_cast<int>(rows.size()), pictureCandidate(rows));
	REQUIRE(rects.size() == 1);
	checkRect(rects[0], 2, 1, 4, 2);
	CHECK(ids.size() == 20);
	CHECK(ids[7] == 0);	   // row 1, col 2 (row-major first cell gets id 0)
	CHECK(ids[9] == 0);	   // row 1, col 4
	CHECK(ids[12] == 0);   // row 2, col 2
	CHECK(ids[0] == -1);   // above the block
	CHECK(ids[6] == -1);   // row 1, col 1 (left of the block)
	CHECK(ids[17] == -1);  // row 3, col 2 (below the block)
}

TEST_CASE("separated-components")
{
	// vertical bar at col 1 plus a 2x2 block at cols 3..4 (both rows)
	const std::vector<std::string> rows = {
		" # ##",
		" # ##",
	};
	const auto [ids, rects] = labelComponents2D(pictureWidth(rows), static_cast<int>(rows.size()), pictureCandidate(rows));
	REQUIRE(rects.size() == 2);
	// ids ascend by row-major first cell: the col-1 cell (i=1) precedes col 3 (i=3)
	checkRect(rects[0], 1, 0, 1, 1);
	checkRect(rects[1], 3, 0, 4, 1);
	CHECK(ids[1] == 0);
	CHECK(ids[6] == 0);	 // col 1, row 2
	CHECK(ids[3] == 1);
	CHECK(ids[9] == 1);
	CHECK(ids[0] == -1);
}

TEST_CASE("diagonal-not-connected")
{
	// (0,0) and (1,1) touch only diagonally; 4-connectivity must NOT merge them
	const std::vector<std::string> rows = {
		"#",
		" #",
	};
	const auto [ids, rects] = labelComponents2D(2, 2, pictureCandidate(rows));
	REQUIRE(rects.size() == 2);
	checkRect(rects[0], 0, 0, 0, 0);
	checkRect(rects[1], 1, 1, 1, 1);
	CHECK(ids[0] == 0);
	CHECK(ids[3] == 1);
	CHECK(ids[1] == -1);  // (1,0): not part of either component
	CHECK(ids[2] == -1);  // (0,1)
}

TEST_CASE("l-shape-bbox")
{
	// L: col 0 on rows 0..2 plus row 2 cols 0..2 -> one 4-connected component
	// whose bounding rect is the full 3x3 square (5 cells inside it)
	const std::vector<std::string> rows = {
		"#",
		"#",
		"###",
	};
	const auto [ids, rects] = labelComponents2D(3, 3, pictureCandidate(rows));
	REQUIRE(rects.size() == 1);
	checkRect(rects[0], 0, 0, 2, 2);
	for (int i : {0, 3, 6, 7, 8})
		CHECK(ids[i] == 0);
	CHECK(ids[1] == -1);  // (1,0) is not part of the L
}

TEST_CASE("empty-and-degenerate")
{
	SUBCASE("no-candidates")
	{
		const std::vector<std::string> rows = {" ", " "};
		const auto [ids, rects] = labelComponents2D(1, 2, pictureCandidate(rows));
		CHECK(rects.empty());
		CHECK(ids.size() == 2);
		CHECK(ids[0] == -1);
		CHECK(ids[1] == -1);
	}
	SUBCASE("single-cell")
	{
		const std::vector<std::string> rows = {"#"};
		const auto [ids, rects] = labelComponents2D(1, 1, pictureCandidate(rows));
		REQUIRE(rects.size() == 1);
		checkRect(rects[0], 0, 0, 0, 0);
		CHECK(ids.size() == 1);
		CHECK(ids[0] == 0);
	}
	SUBCASE("degenerate-dimensions")
	{
		// zero or negative extents produce empty outputs, never a component
		const auto always = [](int, int)
		{
			return true;
		};
		const auto [zero_w_ids, zero_w_rects] = labelComponents2D(0, 3, always);
		CHECK(zero_w_ids.empty());
		CHECK(zero_w_rects.empty());
		const auto [zero_h_ids, zero_h_rects] = labelComponents2D(3, 0, always);
		CHECK(zero_h_ids.empty());
		CHECK(zero_h_rects.empty());
		const auto [neg_ids, neg_rects] = labelComponents2D(-2, 4, always);
		CHECK(neg_ids.empty());
		CHECK(neg_rects.empty());
	}
}

TEST_CASE("authored-complement")
{
	// fully tagged 5x4 plane with the middle column claimed by one authored
	// rect: labeling must see only the unclaimed remainder - two separate
	// side blocks with exact rects
	const std::vector<std::string> rows = {
		"#####",
		"#####",
		"#####",
		"#####",
	};
	const auto tagged = pictureCandidate(rows);
	// the authored claim covers col 2 on every row
	const auto unclaimed = [&tagged](int u, int v)
	{
		return tagged(u, v) && u != 2;
	};
	const auto [ids, rects] = labelComponents2D(5, 4, unclaimed);
	REQUIRE(rects.size() == 2);
	checkRect(rects[0], 0, 0, 1, 3);  // left block, first cell (0,0)
	checkRect(rects[1], 3, 0, 4, 3);  // right block, first cell (3,0)
	CHECK(ids[1] == 0);
	CHECK(ids[15] == 0);  // row 3, col 0
	CHECK(ids[3] == 1);
	CHECK(ids[19] == 1);   // row 3, col 4
	CHECK(ids[2] == -1);   // claimed column, row 0
	CHECK(ids[17] == -1);  // claimed column, row 3
}

TEST_CASE("deterministic-repeat")
{
	// busy picture with five components (rects hand-verified); labeling it
	// twice must give byte-identical ids and rects - all MPI ranks label the
	// same reconstructed global picture
	const std::vector<std::string> rows = {
		"# ##  ##",
		"#  #   #",
		"  ##  #",
		"   ## #",
		" ##   ##",
	};
	const auto labelPicture = [&rows]()
	{
		return labelComponents2D(pictureWidth(rows), static_cast<int>(rows.size()), pictureCandidate(rows));
	};
	const auto [ids_a, rects_a] = labelPicture();
	const auto [ids_b, rects_b] = labelPicture();

	REQUIRE(rects_a.size() == 5);
	checkRect(rects_a[0], 0, 0, 0, 1);	// col 0, rows 0..1
	checkRect(rects_a[1], 2, 0, 4, 3);	// the middle blob
	checkRect(rects_a[2], 6, 0, 7, 1);	// top-right hook
	checkRect(rects_a[3], 6, 2, 7, 4);	// col 6 vertical + row-4 tail
	checkRect(rects_a[4], 1, 4, 2, 4);	// row-4 pair

	CHECK(ids_a == ids_b);
	REQUIRE(rects_b.size() == rects_a.size());
	for (std::size_t i = 0; i < rects_a.size(); i++) {
		CHECK(rects_b[i].lo_u == rects_a[i].lo_u);
		CHECK(rects_b[i].lo_v == rects_a[i].lo_v);
		CHECK(rects_b[i].hi_u == rects_a[i].hi_u);
		CHECK(rects_b[i].hi_v == rects_a[i].hi_v);
	}
}

TEST_SUITE_END();

// openingProfileWeight
TEST_SUITE_BEGIN("inflow_openings_weight");

TEST_CASE("uniform-profile")
{
	// UNIFORM is exactly 1 for any geometry - the branch deliberately never
	// touches the rect/cell arguments, so junk or inverted bounds are safe
	CHECK(openingProfileWeight<int, double>(ProfileType::UNIFORM, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0) == 1.0);
	CHECK(openingProfileWeight<int, double>(ProfileType::UNIFORM, 0.0, 10.0, 0.0, 10.0, 3.0, 7.0) == 1.0);
	CHECK(openingProfileWeight<int, double>(ProfileType::UNIFORM, 5.0, 2.0, -4.0, 9.0, 123.0, -45.0) == 1.0);
	CHECK(openingProfileWeight<int, double>(ProfileType::UNIFORM, 1e300, -1e300, 1e300, -1e300, 0.0, 0.0) == 1.0);
}

TEST_CASE("parabolic-2x2")
{
	// u0=v0=0, u1=v1=1: cell centers xi=eta in {0.25, 0.75} -> factor 0.75
	// per direction; the weight 0.75*0.75 = 0.5625 is exactly representable
	for (int u = 0; u <= 1; u++)
		for (int v = 0; v <= 1; v++)
			CHECK(openingProfileWeight<int, double>(ProfileType::PARABOLIC, 0.0, 1.0, 0.0, 1.0, u, v) == 0.5625);
}

TEST_CASE("parabolic-3x3")
{
	// u0=v0=0, u1=v1=2: xi,eta in {1/6, 1/2, 5/6} -> factors 5/9, 1, 5/9;
	// 5/9 and (5/9)^2 are not representable, so compare to the exact
	// rationals at 1e-15 (the observed rounding error is ~2e-16 relative)
	CHECK(openingProfileWeight<int, double>(ProfileType::PARABOLIC, 0.0, 2.0, 0.0, 2.0, 1.0, 1.0) == 1.0);	// center cell
	for (int u : {0, 2}) {
		for (int v : {0, 2})
			CHECK(openingProfileWeight<int, double>(ProfileType::PARABOLIC, 0.0, 2.0, 0.0, 2.0, u, v) == doctest::Approx(25.0 / 81.0).epsilon(1e-15));
		CHECK(openingProfileWeight<int, double>(ProfileType::PARABOLIC, 0.0, 2.0, 0.0, 2.0, u, 1) == doctest::Approx(5.0 / 9.0).epsilon(1e-15));
		CHECK(openingProfileWeight<int, double>(ProfileType::PARABOLIC, 0.0, 2.0, 0.0, 2.0, 1, u) == doctest::Approx(5.0 / 9.0).epsilon(1e-15));
	}
}

TEST_CASE("parabolic-single-cell-strip")
{
	// 1x3 rect (u0==u1): the single-cell-wide direction contributes factor 1
	// exactly (xi=0.5); along v, eta in {1/6, 1/2, 5/6} -> factors 5/9, 1, 5/9
	const double w0 = openingProfileWeight<int, double>(ProfileType::PARABOLIC, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0);
	const double w1 = openingProfileWeight<int, double>(ProfileType::PARABOLIC, 0.0, 0.0, 0.0, 2.0, 0.0, 1.0);
	const double w2 = openingProfileWeight<int, double>(ProfileType::PARABOLIC, 0.0, 0.0, 0.0, 2.0, 0.0, 2.0);
	CHECK(w0 == doctest::Approx(5.0 / 9.0).epsilon(1e-15));	 // 1 * (1-(2/3)^2)
	CHECK(w1 == 1.0);										 // 1 * 1 exactly (eta=0.5)
	CHECK(w2 == doctest::Approx(5.0 / 9.0).epsilon(1e-15));
	CHECK(w2 == doctest::Approx(w0).epsilon(1e-15));  // symmetric strip ends
}

TEST_CASE("typed-index-safety")
{
	// instantiated exactly as the resolver calls it, <idx, dreal>; the rect
	// offsets enter through REAL subtraction, so shifted rects keep working
	using INDEX = TRAITS::idx;

	// 2x2 rect at (2..3, 5..6), cell (3,5): xi=0.75, eta=0.25 -> exactly 0.5625
	CHECK(openingProfileWeight<INDEX, double>(ProfileType::PARABOLIC, 2, 3, 5, 6, 3, 5) == 0.5625);

	// 4-wide single-row strip at u0=2, v0=v1=7: xi in {1/8,3/8,5/8,7/8} with
	// a unit eta factor -> exactly 7/16, 15/16, 15/16, 7/16
	CHECK(openingProfileWeight<INDEX, double>(ProfileType::PARABOLIC, 2, 5, 7, 7, 2, 7) == 7.0 / 16.0);
	CHECK(openingProfileWeight<INDEX, double>(ProfileType::PARABOLIC, 2, 5, 7, 7, 3, 7) == 15.0 / 16.0);
	CHECK(openingProfileWeight<INDEX, double>(ProfileType::PARABOLIC, 2, 5, 7, 7, 4, 7) == 15.0 / 16.0);
	CHECK(openingProfileWeight<INDEX, double>(ProfileType::PARABOLIC, 2, 5, 7, 7, 5, 7) == 7.0 / 16.0);

	// thirds through idx-typed args: 3x3 corner of a rect shifted to (10..12, 20..22)
	CHECK(openingProfileWeight<INDEX, double>(ProfileType::PARABOLIC, 10, 12, 20, 22, 10, 20) == doctest::Approx(25.0 / 81.0).epsilon(1e-15));
}

TEST_SUITE_END();
