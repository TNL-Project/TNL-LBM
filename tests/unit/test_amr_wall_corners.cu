/*
 * AMR wall-corner C2F destination-partition gate: the symmetric
 * corner-column contract of buildCouplings' six-face patch table under
 * wall-shared (masked) faces -- the generalization of commit 3d963c40
 * ("fill the wall-shared y/z corner columns of the z-face C2F fill").
 *
 * Surface: a fine block's C2F destination band is partitioned disjointly
 * over six per-face rectangles (x-normal faces own the FULL y/z tangents,
 * y-normal faces the interior x range and the full z tangent, z-normal
 * faces the interior x/y ranges). A face masked in fine_wall_masks (its
 * face-adjacent parent plane is GEO_WALL end to end) empties its own
 * patch -- the wall row and its GEO_NOTHING buffer are BC-managed and the
 * first-fluid row is kernel-authored -- so the masked face's patch no
 * longer covers the CORNER COLUMNS where its tangent crossed the other
 * faces' bands: the masked face's first-fluid row (local -1 / local)
 * times the live faces' band rows. 3d963c40 widened the z-normal patches
 * over masked y faces; the symmetric contract this suite locks widens
 * every y/z-normal patch's x tangent over masked x faces as well (the
 * x-normal patches keep their full tangents and need no widening).
 *
 * Scenarios (16^3 mL1, one footprint of gs = 8, walls as interior planes
 * at index 2 (min side) / 13 (max side), the footprint's halo row sitting
 * exactly on the wall plane -- the sim_AMR_channel wall-attachment class
 * rotated onto every axis combination):
 * - y_min / y_max / x_min: single wall-attached faces (the y arms pin
 *   3d963c40's surface; x_min is its symmetric complement),
 * - y_z / x_z / x_y: footprints sitting in a CORNER of two walls,
 * - x_y_z: the three-wall corner,
 * - right_top_front: the all-max mirror (exercises the _hi widening
 *   terms and the max-side wall row at local+1 / buffer at local+2).
 * Walled axes are non-periodic with GEO_NOTHING frames outside the wall
 * planes; free axes stay periodic (the wall-lane fixture layout).
 *
 * Gates, all pattern-agnostic (the binaries are stamped per streaming
 * pattern like the wall-lane suite; the assertions never decode
 * pattern-specific slot layouts):
 * - STRUCTURAL (the primary lock): after SimInit the fine block's wall
 *   mask holds exactly the scenario's masked faces; the coupling's six
 *   patches match the expected rectangle table (the partition spec with
 *   the symmetric widening; masked faces carry the emptied form); the
 *   non-empty rectangles are pairwise disjoint; every corner column
 *   (first-fluid row of a masked face x band row of a live face x
 *   interior of the third axis) is covered by EXACTLY ONE patch, while
 *   the wall rows and GEO_NOTHING buffers crossed with the live y/z
 *   faces' outer band rows are covered by NONE (the widening
 *   deliberately stops at the first-fluid row -- the wall row's band
 *   columns stay BC-managed, the accepted channel semantics; the
 *   x-normal faces' full tangents cover the wall row's x-band columns,
 *   asserted as a positive control of that design).
 * - BEHAVIORAL (the fill-activity witness): with the traveling-gradient
 *   initial condition (the wall-lane IC -- non-stationary, so the fill
 *   content tracks the evolving coarse field) the corner columns' raw
 *   DF slots must change across the driven coarse cycles (a hole in the
 *   partition leaves them at the initial condition forever); under the
 *   push-family patterns the production scatter can author outer-row
 *   slots independently of the fill, so this witness is secondary -- the
 *   structural coverage census above is the discriminating gate.
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include <fmt/core.h>
#include <fmt/format.h>

#include <doctest/doctest.h>

#include "amr_test_fixture.h"

namespace
{

// lattice edge of the level-0 box (the fixture's makeLattice default)
constexpr int WC_N = 16;
// footprint edge in coarse cells (fine interior 2*gs - 2 = 14 per axis)
constexpr int WC_GS = 8;
// wall plane indices: interior planes with >= 3 cells of gap to the
// domain edge on the walled (non-periodic) axes (the V-suite no-warning
// tier; the GEO_NOTHING frames sit outside them)
constexpr int WC_WALL_MIN = 2;
constexpr int WC_WALL_MAX = 13;
// coarse cycles driven per scenario (the gradient IC is measurably
// evolved after 4 coarse steps = 8 fine substeps)
constexpr int WC_CYCLES = 4;

// which of the six footprint faces is wall-attached
struct WallSpec
{
	bool x_min, x_max, y_min, y_max, z_min, z_max;

	bool any_x() const { return x_min || x_max; }
	bool any_y() const { return y_min || y_max; }
	bool any_z() const { return z_min || z_max; }

	// the fine-wall mask bits (fineWallFaceBit order: L R B T Ba F)
	int maskBits() const
	{
		return (x_min ? 1 : 0) | (x_max ? 2 : 0) | (y_min ? 4 : 0) | (y_max ? 8 : 0) | (z_min ? 16 : 0) | (z_max ? 32 : 0);
	}
};

// one scenario: wall placement + footprint origin (coarse cells; the
// walled-side origin puts the footprint's halo row exactly on the wall
// plane, free axes use the wall-lane layout origin 4)
struct Scenario
{
	const char* name;
	WallSpec walls;
	int go[3];
};

// the eight topologies: the three single-wall arms, the three two-wall
// corners, the three-wall corner and the all-max mirror
const std::array<Scenario, 8> wall_corner_scenarios = {{
	{"y_min", {false, false, true, false, false, false}, {4, 3, 4}},
	{"y_max", {false, false, false, true, false, false}, {4, 5, 4}},
	{"x_min", {true, false, false, false, false, false}, {3, 4, 4}},
	{"y_z", {false, false, true, false, true, false}, {4, 3, 3}},
	{"x_z", {true, false, false, false, true, false}, {3, 4, 3}},
	{"x_y", {true, false, true, false, false, false}, {3, 3, 4}},
	{"x_y_z", {true, false, true, false, true, false}, {3, 3, 3}},
	{"right_top_front", {false, true, false, true, false, true}, {5, 5, 5}},
}};

// (axis 0/1/2, min side?) of a halo face
inline std::pair<int, bool> faceAxisSide(TNL::Containers::SyncDirection face)
{
	switch (face) {
		case TNL::Containers::SyncDirection::Left:
			return {0, true};
		case TNL::Containers::SyncDirection::Right:
			return {0, false};
		case TNL::Containers::SyncDirection::Bottom:
			return {1, true};
		case TNL::Containers::SyncDirection::Top:
			return {1, false};
		case TNL::Containers::SyncDirection::Back:
			return {2, true};
		case TNL::Containers::SyncDirection::Front:
			return {2, false};
		default:
			return {-1, true};
	}
}

// the scenario state: periodic on the free axes, walls stamped as level-0
// interior planes with GEO_NOTHING frames outside them, the fine wall
// rows tagged by the face-generic channel stamping (interior tangents
// only), and the wall-lane traveling-gradient initial condition
template <typename NSE>
struct StateWallCorners_AMR : StateSchedule_AMR<NSE>
{
	using Base = StateSchedule_AMR<NSE>;
	using BC_t = typename NSE::BC;
	using idx_t = typename NSE::TRAITS::idx;
	using idx3d_t = typename NSE::TRAITS::idx3d;
	using dreal_t = typename NSE::TRAITS::dreal;
	using lat_local_t = typename Base::lat_t;

	WallSpec walls;

	template <typename... ARGS>
	StateWallCorners_AMR(const WallSpec& walls, ARGS&&... args)
	: Base(std::forward<ARGS>(args)...), walls(walls)
	{}

	void setupBoundaries() override
	{
		// level-0 wall planes (interior, spanning the full cross-section)
		// with the GEO_NOTHING frames outside them; free axes untouched
		// (periodic)
		auto* coarse = this->nse.getBlocksAtLevel(0).front();
		const idx_t N = coarse->local.x();
		const auto stampPlane = [&](int axis, idx_t plane, auto tag)
		{
			for (idx_t z = 0; z < N; z++)
				for (idx_t y = 0; y < N; y++)
					for (idx_t x = 0; x < N; x++) {
						const bool hit = (axis == 0 && x == plane) || (axis == 1 && y == plane) || (axis == 2 && z == plane);
						if (hit)
							coarse->hmap(x, y, z) = tag;
					}
		};
		const struct AxisWalls
		{
			bool min, max;
		} axes[3] = {{walls.x_min, walls.x_max}, {walls.y_min, walls.y_max}, {walls.z_min, walls.z_max}};
		for (int axis = 0; axis < 3; axis++) {
			if (axes[axis].min) {
				stampPlane(axis, WC_WALL_MIN, BC_t::GEO_WALL);
				for (int p = 1; p < WC_WALL_MIN; p++)
					stampPlane(axis, p, BC_t::GEO_NOTHING);
			}
			if (axes[axis].max) {
				stampPlane(axis, WC_WALL_MAX, BC_t::GEO_WALL);
				for (int p = WC_WALL_MAX + 1; p < WC_N - 1; p++)
					stampPlane(axis, p, BC_t::GEO_NOTHING);
			}
		}
		// the single-array (AA/ESO) extra-layer idiom, mirroring sim_AMR_channel:
		// a non-periodic (walled) axis must carry GEO_NOTHING on its outermost
		// planes so the unclamped +-1 neighbor access of the ghost-layer streaming
		// patterns never leaves the overlap-0 stored array of an undistributed
		// level-0 block. The frames are stamped LAST: the wall-plane stamps above
		// span the full cross-section, so a wall plane of one axis would otherwise
		// overwrite the outermost frame rows of another walled axis with GEO_WALL --
		// cells whose unclamped neighbor access sits outside the stored array
		// (exactly the two-wall-corner failure this ordering guards against).
		for (int axis = 0; axis < 3; axis++) {
			if (axes[axis].min || axes[axis].max) {
				stampPlane(axis, 0, BC_t::GEO_NOTHING);
				stampPlane(axis, WC_N - 1, BC_t::GEO_NOTHING);
			}
		}

		// fine-level bounce-back walls -- sim_AMR_channel.cu's
		// face-generic stamping verbatim (interior tangents only): on
		// every footprint face whose face-adjacent parent-level row is
		// GEO_WALL the fine block imposes its own wall at local -2 /
		// local+1 with the GEO_NOTHING streaming buffer one row further
		// out; State_AMR::buildFineWallMasks re-derives the same mask
		// from the same rows at SimInit
		for (auto& fine : this->nse.blocks) {
			if (fine.level == 0)
				continue;
			const idx3d_t& go = fine.global_offset;	// footprint origin in parent-level cells
			const idx3d_t gs{(fine.local.x() + 2) / 2, (fine.local.y() + 2) / 2, (fine.local.z() + 2) / 2};
			const struct FACE
			{
				const char* name;
				char axis_name;
				int axis;
				bool min_side;
			} faces[6] = {
				{"x-min", 'x', 0, true},
				{"x-max", 'x', 0, false},
				{"y-min", 'y', 1, true},
				{"y-max", 'y', 1, false},
				{"z-min", 'z', 2, true},
				{"z-max", 'z', 2, false},
			};
			for (const FACE& f : faces) {
				const int a = f.axis;
				const int b = (a + 1) % 3, c = (a + 2) % 3;
				const idx_t wall = f.min_side ? idx_t(-2) : fine.local[a] + 1;
				const idx_t buffer = f.min_side ? idx_t(-3) : fine.local[a] + 2;
				const idx_t plane = f.min_side ? go[a] - 1 : go[a] + gs[a];
				idx_t n_wall = 0;
				for (idx_t ib = 0; ib < fine.local[b]; ib++)
					for (idx_t ic = 0; ic < fine.local[c]; ic++) {
						idx3d_t fg{0, 0, 0};
						fg[b] = fine.offset[b] + ib;
						fg[c] = fine.offset[c] + ic;
						// the column's wall tag follows the IMMEDIATE
						// PARENT level's map on the face-adjacent plane
						idx3d_t cg{0, 0, 0};
						cg[a] = plane;
						cg[b] = fg[b] / 2;
						cg[c] = fg[c] / 2;
						bool wall_column = false;
						for (auto& coarse2 : this->nse.blocks)
							if (coarse2.level == fine.level - 1 && cg[0] >= coarse2.offset.x() - coarse2.df_overlap_X()
								&& cg[0] < coarse2.offset.x() + coarse2.local.x() + coarse2.df_overlap_X()
								&& cg[1] >= coarse2.offset.y() - coarse2.df_overlap_Y()
								&& cg[1] < coarse2.offset.y() + coarse2.local.y() + coarse2.df_overlap_Y()
								&& cg[2] >= coarse2.offset.z() - coarse2.df_overlap_Z()
								&& cg[2] < coarse2.offset.z() + coarse2.local.z() + coarse2.df_overlap_Z()
								&& coarse2.hmap(cg[0], cg[1], cg[2]) == BC_t::GEO_WALL)
								wall_column = true;
						if (! wall_column)
							continue;
						fg[a] = fine.offset[a] + wall;
						fine.hmap(fg[0], fg[1], fg[2]) = BC_t::GEO_WALL;
						fg[a] = fine.offset[a] + buffer;
						fine.hmap(fg[0], fg[1], fg[2]) = BC_t::GEO_NOTHING;
						n_wall++;
					}
				if (n_wall > 0)
					spdlog::info(
						"amr_wall_corners {}: fine block {} tagged {} {} columns with GEO_WALL (local {}={})",
						pattern_name,
						fine.id,
						n_wall,
						f.name,
						f.axis_name,
						wall
					);
			}
		}
	}

	// the wall-lane traveling-gradient initial condition (non-stationary,
	// incommensurate velocity markers -- the fill content must differ
	// from the initial state and evolve cycle by cycle)
	void resetDFs() override
	{
		using idx = typename NSE::TRAITS::idx;
		using point_t = typename NSE::TRAITS::point_t;

		for (auto& block : this->nse.blocks) {
			const lat_local_t lat_local = (block.level == 0) ? this->nse.lat : block.lat_local;
			const idx3d_t offset = block.offset;

			block.setInitialCondition(
				[lat_local, offset] __cuda_callable__(typename NSE::template KernelStruct<dreal_t> & KS, idx gx, idx gy, idx gz) mutable
				{
					const point_t phys = lat_local.lbm2physPoint(gx - offset.x(), gy - offset.y(), gz - offset.z());
					KS.rho = 1 + 0.01f * TNL::sin(8.0f * phys.x());
					KS.vx = 0.02f * TNL::sin(4.0f * phys.y());
					KS.vy = 0.02f * TNL::cos(6.0f * phys.z());
					KS.vz = 0.015f * TNL::sin(5.0f * phys.x() + 3.0f * phys.y());
					NSE::COLL::setEquilibrium(KS);
				}
			);
		}

		this->nse.copyDFsToHost();
	}
};

// ---- expected patch table: the partition spec with the symmetric
// widening (the same formulas buildCouplings implements; asserted here as
// the contract, so any drift of either side fails loudly) ----

struct ExpPatch
{
	TNL::Containers::SyncDirection face;
	idx3d origin, size;
	bool emptied;
};

template <typename NSE>
std::vector<ExpPatch> expectedPatches(const WallSpec& walls, const LBM_BLOCK<NSE>& fine)
{
	using SD = TNL::Containers::SyncDirection;
	// fov is 2 on every axis of these scenarios (free axes carry the
	// fine-block default overlap 2, walled axes the 3-deep override --
	// both cap at the 2-row C2F band)
	const int fov = 2;
	const int lx = fine.local.x(), ly = fine.local.y(), lz = fine.local.z();
	const int x_lo = walls.x_min ? 1 : 0, x_hi = walls.x_max ? 1 : 0;
	const int y_lo = walls.y_min ? 1 : 0, y_hi = walls.y_max ? 1 : 0;

	std::vector<ExpPatch> table;
	// x-normal faces: FULL y/z tangents, never widened
	table.push_back({SD::Left, idx3d{-fov, -fov, -fov}, idx3d{fov, ly + 2 * fov, lz + 2 * fov}, false});
	table.push_back({SD::Right, idx3d{lx, -fov, -fov}, idx3d{fov, ly + 2 * fov, lz + 2 * fov}, false});
	// y-normal faces: x tangent interior + the masked x faces'
	// first-fluid columns, z tangent full
	table.push_back({SD::Bottom, idx3d{-x_lo, -fov, -fov}, idx3d{lx + x_lo + x_hi, fov, lz + 2 * fov}, false});
	table.push_back({SD::Top, idx3d{-x_lo, ly, -fov}, idx3d{lx + x_lo + x_hi, fov, lz + 2 * fov}, false});
	// z-normal faces: x and y tangents interior + the masked faces'
	// first-fluid columns
	table.push_back({SD::Back, idx3d{-x_lo, -y_lo, -fov}, idx3d{lx + x_lo + x_hi, ly + y_lo + y_hi, fov}, false});
	table.push_back({SD::Front, idx3d{-x_lo, -y_lo, lz}, idx3d{lx + x_lo + x_hi, ly + y_lo + y_hi, fov}, false});
	// masked faces carry the emptied form (origin zeroed, size = the
	// local box with the normal-axis extent zeroed)
	const bool masked[2][3] = {
		{walls.x_min, walls.y_min, walls.z_min},
		{walls.x_max, walls.y_max, walls.z_max},
	};
	for (auto& e : table) {
		const auto [axis, min_side] = faceAxisSide(e.face);
		if (! masked[min_side ? 0 : 1][axis])
			continue;
		e.emptied = true;
		e.origin = idx3d{0, 0, 0};
		e.size = idx3d{lx, ly, lz};
		e.size[axis] = 0;
	}
	return table;
}

// coverage count of one fine-local cell by the non-empty patches
inline int coverCount(const std::vector<ExpPatch>& table, int x, int y, int z)
{
	int count = 0;
	for (const auto& e : table) {
		if (e.emptied)
			continue;
		const auto in = [](int c, int o, int s)
		{
			return s > 0 && c >= o && c < o + s;
		};
		if (in(x, e.origin.x(), e.size.x()) && in(y, e.origin.y(), e.size.y()) && in(z, e.origin.z(), e.size.z()))
			count++;
	}
	return count;
}

// the probe cells of one scenario (fine-local coordinates):
// - positive: the first-fluid row of a masked face crossed with a band
//   row of a live face and the interior of the third axis -- the corner
//   columns the symmetric widening must fill (both band rows of the live
//   face); plus the wall/buffer rows crossed with the x-normal live
//   faces' bands (the full-tangent design covers those, a positive
//   control of the channel's "ghost columns of the wall row stay
//   coupling-driven" contract);
// - negative: the masked face's wall row and GEO_NOTHING buffer crossed
//   with a live y/z face's outer band row -- never a destination (the
//   widening stops at the first-fluid row).
struct ProbeSets
{
	std::vector<idx3d> positive;
	std::vector<idx3d> negative;
};

template <typename NSE>
ProbeSets probeCells(const WallSpec& walls, LBM_BLOCK<NSE>& fine)
{
	ProbeSets out;
	const int local[3] = {static_cast<int>(fine.local.x()), static_cast<int>(fine.local.y()), static_cast<int>(fine.local.z())};
	const int ov[3] = {fine.df_overlap_X(), fine.df_overlap_Y(), fine.df_overlap_Z()};
	const bool masked[2][3] = {
		{walls.x_min, walls.y_min, walls.z_min},
		{walls.x_max, walls.y_max, walls.z_max},
	};
	for (int g_axis = 0; g_axis < 3; g_axis++) {
		for (int g_side = 0; g_side < 2; g_side++) {
			if (! masked[g_side][g_axis])
				continue;
			for (int f_axis = 0; f_axis < 3; f_axis++) {
				if (f_axis == g_axis)
					continue;
				const int third = 3 - g_axis - f_axis;
				for (int f_side = 0; f_side < 2; f_side++) {
					if (masked[f_side][f_axis])
						continue;	// f must be a live face
					for (int g_row = 1; g_row <= 3; g_row++) {
						// g_row: 1 = first-fluid, 2 = wall row, 3 = the
						// GEO_NOTHING buffer (allocated only where the
						// walled axis' overlap is 3)
						if (g_row == 3 && ov[g_axis] < 3)
							continue;
						for (int f_row = 1; f_row <= 2; f_row++) {
							// both band rows of the live face (inner
							// integrated + outer passive)
							for (int t = 0; t < local[third]; t++) {
								int c[3];
								c[g_axis] = g_side == 0 ? -g_row : local[g_axis] + g_row - 1;
								c[f_axis] = f_side == 0 ? -f_row : local[f_axis] + f_row - 1;
								c[third] = t;
								const idx3d cell{c[0], c[1], c[2]};
								// an f patch's tangent on the g axis is
								// FULL (covers even the g wall row)
								// exactly for the x-normal patches on
								// y/z and the y-normal patches on z;
								// the z-normal patches' x/y tangents
								// and every patch's tangent on its own
								// normal axis carry only the widened
								// first-fluid columns
								const bool full_tangent = (f_axis == 0 && g_axis != 0) || (f_axis == 1 && g_axis == 2);
								if (g_row == 1 || (g_row == 2 && full_tangent))
									out.positive.push_back(cell);
								else
									out.negative.push_back(cell);
							}
						}
					}
				}
			}
		}
	}
	return out;
}

// raw DF-slot tuple of one fine-local cell (all frames, all directions;
// the host mirrors take global = offset + local coordinates)
template <typename NSE>
std::vector<double> captureCell(LBM_BLOCK<NSE>& fine, int x, int y, int z)
{
	std::vector<double> snap;
	for (uint8_t d = 0; d < NSE::DFMAX; d++)
		for (int q = 0; q < NSE::Q; q++)
			snap.push_back(fine.hfs[d](q, fine.offset.x() + x, fine.offset.y() + y, fine.offset.z() + z));
	return snap;
}

// one scenario's full assertion body: SimInit, the structural patch
// census, then WC_CYCLES driven coarse cycles with the behavioral
// fill-activity witness
void runWallCornerScenario(const Scenario& sc)
{
	const WallSpec& walls = sc.walls;
	INFO("scenario " << sc.name << " (pattern " << pattern_name << ")");

	const lat_t lat = makeLattice();
	const TRAITS::bool3d periodic{! walls.any_x(), ! walls.any_y(), ! walls.any_z()};
	StateWallCorners_AMR<NSE_CONFIG> state(
		walls, fmt::format("test_amr_wall_corners_{}", sc.name), MPI_COMM_WORLD, lat, "adios2.xml", periodic, 1);
	REQUIRE(state.canCompute());
	createAMRBlocks(
		state.nse, parseAMRConfig<NSE_CONFIG>(fmt::format("1 {} {} {} {} {} {}", sc.go[0], sc.go[1], sc.go[2], WC_GS, WC_GS, WC_GS)));
	// deepen the fine block's overlap on every walled axis (the fine
	// wall's GEO_NOTHING streaming buffer row; SimInit's re-allocation
	// materializes it)
	for (auto& block : state.nse.blocks)
		if (block.level > 0) {
			if (walls.any_x())
				block.storage_overlap_x = 3;
			if (walls.any_y())
				block.storage_overlap_y = 3;
			if (walls.any_z())
				block.storage_overlap_z = 3;
		}
	state.SimInit();
	REQUIRE(! state.nse.terminate);

	// exactly one L0 -> L1 coupling, one block per level, fine interior
	// 2*gs - 2 per axis
	REQUIRE(state.couplings.size() == 1);
	REQUIRE(state.nse.getBlocksAtLevel(0).size() == 1);
	auto* fine = state.nse.getBlocksAtLevel(1).front();
	REQUIRE(fine->local.x() == 2 * WC_GS - 2);
	REQUIRE(fine->local.y() == 2 * WC_GS - 2);
	REQUIRE(fine->local.z() == 2 * WC_GS - 2);

	// the wall mask holds exactly the scenario's masked faces
	CHECK_MESSAGE(
		state.fineWallMask(*fine) == walls.maskBits(),
		fmt::format("wall mask {} != expected {}", state.fineWallMask(*fine), walls.maskBits()));

	// ---- structural: the patch table matches the expected rectangles ----
	const auto& coupling = state.couplings.front();
	REQUIRE(coupling.patches.size() == 6);
	const auto expected = expectedPatches<NSE_CONFIG>(walls, *fine);
	REQUIRE(expected.size() == 6);
	for (const auto& patch : coupling.patches) {
		const auto it = std::find_if(expected.begin(), expected.end(), [&patch](const ExpPatch& e)
		{
			return e.face == patch.face;
		});
		REQUIRE(it != expected.end());
		const bool rect_ok = patch.fine_origin == it->origin && patch.fine_size == it->size;
		CHECK_MESSAGE(
			rect_ok,
			fmt::format(
				"face {} patch origin ({},{},{}) size ({},{},{}) != expected origin ({},{},{}) size ({},{},{}){}",
				static_cast<int>(patch.face),
				patch.fine_origin.x(),
				patch.fine_origin.y(),
				patch.fine_origin.z(),
				patch.fine_size.x(),
				patch.fine_size.y(),
				patch.fine_size.z(),
				it->origin.x(),
				it->origin.y(),
				it->origin.z(),
				it->size.x(),
				it->size.y(),
				it->size.z(),
				it->emptied ? " (emptied face)" : ""
			)
		);
	}

	// ---- structural: disjointness + the corner coverage census ----
	const auto probes = probeCells<NSE_CONFIG>(walls, *fine);
	REQUIRE(! probes.positive.empty());
	REQUIRE(! probes.negative.empty());
	int disjoint_violations = 0, uncovered_corners = 0, covered_negative = 0;
	const int local[3] = {static_cast<int>(fine->local.x()), static_cast<int>(fine->local.y()), static_cast<int>(fine->local.z())};
	const int ov[3] = {fine->df_overlap_X(), fine->df_overlap_Y(), fine->df_overlap_Z()};
	for (int x = -ov[0]; x < local[0] + ov[0]; x++)
		for (int y = -ov[1]; y < local[1] + ov[1]; y++)
			for (int z = -ov[2]; z < local[2] + ov[2]; z++)
				if (coverCount(expected, x, y, z) > 1)
					disjoint_violations++;
	for (const auto& c : probes.positive)
		if (coverCount(expected, c.x(), c.y(), c.z()) != 1)
			uncovered_corners++;
	for (const auto& c : probes.negative)
		if (coverCount(expected, c.x(), c.y(), c.z()) != 0)
			covered_negative++;
	CHECK_MESSAGE(disjoint_violations == 0, fmt::format("{} cells covered by more than one patch", disjoint_violations));
	CHECK_MESSAGE(
		uncovered_corners == 0,
		fmt::format("{} corner columns not covered by exactly one patch (the symmetric corner fill is broken)", uncovered_corners));
	CHECK_MESSAGE(
		covered_negative == 0,
		fmt::format("{} wall-row/buffer cells carry a destination (the widening must stop at the first-fluid row)", covered_negative));

	// ---- behavioral: the corner columns' raw DF slots change across the
	// driven cycles (a partition hole leaves them at the initial state
	// forever; the fill re-authors them from the evolving coarse field
	// every cycle) ----
	fine->copyDFsToHost();
	std::vector<std::vector<double>> initial;
	for (const auto& c : probes.positive)
		initial.push_back(captureCell<NSE_CONFIG>(*fine, c.x(), c.y(), c.z()));
	std::vector<std::vector<double>> final;
	for (int cyc = 0; cyc < WC_CYCLES; cyc++) {
		state.updateKernelData();
		state.SimUpdate();
		REQUIRE(! state.nse.terminate);
	}
	fine->copyDFsToHost();
	for (const auto& c : probes.positive)
		final.push_back(captureCell<NSE_CONFIG>(*fine, c.x(), c.y(), c.z()));
	int frozen_corners = 0, nonfinite = 0;
	for (std::size_t i = 0; i < initial.size(); i++) {
		if (initial[i] == final[i])
			frozen_corners++;
		for (double v : final[i])
			if (! std::isfinite(v))
				nonfinite++;
	}
	CHECK_MESSAGE(
		frozen_corners == 0,
		fmt::format("{} of {} corner columns kept their initial-condition content over {} cycles (unfilled band holes)", frozen_corners, initial.size(), WC_CYCLES));
	CHECK_MESSAGE(nonfinite == 0, fmt::format("{} non-finite DF values in the corner columns", nonfinite));
}

TEST_SUITE_BEGIN("amr_wall_corners");

TEST_CASE("single wall-attached faces: y_min (the 3d963c40 probe topology)")
{
	runWallCornerScenario(wall_corner_scenarios[0]);
}

TEST_CASE("single wall-attached faces: y_max (the max-side mirror)")
{
	runWallCornerScenario(wall_corner_scenarios[1]);
}

TEST_CASE("single wall-attached faces: x_min (the symmetric complement)")
{
	runWallCornerScenario(wall_corner_scenarios[2]);
}

TEST_CASE("two-wall corner: y_z")
{
	runWallCornerScenario(wall_corner_scenarios[3]);
}

TEST_CASE("two-wall corner: x_z")
{
	runWallCornerScenario(wall_corner_scenarios[4]);
}

TEST_CASE("two-wall corner: x_y")
{
	runWallCornerScenario(wall_corner_scenarios[5]);
}

TEST_CASE("three-wall corner: x_y_z")
{
	runWallCornerScenario(wall_corner_scenarios[6]);
}

TEST_CASE("three-wall max-side corner: right_top_front")
{
	runWallCornerScenario(wall_corner_scenarios[7]);
}

TEST_SUITE_END();

}  // namespace

