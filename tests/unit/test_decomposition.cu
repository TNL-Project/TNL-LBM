/*
 * Unit tests for findNeighbors, decomposeLattice_D1Q3, and
 * decomposeLattice_D3Q27 in include/lbm3d/lattice_decomposition.h.
 *
 * ## findNeighbors
 *
 * Verified against a hand-derived expected-neighbor function for uniform
 * decompositions along each single axis (x, y, z), in a 2D plane (xy), and
 * in 3D (xyz), sweeping all 8 periodicity combinations {x,y,z} ∈ {F,T}³.
 *
 * The expected-neighbor function models the vertex-based search:
 * for each axis component of a direction, periodic boundary blocks wrap
 * to the opposite end, non-periodic boundary blocks have no MPI peer
 * (→ -1 for the whole direction), and single-block axes (nblocks == 1)
 * always yield -1 for that axis (periodic → self-match, non-periodic → wall).
 * If the wrapped vertex lands on the own block, the neighbor is -1.
 *
 * ## decomposeLattice_D1Q3 / decomposeLattice_D3Q27 (nproc == 1)
 *
 * With a single MPI rank the block owns the full global lattice, all
 * neighbor IDs are -1 (self-match or wall), and data.periodic must match
 * the input.
 *
 * ## AMR region->block geometry conversion helpers (amr_decomposition.h)
 *
 * Pure per-component integer math, no GPU fixture: the region file is
 * specified in level-0 coordinates for EVERY level, while createAMRBlocks
 * stores the block origin in the immediate PARENT frame and sizes the
 * re-anchored fine interior. The cases below pin the conversion formulas
 *
 *   parent-frame origin:  go     = origin >> (level - 1)
 *   fine offset:          offset = 2 * (origin >> (level - 1)) + 1
 *   fine interior local:  local  = 2 * (size >> (level - 1)) - 2
 *
 * The (level - 1) shift is exact by the phase-1 alignment check
 * (origin/size must be multiples of 2^(level-1)); odd components are only
 * expressible at level 1, where the shift is 0. The level-1 battery asserts
 * the helpers reproduce the pre-refactor formulas bit-for-bit (go =
 * origin, offset = 2*origin + 1, local = 2*size - 2) -- the
 * bit-identity-by-construction evidence of the parent-frame normalization
 * refactor. End-to-end level-1 construction through createAMRBlocks is
 * covered by the gate mocks (test_amr_subcycling / test_amr_coupling
 * fixtures), so it is not duplicated here; levels >= 2 cannot be created
 * end-to-end while the level>1 guard is active, so their coverage is the
 * hand-computed conversion values below.
 */

#include <map>
#include <vector>

#include <doctest/doctest.h>

#include <TNL/Containers/Block.h>
#include <TNL/Containers/DistributedNDArraySyncDirections.h>
#include <TNL/Containers/StaticVector.h>
#include <TNL/MPI/Comm.h>

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
#include "lbm3d/lattice_decomposition.h"
#include "lbm3d/amr_decomposition.h"

using TRAITS = Traits<float, double, int>;
using COLL = D2Q9_SRT<TRAITS>;
using CONFIG =
	LBM_CONFIG<TRAITS, D2Q9_KernelStruct, NSE_Data<TRAITS>, COLL, typename COLL::EQ, D2Q9_STREAMING<TRAITS>, D2Q9_BC_All, D2Q9_MACRO_Default<TRAITS>>;
using BLOCK = LBM_BLOCK<CONFIG>;
using idx = typename TRAITS::idx;
using idx3d = typename TRAITS::idx3d;
using bool3d = typename TRAITS::bool3d;
using Block3 = TNL::Containers::Block<3, idx>;
using SD = TNL::Containers::SyncDirection;

// Build a uniform nx×ny×nz decomposition of [0,gx)×[0,gy)×[0,gz).
// Ranks assigned in z-major, then y-major order:
//   rank = rz * nx*ny + ry * nx + rx
static std::vector<Block3> makeUniformDecomposition(idx3d global, int nx, int ny, int nz)
{
	std::vector<Block3> decomp;
	const idx sx = global.x() / nx;
	const idx sy = global.y() / ny;
	const idx sz = global.z() / nz;
	for (int rz = 0; rz < nz; rz++)
		for (int ry = 0; ry < ny; ry++)
			for (int rx = 0; rx < nx; rx++) {
				idx3d begin{rx * sx, ry * sy, rz * sz};
				idx3d end{(rx + 1) * sx, (ry + 1) * sy, (rz + 1) * sz};
				decomp.emplace_back(begin, end);
			}
	return decomp;
}

static int rankFromXYZ(int rx, int ry, int rz, int nx, int ny)
{
	return rz * nx * ny + ry * nx + rx;
}

// Expected neighbor rank for a direction in a uniform decomposition.
// Returns -1 when the vertex search finds no peer (wall boundary or self-match).
static int expectedNeighbor(int rank, SD dir, int nx, int ny, int nz, bool3d periodic)
{
	const int rx = rank % nx;
	const int ry = (rank / nx) % ny;
	const int rz = rank / (nx * ny);
	int nrx = rx;
	int nry = ry;
	int nrz = rz;

	// For each axis in the direction, resolve the neighbor coordinate.
	// Returns false (→ -1 for the whole direction) when the vertex will not
	// match any block in the decomposition.
	auto resolve = [&](int r, int& nr, int nblocks, bool isPeriodic) -> bool
	{
		if (nblocks == 1) {
			if (isPeriodic) {
				nr = r;	 // wraps to self in this axis
				return true;
			}
			return false;  // wall: vertex won't match any block
		}
		if (nr < 0 || nr >= nblocks) {
			if (isPeriodic) {
				nr = (nr + nblocks) % nblocks;
				return true;
			}
			return false;  // wall
		}
		return true;  // interior
	};

	bool ok = true;
	if ((dir & SD::Left) != SD::None) {
		nrx = rx - 1;
		ok &= resolve(rx, nrx, nx, periodic.x());
	}
	else if ((dir & SD::Right) != SD::None) {
		nrx = rx + 1;
		ok &= resolve(rx, nrx, nx, periodic.x());
	}
	if ((dir & SD::Bottom) != SD::None) {
		nry = ry - 1;
		ok &= resolve(ry, nry, ny, periodic.y());
	}
	else if ((dir & SD::Top) != SD::None) {
		nry = ry + 1;
		ok &= resolve(ry, nry, ny, periodic.y());
	}
	if ((dir & SD::Back) != SD::None) {
		nrz = rz - 1;
		ok &= resolve(rz, nrz, nz, periodic.z());
	}
	else if ((dir & SD::Front) != SD::None) {
		nrz = rz + 1;
		ok &= resolve(rz, nrz, nz, periodic.z());
	}

	if (! ok)
		return -1;

	int neighborRank = rankFromXYZ(nrx, nry, nrz, nx, ny);
	if (neighborRank == rank)
		return -1;	// self-match
	return neighborRank;
}

// Call findNeighbors for every rank and compare against expectedNeighbor
// for all 26 D3Q27 directions.
static void checkDecomposition(const std::vector<Block3>& decomp, const Block3& global, int nx, int ny, int nz, bool3d periodic)
{
	for (int rank = 0; rank < (int) decomp.size(); rank++) {
		auto neighbors = findNeighbors(TNL::Containers::NDArraySyncPatterns::D3Q27, rank, decomp, global, periodic);
		for (SD dir : TNL::Containers::NDArraySyncPatterns::D3Q27) {
			int expected = expectedNeighbor(rank, dir, nx, ny, nz, periodic);
			auto it = neighbors.find(dir);
			if (it == neighbors.end()) {
				FAIL_CHECK(
					fmt::format(
						"rank {} direction {:#x} missing from neighbors map (periodic={},{},{})",
						rank,
						static_cast<unsigned>(dir),
						periodic.x(),
						periodic.y(),
						periodic.z()
					)
				);
			}
			if (it->second != expected) {
				FAIL_CHECK(
					fmt::format(
						"rank {} direction {:#x}: expected {} got {} (periodic={},{},{})",
						rank,
						static_cast<unsigned>(dir),
						expected,
						it->second,
						periodic.x(),
						periodic.y(),
						periodic.z()
					)
				);
			}
		}
	}
}

// All 8 periodicity combinations {x,y,z} ∈ {F,T}³.
static const bool3d PERIODIC_COMBOS[] = {
	{false, false, false},
	{true, false, false},
	{false, true, false},
	{false, false, true},
	{true, true, false},
	{true, false, true},
	{false, true, true},
	{true, true, true},
};

TEST_SUITE_BEGIN("decomposition");

TEST_CASE("findNeighbors: 1D x-split 4 blocks")
{
	const idx3d global{32, 8, 8};
	auto decomp = makeUniformDecomposition(global, 4, 1, 1);
	const Block3 globalBlock{{0, 0, 0}, global};
	for (const bool3d& p : PERIODIC_COMBOS)
		checkDecomposition(decomp, globalBlock, 4, 1, 1, p);
}

TEST_CASE("findNeighbors: 1D y-split 4 blocks")
{
	const idx3d global{8, 32, 8};
	auto decomp = makeUniformDecomposition(global, 1, 4, 1);
	const Block3 globalBlock{{0, 0, 0}, global};
	for (const bool3d& p : PERIODIC_COMBOS)
		checkDecomposition(decomp, globalBlock, 1, 4, 1, p);
}

TEST_CASE("findNeighbors: 1D z-split 4 blocks")
{
	const idx3d global{8, 8, 32};
	auto decomp = makeUniformDecomposition(global, 1, 1, 4);
	const Block3 globalBlock{{0, 0, 0}, global};
	for (const bool3d& p : PERIODIC_COMBOS)
		checkDecomposition(decomp, globalBlock, 1, 1, 4, p);
}

TEST_CASE("findNeighbors: 2D xy-split 2x2 blocks")
{
	const idx3d global{16, 16, 8};
	auto decomp = makeUniformDecomposition(global, 2, 2, 1);
	const Block3 globalBlock{{0, 0, 0}, global};
	for (const bool3d& p : PERIODIC_COMBOS)
		checkDecomposition(decomp, globalBlock, 2, 2, 1, p);
}

TEST_CASE("findNeighbors: 3D xyz-split 2x2x2 blocks")
{
	const idx3d global{16, 16, 16};
	auto decomp = makeUniformDecomposition(global, 2, 2, 2);
	const Block3 globalBlock{{0, 0, 0}, global};
	for (const bool3d& p : PERIODIC_COMBOS)
		checkDecomposition(decomp, globalBlock, 2, 2, 2, p);
}

TEST_CASE("findNeighbors: self-match 1 block")
{
	const idx3d global{8, 8, 8};
	auto decomp = makeUniformDecomposition(global, 1, 1, 1);
	const Block3 globalBlock{{0, 0, 0}, global};
	for (bool3d p : PERIODIC_COMBOS) {
		auto neighbors = findNeighbors(TNL::Containers::NDArraySyncPatterns::D3Q27, 0, decomp, globalBlock, p);
		for (SD dir : TNL::Containers::NDArraySyncPatterns::D3Q27) {
			auto it = neighbors.find(dir);
			if (it == neighbors.end())
				FAIL_CHECK(fmt::format("direction {:#x} missing from neighbors map", static_cast<unsigned>(dir)));
			if (it->second != -1)
				FAIL_CHECK(
					fmt::format(
						"expected -1 for direction {:#x} (single block), got {} (periodic={},{},{})",
						static_cast<unsigned>(dir),
						it->second,
						p.x(),
						p.y(),
						p.z()
					)
				);
		}
	}
}

TEST_CASE("findNeighbors: spot-check 4-block x-split non-periodic")
{
	const idx3d global{32, 8, 8};
	auto decomp = makeUniformDecomposition(global, 4, 1, 1);
	const Block3 globalBlock{{0, 0, 0}, global};
	const bool3d np{false, false, false};
	auto n = findNeighbors(TNL::Containers::NDArraySyncPatterns::D3Q27, 0, decomp, globalBlock, np);
	CHECK(n.at(SD::Left) == -1);
	CHECK(n.at(SD::Right) == 1);
	CHECK(n.at(SD::Top) == -1);
	CHECK(n.at(SD::Bottom) == -1);
	CHECK(n.at(SD::Back) == -1);
	CHECK(n.at(SD::Front) == -1);
	n = findNeighbors(TNL::Containers::NDArraySyncPatterns::D3Q27, 3, decomp, globalBlock, np);
	CHECK(n.at(SD::Left) == 2);
	CHECK(n.at(SD::Right) == -1);
	n = findNeighbors(TNL::Containers::NDArraySyncPatterns::D3Q27, 1, decomp, globalBlock, np);
	CHECK(n.at(SD::Left) == 0);
	CHECK(n.at(SD::Right) == 2);
}

TEST_CASE("findNeighbors: spot-check 4-block x-split x-periodic")
{
	const idx3d global{32, 8, 8};
	auto decomp = makeUniformDecomposition(global, 4, 1, 1);
	const Block3 globalBlock{{0, 0, 0}, global};
	const bool3d p{true, false, false};
	auto n0 = findNeighbors(TNL::Containers::NDArraySyncPatterns::D3Q27, 0, decomp, globalBlock, p);
	CHECK(n0.at(SD::Left) == 3);
	CHECK(n0.at(SD::Right) == 1);
	CHECK(n0.at(SD::Top) == -1);
	CHECK(n0.at(SD::TopLeft) == -1);  // y not periodic → wall
	auto n3 = findNeighbors(TNL::Containers::NDArraySyncPatterns::D3Q27, 3, decomp, globalBlock, p);
	CHECK(n3.at(SD::Left) == 2);
	CHECK(n3.at(SD::Right) == 0);
}

TEST_CASE("findNeighbors: spot-check 4-block x-split all-periodic")
{
	const idx3d global{32, 8, 8};
	auto decomp = makeUniformDecomposition(global, 4, 1, 1);
	const Block3 globalBlock{{0, 0, 0}, global};
	const bool3d p{true, true, true};
	auto n0 = findNeighbors(TNL::Containers::NDArraySyncPatterns::D3Q27, 0, decomp, globalBlock, p);
	CHECK(n0.at(SD::Left) == 3);
	CHECK(n0.at(SD::Right) == 1);
	CHECK(n0.at(SD::Top) == -1);	  // self-match (ny=1)
	CHECK(n0.at(SD::Bottom) == -1);	  // self-match
	CHECK(n0.at(SD::Back) == -1);	  // self-match
	CHECK(n0.at(SD::Front) == -1);	  // self-match
	CHECK(n0.at(SD::TopLeft) == 3);	  // x wraps to 3, y wraps to self
	CHECK(n0.at(SD::TopRight) == 1);  // x stays, y wraps to self
	CHECK(n0.at(SD::BottomLeft) == 3);
	CHECK(n0.at(SD::BottomRight) == 1);
	CHECK(n0.at(SD::BackTopLeft) == 3);	 // all three wrap: x→3, y→self, z→self
	CHECK(n0.at(SD::BackTopRight) == 1);
	CHECK(n0.at(SD::BackBottomLeft) == 3);
	CHECK(n0.at(SD::BackBottomRight) == 1);
	CHECK(n0.at(SD::FrontTopLeft) == 3);
	CHECK(n0.at(SD::FrontTopRight) == 1);
	CHECK(n0.at(SD::FrontBottomLeft) == 3);
	CHECK(n0.at(SD::FrontBottomRight) == 1);
}

TEST_CASE("findNeighbors: spot-check 2x2 xy-split all-periodic")
{
	const idx3d global{16, 16, 8};
	auto decomp = makeUniformDecomposition(global, 2, 2, 1);
	const Block3 globalBlock{{0, 0, 0}, global};
	const bool3d p{true, true, false};
	// rank 0 = block (0,0), rank 1 = (1,0), rank 2 = (0,1), rank 3 = (1,1)
	auto n0 = findNeighbors(TNL::Containers::NDArraySyncPatterns::D3Q27, 0, decomp, globalBlock, p);
	CHECK(n0.at(SD::Left) == 1);  // x wraps
	CHECK(n0.at(SD::Right) == 1);
	CHECK(n0.at(SD::Top) == 2);		 // y stays interior? No, ry=0, Top → ry=1 → rank 2
	CHECK(n0.at(SD::Bottom) == 2);	 // y wraps
	CHECK(n0.at(SD::TopLeft) == 3);	 // (rx=1, ry=1) = rank 3
	CHECK(n0.at(SD::TopRight) == 3);
	CHECK(n0.at(SD::BottomLeft) == 3);
	CHECK(n0.at(SD::BottomRight) == 3);
	CHECK(n0.at(SD::Back) == -1);		  // z not periodic, nz=1 → wall
	CHECK(n0.at(SD::BackTopLeft) == -1);  // z not periodic → directionPeriodic=false
}

TEST_CASE("decomposeLattice_D1Q3: nproc=1 block properties")
{
	const idx3d global{32, 8, 8};
	for (bool3d p : PERIODIC_COMBOS) {
		BLOCK block = decomposeLattice_D1Q3<CONFIG, idx>(MPI_COMM_WORLD, global, p);
		CHECK(block.local == global);
		CHECK(block.offset == idx3d{0, 0, 0});
		CHECK(block.data.periodic == p);
		CHECK(block.neighborIDs.size() == 2);
		CHECK(block.neighborIDs.at(SD::Left) == -1);
		CHECK(block.neighborIDs.at(SD::Right) == -1);
		CHECK(block.neighborRanks.at(SD::Left) == -1);
		CHECK(block.neighborRanks.at(SD::Right) == -1);
	}
}

TEST_CASE("decomposeLattice_D3Q27: nproc=1 block properties")
{
	const idx3d global{16, 16, 16};
	for (bool3d p : PERIODIC_COMBOS) {
		BLOCK block = decomposeLattice_D3Q27<CONFIG, idx>(MPI_COMM_WORLD, global, p);
		CHECK(block.local == global);
		CHECK(block.offset == idx3d{0, 0, 0});
		CHECK(block.data.periodic == p);
		CHECK(block.neighborIDs.size() == 26);
		for (SD dir : TNL::Containers::NDArraySyncPatterns::D3Q27) {
			CHECK(block.neighborIDs.at(dir) == -1);
			CHECK(block.neighborRanks.at(dir) == -1);
		}
	}
}

#ifdef HAVE_MPI

static const std::array<SD, 26>& directions26()
{
	static const std::array<SD, 26> dirs = TNL::Containers::NDArraySyncPatterns::D3Q27;
	return dirs;
}

static int dirToIndex26(SD dir)
{
	const auto& dirs = directions26();
	for (int i = 0; i < 26; i++)
		if (dirs[i] == dir)
			return i;
	return -1;
}

// tiling invariants: total volume == global volume, no pairwise overlap
static void check_tiling(const BLOCK& block, const idx3d& global)
{
	const int nproc = TNL::MPI::GetSize(MPI_COMM_WORLD);

	CHECK(block.global == global);
	CHECK(block.local.x() >= 1);
	CHECK(block.local.y() >= 1);
	CHECK(block.local.z() >= 1);
	CHECK(block.offset.x() >= 0);
	CHECK(block.offset.y() >= 0);
	CHECK(block.offset.z() >= 0);
	CHECK(block.offset.x() + block.local.x() <= global.x());
	CHECK(block.offset.y() + block.local.y() <= global.y());
	CHECK(block.offset.z() + block.local.z() <= global.z());

	idx local_data[6] = {
		block.offset.x(),
		block.offset.y(),
		block.offset.z(),
		block.local.x(),
		block.local.y(),
		block.local.z(),
	};
	std::vector<idx> all_data(6 * nproc);
	MPI_Allgather(local_data, 6, MPI_INT, all_data.data(), 6, MPI_INT, MPI_COMM_WORLD);

	idx total_vol = 0;
	for (int r = 0; r < nproc; r++)
		total_vol += all_data[6 * r + 3] * all_data[6 * r + 4] * all_data[6 * r + 5];
	CHECK(total_vol == global.x() * global.y() * global.z());

	for (int r1 = 0; r1 < nproc; r1++) {
		for (int r2 = r1 + 1; r2 < nproc; r2++) {
			idx x1b = all_data[6 * r1 + 0];
			idx x1e = x1b + all_data[6 * r1 + 3];
			idx y1b = all_data[6 * r1 + 1];
			idx y1e = y1b + all_data[6 * r1 + 4];
			idx z1b = all_data[6 * r1 + 2];
			idx z1e = z1b + all_data[6 * r1 + 5];
			idx x2b = all_data[6 * r2 + 0];
			idx x2e = x2b + all_data[6 * r2 + 3];
			idx y2b = all_data[6 * r2 + 1];
			idx y2e = y2b + all_data[6 * r2 + 4];
			idx z2b = all_data[6 * r2 + 2];
			idx z2e = z2b + all_data[6 * r2 + 5];
			bool overlap = (x1b < x2e && x2b < x1e && y1b < y2e && y2b < y1e && z1b < z2e && z2b < z1e);
			CHECK_FALSE(overlap);
		}
	}
}

// if A→B in direction D, then B→A in opposite(D)
static void check_reciprocity(const BLOCK& block)
{
	const int nproc = TNL::MPI::GetSize(MPI_COMM_WORLD);

	int my_neighbors[26];
	const auto& dirs = directions26();
	for (int i = 0; i < 26; i++) {
		auto it = block.neighborIDs.find(dirs[i]);
		my_neighbors[i] = (it != block.neighborIDs.end()) ? it->second : -1;
	}

	std::vector<int> all_neighbors(26 * nproc);
	MPI_Allgather(my_neighbors, 26, MPI_INT, all_neighbors.data(), 26, MPI_INT, MPI_COMM_WORLD);

	for (int a = 0; a < nproc; a++) {
		for (int d = 0; d < 26; d++) {
			int b = all_neighbors[a * 26 + d];
			if (b < 0 || b >= nproc)
				continue;
			SD opp = TNL::Containers::opposite(dirs[d]);
			int opp_idx = dirToIndex26(opp);
			int back = all_neighbors[b * 26 + opp_idx];
			CHECK(back == a);
		}
	}
}

static void check_multi_rank_tiling()
{
	const idx3d global{16, 16, 16};
	for (bool3d p : PERIODIC_COMBOS) {
		BLOCK block = decomposeLattice_D3Q27<CONFIG, idx>(MPI_COMM_WORLD, global, p);
		CHECK(block.data.periodic == p);
		CHECK(block.rank == TNL::MPI::GetRank(MPI_COMM_WORLD));
		CHECK(block.nproc == TNL::MPI::GetSize(MPI_COMM_WORLD));
		check_tiling(block, global);
	}
}

static void check_multi_rank_reciprocity()
{
	const idx3d global{16, 16, 16};
	for (const bool3d& p : PERIODIC_COMBOS) {
		BLOCK block = decomposeLattice_D3Q27<CONFIG, idx>(MPI_COMM_WORLD, global, p);
		check_reciprocity(block);
	}
}

// In a 2×2×2 grid every block is a corner.  For each axis the block has
// n_valid = 2 (periodic: both ± wrap to the other block) or 1 (non-periodic:
// only the interior direction).  Valid neighbor count over the 26 D3Q27
// directions is (a+b+c) + (ab+ac+bc) + abc, giving 7, 11, 17, or 26.
static int expected_valid_2x2x2(bool3d p)
{
	int a = p.x() ? 2 : 1;
	int b = p.y() ? 2 : 1;
	int c = p.z() ? 2 : 1;
	return (a + b + c) + (a * b + a * c + b * c) + (a * b * c);
}

static void check_multi_rank_2x2x2()
{
	if (TNL::MPI::GetSize(MPI_COMM_WORLD) != 8)
		return;

	const idx3d global{16, 16, 16};
	auto decomp = makeUniformDecomposition(global, 2, 2, 2);
	const Block3 globalBlock{{0, 0, 0}, global};
	const int rank = TNL::MPI::GetRank(MPI_COMM_WORLD);
	const auto& dirs = directions26();

	for (const bool3d& p : PERIODIC_COMBOS) {
		auto neighbors = findNeighbors(TNL::Containers::NDArraySyncPatterns::D3Q27, rank, decomp, globalBlock, p);

		// verify every direction against the reference model
		for (SD dir : dirs) {
			int expected = expectedNeighbor(rank, dir, 2, 2, 2, p);
			CHECK(neighbors.at(dir) == expected);
		}

		// verify reciprocity via Allgather
		int my_neighbors[26];
		for (int i = 0; i < 26; i++)
			my_neighbors[i] = neighbors.at(dirs[i]);
		std::vector<int> all_nb(26 * 8);
		MPI_Allgather(my_neighbors, 26, MPI_INT, all_nb.data(), 26, MPI_INT, MPI_COMM_WORLD);
		for (int a = 0; a < 8; a++) {
			for (int d = 0; d < 26; d++) {
				int b = all_nb[a * 26 + d];
				if (b < 0)
					continue;
				int opp_idx = dirToIndex26(TNL::Containers::opposite(dirs[d]));
				CHECK(all_nb[b * 26 + opp_idx] == a);
			}
		}

		// valid neighbor count matches the 2×2×2 formula
		int valid = 0;
		for (SD dir : dirs)
			if (neighbors.at(dir) >= 0)
				valid++;
		CHECK(valid == expected_valid_2x2x2(p));
	}
}

TEST_CASE("decomposeLattice_D3Q27: multi-rank tiling np2")
{
	check_multi_rank_tiling();
}
TEST_CASE("decomposeLattice_D3Q27: multi-rank tiling np3")
{
	check_multi_rank_tiling();
}
TEST_CASE("decomposeLattice_D3Q27: multi-rank neighbor reciprocity np2")
{
	check_multi_rank_reciprocity();
}
TEST_CASE("decomposeLattice_D3Q27: multi-rank neighbor reciprocity np3")
{
	check_multi_rank_reciprocity();
}
TEST_CASE("decomposeLattice_D3Q27: multi-rank 2x2x2 np8")
{
	check_multi_rank_2x2x2();
}

#endif	// HAVE_MPI

// one row of the hand-computed conversion table: region file values (level-0
// coordinates) -> expected per-component helper results
struct AMRConversionCase
{
	idx3d origin;
	idx3d size;
	idx3d parent_origin;  // amrParentFrameOrigin per component
	idx3d fine_offset;	  // amrFineOffset per component
	idx3d fine_local;	  // amrFineLocal per component
};

static void checkAMRConversionCase(const AMRConversionCase& c, int level)
{
	for (int a = 0; a < 3; a++) {
		CHECK(amrParentFrameOrigin(c.origin[a], level) == c.parent_origin[a]);
		CHECK(amrFineOffset(c.origin[a], level) == c.fine_offset[a]);
		CHECK(amrFineLocal(c.size[a], level) == c.fine_local[a]);
	}
}

TEST_CASE("amr conversion helpers: level-1 identity with the pre-refactor formulas")
{
	// compile-time pins (the helpers are constexpr)
	static_assert(amrParentFrameOrigin(7, 1) == 7);
	static_assert(amrFineOffset(7, 1) == 15);
	static_assert(amrFineLocal(7, 1) == 12);

	// origins/sizes including odd values, which are only legal at level 1
	// (the level-1 alignment multiplier is 1); at level 1 the >> 0 shift is
	// the identity and the helpers must equal the pre-refactor formulas
	const idx origins[] = {0, 1, 2, 3, 5, 7, 24, 33, 100, 201};
	const idx sizes[] = {3, 4, 5, 6, 7, 8, 15, 16, 31, 64};
	for (idx o : origins) {
		CHECK(amrParentFrameOrigin(o, 1) == o);
		CHECK(amrFineOffset(o, 1) == 2 * o + 1);
	}
	for (idx s : sizes)
		CHECK(amrFineLocal(s, 1) == 2 * s - 2);
}

TEST_CASE("amr conversion helpers: level-2 parent-frame values")
{
	// hand-computed with the level-2 parent shift 2^(level-1) = 2:
	// parent = origin >> 1, offset = 2*parent + 1, local = 2*(size >> 1) - 2
	static_assert(amrParentFrameOrigin(24, 2) == 12);
	static_assert(amrFineOffset(24, 2) == 25);
	static_assert(amrFineLocal(16, 2) == 14);

	const AMRConversionCase cases[] = {
		{{24, 8, 4}, {16, 8, 8}, {12, 4, 2}, {25, 9, 5}, {14, 6, 6}},
		{{0, 6, 10}, {4, 6, 12}, {0, 3, 5}, {1, 7, 11}, {2, 4, 10}},
		{{2, 34, 100}, {10, 6, 4}, {1, 17, 50}, {3, 35, 101}, {8, 4, 2}},
	};
	for (const AMRConversionCase& c : cases)
		checkAMRConversionCase(c, 2);
}

TEST_CASE("amr conversion helpers: level-3 parent-frame values incl. zero components")
{
	// hand-computed with the level-3 parent shift 2^(level-1) = 4
	static_assert(amrParentFrameOrigin(40, 3) == 10);
	static_assert(amrFineOffset(40, 3) == 21);
	static_assert(amrFineLocal(24, 3) == 10);

	const AMRConversionCase cases[] = {
		{{40, 16, 8}, {24, 16, 8}, {10, 4, 2}, {21, 9, 5}, {10, 6, 2}},
		{{0, 48, 32}, {32, 8, 16}, {0, 12, 8}, {1, 25, 17}, {14, 2, 6}},
		{{0, 0, 0}, {8, 8, 8}, {0, 0, 0}, {1, 1, 1}, {2, 2, 2}},
	};
	for (const AMRConversionCase& c : cases)
		checkAMRConversionCase(c, 3);
}

TEST_SUITE_END();
