/*
 * Unit tests for applySymmetryCorner in both D2Q9 and D3Q27 BC_All.
 *
 * Tests the generalized symmetry-corner closure: for each direction with a
 * GEO_SYMMETRY neighbor, checks that neighbor's perpendicular directions for
 * GEO_NOTHING to find the mirror half-planes. Verifies:
 * - no symmetry neighbor → no closure
 * - single-plane symmetry → correct mirror direction
 * - same-axis double ghost (bc_face::XM|bc_face::XP) → destructive one-sided copy
 * - GEO_NOTHING without adjacent GEO_SYMMETRY → no closure (ignored)
 * - two-axis symmetry (x and y simultaneously) → both axes mirrored
 */

#include <cstdint>

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
using KS = D2Q9_KernelStruct<typename TRAITS::dreal>;

static BLOCK makeBlock(idx nx, idx ny)
{
	const idx3d global{nx, ny, 1};
	const idx3d loc{nx, ny, 1};
	const idx3d off{0, 0, 0};
	BLOCK block{MPI_COMM_WORLD, global, loc, off, 0};
	block.allocateHostData();
	block.hmap.setValue(BC::GEO_FLUID);
	block.copyMapToDevice();
	// set up data.dmap/indexer for host-side access (no GPU allocation)
#ifdef HAVE_MPI
	block.data.indexer = block.hmap.getLocalView().getIndexer();
#else
	block.data.indexer = block.hmap.getIndexer();
#endif
	block.data.XYZ = block.data.indexer.getStorageSize();
	block.data.dmap = block.hmap.getData();
	return block;
}

static void stamp(BLOCK& block, idx x, idx y, typename BC::map_t tag)
{
	block.hmap(x, y, 0) = tag;
}

static KS makeKS()
{
	KS ks;
	for (int i = 0; i < 9; i++)
		ks.f[i] = static_cast<typename TRAITS::dreal>(i + 1) * 10.0f;
	ks.rho = 1.0;
	ks.vx = 0.0;
	ks.vy = 0.0;
	ks.fx = 0.0;
	ks.fy = 0.0;
	return ks;
}

TEST_SUITE_BEGIN("bcsymmetry");

// no symmetry neighbor → KS unchanged
TEST_CASE("bcsymmetry: no-symmetry-neighbor")
{
	auto block = makeBlock(6, 6);
	stamp(block, 2, 2, BC::GEO_INFLOW_LEFT);

	KS ks = makeKS();
	BC::applySymmetryCorner(block.data, ks, 1, 2, 3, 1, 2, 3, 0, 0, 0);

	for (int i = 0; i < 9; i++)
		CHECK(ks.f[i] == doctest::Approx(makeKS().f[i]));
}

// GEO_NOTHING at y-1 but NO symmetry neighbor → no closure
TEST_CASE("bcsymmetry: ghost-without-symmetry")
{
	auto block = makeBlock(6, 6);
	for (idx x = 0; x < 6; x++)
		stamp(block, x, 0, BC::GEO_NOTHING);
	stamp(block, 1, 1, BC::GEO_INFLOW_LEFT);

	KS ks = makeKS();
	BC::applySymmetryCorner(block.data, ks, 0, 1, 2, 0, 1, 2, 0, 0, 0);

	for (int i = 0; i < 9; i++)
		CHECK(ks.f[i] == doctest::Approx(makeKS().f[i]));
}

// symmetry at x+1=(2,1), ghost at (1,0)=BC cell's y-1 → bc_face::YM
TEST_CASE("bcsymmetry: single-plane-ym")
{
	auto block = makeBlock(6, 6);
	stamp(block, 2, 1, BC::GEO_SYMMETRY);
	stamp(block, 1, 0, BC::GEO_NOTHING);
	stamp(block, 1, 1, BC::GEO_INFLOW_LEFT);

	KS ks = makeKS();
	BC::applySymmetryCorner(block.data, ks, 0, 1, 2, 0, 1, 2, 0, 0, 0);

	// bc_face::YM: zp←zm, pp←pm, mp←mm (only y flipped, x stays)
	CHECK(ks.f[dir9::zp] == doctest::Approx(ks.f[dir9::zm]));
	CHECK(ks.f[dir9::pp] == doctest::Approx(ks.f[dir9::pm]));
	CHECK(ks.f[dir9::mp] == doctest::Approx(ks.f[dir9::mm]));
	CHECK(ks.f[dir9::zz] == doctest::Approx(10.0f));
	CHECK(ks.f[dir9::pz] == doctest::Approx(20.0f));
	CHECK(ks.f[dir9::mz] == doctest::Approx(30.0f));
}

// symmetry at y+1=(1,3), ghost at (0,2)=BC cell's x-1 → bc_face::XM
TEST_CASE("bcsymmetry: single-plane-xm")
{
	auto block = makeBlock(6, 6);
	stamp(block, 1, 3, BC::GEO_SYMMETRY);
	stamp(block, 0, 2, BC::GEO_NOTHING);
	stamp(block, 1, 2, BC::GEO_INFLOW_LEFT);

	KS ks = makeKS();
	BC::applySymmetryCorner(block.data, ks, 0, 1, 2, 1, 2, 3, 0, 0, 0);

	// bc_face::XM: pz←mz=30, pp←mp=90, pm←mm=70
	CHECK(ks.f[dir9::pz] == doctest::Approx(30.0f));
	CHECK(ks.f[dir9::pp] == doctest::Approx(90.0f));
	CHECK(ks.f[dir9::pm] == doctest::Approx(70.0f));
	CHECK(ks.f[dir9::zz] == doctest::Approx(10.0f));
	CHECK(ks.f[dir9::zp] == doctest::Approx(40.0f));
	CHECK(ks.f[dir9::zm] == doctest::Approx(50.0f));
}

// symmetry at x+1, ghost at y+1 → bc_face::YP
TEST_CASE("bcsymmetry: single-plane-yp")
{
	auto block = makeBlock(6, 6);
	stamp(block, 2, 1, BC::GEO_SYMMETRY);
	stamp(block, 1, 2, BC::GEO_NOTHING);
	stamp(block, 1, 1, BC::GEO_INFLOW_LEFT);

	KS ks = makeKS();
	BC::applySymmetryCorner(block.data, ks, 0, 1, 2, 0, 1, 2, 0, 0, 0);

	// bc_face::YP: zm←zp=40, mm←mp=90, pm←pp=60
	CHECK(ks.f[dir9::zm] == doctest::Approx(40.0f));
	CHECK(ks.f[dir9::mm] == doctest::Approx(90.0f));
	CHECK(ks.f[dir9::pm] == doctest::Approx(60.0f));
	CHECK(ks.f[dir9::zz] == doctest::Approx(10.0f));
	CHECK(ks.f[dir9::pz] == doctest::Approx(20.0f));
	CHECK(ks.f[dir9::mz] == doctest::Approx(30.0f));
}

// symmetry at y+1, ghost at x+1 → bc_face::XP
TEST_CASE("bcsymmetry: single-plane-xp")
{
	auto block = makeBlock(6, 6);
	stamp(block, 1, 3, BC::GEO_SYMMETRY);
	stamp(block, 2, 2, BC::GEO_NOTHING);
	stamp(block, 1, 2, BC::GEO_INFLOW_LEFT);

	KS ks = makeKS();
	BC::applySymmetryCorner(block.data, ks, 0, 1, 2, 1, 2, 3, 0, 0, 0);

	// bc_face::XP: mz←pz=20, mm←pm=80, mp←pp=60
	CHECK(ks.f[dir9::mz] == doctest::Approx(20.0f));
	CHECK(ks.f[dir9::mm] == doctest::Approx(80.0f));
	CHECK(ks.f[dir9::mp] == doctest::Approx(60.0f));
	CHECK(ks.f[dir9::zz] == doctest::Approx(10.0f));
	CHECK(ks.f[dir9::pz] == doctest::Approx(20.0f));
	CHECK(ks.f[dir9::zp] == doctest::Approx(40.0f));
}

// symmetry on both x and y axes simultaneously → bc_face::XM|bc_face::YM
TEST_CASE("bcsymmetry: two-axis-symmetry")
{
	auto block = makeBlock(6, 6);
	stamp(block, 2, 1, BC::GEO_SYMMETRY);  // x+1 symmetry
	stamp(block, 1, 2, BC::GEO_SYMMETRY);  // y+1 symmetry
	stamp(block, 0, 1, BC::GEO_NOTHING);   // x-1 ghost
	stamp(block, 1, 0, BC::GEO_NOTHING);    // y-1 ghost
	stamp(block, 1, 1, BC::GEO_INFLOW_LEFT);

	KS ks = makeKS();
	BC::applySymmetryCorner(block.data, ks, 0, 1, 2, 0, 1, 2, 0, 0, 0);

	// ghosts = bc_face::XM | bc_face::YM
	// pp←mm=70 (both x and y flipped)
	CHECK(ks.f[dir9::pp] == doctest::Approx(70.0f));
	// pz←mz=30 (x flipped, y=z)
	CHECK(ks.f[dir9::pz] == doctest::Approx(30.0f));
	// zp←zm=50 (x=z, y flipped)
	CHECK(ks.f[dir9::zp] == doctest::Approx(50.0f));
	// pm←mm=70 (x flipped, y=m)
	CHECK(ks.f[dir9::pm] == doctest::Approx(70.0f));
	// mp←mm=70 (x=m, y flipped)
	CHECK(ks.f[dir9::mp] == doctest::Approx(70.0f));
	// zz unchanged
	CHECK(ks.f[dir9::zz] == doctest::Approx(10.0f));
}

// symmetry at y-1, ghosts at both x-1 and x+1 (BC cell's own neighbors) → bc_face::XM|bc_face::XP
TEST_CASE("bcsymmetry: edge-x-both-sides")
{
	auto block = makeBlock(6, 6);
	stamp(block, 1, 0, BC::GEO_SYMMETRY);
	stamp(block, 0, 1, BC::GEO_NOTHING);
	stamp(block, 2, 1, BC::GEO_NOTHING);
	stamp(block, 1, 1, BC::GEO_INFLOW_LEFT);

	KS ks = makeKS();
	BC::applySymmetryCorner(block.data, ks, 0, 1, 2, 0, 1, 2, 0, 0, 0);

	// bc_face::XM|bc_face::XP: applySymmetry iterates codes 0..8; m-family (codes 0..2) are
	// written first from p-family, then p-family (codes 6..8) read the already-
	// overwritten m-family slots. Net effect is a one-sided destructive copy, not a swap.
	// Original: mm=70, mz=30, mp=90, pm=80, pz=20, pp=60.
	// m-family destroyed: mm←pm=80, mz←pz=20, mp←pp=60.
	// p-family reads overwritten m-family: pm←mm=80, pz←mz=20, pp←mp=60.
	CHECK(ks.f[dir9::mm] == doctest::Approx(80.0f));
	CHECK(ks.f[dir9::mz] == doctest::Approx(20.0f));
	CHECK(ks.f[dir9::mp] == doctest::Approx(60.0f));
	CHECK(ks.f[dir9::pm] == doctest::Approx(80.0f));
	CHECK(ks.f[dir9::pz] == doctest::Approx(20.0f));
	CHECK(ks.f[dir9::pp] == doctest::Approx(60.0f));
	// non-crossing unchanged
	CHECK(ks.f[dir9::zz] == doctest::Approx(10.0f));
	CHECK(ks.f[dir9::zp] == doctest::Approx(40.0f));
	CHECK(ks.f[dir9::zm] == doctest::Approx(50.0f));
}

TEST_SUITE_END();

// D3Q27 applySymmetryCorner tests

#include "lbm3d/d3q27/bc.h"
#include "lbm3d/d3q27/col_srt.h"
#include "lbm3d/d3q27/macro.h"

#ifdef AA_PATTERN
	#include "lbm3d/d3q27/streaming_AA.h"
#endif
#ifdef AB_PATTERN
	#include "lbm3d/d3q27/streaming_AB.h"
#endif

using TRAITS3 = Traits<float, double, int>;
using COLL3 = D3Q27_SRT<TRAITS3>;
using CONFIG3 = LBM_CONFIG<
	TRAITS3,
	D3Q27_KernelStruct,
	NSE_Data<TRAITS3>,
	COLL3,
	typename COLL3::EQ,
	D3Q27_STREAMING<TRAITS3>,
	D3Q27_BC_All,
	D3Q27_MACRO_Default<TRAITS3>>;
using BLOCK3 = LBM_BLOCK<CONFIG3>;
using BC3 = typename CONFIG3::BC;
using KS3 = D3Q27_KernelStruct<typename TRAITS3::dreal>;

static BLOCK3 makeBlock3(idx nx, idx ny, idx nz)
{
	const idx3d global{nx, ny, nz};
	const idx3d loc{nx, ny, nz};
	const idx3d off{0, 0, 0};
	BLOCK3 block{MPI_COMM_WORLD, global, loc, off, 0};
	block.allocateHostData();
	block.hmap.setValue(BC3::GEO_FLUID);
	block.copyMapToDevice();
#ifdef HAVE_MPI
	block.data.indexer = block.hmap.getLocalView().getIndexer();
#else
	block.data.indexer = block.hmap.getIndexer();
#endif
	block.data.XYZ = block.data.indexer.getStorageSize();
	block.data.dmap = block.hmap.getData();
	return block;
}

static void stamp3(BLOCK3& block, idx x, idx y, idx z, typename BC3::map_t tag)
{
	block.hmap(x, y, z) = tag;
}

static KS3 makeKS3()
{
	KS3 ks;
	for (int i = 0; i < 27; i++)
		ks.f[i] = static_cast<typename TRAITS3::dreal>(i + 1) * 10.0f;
	ks.rho = 1.0;
	ks.vx = 0.0;
	ks.vy = 0.0;
	ks.vz = 0.0;
	ks.fx = 0.0;
	ks.fy = 0.0;
	ks.fz = 0.0;
	return ks;
}

TEST_SUITE_BEGIN("bcsymmetry3d");

TEST_CASE("bcsymmetry3d: no-symmetry-neighbor")
{
	auto block = makeBlock3(6, 6, 6);
	stamp3(block, 2, 2, 2, BC3::GEO_INFLOW_LEFT);

	KS3 ks = makeKS3();
	BC3::applySymmetryCorner(block.data, ks, 1, 2, 3, 1, 2, 3, 1, 2, 3);

	for (int i = 0; i < 27; i++)
		CHECK(ks.f[i] == doctest::Approx(makeKS3().f[i]));
}

TEST_CASE("bcsymmetry3d: ghost-without-symmetry")
{
	auto block = makeBlock3(6, 6, 6);
	for (idx x = 0; x < 6; x++)
		stamp3(block, x, 0, 0, BC3::GEO_NOTHING);
	stamp3(block, 1, 1, 0, BC3::GEO_INFLOW_LEFT);

	KS3 ks = makeKS3();
	BC3::applySymmetryCorner(block.data, ks, 0, 1, 2, 0, 1, 2, 0, 0, 0);

	for (int i = 0; i < 27; i++)
		CHECK(ks.f[i] == doctest::Approx(makeKS3().f[i]));
}

// symmetry at x+1, ghost at y-1 → bc_face::YM
TEST_CASE("bcsymmetry3d: single-plane-ym")
{
	auto block = makeBlock3(6, 6, 6);
	stamp3(block, 3, 2, 2, BC3::GEO_SYMMETRY);
	stamp3(block, 2, 1, 2, BC3::GEO_NOTHING);
	stamp3(block, 2, 2, 2, BC3::GEO_INFLOW_LEFT);

	KS3 ks = makeKS3();
	BC3::applySymmetryCorner(block.data, ks, 1, 2, 3, 1, 2, 3, 1, 2, 3);

	// bc_face::YM: y=p directions mirrored from y=m (same x,z)
	// mpm←mmm, mpz←mmz, mpp←mmp, zpm←zmm, zpz←zmz, zpp←zmp, ppm←pmm, ppz←pmz, ppp←pmp
	CHECK(ks.f[mpm] == doctest::Approx(makeKS3().f[mmm]));
	CHECK(ks.f[mpz] == doctest::Approx(makeKS3().f[mmz]));
	CHECK(ks.f[mpp] == doctest::Approx(makeKS3().f[mmp]));
	CHECK(ks.f[zpm] == doctest::Approx(makeKS3().f[zmm]));
	CHECK(ks.f[zpz] == doctest::Approx(makeKS3().f[zmz]));
	CHECK(ks.f[zpp] == doctest::Approx(makeKS3().f[zmp]));
	CHECK(ks.f[ppm] == doctest::Approx(makeKS3().f[pmm]));
	CHECK(ks.f[ppz] == doctest::Approx(makeKS3().f[pmz]));
	CHECK(ks.f[ppp] == doctest::Approx(makeKS3().f[pmp]));
	// y=z and y=m directions unchanged
	CHECK(ks.f[zzz] == doctest::Approx(makeKS3().f[zzz]));
	CHECK(ks.f[pzz] == doctest::Approx(makeKS3().f[pzz]));
	CHECK(ks.f[mzz] == doctest::Approx(makeKS3().f[mzz]));
	CHECK(ks.f[mmm] == doctest::Approx(makeKS3().f[mmm]));
}

// symmetry at y+1, ghost at x-1 → bc_face::XM
TEST_CASE("bcsymmetry3d: single-plane-xm")
{
	auto block = makeBlock3(6, 6, 6);
	stamp3(block, 2, 3, 2, BC3::GEO_SYMMETRY);
	stamp3(block, 1, 2, 2, BC3::GEO_NOTHING);
	stamp3(block, 2, 2, 2, BC3::GEO_INFLOW_LEFT);

	KS3 ks = makeKS3();
	BC3::applySymmetryCorner(block.data, ks, 1, 2, 3, 1, 2, 3, 1, 2, 3);

	// bc_face::XM: x=p directions mirrored from x=m (same y,z)
	// pmm←mmm, pmz←mmz, pmp←mmp, pzm←mzm, pzz←mzz, pzp←mzp, ppm←mpm, ppz←mpz, ppp←mpp
	CHECK(ks.f[pmm] == doctest::Approx(makeKS3().f[mmm]));
	CHECK(ks.f[pmz] == doctest::Approx(makeKS3().f[mmz]));
	CHECK(ks.f[pmp] == doctest::Approx(makeKS3().f[mmp]));
	CHECK(ks.f[pzm] == doctest::Approx(makeKS3().f[mzm]));
	CHECK(ks.f[pzz] == doctest::Approx(makeKS3().f[mzz]));
	CHECK(ks.f[pzp] == doctest::Approx(makeKS3().f[mzp]));
	CHECK(ks.f[ppm] == doctest::Approx(makeKS3().f[mpm]));
	CHECK(ks.f[ppz] == doctest::Approx(makeKS3().f[mpz]));
	CHECK(ks.f[ppp] == doctest::Approx(makeKS3().f[mpp]));
	// x=z and x=m directions unchanged
	CHECK(ks.f[zzz] == doctest::Approx(makeKS3().f[zzz]));
	CHECK(ks.f[zpz] == doctest::Approx(makeKS3().f[zpz]));
	CHECK(ks.f[mmm] == doctest::Approx(makeKS3().f[mmm]));
}

// symmetry at y-1, ghosts at both x-1 and x+1 → bc_face::XM|bc_face::XP (destructive copy)
TEST_CASE("bcsymmetry3d: edge-x-both-sides")
{
	auto block = makeBlock3(6, 6, 6);
	stamp3(block, 2, 1, 2, BC3::GEO_SYMMETRY);
	stamp3(block, 1, 2, 2, BC3::GEO_NOTHING);
	stamp3(block, 3, 2, 2, BC3::GEO_NOTHING);
	stamp3(block, 2, 2, 2, BC3::GEO_INFLOW_LEFT);

	KS3 ks = makeKS3();
	BC3::applySymmetryCorner(block.data, ks, 1, 2, 3, 1, 2, 3, 1, 2, 3);

	// bc_face::XM|bc_face::XP: m-family written first from p-family, then p-family reads overwritten m-family.
	// Destructive one-sided copy, not a swap.
	// m-family ← p-family (first pass, originals)
	CHECK(ks.f[mmm] == doctest::Approx(makeKS3().f[pmm]));
	CHECK(ks.f[mmz] == doctest::Approx(makeKS3().f[pmz]));
	CHECK(ks.f[mmp] == doctest::Approx(makeKS3().f[pmp]));
	CHECK(ks.f[mzm] == doctest::Approx(makeKS3().f[pzm]));
	CHECK(ks.f[mzz] == doctest::Approx(makeKS3().f[pzz]));
	CHECK(ks.f[mzp] == doctest::Approx(makeKS3().f[pzp]));
	CHECK(ks.f[mpm] == doctest::Approx(makeKS3().f[ppm]));
	CHECK(ks.f[mpz] == doctest::Approx(makeKS3().f[ppz]));
	CHECK(ks.f[mpp] == doctest::Approx(makeKS3().f[ppp]));
	// p-family ← m-family (second pass, reads overwritten values = original p-family)
	CHECK(ks.f[pmm] == doctest::Approx(makeKS3().f[pmm]));
	CHECK(ks.f[pmz] == doctest::Approx(makeKS3().f[pmz]));
	CHECK(ks.f[pmp] == doctest::Approx(makeKS3().f[pmp]));
	CHECK(ks.f[pzm] == doctest::Approx(makeKS3().f[pzm]));
	CHECK(ks.f[pzz] == doctest::Approx(makeKS3().f[pzz]));
	CHECK(ks.f[pzp] == doctest::Approx(makeKS3().f[pzp]));
	CHECK(ks.f[ppm] == doctest::Approx(makeKS3().f[ppm]));
	CHECK(ks.f[ppz] == doctest::Approx(makeKS3().f[ppz]));
	CHECK(ks.f[ppp] == doctest::Approx(makeKS3().f[ppp]));
	// x=z unchanged
	CHECK(ks.f[zzz] == doctest::Approx(makeKS3().f[zzz]));
	CHECK(ks.f[zpz] == doctest::Approx(makeKS3().f[zpz]));
}

// symmetry on both x and y axes simultaneously → bc_face::XM|bc_face::YM
TEST_CASE("bcsymmetry3d: two-axis-symmetry")
{
	auto block = makeBlock3(6, 6, 6);
	stamp3(block, 3, 2, 2, BC3::GEO_SYMMETRY);	// x+1 symmetry
	stamp3(block, 2, 3, 2, BC3::GEO_SYMMETRY);	// y+1 symmetry
	stamp3(block, 2, 1, 2, BC3::GEO_NOTHING);	// y-1 ghost
	stamp3(block, 1, 2, 2, BC3::GEO_NOTHING);	// x-1 ghost
	stamp3(block, 2, 2, 2, BC3::GEO_INFLOW_LEFT);

	KS3 ks = makeKS3();
	BC3::applySymmetryCorner(block.data, ks, 1, 2, 3, 1, 2, 3, 1, 2, 3);

	// x-sym block triggers: bc_face::YM (y-1 ghost) and bc_face::ZM/bc_face::ZP checked (neither ghost)
	// y-sym block triggers: bc_face::XM (x-1 ghost) and bc_face::ZM/bc_face::ZP checked (neither ghost)
	// ghosts = bc_face::XM | bc_face::YM
	// x=p, y=p directions mirrored from x=m, y=m (same z)
	// ppm←mmm, ppz←mmz, ppp←mmp
	CHECK(ks.f[ppm] == doctest::Approx(makeKS3().f[mmm]));
	CHECK(ks.f[ppz] == doctest::Approx(makeKS3().f[mmz]));
	CHECK(ks.f[ppp] == doctest::Approx(makeKS3().f[mmp]));
	// x=p, y=z mirrored from x=m, y=z
	CHECK(ks.f[pzm] == doctest::Approx(makeKS3().f[mzm]));
	CHECK(ks.f[pzz] == doctest::Approx(makeKS3().f[mzz]));
	CHECK(ks.f[pzp] == doctest::Approx(makeKS3().f[mzp]));
	// x=z, y=p mirrored from x=z, y=m
	CHECK(ks.f[zpm] == doctest::Approx(makeKS3().f[zmm]));
	CHECK(ks.f[zpz] == doctest::Approx(makeKS3().f[zmz]));
	CHECK(ks.f[zpp] == doctest::Approx(makeKS3().f[zmp]));
	// x=m, y=p mirrored from x=m, y=m
	CHECK(ks.f[mpm] == doctest::Approx(makeKS3().f[mmm]));
	CHECK(ks.f[mpz] == doctest::Approx(makeKS3().f[mmz]));
	CHECK(ks.f[mpp] == doctest::Approx(makeKS3().f[mmp]));
	// x=p, y=m mirrored from x=m, y=m
	CHECK(ks.f[pmm] == doctest::Approx(makeKS3().f[mmm]));
	CHECK(ks.f[pmz] == doctest::Approx(makeKS3().f[mmz]));
	CHECK(ks.f[pmp] == doctest::Approx(makeKS3().f[mmp]));
	// x=z, y=z unchanged
	CHECK(ks.f[zzz] == doctest::Approx(makeKS3().f[zzz]));
}

// symmetry at y+1, ghost at x+1 → bc_face::XP
TEST_CASE("bcsymmetry3d: single-plane-xp")
{
	auto block = makeBlock3(6, 6, 6);
	stamp3(block, 2, 3, 2, BC3::GEO_SYMMETRY);
	stamp3(block, 3, 2, 2, BC3::GEO_NOTHING);
	stamp3(block, 2, 2, 2, BC3::GEO_INFLOW_LEFT);

	KS3 ks = makeKS3();
	BC3::applySymmetryCorner(block.data, ks, 1, 2, 3, 1, 2, 3, 1, 2, 3);

	// bc_face::XP: x=m directions mirrored from x=p (same y,z)
	// mmm←pmm, mmz←pmz, mmp←pmp, mzm←pzm, mzz←pzz, mzp←pzp, mpm←ppm, mpz←ppz, mpp←ppp
	CHECK(ks.f[mmm] == doctest::Approx(makeKS3().f[pmm]));
	CHECK(ks.f[mmz] == doctest::Approx(makeKS3().f[pmz]));
	CHECK(ks.f[mmp] == doctest::Approx(makeKS3().f[pmp]));
	CHECK(ks.f[mzm] == doctest::Approx(makeKS3().f[pzm]));
	CHECK(ks.f[mzz] == doctest::Approx(makeKS3().f[pzz]));
	CHECK(ks.f[mzp] == doctest::Approx(makeKS3().f[pzp]));
	CHECK(ks.f[mpm] == doctest::Approx(makeKS3().f[ppm]));
	CHECK(ks.f[mpz] == doctest::Approx(makeKS3().f[ppz]));
	CHECK(ks.f[mpp] == doctest::Approx(makeKS3().f[ppp]));
	// x=z and x=p unchanged
	CHECK(ks.f[zzz] == doctest::Approx(makeKS3().f[zzz]));
	CHECK(ks.f[pzz] == doctest::Approx(makeKS3().f[pzz]));
}

// symmetry at x+1, ghost at y+1 → bc_face::YP
TEST_CASE("bcsymmetry3d: single-plane-yp")
{
	auto block = makeBlock3(6, 6, 6);
	stamp3(block, 3, 2, 2, BC3::GEO_SYMMETRY);
	stamp3(block, 2, 3, 2, BC3::GEO_NOTHING);
	stamp3(block, 2, 2, 2, BC3::GEO_INFLOW_LEFT);

	KS3 ks = makeKS3();
	BC3::applySymmetryCorner(block.data, ks, 1, 2, 3, 1, 2, 3, 1, 2, 3);

	// bc_face::YP: y=m directions mirrored from y=p (same x,z)
	// mmm←mpm, mmz←mpz, mmp←mpp, zmm←zpm, zmz←zpz, zmp←zpp, pmm←ppm, pmz←ppz, pmp←ppp
	CHECK(ks.f[mmm] == doctest::Approx(makeKS3().f[mpm]));
	CHECK(ks.f[mmz] == doctest::Approx(makeKS3().f[mpz]));
	CHECK(ks.f[mmp] == doctest::Approx(makeKS3().f[mpp]));
	CHECK(ks.f[zmm] == doctest::Approx(makeKS3().f[zpm]));
	CHECK(ks.f[zmz] == doctest::Approx(makeKS3().f[zpz]));
	CHECK(ks.f[zmp] == doctest::Approx(makeKS3().f[zpp]));
	CHECK(ks.f[pmm] == doctest::Approx(makeKS3().f[ppm]));
	CHECK(ks.f[pmz] == doctest::Approx(makeKS3().f[ppz]));
	CHECK(ks.f[pmp] == doctest::Approx(makeKS3().f[ppp]));
	// y=z and y=p unchanged
	CHECK(ks.f[zzz] == doctest::Approx(makeKS3().f[zzz]));
	CHECK(ks.f[zpz] == doctest::Approx(makeKS3().f[zpz]));
}

// symmetry at x+1, ghost at z-1 → bc_face::ZM
TEST_CASE("bcsymmetry3d: single-plane-zm")
{
	auto block = makeBlock3(6, 6, 6);
	stamp3(block, 3, 2, 2, BC3::GEO_SYMMETRY);
	stamp3(block, 2, 2, 1, BC3::GEO_NOTHING);
	stamp3(block, 2, 2, 2, BC3::GEO_INFLOW_LEFT);

	KS3 ks = makeKS3();
	BC3::applySymmetryCorner(block.data, ks, 1, 2, 3, 1, 2, 3, 1, 2, 3);

	// bc_face::ZM: z=p directions mirrored from z=m (same x,y)
	// mmp←mmm, mzp←mzm, mpp←mpm, zmp←zmm, zzp←zzm, zpp←zpm, pmp←pmm, pzp←pzm, ppp←ppm
	CHECK(ks.f[mmp] == doctest::Approx(makeKS3().f[mmm]));
	CHECK(ks.f[mzp] == doctest::Approx(makeKS3().f[mzm]));
	CHECK(ks.f[mpp] == doctest::Approx(makeKS3().f[mpm]));
	CHECK(ks.f[zmp] == doctest::Approx(makeKS3().f[zmm]));
	CHECK(ks.f[zzp] == doctest::Approx(makeKS3().f[zzm]));
	CHECK(ks.f[zpp] == doctest::Approx(makeKS3().f[zpm]));
	CHECK(ks.f[pmp] == doctest::Approx(makeKS3().f[pmm]));
	CHECK(ks.f[pzp] == doctest::Approx(makeKS3().f[pzm]));
	CHECK(ks.f[ppp] == doctest::Approx(makeKS3().f[ppm]));
	// z=z and z=m unchanged
	CHECK(ks.f[zzz] == doctest::Approx(makeKS3().f[zzz]));
	CHECK(ks.f[zzm] == doctest::Approx(makeKS3().f[zzm]));
}

// symmetry at x+1, ghost at z+1 → bc_face::ZP
TEST_CASE("bcsymmetry3d: single-plane-zp")
{
	auto block = makeBlock3(6, 6, 6);
	stamp3(block, 3, 2, 2, BC3::GEO_SYMMETRY);
	stamp3(block, 2, 2, 3, BC3::GEO_NOTHING);
	stamp3(block, 2, 2, 2, BC3::GEO_INFLOW_LEFT);

	KS3 ks = makeKS3();
	BC3::applySymmetryCorner(block.data, ks, 1, 2, 3, 1, 2, 3, 1, 2, 3);

	// bc_face::ZP: z=m directions mirrored from z=p (same x,y)
	// mmm←mmp, mzm←mzp, mpm←mpp, zmm←zmp, zzm←zzp, zpm←zpp, pmm←pmp, pzm←pzp, ppm←ppp
	CHECK(ks.f[mmm] == doctest::Approx(makeKS3().f[mmp]));
	CHECK(ks.f[mzm] == doctest::Approx(makeKS3().f[mzp]));
	CHECK(ks.f[mpm] == doctest::Approx(makeKS3().f[mpp]));
	CHECK(ks.f[zmm] == doctest::Approx(makeKS3().f[zmp]));
	CHECK(ks.f[zzm] == doctest::Approx(makeKS3().f[zzp]));
	CHECK(ks.f[zpm] == doctest::Approx(makeKS3().f[zpp]));
	CHECK(ks.f[pmm] == doctest::Approx(makeKS3().f[pmp]));
	CHECK(ks.f[pzm] == doctest::Approx(makeKS3().f[pzp]));
	CHECK(ks.f[ppm] == doctest::Approx(makeKS3().f[ppp]));
	// z=z and z=p unchanged
	CHECK(ks.f[zzz] == doctest::Approx(makeKS3().f[zzz]));
	CHECK(ks.f[zzp] == doctest::Approx(makeKS3().f[zzp]));
}

TEST_SUITE_END();
