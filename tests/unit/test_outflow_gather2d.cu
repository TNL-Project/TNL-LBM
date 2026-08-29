/*
 * Unit tests for the any-face outflow-pass gathers (GEO_OUTFLOW_RIGHT /
 * GEO_OUTFLOW_RIGHT_INTERP), D2Q9.
 *
 * For every face (XP, XM, YP, YM) the dispatchers streamingOutflow /
 * streamingOutflowInterp run on the device against a synthetic df field
 * holding an injective integer pattern per (slot, x, y, z). The expected
 * KS.f values come from an independent ground truth: direction components
 * are parsed from the direction NAMES (first letter = x, second = y;
 * p = +1, z = 0, m = -1) and the opposite slot is derived by flipping
 * p<->m in the name - so a transcription bug in the production tables or
 * in the gather sites cannot hide behind a circular reference. The blend
 * arithmetic is expected through the same pinned lbm_fma_rn form used by
 * production, so the tests pin the SITES, not the rounding.
 *
 * Under the A-A pattern the gathers are parity-dependent and every case
 * runs for both parities; under A-B both subcases exercise the same
 * parity-free path. Face detection (BC::detectOutflowFace, host-side) is
 * covered for all faces including the symmetry-as-interior rule.
 */

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include <doctest/doctest.h>


#include "lbm3d/lbm_data.h"
#include "lbm3d/d2q9/bc.h"
#include "lbm3d/d2q9/col_srt.h"
#include "lbm3d/d2q9/macro.h"
#include "lbm_common/rounding.h"

// exactly one streaming header must be included
#ifdef AA_PATTERN
	#include "lbm3d/d2q9/streaming_AA.h"
#endif
#ifdef AB_PATTERN
	#include "lbm3d/d2q9/streaming_AB.h"
#endif

#include "lbm3d/lbm_block.h"

using TRAITS = Traits<double>;  // dreal = double: exact comparisons against the double ground truth
using COLL = D2Q9_SRT<TRAITS>;
using CONFIG =
	LBM_CONFIG<TRAITS, D2Q9_KernelStruct, NSE_Data<TRAITS>, COLL, typename COLL::EQ, D2Q9_STREAMING<TRAITS>, D2Q9_BC_All, D2Q9_MACRO_Default<TRAITS>>;
using STREAM = D2Q9_STREAMING<TRAITS>;
using BC = typename CONFIG::BC;
using KS = D2Q9_KernelStruct<typename TRAITS::dreal>;
using idx = typename TRAITS::idx;
using idx3d = typename TRAITS::idx3d;

// direction names in dir9 enum order (must match defs.h); the ground-truth
// components are parsed from these, independently of the production tables
static constexpr const char* dir9_names[9] = { "zz", "pz", "mz", "zp", "zm", "pp", "mm", "pm", "mp" };

static int
nameVal(char c)
{
	return c == 'p' ? 1 : (c == 'm' ? -1 : 0);
}

static int
comp9(int slot, int dim)
{
	return nameVal(dir9_names[slot][dim]);
}

// opposite direction derived from the name (flip p<->m, keep z), independent
// of the production opposite_direction arithmetic
static int
opp9(int slot)
{
	std::string name = dir9_names[slot];
	for (char& c : name)
		if (c == 'p')
			c = 'm';
		else if (c == 'm')
			c = 'p';
	for (int i = 0; i < 9; i++)
		if (name == dir9_names[i])
			return i;
	return -1;
}

// injective pattern over (slot, x, y, z); exact in double for the ranges used
static constexpr int MS = 32;  // mock coordinate space per dimension
static double
pat(int slot, int x, int y, int z)
{
	return 1.0 + 1e6 * slot + 1e4 * x + 1e2 * y + z;
}

// same pinned constant as the production blend helper
static constexpr double SpeedOfSound = 0.5773502691896257;

// device-side df storage mock (POD, passed by value into the kernel)
struct GatherMock
{
	double* mem;
	bool even_iter;

	__cuda_callable__ double&
	df(int, int slot, int x, int y, int z) const
	{
		return mem[(slot * MS + x) * MS * MS + y * MS + z];
	}
};

// device driver: run one outflow gather and copy the kernel struct out
__global__ void
gatherKernel(GatherMock sd, KS* out, int face, bool interp, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx z)
{
	KS ks;
	for (int i = 0; i < 9; i++)
		ks.f[i] = std::numeric_limits<double>::quiet_NaN();
	if (interp)
		STREAM::streamingOutflowInterp(sd, ks, face, xm, x, xp, ym, y, yp, z);
	else
		STREAM::streamingOutflow(sd, ks, face, xm, x, xp, ym, y, yp, z);
	*out = ks;
}

// expected blend values must be computed with the same pinned lbm_fma_rn
// arithmetic as production: host std::fma and device __fma_rn can differ in
// the last bit for the same expression, so blends are resolved on the device
// (in[i] = {A, B} pairs, out[i] = lbm_fma_rn(SpeedOfSound, A, (1-SpeedOfSound)*B))
__global__ void
blendKernel2d(const double* in, double* out, int n)
{
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n)
		out[i] = lbm_fma_rn(SpeedOfSound, in[2 * i], (1 - SpeedOfSound) * in[2 * i + 1]);
}

static void
resolveBlends(double* exp, const double* blendA, const double* blendB, const char* isBlend, int n)
{
	std::vector<double> in;
	std::vector<int> slots;
	for (int i = 0; i < n; i++)
		if (isBlend[i]) {
			in.push_back(blendA[i]);
			in.push_back(blendB[i]);
			slots.push_back(i);
		}
	if (slots.empty())
		return;
	TNL::Containers::Array<double, TNL::Devices::Host> hostIn(in.size());
	for (size_t j = 0; j < in.size(); j++)
		hostIn[j] = in[j];
	TNL::Containers::Array<double, TNL::Devices::Cuda> devIn;
	devIn = hostIn;
	TNL::Containers::Array<double, TNL::Devices::Cuda> devOut(slots.size());
	blendKernel2d<<<1, 9>>>(devIn.getData(), devOut.getData(), (int) slots.size());
	TNL::Backend::deviceSynchronize();
	std::vector<double> res(slots.size());
	TNL::Backend::memcpy(res.data(), devOut.getData(), res.size() * sizeof(double), TNL::Backend::MemcpyDeviceToHost);
	for (size_t j = 0; j < slots.size(); j++)
		exp[slots[j]] = res[j];
}

static void
runGather(int face, bool interp, bool even, int x, int y, int z, KS& out)
{
	std::vector<double> host((size_t) 9 * MS * MS * MS);
	for (int s = 0; s < 9; s++)
		for (int xx = 0; xx < MS; xx++)
			for (int yy = 0; yy < MS; yy++)
				for (int zz = 0; zz < MS; zz++)
					host[((size_t) s * MS + xx) * MS * MS + yy * MS + zz] = pat(s, xx, yy, zz);

	TNL::Containers::Array<double, TNL::Devices::Host> hostArr(host.size());
	for (size_t i = 0; i < host.size(); i++)
		hostArr[i] = host[i];
	TNL::Containers::Array<double, TNL::Devices::Cuda> dev;
	dev = hostArr;
	TNL::Containers::Array<KS, TNL::Devices::Cuda> devOut(1);

	GatherMock sd{dev.getData(), even};
	gatherKernel<<<1, 1>>>(sd, devOut.getData(), face, interp, x - 1, x, x + 1, y - 1, y, y + 1, z);
	TNL::Backend::deviceSynchronize();
	TNL::Backend::memcpy(&out, devOut.getData(), sizeof(KS), TNL::Backend::MemcpyDeviceToHost);
}

// independent ground truth for all 9 slots of one gather
static void
computeExpected(int face, bool interp, bool even, int x, int y, int z, double* exp, double* blendA, double* blendB, char* isBlend)
{
	const int axis = (face & (bc_face::XP | bc_face::XM)) ? 0 : 1;
	const int sgn = (face & (bc_face::XM | bc_face::YM)) ? -1 : 1;
	const int co[2] = { x, y };
	const int anchor = co[axis] - sgn;  // fluid-side neighbor, one cell inward
	for (int i = 0; i < 9; i++) {
		const int c[2] = { comp9(i, 0), comp9(i, 1) };
		const int tang = co[1 - axis] - c[1 - axis];  // tangential -c offset (pull scheme)
		const int cn = c[axis];
		if (! interp) {
#ifdef AB_PATTERN
			int s[2];
			s[axis] = anchor;
			s[1 - axis] = tang;
			exp[i] = pat(i, s[0], s[1], z);
#else
			if (even) {
				// natural layout: slot (i, t + c_i) = postcoll_{n-1}(i, t); the
				// tangential offsets cancel, the normal coordinate is anchor + c_i
				int s[2];
				s[axis] = anchor + c[axis];
				s[1 - axis] = co[1 - axis];
				exp[i] = pat(i, s[0], s[1], z);
			}
			else {
				// twist layout: slot (opp(i), t) = postcoll_{n-1}(i, t)
				int s[2];
				s[axis] = anchor;
				s[1 - axis] = tang;
				exp[i] = pat(opp9(i), s[0], s[1], z);
			}
#endif
		}
		else {
#ifdef AB_PATTERN
			// outward population: anchor column; perpendicular: own column;
			// inward: anchor-column postcoll blended with the own-column postcoll
			if (cn == sgn) {
				int s[2];
				s[axis] = anchor;
				s[1 - axis] = tang;
				exp[i] = pat(i, s[0], s[1], z);
			}
			else if (cn == 0) {
				int s[2];
				s[axis] = co[axis];
				s[1 - axis] = tang;
				exp[i] = pat(i, s[0], s[1], z);
			}
			else {
				int sn[2];
				sn[axis] = anchor;
				sn[1 - axis] = tang;
				int so[2];
				so[axis] = co[axis];
				so[1 - axis] = tang;
				isBlend[i] = 1;
				blendA[i] = pat(i, sn[0], sn[1], z);
				blendB[i] = pat(i, so[0], so[1], z);
			}
#else  // AA
			if (even) {
				// outward- and perpendicular-moving populations take the cell's
				// own postcoll; the inward-moving population blends the
				// pre-anchor column with the anchor column
				if (cn == sgn || cn == 0)
					exp[i] = pat(i, x, y, z);
				else {
					int sa[2];
					sa[axis] = anchor + c[axis];
					sa[1 - axis] = co[1 - axis];
					int sb[2];
					sb[axis] = anchor;
					sb[1 - axis] = co[1 - axis];
					isBlend[i] = 1;
					blendA[i] = pat(i, sa[0], sa[1], z);
					blendB[i] = pat(i, sb[0], sb[1], z);
				}
			}
			else {
				// twist layout: outward from the anchor column, perpendicular
				// from the own column, inward blends the two
				const int slot = opp9(i);
				if (cn == sgn) {
					int s[2];
					s[axis] = anchor;
					s[1 - axis] = tang;
					exp[i] = pat(slot, s[0], s[1], z);
				}
				else if (cn == 0) {
					int s[2];
					s[axis] = co[axis];
					s[1 - axis] = tang;
					exp[i] = pat(slot, s[0], s[1], z);
				}
				else {
					int sn[2];
					sn[axis] = anchor;
					sn[1 - axis] = tang;
					int so[2];
					so[axis] = co[axis];
					so[1 - axis] = tang;
					isBlend[i] = 1;
					blendA[i] = pat(slot, sn[0], sn[1], z);
					blendB[i] = pat(slot, so[0], so[1], z);
				}
			}
#endif
		}
	}
}

TEST_SUITE_BEGIN("outflowgather2d");

TEST_CASE("gather-plain-faces")
{
	const int x = 8, y = 9, z = 3;
	const int faces[4] = { bc_face::XP, bc_face::XM, bc_face::YP, bc_face::YM };
	for (int face : faces) {
		INFO("face=", face);
		for (bool even : { true, false }) {
			INFO("even=", even);
			KS ks;
			runGather(face, /*interp=*/false, even, x, y, z, ks);
			double exp[9];
			double blendA[9], blendB[9];
			char isBlend[9] = {};
			computeExpected(face, /*interp=*/false, even, x, y, z, exp, blendA, blendB, isBlend);
			resolveBlends(exp, blendA, blendB, isBlend, 9);
			for (int i = 0; i < 9; i++)
				CHECK_EQ(ks.f[i], exp[i]);
		}
	}
}

TEST_CASE("gather-interp-faces")
{
	const int x = 8, y = 9, z = 3;
	const int faces[4] = { bc_face::XP, bc_face::XM, bc_face::YP, bc_face::YM };
	for (int face : faces) {
		INFO("face=", face);
		for (bool even : { true, false }) {
			INFO("even=", even);
			KS ks;
			runGather(face, /*interp=*/true, even, x, y, z, ks);
			double exp[9];
			double blendA[9], blendB[9];
			char isBlend[9] = {};
			computeExpected(face, /*interp=*/true, even, x, y, z, exp, blendA, blendB, isBlend);
			resolveBlends(exp, blendA, blendB, isBlend, 9);
			for (int i = 0; i < 9; i++)
				CHECK_EQ(ks.f[i], exp[i]);
		}
	}
}

// host-side face detection through the real DATA type: the block's host map
// is wired into data.dmap (the test_bc_symmetry pattern), so SD.map reads the
// stamped host map; exactly one interior axis-neighbor selects the face and
// symmetry cells count as interior
TEST_CASE("detect-faces")
{
	using BLOCK = LBM_BLOCK<CONFIG>;
	const idx3d global{16, 16, 1};
	BLOCK block{MPI_COMM_WORLD, global, global, idx3d{0, 0, 0}, 0};
	block.allocateHostData();
	block.hmap.setValue(BC::GEO_FLUID);
	block.copyMapToDevice();
#ifdef HAVE_MPI
	block.data.indexer = block.hmap.getLocalView().getIndexer();
#else
	block.data.indexer = block.hmap.getIndexer();
#endif
	block.data.XYZ = block.data.indexer.getStorageSize();
	block.data.dmap = block.hmap.getData();

	const int x = 8, y = 9, z = 0;
	struct Case
	{
		int face;
		int ax, ay;
	};
	const Case cases[4] = {
		{ bc_face::XP, x - 1, y     },
		{ bc_face::XM, x + 1, y     },
		{ bc_face::YP, x,     y - 1 },
		{ bc_face::YM, x,     y + 1 },
	};
	const typename BC::map_t interiorTags[2] = { BC::GEO_FLUID, BC::GEO_SYMMETRY };
	for (const auto& c : cases) {
		INFO("face=", c.face);
		for (auto tag : interiorTags) {
			// walls everywhere, outflow tag on the cell, interior tag on the anchor
			block.hmap.setValue(BC::GEO_WALL);
			block.hmap(x, y, z) = BC::GEO_OUTFLOW_RIGHT_INTERP;
			block.hmap(c.ax, c.ay, z) = tag;
			const int detected = BC::detectOutflowFace(block.data, x - 1, x, x + 1, y - 1, y, y + 1, z);
			CHECK_EQ(detected, c.face);
		}
	}
}

TEST_SUITE_END();
