#pragma once

#include "lbm3d/defs.h"
#include "lbm_common/ciselnik.h"
#include "lbm_common/rounding.h"

template <typename CONFIG>
struct D3Q27_BC_All
{
	using COLL = typename CONFIG::COLL;
	using STREAMING = typename CONFIG::STREAMING;
	using DATA = typename CONFIG::DATA;

	using map_t = typename CONFIG::TRAITS::map_t;
	using idx = typename CONFIG::TRAITS::idx;
	using dreal = typename CONFIG::TRAITS::dreal;

	enum GEO : map_t
	{
		GEO_FLUID,	// compulsory
		GEO_WALL,	// compulsory
		GEO_INFLOW,
		GEO_INFLOW_MOMENT,
		GEO_INFLOW_BOUNCEBACK,
		GEO_INFLOW_EQ_LEFT,
		GEO_OUTFLOW_EQ,
		GEO_OUTFLOW_RIGHT,
		GEO_OUTFLOW_RIGHT_INTERP,
		GEO_NOTHING,
		GEO_SYMMETRY,

		// Adjoint boundary conditions
		GEO_ADJOINT_FLUID,
		GEO_ADJOINT_FLUID_m,
		GEO_ADJOINT_WALL,
		GEO_ADJOINT_INFLOW_BB_LEFT,
		GEO_ADJOINT_OUTFLOW_RIGHT
	};

	__cuda_callable__ static bool isSymmetric(map_t mapgi)
	{
		return mapgi == GEO_SYMMETRY;
	}

	__cuda_callable__ static bool isFluid(map_t mapgi)
	{
		return mapgi == GEO_FLUID;
	}

	__cuda_callable__ static bool isWall(map_t mapgi)
	{
		return mapgi == GEO_WALL;
	}

	// deterministic two-pass outflow: outflow cells are skipped in the main
	// kernel; State::SimUpdate launches cudaLBMKernelOutflow right before it,
	// which applies outflowPass on state finalized by the previous launch
	static constexpr bool use_outflow_pass = true;

	__cuda_callable__ static bool isOutflowPassBC(map_t mapgi)
	{
		return mapgi == GEO_OUTFLOW_RIGHT || mapgi == GEO_OUTFLOW_RIGHT_INTERP;
	}

	// interior side of an outflow plane: fluid, or symmetry (which acts as a
	// fluid cell - outflow planes may cover the full face including
	// symmetry-row corners)
	__cuda_callable__ static bool isOutflowInterior(map_t mapgi)
	{
		return mapgi == GEO_FLUID || mapgi == GEO_SYMMETRY;
	}

	// Outflow face detection from the map: the interior side of the outlet is
	// the axis-neighbor acting as fluid (GEO_FLUID, or GEO_SYMMETRY which acts
	// as a fluid cell - outflow planes may cover the full face including
	// symmetry-row corners), the outward normal points away from it.
	// LBM_BLOCK::validateOutflowPassRegion (called from copyMapToDevice)
	// guarantees exactly one such axis-neighbor per outflow cell, so the
	// check order only matters for unvalidated maps.
	__cuda_callable__ static int detectBCFace(DATA& SD, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm, idx z, idx zp)
	{
		if (isOutflowInterior(SD.map(xm, y, z)))
			return bc_face::XP;
		if (isOutflowInterior(SD.map(xp, y, z)))
			return bc_face::XM;
		if (isOutflowInterior(SD.map(x, ym, z)))
			return bc_face::YP;
		if (isOutflowInterior(SD.map(x, yp, z)))
			return bc_face::YM;
		if (isOutflowInterior(SD.map(x, y, zm)))
			return bc_face::ZP;
		return bc_face::ZM;
	}

	// BC tags whose runtime face detection must find exactly one interior
	// axis-neighbor: the outflow pass and the moment inflow
	__cuda_callable__ static bool isFaceDetectedBC(map_t mapgi)
	{
		return isOutflowPassBC(mapgi) || mapgi == GEO_INFLOW_MOMENT;
	}

	// gathers read the postcollision state finalized by the previous launch
	// and live in streaming_*.h; the BC body then follows the legacy cases
	template <typename LBM_KS>
	__cuda_callable__ static void outflowPass(DATA& SD, LBM_KS& KS, map_t mapgi, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm, idx z, idx zp)
	{
		const int face = detectBCFace(SD, xm, x, xp, ym, y, yp, zm, z, zp);
		switch (mapgi) {
			case GEO_OUTFLOW_RIGHT:
				STREAMING::streamingOutflow(SD, KS, face, xm, x, xp, ym, y, yp, zm, z, zp);
				applySymmetryCorner(SD, KS, xm, x, xp, ym, y, yp, zm, z, zp);
				COLL::computeDensityAndVelocity(KS);
				KS.rho = 1;
				COLL::collision(KS);
				STREAMING::postCollisionStreaming(SD, KS, xm, x, xp, ym, y, yp, zm, z, zp);
				break;
			case GEO_OUTFLOW_RIGHT_INTERP:
				STREAMING::streamingOutflowInterp(SD, KS, face, xm, x, xp, ym, y, yp, zm, z, zp);
				applySymmetryCorner(SD, KS, xm, x, xp, ym, y, yp, zm, z, zp);
				COLL::computeDensityAndVelocity(KS);
				COLL::setEquilibriumDecomposition(KS, 1);
				KS.rho = 1;
				COLL::collision(KS);
				STREAMING::postCollisionStreaming(SD, KS, xm, x, xp, ym, y, yp, zm, z, zp);
				break;
		}
	}

	// direction slot for the letter trits (ex, ey, ez) in {m=0, z=1, p=2} packed as 9*ex + 3*ey + ez
	__cuda_callable__ static constexpr std::uint8_t sym_dir_slot(int code)
	{
		// clang-format off
		switch (code) {
			case 0:  return mmm;
			case 1:  return mmz;
			case 2:  return mmp;
			case 3:  return mzm;
			case 4:  return mzz;
			case 5:  return mzp;
			case 6:  return mpm;
			case 7:  return mpz;
			case 8:  return mpp;
			case 9:  return zmm;
			case 10: return zmz;
			case 11: return zmp;
			case 12: return zzm;
			case 13: return zzz;
			case 14: return zzp;
			case 15: return zpm;
			case 16: return zpz;
			case 17: return zpp;
			case 18: return pmm;
			case 19: return pmz;
			case 20: return pmp;
			case 21: return pzm;
			case 22: return pzz;
			case 23: return pzp;
			case 24: return ppm;
			case 25: return ppz;
			case 26: return ppp;
			default: return zzz;
		}
		// clang-format on
	}

	// Closure for GEO_SYMMETRY cells adjacent to two or three GEO_NOTHING ghost half-spaces
	// (edges and corners of the domain frame).
	// A population is unknown exactly when one of its non-zero components points through a ghost side
	// ('p**' through the x-1 ghost, 'm**' through x+1, '*p*' through y-1, '*m*' through y+1,
	// '**p' through z-1, and '**m' through z+1);
	// each unknown gets the value of the within-cell population with every ghost-crossing component flipped
	// (single reflection per orthogonal directions, double/triple for diagonals).
	// The fills are pure copies whose sources never cross a ghost side,
	// so the result does not depend on where in the domain frame the GEO_SYMMETRY planes lie.
	template <typename LBM_KS>
	__cuda_callable__ static void applySymmetry(LBM_KS& KS, std::uint8_t ghosts)
	{
		for (int code = 0; code < 27; code++) {
			const int ex = code / 9;
			const int ey = code / 3 % 3;
			const int ez = code % 3;
			// cx/cy/cz are set when the direction's component crosses a ghost side
			// ('p' crosses the minus side, 'm' crosses the plus side, 'z' crosses neither)
			const bool cx = ex != 1 && (ghosts & (ex == 2 ? bc_face::XM : bc_face::XP));
			const bool cy = ey != 1 && (ghosts & (ey == 2 ? bc_face::YM : bc_face::YP));
			const bool cz = ez != 1 && (ghosts & (ez == 2 ? bc_face::ZM : bc_face::ZP));
			if (cx || cy || cz) {
				const int src = 9 * (cx ? 2 - ex : ex) + 3 * (cy ? 2 - ey : ey) + (cz ? 2 - ez : ez);
				KS.f[sym_dir_slot(code)] = KS.f[sym_dir_slot(src)];
			}
		}
	}

	// Mirror-precondition for BC cells adjacent to GEO_SYMMETRY planes.
	// Checks the BC cell's own 6 neighbors: if a neighbor is GEO_SYMMETRY,
	// the remaining perpendicular neighbors are checked for GEO_NOTHING to
	// find the mirror half-planes. Symmetry may be on multiple axes;
	// each axis is checked independently.
	// No symmetry neighbor → no closure. The direction from the BC cell to the
	// symmetry cell is never mirrored (the BC handles those populations).
	template <typename LBM_KS>
	__cuda_callable__ static void applySymmetryCorner(DATA& SD, LBM_KS& KS, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm, idx z, idx zp)
	{
		std::uint8_t ghosts = 0;
		if (SD.map(xm, y, z) == GEO_SYMMETRY || SD.map(xp, y, z) == GEO_SYMMETRY) {
			if (SD.map(x, ym, z) == GEO_NOTHING)
				ghosts |= bc_face::YM;
			if (SD.map(x, yp, z) == GEO_NOTHING)
				ghosts |= bc_face::YP;
			if (SD.map(x, y, zm) == GEO_NOTHING)
				ghosts |= bc_face::ZM;
			if (SD.map(x, y, zp) == GEO_NOTHING)
				ghosts |= bc_face::ZP;
		}
		if (SD.map(x, ym, z) == GEO_SYMMETRY || SD.map(x, yp, z) == GEO_SYMMETRY) {
			if (SD.map(xm, y, z) == GEO_NOTHING)
				ghosts |= bc_face::XM;
			if (SD.map(xp, y, z) == GEO_NOTHING)
				ghosts |= bc_face::XP;
			if (SD.map(x, y, zm) == GEO_NOTHING)
				ghosts |= bc_face::ZM;
			if (SD.map(x, y, zp) == GEO_NOTHING)
				ghosts |= bc_face::ZP;
		}
		if (SD.map(x, y, zm) == GEO_SYMMETRY || SD.map(x, y, zp) == GEO_SYMMETRY) {
			if (SD.map(xm, y, z) == GEO_NOTHING)
				ghosts |= bc_face::XM;
			if (SD.map(xp, y, z) == GEO_NOTHING)
				ghosts |= bc_face::XP;
			if (SD.map(x, ym, z) == GEO_NOTHING)
				ghosts |= bc_face::YM;
			if (SD.map(x, yp, z) == GEO_NOTHING)
				ghosts |= bc_face::YP;
		}
		if (ghosts)
			applySymmetry(KS, ghosts);
	}

	// component of direction slot i along axis a (0 = x, 1 = y, 2 = z)
	__cuda_callable__ static constexpr int dcomp(int i, int a)
	{
		return a == 0 ? dir27_cx(i) : a == 1 ? dir27_cy(i) : dir27_cz(i);
	}

	// slot of the direction with the given (normal, t1, t2) components
	// (-1 if not found; the tangential axes t1/t2 are chosen per face)
	__cuda_callable__ static constexpr int dslot(int a, int t1, int t2, int cn, int ct1, int ct2)
	{
		for (int i = 0; i < 27; i++)
			if (dcomp(i, a) == cn && dcomp(i, t1) == ct1 && dcomp(i, t2) == ct2)
				return i;
		return -1;
	}

	// velocity component of the kernel data along axis a (0 = x, 1 = y, 2 = z)
	template <typename LBM_KS>
	__cuda_callable__ static dreal vcomp(const LBM_KS& KS, int a)
	{
		return a == 0 ? KS.vx : a == 1 ? KS.vy : KS.vz;
	}

	// moment boundary condition by Pavel Eichler https://doi.org/10.1016/j.camwa.2024.08.009,
	// generalized from the legacy left-wall inflow (outward normal -x, face XM) to any
	// domain face (tangential axes are cyclic: t1 = axis+1, t2 = axis+2); the single
	// runtime-parameterized body below serves all faces -- the derivation in
	// docs/moment-bc-derivation.md pins its rounding to the pre-generalization XM tree.
	// Unlike D2Q9 (per-face template instantiations, see d2q9/bc.h), this model keeps
	// the runtime face: measured in the fused kernels it is the fastest D3Q27
	// dispatch (sim_2 AB +1.7%, AA neutral vs committed), while per-face
	// instantiation inlines six bodies into the AA kernel and reproduces the ptxas
	// spill regression (sim_2 AA -6.9%)
	template <typename LBM_KS>
	__cuda_callable__ static void inflowMoment(int face, LBM_KS& KS)
	{
		const int AXIS = (face == bc_face::XP || face == bc_face::XM) ? 0 : (face == bc_face::YP || face == bc_face::YM) ? 1 : 2;
		const int SIGN = (face == bc_face::XM || face == bc_face::YM || face == bc_face::ZM) ? -1 : 1;
		const int T1 = (AXIS + 1) % 3;
		const int T2 = (AXIS + 2) % 3;

		const dreal vn = vcomp(KS, AXIS);
		const dreal vt1 = vcomp(KS, T1);
		const dreal vt2 = vcomp(KS, T2);

		// layer slots: Z = populations with cn == 0, W = the outward-moving layer
		// (cn == SIGN); each layer is one axis slot (ct1 == ct2 == 0) plus 8
		// off-axis slots summed as balanced mirror pairs -- corners
		// ((+1,+1)+(-1,-1)) + ((+1,-1)+(-1,+1)) first, then the tangential axis
		// pairs ((+1,0)+(-1,0)) + ((0,+1)+(0,-1)), positive direction first; on
		// the legacy face (XM) this reproduces the verbatim XM expression tree
		// bit-exactly (pair-internal operand order is bitwise-invisible to the
		// commutative IEEE add)
		const int z00 = dslot(AXIS, T1, T2, 0, 0, 0);
		const int zpp = dslot(AXIS, T1, T2, 0, 1, 1);
		const int zmm = dslot(AXIS, T1, T2, 0, -1, -1);
		const int zpm = dslot(AXIS, T1, T2, 0, 1, -1);
		const int zmp = dslot(AXIS, T1, T2, 0, -1, 1);
		const int zpz = dslot(AXIS, T1, T2, 0, 1, 0);
		const int zmz = dslot(AXIS, T1, T2, 0, -1, 0);
		const int zzp = dslot(AXIS, T1, T2, 0, 0, 1);
		const int zzm = dslot(AXIS, T1, T2, 0, 0, -1);
		const int w00 = dslot(AXIS, T1, T2, SIGN, 0, 0);
		const int wpp = dslot(AXIS, T1, T2, SIGN, 1, 1);
		const int wmm = dslot(AXIS, T1, T2, SIGN, -1, -1);
		const int wpm = dslot(AXIS, T1, T2, SIGN, 1, -1);
		const int wmp = dslot(AXIS, T1, T2, SIGN, -1, 1);
		const int wpz = dslot(AXIS, T1, T2, SIGN, 1, 0);
		const int wmz = dslot(AXIS, T1, T2, SIGN, -1, 0);
		const int wzp = dslot(AXIS, T1, T2, SIGN, 0, 1);
		const int wzm = dslot(AXIS, T1, T2, SIGN, 0, -1);

		const dreal zCorners = ((KS.f[zpp] + KS.f[zmm]) + (KS.f[zpm] + KS.f[zmp]));
		const dreal wCorners = ((KS.f[wpp] + KS.f[wmm]) + (KS.f[wpm] + KS.f[wmp]));
		const dreal zSum = zCorners + ((KS.f[zpz] + KS.f[zmz]) + (KS.f[zzp] + KS.f[zzm]));
		const dreal wSum = wCorners + ((KS.f[wpz] + KS.f[wmz]) + (KS.f[wzp] + KS.f[wzm]));

		// reciprocal first, then multiply -- matches the legacy XM denominator
		KS.rho = (dreal) 1.0 / (1 + SIGN * vn) * ((KS.f[z00] + zSum) + 2 * (KS.f[w00] + wSum));

		// lbm_fma_rn pins replicate the fp-contraction spots the compiler picks
		// for the legacy XM body (verified in SASS): a runtime-parameterized
		// formula cannot reproduce those contractions reliably from source, and
		// the intrinsic never re-contracts, so the pinned subexpressions keep
		// the legacy rounding on XM deterministically
		const dreal mT1 = KS.rho * vt1;
		const dreal mT2 = KS.rho * vt2;
		const dreal mT1T1 = n1o3 * KS.rho + KS.rho * (vt1 * vt1);
		const dreal mT2T2 = lbm_fma_rn(KS.rho, vt2 * vt2, n1o3 * KS.rho);
		const dreal mT1T1T2 = lbm_fma_rn(vt2, n1o3 * KS.rho, KS.rho * ((vt1 * vt1) * vt2));
		const dreal mT1T2T2 = lbm_fma_rn(vt1, n1o3 * KS.rho, KS.rho * (vt1 * (vt2 * vt2)));
		const dreal mTT = lbm_fma_rn(KS.rho * (vt1 * vt1), vt2 * vt2, lbm_fma_rn(KS.rho, n1o9, n1o3 * KS.rho * (vt1 * vt1 + vt2 * vt2)));

		// closed-form reconstruction of the unknown layer (populations moving
		// into the domain, cn == -SIGN); partner slots always live in the
		// untouched W / Z layers, so the write order does not matter
		for (int i = 0; i < 27; i++) {
			if (dcomp(i, AXIS) != -SIGN)
				continue;
			const int ct1 = dcomp(i, T1);
			const int ct2 = dcomp(i, T2);
			const int w = dslot(AXIS, T1, T2, SIGN, ct1, ct2);
			const int z = dslot(AXIS, T1, T2, 0, ct1, ct2);
			if (ct1 == 0 && ct2 == 0)
				KS.f[i] = lbm_fma_rn((dreal) (-SIGN) * vn, KS.rho, mTT - (mT1T1 + mT2T2)) + KS.f[w00] + zSum + 2 * wSum;
			else if (ct2 == 0)
				KS.f[i] = (dreal) 0.5 * ((mT1T1 - mTT) + ct1 * (mT1 - mT1T2T2)) - (KS.f[w] + KS.f[z]);
			else if (ct1 == 0)
				KS.f[i] = (dreal) 0.5 * ((mT2T2 - mTT) + ct2 * (mT2 - mT1T1T2)) - (KS.f[w] + KS.f[z]);
			else
				KS.f[i] =
					(dreal) 0.25 * (lbm_fma_rn((dreal) (ct1 * ct2) * KS.rho, vt1 * vt2, mTT) + (ct2 * mT1T1T2 + ct1 * mT1T2T2)) - (KS.f[w] + KS.f[z]);
		}
	}

	template <typename LBM_KS>
	__cuda_callable__ static void preCollision(DATA& SD, LBM_KS& KS, map_t mapgi, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm, idx z, idx zp)
	{
		if (mapgi == GEO_NOTHING || isOutflowPassBC(mapgi)) {
			KS.rho = 1;
			KS.vx = 0;
			KS.vy = 0;
			KS.vz = 0;
			return;
		}

		// modify pull location for streaming
		if (mapgi == GEO_ADJOINT_OUTFLOW_RIGHT)
			xp = x = xm;

		if (mapgi == GEO_ADJOINT_FLUID || mapgi == GEO_ADJOINT_FLUID_m || mapgi == GEO_ADJOINT_WALL || mapgi == GEO_ADJOINT_INFLOW_BB_LEFT
			|| mapgi == GEO_ADJOINT_OUTFLOW_RIGHT)
		{
			STREAMING::streamingAdjoint(SD, KS, xm, x, xp, ym, y, yp, zm, z, zp);
		}
		else if (mapgi != GEO_OUTFLOW_RIGHT_INTERP)
			STREAMING::streaming(SD, KS, xm, x, xp, ym, y, yp, zm, z, zp);

		// boundary conditions
		switch (mapgi) {
			case GEO_INFLOW:
				SD.inflow(KS, x, y, z);
				KS.rho = 1;
				COLL::setEquilibrium(KS);
				break;
			case GEO_INFLOW_MOMENT:
				SD.inflow(KS, x, y, z);
				applySymmetryCorner(SD, KS, xm, x, xp, ym, y, yp, zm, z, zp);
				inflowMoment(detectBCFace(SD, xm, x, xp, ym, y, yp, zm, z, zp), KS);
				break;
			case GEO_INFLOW_BOUNCEBACK:
				SD.inflow(KS, x, y, z);
				applySymmetryCorner(SD, KS, xm, x, xp, ym, y, yp, zm, z, zp);
				// collision step: bounce-back with modified right-hand-side:
				// -2/c_s^2 * rho(x_wall, t_n) * (\xi_k, v_wall)
				{
					dreal t;
					// clang-format off
					t = KS.f[ppp];
					KS.f[ppp] = KS.f[mmm] - no6*KS.rho*n1o216*(- KS.vx - KS.vy - KS.vz);
					KS.f[mmm] = t         - no6*KS.rho*n1o216*(  KS.vx + KS.vy + KS.vz);

					t = KS.f[ppz];
					KS.f[ppz] = KS.f[mmz] - no6*KS.rho*n1o54*(- KS.vx - KS.vy);
					KS.f[mmz] = t         - no6*KS.rho*n1o54*(  KS.vx + KS.vy);

					t = KS.f[ppm];
					KS.f[ppm] = KS.f[mmp] - no6*KS.rho*n1o216*(- KS.vx - KS.vy + KS.vz);
					KS.f[mmp] = t         - no6*KS.rho*n1o216*(  KS.vx + KS.vy - KS.vz);

					t = KS.f[pzp];
					KS.f[pzp] = KS.f[mzm] - no6*KS.rho*n1o54*(- KS.vx - KS.vz);
					KS.f[mzm] = t         - no6*KS.rho*n1o54*(  KS.vx + KS.vz);

					t = KS.f[pzz];
					KS.f[pzz] = KS.f[mzz] - no6*KS.rho*n2o27*(- KS.vx);
					KS.f[mzz] = t         - no6*KS.rho*n2o27*(  KS.vx);

					t = KS.f[pzm];
					KS.f[pzm] = KS.f[mzp] - no6*KS.rho*n1o54*(- KS.vx + KS.vz);
					KS.f[mzp] = t         - no6*KS.rho*n1o54*(  KS.vx - KS.vz);

					t = KS.f[pmp];
					KS.f[pmp] = KS.f[mpm] - no6*KS.rho*n1o216*(- KS.vx + KS.vy - KS.vz);
					KS.f[mpm] = t         - no6*KS.rho*n1o216*(  KS.vx - KS.vy + KS.vz);

					t = KS.f[pmz];
					KS.f[pmz] = KS.f[mpz] - no6*KS.rho*n1o54*(- KS.vx + KS.vy);
					KS.f[mpz] = t         - no6*KS.rho*n1o54*(  KS.vx - KS.vy);

					t = KS.f[pmm];
					KS.f[pmm] = KS.f[mpp] - no6*KS.rho*n1o216*(- KS.vx + KS.vy + KS.vz);
					KS.f[mpp] = t         - no6*KS.rho*n1o216*(  KS.vx - KS.vy - KS.vz);

					t = KS.f[zpp];
					KS.f[zpp] = KS.f[zmm] - no6*KS.rho*n1o54*(- KS.vy - KS.vz);
					KS.f[zmm] = t         - no6*KS.rho*n1o54*(  KS.vy + KS.vz);

					t = KS.f[zpz];
					KS.f[zpz] = KS.f[zmz] - no6*KS.rho*n2o27*(- KS.vy);
					KS.f[zmz] = t         - no6*KS.rho*n2o27*(  KS.vy);

					t = KS.f[zpm];
					KS.f[zpm] = KS.f[zmp] - no6*KS.rho*n1o54*(- KS.vy + KS.vz);
					KS.f[zmp] = t         - no6*KS.rho*n1o54*(  KS.vy - KS.vz);

					t = KS.f[zzp];
					KS.f[zzp] = KS.f[zzm] - no6*KS.rho*n2o27*(- KS.vz);
					KS.f[zzm] = t         - no6*KS.rho*n2o27*(  KS.vz);
					// clang-format on
				}
				break;
			case GEO_INFLOW_EQ_LEFT:
				SD.inflow(KS, x, y, z);
				applySymmetryCorner(SD, KS, xm, x, xp, ym, y, yp, zm, z, zp);
				// clang-format off
				KS.rho = (dreal)1.0/(1-KS.vx) * (
					(
						KS.f[zzz] + (
							+ ((KS.f[zpp] + KS.f[zmm]) + (KS.f[zpm] + KS.f[zmp]))
							+ ((KS.f[zpz] + KS.f[zmz]) + (KS.f[zzp] + KS.f[zzm]))
						)
					)
					+ 2*(
						KS.f[mzz] + (
							+ ((KS.f[mpp] + KS.f[mmm]) + (KS.f[mpm] + KS.f[mmp]))
							+ ((KS.f[mpz] + KS.f[mmz]) + (KS.f[mzp] + KS.f[mzm]))
						)
					)
				);
				// clang-format on
				COLL::setEquilibrium(KS);
				break;
			case GEO_OUTFLOW_EQ:
				applySymmetryCorner(SD, KS, xm, x, xp, ym, y, yp, zm, z, zp);
				COLL::computeDensityAndVelocity(KS);
				KS.rho = 1;
				COLL::setEquilibrium(KS);
				break;
			case GEO_WALL:
				// does not affect the computation, only the output
				KS.rho = 1;
				KS.vx = 0;
				KS.vy = 0;
				KS.vz = 0;
				// collision step: bounce-back
				TNL::swap(KS.f[mmm], KS.f[ppp]);
				TNL::swap(KS.f[mmz], KS.f[ppz]);
				TNL::swap(KS.f[mmp], KS.f[ppm]);
				TNL::swap(KS.f[mzm], KS.f[pzp]);
				TNL::swap(KS.f[mzz], KS.f[pzz]);
				TNL::swap(KS.f[mzp], KS.f[pzm]);
				TNL::swap(KS.f[mpm], KS.f[pmp]);
				TNL::swap(KS.f[mpz], KS.f[pmz]);
				TNL::swap(KS.f[mpp], KS.f[pmm]);
				TNL::swap(KS.f[zmm], KS.f[zpp]);
				TNL::swap(KS.f[zzm], KS.f[zzp]);
				TNL::swap(KS.f[zmz], KS.f[zpz]);
				TNL::swap(KS.f[zmp], KS.f[zpm]);
				break;
			case GEO_SYMMETRY:
				{
					// Detect ghost half-spaces on all six sides, handling edges and corners.
					// The symmetry cell acts as a fluid cell
					// and directions towards GEO_NOTHING determine the normal of the symmetry plane.
					std::uint8_t ghosts = 0;
					if (SD.map(xm, y, z) == GEO_NOTHING)
						ghosts |= bc_face::XM;
					if (SD.map(xp, y, z) == GEO_NOTHING)
						ghosts |= bc_face::XP;
					if (SD.map(x, ym, z) == GEO_NOTHING)
						ghosts |= bc_face::YM;
					if (SD.map(x, yp, z) == GEO_NOTHING)
						ghosts |= bc_face::YP;
					if (SD.map(x, y, zm) == GEO_NOTHING)
						ghosts |= bc_face::ZM;
					if (SD.map(x, y, zp) == GEO_NOTHING)
						ghosts |= bc_face::ZP;
					applySymmetry(KS, ghosts);
					COLL::computeDensityAndVelocity(KS);
					break;
				}

			// Adjoint boundary conditions
			case GEO_ADJOINT_FLUID:
				COLL::collision(KS);
				break;
			case GEO_ADJOINT_FLUID_m:
				COLL::collision(KS);
				COLL::setEquilibrium(KS);  // adds measured data
				break;
			case GEO_ADJOINT_WALL:	// works same as GEO_WALL --- only streaming step is different in adjoint
				// collision step: bounce-back
				TNL::swap(KS.f[mmm], KS.f[ppp]);
				TNL::swap(KS.f[mmz], KS.f[ppz]);
				TNL::swap(KS.f[mmp], KS.f[ppm]);
				TNL::swap(KS.f[mzm], KS.f[pzp]);
				TNL::swap(KS.f[mzz], KS.f[pzz]);
				TNL::swap(KS.f[mzp], KS.f[pzm]);
				TNL::swap(KS.f[mpm], KS.f[pmp]);
				TNL::swap(KS.f[mpz], KS.f[pmz]);
				TNL::swap(KS.f[mpp], KS.f[pmm]);
				TNL::swap(KS.f[zmm], KS.f[zpp]);
				TNL::swap(KS.f[zzm], KS.f[zzp]);
				TNL::swap(KS.f[zmz], KS.f[zpz]);
				TNL::swap(KS.f[zmp], KS.f[zpm]);
				break;
			case GEO_ADJOINT_INFLOW_BB_LEFT:
				{
					// site-local post-collision populations from the wall node are needed
					// for a valid rho in the collision below; the streamed +x populations
					// come from the ghost face (unused in AB) and would pollute the state
					KS.f[mzz] = TNL::Backend::ldg(SD.df(df_cur, pzz, x, y, z));
					KS.f[mpz] = TNL::Backend::ldg(SD.df(df_cur, pmz, x, y, z));
					KS.f[mmz] = TNL::Backend::ldg(SD.df(df_cur, ppz, x, y, z));
					KS.f[mzp] = TNL::Backend::ldg(SD.df(df_cur, pzm, x, y, z));
					KS.f[mzm] = TNL::Backend::ldg(SD.df(df_cur, pzp, x, y, z));
					KS.f[mpp] = TNL::Backend::ldg(SD.df(df_cur, pmm, x, y, z));
					KS.f[mpm] = TNL::Backend::ldg(SD.df(df_cur, pmp, x, y, z));
					KS.f[mmp] = TNL::Backend::ldg(SD.df(df_cur, ppm, x, y, z));
					KS.f[mmm] = TNL::Backend::ldg(SD.df(df_cur, ppp, x, y, z));
					dreal temp_f_mzz = KS.f[mzz];
					dreal temp_f_mpz = KS.f[mpz];
					dreal temp_f_mmz = KS.f[mmz];
					dreal temp_f_mzp = KS.f[mzp];
					dreal temp_f_mzm = KS.f[mzm];
					dreal temp_f_mpp = KS.f[mpp];
					dreal temp_f_mpm = KS.f[mpm];
					dreal temp_f_mmp = KS.f[mmp];
					dreal temp_f_mmm = KS.f[mmm];
					COLL::collision(KS);
					// load current inflow velocity profile
					SD.inflow(KS, x, y, z);
					// do extra collision because of inflow velocity profile
					// clang-format off
					dreal result = n2o27 * temp_f_mzz * no6 * (-KS.vx)
								 + n1o54 * temp_f_mpz * no6 * (-KS.vx + KS.vy)
								 + n1o54 * temp_f_mmz * no6 * (-KS.vx - KS.vy)
								 + n1o54 * temp_f_mzp * no6 * (-KS.vx + KS.vz)
								 + n1o54 * temp_f_mzm * no6 * (-KS.vx - KS.vz)
								 + n1o216 * temp_f_mpp * no6 * (-KS.vx + KS.vy + KS.vz)
								 + n1o216 * temp_f_mpm * no6 * (-KS.vx + KS.vy - KS.vz)
								 + n1o216 * temp_f_mmp * no6 * (-KS.vx - KS.vy + KS.vz)
								 + n1o216 * temp_f_mmm * no6 * (-KS.vx - KS.vy - KS.vz);
					// clang-format on
					KS.f[mmm] -= result;
					KS.f[mmz] -= result;
					KS.f[mmp] -= result;
					KS.f[mzm] -= result;
					KS.f[mzz] -= result;
					KS.f[mzp] -= result;
					KS.f[mpm] -= result;
					KS.f[mpz] -= result;
					KS.f[mpp] -= result;
					KS.f[zmm] -= result;
					KS.f[zmz] -= result;
					KS.f[zmp] -= result;
					KS.f[zzm] -= result;
					KS.f[zzz] -= result;
					KS.f[zzp] -= result;
					KS.f[zpm] -= result;
					KS.f[zpz] -= result;
					KS.f[zpp] -= result;
					KS.f[pmm] -= result;
					KS.f[pmz] -= result;
					KS.f[pmp] -= result;
					KS.f[pzm] -= result;
					KS.f[pzz] -= result;
					KS.f[pzp] -= result;
					KS.f[ppm] -= result;
					KS.f[ppz] -= result;
					KS.f[ppp] -= result;
					// calculate gradient
					SD.inflow(KS, x, y, z);
					// COLL::computeDensityAndVelocity(KS);
					// streaming
					break;
				}
			case GEO_ADJOINT_OUTFLOW_RIGHT:
				COLL::computeDensityAndVelocity_Wall(KS);  //! collision without drho (because K.rho = 1 always)
				break;

			default:
				COLL::computeDensityAndVelocity(KS);
				break;
		}
	}

	__cuda_callable__ static bool doCollision(map_t mapgi)
	{
		// by default, collision is done on non-BC sites only
		// additionally, BCs which include the collision step should be specified here
		return isFluid(mapgi) || isSymmetric(mapgi) || mapgi == GEO_INFLOW_MOMENT;
	}

	template <typename LBM_KS>
	__cuda_callable__ static void
	postCollision(DATA& SD, LBM_KS& KS, map_t mapgi, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm, idx z, idx zp)
	{
		if (mapgi == GEO_NOTHING || isOutflowPassBC(mapgi))
			return;

		STREAMING::postCollisionStreaming(SD, KS, xm, x, xp, ym, y, yp, zm, z, zp);
	}
};
