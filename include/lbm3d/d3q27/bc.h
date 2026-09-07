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

	enum GEO : map_t  // NOLINT(performance-enum-size) base width must match the map_t storage type

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
		GEO_ADJOINT_OUTFLOW_RIGHT,
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

	// BC face detection from the map, hot path of the main kernel: locate the
	// interior side - the axis-neighbor acting as fluid (GEO_FLUID, or
	// GEO_SYMMETRY which acts as a fluid cell - planes may cover the full
	// face including symmetry-row corners) - and point the outward normal
	// away from it. This is sufficient because the validated-map preconditions
	// do the work: GEO_NOTHING (never interior) lies on the outward side of
	// every face-detected BC cell - LBM_BLOCK::validateFaceDetectedBC (called
	// from State::SimInit) guarantees that and exactly one interior
	// axis-neighbor per BC site, and finalizeInletRegions guarantees it for
	// every claimed inflow cell (fluid/symmetry inward, ghost outward) after
	// mirroring this exact first-hit order against the inlet region's (axis,
	// sign). Map-level invariants therefore decide, and the check order only
	// matters for unvalidated maps.
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

	// BC tags whose runtime face detection (detectBCFace) must find exactly
	// one interior axis-neighbor (validated by LBM_BLOCK::validateFaceDetectedBC)
	__cuda_callable__ static bool needsFaceDetection(map_t mapgi)
	{
		return mapgi == GEO_OUTFLOW_RIGHT || mapgi == GEO_OUTFLOW_RIGHT_INTERP || mapgi == GEO_INFLOW_MOMENT;
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
			case 0:  return dir27::mmm;
			case 1:  return dir27::mmz;
			case 2:  return dir27::mmp;
			case 3:  return dir27::mzm;
			case 4:  return dir27::mzz;
			case 5:  return dir27::mzp;
			case 6:  return dir27::mpm;
			case 7:  return dir27::mpz;
			case 8:  return dir27::mpp;
			case 9:  return dir27::zmm;
			case 10: return dir27::zmz;
			case 11: return dir27::zmp;
			case 12: return dir27::zzm;
			case 13: return dir27::zzz;
			case 14: return dir27::zzp;
			case 15: return dir27::zpm;
			case 16: return dir27::zpz;
			case 17: return dir27::zpp;
			case 18: return dir27::pmm;
			case 19: return dir27::pmz;
			case 20: return dir27::pmp;
			case 21: return dir27::pzm;
			case 22: return dir27::pzz;
			case 23: return dir27::pzp;
			case 24: return dir27::ppm;
			case 25: return dir27::ppz;
			case 26: return dir27::ppp;
			default: return dir27::zzz;
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
	// Unlike D2Q9 (per-face template instantiations, see d2q9/bc.h),
	// this model keeps the runtime face: measured in the fused kernels it is the fastest D3Q27 dispatch,
	// while per-face instantiation inlines six bodies into the AA kernel
	// and causes a -6.9% regression due to register spills.
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

		// moment reconstruction requires physical populations (identity for non-well storages)
		const auto f = [&KS](int i) -> dreal
		{
			return COLL::fromStorage(KS.f, i);
		};

		const dreal zCorners = ((f(zpp) + f(zmm)) + (f(zpm) + f(zmp)));
		const dreal wCorners = ((f(wpp) + f(wmm)) + (f(wpm) + f(wmp)));
		const dreal zSum = zCorners + ((f(zpz) + f(zmz)) + (f(zzp) + f(zzm)));
		const dreal wSum = wCorners + ((f(wpz) + f(wmz)) + (f(wzp) + f(wzm)));

		// reciprocal first, then multiply -- matches the legacy XM denominator
		KS.rho = (dreal) 1.0 / (1 + SIGN * vn) * ((f(z00) + zSum) + 2 * (f(w00) + wSum));

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
				KS.f[i] = COLL::toStorage(lbm_fma_rn((dreal) (-SIGN) * vn, KS.rho, mTT - (mT1T1 + mT2T2)) + f(w00) + zSum + 2 * wSum, i);
			else if (ct2 == 0)
				KS.f[i] = COLL::toStorage((dreal) 0.5 * ((mT1T1 - mTT) + ct1 * (mT1 - mT1T2T2)) - (f(w) + f(z)), i);
			else if (ct1 == 0)
				KS.f[i] = COLL::toStorage((dreal) 0.5 * ((mT2T2 - mTT) + ct2 * (mT2 - mT1T1T2)) - (f(w) + f(z)), i);
			else
				KS.f[i] = COLL::toStorage(
					(dreal) 0.25 * (lbm_fma_rn((dreal) (ct1 * ct2) * KS.rho, vt1 * vt2, mTT) + (ct2 * mT1T1T2 + ct1 * mT1T2T2)) - (f(w) + f(z)), i
				);
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
		else if (mapgi != GEO_OUTFLOW_RIGHT_INTERP) {
			STREAMING::streaming(SD, KS, xm, x, xp, ym, y, yp, zm, z, zp);
		}

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
					t = KS.f[dir27::ppp];
					KS.f[dir27::ppp] = KS.f[dir27::mmm] - no6*KS.rho*n1o216*(- KS.vx - KS.vy - KS.vz);
					KS.f[dir27::mmm] = t                - no6*KS.rho*n1o216*(  KS.vx + KS.vy + KS.vz);

					t = KS.f[dir27::ppz];
					KS.f[dir27::ppz] = KS.f[dir27::mmz] - no6*KS.rho*n1o54*(- KS.vx - KS.vy);
					KS.f[dir27::mmz] = t                - no6*KS.rho*n1o54*(  KS.vx + KS.vy);

					t = KS.f[dir27::ppm];
					KS.f[dir27::ppm] = KS.f[dir27::mmp] - no6*KS.rho*n1o216*(- KS.vx - KS.vy + KS.vz);
					KS.f[dir27::mmp] = t                - no6*KS.rho*n1o216*(  KS.vx + KS.vy - KS.vz);

					t = KS.f[dir27::pzp];
					KS.f[dir27::pzp] = KS.f[dir27::mzm] - no6*KS.rho*n1o54*(- KS.vx - KS.vz);
					KS.f[dir27::mzm] = t                - no6*KS.rho*n1o54*(  KS.vx + KS.vz);

					t = KS.f[dir27::pzz];
					KS.f[dir27::pzz] = KS.f[dir27::mzz] - no6*KS.rho*n2o27*(- KS.vx);
					KS.f[dir27::mzz] = t                - no6*KS.rho*n2o27*(  KS.vx);

					t = KS.f[dir27::pzm];
					KS.f[dir27::pzm] = KS.f[dir27::mzp] - no6*KS.rho*n1o54*(- KS.vx + KS.vz);
					KS.f[dir27::mzp] = t                - no6*KS.rho*n1o54*(  KS.vx - KS.vz);

					t = KS.f[dir27::pmp];
					KS.f[dir27::pmp] = KS.f[dir27::mpm] - no6*KS.rho*n1o216*(- KS.vx + KS.vy - KS.vz);
					KS.f[dir27::mpm] = t                - no6*KS.rho*n1o216*(  KS.vx - KS.vy + KS.vz);

					t = KS.f[dir27::pmz];
					KS.f[dir27::pmz] = KS.f[dir27::mpz] - no6*KS.rho*n1o54*(- KS.vx + KS.vy);
					KS.f[dir27::mpz] = t                - no6*KS.rho*n1o54*(  KS.vx - KS.vy);

					t = KS.f[dir27::pmm];
					KS.f[dir27::pmm] = KS.f[dir27::mpp] - no6*KS.rho*n1o216*(- KS.vx + KS.vy + KS.vz);
					KS.f[dir27::mpp] = t                - no6*KS.rho*n1o216*(  KS.vx - KS.vy - KS.vz);

					t = KS.f[dir27::zpp];
					KS.f[dir27::zpp] = KS.f[dir27::zmm] - no6*KS.rho*n1o54*(- KS.vy - KS.vz);
					KS.f[dir27::zmm] = t                - no6*KS.rho*n1o54*(  KS.vy + KS.vz);

					t = KS.f[dir27::zpz];
					KS.f[dir27::zpz] = KS.f[dir27::zmz] - no6*KS.rho*n2o27*(- KS.vy);
					KS.f[dir27::zmz] = t                - no6*KS.rho*n2o27*(  KS.vy);

					t = KS.f[dir27::zpm];
					KS.f[dir27::zpm] = KS.f[dir27::zmp] - no6*KS.rho*n1o54*(- KS.vy + KS.vz);
					KS.f[dir27::zmp] = t                - no6*KS.rho*n1o54*(  KS.vy - KS.vz);

					t = KS.f[dir27::zzp];
					KS.f[dir27::zzp] = KS.f[dir27::zzm] - no6*KS.rho*n2o27*(- KS.vz);
					KS.f[dir27::zzm] = t                - no6*KS.rho*n2o27*(  KS.vz);
					// clang-format on
				}
				break;
			case GEO_INFLOW_EQ_LEFT:
				{
					SD.inflow(KS, x, y, z);
					applySymmetryCorner(SD, KS, xm, x, xp, ym, y, yp, zm, z, zp);
					// moment reconstruction requires physical populations (identity for non-well storages)
					const auto f = [&KS](int i) -> dreal
					{
						return COLL::fromStorage(KS.f, i);
					};
					// clang-format off
					KS.rho = (dreal)1.0/(1-KS.vx) * (
						(
							f(dir27::zzz) + (
								+ ((f(dir27::zpp) + f(dir27::zmm)) + (f(dir27::zpm) + f(dir27::zmp)))
								+ ((f(dir27::zpz) + f(dir27::zmz)) + (f(dir27::zzp) + f(dir27::zzm)))
							)
						)
						+ 2*(
							f(dir27::mzz) + (
								+ ((f(dir27::mpp) + f(dir27::mmm)) + (f(dir27::mpm) + f(dir27::mmp)))
								+ ((f(dir27::mpz) + f(dir27::mmz)) + (f(dir27::mzp) + f(dir27::mzm)))
							)
						)
					);
					// clang-format on
					COLL::setEquilibrium(KS);
					break;
				}
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
				TNL::swap(KS.f[dir27::mmm], KS.f[dir27::ppp]);
				TNL::swap(KS.f[dir27::mmz], KS.f[dir27::ppz]);
				TNL::swap(KS.f[dir27::mmp], KS.f[dir27::ppm]);
				TNL::swap(KS.f[dir27::mzm], KS.f[dir27::pzp]);
				TNL::swap(KS.f[dir27::mzz], KS.f[dir27::pzz]);
				TNL::swap(KS.f[dir27::mzp], KS.f[dir27::pzm]);
				TNL::swap(KS.f[dir27::mpm], KS.f[dir27::pmp]);
				TNL::swap(KS.f[dir27::mpz], KS.f[dir27::pmz]);
				TNL::swap(KS.f[dir27::mpp], KS.f[dir27::pmm]);
				TNL::swap(KS.f[dir27::zmm], KS.f[dir27::zpp]);
				TNL::swap(KS.f[dir27::zzm], KS.f[dir27::zzp]);
				TNL::swap(KS.f[dir27::zmz], KS.f[dir27::zpz]);
				TNL::swap(KS.f[dir27::zmp], KS.f[dir27::zpm]);
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
				TNL::swap(KS.f[dir27::mmm], KS.f[dir27::ppp]);
				TNL::swap(KS.f[dir27::mmz], KS.f[dir27::ppz]);
				TNL::swap(KS.f[dir27::mmp], KS.f[dir27::ppm]);
				TNL::swap(KS.f[dir27::mzm], KS.f[dir27::pzp]);
				TNL::swap(KS.f[dir27::mzz], KS.f[dir27::pzz]);
				TNL::swap(KS.f[dir27::mzp], KS.f[dir27::pzm]);
				TNL::swap(KS.f[dir27::mpm], KS.f[dir27::pmp]);
				TNL::swap(KS.f[dir27::mpz], KS.f[dir27::pmz]);
				TNL::swap(KS.f[dir27::mpp], KS.f[dir27::pmm]);
				TNL::swap(KS.f[dir27::zmm], KS.f[dir27::zpp]);
				TNL::swap(KS.f[dir27::zzm], KS.f[dir27::zzp]);
				TNL::swap(KS.f[dir27::zmz], KS.f[dir27::zpz]);
				TNL::swap(KS.f[dir27::zmp], KS.f[dir27::zpm]);
				break;
			case GEO_ADJOINT_INFLOW_BB_LEFT:
				{
					// site-local post-collision populations from the wall node are needed
					// for a valid rho in the collision below; the streamed +x populations
					// come from the ghost face (unused in AB) and would pollute the state
					KS.f[dir27::mzz] = TNL::Backend::ldg(SD.df(df_cur, dir27::pzz, x, y, z));
					KS.f[dir27::mpz] = TNL::Backend::ldg(SD.df(df_cur, dir27::pmz, x, y, z));
					KS.f[dir27::mmz] = TNL::Backend::ldg(SD.df(df_cur, dir27::ppz, x, y, z));
					KS.f[dir27::mzp] = TNL::Backend::ldg(SD.df(df_cur, dir27::pzm, x, y, z));
					KS.f[dir27::mzm] = TNL::Backend::ldg(SD.df(df_cur, dir27::pzp, x, y, z));
					KS.f[dir27::mpp] = TNL::Backend::ldg(SD.df(df_cur, dir27::pmm, x, y, z));
					KS.f[dir27::mpm] = TNL::Backend::ldg(SD.df(df_cur, dir27::pmp, x, y, z));
					KS.f[dir27::mmp] = TNL::Backend::ldg(SD.df(df_cur, dir27::ppm, x, y, z));
					KS.f[dir27::mmm] = TNL::Backend::ldg(SD.df(df_cur, dir27::ppp, x, y, z));
					dreal temp_f_mzz = KS.f[dir27::mzz];
					dreal temp_f_mpz = KS.f[dir27::mpz];
					dreal temp_f_mmz = KS.f[dir27::mmz];
					dreal temp_f_mzp = KS.f[dir27::mzp];
					dreal temp_f_mzm = KS.f[dir27::mzm];
					dreal temp_f_mpp = KS.f[dir27::mpp];
					dreal temp_f_mpm = KS.f[dir27::mpm];
					dreal temp_f_mmp = KS.f[dir27::mmp];
					dreal temp_f_mmm = KS.f[dir27::mmm];
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
					KS.f[dir27::mmm] -= result;
					KS.f[dir27::mmz] -= result;
					KS.f[dir27::mmp] -= result;
					KS.f[dir27::mzm] -= result;
					KS.f[dir27::mzz] -= result;
					KS.f[dir27::mzp] -= result;
					KS.f[dir27::mpm] -= result;
					KS.f[dir27::mpz] -= result;
					KS.f[dir27::mpp] -= result;
					KS.f[dir27::zmm] -= result;
					KS.f[dir27::zmz] -= result;
					KS.f[dir27::zmp] -= result;
					KS.f[dir27::zzm] -= result;
					KS.f[dir27::zzz] -= result;
					KS.f[dir27::zzp] -= result;
					KS.f[dir27::zpm] -= result;
					KS.f[dir27::zpz] -= result;
					KS.f[dir27::zpp] -= result;
					KS.f[dir27::pmm] -= result;
					KS.f[dir27::pmz] -= result;
					KS.f[dir27::pmp] -= result;
					KS.f[dir27::pzm] -= result;
					KS.f[dir27::pzz] -= result;
					KS.f[dir27::pzp] -= result;
					KS.f[dir27::ppm] -= result;
					KS.f[dir27::ppz] -= result;
					KS.f[dir27::ppp] -= result;
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
