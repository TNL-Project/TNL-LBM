#pragma once

#include "lbm3d/defs.h"
#include "lbm_common/rounding.h"

template <typename CONFIG>
struct D2Q9_BC_All
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
		GEO_OUTFLOW_EQ,
		GEO_OUTFLOW_RIGHT,
		GEO_OUTFLOW_RIGHT_INTERP,
		GEO_NOTHING,
		GEO_SYMMETRY
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

	// Outflow face detection from the map: the interior side of the outlet is
	// the axis-neighbor acting as fluid (GEO_FLUID, or GEO_SYMMETRY which acts
	// as a fluid cell - outflow planes may cover the full face including
	// symmetry-row corners), the outward normal points away from it.
	// LBM_BLOCK::validateOutflowPassRegion (called from copyMapToDevice)
	// guarantees exactly one such axis-neighbor per outflow cell, so the
	// check order only matters for unvalidated maps.
	__cuda_callable__ static bool isOutflowInterior(map_t mapgi)
	{
		return mapgi == GEO_FLUID || mapgi == GEO_SYMMETRY;
	}

	__cuda_callable__ static int detectBCFace(DATA& SD, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx z)
	{
		if (isOutflowInterior(SD.map(xm, y, z)))
			return bc_face::XP;
		if (isOutflowInterior(SD.map(xp, y, z)))
			return bc_face::XM;
		if (isOutflowInterior(SD.map(x, ym, z)))
			return bc_face::YP;
		return bc_face::YM;
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
		const int face = detectBCFace(SD, xm, x, xp, ym, y, yp, z);
		switch (mapgi) {
			case GEO_OUTFLOW_RIGHT:
				STREAMING::streamingOutflow(SD, KS, face, xm, x, xp, ym, y, yp, z);
				applySymmetryCorner(SD, KS, xm, x, xp, ym, y, yp, zm, z, zp);
				COLL::computeDensityAndVelocity(KS);
				KS.rho = 1;
				COLL::collision(KS);
				STREAMING::postCollisionStreaming(SD, KS, xm, x, xp, ym, y, yp, zm, z, zp);
				break;
			case GEO_OUTFLOW_RIGHT_INTERP:
				STREAMING::streamingOutflowInterp(SD, KS, face, xm, x, xp, ym, y, yp, z);
				applySymmetryCorner(SD, KS, xm, x, xp, ym, y, yp, zm, z, zp);
				COLL::computeDensityAndVelocity(KS);
				COLL::setEquilibriumDecomposition(KS, 1);
				KS.rho = 1;
				COLL::collision(KS);
				STREAMING::postCollisionStreaming(SD, KS, xm, x, xp, ym, y, yp, zm, z, zp);
				break;
		}
	}

	// Closure for GEO_SYMMETRY cells adjacent to two GEO_NOTHING ghost half-spaces
	// (corners of the domain frame).
	// A population is unknown exactly when one of its non-zero components points through a ghost side
	// ('p*' through the x-1 ghost, 'm*' through x+1, '*p' through y-1, and '*m' through y+1);
	// each unknown gets the value of the within-cell population with every ghost-crossing component flipped
	// (single reflection per orthogonal directions, double for diagonals).
	// The fills are pure copies whose sources never cross a ghost side,
	// so the result does not depend on where in the domain frame the GEO_SYMMETRY planes lie.
	// direction slot for the letter trits (ex, ey) in {m=0, z=1, p=2} packed as 3*ex + ey
	__cuda_callable__ static constexpr std::uint8_t sym_dir_slot(int code)
	{
		// clang-format off
		switch (code) {
			case 0:  return dir9::mm;
			case 1:  return dir9::mz;
			case 2:  return dir9::mp;
			case 3:  return dir9::zm;
			case 4:  return dir9::zz;
			case 5:  return dir9::zp;
			case 6:  return dir9::pm;
			case 7:  return dir9::pz;
			case 8:  return dir9::pp;
			default: return dir9::zz;
		}
		// clang-format on
	}

	template <typename LBM_KS>
	__cuda_callable__ static void applySymmetry(LBM_KS& KS, std::uint8_t ghosts)
	{
		for (int code = 0; code < 9; code++) {
			if (code == 4)
				continue;  // rest population never crosses a ghost side
			const int ex = code / 3;
			const int ey = code % 3;
			// cx/cy are set when the direction's component crosses a ghost side
			// ('p' crosses the minus side, 'm' crosses the plus side, 'z' crosses neither)
			const bool cx = ex != 1 && (ghosts & (ex == 2 ? bc_face::XM : bc_face::XP));
			const bool cy = ey != 1 && (ghosts & (ey == 2 ? bc_face::YM : bc_face::YP));
			if (cx || cy) {
				const int src = 3 * (cx ? 2 - ex : ex) + (cy ? 2 - ey : ey);
				KS.f[sym_dir_slot(code)] = KS.f[sym_dir_slot(src)];
			}
		}
	}

	// Mirror-precondition for BC cells adjacent to GEO_SYMMETRY planes.
	// Checks the BC cell's own 4 neighbors: if a neighbor is GEO_SYMMETRY,
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
		}
		if (SD.map(x, ym, z) == GEO_SYMMETRY || SD.map(x, yp, z) == GEO_SYMMETRY) {
			if (SD.map(xm, y, z) == GEO_NOTHING)
				ghosts |= bc_face::XM;
			if (SD.map(xp, y, z) == GEO_NOTHING)
				ghosts |= bc_face::XP;
		}
		if (ghosts)
			applySymmetry(KS, ghosts);
	}

	// component of direction slot i along axis a (0 = x, 1 = y)
	__cuda_callable__ static constexpr int dcomp(int i, int a)
	{
		return a == 0 ? dir9_cx(i) : dir9_cy(i);
	}

	// slot of the direction with the given (normal, tangential) components
	// (-1 if not found; the tangential axis is the remaining one of the two)
	__cuda_callable__ static constexpr int dslot(int a, int t, int cn, int ct)
	{
		for (int i = 0; i < 9; i++)
			if (dcomp(i, a) == cn && dcomp(i, t) == ct)
				return i;
		return -1;
	}

	// moment boundary condition by Pavel Eichler https://doi.org/10.1016/j.camwa.2024.08.009
	// (2D reduction: mass + tangential momentum + tangential stress), generalized from
	// the legacy left-wall inflow (outward normal -x, face XM) to any domain face:
	// AXIS is the face-normal axis, SIGN the outward sign of the inflow plane, and
	// every slot depends only on those two parameters, so each instantiation folds
	// to just its own face's code. A shared runtime-face form loses this folding and
	// was measured worse in D2Q9's small fused kernels (hills AA -5% from register
	// and literal pressure) and drifts values off the legacy FP contractions (the
	// hills mass-conservation check failed at final time); the derivation in
	// docs/moment-bc-derivation.md pins the rounding to the pre-generalization XM
	// tree. D3Q27 keeps a runtime-face body instead (see d3q27/bc.h): its fused
	// kernel tolerates it (measured fastest there, sim_2 AA neutral) and per-face
	// instantiation reproduces the ptxas spill regression in the D3Q27 AA kernel
	// (sim_2 -6.9%)
	template <int AXIS, int SIGN, typename LBM_KS>
	__cuda_callable__ static void inflowMoment(LBM_KS& KS)
	{
		constexpr int T = 1 - AXIS;

		// layer slots: Z = populations with cn == 0, W = the outward-moving
		// layer (cn == SIGN); each layer is one axis slot (ct == 0) plus the
		// ct == +-1 pair; the pair is summed before joining the axis slot and
		// the W pair is ordered negative-first so that on the legacy face (XM)
		// the density matches the verbatim XM expression tree bit-exactly
		constexpr int z0 = dslot(AXIS, T, 0, 0);
		constexpr int zp = dslot(AXIS, T, 0, 1);
		constexpr int zm = dslot(AXIS, T, 0, -1);
		constexpr int w0 = dslot(AXIS, T, SIGN, 0);
		constexpr int wp = dslot(AXIS, T, SIGN, 1);
		constexpr int wm = dslot(AXIS, T, SIGN, -1);

		const dreal vn = AXIS == 0 ? KS.vx : KS.vy;
		const dreal vt = AXIS == 0 ? KS.vy : KS.vx;

		KS.rho = (KS.f[z0] + (KS.f[zp] + KS.f[zm]) + 2 * (KS.f[w0] + (KS.f[wm] + KS.f[wp]))) / (1 + SIGN * vn);

		// lbm_fma_rn pins replicate the fp-contraction spots the compiler picks
		// for the legacy XM body (verified in SASS): n1o3*rho fused into the
		// stress and rho*vt fused into each corner pair; which product gets fused
		// depends on the whole-function data flow and cannot be left to the compiler
		const dreal mTT = lbm_fma_rn(KS.rho, n1o3, KS.rho * (vt * vt));

		// closed-form reconstruction of the unknown layer (populations moving
		// into the domain, cn == -SIGN); corners first, the axis slot last
		// because it reads the just-written corner slots
		for (int i = 0; i < 9; i++) {
			if (dcomp(i, AXIS) != -SIGN)
				continue;
			const int ct = dcomp(i, T);
			if (ct == 0)
				continue;
			const int z = dslot(AXIS, T, 0, ct);
			const int w = dslot(AXIS, T, SIGN, ct);
			KS.f[i] = (dreal) 0.5 * (lbm_fma_rn((dreal) ct * vt, KS.rho, mTT) - 2 * KS.f[z] - 2 * KS.f[w]);
		}

		// the axis slot closes the mass budget; the subtraction order and pair
		// grouping reproduce the legacy XM chain (corner writes above are read
		// back here, which is why this slot must come last)
		constexpr int up = dslot(AXIS, T, -SIGN, 1);
		constexpr int um = dslot(AXIS, T, -SIGN, -1);
		constexpr int axisSlot = dslot(AXIS, T, -SIGN, 0);
		KS.f[axisSlot] = KS.rho - KS.f[z0] - KS.f[w0] - (KS.f[zp] + KS.f[zm]) - (KS.f[up] + KS.f[um]) - (KS.f[wm] + KS.f[wp]);
	}
	template <typename LBM_KS>
	__cuda_callable__ static void preCollision(DATA& SD, LBM_KS& KS, map_t mapgi, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm, idx z, idx zp)
	{
		if (mapgi == GEO_NOTHING || isOutflowPassBC(mapgi)) {
			KS.rho = 1;
			KS.vx = 0;
			KS.vy = 0;
			return;
		}

		if (mapgi != GEO_OUTFLOW_RIGHT_INTERP)
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
				switch (detectBCFace(SD, xm, x, xp, ym, y, yp, z)) {
					case bc_face::XM:
						inflowMoment<0, -1>(KS);
						break;
					case bc_face::XP:
						inflowMoment<0, 1>(KS);
						break;
					case bc_face::YP:
						inflowMoment<1, 1>(KS);
						break;
					default:
						inflowMoment<1, -1>(KS);
						break;
				}
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
				// collision step: bounce-back
				TNL::swap(KS.f[dir9::mm], KS.f[dir9::pp]);
				TNL::swap(KS.f[dir9::zm], KS.f[dir9::zp]);
				TNL::swap(KS.f[dir9::mz], KS.f[dir9::pz]);
				TNL::swap(KS.f[dir9::mp], KS.f[dir9::pm]);
				break;
			case GEO_SYMMETRY:
				{
					// Detect ghost half-spaces on all four sides, handling corners.
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
					applySymmetry(KS, ghosts);
					COLL::computeDensityAndVelocity(KS);
					break;
				}
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
