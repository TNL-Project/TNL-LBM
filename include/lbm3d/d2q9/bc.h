#pragma once

#include "lbm3d/defs.h"

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
		GEO_INFLOW_LEFT,
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

	// gathers read the postcollision state finalized by the previous launch
	// and live in streaming_*.h; the BC body then follows the legacy cases
	template <typename LBM_KS>
	__cuda_callable__ static void outflowPass(DATA& SD, LBM_KS& KS, map_t mapgi, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm, idx z, idx zp)
	{
		switch (mapgi) {
			case GEO_OUTFLOW_RIGHT:
				STREAMING::streamingOutflowRight(SD, KS, xm, x, xp, ym, y, yp, zm, z, zp);
				applySymmetryCorner(SD, KS, xm, x, xp, ym, y, yp, zm, z, zp);
				COLL::computeDensityAndVelocity(KS);
				KS.rho = 1;
				COLL::collision(KS);
				STREAMING::postCollisionStreaming(SD, KS, xm, x, xp, ym, y, yp, zm, z, zp);
				break;
			case GEO_OUTFLOW_RIGHT_INTERP:
				STREAMING::streamingOutflowInterpRight(SD, KS, xm, x, xp, ym, y, yp, zm, z, zp);
				applySymmetryCorner(SD, KS, xm, x, xp, ym, y, yp, zm, z, zp);
				COLL::computeDensityAndVelocity(KS);
				COLL::setEquilibriumDecomposition(KS, 1);
				KS.rho = 1;
				COLL::collision(KS);
				STREAMING::postCollisionStreaming(SD, KS, xm, x, xp, ym, y, yp, zm, z, zp);
				break;
		}
	}

	// Bitmask of ghost half-spaces adjacent to a GEO_SYMMETRY cell.
	// Each bit marks a side where the neighbor cell is GEO_NOTHING (the domain-frame ghost layer).
	enum SYM_SIDES : std::uint8_t
	{
		SYM_XM = 1 << 0,  // ghost at x-1
		SYM_XP = 1 << 1,  // ghost at x+1
		SYM_YM = 1 << 2,  // ghost at y-1
		SYM_YP = 1 << 3,  // ghost at y+1
	};

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
			const bool cx = ex != 1 && (ghosts & (ex == 2 ? SYM_XM : SYM_XP));
			const bool cy = ey != 1 && (ghosts & (ey == 2 ? SYM_YM : SYM_YP));
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
				ghosts |= SYM_YM;
			if (SD.map(x, yp, z) == GEO_NOTHING)
				ghosts |= SYM_YP;
		}
		if (SD.map(x, ym, z) == GEO_SYMMETRY || SD.map(x, yp, z) == GEO_SYMMETRY) {
			if (SD.map(xm, y, z) == GEO_NOTHING)
				ghosts |= SYM_XM;
			if (SD.map(xp, y, z) == GEO_NOTHING)
				ghosts |= SYM_XP;
		}
		if (ghosts)
			applySymmetry(KS, ghosts);
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
			case GEO_INFLOW_LEFT:
				{
					SD.inflow(KS, x, y, z);
					applySymmetryCorner(SD, KS, xm, x, xp, ym, y, yp, zm, z, zp);
					// moment boundary condition by Pavel Eichler https://doi.org/10.1016/j.camwa.2024.08.009
					// 2D reduction: mass + y-momentum + Π_yy (normal stress)
					// expressions symetrized: Y-mirror directions paired so float32 summation is commutative
					KS.rho =
						(KS.f[dir9::zz] + (KS.f[dir9::zp] + KS.f[dir9::zm]) + 2 * (KS.f[dir9::mz] + (KS.f[dir9::mm] + KS.f[dir9::mp]))) / (1 - KS.vx);
					dreal m01 = KS.rho * KS.vy;
					dreal m02 = n1o3 * KS.rho + KS.rho * (KS.vy * KS.vy);
					KS.f[dir9::pp] = (dreal) 0.5 * (m02 + m01 - 2 * KS.f[dir9::zp] - 2 * KS.f[dir9::mp]);
					KS.f[dir9::pm] = (dreal) 0.5 * (m02 - m01 - 2 * KS.f[dir9::zm] - 2 * KS.f[dir9::mm]);
					KS.f[dir9::pz] = KS.rho - KS.f[dir9::zz] - KS.f[dir9::mz] - (KS.f[dir9::zp] + KS.f[dir9::zm]) - (KS.f[dir9::pp] + KS.f[dir9::pm])
								   - (KS.f[dir9::mm] + KS.f[dir9::mp]);
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
						ghosts |= SYM_XM;
					if (SD.map(xp, y, z) == GEO_NOTHING)
						ghosts |= SYM_XP;
					if (SD.map(x, ym, z) == GEO_NOTHING)
						ghosts |= SYM_YM;
					if (SD.map(x, yp, z) == GEO_NOTHING)
						ghosts |= SYM_YP;
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
		return isFluid(mapgi) || isSymmetric(mapgi) || mapgi == GEO_INFLOW_LEFT;
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
