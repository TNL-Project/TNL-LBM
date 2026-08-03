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
		GEO_PERIODIC,
		GEO_NOTHING,
		GEO_SYM_TOP,
		GEO_SYM_BOTTOM,
		GEO_SYM_LEFT,
		GEO_SYM_RIGHT
	};

	__cuda_callable__ static bool isPeriodic(map_t mapgi)
	{
		return mapgi == GEO_PERIODIC;
	}

	__cuda_callable__ static bool isSymmetric(map_t mapgi)
	{
		return mapgi == GEO_SYM_TOP || mapgi == GEO_SYM_BOTTOM || mapgi == GEO_SYM_LEFT || mapgi == GEO_SYM_RIGHT;
	}

	__cuda_callable__ static bool isFluid(map_t mapgi)
	{
		return mapgi == GEO_FLUID;
	}

	__cuda_callable__ static bool isWall(map_t mapgi)
	{
		return mapgi == GEO_WALL;
	}

	template <typename LBM_KS>
	__cuda_callable__ static void preCollision(DATA& SD, LBM_KS& KS, map_t mapgi, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm, idx z, idx zp)
	{
		if (mapgi == GEO_NOTHING) {
			// does not affect the computation, only the output
			KS.rho = 1;
			KS.vx = 0;
			KS.vy = 0;
			return;
		}

		// modify pull location for streaming
		if (mapgi == GEO_OUTFLOW_RIGHT)
			xp = x = xm;

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
				COLL::computeDensityAndVelocity(KS);
				KS.rho = 1;
				COLL::setEquilibrium(KS);
				break;
			case GEO_OUTFLOW_RIGHT:
				COLL::computeDensityAndVelocity(KS);
				KS.rho = 1;
				break;
			case GEO_OUTFLOW_RIGHT_INTERP:
				STREAMING::streamingInterpRight(SD, KS, xm, x, xp, ym, y, yp, zm, z, zp);
				COLL::computeDensityAndVelocity(KS);
				COLL::setEquilibriumDecomposition(KS, 1);
				KS.rho = 1;
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
			case GEO_SYM_LEFT:
				KS.f[dir9::pm] = KS.f[dir9::mm];
				KS.f[dir9::pz] = KS.f[dir9::mz];
				KS.f[dir9::pp] = KS.f[dir9::mp];
				COLL::computeDensityAndVelocity(KS);
				break;
			case GEO_SYM_RIGHT:
				KS.f[dir9::mm] = KS.f[dir9::pm];
				KS.f[dir9::mz] = KS.f[dir9::pz];
				KS.f[dir9::mp] = KS.f[dir9::pp];
				COLL::computeDensityAndVelocity(KS);
				break;
			case GEO_SYM_BOTTOM:
				KS.f[dir9::mp] = KS.f[dir9::mm];
				KS.f[dir9::zp] = KS.f[dir9::zm];
				KS.f[dir9::pp] = KS.f[dir9::pm];
				COLL::computeDensityAndVelocity(KS);
				break;
			case GEO_SYM_TOP:
				KS.f[dir9::mm] = KS.f[dir9::mp];
				KS.f[dir9::zm] = KS.f[dir9::zp];
				KS.f[dir9::pm] = KS.f[dir9::pp];
				COLL::computeDensityAndVelocity(KS);
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
		return isFluid(mapgi) || isPeriodic(mapgi) || isSymmetric(mapgi) || mapgi == GEO_INFLOW_LEFT || mapgi == GEO_OUTFLOW_RIGHT
			|| mapgi == GEO_OUTFLOW_RIGHT_INTERP;
	}

	template <typename LBM_KS>
	__cuda_callable__ static void
	postCollision(DATA& SD, LBM_KS& KS, map_t mapgi, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm, idx z, idx zp)
	{
		if (mapgi == GEO_NOTHING)
			return;

		STREAMING::postCollisionStreaming(SD, KS, xm, x, xp, ym, y, yp, zm, z, zp);
	}
};
