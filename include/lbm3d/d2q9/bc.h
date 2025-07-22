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
			// nema zadny vliv na vypocet, jen pro output
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
				TNL::swap(KS.f[mm], KS.f[pp]);
				TNL::swap(KS.f[zm], KS.f[zp]);
				TNL::swap(KS.f[mz], KS.f[pz]);
				TNL::swap(KS.f[mp], KS.f[pm]);
				break;
			case GEO_SYM_LEFT:
				KS.f[pm] = KS.f[mm];
				KS.f[pz] = KS.f[mz];
				KS.f[pp] = KS.f[mp];
				COLL::computeDensityAndVelocity(KS);
				break;
			case GEO_SYM_RIGHT:
				KS.f[mm] = KS.f[pm];
				KS.f[mz] = KS.f[pz];
				KS.f[mp] = KS.f[pp];
				COLL::computeDensityAndVelocity(KS);
				break;
			case GEO_SYM_BOTTOM:
				KS.f[mp] = KS.f[mm];
				KS.f[zp] = KS.f[zm];
				KS.f[pp] = KS.f[pm];
				COLL::computeDensityAndVelocity(KS);
				break;
			case GEO_SYM_TOP:
				KS.f[mm] = KS.f[mp];
				KS.f[zm] = KS.f[zp];
				KS.f[pm] = KS.f[pp];
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
		return isFluid(mapgi) || isPeriodic(mapgi) || mapgi == GEO_OUTFLOW_RIGHT || mapgi == GEO_OUTFLOW_RIGHT_INTERP;
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
