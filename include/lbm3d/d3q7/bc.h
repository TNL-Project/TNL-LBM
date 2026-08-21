#pragma once

#include "lbm3d/defs.h"
#include "lbm_common/ciselnik.h"

template <typename CONFIG>
struct D3Q7_BC_All
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
		GEO_WALL_BODY,
		GEO_SOLID,
		GEO_TRANSFER_FS,
		GEO_TRANSFER_SF,
		GEO_TRANSFER_SW,
		GEO_INFLOW,
		GEO_OUTFLOW_RIGHT,
		GEO_NOTHING,
		GEO_OUTFLOW_PE,
		GEO_SYMMETRY
	};

	__cuda_callable__ static bool isFluid(map_t mapgi)
	{
		return mapgi == GEO_FLUID;
	}

	__cuda_callable__ static bool isWall(map_t mapgi)
	{
		return mapgi == GEO_WALL;
	}

	// d3q7 uses the legacy fused outflow path (see d2q9 bc.h for the pass)
	static constexpr bool use_outflow_pass = false;

	__cuda_callable__ static bool isSolid(map_t mapgi)
	{
		return mapgi == GEO_SOLID;
	}

	__cuda_callable__ static bool isSolidPhase(map_t mapgi)
	{
		return mapgi == GEO_SOLID || mapgi == GEO_TRANSFER_SF || mapgi == GEO_TRANSFER_SW;
	}

	template <typename LBM_KS>
	__cuda_callable__ static void preCollision(DATA& SD, LBM_KS& KS, map_t mapgi, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm, idx z, idx zp)
	{
		if (mapgi == GEO_NOTHING) {
			// does not affect the computation, only the output
			KS.phi = 0;
			return;
		}

		// modify pull location for streaming
		if (mapgi == GEO_OUTFLOW_RIGHT)
			xp = x = xm;

		STREAMING::streaming(SD, KS, xm, x, xp, ym, y, yp, zm, z, zp);

		// boundary conditions
		switch (mapgi) {
			case GEO_INFLOW:
				SD.inflow(KS, x, y, z);
				COLL::setEquilibrium(KS);
				break;
			case GEO_OUTFLOW_PE:
				STREAMING::streaming(SD, KS, xm - 1, xm, x, ym, y, yp, zm, z, zp);
				COLL::computeDensityAndVelocity(KS);
				COLL::setEquilibrium(KS);
				break;
			case GEO_OUTFLOW_RIGHT:
				STREAMING::streaming(SD, KS, xm, x, xm, ym, y, yp, zm, z, zp);
				COLL::computeDensityAndVelocity(KS);
				break;
			case GEO_WALL:
				COLL::computeDensityAndVelocity(KS);
				// collision step: bounce-back
				TNL::swap(KS.f[mzz], KS.f[pzz]);
				TNL::swap(KS.f[zmz], KS.f[zpz]);
				TNL::swap(KS.f[zzm], KS.f[zzp]);
				break;
			case GEO_WALL_BODY:
				COLL::computeDensityAndVelocity(KS);
				// collision step: bounce-back
				TNL::swap(KS.f[mzz], KS.f[pzz]);
				TNL::swap(KS.f[zmz], KS.f[zpz]);
				TNL::swap(KS.f[zzm], KS.f[zzp]);
				// anti-bounce-back (recovers zero gradient across the wall boundary, see Kruger section 8.5.2.1)
				for (int q = 0; q < 7; q++) {
					if (q == zzz)
						KS.f[q] = -KS.f[q] + 2 * n1o4 * KS.phi;
					else
						KS.f[q] = -KS.f[q] + 2 * n1o8 * KS.phi;
				}
				// TODO: Kruger's eq (8.54) includes concentration imposed on the wall - does it diffusively propagate into the domain? -- Yes DH2022
				break;

			case GEO_SYMMETRY:
				// Each cardinal direction going through a GEO_NOTHING ghost side
				// is filled from its mirror partner (no diagonals in D3Q7).
				if (SD.map(xm, y, z) == GEO_NOTHING)
					KS.f[pzz] = KS.f[mzz];
				if (SD.map(xp, y, z) == GEO_NOTHING)
					KS.f[mzz] = KS.f[pzz];
				if (SD.map(x, ym, z) == GEO_NOTHING)
					KS.f[zpz] = KS.f[zmz];
				if (SD.map(x, yp, z) == GEO_NOTHING)
					KS.f[zmz] = KS.f[zpz];
				if (SD.map(x, y, zm) == GEO_NOTHING)
					KS.f[zzp] = KS.f[zzm];
				if (SD.map(x, y, zp) == GEO_NOTHING)
					KS.f[zzm] = KS.f[zzp];
				COLL::computeDensityAndVelocity(KS);
				break;

			case GEO_TRANSFER_FS:
			case GEO_TRANSFER_SF:
				{
					dreal tmp[7];
					for (int q = 0; q < 7; q++)
						tmp[q] = 0;
					for (int q = 0; q < 7; q++) {
						tmp[pzz] += SD.df(df_cur, q, xp, y, z);
						tmp[mzz] += SD.df(df_cur, q, xm, y, z);
						tmp[zpz] += SD.df(df_cur, q, x, yp, z);
						tmp[zmz] += SD.df(df_cur, q, x, ym, z);
						tmp[zzp] += SD.df(df_cur, q, x, y, zp);
						tmp[zzm] += SD.df(df_cur, q, x, y, zm);
					}

					if (SD.phiTransferDirection(pzz, x, y, z))
						KS.f[mzz] = SD.df(df_cur, pzz, x, y, z) + SD.phiTransferCoefficient * (tmp[pzz] - SD.macro(0, x, y, z));
					if (SD.phiTransferDirection(zpz, x, y, z))
						KS.f[zmz] = SD.df(df_cur, zpz, x, y, z) + SD.phiTransferCoefficient * (tmp[zpz] - SD.macro(0, x, y, z));
					if (SD.phiTransferDirection(zzp, x, y, z))
						KS.f[zzm] = SD.df(df_cur, zzp, x, y, z) + SD.phiTransferCoefficient * (tmp[zzp] - SD.macro(0, x, y, z));
					if (SD.phiTransferDirection(mzz, x, y, z))
						KS.f[pzz] = SD.df(df_cur, mzz, x, y, z) + SD.phiTransferCoefficient * (tmp[mzz] - SD.macro(0, x, y, z));
					if (SD.phiTransferDirection(zmz, x, y, z))
						KS.f[zpz] = SD.df(df_cur, zmz, x, y, z) + SD.phiTransferCoefficient * (tmp[zmz] - SD.macro(0, x, y, z));
					if (SD.phiTransferDirection(zzm, x, y, z))
						KS.f[zzp] = SD.df(df_cur, zzm, x, y, z) + SD.phiTransferCoefficient * (tmp[zzm] - SD.macro(0, x, y, z));
					COLL::computeDensityAndVelocity(KS);
					break;
				}

			case GEO_TRANSFER_SW:
				{
					if (SD.phiTransferDirection(pzz, x, y, z))
						KS.f[mzz] = SD.df(df_cur, pzz, x, y, z);
					if (SD.phiTransferDirection(zpz, x, y, z))
						KS.f[zmz] = SD.df(df_cur, zpz, x, y, z);
					if (SD.phiTransferDirection(zzp, x, y, z))
						KS.f[zzm] = SD.df(df_cur, zzp, x, y, z);
					if (SD.phiTransferDirection(mzz, x, y, z))
						KS.f[pzz] = SD.df(df_cur, mzz, x, y, z);
					if (SD.phiTransferDirection(zmz, x, y, z))
						KS.f[zpz] = SD.df(df_cur, zmz, x, y, z);
					if (SD.phiTransferDirection(zzm, x, y, z))
						KS.f[zzp] = SD.df(df_cur, zzm, x, y, z);
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
		return isFluid(mapgi) || isSolid(mapgi) || mapgi == GEO_TRANSFER_SF || mapgi == GEO_TRANSFER_FS || mapgi == GEO_TRANSFER_SW
			|| mapgi == GEO_OUTFLOW_RIGHT;
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
