#pragma once

#include "lbm3d/defs.h"
#include "../defs.h"
#include "lbm_common/ciselnik.h"
#include <TNL/Backend/Macros.h>
#include "../../lbm_common/ciselnik.h"

template <typename CONFIG>
struct D3Q53_BC_All
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
		GEO_INFLOW_LEFT_PRESSURE,
		GEO_OUTFLOW_EQ,
		GEO_OUTFLOW_RIGHT,
		GEO_OUTFLOW_RIGHT_INTERP,
		GEO_PERIODIC,
		GEO_NOTHING,
		GEO_SYM_TOP,
		GEO_SYM_BOTTOM,
		GEO_SYM_LEFT,
		GEO_SYM_RIGHT,
		GEO_SYM_BACK,
		GEO_SYM_FRONT
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
	__cuda_callable__ static void preCollision(DATA& SD, LBM_KS& KS, map_t mapgi, typename LBM_KS::SG streamGrid)
	{
		if (mapgi == GEO_NOTHING) {
			// nema zadny vliv na vypocet, jen pro output
			KS.rho = 1;
			KS.vx = 0;
			KS.vy = 0;
			KS.vz = 0;
			return;
		}

		int x  = streamGrid.x(LBM_KS::NoDV);
		int y  = streamGrid.y(LBM_KS::NoDV);
		int z  = streamGrid.z(LBM_KS::NoDV);

		if (mapgi != GEO_OUTFLOW_RIGHT_INTERP)
			STREAMING::streaming(SD, KS, streamGrid);

		// boundary conditions
		switch (mapgi) {
			case GEO_INFLOW:
				SD.inflow(KS, x, y, z);
				KS.rho = 1;
				COLL::setEquilibrium(KS);
				break;
			case GEO_INFLOW_LEFT:
				SD.inflow(KS, x, y, z);
			case GEO_INFLOW_LEFT_PRESSURE:
				for(int i = 0; i < LBM_KS::ONE_SIZE; i++){
					streamGrid.x(i)+=3;
				}
            	STREAMING::streaming(SD,KS,streamGrid);
				for(int i = 0; i < LBM_KS::ONE_SIZE; i++){
					streamGrid.x(i)-=3;
				}
				COLL::computeDensityAndVelocity(KS);
				SD.inflow(KS, x, y, z);
				COLL::setEquilibrium(KS);
			case GEO_OUTFLOW_EQ:
				COLL::computeDensityAndVelocity(KS);
				KS.rho = 1;
				COLL::setEquilibrium(KS);
				break;
			case GEO_OUTFLOW_RIGHT:
				COLL::computeDensityAndVelocity(KS);
				KS.rho = 1;
				COLL::setEquilibriumDecomposition(KS, 1);
				COLL::computeDensityAndVelocity(KS);
				break;
			case GEO_OUTFLOW_RIGHT_INTERP:
				STREAMING::streamingInterpRight(SD, KS, streamGrid);
				COLL::computeDensityAndVelocity(KS);
				COLL::setEquilibriumDecomposition(KS, 1);
				COLL::computeDensityAndVelocity(KS);
				break;
			case GEO_WALL:
				// does not affect the computation, only the output
				KS.rho = 1;
				KS.vx = 0;
				KS.vy = 0;
				KS.vz = 0;
				// collision step: bounce-back
				#if defined(__CUDA_ARCH__) && defined(UNROLL)
				#pragma unroll 2
				#endif
				for(int id = 0; id < LBM_KS::Qhalf; id++){
					TNL::swap(KS.f[id], KS.f[KS.flip_id(id)]);
				}
				break;

			case GEO_SYM_BOTTOM: // z
				#if defined(__CUDA_ARCH__) && defined(UNROLL)
				#pragma unroll 2
				#endif
				for(int id = 0; id < LBM_KS::Q; id++){
					Coord c = LBM_KS::id_to_dv(id);
					if(c.z < 0){
						KS.f[id] = KS.f[KS.dv_to_id(c.x,c.y,-c.z)];
					}
				}
				COLL::computeDensityAndVelocity(KS);
				break;
			case GEO_SYM_TOP: //z
				#if defined(__CUDA_ARCH__) && defined(UNROLL)
				#pragma unroll 2
				#endif
				for(int id = 0; id < LBM_KS::Q; id++){
					Coord c = LBM_KS::id_to_dv(id);
					if(c.z > 0){
						KS.f[id] = KS.f[KS.dv_to_id(c.x,c.y,-c.z)];
					}
				}
				COLL::computeDensityAndVelocity(KS);
				break;
			case GEO_SYM_BACK: // y
				#if defined(__CUDA_ARCH__) && defined(UNROLL)
				#pragma unroll 2
				#endif
				for(int id = 0; id < LBM_KS::Q; id++){
					Coord c = LBM_KS::id_to_dv(id);
					if(c.y > 0){
						KS.f[id] = KS.f[KS.dv_to_id(c.x,-c.y,c.z)];
					}
				}
				COLL::computeDensityAndVelocity(KS);
				break;
			case GEO_SYM_FRONT: // y
				#if defined(__CUDA_ARCH__) && defined(UNROLL)
				#pragma unroll 2
				#endif
				for(int id = 0; id < LBM_KS::Q; id++){
					Coord c = LBM_KS::id_to_dv(id);
					if(c.y < 0){
						KS.f[id] = KS.f[KS.dv_to_id(c.x,-c.y,c.z)];
					}
				}
				COLL::computeDensityAndVelocity(KS);
				break;
			case GEO_SYM_LEFT: // x
				#if defined(__CUDA_ARCH__) && defined(UNROLL)
				#pragma unroll 2
				#endif
				for(int id = 0; id < LBM_KS::Q; id++){
					Coord c = LBM_KS::id_to_dv(id);
					if(c.x < 0){
						KS.f[id] = KS.f[KS.dv_to_id(-c.x,c.y,c.z)];
					}
				}
				COLL::computeDensityAndVelocity(KS);
				break;
			case GEO_SYM_RIGHT: // x
				#if defined(__CUDA_ARCH__) && defined(UNROLL)
				#pragma unroll 2
				#endif
				for(int id = 0; id < LBM_KS::Q; id++){
					Coord c = LBM_KS::id_to_dv(id);
					if(c.x > 0){
						KS.f[id] = KS.f[KS.dv_to_id(-c.x,c.y,c.z)];
					}
				}
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
		return isFluid(mapgi) || isPeriodic(mapgi) || mapgi == GEO_OUTFLOW_RIGHT || mapgi == GEO_OUTFLOW_RIGHT_INTERP || mapgi == GEO_INFLOW_LEFT;
	}

	template <typename LBM_KS>
	__cuda_callable__ static void
	postCollision(DATA& SD, LBM_KS& KS, map_t mapgi, typename LBM_KS::SG streamGrid)
	{
		if (mapgi == GEO_NOTHING)
			return;

		STREAMING::postCollisionStreaming(SD, KS, streamGrid);
	}
};
