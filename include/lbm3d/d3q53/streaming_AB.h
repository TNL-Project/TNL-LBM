#pragma once

#include "lbm3d/defs.h"
#include "../defs.h"
#include <TNL/Backend/Macros.h>

// pull-scheme
template <typename TRAITS>
struct D3Q53_STREAMING
{
	using idx = typename TRAITS::idx;
	using dreal = typename TRAITS::dreal;

	template <typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void
	postCollisionStreaming(LBM_DATA& SD, LBM_KS& KS, typename LBM_KS::SG streamGrid)
	{
		// no streaming actually, write to the (x,y,z) site
		#ifdef __CUDA_ARCH__
		#pragma unroll
		#endif
		for (int i = 0; i < LBM_KS::Q; i++)
			SD.df(df_out, i, streamGrid.x(LBM_KS::NoDV), streamGrid.y(LBM_KS::NoDV), streamGrid.z(LBM_KS::NoDV)) = KS.f[i];
	}

	template <typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void
	streaming(uint8_t type, LBM_DATA& SD, LBM_KS& KS, typename LBM_KS::SG streamGrid)
	{
		#ifdef __CUDA_ARCH__
		#pragma unroll
		#endif
		for(int id = 0; id < LBM_KS::Q; id++){
			const Coord c = LBM_KS::id_to_coords(id);
			const int flip_id = LBM_KS::flip_id(id);
			KS.f[flip_id] = TNL::Backend::ldg(SD.df(type,flip_id,streamGrid.x(c.x),streamGrid.y(c.y),streamGrid.z(c.z)));
		}
	}

	template <typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void streaming(LBM_DATA& SD, LBM_KS& KS, typename LBM_KS::SG streamGrid)
	{
		streaming(df_cur, SD, KS, streamGrid);
	}

	// streaming with bounce-back rule applied
	template <typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void streamingBounceBack(LBM_DATA& SD, LBM_KS& KS,typename LBM_KS::SG streamGrid)
	{
		#ifdef __CUDA_ARCH__
		#pragma unroll
		#endif
		for(int id = 0; id < LBM_KS::Q; id++){
			const Coord c = LBM_KS::id_to_coords(id);
			const int flip_id = LBM_KS::flip_id(id);
			KS.f[id] = TNL::Backend::ldg(SD.df(df_cur,flip_id,streamGrid.x(c.x),streamGrid.y(c.y),streamGrid.z(c.z)));
		}
	}

	template < typename LBM_DATA, typename LBM_KS  >
	CUDA_HOSTDEV static void interpolatedStreamingOutflowRightX(LBM_DATA &SD, LBM_KS &KS, StreamGrid<int, LBM_KS::NoDV> streamIds)
	{
		for(int id = 0; id < LBM_KS::Q; id++){
			const int i = LBM_KS::id_to_coords(id).x;
			const int j = LBM_KS::id_to_coords(id).y;
			const dreal interCoeffX = MIN(LBM_KS::cs - KS.vx, LBM_KS::cs);
			if(i >= LBM_KS::NoDV){
				KS.f[id] = SD.cdf(id,streamIds.x(KS.flip_coord(i)),streamIds.y(KS.flip_coord(j)));
			}
			else{
				KS.f[id] = (  interCoeffX)*SD.cdf(id,streamIds.x(LBM_KS::NoDV-1),streamIds.y(KS.flip_coord(j))) +(1-interCoeffX)*SD.cdf(id,streamIds.x(LBM_KS::NoDV),streamIds.y(KS.flip_coord(j)));
			}
		}
	}


};
