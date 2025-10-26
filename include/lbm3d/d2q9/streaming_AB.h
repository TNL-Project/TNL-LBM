#pragma once

#include "lbm3d/defs.h"
#include "../defs.h"
#include <TNL/Backend/Macros.h>

// pull-scheme
template <typename TRAITS>
struct D2Q9_STREAMING
{
	using idx = typename TRAITS::idx;
	using dreal = typename TRAITS::dreal;

	template <typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void
	postCollisionStreaming(LBM_DATA& SD, LBM_KS& KS, typename LBM_KS::SG streamGrid)
	{
		// no streaming actually, write to the (x,y,z) site
		for (int i = 0; i < 9; i++)
			SD.df(df_out, i, streamGrid.x(LBM_KS::NoDV), streamGrid.y(LBM_KS::NoDV), streamGrid.z(LBM_KS::NoDV)) = KS.f[i];
	}

	template <typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void
	streaming(uint8_t type, LBM_DATA& SD, LBM_KS& KS, typename LBM_KS::SG streamGrid)
	{
		for(int id = 0; id < LBM_KS::Q; id++){
			const Coord c = LBM_KS::id_to_coords(id);
			KS.f[id] = TNL::Backend::ldg(SD.df(type,id,streamGrid.x(KS.flip_coord(c.x)),streamGrid.y(KS.flip_coord(c.y)),streamGrid.z(KS.flip_coord(c.z))));
		}
	}

	template <typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void streaming(LBM_DATA& SD, LBM_KS& KS, typename LBM_KS::SG streamGrid)
	{
		streaming(df_cur, SD, KS, streamGrid);
	}

	// streaming with bounce-back rule applied
	template <typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void
	streamingBounceBack(LBM_DATA& SD, LBM_KS& KS, typename LBM_KS::SG streamGrid)
	{
		for(int id = 0; id < LBM_KS::Q; id++){
			const Coord c = LBM_KS::id_to_coords(id);
			KS.f[id] = TNL::Backend::ldg(SD.df(df_cur, id, streamGrid.x(c.x),streamGrid.y(c.y),streamGrid.z(c.z)));
		}
	}

	template <typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void
	streamingInterpRight(LBM_DATA& SD, LBM_KS& KS, typename LBM_KS::SG streamGrid)
	{
		idx xp = streamGrid.x(2);
		idx x  = streamGrid.x(1);
		idx xm = streamGrid.x(0);
		idx yp = streamGrid.y(2);
		idx y  = streamGrid.y(1);
		idx ym = streamGrid.y(0);
		idx zp = streamGrid.z(2);
		idx z  = streamGrid.z(1);
		idx zm = streamGrid.z(0);
		// streaming: interpolation from Geier - CuLBM (2015)
		// NOTE: velocity is neglected (for the case velocity << speed of sound)
		constexpr dreal SpeedOfSound = 0.5773502691896257;
		KS.f[mm] = SpeedOfSound * SD.df(df_cur, mm, xm, yp, z) + (1 - SpeedOfSound) * SD.df(df_cur, mm, x, yp, z);
		KS.f[mz] = SpeedOfSound * SD.df(df_cur, mz, xm, y, z) + (1 - SpeedOfSound) * SD.df(df_cur, mz, x, y, z);
		KS.f[mp] = SpeedOfSound * SD.df(df_cur, mp, xm, ym, z) + (1 - SpeedOfSound) * SD.df(df_cur, mp, x, ym, z);
		KS.f[zm] = SD.df(df_cur, zm, x, yp, z);
		KS.f[zz] = SD.df(df_cur, zz, x, y, z);
		KS.f[zp] = SD.df(df_cur, zp, x, ym, z);
		KS.f[pm] = SD.df(df_cur, pm, xm, yp, z);
		KS.f[pz] = SD.df(df_cur, pz, xm, y, z);
		KS.f[pp] = SD.df(df_cur, pp, xm, ym, z);
	}
};
