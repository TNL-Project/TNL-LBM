#pragma once

#include "lbm3d/defs.h"

// pull-scheme
template <typename TRAITS>
struct D2Q9_STREAMING
{
	using idx = typename TRAITS::idx;
	using dreal = typename TRAITS::dreal;

	template <typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void
	postCollisionStreaming(LBM_DATA& SD, LBM_KS& KS, typename LBM_KS::StreamGrid streamGrid)
	{
		// no streaming actually, write to the (x,y,z) site
		for (int i = 0; i < 9; i++)
			SD.df(df_out, i, streamGrid.x[LBM_KS::NoDV], streamGrid.y[LBM_KS::NoDV], streamGrid.z[LBM_KS::NoDV]) = KS.f[i];
	}

	template <typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void
	streaming(uint8_t type, LBM_DATA& SD, LBM_KS& KS, typename LBM_KS::StreamGrid streamGrid)
	{
		for(int id = 0; id < LBM_KS::Q; id++){
			const int i = LBM_KS::id_to_coords(id).x;
			const int j = LBM_KS::id_to_coords(id).y;
			const int k = LBM_KS::id_to_coords(id).z;
			KS.f[KS.coords_to_id(i,j,k)] = TNL::Backend::ldg(SD.df(type,id,streamGrid.x[KS.flip_coord(i)],streamGrid.y[KS.flip_coord(j)],streamGrid.z[KS.flip_coord(k)]));
		}
	}

	template <typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void streaming(LBM_DATA& SD, LBM_KS& KS, typename LBM_KS::StreamGrid streamGrid)
	{
		streaming(df_cur, SD, KS, streamGrid);
	}

	// streaming with bounce-back rule applied
	template <typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void
	streamingBounceBack(LBM_DATA& SD, LBM_KS& KS, typename LBM_KS::StreamGrid streamGrid)
	{
		for(int id = 0; id < LBM_KS::Q; id++){
			const int i = LBM_KS::id_to_coords(id).x;
			const int j = LBM_KS::id_to_coords(id).y;
			const int k = LBM_KS::id_to_coords(id).z;
			KS.f[KS.coords_to_id(i,j,k)] = TNL::Backend::ldg(SD.df(df_cur, id, streamGrid.x[i],streamGrid.y[j],streamGrid.z[k]));
		}
	}

	template <typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void
	streamingInterpRight(LBM_DATA& SD, LBM_KS& KS, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm_unused, idx z, idx zp_unused)
	{
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
