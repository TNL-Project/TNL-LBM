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
	postCollisionStreaming(LBM_DATA& SD, LBM_KS& KS, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm_unused, idx z, idx zp_unused)
	{
		// no streaming actually, write to the (x,y,z) site
		for (int i = 0; i < 9; i++)
			SD.df(df_out, i, x, y, z) = KS.f[i];
	}

	template <typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void
	streaming(uint8_t type, LBM_DATA& SD, LBM_KS& KS, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm_unused, idx z, idx zp_unused)
	{
		KS.f[mm] = TNL::Backend::ldg(SD.df(type, mm, xp, yp, z));
		KS.f[mz] = TNL::Backend::ldg(SD.df(type, mz, xp, y, z));
		KS.f[mp] = TNL::Backend::ldg(SD.df(type, mp, xp, ym, z));
		KS.f[zm] = TNL::Backend::ldg(SD.df(type, zm, x, yp, z));
		KS.f[zz] = TNL::Backend::ldg(SD.df(type, zz, x, y, z));
		KS.f[zp] = TNL::Backend::ldg(SD.df(type, zp, x, ym, z));
		KS.f[pm] = TNL::Backend::ldg(SD.df(type, pm, xm, yp, z));
		KS.f[pz] = TNL::Backend::ldg(SD.df(type, pz, xm, y, z));
		KS.f[pp] = TNL::Backend::ldg(SD.df(type, pp, xm, ym, z));
	}

	template <typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void streaming(LBM_DATA& SD, LBM_KS& KS, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm, idx z, idx zp)
	{
		streaming(df_cur, SD, KS, xm, x, xp, ym, y, yp, zm, z, zp);
	}

	// streaming with bounce-back rule applied
	template <typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void
	streamingBounceBack(LBM_DATA& SD, LBM_KS& KS, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm_unused, idx z, idx zp_unused)
	{
		KS.f[pp] = TNL::Backend::ldg(SD.df(df_cur, mm, xp, yp, z));
		KS.f[pz] = TNL::Backend::ldg(SD.df(df_cur, mz, xp, y, z));
		KS.f[pm] = TNL::Backend::ldg(SD.df(df_cur, mp, xp, ym, z));
		KS.f[zp] = TNL::Backend::ldg(SD.df(df_cur, zm, x, yp, z));
		KS.f[zz] = TNL::Backend::ldg(SD.df(df_cur, zz, x, y, z));
		KS.f[zm] = TNL::Backend::ldg(SD.df(df_cur, zp, x, ym, z));
		KS.f[mp] = TNL::Backend::ldg(SD.df(df_cur, pm, xm, yp, z));
		KS.f[mz] = TNL::Backend::ldg(SD.df(df_cur, pz, xm, y, z));
		KS.f[mm] = TNL::Backend::ldg(SD.df(df_cur, pp, xm, ym, z));
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
