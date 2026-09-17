#pragma once

#include "lbm3d/defs.h"

// pull-scheme
template <typename TRAITS>
struct D3Q7_STREAMING_AB_PULL
{
	static constexpr int DFMAX = 2;
	// DF slot that holds the freshly written field after a kernel launch
	static constexpr std::uint8_t output_df = df_out;

	using idx = typename TRAITS::idx;
	using dreal = typename TRAITS::dreal;

	template <typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void
	postCollisionStreaming(LBM_DATA& SD, LBM_KS& KS, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm, idx z, idx zp)
	{
		// no streaming actually, write to the (x,y,z) site
		for (int i = 0; i < 7; i++)
			SD.df(df_out, i, x, y, z) = KS.f[i];
	}

	template <typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void streaming(LBM_DATA& SD, LBM_KS& KS, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm, idx z, idx zp)
	{
		KS.f[mzz] = SD.df(df_cur, mzz, xp, y, z);
		KS.f[zmz] = SD.df(df_cur, zmz, x, yp, z);
		KS.f[zzm] = SD.df(df_cur, zzm, x, y, zp);
		KS.f[zzz] = SD.df(df_cur, zzz, x, y, z);
		KS.f[zzp] = SD.df(df_cur, zzp, x, y, zm);
		KS.f[zpz] = SD.df(df_cur, zpz, x, ym, z);
		KS.f[pzz] = SD.df(df_cur, pzz, xm, y, z);
	}
};

template <typename TRAITS>
inline constexpr bool is_AB_PULL_v<D3Q7_STREAMING_AB_PULL<TRAITS>> = true;
