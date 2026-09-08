#pragma once

#include "lbm3d/defs.h"

// A-B push scheme: two DF arrays. The post-collision populations are written
// straight to their target sites in the other array, so the next launch reads
// them at its own site (the post-stream layout, like the A-A even sub-step,
// but with separate arrays and no direction twist).
template <typename TRAITS>
struct D3Q7_STREAMING_AB_PUSH
{
	static constexpr int DFMAX = 2;
	// DF slot that holds the freshly written field after a kernel launch
	static constexpr std::uint8_t output_df = df_out;

	using idx = typename TRAITS::idx;
	using dreal = typename TRAITS::dreal;

	// the streaming step itself: write the post-collision populations to the
	// target sites in the other array
	template <typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void
	postCollisionStreaming(LBM_DATA& SD, LBM_KS& KS, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm, idx z, idx zp)
	{
		SD.df(df_out, mzz, xm, y, z) = KS.f[mzz];
		SD.df(df_out, zmz, x, ym, z) = KS.f[zmz];
		SD.df(df_out, zzm, x, y, zm) = KS.f[zzm];
		SD.df(df_out, zzz, x, y, z) = KS.f[zzz];
		SD.df(df_out, zzp, x, y, zp) = KS.f[zzp];
		SD.df(df_out, zpz, x, yp, z) = KS.f[zpz];
		SD.df(df_out, pzz, xp, y, z) = KS.f[pzz];
	}

	// the post-stream populations sit at their own site: identity read
	template <typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void streaming(LBM_DATA& SD, LBM_KS& KS, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm, idx z, idx zp)
	{
		for (int i = 0; i < 7; i++)
			KS.f[i] = SD.df(df_cur, i, x, y, z);
	}
};

template <typename TRAITS>
inline constexpr bool is_AB_PUSH_v<D3Q7_STREAMING_AB_PUSH<TRAITS>> = true;
