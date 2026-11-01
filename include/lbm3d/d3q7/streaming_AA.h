#pragma once

#include "lbm3d/defs.h"

// A-A pattern
template <typename TRAITS>
struct D3Q7_STREAMING_AA
{
	static constexpr int DFMAX = 1;
	// DF slot that holds the freshly written field after a kernel launch
	static constexpr std::uint8_t output_df = df_cur;

	using idx = typename TRAITS::idx;
	using dreal = typename TRAITS::dreal;

	template <typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void
	postCollisionStreaming(LBM_DATA& SD, LBM_KS& KS, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm, idx z, idx zp)
	{
		if (SD.even_iter) {
			// write to the same lattice site, but the opposite DF direction
			SD.df(df_cur, dir27::mzz, x, y, z) = KS.f[dir27::pzz];
			SD.df(df_cur, dir27::zmz, x, y, z) = KS.f[dir27::zpz];
			SD.df(df_cur, dir27::zzm, x, y, z) = KS.f[dir27::zzp];
			SD.df(df_cur, dir27::zzz, x, y, z) = KS.f[dir27::zzz];
			SD.df(df_cur, dir27::zzp, x, y, z) = KS.f[dir27::zzm];
			SD.df(df_cur, dir27::zpz, x, y, z) = KS.f[dir27::zmz];
			SD.df(df_cur, dir27::pzz, x, y, z) = KS.f[dir27::mzz];
		}
		else {
			// write to the neighboring lattice sites, same DF direction
			SD.df(df_cur, dir27::pzz, xp, y, z) = KS.f[dir27::pzz];
			SD.df(df_cur, dir27::zpz, x, yp, z) = KS.f[dir27::zpz];
			SD.df(df_cur, dir27::zzp, x, y, zp) = KS.f[dir27::zzp];
			SD.df(df_cur, dir27::zzz, x, y, z) = KS.f[dir27::zzz];
			SD.df(df_cur, dir27::zzm, x, y, zm) = KS.f[dir27::zzm];
			SD.df(df_cur, dir27::zmz, x, ym, z) = KS.f[dir27::zmz];
			SD.df(df_cur, dir27::mzz, xm, y, z) = KS.f[dir27::mzz];
		}
	}

	template <typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void streaming(LBM_DATA& SD, LBM_KS& KS, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm, idx z, idx zp)
	{
		if (SD.even_iter) {
			// read from the same lattice site, same DF direction
			for (int i = 0; i < 7; i++)
				KS.f[i] = SD.df(df_cur, i, x, y, z);
		}
		else {
			// read from the neighboring lattice sites, but the opposite DF direction
			KS.f[dir27::mzz] = SD.df(df_cur, dir27::pzz, xp, y, z);
			KS.f[dir27::zmz] = SD.df(df_cur, dir27::zpz, x, yp, z);
			KS.f[dir27::zzm] = SD.df(df_cur, dir27::zzp, x, y, zp);
			KS.f[dir27::zzz] = SD.df(df_cur, dir27::zzz, x, y, z);
			KS.f[dir27::zzp] = SD.df(df_cur, dir27::zzm, x, y, zm);
			KS.f[dir27::zpz] = SD.df(df_cur, dir27::zmz, x, ym, z);
			KS.f[dir27::pzz] = SD.df(df_cur, dir27::mzz, xm, y, z);
		}
	}
};

template <typename TRAITS>
inline constexpr bool is_AA_v<D3Q7_STREAMING_AA<TRAITS>> = true;
template <typename TRAITS>
inline constexpr bool twisted_layout_v<D3Q7_STREAMING_AA<TRAITS>> = true;
template <typename TRAITS>
inline constexpr bool requires_ghost_layer_v<D3Q7_STREAMING_AA<TRAITS>> = true;
