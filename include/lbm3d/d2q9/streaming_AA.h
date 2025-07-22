#pragma once

#include "lbm3d/defs.h"

// A-A pattern
template <typename TRAITS>
struct D2Q9_STREAMING
{
	using idx = typename TRAITS::idx;
	using dreal = typename TRAITS::dreal;

	template <typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void
	postCollisionStreaming(LBM_DATA& SD, LBM_KS& KS, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm_unused, idx z, idx zp_unused)
	{
		if (SD.even_iter) {
			// write to the same lattice site, but the opposite DF direction
			SD.df(df_cur, mm, x, y, z) = KS.f[pp];
			SD.df(df_cur, mz, x, y, z) = KS.f[pz];
			SD.df(df_cur, mp, x, y, z) = KS.f[pm];
			SD.df(df_cur, zm, x, y, z) = KS.f[zp];
			SD.df(df_cur, zz, x, y, z) = KS.f[zz];
			SD.df(df_cur, zp, x, y, z) = KS.f[zm];
			SD.df(df_cur, pm, x, y, z) = KS.f[mp];
			SD.df(df_cur, pz, x, y, z) = KS.f[mz];
			SD.df(df_cur, pp, x, y, z) = KS.f[mm];
		}
		else {
			// write to the neighboring lattice sites, same DF direction
			SD.df(df_cur, pp, xp, yp, z) = KS.f[pp];
			SD.df(df_cur, pz, xp, y, z) = KS.f[pz];
			SD.df(df_cur, pm, xp, ym, z) = KS.f[pm];
			SD.df(df_cur, zp, x, yp, z) = KS.f[zp];
			SD.df(df_cur, zz, x, y, z) = KS.f[zz];
			SD.df(df_cur, zm, x, ym, z) = KS.f[zm];
			SD.df(df_cur, mp, xm, yp, z) = KS.f[mp];
			SD.df(df_cur, mz, xm, y, z) = KS.f[mz];
			SD.df(df_cur, mm, xm, ym, z) = KS.f[mm];
		}
	}

	template <typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void
	streaming(LBM_DATA& SD, LBM_KS& KS, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm_unused, idx z, idx zp_unused)
	{
		if (SD.even_iter) {
			// read from the same lattice site, same DF direction
			for (int i = 0; i < 9; i++)
				KS.f[i] = TNL::Backend::ldg(SD.df(df_cur, i, x, y, z));
		}
		else {
			// read from the neighboring lattice sites, but the opposite DF direction
			KS.f[mm] = TNL::Backend::ldg(SD.df(df_cur, pp, xp, yp, z));
			KS.f[mz] = TNL::Backend::ldg(SD.df(df_cur, pz, xp, y, z));
			KS.f[mp] = TNL::Backend::ldg(SD.df(df_cur, pm, xp, ym, z));
			KS.f[zm] = TNL::Backend::ldg(SD.df(df_cur, zp, x, yp, z));
			KS.f[zz] = TNL::Backend::ldg(SD.df(df_cur, zz, x, y, z));
			KS.f[zp] = TNL::Backend::ldg(SD.df(df_cur, zm, x, ym, z));
			KS.f[pm] = TNL::Backend::ldg(SD.df(df_cur, mp, xm, yp, z));
			KS.f[pz] = TNL::Backend::ldg(SD.df(df_cur, mz, xm, y, z));
			KS.f[pp] = TNL::Backend::ldg(SD.df(df_cur, mm, xm, ym, z));
		}
	}
};
