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
			SD.df(df_cur, dir9::mm, x, y, z) = KS.f[dir9::pp];
			SD.df(df_cur, dir9::mz, x, y, z) = KS.f[dir9::pz];
			SD.df(df_cur, dir9::mp, x, y, z) = KS.f[dir9::pm];
			SD.df(df_cur, dir9::zm, x, y, z) = KS.f[dir9::zp];
			SD.df(df_cur, dir9::zz, x, y, z) = KS.f[dir9::zz];
			SD.df(df_cur, dir9::zp, x, y, z) = KS.f[dir9::zm];
			SD.df(df_cur, dir9::pm, x, y, z) = KS.f[dir9::mp];
			SD.df(df_cur, dir9::pz, x, y, z) = KS.f[dir9::mz];
			SD.df(df_cur, dir9::pp, x, y, z) = KS.f[dir9::mm];
		}
		else {
			// write to the neighboring lattice sites, same DF direction
			SD.df(df_cur, dir9::pp, xp, yp, z) = KS.f[dir9::pp];
			SD.df(df_cur, dir9::pz, xp, y, z) = KS.f[dir9::pz];
			SD.df(df_cur, dir9::pm, xp, ym, z) = KS.f[dir9::pm];
			SD.df(df_cur, dir9::zp, x, yp, z) = KS.f[dir9::zp];
			SD.df(df_cur, dir9::zz, x, y, z) = KS.f[dir9::zz];
			SD.df(df_cur, dir9::zm, x, ym, z) = KS.f[dir9::zm];
			SD.df(df_cur, dir9::mp, xm, yp, z) = KS.f[dir9::mp];
			SD.df(df_cur, dir9::mz, xm, y, z) = KS.f[dir9::mz];
			SD.df(df_cur, dir9::mm, xm, ym, z) = KS.f[dir9::mm];
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
			KS.f[dir9::mm] = TNL::Backend::ldg(SD.df(df_cur, dir9::pp, xp, yp, z));
			KS.f[dir9::mz] = TNL::Backend::ldg(SD.df(df_cur, dir9::pz, xp, y, z));
			KS.f[dir9::mp] = TNL::Backend::ldg(SD.df(df_cur, dir9::pm, xp, ym, z));
			KS.f[dir9::zm] = TNL::Backend::ldg(SD.df(df_cur, dir9::zp, x, yp, z));
			KS.f[dir9::zz] = TNL::Backend::ldg(SD.df(df_cur, dir9::zz, x, y, z));
			KS.f[dir9::zp] = TNL::Backend::ldg(SD.df(df_cur, dir9::zm, x, ym, z));
			KS.f[dir9::pm] = TNL::Backend::ldg(SD.df(df_cur, dir9::mp, xm, yp, z));
			KS.f[dir9::pz] = TNL::Backend::ldg(SD.df(df_cur, dir9::mz, xm, y, z));
			KS.f[dir9::pp] = TNL::Backend::ldg(SD.df(df_cur, dir9::mm, xm, ym, z));
		}
	}
};
