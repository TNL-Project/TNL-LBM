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

	template <typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void
	streamingInterpRight(LBM_DATA& SD, LBM_KS& KS, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm_unused, idx z, idx zp_unused)
	{
		constexpr dreal SpeedOfSound = 0.5773502691896257;
		if (SD.even_iter) {
			// AB interpolation pattern (natural interpolation from streaming_AB.h)
			KS.f[dir9::mm] = SpeedOfSound * SD.df(df_cur, dir9::mm, xm, yp, z) + (1 - SpeedOfSound) * SD.df(df_cur, dir9::mm, x, yp, z);
			KS.f[dir9::mz] = SpeedOfSound * SD.df(df_cur, dir9::mz, xm, y, z) + (1 - SpeedOfSound) * SD.df(df_cur, dir9::mz, x, y, z);
			KS.f[dir9::mp] = SpeedOfSound * SD.df(df_cur, dir9::mp, xm, ym, z) + (1 - SpeedOfSound) * SD.df(df_cur, dir9::mp, x, ym, z);
			KS.f[dir9::zm] = SD.df(df_cur, dir9::zm, x, yp, z);
			KS.f[dir9::zz] = SD.df(df_cur, dir9::zz, x, y, z);
			KS.f[dir9::zp] = SD.df(df_cur, dir9::zp, x, ym, z);
			KS.f[dir9::pm] = SD.df(df_cur, dir9::pm, xm, yp, z);
			KS.f[dir9::pz] = SD.df(df_cur, dir9::pz, xm, y, z);
			KS.f[dir9::pp] = SD.df(df_cur, dir9::pp, xm, ym, z);
		}
		else {
			// AA twist transform: direction → opposite(direction), site → site + velocity(direction)
			const idx xmm = xm - 1;
			// 3 interpolated -x dirs (read opp at xmm and xm)
			KS.f[dir9::mm] = SpeedOfSound * SD.df(df_cur, dir9::pp, xmm, y, z) + (1 - SpeedOfSound) * SD.df(df_cur, dir9::pp, xm, y, z);
			KS.f[dir9::mz] = SpeedOfSound * SD.df(df_cur, dir9::pz, xmm, y, z) + (1 - SpeedOfSound) * SD.df(df_cur, dir9::pz, xm, y, z);
			KS.f[dir9::mp] = SpeedOfSound * SD.df(df_cur, dir9::pm, xmm, y, z) + (1 - SpeedOfSound) * SD.df(df_cur, dir9::pm, xm, y, z);
			// 3 zero-x dirs (read opp at (x,y,z))
			KS.f[dir9::zm] = SD.df(df_cur, dir9::zp, x, y, z);
			KS.f[dir9::zz] = SD.df(df_cur, dir9::zz, x, y, z);
			KS.f[dir9::zp] = SD.df(df_cur, dir9::zm, x, y, z);
			// 3 +x dirs (read opp at (x,y,z))
			KS.f[dir9::pm] = SD.df(df_cur, dir9::mp, x, y, z);
			KS.f[dir9::pz] = SD.df(df_cur, dir9::mz, x, y, z);
			KS.f[dir9::pp] = SD.df(df_cur, dir9::mm, x, y, z);
		}
	}
};
