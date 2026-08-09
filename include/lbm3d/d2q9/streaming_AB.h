#pragma once

#include "lbm3d/defs.h"
#include "lbm_common/rounding.h"

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
		KS.f[dir9::mm] = TNL::Backend::ldg(SD.df(type, dir9::mm, xp, yp, z));
		KS.f[dir9::mz] = TNL::Backend::ldg(SD.df(type, dir9::mz, xp, y, z));
		KS.f[dir9::mp] = TNL::Backend::ldg(SD.df(type, dir9::mp, xp, ym, z));
		KS.f[dir9::zm] = TNL::Backend::ldg(SD.df(type, dir9::zm, x, yp, z));
		KS.f[dir9::zz] = TNL::Backend::ldg(SD.df(type, dir9::zz, x, y, z));
		KS.f[dir9::zp] = TNL::Backend::ldg(SD.df(type, dir9::zp, x, ym, z));
		KS.f[dir9::pm] = TNL::Backend::ldg(SD.df(type, dir9::pm, xm, yp, z));
		KS.f[dir9::pz] = TNL::Backend::ldg(SD.df(type, dir9::pz, xm, y, z));
		KS.f[dir9::pp] = TNL::Backend::ldg(SD.df(type, dir9::pp, xm, ym, z));
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
		KS.f[dir9::pp] = TNL::Backend::ldg(SD.df(df_cur, dir9::mm, xp, yp, z));
		KS.f[dir9::pz] = TNL::Backend::ldg(SD.df(df_cur, dir9::mz, xp, y, z));
		KS.f[dir9::pm] = TNL::Backend::ldg(SD.df(df_cur, dir9::mp, xp, ym, z));
		KS.f[dir9::zp] = TNL::Backend::ldg(SD.df(df_cur, dir9::zm, x, yp, z));
		KS.f[dir9::zz] = TNL::Backend::ldg(SD.df(df_cur, dir9::zz, x, y, z));
		KS.f[dir9::zm] = TNL::Backend::ldg(SD.df(df_cur, dir9::zp, x, ym, z));
		KS.f[dir9::mp] = TNL::Backend::ldg(SD.df(df_cur, dir9::pm, xm, yp, z));
		KS.f[dir9::mz] = TNL::Backend::ldg(SD.df(df_cur, dir9::pz, xm, y, z));
		KS.f[dir9::mm] = TNL::Backend::ldg(SD.df(df_cur, dir9::pp, xm, ym, z));
	}

	// outflow pass gathers: the outflow cell takes the pulled state of its
	// upstream neighbor column xm from df_cur (finalized by the previous
	// launch, no race against the df_out writes of the current one)
	template <typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void
	streamingOutflowRight(LBM_DATA& SD, LBM_KS& KS, idx xm, idx x_unused, idx xp_unused, idx ym, idx y, idx yp, idx zm_unused, idx z, idx zp_unused)
	{
		KS.f[dir9::mm] = TNL::Backend::ldg(SD.df(df_cur, dir9::mm, xm, yp, z));
		KS.f[dir9::mz] = TNL::Backend::ldg(SD.df(df_cur, dir9::mz, xm, y, z));
		KS.f[dir9::mp] = TNL::Backend::ldg(SD.df(df_cur, dir9::mp, xm, ym, z));
		KS.f[dir9::zm] = TNL::Backend::ldg(SD.df(df_cur, dir9::zm, xm, yp, z));
		KS.f[dir9::zz] = TNL::Backend::ldg(SD.df(df_cur, dir9::zz, xm, y, z));
		KS.f[dir9::zp] = TNL::Backend::ldg(SD.df(df_cur, dir9::zp, xm, ym, z));
		KS.f[dir9::pm] = TNL::Backend::ldg(SD.df(df_cur, dir9::pm, xm, yp, z));
		KS.f[dir9::pz] = TNL::Backend::ldg(SD.df(df_cur, dir9::pz, xm, y, z));
		KS.f[dir9::pp] = TNL::Backend::ldg(SD.df(df_cur, dir9::pp, xm, ym, z));
	}

	// interpolated outflow (Geier 2015): the m-family blends postcoll_{n-1}
	// from column xm with the outflow cell's own postcoll (column x), the
	// z- and p-families come straight from df_cur like in ordinary streaming
	template <typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void
	streamingOutflowInterpRight(LBM_DATA& SD, LBM_KS& KS, idx xm, idx x, idx xp_unused, idx ym, idx y, idx yp, idx zm_unused, idx z, idx zp_unused)
	{
		// NOTE: velocity is neglected (for the case velocity << speed of sound)
		constexpr dreal SpeedOfSound = 0.5773502691896257;
		KS.f[dir9::mm] = lbm_fma_rn(
			SpeedOfSound,
			TNL::Backend::ldg(SD.df(df_cur, dir9::mm, xm, yp, z)),
			(1 - SpeedOfSound) * TNL::Backend::ldg(SD.df(df_cur, dir9::mm, x, yp, z))
		);
		KS.f[dir9::mz] = lbm_fma_rn(
			SpeedOfSound,
			TNL::Backend::ldg(SD.df(df_cur, dir9::mz, xm, y, z)),
			(1 - SpeedOfSound) * TNL::Backend::ldg(SD.df(df_cur, dir9::mz, x, y, z))
		);
		KS.f[dir9::mp] = lbm_fma_rn(
			SpeedOfSound,
			TNL::Backend::ldg(SD.df(df_cur, dir9::mp, xm, ym, z)),
			(1 - SpeedOfSound) * TNL::Backend::ldg(SD.df(df_cur, dir9::mp, x, ym, z))
		);
		KS.f[dir9::zm] = TNL::Backend::ldg(SD.df(df_cur, dir9::zm, x, yp, z));
		KS.f[dir9::zz] = TNL::Backend::ldg(SD.df(df_cur, dir9::zz, x, y, z));
		KS.f[dir9::zp] = TNL::Backend::ldg(SD.df(df_cur, dir9::zp, x, ym, z));
		KS.f[dir9::pm] = TNL::Backend::ldg(SD.df(df_cur, dir9::pm, xm, yp, z));
		KS.f[dir9::pz] = TNL::Backend::ldg(SD.df(df_cur, dir9::pz, xm, y, z));
		KS.f[dir9::pp] = TNL::Backend::ldg(SD.df(df_cur, dir9::pp, xm, ym, z));
	}
};
