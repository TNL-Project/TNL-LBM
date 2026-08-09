#pragma once

#include "lbm3d/defs.h"
#include "lbm_common/rounding.h"

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

	// Streaming for separate outflow-pass kernel (deterministic BC for A-A pattern):
	// both branches reconstruct the pre-collision populations at the translated
	// A-B pull sites of the outflow cell: postcoll_{n-1}(i) at site (xm, y - cy,i),
	// i.e. the populations that a pull stencil anchored at column xm delivers to column x.
	// The slot layouts provide this only from the finalized previous launch:
	// the required slots are owned by column xm-1/interior threads in both parities,
	// so no race-free in-launch gather exists and the pass must run before the main kernel.
	template <typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void
	streamingOutflowRight(LBM_DATA& SD, LBM_KS& KS, idx xm, idx x, idx xp_unused, idx ym, idx y, idx yp, idx zm_unused, idx z, idx zp_unused)
	{
		if (SD.even_iter) {
			// natural layout: slot (i, t + c_i) = postcoll_{n-1}(i, t);
			// row offsets cancel so every term lands at row y
			const idx xmm = xm - 1;
			KS.f[dir9::pp] = TNL::Backend::ldg(SD.df(df_cur, dir9::pp, x, y, z));
			KS.f[dir9::pz] = TNL::Backend::ldg(SD.df(df_cur, dir9::pz, x, y, z));
			KS.f[dir9::pm] = TNL::Backend::ldg(SD.df(df_cur, dir9::pm, x, y, z));
			KS.f[dir9::zp] = TNL::Backend::ldg(SD.df(df_cur, dir9::zp, xm, y, z));
			KS.f[dir9::zz] = TNL::Backend::ldg(SD.df(df_cur, dir9::zz, xm, y, z));
			KS.f[dir9::zm] = TNL::Backend::ldg(SD.df(df_cur, dir9::zm, xm, y, z));
			KS.f[dir9::mm] = TNL::Backend::ldg(SD.df(df_cur, dir9::mm, xmm, y, z));
			KS.f[dir9::mz] = TNL::Backend::ldg(SD.df(df_cur, dir9::mz, xmm, y, z));
			KS.f[dir9::mp] = TNL::Backend::ldg(SD.df(df_cur, dir9::mp, xmm, y, z));
		}
		else {
			// twist layout: slot (opp(i), t) = postcoll_{n-1}(i, t)
			KS.f[dir9::pp] = TNL::Backend::ldg(SD.df(df_cur, dir9::mm, xm, ym, z));
			KS.f[dir9::pz] = TNL::Backend::ldg(SD.df(df_cur, dir9::mz, xm, y, z));
			KS.f[dir9::pm] = TNL::Backend::ldg(SD.df(df_cur, dir9::mp, xm, yp, z));
			KS.f[dir9::zp] = TNL::Backend::ldg(SD.df(df_cur, dir9::zm, xm, ym, z));
			KS.f[dir9::zz] = TNL::Backend::ldg(SD.df(df_cur, dir9::zz, xm, y, z));
			KS.f[dir9::zm] = TNL::Backend::ldg(SD.df(df_cur, dir9::zp, xm, yp, z));
			KS.f[dir9::mm] = TNL::Backend::ldg(SD.df(df_cur, dir9::pp, xm, yp, z));
			KS.f[dir9::mz] = TNL::Backend::ldg(SD.df(df_cur, dir9::pz, xm, y, z));
			KS.f[dir9::mp] = TNL::Backend::ldg(SD.df(df_cur, dir9::pm, xm, ym, z));
		}
	}

	// interpolated outflow (Geier 2015): reproduces the legacy fused arithmetic exactly
	// (the parity-free body lives in streaming_AB.h::streamingOutflowInterpRight),
	// site-translated like the streamingOutflowRight gather above.
	// The interpolated m-family blends postcoll_{n-1} from column xm with
	// the outflow cell's own postcoll (column x), the z-family takes the cell's own postcoll,
	// the p-family comes from column xm; all of it is previous-launch state finalized before the pass runs.
	template <typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void
	streamingOutflowInterpRight(LBM_DATA& SD, LBM_KS& KS, idx xm, idx x, idx xp_unused, idx ym, idx y, idx yp, idx zm_unused, idx z, idx zp_unused)
	{
		constexpr dreal SpeedOfSound = 0.5773502691896257;
		if (SD.even_iter) {
			// natural layout: slot (i, t + c_i) = postcoll_{n-1}(i, t); row
			// offsets cancel so every term lands at row y
			const idx xmm = xm - 1;
			KS.f[dir9::pp] = TNL::Backend::ldg(SD.df(df_cur, dir9::pp, x, y, z));
			KS.f[dir9::pz] = TNL::Backend::ldg(SD.df(df_cur, dir9::pz, x, y, z));
			KS.f[dir9::pm] = TNL::Backend::ldg(SD.df(df_cur, dir9::pm, x, y, z));
			KS.f[dir9::zp] = TNL::Backend::ldg(SD.df(df_cur, dir9::zp, x, y, z));
			KS.f[dir9::zz] = TNL::Backend::ldg(SD.df(df_cur, dir9::zz, x, y, z));
			KS.f[dir9::zm] = TNL::Backend::ldg(SD.df(df_cur, dir9::zm, x, y, z));
			KS.f[dir9::mm] = lbm_fma_rn(
				SpeedOfSound,
				TNL::Backend::ldg(SD.df(df_cur, dir9::mm, xmm, y, z)),
				(1 - SpeedOfSound) * TNL::Backend::ldg(SD.df(df_cur, dir9::mm, xm, y, z))
			);
			KS.f[dir9::mz] = lbm_fma_rn(
				SpeedOfSound,
				TNL::Backend::ldg(SD.df(df_cur, dir9::mz, xmm, y, z)),
				(1 - SpeedOfSound) * TNL::Backend::ldg(SD.df(df_cur, dir9::mz, xm, y, z))
			);
			KS.f[dir9::mp] = lbm_fma_rn(
				SpeedOfSound,
				TNL::Backend::ldg(SD.df(df_cur, dir9::mp, xmm, y, z)),
				(1 - SpeedOfSound) * TNL::Backend::ldg(SD.df(df_cur, dir9::mp, xm, y, z))
			);
		}
		else {
			// twist layout: slot (opp(i), t) = postcoll_{n-1}(i, t)
			KS.f[dir9::pp] = TNL::Backend::ldg(SD.df(df_cur, dir9::mm, xm, ym, z));
			KS.f[dir9::pz] = TNL::Backend::ldg(SD.df(df_cur, dir9::mz, xm, y, z));
			KS.f[dir9::pm] = TNL::Backend::ldg(SD.df(df_cur, dir9::mp, xm, yp, z));
			KS.f[dir9::zp] = TNL::Backend::ldg(SD.df(df_cur, dir9::zm, x, ym, z));
			KS.f[dir9::zz] = TNL::Backend::ldg(SD.df(df_cur, dir9::zz, x, y, z));
			KS.f[dir9::zm] = TNL::Backend::ldg(SD.df(df_cur, dir9::zp, x, yp, z));
			KS.f[dir9::mm] = lbm_fma_rn(
				SpeedOfSound,
				TNL::Backend::ldg(SD.df(df_cur, dir9::pp, xm, yp, z)),
				(1 - SpeedOfSound) * TNL::Backend::ldg(SD.df(df_cur, dir9::pp, x, yp, z))
			);
			KS.f[dir9::mz] = lbm_fma_rn(
				SpeedOfSound,
				TNL::Backend::ldg(SD.df(df_cur, dir9::pz, xm, y, z)),
				(1 - SpeedOfSound) * TNL::Backend::ldg(SD.df(df_cur, dir9::pz, x, y, z))
			);
			KS.f[dir9::mp] = lbm_fma_rn(
				SpeedOfSound,
				TNL::Backend::ldg(SD.df(df_cur, dir9::pm, xm, ym, z)),
				(1 - SpeedOfSound) * TNL::Backend::ldg(SD.df(df_cur, dir9::pm, x, ym, z))
			);
		}
	}
};
