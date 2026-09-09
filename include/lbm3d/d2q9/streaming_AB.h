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

	// outflow pass gathers for an arbitrary face: the outflow cell takes the
	// pulled state of its anchor column (the fluid-side neighbor, one cell
	// inward) from df_cur (finalized by the previous launch, no race against
	// the df_out writes of the current one).
	// FACE is a compile-time template parameter, so the per-direction
	// components, site offsets and family branches fold to constants.
	template <int FACE, typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void streamingOutflowImpl(LBM_DATA& SD, LBM_KS& KS, idx anchor, idx x, idx y, idx z)
	{
		constexpr bool axis_x = (FACE & (bc_face::XP | bc_face::XM)) != 0;
		for (int i = 0; i < 9; i++) {
			// normal coordinate: the anchor column; tangential: -c offset (pull scheme)
			const idx sx = axis_x ? anchor : x - dir9_cx(i);
			const idx sy = axis_x ? y - dir9_cy(i) : anchor;
			KS.f[i] = TNL::Backend::ldg(SD.df(df_cur, i, sx, sy, z));
		}
	}

	template <typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void streamingOutflow(LBM_DATA& SD, LBM_KS& KS, int face, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx z)
	{
		switch (face) {
			case bc_face::XP:
				streamingOutflowImpl<bc_face::XP>(SD, KS, xm, x, y, z);
				break;
			case bc_face::XM:
				streamingOutflowImpl<bc_face::XM>(SD, KS, xp, x, y, z);
				break;
			case bc_face::YP:
				streamingOutflowImpl<bc_face::YP>(SD, KS, ym, x, y, z);
				break;
			default:
				streamingOutflowImpl<bc_face::YM>(SD, KS, yp, x, y, z);
				break;
		}
	}

	// interpolated-outflow blend in the pinned lbm_fma_rn form:
	// the first site delivers the anchor-column postcoll (weight cs),
	// the second site the own-column postcoll (weight 1-cs)
	template <typename LBM_DATA>
	__cuda_callable__ static dreal outflowInterpBlend(LBM_DATA& SD, int dir, idx anchor_x, idx anchor_y, idx own_x, idx own_y, idx z)
	{
		// NOTE: velocity is neglected (for the case velocity << speed of sound)
		constexpr dreal SpeedOfSound = 0.5773502691896257;
		return lbm_fma_rn(
			SpeedOfSound,
			TNL::Backend::ldg(SD.df(df_cur, dir, anchor_x, anchor_y, z)),
			(1 - SpeedOfSound) * TNL::Backend::ldg(SD.df(df_cur, dir, own_x, own_y, z))
		);
	}

	// interpolated outflow (Geier 2015) for an arbitrary face: the population
	// moving against the outward normal blends postcoll_{n-1} from the anchor
	// column with the outflow cell's own postcoll, the perpendicular population
	// streams ordinarily (own column), the outward-moving population takes the
	// pulled state of the anchor column.
	// FACE is a compile-time template parameter, so the per-direction
	// components, site offsets and family branches fold to constants.
	template <int FACE, typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void streamingOutflowInterpImpl(LBM_DATA& SD, LBM_KS& KS, idx anchor, idx x, idx y, idx z)
	{
		constexpr bool axis_x = (FACE & (bc_face::XP | bc_face::XM)) != 0;
		constexpr int out_sign = (FACE & (bc_face::XM | bc_face::YM)) ? -1 : 1;
		for (int i = 0; i < 9; i++) {
			const int cn = axis_x ? dir9_cx(i) : dir9_cy(i);  // normal component of c_i
			// sites in the anchor column and the own column, tangential -c offsets
			const idx nx = axis_x ? anchor : x - dir9_cx(i);
			const idx ny = axis_x ? y - dir9_cy(i) : anchor;
			const idx ox = axis_x ? x : x - dir9_cx(i);
			const idx oy = axis_x ? y - dir9_cy(i) : y;
			if (cn == out_sign)
				KS.f[i] = TNL::Backend::ldg(SD.df(df_cur, i, nx, ny, z));
			else if (cn == 0)
				KS.f[i] = TNL::Backend::ldg(SD.df(df_cur, i, ox, oy, z));
			else
				KS.f[i] = outflowInterpBlend(SD, i, nx, ny, ox, oy, z);
		}
	}

	template <typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void streamingOutflowInterp(LBM_DATA& SD, LBM_KS& KS, int face, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx z)
	{
		switch (face) {
			case bc_face::XP:
				streamingOutflowInterpImpl<bc_face::XP>(SD, KS, xm, x, y, z);
				break;
			case bc_face::XM:
				streamingOutflowInterpImpl<bc_face::XM>(SD, KS, xp, x, y, z);
				break;
			case bc_face::YP:
				streamingOutflowInterpImpl<bc_face::YP>(SD, KS, ym, x, y, z);
				break;
			default:
				streamingOutflowInterpImpl<bc_face::YM>(SD, KS, yp, x, y, z);
				break;
		}
	}
};
