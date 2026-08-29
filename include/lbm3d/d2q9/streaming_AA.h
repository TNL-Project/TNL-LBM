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

	// Streaming for separate outflow-pass kernel (deterministic BC for A-A pattern),
	// parameterized by the outflow face (outward normal): both branches reconstruct
	// the pre-collision populations at the translated A-B pull sites of the outflow
	// cell: postcoll_{n-1}(i) at site (anchor, tangential -c_i), where the anchor is
	// the fluid-side neighbor column one cell inward.
	// The slot layouts provide this only from the finalized previous launch:
	// the required slots are owned by the pre-anchor/interior threads in both
	// parities, so no race-free in-launch gather exists and the pass must run
	// before the main kernel.
	// FACE is a compile-time template parameter, so the per-direction
	// components, site offsets and family branches fold to constants.
	template <int FACE, typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void streamingOutflowImpl(LBM_DATA& SD, LBM_KS& KS, idx anchor, idx x, idx y, idx z)
	{
		constexpr bool axis_x = (FACE & (bc_face::XP | bc_face::XM)) != 0;
		if (SD.even_iter) {
			// natural layout: slot (i, t + c_i) = postcoll_{n-1}(i, t) with
			// t = (anchor, tangential -c_i); the tangential offsets cancel
			// against +c_i, so the normal coordinate is anchor + c_i[normal]
			// and the tangential coordinates are the cell's own
			for (int i = 0; i < 9; i++) {
				const idx sx = axis_x ? anchor + dir9_cx(i) : x;
				const idx sy = axis_x ? y : anchor + dir9_cy(i);
				KS.f[i] = TNL::Backend::ldg(SD.df(df_cur, i, sx, sy, z));
			}
		}
		else {
			// twist layout: slot (opp(i), t) = postcoll_{n-1}(i, t)
			for (int i = 0; i < 9; i++) {
				const int ct = axis_x ? dir9_cy(i) : dir9_cx(i);
				const idx sx = axis_x ? anchor : x - ct;
				const idx sy = axis_x ? y - ct : anchor;
				KS.f[i] = TNL::Backend::ldg(SD.df(df_cur, opposite_direction(i), sx, sy, z));
			}
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

	// interpolated outflow (Geier 2015) for an arbitrary face: reproduces the
	// legacy fused arithmetic exactly, site-translated like the streamingOutflow
	// gather above. The population moving against the outward normal blends
	// postcoll_{n-1} from the anchor column with the outflow cell's own postcoll,
	// the perpendicular population takes the cell's own postcoll, the outward-
	// moving population comes from the anchor column; all of it is previous-launch
	// state finalized before the pass runs.
	// FACE is a compile-time template parameter, so the per-direction
	// components, site offsets and family branches fold to constants.
	template <int FACE, typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void streamingOutflowInterpImpl(LBM_DATA& SD, LBM_KS& KS, idx anchor, idx x, idx y, idx z)
	{
		constexpr bool axis_x = (FACE & (bc_face::XP | bc_face::XM)) != 0;
		constexpr int out_sign = (FACE & (bc_face::XM | bc_face::YM)) ? -1 : 1;
		if (SD.even_iter) {
			// natural layout: the outward- and perpendicular-moving populations
			// take the cell's own postcoll, the inward-moving population blends
			// the pre-anchor column with the anchor column
			for (int i = 0; i < 9; i++) {
				const int cn = axis_x ? dir9_cx(i) : dir9_cy(i);
				if (cn == out_sign || cn == 0)
					KS.f[i] = TNL::Backend::ldg(SD.df(df_cur, i, x, y, z));
				else if constexpr (axis_x)
					KS.f[i] = outflowInterpBlend(SD, i, anchor + dir9_cx(i), y, anchor, y, z);
				else
					KS.f[i] = outflowInterpBlend(SD, i, x, anchor + dir9_cy(i), x, anchor, z);
			}
		}
		else {
			// twist layout: the outward-moving population comes from the anchor
			// column, the perpendicular population from the own column, the
			// inward-moving population blends the anchor column with the own column
			for (int i = 0; i < 9; i++) {
				const int ct = axis_x ? dir9_cy(i) : dir9_cx(i);
				const int cn = axis_x ? dir9_cx(i) : dir9_cy(i);
				const idx nx = axis_x ? anchor : x - ct;
				const idx ny = axis_x ? y - ct : anchor;
				const idx ox = axis_x ? x : x - ct;
				const idx oy = axis_x ? y - ct : y;
				if (cn == out_sign)
					KS.f[i] = TNL::Backend::ldg(SD.df(df_cur, opposite_direction(i), nx, ny, z));
				else if (cn == 0)
					KS.f[i] = TNL::Backend::ldg(SD.df(df_cur, opposite_direction(i), ox, oy, z));
				else
					KS.f[i] = outflowInterpBlend(SD, opposite_direction(i), nx, ny, ox, oy, z);
			}
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
