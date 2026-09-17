#pragma once

#include "lbm3d/defs.h"
#include "lbm_common/rounding.h"

// A-B push scheme: two DF arrays. The post-collision populations are written
// straight to their target sites in the other array, so the next launch reads
// them at its own site; slot (i, s) then holds the population that arrived at
// s from s - c_i (the post-stream layout, like the A-A even sub-step, but with
// separate arrays and no direction twist).
//
// The outflow-pass gathers read the finalized post-stream array: the
// pre-collision population postcoll_{n-1}(i, s) is found in slot (i, s + c_i).
template <typename TRAITS>
struct D2Q9_STREAMING_AB_PUSH
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
	postCollisionStreaming(LBM_DATA& SD, LBM_KS& KS, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm_unused, idx z, idx zp_unused)
	{
		SD.df(df_out, dir9::zz, x, y, z) = KS.f[dir9::zz];
		SD.df(df_out, dir9::pz, xp, y, z) = KS.f[dir9::pz];
		SD.df(df_out, dir9::mz, xm, y, z) = KS.f[dir9::mz];
		SD.df(df_out, dir9::zp, x, yp, z) = KS.f[dir9::zp];
		SD.df(df_out, dir9::zm, x, ym, z) = KS.f[dir9::zm];
		SD.df(df_out, dir9::pp, xp, yp, z) = KS.f[dir9::pp];
		SD.df(df_out, dir9::mm, xm, ym, z) = KS.f[dir9::mm];
		SD.df(df_out, dir9::pm, xp, ym, z) = KS.f[dir9::pm];
		SD.df(df_out, dir9::mp, xm, yp, z) = KS.f[dir9::mp];
	}

	// the post-stream populations sit at their own site: identity read
	template <typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void
	streaming(uint8_t type, LBM_DATA& SD, LBM_KS& KS, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm_unused, idx z, idx zp_unused)
	{
		for (int i = 0; i < 9; i++)
			KS.f[i] = TNL::Backend::ldg(SD.df(type, i, x, y, z));
	}

	template <typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void streaming(LBM_DATA& SD, LBM_KS& KS, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm, idx z, idx zp)
	{
		streaming(df_cur, SD, KS, xm, x, xp, ym, y, yp, zm, z, zp);
	}

	// streaming with bounce-back rule applied: identity read, then swap all 4
	// opposite DF pairs (the same effect as the GEO_WALL bounce-back collision)
	template <typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void
	streamingBounceBack(LBM_DATA& SD, LBM_KS& KS, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm_unused, idx z, idx zp_unused)
	{
		streaming(SD, KS, xm, x, xp, ym, y, yp, zm_unused, z, zp_unused);
		TNL::swap(KS.f[dir9::mm], KS.f[dir9::pp]);
		TNL::swap(KS.f[dir9::mz], KS.f[dir9::pz]);
		TNL::swap(KS.f[dir9::mp], KS.f[dir9::pm]);
		TNL::swap(KS.f[dir9::zm], KS.f[dir9::zp]);
	}

	// Outflow-pass gather for an arbitrary face, in the post-stream layout
	// (see the file header): the pull-scheme site (anchor, tangential -c_i) is
	// found in slot (i, anchor + c_i[normal], tangential own). The pass reads
	// only the finalized previous launch, so there is no race against the
	// df_out writes of the current one.
	// FACE is a compile-time template parameter, so the per-direction
	// components, site offsets and family branches fold to constants.
	template <int FACE, typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void streamingOutflowImpl(LBM_DATA& SD, LBM_KS& KS, idx anchor, idx x, idx y, idx z)
	{
		constexpr bool axis_x = (FACE & (bc_face::XP | bc_face::XM)) != 0;
		for (int i = 0; i < 9; i++) {
			// normal coordinate: anchor + normal component of c_i; tangential: own
			const idx sx = axis_x ? anchor + dir9_cx(i) : x;
			const idx sy = axis_x ? y : anchor + dir9_cy(i);
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

	// interpolated outflow (Geier 2015) for an arbitrary face, in the
	// post-stream layout: pulls' anchor/own-column sites (s) map to slots
	// (s + c_i), so both columns use the cell's own tangential coordinates and
	// the normal coordinate shifted by the normal component of c_i.
	// FACE is a compile-time template parameter, so the per-direction
	// components, site offsets and family branches fold to constants.
	template <int FACE, typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void streamingOutflowInterpImpl(LBM_DATA& SD, LBM_KS& KS, idx anchor, idx x, idx y, idx z)
	{
		constexpr bool axis_x = (FACE & (bc_face::XP | bc_face::XM)) != 0;
		constexpr int out_sign = (FACE & (bc_face::XM | bc_face::YM)) ? -1 : 1;
		for (int i = 0; i < 9; i++) {
			const int cn = axis_x ? dir9_cx(i) : dir9_cy(i);  // normal component of c_i
			// mapped anchor-column slot and own-column slot
			const idx nx = axis_x ? anchor + cn : x;
			const idx ny = axis_x ? y : anchor + cn;
			const idx ox = axis_x ? x + cn : x;
			const idx oy = axis_x ? y : y + cn;
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

template <typename TRAITS>
inline constexpr bool is_AB_PUSH_v<D2Q9_STREAMING_AB_PUSH<TRAITS>> = true;
