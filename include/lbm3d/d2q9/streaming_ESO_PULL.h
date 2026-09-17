#pragma once

#include "lbm3d/defs.h"
#include "lbm_common/rounding.h"

// Esoteric Pull (Lehmann 2022): single in-place DF array. Directions are
// processed as opposite pairs (head h = the odd-numbered slot, tail
// t = opposite_direction(h)); the rest population is always local.
// - even_iter == false: heads are read at the own site, tails are pulled from
//   the neighbor one step along c_h; after collision each head value is
//   written into the tail slot at that neighbor and each tail value into the
//   head slot at the own site (every memory location is read and written by
//   the same thread, so no race).
// - even_iter == true: the parity counterpart.
// The pre-collision populations read here are identical to the A-B pull
// scheme's at every launch, provided the initial DF field is placed as the
// streamed pull-scheme state: slot (h, s) = eq_h(s - c_h) for heads with
// direction vector c_h, tail and rest slots natural.
template <typename TRAITS>
struct D2Q9_STREAMING_ESO_PULL
{
	static constexpr int DFMAX = 1;
	// DF slot that holds the freshly written field after a kernel launch
	static constexpr std::uint8_t output_df = df_cur;

	using idx = typename TRAITS::idx;
	using dreal = typename TRAITS::dreal;

	using SyncDirection = TNL::Containers::SyncDirection;

	template <typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void
	postCollisionStreaming(LBM_DATA& SD, LBM_KS& KS, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm_unused, idx z, idx zp_unused)
	{
		if (! SD.even_iter) {
			// heads are written to the tail slot at the neighbor along c_h,
			// tails to the head slot at the own site
			SD.df(df_cur, dir9::mz, xp, y, z) = KS.f[dir9::pz];
			SD.df(df_cur, dir9::pz, x, y, z) = KS.f[dir9::mz];

			SD.df(df_cur, dir9::zm, x, yp, z) = KS.f[dir9::zp];
			SD.df(df_cur, dir9::zp, x, y, z) = KS.f[dir9::zm];

			SD.df(df_cur, dir9::mm, xp, yp, z) = KS.f[dir9::pp];
			SD.df(df_cur, dir9::pp, x, y, z) = KS.f[dir9::mm];

			SD.df(df_cur, dir9::mp, xp, ym, z) = KS.f[dir9::pm];
			SD.df(df_cur, dir9::pm, x, y, z) = KS.f[dir9::mp];

			SD.df(df_cur, dir9::zz, x, y, z) = KS.f[dir9::zz];
		}
		else {
			SD.df(df_cur, dir9::pz, xp, y, z) = KS.f[dir9::pz];
			SD.df(df_cur, dir9::mz, x, y, z) = KS.f[dir9::mz];

			SD.df(df_cur, dir9::zp, x, yp, z) = KS.f[dir9::zp];
			SD.df(df_cur, dir9::zm, x, y, z) = KS.f[dir9::zm];

			SD.df(df_cur, dir9::pp, xp, yp, z) = KS.f[dir9::pp];
			SD.df(df_cur, dir9::mm, x, y, z) = KS.f[dir9::mm];

			SD.df(df_cur, dir9::pm, xp, ym, z) = KS.f[dir9::pm];
			SD.df(df_cur, dir9::mp, x, y, z) = KS.f[dir9::mp];

			SD.df(df_cur, dir9::zz, x, y, z) = KS.f[dir9::zz];
		}
	}

	template <typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void
	streaming(LBM_DATA& SD, LBM_KS& KS, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm_unused, idx z, idx zp_unused)
	{
		if (! SD.even_iter) {
			// heads at the own site, tails pulled from the neighbor along c_h
			KS.f[dir9::pz] = TNL::Backend::ldg(SD.df(df_cur, dir9::pz, x, y, z));
			KS.f[dir9::mz] = TNL::Backend::ldg(SD.df(df_cur, dir9::mz, xp, y, z));

			KS.f[dir9::zp] = TNL::Backend::ldg(SD.df(df_cur, dir9::zp, x, y, z));
			KS.f[dir9::zm] = TNL::Backend::ldg(SD.df(df_cur, dir9::zm, x, yp, z));

			KS.f[dir9::pp] = TNL::Backend::ldg(SD.df(df_cur, dir9::pp, x, y, z));
			KS.f[dir9::mm] = TNL::Backend::ldg(SD.df(df_cur, dir9::mm, xp, yp, z));

			KS.f[dir9::pm] = TNL::Backend::ldg(SD.df(df_cur, dir9::pm, x, y, z));
			KS.f[dir9::mp] = TNL::Backend::ldg(SD.df(df_cur, dir9::mp, xp, ym, z));

			KS.f[dir9::zz] = TNL::Backend::ldg(SD.df(df_cur, dir9::zz, x, y, z));
		}
		else {
			KS.f[dir9::pz] = TNL::Backend::ldg(SD.df(df_cur, dir9::mz, x, y, z));
			KS.f[dir9::mz] = TNL::Backend::ldg(SD.df(df_cur, dir9::pz, xp, y, z));

			KS.f[dir9::zp] = TNL::Backend::ldg(SD.df(df_cur, dir9::zm, x, y, z));
			KS.f[dir9::zm] = TNL::Backend::ldg(SD.df(df_cur, dir9::zp, x, yp, z));

			KS.f[dir9::pp] = TNL::Backend::ldg(SD.df(df_cur, dir9::mm, x, y, z));
			KS.f[dir9::mm] = TNL::Backend::ldg(SD.df(df_cur, dir9::pp, xp, yp, z));

			KS.f[dir9::pm] = TNL::Backend::ldg(SD.df(df_cur, dir9::mp, x, y, z));
			KS.f[dir9::mp] = TNL::Backend::ldg(SD.df(df_cur, dir9::pm, xp, ym, z));

			KS.f[dir9::zz] = TNL::Backend::ldg(SD.df(df_cur, dir9::zz, x, y, z));
		}
	}

	// streaming with the bounce-back rule applied: the identity write-back of
	// this pattern preserves the implicit bounce-back of the esoteric schemes
	// (a swapped wall cell writes back exactly what it read)
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

	// slot and site of the pre-collision population of direction i authored at
	// site w, given the layout finalized by the previous launch and the
	// current parity:
	// - even_iter == false (post-phase-2 layout): slot (i, w + [head] c_i)
	// - even_iter == true  (post-phase-1 layout): slot (opp i, w + [head] c_i)
	template <typename LBM_DATA>
	__cuda_callable__ static dreal outflowValue(LBM_DATA& SD, int i, idx wx, idx wy, idx z)
	{
		const idx ox = is_pair_head(i) ? dir9_cx(i) : 0;
		const idx oy = is_pair_head(i) ? dir9_cy(i) : 0;
		const int slot = SD.even_iter ? opposite_direction(i) : i;
		return TNL::Backend::ldg(SD.df(df_cur, slot, wx + ox, wy + oy, z));
	}

	// Outflow-pass gather (a separate kernel launched before the main one for
	// deterministic outflow handling - the outflow pass is a type of
	// processing, not a BC type): reconstructs the pre-collision populations
	// at the translated A-B pull sites of the outflow cell:
	// postcoll_{n-1}(i) at site t_i = (anchor, tangential -c_i), where the
	// anchor is the fluid-side neighbor column one cell inward.
	// FACE is a compile-time template parameter, so the per-direction
	// components and site offsets fold to constants.
	template <int FACE, typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void streamingOutflowImpl(LBM_DATA& SD, LBM_KS& KS, idx anchor, idx x, idx y, idx z)
	{
		constexpr bool axis_x = (FACE & (bc_face::XP | bc_face::XM)) != 0;
		for (int i = 0; i < 9; i++) {
			const idx wx = axis_x ? anchor : x - dir9_cx(i);
			const idx wy = axis_x ? y - dir9_cy(i) : anchor;
			KS.f[i] = outflowValue(SD, i, wx, wy, z);
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
	__cuda_callable__ static dreal outflowInterpBlend(LBM_DATA& SD, int i, idx anchor_x, idx anchor_y, idx own_x, idx own_y, idx z)
	{
		// NOTE: velocity is neglected (for the case velocity << speed of sound)
		constexpr dreal SpeedOfSound = 0.5773502691896257;
		return lbm_fma_rn(SpeedOfSound, outflowValue(SD, i, anchor_x, anchor_y, z), (1 - SpeedOfSound) * outflowValue(SD, i, own_x, own_y, z));
	}

	// interpolated outflow (Geier 2015) for an arbitrary face: the population
	// moving against the outward normal blends postcoll_{n-1} from the anchor
	// column with the outflow cell's own postcoll, the perpendicular population
	// takes the cell's own postcoll, the outward-moving population comes from
	// the anchor column; all of it is previous-launch state finalized before
	// the pass runs.
	// FACE is a compile-time template parameter, so the per-direction
	// components, site offsets and family branches fold to constants.
	template <int FACE, typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void streamingOutflowInterpImpl(LBM_DATA& SD, LBM_KS& KS, idx anchor, idx x, idx y, idx z)
	{
		constexpr bool axis_x = (FACE & (bc_face::XP | bc_face::XM)) != 0;
		constexpr int out_sign = (FACE & (bc_face::XM | bc_face::YM)) ? -1 : 1;
		for (int i = 0; i < 9; i++) {
			const int cn = axis_x ? dir9_cx(i) : dir9_cy(i);  // normal component of c_i
			// value at the anchor column and at the own column, tangential -c offsets
			const idx nx = axis_x ? anchor : x - dir9_cx(i);
			const idx ny = axis_x ? y - dir9_cy(i) : anchor;
			const idx ox = axis_x ? x : x - dir9_cx(i);
			const idx oy = axis_x ? y - dir9_cy(i) : y;
			if (cn == out_sign)
				KS.f[i] = outflowValue(SD, i, nx, ny, z);
			else if (cn == 0)
				KS.f[i] = outflowValue(SD, i, ox, oy, z);
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

	// DF halo exchange descriptors for slot dir on lattice axis a (0=x, 1=y).
	// The fresh cross-boundary populations sit one plane beyond the block
	// boundary on the side opposite to their PARITY-dependent storage layout:
	// - after a phase-1 launch (even_iter == false): slots hold populations of
	//   the opposite direction, exchange mask is counter-canonical, and the
	//   head-side data sits shifted by one (shift = [tail])
	// - after a phase-2 launch (even_iter == true): slots hold their canonical
	//   populations, exchange mask is canonical, shift = [head]
	// SyncDirection::None means no exchange on this axis.
	__cuda_callable__ static constexpr SyncDirection dfSyncDirection(int dir, int axis, bool even_iter)
	{
		const int c = axis == 0 ? dir9_cx(dir) : dir9_cy(dir);
		if (c == 0)
			return SyncDirection::None;
		// after a phase-1 launch the slot carries the opposite direction's
		// population, so the exchange is counter-canonical
		const bool positive = even_iter ? c > 0 : c < 0;
		if (positive)
			return axis == 0 ? SyncDirection::Right : SyncDirection::Top;
		return axis == 0 ? SyncDirection::Left : SyncDirection::Bottom;
	}

	__cuda_callable__ static constexpr int dfSyncOffset(int dir, int axis, bool even_iter)
	{
		(void) axis;
		return even_iter ? int(is_pair_head(dir)) : int(! is_pair_head(dir));
	}
};

template <typename TRAITS>
inline constexpr bool is_ESO_PULL_v<D2Q9_STREAMING_ESO_PULL<TRAITS>> = true;
template <typename TRAITS>
inline constexpr bool is_esoteric_in_place_v<D2Q9_STREAMING_ESO_PULL<TRAITS>> = true;
template <typename TRAITS>
inline constexpr bool requires_ghost_layer_v<D2Q9_STREAMING_ESO_PULL<TRAITS>> = true;
