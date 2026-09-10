#pragma once

#include "lbm3d/defs.h"

// Esoteric Push (Lehmann 2022): single in-place DF array. Directions are
// processed as opposite pairs (head h = the odd-numbered slot, tail
// t = opposite_direction(h)) with the head's direction vector c_h; the rest
// population is always local.
// - even_iter == false: each head is pushed into its own slot at the own
//   site, each tail into the tail slot shifted upstream by -c_h; populations
//   are read from the tail slot upstream/the head slot at the own site
//   (every memory location is read and written by the same thread, so no
//   race).
// - even_iter == true: the parity counterpart with swapped push/pull roles.
// The pre-collision populations read here are identical to the A-B pull
// scheme's at every launch, provided the initial DF field is placed as the
// streamed push-scheme state: slot (opp h, s) = eq_h(s) for heads, natural
// for tails and rest.
template <typename TRAITS>
struct D3Q7_STREAMING_ESO_PUSH
{
	static constexpr int DFMAX = 1;
	// DF slot that holds the freshly written field after a kernel launch
	static constexpr std::uint8_t output_df = df_cur;

	using idx = typename TRAITS::idx;
	using dreal = typename TRAITS::dreal;

	using SyncDirection = TNL::Containers::SyncDirection;

	template <typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void
	postCollisionStreaming(LBM_DATA& SD, LBM_KS& KS, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm, idx z, idx zp)
	{
		if (! SD.even_iter) {
			// heads pushed to their own slot at the own site, tails to the
			// tail slot shifted by -c_h
			SD.df(df_cur, pzz, x, y, z) = KS.f[pzz];
			SD.df(df_cur, mzz, xm, y, z) = KS.f[mzz];
			SD.df(df_cur, zpz, x, y, z) = KS.f[zpz];
			SD.df(df_cur, zmz, x, ym, z) = KS.f[zmz];
			SD.df(df_cur, zzp, x, y, z) = KS.f[zzp];
			SD.df(df_cur, zzm, x, y, zm) = KS.f[zzm];
			SD.df(df_cur, zzz, x, y, z) = KS.f[zzz];
		}
		else {
			SD.df(df_cur, pzz, xm, y, z) = KS.f[mzz];
			SD.df(df_cur, mzz, x, y, z) = KS.f[pzz];
			SD.df(df_cur, zpz, x, ym, z) = KS.f[zmz];
			SD.df(df_cur, zmz, x, y, z) = KS.f[zpz];
			SD.df(df_cur, zzp, x, y, zm) = KS.f[zzm];
			SD.df(df_cur, zzm, x, y, z) = KS.f[zzp];
			SD.df(df_cur, zzz, x, y, z) = KS.f[zzz];
		}
	}

	template <typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void streaming(LBM_DATA& SD, LBM_KS& KS, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm, idx z, idx zp)
	{
		if (! SD.even_iter) {
			// heads read from the tail slot one step upstream (-c_h), tails
			// from the head slot at the own site
			KS.f[pzz] = SD.df(df_cur, mzz, xm, y, z);
			KS.f[mzz] = SD.df(df_cur, pzz, x, y, z);
			KS.f[zpz] = SD.df(df_cur, zmz, x, ym, z);
			KS.f[zmz] = SD.df(df_cur, zpz, x, y, z);
			KS.f[zzp] = SD.df(df_cur, zzm, x, y, zm);
			KS.f[zzm] = SD.df(df_cur, zzp, x, y, z);
			KS.f[zzz] = SD.df(df_cur, zzz, x, y, z);
		}
		else {
			KS.f[pzz] = SD.df(df_cur, pzz, xm, y, z);
			KS.f[mzz] = SD.df(df_cur, mzz, x, y, z);
			KS.f[zpz] = SD.df(df_cur, zpz, x, ym, z);
			KS.f[zmz] = SD.df(df_cur, zmz, x, y, z);
			KS.f[zzp] = SD.df(df_cur, zzp, x, y, zm);
			KS.f[zzm] = SD.df(df_cur, zzm, x, y, z);
			KS.f[zzz] = SD.df(df_cur, zzz, x, y, z);
		}
	}

	// DF halo exchange descriptors for slot dir on lattice axis a (0=x, 1=y,
	// 2=z). The descriptor parity is the parity of the launch that AUTHORED
	// the layout:
	// - phase-1 launch (even_iter == false) layout: canonical mask, shift [tail]
	// - phase-2 launch (even_iter == true) layout: opposite mask, shift [head]
	// SyncDirection::None means no exchange on this axis.
	__cuda_callable__ static constexpr SyncDirection dfSyncDirection(int dir, int axis, bool even_iter)
	{
		const int c = axis == 0 ? dir27_cx(dir) : axis == 1 ? dir27_cy(dir) : dir27_cz(dir);
		if (c == 0)
			return SyncDirection::None;
		const bool positive = even_iter ? c < 0 : c > 0;
		if (positive)
			return axis == 0 ? SyncDirection::Right : axis == 1 ? SyncDirection::Top : SyncDirection::Front;
		return axis == 0 ? SyncDirection::Left : axis == 1 ? SyncDirection::Bottom : SyncDirection::Back;
	}

	__cuda_callable__ static constexpr int dfSyncOffset(int dir, int axis, bool even_iter)
	{
		(void) axis;
		return even_iter ? int(is_pair_head(dir)) : int(! is_pair_head(dir));
	}
};

template <typename TRAITS>
inline constexpr bool is_ESO_PUSH_v<D3Q7_STREAMING_ESO_PUSH<TRAITS>> = true;
template <typename TRAITS>
inline constexpr bool is_esoteric_in_place_v<D3Q7_STREAMING_ESO_PUSH<TRAITS>> = true;
template <typename TRAITS>
inline constexpr bool requires_ghost_layer_v<D3Q7_STREAMING_ESO_PUSH<TRAITS>> = true;
