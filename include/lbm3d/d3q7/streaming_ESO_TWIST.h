#pragma once

#include "lbm3d/defs.h"

// Esoteric Twist (Geier & Schönherr 2017): single in-place DF array.
// Directions with negative components are pulled from the upper-octant
// neighbors while positive directions are read at the own site; after
// collision the values are written back in opposite (twisted) direction, so
// every memory location is read and written by the same thread in each
// launch. The pointer swap of the paper is realized by alternating two
// parities:
// - even_iter == false (phase A): natural slots, read at site n + sigma(c)
//   with sigma(c) = |min(c, 0)| componentwise; write back in opposite slots
//   at the same sites (each site writes post[opp(c)] into slot c).
// - even_iter == true (phase B): the twist counterpart: read from opposite
//   slots at the same sites; write back to own slots at site n + p(c) with
//   p(c) = max(c, 0) componentwise.
// The pre-collision populations read here are identical to the A-B pull
// scheme's at every launch, provided the initial DF field is placed as:
// slot (c, s) = eq_c(s - p(c)).
template <typename TRAITS>
struct D3Q7_STREAMING_ESO_TWIST
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
			// phase A: slot (c, n + sigma(c)) receives post[opp(c)]
			SD.df(df_cur, zzz, x, y, z) = KS.f[zzz];
			SD.df(df_cur, pzz, x, y, z) = KS.f[mzz];
			SD.df(df_cur, mzz, xp, y, z) = KS.f[pzz];
			SD.df(df_cur, zpz, x, y, z) = KS.f[zmz];
			SD.df(df_cur, zmz, x, yp, z) = KS.f[zpz];
			SD.df(df_cur, zzp, x, y, z) = KS.f[zzm];
			SD.df(df_cur, zzm, x, y, zp) = KS.f[zzp];
		}
		else {
			// phase B: slot (d, n + p(d)) keeps post[d]
			SD.df(df_cur, zzz, x, y, z) = KS.f[zzz];
			SD.df(df_cur, pzz, xp, y, z) = KS.f[pzz];
			SD.df(df_cur, mzz, x, y, z) = KS.f[mzz];
			SD.df(df_cur, zpz, x, yp, z) = KS.f[zpz];
			SD.df(df_cur, zmz, x, y, z) = KS.f[zmz];
			SD.df(df_cur, zzp, x, y, zp) = KS.f[zzp];
			SD.df(df_cur, zzm, x, y, z) = KS.f[zzm];
		}
	}

	template <typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void streaming(LBM_DATA& SD, LBM_KS& KS, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm, idx z, idx zp)
	{
		if (! SD.even_iter) {
			// phase A: KS.f[c] = slot (c, n + sigma(c))
			KS.f[zzz] = SD.df(df_cur, zzz, x, y, z);
			KS.f[pzz] = SD.df(df_cur, pzz, x, y, z);
			KS.f[mzz] = SD.df(df_cur, mzz, xp, y, z);
			KS.f[zpz] = SD.df(df_cur, zpz, x, y, z);
			KS.f[zmz] = SD.df(df_cur, zmz, x, yp, z);
			KS.f[zzp] = SD.df(df_cur, zzp, x, y, z);
			KS.f[zzm] = SD.df(df_cur, zzm, x, y, zp);
		}
		else {
			// phase B: KS.f[c] = slot (opp(c), n + sigma(c))
			KS.f[zzz] = SD.df(df_cur, zzz, x, y, z);
			KS.f[pzz] = SD.df(df_cur, mzz, x, y, z);
			KS.f[mzz] = SD.df(df_cur, pzz, xp, y, z);
			KS.f[zpz] = SD.df(df_cur, zmz, x, y, z);
			KS.f[zmz] = SD.df(df_cur, zpz, x, yp, z);
			KS.f[zzp] = SD.df(df_cur, zzm, x, y, z);
			KS.f[zzm] = SD.df(df_cur, zzp, x, y, zp);
		}
	}
};

template <typename TRAITS>
inline constexpr bool is_ESO_TWIST_v<D3Q7_STREAMING_ESO_TWIST<TRAITS>> = true;
template <typename TRAITS>
inline constexpr bool is_esoteric_in_place_v<D3Q7_STREAMING_ESO_TWIST<TRAITS>> = true;
template <typename TRAITS>
inline constexpr bool requires_ghost_layer_v<D3Q7_STREAMING_ESO_TWIST<TRAITS>> = true;
