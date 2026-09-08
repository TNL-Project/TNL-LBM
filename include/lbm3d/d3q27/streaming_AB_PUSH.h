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
struct D3Q27_STREAMING_AB_PUSH
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
	postCollisionStreaming(LBM_DATA& SD, LBM_KS& KS, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm, idx z, idx zp)
	{
		SD.df(df_out, mmm, xm, ym, zm) = KS.f[mmm];
		SD.df(df_out, mmz, xm, ym, z) = KS.f[mmz];
		SD.df(df_out, mmp, xm, ym, zp) = KS.f[mmp];
		SD.df(df_out, mzm, xm, y, zm) = KS.f[mzm];
		SD.df(df_out, mzz, xm, y, z) = KS.f[mzz];
		SD.df(df_out, mzp, xm, y, zp) = KS.f[mzp];
		SD.df(df_out, mpm, xm, yp, zm) = KS.f[mpm];
		SD.df(df_out, mpz, xm, yp, z) = KS.f[mpz];
		SD.df(df_out, mpp, xm, yp, zp) = KS.f[mpp];
		SD.df(df_out, zmm, x, ym, zm) = KS.f[zmm];
		SD.df(df_out, zmz, x, ym, z) = KS.f[zmz];
		SD.df(df_out, zmp, x, ym, zp) = KS.f[zmp];
		SD.df(df_out, zzm, x, y, zm) = KS.f[zzm];
		SD.df(df_out, zzz, x, y, z) = KS.f[zzz];
		SD.df(df_out, zzp, x, y, zp) = KS.f[zzp];
		SD.df(df_out, zpm, x, yp, zm) = KS.f[zpm];
		SD.df(df_out, zpz, x, yp, z) = KS.f[zpz];
		SD.df(df_out, zpp, x, yp, zp) = KS.f[zpp];
		SD.df(df_out, pmm, xp, ym, zm) = KS.f[pmm];
		SD.df(df_out, pmz, xp, ym, z) = KS.f[pmz];
		SD.df(df_out, pmp, xp, ym, zp) = KS.f[pmp];
		SD.df(df_out, pzm, xp, y, zm) = KS.f[pzm];
		SD.df(df_out, pzz, xp, y, z) = KS.f[pzz];
		SD.df(df_out, pzp, xp, y, zp) = KS.f[pzp];
		SD.df(df_out, ppm, xp, yp, zm) = KS.f[ppm];
		SD.df(df_out, ppz, xp, yp, z) = KS.f[ppz];
		SD.df(df_out, ppp, xp, yp, zp) = KS.f[ppp];
	}

	// the post-stream populations sit at their own site: identity read
	template <typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void
	streaming(uint8_t type, LBM_DATA& SD, LBM_KS& KS, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm, idx z, idx zp)
	{
		for (int i = 0; i < 27; i++)
			KS.f[i] = TNL::Backend::ldg(SD.df(type, i, x, y, z));
	}

	template <typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void streaming(LBM_DATA& SD, LBM_KS& KS, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm, idx z, idx zp)
	{
		streaming(df_cur, SD, KS, xm, x, xp, ym, y, yp, zm, z, zp);
	}

	// Bounce-back streaming for the non-Newtonian kernel's wall cells:
	// identity read, then swap all 13 opposite DF pairs (the same effect as
	// the GEO_WALL bounce-back collision).
	template <typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void streamingBounceBack(LBM_DATA& SD, LBM_KS& KS, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm, idx z, idx zp)
	{
		streaming(SD, KS, xm, x, xp, ym, y, yp, zm, z, zp);
		TNL::swap(KS.f[mmm], KS.f[ppp]);
		TNL::swap(KS.f[mmz], KS.f[ppz]);
		TNL::swap(KS.f[mmp], KS.f[ppm]);
		TNL::swap(KS.f[mzm], KS.f[pzp]);
		TNL::swap(KS.f[mzz], KS.f[pzz]);
		TNL::swap(KS.f[mzp], KS.f[pzm]);
		TNL::swap(KS.f[mpm], KS.f[pmp]);
		TNL::swap(KS.f[mpz], KS.f[pmz]);
		TNL::swap(KS.f[mpp], KS.f[pmm]);
		TNL::swap(KS.f[zmm], KS.f[zpp]);
		TNL::swap(KS.f[zzm], KS.f[zzp]);
		TNL::swap(KS.f[zmz], KS.f[zpz]);
		TNL::swap(KS.f[zmp], KS.f[zpm]);
	}

	// Post-stream density at position P = (xp, y, z) (see the pull scheme for
	// the calling context). The post-stream populations sit at their own site
	// in the push layout, so the sum is over the same site.
	template <typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void streamingRho(LBM_DATA& SD, LBM_KS& KS, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm, idx z, idx zp)
	{
		dreal rho = 0;
		for (int i = 0; i < 27; i++)
			rho += SD.df(df_cur, i, xp, y, z);
		KS.rho = rho;
	}

	// Post-stream x-velocity at position P = (xm, y, z) (see the pull scheme
	// for the calling context).
	template <typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void streamingVx(LBM_DATA& SD, LBM_KS& KS, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm, idx z, idx zp)
	{
		dreal vx = 0;
		for (int i = 0; i < 27; i++)
			vx += dir27_cx(i) * TNL::Backend::ldg(SD.df(df_cur, i, xm, y, z));
		KS.vx = vx;
	}

	// Post-stream y-velocity at position P = (xm, y, z) (see the pull scheme
	// for the calling context).
	template <typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void streamingVy(LBM_DATA& SD, LBM_KS& KS, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm, idx z, idx zp)
	{
		dreal vy = 0;
		for (int i = 0; i < 27; i++)
			vy += dir27_cy(i) * TNL::Backend::ldg(SD.df(df_cur, i, xm, y, z));
		KS.vy = vy;
	}

	// Post-stream z-velocity at position P = (xm, y, z) (see the pull scheme
	// for the calling context).
	template <typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void streamingVz(LBM_DATA& SD, LBM_KS& KS, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm, idx z, idx zp)
	{
		dreal vz = 0;
		for (int i = 0; i < 27; i++)
			vz += dir27_cz(i) * TNL::Backend::ldg(SD.df(df_cur, i, xm, y, z));
		KS.vz = vz;
	}

	// Outflow-pass gather for an arbitrary face, in the post-stream layout
	// (see the file header): the pull-scheme site (anchor, tangential -c_i) is
	// found in slot (i, anchor + c_i[normal], tangential own). The pass reads
	// only the finalized previous launch, so there is no race against the
	// df_out writes of the current one.
	// FACE is a compile-time template parameter, so the per-direction
	// components and site offsets fold to constants.
	template <int FACE, typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void streamingOutflowImpl(LBM_DATA& SD, LBM_KS& KS, idx anchor, idx x, idx y, idx z)
	{
		constexpr int axis = (FACE & (bc_face::XP | bc_face::XM)) ? 0
						   : (FACE & (bc_face::YP | bc_face::YM)) ? 1
																  : 2;	// normal axis: 0 = x, 1 = y, 2 = z
		for (int i = 0; i < 27; i++) {
			idx sx = x, sy = y, sz = z;
			if constexpr (axis == 0)
				sx = anchor + dir27_cx(i);
			else if constexpr (axis == 1)
				sy = anchor + dir27_cy(i);
			else
				sz = anchor + dir27_cz(i);
			KS.f[i] = TNL::Backend::ldg(SD.df(df_cur, i, sx, sy, sz));
		}
	}

	template <typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void
	streamingOutflow(LBM_DATA& SD, LBM_KS& KS, int face, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm, idx z, idx zp)
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
			case bc_face::YM:
				streamingOutflowImpl<bc_face::YM>(SD, KS, yp, x, y, z);
				break;
			case bc_face::ZP:
				streamingOutflowImpl<bc_face::ZP>(SD, KS, zm, x, y, z);
				break;
			default:
				streamingOutflowImpl<bc_face::ZM>(SD, KS, zp, x, y, z);
				break;
		}
	}

	// interpolated-outflow blend in the pinned lbm_fma_rn form:
	// the first site delivers the anchor-column postcoll (weight cs),
	// the second site the own-column postcoll (weight 1-cs)
	template <typename LBM_DATA>
	__cuda_callable__ static dreal outflowInterpBlend(LBM_DATA& SD, int dir, idx ax, idx ay, idx az, idx bx, idx by, idx bz)
	{
		// NOTE: velocity is neglected (for the case velocity << speed of sound)
		constexpr dreal SpeedOfSound = 0.5773502691896257;
		return lbm_fma_rn(
			SpeedOfSound, TNL::Backend::ldg(SD.df(df_cur, dir, ax, ay, az)), (1 - SpeedOfSound) * TNL::Backend::ldg(SD.df(df_cur, dir, bx, by, bz))
		);
	}

	// interpolated outflow (Geier 2015) for an arbitrary face, in the
	// post-stream layout: pulls' anchor/own-column sites (s) map to slots
	// (s + c_i), so both columns use the cell's own tangential coordinates and
	// the normal coordinate shifted by the normal component of c_i.
	template <int FACE, typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void streamingOutflowInterpImpl(LBM_DATA& SD, LBM_KS& KS, idx anchor, idx x, idx y, idx z)
	{
		constexpr int axis = (FACE & (bc_face::XP | bc_face::XM)) ? 0 : (FACE & (bc_face::YP | bc_face::YM)) ? 1 : 2;
		constexpr int out_sign = (FACE & (bc_face::XM | bc_face::YM | bc_face::ZM)) ? -1 : 1;
		for (int i = 0; i < 27; i++) {
			const int cn = (axis == 0) ? dir27_cx(i) : (axis == 1) ? dir27_cy(i) : dir27_cz(i);	 // normal component of c_i
			// mapped anchor-column slot and own-column slot
			idx nx = x, ny = y, nz = z, ox = x, oy = y, oz = z;
			if constexpr (axis == 0) {
				nx = anchor + cn;
				ox = x + cn;
			}
			else if constexpr (axis == 1) {
				ny = anchor + cn;
				oy = y + cn;
			}
			else {
				nz = anchor + cn;
				oz = z + cn;
			}
			if (cn == out_sign)
				KS.f[i] = TNL::Backend::ldg(SD.df(df_cur, i, nx, ny, nz));
			else if (cn == 0)
				KS.f[i] = TNL::Backend::ldg(SD.df(df_cur, i, ox, oy, oz));
			else
				KS.f[i] = outflowInterpBlend(SD, i, nx, ny, nz, ox, oy, oz);
		}
	}

	template <typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void
	streamingOutflowInterp(LBM_DATA& SD, LBM_KS& KS, int face, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm, idx z, idx zp)
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
			case bc_face::YM:
				streamingOutflowInterpImpl<bc_face::YM>(SD, KS, yp, x, y, z);
				break;
			case bc_face::ZP:
				streamingOutflowInterpImpl<bc_face::ZP>(SD, KS, zm, x, y, z);
				break;
			default:
				streamingOutflowInterpImpl<bc_face::ZM>(SD, KS, zp, x, y, z);
				break;
		}
	}

	// Adjoint "reversed" streaming. NOTE: the adjoint path is tied to the pull
	// layout (the adjoint simulations pin D3Q27_STREAMING_AB_PULL) and is NOT
	// validated for the push pattern; this body mirrors the pull gather so the
	// BC dispatch compiles for push configs.
	template <typename LBM_DATA, typename LBM_KS>
	CUDA_HOSTDEV static void
	streamingAdjoint(uint8_t type, LBM_DATA& SD, LBM_KS& KS, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm, idx z, idx zp)
	{
		KS.f[mmm] = TNL::Backend::ldg(SD.df(type, mmm, xm, ym, zm));
		KS.f[mmz] = TNL::Backend::ldg(SD.df(type, mmz, xm, ym, z));
		KS.f[mmp] = TNL::Backend::ldg(SD.df(type, mmp, xm, ym, zp));
		KS.f[mzm] = TNL::Backend::ldg(SD.df(type, mzm, xm, y, zm));
		KS.f[mzz] = TNL::Backend::ldg(SD.df(type, mzz, xm, y, z));
		KS.f[mzp] = TNL::Backend::ldg(SD.df(type, mzp, xm, y, zp));
		KS.f[mpm] = TNL::Backend::ldg(SD.df(type, mpm, xm, yp, zm));
		KS.f[mpz] = TNL::Backend::ldg(SD.df(type, mpz, xm, yp, z));
		KS.f[mpp] = TNL::Backend::ldg(SD.df(type, mpp, xm, yp, zp));
		KS.f[zmm] = TNL::Backend::ldg(SD.df(type, zmm, x, ym, zm));
		KS.f[zmz] = TNL::Backend::ldg(SD.df(type, zmz, x, ym, z));
		KS.f[zmp] = TNL::Backend::ldg(SD.df(type, zmp, x, ym, zp));
		KS.f[zzm] = TNL::Backend::ldg(SD.df(type, zzm, x, y, zm));
		KS.f[zzz] = TNL::Backend::ldg(SD.df(type, zzz, x, y, z));
		KS.f[zzp] = TNL::Backend::ldg(SD.df(type, zzp, x, y, zp));
		KS.f[zpm] = TNL::Backend::ldg(SD.df(type, zpm, x, yp, zm));
		KS.f[zpz] = TNL::Backend::ldg(SD.df(type, zpz, x, yp, z));
		KS.f[zpp] = TNL::Backend::ldg(SD.df(type, zpp, x, yp, zp));
		KS.f[pmm] = TNL::Backend::ldg(SD.df(type, pmm, xp, ym, zm));
		KS.f[pmz] = TNL::Backend::ldg(SD.df(type, pmz, xp, ym, z));
		KS.f[pmp] = TNL::Backend::ldg(SD.df(type, pmp, xp, ym, zp));
		KS.f[pzm] = TNL::Backend::ldg(SD.df(type, pzm, xp, y, zm));
		KS.f[pzz] = TNL::Backend::ldg(SD.df(type, pzz, xp, y, z));
		KS.f[pzp] = TNL::Backend::ldg(SD.df(type, pzp, xp, y, zp));
		KS.f[ppm] = TNL::Backend::ldg(SD.df(type, ppm, xp, yp, zm));
		KS.f[ppz] = TNL::Backend::ldg(SD.df(type, ppz, xp, yp, z));
		KS.f[ppp] = TNL::Backend::ldg(SD.df(type, ppp, xp, yp, zp));
	}

	template <typename LBM_DATA, typename LBM_KS>
	CUDA_HOSTDEV static void streamingAdjoint(LBM_DATA& SD, LBM_KS& KS, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm, idx z, idx zp)
	{
		streamingAdjoint(df_cur, SD, KS, xm, x, xp, ym, y, yp, zm, z, zp);
	}
};

template <typename TRAITS>
inline constexpr bool is_AB_PUSH_v<D3Q27_STREAMING_AB_PUSH<TRAITS>> = true;
