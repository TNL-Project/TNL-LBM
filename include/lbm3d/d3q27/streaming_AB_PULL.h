#pragma once

#include "lbm3d/defs.h"
#include "lbm_common/rounding.h"

// pull-scheme
template <typename TRAITS>
struct D3Q27_STREAMING_AB_PULL
{
	static constexpr int DFMAX = 2;
	// DF slot that holds the freshly written field after a kernel launch
	static constexpr std::uint8_t output_df = df_out;

	using idx = typename TRAITS::idx;
	using dreal = typename TRAITS::dreal;

	template <typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void
	postCollisionStreaming(LBM_DATA& SD, LBM_KS& KS, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm, idx z, idx zp)
	{
		// no streaming actually, write to the (x,y,z) site
		for (int i = 0; i < 27; i++)
			SD.df(df_out, i, x, y, z) = KS.f[i];
	}

	template <typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void
	streaming(uint8_t type, LBM_DATA& SD, LBM_KS& KS, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm, idx z, idx zp)
	{
		KS.f[mmm] = TNL::Backend::ldg(SD.df(type, mmm, xp, yp, zp));
		KS.f[mmz] = TNL::Backend::ldg(SD.df(type, mmz, xp, yp, z));
		KS.f[mmp] = TNL::Backend::ldg(SD.df(type, mmp, xp, yp, zm));
		KS.f[mzm] = TNL::Backend::ldg(SD.df(type, mzm, xp, y, zp));
		KS.f[mzz] = TNL::Backend::ldg(SD.df(type, mzz, xp, y, z));
		KS.f[mzp] = TNL::Backend::ldg(SD.df(type, mzp, xp, y, zm));
		KS.f[mpm] = TNL::Backend::ldg(SD.df(type, mpm, xp, ym, zp));
		KS.f[mpz] = TNL::Backend::ldg(SD.df(type, mpz, xp, ym, z));
		KS.f[mpp] = TNL::Backend::ldg(SD.df(type, mpp, xp, ym, zm));
		KS.f[zmm] = TNL::Backend::ldg(SD.df(type, zmm, x, yp, zp));
		KS.f[zmz] = TNL::Backend::ldg(SD.df(type, zmz, x, yp, z));
		KS.f[zmp] = TNL::Backend::ldg(SD.df(type, zmp, x, yp, zm));
		KS.f[zzm] = TNL::Backend::ldg(SD.df(type, zzm, x, y, zp));
		KS.f[zzz] = TNL::Backend::ldg(SD.df(type, zzz, x, y, z));
		KS.f[zzp] = TNL::Backend::ldg(SD.df(type, zzp, x, y, zm));
		KS.f[zpm] = TNL::Backend::ldg(SD.df(type, zpm, x, ym, zp));
		KS.f[zpz] = TNL::Backend::ldg(SD.df(type, zpz, x, ym, z));
		KS.f[zpp] = TNL::Backend::ldg(SD.df(type, zpp, x, ym, zm));
		KS.f[pmm] = TNL::Backend::ldg(SD.df(type, pmm, xm, yp, zp));
		KS.f[pmz] = TNL::Backend::ldg(SD.df(type, pmz, xm, yp, z));
		KS.f[pmp] = TNL::Backend::ldg(SD.df(type, pmp, xm, yp, zm));
		KS.f[pzm] = TNL::Backend::ldg(SD.df(type, pzm, xm, y, zp));
		KS.f[pzz] = TNL::Backend::ldg(SD.df(type, pzz, xm, y, z));
		KS.f[pzp] = TNL::Backend::ldg(SD.df(type, pzp, xm, y, zm));
		KS.f[ppm] = TNL::Backend::ldg(SD.df(type, ppm, xm, ym, zp));
		KS.f[ppz] = TNL::Backend::ldg(SD.df(type, ppz, xm, ym, z));
		KS.f[ppp] = TNL::Backend::ldg(SD.df(type, ppp, xm, ym, zm));
	}

	template <typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void streaming(LBM_DATA& SD, LBM_KS& KS, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm, idx z, idx zp)
	{
		streaming(df_cur, SD, KS, xm, x, xp, ym, y, yp, zm, z, zp);
	}

	// outflow pass gather for an arbitrary face: the outflow cell takes the
	// pulled state of its anchor column (the fluid-side neighbor, one cell
	// inward) from df_cur (finalized by the previous launch, no race against
	// the df_out writes of the current one).
	// FACE is a compile-time template parameter, so the per-direction
	// components and site offsets fold to constants.
	template <int FACE, typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void streamingOutflowImpl(LBM_DATA& SD, LBM_KS& KS, idx anchor, idx x, idx y, idx z)
	{
		constexpr int axis = (FACE & (bc_face::XP | bc_face::XM)) ? 0
						   : (FACE & (bc_face::YP | bc_face::YM)) ? 1
																  : 2;	// normal axis: 0 = x, 1 = y, 2 = z
		for (int i = 0; i < 27; i++) {
			// normal coordinate: the anchor column; tangential: -c offsets (pull scheme)
			idx sx, sy, sz;
			if constexpr (axis == 0) {
				sx = anchor;
				sy = y - dir27_cy(i);
				sz = z - dir27_cz(i);
			}
			else if constexpr (axis == 1) {
				sx = x - dir27_cx(i);
				sy = anchor;
				sz = z - dir27_cz(i);
			}
			else {
				sx = x - dir27_cx(i);
				sy = y - dir27_cy(i);
				sz = anchor;
			}
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

	// interpolated outflow (Geier 2015) for an arbitrary face: the population
	// moving against the outward normal blends postcoll_{n-1} from the anchor
	// column with the outflow cell's own postcoll, the perpendicular population
	// streams ordinarily (own column), the outward-moving population takes the
	// pulled state of the anchor column
	template <int FACE, typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void streamingOutflowInterpImpl(LBM_DATA& SD, LBM_KS& KS, idx anchor, idx x, idx y, idx z)
	{
		constexpr int axis = (FACE & (bc_face::XP | bc_face::XM)) ? 0 : (FACE & (bc_face::YP | bc_face::YM)) ? 1 : 2;
		constexpr int out_sign = (FACE & (bc_face::XM | bc_face::YM | bc_face::ZM)) ? -1 : 1;
		for (int i = 0; i < 27; i++) {
			const int cn = (axis == 0) ? dir27_cx(i) : (axis == 1) ? dir27_cy(i) : dir27_cz(i);	 // normal component of c_i
			// site in the anchor column and site in the own column, tangential -c offsets
			idx nx, ny, nz, ox, oy, oz;
			if constexpr (axis == 0) {
				nx = anchor;
				ny = y - dir27_cy(i);
				nz = z - dir27_cz(i);
				ox = x;
				oy = y - dir27_cy(i);
				oz = z - dir27_cz(i);
			}
			else if constexpr (axis == 1) {
				nx = x - dir27_cx(i);
				ny = anchor;
				nz = z - dir27_cz(i);
				ox = x - dir27_cx(i);
				oy = y;
				oz = z - dir27_cz(i);
			}
			else {
				nx = x - dir27_cx(i);
				ny = y - dir27_cy(i);
				nz = anchor;
				ox = x - dir27_cx(i);
				oy = y - dir27_cy(i);
				oz = z;
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

	// ADJOINT -- "reversed" streaming
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
inline constexpr bool is_AB_PULL_v<D3Q27_STREAMING_AB_PULL<TRAITS>> = true;
