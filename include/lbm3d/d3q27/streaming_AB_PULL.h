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
		KS.f[dir27::mmm] = TNL::Backend::ldg(SD.df(type, dir27::mmm, xp, yp, zp));
		KS.f[dir27::mmz] = TNL::Backend::ldg(SD.df(type, dir27::mmz, xp, yp, z));
		KS.f[dir27::mmp] = TNL::Backend::ldg(SD.df(type, dir27::mmp, xp, yp, zm));
		KS.f[dir27::mzm] = TNL::Backend::ldg(SD.df(type, dir27::mzm, xp, y, zp));
		KS.f[dir27::mzz] = TNL::Backend::ldg(SD.df(type, dir27::mzz, xp, y, z));
		KS.f[dir27::mzp] = TNL::Backend::ldg(SD.df(type, dir27::mzp, xp, y, zm));
		KS.f[dir27::mpm] = TNL::Backend::ldg(SD.df(type, dir27::mpm, xp, ym, zp));
		KS.f[dir27::mpz] = TNL::Backend::ldg(SD.df(type, dir27::mpz, xp, ym, z));
		KS.f[dir27::mpp] = TNL::Backend::ldg(SD.df(type, dir27::mpp, xp, ym, zm));
		KS.f[dir27::zmm] = TNL::Backend::ldg(SD.df(type, dir27::zmm, x, yp, zp));
		KS.f[dir27::zmz] = TNL::Backend::ldg(SD.df(type, dir27::zmz, x, yp, z));
		KS.f[dir27::zmp] = TNL::Backend::ldg(SD.df(type, dir27::zmp, x, yp, zm));
		KS.f[dir27::zzm] = TNL::Backend::ldg(SD.df(type, dir27::zzm, x, y, zp));
		KS.f[dir27::zzz] = TNL::Backend::ldg(SD.df(type, dir27::zzz, x, y, z));
		KS.f[dir27::zzp] = TNL::Backend::ldg(SD.df(type, dir27::zzp, x, y, zm));
		KS.f[dir27::zpm] = TNL::Backend::ldg(SD.df(type, dir27::zpm, x, ym, zp));
		KS.f[dir27::zpz] = TNL::Backend::ldg(SD.df(type, dir27::zpz, x, ym, z));
		KS.f[dir27::zpp] = TNL::Backend::ldg(SD.df(type, dir27::zpp, x, ym, zm));
		KS.f[dir27::pmm] = TNL::Backend::ldg(SD.df(type, dir27::pmm, xm, yp, zp));
		KS.f[dir27::pmz] = TNL::Backend::ldg(SD.df(type, dir27::pmz, xm, yp, z));
		KS.f[dir27::pmp] = TNL::Backend::ldg(SD.df(type, dir27::pmp, xm, yp, zm));
		KS.f[dir27::pzm] = TNL::Backend::ldg(SD.df(type, dir27::pzm, xm, y, zp));
		KS.f[dir27::pzz] = TNL::Backend::ldg(SD.df(type, dir27::pzz, xm, y, z));
		KS.f[dir27::pzp] = TNL::Backend::ldg(SD.df(type, dir27::pzp, xm, y, zm));
		KS.f[dir27::ppm] = TNL::Backend::ldg(SD.df(type, dir27::ppm, xm, ym, zp));
		KS.f[dir27::ppz] = TNL::Backend::ldg(SD.df(type, dir27::ppz, xm, ym, z));
		KS.f[dir27::ppp] = TNL::Backend::ldg(SD.df(type, dir27::ppp, xm, ym, zm));
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
	// components and site offsets fold to constants. Tangential coordinates
	// select among the kernel's already-wrapped neighbors (kernelInitIndices):
	// interior rows read the raw neighbor, periodic-seam rows the wrapped
	// neighbor, distributed axes the halo; boundary rows on non-periodic
	// tangential axes are unreachable (ghost-layer idiom stamps walls/
	// GEO_NOTHING there).
	template <int FACE, typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void streamingOutflowImpl(LBM_DATA& SD, LBM_KS& KS, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm, idx z, idx zp)
	{
		constexpr int axis = (FACE & (bc_face::XP | bc_face::XM)) ? 0
						   : (FACE & (bc_face::YP | bc_face::YM)) ? 1
																  : 2;	// normal axis: 0 = x, 1 = y, 2 = z
		// anchor column: the fluid-side neighbor one cell inward (opposite the outward normal)
		const idx anchor = (FACE == bc_face::XP) ? xm
						 : (FACE == bc_face::XM) ? xp
						 : (FACE == bc_face::YP) ? ym
						 : (FACE == bc_face::YM) ? yp
						 : (FACE == bc_face::ZP) ? zm
												 : zp;
		for (int i = 0; i < 27; i++) {
			// normal coordinate: the anchor column; tangential: -c offsets (pull scheme)
			idx sx;
			idx sy;
			idx sz;
			if constexpr (axis == 0) {
				sx = anchor;
				sy = dir27_cy(i) > 0 ? ym : (dir27_cy(i) < 0 ? yp : y);
				sz = dir27_cz(i) > 0 ? zm : (dir27_cz(i) < 0 ? zp : z);
			}
			else if constexpr (axis == 1) {
				sx = dir27_cx(i) > 0 ? xm : (dir27_cx(i) < 0 ? xp : x);
				sy = anchor;
				sz = dir27_cz(i) > 0 ? zm : (dir27_cz(i) < 0 ? zp : z);
			}
			else {
				sx = dir27_cx(i) > 0 ? xm : (dir27_cx(i) < 0 ? xp : x);
				sy = dir27_cy(i) > 0 ? ym : (dir27_cy(i) < 0 ? yp : y);
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
				streamingOutflowImpl<bc_face::XP>(SD, KS, xm, x, xp, ym, y, yp, zm, z, zp);
				break;
			case bc_face::XM:
				streamingOutflowImpl<bc_face::XM>(SD, KS, xm, x, xp, ym, y, yp, zm, z, zp);
				break;
			case bc_face::YP:
				streamingOutflowImpl<bc_face::YP>(SD, KS, xm, x, xp, ym, y, yp, zm, z, zp);
				break;
			case bc_face::YM:
				streamingOutflowImpl<bc_face::YM>(SD, KS, xm, x, xp, ym, y, yp, zm, z, zp);
				break;
			case bc_face::ZP:
				streamingOutflowImpl<bc_face::ZP>(SD, KS, xm, x, xp, ym, y, yp, zm, z, zp);
				break;
			default:
				streamingOutflowImpl<bc_face::ZM>(SD, KS, xm, x, xp, ym, y, yp, zm, z, zp);
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
	// pulled state of the anchor column; the tangential coordinates select
	// among the wrapped neighbors like in the plain gather above
	template <int FACE, typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void
	streamingOutflowInterpImpl(LBM_DATA& SD, LBM_KS& KS, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm, idx z, idx zp)
	{
		constexpr int axis = (FACE & (bc_face::XP | bc_face::XM)) ? 0 : (FACE & (bc_face::YP | bc_face::YM)) ? 1 : 2;
		constexpr int out_sign = (FACE & (bc_face::XM | bc_face::YM | bc_face::ZM)) ? -1 : 1;
		// anchor column: the fluid-side neighbor one cell inward (opposite the outward normal)
		const idx anchor = (FACE == bc_face::XP) ? xm
						 : (FACE == bc_face::XM) ? xp
						 : (FACE == bc_face::YP) ? ym
						 : (FACE == bc_face::YM) ? yp
						 : (FACE == bc_face::ZP) ? zm
												 : zp;
		for (int i = 0; i < 27; i++) {
			const int cn = (axis == 0) ? dir27_cx(i) : (axis == 1) ? dir27_cy(i) : dir27_cz(i);	 // normal component of c_i
			// site in the anchor column and site in the own column, tangential -c offsets
			idx nx;
			idx ny;
			idx nz;
			idx ox;
			idx oy;
			idx oz;
			if constexpr (axis == 0) {
				nx = anchor;
				ny = dir27_cy(i) > 0 ? ym : (dir27_cy(i) < 0 ? yp : y);
				nz = dir27_cz(i) > 0 ? zm : (dir27_cz(i) < 0 ? zp : z);
				ox = x;
				oy = ny;
				oz = nz;
			}
			else if constexpr (axis == 1) {
				nx = dir27_cx(i) > 0 ? xm : (dir27_cx(i) < 0 ? xp : x);
				ny = anchor;
				nz = dir27_cz(i) > 0 ? zm : (dir27_cz(i) < 0 ? zp : z);
				ox = nx;
				oy = y;
				oz = nz;
			}
			else {
				nx = dir27_cx(i) > 0 ? xm : (dir27_cx(i) < 0 ? xp : x);
				ny = dir27_cy(i) > 0 ? ym : (dir27_cy(i) < 0 ? yp : y);
				nz = anchor;
				ox = nx;
				oy = ny;
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
				streamingOutflowInterpImpl<bc_face::XP>(SD, KS, xm, x, xp, ym, y, yp, zm, z, zp);
				break;
			case bc_face::XM:
				streamingOutflowInterpImpl<bc_face::XM>(SD, KS, xm, x, xp, ym, y, yp, zm, z, zp);
				break;
			case bc_face::YP:
				streamingOutflowInterpImpl<bc_face::YP>(SD, KS, xm, x, xp, ym, y, yp, zm, z, zp);
				break;
			case bc_face::YM:
				streamingOutflowInterpImpl<bc_face::YM>(SD, KS, xm, x, xp, ym, y, yp, zm, z, zp);
				break;
			case bc_face::ZP:
				streamingOutflowInterpImpl<bc_face::ZP>(SD, KS, xm, x, xp, ym, y, yp, zm, z, zp);
				break;
			default:
				streamingOutflowInterpImpl<bc_face::ZM>(SD, KS, xm, x, xp, ym, y, yp, zm, z, zp);
				break;
		}
	}

	// ADJOINT -- "reversed" streaming
	template <typename LBM_DATA, typename LBM_KS>
	CUDA_HOSTDEV static void
	streamingAdjoint(uint8_t type, LBM_DATA& SD, LBM_KS& KS, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm, idx z, idx zp)
	{
		KS.f[dir27::mmm] = TNL::Backend::ldg(SD.df(type, dir27::mmm, xm, ym, zm));
		KS.f[dir27::mmz] = TNL::Backend::ldg(SD.df(type, dir27::mmz, xm, ym, z));
		KS.f[dir27::mmp] = TNL::Backend::ldg(SD.df(type, dir27::mmp, xm, ym, zp));
		KS.f[dir27::mzm] = TNL::Backend::ldg(SD.df(type, dir27::mzm, xm, y, zm));
		KS.f[dir27::mzz] = TNL::Backend::ldg(SD.df(type, dir27::mzz, xm, y, z));
		KS.f[dir27::mzp] = TNL::Backend::ldg(SD.df(type, dir27::mzp, xm, y, zp));
		KS.f[dir27::mpm] = TNL::Backend::ldg(SD.df(type, dir27::mpm, xm, yp, zm));
		KS.f[dir27::mpz] = TNL::Backend::ldg(SD.df(type, dir27::mpz, xm, yp, z));
		KS.f[dir27::mpp] = TNL::Backend::ldg(SD.df(type, dir27::mpp, xm, yp, zp));
		KS.f[dir27::zmm] = TNL::Backend::ldg(SD.df(type, dir27::zmm, x, ym, zm));
		KS.f[dir27::zmz] = TNL::Backend::ldg(SD.df(type, dir27::zmz, x, ym, z));
		KS.f[dir27::zmp] = TNL::Backend::ldg(SD.df(type, dir27::zmp, x, ym, zp));
		KS.f[dir27::zzm] = TNL::Backend::ldg(SD.df(type, dir27::zzm, x, y, zm));
		KS.f[dir27::zzz] = TNL::Backend::ldg(SD.df(type, dir27::zzz, x, y, z));
		KS.f[dir27::zzp] = TNL::Backend::ldg(SD.df(type, dir27::zzp, x, y, zp));
		KS.f[dir27::zpm] = TNL::Backend::ldg(SD.df(type, dir27::zpm, x, yp, zm));
		KS.f[dir27::zpz] = TNL::Backend::ldg(SD.df(type, dir27::zpz, x, yp, z));
		KS.f[dir27::zpp] = TNL::Backend::ldg(SD.df(type, dir27::zpp, x, yp, zp));
		KS.f[dir27::pmm] = TNL::Backend::ldg(SD.df(type, dir27::pmm, xp, ym, zm));
		KS.f[dir27::pmz] = TNL::Backend::ldg(SD.df(type, dir27::pmz, xp, ym, z));
		KS.f[dir27::pmp] = TNL::Backend::ldg(SD.df(type, dir27::pmp, xp, ym, zp));
		KS.f[dir27::pzm] = TNL::Backend::ldg(SD.df(type, dir27::pzm, xp, y, zm));
		KS.f[dir27::pzz] = TNL::Backend::ldg(SD.df(type, dir27::pzz, xp, y, z));
		KS.f[dir27::pzp] = TNL::Backend::ldg(SD.df(type, dir27::pzp, xp, y, zp));
		KS.f[dir27::ppm] = TNL::Backend::ldg(SD.df(type, dir27::ppm, xp, yp, zm));
		KS.f[dir27::ppz] = TNL::Backend::ldg(SD.df(type, dir27::ppz, xp, yp, z));
		KS.f[dir27::ppp] = TNL::Backend::ldg(SD.df(type, dir27::ppp, xp, yp, zp));
	}

	template <typename LBM_DATA, typename LBM_KS>
	CUDA_HOSTDEV static void streamingAdjoint(LBM_DATA& SD, LBM_KS& KS, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm, idx z, idx zp)
	{
		streamingAdjoint(df_cur, SD, KS, xm, x, xp, ym, y, yp, zm, z, zp);
	}
};

template <typename TRAITS>
inline constexpr bool is_AB_PULL_v<D3Q27_STREAMING_AB_PULL<TRAITS>> = true;
