#pragma once

#include "lbm3d/defs.h"
#include "lbm_common/rounding.h"

// A-A pattern
template <typename TRAITS>
struct D3Q27_STREAMING_AA
{
	static constexpr int DFMAX = 1;
	// DF slot that holds the freshly written field after a kernel launch
	static constexpr std::uint8_t output_df = df_cur;

	using idx = typename TRAITS::idx;
	using dreal = typename TRAITS::dreal;

	template <typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void
	postCollisionStreaming(LBM_DATA& SD, LBM_KS& KS, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm, idx z, idx zp)
	{
		if (SD.even_iter) {
			// write to the same lattice site, but the opposite DF direction
			SD.df(df_cur, dir27::mmm, x, y, z) = KS.f[dir27::ppp];
			SD.df(df_cur, dir27::mmz, x, y, z) = KS.f[dir27::ppz];
			SD.df(df_cur, dir27::mmp, x, y, z) = KS.f[dir27::ppm];
			SD.df(df_cur, dir27::mzm, x, y, z) = KS.f[dir27::pzp];
			SD.df(df_cur, dir27::mzz, x, y, z) = KS.f[dir27::pzz];
			SD.df(df_cur, dir27::mzp, x, y, z) = KS.f[dir27::pzm];
			SD.df(df_cur, dir27::mpm, x, y, z) = KS.f[dir27::pmp];
			SD.df(df_cur, dir27::mpz, x, y, z) = KS.f[dir27::pmz];
			SD.df(df_cur, dir27::mpp, x, y, z) = KS.f[dir27::pmm];
			SD.df(df_cur, dir27::zmm, x, y, z) = KS.f[dir27::zpp];
			SD.df(df_cur, dir27::zmz, x, y, z) = KS.f[dir27::zpz];
			SD.df(df_cur, dir27::zmp, x, y, z) = KS.f[dir27::zpm];
			SD.df(df_cur, dir27::zzm, x, y, z) = KS.f[dir27::zzp];
			SD.df(df_cur, dir27::zzz, x, y, z) = KS.f[dir27::zzz];
			SD.df(df_cur, dir27::zzp, x, y, z) = KS.f[dir27::zzm];
			SD.df(df_cur, dir27::zpm, x, y, z) = KS.f[dir27::zmp];
			SD.df(df_cur, dir27::zpz, x, y, z) = KS.f[dir27::zmz];
			SD.df(df_cur, dir27::zpp, x, y, z) = KS.f[dir27::zmm];
			SD.df(df_cur, dir27::pmm, x, y, z) = KS.f[dir27::mpp];
			SD.df(df_cur, dir27::pmz, x, y, z) = KS.f[dir27::mpz];
			SD.df(df_cur, dir27::pmp, x, y, z) = KS.f[dir27::mpm];
			SD.df(df_cur, dir27::pzm, x, y, z) = KS.f[dir27::mzp];
			SD.df(df_cur, dir27::pzz, x, y, z) = KS.f[dir27::mzz];
			SD.df(df_cur, dir27::pzp, x, y, z) = KS.f[dir27::mzm];
			SD.df(df_cur, dir27::ppm, x, y, z) = KS.f[dir27::mmp];
			SD.df(df_cur, dir27::ppz, x, y, z) = KS.f[dir27::mmz];
			SD.df(df_cur, dir27::ppp, x, y, z) = KS.f[dir27::mmm];
		}
		else {
			// write to the neighboring lattice sites, same DF direction
			SD.df(df_cur, dir27::ppp, xp, yp, zp) = KS.f[dir27::ppp];
			SD.df(df_cur, dir27::ppz, xp, yp, z) = KS.f[dir27::ppz];
			SD.df(df_cur, dir27::ppm, xp, yp, zm) = KS.f[dir27::ppm];
			SD.df(df_cur, dir27::pzp, xp, y, zp) = KS.f[dir27::pzp];
			SD.df(df_cur, dir27::pzz, xp, y, z) = KS.f[dir27::pzz];
			SD.df(df_cur, dir27::pzm, xp, y, zm) = KS.f[dir27::pzm];
			SD.df(df_cur, dir27::pmp, xp, ym, zp) = KS.f[dir27::pmp];
			SD.df(df_cur, dir27::pmz, xp, ym, z) = KS.f[dir27::pmz];
			SD.df(df_cur, dir27::pmm, xp, ym, zm) = KS.f[dir27::pmm];
			SD.df(df_cur, dir27::zpp, x, yp, zp) = KS.f[dir27::zpp];
			SD.df(df_cur, dir27::zpz, x, yp, z) = KS.f[dir27::zpz];
			SD.df(df_cur, dir27::zpm, x, yp, zm) = KS.f[dir27::zpm];
			SD.df(df_cur, dir27::zzp, x, y, zp) = KS.f[dir27::zzp];
			SD.df(df_cur, dir27::zzz, x, y, z) = KS.f[dir27::zzz];
			SD.df(df_cur, dir27::zzm, x, y, zm) = KS.f[dir27::zzm];
			SD.df(df_cur, dir27::zmp, x, ym, zp) = KS.f[dir27::zmp];
			SD.df(df_cur, dir27::zmz, x, ym, z) = KS.f[dir27::zmz];
			SD.df(df_cur, dir27::zmm, x, ym, zm) = KS.f[dir27::zmm];
			SD.df(df_cur, dir27::mpp, xm, yp, zp) = KS.f[dir27::mpp];
			SD.df(df_cur, dir27::mpz, xm, yp, z) = KS.f[dir27::mpz];
			SD.df(df_cur, dir27::mpm, xm, yp, zm) = KS.f[dir27::mpm];
			SD.df(df_cur, dir27::mzp, xm, y, zp) = KS.f[dir27::mzp];
			SD.df(df_cur, dir27::mzz, xm, y, z) = KS.f[dir27::mzz];
			SD.df(df_cur, dir27::mzm, xm, y, zm) = KS.f[dir27::mzm];
			SD.df(df_cur, dir27::mmp, xm, ym, zp) = KS.f[dir27::mmp];
			SD.df(df_cur, dir27::mmz, xm, ym, z) = KS.f[dir27::mmz];
			SD.df(df_cur, dir27::mmm, xm, ym, zm) = KS.f[dir27::mmm];
		}
	}

	template <typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void streaming(LBM_DATA& SD, LBM_KS& KS, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm, idx z, idx zp)
	{
		if (SD.even_iter) {
			// read from the same lattice site, same DF direction
			for (int i = 0; i < 27; i++)
				KS.f[i] = TNL::Backend::ldg(SD.df(df_cur, i, x, y, z));
		}
		else {
			// read from the neighboring lattice sites, but the opposite DF direction
			KS.f[dir27::mmm] = TNL::Backend::ldg(SD.df(df_cur, dir27::ppp, xp, yp, zp));
			KS.f[dir27::mmz] = TNL::Backend::ldg(SD.df(df_cur, dir27::ppz, xp, yp, z));
			KS.f[dir27::mmp] = TNL::Backend::ldg(SD.df(df_cur, dir27::ppm, xp, yp, zm));
			KS.f[dir27::mzm] = TNL::Backend::ldg(SD.df(df_cur, dir27::pzp, xp, y, zp));
			KS.f[dir27::mzz] = TNL::Backend::ldg(SD.df(df_cur, dir27::pzz, xp, y, z));
			KS.f[dir27::mzp] = TNL::Backend::ldg(SD.df(df_cur, dir27::pzm, xp, y, zm));
			KS.f[dir27::mpm] = TNL::Backend::ldg(SD.df(df_cur, dir27::pmp, xp, ym, zp));
			KS.f[dir27::mpz] = TNL::Backend::ldg(SD.df(df_cur, dir27::pmz, xp, ym, z));
			KS.f[dir27::mpp] = TNL::Backend::ldg(SD.df(df_cur, dir27::pmm, xp, ym, zm));
			KS.f[dir27::zmm] = TNL::Backend::ldg(SD.df(df_cur, dir27::zpp, x, yp, zp));
			KS.f[dir27::zmz] = TNL::Backend::ldg(SD.df(df_cur, dir27::zpz, x, yp, z));
			KS.f[dir27::zmp] = TNL::Backend::ldg(SD.df(df_cur, dir27::zpm, x, yp, zm));
			KS.f[dir27::zzm] = TNL::Backend::ldg(SD.df(df_cur, dir27::zzp, x, y, zp));
			KS.f[dir27::zzz] = TNL::Backend::ldg(SD.df(df_cur, dir27::zzz, x, y, z));
			KS.f[dir27::zzp] = TNL::Backend::ldg(SD.df(df_cur, dir27::zzm, x, y, zm));
			KS.f[dir27::zpm] = TNL::Backend::ldg(SD.df(df_cur, dir27::zmp, x, ym, zp));
			KS.f[dir27::zpz] = TNL::Backend::ldg(SD.df(df_cur, dir27::zmz, x, ym, z));
			KS.f[dir27::zpp] = TNL::Backend::ldg(SD.df(df_cur, dir27::zmm, x, ym, zm));
			KS.f[dir27::pmm] = TNL::Backend::ldg(SD.df(df_cur, dir27::mpp, xm, yp, zp));
			KS.f[dir27::pmz] = TNL::Backend::ldg(SD.df(df_cur, dir27::mpz, xm, yp, z));
			KS.f[dir27::pmp] = TNL::Backend::ldg(SD.df(df_cur, dir27::mpm, xm, yp, zm));
			KS.f[dir27::pzm] = TNL::Backend::ldg(SD.df(df_cur, dir27::mzp, xm, y, zp));
			KS.f[dir27::pzz] = TNL::Backend::ldg(SD.df(df_cur, dir27::mzz, xm, y, z));
			KS.f[dir27::pzp] = TNL::Backend::ldg(SD.df(df_cur, dir27::mzm, xm, y, zm));
			KS.f[dir27::ppm] = TNL::Backend::ldg(SD.df(df_cur, dir27::mmp, xm, ym, zp));
			KS.f[dir27::ppz] = TNL::Backend::ldg(SD.df(df_cur, dir27::mmz, xm, ym, z));
			KS.f[dir27::ppp] = TNL::Backend::ldg(SD.df(df_cur, dir27::mmm, xm, ym, zm));
		}
	}

	// Interpolation outflow from Geier - CuLBM (2015), velocity neglected.
	// Even: df_cur is natural (post-stream) — AB formula applies directly.
	// Odd: df_cur is twisted; the AA twist transform (dir→opp(dir),
	// site→site+vel(dir)) collapses all y,z to (y,z).  -x dirs interpolate
	// opp(dir) between (xmm,y,z) and (xm,y,z); z/+x dirs read opp(dir) at (x,y,z).
	//
	// NOTE: the odd branch is NOT bit-equivalent to AB for outflow fields that
	// vary in y or z — the twist keeps y,z at the cell (y,z) where AB samples at
	// the pull-source (e.g. (ym,zp) for mpm) and shifts the x interp stencil by
	// one cell.  For uniform or purely axially-varying outflow both patterns
	// agree.  This mirrors the D2Q9 AA implementation (deliberate).
	//
	// LIMITATION: this runs from BC preCollision in the main LBM kernel, where
	// postCollisionStreaming writes every df_cur slot of a site in the same
	// launch.  The interp reads cross sites (e.g. (xm,yp,zp) in the even branch,
	// (xmm,y,z) and 26 slots at (x,y,z) in the odd branch), so they race with
	// writes from other threads.

	// Adjoint "reversed" streaming.  AB reads df_cur[dir] from the neighbor
	// in direction +c_dir (opposite of forward which reads from -c_dir).
	//
	// Odd:  df_cur is twisted — A[opp(i)](s) = pre[i](s), so the adjoint reads
	//       pre[i](x + c_i) = A[opp(i)](x + c_i): one-step (safe in the main kernel).
	// Even: df_cur is natural — A[i](s) = post[i](s) = pre[i](s - c_i), so the
	//       adjoint reads pre[i](x + c_i) = A[i](x + 2*c_i): TWO-step.
	//
	// LIMITATION: the even-branch two-step reads race with postCollisionStreaming
	// in the main LBM kernel (see file header).  AA adjoint is EXPERIMENTAL.
	template <typename LBM_DATA, typename LBM_KS>
	CUDA_HOSTDEV static void
	streamingAdjoint(uint8_t type, LBM_DATA& SD, LBM_KS& KS, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm, idx z, idx zp)
	{
		static_cast<void>(type);
		if (SD.even_iter) {
			// A[i](s) = post[i](s) = pre[i](s - c_i); adjoint needs pre[i](x + c_i) = A[i](x + 2*c_i)
			const idx xmm = xm - 1;
			const idx xpp = xp + 1;
			const idx ymm = ym - 1;
			const idx ypp = yp + 1;
			const idx zmm = zm - 1;
			const idx zpp = zp + 1;
			KS.f[dir27::mmm] = TNL::Backend::ldg(SD.df(df_cur, dir27::mmm, xmm, ymm, zmm));
			KS.f[dir27::mmz] = TNL::Backend::ldg(SD.df(df_cur, dir27::mmz, xmm, ymm, z));
			KS.f[dir27::mmp] = TNL::Backend::ldg(SD.df(df_cur, dir27::mmp, xmm, ymm, zpp));
			KS.f[dir27::mzm] = TNL::Backend::ldg(SD.df(df_cur, dir27::mzm, xmm, y, zmm));
			KS.f[dir27::mzz] = TNL::Backend::ldg(SD.df(df_cur, dir27::mzz, xmm, y, z));
			KS.f[dir27::mzp] = TNL::Backend::ldg(SD.df(df_cur, dir27::mzp, xmm, y, zpp));
			KS.f[dir27::mpm] = TNL::Backend::ldg(SD.df(df_cur, dir27::mpm, xmm, ypp, zmm));
			KS.f[dir27::mpz] = TNL::Backend::ldg(SD.df(df_cur, dir27::mpz, xmm, ypp, z));
			KS.f[dir27::mpp] = TNL::Backend::ldg(SD.df(df_cur, dir27::mpp, xmm, ypp, zpp));
			KS.f[dir27::zmm] = TNL::Backend::ldg(SD.df(df_cur, dir27::zmm, x, ymm, zmm));
			KS.f[dir27::zmz] = TNL::Backend::ldg(SD.df(df_cur, dir27::zmz, x, ymm, z));
			KS.f[dir27::zmp] = TNL::Backend::ldg(SD.df(df_cur, dir27::zmp, x, ymm, zpp));
			KS.f[dir27::zzm] = TNL::Backend::ldg(SD.df(df_cur, dir27::zzm, x, y, zmm));
			KS.f[dir27::zzz] = TNL::Backend::ldg(SD.df(df_cur, dir27::zzz, x, y, z));
			KS.f[dir27::zzp] = TNL::Backend::ldg(SD.df(df_cur, dir27::zzp, x, y, zpp));
			KS.f[dir27::zpm] = TNL::Backend::ldg(SD.df(df_cur, dir27::zpm, x, ypp, zmm));
			KS.f[dir27::zpz] = TNL::Backend::ldg(SD.df(df_cur, dir27::zpz, x, ypp, z));
			KS.f[dir27::zpp] = TNL::Backend::ldg(SD.df(df_cur, dir27::zpp, x, ypp, zpp));
			KS.f[dir27::pmm] = TNL::Backend::ldg(SD.df(df_cur, dir27::pmm, xpp, ymm, zmm));
			KS.f[dir27::pmz] = TNL::Backend::ldg(SD.df(df_cur, dir27::pmz, xpp, ymm, z));
			KS.f[dir27::pmp] = TNL::Backend::ldg(SD.df(df_cur, dir27::pmp, xpp, ymm, zpp));
			KS.f[dir27::pzm] = TNL::Backend::ldg(SD.df(df_cur, dir27::pzm, xpp, y, zmm));
			KS.f[dir27::pzz] = TNL::Backend::ldg(SD.df(df_cur, dir27::pzz, xpp, y, z));
			KS.f[dir27::pzp] = TNL::Backend::ldg(SD.df(df_cur, dir27::pzp, xpp, y, zpp));
			KS.f[dir27::ppm] = TNL::Backend::ldg(SD.df(df_cur, dir27::ppm, xpp, ypp, zmm));
			KS.f[dir27::ppz] = TNL::Backend::ldg(SD.df(df_cur, dir27::ppz, xpp, ypp, z));
			KS.f[dir27::ppp] = TNL::Backend::ldg(SD.df(df_cur, dir27::ppp, xpp, ypp, zpp));
		}
		else {
			KS.f[dir27::mmm] = TNL::Backend::ldg(SD.df(df_cur, dir27::ppp, xm, ym, zm));
			KS.f[dir27::mmz] = TNL::Backend::ldg(SD.df(df_cur, dir27::ppz, xm, ym, z));
			KS.f[dir27::mmp] = TNL::Backend::ldg(SD.df(df_cur, dir27::ppm, xm, ym, zp));
			KS.f[dir27::mzm] = TNL::Backend::ldg(SD.df(df_cur, dir27::pzp, xm, y, zm));
			KS.f[dir27::mzz] = TNL::Backend::ldg(SD.df(df_cur, dir27::pzz, xm, y, z));
			KS.f[dir27::mzp] = TNL::Backend::ldg(SD.df(df_cur, dir27::pzm, xm, y, zp));
			KS.f[dir27::mpm] = TNL::Backend::ldg(SD.df(df_cur, dir27::pmp, xm, yp, zm));
			KS.f[dir27::mpz] = TNL::Backend::ldg(SD.df(df_cur, dir27::pmz, xm, yp, z));
			KS.f[dir27::mpp] = TNL::Backend::ldg(SD.df(df_cur, dir27::pmm, xm, yp, zp));
			KS.f[dir27::zmm] = TNL::Backend::ldg(SD.df(df_cur, dir27::zpp, x, ym, zm));
			KS.f[dir27::zmz] = TNL::Backend::ldg(SD.df(df_cur, dir27::zpz, x, ym, z));
			KS.f[dir27::zmp] = TNL::Backend::ldg(SD.df(df_cur, dir27::zpm, x, ym, zp));
			KS.f[dir27::zzm] = TNL::Backend::ldg(SD.df(df_cur, dir27::zzp, x, y, zm));
			KS.f[dir27::zzz] = TNL::Backend::ldg(SD.df(df_cur, dir27::zzz, x, y, z));
			KS.f[dir27::zzp] = TNL::Backend::ldg(SD.df(df_cur, dir27::zzm, x, y, zp));
			KS.f[dir27::zpm] = TNL::Backend::ldg(SD.df(df_cur, dir27::zmp, x, yp, zm));
			KS.f[dir27::zpz] = TNL::Backend::ldg(SD.df(df_cur, dir27::zmz, x, yp, z));
			KS.f[dir27::zpp] = TNL::Backend::ldg(SD.df(df_cur, dir27::zmm, x, yp, zp));
			KS.f[dir27::pmm] = TNL::Backend::ldg(SD.df(df_cur, dir27::mpp, xp, ym, zm));
			KS.f[dir27::pmz] = TNL::Backend::ldg(SD.df(df_cur, dir27::mpz, xp, ym, z));
			KS.f[dir27::pmp] = TNL::Backend::ldg(SD.df(df_cur, dir27::mpm, xp, ym, zp));
			KS.f[dir27::pzm] = TNL::Backend::ldg(SD.df(df_cur, dir27::mzp, xp, y, zm));
			KS.f[dir27::pzz] = TNL::Backend::ldg(SD.df(df_cur, dir27::mzz, xp, y, z));
			KS.f[dir27::pzp] = TNL::Backend::ldg(SD.df(df_cur, dir27::mzm, xp, y, zp));
			KS.f[dir27::ppm] = TNL::Backend::ldg(SD.df(df_cur, dir27::mmp, xp, yp, zm));
			KS.f[dir27::ppz] = TNL::Backend::ldg(SD.df(df_cur, dir27::mmz, xp, yp, z));
			KS.f[dir27::ppp] = TNL::Backend::ldg(SD.df(df_cur, dir27::mmm, xp, yp, zp));
		}
	}

	template <typename LBM_DATA, typename LBM_KS>
	CUDA_HOSTDEV static void streamingAdjoint(LBM_DATA& SD, LBM_KS& KS, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm, idx z, idx zp)
	{
		streamingAdjoint(df_cur, SD, KS, xm, x, xp, ym, y, yp, zm, z, zp);
	}

	// Gathers for the outflow-pass processing (a separate kernel launched
	// before the main one for deterministic outflow handling - the outflow
	// pass is a type of processing, not a BC type), parameterized by the
	// outflow face (outward normal): both branches reconstruct
	// the pre-collision populations at the translated A-B pull sites of the outflow
	// cell: postcoll_{n-1}(i) at site (anchor, tangential -c_i), where the anchor is
	// the fluid-side neighbor column one cell inward.
	// The slot layouts provide this only from the finalized previous launch:
	// the required slots are owned by the pre-anchor/interior threads in both
	// parities, so no race-free in-launch gather exists and the pass must run
	// before the main kernel.
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
		if (SD.even_iter) {
			// natural layout: slot (i, t + c_i) = postcoll_{n-1}(i, t) with
			// t = (anchor, tangential -c_i); the tangential offsets cancel
			// against +c_i, so the normal coordinate is anchor + c_i[normal]
			// and the tangential coordinates are the cell's own
			for (int i = 0; i < 27; i++) {
				idx sx;
				idx sy;
				idx sz;
				if constexpr (axis == 0) {
					sx = anchor + dir27_cx(i);
					sy = y;
					sz = z;
				}
				else if constexpr (axis == 1) {
					sx = x;
					sy = anchor + dir27_cy(i);
					sz = z;
				}
				else {
					sx = x;
					sy = y;
					sz = anchor + dir27_cz(i);
				}
				KS.f[i] = TNL::Backend::ldg(SD.df(df_cur, i, sx, sy, sz));
			}
		}
		else {
			// twist layout: slot (opp(i), t) = postcoll_{n-1}(i, t)
			for (int i = 0; i < 27; i++) {
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
				KS.f[i] = TNL::Backend::ldg(SD.df(df_cur, opposite_direction(i), sx, sy, sz));
			}
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

	// interpolated outflow (Geier 2015) for an arbitrary face: reproduces the
	// legacy fused arithmetic exactly, site-translated like the streamingOutflow
	// gather above. The population moving against the outward normal blends
	// postcoll_{n-1} from the anchor column with the outflow cell's own postcoll,
	// the perpendicular population takes the cell's own postcoll, the outward-
	// moving population comes from the anchor column; all of it is previous-launch
	// state finalized before the pass runs; the tangential coordinates select among
	// the wrapped neighbors like in the plain gather above.
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
		if (SD.even_iter) {
			// natural layout: the outward- and perpendicular-moving populations
			// take the cell's own postcoll, the inward-moving population blends
			// postcoll_{n-1} from the pre-anchor column with the anchor column
			for (int i = 0; i < 27; i++) {
				const int cn = (axis == 0) ? dir27_cx(i) : (axis == 1) ? dir27_cy(i) : dir27_cz(i);	 // normal component of c_i
				if (cn == out_sign || cn == 0)
					KS.f[i] = TNL::Backend::ldg(SD.df(df_cur, i, x, y, z));
				else if constexpr (axis == 0)
					KS.f[i] = outflowInterpBlend(SD, i, anchor + dir27_cx(i), y, z, anchor, y, z);
				else if constexpr (axis == 1)
					KS.f[i] = outflowInterpBlend(SD, i, x, anchor + dir27_cy(i), z, x, anchor, z);
				else
					KS.f[i] = outflowInterpBlend(SD, i, x, y, anchor + dir27_cz(i), x, y, anchor);
			}
		}
		else {
			// twist layout: the outward-moving population comes from the anchor
			// column, the perpendicular population from the own column, the
			// inward-moving population blends the anchor column with the own column
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
					KS.f[i] = TNL::Backend::ldg(SD.df(df_cur, opposite_direction(i), nx, ny, nz));
				else if (cn == 0)
					KS.f[i] = TNL::Backend::ldg(SD.df(df_cur, opposite_direction(i), ox, oy, oz));
				else
					KS.f[i] = outflowInterpBlend(SD, opposite_direction(i), nx, ny, nz, ox, oy, oz);
			}
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
};

template <typename TRAITS>
inline constexpr bool is_AA_v<D3Q27_STREAMING_AA<TRAITS>> = true;
template <typename TRAITS>
inline constexpr bool twisted_layout_v<D3Q27_STREAMING_AA<TRAITS>> = true;
template <typename TRAITS>
inline constexpr bool requires_ghost_layer_v<D3Q27_STREAMING_AA<TRAITS>> = true;
