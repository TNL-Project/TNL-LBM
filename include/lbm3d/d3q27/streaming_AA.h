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
			SD.df(df_cur, mmm, x, y, z) = KS.f[ppp];
			SD.df(df_cur, mmz, x, y, z) = KS.f[ppz];
			SD.df(df_cur, mmp, x, y, z) = KS.f[ppm];
			SD.df(df_cur, mzm, x, y, z) = KS.f[pzp];
			SD.df(df_cur, mzz, x, y, z) = KS.f[pzz];
			SD.df(df_cur, mzp, x, y, z) = KS.f[pzm];
			SD.df(df_cur, mpm, x, y, z) = KS.f[pmp];
			SD.df(df_cur, mpz, x, y, z) = KS.f[pmz];
			SD.df(df_cur, mpp, x, y, z) = KS.f[pmm];
			SD.df(df_cur, zmm, x, y, z) = KS.f[zpp];
			SD.df(df_cur, zmz, x, y, z) = KS.f[zpz];
			SD.df(df_cur, zmp, x, y, z) = KS.f[zpm];
			SD.df(df_cur, zzm, x, y, z) = KS.f[zzp];
			SD.df(df_cur, zzz, x, y, z) = KS.f[zzz];
			SD.df(df_cur, zzp, x, y, z) = KS.f[zzm];
			SD.df(df_cur, zpm, x, y, z) = KS.f[zmp];
			SD.df(df_cur, zpz, x, y, z) = KS.f[zmz];
			SD.df(df_cur, zpp, x, y, z) = KS.f[zmm];
			SD.df(df_cur, pmm, x, y, z) = KS.f[mpp];
			SD.df(df_cur, pmz, x, y, z) = KS.f[mpz];
			SD.df(df_cur, pmp, x, y, z) = KS.f[mpm];
			SD.df(df_cur, pzm, x, y, z) = KS.f[mzp];
			SD.df(df_cur, pzz, x, y, z) = KS.f[mzz];
			SD.df(df_cur, pzp, x, y, z) = KS.f[mzm];
			SD.df(df_cur, ppm, x, y, z) = KS.f[mmp];
			SD.df(df_cur, ppz, x, y, z) = KS.f[mmz];
			SD.df(df_cur, ppp, x, y, z) = KS.f[mmm];
		}
		else {
			// write to the neighboring lattice sites, same DF direction
			SD.df(df_cur, ppp, xp, yp, zp) = KS.f[ppp];
			SD.df(df_cur, ppz, xp, yp, z) = KS.f[ppz];
			SD.df(df_cur, ppm, xp, yp, zm) = KS.f[ppm];
			SD.df(df_cur, pzp, xp, y, zp) = KS.f[pzp];
			SD.df(df_cur, pzz, xp, y, z) = KS.f[pzz];
			SD.df(df_cur, pzm, xp, y, zm) = KS.f[pzm];
			SD.df(df_cur, pmp, xp, ym, zp) = KS.f[pmp];
			SD.df(df_cur, pmz, xp, ym, z) = KS.f[pmz];
			SD.df(df_cur, pmm, xp, ym, zm) = KS.f[pmm];
			SD.df(df_cur, zpp, x, yp, zp) = KS.f[zpp];
			SD.df(df_cur, zpz, x, yp, z) = KS.f[zpz];
			SD.df(df_cur, zpm, x, yp, zm) = KS.f[zpm];
			SD.df(df_cur, zzp, x, y, zp) = KS.f[zzp];
			SD.df(df_cur, zzz, x, y, z) = KS.f[zzz];
			SD.df(df_cur, zzm, x, y, zm) = KS.f[zzm];
			SD.df(df_cur, zmp, x, ym, zp) = KS.f[zmp];
			SD.df(df_cur, zmz, x, ym, z) = KS.f[zmz];
			SD.df(df_cur, zmm, x, ym, zm) = KS.f[zmm];
			SD.df(df_cur, mpp, xm, yp, zp) = KS.f[mpp];
			SD.df(df_cur, mpz, xm, yp, z) = KS.f[mpz];
			SD.df(df_cur, mpm, xm, yp, zm) = KS.f[mpm];
			SD.df(df_cur, mzp, xm, y, zp) = KS.f[mzp];
			SD.df(df_cur, mzz, xm, y, z) = KS.f[mzz];
			SD.df(df_cur, mzm, xm, y, zm) = KS.f[mzm];
			SD.df(df_cur, mmp, xm, ym, zp) = KS.f[mmp];
			SD.df(df_cur, mmz, xm, ym, z) = KS.f[mmz];
			SD.df(df_cur, mmm, xm, ym, zm) = KS.f[mmm];
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
			KS.f[mmm] = TNL::Backend::ldg(SD.df(df_cur, ppp, xp, yp, zp));
			KS.f[mmz] = TNL::Backend::ldg(SD.df(df_cur, ppz, xp, yp, z));
			KS.f[mmp] = TNL::Backend::ldg(SD.df(df_cur, ppm, xp, yp, zm));
			KS.f[mzm] = TNL::Backend::ldg(SD.df(df_cur, pzp, xp, y, zp));
			KS.f[mzz] = TNL::Backend::ldg(SD.df(df_cur, pzz, xp, y, z));
			KS.f[mzp] = TNL::Backend::ldg(SD.df(df_cur, pzm, xp, y, zm));
			KS.f[mpm] = TNL::Backend::ldg(SD.df(df_cur, pmp, xp, ym, zp));
			KS.f[mpz] = TNL::Backend::ldg(SD.df(df_cur, pmz, xp, ym, z));
			KS.f[mpp] = TNL::Backend::ldg(SD.df(df_cur, pmm, xp, ym, zm));
			KS.f[zmm] = TNL::Backend::ldg(SD.df(df_cur, zpp, x, yp, zp));
			KS.f[zmz] = TNL::Backend::ldg(SD.df(df_cur, zpz, x, yp, z));
			KS.f[zmp] = TNL::Backend::ldg(SD.df(df_cur, zpm, x, yp, zm));
			KS.f[zzm] = TNL::Backend::ldg(SD.df(df_cur, zzp, x, y, zp));
			KS.f[zzz] = TNL::Backend::ldg(SD.df(df_cur, zzz, x, y, z));
			KS.f[zzp] = TNL::Backend::ldg(SD.df(df_cur, zzm, x, y, zm));
			KS.f[zpm] = TNL::Backend::ldg(SD.df(df_cur, zmp, x, ym, zp));
			KS.f[zpz] = TNL::Backend::ldg(SD.df(df_cur, zmz, x, ym, z));
			KS.f[zpp] = TNL::Backend::ldg(SD.df(df_cur, zmm, x, ym, zm));
			KS.f[pmm] = TNL::Backend::ldg(SD.df(df_cur, mpp, xm, yp, zp));
			KS.f[pmz] = TNL::Backend::ldg(SD.df(df_cur, mpz, xm, yp, z));
			KS.f[pmp] = TNL::Backend::ldg(SD.df(df_cur, mpm, xm, yp, zm));
			KS.f[pzm] = TNL::Backend::ldg(SD.df(df_cur, mzp, xm, y, zp));
			KS.f[pzz] = TNL::Backend::ldg(SD.df(df_cur, mzz, xm, y, z));
			KS.f[pzp] = TNL::Backend::ldg(SD.df(df_cur, mzm, xm, y, zm));
			KS.f[ppm] = TNL::Backend::ldg(SD.df(df_cur, mmp, xm, ym, zp));
			KS.f[ppz] = TNL::Backend::ldg(SD.df(df_cur, mmz, xm, ym, z));
			KS.f[ppp] = TNL::Backend::ldg(SD.df(df_cur, mmm, xm, ym, zm));
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
			KS.f[mmm] = TNL::Backend::ldg(SD.df(df_cur, mmm, xmm, ymm, zmm));
			KS.f[mmz] = TNL::Backend::ldg(SD.df(df_cur, mmz, xmm, ymm, z));
			KS.f[mmp] = TNL::Backend::ldg(SD.df(df_cur, mmp, xmm, ymm, zpp));
			KS.f[mzm] = TNL::Backend::ldg(SD.df(df_cur, mzm, xmm, y, zmm));
			KS.f[mzz] = TNL::Backend::ldg(SD.df(df_cur, mzz, xmm, y, z));
			KS.f[mzp] = TNL::Backend::ldg(SD.df(df_cur, mzp, xmm, y, zpp));
			KS.f[mpm] = TNL::Backend::ldg(SD.df(df_cur, mpm, xmm, ypp, zmm));
			KS.f[mpz] = TNL::Backend::ldg(SD.df(df_cur, mpz, xmm, ypp, z));
			KS.f[mpp] = TNL::Backend::ldg(SD.df(df_cur, mpp, xmm, ypp, zpp));
			KS.f[zmm] = TNL::Backend::ldg(SD.df(df_cur, zmm, x, ymm, zmm));
			KS.f[zmz] = TNL::Backend::ldg(SD.df(df_cur, zmz, x, ymm, z));
			KS.f[zmp] = TNL::Backend::ldg(SD.df(df_cur, zmp, x, ymm, zpp));
			KS.f[zzm] = TNL::Backend::ldg(SD.df(df_cur, zzm, x, y, zmm));
			KS.f[zzz] = TNL::Backend::ldg(SD.df(df_cur, zzz, x, y, z));
			KS.f[zzp] = TNL::Backend::ldg(SD.df(df_cur, zzp, x, y, zpp));
			KS.f[zpm] = TNL::Backend::ldg(SD.df(df_cur, zpm, x, ypp, zmm));
			KS.f[zpz] = TNL::Backend::ldg(SD.df(df_cur, zpz, x, ypp, z));
			KS.f[zpp] = TNL::Backend::ldg(SD.df(df_cur, zpp, x, ypp, zpp));
			KS.f[pmm] = TNL::Backend::ldg(SD.df(df_cur, pmm, xpp, ymm, zmm));
			KS.f[pmz] = TNL::Backend::ldg(SD.df(df_cur, pmz, xpp, ymm, z));
			KS.f[pmp] = TNL::Backend::ldg(SD.df(df_cur, pmp, xpp, ymm, zpp));
			KS.f[pzm] = TNL::Backend::ldg(SD.df(df_cur, pzm, xpp, y, zmm));
			KS.f[pzz] = TNL::Backend::ldg(SD.df(df_cur, pzz, xpp, y, z));
			KS.f[pzp] = TNL::Backend::ldg(SD.df(df_cur, pzp, xpp, y, zpp));
			KS.f[ppm] = TNL::Backend::ldg(SD.df(df_cur, ppm, xpp, ypp, zmm));
			KS.f[ppz] = TNL::Backend::ldg(SD.df(df_cur, ppz, xpp, ypp, z));
			KS.f[ppp] = TNL::Backend::ldg(SD.df(df_cur, ppp, xpp, ypp, zpp));
		}
		else {
			KS.f[mmm] = TNL::Backend::ldg(SD.df(df_cur, ppp, xm, ym, zm));
			KS.f[mmz] = TNL::Backend::ldg(SD.df(df_cur, ppz, xm, ym, z));
			KS.f[mmp] = TNL::Backend::ldg(SD.df(df_cur, ppm, xm, ym, zp));
			KS.f[mzm] = TNL::Backend::ldg(SD.df(df_cur, pzp, xm, y, zm));
			KS.f[mzz] = TNL::Backend::ldg(SD.df(df_cur, pzz, xm, y, z));
			KS.f[mzp] = TNL::Backend::ldg(SD.df(df_cur, pzm, xm, y, zp));
			KS.f[mpm] = TNL::Backend::ldg(SD.df(df_cur, pmp, xm, yp, zm));
			KS.f[mpz] = TNL::Backend::ldg(SD.df(df_cur, pmz, xm, yp, z));
			KS.f[mpp] = TNL::Backend::ldg(SD.df(df_cur, pmm, xm, yp, zp));
			KS.f[zmm] = TNL::Backend::ldg(SD.df(df_cur, zpp, x, ym, zm));
			KS.f[zmz] = TNL::Backend::ldg(SD.df(df_cur, zpz, x, ym, z));
			KS.f[zmp] = TNL::Backend::ldg(SD.df(df_cur, zpm, x, ym, zp));
			KS.f[zzm] = TNL::Backend::ldg(SD.df(df_cur, zzp, x, y, zm));
			KS.f[zzz] = TNL::Backend::ldg(SD.df(df_cur, zzz, x, y, z));
			KS.f[zzp] = TNL::Backend::ldg(SD.df(df_cur, zzm, x, y, zp));
			KS.f[zpm] = TNL::Backend::ldg(SD.df(df_cur, zmp, x, yp, zm));
			KS.f[zpz] = TNL::Backend::ldg(SD.df(df_cur, zmz, x, yp, z));
			KS.f[zpp] = TNL::Backend::ldg(SD.df(df_cur, zmm, x, yp, zp));
			KS.f[pmm] = TNL::Backend::ldg(SD.df(df_cur, mpp, xp, ym, zm));
			KS.f[pmz] = TNL::Backend::ldg(SD.df(df_cur, mpz, xp, ym, z));
			KS.f[pmp] = TNL::Backend::ldg(SD.df(df_cur, mpm, xp, ym, zp));
			KS.f[pzm] = TNL::Backend::ldg(SD.df(df_cur, mzp, xp, y, zm));
			KS.f[pzz] = TNL::Backend::ldg(SD.df(df_cur, mzz, xp, y, z));
			KS.f[pzp] = TNL::Backend::ldg(SD.df(df_cur, mzm, xp, y, zp));
			KS.f[ppm] = TNL::Backend::ldg(SD.df(df_cur, mmp, xp, yp, zm));
			KS.f[ppz] = TNL::Backend::ldg(SD.df(df_cur, mmz, xp, yp, z));
			KS.f[ppp] = TNL::Backend::ldg(SD.df(df_cur, mmm, xp, yp, zp));
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
	// components and site offsets fold to constants.
	template <int FACE, typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void streamingOutflowImpl(LBM_DATA& SD, LBM_KS& KS, idx anchor, idx x, idx y, idx z)
	{
		constexpr int axis = (FACE & (bc_face::XP | bc_face::XM)) ? 0
						   : (FACE & (bc_face::YP | bc_face::YM)) ? 1
																  : 2;	// normal axis: 0 = x, 1 = y, 2 = z
		if (SD.even_iter) {
			// natural layout: slot (i, t + c_i) = postcoll_{n-1}(i, t) with
			// t = (anchor, tangential -c_i); the tangential offsets cancel
			// against +c_i, so the normal coordinate is anchor + c_i[normal]
			// and the tangential coordinates are the cell's own
			for (int i = 0; i < 27; i++) {
				idx sx, sy, sz;
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

	// interpolated outflow (Geier 2015) for an arbitrary face: reproduces the
	// legacy fused arithmetic exactly, site-translated like the streamingOutflow
	// gather above. The population moving against the outward normal blends
	// postcoll_{n-1} from the anchor column with the outflow cell's own postcoll,
	// the perpendicular population takes the cell's own postcoll, the outward-
	// moving population comes from the anchor column; all of it is previous-launch
	// state finalized before the pass runs.
	template <int FACE, typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void streamingOutflowInterpImpl(LBM_DATA& SD, LBM_KS& KS, idx anchor, idx x, idx y, idx z)
	{
		constexpr int axis = (FACE & (bc_face::XP | bc_face::XM)) ? 0 : (FACE & (bc_face::YP | bc_face::YM)) ? 1 : 2;
		constexpr int out_sign = (FACE & (bc_face::XM | bc_face::YM | bc_face::ZM)) ? -1 : 1;
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
};

template <typename TRAITS>
inline constexpr bool is_AA_v<D3Q27_STREAMING_AA<TRAITS>> = true;
template <typename TRAITS>
inline constexpr bool twisted_layout_v<D3Q27_STREAMING_AA<TRAITS>> = true;
template <typename TRAITS>
inline constexpr bool requires_ghost_layer_v<D3Q27_STREAMING_AA<TRAITS>> = true;
