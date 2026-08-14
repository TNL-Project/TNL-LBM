#pragma once

#include "lbm3d/defs.h"
#include "lbm_common/rounding.h"

// A-A pattern
//
// LIMITATION: The non-Newtonian methods streamingRho / streamingVx / streamingVy
// / streamingVz read df_cur from two-step neighbors (xp+1, xm-1, etc.).  In the
// AA pattern, postCollisionStreaming writes df_cur at one-step neighbors within
// the SAME kernel launch.  For regular streaming this is safe because each
// df_cur[dir, pos] is read and written by the SAME thread (the one at
// pos - vel(dir)).  The two-step offsets break this property: df_cur[dir, xp+1]
// is read by thread (x,y,z) but written by thread (xp,y,z) — a different thread.
// If these methods were called from the main LBM kernel (where df_cur is
// read-write), the result would be non-deterministic.
//
// Currently safe because these methods are called only from
// computeNonNewtonianKernels (a separate kernel launch that is read-only on
// df_cur and synchronized before the main kernel).  Do NOT move these calls
// into the main LBM kernel or any other kernel that writes df_cur.
template <typename TRAITS>
struct D3Q27_STREAMING
{
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

	// Bounce-back streaming for the non-Newtonian kernel's wall cells.
	// Delegates to streaming() (which handles even/odd branching) and then
	// swaps all 13 opposite DF pairs — the same effect as the GEO_WALL
	// bounce-back collision.  The pair swaps are local to KS.f, so no
	// even/odd distinction is needed for the swap step.
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

	// Computes the post-stream density at position P = (xp, y, z) — the first
	// fluid cell to the right of the inflow boundary.  Used by the non-Newtonian
	// kernel to set KS.rho for inflow cells before calling setEquilibrium.
	//
	// Even: df_cur is post-stream (natural orientation).  All 27 directions
	//   are already at their post-stream positions, so rho = sum_dir df_cur[dir, P].
	//
	// Odd: df_cur is pre-stream (twisted: opp(dir) stored at each site).
	//   The pull-scheme formula rho(P) = sum_dir pre-stream[dir, P - vel(dir)]
	//   becomes rho(P) = sum_dir df_cur[opp(dir), P - vel(dir)], yielding the
	//   same x-offsets as AB: xp+1 / xp / x for m* / z* / p* dirs.
	template <typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void streamingRho(LBM_DATA& SD, LBM_KS& KS, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm, idx z, idx zp)
	{
		dreal rho;
		if (SD.even_iter) {
			// clang-format off
			rho =
				  SD.df(df_cur, mmm, xp, y, z)
				+ SD.df(df_cur, mmz, xp, y, z)
				+ SD.df(df_cur, mmp, xp, y, z)
				+ SD.df(df_cur, mzm, xp, y, z)
				+ SD.df(df_cur, mzz, xp, y, z)
				+ SD.df(df_cur, mzp, xp, y, z)
				+ SD.df(df_cur, mpm, xp, y, z)
				+ SD.df(df_cur, mpz, xp, y, z)
				+ SD.df(df_cur, mpp, xp, y, z)
				+ SD.df(df_cur, zmm, xp, y, z)
				+ SD.df(df_cur, zmz, xp, y, z)
				+ SD.df(df_cur, zmp, xp, y, z)
				+ SD.df(df_cur, zzm, xp, y, z)
				+ SD.df(df_cur, zzp, xp, y, z)
				+ SD.df(df_cur, zzz, xp, y, z)
				+ SD.df(df_cur, zpm, xp, y, z)
				+ SD.df(df_cur, zpz, xp, y, z)
				+ SD.df(df_cur, zpp, xp, y, z)
				+ SD.df(df_cur, pmm, xp, y, z)
				+ SD.df(df_cur, pmz, xp, y, z)
				+ SD.df(df_cur, pmp, xp, y, z)
				+ SD.df(df_cur, pzm, xp, y, z)
				+ SD.df(df_cur, pzz, xp, y, z)
				+ SD.df(df_cur, pzp, xp, y, z)
				+ SD.df(df_cur, ppm, xp, y, z)
				+ SD.df(df_cur, ppz, xp, y, z)
				+ SD.df(df_cur, ppp, xp, y, z);
			// clang-format on
		}
		else {
			const idx xpp = xp + 1;
			// clang-format off
			rho =
				  SD.df(df_cur, ppp, xpp, yp, zp)
				+ SD.df(df_cur, ppz, xpp, yp, z)
				+ SD.df(df_cur, ppm, xpp, yp, zm)
				+ SD.df(df_cur, pzp, xpp, y,  zp)
				+ SD.df(df_cur, pzz, xpp, y,  z)
				+ SD.df(df_cur, pzm, xpp, y,  zm)
				+ SD.df(df_cur, pmp, xpp, ym, zp)
				+ SD.df(df_cur, pmz, xpp, ym, z)
				+ SD.df(df_cur, pmm, xpp, ym, zm)
				+ SD.df(df_cur, zpp, xp,  yp, zp)
				+ SD.df(df_cur, zpz, xp,  yp, z)
				+ SD.df(df_cur, zpm, xp,  yp, zm)
				+ SD.df(df_cur, zzp, xp,  y,  zp)
				+ SD.df(df_cur, zzm, xp,  y,  zm)
				+ SD.df(df_cur, zzz, xp,  y,  z)
				+ SD.df(df_cur, zmp, xp,  ym, zp)
				+ SD.df(df_cur, zmz, xp,  ym, z)
				+ SD.df(df_cur, zmm, xp,  ym, zm)
				+ SD.df(df_cur, mpp, x,   yp, zp)
				+ SD.df(df_cur, mpz, x,   yp, z)
				+ SD.df(df_cur, mpm, x,   yp, zm)
				+ SD.df(df_cur, mzp, x,   y,  zp)
				+ SD.df(df_cur, mzz, x,   y,  z)
				+ SD.df(df_cur, mzm, x,   y,  zm)
				+ SD.df(df_cur, mmp, x,   ym, zp)
				+ SD.df(df_cur, mmz, x,   ym, z)
				+ SD.df(df_cur, mmm, x,   ym, zm);
			// clang-format on
		}
		KS.rho = rho;
	}

	// Computes the post-stream x-velocity at position P = (xm, y, z) — the first
	// fluid cell to the left of the outflow boundary.  Used by the non-Newtonian
	// kernel to set KS.vx for outflow cells.
	//
	// Even: df_cur is post-stream (natural).
	//   vx = sum_p df_cur[p, P] - sum_m df_cur[m, P].
	//
	// Odd: df_cur is pre-stream (twisted).
	//   Same pull-scheme positions as AB (P - vel(dir) → xm-1 / xm / x),
	//   but with opp(dir) direction index.
	template <typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void streamingVx(LBM_DATA& SD, LBM_KS& KS, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm, idx z, idx zp)
	{
		dreal vx;
		if (SD.even_iter) {
			// clang-format off
			vx =
				  SD.df(df_cur, pmm, xm, y, z)
				+ SD.df(df_cur, pmz, xm, y, z)
				+ SD.df(df_cur, pmp, xm, y, z)
				+ SD.df(df_cur, ppm, xm, y, z)
				+ SD.df(df_cur, ppz, xm, y, z)
				+ SD.df(df_cur, ppp, xm, y, z)
				+ SD.df(df_cur, pzm, xm, y, z)
				+ SD.df(df_cur, pzz, xm, y, z)
				+ SD.df(df_cur, pzp, xm, y, z)
				- SD.df(df_cur, mmm, xm, y, z)
				- SD.df(df_cur, mmz, xm, y, z)
				- SD.df(df_cur, mmp, xm, y, z)
				- SD.df(df_cur, mzm, xm, y, z)
				- SD.df(df_cur, mzz, xm, y, z)
				- SD.df(df_cur, mzp, xm, y, z)
				- SD.df(df_cur, mpm, xm, y, z)
				- SD.df(df_cur, mpz, xm, y, z)
				- SD.df(df_cur, mpp, xm, y, z);
			// clang-format on
		}
		else {
			const idx xmm = xm - 1;
			// clang-format off
			vx =
				  SD.df(df_cur, mpp, xmm, yp, zp)
				+ SD.df(df_cur, mpz, xmm, yp, z)
				+ SD.df(df_cur, mpm, xmm, yp, zm)
				+ SD.df(df_cur, mmp, xmm, ym, zp)
				+ SD.df(df_cur, mmz, xmm, ym, z)
				+ SD.df(df_cur, mmm, xmm, ym, zm)
				+ SD.df(df_cur, mzp, xmm, y,  zp)
				+ SD.df(df_cur, mzz, xmm, y,  z)
				+ SD.df(df_cur, mzm, xmm, y,  zm)
				- SD.df(df_cur, pzp, x,   y,  zp)
				- SD.df(df_cur, pzz, x,   y,  z)
				- SD.df(df_cur, pzm, x,   y,  zm)
				- SD.df(df_cur, ppp, x,   yp, zp)
				- SD.df(df_cur, ppz, x,   yp, z)
				- SD.df(df_cur, ppm, x,   yp, zm)
				- SD.df(df_cur, pmp, x,   ym, zp)
				- SD.df(df_cur, pmz, x,   ym, z)
				- SD.df(df_cur, pmm, x,   ym, zm);
			// clang-format on
		}
		KS.vx = vx;
	}

	// Computes the post-stream y-velocity at position P = (xm, y, z) — the first
	// fluid cell to the left of the outflow boundary.  Used by the non-Newtonian
	// kernel to set KS.vy for outflow cells.
	//
	// Even: df_cur is post-stream (natural).
	//   vy = sum_{p_y} df_cur[dir, P] - sum_{m_y} df_cur[dir, P].
	//
	// Odd: df_cur is pre-stream (twisted).
	//   Same pull-scheme positions as AB, but with opp(dir) direction index.
	template <typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void streamingVy(LBM_DATA& SD, LBM_KS& KS, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm, idx z, idx zp)
	{
		dreal vy;
		if (SD.even_iter) {
			// clang-format off
			vy =
				  SD.df(df_cur, mpm, xm, y, z)
				+ SD.df(df_cur, mpz, xm, y, z)
				+ SD.df(df_cur, mpp, xm, y, z)
				+ SD.df(df_cur, zpm, xm, y, z)
				+ SD.df(df_cur, zpz, xm, y, z)
				+ SD.df(df_cur, zpp, xm, y, z)
				+ SD.df(df_cur, ppm, xm, y, z)
				+ SD.df(df_cur, ppz, xm, y, z)
				+ SD.df(df_cur, ppp, xm, y, z)
				- SD.df(df_cur, zmm, xm, y, z)
				- SD.df(df_cur, zmz, xm, y, z)
				- SD.df(df_cur, zmp, xm, y, z)
				- SD.df(df_cur, pmm, xm, y, z)
				- SD.df(df_cur, pmz, xm, y, z)
				- SD.df(df_cur, pmp, xm, y, z)
				- SD.df(df_cur, mmm, xm, y, z)
				- SD.df(df_cur, mmz, xm, y, z)
				- SD.df(df_cur, mmp, xm, y, z);
			// clang-format on
		}
		else {
			const idx xmm = xm - 1;
			// clang-format off
			vy =
				  SD.df(df_cur, pmp, x,   ym, zp)
				+ SD.df(df_cur, pmz, x,   ym, z)
				+ SD.df(df_cur, pmm, x,   ym, zm)
				+ SD.df(df_cur, zmp, xm,  ym, zp)
				+ SD.df(df_cur, zmz, xm,  ym, z)
				+ SD.df(df_cur, zmm, xm,  ym, zm)
				+ SD.df(df_cur, mmp, xmm, ym, zp)
				+ SD.df(df_cur, mmz, xmm, ym, z)
				+ SD.df(df_cur, mmm, xmm, ym, zm)
				- SD.df(df_cur, zpp, xm,  yp, zp)
				- SD.df(df_cur, zpz, xm,  yp, z)
				- SD.df(df_cur, zpm, xm,  yp, zm)
				- SD.df(df_cur, mpp, xmm, yp, zp)
				- SD.df(df_cur, mpz, xmm, yp, z)
				- SD.df(df_cur, mpm, xmm, yp, zm)
				- SD.df(df_cur, ppp, x,   yp, zp)
				- SD.df(df_cur, ppz, x,   yp, z)
				- SD.df(df_cur, ppm, x,   yp, zm);
			// clang-format on
		}
		KS.vy = vy;
	}

	// Computes the post-stream z-velocity at position P = (xm, y, z) — the first
	// fluid cell to the left of the outflow boundary.  Used by the non-Newtonian
	// kernel to set KS.vz for outflow cells.
	//
	// Even: df_cur is post-stream (natural).
	//   vz = sum_{p_z} df_cur[dir, P] - sum_{m_z} df_cur[dir, P].
	//
	// Odd: df_cur is pre-stream (twisted).
	//   Same pull-scheme positions as AB, but with opp(dir) direction index.
	template <typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void streamingVz(LBM_DATA& SD, LBM_KS& KS, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm, idx z, idx zp)
	{
		dreal vz;
		if (SD.even_iter) {
			// clang-format off
			vz =
				  SD.df(df_cur, mmp, xm, y, z)
				+ SD.df(df_cur, pmp, xm, y, z)
				+ SD.df(df_cur, zmp, xm, y, z)
				+ SD.df(df_cur, pzp, xm, y, z)
				+ SD.df(df_cur, zzp, xm, y, z)
				+ SD.df(df_cur, mzp, xm, y, z)
				+ SD.df(df_cur, ppp, xm, y, z)
				+ SD.df(df_cur, zpp, xm, y, z)
				+ SD.df(df_cur, mpp, xm, y, z)
				- SD.df(df_cur, mmm, xm, y, z)
				- SD.df(df_cur, pmm, xm, y, z)
				- SD.df(df_cur, zmm, xm, y, z)
				- SD.df(df_cur, pzm, xm, y, z)
				- SD.df(df_cur, zzm, xm, y, z)
				- SD.df(df_cur, mzm, xm, y, z)
				- SD.df(df_cur, ppm, xm, y, z)
				- SD.df(df_cur, zpm, xm, y, z)
				- SD.df(df_cur, mpm, xm, y, z);
			// clang-format on
		}
		else {
			const idx xmm = xm - 1;
			// clang-format off
			vz =
				  SD.df(df_cur, ppm, x,   yp, zm)
				+ SD.df(df_cur, mpm, xmm, yp, zm)
				+ SD.df(df_cur, zpm, xm,  yp, zm)
				+ SD.df(df_cur, mzm, xmm, y,  zm)
				+ SD.df(df_cur, zzm, xm,  y,  zm)
				+ SD.df(df_cur, pzm, x,   y,  zm)
				+ SD.df(df_cur, mmm, xmm, ym, zm)
				+ SD.df(df_cur, zmm, xm,  ym, zm)
				+ SD.df(df_cur, pmm, x,   ym, zm)
				- SD.df(df_cur, ppp, x,   yp, zp)
				- SD.df(df_cur, mpp, xmm, yp, zp)
				- SD.df(df_cur, zpp, xm,  yp, zp)
				- SD.df(df_cur, mzp, xmm, y,  zp)
				- SD.df(df_cur, zzp, xm,  y,  zp)
				- SD.df(df_cur, pzp, x,   y,  zp)
				- SD.df(df_cur, mmp, xmm, ym, zp)
				- SD.df(df_cur, zmp, xm,  ym, zp)
				- SD.df(df_cur, pmp, x,   ym, zp);
			// clang-format on
		}
		KS.vz = vz;
	}

	// Streaming for separate outflow-pass kernel (deterministic BC for A-A pattern):
	// both branches reconstruct the pre-collision populations at the translated
	// A-B pull sites of the outflow cell: postcoll_{n-1}(i) at site (xm, y - cy,i),
	// i.e. the populations that a pull stencil anchored at column xm delivers to column x.
	// The slot layouts provide this only from the finalized previous launch:
	// the required slots are owned by column xm-1/interior threads in both parities,
	// so no race-free in-launch gather exists and the pass must run before the main kernel.
	template <typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void
	streamingOutflowRight(LBM_DATA& SD, LBM_KS& KS, idx xm, idx x, idx xp_unused, idx ym, idx y, idx yp, idx zm, idx z, idx zp)
	{
		if (SD.even_iter) {
			// natural layout: slot (i, t + c_i) = postcoll_{n-1}(i, t);
			// the row/column offsets cancel so every term lands at (y, z)
			const idx xmm = xm - 1;
			KS.f[ppp] = TNL::Backend::ldg(SD.df(df_cur, ppp, x, y, z));
			KS.f[ppz] = TNL::Backend::ldg(SD.df(df_cur, ppz, x, y, z));
			KS.f[ppm] = TNL::Backend::ldg(SD.df(df_cur, ppm, x, y, z));
			KS.f[pzp] = TNL::Backend::ldg(SD.df(df_cur, pzp, x, y, z));
			KS.f[pzz] = TNL::Backend::ldg(SD.df(df_cur, pzz, x, y, z));
			KS.f[pzm] = TNL::Backend::ldg(SD.df(df_cur, pzm, x, y, z));
			KS.f[pmp] = TNL::Backend::ldg(SD.df(df_cur, pmp, x, y, z));
			KS.f[pmz] = TNL::Backend::ldg(SD.df(df_cur, pmz, x, y, z));
			KS.f[pmm] = TNL::Backend::ldg(SD.df(df_cur, pmm, x, y, z));
			KS.f[zpp] = TNL::Backend::ldg(SD.df(df_cur, zpp, xm, y, z));
			KS.f[zpz] = TNL::Backend::ldg(SD.df(df_cur, zpz, xm, y, z));
			KS.f[zpm] = TNL::Backend::ldg(SD.df(df_cur, zpm, xm, y, z));
			KS.f[zzp] = TNL::Backend::ldg(SD.df(df_cur, zzp, xm, y, z));
			KS.f[zzz] = TNL::Backend::ldg(SD.df(df_cur, zzz, xm, y, z));
			KS.f[zzm] = TNL::Backend::ldg(SD.df(df_cur, zzm, xm, y, z));
			KS.f[zmp] = TNL::Backend::ldg(SD.df(df_cur, zmp, xm, y, z));
			KS.f[zmz] = TNL::Backend::ldg(SD.df(df_cur, zmz, xm, y, z));
			KS.f[zmm] = TNL::Backend::ldg(SD.df(df_cur, zmm, xm, y, z));
			KS.f[mmm] = TNL::Backend::ldg(SD.df(df_cur, mmm, xmm, y, z));
			KS.f[mmz] = TNL::Backend::ldg(SD.df(df_cur, mmz, xmm, y, z));
			KS.f[mmp] = TNL::Backend::ldg(SD.df(df_cur, mmp, xmm, y, z));
			KS.f[mzm] = TNL::Backend::ldg(SD.df(df_cur, mzm, xmm, y, z));
			KS.f[mzz] = TNL::Backend::ldg(SD.df(df_cur, mzz, xmm, y, z));
			KS.f[mzp] = TNL::Backend::ldg(SD.df(df_cur, mzp, xmm, y, z));
			KS.f[mpm] = TNL::Backend::ldg(SD.df(df_cur, mpm, xmm, y, z));
			KS.f[mpz] = TNL::Backend::ldg(SD.df(df_cur, mpz, xmm, y, z));
			KS.f[mpp] = TNL::Backend::ldg(SD.df(df_cur, mpp, xmm, y, z));
		}
		else {
			// twist layout: slot (opp(i), t) = postcoll_{n-1}(i, t)
			KS.f[ppp] = TNL::Backend::ldg(SD.df(df_cur, mmm, xm, ym, zm));
			KS.f[ppz] = TNL::Backend::ldg(SD.df(df_cur, mmz, xm, ym, z));
			KS.f[ppm] = TNL::Backend::ldg(SD.df(df_cur, mmp, xm, ym, zp));
			KS.f[pzp] = TNL::Backend::ldg(SD.df(df_cur, mzm, xm, y, zm));
			KS.f[pzz] = TNL::Backend::ldg(SD.df(df_cur, mzz, xm, y, z));
			KS.f[pzm] = TNL::Backend::ldg(SD.df(df_cur, mzp, xm, y, zp));
			KS.f[pmp] = TNL::Backend::ldg(SD.df(df_cur, mpm, xm, yp, zm));
			KS.f[pmz] = TNL::Backend::ldg(SD.df(df_cur, mpz, xm, yp, z));
			KS.f[pmm] = TNL::Backend::ldg(SD.df(df_cur, mpp, xm, yp, zp));
			KS.f[zpp] = TNL::Backend::ldg(SD.df(df_cur, zmm, xm, ym, zm));
			KS.f[zpz] = TNL::Backend::ldg(SD.df(df_cur, zmz, xm, ym, z));
			KS.f[zpm] = TNL::Backend::ldg(SD.df(df_cur, zmp, xm, ym, zp));
			KS.f[zzp] = TNL::Backend::ldg(SD.df(df_cur, zzm, xm, y, zm));
			KS.f[zzz] = TNL::Backend::ldg(SD.df(df_cur, zzz, xm, y, z));
			KS.f[zzm] = TNL::Backend::ldg(SD.df(df_cur, zzp, xm, y, zp));
			KS.f[zmp] = TNL::Backend::ldg(SD.df(df_cur, zpm, xm, yp, zm));
			KS.f[zmz] = TNL::Backend::ldg(SD.df(df_cur, zpz, xm, yp, z));
			KS.f[zmm] = TNL::Backend::ldg(SD.df(df_cur, zpp, xm, yp, zp));
			KS.f[mmm] = TNL::Backend::ldg(SD.df(df_cur, ppp, xm, yp, zp));
			KS.f[mmz] = TNL::Backend::ldg(SD.df(df_cur, ppz, xm, yp, z));
			KS.f[mmp] = TNL::Backend::ldg(SD.df(df_cur, ppm, xm, yp, zm));
			KS.f[mzm] = TNL::Backend::ldg(SD.df(df_cur, pzp, xm, y, zp));
			KS.f[mzz] = TNL::Backend::ldg(SD.df(df_cur, pzz, xm, y, z));
			KS.f[mzp] = TNL::Backend::ldg(SD.df(df_cur, pzm, xm, y, zm));
			KS.f[mpm] = TNL::Backend::ldg(SD.df(df_cur, pmp, xm, ym, zp));
			KS.f[mpz] = TNL::Backend::ldg(SD.df(df_cur, pmz, xm, ym, z));
			KS.f[mpp] = TNL::Backend::ldg(SD.df(df_cur, pmm, xm, ym, zm));
		}
	}

	// interpolated outflow (Geier 2015): reproduces the legacy fused arithmetic exactly
	// (the parity-free body lives in streaming_AB.h::streamingOutflowInterpRight),
	// site-translated like the streamingOutflowRight gather above.
	// The interpolated m-family blends postcoll_{n-1} from column xm with
	// the outflow cell's own postcoll (column x), the z-family takes the cell's own postcoll,
	// the p-family comes from column xm; all of it is previous-launch state finalized before the pass runs.
	template <typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void
	streamingOutflowInterpRight(LBM_DATA& SD, LBM_KS& KS, idx xm, idx x, idx xp_unused, idx ym, idx y, idx yp, idx zm, idx z, idx zp)
	{
		constexpr dreal SpeedOfSound = 0.5773502691896257;
		if (SD.even_iter) {
			// natural layout: slot (i, t + c_i) = postcoll_{n-1}(i, t); the
			// row/column offsets cancel so every term lands at (y, z)
			const idx xmm = xm - 1;
			KS.f[ppp] = TNL::Backend::ldg(SD.df(df_cur, ppp, x, y, z));
			KS.f[ppz] = TNL::Backend::ldg(SD.df(df_cur, ppz, x, y, z));
			KS.f[ppm] = TNL::Backend::ldg(SD.df(df_cur, ppm, x, y, z));
			KS.f[pzp] = TNL::Backend::ldg(SD.df(df_cur, pzp, x, y, z));
			KS.f[pzz] = TNL::Backend::ldg(SD.df(df_cur, pzz, x, y, z));
			KS.f[pzm] = TNL::Backend::ldg(SD.df(df_cur, pzm, x, y, z));
			KS.f[pmp] = TNL::Backend::ldg(SD.df(df_cur, pmp, x, y, z));
			KS.f[pmz] = TNL::Backend::ldg(SD.df(df_cur, pmz, x, y, z));
			KS.f[pmm] = TNL::Backend::ldg(SD.df(df_cur, pmm, x, y, z));
			KS.f[zpp] = TNL::Backend::ldg(SD.df(df_cur, zpp, x, y, z));
			KS.f[zpz] = TNL::Backend::ldg(SD.df(df_cur, zpz, x, y, z));
			KS.f[zpm] = TNL::Backend::ldg(SD.df(df_cur, zpm, x, y, z));
			KS.f[zzp] = TNL::Backend::ldg(SD.df(df_cur, zzp, x, y, z));
			KS.f[zzz] = TNL::Backend::ldg(SD.df(df_cur, zzz, x, y, z));
			KS.f[zzm] = TNL::Backend::ldg(SD.df(df_cur, zzm, x, y, z));
			KS.f[zmp] = TNL::Backend::ldg(SD.df(df_cur, zmp, x, y, z));
			KS.f[zmz] = TNL::Backend::ldg(SD.df(df_cur, zmz, x, y, z));
			KS.f[zmm] = TNL::Backend::ldg(SD.df(df_cur, zmm, x, y, z));
			KS.f[mmm] = lbm_fma_rn(
				SpeedOfSound, TNL::Backend::ldg(SD.df(df_cur, mmm, xmm, y, z)), (1 - SpeedOfSound) * TNL::Backend::ldg(SD.df(df_cur, mmm, xm, y, z))
			);
			KS.f[mmz] = lbm_fma_rn(
				SpeedOfSound, TNL::Backend::ldg(SD.df(df_cur, mmz, xmm, y, z)), (1 - SpeedOfSound) * TNL::Backend::ldg(SD.df(df_cur, mmz, xm, y, z))
			);
			KS.f[mmp] = lbm_fma_rn(
				SpeedOfSound, TNL::Backend::ldg(SD.df(df_cur, mmp, xmm, y, z)), (1 - SpeedOfSound) * TNL::Backend::ldg(SD.df(df_cur, mmp, xm, y, z))
			);
			KS.f[mzm] = lbm_fma_rn(
				SpeedOfSound, TNL::Backend::ldg(SD.df(df_cur, mzm, xmm, y, z)), (1 - SpeedOfSound) * TNL::Backend::ldg(SD.df(df_cur, mzm, xm, y, z))
			);
			KS.f[mzz] = lbm_fma_rn(
				SpeedOfSound, TNL::Backend::ldg(SD.df(df_cur, mzz, xmm, y, z)), (1 - SpeedOfSound) * TNL::Backend::ldg(SD.df(df_cur, mzz, xm, y, z))
			);
			KS.f[mzp] = lbm_fma_rn(
				SpeedOfSound, TNL::Backend::ldg(SD.df(df_cur, mzp, xmm, y, z)), (1 - SpeedOfSound) * TNL::Backend::ldg(SD.df(df_cur, mzp, xm, y, z))
			);
			KS.f[mpm] = lbm_fma_rn(
				SpeedOfSound, TNL::Backend::ldg(SD.df(df_cur, mpm, xmm, y, z)), (1 - SpeedOfSound) * TNL::Backend::ldg(SD.df(df_cur, mpm, xm, y, z))
			);
			KS.f[mpz] = lbm_fma_rn(
				SpeedOfSound, TNL::Backend::ldg(SD.df(df_cur, mpz, xmm, y, z)), (1 - SpeedOfSound) * TNL::Backend::ldg(SD.df(df_cur, mpz, xm, y, z))
			);
			KS.f[mpp] = lbm_fma_rn(
				SpeedOfSound, TNL::Backend::ldg(SD.df(df_cur, mpp, xmm, y, z)), (1 - SpeedOfSound) * TNL::Backend::ldg(SD.df(df_cur, mpp, xm, y, z))
			);
		}
		else {
			// twist layout: slot (opp(i), t) = postcoll_{n-1}(i, t)
			KS.f[ppp] = TNL::Backend::ldg(SD.df(df_cur, mmm, xm, ym, zm));
			KS.f[ppz] = TNL::Backend::ldg(SD.df(df_cur, mmz, xm, ym, z));
			KS.f[ppm] = TNL::Backend::ldg(SD.df(df_cur, mmp, xm, ym, zp));
			KS.f[pzp] = TNL::Backend::ldg(SD.df(df_cur, mzm, xm, y, zm));
			KS.f[pzz] = TNL::Backend::ldg(SD.df(df_cur, mzz, xm, y, z));
			KS.f[pzm] = TNL::Backend::ldg(SD.df(df_cur, mzp, xm, y, zp));
			KS.f[pmp] = TNL::Backend::ldg(SD.df(df_cur, mpm, xm, yp, zm));
			KS.f[pmz] = TNL::Backend::ldg(SD.df(df_cur, mpz, xm, yp, z));
			KS.f[pmm] = TNL::Backend::ldg(SD.df(df_cur, mpp, xm, yp, zp));
			KS.f[zpp] = TNL::Backend::ldg(SD.df(df_cur, zmm, x, ym, zm));
			KS.f[zpz] = TNL::Backend::ldg(SD.df(df_cur, zmz, x, ym, z));
			KS.f[zpm] = TNL::Backend::ldg(SD.df(df_cur, zmp, x, ym, zp));
			KS.f[zzp] = TNL::Backend::ldg(SD.df(df_cur, zzm, x, y, zm));
			KS.f[zzz] = TNL::Backend::ldg(SD.df(df_cur, zzz, x, y, z));
			KS.f[zzm] = TNL::Backend::ldg(SD.df(df_cur, zzp, x, y, zp));
			KS.f[zmp] = TNL::Backend::ldg(SD.df(df_cur, zpm, x, yp, zm));
			KS.f[zmz] = TNL::Backend::ldg(SD.df(df_cur, zpz, x, yp, z));
			KS.f[zmm] = TNL::Backend::ldg(SD.df(df_cur, zpp, x, yp, zp));
			KS.f[mmm] = lbm_fma_rn(
				SpeedOfSound, TNL::Backend::ldg(SD.df(df_cur, ppp, xm, yp, zp)), (1 - SpeedOfSound) * TNL::Backend::ldg(SD.df(df_cur, ppp, x, yp, zp))
			);
			KS.f[mmz] = lbm_fma_rn(
				SpeedOfSound, TNL::Backend::ldg(SD.df(df_cur, ppz, xm, yp, z)), (1 - SpeedOfSound) * TNL::Backend::ldg(SD.df(df_cur, ppz, x, yp, z))
			);
			KS.f[mmp] = lbm_fma_rn(
				SpeedOfSound, TNL::Backend::ldg(SD.df(df_cur, ppm, xm, yp, zm)), (1 - SpeedOfSound) * TNL::Backend::ldg(SD.df(df_cur, ppm, x, yp, zm))
			);
			KS.f[mzm] = lbm_fma_rn(
				SpeedOfSound, TNL::Backend::ldg(SD.df(df_cur, pzp, xm, y, zp)), (1 - SpeedOfSound) * TNL::Backend::ldg(SD.df(df_cur, pzp, x, y, zp))
			);
			KS.f[mzz] = lbm_fma_rn(
				SpeedOfSound, TNL::Backend::ldg(SD.df(df_cur, pzz, xm, y, z)), (1 - SpeedOfSound) * TNL::Backend::ldg(SD.df(df_cur, pzz, x, y, z))
			);
			KS.f[mzp] = lbm_fma_rn(
				SpeedOfSound, TNL::Backend::ldg(SD.df(df_cur, pzm, xm, y, zm)), (1 - SpeedOfSound) * TNL::Backend::ldg(SD.df(df_cur, pzm, x, y, zm))
			);
			KS.f[mpm] = lbm_fma_rn(
				SpeedOfSound, TNL::Backend::ldg(SD.df(df_cur, pmp, xm, ym, zp)), (1 - SpeedOfSound) * TNL::Backend::ldg(SD.df(df_cur, pmp, x, ym, zp))
			);
			KS.f[mpz] = lbm_fma_rn(
				SpeedOfSound, TNL::Backend::ldg(SD.df(df_cur, pmz, xm, ym, z)), (1 - SpeedOfSound) * TNL::Backend::ldg(SD.df(df_cur, pmz, x, ym, z))
			);
			KS.f[mpp] = lbm_fma_rn(
				SpeedOfSound, TNL::Backend::ldg(SD.df(df_cur, pmm, xm, ym, zm)), (1 - SpeedOfSound) * TNL::Backend::ldg(SD.df(df_cur, pmm, x, ym, zm))
			);
		}
	}
};
