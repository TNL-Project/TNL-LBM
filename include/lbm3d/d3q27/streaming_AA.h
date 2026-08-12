#pragma once

#include "lbm3d/defs.h"

// A-A pattern
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
	template <typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void streamingInterpRight(LBM_DATA& SD, LBM_KS& KS, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm, idx z, idx zp)
	{
		constexpr dreal SpeedOfSound = 0.5773502691896257;
		static_cast<void>(xp);

		if (SD.even_iter) {
			KS.f[mmm] = SpeedOfSound * SD.df(df_cur, mmm, xm, yp, zp) + (1 - SpeedOfSound) * SD.df(df_cur, mmm, x, yp, zp);
			KS.f[mmz] = SpeedOfSound * SD.df(df_cur, mmz, xm, yp, z) + (1 - SpeedOfSound) * SD.df(df_cur, mmz, x, yp, z);
			KS.f[mmp] = SpeedOfSound * SD.df(df_cur, mmp, xm, yp, zm) + (1 - SpeedOfSound) * SD.df(df_cur, mmp, x, yp, zm);
			KS.f[mzm] = SpeedOfSound * SD.df(df_cur, mzm, xm, y, zp) + (1 - SpeedOfSound) * SD.df(df_cur, mzm, x, y, zp);
			KS.f[mzz] = SpeedOfSound * SD.df(df_cur, mzz, xm, y, z) + (1 - SpeedOfSound) * SD.df(df_cur, mzz, x, y, z);
			KS.f[mzp] = SpeedOfSound * SD.df(df_cur, mzp, xm, y, zm) + (1 - SpeedOfSound) * SD.df(df_cur, mzp, x, y, zm);
			KS.f[mpm] = SpeedOfSound * SD.df(df_cur, mpm, xm, ym, zp) + (1 - SpeedOfSound) * SD.df(df_cur, mpm, x, ym, zp);
			KS.f[mpz] = SpeedOfSound * SD.df(df_cur, mpz, xm, ym, z) + (1 - SpeedOfSound) * SD.df(df_cur, mpz, x, ym, z);
			KS.f[mpp] = SpeedOfSound * SD.df(df_cur, mpp, xm, ym, zm) + (1 - SpeedOfSound) * SD.df(df_cur, mpp, x, ym, zm);
			KS.f[zmm] = SD.df(df_cur, zmm, x, yp, zp);
			KS.f[zmz] = SD.df(df_cur, zmz, x, yp, z);
			KS.f[zmp] = SD.df(df_cur, zmp, x, yp, zm);
			KS.f[zzm] = SD.df(df_cur, zzm, x, y, zp);
			KS.f[zzz] = SD.df(df_cur, zzz, x, y, z);
			KS.f[zzp] = SD.df(df_cur, zzp, x, y, zm);
			KS.f[zpm] = SD.df(df_cur, zpm, x, ym, zp);
			KS.f[zpz] = SD.df(df_cur, zpz, x, ym, z);
			KS.f[zpp] = SD.df(df_cur, zpp, x, ym, zm);
			KS.f[pmm] = SD.df(df_cur, pmm, xm, yp, zp);
			KS.f[pmz] = SD.df(df_cur, pmz, xm, yp, z);
			KS.f[pmp] = SD.df(df_cur, pmp, xm, yp, zm);
			KS.f[pzm] = SD.df(df_cur, pzm, xm, y, zp);
			KS.f[pzz] = SD.df(df_cur, pzz, xm, y, z);
			KS.f[pzp] = SD.df(df_cur, pzp, xm, y, zm);
			KS.f[ppm] = SD.df(df_cur, ppm, xm, ym, zp);
			KS.f[ppz] = SD.df(df_cur, ppz, xm, ym, z);
			KS.f[ppp] = SD.df(df_cur, ppp, xm, ym, zm);
		}
		else {
			const idx xmm = xm - 1;
			KS.f[mmm] = SpeedOfSound * SD.df(df_cur, ppp, xmm, y, z) + (1 - SpeedOfSound) * SD.df(df_cur, ppp, xm, y, z);
			KS.f[mmz] = SpeedOfSound * SD.df(df_cur, ppz, xmm, y, z) + (1 - SpeedOfSound) * SD.df(df_cur, ppz, xm, y, z);
			KS.f[mmp] = SpeedOfSound * SD.df(df_cur, ppm, xmm, y, z) + (1 - SpeedOfSound) * SD.df(df_cur, ppm, xm, y, z);
			KS.f[mzm] = SpeedOfSound * SD.df(df_cur, pzp, xmm, y, z) + (1 - SpeedOfSound) * SD.df(df_cur, pzp, xm, y, z);
			KS.f[mzz] = SpeedOfSound * SD.df(df_cur, pzz, xmm, y, z) + (1 - SpeedOfSound) * SD.df(df_cur, pzz, xm, y, z);
			KS.f[mzp] = SpeedOfSound * SD.df(df_cur, pzm, xmm, y, z) + (1 - SpeedOfSound) * SD.df(df_cur, pzm, xm, y, z);
			KS.f[mpm] = SpeedOfSound * SD.df(df_cur, pmp, xmm, y, z) + (1 - SpeedOfSound) * SD.df(df_cur, pmp, xm, y, z);
			KS.f[mpz] = SpeedOfSound * SD.df(df_cur, pmz, xmm, y, z) + (1 - SpeedOfSound) * SD.df(df_cur, pmz, xm, y, z);
			KS.f[mpp] = SpeedOfSound * SD.df(df_cur, pmm, xmm, y, z) + (1 - SpeedOfSound) * SD.df(df_cur, pmm, xm, y, z);
			KS.f[zmm] = SD.df(df_cur, zpp, x, y, z);
			KS.f[zmz] = SD.df(df_cur, zpz, x, y, z);
			KS.f[zmp] = SD.df(df_cur, zpm, x, y, z);
			KS.f[zzm] = SD.df(df_cur, zzp, x, y, z);
			KS.f[zzz] = SD.df(df_cur, zzz, x, y, z);
			KS.f[zzp] = SD.df(df_cur, zzm, x, y, z);
			KS.f[zpm] = SD.df(df_cur, zmp, x, y, z);
			KS.f[zpz] = SD.df(df_cur, zmz, x, y, z);
			KS.f[zpp] = SD.df(df_cur, zmm, x, y, z);
			KS.f[pmm] = SD.df(df_cur, mpp, x, y, z);
			KS.f[pmz] = SD.df(df_cur, mpz, x, y, z);
			KS.f[pmp] = SD.df(df_cur, mpm, x, y, z);
			KS.f[pzm] = SD.df(df_cur, mzp, x, y, z);
			KS.f[pzz] = SD.df(df_cur, mzz, x, y, z);
			KS.f[pzp] = SD.df(df_cur, mzm, x, y, z);
			KS.f[ppm] = SD.df(df_cur, mmp, x, y, z);
			KS.f[ppz] = SD.df(df_cur, mmz, x, y, z);
			KS.f[ppp] = SD.df(df_cur, mmm, x, y, z);
		}
	}

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
};
