#pragma once

#include "lbm3d/defs.h"

// pull-scheme
template <typename TRAITS>
struct D3Q27_STREAMING
{
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

	template <typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void streamingInterpRight(LBM_DATA& SD, LBM_KS& KS, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm, idx z, idx zp)
	{
		// streaming: interpolation from Geier - CuLBM (2015)
		// NOTE: velocity is neglected (for the case velocity << speed of sound)
		constexpr dreal SpeedOfSound = 0.5773502691896257;
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
