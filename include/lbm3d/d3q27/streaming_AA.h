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

	// NOTE: the A-A pattern does not currently support the interpolated outflow
	// boundary condition. We provide a minimal stub so that code paths requiring
	// the method compile; the fallback simply reuses the standard streaming
	// routine. The interpolated boundary must not be used together with the A-A
	// pattern.
	template <typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void streamingInterpRight(
		LBM_DATA& SD,
		LBM_KS& KS,
		idx xm,
		idx x,
		idx xp,
		idx ym,
		idx y,
		idx yp,
		idx zm,
		idx z,
		idx zp
	)
	{
		streaming(SD, KS, xm, x, xp, ym, y, yp, zm, z, zp);
	}

// TODO: implement these methods for A-A pattern
#ifdef TODO
	template <typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void streamingRho(LBM_DATA& SD, LBM_KS& KS, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm, idx z, idx zp)
	{
		// clang-format off
		KS.rho =
			  SD.df(df_cur,mmm,xp+1,yp,zp)
			+ SD.df(df_cur,mmz,xp+1,yp,z )
			+ SD.df(df_cur,mmp,xp+1,yp,zm)
			+ SD.df(df_cur,mzm,xp+1,y ,zp)
			+ SD.df(df_cur,mzz,xp+1,y ,z )
			+ SD.df(df_cur,mzp,xp+1,y ,zm)
			+ SD.df(df_cur,mpm,xp+1,ym,zp)
			+ SD.df(df_cur,mpz,xp+1,ym,z )
			+ SD.df(df_cur,mpp,xp+1,ym,zm)
			+ SD.df(df_cur,zmm,xp  ,yp,zp)
			+ SD.df(df_cur,zmz,xp  ,yp,z )
			+ SD.df(df_cur,zmp,xp  ,yp,zm)
			+ SD.df(df_cur,zzm,xp  ,y ,zp)
			+ SD.df(df_cur,zzp,xp  ,y ,zm)
			+ SD.df(df_cur,zzz,xp  ,y ,z )
			+ SD.df(df_cur,zpm,xp  ,ym,zp)
			+ SD.df(df_cur,zpz,xp  ,ym,z )
			+ SD.df(df_cur,zpp,xp  ,ym,zm)
			+ SD.df(df_cur,pmm,x   ,yp,zp)
			+ SD.df(df_cur,pmz,x   ,yp,z )
			+ SD.df(df_cur,pmp,x   ,yp,zm)
			+ SD.df(df_cur,pzm,x   ,y ,zp)
			+ SD.df(df_cur,pzz,x   ,y ,z )
			+ SD.df(df_cur,pzp,x   ,y ,zm)
			+ SD.df(df_cur,ppm,x   ,ym,zp)
			+ SD.df(df_cur,ppz,x   ,ym,z )
			+ SD.df(df_cur,ppp,x   ,ym,zm);
		// clang-format on
	}

	template <typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void streamingVx(LBM_DATA& SD, LBM_KS& KS, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm, idx z, idx zp)
	{
		// (KS.f[pz] + KS.f[pm] + KS.f[pp] - KS.f[mz] - KS.f[mm] - KS.f[mp] + n1o2*KS.fx)/KS.rho;
		// clang-format off
		KS.vx =
			  SD.df(df_cur,pmm,xm-1,yp,zp)
			+ SD.df(df_cur,pmz,xm-1,yp,z )
			+ SD.df(df_cur,pmp,xm-1,yp,zm)
			+ SD.df(df_cur,ppm,xm-1,ym,zp)
			+ SD.df(df_cur,ppz,xm-1,ym,z )
			+ SD.df(df_cur,ppp,xm-1,ym,zm)
			+ SD.df(df_cur,pzm,xm-1,y ,zp)
			+ SD.df(df_cur,pzz,xm-1,y ,z )
			+ SD.df(df_cur,pzp,xm-1,y ,zm)
			- SD.df(df_cur,mzm,x   ,y ,zp)
			- SD.df(df_cur,mzz,x   ,y ,z )
			- SD.df(df_cur,mzp,x   ,y ,zm)
			- SD.df(df_cur,mmm,x   ,yp,zp)
			- SD.df(df_cur,mmz,x   ,yp,z )
			- SD.df(df_cur,mmp,x   ,yp,zm)
			- SD.df(df_cur,mpm,x   ,ym,zp)
			- SD.df(df_cur,mpz,x   ,ym,z )
			- SD.df(df_cur,mpp,x   ,ym,zm);
		// clang-format on
	}

	template <typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void streamingVy(LBM_DATA& SD, LBM_KS& KS, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm, idx z, idx zp)
	{
		// (KS.f[pz] + KS.f[pm] + KS.f[pp] - KS.f[mz] - KS.f[mm] - KS.f[mp] + n1o2*KS.fx)/KS.rho;
		// clang-format off
		KS.vy =
			  SD.df(df_cur,mpm,x   ,ym,zp)
			+ SD.df(df_cur,mpz,x   ,ym,z )
			+ SD.df(df_cur,mpp,x   ,ym,zm)
			+ SD.df(df_cur,zpm,xm  ,ym,zp)
			+ SD.df(df_cur,zpz,xm  ,ym,z )
			+ SD.df(df_cur,zpp,xm  ,ym,zm)
			+ SD.df(df_cur,ppm,xm-1,ym,zp)
			+ SD.df(df_cur,ppz,xm-1,ym,z )
			+ SD.df(df_cur,ppp,xm-1,ym,zm)
			- SD.df(df_cur,zmm,xm  ,yp,zp)
			- SD.df(df_cur,zmz,xm  ,yp,z )
			- SD.df(df_cur,zmp,xm  ,yp,zm)
			- SD.df(df_cur,pmm,xm-1,yp,zp)
			- SD.df(df_cur,pmz,xm-1,yp,z )
			- SD.df(df_cur,pmp,xm-1,yp,zm)
			- SD.df(df_cur,mmm,x   ,yp,zp)
			- SD.df(df_cur,mmz,x   ,yp,z )
			- SD.df(df_cur,mmp,x   ,yp,zm);
		// clang-format on
	}

	template <typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void streamingVz(LBM_DATA& SD, LBM_KS& KS, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm, idx z, idx zp)
	{
		// (KS.f[pz] + KS.f[pm] + KS.f[pp] - KS.f[mz] - KS.f[mm] - KS.f[mp] + n1o2*KS.fx)/KS.rho;
		// clang-format off
		KS.vz =
			  SD.df(df_cur,mmp,x   ,yp,zm)
			+ SD.df(df_cur,pmp,xm-1,yp,zm)
			+ SD.df(df_cur,zmp,xm  ,yp,zm)
			+ SD.df(df_cur,pzp,xm-1,y ,zm)
			+ SD.df(df_cur,zzp,xm  ,y ,zm)
			+ SD.df(df_cur,mzp,x   ,y ,zm)
			+ SD.df(df_cur,ppp,xm-1,ym,zm)
			+ SD.df(df_cur,zpp,xm  ,ym,zm)
			+ SD.df(df_cur,mpp,x   ,ym,zm)
			- SD.df(df_cur,mmm,x   ,yp,zp)
			- SD.df(df_cur,pmm,xm-1,yp,zp)
			- SD.df(df_cur,zmm,xm  ,yp,zp)
			- SD.df(df_cur,pzm,xm-1,y ,zp)
			- SD.df(df_cur,zzm,xm  ,y ,zp)
			- SD.df(df_cur,mzm,x   ,y ,zp)
			- SD.df(df_cur,ppm,xm-1,ym,zp)
			- SD.df(df_cur,zpm,xm  ,ym,zp)
			- SD.df(df_cur,mpm, x  ,ym,zp);
		// clang-format on
	}
#endif
};
