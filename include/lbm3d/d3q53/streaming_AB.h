#pragma once

#include "lbm3d/defs.h"
#include "../defs.h"
#include <TNL/Backend/Macros.h>

// pull-scheme
template <typename TRAITS>
struct D3Q53_STREAMING
{
	using idx = typename TRAITS::idx;
	using dreal = typename TRAITS::dreal;

	template <typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void
	postCollisionStreaming(LBM_DATA& SD, LBM_KS& KS, typename LBM_KS::SG streamGrid)
	{
		// no streaming actually, write to the (x,y,z) site
		#ifdef __CUDA_ARCH__
		#pragma unroll
		#endif
		for (int i = 0; i < LBM_KS::Q; i++)
			SD.df(df_out, i, streamGrid.x(LBM_KS::NoDV), streamGrid.y(LBM_KS::NoDV), streamGrid.z(LBM_KS::NoDV)) = KS.f[i];
	}

	template <typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void
	streaming(uint8_t type, LBM_DATA& SD, LBM_KS& KS, typename LBM_KS::SG streamGrid)
	{
		#ifdef __CUDA_ARCH__
		#pragma unroll
		#endif
		for(int id = 0; id < LBM_KS::Q; id++){
			const Coord c = LBM_KS::id_to_coords(id);
			const int flip_id = LBM_KS::flip_id(id);
			KS.f[flip_id] = TNL::Backend::ldg(SD.df(type,flip_id,streamGrid.x(c.x),streamGrid.y(c.y),streamGrid.z(c.z)));
		}
	}

	template <typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void streaming(LBM_DATA& SD, LBM_KS& KS, typename LBM_KS::SG streamGrid)
	{
		streaming(df_cur, SD, KS, streamGrid);
	}

	// streaming with bounce-back rule applied
	template <typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void streamingBounceBack(LBM_DATA& SD, LBM_KS& KS,typename LBM_KS::SG streamGrid)
	{
		#ifdef __CUDA_ARCH__
		#pragma unroll
		#endif
		for(int id = 0; id < LBM_KS::Q; id++){
			const Coord c = LBM_KS::id_to_coords(id);
			const int flip_id = LBM_KS::flip_id(id);
			KS.f[id] = TNL::Backend::ldg(SD.df(df_cur,flip_id,streamGrid.x(c.x),streamGrid.y(c.y),streamGrid.z(c.z)));
		}}

	template <typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void streamingRho(LBM_DATA& SD, LBM_KS& KS, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm, idx z, idx zp)
	{
		// clang-format off
		KS.rho =
			  TNL::Backend::ldg(SD.df(df_cur,mmm,xp+1,yp,zp))
			+ TNL::Backend::ldg(SD.df(df_cur,mmz,xp+1,yp,z ))
			+ TNL::Backend::ldg(SD.df(df_cur,mmp,xp+1,yp,zm))
			+ TNL::Backend::ldg(SD.df(df_cur,mzm,xp+1,y ,zp))
			+ TNL::Backend::ldg(SD.df(df_cur,mzz,xp+1,y ,z ))
			+ TNL::Backend::ldg(SD.df(df_cur,mzp,xp+1,y ,zm))
			+ TNL::Backend::ldg(SD.df(df_cur,mpm,xp+1,ym,zp))
			+ TNL::Backend::ldg(SD.df(df_cur,mpz,xp+1,ym,z ))
			+ TNL::Backend::ldg(SD.df(df_cur,mpp,xp+1,ym,zm))
			+ TNL::Backend::ldg(SD.df(df_cur,zmm,xp  ,yp,zp))
			+ TNL::Backend::ldg(SD.df(df_cur,zmz,xp  ,yp,z ))
			+ TNL::Backend::ldg(SD.df(df_cur,zmp,xp  ,yp,zm))
			+ TNL::Backend::ldg(SD.df(df_cur,zzm,xp  ,y ,zp))
			+ TNL::Backend::ldg(SD.df(df_cur,zzp,xp  ,y ,zm))
			+ TNL::Backend::ldg(SD.df(df_cur,zzz,xp  ,y ,z ))
			+ TNL::Backend::ldg(SD.df(df_cur,zpm,xp  ,ym,zp))
			+ TNL::Backend::ldg(SD.df(df_cur,zpz,xp  ,ym,z ))
			+ TNL::Backend::ldg(SD.df(df_cur,zpp,xp  ,ym,zm))
			+ TNL::Backend::ldg(SD.df(df_cur,pmm,x   ,yp,zp))
			+ TNL::Backend::ldg(SD.df(df_cur,pmz,x   ,yp,z ))
			+ TNL::Backend::ldg(SD.df(df_cur,pmp,x   ,yp,zm))
			+ TNL::Backend::ldg(SD.df(df_cur,pzm,x   ,y ,zp))
			+ TNL::Backend::ldg(SD.df(df_cur,pzz,x   ,y ,z ))
			+ TNL::Backend::ldg(SD.df(df_cur,pzp,x   ,y ,zm))
			+ TNL::Backend::ldg(SD.df(df_cur,ppm,x   ,ym,zp))
			+ TNL::Backend::ldg(SD.df(df_cur,ppz,x   ,ym,z ))
			+ TNL::Backend::ldg(SD.df(df_cur,ppp,x   ,ym,zm));
		// clang-format on
	}

	template <typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void streamingVx(LBM_DATA& SD, LBM_KS& KS, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm, idx z, idx zp)
	{
		// (KS.f[pz] + KS.f[pm] + KS.f[pp] - KS.f[mz] - KS.f[mm] - KS.f[mp] + n1o2*KS.fx)/KS.rho;
		// clang-format off
		KS.vx =
			  TNL::Backend::ldg(SD.df(df_cur,pmm,xm-1,yp,zp))
			+ TNL::Backend::ldg(SD.df(df_cur,pmz,xm-1,yp,z ))
			+ TNL::Backend::ldg(SD.df(df_cur,pmp,xm-1,yp,zm))
			+ TNL::Backend::ldg(SD.df(df_cur,ppm,xm-1,ym,zp))
			+ TNL::Backend::ldg(SD.df(df_cur,ppz,xm-1,ym,z ))
			+ TNL::Backend::ldg(SD.df(df_cur,ppp,xm-1,ym,zm))
			+ TNL::Backend::ldg(SD.df(df_cur,pzm,xm-1,y ,zp))
			+ TNL::Backend::ldg(SD.df(df_cur,pzz,xm-1,y ,z ))
			+ TNL::Backend::ldg(SD.df(df_cur,pzp,xm-1,y ,zm))
			- TNL::Backend::ldg(SD.df(df_cur,mzm,x   ,y ,zp))
			- TNL::Backend::ldg(SD.df(df_cur,mzz,x   ,y ,z ))
			- TNL::Backend::ldg(SD.df(df_cur,mzp,x   ,y ,zm))
			- TNL::Backend::ldg(SD.df(df_cur,mmm,x   ,yp,zp))
			- TNL::Backend::ldg(SD.df(df_cur,mmz,x   ,yp,z ))
			- TNL::Backend::ldg(SD.df(df_cur,mmp,x   ,yp,zm))
			- TNL::Backend::ldg(SD.df(df_cur,mpm,x   ,ym,zp))
			- TNL::Backend::ldg(SD.df(df_cur,mpz,x   ,ym,z ))
			- TNL::Backend::ldg(SD.df(df_cur,mpp,x   ,ym,zm));
		// clang-format on
	}

	template <typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void streamingVy(LBM_DATA& SD, LBM_KS& KS, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm, idx z, idx zp)
	{
		// (KS.f[pz] + KS.f[pm] + KS.f[pp] - KS.f[mz] - KS.f[mm] - KS.f[mp] + n1o2*KS.fx)/KS.rho;
		// clang-format off
		KS.vy =
			  TNL::Backend::ldg(SD.df(df_cur,mpm,x   ,ym,zp))
			+ TNL::Backend::ldg(SD.df(df_cur,mpz,x   ,ym,z ))
			+ TNL::Backend::ldg(SD.df(df_cur,mpp,x   ,ym,zm))
			+ TNL::Backend::ldg(SD.df(df_cur,zpm,xm  ,ym,zp))
			+ TNL::Backend::ldg(SD.df(df_cur,zpz,xm  ,ym,z ))
			+ TNL::Backend::ldg(SD.df(df_cur,zpp,xm  ,ym,zm))
			+ TNL::Backend::ldg(SD.df(df_cur,ppm,xm-1,ym,zp))
			+ TNL::Backend::ldg(SD.df(df_cur,ppz,xm-1,ym,z ))
			+ TNL::Backend::ldg(SD.df(df_cur,ppp,xm-1,ym,zm))
			- TNL::Backend::ldg(SD.df(df_cur,zmm,xm  ,yp,zp))
			- TNL::Backend::ldg(SD.df(df_cur,zmz,xm  ,yp,z ))
			- TNL::Backend::ldg(SD.df(df_cur,zmp,xm  ,yp,zm))
			- TNL::Backend::ldg(SD.df(df_cur,pmm,xm-1,yp,zp))
			- TNL::Backend::ldg(SD.df(df_cur,pmz,xm-1,yp,z ))
			- TNL::Backend::ldg(SD.df(df_cur,pmp,xm-1,yp,zm))
			- TNL::Backend::ldg(SD.df(df_cur,mmm,x   ,yp,zp))
			- TNL::Backend::ldg(SD.df(df_cur,mmz,x   ,yp,z ))
			- TNL::Backend::ldg(SD.df(df_cur,mmp,x   ,yp,zm));
		// clang-format on
	}

	template <typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void streamingVz(LBM_DATA& SD, LBM_KS& KS, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm, idx z, idx zp)
	{
		// (KS.f[pz] + KS.f[pm] + KS.f[pp] - KS.f[mz] - KS.f[mm] - KS.f[mp] + n1o2*KS.fx)/KS.rho;
		// clang-format off
		KS.vz =
			  TNL::Backend::ldg(SD.df(df_cur,mmp,x   ,yp,zm))
			+ TNL::Backend::ldg(SD.df(df_cur,pmp,xm-1,yp,zm))
			+ TNL::Backend::ldg(SD.df(df_cur,zmp,xm  ,yp,zm))
			+ TNL::Backend::ldg(SD.df(df_cur,pzp,xm-1,y ,zm))
			+ TNL::Backend::ldg(SD.df(df_cur,zzp,xm  ,y ,zm))
			+ TNL::Backend::ldg(SD.df(df_cur,mzp,x   ,y ,zm))
			+ TNL::Backend::ldg(SD.df(df_cur,ppp,xm-1,ym,zm))
			+ TNL::Backend::ldg(SD.df(df_cur,zpp,xm  ,ym,zm))
			+ TNL::Backend::ldg(SD.df(df_cur,mpp,x   ,ym,zm))
			- TNL::Backend::ldg(SD.df(df_cur,mmm,x   ,yp,zp))
			- TNL::Backend::ldg(SD.df(df_cur,pmm,xm-1,yp,zp))
			- TNL::Backend::ldg(SD.df(df_cur,zmm,xm  ,yp,zp))
			- TNL::Backend::ldg(SD.df(df_cur,pzm,xm-1,y ,zp))
			- TNL::Backend::ldg(SD.df(df_cur,zzm,xm  ,y ,zp))
			- TNL::Backend::ldg(SD.df(df_cur,mzm,x   ,y ,zp))
			- TNL::Backend::ldg(SD.df(df_cur,ppm,xm-1,ym,zp))
			- TNL::Backend::ldg(SD.df(df_cur,zpm,xm  ,ym,zp))
			- TNL::Backend::ldg(SD.df(df_cur,mpm, x  ,ym,zp));
		// clang-format on
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
};
