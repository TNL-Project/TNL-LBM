#pragma once

#include "lbm3d/defs.h"
#include "lbm_common/ciselnik.h"
#include "../defs.h"

template <typename T_TRAITS, typename T_EQ>
struct D3Q343_COMMON_GENERAL
{
	using TRAITS = T_TRAITS;
	using EQ = T_EQ;

	using idx = typename TRAITS::idx;
	using real = typename TRAITS::real;
	using dreal = typename TRAITS::dreal;

	template <typename LBM_KS>
	__cuda_callable__ static void computeDensityAndVelocity(LBM_KS& KS)
	{
		dreal vx  = 0.;
		dreal vy  = 0.;
		dreal vz  = 0.;
		dreal rho = 0.;
		#ifdef __CUDA_ARCH__
		#pragma unroll
		#endif
		for(int id = 0; id < LBM_KS::Q; id++){
			const Coord c = LBM_KS::id_to_dv(id);
			rho += KS.f[id];
			vx  += KS.f[id]*c.x;
			vy  += KS.f[id]*c.y;
			vz  += KS.f[id]*c.z;
		}
		KS.rho = rho;
		KS.vx  = vx/rho + KS.fx/2;
		KS.vy  = vy/rho + KS.fy/2;
		KS.vz  = vz/rho + KS.fz/2;
	}

	template <typename LBM_KS>
	__cuda_callable__ static void computeDensityAndVelocity_Wall(LBM_KS& KS)
	{
		KS.rho = 1;
		KS.vx = 0;
		KS.vy = 0;
		KS.vz = 0;
	}

	template <typename LBM_KS>
	__cuda_callable__ static void setEquilibrium(LBM_KS& KS)
	{
		#ifdef __CUDA_ARCH__
		#pragma unroll
		#endif
		for(int id = 0; id < LBM_KS::Q; id++){
			const Coord c = LBM_KS::id_to_dv(id);
			KS.f[id] = KS.rho*EQ::feq(c.x,c.y,c.z,KS.vx,KS.vy,KS.vz);
		}
	}

	template <typename LBM_KS>
	__cuda_callable__ static void setEquilibriumDecomposition(LBM_KS& KS, dreal rho_out)
	{
		#ifdef __CUDA_ARCH__
		#pragma unroll
		#endif
		for(int id = 0; id < LBM_KS::Q; id++){
			const Coord c = LBM_KS::id_to_dv(id);
            KS.f[id] += (rho_out - KS.rho)*EQ::feq(c.x,c.y,c.z,KS.vx,KS.vy,KS.vz);
        }
	}

	template <typename LBM_KS, typename LAT_DFS>
	__cuda_callable__ static void setEquilibriumLat(LAT_DFS& f, idx x, idx y, idx z, real rho, real vx, real vy, real vz)
	{
		#ifdef __CUDA_ARCH__
		#pragma unroll
		#endif
		for(int id = 0; id < LBM_KS::Q; id++){
			const Coord c = LBM_KS::id_to_dv(id);
            f(id,x,y,z) = rho*EQ::feq(c.x,c.y,c.z,vx,vy,vz);
        }
	}
};
