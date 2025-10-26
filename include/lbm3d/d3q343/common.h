#pragma once

#include "lbm3d/defs.h"
#include "lbm_common/ciselnik.h"
#include "../defs.h"

template <typename T_TRAITS, typename T_EQ>
struct D3Q343_COMMON
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
			KS.f[id] = EQ::feq(KS.rho,c.x,c.y,c.z,KS.vx,KS.vy,KS.vz);
		}
	}

	// used in the "interpolated outflow boundary condition with decomposition" by Eichler https://doi.org/10.1016/j.camwa.2024.08.009
	template <typename LBM_KS>
	__cuda_callable__ static void setEquilibriumDecomposition(LBM_KS& KS, dreal rho_out)
	{
		KS.f[mmm] += EQ::eq_mmm(rho_out, KS.vx, KS.vy, KS.vz) - EQ::eq_mmm(KS.rho, KS.vx, KS.vy, KS.vz);
		KS.f[mmz] += EQ::eq_mmz(rho_out, KS.vx, KS.vy, KS.vz) - EQ::eq_mmz(KS.rho, KS.vx, KS.vy, KS.vz);
		KS.f[mmp] += EQ::eq_mmp(rho_out, KS.vx, KS.vy, KS.vz) - EQ::eq_mmp(KS.rho, KS.vx, KS.vy, KS.vz);
		KS.f[mzm] += EQ::eq_mzm(rho_out, KS.vx, KS.vy, KS.vz) - EQ::eq_mzm(KS.rho, KS.vx, KS.vy, KS.vz);
		KS.f[mzz] += EQ::eq_mzz(rho_out, KS.vx, KS.vy, KS.vz) - EQ::eq_mzz(KS.rho, KS.vx, KS.vy, KS.vz);
		KS.f[mzp] += EQ::eq_mzp(rho_out, KS.vx, KS.vy, KS.vz) - EQ::eq_mzp(KS.rho, KS.vx, KS.vy, KS.vz);
		KS.f[mpm] += EQ::eq_mpm(rho_out, KS.vx, KS.vy, KS.vz) - EQ::eq_mpm(KS.rho, KS.vx, KS.vy, KS.vz);
		KS.f[mpz] += EQ::eq_mpz(rho_out, KS.vx, KS.vy, KS.vz) - EQ::eq_mpz(KS.rho, KS.vx, KS.vy, KS.vz);
		KS.f[mpp] += EQ::eq_mpp(rho_out, KS.vx, KS.vy, KS.vz) - EQ::eq_mpp(KS.rho, KS.vx, KS.vy, KS.vz);
		KS.f[zmm] += EQ::eq_zmm(rho_out, KS.vx, KS.vy, KS.vz) - EQ::eq_zmm(KS.rho, KS.vx, KS.vy, KS.vz);
		KS.f[zmz] += EQ::eq_zmz(rho_out, KS.vx, KS.vy, KS.vz) - EQ::eq_zmz(KS.rho, KS.vx, KS.vy, KS.vz);
		KS.f[zmp] += EQ::eq_zmp(rho_out, KS.vx, KS.vy, KS.vz) - EQ::eq_zmp(KS.rho, KS.vx, KS.vy, KS.vz);
		KS.f[zzm] += EQ::eq_zzm(rho_out, KS.vx, KS.vy, KS.vz) - EQ::eq_zzm(KS.rho, KS.vx, KS.vy, KS.vz);
		KS.f[zzz] += EQ::eq_zzz(rho_out, KS.vx, KS.vy, KS.vz) - EQ::eq_zzz(KS.rho, KS.vx, KS.vy, KS.vz);
		KS.f[zzp] += EQ::eq_zzp(rho_out, KS.vx, KS.vy, KS.vz) - EQ::eq_zzp(KS.rho, KS.vx, KS.vy, KS.vz);
		KS.f[zpm] += EQ::eq_zpm(rho_out, KS.vx, KS.vy, KS.vz) - EQ::eq_zpm(KS.rho, KS.vx, KS.vy, KS.vz);
		KS.f[zpz] += EQ::eq_zpz(rho_out, KS.vx, KS.vy, KS.vz) - EQ::eq_zpz(KS.rho, KS.vx, KS.vy, KS.vz);
		KS.f[zpp] += EQ::eq_zpp(rho_out, KS.vx, KS.vy, KS.vz) - EQ::eq_zpp(KS.rho, KS.vx, KS.vy, KS.vz);
		KS.f[pmm] += EQ::eq_pmm(rho_out, KS.vx, KS.vy, KS.vz) - EQ::eq_pmm(KS.rho, KS.vx, KS.vy, KS.vz);
		KS.f[pmz] += EQ::eq_pmz(rho_out, KS.vx, KS.vy, KS.vz) - EQ::eq_pmz(KS.rho, KS.vx, KS.vy, KS.vz);
		KS.f[pmp] += EQ::eq_pmp(rho_out, KS.vx, KS.vy, KS.vz) - EQ::eq_pmp(KS.rho, KS.vx, KS.vy, KS.vz);
		KS.f[pzm] += EQ::eq_pzm(rho_out, KS.vx, KS.vy, KS.vz) - EQ::eq_pzm(KS.rho, KS.vx, KS.vy, KS.vz);
		KS.f[pzz] += EQ::eq_pzz(rho_out, KS.vx, KS.vy, KS.vz) - EQ::eq_pzz(KS.rho, KS.vx, KS.vy, KS.vz);
		KS.f[pzp] += EQ::eq_pzp(rho_out, KS.vx, KS.vy, KS.vz) - EQ::eq_pzp(KS.rho, KS.vx, KS.vy, KS.vz);
		KS.f[ppm] += EQ::eq_ppm(rho_out, KS.vx, KS.vy, KS.vz) - EQ::eq_ppm(KS.rho, KS.vx, KS.vy, KS.vz);
		KS.f[ppz] += EQ::eq_ppz(rho_out, KS.vx, KS.vy, KS.vz) - EQ::eq_ppz(KS.rho, KS.vx, KS.vy, KS.vz);
		KS.f[ppp] += EQ::eq_ppp(rho_out, KS.vx, KS.vy, KS.vz) - EQ::eq_ppp(KS.rho, KS.vx, KS.vy, KS.vz);
	}

	template <typename LAT_DFS>
	__cuda_callable__ static void setEquilibriumLat(LAT_DFS& f, idx x, idx y, idx z, real rho, real vx, real vy, real vz)
	{
		f(mmm, x, y, z) = EQ::eq_mmm(rho, vx, vy, vz);
		f(zmm, x, y, z) = EQ::eq_zmm(rho, vx, vy, vz);
		f(pmm, x, y, z) = EQ::eq_pmm(rho, vx, vy, vz);
		f(mzm, x, y, z) = EQ::eq_mzm(rho, vx, vy, vz);
		f(zzm, x, y, z) = EQ::eq_zzm(rho, vx, vy, vz);
		f(pzm, x, y, z) = EQ::eq_pzm(rho, vx, vy, vz);
		f(mpm, x, y, z) = EQ::eq_mpm(rho, vx, vy, vz);
		f(zpm, x, y, z) = EQ::eq_zpm(rho, vx, vy, vz);
		f(ppm, x, y, z) = EQ::eq_ppm(rho, vx, vy, vz);

		f(mmz, x, y, z) = EQ::eq_mmz(rho, vx, vy, vz);
		f(zmz, x, y, z) = EQ::eq_zmz(rho, vx, vy, vz);
		f(pmz, x, y, z) = EQ::eq_pmz(rho, vx, vy, vz);
		f(mzz, x, y, z) = EQ::eq_mzz(rho, vx, vy, vz);
		f(zzz, x, y, z) = EQ::eq_zzz(rho, vx, vy, vz);
		f(pzz, x, y, z) = EQ::eq_pzz(rho, vx, vy, vz);
		f(mpz, x, y, z) = EQ::eq_mpz(rho, vx, vy, vz);
		f(zpz, x, y, z) = EQ::eq_zpz(rho, vx, vy, vz);
		f(ppz, x, y, z) = EQ::eq_ppz(rho, vx, vy, vz);

		f(mmp, x, y, z) = EQ::eq_mmp(rho, vx, vy, vz);
		f(zmp, x, y, z) = EQ::eq_zmp(rho, vx, vy, vz);
		f(pmp, x, y, z) = EQ::eq_pmp(rho, vx, vy, vz);
		f(mzp, x, y, z) = EQ::eq_mzp(rho, vx, vy, vz);
		f(zzp, x, y, z) = EQ::eq_zzp(rho, vx, vy, vz);
		f(pzp, x, y, z) = EQ::eq_pzp(rho, vx, vy, vz);
		f(mpp, x, y, z) = EQ::eq_mpp(rho, vx, vy, vz);
		f(zpp, x, y, z) = EQ::eq_zpp(rho, vx, vy, vz);
		f(ppp, x, y, z) = EQ::eq_ppp(rho, vx, vy, vz);
	}
};
