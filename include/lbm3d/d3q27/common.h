#pragma once

#include "lbm3d/defs.h"
#include "lbm_common/ciselnik.h"

template <typename T_TRAITS, typename T_EQ, bool WELL_CONDITIONED = false>
struct D3Q27_COMMON
{
	using TRAITS = T_TRAITS;
	using EQ = T_EQ;

	using idx = typename TRAITS::idx;
	using real = typename TRAITS::real;
	using dreal = typename TRAITS::dreal;

	static constexpr bool is_well_conditioned = WELL_CONDITIONED;

	// storage<->physical DF converters: identity for the standard convention, shifted by +/-w_i when WELL_CONDITIONED
	__cuda_callable__ static constexpr dreal weight(int i)
	{
		return i == dir27::zzz ? n8o27 : (i <= dir27::zzm ? n2o27 : (i <= dir27::zmp ? n1o54 : n1o216));
	}

	__cuda_callable__ static dreal fromStorage(const dreal* f, int i)
	{
		if constexpr (WELL_CONDITIONED)
			return f[i] + weight(i);
		else
			return f[i];
	}

	__cuda_callable__ static dreal toStorage(dreal f, int i)
	{
		if constexpr (WELL_CONDITIONED)
			return f - weight(i);
		else
			return f;
	}

	template <typename LBM_KS>
	__cuda_callable__ static void computeDensityAndVelocity(LBM_KS& KS)
	{
#ifdef USE_HIGH_PRECISION_RHO
		// src: https://en.wikipedia.org/wiki/Kahan_summation_algorithm
		KS.rho = 0;
		dreal c = 0;  // A running compensation for lost low-order bits.
		for (int i = 0; i < 27; i++) {
			dreal y = KS.f[i] - c;
			dreal t = KS.rho + y;
			c = (t - KS.rho) - y;
			KS.rho = t;
		}
#else
		// based on Geier 2015 Appendix J
		KS.rho = ((((KS.f[dir27::ppp] + KS.f[dir27::mmm]) + (KS.f[dir27::pmp] + KS.f[dir27::mpm]))
				   + ((KS.f[dir27::ppm] + KS.f[dir27::mmp]) + (KS.f[dir27::mpp] + KS.f[dir27::pmm])))
				  + (((KS.f[dir27::zpp] + KS.f[dir27::zmm]) + (KS.f[dir27::zpm] + KS.f[dir27::zmp]))
					 + ((KS.f[dir27::pzp] + KS.f[dir27::mzm]) + (KS.f[dir27::pzm] + KS.f[dir27::mzp]))
					 + ((KS.f[dir27::ppz] + KS.f[dir27::mmz]) + (KS.f[dir27::pmz] + KS.f[dir27::mpz])))
				  + ((KS.f[dir27::pzz] + KS.f[dir27::mzz]) + (KS.f[dir27::zpz] + KS.f[dir27::zmz]) + (KS.f[dir27::zzp] + KS.f[dir27::zzm])))
			   + KS.f[dir27::zzz];
#endif

		if constexpr (WELL_CONDITIONED) {
			KS.rho += no1;
		}

		KS.vx = ((((KS.f[dir27::ppp] - KS.f[dir27::mmm]) + (KS.f[dir27::pmp] - KS.f[dir27::mpm]))
				  + ((KS.f[dir27::ppm] - KS.f[dir27::mmp]) + (KS.f[dir27::pmm] - KS.f[dir27::mpp])))
				 + (((KS.f[dir27::pzp] - KS.f[dir27::mzm]) + (KS.f[dir27::pzm] - KS.f[dir27::mzp]))
					+ ((KS.f[dir27::ppz] - KS.f[dir27::mmz]) + (KS.f[dir27::pmz] - KS.f[dir27::mpz])))
				 + (KS.f[dir27::pzz] - KS.f[dir27::mzz]) + KS.fx * n1o2)
			  / KS.rho;
		KS.vy = ((((KS.f[dir27::ppp] - KS.f[dir27::mmm]) + (KS.f[dir27::ppm] - KS.f[dir27::mmp]))
				  + ((KS.f[dir27::mpp] - KS.f[dir27::pmm]) + (KS.f[dir27::mpm] - KS.f[dir27::pmp])))
				 + (((KS.f[dir27::ppz] - KS.f[dir27::mmz]) + (KS.f[dir27::mpz] - KS.f[dir27::pmz]))
					+ ((KS.f[dir27::zpp] - KS.f[dir27::zmm]) + (KS.f[dir27::zpm] - KS.f[dir27::zmp])))
				 + (KS.f[dir27::zpz] - KS.f[dir27::zmz]) + KS.fy * n1o2)
			  / KS.rho;
		KS.vz = ((((KS.f[dir27::ppp] - KS.f[dir27::mmm]) + (KS.f[dir27::mpp] - KS.f[dir27::pmm]))
				  + ((KS.f[dir27::pmp] - KS.f[dir27::mpm]) + (KS.f[dir27::mmp] - KS.f[dir27::ppm])))
				 + (((KS.f[dir27::zpp] - KS.f[dir27::zmm]) + (KS.f[dir27::zmp] - KS.f[dir27::zpm]))
					+ ((KS.f[dir27::pzp] - KS.f[dir27::mzm]) + (KS.f[dir27::mzp] - KS.f[dir27::pzm])))
				 + (KS.f[dir27::zzp] - KS.f[dir27::zzm]) + KS.fz * n1o2)
			  / KS.rho;
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
		KS.f[dir27::mmm] = EQ::eq_mmm(KS.rho, KS.vx, KS.vy, KS.vz);
		KS.f[dir27::mmz] = EQ::eq_mmz(KS.rho, KS.vx, KS.vy, KS.vz);
		KS.f[dir27::mmp] = EQ::eq_mmp(KS.rho, KS.vx, KS.vy, KS.vz);
		KS.f[dir27::mzm] = EQ::eq_mzm(KS.rho, KS.vx, KS.vy, KS.vz);
		KS.f[dir27::mzz] = EQ::eq_mzz(KS.rho, KS.vx, KS.vy, KS.vz);
		KS.f[dir27::mzp] = EQ::eq_mzp(KS.rho, KS.vx, KS.vy, KS.vz);
		KS.f[dir27::mpm] = EQ::eq_mpm(KS.rho, KS.vx, KS.vy, KS.vz);
		KS.f[dir27::mpz] = EQ::eq_mpz(KS.rho, KS.vx, KS.vy, KS.vz);
		KS.f[dir27::mpp] = EQ::eq_mpp(KS.rho, KS.vx, KS.vy, KS.vz);
		KS.f[dir27::zmm] = EQ::eq_zmm(KS.rho, KS.vx, KS.vy, KS.vz);
		KS.f[dir27::zmz] = EQ::eq_zmz(KS.rho, KS.vx, KS.vy, KS.vz);
		KS.f[dir27::zmp] = EQ::eq_zmp(KS.rho, KS.vx, KS.vy, KS.vz);
		KS.f[dir27::zzm] = EQ::eq_zzm(KS.rho, KS.vx, KS.vy, KS.vz);
		KS.f[dir27::zzz] = EQ::eq_zzz(KS.rho, KS.vx, KS.vy, KS.vz);
		KS.f[dir27::zzp] = EQ::eq_zzp(KS.rho, KS.vx, KS.vy, KS.vz);
		KS.f[dir27::zpm] = EQ::eq_zpm(KS.rho, KS.vx, KS.vy, KS.vz);
		KS.f[dir27::zpz] = EQ::eq_zpz(KS.rho, KS.vx, KS.vy, KS.vz);
		KS.f[dir27::zpp] = EQ::eq_zpp(KS.rho, KS.vx, KS.vy, KS.vz);
		KS.f[dir27::pmm] = EQ::eq_pmm(KS.rho, KS.vx, KS.vy, KS.vz);
		KS.f[dir27::pmz] = EQ::eq_pmz(KS.rho, KS.vx, KS.vy, KS.vz);
		KS.f[dir27::pmp] = EQ::eq_pmp(KS.rho, KS.vx, KS.vy, KS.vz);
		KS.f[dir27::pzm] = EQ::eq_pzm(KS.rho, KS.vx, KS.vy, KS.vz);
		KS.f[dir27::pzz] = EQ::eq_pzz(KS.rho, KS.vx, KS.vy, KS.vz);
		KS.f[dir27::pzp] = EQ::eq_pzp(KS.rho, KS.vx, KS.vy, KS.vz);
		KS.f[dir27::ppm] = EQ::eq_ppm(KS.rho, KS.vx, KS.vy, KS.vz);
		KS.f[dir27::ppz] = EQ::eq_ppz(KS.rho, KS.vx, KS.vy, KS.vz);
		KS.f[dir27::ppp] = EQ::eq_ppp(KS.rho, KS.vx, KS.vy, KS.vz);
	}

	// used in the "interpolated outflow boundary condition with decomposition" by Eichler https://doi.org/10.1016/j.camwa.2024.08.009
	template <typename LBM_KS>
	__cuda_callable__ static void setEquilibriumDecomposition(LBM_KS& KS, dreal rho_out)
	{
		KS.f[dir27::mmm] += EQ::eq_mmm(rho_out, KS.vx, KS.vy, KS.vz) - EQ::eq_mmm(KS.rho, KS.vx, KS.vy, KS.vz);
		KS.f[dir27::mmz] += EQ::eq_mmz(rho_out, KS.vx, KS.vy, KS.vz) - EQ::eq_mmz(KS.rho, KS.vx, KS.vy, KS.vz);
		KS.f[dir27::mmp] += EQ::eq_mmp(rho_out, KS.vx, KS.vy, KS.vz) - EQ::eq_mmp(KS.rho, KS.vx, KS.vy, KS.vz);
		KS.f[dir27::mzm] += EQ::eq_mzm(rho_out, KS.vx, KS.vy, KS.vz) - EQ::eq_mzm(KS.rho, KS.vx, KS.vy, KS.vz);
		KS.f[dir27::mzz] += EQ::eq_mzz(rho_out, KS.vx, KS.vy, KS.vz) - EQ::eq_mzz(KS.rho, KS.vx, KS.vy, KS.vz);
		KS.f[dir27::mzp] += EQ::eq_mzp(rho_out, KS.vx, KS.vy, KS.vz) - EQ::eq_mzp(KS.rho, KS.vx, KS.vy, KS.vz);
		KS.f[dir27::mpm] += EQ::eq_mpm(rho_out, KS.vx, KS.vy, KS.vz) - EQ::eq_mpm(KS.rho, KS.vx, KS.vy, KS.vz);
		KS.f[dir27::mpz] += EQ::eq_mpz(rho_out, KS.vx, KS.vy, KS.vz) - EQ::eq_mpz(KS.rho, KS.vx, KS.vy, KS.vz);
		KS.f[dir27::mpp] += EQ::eq_mpp(rho_out, KS.vx, KS.vy, KS.vz) - EQ::eq_mpp(KS.rho, KS.vx, KS.vy, KS.vz);
		KS.f[dir27::zmm] += EQ::eq_zmm(rho_out, KS.vx, KS.vy, KS.vz) - EQ::eq_zmm(KS.rho, KS.vx, KS.vy, KS.vz);
		KS.f[dir27::zmz] += EQ::eq_zmz(rho_out, KS.vx, KS.vy, KS.vz) - EQ::eq_zmz(KS.rho, KS.vx, KS.vy, KS.vz);
		KS.f[dir27::zmp] += EQ::eq_zmp(rho_out, KS.vx, KS.vy, KS.vz) - EQ::eq_zmp(KS.rho, KS.vx, KS.vy, KS.vz);
		KS.f[dir27::zzm] += EQ::eq_zzm(rho_out, KS.vx, KS.vy, KS.vz) - EQ::eq_zzm(KS.rho, KS.vx, KS.vy, KS.vz);
		KS.f[dir27::zzz] += EQ::eq_zzz(rho_out, KS.vx, KS.vy, KS.vz) - EQ::eq_zzz(KS.rho, KS.vx, KS.vy, KS.vz);
		KS.f[dir27::zzp] += EQ::eq_zzp(rho_out, KS.vx, KS.vy, KS.vz) - EQ::eq_zzp(KS.rho, KS.vx, KS.vy, KS.vz);
		KS.f[dir27::zpm] += EQ::eq_zpm(rho_out, KS.vx, KS.vy, KS.vz) - EQ::eq_zpm(KS.rho, KS.vx, KS.vy, KS.vz);
		KS.f[dir27::zpz] += EQ::eq_zpz(rho_out, KS.vx, KS.vy, KS.vz) - EQ::eq_zpz(KS.rho, KS.vx, KS.vy, KS.vz);
		KS.f[dir27::zpp] += EQ::eq_zpp(rho_out, KS.vx, KS.vy, KS.vz) - EQ::eq_zpp(KS.rho, KS.vx, KS.vy, KS.vz);
		KS.f[dir27::pmm] += EQ::eq_pmm(rho_out, KS.vx, KS.vy, KS.vz) - EQ::eq_pmm(KS.rho, KS.vx, KS.vy, KS.vz);
		KS.f[dir27::pmz] += EQ::eq_pmz(rho_out, KS.vx, KS.vy, KS.vz) - EQ::eq_pmz(KS.rho, KS.vx, KS.vy, KS.vz);
		KS.f[dir27::pmp] += EQ::eq_pmp(rho_out, KS.vx, KS.vy, KS.vz) - EQ::eq_pmp(KS.rho, KS.vx, KS.vy, KS.vz);
		KS.f[dir27::pzm] += EQ::eq_pzm(rho_out, KS.vx, KS.vy, KS.vz) - EQ::eq_pzm(KS.rho, KS.vx, KS.vy, KS.vz);
		KS.f[dir27::pzz] += EQ::eq_pzz(rho_out, KS.vx, KS.vy, KS.vz) - EQ::eq_pzz(KS.rho, KS.vx, KS.vy, KS.vz);
		KS.f[dir27::pzp] += EQ::eq_pzp(rho_out, KS.vx, KS.vy, KS.vz) - EQ::eq_pzp(KS.rho, KS.vx, KS.vy, KS.vz);
		KS.f[dir27::ppm] += EQ::eq_ppm(rho_out, KS.vx, KS.vy, KS.vz) - EQ::eq_ppm(KS.rho, KS.vx, KS.vy, KS.vz);
		KS.f[dir27::ppz] += EQ::eq_ppz(rho_out, KS.vx, KS.vy, KS.vz) - EQ::eq_ppz(KS.rho, KS.vx, KS.vy, KS.vz);
		KS.f[dir27::ppp] += EQ::eq_ppp(rho_out, KS.vx, KS.vy, KS.vz) - EQ::eq_ppp(KS.rho, KS.vx, KS.vy, KS.vz);
	}
};

// Legacy alias
template <typename T_TRAITS, typename T_EQ>
struct D3Q27_COMMON_WELL : public D3Q27_COMMON<T_TRAITS, T_EQ, true>
{};
