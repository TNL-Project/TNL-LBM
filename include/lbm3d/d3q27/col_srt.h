#pragma once

#include "common.h"
#include "eq.h"

// improved BRK (SRT) model by Geier 2017
// for standard DF (no well-conditioned)

template <typename TRAITS, typename LBM_EQ = D3Q27_EQ<TRAITS>>
struct D3Q27_SRT : D3Q27_COMMON<TRAITS, LBM_EQ>
{
	using dreal = typename TRAITS::dreal;

	static constexpr const char* id = "SRT";

	template <typename LBM_KS>
	__cuda_callable__ static void collision(LBM_KS& KS)
	{
		const dreal tau = no3 * KS.lbmViscosity + n1o2;

		// NOTE: rho for ADE can be 0 (but all fx,fy,fz are 0...)
		const dreal iRho = no1 / (KS.rho == 0 ? no1 : KS.rho);

		// forcing: vzorce_bgk_force.mw
		const dreal Smmm = (no3 * ((-KS.vx - no1) * KS.fx + (-KS.vy - no1) * KS.fy + (-KS.vz - no1) * KS.fz)) * iRho;
		const dreal Szmm = (no3 * (-KS.vx * KS.fx + (-KS.vy - no1) * KS.fy + (-KS.vz - no1) * KS.fz)) * iRho;
		const dreal Spmm = (no3 * ((-KS.vx + no1) * KS.fx + (-KS.vy - no1) * KS.fy + (-KS.vz - no1) * KS.fz)) * iRho;
		const dreal Smzm = (no3 * ((-KS.vx - no1) * KS.fx - KS.vy * KS.fy + (-KS.vz - no1) * KS.fz)) * iRho;
		const dreal Szzm = (no3 * (-KS.vx * KS.fx - KS.vy * KS.fy + (-KS.vz - no1) * KS.fz)) * iRho;
		const dreal Spzm = (no3 * ((-KS.vx + no1) * KS.fx - KS.vy * KS.fy + (-KS.vz - no1) * KS.fz)) * iRho;
		const dreal Smpm = (no3 * ((-KS.vx - no1) * KS.fx + (-KS.vy + no1) * KS.fy + (-KS.vz - no1) * KS.fz)) * iRho;
		const dreal Szpm = (no3 * (-KS.vx * KS.fx + (-KS.vy + no1) * KS.fy + (-KS.vz - no1) * KS.fz)) * iRho;
		const dreal Sppm = (no3 * ((-KS.vx + no1) * KS.fx + (-KS.vy + no1) * KS.fy + (-KS.vz - no1) * KS.fz)) * iRho;
		const dreal Smmz = (no3 * ((-KS.vx - no1) * KS.fx + (-KS.vy - no1) * KS.fy - KS.vz * KS.fz)) * iRho;
		const dreal Szmz = (no3 * (-KS.vx * KS.fx + (-KS.vy - no1) * KS.fy - KS.vz * KS.fz)) * iRho;
		const dreal Spmz = (no3 * ((-KS.vx + no1) * KS.fx + (-KS.vy - no1) * KS.fy - KS.vz * KS.fz)) * iRho;
		const dreal Smzz = (no3 * ((-KS.vx - no1) * KS.fx - KS.vy * KS.fy - KS.vz * KS.fz)) * iRho;
		const dreal Szzz = (no3 * (-KS.fx * KS.vx - KS.fy * KS.vy - KS.fz * KS.vz)) * iRho;
		const dreal Spzz = (no3 * ((-KS.vx + no1) * KS.fx - KS.vy * KS.fy - KS.vz * KS.fz)) * iRho;
		const dreal Smpz = (no3 * ((-KS.vx - no1) * KS.fx + (-KS.vy + no1) * KS.fy - KS.vz * KS.fz)) * iRho;
		const dreal Szpz = (no3 * (-KS.vx * KS.fx + (-KS.vy + no1) * KS.fy - KS.vz * KS.fz)) * iRho;
		const dreal Sppz = (no3 * ((-KS.vx + no1) * KS.fx + (-KS.vy + no1) * KS.fy - KS.vz * KS.fz)) * iRho;
		const dreal Smmp = (no3 * ((-KS.vx - no1) * KS.fx + (-KS.vy - no1) * KS.fy + (-KS.vz + no1) * KS.fz)) * iRho;
		const dreal Szmp = (no3 * (-KS.vx * KS.fx + (-KS.vy - no1) * KS.fy + (-KS.vz + no1) * KS.fz)) * iRho;
		const dreal Spmp = (no3 * ((-KS.vx + no1) * KS.fx + (-KS.vy - no1) * KS.fy + (-KS.vz + no1) * KS.fz)) * iRho;
		const dreal Smzp = (no3 * ((-KS.vx - no1) * KS.fx - KS.vy * KS.fy + (-KS.vz + no1) * KS.fz)) * iRho;
		const dreal Szzp = (no3 * (-KS.vx * KS.fx - KS.vy * KS.fy + (-KS.vz + no1) * KS.fz)) * iRho;
		const dreal Spzp = (no3 * ((-KS.vx + no1) * KS.fx - KS.vy * KS.fy + (-KS.vz + no1) * KS.fz)) * iRho;
		const dreal Smpp = (no3 * ((-KS.vx - no1) * KS.fx + (-KS.vy + no1) * KS.fy + (-KS.vz + no1) * KS.fz)) * iRho;
		const dreal Szpp = (no3 * (-KS.vx * KS.fx + (-KS.vy + no1) * KS.fy + (-KS.vz + no1) * KS.fz)) * iRho;
		const dreal Sppp = (no3 * ((-KS.vx + no1) * KS.fx + (-KS.vy + no1) * KS.fy + (-KS.vz + no1) * KS.fz)) * iRho;

		const dreal locfeq_mmm = LBM_EQ::eq_mmm(KS.rho, KS.vx, KS.vy, KS.vz);
		const dreal locfeq_mmz = LBM_EQ::eq_mmz(KS.rho, KS.vx, KS.vy, KS.vz);
		const dreal locfeq_mmp = LBM_EQ::eq_mmp(KS.rho, KS.vx, KS.vy, KS.vz);
		const dreal locfeq_mzm = LBM_EQ::eq_mzm(KS.rho, KS.vx, KS.vy, KS.vz);
		const dreal locfeq_mzz = LBM_EQ::eq_mzz(KS.rho, KS.vx, KS.vy, KS.vz);
		const dreal locfeq_mzp = LBM_EQ::eq_mzp(KS.rho, KS.vx, KS.vy, KS.vz);
		const dreal locfeq_mpm = LBM_EQ::eq_mpm(KS.rho, KS.vx, KS.vy, KS.vz);
		const dreal locfeq_mpz = LBM_EQ::eq_mpz(KS.rho, KS.vx, KS.vy, KS.vz);
		const dreal locfeq_mpp = LBM_EQ::eq_mpp(KS.rho, KS.vx, KS.vy, KS.vz);
		const dreal locfeq_zmm = LBM_EQ::eq_zmm(KS.rho, KS.vx, KS.vy, KS.vz);
		const dreal locfeq_zmz = LBM_EQ::eq_zmz(KS.rho, KS.vx, KS.vy, KS.vz);
		const dreal locfeq_zmp = LBM_EQ::eq_zmp(KS.rho, KS.vx, KS.vy, KS.vz);
		const dreal locfeq_zzm = LBM_EQ::eq_zzm(KS.rho, KS.vx, KS.vy, KS.vz);
		const dreal locfeq_zzz = LBM_EQ::eq_zzz(KS.rho, KS.vx, KS.vy, KS.vz);
		const dreal locfeq_zzp = LBM_EQ::eq_zzp(KS.rho, KS.vx, KS.vy, KS.vz);
		const dreal locfeq_zpm = LBM_EQ::eq_zpm(KS.rho, KS.vx, KS.vy, KS.vz);
		const dreal locfeq_zpz = LBM_EQ::eq_zpz(KS.rho, KS.vx, KS.vy, KS.vz);
		const dreal locfeq_zpp = LBM_EQ::eq_zpp(KS.rho, KS.vx, KS.vy, KS.vz);
		const dreal locfeq_pmm = LBM_EQ::eq_pmm(KS.rho, KS.vx, KS.vy, KS.vz);
		const dreal locfeq_pmz = LBM_EQ::eq_pmz(KS.rho, KS.vx, KS.vy, KS.vz);
		const dreal locfeq_pmp = LBM_EQ::eq_pmp(KS.rho, KS.vx, KS.vy, KS.vz);
		const dreal locfeq_pzm = LBM_EQ::eq_pzm(KS.rho, KS.vx, KS.vy, KS.vz);
		const dreal locfeq_pzz = LBM_EQ::eq_pzz(KS.rho, KS.vx, KS.vy, KS.vz);
		const dreal locfeq_pzp = LBM_EQ::eq_pzp(KS.rho, KS.vx, KS.vy, KS.vz);
		const dreal locfeq_ppm = LBM_EQ::eq_ppm(KS.rho, KS.vx, KS.vy, KS.vz);
		const dreal locfeq_ppz = LBM_EQ::eq_ppz(KS.rho, KS.vx, KS.vy, KS.vz);
		const dreal locfeq_ppp = LBM_EQ::eq_ppp(KS.rho, KS.vx, KS.vy, KS.vz);

		KS.f[mmm] += (locfeq_mmm - KS.f[mmm]) / tau + (no1 - n1o2 / tau) * Smmm * locfeq_mmm;
		KS.f[mmz] += (locfeq_mmz - KS.f[mmz]) / tau + (no1 - n1o2 / tau) * Smmz * locfeq_mmz;
		KS.f[mmp] += (locfeq_mmp - KS.f[mmp]) / tau + (no1 - n1o2 / tau) * Smmp * locfeq_mmp;
		KS.f[mzm] += (locfeq_mzm - KS.f[mzm]) / tau + (no1 - n1o2 / tau) * Smzm * locfeq_mzm;
		KS.f[mzz] += (locfeq_mzz - KS.f[mzz]) / tau + (no1 - n1o2 / tau) * Smzz * locfeq_mzz;
		KS.f[mzp] += (locfeq_mzp - KS.f[mzp]) / tau + (no1 - n1o2 / tau) * Smzp * locfeq_mzp;
		KS.f[mpm] += (locfeq_mpm - KS.f[mpm]) / tau + (no1 - n1o2 / tau) * Smpm * locfeq_mpm;
		KS.f[mpz] += (locfeq_mpz - KS.f[mpz]) / tau + (no1 - n1o2 / tau) * Smpz * locfeq_mpz;
		KS.f[mpp] += (locfeq_mpp - KS.f[mpp]) / tau + (no1 - n1o2 / tau) * Smpp * locfeq_mpp;
		KS.f[zmm] += (locfeq_zmm - KS.f[zmm]) / tau + (no1 - n1o2 / tau) * Szmm * locfeq_zmm;
		KS.f[zmz] += (locfeq_zmz - KS.f[zmz]) / tau + (no1 - n1o2 / tau) * Szmz * locfeq_zmz;
		KS.f[zmp] += (locfeq_zmp - KS.f[zmp]) / tau + (no1 - n1o2 / tau) * Szmp * locfeq_zmp;
		KS.f[zzm] += (locfeq_zzm - KS.f[zzm]) / tau + (no1 - n1o2 / tau) * Szzm * locfeq_zzm;
		KS.f[zzz] += (locfeq_zzz - KS.f[zzz]) / tau + (no1 - n1o2 / tau) * Szzz * locfeq_zzz;
		KS.f[zzp] += (locfeq_zzp - KS.f[zzp]) / tau + (no1 - n1o2 / tau) * Szzp * locfeq_zzp;
		KS.f[zpm] += (locfeq_zpm - KS.f[zpm]) / tau + (no1 - n1o2 / tau) * Szpm * locfeq_zpm;
		KS.f[zpz] += (locfeq_zpz - KS.f[zpz]) / tau + (no1 - n1o2 / tau) * Szpz * locfeq_zpz;
		KS.f[zpp] += (locfeq_zpp - KS.f[zpp]) / tau + (no1 - n1o2 / tau) * Szpp * locfeq_zpp;
		KS.f[pmm] += (locfeq_pmm - KS.f[pmm]) / tau + (no1 - n1o2 / tau) * Spmm * locfeq_pmm;
		KS.f[pmz] += (locfeq_pmz - KS.f[pmz]) / tau + (no1 - n1o2 / tau) * Spmz * locfeq_pmz;
		KS.f[pmp] += (locfeq_pmp - KS.f[pmp]) / tau + (no1 - n1o2 / tau) * Spmp * locfeq_pmp;
		KS.f[pzm] += (locfeq_pzm - KS.f[pzm]) / tau + (no1 - n1o2 / tau) * Spzm * locfeq_pzm;
		KS.f[pzz] += (locfeq_pzz - KS.f[pzz]) / tau + (no1 - n1o2 / tau) * Spzz * locfeq_pzz;
		KS.f[pzp] += (locfeq_pzp - KS.f[pzp]) / tau + (no1 - n1o2 / tau) * Spzp * locfeq_pzp;
		KS.f[ppm] += (locfeq_ppm - KS.f[ppm]) / tau + (no1 - n1o2 / tau) * Sppm * locfeq_ppm;
		KS.f[ppz] += (locfeq_ppz - KS.f[ppz]) / tau + (no1 - n1o2 / tau) * Sppz * locfeq_ppz;
		KS.f[ppp] += (locfeq_ppp - KS.f[ppp]) / tau + (no1 - n1o2 / tau) * Sppp * locfeq_ppp;
	}
};
