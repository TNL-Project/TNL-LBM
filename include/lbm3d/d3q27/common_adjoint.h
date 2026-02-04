#pragma once

#include "lbm3d/defs.h"
#include "lbm_common/ciselnik.h"

template <typename T_TRAITS, typename T_EQ>
struct D3Q27_COMMON_ADJOINT
{
	using TRAITS = T_TRAITS;
	using EQ = T_EQ;

	using idx = typename TRAITS::idx;
	using real = typename TRAITS::real;
	using dreal = typename TRAITS::dreal;

	template <typename LBM_KS>
	__cuda_callable__ static void computeDensityAndVelocity(LBM_KS& KS)
	{
		// TODO: if KS had the gradient (gx, gy, gz) it should be computed here
		// computeGradient(KS);
	}

	template <typename LBM_KS>
	__cuda_callable__ static dreal collisionOutflow(LBM_KS& KS, int qx, int qy, int qz, dreal vx, dreal vy, dreal vz)
	{
		// clang-format off
		return    KS.f[zzz]*n8o27*EQ::feq_v( 0, 0, 0, qx, qy, qz, vx, vy, vz)

				+ KS.f[pzz]*n2o27*EQ::feq_v( 1, 0, 0, qx, qy, qz, vx, vy, vz)
				+ KS.f[mzz]*n2o27*EQ::feq_v(-1, 0, 0, qx, qy, qz, vx, vy, vz)
				+ KS.f[zpz]*n2o27*EQ::feq_v( 0, 1, 0, qx, qy, qz, vx, vy, vz)
				+ KS.f[zmz]*n2o27*EQ::feq_v( 0,-1, 0, qx, qy, qz, vx, vy, vz)
				+ KS.f[zzp]*n2o27*EQ::feq_v( 0, 0, 1, qx, qy, qz, vx, vy, vz)
				+ KS.f[zzm]*n2o27*EQ::feq_v( 0, 0,-1, qx, qy, qz, vx, vy, vz)

				+ KS.f[ppz]*n1o54*EQ::feq_v( 1, 1, 0, qx, qy, qz, vx, vy, vz)
				+ KS.f[pmz]*n1o54*EQ::feq_v( 1,-1, 0, qx, qy, qz, vx, vy, vz)
				+ KS.f[mpz]*n1o54*EQ::feq_v(-1, 1, 0, qx, qy, qz, vx, vy, vz)
				+ KS.f[mmz]*n1o54*EQ::feq_v(-1,-1, 0, qx, qy, qz, vx, vy, vz)
				+ KS.f[pzp]*n1o54*EQ::feq_v( 1, 0, 1, qx, qy, qz, vx, vy, vz)
				+ KS.f[mzm]*n1o54*EQ::feq_v(-1, 0,-1, qx, qy, qz, vx, vy, vz)
				+ KS.f[pzm]*n1o54*EQ::feq_v( 1, 0,-1, qx, qy, qz, vx, vy, vz)
				+ KS.f[mzp]*n1o54*EQ::feq_v(-1, 0, 1, qx, qy, qz, vx, vy, vz)
				+ KS.f[zpp]*n1o54*EQ::feq_v( 0, 1, 1, qx, qy, qz, vx, vy, vz)
				+ KS.f[zpm]*n1o54*EQ::feq_v( 0, 1,-1, qx, qy, qz, vx, vy, vz)
				+ KS.f[zmp]*n1o54*EQ::feq_v( 0,-1, 1, qx, qy, qz, vx, vy, vz)
				+ KS.f[zmm]*n1o54*EQ::feq_v( 0,-1,-1, qx, qy, qz, vx, vy, vz)

				+ KS.f[ppp]*n1o216*EQ::feq_v( 1, 1, 1, qx, qy, qz, vx, vy, vz)
				+ KS.f[mmm]*n1o216*EQ::feq_v(-1,-1,-1, qx, qy, qz, vx, vy, vz)
				+ KS.f[ppm]*n1o216*EQ::feq_v( 1, 1,-1, qx, qy, qz, vx, vy, vz)
				+ KS.f[pmp]*n1o216*EQ::feq_v( 1,-1, 1, qx, qy, qz, vx, vy, vz)
				+ KS.f[mpp]*n1o216*EQ::feq_v(-1, 1, 1, qx, qy, qz, vx, vy, vz)
				+ KS.f[mpm]*n1o216*EQ::feq_v(-1, 1,-1, qx, qy, qz, vx, vy, vz)
				+ KS.f[mmp]*n1o216*EQ::feq_v(-1,-1, 1, qx, qy, qz, vx, vy, vz)
				+ KS.f[pmm]*n1o216*EQ::feq_v( 1,-1,-1, qx, qy, qz, vx, vy, vz)
				;
		// clang-format on
	}

	template <typename LBM_KS>
	__cuda_callable__ static void computeDensityAndVelocity_Wall(LBM_KS& KS)
	{
		const dreal tau = no3 * KS.lbmViscosity + n1o2;
		// outflow condition - it is not a wall,  but it is needed
		const dreal locfeq_zzz = collisionOutflow(KS, 0, 0, 0, KS.vx, KS.vy, KS.vz);
		const dreal locfeq_pzz = collisionOutflow(KS, 1, 0, 0, KS.vx, KS.vy, KS.vz);
		const dreal locfeq_mzz = collisionOutflow(KS, -1, 0, 0, KS.vx, KS.vy, KS.vz);
		const dreal locfeq_zpz = collisionOutflow(KS, 0, 1, 0, KS.vx, KS.vy, KS.vz);
		const dreal locfeq_zmz = collisionOutflow(KS, 0, -1, 0, KS.vx, KS.vy, KS.vz);
		const dreal locfeq_zzp = collisionOutflow(KS, 0, 0, 1, KS.vx, KS.vy, KS.vz);
		const dreal locfeq_zzm = collisionOutflow(KS, 0, 0, -1, KS.vx, KS.vy, KS.vz);
		const dreal locfeq_ppz = collisionOutflow(KS, 1, 1, 0, KS.vx, KS.vy, KS.vz);
		const dreal locfeq_pmz = collisionOutflow(KS, 1, -1, 0, KS.vx, KS.vy, KS.vz);
		const dreal locfeq_mpz = collisionOutflow(KS, -1, 1, 0, KS.vx, KS.vy, KS.vz);
		const dreal locfeq_mmz = collisionOutflow(KS, -1, -1, 0, KS.vx, KS.vy, KS.vz);
		const dreal locfeq_pzp = collisionOutflow(KS, 1, 0, 1, KS.vx, KS.vy, KS.vz);
		const dreal locfeq_mzm = collisionOutflow(KS, -1, 0, -1, KS.vx, KS.vy, KS.vz);
		const dreal locfeq_pzm = collisionOutflow(KS, 1, 0, -1, KS.vx, KS.vy, KS.vz);
		const dreal locfeq_mzp = collisionOutflow(KS, -1, 0, 1, KS.vx, KS.vy, KS.vz);
		const dreal locfeq_zpp = collisionOutflow(KS, 0, 1, 1, KS.vx, KS.vy, KS.vz);
		const dreal locfeq_zpm = collisionOutflow(KS, 0, 1, -1, KS.vx, KS.vy, KS.vz);
		const dreal locfeq_zmp = collisionOutflow(KS, 0, -1, 1, KS.vx, KS.vy, KS.vz);
		const dreal locfeq_zmm = collisionOutflow(KS, 0, -1, -1, KS.vx, KS.vy, KS.vz);
		const dreal locfeq_ppp = collisionOutflow(KS, 1, 1, 1, KS.vx, KS.vy, KS.vz);
		const dreal locfeq_mmm = collisionOutflow(KS, -1, -1, -1, KS.vx, KS.vy, KS.vz);
		const dreal locfeq_ppm = collisionOutflow(KS, 1, 1, -1, KS.vx, KS.vy, KS.vz);
		const dreal locfeq_pmp = collisionOutflow(KS, 1, -1, 1, KS.vx, KS.vy, KS.vz);
		const dreal locfeq_mpp = collisionOutflow(KS, -1, 1, 1, KS.vx, KS.vy, KS.vz);
		const dreal locfeq_mpm = collisionOutflow(KS, -1, 1, -1, KS.vx, KS.vy, KS.vz);
		const dreal locfeq_mmp = collisionOutflow(KS, -1, -1, 1, KS.vx, KS.vy, KS.vz);
		const dreal locfeq_pmm = collisionOutflow(KS, 1, -1, -1, KS.vx, KS.vy, KS.vz);

		KS.f[mmm] += (locfeq_mmm - KS.f[mmm]) / tau /* + (no1 - n1o2/tau)*Smmm*locfeq_mmm */;
		KS.f[mmz] += (locfeq_mmz - KS.f[mmz]) / tau /* + (no1 - n1o2/tau)*Smmz*locfeq_mmz */;
		KS.f[mmp] += (locfeq_mmp - KS.f[mmp]) / tau /* + (no1 - n1o2/tau)*Smmp*locfeq_mmp */;
		KS.f[mzm] += (locfeq_mzm - KS.f[mzm]) / tau /* + (no1 - n1o2/tau)*Smzm*locfeq_mzm */;
		KS.f[mzz] += (locfeq_mzz - KS.f[mzz]) / tau /* + (no1 - n1o2/tau)*Smzz*locfeq_mzz */;
		KS.f[mzp] += (locfeq_mzp - KS.f[mzp]) / tau /* + (no1 - n1o2/tau)*Smzp*locfeq_mzp */;
		KS.f[mpm] += (locfeq_mpm - KS.f[mpm]) / tau /* + (no1 - n1o2/tau)*Smpm*locfeq_mpm */;
		KS.f[mpz] += (locfeq_mpz - KS.f[mpz]) / tau /* + (no1 - n1o2/tau)*Smpz*locfeq_mpz */;
		KS.f[mpp] += (locfeq_mpp - KS.f[mpp]) / tau /* + (no1 - n1o2/tau)*Smpp*locfeq_mpp */;
		KS.f[zmm] += (locfeq_zmm - KS.f[zmm]) / tau /* + (no1 - n1o2/tau)*Szmm*locfeq_zmm */;
		KS.f[zmz] += (locfeq_zmz - KS.f[zmz]) / tau /* + (no1 - n1o2/tau)*Szmz*locfeq_zmz */;
		KS.f[zmp] += (locfeq_zmp - KS.f[zmp]) / tau /* + (no1 - n1o2/tau)*Szmp*locfeq_zmp */;
		KS.f[zzm] += (locfeq_zzm - KS.f[zzm]) / tau /* + (no1 - n1o2/tau)*Szzm*locfeq_zzm */;
		KS.f[zzz] += (locfeq_zzz - KS.f[zzz]) / tau /* + (no1 - n1o2/tau)*Szzz*locfeq_zzz */;
		KS.f[zzp] += (locfeq_zzp - KS.f[zzp]) / tau /* + (no1 - n1o2/tau)*Szzp*locfeq_zzp */;
		KS.f[zpm] += (locfeq_zpm - KS.f[zpm]) / tau /* + (no1 - n1o2/tau)*Szpm*locfeq_zpm */;
		KS.f[zpz] += (locfeq_zpz - KS.f[zpz]) / tau /* + (no1 - n1o2/tau)*Szpz*locfeq_zpz */;
		KS.f[zpp] += (locfeq_zpp - KS.f[zpp]) / tau /* + (no1 - n1o2/tau)*Szpp*locfeq_zpp */;
		KS.f[pmm] += (locfeq_pmm - KS.f[pmm]) / tau /* + (no1 - n1o2/tau)*Spmm*locfeq_pmm */;
		KS.f[pmz] += (locfeq_pmz - KS.f[pmz]) / tau /* + (no1 - n1o2/tau)*Spmz*locfeq_pmz */;
		KS.f[pmp] += (locfeq_pmp - KS.f[pmp]) / tau /* + (no1 - n1o2/tau)*Spmp*locfeq_pmp */;
		KS.f[pzm] += (locfeq_pzm - KS.f[pzm]) / tau /* + (no1 - n1o2/tau)*Spzm*locfeq_pzm */;
		KS.f[pzz] += (locfeq_pzz - KS.f[pzz]) / tau /* + (no1 - n1o2/tau)*Spzz*locfeq_pzz */;
		KS.f[pzp] += (locfeq_pzp - KS.f[pzp]) / tau /* + (no1 - n1o2/tau)*Spzp*locfeq_pzp */;
		KS.f[ppm] += (locfeq_ppm - KS.f[ppm]) / tau /* + (no1 - n1o2/tau)*Sppm*locfeq_ppm */;
		KS.f[ppz] += (locfeq_ppz - KS.f[ppz]) / tau /* + (no1 - n1o2/tau)*Sppz*locfeq_ppz */;
		KS.f[ppp] += (locfeq_ppp - KS.f[ppp]) / tau /* + (no1 - n1o2/tau)*Sppp*locfeq_ppp */;
	}

	template <typename LBM_KS>
	__cuda_callable__ static void setEquilibrium(LBM_KS& KS)
	{
		KS.f[mmm] += EQ::eq_mmm(KS.rho, KS.vx, KS.vy, KS.vz, KS.rho_m, KS.vx_m, KS.vy_m, KS.vz_m);
		KS.f[mmz] += EQ::eq_mmz(KS.rho, KS.vx, KS.vy, KS.vz, KS.rho_m, KS.vx_m, KS.vy_m, KS.vz_m);
		KS.f[mmp] += EQ::eq_mmp(KS.rho, KS.vx, KS.vy, KS.vz, KS.rho_m, KS.vx_m, KS.vy_m, KS.vz_m);
		KS.f[mzm] += EQ::eq_mzm(KS.rho, KS.vx, KS.vy, KS.vz, KS.rho_m, KS.vx_m, KS.vy_m, KS.vz_m);
		KS.f[mzz] += EQ::eq_mzz(KS.rho, KS.vx, KS.vy, KS.vz, KS.rho_m, KS.vx_m, KS.vy_m, KS.vz_m);
		KS.f[mzp] += EQ::eq_mzp(KS.rho, KS.vx, KS.vy, KS.vz, KS.rho_m, KS.vx_m, KS.vy_m, KS.vz_m);
		KS.f[mpm] += EQ::eq_mpm(KS.rho, KS.vx, KS.vy, KS.vz, KS.rho_m, KS.vx_m, KS.vy_m, KS.vz_m);
		KS.f[mpz] += EQ::eq_mpz(KS.rho, KS.vx, KS.vy, KS.vz, KS.rho_m, KS.vx_m, KS.vy_m, KS.vz_m);
		KS.f[mpp] += EQ::eq_mpp(KS.rho, KS.vx, KS.vy, KS.vz, KS.rho_m, KS.vx_m, KS.vy_m, KS.vz_m);
		KS.f[zmm] += EQ::eq_zmm(KS.rho, KS.vx, KS.vy, KS.vz, KS.rho_m, KS.vx_m, KS.vy_m, KS.vz_m);
		KS.f[zmz] += EQ::eq_zmz(KS.rho, KS.vx, KS.vy, KS.vz, KS.rho_m, KS.vx_m, KS.vy_m, KS.vz_m);
		KS.f[zmp] += EQ::eq_zmp(KS.rho, KS.vx, KS.vy, KS.vz, KS.rho_m, KS.vx_m, KS.vy_m, KS.vz_m);
		KS.f[zzm] += EQ::eq_zzm(KS.rho, KS.vx, KS.vy, KS.vz, KS.rho_m, KS.vx_m, KS.vy_m, KS.vz_m);
		KS.f[zzz] += EQ::eq_zzz(KS.rho, KS.vx, KS.vy, KS.vz, KS.rho_m, KS.vx_m, KS.vy_m, KS.vz_m);
		KS.f[zzp] += EQ::eq_zzp(KS.rho, KS.vx, KS.vy, KS.vz, KS.rho_m, KS.vx_m, KS.vy_m, KS.vz_m);
		KS.f[zpm] += EQ::eq_zpm(KS.rho, KS.vx, KS.vy, KS.vz, KS.rho_m, KS.vx_m, KS.vy_m, KS.vz_m);
		KS.f[zpz] += EQ::eq_zpz(KS.rho, KS.vx, KS.vy, KS.vz, KS.rho_m, KS.vx_m, KS.vy_m, KS.vz_m);
		KS.f[zpp] += EQ::eq_zpp(KS.rho, KS.vx, KS.vy, KS.vz, KS.rho_m, KS.vx_m, KS.vy_m, KS.vz_m);
		KS.f[pmm] += EQ::eq_pmm(KS.rho, KS.vx, KS.vy, KS.vz, KS.rho_m, KS.vx_m, KS.vy_m, KS.vz_m);
		KS.f[pmz] += EQ::eq_pmz(KS.rho, KS.vx, KS.vy, KS.vz, KS.rho_m, KS.vx_m, KS.vy_m, KS.vz_m);
		KS.f[pmp] += EQ::eq_pmp(KS.rho, KS.vx, KS.vy, KS.vz, KS.rho_m, KS.vx_m, KS.vy_m, KS.vz_m);
		KS.f[pzm] += EQ::eq_pzm(KS.rho, KS.vx, KS.vy, KS.vz, KS.rho_m, KS.vx_m, KS.vy_m, KS.vz_m);
		KS.f[pzz] += EQ::eq_pzz(KS.rho, KS.vx, KS.vy, KS.vz, KS.rho_m, KS.vx_m, KS.vy_m, KS.vz_m);
		KS.f[pzp] += EQ::eq_pzp(KS.rho, KS.vx, KS.vy, KS.vz, KS.rho_m, KS.vx_m, KS.vy_m, KS.vz_m);
		KS.f[ppm] += EQ::eq_ppm(KS.rho, KS.vx, KS.vy, KS.vz, KS.rho_m, KS.vx_m, KS.vy_m, KS.vz_m);
		KS.f[ppz] += EQ::eq_ppz(KS.rho, KS.vx, KS.vy, KS.vz, KS.rho_m, KS.vx_m, KS.vy_m, KS.vz_m);
		KS.f[ppp] += EQ::eq_ppp(KS.rho, KS.vx, KS.vy, KS.vz, KS.rho_m, KS.vx_m, KS.vy_m, KS.vz_m);
	}

	// used in the "interpolated outflow boundary condition with decomposition" by Eichler https://doi.org/10.1016/j.camwa.2024.08.009
	template <typename LBM_KS>
	__cuda_callable__ static void setEquilibriumDecomposition(LBM_KS& KS, dreal rho_out)
	{
		// FIXME: update this for adjoint
#if 0
		KS.f[mmm] += EQ::eq_mmm(rho_out,KS.vx,KS.vy,KS.vz) - EQ::eq_mmm(KS.rho,KS.vx,KS.vy,KS.vz);
		KS.f[mmz] += EQ::eq_mmz(rho_out,KS.vx,KS.vy,KS.vz) - EQ::eq_mmz(KS.rho,KS.vx,KS.vy,KS.vz);
		KS.f[mmp] += EQ::eq_mmp(rho_out,KS.vx,KS.vy,KS.vz) - EQ::eq_mmp(KS.rho,KS.vx,KS.vy,KS.vz);
		KS.f[mzm] += EQ::eq_mzm(rho_out,KS.vx,KS.vy,KS.vz) - EQ::eq_mzm(KS.rho,KS.vx,KS.vy,KS.vz);
		KS.f[mzz] += EQ::eq_mzz(rho_out,KS.vx,KS.vy,KS.vz) - EQ::eq_mzz(KS.rho,KS.vx,KS.vy,KS.vz);
		KS.f[mzp] += EQ::eq_mzp(rho_out,KS.vx,KS.vy,KS.vz) - EQ::eq_mzp(KS.rho,KS.vx,KS.vy,KS.vz);
		KS.f[mpm] += EQ::eq_mpm(rho_out,KS.vx,KS.vy,KS.vz) - EQ::eq_mpm(KS.rho,KS.vx,KS.vy,KS.vz);
		KS.f[mpz] += EQ::eq_mpz(rho_out,KS.vx,KS.vy,KS.vz) - EQ::eq_mpz(KS.rho,KS.vx,KS.vy,KS.vz);
		KS.f[mpp] += EQ::eq_mpp(rho_out,KS.vx,KS.vy,KS.vz) - EQ::eq_mpp(KS.rho,KS.vx,KS.vy,KS.vz);
		KS.f[zmm] += EQ::eq_zmm(rho_out,KS.vx,KS.vy,KS.vz) - EQ::eq_zmm(KS.rho,KS.vx,KS.vy,KS.vz);
		KS.f[zmz] += EQ::eq_zmz(rho_out,KS.vx,KS.vy,KS.vz) - EQ::eq_zmz(KS.rho,KS.vx,KS.vy,KS.vz);
		KS.f[zmp] += EQ::eq_zmp(rho_out,KS.vx,KS.vy,KS.vz) - EQ::eq_zmp(KS.rho,KS.vx,KS.vy,KS.vz);
		KS.f[zzm] += EQ::eq_zzm(rho_out,KS.vx,KS.vy,KS.vz) - EQ::eq_zzm(KS.rho,KS.vx,KS.vy,KS.vz);
		KS.f[zzz] += EQ::eq_zzz(rho_out,KS.vx,KS.vy,KS.vz) - EQ::eq_zzz(KS.rho,KS.vx,KS.vy,KS.vz);
		KS.f[zzp] += EQ::eq_zzp(rho_out,KS.vx,KS.vy,KS.vz) - EQ::eq_zzp(KS.rho,KS.vx,KS.vy,KS.vz);
		KS.f[zpm] += EQ::eq_zpm(rho_out,KS.vx,KS.vy,KS.vz) - EQ::eq_zpm(KS.rho,KS.vx,KS.vy,KS.vz);
		KS.f[zpz] += EQ::eq_zpz(rho_out,KS.vx,KS.vy,KS.vz) - EQ::eq_zpz(KS.rho,KS.vx,KS.vy,KS.vz);
		KS.f[zpp] += EQ::eq_zpp(rho_out,KS.vx,KS.vy,KS.vz) - EQ::eq_zpp(KS.rho,KS.vx,KS.vy,KS.vz);
		KS.f[pmm] += EQ::eq_pmm(rho_out,KS.vx,KS.vy,KS.vz) - EQ::eq_pmm(KS.rho,KS.vx,KS.vy,KS.vz);
		KS.f[pmz] += EQ::eq_pmz(rho_out,KS.vx,KS.vy,KS.vz) - EQ::eq_pmz(KS.rho,KS.vx,KS.vy,KS.vz);
		KS.f[pmp] += EQ::eq_pmp(rho_out,KS.vx,KS.vy,KS.vz) - EQ::eq_pmp(KS.rho,KS.vx,KS.vy,KS.vz);
		KS.f[pzm] += EQ::eq_pzm(rho_out,KS.vx,KS.vy,KS.vz) - EQ::eq_pzm(KS.rho,KS.vx,KS.vy,KS.vz);
		KS.f[pzz] += EQ::eq_pzz(rho_out,KS.vx,KS.vy,KS.vz) - EQ::eq_pzz(KS.rho,KS.vx,KS.vy,KS.vz);
		KS.f[pzp] += EQ::eq_pzp(rho_out,KS.vx,KS.vy,KS.vz) - EQ::eq_pzp(KS.rho,KS.vx,KS.vy,KS.vz);
		KS.f[ppm] += EQ::eq_ppm(rho_out,KS.vx,KS.vy,KS.vz) - EQ::eq_ppm(KS.rho,KS.vx,KS.vy,KS.vz);
		KS.f[ppz] += EQ::eq_ppz(rho_out,KS.vx,KS.vy,KS.vz) - EQ::eq_ppz(KS.rho,KS.vx,KS.vy,KS.vz);
		KS.f[ppp] += EQ::eq_ppp(rho_out,KS.vx,KS.vy,KS.vz) - EQ::eq_ppp(KS.rho,KS.vx,KS.vy,KS.vz);
#endif
	}

	template <typename LAT_DFS>
	__cuda_callable__ static void setEquilibriumLat(LAT_DFS& f, idx x, idx y, idx z, real rho, real vx, real vy, real vz)
	{
		//! only called during initialization
		// TODO: initialize adjoint dfs - 0, before any calculation, there is collision step,
		// where the measured data sets initial dfs for adjoint problem
		f(mmm, x, y, z) = 0;
		f(zmm, x, y, z) = 0;
		f(pmm, x, y, z) = 0;
		f(mzm, x, y, z) = 0;
		f(zzm, x, y, z) = 0;
		f(pzm, x, y, z) = 0;
		f(mpm, x, y, z) = 0;
		f(zpm, x, y, z) = 0;
		f(ppm, x, y, z) = 0;

		f(mmz, x, y, z) = 0;
		f(zmz, x, y, z) = 0;
		f(pmz, x, y, z) = 0;
		f(mzz, x, y, z) = 0;
		f(zzz, x, y, z) = 0;
		f(pzz, x, y, z) = 0;
		f(mpz, x, y, z) = 0;
		f(zpz, x, y, z) = 0;
		f(ppz, x, y, z) = 0;

		f(mmp, x, y, z) = 0;
		f(zmp, x, y, z) = 0;
		f(pmp, x, y, z) = 0;
		f(mzp, x, y, z) = 0;
		f(zzp, x, y, z) = 0;
		f(pzp, x, y, z) = 0;
		f(mpp, x, y, z) = 0;
		f(zpp, x, y, z) = 0;
		f(ppp, x, y, z) = 0;
	}
};
