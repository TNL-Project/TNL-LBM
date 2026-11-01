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

	// false: storage converters are identity (adjoint DFs stored as-is)
	static constexpr bool is_well_conditioned = false;

	// storage<->physical DF converters: identity for the adjoint convention, see COMMON_WELL for the shifted one
	__cuda_callable__ static constexpr dreal weight(int i)
	{
		return i == dir27::zzz ? n8o27 : (i <= dir27::zzm ? n2o27 : (i <= dir27::zmp ? n1o54 : n1o216));
	}

	__cuda_callable__ static dreal fromStorage(const dreal* f, int i)
	{
		return f[i];
	}

	__cuda_callable__ static dreal toStorage(dreal f, int i)
	{
		return f;
	}

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
		return    KS.f[dir27::zzz]*n8o27*EQ::feq_v( 0, 0, 0, qx, qy, qz, vx, vy, vz)

				+ KS.f[dir27::pzz]*n2o27*EQ::feq_v( 1, 0, 0, qx, qy, qz, vx, vy, vz)
				+ KS.f[dir27::mzz]*n2o27*EQ::feq_v(-1, 0, 0, qx, qy, qz, vx, vy, vz)
				+ KS.f[dir27::zpz]*n2o27*EQ::feq_v( 0, 1, 0, qx, qy, qz, vx, vy, vz)
				+ KS.f[dir27::zmz]*n2o27*EQ::feq_v( 0,-1, 0, qx, qy, qz, vx, vy, vz)
				+ KS.f[dir27::zzp]*n2o27*EQ::feq_v( 0, 0, 1, qx, qy, qz, vx, vy, vz)
				+ KS.f[dir27::zzm]*n2o27*EQ::feq_v( 0, 0,-1, qx, qy, qz, vx, vy, vz)

				+ KS.f[dir27::ppz]*n1o54*EQ::feq_v( 1, 1, 0, qx, qy, qz, vx, vy, vz)
				+ KS.f[dir27::pmz]*n1o54*EQ::feq_v( 1,-1, 0, qx, qy, qz, vx, vy, vz)
				+ KS.f[dir27::mpz]*n1o54*EQ::feq_v(-1, 1, 0, qx, qy, qz, vx, vy, vz)
				+ KS.f[dir27::mmz]*n1o54*EQ::feq_v(-1,-1, 0, qx, qy, qz, vx, vy, vz)
				+ KS.f[dir27::pzp]*n1o54*EQ::feq_v( 1, 0, 1, qx, qy, qz, vx, vy, vz)
				+ KS.f[dir27::mzm]*n1o54*EQ::feq_v(-1, 0,-1, qx, qy, qz, vx, vy, vz)
				+ KS.f[dir27::pzm]*n1o54*EQ::feq_v( 1, 0,-1, qx, qy, qz, vx, vy, vz)
				+ KS.f[dir27::mzp]*n1o54*EQ::feq_v(-1, 0, 1, qx, qy, qz, vx, vy, vz)
				+ KS.f[dir27::zpp]*n1o54*EQ::feq_v( 0, 1, 1, qx, qy, qz, vx, vy, vz)
				+ KS.f[dir27::zpm]*n1o54*EQ::feq_v( 0, 1,-1, qx, qy, qz, vx, vy, vz)
				+ KS.f[dir27::zmp]*n1o54*EQ::feq_v( 0,-1, 1, qx, qy, qz, vx, vy, vz)
				+ KS.f[dir27::zmm]*n1o54*EQ::feq_v( 0,-1,-1, qx, qy, qz, vx, vy, vz)

				+ KS.f[dir27::ppp]*n1o216*EQ::feq_v( 1, 1, 1, qx, qy, qz, vx, vy, vz)
				+ KS.f[dir27::mmm]*n1o216*EQ::feq_v(-1,-1,-1, qx, qy, qz, vx, vy, vz)
				+ KS.f[dir27::ppm]*n1o216*EQ::feq_v( 1, 1,-1, qx, qy, qz, vx, vy, vz)
				+ KS.f[dir27::pmp]*n1o216*EQ::feq_v( 1,-1, 1, qx, qy, qz, vx, vy, vz)
				+ KS.f[dir27::mpp]*n1o216*EQ::feq_v(-1, 1, 1, qx, qy, qz, vx, vy, vz)
				+ KS.f[dir27::mpm]*n1o216*EQ::feq_v(-1, 1,-1, qx, qy, qz, vx, vy, vz)
				+ KS.f[dir27::mmp]*n1o216*EQ::feq_v(-1,-1, 1, qx, qy, qz, vx, vy, vz)
				+ KS.f[dir27::pmm]*n1o216*EQ::feq_v( 1,-1,-1, qx, qy, qz, vx, vy, vz)
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

		KS.f[dir27::mmm] += (locfeq_mmm - KS.f[dir27::mmm]) / tau /* + (no1 - n1o2/tau)*Smmm*locfeq_mmm */;
		KS.f[dir27::mmz] += (locfeq_mmz - KS.f[dir27::mmz]) / tau /* + (no1 - n1o2/tau)*Smmz*locfeq_mmz */;
		KS.f[dir27::mmp] += (locfeq_mmp - KS.f[dir27::mmp]) / tau /* + (no1 - n1o2/tau)*Smmp*locfeq_mmp */;
		KS.f[dir27::mzm] += (locfeq_mzm - KS.f[dir27::mzm]) / tau /* + (no1 - n1o2/tau)*Smzm*locfeq_mzm */;
		KS.f[dir27::mzz] += (locfeq_mzz - KS.f[dir27::mzz]) / tau /* + (no1 - n1o2/tau)*Smzz*locfeq_mzz */;
		KS.f[dir27::mzp] += (locfeq_mzp - KS.f[dir27::mzp]) / tau /* + (no1 - n1o2/tau)*Smzp*locfeq_mzp */;
		KS.f[dir27::mpm] += (locfeq_mpm - KS.f[dir27::mpm]) / tau /* + (no1 - n1o2/tau)*Smpm*locfeq_mpm */;
		KS.f[dir27::mpz] += (locfeq_mpz - KS.f[dir27::mpz]) / tau /* + (no1 - n1o2/tau)*Smpz*locfeq_mpz */;
		KS.f[dir27::mpp] += (locfeq_mpp - KS.f[dir27::mpp]) / tau /* + (no1 - n1o2/tau)*Smpp*locfeq_mpp */;
		KS.f[dir27::zmm] += (locfeq_zmm - KS.f[dir27::zmm]) / tau /* + (no1 - n1o2/tau)*Szmm*locfeq_zmm */;
		KS.f[dir27::zmz] += (locfeq_zmz - KS.f[dir27::zmz]) / tau /* + (no1 - n1o2/tau)*Szmz*locfeq_zmz */;
		KS.f[dir27::zmp] += (locfeq_zmp - KS.f[dir27::zmp]) / tau /* + (no1 - n1o2/tau)*Szmp*locfeq_zmp */;
		KS.f[dir27::zzm] += (locfeq_zzm - KS.f[dir27::zzm]) / tau /* + (no1 - n1o2/tau)*Szzm*locfeq_zzm */;
		KS.f[dir27::zzz] += (locfeq_zzz - KS.f[dir27::zzz]) / tau /* + (no1 - n1o2/tau)*Szzz*locfeq_zzz */;
		KS.f[dir27::zzp] += (locfeq_zzp - KS.f[dir27::zzp]) / tau /* + (no1 - n1o2/tau)*Szzp*locfeq_zzp */;
		KS.f[dir27::zpm] += (locfeq_zpm - KS.f[dir27::zpm]) / tau /* + (no1 - n1o2/tau)*Szpm*locfeq_zpm */;
		KS.f[dir27::zpz] += (locfeq_zpz - KS.f[dir27::zpz]) / tau /* + (no1 - n1o2/tau)*Szpz*locfeq_zpz */;
		KS.f[dir27::zpp] += (locfeq_zpp - KS.f[dir27::zpp]) / tau /* + (no1 - n1o2/tau)*Szpp*locfeq_zpp */;
		KS.f[dir27::pmm] += (locfeq_pmm - KS.f[dir27::pmm]) / tau /* + (no1 - n1o2/tau)*Spmm*locfeq_pmm */;
		KS.f[dir27::pmz] += (locfeq_pmz - KS.f[dir27::pmz]) / tau /* + (no1 - n1o2/tau)*Spmz*locfeq_pmz */;
		KS.f[dir27::pmp] += (locfeq_pmp - KS.f[dir27::pmp]) / tau /* + (no1 - n1o2/tau)*Spmp*locfeq_pmp */;
		KS.f[dir27::pzm] += (locfeq_pzm - KS.f[dir27::pzm]) / tau /* + (no1 - n1o2/tau)*Spzm*locfeq_pzm */;
		KS.f[dir27::pzz] += (locfeq_pzz - KS.f[dir27::pzz]) / tau /* + (no1 - n1o2/tau)*Spzz*locfeq_pzz */;
		KS.f[dir27::pzp] += (locfeq_pzp - KS.f[dir27::pzp]) / tau /* + (no1 - n1o2/tau)*Spzp*locfeq_pzp */;
		KS.f[dir27::ppm] += (locfeq_ppm - KS.f[dir27::ppm]) / tau /* + (no1 - n1o2/tau)*Sppm*locfeq_ppm */;
		KS.f[dir27::ppz] += (locfeq_ppz - KS.f[dir27::ppz]) / tau /* + (no1 - n1o2/tau)*Sppz*locfeq_ppz */;
		KS.f[dir27::ppp] += (locfeq_ppp - KS.f[dir27::ppp]) / tau /* + (no1 - n1o2/tau)*Sppp*locfeq_ppp */;
	}

	template <typename LBM_KS>
	__cuda_callable__ static void setEquilibrium(LBM_KS& KS)
	{
		KS.f[dir27::mmm] += EQ::eq_mmm(KS.rho, KS.vx, KS.vy, KS.vz, KS.rho_m, KS.vx_m, KS.vy_m, KS.vz_m);
		KS.f[dir27::mmz] += EQ::eq_mmz(KS.rho, KS.vx, KS.vy, KS.vz, KS.rho_m, KS.vx_m, KS.vy_m, KS.vz_m);
		KS.f[dir27::mmp] += EQ::eq_mmp(KS.rho, KS.vx, KS.vy, KS.vz, KS.rho_m, KS.vx_m, KS.vy_m, KS.vz_m);
		KS.f[dir27::mzm] += EQ::eq_mzm(KS.rho, KS.vx, KS.vy, KS.vz, KS.rho_m, KS.vx_m, KS.vy_m, KS.vz_m);
		KS.f[dir27::mzz] += EQ::eq_mzz(KS.rho, KS.vx, KS.vy, KS.vz, KS.rho_m, KS.vx_m, KS.vy_m, KS.vz_m);
		KS.f[dir27::mzp] += EQ::eq_mzp(KS.rho, KS.vx, KS.vy, KS.vz, KS.rho_m, KS.vx_m, KS.vy_m, KS.vz_m);
		KS.f[dir27::mpm] += EQ::eq_mpm(KS.rho, KS.vx, KS.vy, KS.vz, KS.rho_m, KS.vx_m, KS.vy_m, KS.vz_m);
		KS.f[dir27::mpz] += EQ::eq_mpz(KS.rho, KS.vx, KS.vy, KS.vz, KS.rho_m, KS.vx_m, KS.vy_m, KS.vz_m);
		KS.f[dir27::mpp] += EQ::eq_mpp(KS.rho, KS.vx, KS.vy, KS.vz, KS.rho_m, KS.vx_m, KS.vy_m, KS.vz_m);
		KS.f[dir27::zmm] += EQ::eq_zmm(KS.rho, KS.vx, KS.vy, KS.vz, KS.rho_m, KS.vx_m, KS.vy_m, KS.vz_m);
		KS.f[dir27::zmz] += EQ::eq_zmz(KS.rho, KS.vx, KS.vy, KS.vz, KS.rho_m, KS.vx_m, KS.vy_m, KS.vz_m);
		KS.f[dir27::zmp] += EQ::eq_zmp(KS.rho, KS.vx, KS.vy, KS.vz, KS.rho_m, KS.vx_m, KS.vy_m, KS.vz_m);
		KS.f[dir27::zzm] += EQ::eq_zzm(KS.rho, KS.vx, KS.vy, KS.vz, KS.rho_m, KS.vx_m, KS.vy_m, KS.vz_m);
		KS.f[dir27::zzz] += EQ::eq_zzz(KS.rho, KS.vx, KS.vy, KS.vz, KS.rho_m, KS.vx_m, KS.vy_m, KS.vz_m);
		KS.f[dir27::zzp] += EQ::eq_zzp(KS.rho, KS.vx, KS.vy, KS.vz, KS.rho_m, KS.vx_m, KS.vy_m, KS.vz_m);
		KS.f[dir27::zpm] += EQ::eq_zpm(KS.rho, KS.vx, KS.vy, KS.vz, KS.rho_m, KS.vx_m, KS.vy_m, KS.vz_m);
		KS.f[dir27::zpz] += EQ::eq_zpz(KS.rho, KS.vx, KS.vy, KS.vz, KS.rho_m, KS.vx_m, KS.vy_m, KS.vz_m);
		KS.f[dir27::zpp] += EQ::eq_zpp(KS.rho, KS.vx, KS.vy, KS.vz, KS.rho_m, KS.vx_m, KS.vy_m, KS.vz_m);
		KS.f[dir27::pmm] += EQ::eq_pmm(KS.rho, KS.vx, KS.vy, KS.vz, KS.rho_m, KS.vx_m, KS.vy_m, KS.vz_m);
		KS.f[dir27::pmz] += EQ::eq_pmz(KS.rho, KS.vx, KS.vy, KS.vz, KS.rho_m, KS.vx_m, KS.vy_m, KS.vz_m);
		KS.f[dir27::pmp] += EQ::eq_pmp(KS.rho, KS.vx, KS.vy, KS.vz, KS.rho_m, KS.vx_m, KS.vy_m, KS.vz_m);
		KS.f[dir27::pzm] += EQ::eq_pzm(KS.rho, KS.vx, KS.vy, KS.vz, KS.rho_m, KS.vx_m, KS.vy_m, KS.vz_m);
		KS.f[dir27::pzz] += EQ::eq_pzz(KS.rho, KS.vx, KS.vy, KS.vz, KS.rho_m, KS.vx_m, KS.vy_m, KS.vz_m);
		KS.f[dir27::pzp] += EQ::eq_pzp(KS.rho, KS.vx, KS.vy, KS.vz, KS.rho_m, KS.vx_m, KS.vy_m, KS.vz_m);
		KS.f[dir27::ppm] += EQ::eq_ppm(KS.rho, KS.vx, KS.vy, KS.vz, KS.rho_m, KS.vx_m, KS.vy_m, KS.vz_m);
		KS.f[dir27::ppz] += EQ::eq_ppz(KS.rho, KS.vx, KS.vy, KS.vz, KS.rho_m, KS.vx_m, KS.vy_m, KS.vz_m);
		KS.f[dir27::ppp] += EQ::eq_ppp(KS.rho, KS.vx, KS.vy, KS.vz, KS.rho_m, KS.vx_m, KS.vy_m, KS.vz_m);
	}

	// used in the "interpolated outflow boundary condition with decomposition" by Eichler https://doi.org/10.1016/j.camwa.2024.08.009
	template <typename LBM_KS>
	__cuda_callable__ static void setEquilibriumDecomposition(LBM_KS& KS, dreal rho_out)
	{
		// FIXME: update this for adjoint
#if 0
		KS.f[dir27::mmm] += EQ::eq_mmm(rho_out,KS.vx,KS.vy,KS.vz) - EQ::eq_mmm(KS.rho,KS.vx,KS.vy,KS.vz);
		KS.f[dir27::mmz] += EQ::eq_mmz(rho_out,KS.vx,KS.vy,KS.vz) - EQ::eq_mmz(KS.rho,KS.vx,KS.vy,KS.vz);
		KS.f[dir27::mmp] += EQ::eq_mmp(rho_out,KS.vx,KS.vy,KS.vz) - EQ::eq_mmp(KS.rho,KS.vx,KS.vy,KS.vz);
		KS.f[dir27::mzm] += EQ::eq_mzm(rho_out,KS.vx,KS.vy,KS.vz) - EQ::eq_mzm(KS.rho,KS.vx,KS.vy,KS.vz);
		KS.f[dir27::mzz] += EQ::eq_mzz(rho_out,KS.vx,KS.vy,KS.vz) - EQ::eq_mzz(KS.rho,KS.vx,KS.vy,KS.vz);
		KS.f[dir27::mzp] += EQ::eq_mzp(rho_out,KS.vx,KS.vy,KS.vz) - EQ::eq_mzp(KS.rho,KS.vx,KS.vy,KS.vz);
		KS.f[dir27::mpm] += EQ::eq_mpm(rho_out,KS.vx,KS.vy,KS.vz) - EQ::eq_mpm(KS.rho,KS.vx,KS.vy,KS.vz);
		KS.f[dir27::mpz] += EQ::eq_mpz(rho_out,KS.vx,KS.vy,KS.vz) - EQ::eq_mpz(KS.rho,KS.vx,KS.vy,KS.vz);
		KS.f[dir27::mpp] += EQ::eq_mpp(rho_out,KS.vx,KS.vy,KS.vz) - EQ::eq_mpp(KS.rho,KS.vx,KS.vy,KS.vz);
		KS.f[dir27::zmm] += EQ::eq_zmm(rho_out,KS.vx,KS.vy,KS.vz) - EQ::eq_zmm(KS.rho,KS.vx,KS.vy,KS.vz);
		KS.f[dir27::zmz] += EQ::eq_zmz(rho_out,KS.vx,KS.vy,KS.vz) - EQ::eq_zmz(KS.rho,KS.vx,KS.vy,KS.vz);
		KS.f[dir27::zmp] += EQ::eq_zmp(rho_out,KS.vx,KS.vy,KS.vz) - EQ::eq_zmp(KS.rho,KS.vx,KS.vy,KS.vz);
		KS.f[dir27::zzm] += EQ::eq_zzm(rho_out,KS.vx,KS.vy,KS.vz) - EQ::eq_zzm(KS.rho,KS.vx,KS.vy,KS.vz);
		KS.f[dir27::zzz] += EQ::eq_zzz(rho_out,KS.vx,KS.vy,KS.vz) - EQ::eq_zzz(KS.rho,KS.vx,KS.vy,KS.vz);
		KS.f[dir27::zzp] += EQ::eq_zzp(rho_out,KS.vx,KS.vy,KS.vz) - EQ::eq_zzp(KS.rho,KS.vx,KS.vy,KS.vz);
		KS.f[dir27::zpm] += EQ::eq_zpm(rho_out,KS.vx,KS.vy,KS.vz) - EQ::eq_zpm(KS.rho,KS.vx,KS.vy,KS.vz);
		KS.f[dir27::zpz] += EQ::eq_zpz(rho_out,KS.vx,KS.vy,KS.vz) - EQ::eq_zpz(KS.rho,KS.vx,KS.vy,KS.vz);
		KS.f[dir27::zpp] += EQ::eq_zpp(rho_out,KS.vx,KS.vy,KS.vz) - EQ::eq_zpp(KS.rho,KS.vx,KS.vy,KS.vz);
		KS.f[dir27::pmm] += EQ::eq_pmm(rho_out,KS.vx,KS.vy,KS.vz) - EQ::eq_pmm(KS.rho,KS.vx,KS.vy,KS.vz);
		KS.f[dir27::pmz] += EQ::eq_pmz(rho_out,KS.vx,KS.vy,KS.vz) - EQ::eq_pmz(KS.rho,KS.vx,KS.vy,KS.vz);
		KS.f[dir27::pmp] += EQ::eq_pmp(rho_out,KS.vx,KS.vy,KS.vz) - EQ::eq_pmp(KS.rho,KS.vx,KS.vy,KS.vz);
		KS.f[dir27::pzm] += EQ::eq_pzm(rho_out,KS.vx,KS.vy,KS.vz) - EQ::eq_pzm(KS.rho,KS.vx,KS.vy,KS.vz);
		KS.f[dir27::pzz] += EQ::eq_pzz(rho_out,KS.vx,KS.vy,KS.vz) - EQ::eq_pzz(KS.rho,KS.vx,KS.vy,KS.vz);
		KS.f[dir27::pzp] += EQ::eq_pzp(rho_out,KS.vx,KS.vy,KS.vz) - EQ::eq_pzp(KS.rho,KS.vx,KS.vy,KS.vz);
		KS.f[dir27::ppm] += EQ::eq_ppm(rho_out,KS.vx,KS.vy,KS.vz) - EQ::eq_ppm(KS.rho,KS.vx,KS.vy,KS.vz);
		KS.f[dir27::ppz] += EQ::eq_ppz(rho_out,KS.vx,KS.vy,KS.vz) - EQ::eq_ppz(KS.rho,KS.vx,KS.vy,KS.vz);
		KS.f[dir27::ppp] += EQ::eq_ppp(rho_out,KS.vx,KS.vy,KS.vz) - EQ::eq_ppp(KS.rho,KS.vx,KS.vy,KS.vz);
#endif
	}
};
