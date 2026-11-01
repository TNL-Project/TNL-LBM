#include "common_adjoint.h"
#include "eq_adjoint.h"

template <typename TRAITS, typename LBM_EQ = D3Q27_EQ_ADJOINT<TRAITS>>
struct D3Q27_SRT_ADJOINT : D3Q27_COMMON_ADJOINT<TRAITS, LBM_EQ>
{
	using dreal = typename TRAITS::dreal;

	static constexpr const char* id = "SRT_ADJ";

	template <typename LBM_KS>
	__cuda_callable__ static dreal feq_f(LBM_KS& KS, int qx, int qy, int qz, dreal vx, dreal vy, dreal vz)
	{
		// clang-format off
		return	  KS.f[dir27::zzz]*n8o27*LBM_EQ::feq( 0, 0, 0, qx, qy, qz, vx, vy, vz)

				+ KS.f[dir27::pzz]*n2o27*LBM_EQ::feq( 1, 0, 0, qx, qy, qz, vx, vy, vz)
				+ KS.f[dir27::mzz]*n2o27*LBM_EQ::feq(-1, 0, 0, qx, qy, qz, vx, vy, vz)
				+ KS.f[dir27::zpz]*n2o27*LBM_EQ::feq( 0, 1, 0, qx, qy, qz, vx, vy, vz)
				+ KS.f[dir27::zmz]*n2o27*LBM_EQ::feq( 0,-1, 0, qx, qy, qz, vx, vy, vz)
				+ KS.f[dir27::zzp]*n2o27*LBM_EQ::feq( 0, 0, 1, qx, qy, qz, vx, vy, vz)
				+ KS.f[dir27::zzm]*n2o27*LBM_EQ::feq( 0, 0,-1, qx, qy, qz, vx, vy, vz)

				+ KS.f[dir27::ppz]*n1o54*LBM_EQ::feq( 1, 1, 0, qx, qy, qz, vx, vy, vz)
				+ KS.f[dir27::pmz]*n1o54*LBM_EQ::feq( 1,-1, 0, qx, qy, qz, vx, vy, vz)
				+ KS.f[dir27::mpz]*n1o54*LBM_EQ::feq(-1, 1, 0, qx, qy, qz, vx, vy, vz)
				+ KS.f[dir27::mmz]*n1o54*LBM_EQ::feq(-1,-1, 0, qx, qy, qz, vx, vy, vz)
				+ KS.f[dir27::pzp]*n1o54*LBM_EQ::feq( 1, 0, 1, qx, qy, qz, vx, vy, vz)
				+ KS.f[dir27::mzm]*n1o54*LBM_EQ::feq(-1, 0,-1, qx, qy, qz, vx, vy, vz)
				+ KS.f[dir27::pzm]*n1o54*LBM_EQ::feq( 1, 0,-1, qx, qy, qz, vx, vy, vz)
				+ KS.f[dir27::mzp]*n1o54*LBM_EQ::feq(-1, 0, 1, qx, qy, qz, vx, vy, vz)
				+ KS.f[dir27::zpp]*n1o54*LBM_EQ::feq( 0, 1, 1, qx, qy, qz, vx, vy, vz)
				+ KS.f[dir27::zpm]*n1o54*LBM_EQ::feq( 0, 1,-1, qx, qy, qz, vx, vy, vz)
				+ KS.f[dir27::zmp]*n1o54*LBM_EQ::feq( 0,-1, 1, qx, qy, qz, vx, vy, vz)
				+ KS.f[dir27::zmm]*n1o54*LBM_EQ::feq( 0,-1,-1, qx, qy, qz, vx, vy, vz)

				+ KS.f[dir27::ppp]*n1o216*LBM_EQ::feq( 1, 1, 1, qx, qy, qz, vx, vy, vz)
				+ KS.f[dir27::mmm]*n1o216*LBM_EQ::feq(-1,-1,-1, qx, qy, qz, vx, vy, vz)
				+ KS.f[dir27::ppm]*n1o216*LBM_EQ::feq( 1, 1,-1, qx, qy, qz, vx, vy, vz)
				+ KS.f[dir27::pmp]*n1o216*LBM_EQ::feq( 1,-1, 1, qx, qy, qz, vx, vy, vz)
				+ KS.f[dir27::mpp]*n1o216*LBM_EQ::feq(-1, 1, 1, qx, qy, qz, vx, vy, vz)
				+ KS.f[dir27::mpm]*n1o216*LBM_EQ::feq(-1, 1,-1, qx, qy, qz, vx, vy, vz)
				+ KS.f[dir27::mmp]*n1o216*LBM_EQ::feq(-1,-1, 1, qx, qy, qz, vx, vy, vz)
				+ KS.f[dir27::pmm]*n1o216*LBM_EQ::feq( 1,-1,-1, qx, qy, qz, vx, vy, vz)
				;
		// clang-format on
	}

	template <typename LBM_KS>
	__cuda_callable__ static void collision(LBM_KS& KS)
	{
		const dreal tau = no3 * KS.lbmViscosity + n1o2;

		const dreal locfeq_zzz = feq_f(KS, 0, 0, 0, KS.vx, KS.vy, KS.vz);
		const dreal locfeq_pzz = feq_f(KS, 1, 0, 0, KS.vx, KS.vy, KS.vz);
		const dreal locfeq_mzz = feq_f(KS, -1, 0, 0, KS.vx, KS.vy, KS.vz);
		const dreal locfeq_zpz = feq_f(KS, 0, 1, 0, KS.vx, KS.vy, KS.vz);
		const dreal locfeq_zmz = feq_f(KS, 0, -1, 0, KS.vx, KS.vy, KS.vz);
		const dreal locfeq_zzp = feq_f(KS, 0, 0, 1, KS.vx, KS.vy, KS.vz);
		const dreal locfeq_zzm = feq_f(KS, 0, 0, -1, KS.vx, KS.vy, KS.vz);
		const dreal locfeq_ppz = feq_f(KS, 1, 1, 0, KS.vx, KS.vy, KS.vz);
		const dreal locfeq_pmz = feq_f(KS, 1, -1, 0, KS.vx, KS.vy, KS.vz);
		const dreal locfeq_mpz = feq_f(KS, -1, 1, 0, KS.vx, KS.vy, KS.vz);
		const dreal locfeq_mmz = feq_f(KS, -1, -1, 0, KS.vx, KS.vy, KS.vz);
		const dreal locfeq_pzp = feq_f(KS, 1, 0, 1, KS.vx, KS.vy, KS.vz);
		const dreal locfeq_mzm = feq_f(KS, -1, 0, -1, KS.vx, KS.vy, KS.vz);
		const dreal locfeq_pzm = feq_f(KS, 1, 0, -1, KS.vx, KS.vy, KS.vz);
		const dreal locfeq_mzp = feq_f(KS, -1, 0, 1, KS.vx, KS.vy, KS.vz);
		const dreal locfeq_zpp = feq_f(KS, 0, 1, 1, KS.vx, KS.vy, KS.vz);
		const dreal locfeq_zpm = feq_f(KS, 0, 1, -1, KS.vx, KS.vy, KS.vz);
		const dreal locfeq_zmp = feq_f(KS, 0, -1, 1, KS.vx, KS.vy, KS.vz);
		const dreal locfeq_zmm = feq_f(KS, 0, -1, -1, KS.vx, KS.vy, KS.vz);
		const dreal locfeq_ppp = feq_f(KS, 1, 1, 1, KS.vx, KS.vy, KS.vz);
		const dreal locfeq_mmm = feq_f(KS, -1, -1, -1, KS.vx, KS.vy, KS.vz);
		const dreal locfeq_ppm = feq_f(KS, 1, 1, -1, KS.vx, KS.vy, KS.vz);
		const dreal locfeq_pmp = feq_f(KS, 1, -1, 1, KS.vx, KS.vy, KS.vz);
		const dreal locfeq_mpp = feq_f(KS, -1, 1, 1, KS.vx, KS.vy, KS.vz);
		const dreal locfeq_mpm = feq_f(KS, -1, 1, -1, KS.vx, KS.vy, KS.vz);
		const dreal locfeq_mmp = feq_f(KS, -1, -1, 1, KS.vx, KS.vy, KS.vz);
		const dreal locfeq_pmm = feq_f(KS, 1, -1, -1, KS.vx, KS.vy, KS.vz);

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
};
