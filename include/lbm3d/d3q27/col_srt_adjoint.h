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
		return	  KS.f[zzz]*n8o27*LBM_EQ::feq( 0, 0, 0, qx, qy, qz, vx, vy, vz)

				+ KS.f[pzz]*n2o27*LBM_EQ::feq( 1, 0, 0, qx, qy, qz, vx, vy, vz)
				+ KS.f[mzz]*n2o27*LBM_EQ::feq(-1, 0, 0, qx, qy, qz, vx, vy, vz)
				+ KS.f[zpz]*n2o27*LBM_EQ::feq( 0, 1, 0, qx, qy, qz, vx, vy, vz)
				+ KS.f[zmz]*n2o27*LBM_EQ::feq( 0,-1, 0, qx, qy, qz, vx, vy, vz)
				+ KS.f[zzp]*n2o27*LBM_EQ::feq( 0, 0, 1, qx, qy, qz, vx, vy, vz)
				+ KS.f[zzm]*n2o27*LBM_EQ::feq( 0, 0,-1, qx, qy, qz, vx, vy, vz)

				+ KS.f[ppz]*n1o54*LBM_EQ::feq( 1, 1, 0, qx, qy, qz, vx, vy, vz)
				+ KS.f[pmz]*n1o54*LBM_EQ::feq( 1,-1, 0, qx, qy, qz, vx, vy, vz)
				+ KS.f[mpz]*n1o54*LBM_EQ::feq(-1, 1, 0, qx, qy, qz, vx, vy, vz)
				+ KS.f[mmz]*n1o54*LBM_EQ::feq(-1,-1, 0, qx, qy, qz, vx, vy, vz)
				+ KS.f[pzp]*n1o54*LBM_EQ::feq( 1, 0, 1, qx, qy, qz, vx, vy, vz)
				+ KS.f[mzm]*n1o54*LBM_EQ::feq(-1, 0,-1, qx, qy, qz, vx, vy, vz)
				+ KS.f[pzm]*n1o54*LBM_EQ::feq( 1, 0,-1, qx, qy, qz, vx, vy, vz)
				+ KS.f[mzp]*n1o54*LBM_EQ::feq(-1, 0, 1, qx, qy, qz, vx, vy, vz)
				+ KS.f[zpp]*n1o54*LBM_EQ::feq( 0, 1, 1, qx, qy, qz, vx, vy, vz)
				+ KS.f[zpm]*n1o54*LBM_EQ::feq( 0, 1,-1, qx, qy, qz, vx, vy, vz)
				+ KS.f[zmp]*n1o54*LBM_EQ::feq( 0,-1, 1, qx, qy, qz, vx, vy, vz)
				+ KS.f[zmm]*n1o54*LBM_EQ::feq( 0,-1,-1, qx, qy, qz, vx, vy, vz)

				+ KS.f[ppp]*n1o216*LBM_EQ::feq( 1, 1, 1, qx, qy, qz, vx, vy, vz)
				+ KS.f[mmm]*n1o216*LBM_EQ::feq(-1,-1,-1, qx, qy, qz, vx, vy, vz)
				+ KS.f[ppm]*n1o216*LBM_EQ::feq( 1, 1,-1, qx, qy, qz, vx, vy, vz)
				+ KS.f[pmp]*n1o216*LBM_EQ::feq( 1,-1, 1, qx, qy, qz, vx, vy, vz)
				+ KS.f[mpp]*n1o216*LBM_EQ::feq(-1, 1, 1, qx, qy, qz, vx, vy, vz)
				+ KS.f[mpm]*n1o216*LBM_EQ::feq(-1, 1,-1, qx, qy, qz, vx, vy, vz)
				+ KS.f[mmp]*n1o216*LBM_EQ::feq(-1,-1, 1, qx, qy, qz, vx, vy, vz)
				+ KS.f[pmm]*n1o216*LBM_EQ::feq( 1,-1,-1, qx, qy, qz, vx, vy, vz)
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
};
