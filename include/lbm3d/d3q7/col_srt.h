#pragma once

#include "common.h"
#include "eq.h"
#include "lbm_common/ciselnik.h"

template <typename TRAITS, typename LBM_EQ = D3Q7_EQ<TRAITS>>
struct D3Q7_SRT : D3Q7_COMMON<TRAITS, LBM_EQ>
{
	using dreal = typename TRAITS::dreal;

	static constexpr const char* id = "SRT";

	template <typename LBM_KS>
	__cuda_callable__ static void collision(LBM_KS& KS)
	{
		const dreal tau = n1o2 + LBM_EQ::iCs2 * KS.lbmViscosity;
		const dreal omega = no1 / tau;
		KS.f[dir27::zzz] += omega * (LBM_EQ::eq_zzz(KS.phi, KS.vx, KS.vy, KS.vz) - KS.f[dir27::zzz]);
		KS.f[dir27::pzz] += omega * (LBM_EQ::eq_pzz(KS.phi, KS.vx, KS.vy, KS.vz) - KS.f[dir27::pzz]);
		KS.f[dir27::zpz] += omega * (LBM_EQ::eq_zpz(KS.phi, KS.vx, KS.vy, KS.vz) - KS.f[dir27::zpz]);
		KS.f[dir27::zzp] += omega * (LBM_EQ::eq_zzp(KS.phi, KS.vx, KS.vy, KS.vz) - KS.f[dir27::zzp]);
		KS.f[dir27::mzz] += omega * (LBM_EQ::eq_mzz(KS.phi, KS.vx, KS.vy, KS.vz) - KS.f[dir27::mzz]);
		KS.f[dir27::zmz] += omega * (LBM_EQ::eq_zmz(KS.phi, KS.vx, KS.vy, KS.vz) - KS.f[dir27::zmz]);
		KS.f[dir27::zzm] += omega * (LBM_EQ::eq_zzm(KS.phi, KS.vx, KS.vy, KS.vz) - KS.f[dir27::zzm]);
	}
};
