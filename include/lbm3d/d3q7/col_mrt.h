#pragma once

#include "common.h"
#include "eq.h"
#include "lbm_common/ciselnik.h"

template <typename TRAITS, typename LBM_EQ = D3Q7_EQ<TRAITS>>
struct D3Q7_MRT : D3Q7_COMMON<TRAITS, LBM_EQ>
{
	using dreal = typename TRAITS::dreal;

	static constexpr const char* id = "MRT";

	template <typename LBM_KS>
	__cuda_callable__ static void collision(LBM_KS& KS)
	{
		const dreal Cs2 = no1 / LBM_EQ::iCs2;

		const dreal omega = no1 / (n1o2 + LBM_EQ::iCs2 * KS.lbmViscosity);
		const dreal omega2 = omega;
		const dreal omega3 = omega;
		const dreal omega4 = omega;
		const dreal omega5 = no1;
		const dreal omega6 = no1;
		const dreal omega7 = no1;

		// m := mu_neq = mu_eq - mu = mu_eq - M * f
		// (note that m_000 = 0)
		dreal m_100 = KS.phi * KS.vx + KS.f[dir27::mzz] - KS.f[dir27::pzz];
		dreal m_010 = KS.phi * KS.vy + KS.f[dir27::zmz] - KS.f[dir27::zpz];
		dreal m_001 = KS.phi * KS.vz + KS.f[dir27::zzm] - KS.f[dir27::zzp];
		dreal m_200 = KS.phi * (KS.vx * KS.vx + Cs2) - KS.f[dir27::mzz] - KS.f[dir27::pzz];
		dreal m_020 = KS.phi * (KS.vy * KS.vy + Cs2) - KS.f[dir27::zmz] - KS.f[dir27::zpz];
		dreal m_002 = KS.phi * (KS.vz * KS.vz + Cs2) - KS.f[dir27::zzm] - KS.f[dir27::zzp];
		// collision: m := mu_star = S * mu_neq
		m_100 *= omega2;
		m_010 *= omega3;
		m_001 *= omega4;
		m_200 *= omega5;
		m_020 *= omega6;
		m_002 *= omega7;
		// mu -> DF backtransform: f_new = f_old + M^{-1} * mu_star
		KS.f[dir27::zzz] += -m_200 - m_020 - m_002;
		KS.f[dir27::pzz] += n1o2 * (m_200 + m_100);
		KS.f[dir27::zpz] += n1o2 * (m_020 + m_010);
		KS.f[dir27::zzp] += n1o2 * (m_002 + m_001);
		KS.f[dir27::mzz] += n1o2 * (m_200 - m_100);
		KS.f[dir27::zmz] += n1o2 * (m_020 - m_010);
		KS.f[dir27::zzm] += n1o2 * (m_002 - m_001);
	}
};
