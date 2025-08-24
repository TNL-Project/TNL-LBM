#pragma once

#include "common.h"
#include "eq.h"
#include "lbm_common/ciselnik.h"

// CLBM non-og standard monomials
// central moment equilibria
template <typename TRAITS, typename LBM_EQ = D3Q7_EQ<TRAITS>>
struct D3Q7_CLBM : D3Q7_COMMON<TRAITS, LBM_EQ>
{
	using dreal = typename TRAITS::dreal;

	static constexpr const char* id = "CLBM-RS";

	template <typename LBM_KS>
	__cuda_callable__ static void collision(LBM_KS& KS)
	{
		const dreal omega1 = no1 / (n1o2 + LBM_EQ::iCs2 * KS.lbmViscosity);
		const dreal Qp = 0;	 // source term!
		const dreal gc2e = KS.phi / LBM_EQ::iCs2;
		const dreal omegat5 = no1;	//no3*(omega1 - no2)/(omega1 - no3); //T_xxx ~ 0
		const dreal omegats1 = no1;
		const dreal omegats5 = no1;
		//DF->CM transform
		const dreal gc100 = -KS.phi * KS.vx + KS.f[pzz] - KS.f[mzz];
		const dreal gc010 = -KS.phi * KS.vy + KS.f[zpz] - KS.f[zmz];
		const dreal gc001 = -KS.phi * KS.vz + KS.f[zzp] - KS.f[zzm];
		const dreal gc200 = KS.phi * KS.vx * KS.vx + no2 * (KS.f[mzz] - KS.f[pzz]) * KS.vx + KS.f[pzz] + KS.f[mzz];
		const dreal gc020 = KS.phi * KS.vy * KS.vy + no2 * (KS.f[zmz] - KS.f[zpz]) * KS.vy + KS.f[zpz] + KS.f[zmz];
		const dreal gc002 = KS.phi * KS.vz * KS.vz + no2 * (KS.f[zzm] - KS.f[zzp]) * KS.vz + KS.f[zzp] + KS.f[zzm];
		//collision
		const dreal gc000s = KS.phi + (no1 - n1o2 * omegats1) * Qp;
		const dreal gc100s = (no1 - omega1) * gc100;
		const dreal gc010s = (no1 - omega1) * gc010;
		const dreal gc001s = (no1 - omega1) * gc001;
		const dreal gc200s = gc200 + omegat5 * (gc2e - gc200) + (no1 - n1o2 * omegats5) * Qp / LBM_EQ::iCs2;
		const dreal gc020s = gc020 + omegat5 * (gc2e - gc020) + (no1 - n1o2 * omegats5) * Qp / LBM_EQ::iCs2;
		const dreal gc002s = gc002 + omegat5 * (gc2e - gc002) + (no1 - n1o2 * omegats5) * Qp / LBM_EQ::iCs2;
		//CM->DF backtransform
		KS.f[zzz] = KS.phi * (no1 - KS.vz * KS.vz - KS.vy * KS.vy - KS.vx * KS.vx) - no2 * (gc001s * KS.vz + gc010s * KS.vy + gc100s * KS.vx) - gc200s
				  - gc020s - gc002s;
		KS.f[pzz] = n1o2 * gc000s * (KS.vx * KS.vx + KS.vx) + gc100s * KS.vx + n1o2 * (gc200s + gc100s);
		KS.f[zpz] = n1o2 * gc000s * (KS.vy * KS.vy + KS.vy) + gc010s * KS.vy + n1o2 * (gc020s + gc010s);
		KS.f[zzp] = n1o2 * gc000s * (KS.vz * KS.vz + KS.vz) + gc001s * KS.vz + n1o2 * (gc002s + gc001s);
		KS.f[mzz] = n1o2 * gc000s * (KS.vx * KS.vx - KS.vx) + gc100s * KS.vx + n1o2 * (gc200s - gc100s);
		KS.f[zmz] = n1o2 * gc000s * (KS.vy * KS.vy - KS.vy) + gc010s * KS.vy + n1o2 * (gc020s - gc010s);
		KS.f[zzm] = n1o2 * gc000s * (KS.vz * KS.vz - KS.vz) + gc001s * KS.vz + n1o2 * (gc002s - gc001s);
	}
};
