#pragma once

#include "common.h"
#include "eq.h"

// improved BRK (SRT) model by Geier 2017
// for standard DF (no well-conditioned)

template <typename TRAITS, typename LBM_EQ = D2Q9_EQ<TRAITS>>
struct D2Q9_SRT : D2Q9_COMMON<TRAITS, LBM_EQ>
{
	using dreal = typename TRAITS::dreal;

	static constexpr const char* id = "SRT";

	template <typename LBM_KS>
	__cuda_callable__ static void collision(LBM_KS& KS)
	{
		const dreal tau = no3 * KS.lbmViscosity + n1o2;

		const dreal force_C = (no1 - n1o2 / tau) * no4 / no9 * (no3 * (-KS.vx * KS.fx - KS.vy * KS.fy));
		const dreal force_E = (no1 - n1o2 / tau) / no9 * (no3 * ((no1 - KS.vx) * KS.fx - KS.vy * KS.fy) + no9 * KS.vx * KS.fx);
		const dreal force_W = (no1 - n1o2 / tau) / no9 * (no3 * ((-no1 - KS.vx) * KS.fx - KS.vy * KS.fy) + no9 * KS.vx * KS.fx);
		const dreal force_N = (no1 - n1o2 / tau) / no9 * (no3 * (-KS.vx * KS.fx + (no1 - KS.vy) * KS.fy) + no9 * KS.vy * KS.fy);
		const dreal force_S = (no1 - n1o2 / tau) / no9 * (no3 * (-KS.vx * KS.fx + (-no1 - KS.vy) * KS.fy) + no9 * KS.vy * KS.fy);
		const dreal force_NE =
			(no1 - n1o2 / tau) / no36 * (no3 * ((no1 - KS.vx) * KS.fx + (no1 - KS.vy) * KS.fy) + no9 * (KS.vx + KS.vy) * (KS.fx + KS.fy));
		const dreal force_SW =
			(no1 - n1o2 / tau) / no36 * (no3 * ((-no1 - KS.vx) * KS.fx + (-no1 - KS.vy) * KS.fy) + no9 * (KS.vx + KS.vy) * (KS.fx + KS.fy));
		const dreal force_SE =
			(no1 - n1o2 / tau) / no36 * (no3 * ((no1 - KS.vx) * KS.fx + (-no1 - KS.vy) * KS.fy) + no9 * (KS.vx - KS.vy) * (KS.fx - KS.fy));
		const dreal force_NW =
			(no1 - n1o2 / tau) / no36 * (no3 * ((-no1 - KS.vx) * KS.fx + (no1 - KS.vy) * KS.fy) + no9 * (KS.vx - KS.vy) * (KS.fx - KS.fy));

		KS.f[zz] += (LBM_EQ::eq_zz(KS.rho, KS.vx, KS.vy) - KS.f[zz]) / tau + force_C;
		KS.f[pz] += (LBM_EQ::eq_pz(KS.rho, KS.vx, KS.vy) - KS.f[pz]) / tau + force_E;
		KS.f[mz] += (LBM_EQ::eq_mz(KS.rho, KS.vx, KS.vy) - KS.f[mz]) / tau + force_W;
		KS.f[zm] += (LBM_EQ::eq_zm(KS.rho, KS.vx, KS.vy) - KS.f[zm]) / tau + force_S;
		KS.f[zp] += (LBM_EQ::eq_zp(KS.rho, KS.vx, KS.vy) - KS.f[zp]) / tau + force_N;
		KS.f[pm] += (LBM_EQ::eq_pm(KS.rho, KS.vx, KS.vy) - KS.f[pm]) / tau + force_SE;
		KS.f[pp] += (LBM_EQ::eq_pp(KS.rho, KS.vx, KS.vy) - KS.f[pp]) / tau + force_NE;
		KS.f[mm] += (LBM_EQ::eq_mm(KS.rho, KS.vx, KS.vy) - KS.f[mm]) / tau + force_SW;
		KS.f[mp] += (LBM_EQ::eq_mp(KS.rho, KS.vx, KS.vy) - KS.f[mp]) / tau + force_NW;
	}
};
