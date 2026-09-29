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

	static constexpr const char* id = "CLBM";

	template <typename LBM_KS>
	__cuda_callable__ static void collision(LBM_KS& KS)
	{
		const dreal Cs2 = no1 / LBM_EQ::iCs2;

		const dreal omega = no1 / (n1o2 + LBM_EQ::iCs2 * KS.lbmViscosity);
		const dreal omega2 = omega;
		const dreal omega3 = omega;
		const dreal omega4 = omega;

		// based on LBMAT: coeff before d^3(rho)/d^3(x_1) etc.
		//const dreal omega5 = no3 * (omega2 - no2) / (omega2 - no3);
		//const dreal omega6 = omega5;
		//const dreal omega7 = omega5;

		// based on LBMAT: coeff before d^3(rho)/(d^2(x_1)d^1(x_2))
		//const dreal omega5 = omega2 * (no2 - omega2) / (-omega2*omega2 + no5*omega2 - no4);
		//const dreal omega6 = omega5;
		//const dreal omega7 = omega5;

		// based on LBMAT: coef before d(v_1)/d(x_2) d(rho)/d(x_1,x_2) etc.
		//const dreal omega5 = no2 * omega2 * (omega2 - no2) / (omega2*omega2 - no6*omega2 + no4);
		//const dreal omega6 = omega5;
		//const dreal omega7 = omega5;

		const dreal omega5 = no1;
		//const dreal omega5 = omega2;
		//const dreal omega5 = (no2 - omega2);
		const dreal omega6 = omega5;
		const dreal omega7 = omega5;

		//const dreal omega5 = (TNL::abs(KS.vy * KS.vz) < TNL::abs(KS.vx)) ? omega2 : no3 * (omega2 - no2) / (omega2 - no3);
		//const dreal omega6 = (TNL::abs(KS.vx * KS.vz) < TNL::abs(KS.vy)) ? omega2 : no3 * (omega2 - no2) / (omega2 - no3);
		//const dreal omega7 = (TNL::abs(KS.vx * KS.vy) < TNL::abs(KS.vz)) ? omega2 : no3 * (omega2 - no2) / (omega2 - no3);
		//const dreal omega5 = (TNL::abs(KS.vy * KS.vz) > TNL::abs(KS.vx) || TNL::abs(KS.vx * KS.vz) > TNL::abs(KS.vy) || TNL::abs(KS.vx * KS.vy) >
		//TNL::abs(KS.vz)) ? omega2 : no3 * (omega2 - no2) / (omega2 - no3); const dreal omega5 = !(TNL::abs(KS.vy * KS.vz) > n1o12*TNL::abs(KS.vx) ||
		//TNL::abs(KS.vx * KS.vz) > n1o12*TNL::abs(KS.vy) || TNL::abs(KS.vx * KS.vy) > n1o12*TNL::abs(KS.vz)) ? no1 : (no2 - omega2)*no216; const
		//dreal omega5 = (TNL::abs(KS.qcrit) > 1e-7) ? no1 : (no2 - omega2); const dreal omega5 = (KS.phi < 0 || KS.phi > 1 || KS.phigradmag2 > 1e-13)
		//? no1 : (no2 - omega2); const dreal omega5 = (KS.phi < 0 || KS.phi > 1 || KS.phigradmag2 > 1e-7) ? omega2 : (no2 - omega2); const dreal
		//omega5 = (KS.phigradmag2 > 1e-7 && TNL::abs(KS.qcrit) > 1e-7) ? omega2 : (KS.phigradmag2 > 1e-8 || TNL::abs(KS.qcrit) > 1e-7 ? no1 : no3 *
		//(omega2 - no2) / (omega2 - no3)); const dreal omega5 = (KS.phigradmag2 > 1e-7 && TNL::abs(KS.qcrit) > 1e-7) ? omega2 : no1; dreal omega5 =
		//no1; if (KS.x >= 50 && KS.x < 100) omega5 = no2*omega2-no2; if (KS.x > 25 && KS.x < 50) 	omega5 = (KS.x - 25) * (no2*omega2 - no3) / 25.;
		//if (KS.x < 50) omega5 = no1; if (KS.phigradmag2 > 1e-7 && TNL::abs(KS.qcrit) > 1e-7) omega5 = omega2; const dreal omega6 = omega5; const
		//dreal omega7 = omega5;

		// k := kappa_neq = kappa_eq - kappa = kappa_eq - K * f
		// (note that k_000 = 0)
		dreal k100 = KS.phi * KS.vx + KS.f[mzz] - KS.f[pzz];
		dreal k010 = KS.phi * KS.vy + KS.f[zmz] - KS.f[zpz];
		dreal k001 = KS.phi * KS.vz + KS.f[zzm] - KS.f[zzp];
		dreal k200 = KS.phi * (Cs2 - KS.vx * KS.vx) + no2 * KS.vx * (KS.f[pzz] - KS.f[mzz]) - KS.f[mzz] - KS.f[pzz];
		dreal k020 = KS.phi * (Cs2 - KS.vy * KS.vy) + no2 * KS.vy * (KS.f[zpz] - KS.f[zmz]) - KS.f[zmz] - KS.f[zpz];
		dreal k002 = KS.phi * (Cs2 - KS.vz * KS.vz) + no2 * KS.vz * (KS.f[zzp] - KS.f[zzm]) - KS.f[zzm] - KS.f[zzp];
		// collision: k := kappa_star = S * kappa_neq
		k100 *= omega2;
		k010 *= omega3;
		k001 *= omega4;
		k200 *= omega5;
		k020 *= omega6;
		k002 *= omega7;
		// kappa -> DF backtransform: f_new = f_old + K^{-1} * kappa_star
		KS.f[zzz] += -no2 * (k100 * KS.vx + k010 * KS.vy + k001 * KS.vz) - k200 - k020 - k002;
		KS.f[pzz] += k100 * KS.vx + n1o2 * (k200 + k100);
		KS.f[zpz] += k010 * KS.vy + n1o2 * (k020 + k010);
		KS.f[zzp] += k001 * KS.vz + n1o2 * (k002 + k001);
		KS.f[mzz] += k100 * KS.vx + n1o2 * (k200 - k100);
		KS.f[zmz] += k010 * KS.vy + n1o2 * (k020 - k010);
		KS.f[zzm] += k001 * KS.vz + n1o2 * (k002 - k001);
	}
};
