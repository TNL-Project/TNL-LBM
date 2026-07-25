#pragma once

#include "lbm3d/defs.h"
#include "lbm_common/ciselnik.h"

template <typename T_TRAITS, typename T_EQ>
struct D2Q9_COMMON
{
	using TRAITS = T_TRAITS;
	using EQ = T_EQ;

	using idx = typename TRAITS::idx;
	using real = typename TRAITS::real;
	using dreal = typename TRAITS::dreal;

	template <typename LBM_KS>
	__cuda_callable__ static void computeDensityAndVelocity(LBM_KS& KS)
	{
#ifdef USE_HIGH_PRECISION_RHO
		// src: https://en.wikipedia.org/wiki/Kahan_summation_algorithm
		KS.rho = 0;
		dreal c = 0;  // A running compensation for lost low-order bits.
		for (int i = 0; i < LBM_KS::Q; i++) {
			dreal y = KS.f[i] - c;
			dreal t = KS.rho + y;
			c = (t - KS.rho) - y;
			KS.rho = t;
		}
#else
		KS.rho = KS.f[dir9::zz]
			   + (((KS.f[dir9::pz] + KS.f[dir9::mz]) + (KS.f[dir9::zm] + KS.f[dir9::zp]))
				  + ((KS.f[dir9::pp] + KS.f[dir9::mm]) + (KS.f[dir9::mp] + KS.f[dir9::pm])));
#endif

		KS.vx = ((KS.f[dir9::pz] - KS.f[dir9::mz]) + ((KS.f[dir9::pm] - KS.f[dir9::mp]) + (KS.f[dir9::pp] - KS.f[dir9::mm])) + n1o2 * KS.fx) / KS.rho;
		KS.vy = ((KS.f[dir9::zp] - KS.f[dir9::zm]) + ((KS.f[dir9::mp] - KS.f[dir9::pm]) + (KS.f[dir9::pp] - KS.f[dir9::mm])) + n1o2 * KS.fy) / KS.rho;
	}

	template <typename LBM_KS>
	__cuda_callable__ static void computeDensityAndVelocity_Wall(LBM_KS& KS)
	{
		KS.rho = 1;
		KS.vx = 0;
		KS.vy = 0;
	}

	template <typename LBM_KS>
	__cuda_callable__ static void setEquilibrium(LBM_KS& KS)
	{
		KS.f[dir9::mm] = EQ::eq_mm(KS.rho, KS.vx, KS.vy);
		KS.f[dir9::mz] = EQ::eq_mz(KS.rho, KS.vx, KS.vy);
		KS.f[dir9::mp] = EQ::eq_mp(KS.rho, KS.vx, KS.vy);
		KS.f[dir9::zm] = EQ::eq_zm(KS.rho, KS.vx, KS.vy);
		KS.f[dir9::zz] = EQ::eq_zz(KS.rho, KS.vx, KS.vy);
		KS.f[dir9::zp] = EQ::eq_zp(KS.rho, KS.vx, KS.vy);
		KS.f[dir9::pm] = EQ::eq_pm(KS.rho, KS.vx, KS.vy);
		KS.f[dir9::pz] = EQ::eq_pz(KS.rho, KS.vx, KS.vy);
		KS.f[dir9::pp] = EQ::eq_pp(KS.rho, KS.vx, KS.vy);
	}

	// used in the "interpolated outflow boundary condition with decomposition" by Eichler https://doi.org/10.1016/j.camwa.2024.08.009
	template <typename LBM_KS>
	__cuda_callable__ static void setEquilibriumDecomposition(LBM_KS& KS, dreal rho_out)
	{
		KS.f[dir9::mm] += EQ::eq_mm(rho_out, KS.vx, KS.vy) - EQ::eq_mm(KS.rho, KS.vx, KS.vy);
		KS.f[dir9::mz] += EQ::eq_mz(rho_out, KS.vx, KS.vy) - EQ::eq_mz(KS.rho, KS.vx, KS.vy);
		KS.f[dir9::mp] += EQ::eq_mp(rho_out, KS.vx, KS.vy) - EQ::eq_mp(KS.rho, KS.vx, KS.vy);
		KS.f[dir9::zm] += EQ::eq_zm(rho_out, KS.vx, KS.vy) - EQ::eq_zm(KS.rho, KS.vx, KS.vy);
		KS.f[dir9::zz] += EQ::eq_zz(rho_out, KS.vx, KS.vy) - EQ::eq_zz(KS.rho, KS.vx, KS.vy);
		KS.f[dir9::zp] += EQ::eq_zp(rho_out, KS.vx, KS.vy) - EQ::eq_zp(KS.rho, KS.vx, KS.vy);
		KS.f[dir9::pm] += EQ::eq_pm(rho_out, KS.vx, KS.vy) - EQ::eq_pm(KS.rho, KS.vx, KS.vy);
		KS.f[dir9::pz] += EQ::eq_pz(rho_out, KS.vx, KS.vy) - EQ::eq_pz(KS.rho, KS.vx, KS.vy);
		KS.f[dir9::pp] += EQ::eq_pp(rho_out, KS.vx, KS.vy) - EQ::eq_pp(KS.rho, KS.vx, KS.vy);
	}

	template <typename LAT_DFS>
	__cuda_callable__ static void setEquilibriumLat(LAT_DFS& f, idx x, idx y, idx z, real rho, real vx, real vy, real vz_unused)
	{
		f(dir9::mm, x, y, z) = EQ::eq_mm(rho, vx, vy);
		f(dir9::zm, x, y, z) = EQ::eq_zm(rho, vx, vy);
		f(dir9::pm, x, y, z) = EQ::eq_pm(rho, vx, vy);
		f(dir9::mz, x, y, z) = EQ::eq_mz(rho, vx, vy);
		f(dir9::zz, x, y, z) = EQ::eq_zz(rho, vx, vy);
		f(dir9::pz, x, y, z) = EQ::eq_pz(rho, vx, vy);
		f(dir9::mp, x, y, z) = EQ::eq_mp(rho, vx, vy);
		f(dir9::zp, x, y, z) = EQ::eq_zp(rho, vx, vy);
		f(dir9::pp, x, y, z) = EQ::eq_pp(rho, vx, vy);
	}
};
