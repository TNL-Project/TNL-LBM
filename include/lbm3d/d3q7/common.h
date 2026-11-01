#pragma once

#include "lbm3d/defs.h"

template <typename T_TRAITS, typename T_EQ>
struct D3Q7_COMMON
{
	using TRAITS = T_TRAITS;
	using EQ = T_EQ;

	using idx = typename TRAITS::idx;
	using real = typename TRAITS::real;
	using dreal = typename TRAITS::dreal;

	template <typename LBM_KS>
	__cuda_callable__ static void computeDensityAndVelocity(LBM_KS& KS)
	{
		dreal phi = 0;
		for (int i = 0; i < 7; i++)
			phi += KS.f[i];
		KS.phi = phi;

		// NOTE: does not make sense for ADE
		//KS.vx = ((KS.f[dir27::pzz]-KS.f[dir27::mzz])+KS.fx*n1o2) / KS.phi;
		//KS.vy = ((KS.f[dir27::zpz]-KS.f[dir27::zmz])+KS.fy*n1o2) / KS.phi;
		//KS.vz = ((KS.f[dir27::zzp]-KS.f[dir27::zzm])+KS.fz*n1o2) / KS.phi;
	}

	template <typename LBM_KS>
	__cuda_callable__ static void computeDensityAndVelocity_Wall(LBM_KS& KS)
	{
		KS.phi = 1;
		// NOTE: does not make sense for ADE
		//KS.vx = 0;
		//KS.vy = 0;
		//KS.vz = 0;
	}

	template <typename LBM_KS>
	__cuda_callable__ static void setEquilibrium(LBM_KS& KS)
	{
		KS.f[dir27::mzz] = EQ::eq_mzz(KS.phi, KS.vx, KS.vy, KS.vz);
		KS.f[dir27::zmz] = EQ::eq_zmz(KS.phi, KS.vx, KS.vy, KS.vz);
		KS.f[dir27::zzm] = EQ::eq_zzm(KS.phi, KS.vx, KS.vy, KS.vz);
		KS.f[dir27::zzz] = EQ::eq_zzz(KS.phi, KS.vx, KS.vy, KS.vz);
		KS.f[dir27::zzp] = EQ::eq_zzp(KS.phi, KS.vx, KS.vy, KS.vz);
		KS.f[dir27::zpz] = EQ::eq_zpz(KS.phi, KS.vx, KS.vy, KS.vz);
		KS.f[dir27::pzz] = EQ::eq_pzz(KS.phi, KS.vx, KS.vy, KS.vz);
	}
};
