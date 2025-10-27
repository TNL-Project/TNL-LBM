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
		//KS.vx = ((KS.f[pzz]-KS.f[mzz])+KS.fx*n1o2) / KS.phi;
		//KS.vy = ((KS.f[zpz]-KS.f[zmz])+KS.fy*n1o2) / KS.phi;
		//KS.vz = ((KS.f[zzp]-KS.f[zzm])+KS.fz*n1o2) / KS.phi;
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
		KS.f[mzz] = EQ::eq_mzz(KS.phi, KS.vx, KS.vy, KS.vz);
		KS.f[zmz] = EQ::eq_zmz(KS.phi, KS.vx, KS.vy, KS.vz);
		KS.f[zzm] = EQ::eq_zzm(KS.phi, KS.vx, KS.vy, KS.vz);
		KS.f[zzz] = EQ::eq_zzz(KS.phi, KS.vx, KS.vy, KS.vz);
		KS.f[zzp] = EQ::eq_zzp(KS.phi, KS.vx, KS.vy, KS.vz);
		KS.f[zpz] = EQ::eq_zpz(KS.phi, KS.vx, KS.vy, KS.vz);
		KS.f[pzz] = EQ::eq_pzz(KS.phi, KS.vx, KS.vy, KS.vz);
	}

	template <typename LBM_KS, typename LAT_DFS>
	__cuda_callable__ static void setEquilibriumLat(LAT_DFS& f, idx x, idx y, idx z, real phi, real vx, real vy, real vz)
	{
		f(mzz, x, y, z) = EQ::eq_mzz(phi, vx, vy, vz);
		f(zmz, x, y, z) = EQ::eq_zmz(phi, vx, vy, vz);
		f(zzm, x, y, z) = EQ::eq_zzm(phi, vx, vy, vz);
		f(zzz, x, y, z) = EQ::eq_zzz(phi, vx, vy, vz);
		f(zzp, x, y, z) = EQ::eq_zzp(phi, vx, vy, vz);
		f(zpz, x, y, z) = EQ::eq_zpz(phi, vx, vy, vz);
		f(pzz, x, y, z) = EQ::eq_pzz(phi, vx, vy, vz);
	}
};
