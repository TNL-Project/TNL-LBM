#pragma once

#include <TNL/Backend/Macros.h>

#include "../../lbm_common/ciselnik.h"

// Equilibrium DF based on Equilibrium Cumulants

//#define feq(qx, qy, qz, vx, vy, vz) no1 - n3o2 * ((vx)*(vx) + (vy)*(vy) + (vz)*(vz)) + no3*((qx)*(vx) + (qy)*(vy) + (qz)*(vz)) + n9o2*((qx)*(vx) +
//(qy)*(vy) + (qz)*(vz))*((qx)*(vx) + (qy)*(vy) + (qz)*(vz))

// second order Maxwell-Boltzmann Equilibrium
template <typename TRAITS>
struct D3Q27_EQ_INV_CUM
{
	using dreal = typename TRAITS::dreal;

	__cuda_callable__ static dreal feq_all(dreal rho, int qx, int qy, int qz, dreal vx, dreal vy, dreal vz){
		const dreal w0 = (dreal)(2./3);
    	const dreal w1 = (dreal)(1./6);
		const dreal ws[2] = {w0,w1};
		const dreal eq = no1 - n3o2 * (vx * vx + vy * vy + vz * vz) + no3 * (qx * vx + qy * vy + qz * vz)
		+ n9o2 * (qx * vx + qy * vy + qz * vz) * (qx * vx + qy * vy + qz * vz);
		return ws[abs(qx)]*ws[abs(qy)]*ws[abs(qz)]*rho*eq;
	}

	__cuda_callable__ static dreal feq(int qx, int qy, int qz, dreal vx, dreal vy, dreal vz)
	{
		return no1 - n3o2 * (vx * vx + vy * vy + vz * vz) + no3 * (qx * vx + qy * vy + qz * vz)
			 + n9o2 * (qx * vx + qy * vy + qz * vz) * (qx * vx + qy * vy + qz * vz);
	}

	__cuda_callable__ static dreal eq_zzz(dreal rho, dreal vx, dreal vy, dreal vz)
	{
		return -n1o27 * rho * ((no3 * vx * vx - no2) * (no3 * vy * vy - no2) * (no3 * vz * vz - no2));
	}

	__cuda_callable__ static dreal eq_pzz(dreal rho, dreal vx, dreal vy, dreal vz)
	{
		return n1o54 * rho * ((no3 * vx * vx + no3 * vx + no1) * (no3 * vy * vy - no2) * (no3 * vz * vz - no2));
	}
	__cuda_callable__ static dreal eq_mzz(dreal rho, dreal vx, dreal vy, dreal vz)
	{
		return n1o54 * rho * ((no3 * vx * vx - no3 * vx + no1) * (no3 * vy * vy - no2) * (no3 * vz * vz - no2));
	}
	__cuda_callable__ static dreal eq_zpz(dreal rho, dreal vx, dreal vy, dreal vz)
	{
		return n1o54 * rho * ((no3 * vx * vx - no2) * (no3 * vy * vy + no3 * vy + no1) * (no3 * vz * vz - no2));
	}
	__cuda_callable__ static dreal eq_zmz(dreal rho, dreal vx, dreal vy, dreal vz)
	{
		return n1o54 * rho * ((no3 * vx * vx - no2) * (no3 * vy * vy - no3 * vy + no1) * (no3 * vz * vz - no2));
	}
	__cuda_callable__ static dreal eq_zzp(dreal rho, dreal vx, dreal vy, dreal vz)
	{
		return n1o54 * rho * ((no3 * vx * vx - no2) * (no3 * vy * vy - no2) * (no3 * vz * vz + no3 * vz + no1));
	}
	__cuda_callable__ static dreal eq_zzm(dreal rho, dreal vx, dreal vy, dreal vz)
	{
		return n1o54 * rho * ((no3 * vx * vx - no2) * (no3 * vy * vy - no2) * (no3 * vz * vz - no3 * vz + no1));
	}

	__cuda_callable__ static dreal eq_ppz(dreal rho, dreal vx, dreal vy, dreal vz)
	{
		return -n1o108 * rho * ((no3 * vx * vx + no3 * vx + no1) * (no3 * vy * vy + no3 * vy + no1) * (no3 * vz * vz - no2));
	}
	__cuda_callable__ static dreal eq_pmz(dreal rho, dreal vx, dreal vy, dreal vz)
	{
		return -n1o108 * rho * ((no3 * vx * vx + no3 * vx + no1) * (no3 * vy * vy - no3 * vy + no1) * (no3 * vz * vz - no2));
	}
	__cuda_callable__ static dreal eq_mpz(dreal rho, dreal vx, dreal vy, dreal vz)
	{
		return -n1o108 * rho * ((no3 * vx * vx - no3 * vx + no1) * (no3 * vy * vy + no3 * vy + no1) * (no3 * vz * vz - no2));
	}
	__cuda_callable__ static dreal eq_mmz(dreal rho, dreal vx, dreal vy, dreal vz)
	{
		return -n1o108 * rho * ((no3 * vx * vx - no3 * vx + no1) * (no3 * vy * vy - no3 * vy + no1) * (no3 * vz * vz - no2));
	}

	__cuda_callable__ static dreal eq_pzp(dreal rho, dreal vx, dreal vy, dreal vz)
	{
		return -n1o108 * rho * ((no3 * vx * vx + no3 * vx + no1) * (no3 * vy * vy - no2) * (no3 * vz * vz + no3 * vz + no1));
	}
	__cuda_callable__ static dreal eq_mzm(dreal rho, dreal vx, dreal vy, dreal vz)
	{
		return -n1o108 * rho * ((no3 * vx * vx - no3 * vx + no1) * (no3 * vy * vy - no2) * (no3 * vz * vz - no3 * vz + no1));
	}
	__cuda_callable__ static dreal eq_pzm(dreal rho, dreal vx, dreal vy, dreal vz)
	{
		return -n1o108 * rho * ((no3 * vx * vx + no3 * vx + no1) * (no3 * vy * vy - no2) * (no3 * vz * vz - no3 * vz + no1));
	}
	__cuda_callable__ static dreal eq_mzp(dreal rho, dreal vx, dreal vy, dreal vz)
	{
		return -n1o108 * rho * ((no3 * vx * vx - no3 * vx + no1) * (no3 * vy * vy - no2) * (no3 * vz * vz + no3 * vz + no1));
	}

	__cuda_callable__ static dreal eq_zpp(dreal rho, dreal vx, dreal vy, dreal vz)
	{
		return -n1o108 * rho * ((no3 * vx * vx - no2) * (no3 * vy * vy + no3 * vy + no1) * (no3 * vz * vz + no3 * vz + no1));
	}
	__cuda_callable__ static dreal eq_zpm(dreal rho, dreal vx, dreal vy, dreal vz)
	{
		return -n1o108 * rho * ((no3 * vx * vx - no2) * (no3 * vy * vy + no3 * vy + no1) * (no3 * vz * vz - no3 * vz + no1));
	}
	__cuda_callable__ static dreal eq_zmp(dreal rho, dreal vx, dreal vy, dreal vz)
	{
		return -n1o108 * rho * ((no3 * vx * vx - no2) * (no3 * vy * vy - no3 * vy + no1) * (no3 * vz * vz + no3 * vz + no1));
	}
	__cuda_callable__ static dreal eq_zmm(dreal rho, dreal vx, dreal vy, dreal vz)
	{
		return -n1o108 * rho * ((no3 * vx * vx - no2) * (no3 * vy * vy - no3 * vy + no1) * (no3 * vz * vz - no3 * vz + no1));
	}

	__cuda_callable__ static dreal eq_ppp(dreal rho, dreal vx, dreal vy, dreal vz)
	{
		return n1o216 * rho * ((no3 * vx * vx + no3 * vx + no1) * (no3 * vy * vy + no3 * vy + no1) * (no3 * vz * vz + no3 * vz + no1));
	}
	__cuda_callable__ static dreal eq_mmm(dreal rho, dreal vx, dreal vy, dreal vz)
	{
		return n1o216 * rho * ((no3 * vx * vx - no3 * vx + no1) * (no3 * vy * vy - no3 * vy + no1) * (no3 * vz * vz - no3 * vz + no1));
	}
	__cuda_callable__ static dreal eq_ppm(dreal rho, dreal vx, dreal vy, dreal vz)
	{
		return n1o216 * rho * ((no3 * vx * vx + no3 * vx + no1) * (no3 * vy * vy + no3 * vy + no1) * (no3 * vz * vz - no3 * vz + no1));
	}
	__cuda_callable__ static dreal eq_pmp(dreal rho, dreal vx, dreal vy, dreal vz)
	{
		return n1o216 * rho * ((no3 * vx * vx + no3 * vx + no1) * (no3 * vy * vy - no3 * vy + no1) * (no3 * vz * vz + no3 * vz + no1));
	}
	__cuda_callable__ static dreal eq_mpp(dreal rho, dreal vx, dreal vy, dreal vz)
	{
		return n1o216 * rho * ((no3 * vx * vx - no3 * vx + no1) * (no3 * vy * vy + no3 * vy + no1) * (no3 * vz * vz + no3 * vz + no1));
	}
	__cuda_callable__ static dreal eq_mpm(dreal rho, dreal vx, dreal vy, dreal vz)
	{
		return n1o216 * rho * ((no3 * vx * vx - no3 * vx + no1) * (no3 * vy * vy + no3 * vy + no1) * (no3 * vz * vz - no3 * vz + no1));
	}
	__cuda_callable__ static dreal eq_mmp(dreal rho, dreal vx, dreal vy, dreal vz)
	{
		return n1o216 * rho * ((no3 * vx * vx - no3 * vx + no1) * (no3 * vy * vy - no3 * vy + no1) * (no3 * vz * vz + no3 * vz + no1));
	}
	__cuda_callable__ static dreal eq_pmm(dreal rho, dreal vx, dreal vy, dreal vz)
	{
		return n1o216 * rho * ((no3 * vx * vx + no3 * vx + no1) * (no3 * vy * vy - no3 * vy + no1) * (no3 * vz * vz - no3 * vz + no1));
	}
};
