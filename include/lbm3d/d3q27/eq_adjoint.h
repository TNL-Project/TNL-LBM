#pragma once

#include <TNL/Backend/Macros.h>

#include "../../lbm_common/ciselnik.h"

// second order Maxwell-Boltzmann Equilibrium
template <typename TRAITS>
struct D3Q27_EQ_ADJOINT
{
	using dreal = typename TRAITS::dreal;

	__cuda_callable__ static dreal feq_rho(int kx, int ky, int kz, dreal vx, dreal vy, dreal vz)
	{
		return no1 - n3o2 * (vx * vx + vy * vy + vz * vz) + no3 * (kx * vx + ky * vy + kz * vz)
			 + n9o2 * (kx * vx + ky * vy + kz * vz) * (kx * vx + ky * vy + kz * vz);
	}
	__cuda_callable__ static dreal feq_v(int kx, int ky, int kz, int qx, int qy, int qz, dreal vx, dreal vy, dreal vz)
	{
		return -n3o2 * no2 * (vx * (qx - vx) + vy * (qy - vy) + vz * (qz - vz)) + no3 * (kx * (qx - vx) + ky * (qy - vy) + kz * (qz - vz))
			 + n9o2 * no2 * (kx * vx + ky * vy + kz * vz) * (kx * (qx - vx) + ky * (qy - vy) + kz * (qz - vz));
	}

	__cuda_callable__ static dreal feq(int kx, int ky, int kz, int qx, int qy, int qz, dreal vx, dreal vy, dreal vz)
	{
		return feq_rho(kx, ky, kz, vx, vy, vz) + feq_v(kx, ky, kz, qx, qy, qz, vx, vy, vz);
	}

	__cuda_callable__ static dreal
	collisionWithData(int qx, int qy, int qz, dreal rho, dreal vx, dreal vy, dreal vz, dreal rho_m, dreal vx_m, dreal vy_m, dreal vz_m)
	{
		return (rho - rho_m) + (vx - vx_m) * (qx - vx) / rho + (vy - vy_m) * (qy - vy) / rho + (vz - vz_m) * (qz - vz) / rho;
	}

	// eq_*** in adjoint is difference between measured and primary problem macro
	__cuda_callable__ static dreal eq_zzz(dreal rho, dreal vx, dreal vy, dreal vz, dreal rho_m, dreal vx_m, dreal vy_m, dreal vz_m)
	{
		return collisionWithData(0, 0, 0, rho, vx, vy, vz, rho_m, vx_m, vy_m, vz_m);
	}

	__cuda_callable__ static dreal eq_pzz(dreal rho, dreal vx, dreal vy, dreal vz, dreal rho_m, dreal vx_m, dreal vy_m, dreal vz_m)
	{
		return collisionWithData(1, 0, 0, rho, vx, vy, vz, rho_m, vx_m, vy_m, vz_m);
	}
	__cuda_callable__ static dreal eq_mzz(dreal rho, dreal vx, dreal vy, dreal vz, dreal rho_m, dreal vx_m, dreal vy_m, dreal vz_m)
	{
		return collisionWithData(-1, 0, 0, rho, vx, vy, vz, rho_m, vx_m, vy_m, vz_m);
	}
	__cuda_callable__ static dreal eq_zpz(dreal rho, dreal vx, dreal vy, dreal vz, dreal rho_m, dreal vx_m, dreal vy_m, dreal vz_m)
	{
		return collisionWithData(0, 1, 0, rho, vx, vy, vz, rho_m, vx_m, vy_m, vz_m);
	}
	__cuda_callable__ static dreal eq_zmz(dreal rho, dreal vx, dreal vy, dreal vz, dreal rho_m, dreal vx_m, dreal vy_m, dreal vz_m)
	{
		return collisionWithData(0, -1, 0, rho, vx, vy, vz, rho_m, vx_m, vy_m, vz_m);
	}
	__cuda_callable__ static dreal eq_zzp(dreal rho, dreal vx, dreal vy, dreal vz, dreal rho_m, dreal vx_m, dreal vy_m, dreal vz_m)
	{
		return collisionWithData(0, 0, 1, rho, vx, vy, vz, rho_m, vx_m, vy_m, vz_m);
	}
	__cuda_callable__ static dreal eq_zzm(dreal rho, dreal vx, dreal vy, dreal vz, dreal rho_m, dreal vx_m, dreal vy_m, dreal vz_m)
	{
		return collisionWithData(0, 0, -1, rho, vx, vy, vz, rho_m, vx_m, vy_m, vz_m);
	}

	__cuda_callable__ static dreal eq_ppz(dreal rho, dreal vx, dreal vy, dreal vz, dreal rho_m, dreal vx_m, dreal vy_m, dreal vz_m)
	{
		return collisionWithData(1, 1, 0, rho, vx, vy, vz, rho_m, vx_m, vy_m, vz_m);
	}
	__cuda_callable__ static dreal eq_pmz(dreal rho, dreal vx, dreal vy, dreal vz, dreal rho_m, dreal vx_m, dreal vy_m, dreal vz_m)
	{
		return collisionWithData(1, -1, 0, rho, vx, vy, vz, rho_m, vx_m, vy_m, vz_m);
	}
	__cuda_callable__ static dreal eq_mpz(dreal rho, dreal vx, dreal vy, dreal vz, dreal rho_m, dreal vx_m, dreal vy_m, dreal vz_m)
	{
		return collisionWithData(-1, 1, 0, rho, vx, vy, vz, rho_m, vx_m, vy_m, vz_m);
	}
	__cuda_callable__ static dreal eq_mmz(dreal rho, dreal vx, dreal vy, dreal vz, dreal rho_m, dreal vx_m, dreal vy_m, dreal vz_m)
	{
		return collisionWithData(-1, -1, 0, rho, vx, vy, vz, rho_m, vx_m, vy_m, vz_m);
	}
	__cuda_callable__ static dreal eq_pzp(dreal rho, dreal vx, dreal vy, dreal vz, dreal rho_m, dreal vx_m, dreal vy_m, dreal vz_m)
	{
		return collisionWithData(1, 0, 1, rho, vx, vy, vz, rho_m, vx_m, vy_m, vz_m);
	}
	__cuda_callable__ static dreal eq_mzm(dreal rho, dreal vx, dreal vy, dreal vz, dreal rho_m, dreal vx_m, dreal vy_m, dreal vz_m)
	{
		return collisionWithData(-1, 0, -1, rho, vx, vy, vz, rho_m, vx_m, vy_m, vz_m);
	}
	__cuda_callable__ static dreal eq_pzm(dreal rho, dreal vx, dreal vy, dreal vz, dreal rho_m, dreal vx_m, dreal vy_m, dreal vz_m)
	{
		return collisionWithData(1, 0, -1, rho, vx, vy, vz, rho_m, vx_m, vy_m, vz_m);
	}
	__cuda_callable__ static dreal eq_mzp(dreal rho, dreal vx, dreal vy, dreal vz, dreal rho_m, dreal vx_m, dreal vy_m, dreal vz_m)
	{
		return collisionWithData(-1, 0, 1, rho, vx, vy, vz, rho_m, vx_m, vy_m, vz_m);
	}
	__cuda_callable__ static dreal eq_zpp(dreal rho, dreal vx, dreal vy, dreal vz, dreal rho_m, dreal vx_m, dreal vy_m, dreal vz_m)
	{
		return collisionWithData(0, 1, 1, rho, vx, vy, vz, rho_m, vx_m, vy_m, vz_m);
	}
	__cuda_callable__ static dreal eq_zpm(dreal rho, dreal vx, dreal vy, dreal vz, dreal rho_m, dreal vx_m, dreal vy_m, dreal vz_m)
	{
		return collisionWithData(0, 1, -1, rho, vx, vy, vz, rho_m, vx_m, vy_m, vz_m);
	}
	__cuda_callable__ static dreal eq_zmp(dreal rho, dreal vx, dreal vy, dreal vz, dreal rho_m, dreal vx_m, dreal vy_m, dreal vz_m)
	{
		return collisionWithData(0, -1, 1, rho, vx, vy, vz, rho_m, vx_m, vy_m, vz_m);
	}
	__cuda_callable__ static dreal eq_zmm(dreal rho, dreal vx, dreal vy, dreal vz, dreal rho_m, dreal vx_m, dreal vy_m, dreal vz_m)
	{
		return collisionWithData(0, -1, -1, rho, vx, vy, vz, rho_m, vx_m, vy_m, vz_m);
	}

	__cuda_callable__ static dreal eq_ppp(dreal rho, dreal vx, dreal vy, dreal vz, dreal rho_m, dreal vx_m, dreal vy_m, dreal vz_m)
	{
		return collisionWithData(1, 1, 1, rho, vx, vy, vz, rho_m, vx_m, vy_m, vz_m);
	}
	__cuda_callable__ static dreal eq_mmm(dreal rho, dreal vx, dreal vy, dreal vz, dreal rho_m, dreal vx_m, dreal vy_m, dreal vz_m)
	{
		return collisionWithData(-1, -1, -1, rho, vx, vy, vz, rho_m, vx_m, vy_m, vz_m);
	}
	__cuda_callable__ static dreal eq_ppm(dreal rho, dreal vx, dreal vy, dreal vz, dreal rho_m, dreal vx_m, dreal vy_m, dreal vz_m)
	{
		return collisionWithData(1, 1, -1, rho, vx, vy, vz, rho_m, vx_m, vy_m, vz_m);
	}
	__cuda_callable__ static dreal eq_pmp(dreal rho, dreal vx, dreal vy, dreal vz, dreal rho_m, dreal vx_m, dreal vy_m, dreal vz_m)
	{
		return collisionWithData(1, -1, 1, rho, vx, vy, vz, rho_m, vx_m, vy_m, vz_m);
	}
	__cuda_callable__ static dreal eq_mpp(dreal rho, dreal vx, dreal vy, dreal vz, dreal rho_m, dreal vx_m, dreal vy_m, dreal vz_m)
	{
		return collisionWithData(-1, 1, 1, rho, vx, vy, vz, rho_m, vx_m, vy_m, vz_m);
	}
	__cuda_callable__ static dreal eq_mpm(dreal rho, dreal vx, dreal vy, dreal vz, dreal rho_m, dreal vx_m, dreal vy_m, dreal vz_m)
	{
		return collisionWithData(-1, 1, -1, rho, vx, vy, vz, rho_m, vx_m, vy_m, vz_m);
	}
	__cuda_callable__ static dreal eq_mmp(dreal rho, dreal vx, dreal vy, dreal vz, dreal rho_m, dreal vx_m, dreal vy_m, dreal vz_m)
	{
		return collisionWithData(-1, -1, 1, rho, vx, vy, vz, rho_m, vx_m, vy_m, vz_m);
	}
	__cuda_callable__ static dreal eq_pmm(dreal rho, dreal vx, dreal vy, dreal vz, dreal rho_m, dreal vx_m, dreal vy_m, dreal vz_m)
	{
		return collisionWithData(1, -1, -1, rho, vx, vy, vz, rho_m, vx_m, vy_m, vz_m);
	}
};
