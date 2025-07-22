#pragma once

#include <TNL/Backend/Macros.h>

#include "../../lbm_common/ciselnik.h"

// second order Maxwell-Boltzmann Equilibrium
template <typename TRAITS>
struct D2Q9_EQ
{
	using dreal = typename TRAITS::dreal;

	__cuda_callable__ static dreal feq(int qx, int qy, dreal vx, dreal vy)
	{
		return no1 - n3o2 * (vx * vx + vy * vy) + no3 * (qx * vx + qy * vy) + n9o2 * (qx * vx + qy * vy) * (qx * vx + qy * vy);
	}

	__cuda_callable__ static dreal eq_zz(dreal rho, dreal vx, dreal vy)
	{
		return n4o9 * rho * feq(0, 0, vx, vy);
	}

	__cuda_callable__ static dreal eq_pz(dreal rho, dreal vx, dreal vy)
	{
		return n1o9 * rho * feq(1, 0, vx, vy);
	}

	__cuda_callable__ static dreal eq_mz(dreal rho, dreal vx, dreal vy)
	{
		return n1o9 * rho * feq(-1, 0, vx, vy);
	}

	__cuda_callable__ static dreal eq_zp(dreal rho, dreal vx, dreal vy)
	{
		return n1o9 * rho * feq(0, 1, vx, vy);
	}

	__cuda_callable__ static dreal eq_zm(dreal rho, dreal vx, dreal vy)
	{
		return n1o9 * rho * feq(0, -1, vx, vy);
	}

	__cuda_callable__ static dreal eq_pp(dreal rho, dreal vx, dreal vy)
	{
		return n1o36 * rho * feq(1, 1, vx, vy);
	}

	__cuda_callable__ static dreal eq_pm(dreal rho, dreal vx, dreal vy)
	{
		return n1o36 * rho * feq(1, -1, vx, vy);
	}

	__cuda_callable__ static dreal eq_mp(dreal rho, dreal vx, dreal vy)
	{
		return n1o36 * rho * feq(-1, 1, vx, vy);
	}

	__cuda_callable__ static dreal eq_mm(dreal rho, dreal vx, dreal vy)
	{
		return n1o36 * rho * feq(-1, -1, vx, vy);
	}
};
