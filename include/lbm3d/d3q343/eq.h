#pragma once

#include <TNL/Backend/Macros.h>

#include "../../lbm_common/ciselnik.h"

// second order Maxwell-Boltzmann Equilibrium
template <typename TRAITS>
struct D3Q343_EQ
{
	using dreal = typename TRAITS::dreal;

	static constexpr const dreal T0 = (dreal)0.6979533220196830882384091; // from closure 105T^3 − 210T^2 + 147T − 36 = 0 in D2Q49
    static constexpr const dreal w0 = (dreal)(1.0*(36-49*T0+42*T0*T0-15*T0*T0*T0)/36);
    static constexpr const dreal w1 = (dreal)(1.0*T0*(12-13*T0+5*T0*T0)/16);
    static constexpr const dreal w2 = (dreal)(1.0*T0*(-3+10*T0-5*T0*T0)/40);
    static constexpr const dreal w3 = (dreal)(1.0*T0*(4-15*T0+15*T0*T0)/720);
	static constexpr dreal ws[4] = {
		w0,w1,w2,w3
	};
	__cuda_callable__ static dreal feq(dreal rho, int qx, int qy, int qz, dreal vx, dreal vy, dreal vz)
	{
		const dreal eq = no1 - n3o2 * (vx * vx + vy * vy + vz * vz) + no3 * (qx * vx + qy * vy + qz * vz)
			 + n9o2 * (qx * vx + qy * vy + qz * vz) * (qx * vx + qy * vy + qz * vz);
		return ws[abs(qx)]*ws[abs(qy)]*ws[abs(qz)]*rho*eq;
	}
};
