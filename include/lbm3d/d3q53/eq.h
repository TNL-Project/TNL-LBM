#pragma once

#include <TNL/Backend/Macros.h>

#include "../../lbm_common/ciselnik.h"

// third order Maxwell-Boltzmann Equilibrium
template <typename TRAITS>
struct D3Q53_EQ
{
	using dreal = typename TRAITS::dreal;
	// Contains precomputed values
	static constexpr dreal T = 0.373171943147983;
	static constexpr dreal n1oT = 1./T;
	static constexpr dreal w_1 = 0.000254627832132497;
	static constexpr dreal w_2 = 4.0435346221523e-6;
	static constexpr dreal w_3 = 7.8597574580569e-5;
	static constexpr dreal w_10 = 0.00623707839948295;
	static constexpr dreal w_11 = 0.0209532136880464;
	static constexpr dreal w_14 = 0.0742108949874378;
	static constexpr dreal w_27 = 0.250896152458213;


	__cuda_callable__ static dreal feq(int qx, int qy, int qz, dreal vx, dreal vy, dreal vz, int id){
		const dreal ws[53] = {w_1,w_2,w_3,w_2,w_3,w_3,w_2,w_3,w_2,w_10,w_11,w_10,w_11,w_14,w_11,w_10,w_11,w_10,w_1,w_3,w_3,w_11,w_14,w_11,w_1,w_14,w_27,w_14,w_1,w_11,w_14,w_11,w_3,w_3,w_1,w_10,w_11,w_10,w_11,w_14,w_11,w_10,w_11,w_10,w_2,w_3,w_2,w_3,w_3,w_2,w_3,w_2,w_1};
		const dreal xiu = (qx * vx + qy * vy + qz * vz);
		const dreal uu = (vx * vx + vy * vy + vz * vz);
		const dreal eq = no1 							// 1
		- n1o2 * n1oT * uu + 							// - u2/2cs2
		n1oT * xiu 										// + cu/cs2
		+ n1o2 * n1oT * n1oT * xiu * xiu				// - xiu2/2cs4
		+(xiu*xiu*xiu - 3.*xiu*uu*T)/6.*n1oT*n1oT*n1oT; // third term

 		return ws[id]*eq;
	}
};
