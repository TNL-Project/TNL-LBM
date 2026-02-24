#pragma once

#include <TNL/Backend/Macros.h>

#include "../../lbm_common/ciselnik.h"

// second order Maxwell-Boltzmann Equilibrium
template <typename TRAITS>
struct D3Q53_EQ
{
	using dreal = typename TRAITS::dreal;

	//static constexpr const dreal T = (dreal)0.046649934354587527782; // closure 378*T**5/5 - 381*T**4 + 24893*T**3/40 - 1713*T**2/4 + 921*T/8 - 9/2 = 0
	//static constexpr const dreal T = (dreal)0.95848492022167790724; // closure 378*T**5/5 - 381*T**4 + 24893*T**3/40 - 1713*T**2/4 + 921*T/8 - 9/2 = 0
	//static constexpr const dreal T = (dreal)1./2.6767385370978665559; // closure 378*T**5/5 - 381*T**4 + 24893*T**3/40 - 1713*T**2/4 + 921*T/8 - 9/2 = 0; only root where every weights is positive
	//static constexpr const dreal n1oT = 1./T;
	//static constexpr dreal w_1 = T*T*T/48. - T*T/48. + T/180.;
	//static constexpr dreal w_2 = 3.*T*T*T/64 - 13.*T*T/192. + 3.*T/160;
	//static constexpr dreal w_3 = -5.*T*T*T/64 + 25.*T*T/192 - 3.*T/80;
	//static constexpr dreal w_10 = -23.*T*T*T/8 + 13.*T*T/3 - 6.*T/5;
	//static constexpr dreal w_11 = 11.*T*T*T/2 - 25.*T*T/3 + 12*T/5;
	//static constexpr dreal w_14 = -163.*T*T*T/16 + 243.*T*T/16 - 81.*T/20;
	//static constexpr dreal w_27 = 297.*T*T*T/16 - 427.*T*T/16 + 161.*T/30 + 1;
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
		const dreal eq = no1 - n1o2 * n1oT * (vx * vx + vy * vy + vz * vz) + n1oT * (qx * vx + qy * vy + qz * vz)
		+ n1o2 * n1oT * n1oT * (qx * vx + qy * vy + qz * vz) * (qx * vx + qy * vy + qz * vz);
		return ws[id]*eq;
	}


};
