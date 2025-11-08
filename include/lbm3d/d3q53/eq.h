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
	static constexpr const dreal T = (dreal)2.6767385370978665559; // closure 378*T**5/5 - 381*T**4 + 24893*T**3/40 - 1713*T**2/4 + 921*T/8 - 9/2 = 0; only root where every weights is positive
	static constexpr const dreal n1oT = 1/T;
	static constexpr dreal w_1 = (4*T*T - 15*T + 15)/(720*T*T*T);
	static constexpr dreal w_2 = (18*T*T - 65*T + 45)/(960*T*T*T);
	static constexpr dreal w_3 = (-36*T*T + 125*T - 75)/(960*T*T*T);
	static constexpr dreal w_10 = (-144*T*T + 520*T - 345)/(120*T*T*T);
	static constexpr dreal w_11 = (72*T*T - 250*T + 165)/(30*T*T*T);
	static constexpr dreal w_14 = (-324*T*T + 1215*T - 815)/(80*T*T*T);
	static constexpr dreal w_27 = (240*T*T*T + 1288*T*T - 6405*T + 4455)/(240*T*T*T);


	__cuda_callable__ static dreal feq(dreal rho, int qx, int qy, int qz, dreal vx, dreal vy, dreal vz, int id){
		const dreal ws[53] = {w_1,w_2,w_3,w_2,w_3,w_3,w_2,w_3,w_2,w_10,w_11,w_10,w_11,w_14,w_11,w_10,w_11,w_10,w_1,w_3,w_3,w_11,w_14,w_11,w_1,w_14,w_27,w_14,w_1,w_11,w_14,w_11,w_3,w_3,w_1,w_10,w_11,w_10,w_11,w_14,w_11,w_10,w_11,w_10,w_2,w_3,w_2,w_3,w_3,w_2,w_3,w_2,w_1};
		const dreal eq = no1 - n1o2 * n1oT * (vx * vx + vy * vy + vz * vz) + n1oT * (qx * vx + qy * vy + qz * vz)
		+ n1o2 * n1oT * n1oT * (qx * vx + qy * vy + qz * vz) * (qx * vx + qy * vy + qz * vz);
		return ws[id]*rho*eq;
	}


};
