#pragma once

#include <TNL/Backend/Macros.h>

#include "../../lbm_common/ciselnik.h"

// second order Maxwell-Boltzmann Equilibrium
template <typename TRAITS>
struct D3Q27_EQ_ENTROPIC2
{
	using dreal = typename TRAITS::dreal;

	static constexpr dreal w0 = 2./3;
	static constexpr dreal w1 = 1./6;
    __cuda_callable__ static dreal feq(int qx, int qy, int qz, dreal vx, dreal vy, dreal vz){
        static const dreal ws[2] = {w0,w1};
        const dreal Kx = (dreal)sqrt(1.+3.*vx*vx);
        const dreal Ky = (dreal)sqrt(1.+3.*vy*vy);
        const dreal Kz = (dreal)sqrt(1.+3.*vz*vz);
        const dreal Bx = (dreal)1.0*(2.*vx+Kx)/(1.-vx);
        const dreal Ax = (dreal)1./(w0+w1*Bx+1.0*w1/Bx);
        const dreal By = (dreal)1.0*(2.*vy+Ky)/(1.-vy);
        const dreal Ay = (dreal)1./(w0+w1*By+1.0*w1/By);
		const dreal Bz = (dreal)1.0*(2.*vz+Kz)/(1.-vz);
        const dreal Az = (dreal)1./(w0+w1*Bz+1.0*w1/Bz);

        dreal result = (dreal)Ax*Ay*Az*ws[abs(qx)]*ws[abs(qy)]*ws[abs(qz)];
        if(qx == -1){
            result /= Bx;
        }else if (qx == 1) {
            result *= Bx;
        }
        if(qy == -1){
            result /= By;
        }else if (qy == 1) {
            result *= By;
        }
		if(qz == -1){
            result /= Bz;
        }else if (qz == 1) {
            result *= Bz;
        }
		return result;

    }


};
