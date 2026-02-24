#pragma once

#include "common.h"
#include "eq.h"

// improved BRK (SRT) model by Geier 2017
// for standard DF (no well-conditioned)

template <typename TRAITS, typename LBM_EQ = D3Q53_EQ<TRAITS>>
struct D3Q53_SRT : D3Q53_COMMON<TRAITS, LBM_EQ>
{
	using dreal = typename TRAITS::dreal;

	static constexpr const char* id = "SRT";

	template <typename LBM_KS>
	__cuda_callable__ static void collision(LBM_KS& KS)
	{
		const dreal beta1 = 1./(no2*KS.lbmViscosity/KS.T0 + no1);

		#ifdef __CUDA_ARCH__
		#pragma unroll
		#endif
		for(int id = 0; id < LBM_KS::Q; id++){
			const Coord c = LBM_KS::id_to_dv(id);
            KS.f[id] += (KS.rho*LBM_EQ::feq(c.x,c.y,c.z,KS.vx,KS.vy,KS.vz,id) - KS.f[id])*beta1*2;
        }
	}
};
