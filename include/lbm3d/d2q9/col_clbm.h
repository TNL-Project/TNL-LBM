#pragma once

#include "common.h"
#include "eq.h"

template <typename TRAITS, typename LBM_EQ = D2Q9_EQ<TRAITS>>
struct D2Q9_CLBM : D2Q9_COMMON<TRAITS, LBM_EQ>
{
	using dreal = typename TRAITS::dreal;

	static constexpr const char* id = "CLBM";

	template <typename LBM_KS>
	__cuda_callable__ static void collision(LBM_KS& KS)
	{
		dreal tau = no3 * KS.lbmViscosity + n1o2;
		//      original Robertuv kod podle Geiera + PE Premnath
		dreal P = (dreal) 1. / (dreal) 12.
				* (KS.rho * (KS.vx * KS.vx + KS.vy * KS.vy) - KS.f[pz] - KS.f[zp] - KS.f[zm] - KS.f[mz]
				   - (dreal) 2. * (KS.f[pm] + KS.f[mm] + KS.f[pp] + KS.f[mp] - (dreal) 1. / (dreal) 3. * KS.rho)
				   - (KS.fx * KS.vx + KS.fy * KS.vy));	//FIXME c_s^2 instead of 1/3 - temperature influence on c_s? same for components of matrix S
														//-> diff[pz]rent coef. for P,NE,V...
		dreal NE = (dreal) .25 / tau
				 * (KS.f[zp] + KS.f[zm] - KS.f[pz] - KS.f[mz] + KS.rho * (KS.vx * KS.vx - KS.vy * KS.vy) - (KS.fx * KS.vx - KS.fy * KS.vy));
		dreal V =
			(dreal) .25 / tau * ((KS.f[pp] + KS.f[mm] - KS.f[mp] - KS.f[pm]) - KS.vx * KS.vy * KS.rho + (dreal) .5 * (KS.fx * KS.vy + KS.fy * KS.vx));
		dreal kxxyy = (KS.f[pz] + KS.f[pp] + KS.f[mp] + KS.f[pm] + KS.f[mm] + KS.f[mz] - KS.vx * KS.vx * KS.rho + (dreal) 2. * NE + (dreal) 6. * P)
					* (KS.f[zp] + KS.f[pp] + KS.f[mp] + KS.f[zm] + KS.f[pm] + KS.f[mm] - KS.vy * KS.vy * KS.rho - (dreal) 2. * NE + (dreal) 6. * P);
		//kxxyy = KS.rho/no9;
		dreal UP =
			(-((dreal) .25
				   * (KS.f[pm] + KS.f[mm] - KS.f[pp] - KS.f[mp] - (dreal) 2. * KS.vx * KS.vx * KS.vy * KS.rho
					  + KS.vy * (KS.rho - KS.f[zp] - KS.f[zm] - KS.f[zz]) - (dreal) .5 * (-KS.vx * KS.vx) * KS.fy + KS.fx * KS.vx * KS.vy)
			   - KS.vy * (dreal) .5 * (-(dreal) 3. * P - NE) + KS.vx * ((KS.f[pp] - KS.f[mp] - KS.f[pm] + KS.f[mm]) * (dreal) .5 - (dreal) 2. * V)));
		dreal RIGHT =
			(-((dreal) .25
				   * (KS.f[mm] + KS.f[mp] - KS.f[pm] - KS.f[pp] - (dreal) 2. * KS.vy * KS.vy * KS.vx * KS.rho
					  + KS.vx * (KS.rho - KS.f[zz] - KS.f[mz] - KS.f[pz]) - (dreal) .5 * (-KS.vy * KS.vy) * KS.fx + KS.fy * KS.vy * KS.vx)
			   - KS.vx * (dreal) .5 * (-(dreal) 3. * P + NE) + KS.vy * ((KS.f[pp] + KS.f[mm] - KS.f[pm] - KS.f[mp]) * (dreal) .5 - (dreal) 2. * V)));
		dreal NP =
			((dreal) .25
			 * (kxxyy - KS.f[pp] - KS.f[mp] - KS.f[pm] - KS.f[mm] - (dreal) 8. * P
				+ (dreal) 2.
					  * (KS.vx * (KS.f[pp] - KS.f[mp] + KS.f[pm] - KS.f[mm] - (dreal) 4. * RIGHT)
						 + KS.vy * (KS.f[pp] + KS.f[mp] - KS.f[pm] - KS.f[mm] - (dreal) 4. * UP))
				+ (dreal) 4. * KS.vx * KS.vy * (-KS.f[pp] + KS.f[mp] + KS.f[pm] - KS.f[mm] + (dreal) 4. * V)
				+ KS.vx * KS.vx * (-KS.f[zp] - KS.f[pp] - KS.f[mp] - KS.f[zm] - KS.f[pm] - KS.f[mm] + (dreal) 2. * NE - (dreal) 6. * P)
				+ KS.vy * KS.vy
					  * ((-KS.f[pz] - KS.f[pp] - KS.f[mp] - KS.f[pm] - KS.f[mm] - KS.f[mz] - (dreal) 2. * NE - (dreal) 6. * P)
						 + (dreal) 3. * KS.vx * KS.vx * KS.rho)
				- (KS.fx * KS.vx * KS.vy * KS.vy + KS.fy * KS.vy * KS.vx * KS.vx)));

		KS.f[mp] += (dreal) 2. * P + NP + V - UP + RIGHT;
		KS.f[mz] += -P - (dreal) 2. * NP + NE - (dreal) 2. * RIGHT;
		KS.f[mm] += (dreal) 2. * P + NP - V + UP + RIGHT;
		KS.f[zm] += -P - (dreal) 2. * NP - NE - (dreal) 2. * UP;
		KS.f[pm] += (dreal) 2. * P + NP + V + UP - RIGHT;
		KS.f[pz] += -P - (dreal) 2. * NP + NE + (dreal) 2. * RIGHT;
		KS.f[pp] += (dreal) 2. * P + NP - V - UP - RIGHT;
		KS.f[zp] += -P - (dreal) 2. * NP - NE + (dreal) 2. * UP;
		KS.f[zz] += ((dreal) 4. * (-P + NP));

		// add forcing
		// CLBM Forcing based on Premnath+Banerjee 1202.6087v1.pdf 2012:
		// "Incorporating Forcing Terms in Cascaded Lattice-Boltzmann Approach by Method of Central Moments"
		dreal m1 = KS.fx;
		dreal m2 = KS.fy;
		dreal m3 = (dreal) 6.0 * (KS.fx * KS.vx + KS.fy * KS.vy);
		dreal m4 = (dreal) 2.0 * (KS.fx * KS.vx - KS.fy * KS.vy);
		dreal m5 = KS.fx * KS.vy + KS.fy * KS.vx;
		dreal m6 = ((dreal) 2.0 - (dreal) 3.0 * KS.vx * KS.vx) * KS.fy - (dreal) 6.0 * KS.fx * KS.vx * KS.vy;
		dreal m7 = ((dreal) 2.0 - (dreal) 3.0 * KS.vy * KS.vy) * KS.fx - (dreal) 6.0 * KS.fy * KS.vx * KS.vy;
		dreal m8 =
			(dreal) 6.0 * (((dreal) 3.0 * KS.vy * KS.vy - (dreal) 2.0) * KS.fx * KS.vx + ((dreal) 3.0 * KS.vx * KS.vx - (dreal) 2.0) * KS.fy * KS.vy);

		KS.f[zz] += (-m3 + m8) / (dreal) 9.0;
		KS.f[pz] += ((dreal) 6.0 * m1 - m3 + (dreal) 9.0 * m4 + (dreal) 6.0 * m7 - (dreal) 2.0 * m8) / (dreal) 36.0;
		KS.f[zp] += ((dreal) 6.0 * m2 - m3 - (dreal) 9.0 * m4 + (dreal) 6.0 * m6 - (dreal) 2.0 * m8) / (dreal) 36.0;
		KS.f[mz] += (-(dreal) 6.0 * m1 - m3 + (dreal) 9.0 * m4 - (dreal) 6.0 * m7 - (dreal) 2.0 * m8) / (dreal) 36.0;
		KS.f[zm] += (-(dreal) 6.0 * m2 - m3 - (dreal) 9.0 * m4 - (dreal) 6.0 * m6 - (dreal) 2.0 * m8) / (dreal) 36.0;
		KS.f[pp] +=
			((dreal) 6.0 * m1 + (dreal) 6.0 * m2 + (dreal) 2.0 * m3 + (dreal) 9.0 * m5 - (dreal) 3.0 * m6 - (dreal) 3.0 * m7 + m8) / (dreal) 36.0;
		KS.f[mp] +=
			(-(dreal) 6.0 * m1 + (dreal) 6.0 * m2 + (dreal) 2.0 * m3 - (dreal) 9.0 * m5 - (dreal) 3.0 * m6 + (dreal) 3.0 * m7 + m8) / (dreal) 36.0;
		KS.f[mm] +=
			(-(dreal) 6.0 * m1 - (dreal) 6.0 * m2 + (dreal) 2.0 * m3 + (dreal) 9.0 * m5 + (dreal) 3.0 * m6 + (dreal) 3.0 * m7 + m8) / (dreal) 36.0;
		KS.f[pm] +=
			((dreal) 6.0 * m1 - (dreal) 6.0 * m2 + (dreal) 2.0 * m3 - (dreal) 9.0 * m5 + (dreal) 3.0 * m6 - (dreal) 3.0 * m7 + m8) / (dreal) 36.0;
	}
};
