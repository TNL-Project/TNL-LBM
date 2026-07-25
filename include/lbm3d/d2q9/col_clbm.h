#pragma once

#include "common.h"
#include "eq.h"

template <typename TRAITS, typename LBM_EQ = D2Q9_EQ<TRAITS>>
struct D2Q9_CLBM_Straka2016 : D2Q9_COMMON<TRAITS, LBM_EQ>
{
	using dreal = typename TRAITS::dreal;

	static constexpr const char* id = "CLBM_Straka2016";

	template <typename LBM_KS>
	__cuda_callable__ static void collision(LBM_KS& KS)
	{
		dreal tau = no3 * KS.lbmViscosity + n1o2;
		// based on Geier, Greiner, Korvink 2007 https://doi.org/10.1142/S0129183107010681
		// and on Straka 2016 https://doi.org/10.1051/meca/2015071
		dreal P = (dreal) 1. / (dreal) 12.
				* (KS.rho * (KS.vx * KS.vx + KS.vy * KS.vy) - KS.f[dir9::pz] - KS.f[dir9::zp] - KS.f[dir9::zm] - KS.f[dir9::mz]
				   - (dreal) 2. * (KS.f[dir9::pm] + KS.f[dir9::mm] + KS.f[dir9::pp] + KS.f[dir9::mp] - (dreal) 1. / (dreal) 3. * KS.rho)
				   - (KS.fx * KS.vx + KS.fy * KS.vy));	//FIXME c_s^2 instead of 1/3 - temperature influence on c_s? same for components of matrix S
														//-> different coef. for P,NE,V...
		dreal NE = (dreal) .25 / tau
				 * (KS.f[dir9::zp] + KS.f[dir9::zm] - KS.f[dir9::pz] - KS.f[dir9::mz] + KS.rho * (KS.vx * KS.vx - KS.vy * KS.vy)
					- (KS.fx * KS.vx - KS.fy * KS.vy));
		dreal V = (dreal) .25 / tau
				* ((KS.f[dir9::pp] + KS.f[dir9::mm] - KS.f[dir9::mp] - KS.f[dir9::pm]) - KS.vx * KS.vy * KS.rho
				   + (dreal) .5 * (KS.fx * KS.vy + KS.fy * KS.vx));
		dreal kxxyy = (KS.f[dir9::pz] + KS.f[dir9::pp] + KS.f[dir9::mp] + KS.f[dir9::pm] + KS.f[dir9::mm] + KS.f[dir9::mz] - KS.vx * KS.vx * KS.rho
					   + (dreal) 2. * NE + (dreal) 6. * P)
					* (KS.f[dir9::zp] + KS.f[dir9::pp] + KS.f[dir9::mp] + KS.f[dir9::zm] + KS.f[dir9::pm] + KS.f[dir9::mm] - KS.vy * KS.vy * KS.rho
					   - (dreal) 2. * NE + (dreal) 6. * P);
		//kxxyy = KS.rho/no9;
		dreal UP =
			(-((dreal) .25
				   * (KS.f[dir9::pm] + KS.f[dir9::mm] - KS.f[dir9::pp] - KS.f[dir9::mp] - (dreal) 2. * KS.vx * KS.vx * KS.vy * KS.rho
					  + KS.vy * (KS.rho - KS.f[dir9::zp] - KS.f[dir9::zm] - KS.f[dir9::zz]) - (dreal) .5 * (-KS.vx * KS.vx) * KS.fy
					  + KS.fx * KS.vx * KS.vy)
			   - KS.vy * (dreal) .5 * (-(dreal) 3. * P - NE)
			   + KS.vx * ((KS.f[dir9::pp] - KS.f[dir9::mp] - KS.f[dir9::pm] + KS.f[dir9::mm]) * (dreal) .5 - (dreal) 2. * V)));
		dreal RIGHT =
			(-((dreal) .25
				   * (KS.f[dir9::mm] + KS.f[dir9::mp] - KS.f[dir9::pm] - KS.f[dir9::pp] - (dreal) 2. * KS.vy * KS.vy * KS.vx * KS.rho
					  + KS.vx * (KS.rho - KS.f[dir9::zz] - KS.f[dir9::mz] - KS.f[dir9::pz]) - (dreal) .5 * (-KS.vy * KS.vy) * KS.fx
					  + KS.fy * KS.vy * KS.vx)
			   - KS.vx * (dreal) .5 * (-(dreal) 3. * P + NE)
			   + KS.vy * ((KS.f[dir9::pp] + KS.f[dir9::mm] - KS.f[dir9::pm] - KS.f[dir9::mp]) * (dreal) .5 - (dreal) 2. * V)));
		dreal NP =
			((dreal) .25
			 * (kxxyy - KS.f[dir9::pp] - KS.f[dir9::mp] - KS.f[dir9::pm] - KS.f[dir9::mm] - (dreal) 8. * P
				+ (dreal) 2.
					  * (KS.vx * (KS.f[dir9::pp] - KS.f[dir9::mp] + KS.f[dir9::pm] - KS.f[dir9::mm] - (dreal) 4. * RIGHT)
						 + KS.vy * (KS.f[dir9::pp] + KS.f[dir9::mp] - KS.f[dir9::pm] - KS.f[dir9::mm] - (dreal) 4. * UP))
				+ (dreal) 4. * KS.vx * KS.vy * (-KS.f[dir9::pp] + KS.f[dir9::mp] + KS.f[dir9::pm] - KS.f[dir9::mm] + (dreal) 4. * V)
				+ KS.vx * KS.vx
					  * (-KS.f[dir9::zp] - KS.f[dir9::pp] - KS.f[dir9::mp] - KS.f[dir9::zm] - KS.f[dir9::pm] - KS.f[dir9::mm] + (dreal) 2. * NE
						 - (dreal) 6. * P)
				+ KS.vy * KS.vy
					  * ((-KS.f[dir9::pz] - KS.f[dir9::pp] - KS.f[dir9::mp] - KS.f[dir9::pm] - KS.f[dir9::mm] - KS.f[dir9::mz] - (dreal) 2. * NE
						  - (dreal) 6. * P)
						 + (dreal) 3. * KS.vx * KS.vx * KS.rho)
				- (KS.fx * KS.vx * KS.vy * KS.vy + KS.fy * KS.vy * KS.vx * KS.vx)));

		KS.f[dir9::mp] += (dreal) 2. * P + NP + V - UP + RIGHT;
		KS.f[dir9::mz] += -P - (dreal) 2. * NP + NE - (dreal) 2. * RIGHT;
		KS.f[dir9::mm] += (dreal) 2. * P + NP - V + UP + RIGHT;
		KS.f[dir9::zm] += -P - (dreal) 2. * NP - NE - (dreal) 2. * UP;
		KS.f[dir9::pm] += (dreal) 2. * P + NP + V + UP - RIGHT;
		KS.f[dir9::pz] += -P - (dreal) 2. * NP + NE + (dreal) 2. * RIGHT;
		KS.f[dir9::pp] += (dreal) 2. * P + NP - V - UP - RIGHT;
		KS.f[dir9::zp] += -P - (dreal) 2. * NP - NE + (dreal) 2. * UP;
		KS.f[dir9::zz] += ((dreal) 4. * (-P + NP));

		// add forcing based on Premnath and Banerjee https://doi.org/10.1103/PhysRevE.80.036702
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

		KS.f[dir9::zz] += (-m3 + m8) / (dreal) 9.0;
		KS.f[dir9::pz] += ((dreal) 6.0 * m1 - m3 + (dreal) 9.0 * m4 + (dreal) 6.0 * m7 - (dreal) 2.0 * m8) / (dreal) 36.0;
		KS.f[dir9::zp] += ((dreal) 6.0 * m2 - m3 - (dreal) 9.0 * m4 + (dreal) 6.0 * m6 - (dreal) 2.0 * m8) / (dreal) 36.0;
		KS.f[dir9::mz] += (-(dreal) 6.0 * m1 - m3 + (dreal) 9.0 * m4 - (dreal) 6.0 * m7 - (dreal) 2.0 * m8) / (dreal) 36.0;
		KS.f[dir9::zm] += (-(dreal) 6.0 * m2 - m3 - (dreal) 9.0 * m4 - (dreal) 6.0 * m6 - (dreal) 2.0 * m8) / (dreal) 36.0;
		KS.f[dir9::pp] +=
			((dreal) 6.0 * m1 + (dreal) 6.0 * m2 + (dreal) 2.0 * m3 + (dreal) 9.0 * m5 - (dreal) 3.0 * m6 - (dreal) 3.0 * m7 + m8) / (dreal) 36.0;
		KS.f[dir9::mp] +=
			(-(dreal) 6.0 * m1 + (dreal) 6.0 * m2 + (dreal) 2.0 * m3 - (dreal) 9.0 * m5 - (dreal) 3.0 * m6 + (dreal) 3.0 * m7 + m8) / (dreal) 36.0;
		KS.f[dir9::mm] +=
			(-(dreal) 6.0 * m1 - (dreal) 6.0 * m2 + (dreal) 2.0 * m3 + (dreal) 9.0 * m5 + (dreal) 3.0 * m6 + (dreal) 3.0 * m7 + m8) / (dreal) 36.0;
		KS.f[dir9::pm] +=
			((dreal) 6.0 * m1 - (dreal) 6.0 * m2 + (dreal) 2.0 * m3 - (dreal) 9.0 * m5 + (dreal) 3.0 * m6 - (dreal) 3.0 * m7 + m8) / (dreal) 36.0;
	}
};

template <typename TRAITS, typename LBM_EQ = D2Q9_EQ<TRAITS>>
struct D2Q9_CLBM : D2Q9_COMMON<TRAITS, LBM_EQ>
{
	using dreal = typename TRAITS::dreal;

	static constexpr const char* id = "CLBM";

	template <typename LBM_KS>
	__cuda_callable__ static void collision(LBM_KS& KS)
	{
		// based on Geier 2017 https://doi.org/10.1016/j.jcp.2017.05.040
		// 2D reduction: z-shift omitted, y-shift then x-shift

		// y-shifted partial moments for each x-group
		// x = -1
		const dreal k_m0 = KS.f[dir9::mp] + KS.f[dir9::mm] + KS.f[dir9::mz];
		const dreal k_m1 = (KS.f[dir9::mp] - KS.f[dir9::mm]) - KS.vy * k_m0;
		const dreal k_m2 = (KS.f[dir9::mp] + KS.f[dir9::mm]) - no2 * KS.vy * (KS.f[dir9::mp] - KS.f[dir9::mm]) + KS.vy * KS.vy * k_m0;

		// x = 0
		const dreal k_z0 = KS.f[dir9::zp] + KS.f[dir9::zm] + KS.f[dir9::zz];
		const dreal k_z1 = (KS.f[dir9::zp] - KS.f[dir9::zm]) - KS.vy * k_z0;
		const dreal k_z2 = (KS.f[dir9::zp] + KS.f[dir9::zm]) - no2 * KS.vy * (KS.f[dir9::zp] - KS.f[dir9::zm]) + KS.vy * KS.vy * k_z0;

		// x = +1
		const dreal k_p0 = KS.f[dir9::pp] + KS.f[dir9::pm] + KS.f[dir9::pz];
		const dreal k_p1 = (KS.f[dir9::pp] - KS.f[dir9::pm]) - KS.vy * k_p0;
		const dreal k_p2 = (KS.f[dir9::pp] + KS.f[dir9::pm]) - no2 * KS.vy * (KS.f[dir9::pp] - KS.f[dir9::pm]) + KS.vy * KS.vy * k_p0;

		// x-shift → final central moments
		const dreal k_00 = (k_p0 + k_m0) + k_z0;
		const dreal k_01 = (k_p1 + k_m1) + k_z1;
		const dreal k_02 = (k_p2 + k_m2) + k_z2;
		const dreal k_10 = (k_p0 - k_m0) - KS.vx * k_00;
		const dreal k_11 = (k_p1 - k_m1) - KS.vx * k_01;
		const dreal k_12 = (k_p2 - k_m2) - KS.vx * k_02;
		const dreal k_20 = (k_p0 + k_m0) - no2 * KS.vx * (k_p0 - k_m0) + KS.vx * KS.vx * k_00;
		const dreal k_21 = (k_p1 + k_m1) - no2 * KS.vx * (k_p1 - k_m1) + KS.vx * KS.vx * k_01;
		const dreal k_22 = (k_p2 + k_m2) - no2 * KS.vx * (k_p2 - k_m2) + KS.vx * KS.vx * k_02;

		// relaxation
		const dreal omega1 = no1 / (no3 * KS.lbmViscosity + n1o2);
		const dreal omega3 = no1;  // k_21, k_12 fully relaxed to equilibrium
		const dreal omega4 = no1;  // k_22 fully relaxed to equilibrium

		const dreal ks_00 = k_00;
		const dreal ks_10 = k_10;
		const dreal ks_01 = k_01;

		const dreal ks_20 = KS.rho * n1o3 + n1o2 * (no1 - omega1) * (k_20 - k_02);
		const dreal ks_02 = KS.rho * n1o3 - n1o2 * (no1 - omega1) * (k_20 - k_02);
		const dreal ks_11 = (no1 - omega1) * k_11;

		const dreal ks_21 = (no1 - omega3) * k_21;
		const dreal ks_12 = (no1 - omega3) * k_12;
		const dreal ks_22 = (no1 - omega4) * k_22 + omega4 * KS.rho * n1o9;

		// backward transform: reverse x-shift
		const dreal ks_p0 = (ks_00 * (KS.vx * KS.vx + KS.vx) + ks_10 * (no2 * KS.vx + no1) + ks_20) * n1o2;
		const dreal ks_p1 = (ks_01 * (KS.vx * KS.vx + KS.vx) + ks_11 * (no2 * KS.vx + no1) + ks_21) * n1o2;
		const dreal ks_p2 = (ks_02 * (KS.vx * KS.vx + KS.vx) + ks_12 * (no2 * KS.vx + no1) + ks_22) * n1o2;

		const dreal ks_m0 = (ks_00 * (KS.vx * KS.vx - KS.vx) + ks_10 * (no2 * KS.vx - no1) + ks_20) * n1o2;
		const dreal ks_m1 = (ks_01 * (KS.vx * KS.vx - KS.vx) + ks_11 * (no2 * KS.vx - no1) + ks_21) * n1o2;
		const dreal ks_m2 = (ks_02 * (KS.vx * KS.vx - KS.vx) + ks_12 * (no2 * KS.vx - no1) + ks_22) * n1o2;

		const dreal ks_z0 = ks_00 * (no1 - KS.vx * KS.vx) - no2 * KS.vx * ks_10 - ks_20;
		const dreal ks_z1 = ks_01 * (no1 - KS.vx * KS.vx) - no2 * KS.vx * ks_11 - ks_21;
		const dreal ks_z2 = ks_02 * (no1 - KS.vx * KS.vx) - no2 * KS.vx * ks_12 - ks_22;

		// backward transform: reverse y-shift → populations
		KS.f[dir9::pp] = (ks_p0 * (KS.vy * KS.vy + KS.vy) + ks_p1 * (no2 * KS.vy + no1) + ks_p2) * n1o2;
		KS.f[dir9::pm] = (ks_p0 * (KS.vy * KS.vy - KS.vy) + ks_p1 * (no2 * KS.vy - no1) + ks_p2) * n1o2;
		KS.f[dir9::pz] = ks_p0 * (no1 - KS.vy * KS.vy) - no2 * KS.vy * ks_p1 - ks_p2;

		KS.f[dir9::mp] = (ks_m0 * (KS.vy * KS.vy + KS.vy) + ks_m1 * (no2 * KS.vy + no1) + ks_m2) * n1o2;
		KS.f[dir9::mm] = (ks_m0 * (KS.vy * KS.vy - KS.vy) + ks_m1 * (no2 * KS.vy - no1) + ks_m2) * n1o2;
		KS.f[dir9::mz] = ks_m0 * (no1 - KS.vy * KS.vy) - no2 * KS.vy * ks_m1 - ks_m2;

		KS.f[dir9::zp] = (ks_z0 * (KS.vy * KS.vy + KS.vy) + ks_z1 * (no2 * KS.vy + no1) + ks_z2) * n1o2;
		KS.f[dir9::zm] = (ks_z0 * (KS.vy * KS.vy - KS.vy) + ks_z1 * (no2 * KS.vy - no1) + ks_z2) * n1o2;
		KS.f[dir9::zz] = ks_z0 * (no1 - KS.vy * KS.vy) - no2 * KS.vy * ks_z1 - ks_z2;

		// forcing: Premnath-Banerjee central-moment forcing, 2D reduction
		// source: https://doi.org/10.1103/PhysRevE.80.036702
		const dreal m1 = KS.fx;
		const dreal m2 = KS.fy;
		const dreal m3 = no6 * (KS.fx * KS.vx + KS.fy * KS.vy);
		const dreal m4 = no2 * (KS.fx * KS.vx - KS.fy * KS.vy);
		const dreal m5 = KS.fx * KS.vy + KS.fy * KS.vx;
		const dreal m6 = (no2 - no3 * KS.vx * KS.vx) * KS.fy - no6 * KS.fx * KS.vx * KS.vy;
		const dreal m7 = (no2 - no3 * KS.vy * KS.vy) * KS.fx - no6 * KS.fy * KS.vx * KS.vy;
		const dreal m8 = no6 * ((no3 * KS.vy * KS.vy - no2) * KS.fx * KS.vx + (no3 * KS.vx * KS.vx - no2) * KS.fy * KS.vy);

		KS.f[dir9::zz] += (-m3 + m8) * n1o9;
		KS.f[dir9::pz] += (no6 * m1 - m3 + no9 * m4 + no6 * m7 - no2 * m8) * n1o36;
		KS.f[dir9::zp] += (no6 * m2 - m3 - no9 * m4 + no6 * m6 - no2 * m8) * n1o36;
		KS.f[dir9::mz] += (-no6 * m1 - m3 + no9 * m4 - no6 * m7 - no2 * m8) * n1o36;
		KS.f[dir9::zm] += (-no6 * m2 - m3 - no9 * m4 - no6 * m6 - no2 * m8) * n1o36;
		KS.f[dir9::pp] += (no6 * m1 + no6 * m2 + no2 * m3 + no9 * m5 - no3 * m6 - no3 * m7 + m8) * n1o36;
		KS.f[dir9::mp] += (-no6 * m1 + no6 * m2 + no2 * m3 - no9 * m5 - no3 * m6 + no3 * m7 + m8) * n1o36;
		KS.f[dir9::mm] += (-no6 * m1 - no6 * m2 + no2 * m3 + no9 * m5 + no3 * m6 + no3 * m7 + m8) * n1o36;
		KS.f[dir9::pm] += (no6 * m1 - no6 * m2 + no2 * m3 - no9 * m5 + no3 * m6 - no3 * m7 + m8) * n1o36;
	}
};
