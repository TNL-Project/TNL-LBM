#pragma once

#include "common.h"
#include "eq.h"

template <typename TRAITS, typename LBM_EQ = D3Q27_EQ<TRAITS>>
struct D3Q27_MRT : D3Q27_COMMON<TRAITS, LBM_EQ>
{
	using dreal = typename TRAITS::dreal;

	static constexpr const char* id = "MRT_LES";

	template <typename LBM_KS>
	__cuda_callable__ static void collision(LBM_KS& KS)
	{
		dreal Pxx = KS.f[dir27::mmm] + KS.f[dir27::mmz] + KS.f[dir27::mmp] + KS.f[dir27::mzm] + KS.f[dir27::mzz] + KS.f[dir27::mzp] + KS.f[dir27::mpm]
				  + KS.f[dir27::mpz] + KS.f[dir27::mpp] + KS.f[dir27::pmm] + KS.f[dir27::pmz] + KS.f[dir27::pmp] + KS.f[dir27::pzm] + KS.f[dir27::pzz]
				  + KS.f[dir27::pzp] + KS.f[dir27::ppm] + KS.f[dir27::ppz] + KS.f[dir27::ppp];	//Second order moment Pi
		dreal Pyy = KS.f[dir27::mmm] + KS.f[dir27::mmz] + KS.f[dir27::mmp] + KS.f[dir27::mpm] + KS.f[dir27::mpz] + KS.f[dir27::mpp] + KS.f[dir27::zmm]
				  + KS.f[dir27::zmz] + KS.f[dir27::zmp] + KS.f[dir27::zpm] + KS.f[dir27::zpz] + KS.f[dir27::zpp] + KS.f[dir27::pmm] + KS.f[dir27::pmz]
				  + KS.f[dir27::pmp] + KS.f[dir27::ppm] + KS.f[dir27::ppz] + KS.f[dir27::ppp];
		dreal Pzz = KS.f[dir27::mmm] + KS.f[dir27::mmp] + KS.f[dir27::mzm] + KS.f[dir27::mzp] + KS.f[dir27::mpm] + KS.f[dir27::mpp] + KS.f[dir27::zmm]
				  + KS.f[dir27::zmp] + KS.f[dir27::zzm] + KS.f[dir27::zzp] + KS.f[dir27::zpm] + KS.f[dir27::zpp] + KS.f[dir27::pmm] + KS.f[dir27::pmp]
				  + KS.f[dir27::pzm] + KS.f[dir27::pzp] + KS.f[dir27::ppm] + KS.f[dir27::ppp];
		dreal Pxy = KS.f[dir27::mmm] + KS.f[dir27::mmz] + KS.f[dir27::mmp] - KS.f[dir27::mpm] - KS.f[dir27::mpz] - KS.f[dir27::mpp] - KS.f[dir27::pmm]
				  - KS.f[dir27::pmz] - KS.f[dir27::pmp] + KS.f[dir27::ppm] + KS.f[dir27::ppz] + KS.f[dir27::ppp];
		dreal Pxz = KS.f[dir27::mmm] - KS.f[dir27::mmp] + KS.f[dir27::mzm] - KS.f[dir27::mzp] + KS.f[dir27::mpm] - KS.f[dir27::mpp] - KS.f[dir27::pmm]
				  + KS.f[dir27::pmp] - KS.f[dir27::pzm] + KS.f[dir27::pzp] - KS.f[dir27::ppm] + KS.f[dir27::ppp];
		dreal Pyz = KS.f[dir27::mmm] - KS.f[dir27::mmp] - KS.f[dir27::mpm] + KS.f[dir27::mpp] + KS.f[dir27::zmm] - KS.f[dir27::zmp] - KS.f[dir27::zpm]
				  + KS.f[dir27::zpp] + KS.f[dir27::pmm] - KS.f[dir27::pmp] - KS.f[dir27::ppm] + KS.f[dir27::ppp];
		const dreal Pnxx = Pxx - KS.rho * (n1o3 + KS.vx * KS.vx);  //Non-equilibrium part of Pi
		const dreal Pnyy = Pyy - KS.rho * (n1o3 + KS.vy * KS.vy);
		const dreal Pnzz = Pzz - KS.rho * (n1o3 + KS.vz * KS.vz);
		const dreal Pnxz = Pxz - KS.rho * KS.vx * KS.vz;
		const dreal Pnxy = Pxy - KS.rho * KS.vx * KS.vy;
		const dreal Pnyz = Pyz - KS.rho * KS.vy * KS.vz;

		//FIXME check all LES sometime ];)
		const dreal Q =
			no2 * (Pnxx * Pnxx + Pnyy * Pnyy + Pnzz * Pnzz + no2 * (Pnxy * Pnxy + Pnxz * Pnxz + Pnyz * Pnyz));	//Strain rate tensor magnitude
		const dreal tau = no3 * KS.lbmViscosity + n1o2;
		const dreal C = (dreal) 0.0342;																 //.18^2 Smagorinsky constant .18-.19
		const dreal omega = no2 / (sqrt(tau * tau + no2 * C * no3 * no3 * sqrt(Q) / KS.rho) + tau);	 //Smagorinsky LES
																									 //const dreal omega = no1/tau;

		/*
		T	S = (sqrtf(visclb*visclb+(T)18.*C*C*sqrtf(Q))-visclb)/((T)6.*C);
		omega = (T)1./(iCsq*(visclb+C*S)+(T).5);
		tau = (T)1./omega;
		*/
		/*
		tau = iCsq*visclb+(T).5;
		tau = (T).5*(sqrtf(tau*tau+C*sqrtf((T)648.*Q)/(KS.rho+(T)1.))+tau); //Smagorinsky LES
		omega = (T)1./tau;
		*/
		Pxx -= omega * Pnxx;  //Relaxation, only Pi is relaxed, rho&u are conserved, rest is equilibrated (P.Dellar 2013(talk) -> a'la Ladd 1994)
		Pyy -= omega * Pnyy;
		Pzz -= omega * Pnzz;
		Pxy -= omega * Pnxy;
		Pxz -= omega * Pnxz;
		Pyz -= omega * Pnyz;

		KS.f[dir27::mmm] = n1o216
						 * (KS.rho * (n5o2 - n3o2 * 3 + no3 * (KS.vx * -no1 + KS.vy * -no1 + KS.vz * (-no1)))
							+ n9o2 * (Pxx * no1 + Pyy * no1 + Pzz * no1 + no2 * (Pxy * no1 + Pxz * no1 + Pyz * no1)) - n3o2 * (Pxx + Pyy + Pzz));
		KS.f[dir27::mmz] = n1o54
						 * (KS.rho * (n5o2 - n3o2 * 2 + no3 * (KS.vx * -no1 + KS.vy * -no1 + KS.vz * 0))
							+ n9o2 * (Pxx * no1 + Pyy * no1 + Pzz * 0 + no2 * (Pxy * no1 + Pxz * 0 + Pyz * 0)) - n3o2 * (Pxx + Pyy + Pzz));
		KS.f[dir27::mmp] = n1o216
						 * (KS.rho * (n5o2 - n3o2 * 3 + no3 * (KS.vx * -no1 + KS.vy * -no1 + KS.vz * no1))
							+ n9o2 * (Pxx * no1 + Pyy * no1 + Pzz * no1 + no2 * (Pxy * no1 + Pxz * -no1 + Pyz * (-no1))) - n3o2 * (Pxx + Pyy + Pzz));
		KS.f[dir27::mzm] = n1o54
						 * (KS.rho * (n5o2 - n3o2 * 2 + no3 * (KS.vx * -no1 + KS.vy * 0 + KS.vz * (-no1)))
							+ n9o2 * (Pxx * no1 + Pyy * 0 + Pzz * no1 + no2 * (Pxy * 0 + Pxz * no1 + Pyz * 0)) - n3o2 * (Pxx + Pyy + Pzz));
		KS.f[dir27::mzz] = n2o27
						 * (KS.rho * (n5o2 - n3o2 * no1 + no3 * (KS.vx * -no1 + KS.vy * 0 + KS.vz * 0))
							+ n9o2 * (Pxx * no1 + Pyy * 0 + Pzz * 0 + no2 * (Pxy * 0 + Pxz * 0 + Pyz * 0)) - n3o2 * (Pxx + Pyy + Pzz));
		KS.f[dir27::mzp] = n1o54
						 * (KS.rho * (n5o2 - n3o2 * 2 + no3 * (KS.vx * -no1 + KS.vy * 0 + KS.vz * no1))
							+ n9o2 * (Pxx * no1 + Pyy * 0 + Pzz * no1 + no2 * (Pxy * 0 + Pxz * -no1 + Pyz * 0)) - n3o2 * (Pxx + Pyy + Pzz));
		KS.f[dir27::mpm] = n1o216
						 * (KS.rho * (n5o2 - n3o2 * 3 + no3 * (KS.vx * -no1 + KS.vy * no1 + KS.vz * (-no1)))
							+ n9o2 * (Pxx * no1 + Pyy * no1 + Pzz * no1 + no2 * (Pxy * -no1 + Pxz * no1 + Pyz * (-no1))) - n3o2 * (Pxx + Pyy + Pzz));
		KS.f[dir27::mpz] = n1o54
						 * (KS.rho * (n5o2 - n3o2 * 2 + no3 * (KS.vx * -no1 + KS.vy * no1 + KS.vz * 0))
							+ n9o2 * (Pxx * no1 + Pyy * no1 + Pzz * 0 + no2 * (Pxy * -no1 + Pxz * 0 + Pyz * 0)) - n3o2 * (Pxx + Pyy + Pzz));
		KS.f[dir27::mpp] = n1o216
						 * (KS.rho * (n5o2 - n3o2 * 3 + no3 * (KS.vx * -no1 + KS.vy * no1 + KS.vz * no1))
							+ n9o2 * (Pxx * no1 + Pyy * no1 + Pzz * no1 + no2 * (Pxy * -no1 + Pxz * -no1 + Pyz * no1)) - n3o2 * (Pxx + Pyy + Pzz));
		KS.f[dir27::zmm] = n1o54
						 * (KS.rho * (n5o2 - n3o2 * 2 + no3 * (KS.vx * 0 + KS.vy * -no1 + KS.vz * (-no1)))
							+ n9o2 * (Pxx * 0 + Pyy * no1 + Pzz * no1 + no2 * (Pxy * 0 + Pxz * 0 + Pyz * no1)) - n3o2 * (Pxx + Pyy + Pzz));
		KS.f[dir27::zmz] = n2o27
						 * (KS.rho * (n5o2 - n3o2 * no1 + no3 * (KS.vx * 0 + KS.vy * -no1 + KS.vz * 0))
							+ n9o2 * (Pxx * 0 + Pyy * no1 + Pzz * 0 + no2 * (Pxy * 0 + Pxz * 0 + Pyz * 0)) - n3o2 * (Pxx + Pyy + Pzz));
		KS.f[dir27::zmp] = n1o54
						 * (KS.rho * (n5o2 - n3o2 * 2 + no3 * (KS.vx * 0 + KS.vy * -no1 + KS.vz * no1))
							+ n9o2 * (Pxx * 0 + Pyy * no1 + Pzz * no1 + no2 * (Pxy * 0 + Pxz * 0 + Pyz * (-no1))) - n3o2 * (Pxx + Pyy + Pzz));
		KS.f[dir27::zzm] = n2o27
						 * (KS.rho * (n5o2 - n3o2 * no1 + no3 * (KS.vx * 0 + KS.vy * 0 + KS.vz * (-no1)))
							+ n9o2 * (Pxx * 0 + Pyy * 0 + Pzz * no1 + no2 * (Pxy * 0 + Pxz * 0 + Pyz * 0)) - n3o2 * (Pxx + Pyy + Pzz));
		KS.f[dir27::zzz] = n8o27
						 * (KS.rho * (n5o2 - n3o2 * 0 + no3 * (KS.vx * 0 + KS.vy * 0 + KS.vz * 0))
							+ n9o2 * (Pxx * 0 + Pyy * 0 + Pzz * 0 + no2 * (Pxy * 0 + Pxz * 0 + Pyz * 0)) - n3o2 * (Pxx + Pyy + Pzz));
		KS.f[dir27::zzp] = n2o27
						 * (KS.rho * (n5o2 - n3o2 * no1 + no3 * (KS.vx * 0 + KS.vy * 0 + KS.vz * no1))
							+ n9o2 * (Pxx * 0 + Pyy * 0 + Pzz * no1 + no2 * (Pxy * 0 + Pxz * 0 + Pyz * 0)) - n3o2 * (Pxx + Pyy + Pzz));
		KS.f[dir27::zpm] = n1o54
						 * (KS.rho * (n5o2 - n3o2 * 2 + no3 * (KS.vx * 0 + KS.vy * no1 + KS.vz * (-no1)))
							+ n9o2 * (Pxx * 0 + Pyy * no1 + Pzz * no1 + no2 * (Pxy * 0 + Pxz * 0 + Pyz * (-no1))) - n3o2 * (Pxx + Pyy + Pzz));
		KS.f[dir27::zpz] = n2o27
						 * (KS.rho * (n5o2 - n3o2 * no1 + no3 * (KS.vx * 0 + KS.vy * no1 + KS.vz * 0))
							+ n9o2 * (Pxx * 0 + Pyy * no1 + Pzz * 0 + no2 * (Pxy * 0 + Pxz * 0 + Pyz * 0)) - n3o2 * (Pxx + Pyy + Pzz));
		KS.f[dir27::zpp] = n1o54
						 * (KS.rho * (n5o2 - n3o2 * 2 + no3 * (KS.vx * 0 + KS.vy * no1 + KS.vz * no1))
							+ n9o2 * (Pxx * 0 + Pyy * no1 + Pzz * no1 + no2 * (Pxy * 0 + Pxz * 0 + Pyz * no1)) - n3o2 * (Pxx + Pyy + Pzz));
		KS.f[dir27::pmm] = n1o216
						 * (KS.rho * (n5o2 - n3o2 * 3 + no3 * (KS.vx * no1 + KS.vy * -no1 + KS.vz * (-no1)))
							+ n9o2 * (Pxx * no1 + Pyy * no1 + Pzz * no1 + no2 * (Pxy * -no1 + Pxz * -no1 + Pyz * no1)) - n3o2 * (Pxx + Pyy + Pzz));
		KS.f[dir27::pmz] = n1o54
						 * (KS.rho * (n5o2 - n3o2 * 2 + no3 * (KS.vx * no1 + KS.vy * -no1 + KS.vz * 0))
							+ n9o2 * (Pxx * no1 + Pyy * no1 + Pzz * 0 + no2 * (Pxy * -no1 + Pxz * 0 + Pyz * 0)) - n3o2 * (Pxx + Pyy + Pzz));
		KS.f[dir27::pmp] = n1o216
						 * (KS.rho * (n5o2 - n3o2 * 3 + no3 * (KS.vx * no1 + KS.vy * -no1 + KS.vz * no1))
							+ n9o2 * (Pxx * no1 + Pyy * no1 + Pzz * no1 + no2 * (Pxy * -no1 + Pxz * no1 + Pyz * (-no1))) - n3o2 * (Pxx + Pyy + Pzz));
		KS.f[dir27::pzm] = n1o54
						 * (KS.rho * (n5o2 - n3o2 * 2 + no3 * (KS.vx * no1 + KS.vy * 0 + KS.vz * (-no1)))
							+ n9o2 * (Pxx * no1 + Pyy * 0 + Pzz * no1 + no2 * (Pxy * 0 + Pxz * -no1 + Pyz * 0)) - n3o2 * (Pxx + Pyy + Pzz));
		KS.f[dir27::pzz] = n2o27
						 * (KS.rho * (n5o2 - n3o2 * no1 + no3 * (KS.vx * no1 + KS.vy * 0 + KS.vz * 0))
							+ n9o2 * (Pxx * no1 + Pyy * 0 + Pzz * 0 + no2 * (Pxy * 0 + Pxz * 0 + Pyz * 0)) - n3o2 * (Pxx + Pyy + Pzz));
		KS.f[dir27::pzp] = n1o54
						 * (KS.rho * (n5o2 - n3o2 * 2 + no3 * (KS.vx * no1 + KS.vy * 0 + KS.vz * no1))
							+ n9o2 * (Pxx * no1 + Pyy * 0 + Pzz * no1 + no2 * (Pxy * 0 + Pxz * no1 + Pyz * 0)) - n3o2 * (Pxx + Pyy + Pzz));
		KS.f[dir27::ppm] = n1o216
						 * (KS.rho * (n5o2 - n3o2 * 3 + no3 * (KS.vx * no1 + KS.vy * no1 + KS.vz * (-no1)))
							+ n9o2 * (Pxx * no1 + Pyy * no1 + Pzz * no1 + no2 * (Pxy * no1 + Pxz * -no1 + Pyz * (-no1))) - n3o2 * (Pxx + Pyy + Pzz));
		KS.f[dir27::ppz] = n1o54
						 * (KS.rho * (n5o2 - n3o2 * 2 + no3 * (KS.vx * no1 + KS.vy * no1 + KS.vz * 0))
							+ n9o2 * (Pxx * no1 + Pyy * no1 + Pzz * 0 + no2 * (Pxy * no1 + Pxz * 0 + Pyz * 0)) - n3o2 * (Pxx + Pyy + Pzz));
		KS.f[dir27::ppp] = n1o216
						 * (KS.rho * (n5o2 - n3o2 * 3 + no3 * (KS.vx * no1 + KS.vy * no1 + KS.vz * no1))
							+ n9o2 * (Pxx * no1 + Pyy * no1 + Pzz * no1 + no2 * (Pxy * no1 + Pxz * no1 + Pyz * no1)) - n3o2 * (Pxx + Pyy + Pzz));
	}
};
