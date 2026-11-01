#pragma once

#include "common.h"
#include "eq_well.h"

// improved BGK (SRT) model by Geier 2015 https://doi.org/10.1016/j.camwa.2015.05.001
// Appendix A, page 530 + Appendix J, page 544
// well conditioned

template <typename TRAITS, typename LBM_EQ = D3Q27_EQ_WELL<TRAITS>>
struct D3Q27_BGK_WELL : D3Q27_COMMON_WELL<TRAITS, LBM_EQ>
{
	using dreal = typename TRAITS::dreal;

	static constexpr const char* id = "BGK_WELL";

	template <typename LBM_KS>
	__cuda_callable__ static void collision(LBM_KS& KS)
	{
		const dreal omega1 = no1 / (no3 * KS.lbmViscosity + n1o2);

#ifdef USE_GALILEAN_CORRECTION
		const dreal m_200 = KS.f[dir27::mmm] + KS.f[dir27::mmp] + KS.f[dir27::mmz] + KS.f[dir27::mpm] + KS.f[dir27::mpp] + KS.f[dir27::mpz]
						  + KS.f[dir27::mzm] + KS.f[dir27::mzp] + KS.f[dir27::mzz] + KS.f[dir27::pmm] + KS.f[dir27::pmp] + KS.f[dir27::pmz]
						  + KS.f[dir27::ppm] + KS.f[dir27::ppp] + KS.f[dir27::ppz] + KS.f[dir27::pzm] + KS.f[dir27::pzp] + KS.f[dir27::pzz];
		const dreal m_020 = KS.f[dir27::mmm] + KS.f[dir27::mmp] + KS.f[dir27::mmz] + KS.f[dir27::mpm] + KS.f[dir27::mpp] + KS.f[dir27::mpz]
						  + KS.f[dir27::zmm] + KS.f[dir27::zmp] + KS.f[dir27::zmz] + KS.f[dir27::zpm] + KS.f[dir27::zpp] + KS.f[dir27::zpz]
						  + KS.f[dir27::pmm] + KS.f[dir27::pmp] + KS.f[dir27::pmz] + KS.f[dir27::ppm] + KS.f[dir27::ppp] + KS.f[dir27::ppz];
		const dreal m_002 = KS.f[dir27::mmm] + KS.f[dir27::mmp] + KS.f[dir27::mzm] + KS.f[dir27::mzp] + KS.f[dir27::mpm] + KS.f[dir27::mpp]
						  + KS.f[dir27::zmm] + KS.f[dir27::zmp] + KS.f[dir27::zzm] + KS.f[dir27::zzp] + KS.f[dir27::zpm] + KS.f[dir27::zpp]
						  + KS.f[dir27::pmm] + KS.f[dir27::pmp] + KS.f[dir27::pzm] + KS.f[dir27::pzp] + KS.f[dir27::ppm] + KS.f[dir27::ppp];
		const dreal drho = ((((KS.f[dir27::ppp] + KS.f[dir27::mmm]) + (KS.f[dir27::pmp] + KS.f[dir27::mpm]))
							 + ((KS.f[dir27::ppm] + KS.f[dir27::mmp]) + (KS.f[dir27::mpp] + KS.f[dir27::pmm])))
							+ (((KS.f[dir27::zpp] + KS.f[dir27::zmm]) + (KS.f[dir27::zpm] + KS.f[dir27::zmp]))
							   + ((KS.f[dir27::pzp] + KS.f[dir27::mzm]) + (KS.f[dir27::pzm] + KS.f[dir27::mzp]))
							   + ((KS.f[dir27::ppz] + KS.f[dir27::mmz]) + (KS.f[dir27::pmz] + KS.f[dir27::mpz])))
							+ ((KS.f[dir27::pzz] + KS.f[dir27::mzz]) + (KS.f[dir27::zpz] + KS.f[dir27::zmz]) + (KS.f[dir27::zzp] + KS.f[dir27::zzm])))
						 + KS.f[dir27::zzz];

		const dreal Dxu = -omega1 * n1o2 * (no3 * m_200 / KS.rho - drho / KS.rho - no3 * KS.vx * KS.vx);
		const dreal Dyv = -omega1 * n1o2 * (no3 * m_020 / KS.rho - drho / KS.rho - no3 * KS.vy * KS.vy);
		const dreal Dzw = -omega1 * n1o2 * (no3 * m_002 / KS.rho - drho / KS.rho - no3 * KS.vz * KS.vz);

		const dreal Gx = -no3 * KS.vx * KS.vx * Dxu * (no1 / omega1 - n1o2);
		const dreal Gy = -no3 * KS.vy * KS.vy * Dyv * (no1 / omega1 - n1o2);
		const dreal Gz = -no3 * KS.vz * KS.vz * Dzw * (no1 / omega1 - n1o2);

		const dreal Xz = -n2o3 + KS.vx * KS.vx + Gx;
		const dreal Xp = -n1o2 * (Xz + no1 + KS.vx);
		const dreal Xm = Xp + KS.vx;

		const dreal Yz = -n2o3 + KS.vy * KS.vy + Gy;
		const dreal Yp = -n1o2 * (Yz + no1 + KS.vy);
		const dreal Ym = Yp + KS.vy;

		const dreal Zz = -n2o3 + KS.vz * KS.vz + Gz;
		const dreal Zp = -n1o2 * (Zz + no1 + KS.vz);
		const dreal Zm = Zp + KS.vz;
#else
		const dreal Xz = -n2o3 + KS.vx * KS.vx;
		const dreal Xp = -n1o2 * (Xz + no1 + KS.vx);
		const dreal Xm = Xp + KS.vx;

		const dreal Yz = -n2o3 + KS.vy * KS.vy;
		const dreal Yp = -n1o2 * (Yz + no1 + KS.vy);
		const dreal Ym = Yp + KS.vy;

		const dreal Zz = -n2o3 + KS.vz * KS.vz;
		const dreal Zp = -n1o2 * (Zz + no1 + KS.vz);
		const dreal Zm = Zp + KS.vz;
#endif
		const dreal Smmm = (no3 * ((-KS.vx - no1) * KS.fx + (-KS.vy - no1) * KS.fy + (-KS.vz - no1) * KS.fz));
		const dreal Szmm = (no3 * (-KS.vx * KS.fx + (-KS.vy - no1) * KS.fy + (-KS.vz - no1) * KS.fz));
		const dreal Spmm = (no3 * ((-KS.vx + no1) * KS.fx + (-KS.vy - no1) * KS.fy + (-KS.vz - no1) * KS.fz));
		const dreal Smzm = (no3 * ((-KS.vx - no1) * KS.fx - KS.vy * KS.fy + (-KS.vz - no1) * KS.fz));
		const dreal Szzm = (no3 * (-KS.vx * KS.fx - KS.vy * KS.fy + (-KS.vz - no1) * KS.fz));
		const dreal Spzm = (no3 * ((-KS.vx + no1) * KS.fx - KS.vy * KS.fy + (-KS.vz - no1) * KS.fz));
		const dreal Smpm = (no3 * ((-KS.vx - no1) * KS.fx + (-KS.vy + no1) * KS.fy + (-KS.vz - no1) * KS.fz));
		const dreal Szpm = (no3 * (-KS.vx * KS.fx + (-KS.vy + no1) * KS.fy + (-KS.vz - no1) * KS.fz));
		const dreal Sppm = (no3 * ((-KS.vx + no1) * KS.fx + (-KS.vy + no1) * KS.fy + (-KS.vz - no1) * KS.fz));
		const dreal Smmz = (no3 * ((-KS.vx - no1) * KS.fx + (-KS.vy - no1) * KS.fy - KS.vz * KS.fz));
		const dreal Szmz = (no3 * (-KS.vx * KS.fx + (-KS.vy - no1) * KS.fy - KS.vz * KS.fz));
		const dreal Spmz = (no3 * ((-KS.vx + no1) * KS.fx + (-KS.vy - no1) * KS.fy - KS.vz * KS.fz));
		const dreal Smzz = (no3 * ((-KS.vx - no1) * KS.fx - KS.vy * KS.fy - KS.vz * KS.fz));
		const dreal Szzz = (no3 * (-KS.fx * KS.vx - KS.fy * KS.vy - KS.fz * KS.vz));
		const dreal Spzz = (no3 * ((-KS.vx + no1) * KS.fx - KS.vy * KS.fy - KS.vz * KS.fz));
		const dreal Smpz = (no3 * ((-KS.vx - no1) * KS.fx + (-KS.vy + no1) * KS.fy - KS.vz * KS.fz));
		const dreal Szpz = (no3 * (-KS.vx * KS.fx + (-KS.vy + no1) * KS.fy - KS.vz * KS.fz));
		const dreal Sppz = (no3 * ((-KS.vx + no1) * KS.fx + (-KS.vy + no1) * KS.fy - KS.vz * KS.fz));
		const dreal Smmp = (no3 * ((-KS.vx - no1) * KS.fx + (-KS.vy - no1) * KS.fy + (-KS.vz + no1) * KS.fz));
		const dreal Szmp = (no3 * (-KS.vx * KS.fx + (-KS.vy - no1) * KS.fy + (-KS.vz + no1) * KS.fz));
		const dreal Spmp = (no3 * ((-KS.vx + no1) * KS.fx + (-KS.vy - no1) * KS.fy + (-KS.vz + no1) * KS.fz));
		const dreal Smzp = (no3 * ((-KS.vx - no1) * KS.fx - KS.vy * KS.fy + (-KS.vz + no1) * KS.fz));
		const dreal Szzp = (no3 * (-KS.vx * KS.fx - KS.vy * KS.fy + (-KS.vz + no1) * KS.fz));
		const dreal Spzp = (no3 * ((-KS.vx + no1) * KS.fx - KS.vy * KS.fy + (-KS.vz + no1) * KS.fz));
		const dreal Smpp = (no3 * ((-KS.vx - no1) * KS.fx + (-KS.vy + no1) * KS.fy + (-KS.vz + no1) * KS.fz));
		const dreal Szpp = (no3 * (-KS.vx * KS.fx + (-KS.vy + no1) * KS.fy + (-KS.vz + no1) * KS.fz));
		const dreal Sppp = (no3 * ((-KS.vx + no1) * KS.fx + (-KS.vy + no1) * KS.fy + (-KS.vz + no1) * KS.fz));

		KS.f[dir27::mmm] =
			(no1 - omega1) * KS.f[dir27::mmm] + omega1 * (-KS.rho * Xm * Ym * Zm - n1o216) - (no1 - n1o2 * omega1) * Smmm * Xm * Ym * Zm;
		KS.f[dir27::mmz] =
			(no1 - omega1) * KS.f[dir27::mmz] + omega1 * (-KS.rho * Xm * Ym * Zz - n1o54) - (no1 - n1o2 * omega1) * Smmz * Xm * Ym * Zz;
		KS.f[dir27::mmp] =
			(no1 - omega1) * KS.f[dir27::mmp] + omega1 * (-KS.rho * Xm * Ym * Zp - n1o216) - (no1 - n1o2 * omega1) * Smmp * Xm * Ym * Zp;
		KS.f[dir27::mzm] =
			(no1 - omega1) * KS.f[dir27::mzm] + omega1 * (-KS.rho * Xm * Yz * Zm - n1o54) - (no1 - n1o2 * omega1) * Smzm * Xm * Yz * Zm;
		KS.f[dir27::mzz] =
			(no1 - omega1) * KS.f[dir27::mzz] + omega1 * (-KS.rho * Xm * Yz * Zz - n2o27) - (no1 - n1o2 * omega1) * Smzz * Xm * Yz * Zz;
		KS.f[dir27::mzp] =
			(no1 - omega1) * KS.f[dir27::mzp] + omega1 * (-KS.rho * Xm * Yz * Zp - n1o54) - (no1 - n1o2 * omega1) * Smzp * Xm * Yz * Zp;
		KS.f[dir27::mpm] =
			(no1 - omega1) * KS.f[dir27::mpm] + omega1 * (-KS.rho * Xm * Yp * Zm - n1o216) - (no1 - n1o2 * omega1) * Smpm * Xm * Yp * Zm;
		KS.f[dir27::mpz] =
			(no1 - omega1) * KS.f[dir27::mpz] + omega1 * (-KS.rho * Xm * Yp * Zz - n1o54) - (no1 - n1o2 * omega1) * Smpz * Xm * Yp * Zz;
		KS.f[dir27::mpp] =
			(no1 - omega1) * KS.f[dir27::mpp] + omega1 * (-KS.rho * Xm * Yp * Zp - n1o216) - (no1 - n1o2 * omega1) * Smpp * Xm * Yp * Zp;
		KS.f[dir27::zmm] =
			(no1 - omega1) * KS.f[dir27::zmm] + omega1 * (-KS.rho * Xz * Ym * Zm - n1o54) - (no1 - n1o2 * omega1) * Szmm * Xz * Ym * Zm;
		KS.f[dir27::zmz] =
			(no1 - omega1) * KS.f[dir27::zmz] + omega1 * (-KS.rho * Xz * Ym * Zz - n2o27) - (no1 - n1o2 * omega1) * Szmz * Xz * Ym * Zz;
		KS.f[dir27::zmp] =
			(no1 - omega1) * KS.f[dir27::zmp] + omega1 * (-KS.rho * Xz * Ym * Zp - n1o54) - (no1 - n1o2 * omega1) * Szmp * Xz * Ym * Zp;
		KS.f[dir27::zzm] =
			(no1 - omega1) * KS.f[dir27::zzm] + omega1 * (-KS.rho * Xz * Yz * Zm - n2o27) - (no1 - n1o2 * omega1) * Szzm * Xz * Yz * Zm;
		KS.f[dir27::zzz] =
			(no1 - omega1) * KS.f[dir27::zzz] + omega1 * (-KS.rho * Xz * Yz * Zz - n8o27) - (no1 - n1o2 * omega1) * Szzz * Xz * Yz * Zz;
		KS.f[dir27::zzp] =
			(no1 - omega1) * KS.f[dir27::zzp] + omega1 * (-KS.rho * Xz * Yz * Zp - n2o27) - (no1 - n1o2 * omega1) * Szzp * Xz * Yz * Zp;
		KS.f[dir27::zpm] =
			(no1 - omega1) * KS.f[dir27::zpm] + omega1 * (-KS.rho * Xz * Yp * Zm - n1o54) - (no1 - n1o2 * omega1) * Szpm * Xz * Yp * Zm;
		KS.f[dir27::zpz] =
			(no1 - omega1) * KS.f[dir27::zpz] + omega1 * (-KS.rho * Xz * Yp * Zz - n2o27) - (no1 - n1o2 * omega1) * Szpz * Xz * Yp * Zz;
		KS.f[dir27::zpp] =
			(no1 - omega1) * KS.f[dir27::zpp] + omega1 * (-KS.rho * Xz * Yp * Zp - n1o54) - (no1 - n1o2 * omega1) * Szpp * Xz * Yp * Zp;
		KS.f[dir27::pmm] =
			(no1 - omega1) * KS.f[dir27::pmm] + omega1 * (-KS.rho * Xp * Ym * Zm - n1o216) - (no1 - n1o2 * omega1) * Spmm * Xp * Ym * Zm;
		KS.f[dir27::pmz] =
			(no1 - omega1) * KS.f[dir27::pmz] + omega1 * (-KS.rho * Xp * Ym * Zz - n1o54) - (no1 - n1o2 * omega1) * Spmz * Xp * Ym * Zz;
		KS.f[dir27::pmp] =
			(no1 - omega1) * KS.f[dir27::pmp] + omega1 * (-KS.rho * Xp * Ym * Zp - n1o216) - (no1 - n1o2 * omega1) * Spmp * Xp * Ym * Zp;
		KS.f[dir27::pzm] =
			(no1 - omega1) * KS.f[dir27::pzm] + omega1 * (-KS.rho * Xp * Yz * Zm - n1o54) - (no1 - n1o2 * omega1) * Spzm * Xp * Yz * Zm;
		KS.f[dir27::pzz] =
			(no1 - omega1) * KS.f[dir27::pzz] + omega1 * (-KS.rho * Xp * Yz * Zz - n2o27) - (no1 - n1o2 * omega1) * Spzz * Xp * Yz * Zz;
		KS.f[dir27::pzp] =
			(no1 - omega1) * KS.f[dir27::pzp] + omega1 * (-KS.rho * Xp * Yz * Zp - n1o54) - (no1 - n1o2 * omega1) * Spzp * Xp * Yz * Zp;
		KS.f[dir27::ppm] =
			(no1 - omega1) * KS.f[dir27::ppm] + omega1 * (-KS.rho * Xp * Yp * Zm - n1o216) - (no1 - n1o2 * omega1) * Sppm * Xp * Yp * Zm;
		KS.f[dir27::ppz] =
			(no1 - omega1) * KS.f[dir27::ppz] + omega1 * (-KS.rho * Xp * Yp * Zz - n1o54) - (no1 - n1o2 * omega1) * Sppz * Xp * Yp * Zz;
		KS.f[dir27::ppp] =
			(no1 - omega1) * KS.f[dir27::ppp] + omega1 * (-KS.rho * Xp * Yp * Zp - n1o216) - (no1 - n1o2 * omega1) * Sppp * Xp * Yp * Zp;
	}
};
