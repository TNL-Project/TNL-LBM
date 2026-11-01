#pragma once

#include "common.h"
#include "eq.h"

// improved BGK (SRT) model by Geier 2015 https://doi.org/10.1016/j.camwa.2015.05.001
// Appendix A, page 530,
// for standard DF (not well-conditioned)

template <typename TRAITS, typename EQ = D3Q27_EQ<TRAITS>>
struct D3Q27_BGK : D3Q27_COMMON<TRAITS, EQ>
{
	using dreal = typename TRAITS::dreal;

	static constexpr const char* id = "BGK";

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

		const dreal Dxu = -omega1 * n1o2 * (no3 * m_200 / KS.rho - no1 - no3 * KS.vx * KS.vx);
		const dreal Dyv = -omega1 * n1o2 * (no3 * m_020 / KS.rho - no1 - no3 * KS.vy * KS.vy);
		const dreal Dzw = -omega1 * n1o2 * (no3 * m_002 / KS.rho - no1 - no3 * KS.vz * KS.vz);

		const dreal Gx = -no3 * KS.vx * KS.vx * Dxu * (no1 / omega1 - n1o2);
		const dreal Gy = -no3 * KS.vy * KS.vy * Dyv * (no1 / omega1 - n1o2);
		const dreal Gz = -no3 * KS.vz * KS.vz * Dzw * (no1 / omega1 - n1o2);

		const dreal Xz = n1o3 - no1 + KS.vx * KS.vx + Gx;
		const dreal Xp = -n1o2 * (Xz + no1 + KS.vx);
		const dreal Xm = Xp + KS.vx;

		const dreal Yz = n1o3 - no1 + KS.vy * KS.vy + Gy;
		const dreal Yp = -n1o2 * (Yz + no1 + KS.vy);
		const dreal Ym = Yp + KS.vy;

		const dreal Zz = n1o3 - no1 + KS.vz * KS.vz + Gz;
		const dreal Zp = -n1o2 * (Zz + no1 + KS.vz);
		const dreal Zm = Zp + KS.vz;
#else
		const dreal Xz = n1o3 - no1 + KS.vx * KS.vx;
		const dreal Xp = -n1o2 * (Xz + no1 + KS.vx);
		const dreal Xm = Xp + KS.vx;

		const dreal Yz = n1o3 - no1 + KS.vy * KS.vy;
		const dreal Yp = -n1o2 * (Yz + no1 + KS.vy);
		const dreal Ym = Yp + KS.vy;

		const dreal Zz = n1o3 - no1 + KS.vz * KS.vz;
		const dreal Zp = -n1o2 * (Zz + no1 + KS.vz);
		const dreal Zm = Zp + KS.vz;
#endif
		const dreal Smmm = (no3 * ((-KS.vx - no1) * KS.fx + (-KS.vy - no1) * KS.fy + (-KS.vz - no1) * KS.fz)) / KS.rho;
		const dreal Szmm = (no3 * (-KS.vx * KS.fx + (-KS.vy - no1) * KS.fy + (-KS.vz - no1) * KS.fz)) / KS.rho;
		const dreal Spmm = (no3 * ((-KS.vx + no1) * KS.fx + (-KS.vy - no1) * KS.fy + (-KS.vz - no1) * KS.fz)) / KS.rho;
		const dreal Smzm = (no3 * ((-KS.vx - no1) * KS.fx - KS.vy * KS.fy + (-KS.vz - no1) * KS.fz)) / KS.rho;
		const dreal Szzm = (no3 * (-KS.vx * KS.fx - KS.vy * KS.fy + (-KS.vz - no1) * KS.fz)) / KS.rho;
		const dreal Spzm = (no3 * ((-KS.vx + no1) * KS.fx - KS.vy * KS.fy + (-KS.vz - no1) * KS.fz)) / KS.rho;
		const dreal Smpm = (no3 * ((-KS.vx - no1) * KS.fx + (-KS.vy + no1) * KS.fy + (-KS.vz - no1) * KS.fz)) / KS.rho;
		const dreal Szpm = (no3 * (-KS.vx * KS.fx + (-KS.vy + no1) * KS.fy + (-KS.vz - no1) * KS.fz)) / KS.rho;
		const dreal Sppm = (no3 * ((-KS.vx + no1) * KS.fx + (-KS.vy + no1) * KS.fy + (-KS.vz - no1) * KS.fz)) / KS.rho;
		const dreal Smmz = (no3 * ((-KS.vx - no1) * KS.fx + (-KS.vy - no1) * KS.fy - KS.vz * KS.fz)) / KS.rho;
		const dreal Szmz = (no3 * (-KS.vx * KS.fx + (-KS.vy - no1) * KS.fy - KS.vz * KS.fz)) / KS.rho;
		const dreal Spmz = (no3 * ((-KS.vx + no1) * KS.fx + (-KS.vy - no1) * KS.fy - KS.vz * KS.fz)) / KS.rho;
		const dreal Smzz = (no3 * ((-KS.vx - no1) * KS.fx - KS.vy * KS.fy - KS.vz * KS.fz)) / KS.rho;
		const dreal Szzz = (no3 * (-KS.fx * KS.vx - KS.fy * KS.vy - KS.fz * KS.vz)) / KS.rho;
		const dreal Spzz = (no3 * ((-KS.vx + no1) * KS.fx - KS.vy * KS.fy - KS.vz * KS.fz)) / KS.rho;
		const dreal Smpz = (no3 * ((-KS.vx - no1) * KS.fx + (-KS.vy + no1) * KS.fy - KS.vz * KS.fz)) / KS.rho;
		const dreal Szpz = (no3 * (-KS.vx * KS.fx + (-KS.vy + no1) * KS.fy - KS.vz * KS.fz)) / KS.rho;
		const dreal Sppz = (no3 * ((-KS.vx + no1) * KS.fx + (-KS.vy + no1) * KS.fy - KS.vz * KS.fz)) / KS.rho;
		const dreal Smmp = (no3 * ((-KS.vx - no1) * KS.fx + (-KS.vy - no1) * KS.fy + (-KS.vz + no1) * KS.fz)) / KS.rho;
		const dreal Szmp = (no3 * (-KS.vx * KS.fx + (-KS.vy - no1) * KS.fy + (-KS.vz + no1) * KS.fz)) / KS.rho;
		const dreal Spmp = (no3 * ((-KS.vx + no1) * KS.fx + (-KS.vy - no1) * KS.fy + (-KS.vz + no1) * KS.fz)) / KS.rho;
		const dreal Smzp = (no3 * ((-KS.vx - no1) * KS.fx - KS.vy * KS.fy + (-KS.vz + no1) * KS.fz)) / KS.rho;
		const dreal Szzp = (no3 * (-KS.vx * KS.fx - KS.vy * KS.fy + (-KS.vz + no1) * KS.fz)) / KS.rho;
		const dreal Spzp = (no3 * ((-KS.vx + no1) * KS.fx - KS.vy * KS.fy + (-KS.vz + no1) * KS.fz)) / KS.rho;
		const dreal Smpp = (no3 * ((-KS.vx - no1) * KS.fx + (-KS.vy + no1) * KS.fy + (-KS.vz + no1) * KS.fz)) / KS.rho;
		const dreal Szpp = (no3 * (-KS.vx * KS.fx + (-KS.vy + no1) * KS.fy + (-KS.vz + no1) * KS.fz)) / KS.rho;
		const dreal Sppp = (no3 * ((-KS.vx + no1) * KS.fx + (-KS.vy + no1) * KS.fy + (-KS.vz + no1) * KS.fz)) / KS.rho;

		const dreal feq_mmm = -KS.rho * Xm * Ym * Zm;
		const dreal feq_mmz = -KS.rho * Xm * Ym * Zz;
		const dreal feq_mmp = -KS.rho * Xm * Ym * Zp;
		const dreal feq_mzm = -KS.rho * Xm * Yz * Zm;
		const dreal feq_mzz = -KS.rho * Xm * Yz * Zz;
		const dreal feq_mzp = -KS.rho * Xm * Yz * Zp;
		const dreal feq_mpm = -KS.rho * Xm * Yp * Zm;
		const dreal feq_mpz = -KS.rho * Xm * Yp * Zz;
		const dreal feq_mpp = -KS.rho * Xm * Yp * Zp;
		const dreal feq_zmm = -KS.rho * Xz * Ym * Zm;
		const dreal feq_zmz = -KS.rho * Xz * Ym * Zz;
		const dreal feq_zmp = -KS.rho * Xz * Ym * Zp;
		const dreal feq_zzm = -KS.rho * Xz * Yz * Zm;
		const dreal feq_zzz = -KS.rho * Xz * Yz * Zz;
		const dreal feq_zzp = -KS.rho * Xz * Yz * Zp;
		const dreal feq_zpm = -KS.rho * Xz * Yp * Zm;
		const dreal feq_zpz = -KS.rho * Xz * Yp * Zz;
		const dreal feq_zpp = -KS.rho * Xz * Yp * Zp;
		const dreal feq_pmm = -KS.rho * Xp * Ym * Zm;
		const dreal feq_pmz = -KS.rho * Xp * Ym * Zz;
		const dreal feq_pmp = -KS.rho * Xp * Ym * Zp;
		const dreal feq_pzm = -KS.rho * Xp * Yz * Zm;
		const dreal feq_pzz = -KS.rho * Xp * Yz * Zz;
		const dreal feq_pzp = -KS.rho * Xp * Yz * Zp;
		const dreal feq_ppm = -KS.rho * Xp * Yp * Zm;
		const dreal feq_ppz = -KS.rho * Xp * Yp * Zz;
		const dreal feq_ppp = -KS.rho * Xp * Yp * Zp;

		KS.f[dir27::mmm] += (feq_mmm - KS.f[dir27::mmm]) * omega1 + (no1 - n1o2 * omega1) * Smmm * feq_mmm;
		KS.f[dir27::mmz] += (feq_mmz - KS.f[dir27::mmz]) * omega1 + (no1 - n1o2 * omega1) * Smmz * feq_mmz;
		KS.f[dir27::mmp] += (feq_mmp - KS.f[dir27::mmp]) * omega1 + (no1 - n1o2 * omega1) * Smmp * feq_mmp;
		KS.f[dir27::mzm] += (feq_mzm - KS.f[dir27::mzm]) * omega1 + (no1 - n1o2 * omega1) * Smzm * feq_mzm;
		KS.f[dir27::mzz] += (feq_mzz - KS.f[dir27::mzz]) * omega1 + (no1 - n1o2 * omega1) * Smzz * feq_mzz;
		KS.f[dir27::mzp] += (feq_mzp - KS.f[dir27::mzp]) * omega1 + (no1 - n1o2 * omega1) * Smzp * feq_mzp;
		KS.f[dir27::mpm] += (feq_mpm - KS.f[dir27::mpm]) * omega1 + (no1 - n1o2 * omega1) * Smpm * feq_mpm;
		KS.f[dir27::mpz] += (feq_mpz - KS.f[dir27::mpz]) * omega1 + (no1 - n1o2 * omega1) * Smpz * feq_mpz;
		KS.f[dir27::mpp] += (feq_mpp - KS.f[dir27::mpp]) * omega1 + (no1 - n1o2 * omega1) * Smpp * feq_mpp;
		KS.f[dir27::zmm] += (feq_zmm - KS.f[dir27::zmm]) * omega1 + (no1 - n1o2 * omega1) * Szmm * feq_zmm;
		KS.f[dir27::zmz] += (feq_zmz - KS.f[dir27::zmz]) * omega1 + (no1 - n1o2 * omega1) * Szmz * feq_zmz;
		KS.f[dir27::zmp] += (feq_zmp - KS.f[dir27::zmp]) * omega1 + (no1 - n1o2 * omega1) * Szmp * feq_zmp;
		KS.f[dir27::zzm] += (feq_zzm - KS.f[dir27::zzm]) * omega1 + (no1 - n1o2 * omega1) * Szzm * feq_zzm;
		KS.f[dir27::zzz] += (feq_zzz - KS.f[dir27::zzz]) * omega1 + (no1 - n1o2 * omega1) * Szzz * feq_zzz;
		KS.f[dir27::zzp] += (feq_zzp - KS.f[dir27::zzp]) * omega1 + (no1 - n1o2 * omega1) * Szzp * feq_zzp;
		KS.f[dir27::zpm] += (feq_zpm - KS.f[dir27::zpm]) * omega1 + (no1 - n1o2 * omega1) * Szpm * feq_zpm;
		KS.f[dir27::zpz] += (feq_zpz - KS.f[dir27::zpz]) * omega1 + (no1 - n1o2 * omega1) * Szpz * feq_zpz;
		KS.f[dir27::zpp] += (feq_zpp - KS.f[dir27::zpp]) * omega1 + (no1 - n1o2 * omega1) * Szpp * feq_zpp;
		KS.f[dir27::pmm] += (feq_pmm - KS.f[dir27::pmm]) * omega1 + (no1 - n1o2 * omega1) * Spmm * feq_pmm;
		KS.f[dir27::pmz] += (feq_pmz - KS.f[dir27::pmz]) * omega1 + (no1 - n1o2 * omega1) * Spmz * feq_pmz;
		KS.f[dir27::pmp] += (feq_pmp - KS.f[dir27::pmp]) * omega1 + (no1 - n1o2 * omega1) * Spmp * feq_pmp;
		KS.f[dir27::pzm] += (feq_pzm - KS.f[dir27::pzm]) * omega1 + (no1 - n1o2 * omega1) * Spzm * feq_pzm;
		KS.f[dir27::pzz] += (feq_pzz - KS.f[dir27::pzz]) * omega1 + (no1 - n1o2 * omega1) * Spzz * feq_pzz;
		KS.f[dir27::pzp] += (feq_pzp - KS.f[dir27::pzp]) * omega1 + (no1 - n1o2 * omega1) * Spzp * feq_pzp;
		KS.f[dir27::ppm] += (feq_ppm - KS.f[dir27::ppm]) * omega1 + (no1 - n1o2 * omega1) * Sppm * feq_ppm;
		KS.f[dir27::ppz] += (feq_ppz - KS.f[dir27::ppz]) * omega1 + (no1 - n1o2 * omega1) * Sppz * feq_ppz;
		KS.f[dir27::ppp] += (feq_ppp - KS.f[dir27::ppp]) * omega1 + (no1 - n1o2 * omega1) * Sppp * feq_ppp;
	}
};
