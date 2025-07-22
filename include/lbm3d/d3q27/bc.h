#pragma once

#include "lbm3d/defs.h"
#include "lbm_common/ciselnik.h"

template <typename CONFIG>
struct D3Q27_BC_All
{
	using COLL = typename CONFIG::COLL;
	using STREAMING = typename CONFIG::STREAMING;
	using DATA = typename CONFIG::DATA;

	using map_t = typename CONFIG::TRAITS::map_t;
	using idx = typename CONFIG::TRAITS::idx;
	using dreal = typename CONFIG::TRAITS::dreal;

	enum GEO : map_t
	{
		GEO_FLUID,	// compulsory
		GEO_WALL,	// compulsory
		GEO_INFLOW,
		GEO_INFLOW_LEFT,
		GEO_INFLOW_BB_LEFT,
		GEO_INFLOW_EQ_LEFT,
		GEO_OUTFLOW_EQ,
		GEO_OUTFLOW_RIGHT,
		GEO_OUTFLOW_RIGHT_INTERP,
		GEO_PERIODIC,
		GEO_NOTHING,
		GEO_SYM_TOP,
		GEO_SYM_BOTTOM,
		GEO_SYM_LEFT,
		GEO_SYM_RIGHT,
		GEO_SYM_BACK,
		GEO_SYM_FRONT,

		// Adjoint boundary conditions
		GEO_ADJOINT_FLUID,
		GEO_ADJOINT_FLUID_m,
		GEO_ADJOINT_WALL,
		GEO_ADJOINT_INFLOW_BB_LEFT,
		GEO_ADJOINT_OUTFLOW_RIGHT
	};

	__cuda_callable__ static bool isPeriodic(map_t mapgi)
	{
		return mapgi == GEO_PERIODIC;
	}

	__cuda_callable__ static bool isFluid(map_t mapgi)
	{
		return mapgi == GEO_FLUID;
	}

	__cuda_callable__ static bool isWall(map_t mapgi)
	{
		return mapgi == GEO_WALL;
	}

	template <typename LBM_KS>
	__cuda_callable__ static void preCollision(DATA& SD, LBM_KS& KS, map_t mapgi, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm, idx z, idx zp)
	{
		if (mapgi == GEO_NOTHING) {
			// does not affect the computation, only the output
			KS.rho = 1;
			KS.vx = 0;
			KS.vy = 0;
			KS.vz = 0;
			return;
		}

		// modify pull location for streaming
		if (mapgi == GEO_OUTFLOW_RIGHT || mapgi == GEO_ADJOINT_OUTFLOW_RIGHT)
			xp = x = xm;

		if (mapgi == GEO_ADJOINT_FLUID || mapgi == GEO_ADJOINT_FLUID_m || mapgi == GEO_ADJOINT_WALL || mapgi == GEO_ADJOINT_INFLOW_BB_LEFT
			|| mapgi == GEO_ADJOINT_OUTFLOW_RIGHT)
		{
			STREAMING::streamingAdjoint(SD, KS, xm, x, xp, ym, y, yp, zm, z, zp);
		}
		else if (mapgi != GEO_OUTFLOW_RIGHT_INTERP)
			STREAMING::streaming(SD, KS, xm, x, xp, ym, y, yp, zm, z, zp);

		// boundary conditions
		switch (mapgi) {
			case GEO_INFLOW:
				SD.inflow(KS, x, y, z);
				KS.rho = 1;
				COLL::setEquilibrium(KS);
				break;
			case GEO_INFLOW_LEFT:
				{
					SD.inflow(KS, x, y, z);
					// moment boundary condition by Pavel Eichler https://doi.org/10.1016/j.camwa.2024.08.009
					// expressions symetrized by Jakub Klinkovsky
					// clang-format off
					KS.rho = (dreal)1.0/(1-KS.vx) * (
						(
							KS.f[zzz] + (
								+ ((KS.f[zpp] + KS.f[zmm]) + (KS.f[zpm] + KS.f[zmp]))
								+ ((KS.f[zpz] + KS.f[zmz]) + (KS.f[zzp] + KS.f[zzm]))
							)
						)
						+ 2*(
							KS.f[mzz] + (
								+ ((KS.f[mpp] + KS.f[mmm]) + (KS.f[mpm] + KS.f[mmp]))
								+ ((KS.f[mpz] + KS.f[mmz]) + (KS.f[mzp] + KS.f[mzm]))
							)
						)
					);
					// clang-format on
					dreal m100 = KS.rho * KS.vx;
					dreal m010 = KS.rho * KS.vy;
					dreal m001 = KS.rho * KS.vz;
					dreal m011 = KS.rho * (KS.vy * KS.vz);
					dreal m020 = n1o3 * KS.rho + KS.rho * (KS.vy * KS.vy);
					dreal m002 = n1o3 * KS.rho + KS.rho * (KS.vz * KS.vz);
					dreal m021 = n1o3 * KS.rho * KS.vz + KS.rho * ((KS.vy * KS.vy) * KS.vz);
					dreal m012 = n1o3 * KS.rho * KS.vy + KS.rho * (KS.vy * (KS.vz * KS.vz));
					dreal m022 = n1o9 * KS.rho + n1o3 * KS.rho * (KS.vy * KS.vy + KS.vz * KS.vz) + KS.rho * (KS.vy * KS.vy) * (KS.vz * KS.vz);
					// clang-format off
					KS.f[pzz] = m100 + (m022 - (m020 + m002))
						+ KS.f[mzz]
						+ (
							+ ((KS.f[zpp] + KS.f[zmm]) + (KS.f[zpm] + KS.f[zmp]))
							+ ((KS.f[zzp] + KS.f[zzm]) + (KS.f[zpz] + KS.f[zmz]))
						)
						+ 2*(
							+ ((KS.f[mpp] + KS.f[mmm]) + (KS.f[mpm] + KS.f[mmp]))
							+ ((KS.f[mpz] + KS.f[mmz]) + (KS.f[mzp] + KS.f[mzm]))
						);
					// clang-format on
					KS.f[ppz] = (dreal) 0.5 * ((m020 - m022) + (-m012 + m010)) - (KS.f[mpz] + KS.f[zpz]);
					KS.f[pmz] = (dreal) 0.5 * ((m020 - m022) + (m012 - m010)) - (KS.f[mmz] + KS.f[zmz]);
					KS.f[pzp] = (dreal) 0.5 * ((m002 - m022) + (-m021 + m001)) - (KS.f[mzp] + KS.f[zzp]);
					KS.f[pzm] = (dreal) 0.5 * ((m002 - m022) + (m021 - m001)) - (KS.f[mzm] + KS.f[zzm]);
					KS.f[ppp] = (dreal) 0.25 * ((m022 + m011) + (m021 + m012)) - (KS.f[mpp] + KS.f[zpp]);
					KS.f[ppm] = (dreal) 0.25 * ((m022 - m011) + (-m021 + m012)) - (KS.f[mpm] + KS.f[zpm]);
					KS.f[pmp] = (dreal) 0.25 * ((m022 - m011) + (m021 - m012)) - (KS.f[mmp] + KS.f[zmp]);
					KS.f[pmm] = (dreal) 0.25 * ((m022 + m011) + (-m021 - m012)) - (KS.f[mmm] + KS.f[zmm]);
					break;
				}
			case GEO_INFLOW_BB_LEFT:
				SD.inflow(KS, x, y, z);
				// KS.rho = TNL::Backend::ldg(SD.macro(CONFIG::MACRO::e_rho,xp,y,z));
				/* KS.f[pzz] = KS.f[mzz] - no2 * n2o27  * KS.rho * no3 * ( - KS.vx );
				KS.f[ppz] = KS.f[mmz] - no2 * n1o54  * KS.rho * no3 * ( - KS.vx - KS.vy );
				KS.f[pmz] = KS.f[mpz] - no2 * n1o54  * KS.rho * no3 * ( - KS.vx + KS.vy );
				KS.f[pzp] = KS.f[mzm] - no2 * n1o54  * KS.rho * no3 * ( - KS.vx         - KS.vz );
				KS.f[pzm] = KS.f[mzp] - no2 * n1o54  * KS.rho * no3 * ( - KS.vx         + KS.vz );
				KS.f[ppp] = KS.f[mmm] - no2 * n1o216 * KS.rho * no3 * ( - KS.vx - KS.vy - KS.vz );
				KS.f[ppm] = KS.f[mmp] - no2 * n1o216 * KS.rho * no3 * ( - KS.vx - KS.vy + KS.vz );
				KS.f[pmp] = KS.f[mpm] - no2 * n1o216 * KS.rho * no3 * ( - KS.vx + KS.vy - KS.vz );
				KS.f[pmm] = KS.f[mpp] - no2 * n1o216 * KS.rho * no3 * ( - KS.vx + KS.vy + KS.vz ); */
				KS.f[pzz] = TNL::Backend::ldg(SD.df(df_cur, mzz, x, y, z)) - no2 * n2o27 * KS.rho * no3 * (-KS.vx);
				KS.f[ppz] = TNL::Backend::ldg(SD.df(df_cur, mmz, x, y, z)) - no2 * n1o54 * KS.rho * no3 * (-KS.vx - KS.vy);
				KS.f[pmz] = TNL::Backend::ldg(SD.df(df_cur, mpz, x, y, z)) - no2 * n1o54 * KS.rho * no3 * (-KS.vx + KS.vy);
				KS.f[pzp] = TNL::Backend::ldg(SD.df(df_cur, mzm, x, y, z)) - no2 * n1o54 * KS.rho * no3 * (-KS.vx - KS.vz);
				KS.f[pzm] = TNL::Backend::ldg(SD.df(df_cur, mzp, x, y, z)) - no2 * n1o54 * KS.rho * no3 * (-KS.vx + KS.vz);
				KS.f[ppp] = TNL::Backend::ldg(SD.df(df_cur, mmm, x, y, z)) - no2 * n1o216 * KS.rho * no3 * (-KS.vx - KS.vy - KS.vz);
				KS.f[ppm] = TNL::Backend::ldg(SD.df(df_cur, mmp, x, y, z)) - no2 * n1o216 * KS.rho * no3 * (-KS.vx - KS.vy + KS.vz);
				KS.f[pmp] = TNL::Backend::ldg(SD.df(df_cur, mpm, x, y, z)) - no2 * n1o216 * KS.rho * no3 * (-KS.vx + KS.vy - KS.vz);
				KS.f[pmm] = TNL::Backend::ldg(SD.df(df_cur, mpp, x, y, z)) - no2 * n1o216 * KS.rho * no3 * (-KS.vx + KS.vy + KS.vz);
				COLL::computeDensityAndVelocity(KS);
				break;
			case GEO_INFLOW_EQ_LEFT:
				SD.inflow(KS, x, y, z);
				// clang-format off
				KS.rho = (dreal)1.0/(1-KS.vx) * (
					(
						KS.f[zzz] + (
							+ ((KS.f[zpp] + KS.f[zmm]) + (KS.f[zpm] + KS.f[zmp]))
							+ ((KS.f[zpz] + KS.f[zmz]) + (KS.f[zzp] + KS.f[zzm]))
						)
					)
					+ 2*(
						KS.f[mzz] + (
							+ ((KS.f[mpp] + KS.f[mmm]) + (KS.f[mpm] + KS.f[mmp]))
							+ ((KS.f[mpz] + KS.f[mmz]) + (KS.f[mzp] + KS.f[mzm]))
						)
					)
				);
				// clang-format on
				COLL::setEquilibrium(KS);
				break;
			case GEO_OUTFLOW_EQ:
				COLL::computeDensityAndVelocity(KS);
				KS.rho = 1;
				COLL::setEquilibrium(KS);
				break;
			case GEO_OUTFLOW_RIGHT:
				COLL::computeDensityAndVelocity(KS);
				KS.rho = 1;
				break;
			case GEO_OUTFLOW_RIGHT_INTERP:
				STREAMING::streamingInterpRight(SD, KS, xm, x, xp, ym, y, yp, zm, z, zp);
				COLL::computeDensityAndVelocity(KS);
				COLL::setEquilibriumDecomposition(KS, 1);
				KS.rho = 1;
				break;
			case GEO_WALL:
				// does not affect the computation, only the output
				KS.rho = 1;
				KS.vx = 0;
				KS.vy = 0;
				KS.vz = 0;
				// collision step: bounce-back
				TNL::swap(KS.f[mmm], KS.f[ppp]);
				TNL::swap(KS.f[mmz], KS.f[ppz]);
				TNL::swap(KS.f[mmp], KS.f[ppm]);
				TNL::swap(KS.f[mzm], KS.f[pzp]);
				TNL::swap(KS.f[mzz], KS.f[pzz]);
				TNL::swap(KS.f[mzp], KS.f[pzm]);
				TNL::swap(KS.f[mpm], KS.f[pmp]);
				TNL::swap(KS.f[mpz], KS.f[pmz]);
				TNL::swap(KS.f[mpp], KS.f[pmm]);
				TNL::swap(KS.f[zmm], KS.f[zpp]);
				TNL::swap(KS.f[zzm], KS.f[zzp]);
				TNL::swap(KS.f[zmz], KS.f[zpz]);
				TNL::swap(KS.f[zmp], KS.f[zpm]);
				break;
			case GEO_SYM_TOP:
				KS.f[mmm] = KS.f[mmp];
				KS.f[mzm] = KS.f[mzp];
				KS.f[mpm] = KS.f[mpp];
				KS.f[zmm] = KS.f[zmp];
				KS.f[zzm] = KS.f[zzp];
				KS.f[zpm] = KS.f[zpp];
				KS.f[pmm] = KS.f[pmp];
				KS.f[pzm] = KS.f[pzp];
				KS.f[ppm] = KS.f[ppp];
				COLL::computeDensityAndVelocity(KS);
				break;
			case GEO_SYM_BOTTOM:
				KS.f[mmp] = KS.f[mmm];
				KS.f[mzp] = KS.f[mzm];
				KS.f[mpp] = KS.f[mpm];
				KS.f[zmp] = KS.f[zmm];
				KS.f[zzp] = KS.f[zzm];
				KS.f[zpp] = KS.f[zpm];
				KS.f[pmp] = KS.f[pmm];
				KS.f[pzp] = KS.f[pzm];
				KS.f[ppp] = KS.f[ppm];
				COLL::computeDensityAndVelocity(KS);
				break;
			case GEO_SYM_LEFT:
				KS.f[pmm] = KS.f[mmm];
				KS.f[pmz] = KS.f[mmz];
				KS.f[pmp] = KS.f[mmp];
				KS.f[pzm] = KS.f[mzm];
				KS.f[pzz] = KS.f[mzz];
				KS.f[pzp] = KS.f[mzp];
				KS.f[ppm] = KS.f[mpm];
				KS.f[ppz] = KS.f[mpz];
				KS.f[ppp] = KS.f[mpp];
				COLL::computeDensityAndVelocity(KS);
				break;
			case GEO_SYM_RIGHT:
				KS.f[mmm] = KS.f[pmm];
				KS.f[mmz] = KS.f[pmz];
				KS.f[mmp] = KS.f[pmp];
				KS.f[mzm] = KS.f[pzm];
				KS.f[mzz] = KS.f[pzz];
				KS.f[mzp] = KS.f[pzp];
				KS.f[mpm] = KS.f[ppm];
				KS.f[mpz] = KS.f[ppz];
				KS.f[mpp] = KS.f[ppp];
				COLL::computeDensityAndVelocity(KS);
				break;
			case GEO_SYM_BACK:
				KS.f[mpm] = KS.f[mmm];
				KS.f[mpz] = KS.f[mmz];
				KS.f[mpp] = KS.f[mmp];
				KS.f[zpm] = KS.f[zmm];
				KS.f[zpz] = KS.f[zmz];
				KS.f[zpp] = KS.f[zmp];
				KS.f[ppm] = KS.f[pmm];
				KS.f[ppz] = KS.f[pmz];
				KS.f[ppp] = KS.f[pmp];
				COLL::computeDensityAndVelocity(KS);
				break;
			case GEO_SYM_FRONT:
				KS.f[mmm] = KS.f[mpm];
				KS.f[mmz] = KS.f[mpz];
				KS.f[mmp] = KS.f[mpp];
				KS.f[zmm] = KS.f[zpm];
				KS.f[zmz] = KS.f[zpz];
				KS.f[zmp] = KS.f[zpp];
				KS.f[pmm] = KS.f[ppm];
				KS.f[pmz] = KS.f[ppz];
				KS.f[pmp] = KS.f[ppp];
				COLL::computeDensityAndVelocity(KS);
				break;

			// Adjoint boundary conditions
			case GEO_ADJOINT_FLUID:
				COLL::collision(KS);
				break;
			case GEO_ADJOINT_FLUID_m:
				COLL::collision(KS);
				COLL::setEquilibrium(KS);  // adds measured data
				break;
			case GEO_ADJOINT_WALL:	// works same as GEO_WALL --- only streaming step is different in adjoint
				// collision step: bounce-back
				TNL::swap(KS.f[mmm], KS.f[ppp]);
				TNL::swap(KS.f[mmz], KS.f[ppz]);
				TNL::swap(KS.f[mmp], KS.f[ppm]);
				TNL::swap(KS.f[mzm], KS.f[pzp]);
				TNL::swap(KS.f[mzz], KS.f[pzz]);
				TNL::swap(KS.f[mzp], KS.f[pzm]);
				TNL::swap(KS.f[mpm], KS.f[pmp]);
				TNL::swap(KS.f[mpz], KS.f[pmz]);
				TNL::swap(KS.f[mpp], KS.f[pmm]);
				TNL::swap(KS.f[zmm], KS.f[zpp]);
				TNL::swap(KS.f[zzm], KS.f[zzp]);
				TNL::swap(KS.f[zmz], KS.f[zpz]);
				TNL::swap(KS.f[zmp], KS.f[zpm]);
				break;
			case GEO_ADJOINT_INFLOW_BB_LEFT:
				{
					/* KS.f[mzz] = KS.f[pzz];
					KS.f[mpz] = KS.f[pmz];
					KS.f[mmz] = KS.f[ppz];
					KS.f[mzp] = KS.f[pzm];
					KS.f[mzm] = KS.f[pzp];
					KS.f[mpp] = KS.f[pmm];
					KS.f[mpm] = KS.f[pmp];
					KS.f[mmp] = KS.f[ppm];
					KS.f[mmm] = KS.f[ppp]; */
					KS.f[mzz] = TNL::Backend::ldg(SD.df(df_cur, pzz, x, y, z));
					KS.f[mpz] = TNL::Backend::ldg(SD.df(df_cur, pmz, x, y, z));
					KS.f[mmz] = TNL::Backend::ldg(SD.df(df_cur, ppz, x, y, z));
					KS.f[mzp] = TNL::Backend::ldg(SD.df(df_cur, pzm, x, y, z));
					KS.f[mzm] = TNL::Backend::ldg(SD.df(df_cur, pzp, x, y, z));
					KS.f[mpp] = TNL::Backend::ldg(SD.df(df_cur, pmm, x, y, z));
					KS.f[mpm] = TNL::Backend::ldg(SD.df(df_cur, pmp, x, y, z));
					KS.f[mmp] = TNL::Backend::ldg(SD.df(df_cur, ppm, x, y, z));
					KS.f[mmm] = TNL::Backend::ldg(SD.df(df_cur, ppp, x, y, z));
					dreal temp_f_mzz = KS.f[mzz];
					dreal temp_f_mpz = KS.f[mpz];
					dreal temp_f_mmz = KS.f[mmz];
					dreal temp_f_mzp = KS.f[mzp];
					dreal temp_f_mzm = KS.f[mzm];
					dreal temp_f_mpp = KS.f[mpp];
					dreal temp_f_mpm = KS.f[mpm];
					dreal temp_f_mmp = KS.f[mmp];
					dreal temp_f_mmm = KS.f[mmm];
					COLL::collision(KS);
					// load current inflow velocity profile
					SD.inflow(KS, x, y, z);
					// do extra collision because of inflow velocity profile
					// clang-format off
					dreal result = n2o27 * temp_f_mzz * no2 * no3 * (-KS.vx)
								 + n1o54 * temp_f_mpz * no2 * no3 * (-KS.vx + KS.vy)
								 + n1o54 * temp_f_mmz * no2 * no3 * (-KS.vx - KS.vy)
								 + n1o54 * temp_f_mzp * no2 * no3 * (-KS.vx + KS.vz)
								 + n1o54 * temp_f_mzm * no2 * no3 * (-KS.vx - KS.vz)
								 + n1o216 * temp_f_mpp * no2 * no3 * (-KS.vx + KS.vy + KS.vz)
								 + n1o216 * temp_f_mpm * no2 * no3 * (-KS.vx + KS.vy - KS.vz)
								 + n1o216 * temp_f_mmp * no2 * no3 * (-KS.vx - KS.vy + KS.vz)
								 + n1o216 * temp_f_mmm * no2 * no3 * (-KS.vx - KS.vy - KS.vz);
					// clang-format on
					KS.f[mmm] -= result;
					KS.f[mmz] -= result;
					KS.f[mmp] -= result;
					KS.f[mzm] -= result;
					KS.f[mzz] -= result;
					KS.f[mzp] -= result;
					KS.f[mpm] -= result;
					KS.f[mpz] -= result;
					KS.f[mpp] -= result;
					KS.f[zmm] -= result;
					KS.f[zmz] -= result;
					KS.f[zmp] -= result;
					KS.f[zzm] -= result;
					KS.f[zzz] -= result;
					KS.f[zzp] -= result;
					KS.f[zpm] -= result;
					KS.f[zpz] -= result;
					KS.f[zpp] -= result;
					KS.f[pmm] -= result;
					KS.f[pmz] -= result;
					KS.f[pmp] -= result;
					KS.f[pzm] -= result;
					KS.f[pzz] -= result;
					KS.f[pzp] -= result;
					KS.f[ppm] -= result;
					KS.f[ppz] -= result;
					KS.f[ppp] -= result;
					// calculate gradient
					SD.inflow(KS, x, y, z);
					// COLL::computeDensityAndVelocity(KS);
					// streaming
					break;
				}
			case GEO_ADJOINT_OUTFLOW_RIGHT:
				COLL::computeDensityAndVelocity_Wall(KS);  //! collision without drho (because K.rho = 1 always)
				break;

			default:
				COLL::computeDensityAndVelocity(KS);
				break;
		}
	}

	__cuda_callable__ static bool doCollision(map_t mapgi)
	{
		// by default, collision is done on non-BC sites only
		// additionally, BCs which include the collision step should be specified here
		return isFluid(mapgi) || isPeriodic(mapgi) || mapgi == GEO_OUTFLOW_RIGHT || mapgi == GEO_OUTFLOW_RIGHT_INTERP || mapgi == GEO_INFLOW_LEFT
			|| mapgi == GEO_INFLOW_BB_LEFT;
	}

	template <typename LBM_KS>
	__cuda_callable__ static void
	postCollision(DATA& SD, LBM_KS& KS, map_t mapgi, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm, idx z, idx zp)
	{
		if (mapgi == GEO_NOTHING)
			return;

		STREAMING::postCollisionStreaming(SD, KS, xm, x, xp, ym, y, yp, zm, z, zp);
	}
};
