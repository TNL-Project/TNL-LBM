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
		GEO_INFLOW_BOUNCEBACK,
		GEO_INFLOW_EQ_LEFT,
		GEO_OUTFLOW_EQ,
		GEO_OUTFLOW_RIGHT,
		GEO_OUTFLOW_RIGHT_INTERP,
		GEO_NOTHING,
		GEO_SYMMETRY,

		// Adjoint boundary conditions
		GEO_ADJOINT_FLUID,
		GEO_ADJOINT_FLUID_m,
		GEO_ADJOINT_WALL,
		GEO_ADJOINT_INFLOW_BB_LEFT,
		GEO_ADJOINT_OUTFLOW_RIGHT,

		// coarse cells adjacent to fine blocks - collision-active (streamed
		// and collided like fluid); the inter-level fine-to-coarse coupling
		// kernel additionally overwrites their DFs at the end of each coarse
		// step; appended last to keep the numeric values of existing tags stable
		GEO_AMR_INTERFACE
	};

	__cuda_callable__ static bool isSymmetric(map_t mapgi)
	{
		return mapgi == GEO_SYMMETRY;
	}

	__cuda_callable__ static bool isFluid(map_t mapgi)
	{
		return mapgi == GEO_FLUID;
	}

	__cuda_callable__ static bool isWall(map_t mapgi)
	{
		return mapgi == GEO_WALL;
	}

	// deterministic two-pass outflow: outflow cells are skipped in the main
	// kernel; State::SimUpdate launches cudaLBMKernelOutflow right before it,
	// which applies outflowPass on state finalized by the previous launch
	static constexpr bool use_outflow_pass = true;

	__cuda_callable__ static bool isOutflowPassBC(map_t mapgi)
	{
		return mapgi == GEO_OUTFLOW_RIGHT || mapgi == GEO_OUTFLOW_RIGHT_INTERP;
	}

	// gathers read the postcollision state finalized by the previous launch
	// and live in streaming_*.h; the BC body then follows the legacy cases
	template <typename LBM_KS>
	__cuda_callable__ static void outflowPass(DATA& SD, LBM_KS& KS, map_t mapgi, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm, idx z, idx zp)
	{
		switch (mapgi) {
			case GEO_OUTFLOW_RIGHT:
				STREAMING::streamingOutflowRight(SD, KS, xm, x, xp, ym, y, yp, zm, z, zp);
				applySymmetryCorner(SD, KS, xm, x, xp, ym, y, yp, zm, z, zp);
				COLL::computeDensityAndVelocity(KS);
				KS.rho = 1;
				COLL::collision(KS);
				STREAMING::postCollisionStreaming(SD, KS, xm, x, xp, ym, y, yp, zm, z, zp);
				break;
			case GEO_OUTFLOW_RIGHT_INTERP:
				STREAMING::streamingOutflowInterpRight(SD, KS, xm, x, xp, ym, y, yp, zm, z, zp);
				applySymmetryCorner(SD, KS, xm, x, xp, ym, y, yp, zm, z, zp);
				COLL::computeDensityAndVelocity(KS);
				COLL::setEquilibriumDecomposition(KS, 1);
				KS.rho = 1;
				COLL::collision(KS);
				STREAMING::postCollisionStreaming(SD, KS, xm, x, xp, ym, y, yp, zm, z, zp);
				break;
		}
	}

	// Bitmask of ghost half-spaces adjacent to a GEO_SYMMETRY cell.
	// Each bit marks a side where the neighbor cell is GEO_NOTHING (the domain-frame ghost layer).
	enum SYM_SIDES : std::uint8_t
	{
		SYM_XM = 1 << 0,  // ghost at x-1
		SYM_XP = 1 << 1,  // ghost at x+1
		SYM_YM = 1 << 2,  // ghost at y-1
		SYM_YP = 1 << 3,  // ghost at y+1
		SYM_ZM = 1 << 4,  // ghost at z-1
		SYM_ZP = 1 << 5,  // ghost at z+1
	};

	// direction slot for the letter trits (ex, ey, ez) in {m=0, z=1, p=2} packed as 9*ex + 3*ey + ez
	__cuda_callable__ static constexpr std::uint8_t sym_dir_slot(int code)
	{
		// clang-format off
		switch (code) {
			case 0:  return mmm;
			case 1:  return mmz;
			case 2:  return mmp;
			case 3:  return mzm;
			case 4:  return mzz;
			case 5:  return mzp;
			case 6:  return mpm;
			case 7:  return mpz;
			case 8:  return mpp;
			case 9:  return zmm;
			case 10: return zmz;
			case 11: return zmp;
			case 12: return zzm;
			case 13: return zzz;
			case 14: return zzp;
			case 15: return zpm;
			case 16: return zpz;
			case 17: return zpp;
			case 18: return pmm;
			case 19: return pmz;
			case 20: return pmp;
			case 21: return pzm;
			case 22: return pzz;
			case 23: return pzp;
			case 24: return ppm;
			case 25: return ppz;
			case 26: return ppp;
			default: return zzz;
		}
		// clang-format on
	}

	// Closure for GEO_SYMMETRY cells adjacent to two or three GEO_NOTHING ghost half-spaces
	// (edges and corners of the domain frame).
	// A population is unknown exactly when one of its non-zero components points through a ghost side
	// ('p**' through the x-1 ghost, 'm**' through x+1, '*p*' through y-1, '*m*' through y+1,
	// '**p' through z-1, and '**m' through z+1);
	// each unknown gets the value of the within-cell population with every ghost-crossing component flipped
	// (single reflection per orthogonal directions, double/triple for diagonals).
	// The fills are pure copies whose sources never cross a ghost side,
	// so the result does not depend on where in the domain frame the GEO_SYMMETRY planes lie.
	template <typename LBM_KS>
	__cuda_callable__ static void applySymmetry(LBM_KS& KS, std::uint8_t ghosts)
	{
		for (int code = 0; code < 27; code++) {
			const int ex = code / 9;
			const int ey = code / 3 % 3;
			const int ez = code % 3;
			// cx/cy/cz are set when the direction's component crosses a ghost side
			// ('p' crosses the minus side, 'm' crosses the plus side, 'z' crosses neither)
			const bool cx = ex != 1 && (ghosts & (ex == 2 ? SYM_XM : SYM_XP));
			const bool cy = ey != 1 && (ghosts & (ey == 2 ? SYM_YM : SYM_YP));
			const bool cz = ez != 1 && (ghosts & (ez == 2 ? SYM_ZM : SYM_ZP));
			if (cx || cy || cz) {
				const int src = 9 * (cx ? 2 - ex : ex) + 3 * (cy ? 2 - ey : ey) + (cz ? 2 - ez : ez);
				KS.f[sym_dir_slot(code)] = KS.f[sym_dir_slot(src)];
			}
		}
	}

	// Mirror-precondition for BC cells adjacent to GEO_SYMMETRY planes.
	// Checks the BC cell's own 6 neighbors: if a neighbor is GEO_SYMMETRY,
	// the remaining perpendicular neighbors are checked for GEO_NOTHING to
	// find the mirror half-planes. Symmetry may be on multiple axes;
	// each axis is checked independently.
	// No symmetry neighbor → no closure. The direction from the BC cell to the
	// symmetry cell is never mirrored (the BC handles those populations).
	template <typename LBM_KS>
	__cuda_callable__ static void applySymmetryCorner(DATA& SD, LBM_KS& KS, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm, idx z, idx zp)
	{
		std::uint8_t ghosts = 0;
		if (SD.map(xm, y, z) == GEO_SYMMETRY || SD.map(xp, y, z) == GEO_SYMMETRY) {
			if (SD.map(x, ym, z) == GEO_NOTHING)
				ghosts |= SYM_YM;
			if (SD.map(x, yp, z) == GEO_NOTHING)
				ghosts |= SYM_YP;
			if (SD.map(x, y, zm) == GEO_NOTHING)
				ghosts |= SYM_ZM;
			if (SD.map(x, y, zp) == GEO_NOTHING)
				ghosts |= SYM_ZP;
		}
		if (SD.map(x, ym, z) == GEO_SYMMETRY || SD.map(x, yp, z) == GEO_SYMMETRY) {
			if (SD.map(xm, y, z) == GEO_NOTHING)
				ghosts |= SYM_XM;
			if (SD.map(xp, y, z) == GEO_NOTHING)
				ghosts |= SYM_XP;
			if (SD.map(x, y, zm) == GEO_NOTHING)
				ghosts |= SYM_ZM;
			if (SD.map(x, y, zp) == GEO_NOTHING)
				ghosts |= SYM_ZP;
		}
		if (SD.map(x, y, zm) == GEO_SYMMETRY || SD.map(x, y, zp) == GEO_SYMMETRY) {
			if (SD.map(xm, y, z) == GEO_NOTHING)
				ghosts |= SYM_XM;
			if (SD.map(xp, y, z) == GEO_NOTHING)
				ghosts |= SYM_XP;
			if (SD.map(x, ym, z) == GEO_NOTHING)
				ghosts |= SYM_YM;
			if (SD.map(x, yp, z) == GEO_NOTHING)
				ghosts |= SYM_YP;
		}
		if (ghosts)
			applySymmetry(KS, ghosts);
	}

	template <typename LBM_KS>
	__cuda_callable__ static void preCollision(DATA& SD, LBM_KS& KS, map_t mapgi, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm, idx z, idx zp)
	{
		if (mapgi == GEO_NOTHING || isOutflowPassBC(mapgi)) {
			KS.rho = 1;
			KS.vx = 0;
			KS.vy = 0;
			KS.vz = 0;
			return;
		}

		// GEO_AMR_INTERFACE cells proceed through the normal streaming path:
		// the inter-level fine-to-coarse transfer still overwrites their DF
		// state at the end of each coarse step (collision-active interface)

		// modify pull location for streaming
		if (mapgi == GEO_ADJOINT_OUTFLOW_RIGHT)
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
					applySymmetryCorner(SD, KS, xm, x, xp, ym, y, yp, zm, z, zp);
					// moment reconstruction requires physical populations (identity for non-well storages)
					const auto f = [&KS](int i) -> dreal
					{
						return COLL::fromStorage(KS.f, i);
					};
					// moment boundary condition by Pavel Eichler https://doi.org/10.1016/j.camwa.2024.08.009
					// expressions symetrized by Jakub Klinkovsky
					// clang-format off
					KS.rho = (dreal)1.0/(1-KS.vx) * (
						(
							f(zzz) + (
								+ ((f(zpp) + f(zmm)) + (f(zpm) + f(zmp)))
								+ ((f(zpz) + f(zmz)) + (f(zzp) + f(zzm)))
							)
						)
						+ 2*(
							f(mzz) + (
								+ ((f(mpp) + f(mmm)) + (f(mpm) + f(mmp)))
								+ ((f(mpz) + f(mmz)) + (f(mzp) + f(mzm)))
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
					KS.f[pzz] = COLL::toStorage(
						m100 + (m022 - (m020 + m002))
						+ f(mzz)
						+ (
							+ ((f(zpp) + f(zmm)) + (f(zpm) + f(zmp)))
							+ ((f(zzp) + f(zzm)) + (f(zpz) + f(zmz)))
						)
						+ 2*(
							+ ((f(mpp) + f(mmm)) + (f(mpm) + f(mmp)))
							+ ((f(mpz) + f(mmz)) + (f(mzp) + f(mzm)))
						),
						pzz);
					// clang-format on
					KS.f[ppz] = COLL::toStorage((dreal) 0.5 * ((m020 - m022) + (-m012 + m010)) - (f(mpz) + f(zpz)), ppz);
					KS.f[pmz] = COLL::toStorage((dreal) 0.5 * ((m020 - m022) + (m012 - m010)) - (f(mmz) + f(zmz)), pmz);
					KS.f[pzp] = COLL::toStorage((dreal) 0.5 * ((m002 - m022) + (-m021 + m001)) - (f(mzp) + f(zzp)), pzp);
					KS.f[pzm] = COLL::toStorage((dreal) 0.5 * ((m002 - m022) + (m021 - m001)) - (f(mzm) + f(zzm)), pzm);
					KS.f[ppp] = COLL::toStorage((dreal) 0.25 * ((m022 + m011) + (m021 + m012)) - (f(mpp) + f(zpp)), ppp);
					KS.f[ppm] = COLL::toStorage((dreal) 0.25 * ((m022 - m011) + (-m021 + m012)) - (f(mpm) + f(zpm)), ppm);
					KS.f[pmp] = COLL::toStorage((dreal) 0.25 * ((m022 - m011) + (m021 - m012)) - (f(mmp) + f(zmp)), pmp);
					KS.f[pmm] = COLL::toStorage((dreal) 0.25 * ((m022 + m011) + (-m021 - m012)) - (f(mmm) + f(zmm)), pmm);
					break;
				}
			case GEO_INFLOW_BOUNCEBACK:
				SD.inflow(KS, x, y, z);
				applySymmetryCorner(SD, KS, xm, x, xp, ym, y, yp, zm, z, zp);
				// collision step: bounce-back with modified right-hand-side:
				// -2/c_s^2 * rho(x_wall, t_n) * (\xi_k, v_wall)
				{
					dreal t;
					// clang-format off
					t = KS.f[ppp];
					KS.f[ppp] = KS.f[mmm] - no6*KS.rho*n1o216*(- KS.vx - KS.vy - KS.vz);
					KS.f[mmm] = t         - no6*KS.rho*n1o216*(  KS.vx + KS.vy + KS.vz);

					t = KS.f[ppz];
					KS.f[ppz] = KS.f[mmz] - no6*KS.rho*n1o54*(- KS.vx - KS.vy);
					KS.f[mmz] = t         - no6*KS.rho*n1o54*(  KS.vx + KS.vy);

					t = KS.f[ppm];
					KS.f[ppm] = KS.f[mmp] - no6*KS.rho*n1o216*(- KS.vx - KS.vy + KS.vz);
					KS.f[mmp] = t         - no6*KS.rho*n1o216*(  KS.vx + KS.vy - KS.vz);

					t = KS.f[pzp];
					KS.f[pzp] = KS.f[mzm] - no6*KS.rho*n1o54*(- KS.vx - KS.vz);
					KS.f[mzm] = t         - no6*KS.rho*n1o54*(  KS.vx + KS.vz);

					t = KS.f[pzz];
					KS.f[pzz] = KS.f[mzz] - no6*KS.rho*n2o27*(- KS.vx);
					KS.f[mzz] = t         - no6*KS.rho*n2o27*(  KS.vx);

					t = KS.f[pzm];
					KS.f[pzm] = KS.f[mzp] - no6*KS.rho*n1o54*(- KS.vx + KS.vz);
					KS.f[mzp] = t         - no6*KS.rho*n1o54*(  KS.vx - KS.vz);

					t = KS.f[pmp];
					KS.f[pmp] = KS.f[mpm] - no6*KS.rho*n1o216*(- KS.vx + KS.vy - KS.vz);
					KS.f[mpm] = t         - no6*KS.rho*n1o216*(  KS.vx - KS.vy + KS.vz);

					t = KS.f[pmz];
					KS.f[pmz] = KS.f[mpz] - no6*KS.rho*n1o54*(- KS.vx + KS.vy);
					KS.f[mpz] = t         - no6*KS.rho*n1o54*(  KS.vx - KS.vy);

					t = KS.f[pmm];
					KS.f[pmm] = KS.f[mpp] - no6*KS.rho*n1o216*(- KS.vx + KS.vy + KS.vz);
					KS.f[mpp] = t         - no6*KS.rho*n1o216*(  KS.vx - KS.vy - KS.vz);

					t = KS.f[zpp];
					KS.f[zpp] = KS.f[zmm] - no6*KS.rho*n1o54*(- KS.vy - KS.vz);
					KS.f[zmm] = t         - no6*KS.rho*n1o54*(  KS.vy + KS.vz);

					t = KS.f[zpz];
					KS.f[zpz] = KS.f[zmz] - no6*KS.rho*n2o27*(- KS.vy);
					KS.f[zmz] = t         - no6*KS.rho*n2o27*(  KS.vy);

					t = KS.f[zpm];
					KS.f[zpm] = KS.f[zmp] - no6*KS.rho*n1o54*(- KS.vy + KS.vz);
					KS.f[zmp] = t         - no6*KS.rho*n1o54*(  KS.vy - KS.vz);

					t = KS.f[zzp];
					KS.f[zzp] = KS.f[zzm] - no6*KS.rho*n2o27*(- KS.vz);
					KS.f[zzm] = t         - no6*KS.rho*n2o27*(  KS.vz);
					// clang-format on
				}
				break;
			case GEO_INFLOW_EQ_LEFT:
				{
					SD.inflow(KS, x, y, z);
					applySymmetryCorner(SD, KS, xm, x, xp, ym, y, yp, zm, z, zp);
					// moment reconstruction requires physical populations (identity for non-well storages)
					const auto f = [&KS](int i) -> dreal
					{
						return COLL::fromStorage(KS.f, i);
					};
					// clang-format off
					KS.rho = (dreal)1.0/(1-KS.vx) * (
						(
							f(zzz) + (
								+ ((f(zpp) + f(zmm)) + (f(zpm) + f(zmp)))
								+ ((f(zpz) + f(zmz)) + (f(zzp) + f(zzm)))
							)
						)
						+ 2*(
							f(mzz) + (
								+ ((f(mpp) + f(mmm)) + (f(mpm) + f(mmp)))
								+ ((f(mpz) + f(mmz)) + (f(mzp) + f(mzm)))
							)
						)
					);
					// clang-format on
					COLL::setEquilibrium(KS);
					break;
				}
			case GEO_OUTFLOW_EQ:
				applySymmetryCorner(SD, KS, xm, x, xp, ym, y, yp, zm, z, zp);
				COLL::computeDensityAndVelocity(KS);
				KS.rho = 1;
				COLL::setEquilibrium(KS);
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
			case GEO_SYMMETRY:
				{
					// Detect ghost half-spaces on all six sides, handling edges and corners.
					// The symmetry cell acts as a fluid cell
					// and directions towards GEO_NOTHING determine the normal of the symmetry plane.
					std::uint8_t ghosts = 0;
					if (SD.map(xm, y, z) == GEO_NOTHING)
						ghosts |= SYM_XM;
					if (SD.map(xp, y, z) == GEO_NOTHING)
						ghosts |= SYM_XP;
					if (SD.map(x, ym, z) == GEO_NOTHING)
						ghosts |= SYM_YM;
					if (SD.map(x, yp, z) == GEO_NOTHING)
						ghosts |= SYM_YP;
					if (SD.map(x, y, zm) == GEO_NOTHING)
						ghosts |= SYM_ZM;
					if (SD.map(x, y, zp) == GEO_NOTHING)
						ghosts |= SYM_ZP;
					applySymmetry(KS, ghosts);
					COLL::computeDensityAndVelocity(KS);
					break;
				}

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
					// site-local post-collision populations from the wall node are needed
					// for a valid rho in the collision below; the streamed +x populations
					// come from the ghost face (unused in AB) and would pollute the state
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
					dreal result = n2o27 * temp_f_mzz * no6 * (-KS.vx)
								 + n1o54 * temp_f_mpz * no6 * (-KS.vx + KS.vy)
								 + n1o54 * temp_f_mmz * no6 * (-KS.vx - KS.vy)
								 + n1o54 * temp_f_mzp * no6 * (-KS.vx + KS.vz)
								 + n1o54 * temp_f_mzm * no6 * (-KS.vx - KS.vz)
								 + n1o216 * temp_f_mpp * no6 * (-KS.vx + KS.vy + KS.vz)
								 + n1o216 * temp_f_mpm * no6 * (-KS.vx + KS.vy - KS.vz)
								 + n1o216 * temp_f_mmp * no6 * (-KS.vx - KS.vy + KS.vz)
								 + n1o216 * temp_f_mmm * no6 * (-KS.vx - KS.vy - KS.vz);
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
		// interface cells collide like fluid; the inter-level coupling
		// kernel may still overwrite them at the end of each coarse step
		if (mapgi == GEO_AMR_INTERFACE)
			return true;
		// by default, collision is done on non-BC sites only
		// additionally, BCs which include the collision step should be specified here
		return isFluid(mapgi) || isSymmetric(mapgi) || mapgi == GEO_INFLOW_LEFT;
	}

	template <typename LBM_KS>
	__cuda_callable__ static void
	postCollision(DATA& SD, LBM_KS& KS, map_t mapgi, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm, idx z, idx zp)
	{
		if (mapgi == GEO_NOTHING || isOutflowPassBC(mapgi))
			return;

		// GEO_AMR_INTERFACE cells are written back like fluid; the
		// inter-level fine-to-coarse transfer may overwrite their DF state
		// afterwards (collision-active interface)

		STREAMING::postCollisionStreaming(SD, KS, xm, x, xp, ym, y, yp, zm, z, zp);
	}

	// bitmask of the D3Q27 directions (bit q matches the direction enum in
	// defs.h) in which the cell at (x, y, z) crosses a refinement interface;
	// returns 0 for blocks without interface data (non-AMR runs, fine blocks)
	// and for cells not adjacent to a fine region
	template <typename LBM_DATA>
	__cuda_callable__ static std::uint32_t getInterfaceDir(const LBM_DATA& SD, idx x, idx y, idx z)
	{
		if (SD.dinterface_dir == nullptr)
			return 0;
		return SD.dinterface_dir[SD.indexer.getStorageIndex(x, y, z)];
	}
};
