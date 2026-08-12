#pragma once

#include <TNL/Backend.h>

#include "d3q27/macro.h"
#include "defs.h"
#include "kernels.h"
#include "lbm_common/ciselnik.h"
#include "lbm_data.h"

// Extra kernels for the non-Newtonian fluid model

// LBMKernelStress: computes the strain-rate tensor S from the second moment of f_i^neq
// (local, no finite differences). S is written to the macro array for computeForcing.
template <typename NSE>
#ifdef USE_CUDA
__global__ void cudaLBMKernelStress(
	typename NSE::DATA SD, typename NSE::TRAITS::bool3d distributed, typename NSE::TRAITS::idx3d offset, typename NSE::TRAITS::idx3d end
)
#else
CUDA_HOSTDEV void LBMKernelStress(
	typename NSE::DATA SD,
	typename NSE::TRAITS::idx x,
	typename NSE::TRAITS::idx y,
	typename NSE::TRAITS::idx z,
	typename NSE::TRAITS::bool3d distributed
)
#endif
{
	using dreal = typename NSE::TRAITS::dreal;
	using idx = typename NSE::TRAITS::idx;
	using map_t = typename NSE::TRAITS::map_t;

#ifdef USE_CUDA
	idx x = threadIdx.x + blockIdx.x * blockDim.x + offset.x();
	idx y = threadIdx.y + blockIdx.y * blockDim.y + offset.y();
	idx z = threadIdx.z + blockIdx.z * blockDim.z + offset.z();

	if (x >= end.x() || y >= end.y() || z >= end.z())
		return;
#endif

	map_t gi_map = SD.map(x, y, z);

	typename NSE::template KernelStruct<dreal> KS;
	constexpr int Q = decltype(KS)::Q;

	// copy quantities (loads lbmViscosity, fx, fy, fz)
	NSE::MACRO::copyQuantities(SD, KS, x, y, z);

	idx xp, xm, yp, ym, zp, zm;
	kernelInitIndices<NSE>(SD, gi_map, distributed, x, y, z, xp, xm, yp, ym, zp, zm);

	if (NSE::BC::isFluid(gi_map) || NSE::BC::isPeriodic(gi_map)) {
		// Load post-streaming, pre-collision f_i into KS.f via the pull-scheme streaming.
		// df_cur holds the populations from the previous step; the pull reads from
		// upwind neighbors to give the post-stream f_i at the current cell.
		NSE::STREAMING::streaming(SD, KS, xm, x, xp, ym, y, yp, zm, z, zp);

		// Compute (rho, vx, vy, vz) from the post-stream f_i.
		// This ensures f^eq is consistent with f_i at ALL cells, including walls
		// and inflow/outflow where the macro array may hold BC-imposed values
		// rather than the actual moments of f_i.
		NSE::COLL::computeDensityAndVelocity(KS);

		// Save f_i before setEquilibrium overwrites KS.f
		dreal fneq[Q];
		for (int i = 0; i < Q; i++)
			fneq[i] = KS.f[i];

		// Compute f_i^eq from (rho, vx, vy, vz) — overwrites KS.f
		NSE::COLL::setEquilibrium(KS);

		// fneq[i] = f_i - f_i^eq
		for (int i = 0; i < Q; i++)
			fneq[i] -= KS.f[i];

		// Compute the non-equilibrium second moment:
		//   Pi^neq_αβ = Σ_i c_{iα} c_{iβ} f_i^neq
		//
		// Direction naming: p=+1, z=0, m=-1 along x,y,z.
		// c_α² = 1 for all directions with c_α ≠ 0 (diagonal components).
		// c_α c_β = ±1 for directions with both c_α, c_β ≠ 0 (off-diagonal).

		// Diagonal: sum f^neq over directions with c_x ≠ 0 (18 directions)
		dreal Pi11 = (fneq[pzz] + fneq[mzz]) + (fneq[ppz] + fneq[mmz]) + (fneq[pmz] + fneq[mpz]) + (fneq[pzp] + fneq[mzm]) + (fneq[pzm] + fneq[mzp])
				   + (fneq[ppp] + fneq[mmm]) + (fneq[ppm] + fneq[mmp]) + (fneq[pmp] + fneq[mpm]) + (fneq[pmm] + fneq[mpp]);

		// Diagonal: sum f^neq over directions with c_y ≠ 0 (18 directions)
		dreal Pi22 = (fneq[zpz] + fneq[zmz]) + (fneq[ppz] + fneq[mmz]) + (fneq[pmz] + fneq[mpz]) + (fneq[zpp] + fneq[zmm]) + (fneq[zpm] + fneq[zmp])
				   + (fneq[ppp] + fneq[mmm]) + (fneq[ppm] + fneq[mmp]) + (fneq[pmp] + fneq[mpm]) + (fneq[pmm] + fneq[mpp]);

		// Diagonal: sum f^neq over directions with c_z ≠ 0 (18 directions)
		dreal Pi33 = (fneq[zzp] + fneq[zzm]) + (fneq[pzp] + fneq[mzm]) + (fneq[pzm] + fneq[mzp]) + (fneq[zpp] + fneq[zmm]) + (fneq[zpm] + fneq[zmp])
				   + (fneq[ppp] + fneq[mmm]) + (fneq[ppm] + fneq[mmp]) + (fneq[pmp] + fneq[mpm]) + (fneq[pmm] + fneq[mpp]);

		// Off-diagonal xy: c_x*c_y = +1 for ppz,mmz,ppp,mmm,ppm,mmp; -1 for pmz,mpz,pmp,mpm,pmm,mpp
		dreal Pi12 = ((fneq[ppz] + fneq[mmz]) + (fneq[ppp] + fneq[mmm]) + (fneq[ppm] + fneq[mmp]))
				   - ((fneq[pmz] + fneq[mpz]) + (fneq[pmp] + fneq[mpm]) + (fneq[pmm] + fneq[mpp]));

		// Off-diagonal xz: c_x*c_z = +1 for pzp,mzm,ppp,mmm,pmp,mpm; -1 for pzm,mzp,ppm,mmp,pmm,mpp
		dreal Pi13 = ((fneq[pzp] + fneq[mzm]) + (fneq[ppp] + fneq[mmm]) + (fneq[pmp] + fneq[mpm]))
				   - ((fneq[pzm] + fneq[mzp]) + (fneq[ppm] + fneq[mmp]) + (fneq[pmm] + fneq[mpp]));

		// Off-diagonal zy: c_z*c_y = +1 for zpp,zmm,ppp,mmm,pmm,mpp; -1 for zpm,zmp,ppm,mmp,pmp,mpm
		dreal Pi32 = ((fneq[zpp] + fneq[zmm]) + (fneq[ppp] + fneq[mmm]) + (fneq[pmm] + fneq[mpp]))
				   - ((fneq[zpm] + fneq[zmp]) + (fneq[ppm] + fneq[mmp]) + (fneq[pmp] + fneq[mpm]));

		// Convert Pi^neq to strain rate:
		//   D_αβ = -1/(2 ρ c_s² τ₀) · Pi^neq_αβ = -3/(2 ρ τ₀) · Pi^neq_αβ
		// where τ₀ = 3·lbmViscosity + 1/2 (the relaxation time used in collision).
		// Assumes shear second moments relax at 1/τ₀ (true for SRT and CUM's omega1).
		dreal tau0 = no3 * KS.lbmViscosity + n1o2;
		dreal factor = -n3o2 / (KS.rho * tau0);

		KS.S11 = factor * Pi11;
		KS.S12 = factor * Pi12;
		KS.S13 = factor * Pi13;
		KS.S22 = factor * Pi22;
		KS.S32 = factor * Pi32;
		KS.S33 = factor * Pi33;
	}

	NSE::MACRO::outputMacroStress(SD, KS, x, y, z);
}

template <typename STATE>
void computeNonNewtonianKernels(STATE& state)
{
	using NSE = typename STATE::NSE_type;
	using TRAITS = typename STATE::TRAITS;

	using idx3d = typename TRAITS::idx3d;

	auto& nse = state.nse;

	const auto boundary_directions = {
		TNL::Containers::SyncDirection::Bottom,
		TNL::Containers::SyncDirection::Top,
		TNL::Containers::SyncDirection::Back,
		TNL::Containers::SyncDirection::Front,
		TNL::Containers::SyncDirection::Left,
		TNL::Containers::SyncDirection::Right,
	};

	// LBMKernelStress reads f_i via pull-scheme streaming, which accesses
	// neighbor cells — df_cur must be synced on MPI overlaps first.
#ifdef HAVE_MPI
	if (nse.nproc > 1)
		nse.synchronizeDFsAndMacroDevice(df_cur, true);
#endif

	// compute on boundaries
	for (auto& block : nse.blocks) {
		for (auto direction : boundary_directions)
			if (auto search = block.neighborIDs.find(direction); search != block.neighborIDs.end() && search->second >= 0) {
				const dim3 blockSize = block.computeData.at(direction).blockSize;
				const dim3 gridSize = block.computeData.at(direction).gridSize;
				const auto& stream = block.computeData.at(direction).stream;
				const idx3d offset = block.computeData.at(direction).offset;
				const idx3d size = block.computeData.at(direction).size;
				TNL::Backend::LaunchConfiguration launch_config;
				launch_config.gridSize = gridSize;
				launch_config.blockSize = blockSize;
				launch_config.stream = stream;
				TNL::Backend::launchKernelAsync(cudaLBMKernelStress<NSE>, launch_config, block.data, block.is_distributed(), offset, offset + size);
			}
	}

	// compute on interior lattice sites
	for (auto& block : nse.blocks) {
		const auto direction = TNL::Containers::SyncDirection::None;
		const dim3 blockSize = block.computeData.at(direction).blockSize;
		const dim3 gridSize = block.computeData.at(direction).gridSize;
		const auto& stream = block.computeData.at(direction).stream;
		const idx3d offset = block.computeData.at(direction).offset;
		const idx3d size = block.computeData.at(direction).size;
		TNL::Backend::LaunchConfiguration launch_config;
		launch_config.gridSize = gridSize;
		launch_config.blockSize = blockSize;
		launch_config.stream = stream;
		TNL::Backend::launchKernelAsync(cudaLBMKernelStress<NSE>, launch_config, block.data, block.is_distributed(), offset, offset + size);
	}

	// wait for the computations on boundaries to finish
	for (auto& block : nse.blocks)
		for (auto direction : boundary_directions)
			TNL::Backend::streamSynchronize(block.computeData.at(direction).stream);

	// exchange S tensor on overlaps between blocks (needed by computeForcing in the main kernel)
	// TODO: sync only macro, not DFs
#ifdef HAVE_MPI
	if (nse.nproc > 1)
		nse.synchronizeDFsAndMacroDevice(df_cur, true);
#endif

	// wait for the computation on the interior to finish
	for (auto& block : nse.blocks) {
		const auto& stream = block.computeData.at(TNL::Containers::SyncDirection::None).stream;
		TNL::Backend::streamSynchronize(stream);
	}

	// synchronize the null-stream after all grids
	TNL::Backend::streamSynchronize(0);
	TNL_CHECK_CUDA_DEVICE;
}

// Default "LBM data" class for the non-Newtonian fluid model
template <typename TRAITS>
struct LBM_Data_NonNewtonian : NSE_Data<TRAITS>
{
	using dreal = typename TRAITS::dreal;

	// Parameters for non-Newtonian models (must be initialized from sim)
#if defined(USE_POWERLAW)
	dreal lbm_K;
	dreal lbm_n;
#elif defined(USE_CYMODEL)
	dreal lbm_nu_inf;
	dreal lbm_lambda;
	dreal lbm_a;
	dreal lbm_n;
#elif defined(USE_CASSON)
	dreal lbm_k0;
	dreal lbm_k1;
#endif
};

// Default macro class for the non-Newtonian fluid model
template <typename TRAITS>
struct MacroNonNewtonianDefault : D3Q27_MACRO_Default<TRAITS>
{
	using dreal = typename TRAITS::dreal;
	using idx = typename TRAITS::idx;
	using map_t = typename TRAITS::map_t;

	enum QuantityNames : std::uint8_t
	{
		e_rho,
		e_vx,
		e_vy,
		e_vz,
		e_fx,
		e_fy,
		e_fz,
		e_S11,
		e_S12,
		e_S13,
		e_S22,
		e_S32,
		e_S33,
		N
	};

	template <typename LBM_DATA, typename LBM_KS>
	CUDA_HOSTDEV static void outputMacro(LBM_DATA& SD, LBM_KS& KS, idx x, idx y, idx z)
	{
		SD.macro(e_rho, x, y, z) = KS.rho;
		SD.macro(e_vx, x, y, z) = KS.vx;
		SD.macro(e_vy, x, y, z) = KS.vy;
		SD.macro(e_vz, x, y, z) = KS.vz;

		SD.macro(e_fx, x, y, z) = KS.fx;
		SD.macro(e_fy, x, y, z) = KS.fy;
		SD.macro(e_fz, x, y, z) = KS.fz;
	}

	template <typename LBM_DATA, typename LBM_KS>
	CUDA_HOSTDEV static void outputMacroStress(LBM_DATA& SD, LBM_KS& KS, idx x, idx y, idx z)
	{
		SD.macro(e_S11, x, y, z) = KS.S11;
		SD.macro(e_S12, x, y, z) = KS.S12;
		SD.macro(e_S22, x, y, z) = KS.S22;
		SD.macro(e_S32, x, y, z) = KS.S32;
		SD.macro(e_S13, x, y, z) = KS.S13;
		SD.macro(e_S33, x, y, z) = KS.S33;
	}

	template <typename LBM_DATA, typename LBM_KS>
	CUDA_HOSTDEV static void getStressFromMacro(LBM_DATA& SD, LBM_KS& KS, idx x, idx y, idx z)
	{
		KS.S11 = SD.macro(e_S11, x, y, z);
		KS.S12 = SD.macro(e_S12, x, y, z);
		KS.S22 = SD.macro(e_S22, x, y, z);
		KS.S32 = SD.macro(e_S32, x, y, z);
		KS.S13 = SD.macro(e_S13, x, y, z);
		KS.S33 = SD.macro(e_S33, x, y, z);
	}

	template <typename LBM_DATA, typename LBM_KS>
	CUDA_HOSTDEV static void copyQuantities(LBM_DATA& SD, LBM_KS& KS, idx x, idx y, idx z)
	{
		KS.lbmViscosity = SD.lbmViscosity;
		KS.fx = SD.fx;
		KS.fy = SD.fy;
		KS.fz = SD.fz;
	}

	// computeForcing: applies the split-viscosity body force
	//   f = div( (nu - nu0) * S )
	// via product-rule finite differences of the S tensor.
	template <typename LBM_BC, typename LBM_DATA, typename LBM_KS>
	CUDA_HOSTDEV static void computeForcing(LBM_DATA& SD, LBM_KS& KS, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm, idx z, idx zp)
	{
		map_t gi_map = SD.map(x, y, z);
		map_t gi_map_xp = SD.map(xp, y, z);
		map_t gi_map_xm = SD.map(xm, y, z);
		map_t gi_map_yp = SD.map(x, yp, z);
		map_t gi_map_ym = SD.map(x, ym, z);
		map_t gi_map_zp = SD.map(x, y, zp);
		map_t gi_map_zm = SD.map(x, y, zm);

		LBM_KS KSxp, KSxm, KSyp, KSym, KSzp, KSzm;

		getStressFromMacro(SD, KSxp, xp, y, z);
		getStressFromMacro(SD, KSxm, xm, y, z);
		getStressFromMacro(SD, KSyp, x, yp, z);
		getStressFromMacro(SD, KSym, x, ym, z);
		getStressFromMacro(SD, KSzp, x, y, zp);
		getStressFromMacro(SD, KSzm, x, y, zm);
		getStressFromMacro(SD, KS, x, y, z);

		dreal F1 = 0;
		dreal F2 = 0;
		dreal F3 = 0;

		if (LBM_BC::isFluid(gi_map) || LBM_BC::isPeriodic(gi_map)) {
			// div((nu-nu0)*S) via product rule: differencing (nu-nu0)*S at neighbors.
			// This naturally includes the grad(nu-nu0)·S term via the product rule of
			// finite differences, which the pointwise (nu-nu0)*div(S) form would miss.
			dreal nu0 = KS.lbmViscosity;
			auto compute_dnu = [&](dreal s11, dreal s22, dreal s33, dreal s12, dreal s13, dreal s32) -> dreal
			{
				dreal D_D = s11 * s11 + s22 * s22 + s33 * s33 + no2 * (s12 * s12 + s13 * s13 + s32 * s32);
				dreal gamma_sq = no2 * D_D;
				dreal nu;
#if defined(USE_POWERLAW)
				// Smooth regularization: gd_reg = gd + floor^2/(gd + floor).
				// C^inf smooth (no kink in grad(dnu)), exact for gd >> floor (modification is O(floor^2/gd)),
				// clamps to floor as gd→0. floor set above FP32 noise level.
				dreal gd_floor = (dreal) 1e-10;
				dreal gamma_sq_reg = gamma_sq + gd_floor * gd_floor / (gamma_sq + gd_floor);
				// The power-law formula uses K_lbm (lattice consistency index) rather than nu0.
				// nu0 is subtracted below since the forcing computes div((nu - nu0)·S).
				nu = SD.lbm_K * TNL::pow(gamma_sq_reg, (SD.lbm_n - no1) * n1o2);
#elif defined(USE_CYMODEL)
				dreal gamma = TNL::sqrt(gamma_sq);
				nu = SD.lbm_nu_inf + (nu0 - SD.lbm_nu_inf) * TNL::pow(no1 + TNL::pow(gamma * SD.lbm_lambda, SD.lbm_a), (SD.lbm_n - no1) / SD.lbm_a);
#elif defined(USE_CASSON)
				// Standard Casson: eta = (k0 + k1*sqrt(gamma))^2 / gamma, where gamma = |W'|.
				// Regularization: clamp gamma to >= 1e-10 so eta stays bounded at the plug.
				dreal gamma = TNL::sqrt(gamma_sq);
				dreal gamma_cap = TNL::max(gamma, (dreal) 1e-10);
				nu = TNL::sqr(SD.lbm_k0 + SD.lbm_k1 * TNL::sqrt(gamma_cap)) / gamma_cap;
#endif
				return nu - nu0;
			};

			dreal dnu_c = compute_dnu(KS.S11, KS.S22, KS.S33, KS.S12, KS.S13, KS.S32);
			dreal dnu_xp = compute_dnu(KSxp.S11, KSxp.S22, KSxp.S33, KSxp.S12, KSxp.S13, KSxp.S32);
			dreal dnu_xm = compute_dnu(KSxm.S11, KSxm.S22, KSxm.S33, KSxm.S12, KSxm.S13, KSxm.S32);
			dreal dnu_yp = compute_dnu(KSyp.S11, KSyp.S22, KSyp.S33, KSyp.S12, KSyp.S13, KSyp.S32);
			dreal dnu_ym = compute_dnu(KSym.S11, KSym.S22, KSym.S33, KSym.S12, KSym.S13, KSym.S32);
			dreal dnu_zp = compute_dnu(KSzp.S11, KSzp.S22, KSzp.S33, KSzp.S12, KSzp.S13, KSzp.S32);
			dreal dnu_zm = compute_dnu(KSzm.S11, KSzm.S22, KSzm.S33, KSzm.S12, KSzm.S13, KSzm.S32);

#if defined(NONNEWTONIAN_SKIP_GRAD_DNU)
			// Test: neglect d/dx(nu-nu0), d/dy(nu-nu0), d/dz(nu-nu0) terms.
			// Use the central dnu for all neighbors so the product-rule expansion
			// collapses to dnu_c * div(S) — isolates the dnu*div(S) part only.
			dnu_xp = dnu_c;
			dnu_xm = dnu_c;
			dnu_yp = dnu_c;
			dnu_ym = dnu_c;
			dnu_zp = dnu_c;
			dnu_zm = dnu_c;
#endif

			// Stress tensor S is computed only at fluid/periodic sites by LBMKernelStress.
			auto hasKnownStress = [](map_t m)
			{
				return LBM_BC::isFluid(m) || LBM_BC::isPeriodic(m);
			};

			// x-direction: d/dx[(nu-nu0)*S]
			if (hasKnownStress(gi_map_xm)) {
				if (hasKnownStress(gi_map_xp)) {
					F1 += n1o2 * (dnu_xp * KSxp.S11 - dnu_xm * KSxm.S11);
					F2 += n1o2 * (dnu_xp * KSxp.S12 - dnu_xm * KSxm.S12);
					F3 += n1o2 * (dnu_xp * KSxp.S13 - dnu_xm * KSxm.S13);
				}
				else {
					F1 += dnu_c * KS.S11 - dnu_xm * KSxm.S11;
					F2 += dnu_c * KS.S12 - dnu_xm * KSxm.S12;
					F3 += dnu_c * KS.S13 - dnu_xm * KSxm.S13;
				}
			}
			else if (hasKnownStress(gi_map_xp)) {
				F1 += dnu_xp * KSxp.S11 - dnu_c * KS.S11;
				F2 += dnu_xp * KSxp.S12 - dnu_c * KS.S12;
				F3 += dnu_xp * KSxp.S13 - dnu_c * KS.S13;
			}

			// y-direction: d/dy[(nu-nu0)*S]
			if (hasKnownStress(gi_map_ym)) {
				if (hasKnownStress(gi_map_yp)) {
					F1 += n1o2 * (dnu_yp * KSyp.S12 - dnu_ym * KSym.S12);
					F2 += n1o2 * (dnu_yp * KSyp.S22 - dnu_ym * KSym.S22);
					F3 += n1o2 * (dnu_yp * KSyp.S32 - dnu_ym * KSym.S32);
				}
				else {
					F1 += dnu_c * KS.S12 - dnu_ym * KSym.S12;
					F2 += dnu_c * KS.S22 - dnu_ym * KSym.S22;
					F3 += dnu_c * KS.S32 - dnu_ym * KSym.S32;
				}
			}
			else if (hasKnownStress(gi_map_yp)) {
				F1 += dnu_yp * KSyp.S12 - dnu_c * KS.S12;
				F2 += dnu_yp * KSyp.S22 - dnu_c * KS.S22;
				F3 += dnu_yp * KSyp.S32 - dnu_c * KS.S32;
			}

			// z-direction: d/dz[(nu-nu0)*S]
			if (hasKnownStress(gi_map_zm)) {
				if (hasKnownStress(gi_map_zp)) {
					F1 += n1o2 * (dnu_zp * KSzp.S13 - dnu_zm * KSzm.S13);
					F2 += n1o2 * (dnu_zp * KSzp.S32 - dnu_zm * KSzm.S32);
					F3 += n1o2 * (dnu_zp * KSzp.S33 - dnu_zm * KSzm.S33);
				}
				else {
					F1 += dnu_c * KS.S13 - dnu_zm * KSzm.S13;
					F2 += dnu_c * KS.S32 - dnu_zm * KSzm.S32;
					F3 += dnu_c * KS.S33 - dnu_zm * KSzm.S33;
				}
			}
			else if (hasKnownStress(gi_map_zp)) {
				F1 += dnu_zp * KSzp.S13 - dnu_c * KS.S13;
				F2 += dnu_zp * KSzp.S32 - dnu_c * KS.S32;
				F3 += dnu_zp * KSzp.S33 - dnu_c * KS.S33;
			}
		}

		// NOTE: KS.rho is default-initialized to 1.0 in defs.h
		dreal rho = KS.rho;

		KS.fx += no2 * F1 * rho;
		KS.fy += no2 * F2 * rho;
		KS.fz += no2 * F3 * rho;
	}
};
