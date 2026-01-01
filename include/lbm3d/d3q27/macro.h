#pragma once

#include <cstdint>

#include <TNL/Backend/Macros.h>

// empty Macro containing required forcing quantities for IBM (see lbm.h -> hfx() etc.)
template <typename TRAITS>
struct D3Q27_MACRO_Base
{
	using dreal = typename TRAITS::dreal;
	using idx = typename TRAITS::idx;

	// all quantities after `N` are ignored
	enum QuantityNames : std::uint8_t
	{
		N,
		e_rho,
		e_vx,
		e_vy,
		e_vz,
		e_fx,
		e_fy,
		e_fz
	};

	// specifies if macroscopic quantities are computed in the kernel in each iteration
	static const bool compute_in_each_iteration = false;

	// specifies if the dmacro array is synchronized with MPI in each iteration
	static const bool use_syncMacro = false;

	// maximum width of overlaps for the macro arrays
	static constexpr int overlap_width = 1;

	// compulsory method -- called from cudaLBMComputeVelocitiesStarAndZeroForce kernel
	template <typename LBM_KS>
	__cuda_callable__ static void zeroForcesInKS(LBM_KS& KS)
	{
		KS.fx = 0;
		KS.fy = 0;
		KS.fz = 0;
	}

	// compulsory method -- called from cudaLBMComputeVelocitiesStarAndZeroForce kernel
	template <typename LBM_DATA>
	__cuda_callable__ static void zeroForces(LBM_DATA& SD, idx x, idx y, idx z)
	{}

	template <typename LBM_BC, typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void computeForcing(LBM_DATA& SD, LBM_KS& KS, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm, idx z, idx zp)
	{}
};

template <typename TRAITS>
struct D3Q27_MACRO_Default : D3Q27_MACRO_Base<TRAITS>
{
	using dreal = typename TRAITS::dreal;
	using idx = typename TRAITS::idx;

	enum QuantityNames : std::uint8_t
	{
		e_rho,
		e_vx,
		e_vy,
		e_vz,
		N
	};

	template <typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void outputMacro(LBM_DATA& SD, LBM_KS& KS, idx x, idx y, idx z)
	{
		SD.macro(e_rho, x, y, z) = KS.rho;
		SD.macro(e_vx, x, y, z) = KS.vx;
		SD.macro(e_vy, x, y, z) = KS.vy;
		SD.macro(e_vz, x, y, z) = KS.vz;
	}

	template <typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void copyQuantities(LBM_DATA& SD, LBM_KS& KS, idx x, idx y, idx z)
	{
		KS.lbmViscosity = SD.lbmViscosity;
		KS.fx = SD.fx;
		KS.fy = SD.fy;
		KS.fz = SD.fz;
	}
};

template <typename TRAITS>
struct D3Q27_MACRO_Mean : D3Q27_MACRO_Base<TRAITS>
{
	using dreal = typename TRAITS::dreal;
	using idx = typename TRAITS::idx;

	enum QuantityNames : std::uint8_t
	{
		e_rho,
		e_vx,
		e_vy,
		e_vz,
		e_vm_x,
		e_vm_y,
		e_vm_z,
		e_vm2_xx,
		e_vm2_yy,
		e_vm2_zz,
		e_vm2_xy,
		e_vm2_xz,
		e_vm2_yz,
		N
	};

	// specifies if macroscopic quantities are computed in the kernel in each iteration
	static const bool compute_in_each_iteration = true;

	template <typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void outputMacro(LBM_DATA& SD, LBM_KS& KS, idx x, idx y, idx z)
	{
		// Instant quantities
		SD.macro(e_rho, x, y, z) = KS.rho;
		SD.macro(e_vx, x, y, z) = KS.vx;
		SD.macro(e_vy, x, y, z) = KS.vy;
		SD.macro(e_vz, x, y, z) = KS.vz;

		// Mean quantities and (co)variance

		// We use a simple moving average algorithm in the form of
		// M_mean[n+1] = M_mean[n] + (M[n+1] - M_mean[n]) / (n+1)
		const dreal denominator = dreal(1) / dreal(SD.stat_counter + 1);
		const dreal vm_x_old = SD.macro(e_vm_x, x, y, z);
		const dreal vm_y_old = SD.macro(e_vm_y, x, y, z);
		const dreal vm_z_old = SD.macro(e_vm_z, x, y, z);
		const dreal delta_x = KS.vx - vm_x_old;
		const dreal delta_y = KS.vy - vm_y_old;
		const dreal delta_z = KS.vz - vm_z_old;
		const dreal vm_x_new = vm_x_old + delta_x * denominator;
		const dreal vm_y_new = vm_y_old + delta_y * denominator;
		const dreal vm_z_new = vm_z_old + delta_z * denominator;

		// We use a Welford-like online algorithm for computing the covariance
		// based on https://doi.org/10.1145/3221269.3223036
		// S_ab[n+1] = S_ab[n] + (v_a - vm_a_new) * (v_b - vm_b_old)
		// then Cov(a,b) = S_ab[n+1] / (n+1) and Var(a) = Cov(a,a)
		const dreal vm2_xx_old = SD.macro(e_vm2_xx, x, y, z);
		const dreal vm2_yy_old = SD.macro(e_vm2_yy, x, y, z);
		const dreal vm2_zz_old = SD.macro(e_vm2_zz, x, y, z);
		const dreal vm2_xy_old = SD.macro(e_vm2_xy, x, y, z);
		const dreal vm2_xz_old = SD.macro(e_vm2_xz, x, y, z);
		const dreal vm2_yz_old = SD.macro(e_vm2_yz, x, y, z);
		const dreal delta_new_x = KS.vx - vm_x_new;
		const dreal delta_new_y = KS.vy - vm_y_new;
		const dreal delta_new_z = KS.vz - vm_z_new;
		const dreal vm2_xx_new = vm2_xx_old + delta_new_x * delta_x;
		const dreal vm2_yy_new = vm2_yy_old + delta_new_y * delta_y;
		const dreal vm2_zz_new = vm2_zz_old + delta_new_z * delta_z;
		const dreal vm2_xy_new = vm2_xy_old + delta_new_x * delta_y;
		const dreal vm2_xz_new = vm2_xz_old + delta_new_x * delta_z;
		const dreal vm2_yz_new = vm2_yz_old + delta_new_y * delta_z;

		// write all results
		SD.macro(e_vm_x, x, y, z) = vm_x_new;
		SD.macro(e_vm_y, x, y, z) = vm_y_new;
		SD.macro(e_vm_z, x, y, z) = vm_z_new;
		SD.macro(e_vm2_xx, x, y, z) = vm2_xx_new;
		SD.macro(e_vm2_yy, x, y, z) = vm2_yy_new;
		SD.macro(e_vm2_zz, x, y, z) = vm2_zz_new;
		SD.macro(e_vm2_xy, x, y, z) = vm2_xy_new;
		SD.macro(e_vm2_xz, x, y, z) = vm2_xz_new;
		SD.macro(e_vm2_yz, x, y, z) = vm2_yz_new;
	}

	template <typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void copyQuantities(LBM_DATA& SD, LBM_KS& KS, idx x, idx y, idx z)
	{
		KS.lbmViscosity = SD.lbmViscosity;
		KS.fx = SD.fx;
		KS.fy = SD.fy;
		KS.fz = SD.fz;
	}
};

template <typename TRAITS>
struct D3Q27_MACRO_Void : D3Q27_MACRO_Base<TRAITS>
{
	using dreal = typename TRAITS::dreal;
	using idx = typename TRAITS::idx;

	static const int N = 0;

	template <typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void outputMacro(LBM_DATA& SD, LBM_KS& KS, idx x, idx y, idx z)
	{}

	template <typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void copyQuantities(LBM_DATA& SD, LBM_KS& KS, idx x, idx y, idx z)
	{}
};

// For adjoint simulations
// contains all measured macro quantities and all macro quantities from the primary problem
template <typename TRAITS>
struct D3Q27_MACRO_Adjoint : D3Q27_MACRO_Base<TRAITS>
{
	using dreal = typename TRAITS::dreal;
	using idx = typename TRAITS::idx;

	enum QuantityNames : std::uint8_t
	{
		// NOTE: primary macros must be first
		e_rho,
		e_vx,
		e_vy,
		e_vz,
		// NOTE: measured macros must be the second half
		e_rho_m,
		e_vx_m,
		e_vy_m,
		e_vz_m,
		N
		// NOTE: if anything more is added, the loadPrimaryAndMeasuredMacro function must be generalized!!!
		//gx, gy, gz,
	};

	// specifies if macroscopic quantities are computed in the kernel in each iteration
	static const bool compute_in_each_iteration = true;

	template <typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void outputMacro(LBM_DATA& SD, LBM_KS& KS, idx x, idx y, idx z)
	{
		// NOTE: this is a test
		//SD.loss_function += (dreal)0.5 * (dreal)std::sqrt(((KS.rho-KS.rho_m)*(KS.rho-KS.rho_m) + (KS.vx-KS.vx_m)*(KS.vx-KS.vx_m) +
		//(KS.vy-KS.vy_m)*(KS.vy-KS.vy_m) + (KS.vz-KS.vz_m)*(KS.vz-KS.vz_m)));

		//! otherwise empty, because macro is loaded from primary problem and measured data, not changed in computeInitMacro (lbm.h) or SimUpdate
		//! (state.h) - outputting from KS to SD means overwriting the loaded data
	}

	template <typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void copyLoadedMacro(LBM_DATA& SD, LBM_KS& KS, idx x, idx y, idx z)
	{
		KS.rho = SD.macro(e_rho, x, y, z);
		KS.vx = SD.macro(e_vx, x, y, z);
		KS.vy = SD.macro(e_vy, x, y, z);
		KS.vz = SD.macro(e_vz, x, y, z);

		KS.rho_m = SD.macro(e_rho_m, x, y, z);
		KS.vx_m = SD.macro(e_vx_m, x, y, z);
		KS.vy_m = SD.macro(e_vy_m, x, y, z);
		KS.vz_m = SD.macro(e_vz_m, x, y, z);
	}

	template <typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void copyQuantities(LBM_DATA& SD, LBM_KS& KS, idx x, idx y, idx z)
	{
		KS.lbmViscosity = SD.lbmViscosity;
		KS.fx = SD.fx;
		KS.fy = SD.fy;
		KS.fz = SD.fz;
		copyLoadedMacro(SD, KS, x, y, z);  // copies loaded data to kernel, important for adjoint collisionWithData
	}
};
