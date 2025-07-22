#pragma once

#include <TNL/Backend/Macros.h>

// empty Macro containing required forcing quantities for IBM (see lbm.h -> hfx() etc.)
template <typename TRAITS>
struct D2Q9_MACRO_Base
{
	using dreal = typename TRAITS::dreal;
	using idx = typename TRAITS::idx;

	// all quantities after `N` are ignored
	enum
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

	static const bool use_syncMacro = false;

	// maximum width of overlaps for the macro arrays
	static constexpr int overlap_width = 1;

	// compulsory method -- called from cudaLBMComputeVelocitiesStarAndZeroForce kernel
	template <typename LBM_KS>
	__cuda_callable__ static void zeroForcesInKS(LBM_KS& KS)
	{
		KS.fx = 0;
		KS.fy = 0;
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
struct D2Q9_MACRO_Default : D2Q9_MACRO_Base<TRAITS>
{
	using dreal = typename TRAITS::dreal;
	using idx = typename TRAITS::idx;

	enum
	{
		e_rho,
		e_vx,
		e_vy,
		N
	};

	template <typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void outputMacro(LBM_DATA& SD, LBM_KS& KS, idx x, idx y, idx z)
	{
		SD.macro(e_rho, x, y, z) = KS.rho;
		SD.macro(e_vx, x, y, z) = KS.vx;
		SD.macro(e_vy, x, y, z) = KS.vy;
	}

	template <typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void copyQuantities(LBM_DATA& SD, LBM_KS& KS, idx x, idx y, idx z)
	{
		KS.lbmViscosity = SD.lbmViscosity;
		KS.fx = SD.fx;
		KS.fy = SD.fy;
	}
};

template <typename TRAITS>
struct D2Q9_MACRO_Mean : D2Q9_MACRO_Base<TRAITS>
{
	using dreal = typename TRAITS::dreal;
	using idx = typename TRAITS::idx;

	enum
	{
		e_rho,
		e_vx,
		e_vy,
		e_vm_x,
		e_vm_y,
		e_vm2_xx,
		e_vm2_yy,
		e_vm2_xy,
		N
	};

	template <typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void outputMacro(LBM_DATA& SD, LBM_KS& KS, idx x, idx y, idx z)
	{
		// Instant quantities
		SD.macro(e_rho, x, y, z) = KS.rho;
		SD.macro(e_vx, x, y, z) = KS.vx;
		SD.macro(e_vy, x, y, z) = KS.vy;

		// Mean quantities and (co)variance

		// We use a simple moving average algorithm in the form of
		// M_mean[n+1] = M_mean[n] + (M[n+1] - M_mean[n]) / (n+1)
		const dreal denominator = dreal(1) / dreal(SD.stat_counter + 1);
		const dreal vm_x_old = SD.macro(e_vm_x, x, y, z);
		const dreal vm_y_old = SD.macro(e_vm_y, x, y, z);
		const dreal delta_x = KS.vx - vm_x_old;
		const dreal delta_y = KS.vy - vm_y_old;
		const dreal vm_x_new = vm_x_old + delta_x * denominator;
		const dreal vm_y_new = vm_y_old + delta_y * denominator;

		// We use a Welford-like online algorithm for computing the covariance
		// based on https://doi.org/10.1145/3221269.3223036
		// S_ab[n+1] = S_ab[n] + (v_a - vm_a_new) * (v_b - vm_b_old)
		// then Cov(a,b) = S_ab[n+1] / (n+1) and Var(a) = Cov(a,a)
		const dreal vm2_xx_old = SD.macro(e_vm2_xx, x, y, z);
		const dreal vm2_yy_old = SD.macro(e_vm2_yy, x, y, z);
		const dreal vm2_xy_old = SD.macro(e_vm2_xy, x, y, z);
		const dreal delta_new_x = KS.vx - vm_x_new;
		const dreal delta_new_y = KS.vy - vm_y_new;
		const dreal vm2_xx_new = vm2_xx_old + delta_new_x * delta_x;
		const dreal vm2_yy_new = vm2_yy_old + delta_new_y * delta_y;
		const dreal vm2_xy_new = vm2_xy_old + delta_new_x * delta_y;

		// write all results
		SD.macro(e_vm_x, x, y, z) = vm_x_new;
		SD.macro(e_vm_y, x, y, z) = vm_y_new;
		SD.macro(e_vm2_xx, x, y, z) = vm2_xx_new;
		SD.macro(e_vm2_yy, x, y, z) = vm2_yy_new;
		SD.macro(e_vm2_xy, x, y, z) = vm2_xy_new;
	}

	template <typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void copyQuantities(LBM_DATA& SD, LBM_KS& KS, idx x, idx y, idx z)
	{
		KS.lbmViscosity = SD.lbmViscosity;
		KS.fx = SD.fx;
		KS.fy = SD.fy;
	}
};

template <typename TRAITS>
struct D2Q9_MACRO_Void : D2Q9_MACRO_Base<TRAITS>
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
