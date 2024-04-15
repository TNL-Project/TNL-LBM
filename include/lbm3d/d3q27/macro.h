#pragma once

// empty Macro containing required forcing quantities for IBM (see lbm.h -> hfx() etc.)
template < typename TRAITS >
struct D3Q27_MACRO_Base
{
	using dreal = typename TRAITS::dreal;
	using idx = typename TRAITS::idx;

	// all quantities after `N` are ignored
	enum { N, e_rho, e_vx, e_vy, e_vz, e_fx, e_fy, e_fz };

	static const bool use_syncMacro = false;

	// called from LBMKernelInit
	template < typename LBM_KS >
	CUDA_HOSTDEV static void zeroForcesInKS(LBM_KS &KS)
	{
		KS.fx = 0;
		KS.fy = 0;
		KS.fz = 0;
	}

	// compulsory method -- called from cudaLBMComputeVelocitiesStarAndZeroForce kernel
	template < typename LBM_DATA >
	CUDA_HOSTDEV static void zeroForces(LBM_DATA &SD, idx x, idx y, idx z) {}

	template < typename LBM_BC, typename LBM_DATA, typename LBM_KS >
	CUDA_HOSTDEV static void computeForcing(LBM_DATA &SD, LBM_KS &KS, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm, idx z, idx zp) {}
};


template < typename TRAITS >
struct D3Q27_MACRO_Default : D3Q27_MACRO_Base< TRAITS >
{
	using dreal = typename TRAITS::dreal;
	using idx = typename TRAITS::idx;

	enum { e_rho, e_vx, e_vy, e_vz, N, e_vmx, e_vmy, e_vmz };

	template < typename LBM_DATA, typename LBM_KS >
	CUDA_HOSTDEV static void outputMacro(LBM_DATA &SD, LBM_KS &KS, idx x, idx y, idx z)
	{
		SD.macro(e_rho, x, y, z) = KS.rho;
		SD.macro(e_vx, x, y, z)  = KS.vx;
		SD.macro(e_vy, x, y, z)  = KS.vy;
		SD.macro(e_vz, x, y, z)  = KS.vz;
		SD.macro(e_vmx, x, y, z) += KS.vx;
		SD.macro(e_vmy, x, y, z) += KS.vy;
		SD.macro(e_vmz, x, y, z) += KS.vz;
	}

	template < typename LBM_DATA, typename LBM_KS >
	CUDA_HOSTDEV static void copyQuantities(LBM_DATA &SD, LBM_KS &KS, idx x, idx y, idx z)
	{
		KS.lbmViscosity = SD.lbmViscosity;
		KS.fx = SD.fx;
		KS.fy = SD.fy;
		KS.fz = SD.fz;
	}
};


template < typename TRAITS >
struct D3Q27_MACRO_Void : D3Q27_MACRO_Base< TRAITS >
{
	using dreal = typename TRAITS::dreal;
	using idx = typename TRAITS::idx;

	static const int N=0;

	template < typename LBM_DATA, typename LBM_KS >
	CUDA_HOSTDEV static void outputMacro(LBM_DATA &SD, LBM_KS &KS, idx x, idx y, idx z)
	{
	}

	template < typename LBM_DATA, typename LBM_KS >
	CUDA_HOSTDEV static void copyQuantities(LBM_DATA &SD, LBM_KS &KS, idx x, idx y, idx z)
	{
	}
};
