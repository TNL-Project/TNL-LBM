#pragma once

#include "defs.h"
#include "lbm_common/ciselnik.h"

// only a base type - common for all D3Q* models, cannot be used directly
template <typename TRAITS>
struct LBM_Data
{
	using idx = typename TRAITS::idx;
	using dreal = typename TRAITS::dreal;
	using map_t = typename TRAITS::map_t;
	using indexer_t = typename TRAITS::lattice_indexer_t;

	// even/odd iteration indicator for the A-A pattern
	bool even_iter = true;

	// indexing
	indexer_t indexer;
	idx XYZ;  // precomputed indexer.getStorageSize(), i.e. product of (X+overlaps_x)*(Y+overlaps_y)*(Z+overlaps_z)

	// scalars
	dreal lbmViscosity;
	int stat_counter = 0;  // counter for computing mean quantities in D3Q27_MACRO_Mean - must be set in StateLocal::updateKernelVelocities

	// array pointers
	dreal* dfs[DFMAX];
	dreal* dmacro;
	map_t* dmap;

	// sizes NOT including overlaps
	CUDA_HOSTDEV idx X()
	{
		return indexer.template getSize<0>();
	}
	CUDA_HOSTDEV idx Y()
	{
		return indexer.template getSize<1>();
	}
	CUDA_HOSTDEV idx Z()
	{
		return indexer.template getSize<2>();
	}

	CUDA_HOSTDEV map_t map(idx x, idx y, idx z)
	{
		return dmap[indexer.getStorageIndex(x, y, z)];
	}

	CUDA_HOSTDEV dreal& df(uint8_t type, int q, idx x, idx y, idx z)
	{
		const idx index = q * XYZ + indexer.getStorageIndex(x, y, z);
		return dfs[type][index];
	}

	CUDA_HOSTDEV dreal& macro(int id, idx x, idx y, idx z)
	{
		const idx index = id * XYZ + indexer.getStorageIndex(x, y, z);
		return dmacro[index];
	}
};

// base type for all NSE_Data_* types
template <typename TRAITS>
struct NSE_Data : LBM_Data<TRAITS>
{
	using dreal = typename LBM_Data<TRAITS>::dreal;

	// homogeneous force field
	dreal fx = 0;
	dreal fy = 0;
	dreal fz = 0;
};

template <typename TRAITS>
struct NSE_Data_ConstInflow : NSE_Data<TRAITS>
{
	using idx = typename TRAITS::idx;
	using dreal = typename TRAITS::dreal;

	dreal inflow_vx = 0;
	dreal inflow_vy = 0;
	dreal inflow_vz = 0;

	template <typename LBM_KS>
	CUDA_HOSTDEV void inflow(LBM_KS& KS, idx x, idx y, idx z)
	{
		KS.vx = inflow_vx;
		KS.vy = inflow_vy;
		KS.vz = inflow_vz;
	}
};

template <typename TRAITS>
struct NSE_Data_InflowProfile : NSE_Data<TRAITS>
{
	using idx = typename TRAITS::idx;
	using dreal = typename TRAITS::dreal;

	// arrays for inflow velocity profile
	dreal* inflow_vx = nullptr;
	dreal* inflow_vy = nullptr;
	dreal* inflow_vz = nullptr;

	CUDA_HOSTDEV dreal vel_x(idx y, idx z)
	{
		if (inflow_vx == nullptr)
			return 0;
		return inflow_vx[this->Y() * z + y];
	}

	CUDA_HOSTDEV dreal vel_y(idx y, idx z)
	{
		if (inflow_vy == nullptr)
			return 0;
		return inflow_vy[this->Y() * z + y];
	}

	CUDA_HOSTDEV dreal vel_z(idx y, idx z)
	{
		if (inflow_vz == nullptr)
			return 0;
		return inflow_vz[this->Y() * z + y];
	}

	template <typename LBM_KS>
	CUDA_HOSTDEV void inflow(LBM_KS& KS, idx x, idx y, idx z)
	{
		KS.vx = vel_x(y, z);
		KS.vy = vel_y(y, z);
		KS.vz = vel_z(y, z);
	}
};

template <typename TRAITS>
struct NSE_Data_Adjoint : NSE_Data<TRAITS>
{
	using idx = typename TRAITS::idx;
	using dreal = typename TRAITS::dreal;

	dreal* vx_profile = nullptr;
	dreal* vy_profile = nullptr;
	dreal* vz_profile = nullptr;
	dreal* gx_profile = nullptr;
	dreal* gy_profile = nullptr;
	dreal* gz_profile = nullptr;
	dreal* vx_profile_result = nullptr;
	dreal* vy_profile_result = nullptr;
	dreal* vz_profile_result = nullptr;

	bool* b_profile = nullptr;

	dreal eps = 0;

	dreal loss_function = 0;

	template <typename LBM_KS>
	CUDA_HOSTDEV void inflow(LBM_KS& KS, idx x, idx y, idx z)
	{
		if (! b_profile[this->Y() * z + y]) {
			KS.vx = vx_profile[this->Y() * z + y];
			KS.vy = vy_profile[this->Y() * z + y];
			KS.vz = vz_profile[this->Y() * z + y];
			b_profile[this->Y() * z + y] = true;
		}
		else {
			// clang-format off
			/*
			gx_profile[this->Y() * z + y] = - KS.f[pzz] * no2 * n2o27  * KS.rho * no3
											- KS.f[ppz] * no2 * n1o54  * KS.rho * no3
											- KS.f[pmz] * no2 * n1o54  * KS.rho * no3
											- KS.f[pzp] * no2 * n1o54  * KS.rho * no3
											- KS.f[pzm] * no2 * n1o54  * KS.rho * no3
											- KS.f[ppp] * no2 * n1o216 * KS.rho * no3
											- KS.f[ppm] * no2 * n1o216 * KS.rho * no3
											- KS.f[pmp] * no2 * n1o216 * KS.rho * no3
											- KS.f[pmm] * no2 * n1o216 * KS.rho * no3;
			vx_profile_result[this->Y() * z + y] += eps * gx_profile[this->Y() * z + y];
			*/
			gy_profile[this->Y() * z + y] = - KS.f[ppz] * no2 * n1o54 * KS.rho * no3
											+ KS.f[pmz] * no2 * n1o54 * KS.rho * no3
											- KS.f[ppp] * no2 * n1o216 * KS.rho * no3
											- KS.f[ppm] * no2 * n1o216 * KS.rho * no3
											+ KS.f[pmp] * no2 * n1o216 * KS.rho * no3
											+ KS.f[pmm] * no2 * n1o216 * KS.rho * no3;
			vy_profile_result[this->Y() * z + y] += eps * gy_profile[this->Y() * z + y];
			/*
			gz_profile[this->Y() * z + y] = - KS.f[pzp] * no2 * n1o54  * KS.rho * no3
											+ KS.f[pzm] * no2 * n1o54  * KS.rho * no3
											- KS.f[ppp] * no2 * n1o216 * KS.rho * no3
											+ KS.f[ppm] * no2 * n1o216 * KS.rho * no3
											- KS.f[pmp] * no2 * n1o216 * KS.rho * no3
											+ KS.f[pmm] * no2 * n1o216 * KS.rho * no3;
			vz_profile_result[this->Y() * z + y] += eps * gz_profile[this->Y() * z + y];
			*/
			b_profile[this->Y() * z + y] = false;
			// clang-format on
		}
		/*
		#ifdef USE_CUDA
		atomicAdd(&loss_function, (dreal)0.5 * (dreal)std::sqrt(((KS.rho-KS.rho_m)*(KS.rho-KS.rho_m) + (KS.vx-KS.vx_m)*(KS.vx-KS.vx_m) +
		(KS.vy-KS.vy_m)*(KS.vy-KS.vy_m) + (KS.vz-KS.vz_m)*(KS.vz-KS.vz_m))));
		#else
		loss_function += (dreal)0.5 * (dreal)std::sqrt(((KS.rho-KS.rho_m)*(KS.rho-KS.rho_m) + (KS.vx-KS.vx_m)*(KS.vx-KS.vx_m) +
		(KS.vy-KS.vy_m)*(KS.vy-KS.vy_m) + (KS.vz-KS.vz_m)*(KS.vz-KS.vz_m)));
		#endif
		*/
	}
};

// base type for all ADE_Data_* types
template <typename TRAITS>
struct ADE_Data : LBM_Data<TRAITS>
{
	using idx = typename LBM_Data<TRAITS>::idx;
	using dreal = typename LBM_Data<TRAITS>::dreal;

	// pointer for the variable diffusion coefficient array
	// (can be nullptr in which case it is unused and the lbmViscosity
	// scalar is used instead)
	dreal* diffusion_coefficient_ptr = nullptr;

	// TODO: documentation
	bool* phi_transfer_direction_ptr = nullptr;

	// coefficient for the GEO_TRANSFER_FS and GEO_TRANSFER_SF boundary conditions
	dreal phiTransferCoefficient = 0;

	// TODO: source term on the rhs of the ADE

	CUDA_HOSTDEV dreal diffusionCoefficient(idx x, idx y, idx z)
	{
		if (diffusion_coefficient_ptr == nullptr)
			return this->lbmViscosity;
		else {
			const idx index = this->indexer.getStorageIndex(x, y, z);
			return diffusion_coefficient_ptr[index];
		}
	}

	CUDA_HOSTDEV bool& phiTransferDirection(int q, idx x, idx y, idx z)
	{
		const idx index = q * this->XYZ + this->indexer.getStorageIndex(x, y, z);
		return phi_transfer_direction_ptr[index];
	}
};

template <typename TRAITS>
struct ADE_Data_ConstInflow : ADE_Data<TRAITS>
{
	using idx = typename TRAITS::idx;
	using dreal = typename TRAITS::dreal;

	dreal inflow_phi = 1;

	template <typename LBM_KS>
	CUDA_HOSTDEV void inflow(LBM_KS& KS, idx x, idx y, idx z)
	{
		KS.phi = inflow_phi;
	}
};
