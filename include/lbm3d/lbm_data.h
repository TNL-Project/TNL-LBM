#pragma once

#include "defs.h"

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

	CUDA_HOSTDEV idx Fxyz(int q, idx x, idx y, idx z)
	{
		return q * XYZ + indexer.getStorageIndex(x, y, z);
	}

	CUDA_HOSTDEV dreal& df(uint8_t type, int q, idx x, idx y, idx z)
	{
		return dfs[type][Fxyz(q, x, y, z)];
	}

	CUDA_HOSTDEV dreal& macro(int id, idx x, idx y, idx z)
	{
		return dmacro[Fxyz(id, x, y, z)];
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
		else
			return diffusion_coefficient_ptr[LBM_Data<TRAITS>::indexer.getStorageIndex(x, y, z)];
	}

	CUDA_HOSTDEV bool& phiTransferDirection(int q, idx x, idx y, idx z)
	{
		return phi_transfer_direction_ptr[LBM_Data<TRAITS>::Fxyz(q, x, y, z)];
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
