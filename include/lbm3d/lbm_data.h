#pragma once

#include "defs.h"
#include "state.h"

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

template <typename TRAITS>
struct NSE_Data_ConstInflow_PressureGradient : NSE_Data<TRAITS>
{
	using idx = typename TRAITS::idx;
	using dreal = typename TRAITS::dreal;

	dreal inflow_vx = 0;
	dreal inflow_vy = 0;
	dreal inflow_vz = 0;
	dreal inflow_g = 0.;
	dreal no1oT0 = 3; // set to 1./0.6979533220196830882384091; // FOR D2Q49


	template <typename LBM_KS>
	CUDA_HOSTDEV void inflow(LBM_KS& KS, idx x, idx y, idx z)
	{
		KS.rho += no1oT0*inflow_g;
		KS.vx = inflow_vx;
		KS.vy = inflow_vy;
		KS.vz = inflow_vz;
	}
};

template <typename TRAITS>
struct NSE_Data_Analytical_Solution : NSE_Data<TRAITS>
{
	using idx = typename TRAITS::idx;
	using dreal = typename TRAITS::dreal;

	dreal inflow_vx = 0;
	dreal inflow_vy = 0;
	dreal inflow_vz = 0;
	dreal a = 0; // size of channel
    dreal inflow_g=0;
    dreal InitPoint [3];

	template <typename LBM_KS>
	CUDA_HOSTDEV void inflow(LBM_KS& KS, idx x, idx y, idx z)
	{
		KS.vx = inflow_vx;
		KS.vy = inflow_vy;
		KS.vz = inflow_vz;
	}

	CUDA_HOSTDEV void sum_to_order(dreal a, dreal x, dreal y, int order){
		dreal sum = 0;
		const dreal PI = 3.1415926;
		for(int m = 0;m<order;m++){
		for(int n = 0;n<order;n++){
			sum += coefficient_of_sum(a, x, y, m, n);
		}}
		return -4*a*a/pow(PI,4)*sum;
	}

	CUDA_HOSTDEV void coefficient_of_sum(dreal a, dreal x, dreal y, int m, int n){
		const dreal PI = 3.1415926; // somehow does not work with #define because of conflicting definition
		return sin(1.*m/a*PI)*sin(1.*n/a*PI)/m/n/(m*m+n*n);
	}
};

template <typename TRAITS>
struct NSE_Data_NoInflow : NSE_Data<TRAITS>
{
	using idx = typename TRAITS::idx;
	using dreal = typename TRAITS::dreal;

	template <typename LBM_KS>
	CUDA_HOSTDEV void inflow(LBM_KS& KS, idx x, idx y, idx z)
	{
		KS.rho = 1;
		KS.vx = 0;
		KS.vy = 0;
		KS.vz = 0;
	}
};

template < typename TRAITS>
struct NSE_Data_Parabolic_yconst : NSE_Data<TRAITS>
{
	using idx = typename TRAITS::idx;
	using dreal = typename TRAITS::dreal;
	using point_t = typename TRAITS::point_t;

	dreal inflow_vx = 0;
	dreal inflow_vy = 0;
	dreal inflow_vz = 0;
	dreal inflow_z  = 0;
    dreal inflow_g  = 0;
    point_t InitPoint; // non-dimesional initial point
	dreal no1oT0 = 3;

	template <typename LBM_KS>
	CUDA_HOSTDEV void inflow(LBM_KS& KS, idx x, idx y, idx z)
	{
		KS.rho +=  no1oT0*inflow_g;
		KS.vx  =   4*inflow_vx*(inflow_z*inflow_z*0.25-((z + InitPoint[3])-0.5*inflow_z)*((z + InitPoint[3])-0.5*inflow_z))/(inflow_z*inflow_z);
		KS.vy  =   0.;
		KS.vz  =   0.;
	}
};

template < typename TRAITS>
struct NSE_Data_DoubleParabolic : NSE_Data<TRAITS>
{
	using idx = typename TRAITS::idx;
	using dreal = typename TRAITS::dreal;
	using point_t = typename TRAITS::point_t;

	dreal inflow_vx = 0;
	dreal inflow_vy = 0;
	dreal inflow_vz = 0;
	dreal inflow_y  = 0;
	dreal inflow_z  = 0;
    dreal inflow_g  = 0;
    point_t InitPoint; // non-dimesional initial point
	dreal no1oT0 = 3;

	template <typename LBM_KS>
	CUDA_HOSTDEV void inflow(LBM_KS& KS, idx x, idx y, idx z)
	{
		KS.rho +=  no1oT0*inflow_g;
		KS.vx  =   16*inflow_vx*
			(inflow_z*inflow_z*0.25-((z + InitPoint[3])-0.5*inflow_z)*((z + InitPoint[3])-0.5*inflow_z))/(inflow_z*inflow_z)*
			(inflow_y*inflow_y*0.25-((y + InitPoint[2])-0.5*inflow_y)*((y + InitPoint[2])-0.5*inflow_y))/(inflow_y*inflow_y);
		KS.vy  =   0.;
		KS.vz  =   0.;
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
