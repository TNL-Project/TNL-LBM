//#define AB_PATTERN

#include <argparse/argparse.hpp>

#include "lbm3d/core.h"
#include "lbm3d/d3q7/eq.h"
#include "lbm3d/d3q7/col_srt.h"
#include "lbm3d/d3q7/col_mrt.h"
#include "lbm3d/d3q7/col_clbm.h"
// exactly one streaming header must be included
#ifdef AA_PATTERN
	#include "lbm3d/d3q7/streaming_AA.h"
#endif
#ifdef AB_PATTERN
	#include "lbm3d/d3q7/streaming_AB.h"
#endif
#include "lbm3d/d3q7/bc.h"
#include "lbm3d/d3q7/macro.h"
#include "lbm3d/state_NSE_ADE.h"
#include "lbm3d/obstacles_lbm.h"

template <typename TRAITS>
struct NSE_Data_FreeRhoConstInflow : NSE_Data<TRAITS>
{
	using dreal = typename TRAITS::dreal;
	using idx = typename TRAITS::idx;

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

#if 0
template < typename NSE >
	#ifdef USE_CUDA
__global__ void cudaLBMComputeQCriterion(
	typename NSE::DATA SD,
	short int rank,
	short int nproc
)
	#else
void LBMComputeQCriterion(
	typename NSE::DATA SD,
	typename NSE::TRAITS::idx x,
	typename NSE::TRAITS::idx y,
	typename NSE::TRAITS::idx z,
	short int rank,
	short int nproc
)
	#endif
{
	using dreal = typename NSE::TRAITS::dreal;
	using idx = typename NSE::TRAITS::idx;
	using map_t = typename NSE::TRAITS::map_t;

	#ifdef USE_CUDA
	idx x = threadIdx.x + blockIdx.x * blockDim.x;
	idx y = threadIdx.y + blockIdx.y * blockDim.y;
	idx z = threadIdx.z + blockIdx.z * blockDim.z;
	#endif
	map_t gi_map = SD.map(x, y, z);

	idx xp,xm,yp,ym,zp,zm;
	if (NSE::BC::isPeriodic(gi_map))
	{
		// handle overlaps between GPUs
		//xp = (!SD.overlap_right && x == SD.X-1) ? 0 : (x+1);
		//xm = (!SD.overlap_left && x == 0) ? (SD.X-1) : (x-1);
		xp = (nproc == 1 && x == SD.X()-1) ? 0 : (x+1);
		xm = (nproc == 1 && x == 0) ? (SD.X()-1) : (x-1);
		yp = (y == SD.Y()-1) ? 0 : (y+1);
		ym = (y == 0) ? (SD.Y()-1) : (y-1);
		zp = (z == SD.Z()-1) ? 0 : (z+1);
		zm = (z == 0) ? (SD.Z()-1) : (z-1);
	} else {
		// handle overlaps between GPUs
		// NOTE: ghost layers of lattice sites are assumed in the x-direction, so x+1 and x-1 always work
		xp = x+1;
		xm = x-1;
		yp = TNL::min(y+1, SD.Y()-1);
		ym = TNL::max(y-1,0);
		zp = TNL::min(z+1, SD.Z()-1);
		zm = TNL::max(z-1,0);
	}

		struct Tensor
		{
			dreal xx=0,xy=0,xz=0;
			dreal yx=0,yy=0,yz=0;
			dreal zx=0,zy=0,zz=0;
		};
		Tensor G;

		// grad vel tensor
		if (y == 0 || z == 0 || y == SD.Y()-1 || z == SD.Z()-1)
		{
			// do nothing
			G.xx = 0;
			G.xy = 0;
			G.xz = 0;
			G.yx = 0;
			G.yy = 0;
			G.yz = 0;
			G.zx = 0;
			G.zy = 0;
			G.zz = 0;
		}
		else if (x == 0) {
			// forward difference for x
			G.xx = SD.macro(NSE::MACRO::e_vx,xp,y,z) - SD.macro(NSE::MACRO::e_vx,x,y,z);
			G.yx = SD.macro(NSE::MACRO::e_vy,xp,y,z) - SD.macro(NSE::MACRO::e_vy,x,y,z);
			G.zx = SD.macro(NSE::MACRO::e_vz,xp,y,z) - SD.macro(NSE::MACRO::e_vz,x,y,z);
			// central differences for y,z
			G.xy = n1o2 * (SD.macro(NSE::MACRO::e_vx,x,yp,z) - SD.macro(NSE::MACRO::e_vx,x,ym,z));
			G.yy = n1o2 * (SD.macro(NSE::MACRO::e_vy,x,yp,z) - SD.macro(NSE::MACRO::e_vy,x,ym,z));
			G.zy = n1o2 * (SD.macro(NSE::MACRO::e_vz,x,yp,z) - SD.macro(NSE::MACRO::e_vz,x,ym,z));
			G.xz = n1o2 * (SD.macro(NSE::MACRO::e_vx,x,y,zp) - SD.macro(NSE::MACRO::e_vx,x,y,zm));
			G.yz = n1o2 * (SD.macro(NSE::MACRO::e_vy,x,y,zp) - SD.macro(NSE::MACRO::e_vy,x,y,zm));
			G.zz = n1o2 * (SD.macro(NSE::MACRO::e_vz,x,y,zp) - SD.macro(NSE::MACRO::e_vz,x,y,zm));
		}
		else if (x == SD.X()-1) {
			// backward difference for x
			G.xx = SD.macro(NSE::MACRO::e_vx,x,y,z) - SD.macro(NSE::MACRO::e_vx,xm,y,z);
			G.yx = SD.macro(NSE::MACRO::e_vy,x,y,z) - SD.macro(NSE::MACRO::e_vy,xm,y,z);
			G.zx = SD.macro(NSE::MACRO::e_vz,x,y,z) - SD.macro(NSE::MACRO::e_vz,xm,y,z);
			// central differences for y,z
			G.xy = n1o2 * (SD.macro(NSE::MACRO::e_vx,x,yp,z) - SD.macro(NSE::MACRO::e_vx,x,ym,z));
			G.yy = n1o2 * (SD.macro(NSE::MACRO::e_vy,x,yp,z) - SD.macro(NSE::MACRO::e_vy,x,ym,z));
			G.zy = n1o2 * (SD.macro(NSE::MACRO::e_vz,x,yp,z) - SD.macro(NSE::MACRO::e_vz,x,ym,z));
			G.xz = n1o2 * (SD.macro(NSE::MACRO::e_vx,x,y,zp) - SD.macro(NSE::MACRO::e_vx,x,y,zm));
			G.yz = n1o2 * (SD.macro(NSE::MACRO::e_vy,x,y,zp) - SD.macro(NSE::MACRO::e_vy,x,y,zm));
			G.zz = n1o2 * (SD.macro(NSE::MACRO::e_vz,x,y,zp) - SD.macro(NSE::MACRO::e_vz,x,y,zm));
		}
		else {
			// central differences
			G.xx = n1o2 * (SD.macro(NSE::MACRO::e_vx,xp,y,z) - SD.macro(NSE::MACRO::e_vx,xm,y,z));
			G.xy = n1o2 * (SD.macro(NSE::MACRO::e_vx,x,yp,z) - SD.macro(NSE::MACRO::e_vx,x,ym,z));
			G.xz = n1o2 * (SD.macro(NSE::MACRO::e_vx,x,y,zp) - SD.macro(NSE::MACRO::e_vx,x,y,zm));
			G.yx = n1o2 * (SD.macro(NSE::MACRO::e_vy,xp,y,z) - SD.macro(NSE::MACRO::e_vy,xm,y,z));
			G.yy = n1o2 * (SD.macro(NSE::MACRO::e_vy,x,yp,z) - SD.macro(NSE::MACRO::e_vy,x,ym,z));
			G.yz = n1o2 * (SD.macro(NSE::MACRO::e_vy,x,y,zp) - SD.macro(NSE::MACRO::e_vy,x,y,zm));
			G.zx = n1o2 * (SD.macro(NSE::MACRO::e_vz,xp,y,z) - SD.macro(NSE::MACRO::e_vz,xm,y,z));
			G.zy = n1o2 * (SD.macro(NSE::MACRO::e_vz,x,yp,z) - SD.macro(NSE::MACRO::e_vz,x,ym,z));
			G.zz = n1o2 * (SD.macro(NSE::MACRO::e_vz,x,y,zp) - SD.macro(NSE::MACRO::e_vz,x,y,zm));
		}

		// q criterion from definition: Q = - sum_ij d_i u_j d_j u_i
		const dreal q = G.xx*G.yy + G.yy*G.zz + G.xx*G.zz - G.zx*G.xz - G.yz*G.zy - G.xy*G.yx;
		SD.macro(NSE::MACRO::e_qcrit,x,y,z) = q;
}

template < typename ADE >
	#ifdef USE_CUDA
__global__ void cudaLBMComputePhiGradMag(
	typename ADE::DATA SD,
	short int rank,
	short int nproc
)
	#else
void cudaLBMComputePhiGradMag(
	typename ADE::DATA SD,
	typename ADE::TRAITS::idx x,
	typename ADE::TRAITS::idx y,
	typename ADE::TRAITS::idx z,
	short int rank,
	short int nproc
)
	#endif
{
	using dreal = typename ADE::TRAITS::dreal;
	using idx = typename ADE::TRAITS::idx;
	using map_t = typename ADE::TRAITS::map_t;

	#ifdef USE_CUDA
	idx x = threadIdx.x + blockIdx.x * blockDim.x;
	idx y = threadIdx.y + blockIdx.y * blockDim.y;
	idx z = threadIdx.z + blockIdx.z * blockDim.z;
	#endif
	map_t gi_map = SD.map(x, y, z);

	idx xp,xm,yp,ym,zp,zm;
	if (ADE::BC::isPeriodic(gi_map))
	{
		// handle overlaps between GPUs
		//xp = (!SD.overlap_right && x == SD.X-1) ? 0 : (x+1);
		//xm = (!SD.overlap_left && x == 0) ? (SD.X-1) : (x-1);
		xp = (nproc == 1 && x == SD.X()-1) ? 0 : (x+1);
		xm = (nproc == 1 && x == 0) ? (SD.X()-1) : (x-1);
		yp = (y == SD.Y()-1) ? 0 : (y+1);
		ym = (y == 0) ? (SD.Y()-1) : (y-1);
		zp = (z == SD.Z()-1) ? 0 : (z+1);
		zm = (z == 0) ? (SD.Z()-1) : (z-1);
	} else {
		// handle overlaps between GPUs
		// NOTE: ghost layers of lattice sites are assumed in the x-direction, so x+1 and x-1 always work
		xp = x+1;
		xm = x-1;
		yp = TNL::min(y+1, SD.Y()-1);
		ym = TNL::max(y-1,0);
		zp = TNL::min(z+1, SD.Z()-1);
		zm = TNL::max(z-1,0);
	}

		struct Vector
		{
			dreal x=0;
			dreal y=0;
			dreal z=0;
		};
		Vector G;

		// grad phi vector
		if (y == 0 || z == 0 || y == SD.Y()-1 || z == SD.Z()-1)
		{
			// do nothing
			G.x = 0;
			G.y = 0;
			G.z = 0;
		}
		else if (x == 0) {
			// forward difference for x
			G.x = SD.macro(ADE::MACRO::e_phi,xp,y,z) - SD.macro(ADE::MACRO::e_phi,x,y,z);
			G.y = 0;
			G.z = 0;
		}
		else if (x == SD.X()-1) {
			// backward difference for x
			G.x = SD.macro(ADE::MACRO::e_phi,x,y,z) - SD.macro(ADE::MACRO::e_phi,xm,y,z);
			G.y = 0;
			G.z = 0;
		}
		else {
			// central differences
			G.x = n1o2 * (SD.macro(ADE::MACRO::e_phi,xp,y,z) - SD.macro(ADE::MACRO::e_phi,xm,y,z));
			G.y = n1o2 * (SD.macro(ADE::MACRO::e_phi,x,yp,z) - SD.macro(ADE::MACRO::e_phi,x,ym,z));
			G.z = n1o2 * (SD.macro(ADE::MACRO::e_phi,x,y,zp) - SD.macro(ADE::MACRO::e_phi,x,y,zm));
		}

		SD.macro(ADE::MACRO::e_phigradmag2,x,y,z) = G.x*G.x + G.y*G.y + G.z*G.z;
}

template < typename TRAITS >
struct D3Q27_MACRO_QCriterion : D3Q27_MACRO_Base< TRAITS >
{
	using dreal = typename TRAITS::dreal;
	using idx = typename TRAITS::idx;

	enum QuantityNames : std::uint8_t
	{
		e_rho,
		e_vx,
		e_vy,
		e_vz,
		e_fx,
		e_fy,
		e_fz,
		e_qcrit,
		N
	};

	template < typename LBM_DATA, typename LBM_KS >
	CUDA_HOSTDEV static void outputMacro(LBM_DATA &SD, LBM_KS &KS, idx x, idx y, idx z)
	{
		SD.macro(e_rho, x, y, z) = KS.rho;
		SD.macro(e_vx, x, y, z)  = KS.vx;
		SD.macro(e_vy, x, y, z)  = KS.vy;
		SD.macro(e_vz, x, y, z)  = KS.vz;
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
#endif

template <typename NSE, typename ADE>
struct StateLocal : State_NSE_ADE<NSE, ADE>
{
	using TRAITS = typename NSE::TRAITS;
	using BLOCK_NSE = LBM_BLOCK<NSE>;
	using BLOCK_ADE = LBM_BLOCK<ADE>;

	using State<NSE>::nse;
	using State_NSE_ADE<NSE, ADE>::ade;
	using State<NSE>::cnt;
	using State<NSE>::vtk_helper;

	using idx = typename TRAITS::idx;
	using real = typename TRAITS::real;
	using dreal = typename TRAITS::dreal;
	using point_t = typename TRAITS::point_t;
	using lat_t = Lattice<3, real, idx>;

	dreal lbm_inflow_density = 1;
	dreal lbm_inflow_vx = 0;

	// constructor
	StateLocal(const std::string& id, const TNL::MPI::Comm& communicator, lat_t lat_nse, lat_t lat_ade)
	: State_NSE_ADE<NSE, ADE>(id, communicator, lat_nse, lat_ade)
	{}

	void setupBoundaries() override
	{
		lbmDrawCube(nse, NSE::BC::GEO_WALL, {0.45, 0.2, 0.2}, 0.05);
		lbmDrawCube(ade, ADE::BC::GEO_WALL, {0.45, 0.2, 0.2}, 0.05);
		//lbmDrawSphere(nse, NSE::BC::GEO_WALL, {0.45, 0.2, 0.2}, 0.05);
		//lbmDrawSphere(ade, ADE::BC::GEO_WALL, {0.45, 0.2, 0.2}, 0.05);

		nse.setBoundaryX(0, NSE::BC::GEO_INFLOW);  // left
		nse.setBoundaryX(nse.lat.global.x() - 1, NSE::BC::GEO_OUTFLOW_EQ);
		//nse.setBoundaryX(nse.lat.global.x()-1, NSE::BC::GEO_OUTFLOW_RIGHT);

		nse.setBoundaryZ(1, NSE::BC::GEO_WALL);						  // top
		nse.setBoundaryZ(nse.lat.global.z() - 2, NSE::BC::GEO_WALL);  // bottom
		nse.setBoundaryY(1, NSE::BC::GEO_WALL);						  // back
		nse.setBoundaryY(nse.lat.global.y() - 2, NSE::BC::GEO_WALL);  // front

		// extra layer needed due to A-A pattern
		nse.setBoundaryZ(0, NSE::BC::GEO_NOTHING);						 // top
		nse.setBoundaryZ(nse.lat.global.z() - 1, NSE::BC::GEO_NOTHING);	 // bottom
		nse.setBoundaryY(0, NSE::BC::GEO_NOTHING);						 // back
		nse.setBoundaryY(nse.lat.global.y() - 1, NSE::BC::GEO_NOTHING);	 // front

		// ADE boundaries
		ade.setBoundaryX(0, ADE::BC::GEO_INFLOW);  // left
		ade.setBoundaryX(ade.lat.global.x() - 1, ADE::BC::GEO_OUTFLOW_RIGHT);

		ade.setBoundaryZ(1, ADE::BC::GEO_WALL);						  // top
		ade.setBoundaryZ(ade.lat.global.z() - 2, ADE::BC::GEO_WALL);  // bottom
		ade.setBoundaryY(1, ADE::BC::GEO_WALL);						  // back
		ade.setBoundaryY(ade.lat.global.y() - 2, ADE::BC::GEO_WALL);  // front

		// extra layer needed due to A-A pattern
		ade.setBoundaryZ(0, ADE::BC::GEO_NOTHING);						 // top
		ade.setBoundaryZ(ade.lat.global.z() - 1, ADE::BC::GEO_NOTHING);	 // bottom
		ade.setBoundaryY(0, ADE::BC::GEO_NOTHING);						 // back
		ade.setBoundaryY(ade.lat.global.y() - 1, ADE::BC::GEO_NOTHING);	 // front
	}

	void updateKernelVelocities() override
	{
		for (auto& block : nse.blocks) {
			//block.data.inflow_rho = lbm_inflow_density;
			block.data.inflow_vx = lbm_inflow_vx;
			block.data.inflow_vy = 0;
			block.data.inflow_vz = 0;
		}

		for (auto& block : ade.blocks) {
			// TODO: phys -> lbm conversion for concentration?
			block.data.inflow_phi = 1e-3;
		}
	}

#if 0
	void computeBeforeLBMKernel() override
	{
	#ifdef USE_CUDA
		auto get_grid_size = [] (const auto& block, idx x = 0, idx y = 0, idx z = 0) -> dim3
		{
			dim3 gridSize;
			if (x > 0)
				gridSize.x = x;
			else
				gridSize.x = TNL::roundUpDivision(block.local.x(), block.block_size.x());
			if (y > 0)
				gridSize.y = y;
			else
				gridSize.y = TNL::roundUpDivision(block.local.y(), block.block_size.y());
			if (z > 0)
				gridSize.z = z;
			else
				gridSize.z = TNL::roundUpDivision(block.local.z(), block.block_size.z());

			return gridSize;
		};
	#endif

		for (auto& block : nse.blocks)
		{
	#ifdef USE_CUDA
			const dim3 gridSize = get_grid_size(block);
			cudaLBMComputeQCriterion< NSE ><<<gridSize, block.block_size>>>(block.data, nse.rank, nse.nproc);
			cudaStreamSynchronize(0);
			TNL_CHECK_CUDA_DEVICE;
	#else
		#pragma omp parallel for schedule(static) collapse(2)
			for (idx x = 0; x < block.local.x(); x++)
			for (idx z = 0; z < block.local.z(); z++)
			for (idx y = 0; y < block.local.y(); y++)
				LBMComputeQCriterion< NSE >(block.data, nse.rank, nse.nproc, x, y, z);
	#endif
		}

		for (auto& block : ade.blocks)
		{
	#ifdef USE_CUDA
			const dim3 gridSize = get_grid_size(block);
			cudaLBMComputePhiGradMag< ADE ><<<gridSize, block.block_size>>>(block.data, nse.rank, nse.nproc);
			cudaStreamSynchronize(0);
			TNL_CHECK_CUDA_DEVICE;
	#else
		#pragma omp parallel for schedule(static) collapse(2)
			for (idx x = 0; x < block.local.x(); x++)
			for (idx z = 0; z < block.local.z(); z++)
			for (idx y = 0; y < block.local.y(); y++)
				cudaLBMComputePhiGradMag< ADE >(block.data, nse.rank, nse.nproc, x, y, z);
	#endif
		}
	}
#endif

	bool outputData(const BLOCK_NSE& block, int index, int dof, char* desc, idx x, idx y, idx z, real& value, int& dofs) override
	{
		int k = 0;
		if (index == k++)
			return vtk_helper("lbm_density", block.hmacro(NSE::MACRO::e_rho, x, y, z), 1, desc, value, dofs);
		if (index == k++) {
			switch (dof) {
				case 0:
					return vtk_helper("velocity", nse.lat.lbm2physVelocity(block.hmacro(NSE::MACRO::e_vx, x, y, z)), 3, desc, value, dofs);
				case 1:
					return vtk_helper("velocity", nse.lat.lbm2physVelocity(block.hmacro(NSE::MACRO::e_vy, x, y, z)), 3, desc, value, dofs);
				case 2:
					return vtk_helper("velocity", nse.lat.lbm2physVelocity(block.hmacro(NSE::MACRO::e_vz, x, y, z)), 3, desc, value, dofs);
			}
		}
		//if (index==k++) return vtk_helper("lbm_qcriterion", block.hmacro(NSE::MACRO::e_qcrit,x,y,z), 1, desc, value, dofs);
		return false;
	}

	bool outputData(const BLOCK_ADE& block, int index, int dof, char* desc, idx x, idx y, idx z, real& value, int& dofs) override
	{
		int k = 0;
		if (index == k++)
			return vtk_helper("lbm_phi", block.hmacro(ADE::MACRO::e_phi, x, y, z), 1, desc, value, dofs);
		//if (index==k++) return vtk_helper("lbm_phigradmag2", block.hmacro(ADE::MACRO::e_phigradmag2,x,y,z), 1, desc, value, dofs);
		return false;
	}

	void probe1() override
	{
		if (nse.iterations != 0) {
			// inflow density extrapolation
			idx x = 5;
			idx y = nse.lat.global.y() / 2;
			idx z = nse.lat.global.z() / 2;
			for (auto& block : nse.blocks)
				if (block.isLocalIndex(x, y, z)) {
					real old_lbm_inflow_density = lbm_inflow_density;
					lbm_inflow_density = block.dmacro.getElement(NSE::MACRO::e_rho, x, y, z);
					spdlog::info("probe: lbm inflow density changed from {:e} to {:e}", old_lbm_inflow_density, lbm_inflow_density);
				}
		}
	}
};

template <typename NSE, typename ADE>
int simT1_test(int RESOLUTION = 2)
{
	using idx = typename NSE::TRAITS::idx;
	using real = typename NSE::TRAITS::real;
	using point_t = typename NSE::TRAITS::point_t;
	using lat_t = Lattice<3, real, idx>;

	int block_size = 32;
	int X = 128 * RESOLUTION;  // width in pixels
	//int Y = 41*RESOLUTION;// height in pixels --- top and bottom walls 1px
	//int Z = 41*RESOLUTION;// height in pixels --- top and bottom walls 1px
	int Y = block_size * RESOLUTION;   // height in pixels --- top and bottom walls 1px
	int Z = Y;						   // height in pixels --- top and bottom walls 1px
	real LBM_VISCOSITY = 0.001 / 3.0;  //1.0/6.0; /// GIVEN: optimal is 1/6
	real PHYS_HEIGHT = 0.41;		   // [m] domain height (physical)
	real PHYS_VISCOSITY = 1.552e-5;	   // [m^2/s] fluid viscosity of air
	real PHYS_VELOCITY = 1.0;
	real PHYS_DL = PHYS_HEIGHT / ((real) Y - 2);
	real PHYS_DT = LBM_VISCOSITY / PHYS_VISCOSITY * PHYS_DL * PHYS_DL;	//PHYS_HEIGHT/(real)LBM_HEIGHT;
	real PHYS_DIFFUSION = 2.552e-05;									// [m^2/s] diffusion coeff for the ADE
	point_t PHYS_ORIGIN = {0., 0., 0.};

	// initialize the lattice
	lat_t lat_nse;
	lat_nse.global = typename lat_t::CoordinatesType(X, Y, Z);
	lat_nse.physOrigin = PHYS_ORIGIN;
	lat_nse.physDl = PHYS_DL;
	lat_nse.physDt = PHYS_DT;
	lat_nse.physViscosity = PHYS_VISCOSITY;

	// lattice for the ADE is the same as that for NSE, except for viscosity (diffusion)
	lat_t lat_ade = lat_nse;
	lat_ade.physViscosity = PHYS_DIFFUSION;

	const std::string state_id = fmt::format("sim_T1_res{:02d}_np{:03d}", RESOLUTION, TNL::MPI::GetSize(MPI_COMM_WORLD));
	StateLocal<NSE, ADE> state(state_id, MPI_COMM_WORLD, lat_nse, lat_ade);

	if (! state.canCompute())
		return 0;

	// problem parameters
	state.lbm_inflow_vx = lat_nse.phys2lbmVelocity(PHYS_VELOCITY);

	state.nse.physFinalTime = 10.0;
	state.cnt[PRINT].period = 0.01;
	//state.cnt[PROBE1].period = 0.001;
	// test
	//state.cnt[PRINT].period = 100*PHYS_DT;
	//state.nse.physFinalTime = 1000*PHYS_DT;
	//state.cnt[VTK3D].period = 1000*PHYS_DT;
	//state.cnt[SAVESTATE].period = 600;  // save state every [period] of wall time
	//state.check_savestate_flag = false;
	//state.wallTime = 60;
	// RCI
	//state.nse.physFinalTime = 0.5;
	//state.cnt[VTK3D].period = 0.5;
	//state.cnt[SAVESTATE].period = 3600;  // save state every [period] of wall time
	//state.check_savestate_flag = false;
	//state.wallTime = 3600 * 23.5;

	// add cuts
	state.cnt[VTK2D].period = 0.01;
	state.add2Dcut_X(X / 2, "cutsX/cut_X");
	state.add2Dcut_Y(Y / 2, "cutsY/cut_Y");
	state.add2Dcut_Z(Z / 2, "cutsZ/cut_Z");

	//state.cnt[VTK3D].period = 0.001;
	//state.cnt[VTK3DCUT].period = 0.001;
	//state.add3Dcut(X/4,Y/4,Z/4, X/2,Y/2,Z/2, 2, "box");

	execute(state);

	return 0;
}

//template <typename TRAITS=TraitsSP>
template <typename TRAITS = TraitsDP>
void run(int RES)
{
	using NSE_COLL = D3Q27_CUM<TRAITS, D3Q27_EQ_INV_CUM<TRAITS>>;
	using NSE_CONFIG = LBM_CONFIG<
		TRAITS,
		D3Q27_KernelStruct,
		//NSE_Data_ConstInflow<TRAITS>,
		// FIXME: FreeRho inflow condition leads to lower velocity in the domain (approx 70%)
		NSE_Data_FreeRhoConstInflow<TRAITS>,
		NSE_COLL,
		typename NSE_COLL::EQ,
		D3Q27_STREAMING<TRAITS>,
		D3Q27_BC_All,
		D3Q27_MACRO_Default<TRAITS>
		//D3Q27_MACRO_QCriterion<TRAITS>
		>;

	//using ADE_COLL = D3Q7_SRT<TRAITS>;
	//using ADE_COLL = D3Q7_MRT<TRAITS>;
	using ADE_COLL = D3Q7_CLBM<TRAITS>;
	using ADE_CONFIG = LBM_CONFIG<
		TRAITS,
		D3Q7_KernelStruct,
		ADE_Data_ConstInflow<TRAITS>,
		ADE_COLL,
		typename ADE_COLL::EQ,
		D3Q7_STREAMING<TRAITS>,
		D3Q7_BC_All,
		D3Q7_MACRO_Default<TRAITS>>;

	simT1_test<NSE_CONFIG, ADE_CONFIG>(RES);
}

int main(int argc, char** argv)
{
	TNLMPI_INIT mpi(argc, argv);

	argparse::ArgumentParser program("sim_T1");
	program.add_description("Simple coupled D3Q27-D3Q7 simulation example.");
	program.add_argument("resolution").help("resolution of the lattice").scan<'i', int>().default_value(1);

	try {
		program.parse_args(argc, argv);
	}
	catch (const std::exception& err) {
		std::cerr << err.what() << std::endl;
		std::cerr << program;
		return 1;
	}

	const auto resolution = program.get<int>("resolution");
	if (resolution < 1)
		throw std::invalid_argument("CLI error: resolution must be at least 1");

	run(resolution);

	return 0;
}
