#pragma once

#include "defs.h"

template <typename NSE>
__cuda_callable__ void kernelInitIndices(
	typename NSE::DATA SD,
	typename NSE::TRAITS::map_t map,
	short int nproc,
	typename NSE::TRAITS::idx x,
	typename NSE::TRAITS::idx y,
	typename NSE::TRAITS::idx z,
	typename NSE::LBM_KS::SG &streamGrid
)
{
	if (NSE::BC::isPeriodic(map)) {
		for(int i = -NSE::LBM_KS::NoDV; i <= NSE::LBM_KS::NoDV; i++){
			streamGrid.x[NSE::LBM_KS::NoDV+i] = (x+i+SD.X())%SD.X();
			streamGrid.y[NSE::LBM_KS::NoDV+i] = (y+i+SD.Y())%SD.Y();
			streamGrid.z[NSE::LBM_KS::NoDV+i] = (z+i+SD.Z())%SD.Z();
		}
		// TODO: use nproc_y and nproc_z
		// TODO: use nproc
	}
	else {
#ifdef AA_PATTERN
		// NOTE: ghost layers of lattice sites are assumed in all directions, so these expressions always work
		xp = x + 1;
		xm = x - 1;
		yp = y + 1;
		ym = y - 1;
		zp = z + 1;
		zm = z - 1;
#elif defined(HAVE_MPI)
		const typename NSE::TRAITS::idx& overlap_x = SD.indexer.template getOverlap<0>();
		const typename NSE::TRAITS::idx& overlap_y = SD.indexer.template getOverlap<1>();
		const typename NSE::TRAITS::idx& overlap_z = SD.indexer.template getOverlap<2>();
		streamGrid.x[NSE::LBM_KS::NoDV] = x;
		streamGrid.y[NSE::LBM_KS::NoDV] = y;
		streamGrid.z[NSE::LBM_KS::NoDV] = z;
		for(int i = 1; i <= NSE::LBM_KS::NoDV; i++){
			streamGrid.x[NSE::LBM_KS::NoDV+i] = TNL::min(x + i, SD.X() - 1 + overlap_x);
			streamGrid.x[NSE::LBM_KS::NoDV-i] = TNL::max(x - i, -overlap_x);
			streamGrid.y[NSE::LBM_KS::NoDV+i] = TNL::min(y + i, SD.Y() - 1 + overlap_y);
			streamGrid.y[NSE::LBM_KS::NoDV-i] = TNL::max(y - i, -overlap_y);
			streamGrid.z[NSE::LBM_KS::NoDV+i] = TNL::min(z + i, SD.Z() - 1 + overlap_z);
			streamGrid.z[NSE::LBM_KS::NoDV-i] = TNL::max(z - i, -overlap_z);
		}
#else
		streamGrid.x[NSE::LBM_KS::NoDV] = x;
		streamGrid.y[NSE::LBM_KS::NoDV] = y;
		streamGrid.z[NSE::LBM_KS::NoDV] = z;
		for(int i = 1; i <= NSE::LBM_KS::NoDV; i++){
			streamGrid.x[NSE::LBM_KS::NoDV+i] = TNL::min(x + i, SD.X() - 1);
			streamGrid.x[NSE::LBM_KS::NoDV-i] = TNL::max(x - i, 0);
			streamGrid.y[NSE::LBM_KS::NoDV+i] = TNL::min(y + i, SD.Y() - 1);
			streamGrid.y[NSE::LBM_KS::NoDV-i] = TNL::max(y - i, 0);
			streamGrid.z[NSE::LBM_KS::NoDV+i] = TNL::min(z + i, SD.Z() - 1);
			streamGrid.z[NSE::LBM_KS::NoDV-i] = TNL::max(z - i, 0);
		}
#endif
	}
}

template <typename NSE>
#ifdef USE_CUDA
__global__ void cudaLBMKernel(typename NSE::DATA SD, short int nproc, typename NSE::TRAITS::idx3d offset, typename NSE::TRAITS::idx3d end)
#else
CUDA_HOSTDEV void
LBMKernel(typename NSE::DATA SD, typename NSE::TRAITS::idx x, typename NSE::TRAITS::idx y, typename NSE::TRAITS::idx z, short int nproc)
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

	typename NSE::LBM_KS::SG streamGrid;
	kernelInitIndices<NSE>(SD, gi_map, nproc,x,y,z, streamGrid);

	typename NSE::template KernelStruct<dreal> KS;

	// copy quantities
	NSE::MACRO::copyQuantities(SD, KS, x, y, z);

	// optional computation of the forcing term (e.g. for the non-Newtonian model)
	NSE::MACRO::computeForcing<NSE::BC,NSE::DATA,NSE::LBM_KS>(SD, KS, streamGrid);

	NSE::BC::preCollision(SD, KS, gi_map, streamGrid);
	if (NSE::BC::doCollision(gi_map))
		NSE::COLL::collision(KS);
	NSE::BC::postCollision(SD, KS, gi_map, streamGrid);

	NSE::MACRO::outputMacro(SD, KS, x, y, z);
}

template <typename NSE, typename ADE>
#ifdef USE_CUDA
__global__ void cudaLBMKernel(
	typename NSE::DATA NSE_SD, typename ADE::DATA ADE_SD, short int nproc, typename NSE::TRAITS::idx3d offset, typename NSE::TRAITS::idx3d end
)
#else
CUDA_HOSTDEV void LBMKernel(
	typename NSE::DATA NSE_SD,
	typename ADE::DATA ADE_SD,
	typename NSE::TRAITS::idx x,
	typename NSE::TRAITS::idx y,
	typename NSE::TRAITS::idx z,
	short int nproc
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

	const map_t NSE_mapgi = NSE_SD.map(x, y, z);
	const map_t ADE_mapgi = ADE_SD.map(x, y, z);

	typename NSE::LBM_KS::SG streamGrid;
	kernelInitIndices<NSE>(NSE_SD, NSE_mapgi, nproc,x,y,z, streamGrid);

	// NSE part
	typename NSE::template KernelStruct<dreal> NSE_KS;

	// copy quantities
	NSE::MACRO::copyQuantities(NSE_SD, NSE_KS, x, y, z);

	// optional computation of the forcing term (e.g. for the non-Newtonian model)
	NSE::MACRO::template computeForcing<typename NSE::BC>(NSE_SD, NSE_KS, streamGrid);

	NSE::BC::preCollision(NSE_SD, NSE_KS, NSE_mapgi, streamGrid);
	if (NSE::BC::doCollision(NSE_mapgi))
		NSE::COLL::collision(NSE_KS);
	NSE::BC::postCollision(NSE_SD, NSE_KS, NSE_mapgi, streamGrid);

	NSE::MACRO::outputMacro(NSE_SD, NSE_KS, x, y, z);

	// ADE part
	typename ADE::template KernelStruct<dreal> ADE_KS;
	ADE_KS.vx = NSE_KS.vx;
	ADE_KS.vy = NSE_KS.vy;
	ADE_KS.vz = NSE_KS.vz;
	// NOTE: experiment 2022.04.06: interpolate momentum instead of velocity (LBM conserves momentum, not mass - RF mail 2022.04.01)
	//ADE_KS.vx = NSE_KS.rho * NSE_KS.vx;
	//ADE_KS.vy = NSE_KS.rho * NSE_KS.vy;
	//ADE_KS.vz = NSE_KS.rho * NSE_KS.vz;
	// FIXME this depends on the e_qcrit macro
	//ADE_KS.qcrit = NSE_SD.macro(NSE::MACRO::e_qcrit, x, y, z);
	//ADE_KS.phigradmag2 = ADE_SD.macro(ADE::MACRO::e_phigradmag2, x, y, z);
	//ADE_KS.x = x;

	// copy quantities
	ADE::MACRO::copyQuantities(ADE_SD, ADE_KS, x, y, z);

	ADE::BC::preCollision(ADE_SD, ADE_KS, ADE_mapgi, streamGrid);
	if (ADE::BC::doCollision(ADE_mapgi))
		ADE::COLL::collision(ADE_KS);
	ADE::BC::postCollision(ADE_SD, ADE_KS, ADE_mapgi, streamGrid);

	ADE::MACRO::outputMacro(ADE_SD, ADE_KS, x, y, z);
}

template <typename NSE>
#ifdef USE_CUDA
__global__ void cudaLBMComputeVelocitiesStarAndZeroForce(typename NSE::DATA SD, short int nproc)
#else
void LBMComputeVelocitiesStarAndZeroForce(
	typename NSE::DATA SD, typename NSE::TRAITS::idx x, typename NSE::TRAITS::idx y, typename NSE::TRAITS::idx z, short int nproc
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

	if (x >= SD.X() || y >= SD.Y() || z >= SD.Z())
		return;
#endif

	map_t gi_map = SD.map(x, y, z);

	typename NSE::template KernelStruct<dreal> KS;

	// copy quantities
	NSE::MACRO::copyQuantities(SD, KS, x, y, z);

	typename NSE::LBM_KS::SG streamGrid;
	kernelInitIndices<NSE>(SD, gi_map, nproc,x,y,z, streamGrid);

	NSE::MACRO::zeroForcesInKS(KS);

	// do streaming, compute density and velocity
	NSE::BC::preCollision(SD, KS, gi_map, streamGrid);

	NSE::MACRO::outputMacro(SD, KS, x, y, z);
	// reset forces
	NSE::MACRO::zeroForces(SD, x, y, z);
}
