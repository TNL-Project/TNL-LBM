#pragma once

#include "defs.h"

template < typename NSE >
#ifdef USE_CUDA
__global__ void cudaLBMKernel(
	typename NSE::DATA SD,
	short int rank,
	short int nproc,
	typename NSE::TRAITS::idx offset_x
)
#else
CUDA_HOSTDEV
void LBMKernel(
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
	idx x = threadIdx.x + blockIdx.x * blockDim.x + offset_x;
	idx y = threadIdx.y + blockIdx.y * blockDim.y;
	idx z = threadIdx.z + blockIdx.z * blockDim.z;

	if (x >= SD.X() || y >= SD.Y() || z >= SD.Z())
		return;
	#endif

	map_t gi_map = SD.map(x, y, z);

	idx xp,xm,yp,ym,zp,zm;
	if (NSE::BC::isPeriodic(gi_map))
	{
		// handle overlaps between GPUs
//		xp = (!SD.overlap_right && x == SD.X-1) ? 0 : (x+1);
//		xm = (!SD.overlap_left && x == 0) ? (SD.X-1) : (x-1);
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
		yp = MIN(y+1, SD.Y()-1);
		ym = MAX(y-1,0);
		zp = MIN(z+1, SD.Z()-1);
		zm = MAX(z-1,0);
	}

	typename NSE::template KernelStruct<dreal> KS;

	// copy quantities
	NSE::MACRO::copyQuantities(SD, KS, x, y, z);

	// optional computation of the forcing term (e.g. for the non-Newtonian model)
	NSE::MACRO::template computeForcing<typename NSE::BC>(SD,KS,xm,x,xp,ym,y,yp,zm,z,zp);

	NSE::BC::preCollision(SD,KS,gi_map,xm,x,xp,ym,y,yp,zm,z,zp);
	if (NSE::BC::doCollision(gi_map))
		NSE::COLL::collision(KS);
	NSE::BC::postCollision(SD,KS,gi_map,xm,x,xp,ym,y,yp,zm,z,zp);

	NSE::MACRO::outputMacro(SD, KS, x, y, z);
}


template < typename NSE, typename ADE >
#ifdef USE_CUDA
__global__ void cudaLBMKernel(
	typename NSE::DATA NSE_SD,
	typename ADE::DATA ADE_SD,
	short int rank,
	short int nproc,
	typename NSE::TRAITS::idx offset_x
)
#else
CUDA_HOSTDEV
void LBMKernel(
	typename NSE::DATA NSE_SD,
	typename ADE::DATA ADE_SD,
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
	idx x = threadIdx.x + blockIdx.x * blockDim.x + offset_x;
	idx y = threadIdx.y + blockIdx.y * blockDim.y;
	idx z = threadIdx.z + blockIdx.z * blockDim.z;

	if (x >= NSE_SD.X() || y >= NSE_SD.Y() || z >= NSE_SD.Z())
		return;
	#endif

	const map_t NSE_mapgi = NSE_SD.map(x, y, z);
	const map_t ADE_mapgi = ADE_SD.map(x, y, z);

	idx xp,xm,yp,ym,zp,zm;
	if (NSE::BC::isPeriodic(NSE_mapgi))
	{
		// handle overlaps between GPUs
//		xp = (!NSE_SD.overlap_right && x == NSE_SD.X-1) ? 0 : (x+1);
//		xm = (!NSE_SD.overlap_left && x == 0) ? (NSE_SD.X-1) : (x-1);
		xp = (nproc == 1 && x == NSE_SD.X()-1) ? 0 : (x+1);
		xm = (nproc == 1 && x == 0) ? (NSE_SD.X()-1) : (x-1);
		yp = (y == NSE_SD.Y()-1) ? 0 : (y+1);
		ym = (y == 0) ? (NSE_SD.Y()-1) : (y-1);
		zp = (z == NSE_SD.Z()-1) ? 0 : (z+1);
		zm = (z == 0) ? (NSE_SD.Z()-1) : (z-1);
	} else {
		// handle overlaps between GPUs
		// NOTE: ghost layers of lattice sites are assumed in the x-direction, so x+1 and x-1 always work
		xp = x+1;
		xm = x-1;
		yp = MIN(y+1, NSE_SD.Y()-1);
		ym = MAX(y-1,0);
		zp = MIN(z+1, NSE_SD.Z()-1);
		zm = MAX(z-1,0);
	}

	// NSE part
	typename NSE::template KernelStruct<dreal> NSE_KS;

	// copy quantities
	NSE::MACRO::copyQuantities(NSE_SD, NSE_KS, x, y, z);

	// optional computation of the forcing term (e.g. for the non-Newtonian model)
	NSE::MACRO::template computeForcing<typename NSE::BC>(NSE_SD,NSE_KS,xm,x,xp,ym,y,yp,zm,z,zp);

	NSE::BC::preCollision(NSE_SD,NSE_KS,NSE_mapgi,xm,x,xp,ym,y,yp,zm,z,zp);
	if (NSE::BC::doCollision(NSE_mapgi))
		NSE::COLL::collision(NSE_KS);
	NSE::BC::postCollision(NSE_SD,NSE_KS,NSE_mapgi,xm,x,xp,ym,y,yp,zm,z,zp);

	NSE::MACRO::outputMacro(NSE_SD, NSE_KS, x, y, z);

	// ADE part
	typename ADE::template KernelStruct<dreal> ADE_KS;
	ADE_KS.vx = NSE_KS.vx;
	ADE_KS.vy = NSE_KS.vy;
	ADE_KS.vz = NSE_KS.vz;
	// NOTE: experiment 2022.04.06: interpolate momentum instead of velocity (LBM conserves momentum, not mass - RF mail 2022.04.01)
	// ADE_KS.vx = NSE_KS.rho * NSE_KS.vx;
	// ADE_KS.vy = NSE_KS.rho * NSE_KS.vy;
	// ADE_KS.vz = NSE_KS.rho * NSE_KS.vz;
	// FIXME this depends on the e_qcrit macro
//	ADE_KS.qcrit = NSE_SD.macro(NSE::MACRO::e_qcrit, x, y, z);
//	ADE_KS.phigradmag2 = ADE_SD.macro(ADE::MACRO::e_phigradmag2, x, y, z);
//	ADE_KS.x = x;

	// copy quantities
	ADE::MACRO::copyQuantities(ADE_SD, ADE_KS, x, y, z);

	ADE::BC::preCollision(ADE_SD,ADE_KS,ADE_mapgi,xm,x,xp,ym,y,yp,zm,z,zp);
	if (ADE::BC::doCollision(ADE_mapgi))
		ADE::COLL::collision(ADE_KS);
	ADE::BC::postCollision(ADE_SD,ADE_KS,ADE_mapgi,xm,x,xp,ym,y,yp,zm,z,zp);

	ADE::MACRO::outputMacro(ADE_SD, ADE_KS, x, y, z);
}


// initial condition --> hmacro on CPU
template < typename LBM_TYPE >
void LBMKernelInit(
	typename LBM_TYPE::DATA& SD,
	typename LBM_TYPE::TRAITS::idx x,
	typename LBM_TYPE::TRAITS::idx y,
	typename LBM_TYPE::TRAITS::idx z
)
{
	using dreal = typename LBM_TYPE::TRAITS::dreal;

	typename LBM_TYPE::template KernelStruct<dreal> KS;
	for (int i = 0; i < LBM_TYPE::Q; i++)
		KS.f[i] = SD.df(df_cur, i, x, y, z);

	// copy quantities
	LBM_TYPE::MACRO::copyQuantities(SD, KS, x, y, z);

	LBM_TYPE::MACRO::zeroForcesInKS(KS);

	// compute Density & Velocity
	LBM_TYPE::COLL::computeDensityAndVelocity(KS);

	LBM_TYPE::MACRO::outputMacro(SD, KS, x, y, z);
}


template < typename NSE >
#ifdef USE_CUDA
__global__ void cudaLBMComputeVelocitiesStar(
	typename NSE::DATA SD,
	short int rank,
	short int nproc
)
#else
void LBMComputeVelocitiesStar(
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

	if (x >= SD.X() || y >= SD.Y() || z >= SD.Z())
		return;
	#endif

	map_t gi_map = SD.map(x, y, z);

	typename NSE::template KernelStruct<dreal> KS;

	// copy quantities
	NSE::MACRO::copyQuantities(SD, KS, x, y, z);

	idx xp,xm,yp,ym,zp,zm;
	if (NSE::BC::isPeriodic(gi_map))
	{
		// handle overlaps between GPUs
//		xp = (!SD.overlap_right && x == SD.X-1) ? 0 : (x+1);
//		xm = (!SD.overlap_left && x == 0) ? (SD.X-1) : (x-1);
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
		yp = MIN(y+1, SD.Y()-1);
		ym = MAX(y-1,0);
		zp = MIN(z+1, SD.Z()-1);
		zm = MAX(z-1,0);
	}

	KS.fx = 0;
	KS.fy = 0;
	KS.fz = 0;

	// do streaming, compute density and velocity
	NSE::BC::preCollision(SD,KS,gi_map,xm,x,xp,ym,y,yp,zm,z,zp);

	NSE::MACRO::outputMacro(SD, KS, x, y, z);
}

template < typename NSE >
#ifdef USE_CUDA
__global__ void cudaLBMComputeVelocitiesStarAndZeroForce(
	typename NSE::DATA SD,
	short int rank,
	short int nproc
)
#else
void LBMComputeVelocitiesStarAndZeroForce(
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

	if (x >= SD.X() || y >= SD.Y() || z >= SD.Z())
		return;
	#endif

	map_t gi_map = SD.map(x, y, z);

	typename NSE::template KernelStruct<dreal> KS;

	// copy quantities
	NSE::MACRO::copyQuantities(SD, KS, x, y, z);

	idx xp,xm,yp,ym,zp,zm;
	if (NSE::BC::isPeriodic(gi_map))
	{
		// handle overlaps between GPUs
//		xp = (!SD.overlap_right && x == SD.X-1) ? 0 : (x+1);
//		xm = (!SD.overlap_left && x == 0) ? (SD.X-1) : (x-1);
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
		yp = MIN(y+1, SD.Y()-1);
		ym = MAX(y-1,0);
		zp = MIN(z+1, SD.Z()-1);
		zm = MAX(z-1,0);
	}

	KS.fx = 0;
	KS.fy = 0;
	KS.fz = 0;

	// do streaming, compute density and velocity
	NSE::BC::preCollision(SD,KS,gi_map,xm,x,xp,ym,y,yp,zm,z,zp);

	NSE::MACRO::outputMacro(SD, KS, x, y, z);
	// reset forces
	NSE::MACRO::zeroForces(SD, x, y, z);
}
