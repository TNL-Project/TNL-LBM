#pragma once

#include "defs.h"

template <typename NSE>
__cuda_callable__ void kernelInitIndices(
	typename NSE::DATA SD,
	typename NSE::TRAITS::map_t map,
	typename NSE::TRAITS::bool3d distributed,
	typename NSE::TRAITS::idx x,
	typename NSE::TRAITS::idx y,
	typename NSE::TRAITS::idx z,
	typename NSE::TRAITS::idx& xp,
	typename NSE::TRAITS::idx& xm,
	typename NSE::TRAITS::idx& yp,
	typename NSE::TRAITS::idx& ym,
	typename NSE::TRAITS::idx& zp,
	typename NSE::TRAITS::idx& zm
)
{
	if (NSE::BC::isPeriodic(map)) {
		xp = (! distributed.x() && x == SD.X() - 1) ? 0 : (x + 1);
		xm = (! distributed.x() && x == 0) ? (SD.X() - 1) : (x - 1);
		yp = (! distributed.y() && y == SD.Y() - 1) ? 0 : (y + 1);
		ym = (! distributed.y() && y == 0) ? (SD.Y() - 1) : (y - 1);
		zp = (! distributed.z() && z == SD.Z() - 1) ? 0 : (z + 1);
		zm = (! distributed.z() && z == 0) ? (SD.Z() - 1) : (z - 1);
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
		xp = TNL::min(x + 1, SD.X() - 1 + overlap_x);
		xm = TNL::max(x - 1, -overlap_x);
		yp = TNL::min(y + 1, SD.Y() - 1 + overlap_y);
		ym = TNL::max(y - 1, -overlap_y);
		zp = TNL::min(z + 1, SD.Z() - 1 + overlap_z);
		zm = TNL::max(z - 1, -overlap_z);
#else
		xp = TNL::min(x + 1, SD.X() - 1);
		xm = TNL::max(x - 1, 0);
		yp = TNL::min(y + 1, SD.Y() - 1);
		ym = TNL::max(y - 1, 0);
		zp = TNL::min(z + 1, SD.Z() - 1);
		zm = TNL::max(z - 1, 0);
#endif
	}
}

template <typename NSE>
#ifdef USE_CUDA
__global__ void cudaLBMKernel(
	typename NSE::DATA SD,
	typename NSE::TRAITS::idx3d offset,
	typename NSE::TRAITS::idx3d end,
	typename NSE::TRAITS::bool3d distributed,
	bool output_macro
)
#else
CUDA_HOSTDEV void LBMKernel(
	typename NSE::DATA SD,
	typename NSE::TRAITS::idx x,
	typename NSE::TRAITS::idx y,
	typename NSE::TRAITS::idx z,
	typename NSE::TRAITS::bool3d distributed,
	bool output_macro
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

	idx xp, xm, yp, ym, zp, zm;
	kernelInitIndices<NSE>(SD, gi_map, distributed, x, y, z, xp, xm, yp, ym, zp, zm);

	typename NSE::template KernelStruct<dreal> KS;

	// copy quantities
	NSE::MACRO::copyQuantities(SD, KS, x, y, z);

	// optional computation of the forcing term (e.g. for the non-Newtonian model)
	NSE::MACRO::template computeForcing<typename NSE::BC>(SD, KS, xm, x, xp, ym, y, yp, zm, z, zp);

	NSE::BC::preCollision(SD, KS, gi_map, xm, x, xp, ym, y, yp, zm, z, zp);
	if (NSE::BC::doCollision(gi_map))
		NSE::COLL::collision(KS);
	NSE::BC::postCollision(SD, KS, gi_map, xm, x, xp, ym, y, yp, zm, z, zp);

	if (output_macro)
		NSE::MACRO::outputMacro(SD, KS, x, y, z);
}

template <typename NSE, typename ADE>
#ifdef USE_CUDA
__global__ void cudaLBMKernel(
	typename NSE::DATA NSE_SD,
	typename ADE::DATA ADE_SD,
	typename NSE::TRAITS::idx3d offset,
	typename NSE::TRAITS::idx3d end,
	typename NSE::TRAITS::bool3d distributed,
	bool output_macro
)
#else
CUDA_HOSTDEV void LBMKernel(
	typename NSE::DATA NSE_SD,
	typename ADE::DATA ADE_SD,
	typename NSE::TRAITS::idx x,
	typename NSE::TRAITS::idx y,
	typename NSE::TRAITS::idx z,
	typename NSE::TRAITS::bool3d distributed,
	bool output_macro
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

	idx xp, xm, yp, ym, zp, zm;
	kernelInitIndices<NSE>(NSE_SD, NSE_mapgi, distributed, x, y, z, xp, xm, yp, ym, zp, zm);

	// NSE part
	typename NSE::template KernelStruct<dreal> NSE_KS;

	// copy quantities
	NSE::MACRO::copyQuantities(NSE_SD, NSE_KS, x, y, z);

	// optional computation of the forcing term (e.g. for the non-Newtonian model)
	NSE::MACRO::template computeForcing<typename NSE::BC>(NSE_SD, NSE_KS, xm, x, xp, ym, y, yp, zm, z, zp);

	NSE::BC::preCollision(NSE_SD, NSE_KS, NSE_mapgi, xm, x, xp, ym, y, yp, zm, z, zp);
	if (NSE::BC::doCollision(NSE_mapgi))
		NSE::COLL::collision(NSE_KS);
	NSE::BC::postCollision(NSE_SD, NSE_KS, NSE_mapgi, xm, x, xp, ym, y, yp, zm, z, zp);

	if (output_macro)
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

	ADE::BC::preCollision(ADE_SD, ADE_KS, ADE_mapgi, xm, x, xp, ym, y, yp, zm, z, zp);
	if (ADE::BC::doCollision(ADE_mapgi))
		ADE::COLL::collision(ADE_KS);
	ADE::BC::postCollision(ADE_SD, ADE_KS, ADE_mapgi, xm, x, xp, ym, y, yp, zm, z, zp);

	if (output_macro)
		ADE::MACRO::outputMacro(ADE_SD, ADE_KS, x, y, z);
}

template <typename NSE>
#ifdef USE_CUDA
__global__ void cudaLBMComputeVelocitiesStarAndZeroForce(typename NSE::DATA SD, typename NSE::TRAITS::bool3d distributed)
#else
void LBMComputeVelocitiesStarAndZeroForce(
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

	idx xp, xm, yp, ym, zp, zm;
	kernelInitIndices<NSE>(SD, gi_map, distributed, x, y, z, xp, xm, yp, ym, zp, zm);

	NSE::MACRO::zeroForcesInKS(KS);

	// do streaming, compute density and velocity
	NSE::BC::preCollision(SD, KS, gi_map, xm, x, xp, ym, y, yp, zm, z, zp);

	NSE::MACRO::outputMacro(SD, KS, x, y, z);
	// reset forces
	NSE::MACRO::zeroForces(SD, x, y, z);
}
