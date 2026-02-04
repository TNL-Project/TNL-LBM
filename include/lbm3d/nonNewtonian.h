#pragma once

#include "defs.h"
#include "kernels.h"
#include "lbm_common/ciselnik.h"

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                                                                                                                                                                                                //
//                                                                                                                                                                                                                //
// Extra kernels for the non-Newtonian fluid model
//                                                                                                                                                                                                                //
//                                                                                                                                                                                                                //
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#if 0
template <
	typename LBM_TYPE,
	typename STREAMING,
	typename MACRO,
	typename LBM_DATA,
	typename LBM_BC
>
	#ifdef TODO
CUDA_HOSTDEV
void LBMKernelCheckMap(
	typename LBM_TYPE::T_TRAITS::idx x,
	typename LBM_TYPE::T_TRAITS::idx y,
	typename LBM_TYPE::T_TRAITS::idx z,
	LBM_DATA SD,
	short int rank,
	short int nproc
)
	#else
		#ifdef USE_CUDA
//__launch_bounds__(32, 16)
__global__ void cudaLBMKernelCheckMap(
	LBM_DATA SD,
	short int rank,
	short int nproc,
	typename LBM_TYPE::T_TRAITS::idx offset_x
)
		#else
CUDA_HOSTDEV
void LBMKernelCheckMap(
	LBM_DATA SD,
	typename LBM_TYPE::T_TRAITS::idx x,
	typename LBM_TYPE::T_TRAITS::idx y,
	typename LBM_TYPE::T_TRAITS::idx z,
	short int rank,
	short int nproc
)
		#endif
	#endif
{
	using dreal = typename LBM_TYPE::T_TRAITS::dreal;
	using idx = typename LBM_TYPE::T_TRAITS::idx;
	using map_t = typename LBM_TYPE::T_TRAITS::map_t;

	#ifndef TODO
		#ifdef USE_CUDA
	idx x = threadIdx.x + blockIdx.x * blockDim.x + offset_x;
	idx y = threadIdx.y + blockIdx.y * blockDim.y;
	idx z = threadIdx.z + blockIdx.z * blockDim.z;
		#endif
	#endif
	map_t gi_map = SD.map(x, y, z);

	idx xp,xm,yp,ym,zp,zm;
	if (LBM_BC::isPeriodic(gi_map))
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
//		xp = (SD.overlap_right) ? x+1 : TNL::min(x+1, SD.X-1);
//		xm = (SD.overlap_left) ? x-1 : TNL::max(x-1,0);
		xp = (rank != nproc-1) ? x+1 : TNL::min(x+1, SD.X()-1);
		xm = (rank != 0) ? x-1 : TNL::max(x-1,0);
		yp = TNL::min(y+1, SD.Y()-1);
		ym = TNL::max(y-1,0);
		zp = TNL::min(z+1, SD.Z()-1);
		zm = TNL::max(z-1,0);
	}

	map_t gi_map_xp = SD.map(xp, y, z);
	map_t gi_map_xm = SD.map(xm, y, z);
	map_t gi_map_yp = SD.map(x, yp, z);
	map_t gi_map_ym = SD.map(x, ym, z);
	map_t gi_map_zp = SD.map(x, y, zp);
	map_t gi_map_zm = SD.map(x, y, zm);

	//if(gi_map_xm != gi_map)
	printf("Kontrola mapy:\nxm = %d, x = %d, xp=%d\n gi_xm = %d, gi_x = %d, gi_xp = %d\n",(int)xm,(int)x,(int)xp,gi_map_xm, gi_map, gi_map_xp);
}


template <
	typename LBM_TYPE,
	typename STREAMING,
	typename MACRO,
	typename LBM_DATA,
	typename LBM_BC
>
	#ifdef TODO
CUDA_HOSTDEV
void LBMKernelCheckVelocity(
	typename LBM_TYPE::T_TRAITS::idx x,
	typename LBM_TYPE::T_TRAITS::idx y,
	typename LBM_TYPE::T_TRAITS::idx z,
	LBM_DATA SD,
	short int rank,
	short int nproc,
    int iter
)
	#else
		#ifdef USE_CUDA
//__launch_bounds__(32, 16)
__global__ void cudaLBMKernelCheckVelocity(
	LBM_DATA SD,
	short int rank,
	short int nproc,
	typename LBM_TYPE::T_TRAITS::idx offset_x,
    int iter
)
		#else
CUDA_HOSTDEV
void LBMKernelCheckVelocity(
	LBM_DATA SD,
	typename LBM_TYPE::T_TRAITS::idx x,
	typename LBM_TYPE::T_TRAITS::idx y,
	typename LBM_TYPE::T_TRAITS::idx z,
	short int rank,
	short int nproc,
    int iter
)
		#endif
	#endif
{
	using dreal = typename LBM_TYPE::T_TRAITS::dreal;
	using idx = typename LBM_TYPE::T_TRAITS::idx;
	using map_t = typename LBM_TYPE::T_TRAITS::map_t;

	#ifndef TODO
		#ifdef USE_CUDA
	idx x = threadIdx.x + blockIdx.x * blockDim.x + offset_x;
	idx y = threadIdx.y + blockIdx.y * blockDim.y;
	idx z = threadIdx.z + blockIdx.z * blockDim.z;
		#endif
	#endif
	map_t gi_map = SD.map(x, y, z);

	KernelStruct<dreal> KS;

	KernelStruct<dreal> KSxp, KSxm;

	// copy quantities
	MACRO::copyQuantities(SD, KS, x, y, z);

	idx xp,xm,yp,ym,zp,zm;
	if (LBM_BC::isPeriodic(gi_map))
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
//		xp = (SD.overlap_right) ? x+1 : TNL::min(x+1, SD.X-1);
//		xm = (SD.overlap_left) ? x-1 : TNL::max(x-1,0);
		xp = (rank != nproc-1) ? x+1 : TNL::min(x+1, SD.X()-1);
		xm = (rank != 0) ? x-1 : TNL::max(x-1,0);
		yp = TNL::min(y+1, SD.Y()-1);
		ym = TNL::max(y-1,0);
		zp = TNL::min(z+1, SD.Z()-1);
		zm = TNL::max(z-1,0);
	}

	MACRO::getMacro(SD, KSxp, xp, y, z);
	MACRO::getMacro(SD, KSxm, xm, y, z);

	MACRO::getMacro(SD, KS, x, y, z);

	map_t gi_map_xp = SD.map(xp, y, z);
	map_t gi_map_xm = SD.map(xm, y, z);
	map_t gi_map_yp = SD.map(x, yp, z);
	map_t gi_map_ym = SD.map(x, ym, z);
	map_t gi_map_zp = SD.map(x, y, zp);
	map_t gi_map_zm = SD.map(x, y, zm);

	if(y == (idx)floor(SD.Y()/2.) && z == (idx)floor(SD.Z()/2.))
	{
		if(rank == 0)
		{
			printf("Velocity rank %d, iter = %d vlevo:\n C: vx = %e, vy = %e, vz = %e\n R: vx = %e, vy = %e, vz = %e\nx = %d, xp = %d, y = %d, z = %d\n",(int)rank,iter, KS.vx, KS.vy, KS.vz, KSxp.vx, KSxp.vy, KSxp.vz,(int)x, (int)xp,(int)y, (int)z);
		}
		else if(rank == 1)
		{
			printf("Velocity rank %d, iter = %d vpravo:\n C: vx = %e, vy = %e, vz = %e\n L: vx = %e, vy = %e, vz = %e\nxm = %d, x = %d, y = %d, z = %d\n",(int)rank, iter, KS.vx, KS.vy, KS.vz, KSxm.vx, KSxm.vy, KSxm.vz,(int)xm, (int)x,(int)y, (int)z);
		}
	}
}
#endif

template <typename NSE>
#ifdef USE_CUDA
__global__ void cudaLBMKernelVelocity(typename NSE::DATA SD, short int nproc, typename NSE::TRAITS::idx3d offset, typename NSE::TRAITS::idx3d end)
#else
CUDA_HOSTDEV void
LBMKernelVelocity(typename NSE::DATA SD, typename NSE::TRAITS::idx x, typename NSE::TRAITS::idx y, typename NSE::TRAITS::idx z, short int nproc)
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

	typename NSE::template KernelStruct<dreal> KS;

	// copy quantities
	NSE::MACRO::copyQuantities(SD, KS, x, y, z);

	idx xp, xm, yp, ym, zp, zm;
	kernelInitIndices<NSE>(SD, gi_map, nproc, x, y, z, xp, xm, yp, ym, zp, zm);

	NSE::MACRO::getForce(SD, KS, x, y, z);

	// Streaming
	if (NSE::BC::isStreaming(gi_map))
		NSE::STREAMING::streaming(SD, KS, xm, x, xp, ym, y, yp, zm, z, zp);
	else if (NSE::BC::isWall(gi_map))
		NSE::STREAMING::streamingBounceBack(SD, KS, xm, x, xp, ym, y, yp, zm, z, zp);

	// compute Density & Velocity
	if (NSE::BC::isComputeDensityAndVelocity(gi_map))
		NSE::COLL::computeDensityAndVelocity(KS);
	else if (NSE::BC::isWall(gi_map))
		NSE::COLL::computeDensityAndVelocity_Wall(KS);
	else if (NSE::BC::isInflow(gi_map)) {
		NSE::STREAMING::streamingRho(SD, KS, xm, x, xp, ym, y, yp, zm, z, zp);
		SD.inflow(KS, x, y, z);
	}
	else if (NSE::BC::isOutflowR(gi_map)) {
		NSE::STREAMING::streamingVx(SD, KS, xm, x, xp, ym, y, yp, zm, z, zp);
		NSE::STREAMING::streamingVy(SD, KS, xm, x, xp, ym, y, yp, zm, z, zp);
		NSE::STREAMING::streamingVz(SD, KS, xm, x, xp, ym, y, yp, zm, z, zp);
		KS.rho = no1;
	}

	NSE::MACRO::outputDensityAndVelocity(SD, KS, x, y, z);
}

template <typename NSE>
#ifdef USE_CUDA
__global__ void cudaLBMKernelStress(typename NSE::DATA SD, short int nproc, typename NSE::TRAITS::idx3d offset, typename NSE::TRAITS::idx3d end)
#else
CUDA_HOSTDEV void
LBMKernelStress(typename NSE::DATA SD, typename NSE::TRAITS::idx x, typename NSE::TRAITS::idx y, typename NSE::TRAITS::idx z, short int nproc)
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

	typename NSE::template KernelStruct<dreal> KS;

	typename NSE::template KernelStruct<dreal> KSxp, KSxm, KSyp, KSym, KSzp, KSzm;

	// copy quantities
	NSE::MACRO::copyQuantities(SD, KS, x, y, z);

	idx xp, xm, yp, ym, zp, zm;
	kernelInitIndices<NSE>(SD, gi_map, nproc, x, y, z, xp, xm, yp, ym, zp, zm);

	NSE::MACRO::getMacro(SD, KSxp, xp, y, z);
	NSE::MACRO::getMacro(SD, KSxm, xm, y, z);
	NSE::MACRO::getMacro(SD, KSyp, x, yp, z);
	NSE::MACRO::getMacro(SD, KSym, x, ym, z);
	NSE::MACRO::getMacro(SD, KSzp, x, y, zp);
	NSE::MACRO::getMacro(SD, KSzm, x, y, zm);
	NSE::MACRO::getMacro(SD, KS, x, y, z);

	map_t gi_map_xp = SD.map(xp, y, z);
	map_t gi_map_xm = SD.map(xm, y, z);
	map_t gi_map_yp = SD.map(x, yp, z);
	map_t gi_map_ym = SD.map(x, ym, z);
	map_t gi_map_zp = SD.map(x, y, zp);
	map_t gi_map_zm = SD.map(x, y, zm);

	if (NSE::BC::isFluid(gi_map)) {
		//derivation in x-direction
		if (NSE::BC::isNotFluid(gi_map_xm)) {
			if (NSE::BC::isNotFluid(gi_map_xp)) {
				KS.S11 = 0.;
			}
			else {
				KS.S11 = (KSxp.vx - KS.vx);
				KS.S12 += n1o2 * (KSxp.vy - KS.vy);
				KS.S13 += n1o2 * (KSxp.vz - KS.vz);
			}
		}
		else if (NSE::BC::isNotFluid(gi_map_xp)) {
			KS.S11 = (KS.vx - KSxm.vx);
			KS.S12 += n1o2 * (KS.vy - KSxm.vy);
			KS.S13 += n1o2 * (KS.vz - KSxm.vz);
		}
		else {
			KS.S11 = n1o2 * (KSxp.vx - KSxm.vx);
			KS.S12 += n1o4 * (KSxp.vy - KSxm.vy);
			KS.S13 += n1o4 * (KSxp.vz - KSxm.vz);
		}

		//derivation in y-direction
		if (NSE::BC::isNotFluid(gi_map_ym)) {
			if (NSE::BC::isNotFluid(gi_map_yp)) {
				KS.S22 = 0.;
			}
			else {
				KS.S22 = (KSyp.vy - KS.vy);
				KS.S12 += n1o2 * (KSyp.vx - KS.vx);
				KS.S32 += n1o2 * (KSyp.vz - KS.vz);
			}
		}
		else if (NSE::BC::isNotFluid(gi_map_yp)) {
			KS.S22 = (KS.vy - KSym.vy);
			KS.S12 += n1o2 * (KS.vx - KSym.vx);
			KS.S32 += n1o2 * (KS.vz - KSym.vz);
		}
		else {
			KS.S22 = n1o2 * (KSyp.vy - KSym.vy);
			KS.S12 += n1o4 * (KSyp.vx - KSym.vx);
			KS.S32 += n1o4 * (KSyp.vz - KSym.vz);
		}

		//derivation in z-direction
		if (NSE::BC::isNotFluid(gi_map_zm)) {
			if (NSE::BC::isNotFluid(gi_map_zp)) {
				KS.S33 = 0.;
			}
			else {
				KS.S33 = (KSzp.vz - KS.vz);
				KS.S13 += n1o2 * (KSzp.vx - KS.vx);
				KS.S32 += n1o2 * (KSzp.vy - KS.vy);
			}
		}
		else if (NSE::BC::isNotFluid(gi_map_zp)) {
			KS.S33 = (KS.vz - KSzm.vz);
			KS.S13 += n1o2 * (KS.vx - KSzm.vx);
			KS.S32 += n1o2 * (KS.vy - KSzm.vy);
		}
		else {
			KS.S33 = n1o2 * (KSzp.vz - KSzm.vz);
			KS.S13 += n1o4 * (KSzp.vx - KSzm.vx);
			KS.S32 += n1o4 * (KSzp.vy - KSzm.vy);
		}
	}

	NSE::MACRO::outputMacrodef(SD, KS, x, y, z);
}

template <typename STATE>
void computeNonNewtonianKernels(STATE& state)
{
	using NSE = typename STATE::NSE;
	using TRAITS = typename NSE::TRAITS;

	using idx = typename TRAITS::idx;
	using idx3d = typename TRAITS::idx3d;
	using dreal = typename TRAITS::dreal;

	auto& nse = state.nse;

	const auto boundary_directions = {
		TNL::Containers::SyncDirection::Bottom,
		TNL::Containers::SyncDirection::Top,
		TNL::Containers::SyncDirection::Back,
		TNL::Containers::SyncDirection::Front,
		TNL::Containers::SyncDirection::Left,
		TNL::Containers::SyncDirection::Right,
	};

	// compute on boundaries
	for (auto& block : nse.blocks) {
		for (auto direction : boundary_directions)
			if (auto search = block.neighborIDs.find(direction); search != block.neighborIDs.end() && search->second >= 0) {
				const dim3 blockSize = block.computeData.at(direction).blockSize;
				const dim3 gridSize = block.computeData.at(direction).gridSize;
				const cudaStream_t stream = block.computeData.at(direction).stream;
				const idx3d offset = block.computeData.at(direction).offset;
				const idx3d size = block.computeData.at(direction).size;
				cudaLBMKernelVelocity<NSE><<<gridSize, blockSize, 0, stream>>>(block.data, nse.total_blocks, offset, offset + size);
			}
	}

	// compute on interior lattice sites
	for (auto& block : nse.blocks) {
		const auto direction = TNL::Containers::SyncDirection::None;
		const dim3 blockSize = block.computeData.at(direction).blockSize;
		const dim3 gridSize = block.computeData.at(direction).gridSize;
		const cudaStream_t stream = block.computeData.at(direction).stream;
		const idx3d offset = block.computeData.at(direction).offset;
		const idx3d size = block.computeData.at(direction).size;
		cudaLBMKernelVelocity<NSE><<<gridSize, blockSize, 0, stream>>>(block.data, nse.total_blocks, offset, offset + size);
	}

	// wait for the computations on boundaries to finish
	for (auto& block : nse.blocks)
		for (auto direction : boundary_directions)
			cudaStreamSynchronize(block.computeData.at(direction).stream);

	// exchange macroscopic quantities on overlaps between blocks
	// TODO: avoid communication of DFs here
	nse.synchronizeDFsAndMacroDevice(df_out);

	// wait for the computation on the interior to finish
	for (auto& block : nse.blocks) {
		const cudaStream_t stream = block.computeData.at(TNL::Containers::SyncDirection::None).stream;
		cudaStreamSynchronize(stream);
	}

	// synchronize the whole GPU and check errors
	cudaDeviceSynchronize();
	TNL_CHECK_CUDA_DEVICE;

#if 0
//	cudaLBMKernelCheckMap< NSE, STREAMING, MACRO, LBM_DATA, LBM_BC><<<gridSizeForBoundary, blockSize, 0, cuda_stream_left>>>(lbm.data, lbm.rank, lbm.nproc, lbm.local_X - lbm.df_overlap_X());

	TNL::MPI::Barrier();
	if(lbm.rank == 0)
	{
		cudaLBMKernelCheckVelocity< NSE, STREAMING, MACRO, LBM_DATA, LBM_BC><<<gridSizeForBoundary, blockSize, 0, cuda_stream_left>>>(lbm.data, lbm.rank, lbm.nproc, lbm.local_X - lbm.df_overlap_X(),lbm.iterations);
	}
	else if(lbm.rank == 1)
	{
		cudaLBMKernelCheckVelocity< NSE, STREAMING, MACRO, LBM_DATA, LBM_BC><<<gridSizeForBoundary, blockSize, 0, cuda_stream_right>>>(lbm.data, lbm.rank, lbm.nproc, (idx)0,lbm.iterations);
	}
	cudaStreamSynchronize(cuda_stream_left);
	cudaStreamSynchronize(cuda_stream_right);
	TNL::MPI::Barrier();
#endif

	// compute on boundaries
	for (auto& block : nse.blocks) {
		for (auto direction : boundary_directions)
			if (auto search = block.neighborIDs.find(direction); search != block.neighborIDs.end() && search->second >= 0) {
				const dim3 blockSize = block.computeData.at(direction).blockSize;
				const dim3 gridSize = block.computeData.at(direction).gridSize;
				const cudaStream_t stream = block.computeData.at(direction).stream;
				const idx3d offset = block.computeData.at(direction).offset;
				const idx3d size = block.computeData.at(direction).size;
				cudaLBMKernelStress<NSE><<<gridSize, blockSize, 0, stream>>>(block.data, nse.total_blocks, offset, offset + size);
			}
	}

	// compute on interior lattice sites
	for (auto& block : nse.blocks) {
		const auto direction = TNL::Containers::SyncDirection::None;
		const dim3 blockSize = block.computeData.at(direction).blockSize;
		const dim3 gridSize = block.computeData.at(direction).gridSize;
		const cudaStream_t stream = block.computeData.at(direction).stream;
		const idx3d offset = block.computeData.at(direction).offset;
		const idx3d size = block.computeData.at(direction).size;
		cudaLBMKernelStress<NSE><<<gridSize, blockSize, 0, stream>>>(block.data, nse.total_blocks, offset, offset + size);
	}

	// wait for the computations on boundaries to finish
	for (auto& block : nse.blocks)
		for (auto direction : boundary_directions)
			cudaStreamSynchronize(block.computeData.at(direction).stream);

	// exchange macroscopic quantities on overlaps between blocks
	// TODO: avoid communication of DFs here
	nse.synchronizeDFsAndMacroDevice(df_out);

	// wait for the computation on the interior to finish
	for (auto& block : nse.blocks) {
		const cudaStream_t stream = block.computeData.at(TNL::Containers::SyncDirection::None).stream;
		cudaStreamSynchronize(stream);
	}

	// synchronize the whole GPU and check errors
	cudaDeviceSynchronize();
	TNL_CHECK_CUDA_DEVICE;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                                                                                                                                                                                                //
//                                                                                                                                                                                                                //
// Default "LBM data" class for the non-Newtonian fluid model
//                                                                                                                                                                                                                //
//                                                                                                                                                                                                                //
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "lbm_data.h"

template <typename TRAITS>
struct LBM_Data_NonNewtonian : NSE_Data<TRAITS>
{
	using dreal = typename TRAITS::dreal;

	// Non-Newtonian parameters (only dummy values here -- they have to be initialized from sim)
#if defined(USE_CYMODEL)
	dreal lbm_nu0;
	dreal lbm_lambda;
	dreal lbm_a;
	dreal lbm_n;
#elif defined(USE_CASSON)
	dreal lbm_k0;
	dreal lbm_k1;
#endif
};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                                                                                                                                                                                                //
//                                                                                                                                                                                                                //
// Default macro class for the non-Newtonian fluid model
//                                                                                                                                                                                                                //
//                                                                                                                                                                                                                //
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "d3q27/macro.h"

template <typename TRAITS>
struct MacroNonNewtonianDefault : D3Q27_MACRO_Default<TRAITS>
{
	using dreal = typename TRAITS::dreal;
	using idx = typename TRAITS::idx;
	using map_t = typename TRAITS::map_t;

	enum QuantityNames : std::uint8_t
	{
		e_rho,
		e_vx,
		e_vy,
		e_vz,
		e_vm_plus_x,
		e_vm_minus_x,
		e_vm_y,
		e_vm_z,
		e_vm_xx,
		e_vm_yy,
		e_vm_zz,
		e_fx,
		e_fy,
		e_fz,
		e_S11,
		e_S12,
		e_S13,
		e_S22,
		e_S32,
		e_S33,
		N
	};

	template <typename LBM_DATA, typename LBM_KS>
	CUDA_HOSTDEV static void outputMacro(LBM_DATA& SD, LBM_KS& KS, idx x, idx y, idx z)
	{
		SD.macro(e_rho, x, y, z) = KS.rho;
		SD.macro(e_vx, x, y, z) = KS.vx;
		SD.macro(e_vy, x, y, z) = KS.vy;
		SD.macro(e_vz, x, y, z) = KS.vz;

		SD.macro(e_fx, x, y, z) = KS.fx;
		SD.macro(e_fy, x, y, z) = KS.fy;
		SD.macro(e_fz, x, y, z) = KS.fz;

		SD.macro(e_vm_plus_x, x, y, z) += (KS.vx > 0) ? KS.rho * KS.vx : 0;
		SD.macro(e_vm_minus_x, x, y, z) += (KS.vx < 0) ? KS.rho * KS.vx : 0;

		SD.macro(e_vm_y, x, y, z) += KS.rho * KS.vy;
		SD.macro(e_vm_z, x, y, z) += KS.rho * KS.vz;

		SD.macro(e_vm_xx, x, y, z) += KS.rho * KS.rho * KS.vx * KS.vx;
		SD.macro(e_vm_yy, x, y, z) += KS.rho * KS.rho * KS.vy * KS.vy;
		SD.macro(e_vm_zz, x, y, z) += KS.rho * KS.rho * KS.vz * KS.vz;
	}

	template <typename LBM_DATA, typename LBM_KS>
	CUDA_HOSTDEV static void outputDensityAndVelocity(LBM_DATA& SD, LBM_KS& KS, idx x, idx y, idx z)
	{
		SD.macro(e_rho, x, y, z) = KS.rho;
		SD.macro(e_vx, x, y, z) = KS.vx;
		SD.macro(e_vy, x, y, z) = KS.vy;
		SD.macro(e_vz, x, y, z) = KS.vz;
	}

	template <typename LBM_DATA, typename LBM_KS>
	CUDA_HOSTDEV static void getForce(LBM_DATA& SD, LBM_KS& KS, idx x, idx y, idx z)
	{
		KS.fx = SD.macro(e_fx, x, y, z);
		KS.fy = SD.macro(e_fy, x, y, z);
		KS.fz = SD.macro(e_fz, x, y, z);
	}

	template <typename LBM_DATA, typename LBM_KS>
	CUDA_HOSTDEV static void getMacro(LBM_DATA& SD, LBM_KS& KS, idx x, idx y, idx z)
	{
		KS.rho = SD.macro(e_rho, x, y, z);
		KS.vx = SD.macro(e_vx, x, y, z);
		KS.vy = SD.macro(e_vy, x, y, z);
		KS.vz = SD.macro(e_vz, x, y, z);
	}

	template <typename LBM_DATA, typename LBM_KS>
	CUDA_HOSTDEV static void outputMacrodef(LBM_DATA& SD, LBM_KS& KS, idx x, idx y, idx z)
	{
		SD.macro(e_S11, x, y, z) = KS.S11;
		SD.macro(e_S12, x, y, z) = KS.S12;
		SD.macro(e_S22, x, y, z) = KS.S22;
		SD.macro(e_S32, x, y, z) = KS.S32;
		SD.macro(e_S13, x, y, z) = KS.S13;
		SD.macro(e_S33, x, y, z) = KS.S33;
	}

	template <typename LBM_DATA, typename LBM_KS>
	CUDA_HOSTDEV static void getDef(LBM_DATA& SD, LBM_KS& KS, idx x, idx y, idx z)
	{
		// do globalniho S zapsat S
		KS.S11 = SD.macro(e_S11, x, y, z);
		KS.S12 = SD.macro(e_S12, x, y, z);
		KS.S22 = SD.macro(e_S22, x, y, z);
		KS.S32 = SD.macro(e_S32, x, y, z);
		KS.S13 = SD.macro(e_S13, x, y, z);
		KS.S33 = SD.macro(e_S33, x, y, z);
	}

	template <typename LBM_DATA, typename LBM_KS>
	CUDA_HOSTDEV static void copyQuantities(LBM_DATA& SD, LBM_KS& KS, idx x, idx y, idx z)
	{
		KS.lbmViscosity = SD.lbmViscosity;
		KS.fx = SD.fx;
		KS.fy = SD.fy;
		KS.fz = SD.fz;

#if defined(USE_CYMODEL)
		KS.lbm_nu0 = SD.lbm_nu0;
		KS.lbm_lambda = SD.lbm_lambda;
		KS.lbm_a = SD.lbm_a;
		KS.lbm_n = SD.lbm_n;
#elif defined(USE_CASSON)
		KS.lbm_k0 = SD.lbm_k0;
		KS.lbm_k1 = SD.lbm_k1;
#endif
	}

	template <typename LBM_BC, typename LBM_DATA, typename LBM_KS>
	CUDA_HOSTDEV static void computeForcing(LBM_DATA& SD, LBM_KS& KS, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm, idx z, idx zp)
	{
		map_t gi_map = SD.map(x, y, z);
		map_t gi_map_xp = SD.map(xp, y, z);
		map_t gi_map_xm = SD.map(xm, y, z);
		map_t gi_map_yp = SD.map(x, yp, z);
		map_t gi_map_ym = SD.map(x, ym, z);
		map_t gi_map_zp = SD.map(x, y, zp);
		map_t gi_map_zm = SD.map(x, y, zm);

		LBM_KS KSxp, KSxm, KSyp, KSym, KSzp, KSzm;

		getDef(SD, KSxp, xp, y, z);
		getDef(SD, KSxm, xm, y, z);
		getDef(SD, KSyp, x, yp, z);
		getDef(SD, KSym, x, ym, z);
		getDef(SD, KSzp, x, y, zp);
		getDef(SD, KSzm, x, y, zm);
		getDef(SD, KS, x, y, z);

		dreal F1 = 0;
		dreal F2 = 0;
		dreal F3 = 0;

		if (LBM_BC::isFluid(gi_map)) {
			// derivative in x-direction
			if (LBM_BC::isNotFluid(gi_map_xm)) {
				if (LBM_BC::isNotFluid(gi_map_xp)) {
				}
				else {
					F1 += KSxp.S11 - KS.S11;
					F2 += KSxp.S12 - KS.S12;
					F3 += KSxp.S13 - KS.S13;
				}
			}
			else if (LBM_BC::isNotFluid(gi_map_xp)) {
				F1 += KS.S11 - KSxm.S11;
				F2 += KS.S12 - KSxm.S12;
				F3 += KS.S13 - KSxm.S13;
			}
			else {
				F1 += n1o2 * (KSxp.S11 - KSxm.S11);
				F2 += n1o2 * (KSxp.S12 - KSxm.S12);
				F3 += n1o2 * (KSxp.S13 - KSxm.S13);
			}

			// derivative in y-direction
			if (LBM_BC::isNotFluid(gi_map_ym)) {
				if (LBM_BC::isNotFluid(gi_map_yp)) {
				}
				else {
					F1 += KSyp.S12 - KS.S12;
					F2 += KSyp.S22 - KS.S22;
					F3 += KSyp.S32 - KS.S32;
				}
			}
			else if (LBM_BC::isNotFluid(gi_map_yp)) {
				F1 += KS.S12 - KSym.S12;
				F2 += KS.S22 - KSym.S22;
				F3 += KS.S32 - KSym.S32;
			}
			else {
				F1 += n1o2 * (KSyp.S12 - KSym.S12);
				F2 += n1o2 * (KSyp.S22 - KSym.S22);
				F3 += n1o2 * (KSyp.S32 - KSym.S32);
			}

			// derivative in z-direction
			if (LBM_BC::isNotFluid(gi_map_zm)) {
				if (LBM_BC::isNotFluid(gi_map_zp)) {
				}
				else {
					F1 += KSzp.S13 - KS.S13;
					F2 += KSzp.S32 - KS.S32;
					F3 += KSzp.S33 - KS.S33;
				}
			}
			else if (LBM_BC::isNotFluid(gi_map_zp)) {
				F1 += KS.S13 - KSzm.S13;
				F2 += KS.S32 - KSzm.S32;
				F3 += KS.S33 - KSzm.S33;
			}
			else {
				F1 += n1o2 * (KSzp.S13 - KSzm.S13);
				F2 += n1o2 * (KSzp.S32 - KSzm.S32);
				F3 += n1o2 * (KSzp.S33 - KSzm.S33);
			}
		}

		dreal gamma = sqrt(KS.S11 * KS.S11 + KS.S22 * KS.S22 + KS.S33 * KS.S33 + no2 * (KS.S12 * KS.S12 + KS.S13 * KS.S13 + KS.S32 * KS.S32));

#ifdef USE_CYMODEL
		dreal nu =
			KS.lbmViscosity + (KS.lbm_nu0 - KS.lbmViscosity) * powf((no1 + powf((gamma * KS.lbm_lambda), KS.lbm_a)), (KS.lbm_n - no1) / KS.lbm_a);
#elif USE_CASSON
		dreal nu;
		if (sqrt(gamma) > 1e-10) {
			nu = (KS.lbm_k0 + KS.lbm_k1 * sqrt(gamma)) * (KS.lbm_k0 + KS.lbm_k1 * sqrt(gamma)) / sqrt(gamma);
		}
		else
			nu = KS.lbmViscosity;
#endif

		KS.mu = nu * 1000;

		KS.fx += no2 * (nu - KS.lbmViscosity) * F1 * KS.rho;
		KS.fy += no2 * (nu - KS.lbmViscosity) * F2 * KS.rho;
		KS.fz += no2 * (nu - KS.lbmViscosity) * F3 * KS.rho;
	}
};
