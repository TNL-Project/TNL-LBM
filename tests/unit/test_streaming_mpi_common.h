/*
 * Shared machinery of the streaming-pattern MPI unit test: all six patterns
 * (A-A, A-B pull/push, esoteric twist/pull/push) for both D2Q9 and D3Q27,
 * the unfused gather/collision/scatter kernels replicating State::SimUpdate,
 * the BC-dispatch variants with the production outflow pass, and the
 * snapshot/comparison helpers. See test_streaming_mpi.cu for the test
 * documentation.
 *
 * The BC-dispatch cases are compiled in separate translation units per
 * lattice model (test_streaming_mpi_bc2d.cu / test_streaming_mpi_bc3d.cu):
 * instantiating the per-pattern BC switch 12 times dominates the compile
 * time, and the split lets ninja schedule the model TUs in parallel with
 * this one instead of dragging every per-pattern BC body through a single
 * sequential CUDA compilation.
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <string>

#include <doctest/doctest.h>

#include <fmt/core.h>
#include <spdlog/spdlog.h>

#include <TNL/Containers/DistributedNDArraySyncDirections.h>

#include "lbm3d/lbm.h"
#include "lbm3d/lbm_data.h"
#include "lbm3d/kernels.h"
#include "lbm3d/DataManager.h"

#include "lbm3d/d2q9/bc.h"
#include "lbm3d/d2q9/col_srt.h"
#include "lbm3d/d2q9/macro.h"
#include "lbm3d/d2q9/streaming.h"

#include "lbm3d/d3q27/bc.h"
#include "lbm3d/d3q27/col_srt.h"
#include "lbm3d/d3q27/macro.h"
#include "lbm3d/d3q27/streaming.h"

using TRAITS = Traits<double, double, int>;
using SDirection = TNL::Containers::SyncDirection;

// minimal inflow member is required by the BC interface (unused on a fully
// periodic domain); the values are runtime members so the BC-dispatch channel
// cases can impose a non-zero inflow without instantiating a second set of
// LBM configs
template <typename TRAITS_, int DFS_COUNT>
struct Test2D_Data : NSE_Data<TRAITS_, DFS_COUNT>
{
	using idx = typename TRAITS_::idx;
	using dreal = typename TRAITS_::dreal;

	dreal inflow_vx = 0;
	dreal inflow_vy = 0;

	template <typename LBM_KS>
	CUDA_HOSTDEV void inflow(LBM_KS& KS, idx x, idx y, idx z)
	{
		KS.vx = inflow_vx;
		KS.vy = inflow_vy;
	}
};

template <typename TRAITS_, int DFS_COUNT>
struct Test3D_Data : NSE_Data<TRAITS_, DFS_COUNT>
{
	using idx = typename TRAITS_::idx;
	using dreal = typename TRAITS_::dreal;

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

using COLL2D = D2Q9_SRT<TRAITS>;
template <typename STREAM>
using CONFIG2D = LBM_CONFIG<TRAITS, D2Q9_KernelStruct, Test2D_Data, COLL2D, typename COLL2D::EQ, STREAM, D2Q9_BC_All, D2Q9_MACRO_Default<TRAITS>>;

using COLL3D = D3Q27_SRT<TRAITS>;
template <typename STREAM>
using CONFIG3D = LBM_CONFIG<TRAITS, D3Q27_KernelStruct, Test3D_Data, COLL3D, typename COLL3D::EQ, STREAM, D3Q27_BC_All, D3Q27_MACRO_Default<TRAITS>>;

// the shared collision kernel's configuration per lattice model: the
// streaming member is never invoked inside it, so a single (arbitrary)
// pattern instance carries the collision, equilibrium and macroscopic parts
using COLL_CONFIG2D = CONFIG2D<D2Q9_STREAMING_AB_PULL<TRAITS>>;
using COLL_CONFIG3D = CONFIG3D<D3Q27_STREAMING_AB_PULL<TRAITS>>;

// unfused iteration kernels:

// per-pattern gather: the pattern's streaming loads the populations into
// the KernelStruct registers; they land in a natural-layout scratch array
// for the shared collision kernel
template <typename NSE>
__global__ void cudaStreamingGather(
	typename NSE::DATA SD,
	typename NSE::TRAITS::idx3d offset,
	typename NSE::TRAITS::idx3d end,
	typename NSE::TRAITS::bool3d distributed,
	typename NSE::TRAITS::dreal* scratch
)
{
	using idx = typename NSE::TRAITS::idx;
	using dreal = typename NSE::TRAITS::dreal;

	idx x = threadIdx.x + blockIdx.x * blockDim.x + offset.x();
	idx y = threadIdx.y + blockIdx.y * blockDim.y + offset.y();
	idx z = threadIdx.z + blockIdx.z * blockDim.z + offset.z();

	if (x >= end.x() || y >= end.y() || z >= end.z())
		return;

	idx xp, xm, yp, ym, zp, zm;
	kernelInitIndices<NSE>(SD, distributed, x, y, z, xp, xm, yp, ym, zp, zm);

	typename NSE::template KernelStruct<dreal> KS;
	NSE::STREAMING::streaming(SD, KS, xm, x, xp, ym, y, yp, zm, z, zp);

	const idx lin = SD.indexer.getStorageIndex(x, y, z);
	for (int i = 0; i < NSE::Q; i++)
		scratch[i * SD.XYZ + lin] = KS.f[i];
}

// shared, unfused collision: identical for all streaming patterns of one
// lattice model (macroscopic quantities from the gathered populations, SRT
// relaxation, post-collision populations back to the scratch array)
template <typename NSE>
__global__ void
cudaSharedCollision(typename NSE::DATA SD, typename NSE::TRAITS::idx3d offset, typename NSE::TRAITS::idx3d end, typename NSE::TRAITS::dreal* scratch)
{
	using idx = typename NSE::TRAITS::idx;
	using dreal = typename NSE::TRAITS::dreal;

	idx x = threadIdx.x + blockIdx.x * blockDim.x + offset.x();
	idx y = threadIdx.y + blockIdx.y * blockDim.y + offset.y();
	idx z = threadIdx.z + blockIdx.z * blockDim.z + offset.z();

	if (x >= end.x() || y >= end.y() || z >= end.z())
		return;

	const idx lin = SD.indexer.getStorageIndex(x, y, z);

	typename NSE::template KernelStruct<dreal> KS;
	NSE::MACRO::copyQuantities(SD, KS, x, y, z);
	for (int i = 0; i < NSE::Q; i++)
		KS.f[i] = scratch[i * SD.XYZ + lin];

	NSE::COLL::computeDensityAndVelocity(KS);
	NSE::COLL::collision(KS);

	for (int i = 0; i < NSE::Q; i++)
		scratch[i * SD.XYZ + lin] = KS.f[i];
	NSE::MACRO::outputMacro(SD, KS, x, y, z);
}

// per-pattern scatter: the post-collision populations return to the pattern
// layout via the pattern's own postCollisionStreaming at the current parity
template <typename NSE>
__global__ void cudaStreamingScatter(
	typename NSE::DATA SD,
	typename NSE::TRAITS::idx3d offset,
	typename NSE::TRAITS::idx3d end,
	typename NSE::TRAITS::bool3d distributed,
	const typename NSE::TRAITS::dreal* scratch
)
{
	using idx = typename NSE::TRAITS::idx;
	using dreal = typename NSE::TRAITS::dreal;

	idx x = threadIdx.x + blockIdx.x * blockDim.x + offset.x();
	idx y = threadIdx.y + blockIdx.y * blockDim.y + offset.y();
	idx z = threadIdx.z + blockIdx.z * blockDim.z + offset.z();

	if (x >= end.x() || y >= end.y() || z >= end.z())
		return;

	const idx lin = SD.indexer.getStorageIndex(x, y, z);

	typename NSE::template KernelStruct<dreal> KS;
	for (int i = 0; i < NSE::Q; i++)
		KS.f[i] = scratch[i * SD.XYZ + lin];

	idx xp, xm, yp, ym, zp, zm;
	kernelInitIndices<NSE>(SD, distributed, x, y, z, xp, xm, yp, ym, zp, zm);

	NSE::STREAMING::postCollisionStreaming(SD, KS, xm, x, xp, ym, y, yp, zm, z, zp);
}

// one unfused iteration, replicating State::SimUpdate's CUDA+MPI sequence:
// gather -> shared collision -> scatter launched per compute region
// (boundary slabs on their streams, interior on its stream), boundary
// streams joined before the pipelined DF+macro halo exchange at the
// pattern's parity, the interior stream joined afterwards
template <typename NSE, typename COLL_NSE, typename SCRATCH>
static void unfusedStep(LBM<NSE>& nse, const std::vector<typename COLL_NSE::DATA>& collision_data, std::vector<SCRATCH>& scratch)
{
	using dreal = typename NSE::TRAITS::dreal;
	using idx3d = typename NSE::TRAITS::idx3d;
	using bool3d = typename NSE::TRAITS::bool3d;

	constexpr SDirection boundary_directions[] = {
		SDirection::Bottom, SDirection::Top, SDirection::Back, SDirection::Front, SDirection::Left, SDirection::Right
	};

	auto launch_region = [&](std::size_t i, SDirection direction)
	{
		auto& block = nse.blocks[i];
		const auto& cd = block.computeData.at(direction);
		const bool3d distributed = block.is_distributed();
		dreal* scr = scratch[i].getData();
		TNL::Backend::LaunchConfiguration launch_config;
		launch_config.blockSize = cd.blockSize;
		launch_config.gridSize = cd.gridSize;
		launch_config.stream = cd.stream;
		const idx3d begin = cd.offset;
		const idx3d end = cd.offset + cd.size;
		TNL::Backend::launchKernelAsync(cudaStreamingGather<NSE>, launch_config, block.data, begin, end, distributed, scr);
		TNL::Backend::launchKernelAsync(cudaSharedCollision<COLL_NSE>, launch_config, collision_data[i], begin, end, scr);
		TNL::Backend::launchKernelAsync(cudaStreamingScatter<NSE>, launch_config, block.data, begin, end, distributed, scr);
	};

	for (std::size_t i = 0; i < nse.blocks.size(); i++) {
		// compute on boundaries
		for (const SDirection direction : boundary_directions)
			if (auto search = nse.blocks[i].neighborIDs.find(direction); search != nse.blocks[i].neighborIDs.end() && search->second >= 0)
				launch_region(i, direction);
		// compute on interior lattice sites
		launch_region(i, SDirection::None);
	}

#ifdef HAVE_MPI
	if (nse.nproc > 1) {
		// wait for the computations on boundaries to finish
		for (std::size_t i = 0; i < nse.blocks.size(); i++)
			for (const SDirection direction : boundary_directions)
				if (auto search = nse.blocks[i].neighborIDs.find(direction); search != nse.blocks[i].neighborIDs.end() && search->second >= 0)
					TNL::Backend::streamSynchronize(nse.blocks[i].computeData.at(direction).stream);

		// exchange the latest DFs and dmacro on overlaps between blocks
		nse.synchronizeDFsAndMacroDevice(NSE::STREAMING::output_df, true);
	}
#endif

	// wait for the computation on the interior to finish
	for (auto& block : nse.blocks)
		TNL::Backend::streamSynchronize(block.computeData.at(SDirection::None).stream);
}

// unfused BC-dispatch kernels:
// The same per-site call sequence as the fused cudaLBMKernel and
// cudaLBMKernelOutflow (kernels.h), split at the same seams as the periodic
// pipeline above: the gather runs copyQuantities, the (no-op) forcing hook
// and the pattern's BC::preCollision (which dispatches the whole GEO-tag
// switch - walls, inflow moment, symmetry closure - around the pattern's own
// STREAMING::streaming), the shared collision applies the production
// BC::doCollision gate, and the scatter writes the macroscopic quantities and
// runs BC::postCollision (the pattern's postCollisionStreaming). The
// two-pass outflow mirrors cudaLBMKernelOutflow: a dedicated launch over
// each block's outflow_boxes before the main kernels, running the pattern's
// BC::outflowPass (streamingOutflow{,Interp} gathers, collision,
// postCollisionStreaming) and authoring the outflow cells' macro.
//
// Unlike the periodic pipeline the scratch carries the whole KernelStruct
// state that the later stages read: preCollision already computes the
// density and velocity on BC sites (and overwrites them on walls and ghost
// cells), so the shared collision must not recompute them.

// scratch slot layout: populations [0, NSE::Q), then rho, the D velocity
// components, the D force components and lbmViscosity
template <typename NSE>
constexpr int scratchSlotCount()
{
	return NSE::Q + 2 * NSE::D + 2;
}

template <typename NSE, typename LBM_KS>
__cuda_callable__ void
scratchStoreKS(typename NSE::TRAITS::dreal* scratch, const LBM_KS& KS, typename NSE::TRAITS::idx lin, typename NSE::TRAITS::idx XYZ)
{
	for (int i = 0; i < NSE::Q; i++)
		scratch[i * XYZ + lin] = KS.f[i];
	int slot = NSE::Q;
	scratch[slot++ * XYZ + lin] = KS.rho;
	scratch[slot++ * XYZ + lin] = KS.vx;
	scratch[slot++ * XYZ + lin] = KS.vy;
	if constexpr (NSE::D == 3)
		scratch[slot++ * XYZ + lin] = KS.vz;
	scratch[slot++ * XYZ + lin] = KS.fx;
	scratch[slot++ * XYZ + lin] = KS.fy;
	if constexpr (NSE::D == 3)
		scratch[slot++ * XYZ + lin] = KS.fz;
	scratch[slot++ * XYZ + lin] = KS.lbmViscosity;
}

template <typename NSE, typename LBM_KS>
__cuda_callable__ void
scratchLoadKS(const typename NSE::TRAITS::dreal* scratch, LBM_KS& KS, typename NSE::TRAITS::idx lin, typename NSE::TRAITS::idx XYZ)
{
	for (int i = 0; i < NSE::Q; i++)
		KS.f[i] = scratch[i * XYZ + lin];
	int slot = NSE::Q;
	KS.rho = scratch[slot++ * XYZ + lin];
	KS.vx = scratch[slot++ * XYZ + lin];
	KS.vy = scratch[slot++ * XYZ + lin];
	if constexpr (NSE::D == 3)
		KS.vz = scratch[slot++ * XYZ + lin];
	KS.fx = scratch[slot++ * XYZ + lin];
	KS.fy = scratch[slot++ * XYZ + lin];
	if constexpr (NSE::D == 3)
		KS.fz = scratch[slot++ * XYZ + lin];
	KS.lbmViscosity = scratch[slot++ * XYZ + lin];
}

// per-pattern gather with the production BC dispatch: mirrors the fused
// kernel's prologue (copyQuantities + forcing hook + BC::preCollision)
template <typename NSE>
__global__ void cudaGatherBC(
	typename NSE::DATA SD,
	typename NSE::TRAITS::idx3d offset,
	typename NSE::TRAITS::idx3d end,
	typename NSE::TRAITS::bool3d distributed,
	typename NSE::TRAITS::dreal* scratch
)
{
	using idx = typename NSE::TRAITS::idx;
	using dreal = typename NSE::TRAITS::dreal;
	using map_t = typename NSE::TRAITS::map_t;

	idx x = threadIdx.x + blockIdx.x * blockDim.x + offset.x();
	idx y = threadIdx.y + blockIdx.y * blockDim.y + offset.y();
	idx z = threadIdx.z + blockIdx.z * blockDim.z + offset.z();

	if (x >= end.x() || y >= end.y() || z >= end.z())
		return;

	const map_t gi_map = SD.map(x, y, z);

	idx xp, xm, yp, ym, zp, zm;
	kernelInitIndices<NSE>(SD, distributed, x, y, z, xp, xm, yp, ym, zp, zm);

	typename NSE::template KernelStruct<dreal> KS;
	NSE::MACRO::copyQuantities(SD, KS, x, y, z);
	NSE::MACRO::template computeForcing<typename NSE::BC>(SD, KS, xm, x, xp, ym, y, yp, zm, z, zp);
	NSE::BC::preCollision(SD, KS, gi_map, xm, x, xp, ym, y, yp, zm, z, zp);

	const idx lin = SD.indexer.getStorageIndex(x, y, z);
	scratchStoreKS<NSE>(scratch, KS, lin, SD.XYZ);
}

// shared collision behind the production BC::doCollision gate: BC cells that
// do not collide (walls, ghost cells) keep the KernelStruct state authored by
// preCollision; the outflow-pass cells are not collided either (the outflow
// kernel authors them instead)
template <typename NSE>
__global__ void cudaSharedCollisionBC(
	typename NSE::DATA SD, typename NSE::TRAITS::idx3d offset, typename NSE::TRAITS::idx3d end, typename NSE::TRAITS::dreal* scratch
)
{
	using idx = typename NSE::TRAITS::idx;
	using dreal = typename NSE::TRAITS::dreal;
	using map_t = typename NSE::TRAITS::map_t;

	idx x = threadIdx.x + blockIdx.x * blockDim.x + offset.x();
	idx y = threadIdx.y + blockIdx.y * blockDim.y + offset.y();
	idx z = threadIdx.z + blockIdx.z * blockDim.z + offset.z();

	if (x >= end.x() || y >= end.y() || z >= end.z())
		return;

	const map_t gi_map = SD.map(x, y, z);
	const idx lin = SD.indexer.getStorageIndex(x, y, z);

	typename NSE::template KernelStruct<dreal> KS;
	scratchLoadKS<NSE>(scratch, KS, lin, SD.XYZ);

	if (NSE::BC::doCollision(gi_map))
		NSE::COLL::collision(KS);

	scratchStoreKS<NSE>(scratch, KS, lin, SD.XYZ);
}

// per-pattern scatter with the production BC dispatch: macroscopic output
// (skipped on the outflow-pass cells like the fused kernel), then
// BC::postCollision invoking the pattern's postCollisionStreaming
template <typename NSE>
__global__ void cudaScatterBC(
	typename NSE::DATA SD,
	typename NSE::TRAITS::idx3d offset,
	typename NSE::TRAITS::idx3d end,
	typename NSE::TRAITS::bool3d distributed,
	const typename NSE::TRAITS::dreal* scratch
)
{
	using idx = typename NSE::TRAITS::idx;
	using dreal = typename NSE::TRAITS::dreal;
	using map_t = typename NSE::TRAITS::map_t;

	idx x = threadIdx.x + blockIdx.x * blockDim.x + offset.x();
	idx y = threadIdx.y + blockIdx.y * blockDim.y + offset.y();
	idx z = threadIdx.z + blockIdx.z * blockDim.z + offset.z();

	if (x >= end.x() || y >= end.y() || z >= end.z())
		return;

	const map_t gi_map = SD.map(x, y, z);
	const idx lin = SD.indexer.getStorageIndex(x, y, z);

	typename NSE::template KernelStruct<dreal> KS;
	scratchLoadKS<NSE>(scratch, KS, lin, SD.XYZ);

	idx xp, xm, yp, ym, zp, zm;
	kernelInitIndices<NSE>(SD, distributed, x, y, z, xp, xm, yp, ym, zp, zm);

	NSE::BC::postCollision(SD, KS, gi_map, xm, x, xp, ym, y, yp, zm, z, zp);

	bool skip_macro = false;
	// macro of outflow cells is authored by the outflow pass
	if constexpr (NSE::BC::use_outflow_pass)
		skip_macro = NSE::BC::isOutflowPassBC(gi_map);
	if (! skip_macro)
		NSE::MACRO::outputMacro(SD, KS, x, y, z);
}

// deterministic two-pass outflow: mirrors cudaLBMKernelOutflow (kernels.h) -
// runs on the finalized previous-iteration state before the main launches
template <typename NSE>
__global__ void
cudaOutflowPass(typename NSE::DATA SD, typename NSE::TRAITS::idx3d offset, typename NSE::TRAITS::idx3d end, typename NSE::TRAITS::bool3d distributed)
{
	using idx = typename NSE::TRAITS::idx;
	using dreal = typename NSE::TRAITS::dreal;
	using map_t = typename NSE::TRAITS::map_t;

	idx x = threadIdx.x + blockIdx.x * blockDim.x + offset.x();
	idx y = threadIdx.y + blockIdx.y * blockDim.y + offset.y();
	idx z = threadIdx.z + blockIdx.z * blockDim.z + offset.z();

	if (x >= end.x() || y >= end.y() || z >= end.z())
		return;

	const map_t gi_map = SD.map(x, y, z);
	if (! NSE::BC::isOutflowPassBC(gi_map))
		return;

	idx xp, xm, yp, ym, zp, zm;
	kernelInitIndices<NSE>(SD, distributed, x, y, z, xp, xm, yp, ym, zp, zm);

	typename NSE::template KernelStruct<dreal> KS;
	NSE::MACRO::copyQuantities(SD, KS, x, y, z);
	NSE::MACRO::template computeForcing<typename NSE::BC>(SD, KS, xm, x, xp, ym, y, yp, zm, z, zp);

	NSE::BC::outflowPass(SD, KS, gi_map, xm, x, xp, ym, y, yp, zm, z, zp);

	NSE::MACRO::outputMacro(SD, KS, x, y, z);
}

// one unfused iteration with BC dispatch, replicating the multi-rank branch
// of State::SimUpdate: the outflow pass over each block's outflow_boxes on
// the interior stream (joined right after), then gather -> shared collision
// -> scatter per compute region, boundary streams joined before the
// pipelined DF+macro halo exchange, the interior stream joined afterwards
template <typename NSE, typename COLL_NSE, typename SCRATCH>
static void unfusedStepWithBC(LBM<NSE>& nse, const std::vector<typename COLL_NSE::DATA>& collision_data, std::vector<SCRATCH>& scratch)
{
	using dreal = typename NSE::TRAITS::dreal;
	using idx3d = typename NSE::TRAITS::idx3d;
	using bool3d = typename NSE::TRAITS::bool3d;

	constexpr SDirection boundary_directions[] = {
		SDirection::Bottom, SDirection::Top, SDirection::Back, SDirection::Front, SDirection::Left, SDirection::Right
	};

	// outflow pass on the finalized previous-phase state, before the main launch
	for (std::size_t i = 0; i < nse.blocks.size(); i++) {
		auto& block = nse.blocks[i];
		const auto& cd = block.computeData.at(SDirection::None);
		for (const auto& box : block.outflow_boxes) {
			TNL::Backend::LaunchConfiguration launch_config;
			launch_config.blockSize = cd.blockSize;
			launch_config.gridSize = block.getCudaGridSize(box.end - box.begin, launch_config.blockSize);
			launch_config.stream = cd.stream;
			TNL::Backend::launchKernelAsync(cudaOutflowPass<NSE>, launch_config, block.data, box.begin, box.end, block.is_distributed());
		}
	}
	for (auto& block : nse.blocks)
		TNL::Backend::streamSynchronize(block.computeData.at(SDirection::None).stream);

	auto launch_region = [&](std::size_t i, SDirection direction)
	{
		auto& block = nse.blocks[i];
		const auto& cd = block.computeData.at(direction);
		const bool3d distributed = block.is_distributed();
		dreal* scr = scratch[i].getData();
		TNL::Backend::LaunchConfiguration launch_config;
		launch_config.blockSize = cd.blockSize;
		launch_config.gridSize = cd.gridSize;
		launch_config.stream = cd.stream;
		const idx3d begin = cd.offset;
		const idx3d end = cd.offset + cd.size;
		TNL::Backend::launchKernelAsync(cudaGatherBC<NSE>, launch_config, block.data, begin, end, distributed, scr);
		TNL::Backend::launchKernelAsync(cudaSharedCollisionBC<COLL_NSE>, launch_config, collision_data[i], begin, end, scr);
		TNL::Backend::launchKernelAsync(cudaScatterBC<NSE>, launch_config, block.data, begin, end, distributed, scr);
	};

	for (std::size_t i = 0; i < nse.blocks.size(); i++) {
		// compute on boundaries
		for (const SDirection direction : boundary_directions)
			if (auto search = nse.blocks[i].neighborIDs.find(direction); search != nse.blocks[i].neighborIDs.end() && search->second >= 0)
				launch_region(i, direction);
		// compute on interior lattice sites
		launch_region(i, SDirection::None);
	}

#ifdef HAVE_MPI
	if (nse.nproc > 1) {
		// wait for the computations on boundaries to finish
		for (std::size_t i = 0; i < nse.blocks.size(); i++)
			for (const SDirection direction : boundary_directions)
				if (auto search = nse.blocks[i].neighborIDs.find(direction); search != nse.blocks[i].neighborIDs.end() && search->second >= 0)
					TNL::Backend::streamSynchronize(nse.blocks[i].computeData.at(direction).stream);

		// exchange the latest DFs and dmacro on overlaps between blocks
		nse.synchronizeDFsAndMacroDevice(NSE::STREAMING::output_df, true);
	}
#endif

	// wait for the computation on the interior to finish
	for (auto& block : nse.blocks)
		TNL::Backend::streamSynchronize(block.computeData.at(SDirection::None).stream);
}

struct TGVSetup
{
	int X = 16;
	int Y = 16;
	int Z = 1;
	int iterations = 100;
	double lbm_viscosity = 0.01;
	double phys_height = 1.0;
	double phys_viscosity = 1.5e-5;
	double phys_V_0 = 5e-3;
};

template <typename TRAITS_, typename SETUP>
static Lattice<3, typename TRAITS_::real, typename TRAITS_::idx> makeLat(const SETUP& s)
{
	using real = typename TRAITS_::real;
	using idx = typename TRAITS_::idx;
	using lat_t = Lattice<3, real, idx>;
	const real physDl = s.phys_height / s.Y;
	lat_t lat;
	lat.global = typename lat_t::CoordinatesType(s.X, s.Y, s.Z);
	lat.physOrigin = typename lat_t::PointType(-s.phys_height / 2 + physDl, -s.phys_height / 2 + physDl, s.Z > 1 ? -s.phys_height / 2 + physDl : 0);
	lat.physDl = physDl;
	lat.physDt = s.lbm_viscosity / s.phys_viscosity * physDl * physDl;
	lat.physViscosity = s.phys_viscosity;
	return lat;
}

// when spec is null, the hook is removed so the block gets the
// interface-optimal (1D) split; unset on destruction again
struct ForcedDecomposition
{
	explicit ForcedDecomposition(const char* spec)
	{
		if (spec == nullptr)
			unsetenv("TNL_LBM_FORCE_DECOMPOSITION");
		else
			setenv("TNL_LBM_FORCE_DECOMPOSITION", spec, 1);
	}
	~ForcedDecomposition()
	{
		unsetenv("TNL_LBM_FORCE_DECOMPOSITION");
	}
};

// all macroscopic quantities over the owned sites plus, along distributed
// axes only, the exchanged one-cell ghost planes (non-distributed axes have
// no halo allocated; non-periodic ghost values are never authored)
template <typename NSE>
static std::vector<double> snapshotMacro(const LBM_BLOCK<NSE>& block)
{
	using idx = typename NSE::TRAITS::idx;
	using dreal = typename NSE::TRAITS::dreal;
	static_assert(std::is_same_v<dreal, double>);
	const auto distributed = block.is_distributed();
	const idx x0 = block.offset.x() - (distributed.x() ? 1 : 0);
	const idx x1 = block.offset.x() + block.local.x() - 1 + (distributed.x() ? 1 : 0);
	const idx y0 = block.offset.y() - (distributed.y() ? 1 : 0);
	const idx y1 = block.offset.y() + block.local.y() - 1 + (distributed.y() ? 1 : 0);
	const idx z0 = block.offset.z() - (distributed.z() ? 1 : 0);
	const idx z1 = block.offset.z() + block.local.z() - 1 + (distributed.z() ? 1 : 0);
	std::vector<double> out;
	out.reserve(NSE::MACRO::N * (x1 - x0 + 1) * (y1 - y0 + 1) * (z1 - z0 + 1));
	for (int m = 0; m < NSE::MACRO::N; m++)
		for (idx z = z0; z <= z1; z++)
			for (idx y = y0; y <= y1; y++)
				for (idx x = x0; x <= x1; x++)
					out.push_back(block.hmacro(m, x, y, z));
	return out;
}

static void checkSnapshotBitwiseMatch(const char* tag, const std::vector<double>& candidate, const std::vector<double>& reference)
{
	INFO("pattern: ", tag);
	REQUIRE(candidate.size() == reference.size());
	std::size_t ndiff = 0;
	double maxdiff = 0;
	for (std::size_t i = 0; i < candidate.size(); i++)
		if (candidate[i] != reference[i]) {
			ndiff++;
			maxdiff = std::max(maxdiff, (double) std::abs(candidate[i] - reference[i]));
		}
	INFO("differing cells: ", ndiff, " of ", candidate.size(), " (max|diff| = ", maxdiff, ")");
	CHECK(ndiff == 0);
}

// Taylor-Green initial condition as a named functor: shared by the TGV
// runner and the frame-0 output runner so that both reuse the same
// setInitialCondition kernel instantiations
template <typename NSE>
struct TGVEq
{
	using idx = typename NSE::TRAITS::idx;
	using dreal = typename NSE::TRAITS::dreal;
	using lat_t = Lattice<3, typename NSE::TRAITS::real, idx>;

	lat_t lat;
	dreal V_0;
	dreal rho_0;

	// Taylor-Green vortex with one full spatial period along each model axis
	__cuda_callable__ void operator()(typename NSE::template KernelStruct<dreal>& KS, idx gx, idx gy, idx gz) const
	{
		const dreal L = lat.global.x() * lat.physDl / (2 * TNL::pi);
		const dreal x = lat.lbm2physX(gx);
		const dreal y = lat.lbm2physY(gy);
		if constexpr (NSE::D == 3) {
			const dreal z = lat.lbm2physZ(gz);
			KS.vx = V_0 * TNL::sin(x / L) * TNL::cos(y / L) * TNL::cos(z / L);
			KS.vy = -V_0 * TNL::cos(x / L) * TNL::sin(y / L) * TNL::cos(z / L);
			KS.vz = 0;
			KS.rho = rho_0 + 3 * (V_0 * V_0 / 16) * (TNL::cos(2 * x / L) + TNL::cos(2 * y / L)) * (TNL::cos(2 * z / L) + 2);
		}
		else {
			KS.vx = V_0 * TNL::sin(x / L) * TNL::cos(y / L);
			KS.vy = -V_0 * TNL::cos(x / L) * TNL::sin(y / L);
			KS.rho = rho_0;
		}
		NSE::COLL::setEquilibrium(KS);
	}
};

// run one pattern instance: init snapshot (initial DF+macro exchange effects)
// concatenated with the snapshot after `iterations` unfused steps
template <typename NSE, typename COLL_NSE>
static std::vector<double> runTGV(const std::string& id, const TGVSetup& s)
{
	using TRAITS_ = typename NSE::TRAITS;
	using real = typename TRAITS_::real;
	using idx = typename TRAITS_::idx;
	using dreal = typename TRAITS_::dreal;
	using bool3d = typename TRAITS_::bool3d;
	using lat_t = Lattice<3, real, idx>;

	INFO("pattern instance: ", id);
	lat_t lat = makeLat<TRAITS_>(s);
	const bool3d periodic = (NSE::D == 3) ? bool3d{true, true, true} : bool3d{true, true, false};
	LBM<NSE> nse(MPI_COMM_WORLD, lat, periodic);
	nse.allocateHostData();
	nse.allocateDeviceData();
	for (auto& block : nse.blocks)
		block.data.lbmViscosity = lat.lbmViscosity();

	// initial equilibrium placement: the pattern's own parity-0 layout,
	// authoring also the initial macroscopic quantities on the owned sites
	const dreal V_0 = lat.phys2lbmVelocity(s.phys_V_0);
	const dreal rho_0 = 1;
	nse.setInitialCondition(TGVEq<NSE>{lat, V_0, rho_0});

#ifdef HAVE_MPI
	if (nse.nproc > 1)
		// finalize the initial layout-0 field on the subdomain overlaps
		nse.synchronizeDFsAndMacroDevice(df_cur, true);
#endif

	CHECK(nse.iterations == 0);
	nse.copyMacroToHost();
	std::vector<double> snapshot = snapshotMacro<NSE>(nse.blocks.front());
	const std::size_t init_size = snapshot.size();

	// one natural-layout DF scratch array per block plus the kernel data of
	// the shared collision kernel (only the base members it reads: indexer,
	// XYZ storage size, macro pointer and the physical parameters)
	using scratch_array_t = TNL::Containers::Array<dreal, DeviceType, idx>;
	std::vector<scratch_array_t> scratch(nse.blocks.size());
	std::vector<typename COLL_NSE::DATA> collision_data(nse.blocks.size());
	for (std::size_t i = 0; i < nse.blocks.size(); i++) {
		const auto& block = nse.blocks[i];
		scratch[i].setSize(static_cast<idx>(NSE::Q) * block.data.XYZ);
		collision_data[i].indexer = block.data.indexer;
		collision_data[i].XYZ = block.data.XYZ;
		collision_data[i].dmacro = block.data.dmacro;
		collision_data[i].lbmViscosity = block.data.lbmViscosity;
		collision_data[i].periodic = block.data.periodic;
	}

	for (int it = 0; it < s.iterations; it++) {
		nse.updateKernelData();
		unfusedStep<NSE, COLL_NSE>(nse, collision_data, scratch);
		nse.iterations++;
	}
	CHECK(nse.iterations == s.iterations);
	nse.copyMacroToHost();
	const std::vector<double> snapshot_final = snapshotMacro<NSE>(nse.blocks.front());
	REQUIRE(snapshot_final.size() == init_size);
	snapshot.insert(snapshot.end(), snapshot_final.begin(), snapshot_final.end());
	return snapshot;
}

// every candidate pattern must produce the same initial and final fields as
// the A-B pull reference of the same binary, bitwise
template <typename COLL_NSE, typename REF, typename... CANDS>
static void checkTGV(const char* model_tag, const TGVSetup& setup, std::initializer_list<const char*> pattern_tags)
{
	const std::vector<double> reference = runTGV<REF, COLL_NSE>(fmt::format("{}_ab_pull", model_tag), setup);
	auto tag = pattern_tags.begin();
	using expand = int[];
	(void) expand{
		0, (checkSnapshotBitwiseMatch(*tag, runTGV<CANDS, COLL_NSE>(fmt::format("{}_{}", model_tag, *tag), setup), reference), tag++, 0)...
	};
}

// non-periodic channel exercising the real BC dispatch per pattern: walls on
// the y (2D and 3D) and z (3D) faces, a moment inflow on the left x face and
// a two-pass outflow (plain or interpolated) on the right x face; the map
// follows the ghost-layer idiom required by the single-array patterns
struct ChannelSetup
{
	int X = 24;
	int Y = 12;
	int Z = 1;
	int iterations = 60;
	double lbm_viscosity = 0.01;
	double phys_height = 1.0;
	double phys_viscosity = 1.5e-5;
	double phys_V_0 = 5e-3;
	bool interp_outflow = false;
};

// run one pattern instance through the channel: init snapshot (the map, the
// initial DF+macro exchange effects) concatenated with the snapshot after
// `iterations` unfused BC-dispatch steps
template <typename NSE, typename COLL_NSE>
static std::vector<double> runChannel(const std::string& id, const ChannelSetup& s)
{
	using TRAITS_ = typename NSE::TRAITS;
	using real = typename TRAITS_::real;
	using idx = typename TRAITS_::idx;
	using dreal = typename TRAITS_::dreal;
	using lat_t = Lattice<3, real, idx>;
	using BC = typename NSE::BC;

	INFO("pattern instance: ", id);
	lat_t lat = makeLat<TRAITS_>(s);
	// non-periodic on every axis: all domain boundaries come from the map
	LBM<NSE> nse(MPI_COMM_WORLD, lat);
	nse.allocateHostData();
	nse.allocateDeviceData();
	for (auto& block : nse.blocks)
		block.data.lbmViscosity = lat.lbmViscosity();

	// stamp order: inflow/outflow first, walls next (they win the shared
	// edges/corners), the GEO_NOTHING ghost frame always last (see AGENTS.md,
	// the setBoundary* call-order anti-pattern)
	nse.resetMap(BC::GEO_FLUID);
	nse.setBoundaryX(1, BC::GEO_INFLOW_MOMENT);
	nse.setBoundaryX(s.X - 2, s.interp_outflow ? BC::GEO_OUTFLOW_RIGHT_INTERP : BC::GEO_OUTFLOW_RIGHT);
	nse.setBoundaryY(1, BC::GEO_WALL);
	nse.setBoundaryY(s.Y - 2, BC::GEO_WALL);
	if constexpr (NSE::D == 3) {
		nse.setBoundaryZ(1, BC::GEO_WALL);
		nse.setBoundaryZ(s.Z - 2, BC::GEO_WALL);
	}
	nse.setBoundaryX(0, BC::GEO_NOTHING);
	nse.setBoundaryX(s.X - 1, BC::GEO_NOTHING);
	nse.setBoundaryY(0, BC::GEO_NOTHING);
	nse.setBoundaryY(s.Y - 1, BC::GEO_NOTHING);
	if constexpr (NSE::D == 3) {
		nse.setBoundaryZ(0, BC::GEO_NOTHING);
		nse.setBoundaryZ(s.Z - 1, BC::GEO_NOTHING);
	}
	// builds the outflow-pass rectangle cover (updateOutflowPassRegion)
	nse.copyMapToDevice();

#ifdef HAVE_MPI
	if (nse.nproc > 1)
		// finalize the map on the subdomain overlaps (State::SimInit order)
		nse.synchronizeMapDevice();
#endif

	// the runtime face detection must find exactly one interior-side
	// axis-neighbor per inflow/outflow site in the synchronized map
	nse.validateFaceDetectedBC();

	// constant initial equilibrium matching the imposed inflow profile; the
	// tangential components engage the tangential terms of the moment BC
	const dreal V_0 = lat.phys2lbmVelocity(s.phys_V_0);
	const dreal inflow_vy = V_0 / 4;
	const dreal inflow_vz = V_0 / 8;
	for (auto& block : nse.blocks) {
		block.data.inflow_vx = V_0;
		block.data.inflow_vy = inflow_vy;
		if constexpr (NSE::D == 3)
			block.data.inflow_vz = inflow_vz;
	}
	nse.setInitialCondition(
		[V_0, inflow_vy, inflow_vz] __cuda_callable__(typename NSE::template KernelStruct<dreal> & KS, idx, idx, idx) mutable
		{
			KS.rho = 1;
			KS.vx = V_0;
			KS.vy = inflow_vy;
			if constexpr (NSE::D == 3)
				KS.vz = inflow_vz;
			NSE::COLL::setEquilibrium(KS);
		}
	);

#ifdef HAVE_MPI
	if (nse.nproc > 1)
		// finalize the initial layout-0 field on the subdomain overlaps
		nse.synchronizeDFsAndMacroDevice(df_cur, true);
#endif

	CHECK(nse.iterations == 0);
	nse.copyMacroToHost();
	std::vector<double> snapshot = snapshotMacro<NSE>(nse.blocks.front());
	const std::size_t init_size = snapshot.size();

	// widened natural-layout scratch (population slots plus the KernelStruct
	// macro state) and the kernel data of the shared collision kernel (it
	// reads only indexer, XYZ storage size, the macro and map arrays and the
	// physical parameters)
	using scratch_array_t = TNL::Containers::Array<dreal, DeviceType, idx>;
	std::vector<scratch_array_t> scratch(nse.blocks.size());
	std::vector<typename COLL_NSE::DATA> collision_data(nse.blocks.size());
	for (std::size_t i = 0; i < nse.blocks.size(); i++) {
		const auto& block = nse.blocks[i];
		scratch[i].setSize(static_cast<idx>(scratchSlotCount<NSE>()) * block.data.XYZ);
		collision_data[i].indexer = block.data.indexer;
		collision_data[i].XYZ = block.data.XYZ;
		collision_data[i].dmacro = block.data.dmacro;
		collision_data[i].dmap = block.data.dmap;
		collision_data[i].lbmViscosity = block.data.lbmViscosity;
		collision_data[i].periodic = block.data.periodic;
	}

	for (int it = 0; it < s.iterations; it++) {
		nse.updateKernelData();
		unfusedStepWithBC<NSE, COLL_NSE>(nse, collision_data, scratch);
		nse.iterations++;
	}
	CHECK(nse.iterations == s.iterations);
	nse.copyMacroToHost();
	const std::vector<double> snapshot_final = snapshotMacro<NSE>(nse.blocks.front());
	REQUIRE(snapshot_final.size() == init_size);
	snapshot.insert(snapshot.end(), snapshot_final.begin(), snapshot_final.end());
	return snapshot;
}

// every candidate pattern must produce the same initial and final channel
// fields as the A-B pull reference of the same binary, bitwise; this asserts
// that the BC handling is pattern-equivalent (the BC bodies are shared code
// across the patterns), not that the unfused dispatch reproduces the
// production fused kernels - the production FP contract stays with the
// test_nse/test_d2q9 regression suites
template <typename COLL_NSE, typename REF, typename... CANDS>
static void checkChannel(const char* model_tag, const ChannelSetup& setup, std::initializer_list<const char*> pattern_tags)
{
	const std::vector<double> reference = runChannel<REF, COLL_NSE>(fmt::format("{}_ab_pull", model_tag), setup);
	auto tag = pattern_tags.begin();
	using expand = int[];
	(void) expand{
		0, (checkSnapshotBitwiseMatch(*tag, runChannel<CANDS, COLL_NSE>(fmt::format("{}_{}", model_tag, *tag), setup), reference), tag++, 0)...
	};
}

// frame-0 output read-back.
// The historical frame-0 bug class: the t=0 macroscopic quantities were
// written to the output before the streaming pattern's init permutation
// authored them, so the output carried stale values only at the (pattern
// init layout) x (output serialization) intersection. The runner below
// initializes the TGV field through the production LBM_BLOCK::setInitialCondition
// path (per-pattern parity-0 layout), finalizes the layout on the subdomain
// overlaps, copies the macro to the host, and then passes the block's host
// macro arrays through the real serialization machinery: one BP5 step via
// DataManager (variables defined with the production global-shape /
// block-offset / block-extent selections met in State::write3D), followed by
// an adios2 read-back of the same selection on this rank. The round-trip is
// asserted bitwise against the in-memory snapshot, and every pattern's
// round-trip is asserted bitwise against the A-B pull instance.

// name per macro component, mirroring the production getOutputDataNames
inline const char* frame0VariableName(int m)
{
	static const char* names[] = {"lbm_density", "velocity_x", "velocity_y", "velocity_z"};
	return names[m];
}

// the DataManager picks up the engine from the "Output" io of the ADIOS
// config, like State's adios does; write the minimal BP5 config in the
// current working directory on rank 0 (all ranks must see it before the
// ADIOS object is constructed)
static void writeFrame0AdiosConfig(const std::string& config_path = "adios2-unitmpi.xml")
{
	bool is_rank0 = true;
#ifdef HAVE_MPI
	is_rank0 = TNL::MPI::GetRank(MPI_COMM_WORLD) == 0;
#endif
	if (is_rank0) {
		std::ofstream config(config_path);
		config << "<?xml version=\"1.0\"?>\n"
				  "<adios-config>\n"
				  "    <io name=\"Output\">\n"
				  "        <engine type=\"BP5\"/>\n"
				  "    </io>\n"
				  "</adios-config>\n";
	}
#ifdef HAVE_MPI
	TNL::MPI::Barrier(MPI_COMM_WORLD);
#endif
}

// write one frame with per-macro-component variables through DataManager and
// read the same selection back with adios2; `memory` holds the block's host
// macro in the serialization layout (component-major, x fastest) both for
// the write and for the caller's bitwise comparison
template <typename NSE>
static std::vector<double> frame0RoundTrip(LBM<NSE>& nse, const std::string& ioName, const std::vector<double>& memory)
{
	using idx = typename NSE::TRAITS::idx;
	using dreal = typename NSE::TRAITS::dreal;
	static_assert(std::is_same_v<dreal, double>);
	const auto& block = nse.blocks.front();

	writeFrame0AdiosConfig();

#ifdef HAVE_MPI
	adios2::ADIOS adios("adios2-unitmpi.xml", nse.communicator);
#else
	adios2::ADIOS adios("adios2-unitmpi.xml");
#endif
	DataManager dataManager(&adios);
	dataManager.prepareIO(ioName);

	// the production block-selection layout (State::predefine3D): global
	// shape, this block's offset and extent, (z, y, x) ordering
	const adios2::Dims shape{
		static_cast<std::size_t>(block.global.z()), static_cast<std::size_t>(block.global.y()), static_cast<std::size_t>(block.global.x())
	};
	const adios2::Dims start{
		static_cast<std::size_t>(block.offset.z()), static_cast<std::size_t>(block.offset.y()), static_cast<std::size_t>(block.offset.x())
	};
	const adios2::Dims count{
		static_cast<std::size_t>(block.local.z()), static_cast<std::size_t>(block.local.y()), static_cast<std::size_t>(block.local.x())
	};
	const std::size_t cell_count = count[0] * count[1] * count[2];
	REQUIRE(memory.size() == NSE::MACRO::N * cell_count);
	for (int m = 0; m < NSE::MACRO::N; m++)
		dataManager.template defineData<dreal>(frame0VariableName(m), shape, start, count, ioName);

	dataManager.openEngine(ioName);
	dataManager.beginStep(ioName);
	for (int m = 0; m < NSE::MACRO::N; m++) {
		// the buffer must outlive the step (Puts are deferred), so it lives
		// in the DataManager's step buffer like the production writers' do
		auto& buffer = dataManager.template newStepBuffer<dreal>(cell_count);
		buffer.assign(memory.begin() + m * cell_count, memory.begin() + (m + 1) * cell_count);
		dataManager.template outputData<dreal>(frame0VariableName(m), buffer.data(), ioName);
	}
	dataManager.endStep(ioName);
	dataManager.closeEngine(ioName);

	// read the same selection back with adios2 (random-access file mode: no
	// BeginStep needed and InquireVariable is legal outside a step)
	adios2::IO read_io = adios.DeclareIO(fmt::format("read_{}", ioName));
	adios2::Engine read_engine = read_io.Open(ioName + ".bp", adios2::Mode::ReadRandomAccess);
	std::vector<double> readback;
	readback.reserve(memory.size());
	for (int m = 0; m < NSE::MACRO::N; m++) {
		adios2::Variable<dreal> var = read_io.InquireVariable<dreal>(frame0VariableName(m));
		REQUIRE(var);
		var.SetSelection({start, count});
		std::vector<dreal> buffer(cell_count);
		read_engine.Get(var, buffer.data());
		read_engine.PerformGets();
		readback.insert(readback.end(), buffer.begin(), buffer.end());
	}
	read_engine.Close();
	return readback;
}

// initialize the TGV field, write frame 0 through the real output machinery
// and return the serialization round-trip; asserts the round-trip reproduces
// the in-memory host macro bitwise
template <typename NSE>
static std::vector<double> runFrame0(const std::string& id, const TGVSetup& s)
{
	using TRAITS_ = typename NSE::TRAITS;
	using real = typename TRAITS_::real;
	using idx = typename TRAITS_::idx;
	using dreal = typename TRAITS_::dreal;
	using bool3d = typename TRAITS_::bool3d;
	using idx3d = typename TRAITS_::idx3d;
	using lat_t = Lattice<3, real, idx>;

	INFO("pattern instance: ", id);
	lat_t lat = makeLat<TRAITS_>(s);
	const bool3d periodic = (NSE::D == 3) ? bool3d{true, true, true} : bool3d{true, true, false};
	LBM<NSE> nse(MPI_COMM_WORLD, lat, periodic);
	nse.allocateHostData();
	nse.allocateDeviceData();
	for (auto& block : nse.blocks)
		block.data.lbmViscosity = lat.lbmViscosity();

	// the same production init path as the TGV rows: pattern-native parity-0
	// layout authoring the initial macroscopic quantities on the owned sites
	const dreal V_0 = lat.phys2lbmVelocity(s.phys_V_0);
	const dreal rho_0 = 1;
	nse.setInitialCondition(TGVEq<NSE>{lat, V_0, rho_0});

#ifdef HAVE_MPI
	if (nse.nproc > 1)
		// finalize the initial layout-0 field on the subdomain overlaps
		nse.synchronizeDFsAndMacroDevice(df_cur, true);
#endif

	CHECK(nse.iterations == 0);
	nse.copyMacroToHost();

	// the in-memory snapshot of the owned sites in the serialization layout
	// (component-major, x fastest within the block extent)
	const auto& block = nse.blocks.front();
	const idx3d begin = block.offset;
	const idx3d end = block.offset + block.local;
	std::vector<double> memory;
	memory.reserve(NSE::MACRO::N * block.local.x() * block.local.y() * block.local.z());
	for (int m = 0; m < NSE::MACRO::N; m++)
		for (idx z = begin.z(); z < end.z(); z++)
			for (idx y = begin.y(); y < end.y(); y++)
				for (idx x = begin.x(); x < end.x(); x++)
					memory.push_back(block.hmacro(m, x, y, z));

	const std::vector<double> readback = frame0RoundTrip<NSE>(nse, fmt::format("frame0_{}", id), memory);
	checkSnapshotBitwiseMatch("round-trip vs in-memory", readback, memory);
	return readback;
}

// every candidate pattern's frame-0 round-trip must match the A-B pull
// reference's round-trip of the same binary, bitwise
template <typename REF, typename... CANDS>
static void checkFrame0(const char* model_tag, const TGVSetup& setup, std::initializer_list<const char*> pattern_tags)
{
	const std::vector<double> reference = runFrame0<REF>(fmt::format("{}_ab_pull", model_tag), setup);
	auto tag = pattern_tags.begin();
	using expand = int[];
	(void) expand{0, (checkSnapshotBitwiseMatch(*tag, runFrame0<CANDS>(fmt::format("{}_{}", model_tag, *tag), setup), reference), tag++, 0)...};
}
