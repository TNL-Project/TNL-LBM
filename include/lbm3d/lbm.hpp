#pragma once

#include "lbm.h"
#include "lattice_decomposition.h"

template <typename CONFIG>
LBM<CONFIG>::LBM(const TNL::MPI::Comm& communicator, lat_t lat, const bool3d& periodic)
: communicator(communicator),
  lat(lat)
{
	// initialize MPI info
	rank = communicator.rank();
	nproc = communicator.size();

	// default lattice decomposition
	//BLOCK block = decomposeLattice_D1Q3<CONFIG>(communicator, lat.global, periodic);
	BLOCK block = decomposeLattice_D3Q27<CONFIG>(communicator, lat.global, periodic);
	blocks.push_back(std::move(block));
	total_blocks = nproc;

	physCharLength = lat.physDl * lat.global.y();
}

template <typename CONFIG>
LBM<CONFIG>::LBM(const TNL::MPI::Comm& communicator, lat_t lat, std::vector<BLOCK>&& blocks)
: communicator(communicator),
  lat(lat),
  blocks(std::forward<std::vector<BLOCK>>(blocks))
{
	// initialize MPI info
	rank = communicator.rank();
	nproc = communicator.size();

	total_blocks = TNL::MPI::reduce(blocks.size(), MPI_SUM, communicator);

	physCharLength = lat.physDl * (real) lat.global.y();
}

template <typename CONFIG>
LBM<CONFIG>::LBM(const TNL::MPI::Comm& communicator, lat_t lat, const bool3d& periodic, int max_level)
: LBM(communicator, lat, periodic)
{
	// fine-level blocks are created later by the caller (AMR setup code), which
	// is also responsible for maintaining level_block_counts
	this->max_level = max_level;
	level_block_counts = std::vector<int>(max_level + 1, 0);
}

template <typename CONFIG>
bool LBM<CONFIG>::isAnyLocalIndex(idx x, idx y, idx z)
{
	for (auto& block : blocks)
		if (block.isLocalIndex(x, y, z))
			return true;
	return false;
}

template <typename CONFIG>
bool LBM<CONFIG>::isAnyLocalX(idx x)
{
	for (auto& block : blocks)
		if (block.isLocalX(x))
			return true;
	return false;
}

template <typename CONFIG>
bool LBM<CONFIG>::isAnyLocalY(idx y)
{
	for (auto& block : blocks)
		if (block.isLocalY(y))
			return true;
	return false;
}

template <typename CONFIG>
bool LBM<CONFIG>::isAnyLocalZ(idx z)
{
	for (auto& block : blocks)
		if (block.isLocalZ(z))
			return true;
	return false;
}

template <typename CONFIG>
std::vector<typename LBM<CONFIG>::BLOCK*> LBM<CONFIG>::getBlocksAtLevel(int level)
{
	std::vector<BLOCK*> blocks_at_level;
	for (auto& block : blocks)
		if (block.level == level)
			blocks_at_level.push_back(&block);
	return blocks_at_level;
}

template <typename CONFIG>
typename LBM<CONFIG>::BLOCK* LBM<CONFIG>::findBlockContaining(idx x, idx y, idx z, int level)
{
	for (auto& block : blocks)
		if (block.level == level && block.isLocalIndex(x, y, z))
			return &block;
	return nullptr;
}

template <typename CONFIG>
void LBM<CONFIG>::setMap(idx x, idx y, idx z, map_t value)
{
	for (auto& block : blocks)
		block.setMap(x, y, z, value);
}

template <typename CONFIG>
void LBM<CONFIG>::allocatePhiTransferDirectionArrays()
{
	for (auto& block : blocks)
		block.allocatePhiTransferDirectionArrays();
}

template <typename CONFIG>
void LBM<CONFIG>::setBoundaryX(idx x, map_t value)
{
	// boundary planes are expressed in level-0 global indices and apply only
	// to level-0 blocks: fine blocks must keep the all-FLUID interior map set
	// by createAMRBlocks (their ghost rows are owned by the coupling). Without
	// the guard, a plane index that numerically falls inside a fine block's
	// local range would tag a spurious BC slab through the patch interior.
	for (auto& block : blocks)
		if (block.level == 0)
			block.setBoundaryX(x, value);
}

template <typename CONFIG>
void LBM<CONFIG>::setBoundaryY(idx y, map_t value)
{
	for (auto& block : blocks)
		if (block.level == 0)
			block.setBoundaryY(y, value);
}

template <typename CONFIG>
void LBM<CONFIG>::setBoundaryZ(idx z, map_t value)
{
	for (auto& block : blocks)
		if (block.level == 0)
			block.setBoundaryZ(z, value);
}

template <typename CONFIG>
void LBM<CONFIG>::resetMap(map_t geo_type)
{
	for (auto& block : blocks)
		block.resetMap(geo_type);
}

template <typename CONFIG>
void LBM<CONFIG>::setEquilibrium(real rho, real vx, real vy, real vz)
{
	for (auto& block : blocks)
		block.setEquilibrium(rho, vx, vy, vz);
}

template <typename CONFIG>
void LBM<CONFIG>::computeInitialMacro()
{
	for (auto& block : blocks)
		block.computeInitialMacro();
}

template <typename CONFIG>
void LBM<CONFIG>::copyMapToHost()
{
	for (auto& block : blocks)
		block.copyMapToHost();
}

template <typename CONFIG>
void LBM<CONFIG>::copyMapToDevice()
{
	for (auto& block : blocks)
		block.copyMapToDevice();
}

template <typename CONFIG>
void LBM<CONFIG>::copyMacroToHost()
{
	for (auto& block : blocks)
		block.copyMacroToHost();
}

template <typename CONFIG>
void LBM<CONFIG>::copyMacroToDevice()
{
	for (auto& block : blocks)
		block.copyMacroToDevice();
}

template <typename CONFIG>
void LBM<CONFIG>::copyDFsToHost(uint8_t dfty)
{
	for (auto& block : blocks)
		block.copyDFsToHost(dfty);
}

template <typename CONFIG>
void LBM<CONFIG>::copyDFsToDevice(uint8_t dfty)
{
	for (auto& block : blocks)
		block.copyDFsToDevice(dfty);
}

template <typename CONFIG>
void LBM<CONFIG>::copyDFsToHost()
{
	for (auto& block : blocks)
		block.copyDFsToHost();
}

template <typename CONFIG>
void LBM<CONFIG>::copyDFsToDevice()
{
	for (auto& block : blocks)
		block.copyDFsToDevice();
}

#ifdef HAVE_MPI
template <typename CONFIG>
void LBM<CONFIG>::synchronizeDFsAndMacroDevice(uint8_t dftype, bool sync_macro)
{
	TNL::Timer t;
	t.start();

	// stage 0: set inputs, allocate buffers
	// stage 1: fill send buffers
	for (auto& block : blocks) {
		block.synchronizeDFsDevice_start(dftype);
		if (sync_macro)
			block.synchronizeMacroDevice_start();
	}

	// stage 2: issue all send and receive async operations
	for (auto& block : blocks) {
		for (int i = 0; i < CONFIG::Q; i++)
			block.df_sync[i].stage_2();
		if (sync_macro)
			for (int i = 0; i < MACRO::N; i++)
				block.macro_sync[i].stage_2();
	}

	// stage 3: copy data from receive buffers
	for (auto& block : blocks) {
		for (int i = 0; i < CONFIG::Q; i++)
			block.df_sync[i].stage_3();
		if (sync_macro)
			for (int i = 0; i < MACRO::N; i++)
				block.macro_sync[i].stage_3();
	}

	// stage 4: ensure everything has finished
	for (auto& block : blocks) {
		for (int i = 0; i < CONFIG::Q; i++)
			block.df_sync[i].stage_4();
		if (sync_macro)
			for (int i = 0; i < MACRO::N; i++)
				block.macro_sync[i].stage_4();
	}

	t.stop();

	auto profile_logger = spdlog::get("profile");
	if (profile_logger && nproc > 1 && iterations % 100 == 0) {
		// count the data volume
		std::size_t total_sent_bytes = 0;
		std::size_t total_recv_bytes = 0;
		std::size_t total_sent_messages = 0;
		std::size_t total_recv_messages = 0;
		for (auto& block : blocks) {
			for (int i = 0; i < CONFIG::Q; i++) {
				total_sent_bytes += block.df_sync[i].sent_bytes;
				total_recv_bytes += block.df_sync[i].recv_bytes;
				total_sent_messages += block.df_sync[i].sent_messages;
				total_recv_messages += block.df_sync[i].recv_messages;
			}
			if (sync_macro)
				for (int i = 0; i < MACRO::N; i++) {
					total_sent_bytes += block.macro_sync[i].sent_bytes;
					total_recv_bytes += block.macro_sync[i].recv_bytes;
					total_sent_messages += block.macro_sync[i].sent_messages;
					total_recv_messages += block.macro_sync[i].recv_messages;
				}
		}

		// print stats
		const double sent_GB = total_sent_bytes * 1e-9;
		const double recv_GB = total_recv_bytes * 1e-9;
		const double sent_GBps = sent_GB / t.getRealTime();
		const double recv_GBps = recv_GB / t.getRealTime();
		const double total_GBps = sent_GBps + recv_GBps;
		profile_logger->info(
			"MPI synchronization stats (last iteration):\n"
			"sent {} GB in {} messages, received {} GB in {} messages, in {} seconds\n"
			"bandwidth: unidirectional {} GB/s, bidirectional {} GB/s",
			sent_GB,
			total_sent_messages,
			recv_GB,
			total_recv_messages,
			t.getRealTime(),
			recv_GBps,
			total_GBps
		);
	}
}

template <typename CONFIG>
void LBM<CONFIG>::synchronizeDFsAndMacroDeviceForLevel(int level, uint8_t dftype, bool sync_macro)
{
	const std::vector<BLOCK*> level_blocks = getBlocksAtLevel(level);
	if (level_blocks.empty())
		return;

	TNL::Timer t;
	t.start();

	// stage 0: set inputs, allocate buffers
	// stage 1: fill send buffers
	for (auto* block : level_blocks) {
		block->synchronizeDFsDevice_start(dftype);
		if (sync_macro)
			block->synchronizeMacroDevice_start();
	}

	// stage 2: issue all send and receive async operations
	for (auto* block : level_blocks) {
		for (int i = 0; i < CONFIG::Q; i++)
			block->df_sync[i].stage_2();
		if (sync_macro)
			for (int i = 0; i < MACRO::N; i++)
				block->macro_sync[i].stage_2();
	}

	// stage 3: copy data from receive buffers
	for (auto* block : level_blocks) {
		for (int i = 0; i < CONFIG::Q; i++)
			block->df_sync[i].stage_3();
		if (sync_macro)
			for (int i = 0; i < MACRO::N; i++)
				block->macro_sync[i].stage_3();
	}

	// stage 4: ensure everything has finished
	for (auto* block : level_blocks) {
		for (int i = 0; i < CONFIG::Q; i++)
			block->df_sync[i].stage_4();
		if (sync_macro)
			for (int i = 0; i < MACRO::N; i++)
				block->macro_sync[i].stage_4();
	}

	t.stop();

	auto profile_logger = spdlog::get("profile");
	if (profile_logger && nproc > 1 && iterations % 100 == 0) {
		// count the data volume
		std::size_t total_sent_bytes = 0;
		std::size_t total_recv_bytes = 0;
		std::size_t total_sent_messages = 0;
		std::size_t total_recv_messages = 0;
		for (auto* block : level_blocks) {
			for (int i = 0; i < CONFIG::Q; i++) {
				total_sent_bytes += block->df_sync[i].sent_bytes;
				total_recv_bytes += block->df_sync[i].recv_bytes;
				total_sent_messages += block->df_sync[i].sent_messages;
				total_recv_messages += block->df_sync[i].recv_messages;
			}
			if (sync_macro)
				for (int i = 0; i < MACRO::N; i++) {
					total_sent_bytes += block->macro_sync[i].sent_bytes;
					total_recv_bytes += block->macro_sync[i].recv_bytes;
					total_sent_messages += block->macro_sync[i].sent_messages;
					total_recv_messages += block->macro_sync[i].recv_messages;
				}
		}

		// print stats
		const double sent_GB = total_sent_bytes * 1e-9;
		const double recv_GB = total_recv_bytes * 1e-9;
		const double sent_GBps = sent_GB / t.getRealTime();
		const double recv_GBps = recv_GB / t.getRealTime();
		const double total_GBps = sent_GBps + recv_GBps;
		profile_logger->info(
			"MPI synchronization stats for level {} (last iteration):\n"
			"sent {} GB in {} messages, received {} GB in {} messages, in {} seconds\n"
			"bandwidth: unidirectional {} GB/s, bidirectional {} GB/s",
			level,
			sent_GB,
			total_sent_messages,
			recv_GB,
			total_recv_messages,
			t.getRealTime(),
			recv_GBps,
			total_GBps
		);
	}
}

template <typename CONFIG>
void LBM<CONFIG>::synchronizeMapDevice()
{
	for (auto& block : blocks)
		block.synchronizeMapDevice_start();
	for (auto& block : blocks)
		block.map_sync.wait();
}
#endif	// HAVE_MPI

template <typename CONFIG>
void LBM<CONFIG>::allocateHostData()
{
	for (auto& block : blocks)
		block.allocateHostData();
}

template <typename CONFIG>
void LBM<CONFIG>::allocateDeviceData()
{
	for (auto& block : blocks)
		block.allocateDeviceData();
}

template <typename CONFIG>
void LBM<CONFIG>::allocateDiffusionCoefficientArrays()
{
	for (auto& block : blocks)
		block.allocateDiffusionCoefficientArrays();
}

template <typename CONFIG>
void LBM<CONFIG>::updateKernelData()
{
	for (auto& block : blocks) {
		// needed for A-A pattern
		// The A-A cycle must start with the spatial sub-step (even_iter == false), not the reflect sub-step.
		// The spatial sub-step reads A[opposite(i)](x - c_i), which performs streaming from the twisted array;
		// the reflect sub-step reads A[i](x), an identity read with no streaming.
		// If reflect runs first, it collides the initial state without streaming, whereas the A-B pattern streams before its first collision.
		// This would produce a systematic error that propagates through the entire simulation.
		// Starting with the spatial sub-step, with DFs initialized in twisted orientation (A[opposite(i)] = eq_i via setEquilibriumLat),
		// makes the first read A[opposite(i)](x - c_i) = eq_i(x - c_i), matching A-B's streamed pull exactly.
		// updateKernelData is called before SimUpdate increments iterations,
		// so even_iter is based on the pre-increment counter.
		// iterations == 0 → even_iter = false → spatial sub-step first.
		block.data.even_iter = (iterations % 2) == 1;

		// rotation (no-op for A-A pattern ... DFMAX=1)
		int i = iterations % DFMAX;	 // i = 0, 1, 2, ... DMAX-1

		for (int k = 0; k < DFMAX; k++) {
			int knew = (k - i) <= 0 ? (k - i + DFMAX) % DFMAX : k - i;
			//block.data.dfs[k] = block.dfs[knew];
			block.data.dfs[k] = block.dfs[knew].getData();
			//printf("updateKernelData:: assigning data.dfs[%d] = dfs[%d]\n",k, knew);
		}
	}
}

template <typename CONFIG>
void LBM<CONFIG>::updateKernelDataForLevel(int level, int substep)
{
	// Per-level variant of updateKernelData() for Berger-Colella subcycling:
	// the global `iterations` counter increments only once per coarse step, but
	// fine levels perform multiple substeps per coarse step, so the parity
	// driving even_iter (A-A pattern) and the DF pointer rotation (A-B pattern)
	// must be based on the level-local `substep` counter instead.
	// Only blocks at the given level are updated - the coarse level keeps using
	// the global updateKernelData() driven by `iterations`.
	for (auto& block : blocks) {
		if (block.level != level)
			continue;

		// restore the per-level lattice viscosity, which State::updateKernelData()
		// overwrites from the level-0 lattice every iteration
		block.data.lbmViscosity = block.lat_local.lbmViscosity();

#ifdef AA_PATTERN
		// A-A pattern: DF rotation is a no-op (DFMAX=1), only even_iter toggles;
		// see updateKernelData() for the sub-step ordering requirements
		block.data.even_iter = (substep % 2) == 1;
#endif

#ifdef AB_PATTERN
		// A-B pattern: absolute DF pointer rotation, mirroring updateKernelData();
		// the source must be the stored dfs arrays, because data.dfs are
		// already-rotated raw pointers
		int i = substep % DFMAX;  // i = 0, 1, 2, ... DFMAX-1

		for (int k = 0; k < DFMAX; k++) {
			int knew = (k - i) <= 0 ? (k - i + DFMAX) % DFMAX : k - i;
			block.data.dfs[k] = block.dfs[knew].getData();
		}
#endif
	}
}

template <typename CONFIG>
template <typename F>
void LBM<CONFIG>::forLocalLatticeSites(F f)
{
	for (auto& block : blocks)
		block.forLocalLatticeSites(f);
}

template <typename CONFIG>
template <typename F>
void LBM<CONFIG>::forAllLatticeSites(F f)
{
	for (auto& block : blocks)
		block.forAllLatticeSites(f);
}
