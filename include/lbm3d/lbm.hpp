#pragma once

#include "lbm.h"
#include "lattice_decomposition.h"

template <typename CONFIG>
LBM<CONFIG>::LBM(const TNL::MPI::Comm& communicator, lat_t lat, bool periodic_lattice)
: communicator(communicator),
  lat(lat)
{
	// initialize MPI info
	rank = communicator.rank();
	nproc = communicator.size();

	// default lattice decomposition
	//BLOCK block = decomposeLattice_D1Q3<CONFIG>(communicator, lat.global, periodic_lattice);
	BLOCK block = decomposeLattice_D3Q27<CONFIG>(communicator, lat.global, periodic_lattice);
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
	for (auto& block : blocks)
		block.setBoundaryX(x, value);
}

template <typename CONFIG>
void LBM<CONFIG>::setBoundaryY(idx y, map_t value)
{
	for (auto& block : blocks)
		block.setBoundaryY(y, value);
}

template <typename CONFIG>
void LBM<CONFIG>::setBoundaryZ(idx z, map_t value)
{
	for (auto& block : blocks)
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
void LBM<CONFIG>::synchronizeDFsAndMacroDevice(uint8_t dftype)
{
	TNL::Timer t;
	t.start();

	// stage 0: set inputs, allocate buffers
	// stage 1: fill send buffers
	for (auto& block : blocks) {
		block.synchronizeDFsDevice_start(dftype);
		if (MACRO::use_syncMacro)
			block.synchronizeMacroDevice_start();
	}

	// stage 2: issue all send and receive async operations
	for (auto& block : blocks) {
		for (int i = 0; i < CONFIG::Q; i++)
			block.df_sync[i].stage_2();
		if (MACRO::use_syncMacro)
			for (int i = 0; i < MACRO::N; i++)
				block.macro_sync[i].stage_2();
	}

	// stage 3: copy data from receive buffers
	for (auto& block : blocks) {
		for (int i = 0; i < CONFIG::Q; i++)
			block.df_sync[i].stage_3();
		if (MACRO::use_syncMacro)
			for (int i = 0; i < MACRO::N; i++)
				block.macro_sync[i].stage_3();
	}

	// stage 4: ensure everything has finished
	for (auto& block : blocks) {
		for (int i = 0; i < CONFIG::Q; i++)
			block.df_sync[i].stage_4();
		if (MACRO::use_syncMacro)
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
			if (MACRO::use_syncMacro)
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
		block.data.even_iter = (iterations % 2) == 0;

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
