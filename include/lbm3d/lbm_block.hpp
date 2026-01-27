#pragma once

#include "lbm_block.h"
#include "UniformDataWriter.h"
#include "block_size_optimizer.h"

template <typename CONFIG>
LBM_BLOCK<CONFIG>::LBM_BLOCK(const TNL::MPI::Comm& communicator, idx3d global, idx3d local, idx3d offset, int this_id)
: communicator(communicator),
  global(global),
  local(local),
  offset(offset),
  id(this_id)
{
	// initialize MPI info
	rank = communicator.rank();
	nproc = communicator.size();
}

template <typename CONFIG>
template <typename Pattern>
void LBM_BLOCK<CONFIG>::setLatticeDecomposition(
	const Pattern& pattern,
	const std::map<TNL::Containers::SyncDirection, int>& neighborIDs,
	const std::map<TNL::Containers::SyncDirection, int>& neighborRanks
)
{
	this->neighborIDs = neighborIDs;
	this->neighborRanks = neighborRanks;

#ifdef HAVE_MPI
	// set communication pattern for all synchronizers
	map_sync.setSynchronizationPattern(pattern);
	for (int i = 0; i < CONFIG::Q + CONFIG::MACRO::N; i++)
		dreal_sync[i].setSynchronizationPattern(pattern);

	// set neighbors for all synchronizers
	for (auto [direction, rank] : neighborRanks) {
		map_sync.setNeighbor(direction, rank);
		for (int i = 0; i < CONFIG::Q + CONFIG::MACRO::N; i++)
			dreal_sync[i].setNeighbor(direction, rank);
	}

	auto isPrimaryDirection = [](TNL::Containers::SyncDirection direction) -> bool
	{
		if ((direction & TNL::Containers::SyncDirection::Left) != TNL::Containers::SyncDirection::None)
			return true;
		else if ((direction & TNL::Containers::SyncDirection::Right) != TNL::Containers::SyncDirection::None)
			return false;
		else if ((direction & TNL::Containers::SyncDirection::Bottom) != TNL::Containers::SyncDirection::None)
			return true;
		else if ((direction & TNL::Containers::SyncDirection::Top) != TNL::Containers::SyncDirection::None)
			return false;
		else if ((direction & TNL::Containers::SyncDirection::Back) != TNL::Containers::SyncDirection::None)
			return true;
		else  // direction & TNL::Containers::SyncDirection::Front
			return false;
	};

	// TODO: make this a general parameter (for now we set an upper bound)
	constexpr int blocks_per_rank = 32;

	// set tags for map_sync
	for (auto [direction, neighbor_id] : neighborIDs) {
		if (! isPrimaryDirection(direction) ^ isPrimaryDirection(opposite(direction)))
			throw std::logic_error("Bug in isPrimaryDirection!!!");
		if (neighbor_id < 0)
			map_sync.setTags(direction, -1, -1);
		else if (isPrimaryDirection(direction))
			map_sync.setTags(direction, blocks_per_rank * nproc + neighbor_id, this->id);
		else
			map_sync.setTags(direction, neighbor_id, blocks_per_rank * nproc + this->id);
		// disable DistributedNDArraySynchronizer initializing explicit -1 tags based on the tag_offset
		map_sync.setTagOffset(-1);
	}

	// set tags
	for (int i = 0; i < CONFIG::Q + CONFIG::MACRO::N; i++) {
		for (auto [direction, neighbor_id] : neighborIDs) {
			if (! isPrimaryDirection(direction) ^ isPrimaryDirection(opposite(direction)))
				throw std::logic_error("Bug in isPrimaryDirection!!!");
			if (neighbor_id < 0)
				dreal_sync[i].setTags(direction, -1, -1);
			else {
				const int offset0 = (2 * i + 0) * blocks_per_rank * nproc;
				const int offset1 = (2 * i + 1) * blocks_per_rank * nproc;
				if (isPrimaryDirection(direction))
					dreal_sync[i].setTags(direction, offset1 + neighbor_id, offset0 + this->id);
				else
					dreal_sync[i].setTags(direction, offset0 + neighbor_id, offset1 + this->id);
			}
			// disable DistributedNDArraySynchronizer initializing explicit -1 tags based on the tag_offset
			dreal_sync[i].setTagOffset(-1);
		}
	}
#endif

	// re-initialize compute data
	computeData.clear();
	for (auto direction : pattern)
		computeData[direction] = {};
	computeData[TNL::Containers::SyncDirection::None] = {};

	// create CUDA streams
#ifdef USE_CUDA
	// get the range of stream priorities for current GPU
	int priority_high;
	int priority_low;
	cudaDeviceGetStreamPriorityRange(&priority_low, &priority_high);
	// low-priority stream for the interior
	computeData.at(TNL::Containers::SyncDirection::None).stream = TNL::Backend::Stream::create(TNL::Backend::StreamNonBlocking, priority_low);
	// high-priority streams for boundaries
	for (auto direction : pattern) {
		computeData.at(direction).stream = TNL::Backend::Stream::create(TNL::Backend::StreamNonBlocking, priority_high);
	#ifdef HAVE_MPI
		// set the stream to the synchronizer
		for (int i = 0; i < CONFIG::Q + CONFIG::MACRO::N; i++)
			dreal_sync[i].setCudaStream(direction, computeData.at(direction).stream);
	#endif
	}
#endif

	// set the multi-index attributes in compute data
	idx3d interior_offset = {0, 0, 0};
	idx3d interior_size = local;

	// for compute on boundaries
	auto direction = TNL::Containers::SyncDirection::Left;
	if (auto search = neighborIDs.find(direction); search != neighborIDs.end() && search->second >= 0) {
		computeData.at(direction).offset = idx3d{0, 0, 0};
		computeData.at(direction).size = idx3d{overlap_width, local.y(), local.z()};
		interior_offset.x()++;
		interior_size.x() -= computeData.at(direction).size.x();
		computeData.at(direction).blockSize = getCudaBlockSize(computeData.at(direction).size);
		computeData.at(direction).gridSize = getCudaGridSize(computeData.at(direction).size, computeData.at(direction).blockSize, overlap_width);
	}
	direction = TNL::Containers::SyncDirection::Right;
	if (auto search = neighborIDs.find(direction); search != neighborIDs.end() && search->second >= 0) {
		computeData.at(direction).offset = idx3d{local.x() - overlap_width, 0, 0};
		computeData.at(direction).size = idx3d{overlap_width, local.y(), local.z()};
		interior_size.x() -= computeData.at(direction).size.x();
		computeData.at(direction).blockSize = getCudaBlockSize(computeData.at(direction).size);
		computeData.at(direction).gridSize = getCudaGridSize(computeData.at(direction).size, computeData.at(direction).blockSize, overlap_width);
	}
	direction = TNL::Containers::SyncDirection::Bottom;
	if (auto search = neighborIDs.find(direction); search != neighborIDs.end() && search->second >= 0) {
		computeData.at(direction).offset = idx3d{interior_offset.x(), 0, 0};
		computeData.at(direction).size = idx3d{interior_size.x(), overlap_width, local.z()};
		interior_offset.y()++;
		interior_size.y() -= computeData.at(direction).size.y();
		computeData.at(direction).blockSize = getCudaBlockSize(computeData.at(direction).size);
		computeData.at(direction).gridSize = getCudaGridSize(computeData.at(direction).size, computeData.at(direction).blockSize, 0, overlap_width);
	}
	direction = TNL::Containers::SyncDirection::Top;
	if (auto search = neighborIDs.find(direction); search != neighborIDs.end() && search->second >= 0) {
		computeData.at(direction).offset = idx3d{interior_offset.x(), local.y() - overlap_width, 0};
		computeData.at(direction).size = idx3d{interior_size.x(), overlap_width, local.z()};
		interior_size.y() -= computeData.at(direction).size.y();
		computeData.at(direction).blockSize = getCudaBlockSize(computeData.at(direction).size);
		computeData.at(direction).gridSize = getCudaGridSize(computeData.at(direction).size, computeData.at(direction).blockSize, 0, overlap_width);
	}
	direction = TNL::Containers::SyncDirection::Back;
	if (auto search = neighborIDs.find(direction); search != neighborIDs.end() && search->second >= 0) {
		computeData.at(direction).offset = idx3d{interior_offset.x(), interior_offset.y(), 0};
		computeData.at(direction).size = idx3d{interior_size.x(), interior_size.y(), overlap_width};
		interior_offset.z()++;
		interior_size.z() -= computeData.at(direction).size.z();
		computeData.at(direction).blockSize = getCudaBlockSize(computeData.at(direction).size);
		computeData.at(direction).gridSize =
			getCudaGridSize(computeData.at(direction).size, computeData.at(direction).blockSize, 0, 0, overlap_width);
	}
	direction = TNL::Containers::SyncDirection::Front;
	if (auto search = neighborIDs.find(direction); search != neighborIDs.end() && search->second >= 0) {
		computeData.at(direction).offset = idx3d{interior_offset.x(), interior_offset.y(), local.z() - overlap_width};
		computeData.at(direction).size = idx3d{interior_size.x(), interior_size.y(), overlap_width};
		interior_size.z() -= computeData.at(direction).size.z();
		computeData.at(direction).blockSize = getCudaBlockSize(computeData.at(direction).size);
		computeData.at(direction).gridSize =
			getCudaGridSize(computeData.at(direction).size, computeData.at(direction).blockSize, 0, 0, overlap_width);
	}

	// for compute on interior lattice sites
	direction = TNL::Containers::SyncDirection::None;
	computeData.at(direction).offset = interior_offset;
	computeData.at(direction).size = interior_size;
	computeData.at(direction).blockSize = getCudaBlockSize(interior_size);
	computeData.at(direction).gridSize = getCudaGridSize(interior_size, computeData.at(direction).blockSize);
}

template <typename CONFIG>
dim3 LBM_BLOCK<CONFIG>::getCudaBlockSize(const idx3d& local_size)
{
	// find optimal thread block size for the LBM kernel
	// use 256 threads for SP and 128 threads for DP
	constexpr int max_threads = 256 / (sizeof(dreal) / sizeof(float));
	const idx3d result = get_optimal_block_size<typename TRAITS::xyz_permutation>(local_size, max_threads);
	return {unsigned(result.x()), unsigned(result.y()), unsigned(result.z())};
}

template <typename CONFIG>
dim3 LBM_BLOCK<CONFIG>::getCudaGridSize(const idx3d& local_size, const dim3& block_size, idx x, idx y, idx z)
{
	dim3 gridSize;
	if (x > 0)
		gridSize.x = x;
	else
		gridSize.x = TNL::roundUpDivision(local_size.x(), block_size.x);
	if (y > 0)
		gridSize.y = y;
	else
		gridSize.y = TNL::roundUpDivision(local_size.y(), block_size.y);
	if (z > 0)
		gridSize.z = z;
	else
		gridSize.z = TNL::roundUpDivision(local_size.z(), block_size.z);
	return gridSize;
}

template <typename CONFIG>
void LBM_BLOCK<CONFIG>::setEquilibrium(real rho, real vx, real vy, real vz)
{
// extract variables and views for capturing in the lambda function
#ifdef HAVE_MPI
	auto local_df = dfs[0].getLocalView();
#else
	auto local_df = dfs[0].getView();
#endif

	// NOTE: it is important to reset *all* lattice sites (i.e. including ghost layers) when using the A-A pattern
	// (because GEO_INFLOW and GEO_OUTFLOW_EQ access the ghost layer in streaming)
	const int overlap_x = local_df.template getOverlap<1>();
	const int overlap_y = local_df.template getOverlap<2>();
	const int overlap_z = local_df.template getOverlap<3>();
	const idx3d begin = {-overlap_y, -overlap_z, -overlap_x};
	const idx3d end = {local.y() + overlap_y, local.z() + overlap_z, local.x() + overlap_x};

	TNL::Algorithms::parallelFor<DeviceType>(
		begin,
		end,
		[local_df, rho, vx, vy, vz] __cuda_callable__(idx3d yzx) mutable
		{
			const auto& [y, z, x] = yzx;
			CONFIG::COLL::setEquilibriumLat(local_df, x, y, z, rho, vx, vy, vz);
		}
	);

	// copy the initialized DFs so that they are not overridden
	for (uint8_t dftype = 1; dftype < DFMAX; dftype++)
		dfs[dftype] = dfs[0];
}

template <typename CONFIG>
void LBM_BLOCK<CONFIG>::computeInitialMacro()
{
	// extract variables and views for capturing in the lambda function
	auto SD = data;

	const idx3d begin = {0, 0, 0};
	const idx3d end = {local.y(), local.z(), local.x()};

	TNL::Algorithms::parallelFor<DeviceType>(
		begin,
		end,
		[SD] __cuda_callable__(idx3d yzx) mutable
		{
			const auto& [y, z, x] = yzx;
			typename CONFIG::template KernelStruct<dreal> KS;
			for (int i = 0; i < CONFIG::Q; i++)
				KS.f[i] = SD.df(df_cur, i, x, y, z);

			CONFIG::MACRO::copyQuantities(SD, KS, x, y, z);
			CONFIG::MACRO::zeroForcesInKS(KS);
			CONFIG::COLL::computeDensityAndVelocity(KS);
			CONFIG::MACRO::outputMacro(SD, KS, x, y, z);
		}
	);
}

template <typename CONFIG>
bool LBM_BLOCK<CONFIG>::isLocalIndex(idx x, idx y, idx z) const
{
	return x >= offset.x() && x < offset.x() + local.x() && y >= offset.y() && y < offset.y() + local.y() && z >= offset.z()
		&& z < offset.z() + local.z();
}

template <typename CONFIG>
bool LBM_BLOCK<CONFIG>::isLocalX(idx x) const
{
	return x >= offset.x() && x < offset.x() + local.x();
}

template <typename CONFIG>
bool LBM_BLOCK<CONFIG>::isLocalY(idx y) const
{
	return y >= offset.y() && y < offset.y() + local.y();
}

template <typename CONFIG>
bool LBM_BLOCK<CONFIG>::isLocalZ(idx z) const
{
	return z >= offset.z() && z < offset.z() + local.z();
}

template <typename CONFIG>
void LBM_BLOCK<CONFIG>::setMap(idx x, idx y, idx z, map_t value)
{
	if (isLocalIndex(x, y, z))
		hmap(x, y, z) = value;
}

template <typename CONFIG>
void LBM_BLOCK<CONFIG>::setBoundaryX(idx x, map_t value)
{
	if (isLocalX(x))
		for (idx y = offset.y(); y < offset.y() + local.y(); y++)
			for (idx z = offset.z(); z < offset.z() + local.z(); z++)
				hmap(x, y, z) = value;
}

template <typename CONFIG>
void LBM_BLOCK<CONFIG>::setBoundaryY(idx y, map_t value)
{
	if (isLocalY(y))
		for (idx x = offset.x(); x < offset.x() + local.x(); x++)
			for (idx z = offset.z(); z < offset.z() + local.z(); z++)
				hmap(x, y, z) = value;
}

template <typename CONFIG>
void LBM_BLOCK<CONFIG>::setBoundaryZ(idx z, map_t value)
{
	if (isLocalZ(z))
		for (idx x = offset.x(); x < offset.x() + local.x(); x++)
			for (idx y = offset.y(); y < offset.y() + local.y(); y++)
				hmap(x, y, z) = value;
}

template <typename CONFIG>
void LBM_BLOCK<CONFIG>::resetMap(map_t geo_type)
{
	hmap.setValue(geo_type);
}

template <typename CONFIG>
void LBM_BLOCK<CONFIG>::copyMapToHost()
{
	hmap = dmap;
	hdiffusionCoeff = ddiffusionCoeff;
	hphiTransferDirection = dphiTransferDirection;
}

template <typename CONFIG>
void LBM_BLOCK<CONFIG>::copyMapToDevice()
{
	dmap = hmap;
	ddiffusionCoeff = hdiffusionCoeff;
	dphiTransferDirection = hphiTransferDirection;
}

template <typename CONFIG>
void LBM_BLOCK<CONFIG>::copyMacroToHost()
{
	hmacro = dmacro;
}

template <typename CONFIG>
void LBM_BLOCK<CONFIG>::copyMacroToDevice()
{
	dmacro = hmacro;
}

template <typename CONFIG>
void LBM_BLOCK<CONFIG>::copyDFsToHost(uint8_t dfty)
{
	dlat_view_t df = dfs[0].getView();
	df.bind(data.dfs[dfty]);
	hfs[dfty] = df;
}

template <typename CONFIG>
void LBM_BLOCK<CONFIG>::copyDFsToDevice(uint8_t dfty)
{
	dlat_view_t df = dfs[0].getView();
	df.bind(data.dfs[dfty]);
	df = hfs[dfty];
}

template <typename CONFIG>
void LBM_BLOCK<CONFIG>::copyDFsToHost()
{
	for (uint8_t dfty = 0; dfty < DFMAX; dfty++)
		hfs[dfty] = dfs[dfty];
}

template <typename CONFIG>
void LBM_BLOCK<CONFIG>::copyDFsToDevice()
{
	for (uint8_t dfty = 0; dfty < DFMAX; dfty++)
		dfs[dfty] = hfs[dfty];
}

#ifdef HAVE_MPI

template <typename CONFIG>
template <typename Array>
void LBM_BLOCK<CONFIG>::startDrealArraySynchronization(Array& array, int sync_offset)
{
	static_assert(Array::getDimension() == 4, "4D array expected");
	constexpr int N = Array::SizesHolderType::template getStaticSize<0>();
	static_assert(N > 0, "the first dimension must be static");
	constexpr bool is_df = std::is_same<typename Array::ConstViewType, typename dlat_array_t::ConstViewType>::value;

	// empty view, but with correct sizes
	typename dreal_array_t::LocalViewType localView(nullptr, data.indexer);
	typename dreal_array_t::ViewType view(localView, dmap.getSizes(), dmap.getLocalBegins(), dmap.getLocalEnds(), dmap.getCommunicator());

	for (int i = 0; i < N; i++) {
		// rebind just the data pointer
		view.bind(array.getData() + i * data.indexer.getStorageSize());
		// determine sync direction
		TNL::Containers::SyncDirection sync_direction = (is_df) ? df_sync_directions[i] : TNL::Containers::SyncDirection::All;
	#ifdef AA_PATTERN
		// reset shift of the lattice sites
		dreal_sync[i + sync_offset].setBufferOffsets(0);
		if (is_df) {
			if (data.even_iter) {
				// lattice sites for synchronization are not shifted, but DFs have opposite directions
				sync_direction = opposite(sync_direction);
			}
			else {
				// DFs have canonical directions, but lattice sites for synchronization are shifted
				// (values to be synchronized were written to the neighboring sites)
				dreal_sync[i + sync_offset].setBufferOffsets(1);
			}
		}
	#endif
		// start the synchronization
		// NOTE: we don't use synchronize with policy because we need pipelining
		// NOTE: we could use only synchronize with policy=deferred, because threadpool and async require MPI_THREAD_MULTIPLE which is slow
		// stage 0: set inputs, allocate buffers
		dreal_sync[i + sync_offset].stage_0(view, sync_direction);
		// stage 1: fill send buffers
		dreal_sync[i + sync_offset].stage_1();
	}
}

template <typename CONFIG>
void LBM_BLOCK<CONFIG>::synchronizeDFsDevice_start(uint8_t dftype)
{
	auto df = dfs[0].getView();
	df.bind(data.dfs[dftype]);
	startDrealArraySynchronization(df, 0);
}

template <typename CONFIG>
void LBM_BLOCK<CONFIG>::synchronizeMacroDevice_start()
{
	startDrealArraySynchronization(dmacro, CONFIG::Q);
}

template <typename CONFIG>
void LBM_BLOCK<CONFIG>::synchronizeMapDevice_start()
{
	// NOTE: threadpool and async require MPI_THREAD_MULTIPLE which is slow
	constexpr auto policy = std::decay_t<decltype(map_sync)>::AsyncPolicy::deferred;
	map_sync.synchronize(policy, dmap);
}
#endif	// HAVE_MPI

template <typename CONFIG>
void LBM_BLOCK<CONFIG>::allocateHostData()
{
	for (uint8_t dfty = 0; dfty < DFMAX; dfty++) {
		hfs[dfty].setSizes(0, global.x(), global.y(), global.z());
#ifdef HAVE_MPI
		if (local.x() != global.x())
			hfs[dfty].getOverlaps().template setSize<1>(overlap_width);
		if (local.y() != global.y())
			hfs[dfty].getOverlaps().template setSize<2>(overlap_width);
		if (local.z() != global.z())
			hfs[dfty].getOverlaps().template setSize<3>(overlap_width);
		hfs[dfty].template setDistribution<1>(offset.x(), offset.x() + local.x(), communicator);
		hfs[dfty].template setDistribution<2>(offset.y(), offset.y() + local.y(), communicator);
		hfs[dfty].template setDistribution<3>(offset.z(), offset.z() + local.z(), communicator);
		hfs[dfty].allocate();
#endif
	}

	hmap.setSizes(global.x(), global.y(), global.z());
#ifdef HAVE_MPI
	if (local.x() != global.x())
		hmap.getOverlaps().template setSize<0>(overlap_width);
	if (local.y() != global.y())
		hmap.getOverlaps().template setSize<1>(overlap_width);
	if (local.z() != global.z())
		hmap.getOverlaps().template setSize<2>(overlap_width);
	hmap.template setDistribution<0>(offset.x(), offset.x() + local.x(), communicator);
	hmap.template setDistribution<1>(offset.y(), offset.y() + local.y(), communicator);
	hmap.template setDistribution<2>(offset.z(), offset.z() + local.z(), communicator);
	hmap.allocate();
#endif

	hmacro.setSizes(0, global.x(), global.y(), global.z());
#ifdef HAVE_MPI
	if (local.x() != global.x())
		hmacro.getOverlaps().template setSize<1>(macro_overlap_width);
	if (local.y() != global.y())
		hmacro.getOverlaps().template setSize<2>(macro_overlap_width);
	if (local.z() != global.z())
		hmacro.getOverlaps().template setSize<3>(macro_overlap_width);
	hmacro.template setDistribution<1>(offset.x(), offset.x() + local.x(), communicator);
	hmacro.template setDistribution<2>(offset.y(), offset.y() + local.y(), communicator);
	hmacro.template setDistribution<3>(offset.z(), offset.z() + local.z(), communicator);
	hmacro.allocate();
#endif
	hmacro.setValue(0);
}

template <typename CONFIG>
void LBM_BLOCK<CONFIG>::allocateDeviceData()
{
//#ifdef USE_CUDA
#if 1
	dmap.setSizes(global.x(), global.y(), global.z());
	#ifdef HAVE_MPI
	if (local.x() != global.x())
		dmap.getOverlaps().template setSize<0>(overlap_width);
	if (local.y() != global.y())
		dmap.getOverlaps().template setSize<1>(overlap_width);
	if (local.z() != global.z())
		dmap.getOverlaps().template setSize<2>(overlap_width);
	dmap.template setDistribution<0>(offset.x(), offset.x() + local.x(), communicator);
	dmap.template setDistribution<1>(offset.y(), offset.y() + local.y(), communicator);
	dmap.template setDistribution<2>(offset.z(), offset.z() + local.z(), communicator);
	dmap.allocate();
	#endif

	for (auto& df : dfs) {
		df.setSizes(0, global.x(), global.y(), global.z());
	#ifdef HAVE_MPI
		if (local.x() != global.x())
			df.getOverlaps().template setSize<1>(overlap_width);
		if (local.y() != global.y())
			df.getOverlaps().template setSize<2>(overlap_width);
		if (local.z() != global.z())
			df.getOverlaps().template setSize<3>(overlap_width);
		df.template setDistribution<1>(offset.x(), offset.x() + local.x(), communicator);
		df.template setDistribution<2>(offset.y(), offset.y() + local.y(), communicator);
		df.template setDistribution<3>(offset.z(), offset.z() + local.z(), communicator);
		df.allocate();
	#endif
	}

	dmacro.setSizes(0, global.x(), global.y(), global.z());
	#ifdef HAVE_MPI
	if (local.x() != global.x())
		dmacro.getOverlaps().template setSize<1>(macro_overlap_width);
	if (local.y() != global.y())
		dmacro.getOverlaps().template setSize<2>(macro_overlap_width);
	if (local.z() != global.z())
		dmacro.getOverlaps().template setSize<3>(macro_overlap_width);
	dmacro.template setDistribution<1>(offset.x(), offset.x() + local.x(), communicator);
	dmacro.template setDistribution<2>(offset.y(), offset.y() + local.y(), communicator);
	dmacro.template setDistribution<3>(offset.z(), offset.z() + local.z(), communicator);
	dmacro.allocate();
	#endif
#else
	// TODO: skip double allocation !!!
//	dmap=hmap;
//	dmacro=hmacro;
//	for (uint8_t dfty=0;dfty<DFMAX;dfty++)
//		dfs[dfty] = (dreal*)malloc(27*size_dreal);
////	df1 = (dreal*)malloc(27*size_dreal);
////	df2 = (dreal*)malloc(27*size_dreal);
#endif

	// initialize data pointers
	for (uint8_t dfty = 0; dfty < DFMAX; dfty++)
		data.dfs[dfty] = dfs[dfty].getData();
#ifdef HAVE_MPI
	data.indexer = dmap.getLocalView().getIndexer();
#else
	data.indexer = dmap.getIndexer();
#endif
	data.XYZ = data.indexer.getStorageSize();
	data.dmap = dmap.getData();
	data.dmacro = dmacro.getData();
}

template <typename CONFIG>
void LBM_BLOCK<CONFIG>::allocateDiffusionCoefficientArrays()
{
	hdiffusionCoeff.setSizes(global.x(), global.y(), global.z());
	ddiffusionCoeff.setSizes(global.x(), global.y(), global.z());
#ifdef HAVE_MPI
	hdiffusionCoeff.template setDistribution<0>(offset.x(), offset.x() + local.x(), communicator);
	hdiffusionCoeff.allocate();
	ddiffusionCoeff.template setDistribution<0>(offset.x(), offset.x() + local.x(), communicator);
	ddiffusionCoeff.allocate();
#endif
}

template <typename CONFIG>
void LBM_BLOCK<CONFIG>::allocatePhiTransferDirectionArrays()
{
	using hbool_array_t = typename CONFIG::hbool_array_t;
	hbool_array_t TransferFS;
	hbool_array_t TransferSF;
	hbool_array_t TransferSW;
	TransferFS.setSizes(global.x(), global.y(), global.z());
	TransferSF.setSizes(global.x(), global.y(), global.z());
	TransferSW.setSizes(global.x(), global.y(), global.z());

#ifdef HAVE_MPI
	TransferFS.template setDistribution<0>(offset.x(), offset.x() + local.x(), communicator);
	TransferFS.allocate();
	TransferSF.template setDistribution<0>(offset.x(), offset.x() + local.x(), communicator);
	TransferSF.allocate();
	TransferSW.template setDistribution<0>(offset.x(), offset.x() + local.x(), communicator);
	TransferSW.allocate();
#endif

	TransferFS.setValue(false);
	TransferSF.setValue(false);
	TransferSW.setValue(false);

	hphiTransferDirection.setSizes(0, global.x(), global.y(), global.z());
	dphiTransferDirection.setSizes(0, global.x(), global.y(), global.z());
#ifdef HAVE_MPI
	hphiTransferDirection.template setDistribution<1>(offset.x(), offset.x() + local.x(), communicator);
	hphiTransferDirection.allocate();
	dphiTransferDirection.template setDistribution<1>(offset.x(), offset.x() + local.x(), communicator);
	dphiTransferDirection.allocate();
#endif

	forLocalLatticeSites(
		[&](LBM_BLOCK& block, idx x, idx y, idx z)
		{
			if (CONFIG::BC::isFluid(hmap(x, y, z))) {
				if (CONFIG::BC::isSolid(hmap(x + 1, y, z))) {
					TransferFS(x, y, z) = true;
					hphiTransferDirection(pzz, x, y, z) = true;
				}
				if (CONFIG::BC::isSolid(hmap(x, y + 1, z))) {
					TransferFS(x, y, z) = true;
					hphiTransferDirection(zpz, x, y, z) = true;
				}
				if (CONFIG::BC::isSolid(hmap(x, y, z + 1))) {
					TransferFS(x, y, z) = true;
					hphiTransferDirection(zzp, x, y, z) = true;
				}
				if (CONFIG::BC::isSolid(hmap(x - 1, y, z))) {
					TransferFS(x, y, z) = true;
					hphiTransferDirection(mzz, x, y, z) = true;
				}
				if (CONFIG::BC::isSolid(hmap(x, y - 1, z))) {
					TransferFS(x, y, z) = true;
					hphiTransferDirection(zmz, x, y, z) = true;
				}
				if (CONFIG::BC::isSolid(hmap(x, y, z - 1))) {
					TransferFS(x, y, z) = true;
					hphiTransferDirection(zzm, x, y, z) = true;
				}
			}
			if (CONFIG::BC::isSolid(hmap(x, y, z))) {
				if (CONFIG::BC::isFluid(hmap(x + 1, y, z))) {
					TransferSF(x, y, z) = true;
					hphiTransferDirection(pzz, x, y, z) = true;
				}
				if (CONFIG::BC::isFluid(hmap(x, y + 1, z))) {
					TransferSF(x, y, z) = true;
					hphiTransferDirection(zpz, x, y, z) = true;
				}
				if (CONFIG::BC::isFluid(hmap(x, y, z + 1))) {
					TransferSF(x, y, z) = true;
					hphiTransferDirection(zzp, x, y, z) = true;
				}
				if (CONFIG::BC::isFluid(hmap(x - 1, y, z))) {
					TransferSF(x, y, z) = true;
					hphiTransferDirection(mzz, x, y, z) = true;
				}
				if (CONFIG::BC::isFluid(hmap(x, y - 1, z))) {
					TransferSF(x, y, z) = true;
					hphiTransferDirection(zmz, x, y, z) = true;
				}
				if (CONFIG::BC::isFluid(hmap(x, y, z - 1))) {
					TransferSF(x, y, z) = true;
					hphiTransferDirection(zzm, x, y, z) = true;
				}

				if (CONFIG::BC::isWall(hmap(x + 1, y, z))) {
					TransferSW(x, y, z) = true;
					hphiTransferDirection(pzz, x, y, z) = true;
				}
				if (CONFIG::BC::isWall(hmap(x, y + 1, z))) {
					TransferSW(x, y, z) = true;
					hphiTransferDirection(zpz, x, y, z) = true;
				}
				if (CONFIG::BC::isWall(hmap(x, y, z + 1))) {
					TransferSW(x, y, z) = true;
					hphiTransferDirection(zzp, x, y, z) = true;
				}
				if (CONFIG::BC::isWall(hmap(x - 1, y, z))) {
					TransferSW(x, y, z) = true;
					hphiTransferDirection(mzz, x, y, z) = true;
				}
				if (CONFIG::BC::isWall(hmap(x, y - 1, z))) {
					TransferSW(x, y, z) = true;
					hphiTransferDirection(zmz, x, y, z) = true;
				}
				if (CONFIG::BC::isWall(hmap(x, y, z - 1))) {
					TransferSW(x, y, z) = true;
					hphiTransferDirection(zzm, x, y, z) = true;
				}
			}
		}
	);

	forLocalLatticeSites(
		[&](LBM_BLOCK& block, idx x, idx y, idx z)
		{
			if (TransferFS(x, y, z))
				hmap(x, y, z) = CONFIG::BC::GEO_TRANSFER_FS;
			if (TransferSF(x, y, z))
				hmap(x, y, z) = CONFIG::BC::GEO_TRANSFER_SF;
			if (TransferSW(x, y, z))
				hmap(x, y, z) = CONFIG::BC::GEO_TRANSFER_SW;
		}
	);
}

template <typename CONFIG>
template <typename F>
void LBM_BLOCK<CONFIG>::forLocalLatticeSites(F f)
{
#pragma omp parallel for schedule(static) collapse(2)
	for (idx x = offset.x(); x < offset.x() + local.x(); x++)
		for (idx z = offset.z(); z < offset.z() + local.z(); z++)
			for (idx y = offset.y(); y < offset.y() + local.y(); y++)
				f(*this, x, y, z);
}

template <typename CONFIG>
template <typename F>
void LBM_BLOCK<CONFIG>::forAllLatticeSites(F f)
{
	const int overlap_x = hmap.template getOverlap<0>();
	const int overlap_y = hmap.template getOverlap<1>();
	const int overlap_z = hmap.template getOverlap<2>();

#pragma omp parallel for schedule(static) collapse(2)
	for (idx x = offset.x() - overlap_x; x < offset.x() + local.x() + overlap_x; x++)
		for (idx z = offset.z() - overlap_z; z < offset.z() + local.z() + overlap_z; z++)
			for (idx y = offset.y() - overlap_y; y < offset.y() + local.y() + overlap_y; y++)
				f(*this, x, y, z);
}

template <typename CONFIG>
template <typename Output>
void LBM_BLOCK<CONFIG>::writeVTK_3D(lat_t lat, Output&& outputData, const std::string& filename, real time, int cycle, DataManager& dataManager) const
{
	std::vector<int> tempIData;
	std::vector<float> tempFData;
	const std::size_t nCells = static_cast<std::size_t>(local.x()) * static_cast<std::size_t>(local.y()) * static_cast<std::size_t>(local.z());
	tempIData.reserve(nCells);
	tempFData.reserve(nCells);
	const point_t origin = lat.lbm2physPoint(0, 0, 0);
	UniformDataWriter<TRAITS> writer(global, local, offset, origin, lat.physDl, dataManager, filename);

	for (idx z = offset.z(); z < offset.z() + local.z(); z++)
		for (idx y = offset.y(); y < offset.y() + local.y(); y++)
			for (idx x = offset.x(); x < offset.x() + local.x(); x++)
				tempIData.push_back(hmap(x, y, z));

	writer.write("wall", tempIData, 1);
	tempIData.clear();

	OutputDataDescriptor<typename hmacro_array_t::ValueType> desc;
	int index = 0;
	while (outputData(*this, index++, 0, offset.x(), offset.y(), offset.z(), desc)) {
		for (int dof = 0; dof < desc.dofs; dof++) {
			for (idx z = offset.z(); z < offset.z() + local.z(); z++)
				for (idx y = offset.y(); y < offset.y() + local.y(); y++)
					for (idx x = offset.x(); x < offset.x() + local.x(); x++) {
						outputData(*this, index - 1, dof, x, y, z, desc);
						tempFData.push_back(desc.value);
					}
			// FIXME: the VTX reader does not support vector fields on ImageData
			// https://github.com/ornladios/ADIOS2/discussions/4117
			if (desc.dofs > 1)
				switch (dof) {
					case 0:
						desc.quantity += "X";
						break;
					case 1:
						desc.quantity += "Y";
						break;
					case 2:
						desc.quantity += "Z";
						break;
				}
			writer.write(desc.quantity, tempFData, desc.dofs);
			tempFData.clear();
		}
	}

	writer.write("TIME", time);
}

template <typename CONFIG>
template <typename Output>
void LBM_BLOCK<CONFIG>::writeVTK_3Dcut(
	lat_t lat,
	Output&& outputData,
	const std::string& filename,
	real time,
	int cycle,
	idx ox,
	idx oy,
	idx oz,
	idx gx,
	idx gy,
	idx gz,
	DataManager& dataManager
) const
{
	bool overlapT =
		! (ox + gx <= offset.x() || offset.x() + local.x() <= ox || oy + gy <= offset.y() || offset.y() + local.y() <= oy || oz + gz <= offset.z()
		   || offset.z() + local.z() <= oz);
	int color = overlapT ? 1 : MPI_UNDEFINED;
	TNL::MPI::Comm communicator = TNL::MPI::Comm::split(MPI_COMM_WORLD, color, 0);
	if (! overlapT)
		return;

	// intersection of the local domain with the box
	idx lx = TNL::min(ox + gx, offset.x() + local.x()) - TNL::max(ox, offset.x());
	idx ly = TNL::min(oy + gy, offset.y() + local.y()) - TNL::max(oy, offset.y());
	idx lz = TNL::min(oz + gz, offset.z() + local.z()) - TNL::max(oz, offset.z());

	idx oX = TNL::max(0, offset.x() - ox);
	idx oY = TNL::max(0, offset.y() - oy);
	idx oZ = TNL::max(0, offset.z() - oz);

	std::vector<int> tempIData;
	std::vector<float> tempFData;
	const std::size_t nCells = static_cast<std::size_t>(lx) * static_cast<std::size_t>(ly) * static_cast<std::size_t>(lz);
	tempIData.reserve(nCells);
	tempFData.reserve(nCells);
	const point_t origin = lat.lbm2physPoint(ox, oy, oz);
	UniformDataWriter<TRAITS> writer({gx, gy, gz}, {lx, ly, lz}, {oX, oY, oZ}, origin, lat.physDl, dataManager, filename);

	ox = TNL::max(ox, offset.x());
	oy = TNL::max(oy, offset.y());
	oz = TNL::max(oz, offset.z());

	for (idx z = oz; z < oz + lz; z++)
		for (idx y = oy; y < oy + ly; y++)
			for (idx x = ox; x < ox + lx; x++)
				tempIData.push_back(hmap(x, y, z));
	writer.write("wall", tempIData, 1);
	tempIData.clear();

	OutputDataDescriptor<typename hmacro_array_t::ValueType> desc;
	int index = 0;
	while (outputData(*this, index++, 0, ox, oy, oz, desc)) {
		for (int dof = 0; dof < desc.dofs; dof++) {
			for (idx z = oz; z < oz + lz; z++)
				for (idx y = oy; y < oy + ly; y++)
					for (idx x = ox; x < ox + lx; x++) {
						outputData(*this, index - 1, dof, x, y, z, desc);
						tempFData.push_back(desc.value);
					}
			// FIXME: the VTX reader does not support vector fields on ImageData
			// https://github.com/ornladios/ADIOS2/discussions/4117
			if (desc.dofs > 1)
				switch (dof) {
					case 0:
						desc.quantity += "X";
						break;
					case 1:
						desc.quantity += "Y";
						break;
					case 2:
						desc.quantity += "Z";
						break;
				}
			writer.write(desc.quantity, tempFData, desc.dofs);
			tempFData.clear();
		}
	}

	writer.write("TIME", time);
}

template <typename CONFIG>
template <typename Output>
void LBM_BLOCK<CONFIG>::writeVTK_2DcutX(
	lat_t lat, Output&& outputData, const std::string& filename, real time, int cycle, idx XPOS, DataManager& dataManager
) const
{
	int color = isLocalX(XPOS) ? 1 : MPI_UNDEFINED;
	TNL::MPI::Comm communicator = TNL::MPI::Comm::split(MPI_COMM_WORLD, color, 1);
	if (! isLocalX(XPOS))
		return;

	std::vector<int> tempIData;
	std::vector<float> tempFData;
	const std::size_t nCells = static_cast<std::size_t>(local.y()) * static_cast<std::size_t>(local.z());
	tempIData.reserve(nCells);
	tempFData.reserve(nCells);
	const point_t origin = lat.lbm2physPoint(XPOS, 0, 0);
	UniformDataWriter<TRAITS> writer(
		{1, global.y(), global.z()}, {1, local.y(), local.z()}, {0, offset.y(), offset.z()}, origin, lat.physDl, dataManager, filename
	);

	idx x = XPOS;
	for (idx z = offset.z(); z < offset.z() + local.z(); z++)
		for (idx y = offset.y(); y < offset.y() + local.y(); y++)
			tempIData.push_back(hmap(x, y, z));
	writer.write("wall", tempIData, 1);
	tempIData.clear();

	OutputDataDescriptor<typename hmacro_array_t::ValueType> desc;
	int index = 0;
	while (outputData(*this, index++, 0, offset.x(), offset.y(), offset.z(), desc)) {
		for (int dof = 0; dof < desc.dofs; dof++) {
			for (idx z = offset.z(); z < offset.z() + local.z(); z++)
				for (idx y = offset.y(); y < offset.y() + local.y(); y++) {
					outputData(*this, index - 1, dof, x, y, z, desc);
					tempFData.push_back(desc.value);
				}
			// FIXME: the VTX reader does not support vector fields on ImageData
			// https://github.com/ornladios/ADIOS2/discussions/4117
			if (desc.dofs > 1)
				switch (dof) {
					case 0:
						desc.quantity += "X";
						break;
					case 1:
						desc.quantity += "Y";
						break;
					case 2:
						desc.quantity += "Z";
						break;
				}
			writer.write(desc.quantity, tempFData, desc.dofs);
			tempFData.clear();
		}
	}

	writer.write("TIME", time);
}

template <typename CONFIG>
template <typename Output>
void LBM_BLOCK<CONFIG>::writeVTK_2DcutY(
	lat_t lat, Output&& outputData, const std::string& filename, real time, int cycle, idx YPOS, DataManager& dataManager
) const
{
	int color = isLocalY(YPOS) ? 1 : MPI_UNDEFINED;
	TNL::MPI::Comm communicator = TNL::MPI::Comm::split(MPI_COMM_WORLD, color, 2);
	if (! isLocalY(YPOS))
		return;

	std::vector<int> tempIData;
	std::vector<float> tempFData;
	const std::size_t nCells = static_cast<std::size_t>(local.x()) * static_cast<std::size_t>(local.z());
	tempIData.reserve(nCells);
	tempFData.reserve(nCells);
	const point_t origin = lat.lbm2physPoint(0, YPOS, 0);
	UniformDataWriter<TRAITS> writer(
		{global.x(), 1, global.z()}, {local.x(), 1, local.z()}, {offset.x(), 0, offset.z()}, origin, lat.physDl, dataManager, filename
	);

	idx y = YPOS;
	for (idx z = offset.z(); z < offset.z() + local.z(); z++)
		for (idx x = offset.x(); x < offset.x() + local.x(); x++)
			tempIData.push_back(hmap(x, y, z));
	writer.write("wall", tempIData, 1);
	tempIData.clear();

	OutputDataDescriptor<typename hmacro_array_t::ValueType> desc;
	int index = 0;
	while (outputData(*this, index++, 0, offset.x(), offset.y(), offset.z(), desc)) {
		for (int dof = 0; dof < desc.dofs; dof++) {
			for (idx z = offset.z(); z < offset.z() + local.z(); z++)
				for (idx x = offset.x(); x < offset.x() + local.x(); x++) {
					outputData(*this, index - 1, dof, x, y, z, desc);
					tempFData.push_back(desc.value);
				}
			// FIXME: the VTX reader does not support vector fields on ImageData
			// https://github.com/ornladios/ADIOS2/discussions/4117
			if (desc.dofs > 1)
				switch (dof) {
					case 0:
						desc.quantity += "X";
						break;
					case 1:
						desc.quantity += "Y";
						break;
					case 2:
						desc.quantity += "Z";
						break;
				}
			writer.write(desc.quantity, tempFData, desc.dofs);
			tempFData.clear();
		}
	}

	writer.write("TIME", time);
}

template <typename CONFIG>
template <typename Output>
void LBM_BLOCK<CONFIG>::writeVTK_2DcutZ(
	lat_t lat, Output&& outputData, const std::string& filename, real time, int cycle, idx ZPOS, DataManager& dataManager
) const
{
	int color = isLocalZ(ZPOS) ? 1 : MPI_UNDEFINED;
	TNL::MPI::Comm communicator = TNL::MPI::Comm::split(MPI_COMM_WORLD, color, 3);
	if (! isLocalZ(ZPOS))
		return;

	std::vector<int> tempIData;
	std::vector<float> tempFData;
	const std::size_t nCells = static_cast<std::size_t>(local.x()) * static_cast<std::size_t>(local.y());
	tempIData.reserve(nCells);
	tempFData.reserve(nCells);
	const point_t origin = lat.lbm2physPoint(0, 0, ZPOS);
	UniformDataWriter<TRAITS> writer(
		{global.x(), global.y(), 1}, {local.x(), local.y(), 1}, {offset.x(), offset.y(), 0}, origin, lat.physDl, dataManager, filename
	);

	idx z = ZPOS;
	for (idx y = offset.y(); y < offset.y() + local.y(); y++)
		for (idx x = offset.x(); x < offset.x() + local.x(); x++)
			tempIData.push_back(hmap(x, y, z));
	writer.write("wall", tempIData, 1);
	tempIData.clear();

	OutputDataDescriptor<typename hmacro_array_t::ValueType> desc;
	int index = 0;
	while (outputData(*this, index++, 0, offset.x(), offset.y(), offset.z(), desc)) {
		for (int dof = 0; dof < desc.dofs; dof++) {
			for (idx y = offset.y(); y < offset.y() + local.y(); y++)
				for (idx x = offset.x(); x < offset.x() + local.x(); x++) {
					outputData(*this, index - 1, dof, x, y, z, desc);
					tempFData.push_back(desc.value);
				}
			// FIXME: the VTX reader does not support vector fields on ImageData
			// https://github.com/ornladios/ADIOS2/discussions/4117
			if (desc.dofs > 1)
				switch (dof) {
					case 0:
						desc.quantity += "X";
						break;
					case 1:
						desc.quantity += "Y";
						break;
					case 2:
						desc.quantity += "Z";
						break;
				}
			writer.write(desc.quantity, tempFData, desc.dofs);
			tempFData.clear();
		}
	}

	writer.write("TIME", time);
}
