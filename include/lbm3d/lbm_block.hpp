#pragma once

#include "lbm_block.h"
#include "vtk_writer.h"
#include "block_size_optimizer.h"

template< typename CONFIG >
LBM_BLOCK<CONFIG>::LBM_BLOCK(const TNL::MPI::Comm& communicator, idx3d global, idx3d local, idx3d offset, int this_id)
: communicator(communicator), global(global), local(local), offset(offset), id(this_id)
{
	// initialize MPI info
	rank = communicator.rank();
	nproc = communicator.size();
}

template< typename CONFIG >
template< typename Pattern >
void LBM_BLOCK<CONFIG>::setLatticeDecomposition(
	const Pattern& pattern,
	const std::map< TNL::Containers::SyncDirection, int >& neighborIDs,
	const std::map< TNL::Containers::SyncDirection, int >& neighborRanks)
{
	this->neighborIDs = neighborIDs;
	this->neighborRanks = neighborRanks;

#ifdef HAVE_MPI
	// set communication pattern for all synchronizers
	map_sync.setSynchronizationPattern(pattern);
	for (int i = 0; i < CONFIG::Q + MACRO::N; i++)
		dreal_sync[i].setSynchronizationPattern(pattern);

	// set neighbors for all synchronizers
	for (auto [direction, rank] : neighborRanks) {
		map_sync.setNeighbor(direction, rank);
		for (int i = 0; i < CONFIG::Q + MACRO::N; i++)
			dreal_sync[i].setNeighbor(direction, rank);
	}

	auto isPrimaryDirection = [] (TNL::Containers::SyncDirection direction) -> bool
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
		else // direction & TNL::Containers::SyncDirection::Front
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
		map_sync.setTagOffset( -1 );
	}

	// set tags
	for (int i = 0; i < CONFIG::Q + MACRO::N; i++) {
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
			dreal_sync[i].setTagOffset( -1 );
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
	int priority_high, priority_low;
	cudaDeviceGetStreamPriorityRange(&priority_low, &priority_high);
	// low-priority stream for the interior
	computeData.at(TNL::Containers::SyncDirection::None).stream = TNL::Backend::Stream::create(TNL::Backend::StreamNonBlocking, priority_low);
	// high-priority streams for boundaries
	for (auto direction : pattern) {
		computeData.at(direction).stream = TNL::Backend::Stream::create(TNL::Backend::StreamNonBlocking, priority_high);
		#ifdef HAVE_MPI
		// set the stream to the synchronizer
		for (int i = 0; i < CONFIG::Q + MACRO::N; i++)
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
		computeData.at(direction).gridSize = getCudaGridSize(computeData.at(direction).size, computeData.at(direction).blockSize, 0, 0, overlap_width);
	}
	direction = TNL::Containers::SyncDirection::Front;
	if (auto search = neighborIDs.find(direction); search != neighborIDs.end() && search->second >= 0) {
		computeData.at(direction).offset = idx3d{interior_offset.x(), interior_offset.y(), local.z() - overlap_width};
		computeData.at(direction).size = idx3d{interior_size.x(), interior_size.y(), overlap_width};
		interior_size.z() -= computeData.at(direction).size.z();
		computeData.at(direction).blockSize = getCudaBlockSize(computeData.at(direction).size);
		computeData.at(direction).gridSize = getCudaGridSize(computeData.at(direction).size, computeData.at(direction).blockSize, 0, 0, overlap_width);
	}

	// for compute on interior lattice sites
	direction = TNL::Containers::SyncDirection::None;
	computeData.at(direction).offset = interior_offset;
	computeData.at(direction).size = interior_size;
	computeData.at(direction).blockSize = getCudaBlockSize(interior_size);
	computeData.at(direction).gridSize = getCudaGridSize(interior_size, computeData.at(direction).blockSize);
}

template< typename CONFIG >
dim3 LBM_BLOCK<CONFIG>::getCudaBlockSize(const idx3d& local_size)
{
	// find optimal thread block size for the LBM kernel
	// use 256 threads for SP and 128 threads for DP
	constexpr int max_threads = 256 / (sizeof(dreal) / sizeof(float));
	const idx3d result = get_optimal_block_size< typename TRAITS::xyz_permutation >(local_size, max_threads);
	return {unsigned(result.x()), unsigned(result.y()), unsigned(result.z())};
}

template< typename CONFIG >
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

template< typename CONFIG >
void LBM_BLOCK<CONFIG>::setEqLat(idx x, idx y, idx z, real rho, real vx, real vy, real vz)
{
	for (uint8_t dfty=0; dfty<DFMAX; dfty++) {
		#ifdef HAVE_MPI
		// shift global indices to local
		const auto local_begins = hfs[dfty].getLocalBegins();
		const idx lx = x - local_begins.template getSize< 1 >();
		const idx ly = y - local_begins.template getSize< 2 >();
		const idx lz = z - local_begins.template getSize< 3 >();
		// call setEquilibriumLat on the local array view
		auto local_view = hfs[dfty].getLocalView();
		CONFIG::COLL::setEquilibriumLat(local_view, lx, ly, lz, rho, vx, vy, vz);
		#else
		// without MPI, global array = local array
		auto local_view = hfs[dfty].getView();
		CONFIG::COLL::setEquilibriumLat(local_view, x, y, z, rho, vx, vy, vz);
		#endif
	}
}

template< typename CONFIG >
void LBM_BLOCK<CONFIG>::resetForces(real ifx, real ify, real ifz)
{
	/// Reset forces - This is necessary since '+=' is used afterwards.
	#pragma omp parallel for schedule(static) collapse(2)
	for (idx x = offset.x(); x < offset.x() + local.x(); x++)
	for (idx z = offset.z(); z < offset.z() + local.z(); z++)
	for (idx y = offset.y(); y < offset.y() + local.y(); y++)
	{
		hmacro(MACRO::e_fx, x, y, z) = ifx;
		hmacro(MACRO::e_fy, x, y, z) = ify;
		hmacro(MACRO::e_fz, x, y, z) = ifz;
	}
}

template< typename CONFIG >
void LBM_BLOCK<CONFIG>::copyForcesToDevice()
{
	// FIXME: overlaps
	#ifdef USE_CUDA
	TNL::Backend::memcpy(dfx(), hfx(), local.x()*local.y()*local.z()*sizeof(dreal), TNL::Backend::MemcpyHostToDevice);
	TNL::Backend::memcpy(dfy(), hfy(), local.x()*local.y()*local.z()*sizeof(dreal), TNL::Backend::MemcpyHostToDevice);
	TNL::Backend::memcpy(dfz(), hfz(), local.x()*local.y()*local.z()*sizeof(dreal), TNL::Backend::MemcpyHostToDevice);
	TNL_CHECK_CUDA_DEVICE;
	#endif
}


template< typename CONFIG >
bool LBM_BLOCK<CONFIG>::isLocalIndex(idx x, idx y, idx z) const
{
	return x >= offset.x() && x < offset.x() + local.x() &&
		y >= offset.y() && y < offset.y() + local.y() &&
		z >= offset.z() && z < offset.z() + local.z();
}

template< typename CONFIG >
bool LBM_BLOCK<CONFIG>::isLocalX(idx x) const
{
	return x >= offset.x() && x < offset.x() + local.x();
}

template< typename CONFIG >
bool LBM_BLOCK<CONFIG>::isLocalY(idx y) const
{
	return y >= offset.y() && y < offset.y() + local.y();
}

template< typename CONFIG >
bool LBM_BLOCK<CONFIG>::isLocalZ(idx z) const
{
	return z >= offset.z() && z < offset.z() + local.z();
}

template< typename CONFIG >
void LBM_BLOCK<CONFIG>::setMap(idx x, idx y, idx z, map_t value)
{
	if (isLocalIndex(x, y, z)) hmap(x, y, z) = value;
}

template< typename CONFIG >
void LBM_BLOCK<CONFIG>::setBoundaryX(idx x, map_t value)
{
	if (isLocalX(x))
		for (idx y = offset.y(); y < offset.y() + local.y(); y++)
		for (idx z = offset.z(); z < offset.z() + local.z(); z++)
			hmap(x, y, z) = value;
}

template< typename CONFIG >
void LBM_BLOCK<CONFIG>::setBoundaryY(idx y, map_t value)
{
	if (isLocalY(y))
		for (idx x = offset.x(); x < offset.x() + local.x(); x++)
		for (idx z = offset.z(); z < offset.z() + local.z(); z++)
			hmap(x, y, z) = value;
}

template< typename CONFIG >
void LBM_BLOCK<CONFIG>::setBoundaryZ(idx z, map_t value)
{
	if (isLocalZ(z))
		for (idx x = offset.x(); x < offset.x() + local.x(); x++)
		for (idx y = offset.y(); y < offset.y() + local.y(); y++)
			hmap(x, y, z) = value;
}

template< typename CONFIG >
bool LBM_BLOCK<CONFIG>::isFluid(idx x, idx y, idx z) const
{
	if (!isLocalIndex(x, y, z)) return false;
	return CONFIG::BC::isFluid(hmap(x,y,z));
}

template< typename CONFIG >
void LBM_BLOCK<CONFIG>::resetMap(map_t geo_type)
{
	hmap.setValue(geo_type);
}


template< typename CONFIG >
void  LBM_BLOCK<CONFIG>::copyMapToHost()
{
	hmap = dmap;
}

template< typename CONFIG >
void  LBM_BLOCK<CONFIG>::copyMapToDevice()
{
	dmap = hmap;
}

template< typename CONFIG >
void LBM_BLOCK<CONFIG>::copyMacroToHost()
{
	hmacro = dmacro;
}

template< typename CONFIG >
void LBM_BLOCK<CONFIG>::copyMacroToDevice()
{
	dmacro = hmacro;
}

template< typename CONFIG >
void LBM_BLOCK<CONFIG>::copyDFsToHost(uint8_t dfty)
{
	dlat_view_t df = dfs[0].getView();
	df.bind(data.dfs[dfty]);
	hfs[dfty] = df;
}

template< typename CONFIG >
void LBM_BLOCK<CONFIG>::copyDFsToDevice(uint8_t dfty)
{
	dlat_view_t df = dfs[0].getView();
	df.bind(data.dfs[dfty]);
	df = hfs[dfty];
}

template< typename CONFIG >
void LBM_BLOCK<CONFIG>::copyDFsToHost()
{
	for (uint8_t dfty=0;dfty<DFMAX;dfty++)
		hfs[dfty] = dfs[dfty];
}

template< typename CONFIG >
void LBM_BLOCK<CONFIG>::copyDFsToDevice()
{
	for (uint8_t dfty=0;dfty<DFMAX;dfty++)
		dfs[dfty] = hfs[dfty];
}

#ifdef HAVE_MPI

template< typename CONFIG >
	template< typename Array >
void LBM_BLOCK<CONFIG>::startDrealArraySynchronization(Array& array, int sync_offset)
{
	static_assert( Array::getDimension() == 4, "4D array expected" );
	constexpr int N = Array::SizesHolderType::template getStaticSize<0>();
	static_assert( N > 0, "the first dimension must be static" );
	constexpr bool is_df = std::is_same< typename Array::ConstViewType, typename dlat_array_t::ConstViewType >::value;

	// empty view, but with correct sizes
	typename sync_array_t::LocalViewType localView(nullptr, data.indexer);
	typename sync_array_t::ViewType view(localView, dmap.getSizes(), dmap.getLocalBegins(), dmap.getLocalEnds(), dmap.getCommunicator());

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

template< typename CONFIG >
void LBM_BLOCK<CONFIG>::synchronizeDFsDevice_start(uint8_t dftype)
{
	auto df = dfs[0].getView();
	df.bind(data.dfs[dftype]);
	startDrealArraySynchronization(df, 0);
}

template< typename CONFIG >
void LBM_BLOCK<CONFIG>::synchronizeMacroDevice_start()
{
	startDrealArraySynchronization(dmacro, CONFIG::Q);
}

template< typename CONFIG >
void LBM_BLOCK<CONFIG>::synchronizeMapDevice_start()
{
	// NOTE: threadpool and async require MPI_THREAD_MULTIPLE which is slow
	constexpr auto policy = std::decay_t<decltype(map_sync)>::AsyncPolicy::deferred;
	map_sync.synchronize(policy, dmap);
}
#endif  // HAVE_MPI

template< typename CONFIG >
void LBM_BLOCK<CONFIG>::computeCPUMacroFromLat()
{
	// take Lat, compute KS and then CPU_MACRO
	if (CPU_MACRO::N > 0)
	{
		typename CONFIG::DATA SD;
		for (uint8_t dfty=0;dfty<DFMAX;dfty++)
			SD.dfs[dfty] = hfs[dfty].getData();
		#ifdef HAVE_MPI
		SD.indexer = hmap.getLocalView().getIndexer();
		#else
		SD.indexer = hmap.getIndexer();
		#endif
		SD.XYZ = SD.indexer.getStorageSize();
		SD.dmap = hmap.getData();
		SD.dmacro = cpumacro.getData();

		#pragma omp parallel for schedule(static) collapse(2)
		for (idx x=0; x<local.x(); x++)
		for (idx z=0; z<local.z(); z++)
		for (idx y=0; y<local.y(); y++)
		{
			typename CONFIG::template KernelStruct<dreal> KS;
			KS.fx=0;
			KS.fy=0;
			KS.fz=0;
			CONFIG::COLL::copyDFcur2KS(SD, KS, x, y, z);
			CONFIG::COLL::computeDensityAndVelocity(KS);
			CPU_MACRO::outputMacro(SD, KS, x, y, z);
//			if (x==128 && y==23 && z==103)
//			printf("KS: %e %e %e %e vs. cpumacro %e %e %e %e [at %d %d %d]\n", KS.vx, KS.vy, KS.vz, KS.rho, cpumacro[mpos(CPU_MACRO::e_vx,x,y,z)], cpumacro[mpos(CPU_MACRO::e_vy,x,y,z)], cpumacro[mpos(CPU_MACRO::e_vz,x,y,z)],cpumacro[mpos(CPU_MACRO::e_rho,x,y,z)],x,y,z);
		}
//                printf("computeCPUMAcroFromLat done.\n");
	}
}

template< typename CONFIG >
void LBM_BLOCK<CONFIG>::allocateHostData()
{
	for (uint8_t dfty=0;dfty<DFMAX;dfty++)
	{
		hfs[dfty].setSizes(0, global.x(), global.y(), global.z());
		#ifdef HAVE_MPI
		if (local.x() != global.x())
			hfs[dfty].getOverlaps().template setSize< 1 >( 1 );
		if (local.y() != global.y())
			hfs[dfty].getOverlaps().template setSize< 2 >( 1 );
		if (local.z() != global.z())
			hfs[dfty].getOverlaps().template setSize< 3 >( 1 );
		hfs[dfty].template setDistribution< 1 >(offset.x(), offset.x() + local.x(), communicator);
		hfs[dfty].template setDistribution< 2 >(offset.y(), offset.y() + local.y(), communicator);
		hfs[dfty].template setDistribution< 3 >(offset.z(), offset.z() + local.z(), communicator);
		hfs[dfty].allocate();
		#endif
	}

	hmap.setSizes(global.x(), global.y(), global.z());
#ifdef HAVE_MPI
	if (local.x() != global.x())
		hmap.getOverlaps().template setSize< 0 >( 1 );
	if (local.y() != global.y())
		hmap.getOverlaps().template setSize< 1 >( 1 );
	if (local.z() != global.z())
		hmap.getOverlaps().template setSize< 2 >( 1 );
	hmap.template setDistribution< 0 >(offset.x(), offset.x() + local.x(), communicator);
	hmap.template setDistribution< 1 >(offset.y(), offset.y() + local.y(), communicator);
	hmap.template setDistribution< 2 >(offset.z(), offset.z() + local.z(), communicator);
	hmap.allocate();
#endif

	hmacro.setSizes(0, global.x(), global.y(), global.z());
	cpumacro.setSizes(0, global.x(), global.y(), global.z());
#ifdef HAVE_MPI
	if (local.x() != global.x())
		hmacro.getOverlaps().template setSize< 1 >( 1 );
	if (local.y() != global.y())
		hmacro.getOverlaps().template setSize< 2 >( 1 );
	if (local.z() != global.z())
		hmacro.getOverlaps().template setSize< 3 >( 1 );
	hmacro.template setDistribution< 1 >(offset.x(), offset.x() + local.x(), communicator);
	hmacro.template setDistribution< 2 >(offset.y(), offset.y() + local.y(), communicator);
	hmacro.template setDistribution< 3 >(offset.z(), offset.z() + local.z(), communicator);
	hmacro.allocate();
	if (local.x() != global.x())
		cpumacro.getOverlaps().template setSize< 1 >( 1 );
	if (local.y() != global.y())
		cpumacro.getOverlaps().template setSize< 2 >( 1 );
	if (local.z() != global.z())
		cpumacro.getOverlaps().template setSize< 3 >( 1 );
	cpumacro.template setDistribution< 1 >(offset.x(), offset.x() + local.x(), communicator);
	cpumacro.template setDistribution< 2 >(offset.y(), offset.y() + local.y(), communicator);
	cpumacro.template setDistribution< 3 >(offset.z(), offset.z() + local.z(), communicator);
	cpumacro.allocate();
#endif
	hmacro.setValue(0);
	// avoid setting empty array
	if (CPU_MACRO::N)
		cpumacro.setValue(0);
}

template< typename CONFIG >
void LBM_BLOCK<CONFIG>::allocateDeviceData()
{
//#ifdef USE_CUDA
#if 1
	dmap.setSizes(global.x(), global.y(), global.z());
	#ifdef HAVE_MPI
	if (local.x() != global.x())
		dmap.getOverlaps().template setSize< 0 >( 1 );
	if (local.y() != global.y())
		dmap.getOverlaps().template setSize< 1 >( 1 );
	if (local.z() != global.z())
		dmap.getOverlaps().template setSize< 2 >( 1 );
	dmap.template setDistribution< 0 >(offset.x(), offset.x() + local.x(), communicator);
	dmap.template setDistribution< 1 >(offset.y(), offset.y() + local.y(), communicator);
	dmap.template setDistribution< 2 >(offset.z(), offset.z() + local.z(), communicator);
	dmap.allocate();
	#endif

	for (uint8_t dfty=0;dfty<DFMAX;dfty++)
	{
		dfs[dfty].setSizes(0, global.x(), global.y(), global.z());
		#ifdef HAVE_MPI
		if (local.x() != global.x())
			dfs[dfty].getOverlaps().template setSize< 1 >( 1 );
		if (local.y() != global.y())
			dfs[dfty].getOverlaps().template setSize< 2 >( 1 );
		if (local.z() != global.z())
			dfs[dfty].getOverlaps().template setSize< 3 >( 1 );
		dfs[dfty].template setDistribution< 1 >(offset.x(), offset.x() + local.x(), communicator);
		dfs[dfty].template setDistribution< 2 >(offset.y(), offset.y() + local.y(), communicator);
		dfs[dfty].template setDistribution< 3 >(offset.z(), offset.z() + local.z(), communicator);
		dfs[dfty].allocate();
		#endif
	}

	dmacro.setSizes(0, global.x(), global.y(), global.z());
	#ifdef HAVE_MPI
	if (local.x() != global.x())
		dmacro.getOverlaps().template setSize< 1 >( 1 );
	if (local.y() != global.y())
		dmacro.getOverlaps().template setSize< 2 >( 1 );
	if (local.z() != global.z())
		dmacro.getOverlaps().template setSize< 3 >( 1 );
	dmacro.template setDistribution< 1 >(offset.x(), offset.x() + local.x(), communicator);
	dmacro.template setDistribution< 2 >(offset.y(), offset.y() + local.y(), communicator);
	dmacro.template setDistribution< 3 >(offset.z(), offset.z() + local.z(), communicator);
	dmacro.allocate();
	#endif
#else
	// TODO: skip douple allocation !!!
//	dmap=hmap;
//	dmacro=hmacro;
//	for (uint8_t dfty=0;dfty<DFMAX;dfty++)
//		dfs[dfty] = (dreal*)malloc(27*size_dreal);
////	df1 = (dreal*)malloc(27*size_dreal);
////	df2 = (dreal*)malloc(27*size_dreal);
#endif

	// initialize data pointers
	for (uint8_t dfty=0;dfty<DFMAX;dfty++)
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

template< typename CONFIG >
	template< typename F >
void LBM_BLOCK<CONFIG>::forLocalLatticeSites(F f)
{
	#pragma omp parallel for schedule(static) collapse(2)
	for (idx x = offset.x(); x < offset.x() + local.x(); x++)
	for (idx z = offset.z(); z < offset.z() + local.z(); z++)
	for (idx y = offset.y(); y < offset.y() + local.y(); y++)
		f(*this, x, y, z);
}

template< typename CONFIG >
	template< typename F >
void LBM_BLOCK<CONFIG>::forAllLatticeSites(F f)
{
	const int overlap_x = hmap.template getOverlap< 0 >();
	const int overlap_y = hmap.template getOverlap< 1 >();
	const int overlap_z = hmap.template getOverlap< 2 >();

	#pragma omp parallel for schedule(static) collapse(2)
	for (idx x = offset.x() - overlap_x; x < offset.x() + local.x() + overlap_x; x++)
	for (idx z = offset.z() - overlap_z; z < offset.z() + local.z() + overlap_z; z++)
	for (idx y = offset.y() - overlap_y; y < offset.y() + local.y() + overlap_y; y++)
		f(*this, x, y, z);
}

template< typename CONFIG >
	template< typename Output >
void LBM_BLOCK<CONFIG>::writeVTK_3D(lat_t lat, Output&& outputData, const std::string& filename, real time, int cycle) const
{
	VTKWriter vtk;

	FILE* fp = fopen(filename.c_str(), "w+");
	vtk.writeHeader(fp);
	fprintf(fp,"DATASET RECTILINEAR_GRID\n");
	fprintf(fp,"DIMENSIONS %d %d %d\n", (int)local.x(), (int)local.y(), (int)local.z());
	fprintf(fp,"X_COORDINATES %d float\n", (int)local.x());
	for (idx x = offset.x(); x < offset.x() + local.x(); x++)
		vtk.writeFloat(fp, lat.lbm2physX(x));
	vtk.writeBuffer(fp);

	fprintf(fp,"Y_COORDINATES %d float\n", (int)local.y());
	for (idx y = offset.y(); y < offset.y() + local.y(); y++)
		vtk.writeFloat(fp, lat.lbm2physY(y));
	vtk.writeBuffer(fp);

	fprintf(fp,"Z_COORDINATES %d float\n", (int)local.z());
	for (idx z = offset.z(); z < offset.z() + local.z(); z++)
		vtk.writeFloat(fp, lat.lbm2physZ(z));
	vtk.writeBuffer(fp);

	fprintf(fp,"FIELD FieldData %d\n",2);
	fprintf(fp,"TIME %d %d float\n",1,1);
	vtk.writeFloat(fp, time);
	vtk.writeBuffer(fp);

	fprintf(fp,"CYCLE %d %d float\n",1,1);
	vtk.writeFloat(fp, cycle);
	vtk.writeBuffer(fp);

	fprintf(fp,"POINT_DATA %d\n", (int)(local.x()*local.y()*local.z()));

	fprintf(fp,"SCALARS wall int 1\n");
	fprintf(fp,"LOOKUP_TABLE default\n");
	for (idx z = offset.z(); z < offset.z() + local.z(); z++)
	for (idx y = offset.y(); y < offset.y() + local.y(); y++)
	for (idx x = offset.x(); x < offset.x() + local.x(); x++)
		vtk.writeInt(fp, hmap(x,y,z));

	char idd[500];
	real value;
	int dofs;
	int index=0;
	while (outputData(*this, index++, 0, idd, offset.x(), offset.y(), offset.z(), value, dofs))
	{
		// insert description
		if (dofs==1)
		{
			fprintf(fp,"SCALARS %s float 1\n",idd);
			fprintf(fp,"LOOKUP_TABLE default\n");
		}
		else
			fprintf(fp,"VECTORS %s float\n",idd);

		for (idx z = offset.z(); z < offset.z() + local.z(); z++)
		for (idx y = offset.y(); y < offset.y() + local.y(); y++)
		for (idx x = offset.x(); x < offset.x() + local.x(); x++)
		{
			for (int dof=0;dof<dofs;dof++)
			{
				outputData(*this, index-1, dof, idd, x, y, z, value, dofs);
				vtk.writeFloat(fp, value);
			}
		}
		vtk.writeBuffer(fp);
	}

	fclose(fp);
}

template< typename CONFIG >
	template< typename Output >
void LBM_BLOCK<CONFIG>::writeVTK_3Dcut(lat_t lat, Output&& outputData, const std::string& filename, real time, int cycle, idx ox, idx oy, idx oz, idx lx, idx ly, idx lz, idx step) const
{
	if (!isLocalIndex(ox, oy, oz)) return;

	VTKWriter vtk;

	// intersection of the local domain with the box
	lx = MIN(ox + lx, offset.x() + local.x()) - MAX(ox, offset.x());
	ly = MIN(oy + ly, offset.y() + local.y()) - MAX(oy, offset.y());
	lz = MIN(oz + lz, offset.z() + local.z()) - MAX(oz, offset.z());
	ox = MAX(ox, offset.x());
	oy = MAX(oy, offset.y());
	oz = MAX(oz, offset.z());

	// box dimensions (round-up integer division)
	idx X = lx / step + (lx % step != 0);
	idx Y = ly / step + (ly % step != 0);
	idx Z = lz / step + (lz % step != 0);

	FILE* fp = fopen(filename.c_str(), "w+");
	vtk.writeHeader(fp);
	fprintf(fp,"DATASET RECTILINEAR_GRID\n");
	fprintf(fp,"DIMENSIONS %d %d %d\n", (int)X, (int)Y, (int)Z);
	fprintf(fp,"X_COORDINATES %d float\n", (int)X);
	for (idx x = ox; x < ox + lx; x += step)
		vtk.writeFloat(fp, lat.lbm2physX(x));
	vtk.writeBuffer(fp);

	fprintf(fp,"Y_COORDINATES %d float\n", (int)Y);
	for (idx y = oy; y < oy + ly; y += step)
		vtk.writeFloat(fp, lat.lbm2physY(y));
	vtk.writeBuffer(fp);

	fprintf(fp,"Z_COORDINATES %d float\n", (int)Z);
	for (idx z = oz; z < oz + lz; z += step)
		vtk.writeFloat(fp, lat.lbm2physZ(z));
	vtk.writeBuffer(fp);

	fprintf(fp,"FIELD FieldData %d\n",2);
	fprintf(fp,"TIME %d %d float\n",1,1);
	vtk.writeFloat(fp, time);
	vtk.writeBuffer(fp);

	fprintf(fp,"CYCLE %d %d float\n",1,1);
	vtk.writeFloat(fp, cycle);
	vtk.writeBuffer(fp);

	fprintf(fp,"POINT_DATA %d\n", (int)(X*Y*Z));

	fprintf(fp,"SCALARS wall int 1\n");
	fprintf(fp,"LOOKUP_TABLE default\n");
	for (idx z = oz; z < oz + lz; z += step)
	for (idx y = oy; y < oy + ly; y += step)
	for (idx x = ox; x < ox + lx; x += step)
		vtk.writeInt(fp, hmap(x,y,z));

	char idd[500];
	real value;
	int dofs;
	int index=0;
	while (outputData(*this, index++, 0, idd, ox, oy, oz, value, dofs))
	{
		// insert description
		if (dofs==1)
		{
			fprintf(fp,"SCALARS %s float 1\n",idd);
			fprintf(fp,"LOOKUP_TABLE default\n");
		}
		else
			fprintf(fp,"VECTORS %s float\n",idd);

		for (idx z = oz; z < oz + lz; z += step)
		for (idx y = oy; y < oy + ly; y += step)
		for (idx x = ox; x < ox + lx; x += step)
		{
			for (int dof=0;dof<dofs;dof++)
			{
				outputData(*this, index-1, dof, idd, x, y, z, value,dofs);
				vtk.writeFloat(fp, value);
			}
		}
		vtk.writeBuffer(fp);
	}

	fclose(fp);
}

template< typename CONFIG >
	template< typename Output >
void LBM_BLOCK<CONFIG>::writeVTK_2DcutX(lat_t lat, Output&& outputData, const std::string& filename, real time, int cycle, idx XPOS) const
{
	if (!isLocalX(XPOS)) return;

	VTKWriter vtk;

	FILE* fp = fopen(filename.c_str(), "w+");
	vtk.writeHeader(fp);
	fprintf(fp,"DATASET RECTILINEAR_GRID\n");
	fprintf(fp,"DIMENSIONS %d %d %d\n",1, (int)local.y(), (int)local.z());

	fprintf(fp,"X_COORDINATES %d float\n", 1);
	vtk.writeFloat(fp, lat.lbm2physX(XPOS));
	vtk.writeBuffer(fp);

	fprintf(fp,"Y_COORDINATES %d float\n", (int)local.y());
	for (idx y = offset.y(); y < offset.y() + local.y(); y++)
		vtk.writeFloat(fp, lat.lbm2physY(y));
	vtk.writeBuffer(fp);

	fprintf(fp,"Z_COORDINATES %d float\n", (int)local.z());
	for (idx z = offset.z(); z < offset.z() + local.z(); z++)
		vtk.writeFloat(fp, lat.lbm2physZ(z));
	vtk.writeBuffer(fp);


	fprintf(fp,"FIELD FieldData %d\n",2);
	fprintf(fp,"TIME %d %d float\n",1,1);
	vtk.writeFloat(fp, time);
	vtk.writeBuffer(fp);

	fprintf(fp,"CYCLE %d %d float\n",1,1);
	vtk.writeFloat(fp, cycle);
	vtk.writeBuffer(fp);

	fprintf(fp,"POINT_DATA %d\n", (int)(1*local.y()*local.z()));

	fprintf(fp,"SCALARS wall int 1\n");
	fprintf(fp,"LOOKUP_TABLE default\n");
	idx x=XPOS;
	for (idx z = offset.z(); z < offset.z() + local.z(); z++)
	for (idx y = offset.y(); y < offset.y() + local.y(); y++)
		vtk.writeInt(fp, hmap(x,y,z));

	int index=0;
	char idd[500];
	real value;
	int dofs;
	while (outputData(*this, index++, 0, idd, offset.x(),offset.y(),offset.z(), value, dofs))
	{
		// insert description
		if (dofs==1)
		{
			fprintf(fp,"SCALARS %s float 1\n",idd);
			fprintf(fp,"LOOKUP_TABLE default\n");
		} else
		{
			fprintf(fp,"VECTORS %s float\n",idd);
		}
		for (idx z = offset.z(); z < offset.z() + local.z(); z++)
		for (idx y = offset.y(); y < offset.y() + local.y(); y++)
		{
			for (int dof=0;dof<dofs;dof++)
			{
				outputData(*this, index-1,dof,idd,x,y,z,value,dofs);
				vtk.writeFloat(fp, value);
			}
		}
		vtk.writeBuffer(fp);
	}

	fclose(fp);
}

template< typename CONFIG >
	template< typename Output >
void LBM_BLOCK<CONFIG>::writeVTK_2DcutY(lat_t lat, Output&& outputData, const std::string& filename, real time, int cycle, idx YPOS) const
{
	if (!isLocalY(YPOS)) return;

	VTKWriter vtk;

	FILE* fp = fopen(filename.c_str(), "w+");
	vtk.writeHeader(fp);
	fprintf(fp,"DATASET RECTILINEAR_GRID\n");
	fprintf(fp,"DIMENSIONS %d %d %d\n", (int)local.x(), 1, (int)local.z());
	fprintf(fp,"X_COORDINATES %d float\n", (int)local.x());
	for (idx x = offset.x(); x < offset.x() + local.x(); x++)
		vtk.writeFloat(fp, lat.lbm2physX(x));
	vtk.writeBuffer(fp);

	fprintf(fp,"Y_COORDINATES 1 float\n");
	vtk.writeFloat(fp, lat.lbm2physY(YPOS));
	vtk.writeBuffer(fp);

	fprintf(fp,"Z_COORDINATES %d float\n", (int)local.z());
	for (idx z = offset.z(); z < offset.z() + local.z(); z++)
		vtk.writeFloat(fp, lat.lbm2physZ(z));
	vtk.writeBuffer(fp);


	fprintf(fp,"FIELD FieldData %d\n",2);
	fprintf(fp,"TIME %d %d float\n",1,1);
	vtk.writeFloat(fp, time);
	vtk.writeBuffer(fp);

	fprintf(fp,"CYCLE %d %d float\n",1,1);
	vtk.writeFloat(fp, cycle);
	vtk.writeBuffer(fp);

	fprintf(fp,"POINT_DATA %d\n", (int)(1*local.x()*local.z()));

	fprintf(fp,"SCALARS wall int 1\n");
	fprintf(fp,"LOOKUP_TABLE default\n");
	idx y=YPOS;
	for (idx z = offset.z(); z < offset.z() + local.z(); z++)
	for (idx x = offset.x(); x < offset.x() + local.x(); x++)
		vtk.writeInt(fp, hmap(x,y,z));

	int index=0;
	char idd[500];
	real value;
	int dofs;
	while (outputData(*this, index++, 0, idd, offset.x(),offset.y(),offset.z(), value, dofs))
	{
		// insert description
		if (dofs==1)
		{
			fprintf(fp,"SCALARS %s float 1\n",idd);
			fprintf(fp,"LOOKUP_TABLE default\n");
		} else
			fprintf(fp,"VECTORS %s float\n",idd);

		for (idx z = offset.z(); z < offset.z() + local.z(); z++)
		for (idx x = offset.x(); x < offset.x() + local.x(); x++)
		{
			for (int dof=0;dof<dofs;dof++)
			{
				outputData(*this, index-1,dof,idd,x,y,z,value,dofs);
				vtk.writeFloat(fp, value);
			}
		}
		vtk.writeBuffer(fp);
	}

	fclose(fp);
}

template< typename CONFIG >
	template< typename Output >
void LBM_BLOCK<CONFIG>::writeVTK_2DcutZ(lat_t lat, Output&& outputData, const std::string& filename, real time, int cycle, idx ZPOS) const
{
	if (!isLocalZ(ZPOS)) return;

	VTKWriter vtk;

	FILE* fp = fopen(filename.c_str(), "w+");
	vtk.writeHeader(fp);
	fprintf(fp,"DATASET RECTILINEAR_GRID\n");
	fprintf(fp,"DIMENSIONS %d %d %d\n", (int)local.x(), (int)local.y(), 1);
	fprintf(fp,"X_COORDINATES %d float\n", (int)local.x());
	for (idx x = offset.x(); x < offset.x() + local.x(); x++)
		vtk.writeFloat(fp, lat.lbm2physX(x));
	vtk.writeBuffer(fp);

	fprintf(fp,"Y_COORDINATES %d float\n", (int)local.y());
	for (idx y = offset.y(); y < offset.y() + local.y(); y++)
		vtk.writeFloat(fp, lat.lbm2physY(y));
	vtk.writeBuffer(fp);

	fprintf(fp,"Z_COORDINATES %d float\n", 1);
	vtk.writeFloat(fp, lat.lbm2physZ(ZPOS));
	vtk.writeBuffer(fp);

	fprintf(fp,"FIELD FieldData %d\n",2);
	fprintf(fp,"TIME %d %d float\n",1,1);
	vtk.writeFloat(fp, time);
	vtk.writeBuffer(fp);

	fprintf(fp,"CYCLE %d %d float\n",1,1);
	vtk.writeFloat(fp, cycle);
	vtk.writeBuffer(fp);

	fprintf(fp,"POINT_DATA %d\n", (int)(1*local.x()*local.y()));

	fprintf(fp,"SCALARS wall int 1\n");
	fprintf(fp,"LOOKUP_TABLE default\n");
	idx z=ZPOS;
	for (idx y = offset.y(); y < offset.y() + local.y(); y++)
	for (idx x = offset.x(); x < offset.x() + local.x(); x++)
		vtk.writeInt(fp, hmap(x,y,z));

	int index=0;
	char idd[500];
	real value;
	int dofs;
	while (outputData(*this, index++, 0, idd, offset.x(),offset.y(),offset.z(), value, dofs))
	{
		// insert description
		if (dofs==1)
		{
			fprintf(fp,"SCALARS %s float 1\n",idd);
			fprintf(fp,"LOOKUP_TABLE default\n");
		} else
			fprintf(fp,"VECTORS %s float\n",idd);

		for (idx y = offset.y(); y < offset.y() + local.y(); y++)
		for (idx x = offset.x(); x < offset.x() + local.x(); x++)
		{
			for (int dof=0;dof<dofs;dof++)
			{
				outputData(*this, index-1,dof,idd,x,y,z,value,dofs);
				vtk.writeFloat(fp, value);
			}
		}
		vtk.writeBuffer(fp);
	}

	fclose(fp);
}
