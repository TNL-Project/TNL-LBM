#pragma once

#include "lbm_block.h"
#include "vtk_writer.h"
#include "block_size_optimizer.h"

template< typename CONFIG >
LBM_BLOCK<CONFIG>::LBM_BLOCK(const TNL::MPI::Comm& communicator, idx3d global, idx3d local, idx3d offset, int neighbour_left, int neighbour_right, int left_id, int this_id, int right_id)
: communicator(communicator), global(global), local(local), offset(offset), left_id(left_id), id(this_id), right_id(right_id)
{
	// initialize MPI info
	rank = communicator.rank();
	nproc = communicator.size();

	// initialize neighbours
	if (neighbour_left < 0)
		this->neighbour_left = (rank + nproc - 1) % nproc;
	else
		this->neighbour_left = neighbour_left;
	if (neighbour_right < 0)
		this->neighbour_right = (rank + 1) % nproc;
	else
		this->neighbour_right = neighbour_right;

#ifdef USE_CUDA
	// initialize optimal thread block size for the LBM kernel
	constexpr int max_threads = 256 / (sizeof(dreal) / sizeof(float));  // use 256 threads for SP and 128 threads for DP
	block_size = get_optimal_block_size< typename TRAITS::xyz_permutation >(local, max_threads);
#endif
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
	cudaMemcpy(dfx(), hfx(), local.x()*local.y()*local.z()*sizeof(dreal), cudaMemcpyHostToDevice);
	cudaMemcpy(dfy(), hfy(), local.x()*local.y()*local.z()*sizeof(dreal), cudaMemcpyHostToDevice);
	cudaMemcpy(dfz(), hfz(), local.x()*local.y()*local.z()*sizeof(dreal), cudaMemcpyHostToDevice);
	checkCudaDevice;
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
void LBM_BLOCK<CONFIG>::setBoundaryTransfer()
{
			std::cout << "Hey" << std::endl;
	
	TransferFS.setSizes(global.x(), global.y(), global.z());
	TransferSF.setSizes(global.x(), global.y(), global.z());
	TransferSW.setSizes(global.x(), global.y(), global.z());
			std::cout << "He" << std::endl;

	#ifdef HAVE_MPI
	TransferFS.template setDistribution< 0 >(offset.x(), offset.x() + local.x(), communicator);
	TransferFS.allocate();
	TransferSF.template setDistribution< 0 >(offset.x(), offset.x() + local.x(), communicator);
	TransferSF.allocate();
	TransferSW.template setDistribution< 0 >(offset.x(), offset.x() + local.x(), communicator);
	TransferSW.allocate();
	#endif
		std::cout << "Heyda" << std::endl;

	for(idx x = offset.x(); x < offset.x() + local.x(); x++)
	for(idx y = offset.y(); y < offset.y() + local.y(); y++)
	for(idx z = offset.z(); z < offset.z() + local.z(); z++)
		if (isLocalIndex(x, y, z)) {
			TransferFS(x, y, z) = false;
			TransferSF(x, y, z) = false;
			TransferSW(x, y, z) = false;
		}
	for(idx x = offset.x(); x < offset.x() + local.x(); x++){
		for(idx y = offset.y(); y < offset.y() + local.y(); y++){
			for(idx z = offset.z(); z < offset.z() + local.z(); z++){
				if(isLocalIndex(x, y, z)){
					if(isFluid(x,y,z)){
						if(isSolid(x+1,y,z)){
							TransferFS(x, y, z) = true;
							transferDIR(pzz,x,y,z) = true;
						}
						if(isSolid(x,y+1,z)){
							TransferFS(x, y, z) = true;
							transferDIR(zpz,x,y,z) = true;
						}
						if(isSolid(x,y,z+1)){
							TransferFS(x, y, z) = true;
							transferDIR(zzp,x,y,z) = true;
						}
						if(isSolid(x-1,y,z)){
							TransferFS(x, y, z) = true;
							transferDIR(mzz,x,y,z) = true;
						}
						if(isSolid(x,y-1,z)){
							TransferFS(x, y, z) = true;
							transferDIR(zmz,x,y,z) = true;
						}
						if(isSolid(x,y,z-1)){
							TransferFS(x, y, z) = true;
							transferDIR(zzm,x,y,z) = true;
						}
					}
					if(isSolid(x,y,z)){
						if(isFluid(x+1,y,z)){
							TransferSF(x, y, z) = true;
							transferDIR(pzz,x,y,z) = true;
						}
						if(isFluid(x,y+1,z)){
							TransferSF(x, y, z) = true;
							transferDIR(zpz,x,y,z) = true;
						}
						if(isFluid(x,y,z+1)){
							TransferSF(x, y, z) = true;
							transferDIR(zzp,x,y,z) = true;
						}
						if(isFluid(x-1,y,z)){
							TransferSF(x, y, z) = true;
							transferDIR(mzz,x,y,z) = true;
						}
						if(isFluid(x,y-1,z)){
							TransferSF(x, y, z) = true;
							transferDIR(zmz,x,y,z) = true;
						}
						if(isFluid(x,y,z-1)){
							TransferSF(x, y, z) = true;
							transferDIR(zzm,x,y,z) = true;
						}

						if(isWall(x+1,y,z)){
							TransferSW(x, y, z) = true;
							transferDIR(pzz,x,y,z) = true;
						}
						if(isWall(x,y+1,z)){
							TransferSW(x, y, z) = true;
							transferDIR(zpz,x,y,z) = true;
						}
						if(isWall(x,y,z+1)){
							TransferSW(x, y, z) = true;
							transferDIR(zzp,x,y,z) = true;
						}
						if(isWall(x-1,y,z)){
							TransferSW(x, y, z) = true;
							transferDIR(mzz,x,y,z) = true;
						}
						if(isWall(x,y-1,z)){
							TransferSW(x, y, z) = true;
							transferDIR(zmz,x,y,z) = true;
						}
						if(isWall(x,y,z-1)){
							TransferSW(x, y, z) = true;
							transferDIR(zzm,x,y,z) = true;
						}
					}
				}
			}
		}
	}
	
	for(idx x = offset.x(); x < offset.x() + local.x(); x++){
		for(idx y = offset.y(); y < offset.y() + local.y(); y++){
			for(idx z = offset.z(); z < offset.z() + local.z(); z++){
				if(TransferFS(x,y,z))
					setMap(x,y,z, CONFIG::BC::GEO_TRANSFER_FS);
				if(TransferSF(x,y,z))	
					setMap(x,y,z, CONFIG::BC::GEO_TRANSFER_SF);
				if(TransferSW(x,y,z))
					setMap(x,y,z, CONFIG::BC::GEO_TRANSFER_SW);
			}
		}
	}		
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
	std::cout << "Heyda and y value: " << y << " and wall: " <<  value << std::endl;
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
	return CONFIG::BC::isFluid(map(x,y,z));
}

template< typename CONFIG >
bool LBM_BLOCK<CONFIG>::isWall(idx x, idx y, idx z) const
{
	if (!isLocalIndex(x, y, z)) return false;
	return CONFIG::BC::isWall(map(x,y,z));
}

template< typename CONFIG >
bool LBM_BLOCK<CONFIG>::isSolid(idx x, idx y, idx z) const
{
	if (!isLocalIndex(x, y, z)) return false;
	return CONFIG::BC::isSolid(map(x,y,z));
}


template< typename CONFIG >
bool LBM_BLOCK<CONFIG>::isSolidPhase(idx x, idx y, idx z) const
{
	if (!isLocalIndex(x, y, z)) return false;
	return CONFIG::BC::isSolidPhase(map(x,y,z));
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
	hdifmap = difmap;
	transferDIR = dtransferDIR;	
}

template< typename CONFIG >
void  LBM_BLOCK<CONFIG>::copyMapToDevice()
{
	dmap = hmap;
	difmap = hdifmap;
	dtransferDIR = transferDIR;
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
	#ifdef HAVE_MPI
	typename sync_array_t::LocalViewType localView(nullptr, data.indexer);
	typename sync_array_t::ViewType view(localView, dmap.getSizes(), dmap.getLocalBegins(), dmap.getLocalEnds(), dmap.getCommunicator());
	#else
	typename sync_array_t::ViewType view(nullptr, data.indexer);
	#endif

	for (int i = 0; i < N; i++) {
		// set neighbors (0 = x-direction)
		dreal_sync[i + sync_offset].template setNeighbors< 0 >( neighbour_left, neighbour_right );
		// TODO: make this a general parameter (for now we set an upper bound)
		constexpr int blocks_per_rank = 32;
		dreal_sync[i + sync_offset].setTags(
			left_id < 0 ? -1 :
				(2 * i + 1) * blocks_per_rank * nproc + left_id,   // from left
			left_id < 0 ? -1 :
				(2 * i + 0) * blocks_per_rank * nproc + id,        // to left
			right_id < 0 ? -1 :
				(2 * i + 0) * blocks_per_rank * nproc + right_id,  // from right
			right_id < 0 ? -1 :
				(2 * i + 1) * blocks_per_rank * nproc + id );      // to right
		// rebind just the data pointer
		view.bind(array.getData() + i * data.indexer.getStorageSize());
		// determine sync direction
		TNL::Containers::SyncDirection sync_direction = (is_df) ? df_sync_directions[i] : TNL::Containers::SyncDirection::All;
		#ifdef AA_PATTERN
		// reset shift of the lattice sites
		dreal_sync[i + sync_offset].template setBuffersShift<0>(0);
		if (is_df) {
			if (data.even_iter) {
				// lattice sites for synchronization are not shifted, but DFs have opposite directions
				if (sync_direction == TNL::Containers::SyncDirection::Right)
					sync_direction = TNL::Containers::SyncDirection::Left;
				else if (sync_direction == TNL::Containers::SyncDirection::Left)
					sync_direction = TNL::Containers::SyncDirection::Right;
			}
			else {
				// DFs have canonical directions, but lattice sites for synchronization are shifted
				// (values to be synchronized were written to the neighboring sites)
				dreal_sync[i + sync_offset].template setBuffersShift<0>(1);
			}
		}
		#endif
		#ifdef USE_CUDA
		// lazy creation of CUDA streams
		if (streams.empty()) {
			// get the range of stream priorities for current GPU
			int priority_high, priority_low;
			cudaDeviceGetStreamPriorityRange(&priority_low, &priority_high);
			// low-priority stream for the interior
			streams.emplace( id, TNL::Cuda::Stream::create(cudaStreamNonBlocking, priority_low) );
			// high-priority streams for boundaries
			streams.emplace( left_id, TNL::Cuda::Stream::create(cudaStreamNonBlocking, priority_high) );
			streams.emplace( right_id, TNL::Cuda::Stream::create(cudaStreamNonBlocking, priority_high) );
		}
		// set the CUDA stream
		dreal_sync[i + sync_offset].setCudaStreams(streams.at(left_id), streams.at(right_id));
		#endif
		// start the synchronization
		// NOTE: we don't use synchronizeAsync because we need pipelining
		// NOTE: we could use only policy=deferred for synchronizeAsync, because  threadpool and async require MPI_THREAD_MULTIPLE which is slow
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

	// set neighbors (0 = x-direction)
	map_sync.template setNeighbors< 0 >( neighbour_left, neighbour_right );
	map_sync.setTags(
			left_id < 0 ? -1 : nproc + left_id,  // from left
			left_id < 0 ? -1 : id,               // to left
			right_id < 0 ? -1 : right_id,        // from right
			right_id < 0 ? -1 : nproc + id );    // to right
	map_sync.synchronizeAsync(dmap, policy);
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
		hfs[dfty].template setDistribution< 1 >(offset.x(), offset.x() + local.x(), communicator);
		hfs[dfty].allocate();
		#endif
	}

	hmap.setSizes(global.x(), global.y(), global.z());
	hdifmap.setSizes(global.x(), global.y(), global.z());
	transferDIR.setSizes(0, global.x(), global.y(), global.z());
#ifdef HAVE_MPI
	hmap.template setDistribution< 0 >(offset.x(), offset.x() + local.x(), communicator);
	hmap.allocate();
	hdifmap.template setDistribution< 0 >(offset.x(), offset.x() + local.x(), communicator);
	hdifmap.allocate();
	transferDIR.template setDistribution< 1 >(offset.x(), offset.x() + local.x(), communicator);
	transferDIR.allocate();
#endif

	hmacro.setSizes(0, global.x(), global.y(), global.z());
	cpumacro.setSizes(0, global.x(), global.y(), global.z());
#ifdef HAVE_MPI
	hmacro.template setDistribution< 1 >(offset.x(), offset.x() + local.x(), communicator);
	hmacro.allocate();
	cpumacro.template setDistribution< 1 >(offset.x(), offset.x() + local.x(), communicator);
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
	difmap.setSizes(global.x(), global.y(), global.z());
	dtransferDIR.setSizes(0, global.x(), global.y(), global.z());
	#ifdef HAVE_MPI
	dmap.template setDistribution< 0 >(offset.x(), offset.x() + local.x(), communicator);
	dmap.allocate();
	difmap.template setDistribution< 0 >(offset.x(), offset.x() + local.x(), communicator);
	difmap.allocate();
	dtransferDIR.template setDistribution< 1 >(offset.x(), offset.x() + local.x(), communicator);
	dtransferDIR.allocate();
	#endif

	for (uint8_t dfty=0;dfty<DFMAX;dfty++)
	{
		dfs[dfty].setSizes(0, global.x(), global.y(), global.z());
		#ifdef HAVE_MPI
		dfs[dfty].template setDistribution< 1 >(offset.x(), offset.x() + local.x(), communicator);
		dfs[dfty].allocate();
		#endif
	}

	dmacro.setSizes(0, global.x(), global.y(), global.z());
	#ifdef HAVE_MPI
	dmacro.template setDistribution< 1 >(offset.x(), offset.x() + local.x(), communicator);
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
	data.difMap = difmap.getData();
	data.dmacro = dmacro.getData();
	data.dtransferDIR = dtransferDIR.getData();
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
#ifdef HAVE_MPI
	const int overlap_x = hmap.getLocalView().getIndexer().template getOverlap< 0 >();
	const int overlap_y = hmap.getLocalView().getIndexer().template getOverlap< 1 >();
	const int overlap_z = hmap.getLocalView().getIndexer().template getOverlap< 2 >();
#else
	const int overlap_x = hmap.getIndexer().template getOverlap< 0 >();
	const int overlap_y = hmap.getIndexer().template getOverlap< 1 >();
	const int overlap_z = hmap.getIndexer().template getOverlap< 2 >();
#endif

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
