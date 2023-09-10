#include "state.h"

template < typename NSE, typename ADE >
struct State_NSE_ADE : State<NSE>
{
	// using different TRAITS is not implemented (probably does not make sense...)
	static_assert(std::is_same<typename NSE::TRAITS, typename ADE::TRAITS>::value,
			"TRAITS must be the same type in NSE and ADE.");
	using TRAITS = typename NSE::TRAITS;
	using BLOCK_NSE = LBM_BLOCK< NSE >;
	using BLOCK_ADE = LBM_BLOCK< ADE >;

	using State<NSE>::nse;
	using State<NSE>::cnt;
	using State<NSE>::vtk_helper;
	using State<NSE>::log;

	using idx = typename TRAITS::idx;
	using real = typename TRAITS::real;
	using lat_t = Lattice<3, real, idx>;

	LBM<ADE> ade;

	// constructor
	State_NSE_ADE(const TNL::MPI::Comm& communicator, lat_t ilat, real iphysViscosity, real iphysDt, real iphysDiffusion)
		: State<NSE>(communicator, ilat, iphysViscosity, iphysDt),
		  ade(communicator, ilat, iphysDiffusion, iphysDt)
	{
		// ADE allocation
		ade.allocateHostData();
	}

	// TODO: override estimateMemoryDemands

	void reset() override
	{
		nse.resetMap(NSE::BC::GEO_FLUID);
		ade.resetMap(ADE::BC::GEO_FLUID);
		this->setupBoundaries();

		// reset lattice for NSE and ADE
		// NOTE: it is important to reset *all* lattice sites (i.e. including ghost layers) when using the A-A pattern
		// (because GEO_INFLOW and GEO_OUTFLOW_EQ access the ghost layer in streaming)
		nse.forAllLatticeSites( [&] (BLOCK_NSE& block, idx x, idx y, idx z) {
			block.setEqLat(x,y,z,1,0,0,0);//rho,vx,vy,vz);
		} );
		ade.forAllLatticeSites( [&] (BLOCK_ADE& block, idx x, idx y, idx z) {
			block.setEqLat(x,y,z,0,0,0,0);//phi,vx,vy,vz);
		} );
	}

	void SimInit() override
	{
		log("MPI info: rank={:d}, nproc={:d}, lat.global=[{:d},{:d},{:d}]", nse.rank, nse.nproc, nse.lat.global.x(), nse.lat.global.y(), nse.lat.global.z());
		for (auto& block : nse.blocks)
			log("LBM block {:d}: local=[{:d},{:d},{:d}], offset=[{:d},{:d},{:d}]", block.id, block.local.x(), block.local.y(), block.local.z(), block.offset.x(), block.offset.y(), block.offset.z());

		log("\nSTART: simulation NSE:{}-ADE:{} lbmViscosity {:e} lbmDiffusion {:e} physDl {:e} physDt {:e}", NSE::COLL::id, ADE::COLL::id, nse.lbmViscosity(), ade.lbmViscosity(), nse.lat.physDl, nse.physDt);

		// reset counters
		for (int c=0;c<MAX_COUNTER;c++) cnt[c].count = 0;
		cnt[SAVESTATE].count = 1;  // skip initial save of state
		nse.iterations = 0;

		// setup map and DFs in CPU memory
		reset();

		// init NSE
		for (auto& block : nse.blocks)
		{
			// create LBM_DATA with host pointers
			typename NSE::DATA SD;
			for (uint8_t dfty=0;dfty<DFMAX;dfty++)
				SD.dfs[dfty] = block.hfs[dfty].getData();
			#ifdef HAVE_MPI
			SD.indexer = block.hmap.getLocalView().getIndexer();
			#else
			SD.indexer = block.hmap.getIndexer();
			#endif
			SD.XYZ = SD.indexer.getStorageSize();
			SD.dmap = block.hmap.getData();
			SD.dmacro = block.hmacro.getData();

			// initialize macroscopic quantities on CPU
			#pragma omp parallel for schedule(static) collapse(2)
			for (idx x = 0; x < block.local.x(); x++)
			for (idx z = 0; z < block.local.z(); z++)
			for (idx y = 0; y < block.local.y(); y++)
				LBMKernelInit<NSE>(SD, x, y, z);
		}

		// init ADE
		for (auto& block : ade.blocks)
		{
			// create LBM_DATA with host pointers
			typename ADE::DATA SD;
			for (uint8_t dfty=0;dfty<DFMAX;dfty++)
				SD.dfs[dfty] = block.hfs[dfty].getData();
			#ifdef HAVE_MPI
			SD.indexer = block.hmap.getLocalView().getIndexer();
			#else
			SD.indexer = block.hmap.getIndexer();
			#endif
			SD.XYZ = SD.indexer.getStorageSize();
			SD.dmap = block.hmap.getData();
			SD.dmacro = block.hmacro.getData();

			// initialize macroscopic quantities on CPU
			#pragma omp parallel for schedule(static) collapse(2)
			for (idx x = 0; x < block.local.x(); x++)
			for (idx z = 0; z < block.local.z(); z++)
			for (idx y = 0; y < block.local.y(); y++)
				LBMKernelInit<ADE>(SD, x, y, z);
		}

		nse.allocateDeviceData();
		ade.allocateDeviceData();
		this->copyAllToDevice();

#ifdef HAVE_MPI
		// synchronize overlaps with MPI (initial synchronization can be synchronous)
		nse.synchronizeMapDevice();
		nse.synchronizeDFsAndMacroDevice(df_cur);

		ade.synchronizeMapDevice();
		ade.synchronizeDFsAndMacroDevice(df_cur);
#endif
	}

	void updateKernelData() override
	{
		// general update (even_iter, dfs pointer)
		nse.updateKernelData();
		ade.updateKernelData();

		// update LBM viscosity/diffusivity
		for( auto& block : nse.blocks )
			block.data.lbmViscosity = nse.lbmViscosity();
		for( auto& block : ade.blocks )
			block.data.lbmViscosity = ade.lbmViscosity();
	}

	void SimUpdate() override
	{
		// debug
		for (auto& block : nse.blocks)
		if (block.data.lbmViscosity == 0) {
			log("error: NSE viscosity is 0");
			nse.terminate = true;
			return;
		}
		for (auto& block : ade.blocks)
		if (block.data.lbmViscosity == 0) {
			log("error: ADE diffusion is 0");
			nse.terminate = true;
			return;
		}

		#ifdef USE_CUDA
		auto get_grid_size = [] (const auto& block, idx x = 0, idx y = 0, idx z = 0) -> dim3
		{
			dim3 gridSize;
			if (x > 0)
				gridSize.x = x;
			else
				gridSize.x = TNL::roundUpDivision(block.local.x(), block.block_size.x());
			if (y > 0)
				gridSize.y = y;
			else
				gridSize.y = TNL::roundUpDivision(block.local.y(), block.block_size.y());
			if (z > 0)
				gridSize.z = z;
			else
				gridSize.z = TNL::roundUpDivision(block.local.z(), block.block_size.z());

			return gridSize;
		};
		#endif


		// call hook method (used e.g. for extra kernels in the non-Newtonian model)
		this->computeBeforeLBMKernel();


		#ifdef AA_PATTERN
		uint8_t output_df = df_cur;
		#endif
		#ifdef AB_PATTERN
		uint8_t output_df = df_out;
		#endif

		if (nse.blocks.size() != ade.blocks.size())
			throw std::logic_error("vectors of nse.blocks and ade.blocks must have equal sizes");

#ifdef USE_CUDA
		#ifdef HAVE_MPI
		if (nse.nproc == 1)
		{
		#endif
			for (std::size_t b = 0; b < nse.blocks.size(); b++)
			{
				auto& block_nse = nse.blocks[b];
				auto& block_ade = ade.blocks[b];
				// TODO: check that block_nse and block_ade have the same sizes

				const dim3 blockSize = {unsigned(block_nse.block_size.x()), unsigned(block_nse.block_size.y()), unsigned(block_nse.block_size.z())};
				const dim3 gridSize = get_grid_size(block_nse);
				cudaLBMKernel< NSE, ADE ><<<gridSize, blockSize>>>(block_nse.data, block_ade.data, nse.rank, nse.nproc, (idx) 0);
			}
			cudaDeviceSynchronize();
			checkCudaDevice;
			// copying of overlaps is not necessary for nproc == 1 (nproc is checked in streaming as well)
		#ifdef HAVE_MPI
		}
		else
		{
			for (std::size_t b = 0; b < nse.blocks.size(); b++)
			{
				auto& block_nse = nse.blocks[b];
				auto& block_ade = ade.blocks[b];
				// TODO: check that block_nse and block_ade have the same sizes

				const dim3 blockSize = {unsigned(block_nse.block_size.x()), unsigned(block_nse.block_size.y()), unsigned(block_nse.block_size.z())};
				const dim3 gridSizeForBoundary = get_grid_size(block_nse, block_nse.df_overlap_X());
				const dim3 gridSizeForInternal = get_grid_size(block_nse, block_nse.local.x() - 2*block_nse.df_overlap_X());

				// get CUDA streams
				const cudaStream_t cuda_stream_left = block_nse.streams.at(block_nse.left_id);
				const cudaStream_t cuda_stream_right = block_nse.streams.at(block_nse.right_id);
				const cudaStream_t cuda_stream_main = block_nse.streams.at(block_nse.id);

				// compute on boundaries (NOTE: 1D distribution is assumed)
				cudaLBMKernel< NSE, ADE ><<<gridSizeForBoundary, blockSize, 0, cuda_stream_left>>>(block_nse.data, block_ade.data, block_nse.id, nse.total_blocks, (idx) 0);
				cudaLBMKernel< NSE, ADE ><<<gridSizeForBoundary, blockSize, 0, cuda_stream_right>>>(block_nse.data, block_ade.data, block_nse.id, nse.total_blocks, block_nse.local.x() - block_nse.df_overlap_X());

				// compute on internal lattice sites
				cudaLBMKernel< NSE, ADE ><<<gridSizeForInternal, blockSize, 0, cuda_stream_main>>>(block_nse.data, block_ade.data, block_nse.id, nse.total_blocks, block_nse.df_overlap_X());
			}

			// wait for the computations on boundaries to finish
			for (auto& block : nse.blocks)
			{
				const cudaStream_t cuda_stream_left = block.streams.at(block.left_id);
				const cudaStream_t cuda_stream_right = block.streams.at(block.right_id);

				cudaStreamSynchronize(cuda_stream_left);
				cudaStreamSynchronize(cuda_stream_right);
			}

			// exchange the latest DFs and dmacro on overlaps between blocks
			// (it is important to wait for the communication before waiting for the computation, otherwise MPI won't progress)
			synchronizeDFsAndMacroDevice(output_df);

			// wait for the computation on the interior to finish
			for (auto& block : nse.blocks)
			{
				const cudaStream_t cuda_stream_main = block.streams.at(block.id);
				cudaStreamSynchronize(cuda_stream_main);
			}

			// synchronize the whole GPU and check errors
			cudaDeviceSynchronize();
			checkCudaDevice;
		}
		#endif
#else
		for (std::size_t b = 0; b < nse.blocks.size(); b++)
		{
			auto& block_nse = nse.blocks[b];
			auto& block_ade = ade.blocks[b];
			// TODO: check that block_nse and block_ade have the same sizes

//			#pragma omp parallel for schedule(static) collapse(2)
			for (idx x=0; x<block_nse.local.x(); x++)
			for (idx z=0; z<block_nse.local.z(); z++)
			for (idx y=0; y<block_nse.local.y(); y++)
			{
				LBMKernel< NSE, ADE >(block_nse.data, block_ade.data, x, y, z, nse.rank, nse.nproc);
			}
		}
		#ifdef HAVE_MPI
		// TODO: overlap computation with synchronization, just like above
		nse.synchronizeDFsDevice(output_df);
		ade.synchronizeDFsDevice(output_df);
		#endif
#endif

		nse.iterations++;
		ade.iterations = nse.iterations;

		bool doit=false;
		for (int c=0;c<MAX_COUNTER;c++) if (c!=PRINT && c!=SAVESTATE) if (cnt[c].action(nse.physTime())) doit = true;
		if (doit)
		{
			// common copy
			nse.copyMacroToHost();
			ade.copyMacroToHost();
			// to be able to compute rho, vx, vy, vz etc... based on DFs on CPU to save GPU memory FIXME may not work with ESOTWIST
			if (NSE::CPU_MACRO::N>0)
				nse.copyDFsToHost(output_df);
			if (ADE::CPU_MACRO::N>0)
				ade.copyDFsToHost(output_df);
			#ifdef USE_CUDA
			checkCudaDevice;
			#endif
		}
	}

	void AfterSimUpdate() override
	{
		State< NSE >::AfterSimUpdate();
		// TODO: figure out what should be done for ade here...
	}

	// called from SimInit -- copy the initial state to the GPU
	void copyAllToDevice() override
	{
		nse.copyMapToDevice();
		nse.copyDFsToDevice();
		nse.copyMacroToDevice();
		ade.copyMapToDevice();
		ade.copyDFsToDevice();
		ade.copyMacroToDevice();
	}

	// called from core.h -- inside the time loop before saving state
	void copyAllToHost() override
	{
		nse.copyMapToHost();
		nse.copyDFsToHost();
		nse.copyMacroToHost();
		ade.copyMapToHost();
		ade.copyDFsToHost();
		ade.copyMacroToHost();
	}

	void writeVTKs_3D() override
	{
		const std::string dir = fmt::format("results_{}/vtk3D", this->id);
		mkdir_p(dir.c_str(), 0755);

		for (const auto& block : nse.blocks)
		{
			const std::string basename = fmt::format("NSE_block{:03d}_{:d}.vtk", block.id, cnt[VTK3D].count);
			const std::string filename = fmt::format("{}/{}", dir, basename);
			auto outputData = [this] (const BLOCK_NSE& block, int index, int dof, char* desc, idx x, idx y, idx z, real& value, int& dofs) mutable
			{
				return this->outputData(block, index, dof, desc, x, y, z, value, dofs);
			};
			block.writeVTK_3D(nse.lat, outputData, filename, nse.physTime(), cnt[VTK3D].count);
			this->log("[vtk {} written, time {:f}, cycle {:d}] ", filename, nse.physTime(), cnt[VTK3D].count);
		}
		for (const auto& block : ade.blocks)
		{
			const std::string basename = fmt::format("ADE_block{:03d}_{:d}.vtk", block.id, cnt[VTK3D].count);
			const std::string filename = fmt::format("{}/{}", dir, basename);
			auto outputData = [this] (const BLOCK_ADE& block, int index, int dof, char* desc, idx x, idx y, idx z, real& value, int& dofs) mutable
			{
				return this->outputData(block, index, dof, desc, x, y, z, value, dofs);
			};
			block.writeVTK_3D(ade.lat, outputData, filename, nse.physTime(), cnt[VTK3D].count);
			this->log("[vtk {} written, time {:f}, cycle {:d}] ", filename, nse.physTime(), cnt[VTK3D].count);
		}
	}

	void writeVTKs_3Dcut() override
	{
		if (this->probe3Dvec.size()<=0) return;
		// browse all 3D vtk cuts
		for (auto& probevec : this->probe3Dvec)
		{
			for (const auto& block : nse.blocks)
			{
				const std::string fname = fmt::format("results_{}/vtk3Dcut/{}_NSE_block{:03d}_{:d}.vtk", this->id, probevec.name, block.id, probevec.cycle);
				// create parent directories
				create_file(fname.c_str());
				auto outputData = [this] (const BLOCK_NSE& block, int index, int dof, char* desc, idx x, idx y, idx z, real& value, int& dofs) mutable
				{
					return this->outputData(block, index, dof, desc, x, y, z, value, dofs);
				};
				block.writeVTK_3Dcut(
					nse.lat,
					outputData,
					fname,
					nse.physTime(),
					probevec.cycle,
					probevec.ox,
					probevec.oy,
					probevec.oz,
					probevec.lx,
					probevec.ly,
					probevec.lz,
					probevec.step
				);
				this->log("[vtk {} written, time {:f}, cycle {:d}] ", fname, nse.physTime(), probevec.cycle);
			}
			for (const auto& block : ade.blocks)
			{
				const std::string fname = fmt::format("results_{}/vtk3Dcut/{}_ADE_block{:03d}_{:d}.vtk", this->id, probevec.name, block.id, probevec.cycle);
				// create parent directories
				create_file(fname.c_str());
				auto outputData = [this] (const BLOCK_ADE& block, int index, int dof, char* desc, idx x, idx y, idx z, real& value, int& dofs) mutable
				{
					return this->outputData(block, index, dof, desc, x, y, z, value, dofs);
				};
				block.writeVTK_3Dcut(
					ade.lat,
					outputData,
					fname,
					nse.physTime(),
					probevec.cycle,
					probevec.ox,
					probevec.oy,
					probevec.oz,
					probevec.lx,
					probevec.ly,
					probevec.lz,
					probevec.step
				);
				this->log("[vtk {} written, time {:f}, cycle {:d}] ", fname, nse.physTime(), probevec.cycle);
			}
			probevec.cycle++;
		}
	}

	void writeVTKs_2D() override
	{
		if (this->probe2Dvec.size()<=0) return;
		// browse all 2D vtk cuts
		for (auto& probevec : this->probe2Dvec)
		{
			for (const auto& block : nse.blocks)
			{
				const std::string fname = fmt::format("results_{}/vtk2D/{}_NSE_block{:03d}_{:d}.vtk", this->id, probevec.name, block.id, probevec.cycle);
				// create parent directories
				create_file(fname.c_str());
				auto outputData = [this] (const BLOCK_NSE& block, int index, int dof, char* desc, idx x, idx y, idx z, real& value, int& dofs) mutable
				{
					return this->outputData(block, index, dof, desc, x, y, z, value, dofs);
				};
				switch (probevec.type)
				{
					case 0: block.writeVTK_2DcutX(nse.lat, outputData, fname, nse.physTime(), probevec.cycle, probevec.position);
						break;
					case 1: block.writeVTK_2DcutY(nse.lat, outputData, fname, nse.physTime(), probevec.cycle, probevec.position);
						break;
					case 2: block.writeVTK_2DcutZ(nse.lat, outputData, fname, nse.physTime(), probevec.cycle, probevec.position);
						break;
				}
				this->log("[vtk {} written, time {:f}, cycle {:d}] ", fname, nse.physTime(), probevec.cycle);
			}
			for (const auto& block : ade.blocks)
			{
				const std::string fname = fmt::format("results_{}/vtk2D/{}_ADE_block{:03d}_{:d}.vtk", this->id, probevec.name, block.id, probevec.cycle);
				// create parent directories
				create_file(fname.c_str());
				auto outputData = [this] (const BLOCK_ADE& block, int index, int dof, char* desc, idx x, idx y, idx z, real& value, int& dofs) mutable
				{
					return this->outputData(block, index, dof, desc, x, y, z, value, dofs);
				};
				switch (probevec.type)
				{
					case 0: block.writeVTK_2DcutX(nse.lat, outputData, fname, nse.physTime(), probevec.cycle, probevec.position);
						break;
					case 1: block.writeVTK_2DcutY(nse.lat, outputData, fname, nse.physTime(), probevec.cycle, probevec.position);
						break;
					case 2: block.writeVTK_2DcutZ(nse.lat, outputData, fname, nse.physTime(), probevec.cycle, probevec.position);
						break;
				}
				this->log("[vtk {} written, time {:f}, cycle {:d}] ", fname, nse.physTime(), probevec.cycle);
			}
			probevec.cycle++;
		}
	}

	bool outputData(const BLOCK_NSE& block, int index, int dof, char *desc, idx x, idx y, idx z, real &value, int &dofs) override { return false; }
	virtual bool outputData(const BLOCK_ADE& block, int index, int dof, char *desc, idx x, idx y, idx z, real &value, int &dofs) { return false; }

// TODO: ensure that NSE and ADE synchronizers use different MPI tags for communication
private:
#ifdef HAVE_MPI
	void synchronizeDFsAndMacroDevice(uint8_t dftype)
	{
		// stage 0: set inputs, allocate buffers
		// stage 1: fill send buffers
		for( auto& block : nse.blocks ) {
			block.synchronizeDFsDevice_start(dftype);
			if (NSE::MACRO::use_syncMacro)
				block.synchronizeMacroDevice_start();
		}
		for( auto& block : ade.blocks ) {
			block.synchronizeDFsDevice_start(dftype);
			if (ADE::MACRO::use_syncMacro)
				block.synchronizeMacroDevice_start();
		}

		// stage 2: issue all send and receive async operations
		for( auto& block : nse.blocks ) {
			for (int i = 0; i < NSE::Q; i++)
				block.dreal_sync[i].stage_2();
			if (NSE::MACRO::use_syncMacro)
				for (int i = 0; i < NSE::MACRO::N; i++)
					block.dreal_sync[NSE::Q + i].stage_2();
		}
		for( auto& block : ade.blocks ) {
			for (int i = 0; i < ADE::Q; i++)
				block.dreal_sync[i].stage_2();
			if (ADE::MACRO::use_syncMacro)
				for (int i = 0; i < ADE::MACRO::N; i++)
					block.dreal_sync[ADE::Q + i].stage_2();
		}

		// stage 3: copy data from receive buffers
		for( auto& block : nse.blocks ) {
			for (int i = 0; i < NSE::Q; i++)
				block.dreal_sync[i].stage_3();
			if (NSE::MACRO::use_syncMacro)
				for (int i = 0; i < NSE::MACRO::N; i++)
					block.dreal_sync[NSE::Q + i].stage_3();
		}
		for( auto& block : ade.blocks ) {
			for (int i = 0; i < ADE::Q; i++)
				block.dreal_sync[i].stage_3();
			if (ADE::MACRO::use_syncMacro)
				for (int i = 0; i < ADE::MACRO::N; i++)
					block.dreal_sync[ADE::Q + i].stage_3();
		}

		// stage 4: ensure everything has finished
		for( auto& block : nse.blocks ) {
			for (int i = 0; i < NSE::Q; i++)
				block.dreal_sync[i].stage_4();
			if (NSE::MACRO::use_syncMacro)
				for (int i = 0; i < NSE::MACRO::N; i++)
					block.dreal_sync[NSE::Q + i].stage_4();
		}
		for( auto& block : ade.blocks ) {
			for (int i = 0; i < ADE::Q; i++)
				block.dreal_sync[i].stage_4();
			if (ADE::MACRO::use_syncMacro)
				for (int i = 0; i < ADE::MACRO::N; i++)
					block.dreal_sync[ADE::Q + i].stage_4();
		}
	}
#endif
};
