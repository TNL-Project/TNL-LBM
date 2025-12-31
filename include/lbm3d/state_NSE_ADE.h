#include "state.h"

template <typename NSE, typename ADE>
struct State_NSE_ADE : State<NSE>
{
	// using different TRAITS is not implemented (probably does not make sense...)
	static_assert(std::is_same<typename NSE::TRAITS, typename ADE::TRAITS>::value, "TRAITS must be the same type in NSE and ADE.");
	using TRAITS = typename NSE::TRAITS;
	using BLOCK_NSE = LBM_BLOCK<NSE>;
	using BLOCK_ADE = LBM_BLOCK<ADE>;

	using State<NSE>::id;
	using State<NSE>::dataManager;
	using State<NSE>::nse;
	using State<NSE>::cnt;

	using idx = typename TRAITS::idx;
	using idx3d = typename TRAITS::idx3d;
	using real = typename TRAITS::real;
	using dreal = typename TRAITS::dreal;
	using lat_t = Lattice<3, real, idx>;

	LBM<ADE> ade;

	// constructor
	State_NSE_ADE(
		const std::string& id, const TNL::MPI::Comm& communicator, lat_t lat_nse, lat_t lat_ade, const std::string& adiosConfigPath = "adios2.xml"
	)
	: State<NSE>(id, communicator, lat_nse, adiosConfigPath),
	  ade(communicator, lat_ade)
	{
		// ADE allocation
		ade.allocateHostData();
	}

	// TODO: override estimateMemoryDemands

	void reset() override
	{
		// compute initial DFs on GPU and copy to CPU
		nse.setEquilibrium(1, 0, 0, 0);	 // rho, vx, vy, vz
		ade.setEquilibrium(1, 0, 0, 0);	 // rho, vx, vy, vz

		nse.resetMap(NSE::BC::GEO_FLUID);
		ade.resetMap(ADE::BC::GEO_FLUID);

		// setup domain geometry after all resets, including setEquilibrium,
		// so it can override the defaults with different initial condition
		this->setupBoundaries();

		nse.copyMapToDevice();
		ade.copyMapToDevice();

		// compute initial macroscopic quantities on GPU and copy to CPU
		nse.computeInitialMacro();
		ade.computeInitialMacro();
		nse.copyMacroToHost();
		ade.copyMacroToHost();
	}

	// resetDFs is not used in NSE-ADE - make sure other subclasses don't accidentally override it
	void resetDFs() final {}

	void SimInit() override
	{
		spdlog::info(
			"MPI info: rank={:d}, nproc={:d}, lat.global=[{:d},{:d},{:d}]",
			nse.rank,
			nse.nproc,
			nse.lat.global.x(),
			nse.lat.global.y(),
			nse.lat.global.z()
		);
		for (auto& block : nse.blocks)
			spdlog::info(
				"LBM block {:d}: local=[{:d},{:d},{:d}], offset=[{:d},{:d},{:d}]",
				block.id,
				block.local.x(),
				block.local.y(),
				block.local.z(),
				block.offset.x(),
				block.offset.y(),
				block.offset.z()
			);

		spdlog::info(
			"\nSTART: simulation NSE:{}-ADE:{} lbmViscosity {:e} lbmDiffusion {:e} physDl {:e} physDt {:e}",
			NSE::COLL::id,
			ADE::COLL::id,
			nse.lat.lbmViscosity(),
			ade.lat.lbmViscosity(),
			nse.lat.physDl,
			nse.lat.physDt
		);

		// reset counters
		for (int c = 0; c < MAX_COUNTER; c++)
			cnt[c].count = 0;
		cnt[SAVESTATE].count = 1;  // skip initial save of state
		nse.iterations = 0;

		// allocate before reset - it might initialize on the GPU...
		nse.allocateDeviceData();
		ade.allocateDeviceData();

		// initialize map, DFs, and macro both in CPU and GPU memory
		reset();

#ifdef HAVE_MPI
		if (nse.nproc > 1) {
			// synchronize overlaps with MPI (initial synchronization can be synchronous)
			nse.synchronizeMapDevice();
			nse.synchronizeDFsAndMacroDevice(df_cur);

			ade.synchronizeMapDevice();
			ade.synchronizeDFsAndMacroDevice(df_cur);
		}
#endif

		spdlog::info("Finished SimInit");
	}

	void updateKernelData() override
	{
		// general update (even_iter, dfs pointer)
		nse.updateKernelData();
		ade.updateKernelData();

		// update LBM viscosity/diffusivity
		for (auto& block : nse.blocks)
			block.data.lbmViscosity = nse.lat.lbmViscosity();
		for (auto& block : ade.blocks)
			block.data.lbmViscosity = ade.lat.lbmViscosity();

		// update ADE-specific data
		for (auto& block : ade.blocks) {
			block.data.diffusion_coefficient_ptr = block.ddiffusionCoeff.getData();
			block.data.phi_transfer_direction_ptr = block.dphiTransferDirection.getData();
		}
	}

	void SimUpdate() override
	{
		// debug
		for (auto& block : nse.blocks)
			if (block.data.lbmViscosity == 0) {
				spdlog::error("error: NSE viscosity is 0");
				nse.terminate = true;
				return;
			}
		for (auto& block : ade.blocks)
			if (block.data.lbmViscosity == 0 && block.data.diffusion_coefficient_ptr == nullptr) {
				spdlog::error("error: ADE diffusion is 0");
				nse.terminate = true;
				return;
			}

		// call hook method (used e.g. for extra kernels in the non-Newtonian model)
		this->computeBeforeLBMKernel();

#ifdef HAVE_MPI
	#ifdef AA_PATTERN
		uint8_t output_df = df_cur;
	#endif
	#ifdef AB_PATTERN
		uint8_t output_df = df_out;
	#endif
#endif

		if (nse.blocks.size() != ade.blocks.size())
			throw std::logic_error("vectors of nse.blocks and ade.blocks must have equal sizes");

#ifdef USE_CUDA
	#ifdef HAVE_MPI
		if (nse.nproc == 1) {
	#endif
			for (std::size_t b = 0; b < nse.blocks.size(); b++) {
				auto& block_nse = nse.blocks[b];
				auto& block_ade = ade.blocks[b];
				// TODO: check that block_nse and block_ade have the same sizes

				const auto direction = TNL::Containers::SyncDirection::None;
				TNL::Backend::LaunchConfiguration launch_config;
				launch_config.blockSize = block_nse.computeData.at(direction).blockSize;
				launch_config.gridSize = block_nse.computeData.at(direction).gridSize;
				TNL::Backend::launchKernelAsync(
					cudaLBMKernel<NSE, ADE>,
					launch_config,
					block_nse.data,
					block_ade.data,
					idx3d{0, 0, 0},
					block_nse.local,
					block_nse.is_distributed()
				);
			}
			// synchronize the null-stream after all grids
			TNL::Backend::streamSynchronize(0);
			// copying of overlaps is not necessary for nproc == 1 (nproc is checked in streaming as well)
	#ifdef HAVE_MPI
		}
		else {
			const auto boundary_directions = {
				TNL::Containers::SyncDirection::Bottom,
				TNL::Containers::SyncDirection::Top,
				TNL::Containers::SyncDirection::Back,
				TNL::Containers::SyncDirection::Front,
				TNL::Containers::SyncDirection::Left,
				TNL::Containers::SyncDirection::Right,
			};

			// compute on boundaries
			for (std::size_t b = 0; b < nse.blocks.size(); b++) {
				auto& block_nse = nse.blocks[b];
				auto& block_ade = ade.blocks[b];
				// TODO: check that block_nse and block_ade have the same sizes

				for (auto direction : boundary_directions)
					if (auto search = block_nse.neighborIDs.find(direction); search != block_nse.neighborIDs.end() && search->second >= 0) {
						TNL::Backend::LaunchConfiguration launch_config;
						launch_config.blockSize = block_nse.computeData.at(direction).blockSize;
						launch_config.gridSize = block_nse.computeData.at(direction).gridSize;
						launch_config.stream = block_nse.computeData.at(direction).stream;
						const idx3d offset = block_nse.computeData.at(direction).offset;
						const idx3d size = block_nse.computeData.at(direction).size;
						TNL::Backend::launchKernelAsync(
							cudaLBMKernel<NSE, ADE>, launch_config, block_nse.data, block_ade.data, offset, offset + size, block_nse.is_distributed()
						);
					}
			}

			// compute on interior lattice sites
			for (std::size_t b = 0; b < nse.blocks.size(); b++) {
				auto& block_nse = nse.blocks[b];
				auto& block_ade = ade.blocks[b];
				const auto direction = TNL::Containers::SyncDirection::None;
				TNL::Backend::LaunchConfiguration launch_config;
				launch_config.blockSize = block_nse.computeData.at(direction).blockSize;
				launch_config.gridSize = block_nse.computeData.at(direction).gridSize;
				launch_config.stream = block_nse.computeData.at(direction).stream;
				const idx3d offset = block_nse.computeData.at(direction).offset;
				const idx3d size = block_nse.computeData.at(direction).size;
				TNL::Backend::launchKernelAsync(
					cudaLBMKernel<NSE, ADE>, launch_config, block_nse.data, block_ade.data, offset, offset + size, block_nse.is_distributed()
				);
			}

			// wait for the computations on boundaries to finish
			for (auto& block : nse.blocks)
				for (auto direction : boundary_directions)
					if (auto search = block.neighborIDs.find(direction); search != block.neighborIDs.end() && search->second >= 0) {
						const auto& stream = block.computeData.at(direction).stream;
						TNL::Backend::streamSynchronize(stream);
					}

			// exchange the latest DFs and dmacro on overlaps between blocks
			// (it is important to wait for the communication before waiting for the computation, otherwise MPI won't progress)
			// TODO: merge the pipelining of the communication in the NSE and ADE into one
			nse.synchronizeDFsAndMacroDevice(output_df);
			ade.synchronizeDFsAndMacroDevice(output_df);

			// wait for the computation on the interior to finish
			for (auto& block : nse.blocks) {
				const auto& stream = block.computeData.at(TNL::Containers::SyncDirection::None).stream;
				TNL::Backend::streamSynchronize(stream);
			}
		}
	#endif
#else
		for (std::size_t b = 0; b < nse.blocks.size(); b++) {
			auto& block_nse = nse.blocks[b];
			auto& block_ade = ade.blocks[b];
			// TODO: check that block_nse and block_ade have the same sizes

			//#pragma omp parallel for schedule(static) collapse(2)
			for (idx x = 0; x < block_nse.local.x(); x++)
				for (idx z = 0; z < block_nse.local.z(); z++)
					for (idx y = 0; y < block_nse.local.y(); y++) {
						LBMKernel<NSE, ADE>(block_nse.data, block_ade.data, x, y, z, block_nse.is_distributed());
					}
		}
	#ifdef HAVE_MPI
		// TODO: overlap computation with synchronization, just like above
		nse.synchronizeDFsAndMacroDevice(output_df);
		ade.synchronizeDFsAndMacroDevice(output_df);
	#endif
#endif

		nse.iterations++;
		ade.iterations = nse.iterations;

		bool doit = false;
		for (int c = 0; c < MAX_COUNTER; c++)
			if (c != PRINT && c != SAVESTATE)
				if (cnt[c].action(nse.physTime()))
					doit = true;
		if (doit) {
			// common copy
			nse.copyMacroToHost();
			ade.copyMacroToHost();
		}
	}

	void AfterSimUpdate() override
	{
		State<NSE>::AfterSimUpdate();
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
		dataManager.initEngine(fmt::format("results_{}/output_NSE_3D", id));
		for (const auto& block : nse.blocks) {
			const std::string fname = fmt::format("results_{}/output_NSE_3D", id);
			create_parent_directories(fname.c_str());
			auto outputData = [this](const BLOCK_NSE& block, int index, int dof, idx x, idx y, idx z, OutputDataDescriptor<dreal>& desc) mutable
			{
				return this->outputData(block, index, dof, x, y, z, desc);
			};
			block.writeVTK_3D(nse.lat, outputData, fname, nse.physTime(), cnt[VTK3D].count, dataManager);
			spdlog::info("[vtk {} written, time {:f}, cycle {:d}] ", fname, nse.physTime(), cnt[VTK3D].count);
		}

		dataManager.initEngine(fmt::format("results_{}/output_ADE_3D", id));
		for (const auto& block : ade.blocks) {
			const std::string fname = fmt::format("results_{}/output_ADE_3D", id);
			create_parent_directories(fname.c_str());
			auto outputData = [this](const BLOCK_ADE& block, int index, int dof, idx x, idx y, idx z, OutputDataDescriptor<dreal>& desc) mutable
			{
				return this->outputData(block, index, dof, x, y, z, desc);
			};
			block.writeVTK_3D(ade.lat, outputData, fname, nse.physTime(), cnt[VTK3D].count, dataManager);
			spdlog::info("[vtk {} written, time {:f}, cycle {:d}] ", fname, nse.physTime(), cnt[VTK3D].count);
		}
	}

	void writeVTKs_3Dcut() override
	{
		if (this->probe3Dvec.size() <= 0)
			return;

		// browse all 3D vtk cuts
		for (auto& probevec : this->probe3Dvec) {
			dataManager.initEngine(fmt::format("results_{}/output_NSE_3Dcut_{}", id, probevec.name));
			for (const auto& block : nse.blocks) {
				const std::string fname = fmt::format("results_{}/output_NSE_3Dcut_{}", id, probevec.name);
				// create parent directories
				create_file(fname.c_str());
				auto outputData = [this](const BLOCK_NSE& block, int index, int dof, idx x, idx y, idx z, OutputDataDescriptor<dreal>& desc) mutable
				{
					return this->outputData(block, index, dof, x, y, z, desc);
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
					probevec.step,
					dataManager
				);
				spdlog::info("[vtk {} written, time {:f}, cycle {:d}] ", fname, nse.physTime(), probevec.cycle);
			}

			dataManager.initEngine(fmt::format("results_{}/output_ADE_3Dcut_{}", id, probevec.name));
			for (const auto& block : ade.blocks) {
				const std::string fname = fmt::format("results_{}/output_ADE_3Dcut_{}", id, probevec.name);
				// create parent directories
				create_file(fname.c_str());
				auto outputData = [this](const BLOCK_ADE& block, int index, int dof, idx x, idx y, idx z, OutputDataDescriptor<dreal>& desc) mutable
				{
					return this->outputData(block, index, dof, x, y, z, desc);
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
					probevec.step,
					dataManager
				);
				spdlog::info("[vtk {} written, time {:f}, cycle {:d}] ", fname, nse.physTime(), probevec.cycle);
			}
			probevec.cycle++;
		}
	}

	void writeVTKs_2D() override
	{
		if (this->probe2Dvec.size() <= 0)
			return;

		// browse all 2D vtk cuts
		for (auto& probevec : this->probe2Dvec) {
			dataManager.initEngine(fmt::format("results_{}/output_NSE_2D_{}", id, probevec.name));
			for (const auto& block : nse.blocks) {
				const std::string fname = fmt::format("results_{}/output_NSE_2D_{}", id, probevec.name);
				// create parent directories
				create_file(fname.c_str());
				auto outputData = [this](const BLOCK_NSE& block, int index, int dof, idx x, idx y, idx z, OutputDataDescriptor<dreal>& desc) mutable
				{
					return this->outputData(block, index, dof, x, y, z, desc);
				};
				switch (probevec.type) {
					case 0:
						block.writeVTK_2DcutX(nse.lat, outputData, fname, nse.physTime(), probevec.cycle, probevec.position, dataManager);
						break;
					case 1:
						block.writeVTK_2DcutY(nse.lat, outputData, fname, nse.physTime(), probevec.cycle, probevec.position, dataManager);
						break;
					case 2:
						block.writeVTK_2DcutZ(nse.lat, outputData, fname, nse.physTime(), probevec.cycle, probevec.position, dataManager);
						break;
				}
				spdlog::info("[vtk {} written, time {:f}, cycle {:d}] ", fname, nse.physTime(), probevec.cycle);
			}

			dataManager.initEngine(fmt::format("results_{}/output_ADE_2D_{}", id, probevec.name));
			for (const auto& block : ade.blocks) {
				const std::string fname = fmt::format("results_{}/output_ADE_2D_{}", id, probevec.name);
				// create parent directories
				create_file(fname.c_str());
				auto outputData = [this](const BLOCK_ADE& block, int index, int dof, idx x, idx y, idx z, OutputDataDescriptor<dreal>& desc) mutable
				{
					return this->outputData(block, index, dof, x, y, z, desc);
				};
				switch (probevec.type) {
					case 0:
						block.writeVTK_2DcutX(nse.lat, outputData, fname, nse.physTime(), probevec.cycle, probevec.position, dataManager);
						break;
					case 1:
						block.writeVTK_2DcutY(nse.lat, outputData, fname, nse.physTime(), probevec.cycle, probevec.position, dataManager);
						break;
					case 2:
						block.writeVTK_2DcutZ(nse.lat, outputData, fname, nse.physTime(), probevec.cycle, probevec.position, dataManager);
						break;
				}
				spdlog::info("[vtk {} written, time {:f}, cycle {:d}] ", fname, nse.physTime(), probevec.cycle);
			}
			probevec.cycle++;
		}
	}

	bool outputData(const BLOCK_NSE& block, int index, int dof, idx x, idx y, idx z, OutputDataDescriptor<dreal>& desc) override
	{
		return false;
	}
	virtual bool outputData(const BLOCK_ADE& block, int index, int dof, idx x, idx y, idx z, OutputDataDescriptor<dreal>& desc)
	{
		return false;
	}
};
