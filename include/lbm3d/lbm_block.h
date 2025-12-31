#pragma once

#include "defs.h"
#include "lattice.h"

template <typename CONFIG>
struct LBM_BLOCK
{
	using TRAITS = typename CONFIG::TRAITS;

	using idx = typename TRAITS::idx;
	using dreal = typename TRAITS::dreal;
	using real = typename TRAITS::real;
	using map_t = typename TRAITS::map_t;
	using point_t = typename TRAITS::point_t;
	using idx3d = typename TRAITS::idx3d;
	using bool3d = typename TRAITS::bool3d;
	using lat_t = Lattice<3, real, idx>;

	using hmap_array_t = typename CONFIG::hmap_array_t;
	using dmap_array_t = typename CONFIG::dmap_array_t;
	using hlat_array_t = typename CONFIG::hlat_array_t;
	using dlat_array_t = typename CONFIG::dlat_array_t;
	using dlat_view_t = typename CONFIG::dlat_view_t;
	using hmacro_array_t = typename CONFIG::hmacro_array_t;
	using dmacro_array_t = typename CONFIG::dmacro_array_t;
	using dreal_array_t = typename CONFIG::dreal_array_t;
	using hreal_array_t = typename CONFIG::hreal_array_t;
	using hboollat_array_t = typename CONFIG::hboollat_array_t;
	using dboollat_array_t = typename CONFIG::dboollat_array_t;

	// KernelData contains only the necessary data for the CUDA kernel. these are copied just before the kernel is called
	typename CONFIG::DATA data;

	hmap_array_t hmap;
	dmap_array_t dmap;

	// macroscopic quantities
	hmacro_array_t hmacro;
	dmacro_array_t dmacro;

	// Arrays for non-constant diffusion coefficient depending on spatial coordinates.
	// Note that these arrays are empty (zero size) by default and `lat.lbmViscosity`
	// is used instead as a constant throughout the domain. A convenient way to
	// allocate these arrays is to call `allocateDiffusionCoefficientArrays` from
	// `setupBoundaries`.
	hreal_array_t hdiffusionCoeff;
	dreal_array_t ddiffusionCoeff;

	// Arrays for the heat/mass transfer boundary condition.
	// Note that these arrays are empty (zero size) by default and the
	// simulation that wants to use the boundary condition must call
	// `allocatePhiTransferDirectionArrays` to initialize these arrays.
	hboollat_array_t hphiTransferDirection;
	dboollat_array_t dphiTransferDirection;

	// distribution functions
	hlat_array_t hfs[DFMAX];
	dlat_array_t dfs[DFMAX];

	// MPI
	TNL::MPI::Comm communicator = MPI_COMM_WORLD;
	int rank = 0;
	int nproc = 1;

	// lattice sizes and offsets
	idx3d global;
	idx3d local;
	idx3d offset;

	// index of this block
	int id;

	// indices of the neighboring blocks
	std::map<TNL::Containers::SyncDirection, int> neighborIDs;

	// owners of the neighboring blocks
	std::map<TNL::Containers::SyncDirection, int> neighborRanks;

#ifdef HAVE_MPI
	// synchronizers for dfs, macro and map
	TNL::Containers::DistributedNDArraySynchronizer<typename dreal_array_t::ViewType> dreal_sync[CONFIG::Q + CONFIG::MACRO::N];
	TNL::Containers::DistributedNDArraySynchronizer<dmap_array_t> map_sync;
#endif

	// data for compute for the block itself and each neighbor
	struct COMPUTE_DATA
	{
		// parameters for CUDA kernel launch
		dim3 gridSize;
		dim3 blockSize;
		TNL::Backend::Stream stream;
		// parameters for cudaLBMKernel
		idx3d offset = 0;
		idx3d size = 0;
	};
	std::map<TNL::Containers::SyncDirection, COMPUTE_DATA> computeData;

	// constructors
	LBM_BLOCK() = delete;
	LBM_BLOCK(const LBM_BLOCK&) = delete;
	LBM_BLOCK(LBM_BLOCK&&) = default;
	LBM_BLOCK(const TNL::MPI::Comm& communicator, idx3d global, idx3d local, idx3d offset, int this_id = 0);

	// initialization method for MPI synchronization - must be called before starting the simulation!
	template <typename Pattern>
	void setLatticeDecomposition(
		const Pattern& pattern,	 // communication pattern for MPI synchronization - must be consistent with the lattice decomposition
		const std::map<TNL::Containers::SyncDirection, int>& neighborIDs,
		const std::map<TNL::Containers::SyncDirection, int>& neighborRanks
	);

	// auxiliary
	dim3 getCudaBlockSize(const idx3d& local_size);
	dim3 getCudaGridSize(const idx3d& local_size, const dim3& block_size, idx x = 0, idx y = 0, idx z = 0);

// maximum width of overlaps for the map and fs arrays
// (the real overlap may still be 0 if there is no neighbor in the particular direction)
#ifdef HAVE_MPI
	static constexpr int overlap_width = 1;
#else
	static constexpr int overlap_width = 0;
#endif
	// maximum width of overlaps for the macro arrays
	static constexpr int macro_overlap_width = CONFIG::MACRO::overlap_width;

	int df_overlap_X()
	{
		return data.indexer.template getOverlap<0>();
	}
	int df_overlap_Y()
	{
		return data.indexer.template getOverlap<1>();
	}
	int df_overlap_Z()
	{
		return data.indexer.template getOverlap<2>();
	}

	// returns a tuple of bools indicating if the lattice is distributed along each dimension
	bool3d is_distributed() const
	{
		return TNL::notEqualTo(local, global);
	}

#ifdef HAVE_MPI
	// synchronization methods
	template <typename Array>
	void startDrealArraySynchronization(Array& array, int sync_offset, bool is_df);
	void synchronizeDFsDevice_start(uint8_t dftype);
	void synchronizeMacroDevice_start();
	void synchronizeMapDevice_start();
#endif

	void copyMapToHost();
	void copyMapToDevice();
	void copyMacroToHost();
	void copyMacroToDevice();
	void copyDFsToHost(uint8_t dfty);
	void copyDFsToDevice(uint8_t dfty);
	void copyDFsToHost();
	void copyDFsToDevice();

	// Helpers for indexing - methods check if the given GLOBAL (multi)index is in the local range
	bool isLocalIndex(idx x, idx y, idx z) const;
	bool isLocalX(idx x) const;
	bool isLocalY(idx y) const;
	bool isLocalZ(idx z) const;

	// Global methods - use GLOBAL indices !!!
	void setMap(idx x, idx y, idx z, map_t value);
	void setBoundaryX(idx x, map_t value);
	void setBoundaryY(idx y, map_t value);
	void setBoundaryZ(idx z, map_t value);

	void resetMap(map_t geo_type);
	void setEquilibrium(real rho, real vx, real vy, real vz);
	void computeInitialMacro();

	void allocateHostData();
	void allocateDeviceData();
	void allocateDiffusionCoefficientArrays();
	void allocatePhiTransferDirectionArrays();

	template <typename F>
	void forLocalLatticeSites(F f);

	template <typename F>
	void forAllLatticeSites(F f);

	~LBM_BLOCK() = default;
};

#include "lbm_block.hpp"
