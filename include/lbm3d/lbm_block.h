#pragma once

#include "defs.h"
#include "lattice.h"
#include <vector>

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

	using hmap_array_t = typename TRAITS::hmap_array_t;
	using dmap_array_t = typename TRAITS::dmap_array_t;
	using hlat_array_t = typename TRAITS::hlat_array_t;
	using dlat_array_t = typename TRAITS::dlat_array_t;
	using dlat_view_t = typename TRAITS::dlat_view_t;
	using hmacro_array_t = typename TRAITS::hmacro_array_t;
	using dmacro_array_t = typename TRAITS::dmacro_array_t;
	using dreal_array_t = typename TRAITS::dreal_array_t;
	using hreal_array_t = typename TRAITS::hreal_array_t;
	using hboollat_array_t = typename TRAITS::hboollat_array_t;
	using dboollat_array_t = typename TRAITS::dboollat_array_t;
	using dreal_view_t = typename dreal_array_t::ViewType;

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

	// AMR: refinement level of this block (0 = coarsest level, default for backward compatibility)
	int level = 0;

	// indices of the neighboring blocks
	std::map<TNL::Containers::SyncDirection, int> neighborIDs;

	// owners of the neighboring blocks
	std::map<TNL::Containers::SyncDirection, int> neighborRanks;

	// AMR: refinement-level state
	// per-block lattice parameters for this refinement level (defaults to all
	// zeros until initLevelLattice is called; identical semantics to the
	// global LBM::lat when level == 0)
	lat_t lat_local;
	// offset of this block in the parent (coarse) level's coordinate system
	// (used for interface location matching in Wave 3; defaults to `offset`)
	idx3d global_offset;

#ifdef HAVE_MPI
	// synchronizers for dfs, macro and map
	TNL::Containers::DistributedNDArraySynchronizer<dreal_view_t> df_sync[CONFIG::Q];
	TNL::Containers::DistributedNDArraySynchronizer<dreal_view_t> macro_sync[CONFIG::MACRO::N];
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

	// disjoint [begin, end) boxes in *local* indices covering all outflow-pass sites;
	// boxes may include non-outflow padding cells (the kernel early-outs those on the per-cell isOutflowPassBC check)
	struct OutflowBox
	{
		idx3d begin = 0;
		idx3d end = 0;
	};
	std::vector<OutflowBox> outflow_boxes;
	// minimum extent of each box side: sides of a single cell (the degenerate
	// mask direction, e.g. the wall-normal of a plane outlet) are exempt,
	// and on smaller local grids the side is only grown to the grid extent
	static constexpr idx min_outflow_box_extent = 32;
	// upper bound of the rectangle cover: if the cover has more boxes,
	// the pair whose bounding box adds the least dead volume is merged until it fits the bound
	// (merging may result in box sides below min_outflow_box_extent — minimizing the covered volume takes precedence)
	static constexpr idx max_outflow_boxes = 64;

	// constructors
	LBM_BLOCK() = delete;
	LBM_BLOCK(const LBM_BLOCK&) = delete;
	LBM_BLOCK(LBM_BLOCK&&) = default;
	LBM_BLOCK(const TNL::MPI::Comm& communicator, idx3d global, idx3d local, idx3d offset, int this_id = 0);
	// AMR: constructor with explicit refinement level - computes per-level lattice parameters from the base (coarsest-level) lattice
	LBM_BLOCK(const TNL::MPI::Comm& communicator, idx3d global, idx3d local, idx3d offset, const lat_t& base_lat, int level, int this_id = 0);

	// AMR: (re)initializes the refinement-level state (`level`, `lat_local` and `data.lbmViscosity`) from the base (coarsest-level) lattice
	void initLevelLattice(const lat_t& base_lat, int level);

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

	// Actual width of overlaps allocated for this block's arrays (map, DF and
	// macro storages all share one indexer, so they must all use this width).
	// Defaults to `overlap_width`; blocks at refinement level > 0 use 1 so
	// that the inter-level coupling kernels can fill (coarse-to-fine) and read
	// (fine-to-coarse) the ghost ring around the block's footprint (since
	// change 4 of the interface redesign: a 1-cell ring suffices once the
	// ring fine-to-coarse launch is retired; the only kernel still reaching
	// a ghost cell is the max-side skin window, +1 deep, and the streaming
	// 1-hop neighborhood) -- see initLevelLattice and the kernels in
	// d3q27/amr_coupling.h. The allocation only materializes the overlap on
	// axes where the block is a proper subdomain (`local != global`), so
	// always query the allocated indexer (`df_overlap_X/Y/Z`) instead of
	// assuming this value everywhere.
	int storage_overlap = overlap_width;

	// Per-axis overrides of the allocated overlap depth (-1 = use
	// `storage_overlap` on that axis). A fine-level bounce-back wall on a
	// footprint min face (State_AMR's fine_wall_masks, imposed e.g. by
	// sim_AMR/sim_AMR_channel.cu) places its GEO_WALL row at local index -2
	// of the walled axis with a GEO_NOTHING streaming buffer at -3 (on a
	// max face at local+1 with the buffer at local+2): the AA-pattern
	// neighbor reads are unclamped (kernels.h), so a processed wall cell
	// needs one allocated row beyond it that the kernel never processes,
	// i.e. a 3-deep overlap on the walled axis. A simulation sets the
	// override AFTER createAMRBlocks and BEFORE execute() -- State::SimInit
	// re-runs the allocation for all blocks, materializing the deeper
	// overlap then.
	int storage_overlap_x = -1;
	int storage_overlap_y = -1;
	int storage_overlap_z = -1;

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
	template <typename Array, typename view_t, typename XYZIndexer>
	void start4DArraySynchronization(Array& array, TNL::Containers::DistributedNDArraySynchronizer<view_t>* sync, XYZIndexer indexer, bool is_df);
	void synchronizeDFsDevice_start(uint8_t dftype);
	void synchronizeMacroDevice_start();
	void synchronizeMapDevice_start();
#endif

	void copyMapToHost();
	void copyMapToDevice();
	// recompute the rectangle cover of outflow-pass sites from the host map
	void updateOutflowPassRegion();
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
	// compute macroscopic quantities from the stored device DFs over an
	// arbitrary [begin, end) window of LOCAL (indexer) coordinates -- same
	// device pass and math as the no-arg overload, just extent-parameterized
	// (the no-arg overload delegates with the interior [0, local) extent).
	// begin/end are in x/y/z axis order; the window may extend into the
	// ghost rows (negative / >= local coordinates) as long as they are
	// within the block's allocated overlap storage.
	void computeInitialMacro(const idx3d& begin, const idx3d& end);

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
