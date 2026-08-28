#pragma once

#include "lattice.h"
#include "lbm_block.h"
#include <type_traits>

template <typename CONFIG>
struct LBM
{
	using MACRO = typename CONFIG::MACRO;
	using TRAITS = typename CONFIG::TRAITS;
	using BLOCK = LBM_BLOCK<CONFIG>;
	static_assert(std::is_move_constructible_v<BLOCK>, "LBM_BLOCK must be move-constructible");

	using idx = typename TRAITS::idx;
	using dreal = typename TRAITS::dreal;
	using real = typename TRAITS::real;
	using map_t = typename TRAITS::map_t;
	using point_t = typename TRAITS::point_t;
	using idx3d = typename TRAITS::idx3d;
	using bool3d = typename TRAITS::bool3d;
	using lat_t = Lattice<3, real, idx>;

	// MPI
	TNL::MPI::Comm communicator = MPI_COMM_WORLD;
	int rank = 0;
	int nproc = 1;

	// global lattice size and physical units conversion
	lat_t lat;

	// local lattice blocks (subdomains)
	std::vector<BLOCK> blocks;
	int total_blocks = 0;

	// AMR: maximum refinement level present among the blocks and the number of
	// blocks at each level (blocks at all levels share the `blocks` vector and
	// each block carries its own `level` field; level_block_counts is maintained
	// by the caller when fine-level blocks are created)
	int max_level = 0;
	std::vector<int> level_block_counts;

	// AMR subcycling: cumulative per-level substep counters, one per
	// refinement level (level 0 advances on the global `iterations` clock
	// instead, so totalSubstepCount[0] is never consumed). The counters drive
	// the parity/rotation argument of updateKernelDataForLevel and the
	// write-side parity of the fine-to-coarse launches; a counter holds the
	// level's COMPLETED-substep count, i.e. the index of its next substep
	// (post-increment semantics mirroring how the global `iterations` clock
	// relates to the next level-0 step). Reset only on construction/restart:
	// every coarse cycle adds 2^L substeps at level L (an even count), so the
	// parity/rotation state is cycle-invariant and a restart at a cycle
	// boundary with zeroed counters is parity-consistent by induction.
	std::vector<int> totalSubstepCount;

#ifdef HAVE_MPI
	// synchronization methods
	void synchronizeDFsAndMacroDevice(uint8_t dftype, bool sync_macro);
	// AMR: per-level variant of synchronizeDFsAndMacroDevice for subcycling -
	// synchronizes only the blocks at `level` (early no-op when the level has
	// no blocks on this rank)
	void synchronizeDFsAndMacroDeviceForLevel(int level, uint8_t dftype, bool sync_macro);
	void synchronizeMapDevice();
#endif

	// input parameters: constant in time
	real physCharLength;		// characteristic length used for Re calculation, default is physDl * (real)Y but you can specify that manually
	real physFinalTime = 1e10;	// default 1e10
	real physStartTime = 0;		// used for ETA calculation only (default is 0)
	int iterations = 0;			// number of lbm iterations
	int startIterations = 0;	// number of lbm iterations at the start (physStartTime) -- used for GLUPS calculation only

	bool terminate = false;	 // flag for terminal error detection

	// constructors
	LBM() = delete;
	LBM(const LBM&) = delete;
	LBM(LBM&&) = default;
	LBM(const TNL::MPI::Comm& communicator, lat_t lat, const bool3d& periodic = {false, false, false});
	LBM(const TNL::MPI::Comm& communicator, lat_t lat, std::vector<BLOCK>&& blocks);
	// AMR: delegates to the default-decomposition constructor (level-0 blocks
	// are created as usual) and pre-allocates the per-level bookkeeping;
	// fine-level block creation is left to the caller (AMR setup code)
	LBM(const TNL::MPI::Comm& communicator, lat_t lat, const bool3d& periodic, int max_level);

	real Re(real physvel)
	{
		return fabs(physvel) * physCharLength / lat.physViscosity;
	}
	real physTime()
	{
		return lat.physDt * (real) iterations;
	}

	void copyMapToHost();
	void copyMapToDevice();
	void copyMacroToHost();
	void copyMacroToDevice();
	void copyDFsToHost(uint8_t dfty);
	void copyDFsToDevice(uint8_t dfty);
	void copyDFsToHost();
	void copyDFsToDevice();

	// Helpers for indexing - methods check if the given GLOBAL (multi)index is in the local range
	bool isAnyLocalIndex(idx x, idx y, idx z);
	bool isAnyLocalX(idx x);
	bool isAnyLocalY(idx y);
	bool isAnyLocalZ(idx z);

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
	void updateKernelData();  // copy physical parameters to data structure accessible by the CUDA kernel

	// AMR helpers for multi-level block management
	std::vector<BLOCK*> getBlocksAtLevel(int level);			 // non-owning pointers to blocks at the given level (empty for out-of-range levels)
	BLOCK* findBlockContaining(idx x, idx y, idx z, int level);	 // block at `level` whose local range contains the coordinate (nullptr if none)
	// per-level variant of updateKernelData() for subcycling: even_iter parity
	// and DF pointer rotation are driven by the level-local `substep` counter,
	// not the global `iterations`
	void updateKernelDataForLevel(int level, int substep);

	// lattice-update census of one (coarse) iteration -- the GLUPS basis:
	// the level-0 global lattice updated once (the historical basis,
	// MPI-global by construction) plus every fine block's interior once per
	// substep it advances (a level-L block runs 2^L substeps per coarse
	// iteration under the AMR subcycling of State_AMR::advancePair). Fine
	// blocks live on the rank that owns them (AMR is single-rank in v1), and
	// non-AMR runs keep the historical basis exactly (no blocks at level > 0).
	double totalLatticeUpdatesPerIteration() const;

	template <typename F>
	void forLocalLatticeSites(F f);

	template <typename F>
	void forAllLatticeSites(F f);

	~LBM() = default;
};

#include "lbm.hpp"
