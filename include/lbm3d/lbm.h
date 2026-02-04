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

#ifdef HAVE_MPI
	// synchronization methods
	void synchronizeDFsAndMacroDevice(uint8_t dftype);
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
	LBM(const TNL::MPI::Comm& communicator, lat_t lat, bool periodic_lattice = false);
	LBM(const TNL::MPI::Comm& communicator, lat_t lat, std::vector<BLOCK>&& blocks);

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

	template <typename F>
	void forLocalLatticeSites(F f);

	template <typename F>
	void forAllLatticeSites(F f);

	~LBM() = default;
};

#include "lbm.hpp"
