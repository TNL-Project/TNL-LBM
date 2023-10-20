#pragma once

#include "defs.h"
#include "lattice.h"

template< typename CONFIG >
struct LBM_BLOCK
{
	using MACRO = typename CONFIG::MACRO;
	using CPU_MACRO = typename CONFIG::CPU_MACRO;
	using TRAITS = typename CONFIG::TRAITS;

	using idx = typename TRAITS::idx;
	using dreal = typename TRAITS::dreal;
	using real = typename TRAITS::real;
	using map_t = typename TRAITS::map_t;
	using idx3d = typename TRAITS::idx3d;
	using lat_t = Lattice<3, real, idx>;

	using hmap_array_t = typename CONFIG::hmap_array_t;
	using dmap_array_t = typename CONFIG::dmap_array_t;
	using hlat_array_t = typename CONFIG::hlat_array_t;
	using dlat_array_t = typename CONFIG::dlat_array_t;
	using bool_array_t = typename CONFIG::bool_array_t;
	using dbool_array_t = typename CONFIG::dbool_array_t;
	using dreal_array_t = typename CONFIG::dreal_array_t;
	using hreal_array_t = typename CONFIG::hreal_array_t;
	using dlat_view_t = typename CONFIG::dlat_view_t;
	using hmacro_array_t = typename CONFIG::hmacro_array_t;
	using dmacro_array_t = typename CONFIG::dmacro_array_t;
	using cpumacro_array_t = typename CONFIG::cpumacro_array_t;
	using sync_array_t = typename CONFIG::sync_array_t;
	using boollat_array_t = typename CONFIG::boollat_array_t;
	using hboollat_array_t = typename CONFIG::hboollat_array_t;

	// KernelData contains only the necessary data for the CUDA kernel. these are copied just before the kernel is called
	typename CONFIG::DATA data;
#ifdef USE_CUDA
	// CUDA thread block size for the LBM kernel
	idx3d block_size{0, 0, 0};
#endif

	hmap_array_t hmap;
	dmap_array_t dmap;
	dreal_array_t difmap;
	hreal_array_t hdifmap;

	map_t& map(idx x, idx y, idx z) { return hmap(x,y,z); }
	const map_t& map(idx x, idx y, idx z) const { return hmap(x,y,z); }	

	// macroscopic quantities
	hmacro_array_t hmacro;
	dmacro_array_t dmacro;
	cpumacro_array_t cpumacro;

	//Transfer BC
	bool_array_t TransferFS;
	bool_array_t TransferSF;
	bool_array_t TransferSW;
	hboollat_array_t transferDIR;
	boollat_array_t dtransferDIR;

	// distribution functions
	hlat_array_t hfs[DFMAX];
	dlat_array_t dfs[DFMAX];

	// MPI
	TNL::MPI::Comm communicator = MPI_COMM_WORLD;
	int rank = 0;
	int nproc = 1;
	int neighbour_left = 0;
	int neighbour_right = 0;

	// lattice sizes and offsets
	idx3d global;
	idx3d local;
	idx3d offset;

	// block indices
	int left_id;  // index of the left neighbor block
	int id;       // index of this block
	int right_id; // index of the right neighbor block

	// constructors
	LBM_BLOCK() = delete;
	LBM_BLOCK(const LBM_BLOCK&) = delete;
	LBM_BLOCK(LBM_BLOCK&&) = default;
	LBM_BLOCK(const TNL::MPI::Comm& communicator, idx3d global, idx3d local, idx3d offset, int neighbour_left = -1, int neighbour_right = -1, int left_id = 0, int this_id = 0, int right_id = 0);

	int df_overlap_X() { return data.indexer.template getOverlap< 0 >(); }
	int df_overlap_Y() { return data.indexer.template getOverlap< 1 >(); }
	int df_overlap_Z() { return data.indexer.template getOverlap< 2 >(); }

#ifdef HAVE_MPI
	// synchronizers for dfs, macro and map
	TNL::Containers::DistributedNDArraySynchronizer< typename sync_array_t::ViewType > dreal_sync[CONFIG::Q + MACRO::N];
	TNL::Containers::DistributedNDArraySynchronizer< dmap_array_t > map_sync;

	// CUDA streams for the block itself and each neighbor
	std::map< int, TNL::Cuda::Stream > streams;

	// synchronization methods
	template< typename Array >
	void startDrealArraySynchronization(Array& array, int sync_offset);
	void synchronizeDFsDevice_start(uint8_t dftype);
	void synchronizeMacroDevice_start();
	void synchronizeMapDevice_start();
#endif

	void resetForces() { resetForces(0,0,0);}
	void resetForces(real ifx, real ify, real ifz);
	void copyForcesToDevice();

	// all this is needed for IBM only and forcing
	dreal* hrho() { return &hmacro(MACRO::e_rho, offset.x(), offset.y(), offset.z()); }
	dreal* hvx() { return &hmacro(MACRO::e_vx, offset.x(), offset.y(), offset.z()); }
	dreal* hvy() { return &hmacro(MACRO::e_vy, offset.x(), offset.y(), offset.z()); }
	dreal* hvz() { return &hmacro(MACRO::e_vz, offset.x(), offset.y(), offset.z()); }
	dreal* hfx() { return &hmacro(MACRO::e_fx, offset.x(), offset.y(), offset.z()); }
	dreal* hfy() { return &hmacro(MACRO::e_fy, offset.x(), offset.y(), offset.z()); }
	dreal* hfz() { return &hmacro(MACRO::e_fz, offset.x(), offset.y(), offset.z()); }

	dreal* drho() { return &dmacro(MACRO::e_rho, offset.x(), offset.y(), offset.z()); }
	dreal* dvx() { return &dmacro(MACRO::e_vx, offset.x(), offset.y(), offset.z()); }
	dreal* dvy() { return &dmacro(MACRO::e_vy, offset.x(), offset.y(), offset.z()); }
	dreal* dvz() { return &dmacro(MACRO::e_vz, offset.x(), offset.y(), offset.z()); }
	dreal* dfx() { return &dmacro(MACRO::e_fx, offset.x(), offset.y(), offset.z()); }
	dreal* dfy() { return &dmacro(MACRO::e_fy, offset.x(), offset.y(), offset.z()); }
	dreal* dfz() { return &dmacro(MACRO::e_fz, offset.x(), offset.y(), offset.z()); }

	void copyMapToHost();
	void copyMapToDevice();
	void copyMacroToHost();
	void copyMacroToDevice();
	void copyDFsToHost(uint8_t dfty);
	void copyDFsToDevice(uint8_t dfty);
	void copyDFsToHost();
	void copyDFsToDevice();

	void computeCPUMacroFromLat();

	// Helpers for indexing - methods check if the given GLOBAL (multi)index is in the local range
	bool isLocalIndex(idx x, idx y, idx z) const;
	bool isLocalX(idx x) const;
	bool isLocalY(idx y) const;
	bool isLocalZ(idx z) const;

	// Global methods - use GLOBAL indices !!!
	void setMap(idx x, idx y, idx z, map_t value);
	void setBoundaryTransfer();	
	void setBoundaryX(idx x, map_t value);
	void setBoundaryY(idx y, map_t value);
	void setBoundaryZ(idx z, map_t value);

	bool isFluid(idx x, idx y, idx z) const;
	bool isWall(idx x, idx y, idx z) const;
	bool isSolid(idx x, idx y, idx z) const;
	bool isSolidPhase(idx x, idx y, idx z) const;

	void resetMap(map_t geo_type);
	void setEqLat(idx x, idx y, idx z, real rho, real vx, real vy, real vz); // prescribe rho,vx,vy,vz at a given point into "hfs" array

	void allocateHostData();
	void allocateDeviceData();

	template< typename F >
	void forLocalLatticeSites(F f);

	template< typename F >
	void forAllLatticeSites(F f);

	// VTK output
	template< typename Output >
	void writeVTK_3D(lat_t lat, Output&& outputData, const std::string& filename, real time, int cycle) const;
	template< typename Output >
	void writeVTK_3Dcut(lat_t lat, Output&& outputData, const std::string& filename, real time, int cycle, idx ox, idx oy, idx oz, idx lx, idx ly, idx lz, idx step) const;
	template< typename Output >
	void writeVTK_2DcutX(lat_t lat, Output&& outputData, const std::string& filename, real time, int cycle, idx XPOS) const;
	template< typename Output >
	void writeVTK_2DcutY(lat_t lat, Output&& outputData, const std::string& filename, real time, int cycle, idx YPOS) const;
	template< typename Output >
	void writeVTK_2DcutZ(lat_t lat, Output&& outputData, const std::string& filename, real time, int cycle, idx ZPOS) const;

	~LBM_BLOCK() = default;
};

#include "lbm_block.hpp"
