#pragma once

#include <vector>
#include <deque>
#include <limits>
#include <string>

#include <sys/stat.h>
#include <sys/wait.h>

#include <TNL/Timer.h>
#include <fmt/core.h>

#include "defs.h"
#include "kernels.h"
#include "lbm.h"
#include "vtk_writer.h"

// ibm: lagrangian filament/surface
#include "lagrange_3D.h"

// sparse box origin+length where to plot - can be just a part of the domain
template < typename IDX >
struct probe3Dcut
{
	IDX ox,oy,oz; // lower left front point
	IDX lx,ly,lz; // length
	IDX step; // 1: every voxel 2: every 3 voxels etc.
	std::string name;
	int cycle;
};

template < typename IDX >
struct probe2Dcut
{
	int type; // 0=X, 1=Y, 2=Z
	std::string name;
	IDX position; // x/y/z ... LBM units ... int
	int cycle;
};

template < typename IDX >
struct probe1Dcut
{
	int type; // 0=X, 1=Y, 2=Z
	std::string name;
	IDX pos1; // x/y/z
	IDX pos2; // y/z
	int cycle;
};

template < typename REAL >
struct probe1Dlinecut
{
	std::string name;
	using point_t = TNL::Containers::StaticVector< 3, REAL >;
	point_t from; // physical units
	point_t to;   // physical units
	int cycle;
};

// for print/stat/write/reset counters
template < typename REAL >
struct counter
{
	int count=0;
	REAL period=-1.0;
	bool action(REAL time) { return (period>0 && time >= count * period) ? true : false; }
};

enum { STAT_RESET, STAT2_RESET, PRINT, VTK1D, VTK2D, VTK3D, PROBE1, PROBE2, PROBE3, SAVESTATE, VTK3DCUT, MAX_COUNTER };
enum { MemoryToFile, FileToMemory };


template< typename NSE >
struct State
{
	using TRAITS = typename NSE::TRAITS;
	using BLOCK_NSE = LBM_BLOCK< NSE >;
	using Lagrange3D = ::Lagrange3D< LBM<NSE> >;

	using map_t = typename TRAITS::map_t;
	using idx = typename TRAITS::idx;
	using dreal = typename TRAITS::dreal;
	using real = typename TRAITS::real;
	using point_t = typename TRAITS::point_t;
	using lat_t = typename LBM< NSE >::lat_t;

	using T_PROBE3DCUT = probe3Dcut<idx>;
	using T_PROBE2DCUT = probe2Dcut<idx>;
	using T_PROBE1DCUT = probe1Dcut<idx>;
	using T_PROBE1DLINECUT = probe1Dlinecut<real>;
	using T_COUNTER = counter<real>;

	LBM<NSE> nse;

	std::vector< T_PROBE3DCUT > probe3Dvec;
	std::vector< T_PROBE2DCUT > probe2Dvec;
	std::vector< T_PROBE1DCUT > probe1Dvec;
	std::vector< T_PROBE1DLINECUT > probe1Dlinevec;

	// Lagrange
	std::deque<Lagrange3D> FF;			// array of filaments (std::deque instead of std::vector because Lagrange3D is not copy-constructible)
	int addLagrange3D();				// add a filament into the array and returns its index
	void computeAllLagrangeForces();

	// vtk surface rotational
	void writeVTK_Surface(const char* name, real time, int cycle, Lagrange3D &fil);
	void writeVTK_Points(const char* name, real time, int cycle, Lagrange3D &fil);

	// how often to probe/print/write/stat
	T_COUNTER cnt[MAX_COUNTER];
	virtual void probe1() {  }
	virtual void probe2() {  }
	virtual void probe3() {  }
	virtual void statReset() { }
	virtual void stat2Reset() { }

	// vtk export
	template< typename real1, typename real2 >
	bool vtk_helper(const char* iid, real1 ivalue, int idofs, char* id, real2 &value, int &dofs) /// simplifies data output routine
	{
		sprintf(id,"%s",iid);
		dofs=idofs;
		value=ivalue;
		return true;
	}
	virtual void writeVTKs_2D();
	template < typename... ARGS >
	void add2Dcut_X(idx x, const char* fmt, ARGS... args);
	template < typename... ARGS >
	void add2Dcut_Y(idx y, const char* fmt, ARGS... args);
	template < typename... ARGS >
	void add2Dcut_Z(idx z, const char* fmt, ARGS... args);

	virtual void writeVTKs_3D();

	// 3D cuts
	virtual void writeVTKs_3Dcut();
	template < typename... ARGS >
	void add3Dcut(idx ox, idx oy, idx oz, idx lx, idx ly, idx lz, idx step, const char* fmt, ARGS... args);

	virtual void writeVTKs_1D();

	template < typename... ARGS >
	void add1Dcut(point_t from, point_t to, const char* fmt, ARGS... args);
	template < typename... ARGS >
	void add1Dcut_X(real y, real z, const char* fmt, ARGS... args);
	template < typename... ARGS >
	void add1Dcut_Y(real x, real z, const char* fmt, ARGS... args);
	template < typename... ARGS >
	void add1Dcut_Z(real x, real y, const char* fmt, ARGS... args);
	void write1Dcut(point_t from, point_t to, const std::string& fname);
	void write1Dcut_X(idx y, idx z, const std::string& fname);
	void write1Dcut_Y(idx y, idx z, const std::string& fname);
	void write1Dcut_Z(idx x, idx y, const std::string& fname);


	template< typename... ARGS >
	void WriteTempAVG(const char* filename, dreal AvgTemp, double height, double width, double res);
	
	template< typename... ARGS >
	void WriteTempInFile(const char* filename, dreal AvgTemp, double height, double width, double res, double physDt, double physDl, double physDif, double transfer, double vel, double time);




	int verbosity=1;
	std::string id = "default";

	virtual bool outputData(const BLOCK_NSE& block, int index, int dof, char *desc, idx x, idx y, idx z, real &value, int &dofs) { return false; }

	bool getPNGdimensions(const char * filename, int &w, int &h);

	bool projectPNG_X(const std::string& filename, idx x0, bool rotate=false, bool mirror=false, bool flip=false,
	                  real amin=0, real amax=1, real bmin=0, real bmax=1);  // amin, amax, bmin, bmax ... used for cropping, see the code
	bool projectPNG_Y(const std::string& filename, idx y0, bool rotate=false, bool mirror=false, bool flip=false,
	                  real amin=0, real amax=1, real bmin=0, real bmax=1);  // amin, amax, bmin, bmax ... used for cropping, see the code
	bool projectPNG_Z(const std::string& filename, idx z0, bool rotate=false, bool mirror=false, bool flip=false,
	                  real amin=0, real amax=1, real bmin=0, real bmax=1);  // amin, amax, bmin, bmax ... used for cropping, see the code

	// simulation control
	virtual bool estimateMemoryDemands(); // called from State constructor
	virtual void reset();
	virtual void setupBoundaries() { } // called from State::reset
	virtual void resetLattice(real irho, real ivx, real ivy, real ivz); // called from State::reset
	virtual void SimInit(); // called from core.h -- before time loop
	virtual void updateKernelData(); // called from core.h -- calls updateKernelData on all LBM blocks
	virtual void updateKernelVelocities() { } // called from core.h -- setup current velocity profile for the Kernel
	virtual void SimUpdate(); // called from core.h -- from the time loop, once per time step
	virtual void AfterSimUpdate(); // called from core.h -- once before the time loop and then after each SimUpdate() call
	virtual void AfterSimFinished(); // called from core.h -- at the end of the simulation (after the time loop)
	virtual void computeBeforeLBMKernel() { } // called from core.h just before the main LBMKernel -- extra kernels e.g. for the non-Newtonian model
	virtual void computeAfterLBMKernel() { } // called from core.h after the main LBMKernel -- extra kernels e.g. for the coupled LBM-MHFEM solver
	virtual void copyAllToDevice(); // called from SimInit -- copy the initial state to the GPU
	virtual void copyAllToHost(); // called from core.h -- inside the time loop before saving state

	template < typename... ARGS >
	void mark(const char* fmt, ARGS... args);
	bool isMark();

	void flagCreate(const char*flagname);
	void flagDelete(const char*flagname);
	bool flagExists(const char*flagname);


	template < typename... ARGS >
	void setid(const char* fmt, ARGS... args);

	template < typename... ARGS >
	void log(const char* fmt, ARGS... args);

	// save & load state
	void move(const std::string& srcdir, const std::string& dstdir, const std::string& srcfilename, const std::string& dstfilename);
	template < typename VARTYPE >
	int saveloadBinaryData(int direction, const std::string& dirname, const std::string& filename, VARTYPE* data, idx length);

	template< typename... ARGS >
	int saveLoadTextData(int direction, const std::string& subdirname, const std::string& filename, ARGS&... args);

	// old version
	//template < typename... ARGS >
	//int saveLoadTextData(int direction, const char*subdirname, const char*filename, const char*fmt, ARGS&... args);

	void saveAndLoadState(int direction, const char*subdirname);
//	void loadInitState(int direction, const char*subdirname);

	// called periodically thru cnt[SAVESTATE]
	bool check_savestate_flag=true;         // false = output savestate every cnt[SAVESTATE].period, true = output savestate only if "savestate" file exists
	bool delete_savestate_flag=true;        // true = delete "savestate" flag file after savestate is completed
	virtual void saveState(bool forced=false);
	virtual void loadState(bool forced=false);

	// JK magic starts from here
	std::string getFmt(short)               { return "%hd"; }
	std::string getFmt(int)                 { return "%d"; }
	std::string getFmt(long)                { return "%ld"; }
	std::string getFmt(long long)           { return "%lld"; }
	std::string getFmt(unsigned short)      { return "%hu"; }
	std::string getFmt(unsigned int)        { return "%u"; }
	std::string getFmt(unsigned long)       { return "%lu"; }
	std::string getFmt(unsigned long long)  { return "%llu"; }
	std::string getFmt(float)               { return "%e"; }
	std::string getFmt(double)              { return "%le"; }

	template< typename ARG0 >
	std::string getSaveLoadFmt(ARG0 arg0)	{ return getFmt(arg0) + "\n"; }

	template< typename ARG0, typename... ARGS >
	std::string getSaveLoadFmt(ARG0 arg0, ARGS... args) {	return getFmt(arg0) + "\n" + getSaveLoadFmt(args...); }
	// JK magic ends here

	// timers for walltime, GLUPS and ETA calculations
	TNL::Timer timer_total;
	long wallTime = -1;  // maximum allowed wallTime in seconds, use negative value to disable wall time check
	bool wallTimeReached();
	double getWallTime(bool collective=false);	// collective: must be true when called by all MPI ranks and false otherwise (e.g. when called only by rank 0)

	// helpers for incremental GLUPS calculation
	int glups_prev_iterations = 0;
	double glups_prev_time = 0;

	// timers for profiling
	TNL::Timer timer_SimInit, timer_SimUpdate, timer_AfterSimUpdate, timer_compute, timer_compute_overlaps, timer_wait_communication, timer_wait_computation;

	// constructors
	template< typename... ARGS >
	State(const TNL::MPI::Comm& communicator, lat_t ilat, ARGS&&... args)
	: nse(communicator, ilat, std::forward<ARGS>(args)...)
	{
		bool local_estimate = estimateMemoryDemands();
		bool global_result = TNL::MPI::reduce(local_estimate, MPI_LAND, communicator);
		if (!local_estimate)
			log("Not enough memory available (CPU or GPU). [disable this check in lbm3d/state.h -> State constructor]");
		if (!global_result)
			throw std::runtime_error("Not enough memory available (CPU or GPU).");

		// allocate host data -- after the estimate
		nse.allocateHostData();

		// initial time of current simulation
		timer_total.start();
	}
};

#include "state.hpp"
