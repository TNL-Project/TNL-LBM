#pragma once

#include <cstdint>
#include <utility>

#include <TNL/Backend/Stream.h>
#include <TNL/Backend/Types.h>
#include <TNL/Containers/BlockPartitioning.h>
#include <TNL/Containers/DistributedNDArray.h>
#include <TNL/Containers/DistributedNDArraySyncDirections.h>
#include <TNL/Containers/DistributedNDArraySynchronizer.h>
#include <TNL/Containers/NDArray.h>
#include <TNL/Containers/StaticVector.h>
#include <TNL/Containers/ndarray/SizesHolder.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Devices/Host.h>
#include <TNL/MPI.h>
#include <TNL/MPI/ScopedInitializer.h>

#if ! defined(AB_PATTERN) && ! defined(AA_PATTERN)
	// TODO: update multidimensional MPI synchronization for AA pattern
	// (for the even time step which is similar to a "push scheme", we need to
	// avoid race conditions on the corners - set smaller buffer size and adjust the offsets)
	//#define AA_PATTERN
	#define AB_PATTERN
#endif

#if ! defined(__CUDACC__) && ! defined(__HIP__)
using TNL::dim3;
#endif

using TNLMPI_INIT = TNL::MPI::ScopedInitializer;

#ifdef __CUDACC__
	#define CUDA_HOSTDEV __host__ __device__
#else
	#define CUDA_HOSTDEV
#endif

#ifdef __CUDACC__
	#include <cuda_profiler_api.h>
#endif

#define CONSTFUNC __attribute__((always_inline)) static constexpr

// number of dist. functions, default=2
// quick fix, use templates to define DFMAX ... through TRAITS maybe ?
#ifdef USE_DFMAX3  // special 3 dfs
enum : std::uint8_t
{
	df_cur,
	df_out,
	df_prev,
	DFMAX
};
#elif defined(AB_PATTERN)  // default 2 dfs
enum : std::uint8_t
{
	df_cur,
	df_out,
	DFMAX
};
#elif defined(AA_PATTERN)
enum : std::uint8_t
{
	df_cur,
	DFMAX
};
#endif

#ifdef USE_CUDA
using DeviceType = TNL::Devices::Cuda;
#else
using DeviceType = TNL::Devices::Host;
#endif

template <
	typename _dreal = float,   // real number representation on GPU
	typename _real = double,   // real number representation on CPU
	typename _idx = long int,  // array index on CPU and GPU (can be very large)
	typename _map_t = short int>
struct Traits
{
	using real = _real;
	using dreal = _dreal;
	using idx = _idx;
	using map_t = _map_t;
	using point_t = TNL::Containers::StaticVector<3, real>;
	using idx3d = TNL::Containers::StaticVector<3, idx>;
	using bool3d = TNL::Containers::StaticVector<3, bool>;

	using xyz_permutation = std::index_sequence<0, 2, 1>;	 // x, z, y
	using d4_permutation = std::index_sequence<0, 1, 3, 2>;	 // id, x, z, y

#ifdef HAVE_MPI
	// all overlaps are set at run-time
	using xyz_overlaps = TNL::Containers::SizesHolder<idx, 0, 0, 0>;	// x, y, z
	using d4_overlaps = TNL::Containers::SizesHolder<idx, 0, 0, 0, 0>;	// id, x, y, z
#else
	// all overlaps are zero
	using xyz_overlaps = TNL::Containers::ConstStaticSizesHolder<idx, 3, 0>;
	using d4_overlaps = TNL::Containers::ConstStaticSizesHolder<idx, 4, 0>;
#endif

	template <typename Value, typename Device>
	using array3d = TNL::Containers::NDArray<
		Value,
		TNL::Containers::SizesHolder<idx, 0, 0, 0>,	 // x, y, z
		xyz_permutation,
		Device,
		idx,
		xyz_overlaps>;
	template <std::size_t N, typename Value, typename Device>
	using array4d = TNL::Containers::NDArray<
		Value,
		TNL::Containers::SizesHolder<idx, N, 0, 0, 0>,	// N, x, y, z
		d4_permutation,
		Device,
		idx,
		d4_overlaps>;

	using lattice_indexer_t = typename array3d<dreal, DeviceType>::IndexerType;
};

using TraitsSP = Traits<float>;	 //_dreal is float only
using TraitsDP = Traits<double>;

struct Coord{
	int x,y,z;
};

template < typename INDEX, int NoDV >
struct StreamGrid
{
	INDEX ids[6*NoDV+3];
	CUDA_HOSTDEV INDEX& x(int id){return ids[id];};
	CUDA_HOSTDEV INDEX& y(int id){return ids[2*NoDV+1+id];};
	CUDA_HOSTDEV INDEX& z(int id){return ids[4*NoDV+2+id];};
};

// KernelStruct - D2Q9
template < typename REAL >
struct D2Q9_KernelStruct
{

	static constexpr int NoDV = 1;
	static constexpr int ONE_SIZE = 2*NoDV + 1;
	static constexpr int Q = ONE_SIZE*ONE_SIZE;
	static constexpr int Qhalf = (Q-1)/2;

	using SG = StreamGrid<int, NoDV>;

	// Same for all models, always can be ordered in the way that flipping id means flipping discrete velocity
	CUDA_HOSTDEV CONSTFUNC int flip_coord(int val){return ONE_SIZE-val-1;}
	CUDA_HOSTDEV CONSTFUNC int flip_id(int id){return Q - id - 1;}

	CUDA_HOSTDEV Coord id_to_dv(int id){
		return {id/ONE_SIZE-NoDV,id%ONE_SIZE-NoDV,0};
	}

	CUDA_HOSTDEV Coord id_to_coords(int id){
		return {id/ONE_SIZE,id%ONE_SIZE,NoDV};
	}

	CUDA_HOSTDEV Coord id_to_flip_coords(int id){
		return {flip_coord(id/ONE_SIZE),flip_coord(id%ONE_SIZE),NoDV};
	}

	CUDA_HOSTDEV CONSTFUNC int dv_to_id(int cx, int cy, int cz){
		return cy+NoDV + ONE_SIZE*(cx+NoDV);
	}

	CUDA_HOSTDEV CONSTFUNC int coords_to_id(int cx, int cy, int cz){
		return cy + ONE_SIZE*cx;
	}

	REAL fx=0.,fy=0.,fz=0.;
	REAL f[Q];
	REAL vx=0., vy=0., vz=0., rho=1.0, lbmViscosity=1.0, T=1./3;
};
// helper function for getting a 3D array view from a 4D distributed array
template <typename TRAITS, typename Array>
auto getMacroView(const Array& array, std::uint8_t id)
{
	using local_array_t = typename TRAITS::template array3d<typename Array::ValueType, typename Array::DeviceType>;
	using local_view_t = typename local_array_t::ViewType;
#ifdef HAVE_MPI
	using view_t = TNL::Containers::DistributedNDArrayView<local_view_t>;
#else
	using view_t = local_view_t;
#endif

	// getSubarrayView does not handle overlaps :-(
	typename TRAITS::xyz_overlaps overlaps;
#ifdef HAVE_MPI
	overlaps.template setSize<0>(array.getOverlaps().template getSize<1>());
	overlaps.template setSize<1>(array.getOverlaps().template getSize<2>());
	overlaps.template setSize<2>(array.getOverlaps().template getSize<3>());
	const auto subarray = array.getConstLocalView().template getSubarrayView<1, 2, 3>(id, 0, 0, 0);
#else
	const auto subarray = array.getConstView().template getSubarrayView<1, 2, 3>(id, 0, 0, 0);
#endif
	local_view_t local_view(const_cast<typename Array::ValueType*>(subarray.getData()), subarray.getSizes(), subarray.getStrides(), overlaps);

#ifdef HAVE_MPI
	typename view_t::SizesHolderType global_sizes;
	global_sizes.template setSize<0>(array.template getSize<1>());
	global_sizes.template setSize<1>(array.template getSize<2>());
	global_sizes.template setSize<2>(array.template getSize<3>());
	typename view_t::LocalBeginsType local_begins;
	local_begins.template setSize<0>(array.getLocalBegins().template getSize<1>());
	local_begins.template setSize<1>(array.getLocalBegins().template getSize<2>());
	local_begins.template setSize<2>(array.getLocalBegins().template getSize<3>());
	typename view_t::SizesHolderType local_ends;
	local_ends.template setSize<0>(array.getLocalEnds().template getSize<1>());
	local_ends.template setSize<1>(array.getLocalEnds().template getSize<2>());
	local_ends.template setSize<2>(array.getLocalEnds().template getSize<3>());
	return view_t(local_view, global_sizes, local_begins, local_ends, array.getCommunicator());
#else
	return local_view;
#endif
}

// KernelStruct - D3Q7
template <typename REAL>
struct D3Q7_KernelStruct
{
	static constexpr int D = 3;
	static constexpr int Q = 7;
	static constexpr int NoDV = 1;

	using SG = StreamGrid<int, NoDV>;
	REAL f[Q];
	REAL vx = 0, vy = 0, vz = 0;
	REAL phi = 1.0, lbmViscosity = 1.0;
	// FIXME
	//REAL qcrit=0, phigradmag2=0;
};

// KernelStruct - D3Q27
template <typename REAL>
struct D3Q27_KernelStruct
{
	static constexpr int D = 3;
	static constexpr int Q = 27;
	static constexpr int Qhalf = (Q-1)/2;
	static constexpr int NoDV = 1;
	static constexpr int ONE_SIZE = 2*NoDV + 1;
	static constexpr REAL T0 = 1./3;
	static constexpr REAL cs = 0.5773502691896258;

	__cuda_callable__ CONSTFUNC int flip_coord(int val){return ONE_SIZE-val-1;}
	__cuda_callable__ CONSTFUNC int flip_id(int id){return Q - id - 1;}
	__cuda_callable__ CONSTFUNC int flip_id_x(int id){
		Coord c = id_to_coords(id);
		int nx = flip_coord(c.x);
		return coords_to_id(nx, c.y, c.z);
	}
	__cuda_callable__ CONSTFUNC int flip_id_y(int id){
		Coord c = id_to_coords(id);
		int ny = flip_coord(c.y);
		return coords_to_id(c.x, ny, c.z);
	}
	__cuda_callable__ CONSTFUNC int flip_id_z(int id){
		Coord c = id_to_coords(id);
		int nz = flip_coord(c.z);
		return coords_to_id(c.x, c.y, nz);
	}

	__cuda_callable__ CONSTFUNC Coord id_to_dv(int id){
		int x = id / (ONE_SIZE * ONE_SIZE);
		int y = (id / ONE_SIZE) % ONE_SIZE;
		int z = id % ONE_SIZE;
		return {x-NoDV,y-NoDV,z-NoDV};
	}

	__cuda_callable__ CONSTFUNC Coord id_to_coords(int id){
		int x = id / (ONE_SIZE * ONE_SIZE);
		int y = (id / ONE_SIZE) % ONE_SIZE;
		int z = id % ONE_SIZE;
		return {x,y,z};
	}

	__cuda_callable__ CONSTFUNC int dv_to_id(int cx, int cy, int cz){
		return (cx + NoDV) * ONE_SIZE * ONE_SIZE + (cy + NoDV) * ONE_SIZE + (cz + NoDV);
	}
	__cuda_callable__ CONSTFUNC int coords_to_id(int cx, int cy, int cz){
		return cx * ONE_SIZE * ONE_SIZE + cy * ONE_SIZE + cz;
	}

	using SG = StreamGrid<int, 1>;

	REAL f[Q];
	REAL fx = 0, fy = 0, fz = 0;
	REAL vx = 0, vy = 0, vz = 0;
	REAL rho = 1.0, lbmViscosity = 1.0;

#if defined(USE_CYMODEL) || defined(USE_CASSON)
	REAL S11 = 0., S12 = 0., S22 = 0., S32 = 0., S13 = 0., S33 = 0.;

	//Non-Newtonian parameters
	#if defined(USE_CYMODEL)
	REAL lbm_nu0 = 0, lbm_lambda = 0, lbm_a = 0, lbm_n = 0;
	#elif defined(USE_CASSON)
	REAL lbm_k0 = 0, lbm_k1 = 0;
	#endif

	REAL mu;
#endif
};

// KernelStruct - D3Q343
template <typename REAL>
struct D3Q343_KernelStruct
{
	static constexpr int D = 3;
	static constexpr int Q = 343;
	static constexpr int Qhalf = (Q-1)/2;
	static constexpr int NoDV = 3;
	static constexpr int ONE_SIZE = 2*NoDV + 1;
	static constexpr REAL T0 = 0.6979533220196830882384091;
	static constexpr REAL cs = 0.8354360071362038; // sqrt(T0)

	__cuda_callable__ CONSTFUNC int flip_coord(int val){return ONE_SIZE-val-1;}
	__cuda_callable__ CONSTFUNC int flip_id(int id){return Q - id - 1;}

	__cuda_callable__ CONSTFUNC Coord id_to_dv(int id){
		int x = id / (ONE_SIZE * ONE_SIZE);
		int y = (id / ONE_SIZE) % ONE_SIZE;
		int z = id % ONE_SIZE;
		return {x-NoDV,y-NoDV,z-NoDV};
	}

	__cuda_callable__ CONSTFUNC Coord id_to_coords(int id){
		int x = id / (ONE_SIZE * ONE_SIZE);
		int y = (id / ONE_SIZE) % ONE_SIZE;
		int z = id % ONE_SIZE;
		return {x,y,z};
	}

	__cuda_callable__ CONSTFUNC int dv_to_id(int cx, int cy, int cz){
		return (cx + NoDV) * ONE_SIZE * ONE_SIZE + (cy + NoDV) * ONE_SIZE + (cz + NoDV);
	}
	__cuda_callable__ CONSTFUNC int coords_to_id(int cx, int cy, int cz){
		return cx * ONE_SIZE * ONE_SIZE + cy * ONE_SIZE + cz;
	}

	using SG = StreamGrid<int, 3>;

	REAL f[Q];
	REAL fx = 0, fy = 0, fz = 0;
	REAL vx = 0, vy = 0, vz = 0;
	REAL rho = 1.0, lbmViscosity = 1.0;
	// ELBM Lagrange multipliers
	REAL A = 1.0, B1 = 1.0 , B2 = 1.0, B3 = 1.0;
	REAL alpha = 2.0;
};

template <typename REAL>
struct D3Q53_KernelStruct_ELBM
{
	static constexpr int D = 3;
	static constexpr REAL T0 = 1./2.67972986276583;
	static constexpr int Q = 53;
	static constexpr int Qhalf = (Q-1)/2;
	static constexpr int NoDV = 3;
	static constexpr int ONE_SIZE = 2*NoDV + 1;
	static constexpr REAL cs = 0.6108780100379961; // sqrt(T0)

	__cuda_callable__ CONSTFUNC int flip_coord(int val){return ONE_SIZE-val-1;}
	__cuda_callable__ CONSTFUNC int flip_id(int id){return Q - id - 1;}
	__cuda_callable__ CONSTFUNC int flip_id_x(int id){
		Coord c = id_to_coords(id);
		int nx = flip_coord(c.x);
		return coords_to_id(nx, c.y, c.z);
	}
	__cuda_callable__ CONSTFUNC int flip_id_y(int id){
		Coord c = id_to_coords(id);
		int ny = flip_coord(c.y);
		return coords_to_id(c.x, ny, c.z);
	}
	__cuda_callable__ CONSTFUNC int flip_id_z(int id){
		Coord c = id_to_coords(id);
		int nz = flip_coord(c.z);
		return coords_to_id(c.x, c.y, nz);
	}

	__cuda_callable__ CONSTFUNC Coord id_to_dv(int id){
		Coord c = id_to_coords(id);
		return {c.x-NoDV,c.y-NoDV,c.z-NoDV};
	}

	__cuda_callable__ CONSTFUNC Coord id_to_coords(int id){
        const int XS[Q] = {0,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4,5,5,5,5,5,5,5,5,6};
        const int YS[Q] = {3,1,1,1,3,3,5,5,5,2,2,2,3,3,3,4,4,4,0,1,1,2,2,2,3,3,3,3,3,4,4,4,5,5,6,2,2,2,3,3,3,4,4,4,1,1,1,3,3,5,5,5,3};
        const int ZS[Q] = {3,1,3,5,1,5,1,3,5,2,3,4,2,3,4,2,3,4,3,1,5,2,3,4,0,2,3,4,6,2,3,4,1,5,3,2,3,4,2,3,4,2,3,4,1,3,5,1,5,1,3,5,3};
		int x = XS[id];
		int y = YS[id];
		int z = ZS[id];
		return {x,y,z};
	}


	__cuda_callable__ CONSTFUNC int dv_to_id(int cx, int cy, int cz){
		if (cx == -3 && cy == 0 && cz == 0) return 0;
		if (cx == -2 && cy == -2 && cz == -2) return 1;
		if (cx == -2 && cy == -2 && cz == 0) return 2;
		if (cx == -2 && cy == -2 && cz == 2) return 3;
		if (cx == -2 && cy == 0 && cz == -2) return 4;
		if (cx == -2 && cy == 0 && cz == 2) return 5;
		if (cx == -2 && cy == 2 && cz == -2) return 6;
		if (cx == -2 && cy == 2 && cz == 0) return 7;
		if (cx == -2 && cy == 2 && cz == 2) return 8;
		if (cx == -1 && cy == -1 && cz == -1) return 9;
		if (cx == -1 && cy == -1 && cz == 0) return 10;
		if (cx == -1 && cy == -1 && cz == 1) return 11;
		if (cx == -1 && cy == 0 && cz == -1) return 12;
		if (cx == -1 && cy == 0 && cz == 0) return 13;
		if (cx == -1 && cy == 0 && cz == 1) return 14;
		if (cx == -1 && cy == 1 && cz == -1) return 15;
		if (cx == -1 && cy == 1 && cz == 0) return 16;
		if (cx == -1 && cy == 1 && cz == 1) return 17;
		if (cx == 0 && cy == -3 && cz == 0) return 18;
		if (cx == 0 && cy == -2 && cz == -2) return 19;
		if (cx == 0 && cy == -2 && cz == 2) return 20;
		if (cx == 0 && cy == -1 && cz == -1) return 21;
		if (cx == 0 && cy == -1 && cz == 0) return 22;
		if (cx == 0 && cy == -1 && cz == 1) return 23;
		if (cx == 0 && cy == 0 && cz == -3) return 24;
		if (cx == 0 && cy == 0 && cz == -1) return 25;
		if (cx == 0 && cy == 0 && cz == 0) return 26;
		if (cx == 0 && cy == 0 && cz == 1) return 27;
		if (cx == 0 && cy == 0 && cz == 3) return 28;
		if (cx == 0 && cy == 1 && cz == -1) return 29;
		if (cx == 0 && cy == 1 && cz == 0) return 30;
		if (cx == 0 && cy == 1 && cz == 1) return 31;
		if (cx == 0 && cy == 2 && cz == -2) return 32;
		if (cx == 0 && cy == 2 && cz == 2) return 33;
		if (cx == 0 && cy == 3 && cz == 0) return 34;
		if (cx == 1 && cy == -1 && cz == -1) return 35;
		if (cx == 1 && cy == -1 && cz == 0) return 36;
		if (cx == 1 && cy == -1 && cz == 1) return 37;
		if (cx == 1 && cy == 0 && cz == -1) return 38;
		if (cx == 1 && cy == 0 && cz == 0) return 39;
		if (cx == 1 && cy == 0 && cz == 1) return 40;
		if (cx == 1 && cy == 1 && cz == -1) return 41;
		if (cx == 1 && cy == 1 && cz == 0) return 42;
		if (cx == 1 && cy == 1 && cz == 1) return 43;
		if (cx == 2 && cy == -2 && cz == -2) return 44;
		if (cx == 2 && cy == -2 && cz == 0) return 45;
		if (cx == 2 && cy == -2 && cz == 2) return 46;
		if (cx == 2 && cy == 0 && cz == -2) return 47;
		if (cx == 2 && cy == 0 && cz == 2) return 48;
		if (cx == 2 && cy == 2 && cz == -2) return 49;
		if (cx == 2 && cy == 2 && cz == 0) return 50;
		if (cx == 2 && cy == 2 && cz == 2) return 51;
		if (cx == 3 && cy == 0 && cz == 0) return 52;

	}

	__cuda_callable__ CONSTFUNC int coords_to_id(int cx, int cy, int cz){
		return dv_to_id(cx-NoDV,cy-NoDV,cz-NoDV);
	}

    __cuda_callable__ CONSTFUNC REAL id_to_weight(int id){
		if (id == 0) return 0.000254627832132497;
		if (id == 1) return 0.00000404353462215176;
		if (id == 2) return 0.0000785975745805697;
		if (id == 3) return 0.00000404353462215176;
		if (id == 4) return 0.0000785975745805697;
		if (id == 5) return 0.0000785975745805697;
		if (id == 6) return 0.00000404353462215176;
		if (id == 7) return 0.0000785975745805697;
		if (id == 8) return 0.00000404353462215176;
		if (id == 9) return 0.00623707839948299;
		if (id == 10) return 0.0209532136880463;
		if (id == 11) return 0.00623707839948299;
		if (id == 12) return 0.0209532136880463;
		if (id == 13) return 0.0742108949874377;
		if (id == 14) return 0.0209532136880463;
		if (id == 15) return 0.00623707839948299;
		if (id == 16) return 0.0209532136880463;
		if (id == 17) return 0.00623707839948299;
		if (id == 18) return 0.000254627832132497;
		if (id == 19) return 0.0000785975745805697;
		if (id == 20) return 0.0000785975745805697;
		if (id == 21) return 0.0209532136880463;
		if (id == 22) return 0.0742108949874377;
		if (id == 23) return 0.0209532136880463;
		if (id == 24) return 0.000254627832132497;
		if (id == 25) return 0.0742108949874377;
		if (id == 26) return 0.250896152458214;
		if (id == 27) return 0.0742108949874377;
		if (id == 28) return 0.000254627832132497;
		if (id == 29) return 0.0209532136880463;
		if (id == 30) return 0.0742108949874377;
		if (id == 31) return 0.0209532136880463;
		if (id == 32) return 0.0000785975745805697;
		if (id == 33) return 0.0000785975745805697;
		if (id == 34) return 0.000254627832132497;
		if (id == 35) return 0.00623707839948299;
		if (id == 36) return 0.0209532136880463;
		if (id == 37) return 0.00623707839948299;
		if (id == 38) return 0.0209532136880463;
		if (id == 39) return 0.0742108949874377;
		if (id == 40) return 0.0209532136880463;
		if (id == 41) return 0.00623707839948299;
		if (id == 42) return 0.0209532136880463;
		if (id == 43) return 0.00623707839948299;
		if (id == 44) return 0.00000404353462215176;
		if (id == 45) return 0.0000785975745805697;
		if (id == 46) return 0.00000404353462215176;
		if (id == 47) return 0.0000785975745805697;
		if (id == 48) return 0.0000785975745805697;
		if (id == 49) return 0.00000404353462215176;
		if (id == 50) return 0.0000785975745805697;
		if (id == 51) return 0.00000404353462215176;
		if (id == 52) return 0.000254627832132497;

	}

	using SG = StreamGrid<int, 3>;

	REAL f[Q];
	REAL fx = 0, fy = 0, fz = 0;
	REAL vx = 0, vy = 0, vz = 0;
	REAL rho = 1.0, lbmViscosity = 1.0;
	REAL A = 1.0, B1 = 1.0 , B2 = 1.0, B3 = 1.0;
	REAL alpha = 2.;
};

// KernelStruct - D3Q27
template <typename REAL>
struct D3Q27_KernelStruct_Adjoint : public D3Q27_KernelStruct<REAL>
{
	// ..._m = measured velocities
	REAL vz_m = 0, vx_m = 0, vy_m = 0;
	REAL rho_m = 1.0;
	// REAL gx=0, gy=0, gz=0; //! adjoint gradient - if gradient update should not be done on kernel -> remove
};

template <
	typename _TRAITS,
	template <typename> class _KERNEL_STRUCT,
	typename _DATA,
	typename _COLL,
	typename _EQ,
	typename _STREAMING,
	template <typename> class _BC,
	typename _MACRO>
struct LBM_CONFIG
{
	using TRAITS = _TRAITS;
	template <typename REAL>
	using KernelStruct = _KERNEL_STRUCT<REAL>;
	using LBM_KS = KernelStruct<typename TRAITS::dreal>;
	using DATA = _DATA;
	using COLL = _COLL;
	using EQ = _EQ;
	using STREAMING = _STREAMING;
	using BC = _BC<LBM_CONFIG>;
	using MACRO = _MACRO;
	using SG = StreamGrid<int, KernelStruct<typename TRAITS::dreal>::NoDV>; // whyyyy

	static constexpr int D = KernelStruct<typename TRAITS::dreal>::D;
	static constexpr int Q = KernelStruct<typename TRAITS::dreal>::Q;

	using __hmap_array_t = typename TRAITS::template array3d<typename TRAITS::map_t, TNL::Devices::Host>;
	using __dmap_array_t = typename TRAITS::template array3d<typename TRAITS::map_t, DeviceType>;
	using __hbool_array_t = typename TRAITS::template array3d<bool, TNL::Devices::Host>;
	using __dbool_array_t = typename TRAITS::template array3d<bool, DeviceType>;
	using __hreal_array_t = typename TRAITS::template array3d<typename TRAITS::dreal, TNL::Devices::Host>;
	using __dreal_array_t = typename TRAITS::template array3d<typename TRAITS::dreal, DeviceType>;

	using __hlat_array_t = typename TRAITS::template array4d<Q, typename TRAITS::dreal, TNL::Devices::Host>;
	using __dlat_array_t = typename TRAITS::template array4d<Q, typename TRAITS::dreal, DeviceType>;
	using __hboollat_array_t = typename TRAITS::template array4d<Q, bool, TNL::Devices::Host>;
	using __dboollat_array_t = typename TRAITS::template array4d<Q, bool, DeviceType>;

	using __hmacro_array_t = typename TRAITS::template array4d<MACRO::N, typename TRAITS::dreal, TNL::Devices::Host>;
	using __dmacro_array_t = typename TRAITS::template array4d<MACRO::N, typename TRAITS::dreal, DeviceType>;

#ifdef HAVE_MPI
	using hmap_array_t = TNL::Containers::DistributedNDArray<__hmap_array_t>;
	using dmap_array_t = TNL::Containers::DistributedNDArray<__dmap_array_t>;
	using hbool_array_t = TNL::Containers::DistributedNDArray<__hbool_array_t>;
	using dbool_array_t = TNL::Containers::DistributedNDArray<__dbool_array_t>;
	using dreal_array_t = TNL::Containers::DistributedNDArray<__dreal_array_t>;
	using hreal_array_t = TNL::Containers::DistributedNDArray<__hreal_array_t>;

	using hlat_array_t = TNL::Containers::DistributedNDArray<__hlat_array_t>;
	using dlat_array_t = TNL::Containers::DistributedNDArray<__dlat_array_t>;
	using hboollat_array_t = TNL::Containers::DistributedNDArray<__hboollat_array_t>;
	using dboollat_array_t = TNL::Containers::DistributedNDArray<__dboollat_array_t>;

	using hmacro_array_t = TNL::Containers::DistributedNDArray<__hmacro_array_t>;
	using dmacro_array_t = TNL::Containers::DistributedNDArray<__dmacro_array_t>;
#else
	using hmap_array_t = __hmap_array_t;
	using dmap_array_t = __dmap_array_t;
	using hbool_array_t = __hbool_array_t;
	using dbool_array_t = __dbool_array_t;
	using dreal_array_t = __dreal_array_t;
	using hreal_array_t = __hreal_array_t;

	using hlat_array_t = __hlat_array_t;
	using dlat_array_t = __dlat_array_t;
	using hboollat_array_t = __hboollat_array_t;
	using dboollat_array_t = __dboollat_array_t;

	using hmacro_array_t = __hmacro_array_t;
	using dmacro_array_t = __dmacro_array_t;
#endif

	using hmap_view_t = typename hmap_array_t::ViewType;
	using dmap_view_t = typename dmap_array_t::ViewType;

	using hlat_view_t = typename hlat_array_t::ViewType;
	using dlat_view_t = typename dlat_array_t::ViewType;
};

//#define USE_HIGH_PRECISION_RHO // use num value ordering to compute rho inlbm_common.h .. slow!!!
//#define USE_GALILEAN_CORRECTION // Geier 2015: use Gal correction in BKG and CUM?
//#define USE_GEIER_CUM_2017 // use Geier 2017 Cummulant improvement A,B terms
//#define USE_GEIER_CUM_ANTIALIAS // use antialiasing Dxu, Dyv, Dzw from Geier 2015/2017

enum : std::uint8_t
{
	// Q5
	zz = 0,
	pz = 1,
	mz = 2,
	zp = 3,
	zm = 4,
	// +Q9
	pp = 5,
	mm = 6,
	pm = 7,
	mp = 8,
};


// NOTE: df_sync_directions must be kept consistent with this enum!
enum : std::uint8_t
{
	mmm,
	mmz,
	mmp,
	mzm,
	mzz,
	mzp,
	mpm,
	mpz,
	mpp,
	zmm,
	zmz,
	zmp,
	zzm,
	zzz,
	zzp,
	zpm,
	zpz,
	zpp,
	pmm,
	pmz,
	pmp,
	pzm,
	pzz,
	pzp,
	ppm,
	ppz,
	ppp
};

// NOTE: df_sync_directions must be kept consistent with this enum!
enum : std::uint16_t
{
	mmm_mmm_mmm,
mmm_mmm_mm,
mmm_mmm_m,
mmm_mmm_z,
mmm_mmm_p,
mmm_mmm_pp,
mmm_mmm_ppp,
mmm_mm_mmm,
mmm_mm_mm,
mmm_mm_m,
mmm_mm_z,
mmm_mm_p,
mmm_mm_pp,
mmm_mm_ppp,
mmm_m_mmm,
mmm_m_mm,
mmm_m_m,
mmm_m_z,
mmm_m_p,
mmm_m_pp,
mmm_m_ppp,
mmm_z_mmm,
mmm_z_mm,
mmm_z_m,
mmm_z_z,
mmm_z_p,
mmm_z_pp,
mmm_z_ppp,
mmm_p_mmm,
mmm_p_mm,
mmm_p_m,
mmm_p_z,
mmm_p_p,
mmm_p_pp,
mmm_p_ppp,
mmm_pp_mmm,
mmm_pp_mm,
mmm_pp_m,
mmm_pp_z,
mmm_pp_p,
mmm_pp_pp,
mmm_pp_ppp,
mmm_ppp_mmm,
mmm_ppp_mm,
mmm_ppp_m,
mmm_ppp_z,
mmm_ppp_p,
mmm_ppp_pp,
mmm_ppp_ppp,
mm_mmm_mmm,
mm_mmm_mm,
mm_mmm_m,
mm_mmm_z,
mm_mmm_p,
mm_mmm_pp,
mm_mmm_ppp,
mm_mm_mmm,
mm_mm_mm,
mm_mm_m,
mm_mm_z,
mm_mm_p,
mm_mm_pp,
mm_mm_ppp,
mm_m_mmm,
mm_m_mm,
mm_m_m,
mm_m_z,
mm_m_p,
mm_m_pp,
mm_m_ppp,
mm_z_mmm,
mm_z_mm,
mm_z_m,
mm_z_z,
mm_z_p,
mm_z_pp,
mm_z_ppp,
mm_p_mmm,
mm_p_mm,
mm_p_m,
mm_p_z,
mm_p_p,
mm_p_pp,
mm_p_ppp,
mm_pp_mmm,
mm_pp_mm,
mm_pp_m,
mm_pp_z,
mm_pp_p,
mm_pp_pp,
mm_pp_ppp,
mm_ppp_mmm,
mm_ppp_mm,
mm_ppp_m,
mm_ppp_z,
mm_ppp_p,
mm_ppp_pp,
mm_ppp_ppp,
m_mmm_mmm,
m_mmm_mm,
m_mmm_m,
m_mmm_z,
m_mmm_p,
m_mmm_pp,
m_mmm_ppp,
m_mm_mmm,
m_mm_mm,
m_mm_m,
m_mm_z,
m_mm_p,
m_mm_pp,
m_mm_ppp,
m_m_mmm,
m_m_mm,
m_m_m,
m_m_z,
m_m_p,
m_m_pp,
m_m_ppp,
m_z_mmm,
m_z_mm,
m_z_m,
m_z_z,
m_z_p,
m_z_pp,
m_z_ppp,
m_p_mmm,
m_p_mm,
m_p_m,
m_p_z,
m_p_p,
m_p_pp,
m_p_ppp,
m_pp_mmm,
m_pp_mm,
m_pp_m,
m_pp_z,
m_pp_p,
m_pp_pp,
m_pp_ppp,
m_ppp_mmm,
m_ppp_mm,
m_ppp_m,
m_ppp_z,
m_ppp_p,
m_ppp_pp,
m_ppp_ppp,
z_mmm_mmm,
z_mmm_mm,
z_mmm_m,
z_mmm_z,
z_mmm_p,
z_mmm_pp,
z_mmm_ppp,
z_mm_mmm,
z_mm_mm,
z_mm_m,
z_mm_z,
z_mm_p,
z_mm_pp,
z_mm_ppp,
z_m_mmm,
z_m_mm,
z_m_m,
z_m_z,
z_m_p,
z_m_pp,
z_m_ppp,
z_z_mmm,
z_z_mm,
z_z_m,
z_z_z,
z_z_p,
z_z_pp,
z_z_ppp,
z_p_mmm,
z_p_mm,
z_p_m,
z_p_z,
z_p_p,
z_p_pp,
z_p_ppp,
z_pp_mmm,
z_pp_mm,
z_pp_m,
z_pp_z,
z_pp_p,
z_pp_pp,
z_pp_ppp,
z_ppp_mmm,
z_ppp_mm,
z_ppp_m,
z_ppp_z,
z_ppp_p,
z_ppp_pp,
z_ppp_ppp,
p_mmm_mmm,
p_mmm_mm,
p_mmm_m,
p_mmm_z,
p_mmm_p,
p_mmm_pp,
p_mmm_ppp,
p_mm_mmm,
p_mm_mm,
p_mm_m,
p_mm_z,
p_mm_p,
p_mm_pp,
p_mm_ppp,
p_m_mmm,
p_m_mm,
p_m_m,
p_m_z,
p_m_p,
p_m_pp,
p_m_ppp,
p_z_mmm,
p_z_mm,
p_z_m,
p_z_z,
p_z_p,
p_z_pp,
p_z_ppp,
p_p_mmm,
p_p_mm,
p_p_m,
p_p_z,
p_p_p,
p_p_pp,
p_p_ppp,
p_pp_mmm,
p_pp_mm,
p_pp_m,
p_pp_z,
p_pp_p,
p_pp_pp,
p_pp_ppp,
p_ppp_mmm,
p_ppp_mm,
p_ppp_m,
p_ppp_z,
p_ppp_p,
p_ppp_pp,
p_ppp_ppp,
pp_mmm_mmm,
pp_mmm_mm,
pp_mmm_m,
pp_mmm_z,
pp_mmm_p,
pp_mmm_pp,
pp_mmm_ppp,
pp_mm_mmm,
pp_mm_mm,
pp_mm_m,
pp_mm_z,
pp_mm_p,
pp_mm_pp,
pp_mm_ppp,
pp_m_mmm,
pp_m_mm,
pp_m_m,
pp_m_z,
pp_m_p,
pp_m_pp,
pp_m_ppp,
pp_z_mmm,
pp_z_mm,
pp_z_m,
pp_z_z,
pp_z_p,
pp_z_pp,
pp_z_ppp,
pp_p_mmm,
pp_p_mm,
pp_p_m,
pp_p_z,
pp_p_p,
pp_p_pp,
pp_p_ppp,
pp_pp_mmm,
pp_pp_mm,
pp_pp_m,
pp_pp_z,
pp_pp_p,
pp_pp_pp,
pp_pp_ppp,
pp_ppp_mmm,
pp_ppp_mm,
pp_ppp_m,
pp_ppp_z,
pp_ppp_p,
pp_ppp_pp,
pp_ppp_ppp,
ppp_mmm_mmm,
ppp_mmm_mm,
ppp_mmm_m,
ppp_mmm_z,
ppp_mmm_p,
ppp_mmm_pp,
ppp_mmm_ppp,
ppp_mm_mmm,
ppp_mm_mm,
ppp_mm_m,
ppp_mm_z,
ppp_mm_p,
ppp_mm_pp,
ppp_mm_ppp,
ppp_m_mmm,
ppp_m_mm,
ppp_m_m,
ppp_m_z,
ppp_m_p,
ppp_m_pp,
ppp_m_ppp,
ppp_z_mmm,
ppp_z_mm,
ppp_z_m,
ppp_z_z,
ppp_z_p,
ppp_z_pp,
ppp_z_ppp,
ppp_p_mmm,
ppp_p_mm,
ppp_p_m,
ppp_p_z,
ppp_p_p,
ppp_p_pp,
ppp_p_ppp,
ppp_pp_mmm,
ppp_pp_mm,
ppp_pp_m,
ppp_pp_z,
ppp_pp_p,
ppp_pp_pp,
ppp_pp_ppp,
ppp_ppp_mmm,
ppp_ppp_mm,
ppp_ppp_m,
ppp_ppp_z,
ppp_ppp_p,
ppp_ppp_pp,
ppp_ppp_ppp
};

// SYNC DIRECTIONS NOW IMPORTED IN SIMULATION
