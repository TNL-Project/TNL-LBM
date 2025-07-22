#pragma once

#if ! defined(AB_PATTERN) && ! defined(AA_PATTERN)
	// TODO: update multidimensional MPI synchronization for AA pattern
	// (for the even time step which is similar to a "push scheme", we need to
	// avoid race conditions on the corners - set smaller buffer size and adjust the offsets)
	//#define AA_PATTERN
	#define AB_PATTERN
#endif

#include <cstdio>
#include <cstdlib>

#include <utility>

#include <TNL/Containers/StaticVector.h>
#include <TNL/Containers/NDArray.h>
#include <TNL/Containers/DistributedNDArray.h>
#include <TNL/Containers/DistributedNDArraySynchronizer.h>
#include <TNL/Containers/BlockPartitioning.h>
#include <TNL/MPI.h>
#include <TNL/Backend/Stream.h>
#include <TNL/Backend/Types.h>
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

// KernelStruct - D2Q9
template <typename REAL>
struct D2Q9_KernelStruct
{
	static constexpr int D = 2;
	static constexpr int Q = 9;
	REAL f[Q];
	REAL fx = 0, fy = 0;
	REAL vx = 0, vy = 0;
	REAL rho = 1.0, lbmViscosity = 1.0;
};

// KernelStruct - D3Q7
template <typename REAL>
struct D3Q7_KernelStruct
{
	static constexpr int D = 3;
	static constexpr int Q = 7;
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
	using DATA = _DATA;
	using COLL = _COLL;
	using EQ = _EQ;
	using STREAMING = _STREAMING;
	using BC = _BC<LBM_CONFIG>;
	using MACRO = _MACRO;

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
	// Q7
	zzz = 0,
	pzz = 1,
	mzz = 2,
	zpz = 3,
	zmz = 4,
	zzp = 5,
	zzm = 6,
	// +Q19
	ppz = 7,
	mmz = 8,
	pmz = 9,
	mpz = 10,
	pzp = 11,
	mzm = 12,
	pzm = 13,
	mzp = 14,
	zpp = 15,
	zmm = 16,
	zpm = 17,
	zmp = 18,
	// +Q27
	ppp = 19,
	mmm = 20,
	ppm = 21,
	mmp = 22,
	pmp = 23,
	mpm = 24,
	pmm = 25,
	mpp = 26,
};

// array of sync directions for the MPI synchronizer
// (indexing must correspond to the enum above)
inline constexpr TNL::Containers::SyncDirection df_sync_directions[27] = {
	// Q7
	TNL::Containers::SyncDirection::None,
	TNL::Containers::SyncDirection::Right,
	TNL::Containers::SyncDirection::Left,
	TNL::Containers::SyncDirection::Top,
	TNL::Containers::SyncDirection::Bottom,
	TNL::Containers::SyncDirection::Front,
	TNL::Containers::SyncDirection::Back,
	// +Q19
	TNL::Containers::SyncDirection::TopRight,
	TNL::Containers::SyncDirection::BottomLeft,
	TNL::Containers::SyncDirection::BottomRight,
	TNL::Containers::SyncDirection::TopLeft,
	TNL::Containers::SyncDirection::FrontRight,
	TNL::Containers::SyncDirection::BackLeft,
	TNL::Containers::SyncDirection::BackRight,
	TNL::Containers::SyncDirection::FrontLeft,
	TNL::Containers::SyncDirection::FrontTop,
	TNL::Containers::SyncDirection::BackBottom,
	TNL::Containers::SyncDirection::BackTop,
	TNL::Containers::SyncDirection::FrontBottom,
	// +Q27
	TNL::Containers::SyncDirection::FrontTopRight,
	TNL::Containers::SyncDirection::BackBottomLeft,
	TNL::Containers::SyncDirection::BackTopRight,
	TNL::Containers::SyncDirection::FrontBottomLeft,
	TNL::Containers::SyncDirection::FrontBottomRight,
	TNL::Containers::SyncDirection::BackTopLeft,
	TNL::Containers::SyncDirection::BackBottomRight,
	TNL::Containers::SyncDirection::FrontTopLeft,
};
