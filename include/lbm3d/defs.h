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

// helper function for getting a 3D array view from a 4D distributed array
template <typename TRAITS, typename Array>
auto getMacroView(const Array& array, std::uint8_t id)
{
	using holder_t = TNL::Containers::SizesHolder<typename TRAITS::idx, 0, 0, 0>;
	using local_array_t = typename TRAITS::template array3d<typename Array::ValueType, typename Array::DeviceType>;
	using local_view_t = typename local_array_t::ViewType;
#ifdef HAVE_MPI
	using view_t = TNL::Containers::DistributedNDArrayView<local_view_t>;
#else
	using view_t = local_view_t;
#endif

#ifdef HAVE_MPI
	const auto local_4d_view = array.getConstLocalView();
#else
	const auto local_4d_view = array.getConstView();
#endif

	holder_t sizes;
	sizes.template setSize<0>(local_4d_view.template getSize<1>());
	sizes.template setSize<1>(local_4d_view.template getSize<2>());
	sizes.template setSize<2>(local_4d_view.template getSize<3>());

	holder_t strides;
	// NOTE: this works only because the static 0-dimension is the slowest
	strides.template setSize<0>(local_4d_view.template getStride<1>());
	strides.template setSize<1>(local_4d_view.template getStride<2>());
	strides.template setSize<2>(local_4d_view.template getStride<3>());

	typename TRAITS::xyz_overlaps overlaps;
#ifdef HAVE_MPI
	overlaps.template setSize<0>(array.getOverlaps().template getSize<1>());
	overlaps.template setSize<1>(array.getOverlaps().template getSize<2>());
	overlaps.template setSize<2>(array.getOverlaps().template getSize<3>());
#endif

	const typename Array::IndexType offset =
		local_4d_view.getStorageIndex(id, -overlaps.template getSize<0>(), -overlaps.template getSize<1>(), -overlaps.template getSize<2>());
	const typename Array::ValueType* begin = local_4d_view.getData() + offset;

	// getSubarrayView does not handle overlaps so we must get the subarray view this way
	local_view_t local_view(const_cast<typename Array::ValueType*>(begin), sizes, strides, overlaps);

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
	static constexpr int Q = 7;
	REAL f[Q];
	REAL vz = 0, vx = 0, vy = 0;
	REAL phi = 1.0, lbmViscosity = 1.0;
	// FIXME
	//REAL qcrit=0, phigradmag2=0;
};

// KernelStruct - D3Q27
template <typename REAL>
struct D3Q27_KernelStruct
{
	static constexpr int Q = 27;
	REAL f[Q];
	REAL fz = 0, fx = 0, fy = 0;
	REAL vz = 0, vx = 0, vy = 0;
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
	using DATA = _DATA;
	using COLL = _COLL;
	using EQ = _EQ;
	using STREAMING = _STREAMING;
	using BC = _BC<LBM_CONFIG>;
	using MACRO = _MACRO;

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
