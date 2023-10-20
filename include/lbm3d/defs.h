#pragma once

#if !defined(AB_PATTERN) && !defined(AA_PATTERN)
	#define AA_PATTERN
#endif

#include <cstdio>
#include <cstdlib>
#include <cmath>

#include <string>
#include <iostream>
#include <utility>

#include <png.h>

#include <TNL/Containers/StaticVector.h>
#include <TNL/Containers/NDArray.h>
#include <TNL/Containers/DistributedNDArray.h>
#include <TNL/Containers/DistributedNDArraySynchronizer.h>
#include <TNL/Containers/Partitioner.h>
#include <TNL/MPI.h>
#include <TNL/Cuda/Stream.h>

#include "../lbm_common/ciselnik.h"

using TNLMPI_INIT = TNL::MPI::ScopedInitializer;


#ifdef __CUDACC__
	#define CUDA_HOSTDEV __host__ __device__
	#define CUDA_HOSTDEV_NOINLINE CUDA_HOSTDEV __noinline__
#else
	#define CUDA_HOSTDEV
	#define CUDA_HOSTDEV_NOINLINE
#endif

#ifdef USE_CUDA
	#define checkCudaDevice TNL_CHECK_CUDA_DEVICE
	#include <cuda_profiler_api.h>
#endif // USE_CUDA


// number of dist. functions, default=2
// quick fix, use templates to define DFMAX ... through TRAITS maybe ?
#ifdef USE_DFMAX3
enum : uint8_t { df_cur, df_out, df_prev, DFMAX }; // special 3 dfs
#elif defined(AB_PATTERN)
enum : uint8_t { df_cur, df_out, DFMAX }; // default 2 dfs
#elif defined(AA_PATTERN)
enum : uint8_t { df_cur, DFMAX }; // default 1 dfs
#endif

#ifdef USE_CUDA
	using DeviceType = TNL::Devices::Cuda;
#else
	using DeviceType = TNL::Devices::Host;
#endif

template <
	typename _dreal = float,	// real number representation on GPU
	typename _real = double,	// real number representation on CPU
	typename _idx = long int,	// array index on CPU and GPU (can be very large)
	typename _map_t = short int,
	unsigned _overlap_width = 1
>
struct Traits
{
	using real = _real;
	using dreal = _dreal;
	using idx = _idx;
	using map_t = _map_t;
	using point_t = TNL::Containers::StaticVector< 3, real >;
	using idx3d = TNL::Containers::StaticVector< 3, idx >;
	using idx2d = TNL::Containers::StaticVector< 2, idx >;	// this is just for convenience

	using xyz_permutation = std::index_sequence< 0, 2, 1 >;		// x, z, y
	using d4_permutation = std::index_sequence< 0, 1, 3, 2 >;		// id, x, z, y

#ifdef HAVE_MPI
	using xyz_overlaps = std::index_sequence< _overlap_width, 0, 0 >;	// x, y, z
	using d4_overlaps = std::index_sequence< 0, _overlap_width, 0, 0 >;	// id, x, y, z
#else
	using xyz_overlaps = std::index_sequence< 0, 0, 0 >;	// x, y, z
	using d4_overlaps = std::index_sequence< 0, 0, 0, 0 >;	// id, x, y, z
#endif

	template< typename Value, typename Device >
	using array3d = TNL::Containers::NDArray<
		Value,
		TNL::Containers::SizesHolder< idx, 0, 0, 0 >,	// x, y, z
		xyz_permutation,
		Device,
		idx,
		xyz_overlaps >;
	template< std::size_t N, typename Value, typename Device >
	using array4d = TNL::Containers::NDArray<
		Value,
		TNL::Containers::SizesHolder< idx, N, 0, 0, 0 >,	// N, x, y, z
		d4_permutation,
		Device,
		idx,
		d4_overlaps >;

	using xyz_indexer_t = typename array3d<dreal, DeviceType>::IndexerType;
};

using TraitsSP = Traits<float>; //_dreal is float only
using TraitsDP = Traits<double>;

// KernelStruct - D3Q7
template < typename REAL >
struct D3Q7_KernelStruct
{
	static constexpr int Q = 7;
	REAL f[Q];
	REAL vz=0, vx=0, vy=0;
	REAL phi=1.0, lbmViscosity=1.0;
	// FIXME
//	REAL qcrit=0, phigradmag2=0;
};

// KernelStruct - D3Q27
template < typename REAL >
struct D3Q27_KernelStruct
{
	static constexpr int Q = 27;
	REAL f[Q];
	REAL fz=0, fx=0, fy=0;
	REAL vz=0, vx=0, vy=0;
	REAL rho=1.0, lbmViscosity=1.0;

#if defined(USE_CYMODEL) || defined(USE_CASSON)
	REAL S11=0.,S12=0.,S22=0.,S32=0.,S13=0.,S33=0.;

	//Non-Newtonian parameters
	#if defined(USE_CYMODEL)
	REAL lbm_nu0=0, lbm_lambda=0, lbm_a=0, lbm_n=0;
	#elif defined(USE_CASSON)
	REAL lbm_k0=0, lbm_k1=0;
	#endif

	REAL mu;
#endif
};

template<
	typename _TRAITS,
	template<typename> class _KERNEL_STRUCT,
	typename _DATA,
	typename _COLL,
	typename _EQ,
	typename _STREAMING,
	template<typename> class _BC,
	typename _MACRO,
	typename _CPU_MACRO
>
struct LBM_CONFIG
{
	using TRAITS = _TRAITS;
	template< typename REAL >
	using KernelStruct = _KERNEL_STRUCT<REAL>;
	using DATA = _DATA;
	using COLL = _COLL;
	using EQ = _EQ;
	using STREAMING = _STREAMING;
	using BC = _BC<LBM_CONFIG>;
	using MACRO = _MACRO;
	using CPU_MACRO = _CPU_MACRO;

	static constexpr int Q = KernelStruct<typename TRAITS::dreal>::Q;

	using __hmap_array_t = typename TRAITS::template array3d<typename TRAITS::map_t, TNL::Devices::Host>;
	using __dmap_array_t = typename TRAITS::template array3d<typename TRAITS::map_t, DeviceType>;
	using __bool_array_t = typename TRAITS::template array3d<bool, TNL::Devices::Host>;
	using __dbool_array_t = typename TRAITS::template array3d<bool, DeviceType>;
	using __hreal_array_t = typename TRAITS::template array3d<typename TRAITS::dreal, TNL::Devices::Host>;
	using __dreal_array_t = typename TRAITS::template array3d<typename TRAITS::dreal, DeviceType>;

	using __hlat_array_t = typename TRAITS::template array4d<Q, typename TRAITS::dreal, TNL::Devices::Host>;
	using __dlat_array_t = typename TRAITS::template array4d<Q, typename TRAITS::dreal, DeviceType>;
	using __boollat_array_t = typename TRAITS::template array4d<Q,bool, DeviceType>;
	using __hboollat_array_t = typename TRAITS::template array4d<Q,bool, TNL::Devices::Host>;

	using __hmacro_array_t = typename TRAITS::template array4d<MACRO::N, typename TRAITS::dreal, TNL::Devices::Host>;
	using __dmacro_array_t = typename TRAITS::template array4d<MACRO::N, typename TRAITS::dreal, DeviceType>;
	using __cpumacro_array_t = typename TRAITS::template array4d<CPU_MACRO::N, typename TRAITS::dreal, TNL::Devices::Host>;

#ifdef HAVE_MPI
	using sync_array_t = TNL::Containers::DistributedNDArray< typename TRAITS::template array3d<typename TRAITS::dreal, DeviceType > >;

	using hmap_array_t = TNL::Containers::DistributedNDArray< __hmap_array_t >;
	using dmap_array_t = TNL::Containers::DistributedNDArray< __dmap_array_t >;
	using bool_array_t = TNL::Containers::DistributedNDArray< __bool_array_t >;
	using dbool_array_t = TNL::Containers::DistributedNDArray< __dbool_array_t >;
	using dreal_array_t = TNL::Containers::DistributedNDArray< __dreal_array_t >;
	using hreal_array_t = TNL::Containers::DistributedNDArray< __hreal_array_t >;


	using hlat_array_t = TNL::Containers::DistributedNDArray< __hlat_array_t >;
	using dlat_array_t = TNL::Containers::DistributedNDArray< __dlat_array_t >;
	using boollat_array_t = TNL::Containers::DistributedNDArray< __boollat_array_t >;
	using hboollat_array_t = TNL::Containers::DistributedNDArray< __hboollat_array_t >;

	using hmacro_array_t = TNL::Containers::DistributedNDArray< __hmacro_array_t >;
	using dmacro_array_t = TNL::Containers::DistributedNDArray< __dmacro_array_t >;
	using cpumacro_array_t = TNL::Containers::DistributedNDArray< __cpumacro_array_t >;
#else
	using sync_array_t = typename TRAITS::template array3d<typename TRAITS::dreal, DeviceType>;

	using hmap_array_t = __hmap_array_t;
	using dmap_array_t = __dmap_array_t;
	using bool_array_t = __bool_array_t;
	using dbool_array_t = __dbool_array_t;
	using dreal_array_t = __dreal_array_t;
	using hreal_array_t = __hreal_array_t;

	using hlat_array_t = __hlat_array_t;
	using dlat_array_t = __dlat_array_t;
	using boollat_array_t = __boollat_array_t;
	using hboollat_array_t = __hboollat_array_t;

	using hmacro_array_t = __hmacro_array_t;
	using dmacro_array_t = __dmacro_array_t;
	using cpumacro_array_t = __cpumacro_array_t;
#endif

	using hmap_view_t = typename hmap_array_t::ViewType;
	using dmap_view_t = typename dmap_array_t::ViewType;
	using bool_view_t = typename bool_array_t::ViewType;
	using dbool_view_t = typename dbool_array_t::ViewType;

	using hlat_view_t = typename hlat_array_t::ViewType;
	using dlat_view_t = typename dlat_array_t::ViewType;
};


//#define USE_HIGH_PRECISION_RHO // use num value ordering to compute rho inlbm_common.h .. slow!!!
//#define USE_GALILEAN_CORRECTION // Geier 2015: use Gal correction in BKG and CUM?
//#define USE_GEIER_CUM_2017 // use Geier 2017 Cummulant improvement A,B terms
//#define USE_GEIER_CUM_ANTIALIAS // use antialiasing Dxu, Dyv, Dzw from Geier 2015/2017

// TODO: replace these macros with functions TNL::min and TNL::max
#define MAX( a , b) (((a)>(b))?(a):(b))
#define MIN( a , b) (((a)<(b))?(a):(b))

#define SQ(x) ((x) * (x)) // square function; replaces SQ(x) by ((x) * (x)) in the code
#define NORM(x, y, z) sqrt(SQ(x) + SQ(y) + SQ(z))

enum { SOLVER_UMFPACK, SOLVER_PETSC };

// NOTE: df_sync_directions must be kept consistent with this enum!
enum
{
	// Q7
	zzz=0,
	pzz=1,
	mzz=2,
	zpz=3,
	zmz=4,
	zzp=5,
	zzm=6,
	// +Q19
	ppz=7,
	mmz=8,
	pmz=9,
	mpz=10,
	pzp=11,
	mzm=12,
	pzm=13,
	mzp=14,
	zpp=15,
	zmm=16,
	zpm=17,
	zmp=18,
	// +Q27
	ppp=19,
	mmm=20,
	ppm=21,
	mmp=22,
	pmp=23,
	mpm=24,
	pmm=25,
	mpp=26,
};

// static array of sync directions for the MPI synchronizer
// (indexing must correspond to the enum above)
static TNL::Containers::SyncDirection df_sync_directions[27] = {
	// Q7
	TNL::Containers::SyncDirection::None,
	TNL::Containers::SyncDirection::Right,
	TNL::Containers::SyncDirection::Left,
	TNL::Containers::SyncDirection::None,
	TNL::Containers::SyncDirection::None,
	TNL::Containers::SyncDirection::None,
	TNL::Containers::SyncDirection::None,
	// +Q19
	TNL::Containers::SyncDirection::Right,
	TNL::Containers::SyncDirection::Left,
	TNL::Containers::SyncDirection::Right,
	TNL::Containers::SyncDirection::Left,
	TNL::Containers::SyncDirection::Right,
	TNL::Containers::SyncDirection::Left,
	TNL::Containers::SyncDirection::Right,
	TNL::Containers::SyncDirection::Left,
	TNL::Containers::SyncDirection::None,
	TNL::Containers::SyncDirection::None,
	TNL::Containers::SyncDirection::None,
	TNL::Containers::SyncDirection::None,
	// +Q27
	TNL::Containers::SyncDirection::Right,
	TNL::Containers::SyncDirection::Left,
	TNL::Containers::SyncDirection::Right,
	TNL::Containers::SyncDirection::Left,
	TNL::Containers::SyncDirection::Right,
	TNL::Containers::SyncDirection::Left,
	TNL::Containers::SyncDirection::Right,
	TNL::Containers::SyncDirection::Left,
};


// default
#include "lbm_data.h"  // LBM_Data is a general template (for any Q)
#include "d3q27/macro.h"
#include "d3q27/bc.h"

#include "d3q27/eq.h"
#include "d3q27/eq_inv_cum.h"
#include "d3q27/eq_well.h"
#include "d3q27/eq_entropic.h"

// exactly one streaming header must be included
#ifdef AA_PATTERN
	#include "d3q27/streaming_AA.h"
#endif
#ifdef AB_PATTERN
	#include "d3q27/streaming_AB.h"
#endif

#include "d3q27/col_cum.h"
#include "d3q27/col_bgk.h"
#include "d3q27/col_clbm.h"
#include "d3q27/col_fclbm.h"
#include "d3q27/col_mrt.h"
#include "d3q27/col_srt.h"
#include "d3q27/col_cum_sgs.h"
#include "d3q27/col_kbc_n.h"
#include "d3q27/col_kbc_c.h"
#include "d3q27/col_srt_modif_force.h"
#include "d3q27/col_clbm_fei.h"

#include "d3q27/col_srt_well.h"
#include "d3q27/col_clbm_well.h"
#include "d3q27/col_cum_well.h"
#include "d3q27/col_bgk_well.h"
