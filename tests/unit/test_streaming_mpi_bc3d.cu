/*
 * D3Q27 BC-dispatch cases of the streaming-pattern MPI unit test (see
 * test_streaming_mpi.cu for the test documentation); separate translation
 * unit for compile-time parallelism across the lattice models.
 */
#include "test_streaming_mpi_common.h"

TEST_SUITE_BEGIN("streamingmpibc3d");

TEST_CASE("channel 3D wall+inflow+outflow right vs AB_PULL (single-rank)")
{
	ForcedDecomposition hook(nullptr);
	const ChannelSetup d3d{24, 8, 8, 30};
	checkChannel<
		COLL_CONFIG3D,
		CONFIG3D<D3Q27_STREAMING_AB_PULL<TRAITS>>,
		CONFIG3D<D3Q27_STREAMING_AA<TRAITS>>,
		CONFIG3D<D3Q27_STREAMING_AB_PUSH<TRAITS>>,
		CONFIG3D<D3Q27_STREAMING_ESO_TWIST<TRAITS>>,
		CONFIG3D<D3Q27_STREAMING_ESO_PULL<TRAITS>>,
		CONFIG3D<D3Q27_STREAMING_ESO_PUSH<TRAITS>>>("d3q27", d3d, {"aa", "ab_push", "eso_twist", "eso_pull", "eso_push"});
}

TEST_CASE("channel 3D wall+inflow+outflow right interp vs AB_PULL (single-rank)")
{
	ForcedDecomposition hook(nullptr);
	const ChannelSetup d3d{24, 8, 8, 30, 0.01, 1.0, 1.5e-5, 5e-3, true};
	checkChannel<
		COLL_CONFIG3D,
		CONFIG3D<D3Q27_STREAMING_AB_PULL<TRAITS>>,
		CONFIG3D<D3Q27_STREAMING_AA<TRAITS>>,
		CONFIG3D<D3Q27_STREAMING_AB_PUSH<TRAITS>>,
		CONFIG3D<D3Q27_STREAMING_ESO_TWIST<TRAITS>>,
		CONFIG3D<D3Q27_STREAMING_ESO_PULL<TRAITS>>,
		CONFIG3D<D3Q27_STREAMING_ESO_PUSH<TRAITS>>>("d3q27", d3d, {"aa", "ab_push", "eso_twist", "eso_pull", "eso_push"});
}

// periodic tangential axis: the outflow plane spans the periodic seam rows,
// where the outflow-pass gathers must wrap the tangential coordinates across
// the global seam like kernelInitIndices does everywhere else
TEST_CASE("channel 3D periodic-y+inflow+outflow right vs AB_PULL (single-rank)")
{
	ForcedDecomposition hook(nullptr);
	ChannelSetup d3d{24, 8, 8, 30};
	d3d.periodic_y = true;
	checkChannel<
		COLL_CONFIG3D,
		CONFIG3D<D3Q27_STREAMING_AB_PULL<TRAITS>>,
		CONFIG3D<D3Q27_STREAMING_AA<TRAITS>>,
		CONFIG3D<D3Q27_STREAMING_AB_PUSH<TRAITS>>,
		CONFIG3D<D3Q27_STREAMING_ESO_TWIST<TRAITS>>,
		CONFIG3D<D3Q27_STREAMING_ESO_PULL<TRAITS>>,
		CONFIG3D<D3Q27_STREAMING_ESO_PUSH<TRAITS>>>("d3q27", d3d, {"aa", "ab_push", "eso_twist", "eso_pull", "eso_push"});
}

TEST_CASE("channel 3D periodic-y+inflow+outflow right interp vs AB_PULL (single-rank)")
{
	ForcedDecomposition hook(nullptr);
	ChannelSetup d3d{24, 8, 8, 30, 0.01, 1.0, 1.5e-5, 5e-3, true};
	d3d.periodic_y = true;
	checkChannel<
		COLL_CONFIG3D,
		CONFIG3D<D3Q27_STREAMING_AB_PULL<TRAITS>>,
		CONFIG3D<D3Q27_STREAMING_AA<TRAITS>>,
		CONFIG3D<D3Q27_STREAMING_AB_PUSH<TRAITS>>,
		CONFIG3D<D3Q27_STREAMING_ESO_TWIST<TRAITS>>,
		CONFIG3D<D3Q27_STREAMING_ESO_PULL<TRAITS>>,
		CONFIG3D<D3Q27_STREAMING_ESO_PUSH<TRAITS>>>("d3q27", d3d, {"aa", "ab_push", "eso_twist", "eso_pull", "eso_push"});
}

#ifdef HAVE_MPI

TEST_CASE("channel 3D wall+inflow+outflow right vs AB_PULL multi-rank 2x2x2 np8")
{
	// forced 2x2x2 split: DF halo exists on all three axes
	ForcedDecomposition hook("2,2,2");
	const ChannelSetup d3d{24, 8, 8, 10};
	checkChannel<
		COLL_CONFIG3D,
		CONFIG3D<D3Q27_STREAMING_AB_PULL<TRAITS>>,
		CONFIG3D<D3Q27_STREAMING_AA<TRAITS>>,
		CONFIG3D<D3Q27_STREAMING_AB_PUSH<TRAITS>>,
		CONFIG3D<D3Q27_STREAMING_ESO_TWIST<TRAITS>>,
		CONFIG3D<D3Q27_STREAMING_ESO_PULL<TRAITS>>,
		CONFIG3D<D3Q27_STREAMING_ESO_PUSH<TRAITS>>>("d3q27", d3d, {"aa", "ab_push", "eso_twist", "eso_pull", "eso_push"});
}

TEST_CASE("channel 3D wall+inflow+outflow right interp vs AB_PULL multi-rank 2x2x2 np8")
{
	ForcedDecomposition hook("2,2,2");
	const ChannelSetup d3d{24, 8, 8, 10, 0.01, 1.0, 1.5e-5, 5e-3, true};
	checkChannel<
		COLL_CONFIG3D,
		CONFIG3D<D3Q27_STREAMING_AB_PULL<TRAITS>>,
		CONFIG3D<D3Q27_STREAMING_AA<TRAITS>>,
		CONFIG3D<D3Q27_STREAMING_AB_PUSH<TRAITS>>,
		CONFIG3D<D3Q27_STREAMING_ESO_TWIST<TRAITS>>,
		CONFIG3D<D3Q27_STREAMING_ESO_PULL<TRAITS>>,
		CONFIG3D<D3Q27_STREAMING_ESO_PUSH<TRAITS>>>("d3q27", d3d, {"aa", "ab_push", "eso_twist", "eso_pull", "eso_push"});
}

// distributed periodic axis: the seam values arrive through the inter-rank
// halo synchronization, so the outflow-pass gathers must keep reading the
// raw neighbor column (the wrap engages only on non-distributed axes)
TEST_CASE("channel 3D periodic-y+inflow+outflow right vs AB_PULL multi-rank 2x2x2 np8")
{
	// forced 2x2x2 split: the periodic y axis is distributed and its seam is
	// covered by the DF halo exchange
	ForcedDecomposition hook("2,2,2");
	ChannelSetup d3d{24, 8, 8, 10};
	d3d.periodic_y = true;
	checkChannel<
		COLL_CONFIG3D,
		CONFIG3D<D3Q27_STREAMING_AB_PULL<TRAITS>>,
		CONFIG3D<D3Q27_STREAMING_AA<TRAITS>>,
		CONFIG3D<D3Q27_STREAMING_AB_PUSH<TRAITS>>,
		CONFIG3D<D3Q27_STREAMING_ESO_TWIST<TRAITS>>,
		CONFIG3D<D3Q27_STREAMING_ESO_PULL<TRAITS>>,
		CONFIG3D<D3Q27_STREAMING_ESO_PUSH<TRAITS>>>("d3q27", d3d, {"aa", "ab_push", "eso_twist", "eso_pull", "eso_push"});
}

TEST_CASE("channel 3D periodic-y+inflow+outflow right interp vs AB_PULL multi-rank 2x2x2 np8")
{
	ForcedDecomposition hook("2,2,2");
	ChannelSetup d3d{24, 8, 8, 10, 0.01, 1.0, 1.5e-5, 5e-3, true};
	d3d.periodic_y = true;
	checkChannel<
		COLL_CONFIG3D,
		CONFIG3D<D3Q27_STREAMING_AB_PULL<TRAITS>>,
		CONFIG3D<D3Q27_STREAMING_AA<TRAITS>>,
		CONFIG3D<D3Q27_STREAMING_AB_PUSH<TRAITS>>,
		CONFIG3D<D3Q27_STREAMING_ESO_TWIST<TRAITS>>,
		CONFIG3D<D3Q27_STREAMING_ESO_PULL<TRAITS>>,
		CONFIG3D<D3Q27_STREAMING_ESO_PUSH<TRAITS>>>("d3q27", d3d, {"aa", "ab_push", "eso_twist", "eso_pull", "eso_push"});
}

#endif	// HAVE_MPI

TEST_SUITE_END();
