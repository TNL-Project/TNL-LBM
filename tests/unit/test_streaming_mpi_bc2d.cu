/*
 * D2Q9 BC-dispatch cases of the streaming-pattern MPI unit test (see
 * test_streaming_mpi.cu for the test documentation); separate translation
 * unit for compile-time parallelism across the lattice models.
 */
#include "test_streaming_mpi_common.h"

TEST_SUITE_BEGIN("streamingmpibc");

TEST_CASE("channel 2D wall+inflow+outflow right vs AB_PULL (single-rank)")
{
	ForcedDecomposition hook(nullptr);
	const ChannelSetup d2d{24, 12, 1, 60};
	checkChannel<
		COLL_CONFIG2D,
		CONFIG2D<D2Q9_STREAMING_AB_PULL<TRAITS>>,
		CONFIG2D<D2Q9_STREAMING_AA<TRAITS>>,
		CONFIG2D<D2Q9_STREAMING_AB_PUSH<TRAITS>>,
		CONFIG2D<D2Q9_STREAMING_ESO_TWIST<TRAITS>>,
		CONFIG2D<D2Q9_STREAMING_ESO_PULL<TRAITS>>,
		CONFIG2D<D2Q9_STREAMING_ESO_PUSH<TRAITS>>>("d2q9", d2d, {"aa", "ab_push", "eso_twist", "eso_pull", "eso_push"});
}

TEST_CASE("channel 2D wall+inflow+outflow right interp vs AB_PULL (single-rank)")
{
	ForcedDecomposition hook(nullptr);
	const ChannelSetup d2d{24, 12, 1, 60, 0.01, 1.0, 1.5e-5, 5e-3, true};
	checkChannel<
		COLL_CONFIG2D,
		CONFIG2D<D2Q9_STREAMING_AB_PULL<TRAITS>>,
		CONFIG2D<D2Q9_STREAMING_AA<TRAITS>>,
		CONFIG2D<D2Q9_STREAMING_AB_PUSH<TRAITS>>,
		CONFIG2D<D2Q9_STREAMING_ESO_TWIST<TRAITS>>,
		CONFIG2D<D2Q9_STREAMING_ESO_PULL<TRAITS>>,
		CONFIG2D<D2Q9_STREAMING_ESO_PUSH<TRAITS>>>("d2q9", d2d, {"aa", "ab_push", "eso_twist", "eso_pull", "eso_push"});
}

#ifdef HAVE_MPI

TEST_CASE("channel 2D wall+inflow+outflow right vs AB_PULL multi-rank 2x2 np4")
{
	// forced 2x2 split: DF halo exists on both axes, the BC dispatch, the
	// map sync and the diagonal corner exchanges run across the interfaces,
	// and all six patterns init safely
	ForcedDecomposition hook("2,2,1");
	const ChannelSetup d2d{24, 12, 1, 60};
	checkChannel<
		COLL_CONFIG2D,
		CONFIG2D<D2Q9_STREAMING_AB_PULL<TRAITS>>,
		CONFIG2D<D2Q9_STREAMING_AA<TRAITS>>,
		CONFIG2D<D2Q9_STREAMING_AB_PUSH<TRAITS>>,
		CONFIG2D<D2Q9_STREAMING_ESO_TWIST<TRAITS>>,
		CONFIG2D<D2Q9_STREAMING_ESO_PULL<TRAITS>>,
		CONFIG2D<D2Q9_STREAMING_ESO_PUSH<TRAITS>>>("d2q9", d2d, {"aa", "ab_push", "eso_twist", "eso_pull", "eso_push"});
}

TEST_CASE("channel 2D wall+inflow+outflow right interp vs AB_PULL multi-rank 2x2 np4")
{
	ForcedDecomposition hook("2,2,1");
	const ChannelSetup d2d{24, 12, 1, 60, 0.01, 1.0, 1.5e-5, 5e-3, true};
	checkChannel<
		COLL_CONFIG2D,
		CONFIG2D<D2Q9_STREAMING_AB_PULL<TRAITS>>,
		CONFIG2D<D2Q9_STREAMING_AA<TRAITS>>,
		CONFIG2D<D2Q9_STREAMING_AB_PUSH<TRAITS>>,
		CONFIG2D<D2Q9_STREAMING_ESO_TWIST<TRAITS>>,
		CONFIG2D<D2Q9_STREAMING_ESO_PULL<TRAITS>>,
		CONFIG2D<D2Q9_STREAMING_ESO_PUSH<TRAITS>>>("d2q9", d2d, {"aa", "ab_push", "eso_twist", "eso_pull", "eso_push"});
}

#endif	// HAVE_MPI

TEST_SUITE_END();
