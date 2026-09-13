/*
 * MPI unit tests for the streaming patterns' exchange machinery: all six
 * patterns (A-A, A-B pull, A-B push, esoteric twist/pull/push) instantiated
 * in ONE translation unit, for both D2Q9 and D3Q27, each compared against
 * the A-B pull instance of the same binary (the built-in reference) on a
 * small periodic Taylor-Green vortex.
 *
 * The per-iteration work is UNFUSED: a streaming pattern changes where
 * populations live in memory, never the arithmetic, so the heavy collision
 * operator does not need a per-pattern instantiation. Each iteration runs
 * three kernels replicating the production SimUpdate sequence:
 *   1. a per-pattern GATHER kernel that only invokes the pattern's
 *      STREAMING::streaming (population loads into the KernelStruct
 *      registers) and stores the natural-layout populations into a scratch
 *      array,
 *   2. one SHARED collision kernel per lattice model (2 instantiations for
 *      the whole binary, not 12) that reads the natural populations back,
 *      computes macroscopic quantities, applies the SRT collision and
 *      writes both results out,
 *   3. a per-pattern SCATTER kernel that loads the post-collision natural
 *      populations and calls the pattern's own postCollisionStreaming.
 * The gather/scatter kernels are pure data movement, so every pattern
 * consumes bitwise-identical collision output by construction and any
 * cross-pattern difference must originate in the streaming placement or
 * the halo exchange - exactly what this test asserts.
 *
 * Initialization reuses LBM_BLOCK::setInitialCondition (the virtual "-1 ->
 * 0" iteration producing each pattern's parity-0 layout from the common
 * equilibrium; EsoTwist's staged placement included). The exchange
 * machinery is exercised exactly as in production: LBM::updateKernelData
 * parity/array rotation every iteration, region launches mirroring
 * State::SimUpdate (boundary slabs on their streams, interior last),
 * boundary streams joined before LBM::synchronizeDFsAndMacroDevice runs
 * the pipelined per-slot DF exchange with the parity-dependent
 * dfSyncDirection/dfSyncOffset descriptors plus the macro exchange, and
 * the interior stream joined afterwards - at BOTH parities (iterations
 * alternate even_iter by construction).
 *
 * The initial and final macroscopic fields (owned sites plus, along
 * distributed axes, the exchanged ghost planes) must be BITWISE identical
 * between every pattern and A-B pull.
 *
 * The multi-rank cases force multi-dimensional decompositions with the
 * TNL_LBM_FORCE_DECOMPOSITION hook (2x2 for 2D at np=4, 2x2x2 for 3D at
 * np=8), because only multi-axis cuts allocate the diagonal edge/corner
 * exchange buffers where the per-slot mask restriction, the EsoTwist staged
 * passes and the shared per-buffer sequencing stream apply. The A-A
 * multi-dimensional cases are separate test cases on purpose: A-A was the
 * last pattern to fail bitwise identity under multi-dimensional
 * decompositions (see AGENTS.md, "Known limitations under A-A"), and with
 * the include/lbm3d/ fix these rows assert the correct behavior
 * unconditionally - their isolation keeps a regression attributable to A-A
 * alone.
 *
 * The BC-dispatch cases (compiled in the separate per-model translation
 * units test_streaming_mpi_bc2d.cu / test_streaming_mpi_bc3d.cu) run the
 * same unfused structure on a non-periodic channel (walls, a moment inflow
 * and a two-pass outflow under the ghost-layer idiom), but with the
 * production BC dispatch in the kernels: the gather calls
 * NSE::BC::preCollision (the full GEO-tag switch around the pattern's
 * streaming), the shared collision applies the production BC::doCollision
 * gate, the scatter calls BC::postCollision, and the deterministic two-pass
 * outflow is launched over each block's outflow_boxes before the main
 * kernels, mirroring State::SimUpdate. Every pattern present must again
 * match A-B pull bitwise on the initial and final macroscopic fields; this
 * asserts pattern-equivalence of the BC dispatch (shared BC code across
 * patterns), never that the unfused pipeline reproduces the fused
 * production kernels' FP contractions - that contract is owned by the
 * test_nse/test_d2q9 regression suites.
 *
 * The frame-0 output cases target the historical bug class at the
 * (pattern init layout) x (output serialization) intersection - t=0
 * macroscopic quantities serialized before the pattern's init permutation
 * authored them. The TGV field is initialized through the production
 * setInitialCondition and the initial DF+macro halo exchange, copied to
 * the host, and the block's host macro arrays are passed through the real
 * output machinery WITHOUT instantiating State: variables defined with
 * the production global-shape / block-offset / block-extent selections
 * (State::predefine3D), one BP5 step written via DataManager, the file
 * read back with adios2 in random-access mode. Each pattern's round-trip
 * must reproduce its in-memory host macro bitwise, and every pattern's
 * round-trip must match the A-B pull instance bitwise.
 *
 * Additional host-only exchange-machinery assertions (single-rank): the
 * ESO_PULL/ESO_PUSH dfSyncDirection/dfSyncOffset descriptors against their
 * documented semantics, and the per-slot exchange-mask partition invariants
 * for all patterns whose setLatticeDecomposition restricts the DF exchange
 * buffers (esoteric in-place, A-B push) - every halo direction must be
 * carried by exactly one slot per parity phase.
 */
#include "test_streaming_mpi_common.h"
TEST_SUITE_BEGIN("streamingmpi");

TEST_CASE("TGV 2D all patterns vs AB_PULL (single-rank)")
{
	ForcedDecomposition hook(nullptr);
	const TGVSetup d2d{16, 16, 1, 100};
	checkTGV<
		COLL_CONFIG2D,
		CONFIG2D<D2Q9_STREAMING_AB_PULL<TRAITS>>,
		CONFIG2D<D2Q9_STREAMING_AA<TRAITS>>,
		CONFIG2D<D2Q9_STREAMING_AB_PUSH<TRAITS>>,
		CONFIG2D<D2Q9_STREAMING_ESO_TWIST<TRAITS>>,
		CONFIG2D<D2Q9_STREAMING_ESO_PULL<TRAITS>>,
		CONFIG2D<D2Q9_STREAMING_ESO_PUSH<TRAITS>>>("d2q9", d2d, {"aa", "ab_push", "eso_twist", "eso_pull", "eso_push"});
}

#ifdef HAVE_MPI

TEST_CASE("TGV 2D all patterns vs AB_PULL multi-rank np2")
{
	// 1D split: only the distributed axis carries overlap; A-A must pass
	ForcedDecomposition hook(nullptr);
	const TGVSetup d2d{16, 16, 1, 100};
	checkTGV<
		COLL_CONFIG2D,
		CONFIG2D<D2Q9_STREAMING_AB_PULL<TRAITS>>,
		CONFIG2D<D2Q9_STREAMING_AA<TRAITS>>,
		CONFIG2D<D2Q9_STREAMING_AB_PUSH<TRAITS>>,
		CONFIG2D<D2Q9_STREAMING_ESO_TWIST<TRAITS>>,
		CONFIG2D<D2Q9_STREAMING_ESO_PULL<TRAITS>>,
		CONFIG2D<D2Q9_STREAMING_ESO_PUSH<TRAITS>>>("d2q9", d2d, {"aa", "ab_push", "eso_twist", "eso_pull", "eso_push"});
}

TEST_CASE("TGV 2D non-AA patterns vs AB_PULL multi-rank 2x2 np4")
{
	// 2x2 split: diagonal ghost-corner exchanges fire for the first time
	ForcedDecomposition hook("2,2,1");
	const TGVSetup d2d{16, 16, 1, 100};
	checkTGV<
		COLL_CONFIG2D,
		CONFIG2D<D2Q9_STREAMING_AB_PULL<TRAITS>>,
		CONFIG2D<D2Q9_STREAMING_AB_PUSH<TRAITS>>,
		CONFIG2D<D2Q9_STREAMING_ESO_TWIST<TRAITS>>,
		CONFIG2D<D2Q9_STREAMING_ESO_PULL<TRAITS>>,
		CONFIG2D<D2Q9_STREAMING_ESO_PUSH<TRAITS>>>("d2q9", d2d, {"ab_push", "eso_twist", "eso_pull", "eso_push"});
}

TEST_CASE("TGV 2D AA vs AB_PULL multi-rank 2x2 np4")
{
	ForcedDecomposition hook("2,2,1");
	const TGVSetup d2d{16, 16, 1, 100};
	checkTGV<COLL_CONFIG2D, CONFIG2D<D2Q9_STREAMING_AB_PULL<TRAITS>>, CONFIG2D<D2Q9_STREAMING_AA<TRAITS>>>("d2q9", d2d, {"aa"});
}

#endif	// HAVE_MPI

TEST_SUITE_END();

TEST_SUITE_BEGIN("streamingmpi3d");

TEST_CASE("TGV 3D all patterns vs AB_PULL (single-rank)")
{
	ForcedDecomposition hook(nullptr);
	const TGVSetup d3d{8, 8, 8, 60};
	checkTGV<
		COLL_CONFIG3D,
		CONFIG3D<D3Q27_STREAMING_AB_PULL<TRAITS>>,
		CONFIG3D<D3Q27_STREAMING_AA<TRAITS>>,
		CONFIG3D<D3Q27_STREAMING_AB_PUSH<TRAITS>>,
		CONFIG3D<D3Q27_STREAMING_ESO_TWIST<TRAITS>>,
		CONFIG3D<D3Q27_STREAMING_ESO_PULL<TRAITS>>,
		CONFIG3D<D3Q27_STREAMING_ESO_PUSH<TRAITS>>>("d3q27", d3d, {"aa", "ab_push", "eso_twist", "eso_pull", "eso_push"});
}

#ifdef HAVE_MPI

TEST_CASE("TGV 3D all patterns vs AB_PULL multi-rank np2")
{
	ForcedDecomposition hook(nullptr);
	const TGVSetup d3d{8, 8, 8, 60};
	checkTGV<
		COLL_CONFIG3D,
		CONFIG3D<D3Q27_STREAMING_AB_PULL<TRAITS>>,
		CONFIG3D<D3Q27_STREAMING_AA<TRAITS>>,
		CONFIG3D<D3Q27_STREAMING_AB_PUSH<TRAITS>>,
		CONFIG3D<D3Q27_STREAMING_ESO_TWIST<TRAITS>>,
		CONFIG3D<D3Q27_STREAMING_ESO_PULL<TRAITS>>,
		CONFIG3D<D3Q27_STREAMING_ESO_PUSH<TRAITS>>>("d3q27", d3d, {"aa", "ab_push", "eso_twist", "eso_pull", "eso_push"});
}

TEST_CASE("TGV 3D non-AA patterns vs AB_PULL multi-rank 2x2x2 np8")
{
	// 2x2x2 split: all three axes cut, 3D edge and corner buffers fire
	ForcedDecomposition hook("2,2,2");
	const TGVSetup d3d{8, 8, 8, 10};
	checkTGV<
		COLL_CONFIG3D,
		CONFIG3D<D3Q27_STREAMING_AB_PULL<TRAITS>>,
		CONFIG3D<D3Q27_STREAMING_AB_PUSH<TRAITS>>,
		CONFIG3D<D3Q27_STREAMING_ESO_TWIST<TRAITS>>,
		CONFIG3D<D3Q27_STREAMING_ESO_PULL<TRAITS>>,
		CONFIG3D<D3Q27_STREAMING_ESO_PUSH<TRAITS>>>("d3q27", d3d, {"ab_push", "eso_twist", "eso_pull", "eso_push"});
}

TEST_CASE("TGV 3D AA vs AB_PULL multi-rank 2x2x2 np8")
{
	ForcedDecomposition hook("2,2,2");
	const TGVSetup d3d{8, 8, 8, 10};
	checkTGV<COLL_CONFIG3D, CONFIG3D<D3Q27_STREAMING_AB_PULL<TRAITS>>, CONFIG3D<D3Q27_STREAMING_AA<TRAITS>>>("d3q27", d3d, {"aa"});
}

#endif	// HAVE_MPI

TEST_SUITE_END();

TEST_SUITE_BEGIN("streamingmpiframe0");

TEST_CASE("frame-0 output 2D vs AB_PULL (single-rank)")
{
	ForcedDecomposition hook(nullptr);
	const TGVSetup d2d{16, 16, 1, 0};
	checkFrame0<
		CONFIG2D<D2Q9_STREAMING_AB_PULL<TRAITS>>,
		CONFIG2D<D2Q9_STREAMING_AA<TRAITS>>,
		CONFIG2D<D2Q9_STREAMING_AB_PUSH<TRAITS>>,
		CONFIG2D<D2Q9_STREAMING_ESO_TWIST<TRAITS>>,
		CONFIG2D<D2Q9_STREAMING_ESO_PULL<TRAITS>>,
		CONFIG2D<D2Q9_STREAMING_ESO_PUSH<TRAITS>>>("d2q9", d2d, {"aa", "ab_push", "eso_twist", "eso_pull", "eso_push"});
}

#ifdef HAVE_MPI

TEST_CASE("frame-0 output 2D vs AB_PULL multi-rank np2")
{
	// the block selections and the macro ghost exchange distribute with the
	// lattice; each rank writes and reads back its own selection
	ForcedDecomposition hook(nullptr);
	const TGVSetup d2d{16, 16, 1, 0};
	checkFrame0<
		CONFIG2D<D2Q9_STREAMING_AB_PULL<TRAITS>>,
		CONFIG2D<D2Q9_STREAMING_AA<TRAITS>>,
		CONFIG2D<D2Q9_STREAMING_AB_PUSH<TRAITS>>,
		CONFIG2D<D2Q9_STREAMING_ESO_TWIST<TRAITS>>,
		CONFIG2D<D2Q9_STREAMING_ESO_PULL<TRAITS>>,
		CONFIG2D<D2Q9_STREAMING_ESO_PUSH<TRAITS>>>("d2q9", d2d, {"aa", "ab_push", "eso_twist", "eso_pull", "eso_push"});
}

#endif	// HAVE_MPI

TEST_SUITE_END();

TEST_SUITE_BEGIN("streamingmpiframe03d");

TEST_CASE("frame-0 output 3D vs AB_PULL (single-rank)")
{
	ForcedDecomposition hook(nullptr);
	const TGVSetup d3d{8, 8, 8, 0};
	checkFrame0<
		CONFIG3D<D3Q27_STREAMING_AB_PULL<TRAITS>>,
		CONFIG3D<D3Q27_STREAMING_AA<TRAITS>>,
		CONFIG3D<D3Q27_STREAMING_AB_PUSH<TRAITS>>,
		CONFIG3D<D3Q27_STREAMING_ESO_TWIST<TRAITS>>,
		CONFIG3D<D3Q27_STREAMING_ESO_PULL<TRAITS>>,
		CONFIG3D<D3Q27_STREAMING_ESO_PUSH<TRAITS>>>("d3q27", d3d, {"aa", "ab_push", "eso_twist", "eso_pull", "eso_push"});
}

#ifdef HAVE_MPI

TEST_CASE("frame-0 output 3D vs AB_PULL multi-rank np2")
{
	ForcedDecomposition hook(nullptr);
	const TGVSetup d3d{8, 8, 8, 0};
	checkFrame0<
		CONFIG3D<D3Q27_STREAMING_AB_PULL<TRAITS>>,
		CONFIG3D<D3Q27_STREAMING_AA<TRAITS>>,
		CONFIG3D<D3Q27_STREAMING_AB_PUSH<TRAITS>>,
		CONFIG3D<D3Q27_STREAMING_ESO_TWIST<TRAITS>>,
		CONFIG3D<D3Q27_STREAMING_ESO_PULL<TRAITS>>,
		CONFIG3D<D3Q27_STREAMING_ESO_PUSH<TRAITS>>>("d3q27", d3d, {"aa", "ab_push", "eso_twist", "eso_pull", "eso_push"});
}

#endif	// HAVE_MPI

TEST_SUITE_END();

// velocity components per lattice model family (Q=9 shares dir9_*, the rest dir27_*)
static int dirComp(int slot, int axis, int Q)
{
	if (Q == 9) {
		if (axis == 0)
			return dir9_cx(slot);
		if (axis == 1)
			return dir9_cy(slot);
		return 0;
	}
	if (axis == 0)
		return dir27_cx(slot);
	if (axis == 1)
		return dir27_cy(slot);
	return dir27_cz(slot);
}

static int dirCount(int Q)
{
	return Q == 9 ? 2 : 3;
}

// face direction of one lattice axis: x -> Left/Right, y -> Bottom/Top, z -> Back/Front
static SDirection faceDirection(int axis, bool positive)
{
	static const SDirection POSITIVE[3] = {SDirection::Right, SDirection::Top, SDirection::Front};
	static const SDirection NEGATIVE[3] = {SDirection::Left, SDirection::Bottom, SDirection::Back};
	return positive ? POSITIVE[axis] : NEGATIVE[axis];
}

// combined mask of the canonical slot direction over all non-trivial axes
static SDirection canonicalMask(int slot, int Q, bool flip)
{
	SDirection mask = SDirection::None;
	for (int a = 0; a < dirCount(Q); a++) {
		const int c = dirComp(slot, a, Q);
		if (c == 0)
			continue;
		const bool positive = flip ? c < 0 : c > 0;
		mask = mask | faceDirection(a, positive);
	}
	return mask;
}

// direction table of the model family (canonical slot directions)
static const SDirection* canonicalDirections(int Q)
{
	return Q == 9 ? df_sync_directions_d2q9 : df_sync_directions;
}

// combined per-parity exchange mask of one slot, from the pattern's own descriptors
template <typename CONFIG>
static SDirection combinedSyncMask(int slot, bool even)
{
	using S = typename CONFIG::STREAMING;
	if constexpr (is_ESO_TWIST_v<S>) {
		SDirection mask = SDirection::None;
		for (int pass = 0; pass < 2; pass++)
			mask = mask | eso_twist_pass_mask<CONFIG>(slot, pass, even);
		return mask;
	}
	else {
		SDirection mask = SDirection::None;
		for (int a = 0; a < dirCount(CONFIG::Q); a++)
			mask = mask | S::dfSyncDirection(slot, a, even);
		return mask;
	}
}

// per-parity exchange masks of one slot, at the granularity the containment
// restriction in setLatticeDecomposition evaluates them: esoteric pull/push
// run one combined-mask pass per parity, EsoTwist runs two staged passes
// (mixed-sign diagonal payloads are carried by their face sub-buffers, never
// by a diagonal buffer), and A-B push exchanges the canonical slot direction
// and owns the opposite buffer for the stage_2 receive lookup
template <typename CONFIG>
static int slotSyncMasks(int slot, bool even, std::array<SDirection, 4>& masks)
{
	using S = typename CONFIG::STREAMING;
	int n = 0;
	if constexpr (is_ESO_TWIST_v<S>) {
		for (int pass = 0; pass < 2; pass++) {
			const SDirection mask = eso_twist_pass_mask<CONFIG>(slot, pass, even);
			if (mask != SDirection::None)
				masks[n++] = mask;
		}
	}
	else if constexpr (is_AB_PUSH_v<S>) {
		(void) even;
		const SDirection dir = canonicalDirections(CONFIG::Q)[slot];
		if (dir != SDirection::None) {
			masks[n++] = dir;
			masks[n++] = opposite(dir);
		}
	}
	else {
		SDirection mask = SDirection::None;
		for (int a = 0; a < dirCount(CONFIG::Q); a++)
			mask = mask | S::dfSyncDirection(slot, a, even);
		if (mask != SDirection::None)
			masks[n++] = mask;
	}
	return n;
}

template <typename CONFIG>
static void checkEsoDescriptorTable()
{
	using S = typename CONFIG::STREAMING;
	for (int slot = 0; slot < CONFIG::Q; slot++)
		for (int a = 0; a < dirCount(CONFIG::Q); a++)
			for (const bool even : {false, true}) {
				INFO("slot ", slot, " axis ", a, " even_iter ", even);
				const int c = dirComp(slot, a, CONFIG::Q);
				const SDirection dir = S::dfSyncDirection(slot, a, even);
				const int offset = S::dfSyncOffset(slot, a, even);
				// shift by one on the head side of the pair post-phase-2, on the tail side post-phase-1
				CHECK(offset == (even ? int(is_pair_head(slot)) : int(! is_pair_head(slot))));
				if (c == 0) {
					CHECK(dir == SDirection::None);
					continue;
				}
				// ESO_PULL: phase-1 layout is counter-canonical, phase-2 canonical; ESO_PUSH mirrors that
				const bool positive = is_ESO_PULL_v<S> ? (even ? c > 0 : c < 0) : (even ? c < 0 : c > 0);
				CHECK(dir == faceDirection(a, positive));
			}
}

template <typename CONFIG>
static void checkCombinedMasks()
{
	using S = typename CONFIG::STREAMING;
	for (int slot = 0; slot < CONFIG::Q; slot++)
		for (const bool even : {false, true}) {
			INFO("slot ", slot, " even_iter ", even);
			const SDirection expected =
				(is_ESO_PULL_v<S> || is_ESO_TWIST_v<S>) ? canonicalMask(slot, CONFIG::Q, ! even) : canonicalMask(slot, CONFIG::Q, even);
			CHECK(combinedSyncMask<CONFIG>(slot, even) == expected);
			if constexpr (is_ESO_TWIST_v<S>) {
				// the two staged passes partition the axes: disjoint masks
				const SDirection pass0 = eso_twist_pass_mask<CONFIG>(slot, 0, even);
				const SDirection pass1 = eso_twist_pass_mask<CONFIG>(slot, 1, even);
				CHECK((pass0 & pass1) == SDirection::None);
			}
		}
}

static int bitCount(SDirection dir)
{
	unsigned v = static_cast<unsigned>(dir);
	int bits = 0;
	while (v != 0) {
		bits += v & 1;
		v >>= 1;
	}
	return bits;
}

// all non-zero velocity components of the canonical slot share one sign
// (the uniform-sign diagonals are the only payloads EsoTwist can ship with a
// diagonal buffer; mixed-sign diagonal cells ride the pass-sequenced face
// buffers instead)
static bool uniformSignDirection(int slot, int Q)
{
	int sign = 0;
	for (int a = 0; a < dirCount(Q); a++) {
		const int c = dirComp(slot, a, Q);
		if (c == 0)
			continue;
		if (sign == 0)
			sign = c;
		else if ((sign < 0) != (c < 0))
			return false;
	}
	return sign != 0;
}

// Expected number of slots claiming a diagonal (multi-face) halo direction
// per parity phase, by pattern and geometry class:
// - esoteric pull/push: the corner (all-axes) payload is carried by exactly
//   one slot per phase (two carriers would overwrite each other
//   nondeterministically); in 3D an edge payload additionally rides the two
//   adjacent corner masks as a shared sub-buffer, hence three claimants,
// - EsoTwist ships a diagonal buffer only for the uniform-sign diagonals
//   (its staged passes split the axes of mixed-sign slots, whose diagonal
//   cells then ride the pass-sequenced face buffers instead), with the same
//   edge multiplicity as pull/push for the shipped ones,
// - zero diagonal claimants would leave the halo hole unshipped.
// Single-face directions are read-only shared spans that diagonal slots
// legitimately carry as sub-buffers, so only coverage (at least one
// claimant) is required of them.
template <typename CONFIG>
static void checkMaskPartition()
{
	using S = typename CONFIG::STREAMING;
	constexpr int DIM = (CONFIG::Q == 9) ? 2 : 3;
	for (const bool even : {false, true})
		for (int d = 0; d < CONFIG::Q; d++) {
			const SDirection direction = canonicalDirections(CONFIG::Q)[d];
			if (direction == SDirection::None)
				continue;
			int claims = 0;
			for (int slot = 0; slot < CONFIG::Q; slot++) {
				std::array<SDirection, 4> masks{};
				const int nmasks = slotSyncMasks<CONFIG>(slot, even, masks);
				for (int m = 0; m < nmasks; m++)
					if ((direction & masks[m]) == direction) {
						claims++;
						break;
					}
			}
			INFO("direction ", static_cast<unsigned>(direction), " even_iter ", even, " claimed by ", claims, " slots");
			const int bits = bitCount(direction);
			if (bits == 1) {
				CHECK(claims >= 1);
				continue;
			}
			if constexpr (is_ESO_TWIST_v<S>) {
				if (! uniformSignDirection(d, CONFIG::Q))
					CHECK(claims == 0);
				else if (DIM == 3 && bits == 2)
					CHECK(claims == 3);
				else
					CHECK(claims == 1);
			}
			else {
				if (DIM == 3 && bits == 2)
					CHECK(claims == 3);
				else
					CHECK(claims == 1);
			}
		}
}

TEST_SUITE_BEGIN("streamingsyncdesc");

TEST_CASE("DF sync descriptors: ESO_PULL/ESO_PUSH sanity")
{
	checkEsoDescriptorTable<CONFIG2D<D2Q9_STREAMING_ESO_PULL<TRAITS>>>();
	checkEsoDescriptorTable<CONFIG2D<D2Q9_STREAMING_ESO_PUSH<TRAITS>>>();
	checkEsoDescriptorTable<CONFIG3D<D3Q27_STREAMING_ESO_PULL<TRAITS>>>();
	checkEsoDescriptorTable<CONFIG3D<D3Q27_STREAMING_ESO_PUSH<TRAITS>>>();
}

TEST_CASE("DF sync mask invariants: containment partition")
{
	checkCombinedMasks<CONFIG2D<D2Q9_STREAMING_ESO_PULL<TRAITS>>>();
	checkCombinedMasks<CONFIG2D<D2Q9_STREAMING_ESO_PUSH<TRAITS>>>();
	checkCombinedMasks<CONFIG2D<D2Q9_STREAMING_ESO_TWIST<TRAITS>>>();
	checkCombinedMasks<CONFIG3D<D3Q27_STREAMING_ESO_PULL<TRAITS>>>();
	checkCombinedMasks<CONFIG3D<D3Q27_STREAMING_ESO_PUSH<TRAITS>>>();
	checkCombinedMasks<CONFIG3D<D3Q27_STREAMING_ESO_TWIST<TRAITS>>>();

	checkMaskPartition<CONFIG2D<D2Q9_STREAMING_ESO_PULL<TRAITS>>>();
	checkMaskPartition<CONFIG2D<D2Q9_STREAMING_ESO_PUSH<TRAITS>>>();
	checkMaskPartition<CONFIG2D<D2Q9_STREAMING_ESO_TWIST<TRAITS>>>();
	checkMaskPartition<CONFIG3D<D3Q27_STREAMING_ESO_PULL<TRAITS>>>();
	checkMaskPartition<CONFIG3D<D3Q27_STREAMING_ESO_PUSH<TRAITS>>>();
	checkMaskPartition<CONFIG3D<D3Q27_STREAMING_ESO_TWIST<TRAITS>>>();

	// A-A and the A-B patterns carry no descriptor machinery; their exchanges
	// key on these canonical slot-direction tables, which must agree with the
	// direction enums, and the table must close under the direction pairing
	// (opposite_direction matches the enum layout; see defs.h)
	for (int slot = 0; slot < 9; slot++) {
		INFO("slot ", slot);
		CHECK(df_sync_directions_d2q9[slot] == canonicalMask(slot, 9, false));
		CHECK(df_sync_directions_d2q9[opposite_direction(slot)] == opposite(df_sync_directions_d2q9[slot]));
	}
	for (int slot = 0; slot < 27; slot++) {
		INFO("slot ", slot);
		CHECK(df_sync_directions[slot] == canonicalMask(slot, 27, false));
		CHECK(df_sync_directions[opposite_direction(slot)] == opposite(df_sync_directions[slot]));
	}
}

TEST_SUITE_END();
