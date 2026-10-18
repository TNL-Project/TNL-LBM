/*
 * AMR wall-refinement lane: failing-first in-situ gate of the frozen-cell
 * C2F read under the single-array streaming patterns (SP-1 of
 * results_drag_crisis/out_of_scope_survey.md; Defect-1 of
 * docs/AMR-for-LBM-implementation.md §10.1 with the D.4 revisit record in
 * §9.3).
 *
 * Surface: under the Schönherr-ch7 band registration the ONLY live
 * frozen-cell C2F read is the Sec-7.3 wall guard's shifted source window
 * {c = 0 ring, c = 1 frozen skin} at wall-adjacent coupling (the fully
 * periodic mL1 bitwise lock cannot see it: it has no physical-BC rows
 * adjacent to a footprint). The frozen GEO_NOTHING skin row is authored
 * ONLY by the once-per-cycle-end skin F2C (and by the pattern-native
 * initial-condition at cycle 0); it is never collided or streamed, so it
 * does not track the coarse-step parity of a single-array (in-place)
 * pattern the way live cells do. The C2F fill's coarse-side DF read
 * (`read_coarse_df` of cudaAMR_CoarseToFine,
 * include/lbm3d/d3q27/amr_coupling.h) resolves the source sites through
 * `STREAMING::postCollisionSlot(q, site, coarse_even_iter)` -- the parity
 * discipline of the JUST-RUN coarse launch. On a live cell that launch
 * re-wrote the slots in its own convention; on a frozen cell the slots
 * still hold the last coupling-authored (or kernel-pushed) content, whose
 * slot convention belongs to a different phase of the pattern's parity
 * pair -- the read-vs-store convention mismatch this suite measures
 * in-situ.
 *
 * Closed loop (the D.4 gap-list marker mock): per cycle N the transfer
 * phase is driven in the sanctioned reorder (the cycle contract declares
 * the F2C/C2F touched sets disjoint and freely reorderable), with the
 * cycle-end skin F2C deferred until AFTER the C2F fill's read point:
 *
 *   F2C store of cycle N-1 -> coarse LBM step N -> C2F fill N (audit) -> F2C N (capture)
 *
 * so the frozen-skin slot state the fill reads is exactly what the
 * previous cycles' F2C stores plus the just-ran kernel's parity-phase
 * writes left -- deterministic under both the A-A and the esoteric
 * in-place patterns (the production streams otherwise race the two
 * transfer directions).
 *
 * Fixture: the subcycling/nesting 16^3 -> "1 4 4 4 8 8 8" mL1 geometry,
 * with a full GEO_INFLOW slab plane stamped on the coarse plane x = 3
 * (one coarse cell outside the footprint's min-x face, in the sim_AMR_ball
 * R = 1 adjacency class) so the thesis Sec-7.3 guard steers that face's
 * source windows onto the {c = 0 ring (x = 4), c = 1 frozen skin (x = 5)}
 * pair. GEO_INFLOW is selected (not GEO_WALL) precisely because the
 * fine-level wall mask only counts GEO_WALL columns and would otherwise
 * drop the face's C2F fill entirely (the guard would then never fire);
 * the slab plane's own equilibrium-at-inflow evolution is an inert side
 * condition here. The registration is accepted by
 * checkCouplingMapPattern's rail (a) at SimInit.
 *
 * Audit (host-side, pattern-decoded through the streaming headers' own
 * slot identities):
 * - CAPTURE (after each cycle's deferred F2C): the skin row's authored
 *   content A^N(q, x) = the value the F2C store deposited for direction q
 *   of cell x, decoded through the pattern's store->read slot identity
 *   (preCollisionSlot(q, x + c_q, W) == postCollisionSlot(q, x, !W): every
 *   streaming header documents its postCollisionSlot as "the consistent
 *   counterpart of preCollisionSlot"). The cycle-0 capture A^0 is the
 *   SimInit fill's read-consistent (initial-condition) skin state.
 * - READ (at the C2F fill's read point, F2C(N) not yet launched):
 *   read^N(q, x) = what the fill's read_coarse_df resolves for a frozen
 *   source. Post-fix the kernel resolves frozen (GEO_NOTHING) sources
 *   through the parity-inverted discipline postCollisionSlot(q, x,
 *   !coarse_even_iter) -- by the store->read slot identity that is exactly
 *   the address family the coupling's own F2C store deposits into, which
 *   the kernel's parity-phase writes never author at a frozen site; the
 *   host replica below mirrors that fixed read (the audited skin plate is
 *   all GEO_NOTHING by the census case).
 * - CLASSIFY per (cycle, cell, direction): fresh == A^{N-1}(q),
 *   stale == A^{N-2}(q), flip == A^{*}(opp(q)) (the catalogued Defect-1
 *   direction reversal), live == a kernel-aliased slot: its storage
 *   address in the shared single-array frame is owned by the cell one
 *   step toward c_q (the natural-downwind slot itself, or the live
 *   pusher of a ring cell's parity-phase push), so the frame holds the
 *   live cell's current data there and no read discipline could recover
 *   a frozen era content from it; other == none of them (true
 *   era/convention corruption). The armed assertion is flip == 0 and
 *   other == 0: direction-correctness (no reversal) and total coverage
 *   (every audited slot read either authored or legitimately live
 *   content). Vintage comparisons accept every era and only ever fail
 *   on convention-level wrongness, never on 1-ulp arithmetic.
 *
 * The read evaluation below is a HOST replica of the
 * STREAMING::postCollisionSlot addressing (the device versions' clamps
 * are inert at the audited mid-block row). The AB-pull arm is the
 * reference: its two physical DF frames let the stale read vintage stay
 * direction-correct everywhere, so it must classify 100 % fresh/stale at
 * every strictly-armed cycle (the documented 2-cycle staleness). The
 * single-array arms must classify flip == 0 and other == 0, with the
 * reads partitioned into fresh (frozen-owned slots, era-correct) and
 * live (kernel-aliased slots) by the hmap predicate.
 */

#include <array>
#include <cstdint>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include <fmt/core.h>
#include <fmt/format.h>

#include <doctest/doctest.h>

#include "amr_test_fixture.h"

namespace
{

// the audited skin plate: the c = 1 frozen row behind the Sec-7.3 guarded
// window of the footprint's min-x face (coarse x = 5, y/z in [5, 11) --
// the F2C-written depth-1 row; footprint [4, 12), x = 3 is the GEO_INFLOW
// slab, x = 4 the reactivated ring row)
constexpr int SKIN_X = 5;
constexpr int SKIN_LO = 5;
constexpr int SKIN_HI = 11;

// the GEO_INFLOW slab plane (one coarse cell outside the footprint's
// min-x face; not GEO_WALL -- see the file docstring)
constexpr int WALL_X = 3;

// cycles driven per scenario run (both coarse parities covered twice;
// strict assertions arm at cycle 3, the first closed loop whose read-side
// vintages are all F2C-authored: F2C(2) -> coarse(3) -> C2F(3))
constexpr int WALL_LANE_CYCLES = 5;
constexpr int WALL_LANE_ARM_FROM = 3;

// wall-lane scenario state: the subcycling spy's census events plus (a) the
// GEO_INFLOW slab and a non-uniform VELOCITY-marker initial condition (so
// the momentum direction is observable in every slot -- direction reversal
// of a zero-velocity field would be invisible), (b) the sanctioned-reorder
// transfer phase that defers the cycle-end skin F2C until after the C2F
// fill's read point and runs the slot audit/capture around it.
template <typename NSE>
struct StateWallLane_AMR : StateSchedule_AMR<NSE>
{
	using Base = StateSchedule_AMR<NSE>;
	using BLOCK_NSE = LBM_BLOCK<NSE>;
	using idx_t = typename NSE::TRAITS::idx;
	using idx3d_t = typename NSE::TRAITS::idx3d;
	using dreal_t = typename NSE::TRAITS::dreal;
	using lat_local_t = typename Base::lat_t;
	using BC_t = typename NSE::BC;

	template <typename... ARGS>
	StateWallLane_AMR(ARGS&&... args)
	: Base(std::forward<ARGS>(args)...)
	{}

	void setupBoundaries() override
	{
		// periodic-base fixture (the empty base body); stamp the full
		// GEO_INFLOW slab plane on the level-0 plane WALL_X (all y, z): the
		// guard's taint scan then steers the min-x face's windows onto the
		// {c = 0 ring, c = 1 skin} pair (sim_AMR_ball's R = 1 class)
		for (auto& block : this->nse.blocks)
			if (block.level == 0)
				for (idx_t z = block.offset.z(); z < block.offset.z() + block.local.z(); z++)
					for (idx_t y = block.offset.y(); y < block.offset.y() + block.local.y(); y++)
						block.hmap(WALL_X, y, z) = BC_t::GEO_INFLOW;
	}

	void resetDFs() override
	{
		using idx = typename NSE::TRAITS::idx;
		using point_t = typename NSE::TRAITS::point_t;

		for (auto& block : this->nse.blocks) {
			const lat_local_t lat_local = (block.level == 0) ? this->nse.lat : block.lat_local;
			const idx3d_t offset = block.offset;

			block.setInitialCondition(
				[lat_local, offset] __cuda_callable__(typename NSE::template KernelStruct<dreal_t> & KS, idx gx, idx gy, idx gz) mutable
				{
					const point_t phys = lat_local.lbm2physPoint(gx - offset.x(), gy - offset.y(), gz - offset.z());
					KS.rho = 1 + 0.01f * TNL::sin(8.0f * phys.x());
					// non-uniform velocity markers (per-component wave shapes
					// chosen incommensurate so a slot of direction q can
					// never be confused with the opposite direction's slot
					// or with a foreign cell's collision output by
					// coincidence)
					KS.vx = 0.02f * TNL::sin(4.0f * phys.y());
					KS.vy = 0.02f * TNL::cos(6.0f * phys.z());
					KS.vz = 0.015f * TNL::sin(5.0f * phys.x() + 3.0f * phys.y());
					NSE::COLL::setEquilibrium(KS);
				}
			);
		}

		this->nse.copyDFsToHost();
	}

	// ---- the sanctioned-reorder transfer phase (see the file docstring
	// for the closed-loop order) ----

	std::vector<std::array<int, 3>> skin;  // filled at the first (SimInit) audit
	int fill_calls = 0;					   // SimInit fill == 0, then one per coarse cycle
	bool f2c_pending = false;
	int f2c_pending_level = -1;
	std::map<int, std::vector<float>> authored;	 // A^cycle per capture point (cycle 0 = IC read face)
	std::map<int, bool> authored_next_even;		 // the store parity W of each capture
	struct Audit
	{
		int cycle = -1;
		bool coarse_even = false;
		std::vector<float> read;  // [cells x 27]
	};
	std::vector<Audit> audits;

	void launchFineToCoarseTransfersInterior(int fine_level) override
	{
		// do NOT launch here: the sanctioned reorder defers the cycle-end
		// skin F2C to the post-C2F point of the same transfer phase (the
		// spy event is still recorded at the production call site)
		Base::record(Base::Stage::f2c, fine_level);
		f2c_pending = true;
		f2c_pending_level = fine_level;
	}

	void launchCoarseToFineTransfers(int fine_level) override
	{
		// drain anything in flight, run the C2F fill, drain it, then audit
		// the read-consistent frozen-skin slot state (the fill itself never
		// writes coarse cells, so the post-launch state is exactly what the
		// fill's read_coarse_df resolved); the deferred cycle-end F2C then
		// runs and its deposit is captured
		State_AMR<NSE>::synchronizeTransfers();
		Base::launchCoarseToFineTransfers(fine_level);
		State_AMR<NSE>::synchronizeTransfers();
		auditRead();
		if (f2c_pending) {
			const int level = f2c_pending_level;
			f2c_pending = false;
			f2c_pending_level = -1;
			Base::launchFineToCoarseTransfersInterior(level);
			State_AMR<NSE>::synchronizeTransfers();
			captureAuthored();
		}
	}

	BLOCK_NSE* coarseBlock()
	{
		return this->nse.getBlocksAtLevel(0).front();
	}

	// ---- host replicas of the STREAMING::postCollisionSlot addressing

	// physical DF array index of logical df_out at the current rotation
	// (A-B two-frame patterns only); single-array patterns read frame 0
	int physOutFrame()
	{
		return dfsSwapped(*this->nse.getBlocksAtLevel(0).front()) ? 0 : 1;
	}

	// evaluate direction q's slot at coarse local cell (x, y, z) through
	// the pattern's postCollisionSlot parity discipline
	float postSlot(int q, int x, int y, int z, bool even)
	{
		const BLOCK_NSE* coarse = this->nse.getBlocksAtLevel(0).front();
		using STREAMING = typename NSE::STREAMING;
		if constexpr (NSE::DFMAX >= 2) {
			// A-B pull: postCollisionSlot ignores the parity argument and
			// reads the own-site slot of logical df_out
			(void) even;
			return coarse->hfs[physOutFrame()](q, x, y, z);
		}
		else if constexpr (is_AA_v<STREAMING>) {
			if (even)
				return coarse->hfs[0](opposite_direction(q), x, y, z);
			return coarse->hfs[0](q, x + dir27_cx(q), y + dir27_cy(q), z + dir27_cz(q));
		}
		else if constexpr (is_ESO_TWIST_v<STREAMING>) {
			const int ox = x + (dir27_cx(q) > 0 ? 1 : 0);
			const int oy = y + (dir27_cy(q) > 0 ? 1 : 0);
			const int oz = z + (dir27_cz(q) > 0 ? 1 : 0);
			return coarse->hfs[0](even ? q : opposite_direction(q), ox, oy, oz);
		}
		else if constexpr (is_ESO_PULL_v<STREAMING>) {
			const int ox = x + (is_pair_head(q) ? dir27_cx(q) : 0);
			const int oy = y + (is_pair_head(q) ? dir27_cy(q) : 0);
			const int oz = z + (is_pair_head(q) ? dir27_cz(q) : 0);
			return coarse->hfs[0](even ? q : opposite_direction(q), ox, oy, oz);
		}
		else if constexpr (is_ESO_PUSH_v<STREAMING>) {
			const int ox = x + (is_pair_head(q) ? 0 : dir27_cx(q));
			const int oy = y + (is_pair_head(q) ? 0 : dir27_cy(q));
			const int oz = z + (is_pair_head(q) ? 0 : dir27_cz(q));
			return coarse->hfs[0](even ? opposite_direction(q) : q, ox, oy, oz);
		}
		else {
			static_assert(sizeof(STREAMING) == 0, "host postSlot is not specialized for this streaming pattern");
		}
	}

	// raw-slot probe for the writer search (DEBUG only): dump a fixed
	// address table at every checkpoint
	void probeSlots(const char* tag)
	{
		if (std::getenv("AMR_WALL_LANE_DEBUG") == nullptr)
			return;
		const BLOCK_NSE* coarse = this->nse.getBlocksAtLevel(0).front();
		static const int probes[4][4] = {
			{1, 6, 5, 7},  // (pzz, (6,5,7)) -- natural deposit slot of cell (5,5,7)
			{2, 5, 5, 7},  // (mzz, x itself)  -- twisted-side deposit slot
			{1, 5, 5, 7},  // (pzz, x itself)  -- push-target slot at the skin site
			{7, 6, 6, 7},  // (ppz, (6,6,7)) -- a hit-family deposit slot
		};
		fmt::print("  PROBE {}:", tag);
		for (const auto& p : probes)
			fmt::print("  ({},{},{}) q {} = {:.9e}", p[1], p[2], p[3], p[0], static_cast<double>(coarse->hfs[0](p[0], p[1], p[2], p[3])));
		fmt::print("\n");
	}

	void auditRead()
	{
		BLOCK_NSE* coarse = coarseBlock();
		// the audited skin plate is constant per fixture; collect it once
		if (skin.empty())
			for (int z = SKIN_LO; z < SKIN_HI; z++)
				for (int y = SKIN_LO; y < SKIN_HI; y++)
					skin.push_back({SKIN_X, y, z});
		coarse->copyDFsToHost();
		coarse->copyMapToHost();
		if (std::getenv("AMR_WALL_LANE_DEBUG") != nullptr) {
			fmt::print("  PROBE point A (post-C2F read, cycle {})\n", fill_calls);
			probeSlots("A");
		}

		Audit a;
		a.cycle = fill_calls;
		a.coarse_even = coarse->data.even_iter;
		// the fixed kernel reads frozen (GEO_NOTHING) sources with the
		// parity-inverted discipline; the audited skin plate is all frozen,
		// so the replica evaluates the read at !coarse_even as the kernel does
		for (const auto& c : skin)
			for (int q = 0; q < NSE::Q; q++)
				a.read.push_back(postSlot(q, c[0], c[1], c[2], ! a.coarse_even));
		audits.push_back(std::move(a));
		// cycle 0 (the SimInit fill): adopt the initial-condition
		// read-consistent skin state as the first authored vintage
		// (direction-correct by each pattern's layout-0 construction)
		if (fill_calls == 0)
			authored[0] = audits.back().read;
		fill_calls++;
	}

	void captureAuthored()
	{
		// the F2C store deposited each skin cell's direction-q content at
		// preCollisionSlot(q, x + c_q, W) == postCollisionSlot(q, x, !W)
		// (each streaming header's consistent-counterpart identity), with W
		// the next-kernel parity: level-0 parents take it from the global
		// clock ((iterations % 2) == 1, post-increment inside SimUpdate)
		coarseBlock()->copyDFsToHost();
		if (std::getenv("AMR_WALL_LANE_DEBUG") != nullptr) {
			fmt::print("  PROBE point B (post-F2C capture, cycle {})\n", fill_calls - 1);
			probeSlots("B");
		}
		const bool W = (this->nse.iterations % 2) == 1;
		std::vector<float> snap;
		for (const auto& c : skin)
			for (int q = 0; q < NSE::Q; q++)
				snap.push_back(postSlot(q, c[0], c[1], c[2], ! W));
		if (std::getenv("AMR_WALL_LANE_DEBUG") != nullptr) {
			BLOCK_NSE* coarse = coarseBlock();
			coarse->copyMacroToHost();
			const std::size_t probe[3] = {0, 12, 35};
			for (int pi = 0; pi < 3; pi++) {
				const std::size_t ci = probe[pi];
				dreal_t s = 0;
				for (int q = 0; q < NSE::Q; q++)
					s += snap[ci * NSE::Q + q];
				fmt::print(
					"  CAPTURE cycle {} cell ({},{},{}) W={}: sum_f = {:.9e} vs F2C macro rho = {:.9e}\n",
					fill_calls - 1,
					skin[ci][0],
					skin[ci][1],
					skin[ci][2],
					W,
					static_cast<double>(s),
					static_cast<double>(coarse->hmacro(NSE::MACRO::e_rho, skin[ci][0], skin[ci][1], skin[ci][2]))
				);
			}
		}
		// keyed by the cycle whose deferred F2C just landed (auditRead of
		// the same cycle already counted it, hence -1): authored[k] holds
		// the F2C-authored skin state deposited at the end of cycle k
		authored[fill_calls - 1] = std::move(snap);
		authored_next_even[fill_calls - 1] = W;
	}

	// kernel-aliasing predicate of the fixed read: in a single-array frame
	// the audited direction's slot (and any write of the last kernel phase
	// that can overwrite it) reaches exactly the cell one step toward c_q --
	// at even parities it IS the natural-downwind read target, at odd
	// parities it is the live pusher whose odd-phase push lands in the
	// twisted own-site slot. If that cell is not GEO_NOTHING it is
	// kernel-owned live storage: the in-place frame cannot hold the frozen
	// cell's authored era content AND the live solver's phase data at the
	// same address, so the read legitimately returns the live cell's
	// current content (the population streaming across the ring/skin seam),
	// not a frozen authored vintage. A-B frames keep the frozen slots
	// disjoint from kernel writes, so the alias class is empty there.
	bool readAliasedByKernel(int q, const std::array<int, 3>& c)
	{
		if constexpr (NSE::DFMAX >= 2) {
			return false;
		}
		else {
			const BLOCK_NSE* coarse = coarseBlock();
			return coarse->hmap(c[0] + dir27_cx(q), c[1] + dir27_cy(q), c[2] + dir27_cz(q)) != BC_t::GEO_NOTHING;
		}
	}

	// per-cycle classification census of the audited reads against the two
	// most recent authored vintages
	struct Census
	{
		long fresh = 0, stale = 0, flip = 0, live = 0, other = 0;
		std::vector<std::string> first_other;
	};
	Census classify(int audit_idx)
	{
		Census out;
		const Audit& a = audits[audit_idx];
		int latest = -1, previous = -1;
		for (const auto& entry : authored) {
			if (entry.first < a.cycle) {
				previous = latest;
				latest = entry.first;
			}
		}
		for (std::size_t ci = 0; ci < skin.size(); ci++)
			for (int q = 0; q < NSE::Q; q++) {
				const float v = a.read[ci * NSE::Q + q];
				bool hit = false;
				if (latest >= 0 && std::memcmp(&v, &authored.at(latest)[ci * NSE::Q + q], sizeof v) == 0) {
					out.fresh++;
					hit = true;
				}
				else if (previous >= 0 && std::memcmp(&v, &authored.at(previous)[ci * NSE::Q + q], sizeof v) == 0) {
					out.stale++;
					hit = true;
				}
				else if (dir27_cx(q) || dir27_cy(q) || dir27_cz(q)) {
					const int oq = opposite_direction(q);
					if (latest >= 0 && std::memcmp(&v, &authored.at(latest)[ci * NSE::Q + oq], sizeof v) == 0) {
						out.flip++;
						hit = true;
					}
					else if (previous >= 0 && std::memcmp(&v, &authored.at(previous)[ci * NSE::Q + oq], sizeof v) == 0) {
						out.flip++;
						hit = true;
					}
				}
				if (! hit && readAliasedByKernel(q, skin[ci])) {
					out.live++;
					hit = true;
				}
				if (! hit) {
					out.other++;
					if (out.first_other.size() < 6)
						out.first_other.push_back(fmt::format(
							"cycle {} cell ({},{},{}) q {}: read {:.9e}",
							a.cycle,
							skin[ci][0],
							skin[ci][1],
							skin[ci][2],
							q,
							static_cast<double>(v)
						));
				}
			}
		return out;
	}
};

TEST_SUITE_BEGIN("amr_wall_lane");

TEST_CASE("wall-adjacent mL1 registration census (guarded window reads the frozen skin)")
{
	const lat_t lat = makeLattice();
	StateWallLane_AMR<NSE_CONFIG> state("test_amr_wall_lane_census", MPI_COMM_WORLD, lat, "adios2.xml", TRAITS::bool3d{true, true, true}, 1);
	REQUIRE(state.canCompute());
	createAMRBlocks(state.nse, parseAMRConfig<NSE_CONFIG>("1 4 4 4 8 8 8"));
	state.SimInit();
	REQUIRE(! state.nse.terminate);

	// the wall-lane registration survived: one L0->L1 coupling with its
	// halo and skin patch sets on every face (the guarded min-x face keeps
	// its fill -- GEO_INFLOW is not counted by the fine-level wall mask)
	REQUIRE(state.couplings.size() == 1);
	CHECK(state.nse.getBlocksAtLevel(0).size() == 1);
	CHECK(state.nse.getBlocksAtLevel(1).size() == 1);
	std::size_t halo_patches = 0, skin_patches = 0;
	for (const auto& c : state.couplings) {
		halo_patches += c.patches.size();
		skin_patches += c.interior_patches.size();
	}
	CHECK(halo_patches == 6);
	CHECK(skin_patches == 6);

	// map census: the GEO_INFLOW slab plane is intact (markAMRInterface
	// only re-tags GEO_FLUID cells), the ring row behind it is
	// GEO_AMR_INTERFACE, the guarded skin row is frozen GEO_NOTHING
	auto* coarse = state.nse.getBlocksAtLevel(0).front();
	coarse->copyMapToHost();
	CHECK(coarse->hmap(WALL_X, 6, 6) == NSE_CONFIG::BC::GEO_INFLOW);
	CHECK(coarse->hmap(SKIN_X - 1, 6, 6) == NSE_CONFIG::BC::GEO_AMR_INTERFACE);
	int nothing_plate = 0, ring_plate = 0;
	for (int z = SKIN_LO; z < SKIN_HI; z++)
		for (int y = SKIN_LO; y < SKIN_HI; y++) {
			if (coarse->hmap(SKIN_X, y, z) == NSE_CONFIG::BC::GEO_NOTHING)
				nothing_plate++;
			if (coarse->hmap(SKIN_X - 1, y, z) == NSE_CONFIG::BC::GEO_AMR_INTERFACE)
				ring_plate++;
		}
	CHECK(nothing_plate == (SKIN_HI - SKIN_LO) * (SKIN_HI - SKIN_LO));
	CHECK(ring_plate == (SKIN_HI - SKIN_LO) * (SKIN_HI - SKIN_LO));
}

TEST_CASE("frozen-skin C2F read audit: closed loop F2C(N-1) -> coarse N -> C2F N (the SP-1 / Defect-1 failing-first gate)")
{
	const lat_t lat = makeLattice();
	StateWallLane_AMR<NSE_CONFIG> state("test_amr_wall_lane_audit", MPI_COMM_WORLD, lat, "adios2.xml", TRAITS::bool3d{true, true, true}, 1);
	REQUIRE(state.canCompute());
	createAMRBlocks(state.nse, parseAMRConfig<NSE_CONFIG>("1 4 4 4 8 8 8"));
	state.SimInit();
	REQUIRE(! state.nse.terminate);

	// the SimInit fill ran through the same override (audits[0]; A^0 is
	// the initial-condition read face, direction-correct by construction)
	REQUIRE(state.audits.size() == 1);

	for (int c = 0; c < WALL_LANE_CYCLES; c++) {
		state.updateKernelData();
		state.SimUpdate();
		REQUIRE(! state.nse.terminate);

		REQUIRE(state.audits.size() == static_cast<std::size_t>(c + 2));
		const auto& audit = state.audits.back();
		const auto census = state.classify(static_cast<int>(state.audits.size()) - 1);
		const long total = census.fresh + census.stale + census.flip + census.live + census.other;
		fmt::print(
			"amr_wall_lane {}: cycle {} (coarse_even_iter = {}): fresh {} stale {} flip {} live {} other {} of {} audited slot reads\n",
			pattern_name,
			audit.cycle,
			audit.coarse_even,
			census.fresh,
			census.stale,
			census.flip,
			census.live,
			census.other,
			total
		);
		for (const auto& line : census.first_other)
			fmt::print("  {}\n", line);
		if (std::getenv("AMR_WALL_LANE_DEBUG") != nullptr && audit.cycle >= 1) {
			const int ci = 12;  // cell (5,5,7) mid-plate
			fmt::print("  DEBUG cell ({},{},{}) cycle {}\n", state.skin[ci][0], state.skin[ci][1], state.skin[ci][2], audit.cycle);
			for (int q = 0; q < NSE_CONFIG::Q; q++) {
				const float v = audit.read[ci * NSE_CONFIG::Q + q];
				fmt::print("   q {:2d} read {: .9e}", q, static_cast<double>(v));
				for (int k = 0; k < 3; k++) {
					auto it = state.authored.find(audit.cycle - 1 - k);
					if (it != state.authored.end())
						fmt::print(" | A^{} {: .9e}", audit.cycle - 1 - k, static_cast<double>(it->second[ci * NSE_CONFIG::Q + q]));
					else
						fmt::print(" | A^{} (missing)", audit.cycle - 1 - k);
				}
				fmt::print("  (opp {})\n", opposite_direction(q));
			}
		}
		// THE GATE, strictly armed from the first fully F2C-authored closed
		// loop: every audited frozen-skin read must resolve to either a
		// recently authored direction-q value of the same cell (any
		// vintage; a direction reversal cannot satisfy this) or a
		// legitimately kernel-aliased live value (single-array frames
		// share those addresses with the live solver's working slots).
		// Cycles 1-2 still print their censuses (the initial-condition
		// transient evidence)
		if (audit.cycle >= WALL_LANE_ARM_FROM) {
			CHECK_MESSAGE(
				census.flip == 0,
				fmt::format(
					"amr_wall_lane cycle {}: {} frozen-skin C2F reads resolved to the OPPOSITE direction's authored value (Defect-1 direction reversal)",
					audit.cycle,
					census.flip
				)
			);
			CHECK_MESSAGE(
				census.other == 0,
				fmt::format(
					"amr_wall_lane cycle {}: {} of {} frozen-skin C2F reads are neither a recently authored vintage nor legitimately kernel-aliased live content (the era-correct frozen read is broken)",
					audit.cycle,
					census.other,
					total
				)
			);
		}
	}
}

TEST_SUITE_END();

}  // namespace
