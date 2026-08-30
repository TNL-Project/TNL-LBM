#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <omp.h>

#include <spdlog/spdlog.h>

#include "state.h"
#include "amr_decomposition.h"
#include "d3q27/amr_coupling.h"
#include "viz/OverlappingAMRWriter.h"

/**
 * \brief Conservation statistics aggregated over the AMR block hierarchy.
 *
 * Global mass and momentum are volume-weighted (a level-L cell carries
 * `1/8^L` of the coarse-cell volume with the 2:1 refinement ratio); the
 * per-level kinetic energy is a plain (unweighted) level diagnostic.
 */
struct AMRConservationStats
{
	double total_mass = 0;
	double total_momentum_x = 0, total_momentum_y = 0, total_momentum_z = 0;
	std::vector<double> per_level_kinetic_energy;
};

/**
 * \brief Berger-Colella time subcycling driver for AMR simulations.
 *
 * \ref State_AMR inherits \ref State and overrides \ref SimUpdate to advance
 * a multi-level block hierarchy in time with the classic Berger & Colella
 * (1989) subcycling schedule, ordered into the Schönherr cycle with
 * SIMULATED band (Schönherr 2015 thesis ch.7; the cycle contract of
 * docs/AMR-schonherr-ch7-target-contract.md / plan
 * .omo/plans/schonherr-ch7-conversion.md sec. 1.3; the passive-band form
 * shipped by the conversion was flipped to the simulated band after the
 * T16 20-tc null verdict - contract sec. 4 fork row (c)). One global
 * iteration performs the Berger-Colella PAIR recursion (\ref advancePair):
 * one pair at level L covers 2 dt_L = dt_{L-1} and nests the finer level's
 * own pairs and sync points inside, so each cycle performs per level pair
 * (L-1, L), with s_L the level's cumulative substep counter
 * (LBM::totalSubstepCount; parity of the level's `substep % 2` rotation):
 *
 * 1. fine substep 1 of 2 at level L (substep A of the pair):
 *    `updateKernelDataForLevel(L, s_L)`
 *    selects the fine level's substep-0 parity / DF rotation (MANDATORY
 *    before the kernel - for the A-B pattern the rotation selects the
 *    physical array `df_cur` refers to; the global `updateKernelData()` is
 *    driven by the coarse clock and must not drive the fine substeps),
 *    then `cudaLBMKernel` launched on the WIDENED extent
 *    [-1, local+1) per axis (`ghost_layers = 1`): the inner ghost rows
 *    are INTEGRATED like interior fluid (they are GEO_FLUID and collide
 *    + stream), pulling their streaming input from the outer ghost row
 *    filled at the END of the previous cycle (step 5 below; cycle 0
 *    reads the initial fill \ref SimInit performed). The interior reads
 *    the inner ghost row of the same frame -- that frame's fill,
 * 2. fine substep 2 of 2 (substep B of the pair):
 *    `updateKernelDataForLevel(L, s_L + 1)` +
 *    `cudaLBMKernel` on the interior-only extent (same rotation
 *    requirement). The boundary data of this substep is substep 1's
 *    kernel-updated inner ghost rows in the OTHER AB frame, so the band
 *    advances synchronously with the fine clock and no fill is needed
 *    for this frame,
 * 3. ONE coarse (level 0) LBM step on all level-0 blocks
 *    (`updateKernelData()` was already called by `execute()` in `core.h`
 *    and set the level-0 even_iter parity / DF rotation from the global
 *    `iterations` counter). Coarse cells tagged `GEO_AMR_INTERFACE` (the
 *    interface ring around each fine footprint) are collision-active
 *    inside the kernel (they stream and collide like fluid); since the
 *    ring fine-to-coarse launch was removed (gate B ruling, 2026-08-16)
 *    the coarse kernel is their only writer -- fine feedback reaches them
 *    through streaming from the skin cells the interior F2C wrote at the
 *    end of the previous cycle (step 4 below),
 * 4. fine-to-coarse transfer at every pair sync point of the pair (L-1,
 *    L): `cudaAMR_FineToCoarse` fires once per level-(L-1) substep,
 *    immediately when both levels are time-aligned (twice per pair of
 *    level L -- for L == 1 that is the historical ONCE-per-cycle cadence),
 *    transferring the strategy-selected fine state (the Schönherr
 *    compact-moment reconstruction of the F2C_SCHONHERR default; the
 *    Lagrava-filtered fine state under the F2C_LAGRAVA opt-out) back
 *    onto the frozen GEO_NOTHING skin cells of level L-1, reading the
 *    fine level's rotation-1 frame (the post-substep array of the pair
 *    that just completed; see `launchFineToCoarseTransfersInterior`),
 * 5. coarse-to-fine transfer, the single-frame fill per fine pair:
 *    `updateKernelDataForLevel(L, s_L)` + `cudaAMR_CoarseToFine` fills the
 *    fine ghost rows (both overlap rows) in the physical frame the next
 *    pair's substep A consumes -- the mid-cycle fill inside \ref
 *    advancePair (the band for the finer pair #2) and the level-ascending
 *    cycle-end cascade (the band for pair #1 of the next cycle). s_L is
 *    always even at a pair boundary, so the frame is parity-identical to
 *    the historical positional substep 0. The other frame needs no fill:
 *    substep 2 consumes substep 1's updated inner ghost rows from it, and
 *    its outer row is unreachable (substep 2 is interior-only).
 *
 * The relative order of step 4 (F2C) against step 5 (C2F) is
 * irrelevant: their touched sets are disjoint (F2C writes coarse skin
 * cells of the coarse post-step array, C2F writes fine ghost rows) -
 * declared per the cycle contract. This disjointness is EXPLOITED: the
 * two directions run CONCURRENTLY, each on its own dedicated stream
 * (stream_f2c / stream_c2f), with a single barrier per transfer phase
 * (synchronizeTransfers) before the next dependent kernel launch -
 * Schönherr's asynchronous coupling overlap. \ref SimInit performs the
 * same single-frame fill (step 5) after building \ref couplings, so
 * cycle 0 starts with valid destinations in the substep-0 frame (the
 * fine substep 1 of cycle 0 therefore reads a t_0 fill - the startup
 * transient of the contract sec. 1.3).
 *
 * `nse.iterations` counts COARSE steps only; fine substeps advance the
 * level clocks, not the global counter. With the 2:1 refinement ratio the
 * physical time stays synchronized: `physDt_coarse = 2 * physDt_fine`, so
 * two fine substeps at level L match one step at level L-1.
 *
 * Coupling parameters (conventions documented in `d3q27/amr_coupling.h`):
 * - `tau = 3 * nu_lb + 0.5` per level; `nu_lb_f = 2 * nu_lb_c` with 2:1
 *   refinement, so `tau_fine != tau_coarse` and both are computed from the
 *   per-level `lbmViscosity()` of the SOURCE blocks of each transfer.
 * - The read-side parity argument is the CURRENT `data.even_iter` of the
 *   source block (the parity of the kernel launch that produced the data).
 *   For `cudaAMR_FineToCoarse` the write-side `coarse_even_iter` is the
 *   parity of the NEXT consuming coarse substep.
 * - Kernel extents are passed in the target block's indexer coordinates
 *   and both blocks' `offset` values are passed to the kernels, which map
 *   between the two indexer frames in the global coordinates of each level
 *   (nested fine blocks have non-corresponding indexer origins - see the
 *   kernel docstrings).
 *
 * v1 scope (same single-GPU scope as `createAMRBlocks`/`markAMRInterface`):
 * - `max_level == 0` falls back to the base `State<NSE>::SimUpdate()`
 *   unchanged, `max_level >= 1` requires CUDA (`USE_CUDA`);
 * - kernel launches use the null stream with the interior
 *   (`SyncDirection::None`) launch configuration over the FULL block
 *   extents, followed by `TNL::Backend::streamSynchronize(0)` - this
 *   mirrors the `nproc == 1` path of the base driver (per-direction streams
 *   for MPI boundary neighbors are out of scope);
 * - ghost/interface extents are built ONCE in `SimInit()` by scanning the
 *   `GEO_AMR_INTERFACE` markings produced by `markAMRInterface`: one
 *   `AMR_InterfacePatch` per (fine block, parent block, halo face) triple
 *   whose face rectangle holds at least one marked cell, stored in
 *   \ref couplings and consumed by all transfer launches (todo 8);
 * - halo exchange between blocks uses the PER-LEVEL variant
 *   `nse.synchronizeDFsAndMacroDeviceForLevel()` (todo 8): level 0 after
 *   the coarse step, level L after each fine substep. It is a no-op for
 *   fine blocks in the v1 single-rank setup (fine blocks have no MPI
 *   neighbors) and is only invoked for `nproc > 1` (the base driver needs
 *   no overlap copying for `nproc == 1` either);
 * - the transfer extents are additionally clipped at launch time by a
 *   storability guard derived from the block's overlap storage (see the
 *   launch helpers): coarse-to-fine fills at most the 2-cell-deep fine
 *   ghost storage (`LBM_BLOCK::storage_overlap` on refinement-level
 *   blocks; the face-aware clip of `launchCoarseToFineTransfers`) and
 *   fine-to-coarse is clipped to coarse cells whose full 2x2x2 fine
 *   subcell block is a valid storage index of the fine block (the
 *   kernels' documented preconditions);
 * - `computeBeforeLBMKernel()` is called once per coarse step before the
 *   level-0 kernel, exactly as in the base driver; `computeAfterLBMKernel()`
 *   remains in the inherited `AfterSimUpdate()`. The subclass contract
 *   (e.g. the non-Newtonian model) is unchanged.
 */

template <typename NSE>
struct State_AMR : State<NSE>
{
	using Base = State<NSE>;
	using TRAITS = typename NSE::TRAITS;
	using BLOCK_NSE = LBM_BLOCK<NSE>;

	using idx = typename TRAITS::idx;
	using dreal = typename TRAITS::dreal;
	using real = typename TRAITS::real;
	using idx3d = typename TRAITS::idx3d;
	using lat_t = typename Base::lat_t;

	/**
	 * \brief Host-side descriptor of all coarse-fine interfaces between one
	 * pair of consecutive levels.
	 *
	 * Built ONCE in \ref SimInit (todo 8): \ref buildCouplings scans the
	 * `GEO_AMR_INTERFACE` markings produced by `markAMRInterface` and
	 * appends one `AMR_InterfacePatch` per (fine block, parent block, halo
	 * face) triple whose face rectangle holds at least one marked cell.
	 * Consumed by all coupling launches in `SimUpdate()`.
	 *
	 * The patches store rectangles only; the two `_block_ids` arrays
	 * (parallel to `patches`) record which blocks each patch couples, so
	 * the launches can pass the correct `data` of both blocks without
	 * re-deriving ownership from the geometry.
	 */
	struct InterLevelCoupling
	{
		int coarse_level = 0;
		int fine_level = 0;
		std::vector<AMR_InterfacePatch<NSE>> patches;
		std::vector<int> coarse_block_ids;
		std::vector<int> fine_block_ids;
		// interior (under-footprint) patches: frozen GEO_NOTHING cells that
		// receive fine-averaged DFs via F2C each cycle (two-way feedback)
		std::vector<AMR_InterfacePatch<NSE>> interior_patches;
		std::vector<int> interior_coarse_block_ids;
		std::vector<int> interior_fine_block_ids;
	};

	// all inter-level couplings of the simulation (empty when max_level == 0
	// or when no GEO_AMR_INTERFACE cells were found in SimInit)
	std::vector<InterLevelCoupling> couplings;

	// pass-through constructor: State_AMR adds no physical state of its own
	template <typename... ARGS>
	State_AMR(ARGS&&... args)
	: Base(std::forward<ARGS>(args)...)
	{}

	// runs the base SimInit (memory estimate, allocation, counter reset,
	// boundary setup via reset()->setupBoundaries()), then - for
	// `nse.max_level > 0` only - tags the coarse-fine interfaces via
	// `markAMRInterface` and builds \ref couplings; with max_level == 0 the
	// initialization is identical to the base driver (sim_1 fallback)
	void SimInit() override;

	// Berger-Colella subcycling (see the file docstring); delegates to the
	// base driver when `nse.max_level == 0`
	void SimUpdate() override;

	// runs the base AfterSimUpdate (I/O, probes, checkpoints), then - for
	// `nse.max_level > 0` only - logs the conservation statistics of
	// \ref computeConservationStats at the PRINT interval
	void AfterSimUpdate() override;

	// Write AMR visualization output in VTKHDF (OverlappingAMR) format.
	// Static single-GPU version (v1): one file per OUT3D cycle.
	// No-op when nse.max_level == 0 (no AMR).
	void write3D_AMR(real time, int cycle);

	// implementation details of the subcycling stages. They are public
	// and virtual: schedule-observation test subclasses override them to
	// record the launch sequence and AB parity at each call site, then
	// delegate to these implementations (the schedule census of
	// tests/test_amr_subcycling.cu).

	// launch `cudaLBMKernel` on every block at `level` (null-stream,
	// interior launch configuration) and synchronize. With
	// `ghost_layers > 0` the launch extent is widened by that many overlap
	// cells per face (bounded by the allocated overlap of each block) --
	// the Schönherr-simulated-band substep-1 launch at fine levels
	// integrates the inner ghost rows (collide+stream like interior
	// fluid, their streaming source is the outer overlap row); all other
	// launches pass 0 and cover the interior [0, local) only. An axis
	// whose face is masked in fine_wall_masks is deepened to the fine
	// wall's GEO_WALL row in BOTH launch classes (kernelLaunchWindow).
	virtual void launchLBMKernelForLevel(int level, bool compute_macro, int ghost_layers);
	// fill the ghost layer of every level-`fine_level` block from level
	// `fine_level - 1` via `cudaAMR_CoarseToFine`, iterating the
	// `AMR_InterfacePatch` descriptors of \ref couplings
	virtual void launchCoarseToFineTransfers(int fine_level);
	// project fine-averaged DFs onto the frozen GEO_NOTHING skin cells of
	// each fine footprint (interior_patches of \ref couplings) -- the ONLY
	// fine-to-coarse channel since the ring F2C launch was removed (gate B
	// ruling, D.1 hard-delete)
	virtual void launchFineToCoarseTransfersInterior(int fine_level);

	// per-direction transfer streams of the asynchronous coupling overlap
	// (Schönherr's F2C/C2F concurrency): the two directions' kernels touch
	// disjoint cell sets (F2C writes the parent skin reading fine
	// interiors, C2F writes the fine ghost rows reading the parent ring),
	// so each transfer phase launches both directions with no intermediate
	// sync -- one per stream -- and drains them at the single barrier
	// synchronizeTransfers() before the next dependent kernel launch.
	// Created lazily as BLOCKING streams (StreamDefault): they implicitly
	// order after all prior null-stream work, so a transfer can never race
	// the producing LBM kernels (which are null-stream synced at every
	// launch anyway). Stream 0 until created: the spy fixtures that
	// override the launchers leave them at 0 and the barrier degrades to
	// a null-stream sync.
	TNL::Backend::stream_t stream_f2c = 0;
	TNL::Backend::stream_t stream_c2f = 0;

	void ensureTransferStreams()
	{
		if (stream_f2c == 0) {
			stream_f2c = TNL::Backend::streamCreateWithPriority(TNL::Backend::StreamDefault, 0);
			stream_c2f = TNL::Backend::streamCreateWithPriority(TNL::Backend::StreamDefault, 0);
		}
	}

	// the single sync point of a transfer phase: both directions' streams
	// drained before the next dependent kernel launch (substep B / the next
	// cycle / SimInit's post-fill seeding) and before any host-side read
	// of the transferred data
	void synchronizeTransfers();

	~State_AMR() override
	{
		TNL::Backend::streamDestroy(stream_f2c);
		TNL::Backend::streamDestroy(stream_c2f);
	}

	// the Berger-Colella pair recursion driving the launch helpers above
	// (the file docstring): advance `level` by one pair of substeps (2 dt_L
	// = dt_{L-1}, the widened simulated-band substep then the interior-only
	// substep), nesting the finer level's pairs and the mid/end-sync
	// transfers inside; the per-level cumulative substep counters
	// (LBM::totalSubstepCount) drive every parity/rotation argument.
	// max_level == L means the inner blocks are unreachable and the emission
	// is exactly the historical two-substep sequence at that level
	void advancePair(int level, bool compute_macro, bool sync_macro);

	// build \ref couplings from the `GEO_AMR_INTERFACE` markings (called by
	// SimInit AFTER `markAMRInterface` ran); see the implementation for the
	// face partition and the validity test
	void buildCouplings();

#ifndef NDEBUG
	// registration map-pattern guard (Schönherr-ch7 conversion, plan
	// T6/Oracle F2, debug builds only): static structural check that the
	// \ref buildCouplings patch geometry agrees with the map tags; returns
	// false on the first violation (SimInit sets the terminate flag then)
	bool checkCouplingMapPattern();
#endif
	// coarse cell (x,y,z) (parent-level global coordinates) is a valid
	// PARENT-level interface cell only if no other fine block at
	// `fine_level` covers it (same-level abutment: the cell is shadowed by
	// the other fine block's footprint and must not couple to `owner`)
	bool isShadowedBySameLevelBlock(idx x, idx y, idx z, int fine_level, const BLOCK_NSE* owner);
	// block at `level` with block id `id` (nullptr if none; ids are unique
	// across all levels)
	BLOCK_NSE* findBlockById(int level, int id);

	// per-level lattice viscosity nu_lb of a block: level-0 blocks use the
	// global lattice (their lat_local is only initialized by the level-aware
	// block constructor); finer levels use their per-level lat_local
	real blockLbmViscosity(const BLOCK_NSE& block) const
	{
		return block.level == 0 ? this->nse.lat.lbmViscosity() : block.lat_local.lbmViscosity();
	}

	// per-fine-block 6-bit mask of the fine-level bounce-back walls a
	// simulation imposes on the footprint's faces (sim_AMR/sim_AMR_channel.cu
	// imposes one on the z-min face): bit F set means the block holds its
	// own GEO_WALL row one row OUTSIDE face F's C2F destination band (local
	// index -2 on a min face / local+1 on a max face -- the wall link plane
	// then coincides with the coarse wall's link plane), backed by a
	// GEO_NOTHING streaming buffer one row further out (the AA pattern's
	// unclamped neighbor reads, kernels.h, need one allocated row beyond
	// any processed cell). The masked face is BC-managed end to end: it
	// receives no coarse-to-fine fill (the destination is emptied in
	// buildCouplings) and the fine kernel processes the wall row in every
	// fine substep (kernelLaunchWindow). Filled by buildFineWallMasks at
	// SimInit from the IMMEDIATE PARENT level's boundary map (the wall
	// chain: a nested wall-shared face is backed by the parent's own fine
	// wall row, plan sec. 5.1); keyed by block id, missing entries read as
	// 0 (no fine wall).
	std::map<int, std::uint8_t> fine_wall_masks;

	// bit of a footprint face in fine_wall_masks, in the face order of
	// buildCouplings' faces[] array (the two bits of one axis are adjacent,
	// so bit / 2 is the face's normal axis)
	static constexpr int fineWallFaceBit(TNL::Containers::SyncDirection face)
	{
		using SyncDirection = TNL::Containers::SyncDirection;
		switch (face) {
			case SyncDirection::Left:  // x-min
				return 0;
			case SyncDirection::Right:	// x-max
				return 1;
			case SyncDirection::Bottom:	 // y-min
				return 2;
			case SyncDirection::Top:  // y-max
				return 3;
			case SyncDirection::Back:  // z-min
				return 4;
			case SyncDirection::Front:	// z-max
				return 5;
			default:  // not a face normal (diagonal or None sync direction)
				return -1;
		}
	}

	// face names matching the bit order of fineWallFaceBit (the log lines
	// and the fail-fast assertion messages)
	static const char* fineWallFaceName(TNL::Containers::SyncDirection face)
	{
		using SyncDirection = TNL::Containers::SyncDirection;
		switch (face) {
			case SyncDirection::Left:
				return "x-min";
			case SyncDirection::Right:
				return "x-max";
			case SyncDirection::Bottom:
				return "y-min";
			case SyncDirection::Top:
				return "y-max";
			case SyncDirection::Back:
				return "z-min";
			case SyncDirection::Front:
				return "z-max";
			default:
				return "unknown";
		}
	}

	// the block's fine-wall mask (0 when the block carries no fine wall)
	std::uint8_t fineWallMask(const BLOCK_NSE& block) const
	{
		const auto it = fine_wall_masks.find(block.id);
		return it == fine_wall_masks.end() ? 0 : it->second;
	}

	// per-axis (begin, size) launch window of the block's cudaLBMKernel: an
	// unmasked axis covers [-g, local+g) with g = min(ghost_layers,
	// allocated overlap) -- the widened simulated-band extent of substep 1
	// at g = 1, the interior-only extent of substep 2 at g = 0. An axis
	// whose min face is masked in fine_wall_masks is deepened to the
	// GEO_WALL row at local -2, an axis whose max face is masked to the
	// GEO_WALL row at local+1 (the wall rows are processed in BOTH
	// substeps: the bounce-back refreshes the wall's slots in each
	// substep's frame; the GEO_NOTHING buffer row one row further out is
	// never processed -- a processed cell's streaming gather reaches at
	// most one row beyond the wall).
	std::pair<idx3d, idx3d> kernelLaunchWindow(BLOCK_NSE& block, int ghost_layers) const
	{
		using SyncDirection = TNL::Containers::SyncDirection;
		static constexpr SyncDirection min_face[3] = {SyncDirection::Left, SyncDirection::Bottom, SyncDirection::Back};
		static constexpr SyncDirection max_face[3] = {SyncDirection::Right, SyncDirection::Top, SyncDirection::Front};
		const std::uint8_t mask = fineWallMask(block);
		const idx local[3] = {block.local.x(), block.local.y(), block.local.z()};
		const idx ov[3] = {block.df_overlap_X(), block.df_overlap_Y(), block.df_overlap_Z()};
		idx3d begin, size;
		for (int a = 0; a < 3; a++) {
			const idx g = std::min<idx>(ghost_layers, ov[a]);
			begin[a] = (mask & (1 << fineWallFaceBit(min_face[a]))) ? idx(-2) : -g;
			const idx end = (mask & (1 << fineWallFaceBit(max_face[a]))) ? local[a] + 2 : local[a] + g;
			size[a] = end - begin[a];
		}
		return {begin, size};
	}

	// the interior [0, local) window is the only launch window the block's
	// precomputed grid covers: that grid rounds `local` up to a whole number
	// of blocks with zero slack, so any window extending past `local`
	// (a widened substep-1 extent or a wall-deepened MAX face whose begin
	// still starts at 0) would leave its outer rows unlaunched if the
	// precomputed grid were reused
	static bool isInteriorLaunchWindow(const idx3d& begin, const idx3d& size, const idx3d& local)
	{
		return begin.x() == 0 && begin.y() == 0 && begin.z() == 0 && size == local;
	}

	// derive fine_wall_masks from the COARSE boundary map (see SimInit for
	// the ordering); hard-fails with std::runtime_error on a partial wall
	// or on a wall without the per-axis storage override
	void buildFineWallMasks();

	// R4 wall-pedestal prisms of one fine block (the plan's sec. 5.3 refill
	// mechanism): on a wall-shared face of a NESTED block (level >= 2) the
	// parent's own-8 upward fine-to-coarse window reads the footprint's
	// frozen rows at face-normal depths {2,3} below the depth-1 skin
	// (relative rows {1,2} of the wall pedestal), which are never authored
	// by the standard transfers -- the prisms are the extra F2C destination
	// rectangles covering exactly those rows. Returns the disjoint
	// parent-frame [begin, end) prism rectangles ([go+2, go+4) on the
	// face-normal axis of a masked min face, clamped by the opposite face's
	// structures; twice-inset tangents [go+2, go+gs-2) on both tangent
	// axes), empty for level-1 blocks (their parent's level-0 lattice has
	// no upward transfer to feed) and for axes with no twice-inset tangent
	// range (a thin footprint carries no unreachable deep core at all)
	static std::vector<std::pair<idx3d, idx3d>> wallPedestalPrismRects(const BLOCK_NSE& fine, std::uint8_t mask);

	// host-side reduction over all blocks: volume-weighted global mass and
	// momentum plus per-level kinetic energy (see AfterSimUpdate); kept
	// public like the other implementation details above so that unit tests
	// can drive it directly (test_amr_subcycling)
	AMRConservationStats computeConservationStats();
};

/**
 * \brief Launch `cudaLBMKernel` on every block at the given level.
 *
 * v1 single-GPU: all blocks are local, so the interior
 * (`SyncDirection::None`) launch configuration is used over the FULL block
 * extent on the null stream (matches the `nproc == 1` path of the base
 * driver). `GEO_AMR_INTERFACE` cells are collision-active inside the kernel
 * (they stream and collide like fluid) and may additionally be overwritten
 * by the inter-level fine-to-coarse transfer; fine-level ghost cells are
 * maintained by the coupling kernels, not by MPI overlap synchronization.
 */
template <typename NSE>
void State_AMR<NSE>::launchLBMKernelForLevel(int level, bool compute_macro, int ghost_layers)
{
	for (auto* block : this->nse.getBlocksAtLevel(level)) {
		const auto direction = TNL::Containers::SyncDirection::None;
		TNL::Backend::LaunchConfiguration launch_config;
		launch_config.blockSize = block->computeData.at(direction).blockSize;
		// launch window: cover the inner overlap rows (bounded by the
		// allocated overlap, which is 2 on refinement-level blocks); the
		// kernel's neighbor clamps (kernels.h) then reach the outer
		// overlap row as the streaming source, filled by the C2F transfer.
		// An axis masked in fine_wall_masks is deepened to the fine wall's
		// GEO_WALL row instead (see kernelLaunchWindow)
		const auto [begin, size] = kernelLaunchWindow(*block, ghost_layers);
		launch_config.gridSize = isInteriorLaunchWindow(begin, size, block->local) ? block->computeData.at(direction).gridSize
																				   : block->getCudaGridSize(size, launch_config.blockSize);
		TNL::Backend::launchKernelAsync(cudaLBMKernel<NSE>, launch_config, block->data, begin, begin + size, block->is_distributed(), compute_macro);
	}
	// synchronize the null-stream after all grids (same as the base driver)
	TNL::Backend::streamSynchronize(0);
}

/**
 * \brief Berger-Colella initialization hook: tags the coarse-fine
 * interfaces and builds the coupling descriptors (todo 8).
 *
 * The base SimInit establishes everything this hook needs:
 * - `State::SimInit` calls `reset()`, which calls `setupBoundaries()` and
 *   `copyMapToDevice()` (skipped on the `loadstate` path, where the map is
 *   loaded from the checkpoint instead) - i.e. the COARSE BOUNDARY MAP IS
 *   COMPLETE IN HOST MEMORY when Base::SimInit() returns;
 * - all fine-level blocks were created by `createAMRBlocks` before
 *   `execute()` and are allocated by the base SimInit together with the
 *   level-0 blocks.
 *
 * ORDERING CONTRACT: `markAMRInterface` REQUIRES the coarse boundary map
 * to be set (it only re-tags GEO_FLUID cells and must not clobber walls /
 * inflows), so it CANNOT run before `Base::SimInit()`. The base SimInit's
 * initial `synchronizeDFsAndMacroDevice(df_cur, true)` for `nproc > 1` runs
 * before the re-tagging; AMR v1 is single-rank (`createAMRBlocks` rejects
 * `nproc > 1`), so the ordering is harmless there.
 *
 * With `max_level == 0` the initialization is byte-identical to the base
 * driver (acceptance criterion: sim_1 runs unchanged).
 */
template <typename NSE>
void State_AMR<NSE>::SimInit()
{
	Base::SimInit();

	// no refinement levels: identical to the base driver
	if (this->nse.max_level == 0)
		return;

	// tag coarse cells adjacent to fine blocks as GEO_AMR_INTERFACE and
	// upload the updated map to the device (POSITION REQUIREMENT: after the
	// base SimInit set the coarse boundary map - see the docstring)
	markAMRInterface(this->nse);

	// derive the fine-level wall masks AFTER the coarse boundary map is
	// complete (the base SimInit's reset()->setupBoundaries();
	// markAMRInterface above only re-tags GEO_FLUID cells, so the coarse
	// wall planes the scan keys on survive) and BEFORE buildCouplings (the
	// masks empty the masked faces' C2F destinations)
	buildFineWallMasks();

#ifdef F2C_LAGRAVA
	// strategy coupling guard (the plan's sec. 5.4 v1 ruling): the
	// F2C_LAGRAVA filter reads a 4-node window one fine cell deeper than
	// F2C_SCHONHERR's own-8, so a nested wall-shared face's 3-row pedestal
	// (the depth-{1,2,3} destination set of the R4 prisms) underflows it by
	// one row -- the parent's upward fine-to-coarse window would then read
	// a never-written deep-core cell. Nested wall sharing therefore
	// requires the F2C_SCHONHERR default; non-nested (level-1-only) wall
	// runs under the Lagrava opt-out are unaffected
	for (const auto& block : this->nse.blocks)
		if (block.level >= 2 && fineWallMask(block) != 0) {
			const std::string message = fmt::format(
				"State_AMR: wall-shared nesting requires F2C_SCHONHERR: fine block {} at level {} carries wall-shared faces "
				"(fine_wall_masks = {}), but the active fine-to-coarse strategy is F2C_LAGRAVA, whose 4-node window underflows "
				"the wall pedestal's 3-row depth and would read a never-written deep-core cell; rebuild with "
				"-DTNL_LBM_F2C_STRATEGY=F2C_SCHONHERR or remove the nested wall-shared face",
				block.id,
				block.level,
				static_cast<int>(fineWallMask(block))
			);
			spdlog::error("{}", message);
			throw std::runtime_error(message);
		}
#endif

	// build the inter-level coupling descriptors consumed by the transfer
	// launches in SimUpdate()
	buildCouplings();

#ifndef NDEBUG
	// registration map-pattern guard (Schönherr-ch7 conversion, plan T6 /
	// Oracle F2): static structural check that the patch geometry agrees
	// with the map tags -- a host-side abort cannot be xfail-marked, so it
	// lands with the commit-7 band geometry whose nominal source pairs and
	// skin destinations satisfy the asserted pattern by construction
	if (! checkCouplingMapPattern()) {
		spdlog::error("State_AMR: registration map-pattern assertion FAILED (coupling patch geometry disagrees with the map tags); terminating");
		this->nse.terminate = true;
		return;
	}
#endif

	std::size_t total_patches = 0;
	std::size_t total_interior_patches = 0;
	for (const InterLevelCoupling& coupling : couplings) {
		total_patches += coupling.patches.size();
		total_interior_patches += coupling.interior_patches.size();
		spdlog::info(
			"State_AMR: coupling level {} -> {} has {} interface patches, {} interior patches",
			coupling.coarse_level,
			coupling.fine_level,
			coupling.patches.size(),
			coupling.interior_patches.size()
		);
	}
	if (total_patches == 0 && total_interior_patches == 0)
		spdlog::warn(
			"State_AMR: no AMR coupling patches found for max_level = {} - inter-level coupling kernels will not launch", this->nse.max_level
		);

#ifdef USE_CUDA
	// Schönherr cycle (the cycle contract of
	// docs/AMR-schonherr-ch7-target-contract.md sec. 1.3): initial
	// coarse-to-fine fill of the substep-0 rotation frame, identical to
	// step 5 of SimUpdate - cycle 0's substep 1 reads the destinations
	// filled here (substep 2 consumes substep 1's kernel-updated inner
	// ghost rows from the other frame, so no second fill is needed).
	// The fill reads the initial coarse state (the t_0 fill; the startup
	// transient of the contract). Without it cycle 0 would read
	// uninitialized ghost rows.
	for (int L = 1; L <= this->nse.max_level; L++) {
		// the counters are zero at construction (and parity-consistent on a
		// restart landing at a cycle boundary), so this argument is 0 --
		// the same substep-0 rotation the positional preparation selected
		this->nse.updateKernelDataForLevel(L, this->nse.totalSubstepCount[L]);
		launchCoarseToFineTransfers(L);
	}
	// the SimInit fill's single sync point: the ghost-macro seeding below
	// reads the filled ghost DFs, so both transfer streams must be drained
	synchronizeTransfers();

	// seed the ghost-row macros from the SimInit C2F fill: frame 0000 is
	// emitted before any kernel ran, and computeInitialMacro covers only the
	// interior [0, local), so without this the fine ghost rows would carry
	// the zero-init dmacro in the t=0 snapshot. Recompute the SAME window
	// the cycle-0 substep-1 kernel will use (ghost_layers = 1), which yields
	// identical macros from the C2F-seeded DFs with no physics run; the
	// interior macros are recomputed identically too. fine_wall_masks is
	// already built, so masked-wall faces carry their GEO_WALL row in the
	// window consistently.
	for (auto& block : this->nse.blocks) {
		if (block.level == 0)
			continue;
		const auto [begin, size] = kernelLaunchWindow(block, /*ghost_layers=*/1);
		block.computeInitialMacro(begin, begin + size);
	}
#endif
}

/**
 * \brief Derive \ref fine_wall_masks from the IMMEDIATE PARENT's boundary
 * map (the wall-chain resolution rule of the amr-nlevel-nesting plan, sec.
 * 5.1: key on the parent level's map, never the level-0 map at depth).
 *
 * For each fine block and each of the footprint's six faces the block's
 * interior cross-section columns are scanned (tangential fine-local
 * indices in [0, local), parent-frame cross-coordinates floor(fine/2)): a
 * column is "wall" iff the matching parent-level column on the
 * face-adjacent parent plane (the halo row go_a - 1 on a min face /
 * go_a + gs_a on a max face) is GEO_WALL. The band geometry places the
 * fine wall row one row OUTSIDE the face's C2F destination band, so the
 * wall link plane coincides with the parent wall's link plane at every
 * level (the fine rows themselves are tagged by the simulation's
 * setupBoundaries).
 *
 * The parent-plane scan is a SINGLE division-by-2 hop from the block's own
 * fine-global coordinates (the Commit-A parent-frame normalization makes
 * the re-anchored offset positive at every level, so floor(fine/2) is
 * exact at any depth) against the parent's STORAGE extent
 * [offset - ov, offset + local + ov): on a wall-shared nested face the
 * backing wall lives on the parent's OWN fine wall row at parent-local -2
 * (min face) / local+1 (max face), inside the overlap-extended host-map
 * storage but outside the interior clip, so the scan must not be clipped
 * to the interior (at level 1 the parent planes lie inside the level-0
 * interior and the storage-extent bounds reduce to the historical scan
 * column-for-column). Blocks are visited in creation order, which is
 * level-ascending by V5: the parent's own map is fully tagged before any
 * deeper level's scan reads it.
 *
 * Fail-fast, no silent lanes:
 * - a PARTIAL wall (count strictly between 0 and the full cross-section)
 *   throws std::runtime_error naming block, face, count, and expected
 *   count -- the launch window and the coarse-to-fine fill are face-wide
 *   decisions, a column-wise mixture has no defined contract;
 * - a full wall without the per-axis storage override (df_overlap < 3 on
 *   the walled axis) throws std::runtime_error naming block, face, and
 *   the override to set -- with a tagged wall row and only the 2-deep C2F
 *   band allocated, the GEO_NOTHING streaming-buffer row would lie
 *   outside the storage and the coupling patch would overwrite the wall
 *   columns;
 * - a fine block whose own wall row of a face carries GEO_WALL tags
 *   WITHOUT any wall backing on the parent hop (count == 0) throws
 *   std::runtime_error naming block and face: the mask machinery would
 *   silently leave the face unmasked while its wall row is BC-affected;
 *   the wall chain is broken at exactly this level and the unhandled wall
 *   columns are a mass source.
 *
 * Host-side only (parent-level hmap reads); runs once per SimInit.
 */
template <typename NSE>
void State_AMR<NSE>::buildFineWallMasks()
{
	using SyncDirection = TNL::Containers::SyncDirection;

	fine_wall_masks.clear();

	// (face, normal axis, min/max side) in the face order of
	// fineWallFaceBit / buildCouplings' faces[] array
	const struct FACE_SCAN
	{
		SyncDirection face;
		int axis;
		bool min_side;
	} scan[6] = {
		{SyncDirection::Left, 0, true},
		{SyncDirection::Right, 0, false},
		{SyncDirection::Bottom, 1, true},
		{SyncDirection::Top, 1, false},
		{SyncDirection::Back, 2, true},
		{SyncDirection::Front, 2, false},
	};

	for (auto& fine : this->nse.blocks) {
		if (fine.level == 0)
			continue;
		const std::vector<BLOCK_NSE*> parents = this->nse.getBlocksAtLevel(fine.level - 1);
		// footprint origin/extent in parent-level cells (the same
		// re-anchored registration as buildCouplings)
		const idx3d& go = fine.global_offset;
		const idx3d gs{(fine.local.x() + 2) / 2, (fine.local.y() + 2) / 2, (fine.local.z() + 2) / 2};
		const idx go3[3] = {go.x(), go.y(), go.z()};
		const idx gs3[3] = {gs.x(), gs.y(), gs.z()};
		const idx off3[3] = {fine.offset.x(), fine.offset.y(), fine.offset.z()};
		const idx loc3[3] = {fine.local.x(), fine.local.y(), fine.local.z()};
		const idx ov3[3] = {fine.df_overlap_X(), fine.df_overlap_Y(), fine.df_overlap_Z()};

		std::uint8_t mask = 0;
		for (const FACE_SCAN& fs : scan) {
			const int a = fs.axis;
			const int b = (a + 1) % 3;
			const int c = (a + 2) % 3;
			// face-adjacent parent plane: the halo row one coarse cell OUTSIDE the footprint
			const idx plane = fs.min_side ? go3[a] - 1 : go3[a] + gs3[a];
			const idx expected = loc3[b] * loc3[c];
			idx count = 0;
			for (idx ib = 0; ib < loc3[b]; ib++)
				for (idx ic = 0; ic < loc3[c]; ic++) {
					idx cg[3];
					cg[a] = plane;
					// floor(fine/2) is exact integer division at every level:
					// the re-anchored fine-global coordinates of the parent-frame
					// normalization are positive on all levels (offset = 2 *
					// (origin >> (level - 1)) + 1, amrFineOffset) -- a single
					// hop into the parent frame, no 2^(L-1) chain arithmetic
					cg[b] = (off3[b] + ib) / 2;
					cg[c] = (off3[c] + ic) / 2;
					for (BLOCK_NSE* coarse : parents) {
						// bounds check first against the parent's STORAGE
						// extent [offset - ov, offset + local + ov), not the
						// interior clip: on a wall-shared nested face the
						// parent's own fine wall row (parent-local -2 /
						// local+1) carries the backing tag inside the
						// overlap-extended host-map storage (at level 1 the
						// parent planes lie inside the level-0 interior and
						// this admits exactly the historical column set)
						const idx cbeg[3] = {
							coarse->offset.x() - coarse->df_overlap_X(),
							coarse->offset.y() - coarse->df_overlap_Y(),
							coarse->offset.z() - coarse->df_overlap_Z()
						};
						const idx cend[3] = {
							coarse->offset.x() + coarse->local.x() + coarse->df_overlap_X(),
							coarse->offset.y() + coarse->local.y() + coarse->df_overlap_Y(),
							coarse->offset.z() + coarse->local.z() + coarse->df_overlap_Z()
						};
						if (cg[0] < cbeg[0] || cg[0] >= cend[0] || cg[1] < cbeg[1] || cg[1] >= cend[1] || cg[2] < cbeg[2] || cg[2] >= cend[2])
							continue;
						if (coarse->hmap(cg[0], cg[1], cg[2]) == NSE::BC::GEO_WALL) {
							count++;
							break;	// one parent block owns the column (v1 single-block parent level)
						}
					}
				}
			if (count == 0) {
				// no parent wall on this hop: a fine-level wall the mask
				// machinery cannot see has no defined contract -- if the
				// block's own wall row of this face carries GEO_WALL tags
				// the wall chain is silently broken here (the face would
				// stay unmasked while the wall row is BC-affected: unfilled,
				// unlaunched wall columns are a mass source). A plain-fluid
				// own row is the ordinary no-wall lane and continues
				// silently
				const idx own_row = fs.min_side ? off3[a] - 2 : off3[a] + loc3[a] + 1;
				idx own_wall = 0;
				for (idx ib = 0; ib < loc3[b]; ib++)
					for (idx ic = 0; ic < loc3[c]; ic++) {
						idx fg[3];
						fg[a] = own_row;
						fg[b] = off3[b] + ib;
						fg[c] = off3[c] + ic;
						if (fine.hmap(fg[0], fg[1], fg[2]) == NSE::BC::GEO_WALL)
							own_wall++;
					}
				if (own_wall > 0) {
					const std::string message = fmt::format(
						"State_AMR: fine block {} has GEO_WALL tags on its own {} wall row but no wall backing on the parent level "
						"(0 of {} face-adjacent parent columns are GEO_WALL): a fine wall the mask machinery cannot see silently "
						"breaks the wall chain (the face stays unmasked, its wall columns receive no coarse-to-fine fill and the "
						"launch windows do not cover them -- a mass-source error); tag the parent level's wall row or remove the "
						"fine wall tags",
						fine.id,
						fineWallFaceName(fs.face),
						expected
					);
					spdlog::error("{}", message);
					throw std::runtime_error(message);
				}
				continue;
			}
			if (count != expected) {
				const std::string message = fmt::format(
					"State_AMR: fine block {} has a PARTIAL fine-level wall on the {} face: {} of {} interior cross-section "
					"columns are backed by GEO_WALL on the face-adjacent coarse plane (a fine wall must cover the full face "
					"cross-section -- the launch window and the coarse-to-fine fill are face-wide decisions)",
					fine.id,
					fineWallFaceName(fs.face),
					count,
					expected
				);
				spdlog::error("{}", message);
				throw std::runtime_error(message);
			}
			// a masked wall REQUIRES the 3-deep per-axis overlap on the
			// walled axis (see LBM_BLOCK::storage_overlap_*): the GEO_WALL
			// row sits one row outside the 2-deep C2F band and the
			// GEO_NOTHING streaming buffer one row further out
			if (ov3[a] < 3) {
				const std::string message = fmt::format(
					"State_AMR: fine block {} imposes its own GEO_WALL on the {} face but the {}-axis overlap is {} (< 3); set "
					"storage_overlap_{} = 3 on the fine block after createAMRBlocks so the wall's GEO_NOTHING streaming-buffer "
					"row is allocated",
					fine.id,
					fineWallFaceName(fs.face),
					static_cast<char>('x' + a),
					ov3[a],
					static_cast<char>('x' + a)
				);
				spdlog::error("{}", message);
				throw std::runtime_error(message);
			}
			mask |= std::uint8_t(1) << fineWallFaceBit(fs.face);
			spdlog::info(
				"State_AMR: fine block {} imposes its own GEO_WALL on the {} face; the face's coarse-to-fine fill is dropped",
				fine.id,
				fineWallFaceName(fs.face)
			);
		}
		if (mask != 0)
			fine_wall_masks[fine.id] = mask;
	}
}

template <typename NSE>
std::vector<std::pair<typename State_AMR<NSE>::idx3d, typename State_AMR<NSE>::idx3d>>
State_AMR<NSE>::wallPedestalPrismRects(const BLOCK_NSE& fine, std::uint8_t mask)
{
	std::vector<std::pair<idx3d, idx3d>> prisms;
	if (fine.level < 2 || mask == 0)
		return prisms;

	const idx3d& go = fine.global_offset;
	const idx3d gs{(fine.local.x() + 2) / 2, (fine.local.y() + 2) / 2, (fine.local.z() + 2) / 2};

	for (int a = 0; a < 3; a++) {
		const int b = (a + 1) % 3;
		const int c = (a + 2) % 3;
		// pedestal rows on the face-normal axis: face-normal depths {2,3}
		// (relative rows {1,2} behind the depth-1 skin), clamped into the
		// deep band [go+2, go+gs-2) so they never re-author the opposite
		// face's skin or either ring row; the min-side and max-side ranges
		// of one axis are merged when both faces are wall-shared on a thin
		// axis (the authorings are identical F2C projections, but the
		// rectangles must stay disjoint for the census)
		const bool min_masked = (mask & (std::uint8_t(1) << (2 * a))) != 0;
		const bool max_masked = (mask & (std::uint8_t(1) << (2 * a + 1))) != 0;
		idx p_begin = 0, p_end = 0;
		if (min_masked && max_masked) {
			p_begin = go[a] + 2;
			p_end = go[a] + gs[a] - 2;
		}
		else if (min_masked) {
			p_begin = go[a] + 2;
			p_end = std::min(go[a] + 4, go[a] + gs[a] - 2);
		}
		else if (max_masked) {
			p_begin = std::max(go[a] + gs[a] - 4, go[a] + 2);
			p_end = go[a] + gs[a] - 2;
		}
		else
			continue;
		idx3d begin{0, 0, 0}, end{0, 0, 0};
		begin[a] = p_begin;
		end[a] = p_end;
		// twice-inset tangents (the wall-face tangent rectangle idiom):
		// empty on any axis whose footprint is too thin to carry a deep
		// core -- such a prism must not be emitted at all
		for (int t : {b, c}) {
			begin[t] = go[t] + 2;
			end[t] = go[t] + gs[t] - 2;
		}
		if (begin.x() >= end.x() || begin.y() >= end.y() || begin.z() >= end.z())
			continue;
		prisms.emplace_back(begin, end);
	}
	return prisms;
}

/**
 * \brief Logs the AMR conservation statistics at the PRINT interval.
 *
 * The PRINT trigger must be captured BEFORE calling the base
 * implementation, because `State::AfterSimUpdate()` increments
 * `cnt[PRINT].count` (state.hpp) and afterwards `action()` no longer
 * reports the trigger for this step - the same capture-before-base pattern
 * as in `State_NSE_ADE::AfterSimUpdate()`. With `max_level == 0` the
 * behavior is identical to the base driver.
 */
template <typename NSE>
void State_AMR<NSE>::AfterSimUpdate()
{
	// VTKHDF AMR output at the OUT3D cadence (co-exists with base BP5 output);
	// trigger check BEFORE Base (Base increments count)
	if (this->nse.max_level > 0 && this->cnt[OUT3D].action(this->nse.physTime()))
		this->write3D_AMR(this->nse.physTime(), this->cnt[OUT3D].count);

	// trigger check BEFORE Base (Base increments count)
	const bool do_amr_report = this->nse.max_level > 0 && this->cnt[PRINT].action(this->nse.physTime());

	Base::AfterSimUpdate();

	if (! do_amr_report)
		return;

	AMRConservationStats s = computeConservationStats();
	spdlog::info("AMR conservation: mass = {:.6e}", s.total_mass);
	for (std::size_t L = 0; L < s.per_level_kinetic_energy.size(); L++)
		spdlog::info("AMR level {}: kinetic energy = {:.6e}", L, s.per_level_kinetic_energy[L]);
}

/**
 * \brief Write the AMR block hierarchy to one VTKHDF OverlappingAMR file.
 *
 * Called from \ref AfterSimUpdate at the OUT3D cadence (co-exists with the
 * base driver's BP5 output). The host mirrors of the macroscopic quantities
 * are refreshed explicitly because macros are not computed in every
 * iteration when `MACRO::compute_in_each_iteration == false` (e.g.
 * D3Q27_MACRO_Default); the writer itself only copies, it does not
 * recompute.
 */
template <typename NSE>
void State_AMR<NSE>::write3D_AMR(real time, int cycle)
{
	if (this->nse.max_level == 0)
		return;

	// single-file VTKHDF per step; cycle is OUT3D's current counter
	// (pre-increment - Base::AfterSimUpdate increments it later)
	const std::string fname = fmt::format("results_{}/output_amr_{:04d}.vtkhdf", this->id, cycle);

	// ensure the host-side macros are up to date before the writer pulls
	// them (copyMacroToHost() is non-const; the writer takes care of the rest)
	for (auto& block : this->nse.blocks)
		block.copyMacroToHost();

	OverlappingAMRWriter<TRAITS>::write(fname, this->nse, time);
}

/**
 * \brief Host-side reduction of the conservation quantities over all blocks.
 *
 * Each block's macroscopic quantities are copied to the host and summed in
 * physical-volume units: the cell volume scales as `(1/2)^3` per refinement
 * level with the 2:1 ratio, so a level-L cell weights `1/8^L` of a coarse
 * cell. The per-level kinetic energy sums `0.5 * rho * |u|^2` without the
 * volume weight (per-level diagnostic).
 *
 * Coarse cells tagged `GEO_NOTHING` (hidden under a fine footprint - the
 * fine level holds the authoritative solution there) are excluded from all
 * sums: counting them alongside the fine level would double-count the
 * refined region. `GEO_AMR_INTERFACE` ring cells and physical-BC cells are
 * real coarse cells and keep contributing.
 *
 * Frame note (nesting): the metric is evaluated exclusively at cycle
 * boundaries (the coarse-clock hook of \ref AfterSimUpdate), where every
 * level's last launch of the cycle was its interior substep at the cycle's
 * end time and every fine level sits in its substep-0 rotation (each cycle
 * advances each level's counter by an even 2^L), so the per-level sums at
 * any nesting depth mix macros of identical physical time -- the same
 * end-of-cycle snapshot the max_level == 1 schedule produced, with the
 * macro array itself carrying no rotation/parity dependence.
 */
template <typename NSE>
auto State_AMR<NSE>::computeConservationStats() -> AMRConservationStats
{
	using MACRO = typename NSE::MACRO;
	AMRConservationStats s;
	s.per_level_kinetic_energy.resize(this->nse.max_level + 1, 0.0);

	for (auto& block : this->nse.blocks) {
		block.copyMacroToHost();
		// cell volume scales as (1/2)^3 per refinement level (2:1 ratio)
		const double volume_factor = std::pow(0.5, 3.0 * block.level);

		// conservation accumulated in a DETERMINISTIC order:
		// forLocalLatticeSites is OpenMP-parallel with schedule(static), so
		// the site->thread assignment is fixed; per-thread partials are
		// written by their own thread only (no atomics, no arrival-order
		// reassociation) and merged in thread-index order, reproducing one
		// summation order bit-for-bit regardless of host timing (the
		// previous atomic accumulations reassociated with the barrier
		// timing of the asynchronous-coupling era)
		const int threads = omp_get_max_threads();
		std::vector<double> p_mass(threads, 0.0), p_mx(threads, 0.0), p_my(threads, 0.0), p_mz(threads, 0.0), p_ke(threads, 0.0);

		block.forLocalLatticeSites(
			[&](BLOCK_NSE& b, idx x, idx y, idx z)
			{
				// hidden cells (frozen GEO_NOTHING under a fine footprint)
				// are already counted on the fine level - skip them to avoid
				// double-counting the refined region (see the docstring);
				// the host map holds the tags (markAMRInterface tags
				// host-side before uploading to the device)
				if (b.hmap(x, y, z) == NSE::BC::GEO_NOTHING)
					return;

				const double rho = b.hmacro(MACRO::e_rho, x, y, z);
				const double vx = b.hmacro(MACRO::e_vx, x, y, z);
				const double vy = b.hmacro(MACRO::e_vy, x, y, z);
				const double vz = b.hmacro(MACRO::e_vz, x, y, z);

				const int t = omp_get_thread_num();
				p_mass[t] += rho * volume_factor;
				p_mx[t] += rho * vx * volume_factor;
				p_my[t] += rho * vy * volume_factor;
				p_mz[t] += rho * vz * volume_factor;
				p_ke[t] += 0.5 * rho * (vx * vx + vy * vy + vz * vz);
			}
		);

		double block_ke = 0.0;
		for (int t = 0; t < threads; t++) {
			s.total_mass += p_mass[t];
			s.total_momentum_x += p_mx[t];
			s.total_momentum_y += p_my[t];
			s.total_momentum_z += p_mz[t];
			block_ke += p_ke[t];
		}
		s.per_level_kinetic_energy[block.level] += block_ke;
	}
	return s;
}

/**
 * \brief Build \ref couplings from the `GEO_AMR_INTERFACE` markings.
 *
 * For each fine block at level L and each of the six faces of its parent
 * footprint's 1-cell halo box (disjoint partition: the x-faces span the
 * full y/z halo, the y-faces the interior x-range, the z-faces the
 * interior x/y range), the face rectangle is clipped against every
 * level-(L-1) block's range (global parent-level coordinates - the same
 * overlap test as `markAMRInterface`). A patch is appended iff the clipped
 * rectangle holds at least one VALID parent-level interface cell:
 * - tagged `GEO_AMR_INTERFACE` on the parent block's (host) map, and
 * - not shadowed by another fine block at level L (same-level abutment:
 *   cells inside another fine block's footprint are marked by
 *   `markAMRInterface` but must not couple to the PARENT level's data).
 *
 * Each patch covers a rectangle of PARENT-level ring cells (TWO cells
 * thick in the face normal, spanning the nominal coarse-to-fine source
 * pair {c=-1, c=0} of the Schönherr ch.7 band map on the min faces,
 * mirrored at {c=gs-1, c=gs} on the max faces) and the matching
 * fine-level destination rectangle of the face (the face's disjoint
 * partition of the overlap complement: refinement-level blocks allocate
 * a 2-deep DF overlap (see `LBM_BLOCK::storage_overlap`), so the
 * destination spans the face's own ghost rows [-ov, 0) / [local,
 * local+ov) with the tangential ownership cascading x -> y -> z as for
 * the coarse ring faces). The coarse-to-fine launch clips each such
 * rectangle face-aware to the block's allocated ghost STORAGE (see
 * `launchCoarseToFineTransfers`): the filled destination rows are read
 * by the fine-level streaming and by the max-side skin fine-to-coarse
 * windows. Both rectangles are stored in the two
 * blocks' indexer coordinates: with the 2:1 refinement ratio, fine cell
 * `2c` covers coarse cell `c`. The launches map
 * `begin = fine_origin, end = fine_origin + fine_size` for coarse-to-fine
 * and `begin = coarse_origin, end = coarse_origin + coarse_size` for
 * fine-to-coarse, subject to the storability guards documented at the
 * launch helpers.
 *
 * The INTERIOR (under-footprint) patch list holds the footprint's 6
 * disjoint inset-face SKIN rectangles (the depth-1 shell one coarse row
 * inside the reactivated c=0 ring row, the same disjoint partition idiom
 * as the halo faces above).
 * They carry the ONLY fine-to-coarse feedback channel (changes 2+3 of the
 * AMR interface redesign): the ring fine-to-coarse launch was removed
 * (gate B ruling, D.1 hard-delete), the deep frozen core is never written,
 * and the collision-active ring cells are driven by the coarse kernel only
 * (see `launchFineToCoarseTransfersInterior`).
 *
 * The whole function is host-side (hmap reads); it runs once per
 * simulation.
 */
template <typename NSE>
void State_AMR<NSE>::buildCouplings()
{
	using SyncDirection = TNL::Containers::SyncDirection;

	couplings.clear();

	for (int fine_level = 1; fine_level <= this->nse.max_level; fine_level++) {
		const int coarse_level = fine_level - 1;

		InterLevelCoupling coupling;
		coupling.coarse_level = coarse_level;
		coupling.fine_level = fine_level;

		for (auto* fine : this->nse.getBlocksAtLevel(fine_level)) {
			// footprint of the fine block in parent-level global coordinates
			// (createAMRBlocks stores the parent-level origin in global_offset;
			// the footprint size is (fine.local + 2) / 2 with the 2:1 ratio --
			// the re-anchored interior local = 2*gs - 2 is inset one fine
			// cell per face)
			const idx3d& go = fine->global_offset;
			const idx3d gs{(fine->local.x() + 2) / 2, (fine->local.y() + 2) / 2, (fine->local.z() + 2) / 2};

			// the C2F destination band is structurally TWO ghost rows deep
			// per face (the simulated band of the cycle contract: one
			// integrated inner row + one passive streaming-source row);
			// deeper allocated overlap on an axis is a pure streaming
			// buffer (e.g. the GEO_NOTHING row below a fine wall) and
			// never carries destinations
			const idx3d fov{std::min<idx>(fine->df_overlap_X(), 2), std::min<idx>(fine->df_overlap_Y(), 2), std::min<idx>(fine->df_overlap_Z(), 2)};
			// faces masked in fine_wall_masks are BC-managed end to end
			// (see buildFineWallMasks): their destination is emptied
			// below, so no coarse-to-fine fill reaches them
			const std::uint8_t fine_wall_mask = fineWallMask(*fine);

			// the six faces of the footprint's RING in parent-level global
			// coordinates (the disjoint ring partition of the Schönherr
			// ch.7 band map, docs/AMR-schonherr-ch7-target-contract.md
			// sec. 2.1: ring = (K+2)^3 - (K-2)^3 = halo + reactivated c=0
			// shell): each rectangle is TWO cells thick in the face normal
			// and spans the C2F nominal source pair {c=-1, c=0} (min faces)
			// / {c=gs-1, c=gs} (max faces); the x-normal faces own the full
			// y/z halo tangent [go-1, go+gs+1), the y-normal faces the
			// once-inset x range [go+1, go+gs-1), the z-normal faces the
			// twice-inset x/y ranges, so the six faces partition the ring
			// disjointly (e.g. a K = 8 footprint: 200+200+120+120+72+72 =
			// 784 cells)
			const struct FACE
			{
				SyncDirection face;
				idx3d begin, end;
			} faces[6] = {
				{SyncDirection::Left, {go.x() - 1, go.y() - 1, go.z() - 1}, {go.x() + 1, go.y() + gs.y() + 1, go.z() + gs.z() + 1}},
				{SyncDirection::Right,
				 {go.x() + gs.x() - 1, go.y() - 1, go.z() - 1},
				 {go.x() + gs.x() + 1, go.y() + gs.y() + 1, go.z() + gs.z() + 1}},
				{SyncDirection::Bottom, {go.x() + 1, go.y() - 1, go.z() - 1}, {go.x() + gs.x() - 1, go.y() + 1, go.z() + gs.z() + 1}},
				{SyncDirection::Top, {go.x() + 1, go.y() + gs.y() - 1, go.z() - 1}, {go.x() + gs.x() - 1, go.y() + gs.y() + 1, go.z() + gs.z() + 1}},
				{SyncDirection::Back, {go.x() + 1, go.y() + 1, go.z() - 1}, {go.x() + gs.x() - 1, go.y() + gs.y() - 1, go.z() + 1}},
				{SyncDirection::Front,
				 {go.x() + 1, go.y() + 1, go.z() + gs.z() - 1},
				 {go.x() + gs.x() - 1, go.y() + gs.y() - 1, go.z() + gs.z() + 1}},
			};

			for (const FACE& f : faces) {
				for (auto* coarse : this->nse.getBlocksAtLevel(coarse_level)) {
					// clip the face rectangle to this coarse block's range
					// (global parent-level coordinates)
					const idx3d begin{
						std::max(f.begin.x(), coarse->offset.x()),
						std::max(f.begin.y(), coarse->offset.y()),
						std::max(f.begin.z(), coarse->offset.z())
					};
					const idx3d end{
						std::min(f.end.x(), coarse->offset.x() + coarse->local.x()),
						std::min(f.end.y(), coarse->offset.y() + coarse->local.y()),
						std::min(f.end.z(), coarse->offset.z() + coarse->local.z())
					};
					if (begin.x() >= end.x() || begin.y() >= end.y() || begin.z() >= end.z())
						continue;

					// the face couples only where the parent level actually
					// exposes valid interface cells (see the docstring)
					bool has_interface_cell = false;
					for (idx x = begin.x(); x < end.x() && ! has_interface_cell; x++)
						for (idx y = begin.y(); y < end.y() && ! has_interface_cell; y++)
							for (idx z = begin.z(); z < end.z() && ! has_interface_cell; z++)
								if (coarse->hmap(x, y, z) == NSE::BC::GEO_AMR_INTERFACE && ! isShadowedBySameLevelBlock(x, y, z, fine_level, fine))
									has_interface_cell = true;
					if (! has_interface_cell)
						continue;

					AMR_InterfacePatch<NSE> patch;
					// indexer-coordinates rectangles of the two blocks (see
					// the docstring); the COARSE rectangle spans the face's
					// nominal C2F source pair (2 cells thick in the normal)
					patch.coarse_origin = {begin.x() - coarse->offset.x(), begin.y() - coarse->offset.y(), begin.z() - coarse->offset.z()};
					patch.coarse_size = {end.x() - begin.x(), end.y() - begin.y(), end.z() - begin.z()};
					// the FINE rectangle is the face's C2F destination
					// partition of the overlap complement (contract
					// sec. 2.4: stored (local + 2*ov) rows per axis minus
					// the simulated interior; the x-normal faces own the
					// full stored tangent, the y-normal faces the interior
					// x range, the z-normal faces the interior x/y ranges,
					// and the face-normal direction spans this face's ov
					// ghost rows only -- e.g. a K = 8 footprint emits
					// 18^3 - 14^3 = 3,088 cells {648, 648, 504, 504, 392,
					// 392}, written once each). Unlike the coarse
					// rectangle, the destination rectangle belongs to the
					// face and not to the clipped coarse block: with the
					// v1 single-block scope (nproc == 1 only) exactly one
					// coarse block intersects each face, so every
					// destination cell is still written exactly once.
					switch (f.face) {
						case SyncDirection::Left:
							patch.fine_origin = {-fov.x(), -fov.y(), -fov.z()};
							patch.fine_size = {fov.x(), fine->local.y() + 2 * fov.y(), fine->local.z() + 2 * fov.z()};
							break;
						case SyncDirection::Right:
							patch.fine_origin = {fine->local.x(), -fov.y(), -fov.z()};
							patch.fine_size = {fov.x(), fine->local.y() + 2 * fov.y(), fine->local.z() + 2 * fov.z()};
							break;
						case SyncDirection::Bottom:
							patch.fine_origin = {0, -fov.y(), -fov.z()};
							patch.fine_size = {fine->local.x(), fov.y(), fine->local.z() + 2 * fov.z()};
							break;
						case SyncDirection::Top:
							patch.fine_origin = {0, fine->local.y(), -fov.z()};
							patch.fine_size = {fine->local.x(), fov.y(), fine->local.z() + 2 * fov.z()};
							break;
						case SyncDirection::Back:
							patch.fine_origin = {0, 0, -fov.z()};
							patch.fine_size = {fine->local.x(), fine->local.y(), fov.z()};
							break;
						case SyncDirection::Front:
							patch.fine_origin = {0, 0, fine->local.z()};
							patch.fine_size = {fine->local.x(), fine->local.y(), fov.z()};
							break;
						default:  // unreachable: the halo faces carry the 6 axis normals
							patch.fine_origin = {0, 0, 0};
							patch.fine_size = {0, 0, 0};
							break;
					}
					// a face masked in fine_wall_masks receives NO
					// coarse-to-fine fill: the wall row and its streaming
					// buffer are BC-managed, and the first-fluid row's
					// state is authored by the fine kernel every substep
					// (a fill would clobber it with coarse-converted
					// data). Empty destination: the zero face-normal
					// extent makes the transfer launch's clip skip the
					// patch
					if (fine_wall_mask & (1 << fineWallFaceBit(f.face))) {
						patch.fine_origin = {0, 0, 0};
						patch.fine_size = {fine->local.x(), fine->local.y(), fine->local.z()};
						patch.fine_size[fineWallFaceBit(f.face) / 2] = 0;
					}
					patch.face = f.face;
					coupling.patches.push_back(patch);
					coupling.coarse_block_ids.push_back(coarse->id);
					coupling.fine_block_ids.push_back(fine->id);
				}
			}

			// interior patches (changes 2+3 of the AMR interface redesign,
			// docs/AMR-interface-proposed-diagram.md §3/§7 — unconditional
			// since the ring F2C path was removed, gate B ruling + D.1
			// hard-delete): the one-coarse-cell-deep SKIN of the fine
			// footprint AT DEPTH 1 (frozen GEO_NOTHING cells one coarse row
			// INSIDE the reactivated c=0 ring row of
			// docs/AMR-schonherr-ch7-target-contract.md sec. 2.1,
			// F2C-injected with fine-filtered DFs each cycle) as a DISJOINT
			// partition of 6 inset-face rectangles — the same disjoint
			// face-partition idiom as the halo ring above, inset one more
			// coarse cell INTO the footprint: the x-normal faces own the
			// full depth-1 tangent range [go+1, go+gs-1), the y-normal
			// faces the twice-inset interior x-range [go+2, go+gs-2), the
			// z-normal faces the twice-inset interior x/y ranges. The
			// reactivated c=0 surface shell in between is collision-active
			// (coarse-kernel-driven GEO_AMR_INTERFACE) and never
			// F2C-written; the c>=2 deep frozen core is never F2C-written
			// and never read either — e.g. a K=8 footprint emits its
			// (K-2)^3-(K-4)^3 = 6^3-4^3 = 152 skin cells in 6 rectangles
			// (K=32: 5,048).
			// Degenerate thin footprints clamp to EMPTY rectangles (skipped
			// by the clip below, never pushed): with gs.a < 5 the
			// twice-inset tangent ranges [go.a+2, go.a+gs.a-2) of the other
			// axes' faces are empty, and the max(..., go.a+2) clamp on the
			// max-side slab origin keeps a gs.a == 3 footprint from
			// emitting the same axis-plane twice (the min-side face then
			// owns the whole depth-1 slab) — no rectangle therefore carries
			// a negative extent and no coarse cell is written twice.
			const idx xi0 = go.x() + 1, xi1 = go.x() + gs.x() - 1;
			const idx yi0 = go.y() + 1, yi1 = go.y() + gs.y() - 1;
			const idx zi0 = go.z() + 1, zi1 = go.z() + gs.z() - 1;
			const idx xi2 = go.x() + 2, xi2e = go.x() + gs.x() - 2;
			const idx yi2 = go.y() + 2, yi2e = go.y() + gs.y() - 2;
			const idx xr0 = std::max(go.x() + gs.x() - 2, go.x() + 2);
			const idx yr0 = std::max(go.y() + gs.y() - 2, go.y() + 2);
			const idx zr0 = std::max(go.z() + gs.z() - 2, go.z() + 2);
			const struct SKIN
			{
				idx3d begin, end;
			} skins[6] = {
				{{xi0, yi0, zi0}, {go.x() + 2, yi1, zi1}},	  // x-min face (full depth-1 tangent)
				{{xr0, yi0, zi0}, {xi1, yi1, zi1}},			  // x-max face (full depth-1 tangent)
				{{xi2, yi0, zi0}, {xi2e, go.y() + 2, zi1}},	  // y-min face (twice-inset x)
				{{xi2, yr0, zi0}, {xi2e, yi1, zi1}},		  // y-max face (twice-inset x)
				{{xi2, yi2, zi0}, {xi2e, yi2e, go.z() + 2}},  // z-min face (twice-inset x/y)
				{{xi2, yi2, zr0}, {xi2e, yi2e, zi1}},		  // z-max face (twice-inset x/y)
			};
			for (const SKIN& skin : skins) {
				for (auto* coarse : this->nse.getBlocksAtLevel(coarse_level)) {
					// clip the skin rectangle to this coarse block's range
					// (global parent-level coordinates — same overlap test
					// as the ring faces and the full-footprint interior)
					const idx3d cbegin{
						std::max(skin.begin.x(), coarse->offset.x()),
						std::max(skin.begin.y(), coarse->offset.y()),
						std::max(skin.begin.z(), coarse->offset.z())
					};
					const idx3d cend{
						std::min(skin.end.x(), coarse->offset.x() + coarse->local.x()),
						std::min(skin.end.y(), coarse->offset.y() + coarse->local.y()),
						std::min(skin.end.z(), coarse->offset.z() + coarse->local.z())
					};
					if (cbegin.x() >= cend.x() || cbegin.y() >= cend.y() || cbegin.z() >= cend.z())
						continue;

					AMR_InterfacePatch<NSE> patch;
					// indexer-coordinates rectangles of the two blocks; the
					// fine rectangle covers 2 fine cells per coarse cell on
					// each axis (bookkeeping — the F2C launch derives the
					// fine window from the coarse coordinates and offsets)
					patch.coarse_origin = {cbegin.x() - coarse->offset.x(), cbegin.y() - coarse->offset.y(), cbegin.z() - coarse->offset.z()};
					patch.coarse_size = {cend.x() - cbegin.x(), cend.y() - cbegin.y(), cend.z() - cbegin.z()};
					patch.fine_origin = {2 * cbegin.x() - fine->offset.x(), 2 * cbegin.y() - fine->offset.y(), 2 * cbegin.z() - fine->offset.z()};
					patch.fine_size = {2 * patch.coarse_size.x(), 2 * patch.coarse_size.y(), 2 * patch.coarse_size.z()};
					patch.face = SyncDirection::None;
					coupling.interior_patches.push_back(patch);
					coupling.interior_coarse_block_ids.push_back(coarse->id);
					coupling.interior_fine_block_ids.push_back(fine->id);
				}
			}

			// R4 wall-pedestal prisms (the plan's sec. 5.3 refill mechanism,
			// nested wall-shared faces only -- the helper returns nothing on
			// level-1 blocks, so the single-fine-level regression contract
			// keeps its exact interior-patch census): the frozen rows at
			// face-normal depths {2,3} below the depth-1 skin become F2C
			// destinations, disjoint from the six skins above, so the
			// parent's own-8 upward fine-to-coarse window never reads a
			// never-written deep-core cell
			for (const auto& [prism_begin, prism_end] : wallPedestalPrismRects(*fine, fine_wall_mask)) {
				for (auto* coarse : this->nse.getBlocksAtLevel(coarse_level)) {
					// same clipping / bookkeeping idiom as the skins above
					const idx3d pbegin{
						std::max(prism_begin.x(), coarse->offset.x()),
						std::max(prism_begin.y(), coarse->offset.y()),
						std::max(prism_begin.z(), coarse->offset.z())
					};
					const idx3d pend{
						std::min(prism_end.x(), coarse->offset.x() + coarse->local.x()),
						std::min(prism_end.y(), coarse->offset.y() + coarse->local.y()),
						std::min(prism_end.z(), coarse->offset.z() + coarse->local.z())
					};
					if (pbegin.x() >= pend.x() || pbegin.y() >= pend.y() || pbegin.z() >= pend.z())
						continue;

					AMR_InterfacePatch<NSE> patch;
					patch.coarse_origin = {pbegin.x() - coarse->offset.x(), pbegin.y() - coarse->offset.y(), pbegin.z() - coarse->offset.z()};
					patch.coarse_size = {pend.x() - pbegin.x(), pend.y() - pbegin.y(), pend.z() - pbegin.z()};
					patch.fine_origin = {2 * pbegin.x() - fine->offset.x(), 2 * pbegin.y() - fine->offset.y(), 2 * pbegin.z() - fine->offset.z()};
					patch.fine_size = {2 * patch.coarse_size.x(), 2 * patch.coarse_size.y(), 2 * patch.coarse_size.z()};
					patch.face = SyncDirection::None;
					coupling.interior_patches.push_back(patch);
					coupling.interior_coarse_block_ids.push_back(coarse->id);
					coupling.interior_fine_block_ids.push_back(fine->id);
				}
			}
		}

		couplings.push_back(std::move(coupling));
	}
}

#ifndef NDEBUG
/**
 * \brief Registration map-pattern guard (Schönherr-ch7 conversion, plan
 * T6/Oracle F2) -- debug builds only.
 *
 * Static structural check that the coupling patch geometry built by
 * \ref buildCouplings agrees with the map tags: a one-cell registration
 * error is invisible to the kernel-window mocks but breaks every nominal
 * source window, so it is asserted here on the host before the first
 * SimUpdate. Two rails:
 * - (a) for every coarse-to-fine (halo patch) destination cell, fold the
 *   destination through the kernel's full window rule (the same fdiv2 /
 *   axis_window nominal solve + storage clamp, then the thesis Sec. 7.3
 *   wall shift of `cudaAMR_CoarseToFine`) and assert the final source
 *   tuple lies on the coupling band: every tuple cell must be
 *   `GEO_AMR_INTERFACE` (halo or reactivated shell) or a `GEO_NOTHING`
 *   cell inside the footprint (the F2C-refilled skin). At interior faces
 *   the fold is identity and the tuple is the ring band of the original
 *   form; at faces co-located with a physical boundary (wall refinement)
 *   the nominal tuple covers the boundary row and the wall shift steers
 *   it onto the {c = 0 ring, c = 1 skin} pair exactly as in the kernel;
 * - (b) every fine-to-coarse (interior patch) destination cell is
 *   `GEO_NOTHING` at EXACTLY footprint surface-depth 1 (c = 1 / c =
 *   gs-2), or, on a nested wall-shared face, one of the R4 wall-pedestal
 *   prism cells at face-normal depths {2,3} (the plan's sec. 5.3 refill
 *   mechanism; the depth set is {1} normally, {1,2,3} on the pedestal
 *   rows).
 *
 * Returns false on the first violation (the offending cell is logged);
 * the caller (SimInit) sets the terminate flag on false.
 */
template <typename NSE>
bool State_AMR<NSE>::checkCouplingMapPattern()
{
	// kernel fdiv2 (valid for negative fine-global coordinates, unlike the
	// truncating integer division of C++)
	const auto fdiv2 = [](idx v) -> idx
	{
		return v >= 0 ? v / 2 : -((-v + 1) / 2);
	};

	for (const InterLevelCoupling& coupling : couplings) {
		// (a) C2F halo patches: fold every destination through the kernel's
		// full window rule (nominal window + storage clamp + the thesis
		// Sec. 7.3 wall shift) and assert the final source tuple lies on the
		// coupling band: every cell must be GEO_AMR_INTERFACE (halo or
		// reactivated shell) or a GEO_NOTHING cell inside the footprint (the
		// F2C-refilled skin); physical-BC cells in the final tuple, dead
		// frozen cells outside the footprint, and plain GEO_FLUID tuple
		// cells are all registration errors
		for (std::size_t i = 0; i < coupling.patches.size(); i++) {
			const AMR_InterfacePatch<NSE>& patch = coupling.patches[i];
			BLOCK_NSE* fine = findBlockById(coupling.fine_level, coupling.fine_block_ids[i]);
			BLOCK_NSE* coarse = findBlockById(coupling.coarse_level, coupling.coarse_block_ids[i]);
			if (fine == nullptr || coarse == nullptr)
				return false;
			const idx3d& go = fine->global_offset;
			const idx3d gs{(fine->local.x() + 2) / 2, (fine->local.y() + 2) / 2, (fine->local.z() + 2) / 2};
			const idx csize[3] = {coarse->local.x(), coarse->local.y(), coarse->local.z()};
			const idx cov[3] = {coarse->df_overlap_X(), coarse->df_overlap_Y(), coarse->df_overlap_Z()};
			const idx coff[3] = {coarse->offset.x(), coarse->offset.y(), coarse->offset.z()};
			// the kernel's live-source predicate of the wall guard (a
			// physical-BC tag makes the cell a non-source). The window cells
			// w below are coarse-LOCAL (home = fdiv2(fg) - coff), but the
			// host map is GLOBAL-indexed with storage bounds
			// [offset - ov, offset + local + ov) -- the read must carry the
			// offset back (an identity when the coarse parent is a level-0
			// block, mandatory for level >= 1 parents of nested couplings)
			const auto is_source = [&coarse, &coff](idx cx, idx cy, idx cz) -> bool
			{
				const auto mapgi = coarse->hmap(cx + coff[0], cy + coff[1], cz + coff[2]);
				return NSE::BC::isFluid(mapgi) || mapgi == NSE::BC::GEO_AMR_INTERFACE || mapgi == NSE::BC::GEO_NOTHING;
			};
			// final-tuple band membership: the ring (either halo or
			// reactivated shell), the F2C-refilled skin inside this patch's
			// footprint, OR a plain-fluid parent row -- the kernel's own
			// source predicate accepts plain fluid, which is exactly what a
			// nested parent's ghost-attached rows hold (e.g. the parent's own
			// coarse-to-fine band rows below/above a footprint-shared face;
			// the launch clip drops the shared faces' own destinations but
			// the full-tangent destinations of the other faces still map
			// there). On level-0 parents every final tuple is a ring/skin
			// cell, so the fluid lane never fires on level-1 configurations.
			// Physical-BC tags (non-sources per the kernel predicate) and
			// footprint-external frozen rows remain registration errors.
			const auto on_band = [&coarse, &go, &gs, &coff](idx cx, idx cy, idx cz) -> bool
			{
				const idx gx = cx + coff[0];
				const idx gy = cy + coff[1];
				const idx gz = cz + coff[2];
				const auto mapgi = coarse->hmap(gx, gy, gz);
				if (mapgi == NSE::BC::GEO_AMR_INTERFACE || NSE::BC::isFluid(mapgi))
					return true;
				if (mapgi != NSE::BC::GEO_NOTHING)
					return false;
				return gx >= go.x() && gx < go.x() + gs.x() && gy >= go.y() && gy < go.y() + gs.y() && gz >= go.z() && gz < go.z() + gs.z();
			};
			// the kernel wall shift, host replica (one cell away from the
			// tainted end, or mirrored home at the storage edge)
			const auto shift_pair = [](idx* nodes, bool taint_lo, bool taint_hi, idx lo, idx hi, idx home) -> void
			{
				if (nodes[0] == nodes[1] || taint_lo == taint_hi)
					return;
				const idx start = taint_hi ? nodes[0] - 1 : nodes[0] + 1;
				if (start < lo || start + 1 > hi) {
					nodes[0] = home;
					nodes[1] = home;
					return;
				}
				nodes[0] = start;
				nodes[1] = start + 1;
			};
			for (idx x = patch.fine_origin.x(); x < patch.fine_origin.x() + patch.fine_size.x(); x++)
				for (idx y = patch.fine_origin.y(); y < patch.fine_origin.y() + patch.fine_size.y(); y++)
					for (idx z = patch.fine_origin.z(); z < patch.fine_origin.z() + patch.fine_size.z(); z++) {
						const idx fg[3] = {fine->offset.x() + x, fine->offset.y() + y, fine->offset.z() + z};
						// the kernel's axis_window nominal solve + storage clamp
						idx w[3][2], ho[3];
						for (int a = 0; a < 3; a++) {
							const idx home = fdiv2(fg[a]) - coff[a];
							const idx p = fg[a] & 1;
							ho[a] = home;
							const int extent = static_cast<int>(csize[a] + 2 * cov[a]);
							const int n = 2 < extent ? 2 : extent;
							const idx lo = -cov[a];
							const idx hi = csize[a] - 1 + cov[a] - (n - 1);
							idx start = home - 1 + p;
							start = start < lo ? lo : (start > hi ? hi : start);
							w[a][0] = start;
							w[a][1] = start + (n - 1);
						}
						// taint scan of the nominal tuple + the wall shift per axis
						bool inv[2][2][2];
						bool tainted = false;
						for (int ibz = 0; ibz < 2; ibz++)
							for (int iby = 0; iby < 2; iby++)
								for (int ibx = 0; ibx < 2; ibx++) {
									inv[ibx][iby][ibz] = ! is_source(w[0][ibx], w[1][iby], w[2][ibz]);
									tainted = tainted || inv[ibx][iby][ibz];
								}
						if (tainted) {
							shift_pair(
								w[0],
								inv[0][0][0] || inv[0][0][1] || inv[0][1][0] || inv[0][1][1],
								inv[1][0][0] || inv[1][0][1] || inv[1][1][0] || inv[1][1][1],
								-cov[0],
								csize[0] - 1 + cov[0],
								ho[0]
							);
							shift_pair(
								w[1],
								inv[0][0][0] || inv[0][0][1] || inv[1][0][0] || inv[1][0][1],
								inv[0][1][0] || inv[0][1][1] || inv[1][1][0] || inv[1][1][1],
								-cov[1],
								csize[1] - 1 + cov[1],
								ho[1]
							);
							shift_pair(
								w[2],
								inv[0][0][0] || inv[0][1][0] || inv[1][0][0] || inv[1][1][0],
								inv[0][0][1] || inv[0][1][1] || inv[1][0][1] || inv[1][1][1],
								-cov[2],
								csize[2] - 1 + cov[2],
								ho[2]
							);
							// the kernel's residual collapse (a physical BC
							// thicker than one cell, or a mid-window straddle)
							bool residual = false;
							for (int ibz = 0; ibz < 2 && ! residual; ibz++)
								for (int iby = 0; iby < 2 && ! residual; iby++)
									for (int ibx = 0; ibx < 2 && ! residual; ibx++)
										residual = ! is_source(w[0][ibx], w[1][iby], w[2][ibz]);
							if (residual)
								for (int a = 0; a < 3; a++) {
									w[a][0] = ho[a];
									w[a][1] = ho[a];
								}
						}
						// final-tuple band membership over all 8 source cells
						for (int ibz = 0; ibz < 2; ibz++)
							for (int iby = 0; iby < 2; iby++)
								for (int ibx = 0; ibx < 2; ibx++)
									if (! on_band(w[0][ibx], w[1][iby], w[2][ibz])) {
										const idx cell[3] = {w[0][ibx] + coff[0], w[1][iby] + coff[1], w[2][ibz] + coff[2]};
										spdlog::error(
											"checkCouplingMapPattern: C2F destination ({},{},{}) of face {} has a source cell off the "
											"coupling band (coarse cell ({},{},{}): tag {}, footprint [{},{},{}] + [{},{},{}])",
											x,
											y,
											z,
											static_cast<int>(patch.face),
											cell[0],
											cell[1],
											cell[2],
											static_cast<int>(coarse->hmap(cell[0], cell[1], cell[2])),
											go.x(),
											go.y(),
											go.z(),
											gs.x(),
											gs.y(),
											gs.z()
										);
										return false;
									}
					}
		}

		// (b) F2C interior patches: every destination cell must be frozen
		// GEO_NOTHING at exactly footprint surface-depth 1 (c=1 / c=gs-2),
		// except the R4 wall-pedestal prism cells of a nested wall-shared face,
		// which sit at face-normal depths {2,3} (plan sec. 5.3: the face-
		// specific depth set is {1} normally, {1,2,3} on the pedestal rows;
		// every destination stays frozen)
		for (std::size_t i = 0; i < coupling.interior_patches.size(); i++) {
			const AMR_InterfacePatch<NSE>& patch = coupling.interior_patches[i];
			BLOCK_NSE* fine = findBlockById(coupling.fine_level, coupling.interior_fine_block_ids[i]);
			BLOCK_NSE* coarse = findBlockById(coupling.coarse_level, coupling.interior_coarse_block_ids[i]);
			if (fine == nullptr || coarse == nullptr)
				return false;
			const idx3d& go = fine->global_offset;
			const idx3d gs{(fine->local.x() + 2) / 2, (fine->local.y() + 2) / 2, (fine->local.z() + 2) / 2};
			// the prism set this coupling was built with (empty on level-1
			// blocks and unmasked faces): pedestal destinations may sit deeper
			// than 1 exactly inside these rectangles
			const auto prisms = wallPedestalPrismRects(*fine, fineWallMask(*fine));
			for (idx x = patch.coarse_origin.x(); x < patch.coarse_origin.x() + patch.coarse_size.x(); x++)
				for (idx y = patch.coarse_origin.y(); y < patch.coarse_origin.y() + patch.coarse_size.y(); y++)
					for (idx z = patch.coarse_origin.z(); z < patch.coarse_origin.z() + patch.coarse_size.z(); z++) {
						const idx gx = coarse->offset.x() + x;
						const idx gy = coarse->offset.y() + y;
						const idx gz = coarse->offset.z() + z;
						const idx depth = std::min(
							std::min(std::min(gx - go.x(), go.x() + gs.x() - 1 - gx), std::min(gy - go.y(), go.y() + gs.y() - 1 - gy)),
							std::min(gz - go.z(), go.z() + gs.z() - 1 - gz)
						);
						bool depth_ok = depth == 1;
						if (! depth_ok)
							for (const auto& [pbegin, pend] : prisms)
								if (gx >= pbegin.x() && gx < pend.x() && gy >= pbegin.y() && gy < pend.y() && gz >= pbegin.z() && gz < pend.z()) {
									depth_ok = true;
									break;
								}
						if (coarse->hmap(gx, gy, gz) != NSE::BC::GEO_NOTHING || ! depth_ok) {
							spdlog::error(
								"checkCouplingMapPattern: F2C destination ({},{},{}) of an interior patch is not a depth-1 frozen cell "
								"or an R4 wall-pedestal cell (tag {}, depth {})",
								gx,
								gy,
								gz,
								static_cast<int>(coarse->hmap(gx, gy, gz)),
								depth
							);
							return false;
						}
					}
		}
	}
	return true;
}
#endif

template <typename NSE>
bool State_AMR<NSE>::isShadowedBySameLevelBlock(idx x, idx y, idx z, int fine_level, const BLOCK_NSE* owner)
{
	for (auto* other : this->nse.getBlocksAtLevel(fine_level)) {
		if (other == owner)
			continue;
		// footprint of the other fine block in parent-level global coordinates
		// ((local + 2) / 2 -- the re-anchored interior is inset one fine cell
		// per face, see buildCouplings)
		const idx3d& oo = other->global_offset;
		const idx3d os{(other->local.x() + 2) / 2, (other->local.y() + 2) / 2, (other->local.z() + 2) / 2};
		if (x >= oo.x() && x < oo.x() + os.x() && y >= oo.y() && y < oo.y() + os.y() && z >= oo.z() && z < oo.z() + os.z())
			return true;
	}
	return false;
}

template <typename NSE>
typename State_AMR<NSE>::BLOCK_NSE* State_AMR<NSE>::findBlockById(int level, int id)
{
	for (auto& block : this->nse.blocks)
		if (block.level == level && block.id == id)
			return &block;
	return nullptr;
}

/**
 * \brief Coarse-to-fine ghost-layer fill for one level (step 5 of the
 * Schönherr simulated-band cycle - and the identical initial fill of
 * \ref SimInit).
 *
 * Iterates the `AMR_InterfacePatch` descriptors of \ref couplings matching
 * `fine_level` and launches one `cudaAMR_CoarseToFine` per patch with
 * `begin = fine_origin, end = fine_origin + fine_size` (fine indexer
 * coordinates).
 *
 * Storability guard: the patch's fine rectangle is clipped FACE-AWARE to
 * the fine block's ALLOCATED ghost STORAGE (the overlap depth of the
 * block's indexer, 2 cells deep on refinement-level blocks -- see
 * `LBM_BLOCK::storage_overlap`): tangential axes admit the full stored
 * window `[-ov, local+ov)`, the face-normal axis clips to the face's own
 * destination band (`[-ov, 0)` on the min face / `[local, local+ov)` on
 * the max face). The filled rows feed the fine-level streaming directly,
 * and the inner ghost layer is also the layer the max-side skin
 * fine-to-coarse windows read. Axes where the block spans
 * its global extent have no overlap allocated there and get no fill
 * (streaming then consumes exterior-boundary data instead). When
 * \ref couplings is empty (no marked interface cells), this is a silent
 * no-op (SimInit logged a warning).
 */
template <typename NSE>
void State_AMR<NSE>::launchCoarseToFineTransfers(int fine_level)
{
	using SyncDirection = TNL::Containers::SyncDirection;

	ensureTransferStreams();

	for (InterLevelCoupling& coupling : couplings) {
		if (coupling.fine_level != fine_level)
			continue;

		for (std::size_t i = 0; i < coupling.patches.size(); i++) {
			AMR_InterfacePatch<NSE>& patch = coupling.patches[i];
			BLOCK_NSE* fine = findBlockById(coupling.fine_level, coupling.fine_block_ids[i]);
			BLOCK_NSE* coarse = findBlockById(coupling.coarse_level, coupling.coarse_block_ids[i]);
			if (fine == nullptr || coarse == nullptr)
				continue;

			// tau = 3*nu_lb + 0.5 per level (see the file docstring)
			const dreal tau_fine = static_cast<dreal>(3 * blockLbmViscosity(*fine) + 0.5);
			const dreal tau_coarse = static_cast<dreal>(3 * blockLbmViscosity(*coarse) + 0.5);
			// parity of the kernel launch that produced the current coarse
			// data (AA-pattern state; ignored by the kernel for AB)
			const bool coarse_even_iter = coarse->data.even_iter;

			// launch extent in the fine block's indexer coordinates, clipped
			// FACE-AWARE to the fine block's overlap storage (the band
			// registration of the contract): tangential axes admit the full
			// stored overlap window [-ov, local+ov); the face-normal axis is
			// clipped to the face's OWN destination band -- [-ov, 0) on the
			// min face / [local, local+ov) on the max face (the patch's fine
			// rectangle already lies inside these windows on valid
			// geometries; the clip is the storability guard for axes where
			// the block spans its global extent and carries no overlap)
			const idx3d ov{fine->df_overlap_X(), fine->df_overlap_Y(), fine->df_overlap_Z()};
			idx cx_lo = -ov.x(), cx_hi = fine->local.x() + ov.x();
			idx cy_lo = -ov.y(), cy_hi = fine->local.y() + ov.y();
			idx cz_lo = -ov.z(), cz_hi = fine->local.z() + ov.z();
			if (patch.face == SyncDirection::Left)
				cx_hi = 0;
			else if (patch.face == SyncDirection::Right)
				cx_lo = fine->local.x();
			if (patch.face == SyncDirection::Bottom)
				cy_hi = 0;
			else if (patch.face == SyncDirection::Top)
				cy_lo = fine->local.y();
			if (patch.face == SyncDirection::Back)
				cz_hi = 0;
			else if (patch.face == SyncDirection::Front)
				cz_lo = fine->local.z();
			const idx3d begin{std::max(patch.fine_origin.x(), cx_lo), std::max(patch.fine_origin.y(), cy_lo), std::max(patch.fine_origin.z(), cz_lo)};
			const idx3d end{
				std::min(patch.fine_origin.x() + patch.fine_size.x(), cx_hi),
				std::min(patch.fine_origin.y() + patch.fine_size.y(), cy_hi),
				std::min(patch.fine_origin.z() + patch.fine_size.z(), cz_hi)
			};
			if (begin.x() >= end.x() || begin.y() >= end.y() || begin.z() >= end.z())
				continue;

			const idx3d size{end.x() - begin.x(), end.y() - begin.y(), end.z() - begin.z()};

			// call-site launch-config cache (2026-08-18): the patch extent is
			// immutable after SimInit, so the logging optimizer runs once
			// per patch instead of once per fill launch (~12 of 18
			// per-iteration optimizer calls in the AMR schedule)
			if (patch.cached_block_size.x == 0)
				patch.cached_block_size = fine->getCudaBlockSize(size);
			if (patch.cached_grid_size.x == 0)
				patch.cached_grid_size = fine->getCudaGridSize(size, patch.cached_block_size);

			TNL::Backend::LaunchConfiguration launch_config;
			launch_config.blockSize = patch.cached_block_size;
			launch_config.gridSize = patch.cached_grid_size;
			launch_config.stream = stream_c2f;
			TNL::Backend::launchKernelAsync(
				cudaAMR_CoarseToFine<NSE>,
				launch_config,
				fine->data,
				coarse->data,
				begin,
				end,
				tau_fine,
				tau_coarse,
				coarse_even_iter,
				fine->offset,
				coarse->offset
			);
		}
	}
	// NO trailing sync: the C2F kernels run concurrently with the F2C
	// direction on their own streams (disjoint cell sets); the phase's
	// single barrier is the caller's synchronizeTransfers()
}

// D.1 hard-delete (gate-B ruling, 2026-08-16): the ring fine-to-coarse
// launch (`launchFineToCoarseTransfers` over the halo `patches`) was
// removed -- the skin F2C of \ref launchFineToCoarseTransfersInterior is
// now the ONLY fine-to-coarse channel. The halo `patches` of \ref couplings
// remain: they are the coarse-to-fine ghost-fill geometry of
// \ref launchCoarseToFineTransfers.
/**
 * \brief Skin (under-footprint) fine-to-coarse transfer -- the ONLY
 * fine-to-coarse channel (gate B ruling, ring path removed in D.1).
 *
 * Iterates the interior_patches of \ref couplings (the 6 disjoint inset-face
 * SKIN rectangles built in `buildCouplings`, one coarse cell deep inside
 * the footprint) and launches `cudaAMR_FineToCoarse` over each rectangle.
 * The under-footprint coarse cells are frozen as GEO_NOTHING (no
 * stream/collide); the written cells' DFs are set exclusively by this
 * transfer — the strategy-selected F2C state (F2C_SCHONHERR default;
 * Lagrava-filtered fine-averaged DFs under the F2C_LAGRAVA opt-out),
 * full overwrite. The deep frozen
 * core is never written (and never read: the coarse C2F stencil
 * reaches at most 1 cell into the footprint). This is the two-way feedback
 * channel: fine-interior information reaches the coarse lattice through
 * the written skin cells, which the collision-active ring cells stream
 * from at the next coarse step. Every written skin cell reads fine state
 * of the fine interior only (its own 8 subcells under the F2C_SCHONHERR
 * default; the window with the kernel's lo = 0 interior clamp under the
 * Lagrava opt-out). When \ref couplings is empty (no marked interface
 * cells), this is a silent no-op (SimInit logged a warning).
 */
template <typename NSE>
void State_AMR<NSE>::launchFineToCoarseTransfersInterior(int fine_level)
{
	const int coarse_level = fine_level - 1;

	ensureTransferStreams();

	for (InterLevelCoupling& coupling : couplings) {
		if (coupling.fine_level != fine_level)
			continue;

		for (std::size_t i = 0; i < coupling.interior_patches.size(); i++) {
			AMR_InterfacePatch<NSE>& patch = coupling.interior_patches[i];
			BLOCK_NSE* fine = findBlockById(coupling.fine_level, coupling.interior_fine_block_ids[i]);
			BLOCK_NSE* coarse = findBlockById(coupling.coarse_level, coupling.interior_coarse_block_ids[i]);
			if (fine == nullptr || coarse == nullptr)
				continue;

			const dreal tau_fine = static_cast<dreal>(3 * blockLbmViscosity(*fine) + 0.5);
			const dreal tau_coarse = static_cast<dreal>(3 * blockLbmViscosity(*coarse) + 0.5);
			const bool fine_even_iter = fine->data.even_iter;
			// parity of the NEXT consuming substep on the coarse side (the
			// write side of cudaAMR_FineToCoarse): level 0 is driven by the
			// global `iterations` clock -- SimUpdate post-incremented it for
			// this cycle before any launch, so the level-0 arm reproduces
			// the historical `(iterations % 2) == 1` verbatim; finer parents
			// (nested couplings) are driven by their cumulative substep
			// counter, which holds the completed-substep count = the next
			// substep's index with the same post-increment semantics. The
			// historical `: false` arm for coarse_level >= 1 was the frozen
			// parity bug this fixes
			const int next_coarse_substep = (coarse_level == 0) ? this->nse.iterations : this->nse.totalSubstepCount[coarse_level];
			const bool next_coarse_even_iter = (next_coarse_substep % 2) == 1;

			const idx3d ov{fine->df_overlap_X(), fine->df_overlap_Y(), fine->df_overlap_Z()};
			const idx3d begin = patch.coarse_origin;
			const idx3d end{
				patch.coarse_origin.x() + patch.coarse_size.x(),
				patch.coarse_origin.y() + patch.coarse_size.y(),
				patch.coarse_origin.z() + patch.coarse_size.z()
			};
			if (begin.x() >= end.x() || begin.y() >= end.y() || begin.z() >= end.z())
				continue;

			const idx3d size{end.x() - begin.x(), end.y() - begin.y(), end.z() - begin.z()};
			// call-site launch-config cache (2026-08-18), same rationale as
			// the C2F launcher (6 of 18 per-iteration optimizer calls)
			if (patch.cached_block_size.x == 0)
				patch.cached_block_size = coarse->getCudaBlockSize(size);
			if (patch.cached_grid_size.x == 0)
				patch.cached_grid_size = coarse->getCudaGridSize(size, patch.cached_block_size);

			TNL::Backend::LaunchConfiguration launch_config;
			launch_config.blockSize = patch.cached_block_size;
			launch_config.gridSize = patch.cached_grid_size;
			launch_config.stream = stream_f2c;
			TNL::Backend::launchKernelAsync(
				cudaAMR_FineToCoarse<NSE>,
				launch_config,
				coarse->data,
				fine->data,
				begin,
				end,
				tau_coarse,
				tau_fine,
				fine_even_iter,
				next_coarse_even_iter,
				fine->offset,
				coarse->offset,
				fine->local,
				ov
			);
		}
	}
	// NO trailing sync: the F2C kernels run concurrently with the C2F
	// direction on their own streams (disjoint cell sets); the phase's
	// single barrier is the caller's synchronizeTransfers()
}

template <typename NSE>
void State_AMR<NSE>::synchronizeTransfers()
{
	// the single sync point of a transfer phase: draining both directions
	// ends the F2C/C2F overlap window exactly here (uncreated spy-fixture
	// streams are the null stream, so this degrades to a null-stream sync)
	TNL::Backend::streamSynchronize(stream_f2c);
	TNL::Backend::streamSynchronize(stream_c2f);
}

/**
 * \brief Advance one refinement level by one pair of substeps (the
 * Berger-Colella pair recursion).
 *
 * One pair at level L covers 2 dt_L = dt_{L-1}: substep A on the widened
 * simulated-band extent (the inner ghost rows are INTEGRATED, collide +
 * stream like interior fluid, sourcing the outer ghost row filled by the
 * previous coarse-to-fine fill; on a face masked in fine_wall_masks the
 * window is deepened to the fine wall's GEO_WALL row), then substep B on
 * the interior-only extent (its boundary data is substep A's updated inner
 * ghost rows in the other AB frame, so it needs no fill). With
 * L < max_level the pair nests the finer level's own pairs per the plan's
 * schedule tables (.omo/plans/amr-nlevel-nesting.md sec. 3):
 *
 * - after substep A and before substep B, the finer pair covering
 *   [t, t+dt_L) runs, followed by the mid-sync F2C(L+1 -> L) and the
 *   mid-cycle C2F fill of the finer band (its source is L's live
 *   post-substep-A state; the fill covers pair #2 of the finer level),
 * - after substep B, the finer pair covering [t+dt_L, t+2dt_L) runs,
 *   followed by the end-sync F2C(L+1 -> L).
 *
 * Every fine-to-coarse launch therefore fires once per level-L substep,
 * immediately when both levels are time-aligned (frame-forced by the
 * AB-ring streaming discipline: each level's ring cells stream from the
 * skin in their own current frame). At L == max_level the inner blocks are
 * unreachable and the emission is the historical two-substep sequence.
 *
 * The parity/rotation preparation of each launch reads the level's
 * cumulative substep counter (LBM::totalSubstepCount): the counter holds
 * the completed-substep count, so its value before the increments in this
 * function is the substep's index; updateKernelDataForLevel is an absolute
 * setter on substep mod 2 / mod DFMAX, so the parity stays cycle-invariant
 * (every cycle advances the counter by an even 2^L).
 */
template <typename NSE>
void State_AMR<NSE>::advancePair(int level, bool compute_macro, bool sync_macro)
{
#ifdef HAVE_MPI
	#ifdef AA_PATTERN
	uint8_t output_df = df_cur;
	#endif
	#ifdef AB_PATTERN
	uint8_t output_df = df_out;
	#endif
#else
	static_cast<void>(sync_macro);	// consumed only by the MPI overlap exchange
#endif

	// substep A: toggle the level's even_iter parity / DF rotation to the
	// substep's index BEFORE the kernel (CRITICAL: for the A-B pattern the
	// rotation selects the physical array df_cur refers to; the global
	// updateKernelData() is driven by the coarse clock and must not drive
	// the fine substeps), then the widened simulated-band launch
	this->nse.updateKernelDataForLevel(level, this->nse.totalSubstepCount[level]);
	launchLBMKernelForLevel(level, compute_macro, /*ghost_layers=*/1);
	this->nse.totalSubstepCount[level]++;

#ifdef HAVE_MPI
	// exchange the latest DFs and dmacro on overlaps between the
	// level-L blocks (no-op for fine blocks in the v1 single-rank setup)
	if (this->nse.nproc > 1) {
		this->timer_wait_communication.start();
		this->nse.synchronizeDFsAndMacroDeviceForLevel(level, output_df, sync_macro);
		this->timer_wait_communication.stop();
	}
#endif

	if (level < this->nse.max_level) {
		// finer pair #1 covers [t, t+dt_L); the pair's inner structure
		// (its own even-finer pairs and syncs) is consumed by the recursion
		advancePair(level + 1, compute_macro, sync_macro);
		// mid-sync at t+dt_L, both levels time-aligned: fill the skin from
		// the finer pair, then refill the finer band for its pair #2 from
		// this level's live post-substep-A state (single-frame fill of the
		// finer level's substep-0 rotation). The two directions run
		// CONCURRENTLY on their per-direction streams (disjoint cell sets)
		// and are drained together below -- the phase's single sync point
		launchFineToCoarseTransfersInterior(level + 1);
		this->nse.updateKernelDataForLevel(level + 1, this->nse.totalSubstepCount[level + 1]);
		launchCoarseToFineTransfers(level + 1);
		// substep B reads the skin the F2C just wrote (its ring streams
		// from it) and the finer pair #2 reads the ghost rows the C2F
		// filled -- both must be complete
		synchronizeTransfers();
	}

	// substep B: interior-only launch (on a face masked in fine_wall_masks
	// the window still covers the GEO_WALL row: the bounce-back refreshes
	// the wall's slots in every substep's frame)
	this->nse.updateKernelDataForLevel(level, this->nse.totalSubstepCount[level]);
	launchLBMKernelForLevel(level, compute_macro, /*ghost_layers=*/0);
	this->nse.totalSubstepCount[level]++;

#ifdef HAVE_MPI
	// exchange the latest DFs and dmacro on overlaps between the
	// level-L blocks (no-op for fine blocks in the v1 single-rank setup)
	if (this->nse.nproc > 1) {
		this->timer_wait_communication.start();
		this->nse.synchronizeDFsAndMacroDeviceForLevel(level, output_df, sync_macro);
		this->timer_wait_communication.stop();
	}
#endif

	if (level < this->nse.max_level) {
		// finer pair #2 covers [t+dt_L, t+2dt_L), then the end-sync at
		// t+2dt_L (the band for the finer level's next pair #1 arrives with
		// the cycle-end cascade at the coarse clock, or with a later
		// mid-sync when this pair itself runs nested inside a coarser one)
		advancePair(level + 1, compute_macro, sync_macro);
		launchFineToCoarseTransfersInterior(level + 1);
	}
}

template <typename NSE>
void State_AMR<NSE>::SimUpdate()
{
	// no refinement levels: the driver is identical to the base one
	// (acceptance criterion of todo 7: sim_1 runs unchanged)
	if (this->nse.max_level == 0) {
		Base::SimUpdate();
		return;
	}

#ifndef USE_CUDA
	// the AMR coupling kernels are CUDA-only in v1 (same scope as
	// createAMRBlocks/markAMRInterface)
	spdlog::error("State_AMR: AMR subcycling (max_level = {}) requires a CUDA build (USE_CUDA)", this->nse.max_level);
	this->nse.terminate = true;
#else
	this->timer_SimUpdate.start();

	// debug (same zero-viscosity guard as the base driver)
	for (auto& block : this->nse.blocks)
		if (block.data.lbmViscosity == 0) {
			spdlog::error("error: LBM viscosity is 0");
			this->nse.terminate = true;
			return;
		}

	// NOTE: all Lagrangian points are assumed to be on the first GPU
	// (inert for v1 AMR sims without Lagrangian points, same as the base driver)
	if (this->ibm.LL.size() > 0) {
		for (auto& block : this->nse.blocks) {
			const auto direction = TNL::Containers::SyncDirection::None;
			TNL::Backend::LaunchConfiguration launch_config;
			launch_config.blockSize = block.computeData.at(direction).blockSize;
			launch_config.gridSize = block.computeData.at(direction).gridSize;
			TNL::Backend::launchKernelAsync(cudaLBMComputeVelocitiesStarAndZeroForce<NSE>, launch_config, block.data, block.is_distributed());
		}
		// synchronize the null-stream after all grids
		TNL::Backend::streamSynchronize(0);

		this->ibm.computeForces(this->nse.physTime());
	}

	// hook, same position as in the base driver (once per coarse step,
	// before the level-0 kernel); computeAfterLBMKernel() stays in the
	// inherited AfterSimUpdate(), exactly as in the base flow
	this->computeBeforeLBMKernel();

	// technically this should happen after the LBM kernel, but we need to
	// check actions beforehand (same as the base driver); the counter
	// increments ONCE per coarse step - fine substeps advance the level
	// clocks, not the global counter (physDt_fine = physDt_coarse / 2^L)
	this->nse.iterations++;

	// macroscopic output/synchronization flags, computed once per coarse
	// step from the level-0 decision of the base driver and reused at all
	// levels (v1 simplification: macro output cadence follows the coarse clock)
	bool sync_macro = NSE::MACRO::use_syncMacro;
	for (int c = 0; c < MAX_COUNTER; c++)
		if (c != PRINT && c != SAVESTATE)
			if (this->cnt[c].action(this->nse.physTime()))
				sync_macro = true;
	bool compute_macro = NSE::MACRO::compute_in_each_iteration || sync_macro;

	#ifdef HAVE_MPI
		#ifdef AA_PATTERN
	uint8_t output_df = df_cur;
		#endif
		#ifdef AB_PATTERN
	uint8_t output_df = df_out;
		#endif
	#endif

	this->timer_compute.start();

	// Schönherr cycle: fine levels advance first.
	// the Berger-Colella pair recursion (the file docstring): level 1's pair
	// consumes 2 dt_1 = dt_0 per coarse step, with the finer levels' pairs,
	// the mid-sync F2C + fill, and the end-sync F2C nested inside per the
	// plan's schedule tables (.omo/plans/amr-nlevel-nesting.md sec. 3)
	advancePair(1, compute_macro, sync_macro);

	// ---------- Schönherr cycle step 3: one coarse (level 0) LBM step ----------
	// execute() (core.h) already called updateKernelData(), which set the
	// level-0 even_iter parity / DF rotation from the global `iterations`.
	// GEO_AMR_INTERFACE cells are collision-active inside the kernel
	// (BC::doCollision == true); the coarse kernel is their only writer
	// since the ring fine-to-coarse launch was removed (D.1) - fine
	// feedback reaches them through streaming from the skin cells the
	// interior F2C wrote at the end of the previous cycle
	launchLBMKernelForLevel(0, compute_macro, /*ghost_layers=*/0);

	#ifdef HAVE_MPI
	// exchange the latest DFs and dmacro on overlaps between the LEVEL-0
	// blocks (the per-level variant skips fine blocks, which have no MPI
	// neighbors in the v1 single-rank setup; skipped entirely for
	// nproc == 1 like the base driver)
	if (this->nse.nproc > 1) {
		this->timer_wait_communication.start();
		this->nse.synchronizeDFsAndMacroDeviceForLevel(0, output_df, sync_macro);
		this->timer_wait_communication.stop();
	}
	#endif

	// ---------- Schönherr cycle steps 4-5: transfers at the cycle end ----------
	// end-sync fine-to-coarse, reading the level-1 rotation-1 frame (the
	// post-substep-2 array): inject the strategy-selected fine state
	// (F2C_SCHONHERR default; Lagrava-filtered under the F2C_LAGRAVA
	// opt-out) into the frozen GEO_NOTHING skin cells of each level-1
	// footprint (two-way feedback; the nested couplings' end-syncs already
	// ran inside the recursion above). Ring cells stream+collide only --
	// the ring F2C launch was removed (gate B ruling, D.1 hard-delete) and
	// the fine feedback reaches them through streaming from the freshly
	// F2C-written skin on the next coarse step (df_out -> next df_cur
	// convention, no kernel change). The relative order of the F2C against
	// the cascade below is IRRELEVANT: the touched sets are disjoint (F2C
	// writes skin cells in the parent's post-step array, C2F writes fine
	// ghost rows)
	launchFineToCoarseTransfersInterior(1);

	// the cycle-end coarse-to-fine cascade, level-ascending: fill the ghost
	// rows of each level's substep-0 rotation frame -- the band for that
	// level's FIRST pair of the next cycle (Schönherr simulated band; the
	// other frame needs no fill: substep B reads substep A's kernel-updated
	// inner ghost rows from it and its outer row is unreachable). The
	// counter argument is the level's cumulative substep count, always even
	// at a cycle boundary, so the parity/rotation matches the historical
	// positional-0 preparation bit-for-bit (updateKernelDataForLevel is an
	// absolute setter on substep mod 2 / mod DFMAX). The cascade overlaps
	// the F2C above on its own stream (disjoint cell sets)
	for (int M = 1; M <= this->nse.max_level; M++) {
		this->nse.updateKernelDataForLevel(M, this->nse.totalSubstepCount[M]);
		launchCoarseToFineTransfers(M);
	}

	// the cycle-end transfer phase's single sync point: the end-sync F2C(1)
	// and the whole C2F cascade ran concurrently on their per-direction
	// streams -- drain both before the timers stop and AfterSimUpdate's
	// host-side reads (conservation stats, output frames, checkpoints)
	synchronizeTransfers();

	this->timer_compute.stop();
	this->timer_SimUpdate.stop();
#endif
}
