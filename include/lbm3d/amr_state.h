#pragma once

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

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
 * (1989) subcycling schedule. One global iteration performs
 *
 * 1. ONE coarse (level 0) LBM step on all level-0 blocks
 *    (`updateKernelData()` was already called by `execute()` in `core.h` and
 *    set the level-0 even_iter parity / DF rotation from the global
 *    `iterations` counter). Coarse cells tagged `GEO_AMR_INTERFACE` (the
 *    interface ring around each fine footprint) are collision-active inside
 *    the kernel (they stream and collide like fluid); since the ring
 *    fine-to-coarse launch was removed (gate B ruling, 2026-08-16) the coarse
 *    kernel is their only writer -- fine feedback reaches them through
 *    streaming from the skin cells the interior F2C writes (step 8 of the
 *    recursion below).
 * 2. For each finer level L = 1..max_level:
 *    a. `updateKernelDataForLevel(L, 0)` toggles the fine level's even_iter
 *       parity / DF rotation to substep 0 (MANDATORY before the ghost fill
 *       - for the A-B pattern the rotation selects the physical array
 *       `df_cur` refers to, so the fill must land in the array the
 *       upcoming substep reads; the global `updateKernelData()` is driven
 *       by the coarse clock and must not drive the fine substeps),
 *    b. coarse-to-fine transfer: `cudaAMR_CoarseToFine` fills the fine
 *       ghost layer from level L-1 (see `d3q27/amr_coupling.h`),
 *    c. fine substep 1 of 2: `cudaLBMKernel` on all blocks at level L,
 *    d. `updateKernelDataForLevel(L, 1)` toggles the parity again (BEFORE
 *       the BVP fill, same reason),
 *    e. BVP: the coarse-to-fine transfer re-fills the fine ghost layer
 *       (substep 1 consumed the ghost DFs and streamed outward into them),
 *    f. fine substep 2 of 2,
 *    g. fine-to-coarse transfer: `cudaAMR_FineToCoarse` projects the
 *       Lagrava-filtered fine state back onto level L-1.
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
 *   storability guard derived from `LBM_BLOCK::overlap_width` (see the
 *   launch helpers): coarse-to-fine fills at most the 1-cell-deep fine
 *   ghost storage and fine-to-coarse is clipped to coarse cells whose full
 *   2x2x2 fine subcell block is a valid storage index of the fine block
 *   (the kernels' documented preconditions);
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

	// implementation details of the subcycling stages (kept public for
	// future subclass overrides, same as the base State members)

	// launch `cudaLBMKernel` on the full extent of every block at `level`
	// (null-stream, interior launch configuration) and synchronize
	void launchLBMKernelForLevel(int level, bool compute_macro);
	// fill the ghost layer of every level-`fine_level` block from level
	// `fine_level - 1` via `cudaAMR_CoarseToFine`, iterating the
	// `AMR_InterfacePatch` descriptors of \ref couplings
	void launchCoarseToFineTransfers(int fine_level, bool c2f_time_centered);
	// project fine-averaged DFs onto the frozen GEO_NOTHING skin cells of
	// each fine footprint (interior_patches of \ref couplings) -- the ONLY
	// fine-to-coarse channel since the ring F2C launch was removed (gate B
	// ruling, D.1 hard-delete)
	void launchFineToCoarseTransfersInterior(int fine_level);

	// build \ref couplings from the `GEO_AMR_INTERFACE` markings (called by
	// SimInit AFTER `markAMRInterface` ran); see the implementation for the
	// face partition and the validity test
	void buildCouplings();
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
void State_AMR<NSE>::launchLBMKernelForLevel(int level, bool compute_macro)
{
	for (auto* block : this->nse.getBlocksAtLevel(level)) {
		const auto direction = TNL::Containers::SyncDirection::None;
		TNL::Backend::LaunchConfiguration launch_config;
		launch_config.blockSize = block->computeData.at(direction).blockSize;
		launch_config.gridSize = block->computeData.at(direction).gridSize;
		TNL::Backend::launchKernelAsync(
			cudaLBMKernel<NSE>, launch_config, block->data, idx3d{0, 0, 0}, block->local, block->is_distributed(), compute_macro
		);
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

	// build the inter-level coupling descriptors consumed by the transfer
	// launches in SimUpdate()
	buildCouplings();

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
		double block_ke = 0.0;

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

// forLocalLatticeSites is OpenMP-parallel: accumulate atomically
#pragma omp atomic update
				s.total_mass += rho * volume_factor;
#pragma omp atomic update
				s.total_momentum_x += rho * vx * volume_factor;
#pragma omp atomic update
				s.total_momentum_y += rho * vy * volume_factor;
#pragma omp atomic update
				s.total_momentum_z += rho * vz * volume_factor;
#pragma omp atomic update
				block_ke += 0.5 * rho * (vx * vx + vy * vy + vz * vz);
			}
		);

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
 * Each patch covers a rectangle of PARENT-level halo cells (1 cell thick
 * in the face normal) and the matching fine-level ghost rectangle (2 fine
 * cells per coarse cell). Refinement-level blocks allocate a 1-cell-deep
 * DF overlap (see `LBM_BLOCK::storage_overlap`), so the coarse-to-fine
 * launch clips the fill to that single layer: it is the ghost layer read
 * by the fine-level streaming and by the max-side skin fine-to-coarse
 * windows. Both rectangles are stored in the two
 * blocks' indexer coordinates: with the 2:1 refinement ratio, fine cell
 * `2c` covers coarse cell `c`, and `fine_origin = 2 * coarse_rect_begin -
 * fine.offset` (per axis). The launches map
 * `begin = fine_origin, end = fine_origin + fine_size` for coarse-to-fine
 * and `begin = coarse_origin, end = coarse_origin + coarse_size` for
 * fine-to-coarse, subject to the storability guards documented at the
 * launch helpers.
 *
 * The INTERIOR (under-footprint) patch list holds the footprint's 6
 * disjoint inset-face SKIN rectangles (one coarse cell deep inside the
 * footprint, the same disjoint partition idiom as the halo faces above).
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
			// the footprint size is fine.local / 2 with the 2:1 ratio)
			const idx3d& go = fine->global_offset;
			const idx3d gs{fine->local.x() / 2, fine->local.y() / 2, fine->local.z() / 2};

			// the six faces of the footprint's 1-cell halo box in parent-level
			// global coordinates (disjoint partition, see the docstring)
			const struct FACE
			{
				SyncDirection face;
				idx3d begin, end;
			} faces[6] = {
				{SyncDirection::Left, {go.x() - 1, go.y() - 1, go.z() - 1}, {go.x(), go.y() + gs.y() + 1, go.z() + gs.z() + 1}},
				{SyncDirection::Right, {go.x() + gs.x(), go.y() - 1, go.z() - 1}, {go.x() + gs.x() + 1, go.y() + gs.y() + 1, go.z() + gs.z() + 1}},
				{SyncDirection::Bottom, {go.x(), go.y() - 1, go.z() - 1}, {go.x() + gs.x(), go.y(), go.z() + gs.z() + 1}},
				{SyncDirection::Top, {go.x(), go.y() + gs.y(), go.z() - 1}, {go.x() + gs.x(), go.y() + gs.y() + 1, go.z() + gs.z() + 1}},
				{SyncDirection::Back, {go.x(), go.y(), go.z() - 1}, {go.x() + gs.x(), go.y() + gs.y(), go.z()}},
				{SyncDirection::Front, {go.x(), go.y(), go.z() + gs.z()}, {go.x() + gs.x(), go.y() + gs.y(), go.z() + gs.z() + 1}},
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
					// the docstring); the fine rectangle covers 2 fine cells
					// per coarse cell on each axis
					patch.coarse_origin = {begin.x() - coarse->offset.x(), begin.y() - coarse->offset.y(), begin.z() - coarse->offset.z()};
					patch.coarse_size = {end.x() - begin.x(), end.y() - begin.y(), end.z() - begin.z()};
					patch.fine_origin = {2 * begin.x() - fine->offset.x(), 2 * begin.y() - fine->offset.y(), 2 * begin.z() - fine->offset.z()};
					patch.fine_size = {2 * patch.coarse_size.x(), 2 * patch.coarse_size.y(), 2 * patch.coarse_size.z()};
					patch.face = f.face;
					coupling.patches.push_back(patch);
					coupling.coarse_block_ids.push_back(coarse->id);
					coupling.fine_block_ids.push_back(fine->id);
				}
			}

			// interior patches (changes 2+3 of the AMR interface redesign,
			// docs/AMR-interface-proposed-diagram.md §3/§7 — unconditional
			// since the ring F2C path was removed, gate B ruling + D.1
			// hard-delete): the
			// one-coarse-cell-deep SKIN of the fine footprint (frozen
			// GEO_NOTHING cells, F2C-injected with fine-filtered DFs each
			// cycle) as a DISJOINT partition of 6 inset-face rectangles —
			// the same disjoint face-partition idiom as the halo ring above,
			// inset one coarse cell INTO the footprint: the x-normal faces
			// own the full footprint y/z range, the y-normal faces the
			// interior x-range, the z-normal faces the interior x/y range.
			// The deep frozen core is never F2C-written (the coarse C2F
			// stencil reaches only 1 cell into the footprint, so the core
			// is never read either) — e.g. a 32^3 footprint emits its
			// 32^3-30^3 = 5,768 skin cells in 6 rectangles.
			// Degenerate thin footprints clamp to EMPTY rectangles (skipped
			// by the clip below, never pushed): with gs.a < 3 the tangent
			// interior ranges [go.a+1, go.a+gs.a-1) of the other axes'
			// faces are empty, and the max(..., go.a+1) clamp on the
			// max-side slab origin keeps a 1-cell-thin footprint from
			// emitting the same axis-plane twice (the min-side face wins)
			// — no rectangle therefore carries a negative extent and no
			// coarse cell is written twice.
			const idx xi0 = go.x() + 1, xi1 = go.x() + gs.x() - 1;
			const idx yi0 = go.y() + 1, yi1 = go.y() + gs.y() - 1;
			const idx zi0 = go.z() + 1, zi1 = go.z() + gs.z() - 1;
			const idx xr0 = std::max(go.x() + gs.x() - 1, go.x() + 1);
			const idx yr0 = std::max(go.y() + gs.y() - 1, go.y() + 1);
			const idx zr0 = std::max(go.z() + gs.z() - 1, go.z() + 1);
			const struct SKIN
			{
				idx3d begin, end;
			} skins[6] = {
				{{go.x(), go.y(), go.z()}, {go.x() + 1, go.y() + gs.y(), go.z() + gs.z()}},	   // x-min face (full y/z)
				{{xr0, go.y(), go.z()}, {go.x() + gs.x(), go.y() + gs.y(), go.z() + gs.z()}},  // x-max face (full y/z)
				{{xi0, go.y(), go.z()}, {xi1, go.y() + 1, go.z() + gs.z()}},				   // y-min face (interior x)
				{{xi0, yr0, go.z()}, {xi1, go.y() + gs.y(), go.z() + gs.z()}},				   // y-max face (interior x)
				{{xi0, yi0, go.z()}, {xi1, yi1, go.z() + 1}},								   // z-min face (interior x/y)
				{{xi0, yi0, zr0}, {xi1, yi1, go.z() + gs.z()}},								   // z-max face (interior x/y)
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
		}

		couplings.push_back(std::move(coupling));
	}
}

template <typename NSE>
bool State_AMR<NSE>::isShadowedBySameLevelBlock(idx x, idx y, idx z, int fine_level, const BLOCK_NSE* owner)
{
	for (auto* other : this->nse.getBlocksAtLevel(fine_level)) {
		if (other == owner)
			continue;
		// footprint of the other fine block in parent-level global coordinates
		const idx3d& oo = other->global_offset;
		const idx3d os{other->local.x() / 2, other->local.y() / 2, other->local.z() / 2};
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
 * \brief Coarse-to-fine ghost-layer fill for one level (step 2a/2d of the
 * subcycling schedule).
 *
 * Iterates the `AMR_InterfacePatch` descriptors of \ref couplings matching
 * `fine_level` and launches one `cudaAMR_CoarseToFine` per patch with
 * `begin = fine_origin, end = fine_origin + fine_size` (fine indexer
 * coordinates).
 *
 * Storability guard: the patch's fine rectangle is clipped per axis to the
 * fine block's ALLOCATED ghost STORAGE (the overlap depth of the block's
 * indexer, 1 cell deep on refinement-level blocks -- see
 * `LBM_BLOCK::storage_overlap`; the inner layer feeds the fine-level
 * streaming directly, and it is also exactly the single ghost layer the
 * max-side skin fine-to-coarse windows read). Axes where the block spans
 * its global extent have no overlap allocated there and get no fill
 * (streaming then consumes exterior-boundary data instead). When
 * \ref couplings is empty (no marked interface cells), this is a silent
 * no-op (SimInit logged a warning).
 */
template <typename NSE>
void State_AMR<NSE>::launchCoarseToFineTransfers(int fine_level, bool c2f_time_centered)
{
	for (const InterLevelCoupling& coupling : couplings) {
		if (coupling.fine_level != fine_level)
			continue;

		for (std::size_t i = 0; i < coupling.patches.size(); i++) {
			const AMR_InterfacePatch<NSE>& patch = coupling.patches[i];
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
			// to the fine block's overlap storage (see the docstring)
			const idx3d ov{fine->df_overlap_X(), fine->df_overlap_Y(), fine->df_overlap_Z()};
			const idx3d begin{
				std::max(patch.fine_origin.x(), -ov.x()), std::max(patch.fine_origin.y(), -ov.y()), std::max(patch.fine_origin.z(), -ov.z())
			};
			const idx3d end{
				std::min(patch.fine_origin.x() + patch.fine_size.x(), fine->local.x() + ov.x()),
				std::min(patch.fine_origin.y() + patch.fine_size.y(), fine->local.y() + ov.y()),
				std::min(patch.fine_origin.z() + patch.fine_size.z(), fine->local.z() + ov.z())
			};
			if (begin.x() >= end.x() || begin.y() >= end.y() || begin.z() >= end.z())
				continue;

			const idx3d size{end.x() - begin.x(), end.y() - begin.y(), end.z() - begin.z()};

			TNL::Backend::LaunchConfiguration launch_config;
			launch_config.blockSize = fine->getCudaBlockSize(size);
			launch_config.gridSize = fine->getCudaGridSize(size, launch_config.blockSize);
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
				c2f_time_centered,
				fine->offset,
				coarse->offset
			);
		}
	}
	// the fine substeps below consume the ghost DFs on the same stream, but
	// keep the same discipline as the base driver (null-stream sync)
	TNL::Backend::streamSynchronize(0);
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
 * transfer — Lagrava-filtered fine-averaged DFs, full overwrite. The deep
 * frozen core is never written (and never read: the coarse C2F stencil
 * reaches at most 1 cell into the footprint). This is the two-way feedback
 * channel: fine-interior information reaches the coarse lattice through
 * the written skin cells, which the collision-active ring cells stream
 * from at the next coarse step. Every written skin cell reads its own
 * fine subcells plus the window clamp of the kernel (lo = 0, fine
 * interior only). When \ref couplings is empty (no marked interface
 * cells), this is a silent no-op (SimInit logged a warning).
 */
template <typename NSE>
void State_AMR<NSE>::launchFineToCoarseTransfersInterior(int fine_level)
{
	const int coarse_level = fine_level - 1;

	for (const InterLevelCoupling& coupling : couplings) {
		if (coupling.fine_level != fine_level)
			continue;

		for (std::size_t i = 0; i < coupling.interior_patches.size(); i++) {
			const AMR_InterfacePatch<NSE>& patch = coupling.interior_patches[i];
			BLOCK_NSE* fine = findBlockById(coupling.fine_level, coupling.interior_fine_block_ids[i]);
			BLOCK_NSE* coarse = findBlockById(coupling.coarse_level, coupling.interior_coarse_block_ids[i]);
			if (fine == nullptr || coarse == nullptr)
				continue;

			const dreal tau_fine = static_cast<dreal>(3 * blockLbmViscosity(*fine) + 0.5);
			const dreal tau_coarse = static_cast<dreal>(3 * blockLbmViscosity(*coarse) + 0.5);
			const bool fine_even_iter = fine->data.even_iter;
			const bool next_coarse_even_iter = (coarse_level == 0) ? ((this->nse.iterations % 2) == 1) : false;

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
			TNL::Backend::LaunchConfiguration launch_config;
			launch_config.blockSize = coarse->getCudaBlockSize(size);
			launch_config.gridSize = coarse->getCudaGridSize(size, launch_config.blockSize);
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
	TNL::Backend::streamSynchronize(0);
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

	// ---------- Berger-Colella step 1: one coarse (level 0) LBM step ----------
	// execute() (core.h) already called updateKernelData(), which set the
	// level-0 even_iter parity / DF rotation from the global `iterations`.
	// GEO_AMR_INTERFACE cells are collision-active inside the kernel
	// (BC::doCollision == true); the coarse kernel is their only writer
	// since the ring fine-to-coarse launch was removed (D.1)
	launchLBMKernelForLevel(0, compute_macro);

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

	// ---------- Berger-Colella recursion: finer levels ----------
	// 2026-08-18 H9 retry (user directive): under -DC2F_H9 the FIRST fill
	// of each cycle reads the time-centered (t_n + t_{n+1})/2 coarse state;
	// the BVP re-fill stays on the post-step state. AB-pattern only (the
	// D.4 defect makes AMR+AA unsupported).
	#ifdef C2F_H9
		#ifdef AA_PATTERN
			#error "C2F_H9 is AB-pattern only (AMR under AA carries the D.4 defect)"
		#endif
	constexpr bool h9_first_fill = true;
	#else
	constexpr bool h9_first_fill = false;
	#endif
	for (int L = 1; L <= this->nse.max_level; L++) {
		// 1. toggle the fine level's even_iter parity / DF rotation to
		// substep 0 BEFORE the ghost fill (CRITICAL: for the A-B pattern
		// the rotation selects the physical array df_cur refers to, and
		// the ghost fill must land in the array the upcoming substep
		// reads; the global updateKernelData() is driven by the coarse
		// clock and must not drive the fine substeps)
		this->nse.updateKernelDataForLevel(L, 0);

		// 2. coarse-to-fine: fill the fine ghost layer from level L-1
		// (patch rectangles of the level coupling built in SimInit)
		launchCoarseToFineTransfers(L, h9_first_fill);

		// 3. fine substep 1 of 2
		launchLBMKernelForLevel(L, compute_macro);

	#ifdef HAVE_MPI
		// exchange the latest DFs and dmacro on overlaps between the
		// level-L blocks (no-op for fine blocks in the v1 single-rank setup)
		if (this->nse.nproc > 1) {
			this->timer_wait_communication.start();
			this->nse.synchronizeDFsAndMacroDeviceForLevel(L, output_df, sync_macro);
			this->timer_wait_communication.stop();
		}
	#endif

		// 4. toggle the parity for substep 1 BEFORE the BVP re-fill (same
		// reason as above: the fill must target the df_cur frame the
		// upcoming substep reads)
		this->nse.updateKernelDataForLevel(L, 1);

		// 5. BVP: re-fill the fine ghost layer between the substeps (the
		// first substep's streaming consumed the ghost DFs and streamed
		// outward into them)
		launchCoarseToFineTransfers(L, false);

		// 6. fine substep 2 of 2
		launchLBMKernelForLevel(L, compute_macro);

	#ifdef HAVE_MPI
		// exchange the latest DFs and dmacro on overlaps between the
		// level-L blocks (no-op for fine blocks in the v1 single-rank setup)
		if (this->nse.nproc > 1) {
			this->timer_wait_communication.start();
			this->nse.synchronizeDFsAndMacroDeviceForLevel(L, output_df, sync_macro);
			this->timer_wait_communication.stop();
		}
	#endif

		// 7. fine-to-coarse: inject the (Lagrava-filtered) fine state into
		// the frozen GEO_NOTHING cells of the 6 skin rectangles of each
		// fine footprint (two-way feedback). Ring cells stream+collide
		// only -- the ring F2C launch was removed (gate B ruling, D.1
		// hard-delete) and the fine feedback reaches them through
		// streaming from the freshly F2C-written skin on the next coarse
		// step (df_out -> next df_cur convention, no kernel change)
		launchFineToCoarseTransfersInterior(L);
	}

	this->timer_compute.stop();
	this->timer_SimUpdate.stop();
#endif
}
