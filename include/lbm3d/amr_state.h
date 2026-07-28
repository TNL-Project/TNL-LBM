#pragma once

#include <algorithm>
#include <cmath>
#include <utility>
#include <vector>

#include <spdlog/spdlog.h>

#include "state.h"
#include "amr_decomposition.h"
#include "d3q27/amr_coupling.h"

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
 *    `iterations` counter). Coarse cells tagged `GEO_AMR_INTERFACE` are
 *    skipped inside the kernel via `BC::doCollision == false` (Wave 2).
 * 2. For each finer level L = 1..max_level:
 *    a. coarse-to-fine transfer: `cudaAMR_CoarseToFine` fills the fine
 *       ghost layer from level L-1 (see `d3q27/amr_coupling.h`),
 *    b. `updateKernelDataForLevel(L, 0)` toggles the fine level's even_iter
 *       parity / DF rotation to substep 0 (MANDATORY before the substep -
 *       the global `updateKernelData()` is driven by the coarse clock),
 *    c. fine substep 1 of 2: `cudaLBMKernel` on all blocks at level L,
 *    d. BVP: the coarse-to-fine transfer re-fills the fine ghost layer
 *       (substep 1 consumed the ghost DFs and streamed outward into them),
 *    e. `updateKernelDataForLevel(L, 1)` toggles the parity again,
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
 * - Kernel extents are passed in the target block's indexer coordinates;
 *   the kernels assume the two blocks' indexer origins correspond (see the
 *   precondition in the kernel docstrings - a Wave 5/6 concern).
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

	// implementation details of the subcycling stages (kept public for
	// future subclass overrides, same as the base State members)

	// launch `cudaLBMKernel` on the full extent of every block at `level`
	// (null-stream, interior launch configuration) and synchronize
	void launchLBMKernelForLevel(int level, bool compute_macro);
	// fill the ghost layer of every level-`fine_level` block from level
	// `fine_level - 1` via `cudaAMR_CoarseToFine`, iterating the
	// `AMR_InterfacePatch` descriptors of \ref couplings
	void launchCoarseToFineTransfers(int fine_level);
	// project the filtered fine state of every level-`fine_level` block back
	// onto level `fine_level - 1` via `cudaAMR_FineToCoarse`, iterating the
	// `AMR_InterfacePatch` descriptors of \ref couplings
	void launchFineToCoarseTransfers(int fine_level);

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

private:
	// host-side reduction over all blocks: volume-weighted global mass and
	// momentum plus per-level kinetic energy (see AfterSimUpdate)
	AMRConservationStats computeConservationStats();
};

/**
 * \brief Launch `cudaLBMKernel` on every block at the given level.
 *
 * v1 single-GPU: all blocks are local, so the interior
 * (`SyncDirection::None`) launch configuration is used over the FULL block
 * extent on the null stream (matches the `nproc == 1` path of the base
 * driver). `GEO_AMR_INTERFACE` cells are skipped inside the kernel via
 * `BC::doCollision == false`; fine-level ghost cells are maintained by the
 * coupling kernels, not by MPI overlap synchronization.
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
	for (const InterLevelCoupling& coupling : couplings) {
		total_patches += coupling.patches.size();
		spdlog::info(
			"State_AMR: coupling level {} -> {} has {} interface patches", coupling.coarse_level, coupling.fine_level, coupling.patches.size()
		);
	}
	if (total_patches == 0)
		spdlog::warn(
			"State_AMR: no GEO_AMR_INTERFACE cells found for max_level = {} - inter-level coupling kernels will not launch", this->nse.max_level
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
 * \brief Host-side reduction of the conservation quantities over all blocks.
 *
 * Each block's macroscopic quantities are copied to the host and summed in
 * physical-volume units: the cell volume scales as `(1/2)^3` per refinement
 * level with the 2:1 ratio, so a level-L cell weights `1/8^L` of a coarse
 * cell. The per-level kinetic energy sums `0.5 * rho * |u|^2` without the
 * volume weight (per-level diagnostic).
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
 * cells per coarse cell, i.e. 2 cells thick - the outer cell layer feeds
 * only the fine-to-coarse filter, the inner layer is the ghost layer read
 * by the fine-level streaming). Both rectangles are stored in the two
 * blocks' indexer coordinates: with the 2:1 refinement ratio, fine cell
 * `2c` covers coarse cell `c`, and `fine_origin = 2 * coarse_rect_begin -
 * fine.offset` (per axis). The launches map
 * `begin = fine_origin, end = fine_origin + fine_size` for coarse-to-fine
 * and `begin = coarse_origin, end = coarse_origin + coarse_size` for
 * fine-to-coarse, subject to the storability guards documented at the
 * launch helpers.
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
 * Storability guard: the patch's fine rectangle is clipped to the fine
 * block's ghost STORAGE, which is only `overlap_width` cells deep around
 * `[0, fine.local)`; with `overlap_width == 1` the outer cell layer of the
 * 2-cell-deep rectangle is skipped (the kernels read only the 1-cell-deep
 * ghost layer during streaming, so no consumed data is lost). When
 * \ref couplings is empty (no marked interface cells), this is a silent
 * no-op (SimInit logged a warning).
 */
template <typename NSE>
void State_AMR<NSE>::launchCoarseToFineTransfers(int fine_level)
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
			const idx ov = BLOCK_NSE::overlap_width;
			const idx3d begin{std::max(patch.fine_origin.x(), -ov), std::max(patch.fine_origin.y(), -ov), std::max(patch.fine_origin.z(), -ov)};
			const idx3d end{
				std::min(patch.fine_origin.x() + patch.fine_size.x(), fine->local.x() + ov),
				std::min(patch.fine_origin.y() + patch.fine_size.y(), fine->local.y() + ov),
				std::min(patch.fine_origin.z() + patch.fine_size.z(), fine->local.z() + ov)
			};
			if (begin.x() >= end.x() || begin.y() >= end.y() || begin.z() >= end.z())
				continue;

			const idx3d size{end.x() - begin.x(), end.y() - begin.y(), end.z() - begin.z()};

			TNL::Backend::LaunchConfiguration launch_config;
			launch_config.blockSize = fine->getCudaBlockSize(size);
			launch_config.gridSize = fine->getCudaGridSize(size, launch_config.blockSize);
			TNL::Backend::launchKernelAsync(
				cudaAMR_CoarseToFine<NSE>, launch_config, fine->data, coarse->data, begin, end, tau_fine, tau_coarse, coarse_even_iter
			);
		}
	}
	// the fine substeps below consume the ghost DFs on the same stream, but
	// keep the same discipline as the base driver (null-stream sync)
	TNL::Backend::streamSynchronize(0);
}

/**
 * \brief Fine-to-coarse transfer for one level (step 2g of the subcycling
 * schedule).
 *
 * Iterates the `AMR_InterfacePatch` descriptors of \ref couplings matching
 * `fine_level` and launches one `cudaAMR_FineToCoarse` per patch with
 * `begin = coarse_origin, end = coarse_origin + coarse_size` (coarse
 * indexer coordinates) - the patch rectangles ARE the interface cells
 * tagged by `markAMRInterface` (one patch per (fine block, parent block,
 * halo face) triple).
 *
 * Storability guard: the kernel reads the 2x2x2 fine subcells {2c, 2c+1}
 * (per axis, in the kernel's shared indexer frame) of every coarse cell c
 * in the extent, so the extent is clipped to cells whose subcells are all
 * valid fine storage indices - the fine overlap storage is
 * `overlap_width` deep around `[0, fine.local)`. When \ref couplings is
 * empty (no marked interface cells), this is a silent no-op (SimInit
 * logged a warning).
 */
template <typename NSE>
void State_AMR<NSE>::launchFineToCoarseTransfers(int fine_level)
{
	const int coarse_level = fine_level - 1;

	for (const InterLevelCoupling& coupling : couplings) {
		if (coupling.fine_level != fine_level)
			continue;

		for (std::size_t i = 0; i < coupling.patches.size(); i++) {
			const AMR_InterfacePatch<NSE>& patch = coupling.patches[i];
			BLOCK_NSE* fine = findBlockById(coupling.fine_level, coupling.fine_block_ids[i]);
			BLOCK_NSE* coarse = findBlockById(coupling.coarse_level, coupling.coarse_block_ids[i]);
			if (fine == nullptr || coarse == nullptr)
				continue;

			const dreal tau_fine = static_cast<dreal>(3 * blockLbmViscosity(*fine) + 0.5);
			const dreal tau_coarse = static_cast<dreal>(3 * blockLbmViscosity(*coarse) + 0.5);
			// parity of the stored fine data produced by fine substep 2
			// (AA-pattern state; ignored by the kernel for AB)
			const bool fine_even_iter = fine->data.even_iter;
			// parity of the NEXT consuming coarse substep: for level 0 the
			// next launch is the next global iteration's coarse step (the
			// counter was already incremented above); finer source levels
			// start their next subcycling cycle with substep 0
			// (even_iter == false)
			const bool next_coarse_even_iter = (coarse_level == 0) ? ((this->nse.iterations % 2) == 1) : false;

			// launch extent in the coarse block's indexer coordinates,
			// clipped to coarse cells whose 2x2x2 fine subcell block is
			// storable in the fine block, incl. its overlap layer (see the
			// docstring):
			//   2c >= -ov  &&  2c+1 < fine.local + ov   per axis,
			// where 2c, 2c+1 are the fine indexer coordinates the kernel
			// reads; the lower bound is x >= -ov/2, i.e. x >= 0 for
			// overlap_width <= 1, and the patch origins are non-negative
			const idx ov = BLOCK_NSE::overlap_width;
			const idx3d begin{
				std::max(patch.coarse_origin.x(), idx(0)), std::max(patch.coarse_origin.y(), idx(0)), std::max(patch.coarse_origin.z(), idx(0))
			};
			const idx3d end{
				std::min(patch.coarse_origin.x() + patch.coarse_size.x(), (fine->local.x() + ov - 2) / 2 + 1),
				std::min(patch.coarse_origin.y() + patch.coarse_size.y(), (fine->local.y() + ov - 2) / 2 + 1),
				std::min(patch.coarse_origin.z() + patch.coarse_size.z(), (fine->local.z() + ov - 2) / 2 + 1)
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
				next_coarse_even_iter
			);
		}
	}
	// synchronize the null-stream after all grids (same as the base driver)
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
	// GEO_AMR_INTERFACE cells are skipped inside the kernel (BC::doCollision
	// == false); their DFs are maintained by the fine-to-coarse transfer.
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
	for (int L = 1; L <= this->nse.max_level; L++) {
		// 1. coarse-to-fine: fill the fine ghost layer from level L-1
		// (patch rectangles of the level coupling built in SimInit).
		// v1 fills the PRE-toggle df_cur frame (exact plan ordering); for
		// the AB pattern the toggle below may rotate df_cur to the other
		// physical array - the ghost frame targeting is deferred to the
		// AA/AB review of Wave 5 (same v1 caveat as the fine-ghost
		// orientation in the coupling-kernel docstrings)
		launchCoarseToFineTransfers(L);

		// 2. fine substep 1 of 2: toggle the fine level's even_iter parity /
		// DF rotation to substep 0 BEFORE the kernel (CRITICAL: the global
		// updateKernelData() is driven by the coarse clock and must not
		// drive the fine substeps)
		this->nse.updateKernelDataForLevel(L, 0);
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

		// 3. BVP: re-fill the fine ghost layer between the substeps (the
		// first substep's streaming consumed the ghost DFs and streamed
		// outward into them)
		launchCoarseToFineTransfers(L);

		// 4. fine substep 2 of 2 (toggle the parity again)
		this->nse.updateKernelDataForLevel(L, 1);
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

		// 5. fine-to-coarse: project the (Lagrava-filtered) fine state back
		// onto the level L-1 interface cells of the coupling patches
		launchFineToCoarseTransfers(L);
	}

	this->timer_compute.stop();
	this->timer_SimUpdate.stop();
#endif
}
