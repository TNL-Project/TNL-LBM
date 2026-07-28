#pragma once

#include <algorithm>
#include <utility>
#include <vector>

#include <spdlog/spdlog.h>

#include "state.h"
#include "d3q27/amr_coupling.h"

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
 * - ghost/interface extents are derived INLINE from block geometry
 *   (`fine.offset + {-1, fine.local + 2}` ghost shell + a manual overlap
 *   test against the parent level's blocks via `getBlocksAtLevel`);
 *   \ref couplings (todo 8) will replace this derivation with proper
 *   `AMR_InterfacePatch` iterations;
 * - `nse.synchronizeDFsAndMacroDevice()` synchronizes ALL blocks - there
 *   is no per-level variant yet (todo 8); it is a no-op for fine blocks in
 *   the v1 single-rank setup and is only invoked for `nproc > 1` (the base
 *   driver needs no overlap copying for `nproc == 1` either);
 * - the fine-to-coarse staging is clipped to coarse cells whose full
 *   2x2x2 fine subcell block is a valid storage index of the fine block
 *   (the kernel's precondition). With `overlap_width == 1` the strict-halo
 *   cells tagged by `markAMRInterface` do not survive this clip (their
 *   fine subcells need a 2-cell-deep fine ghost layer), so the surviving
 *   footprint-interior cells receive the classic Berger-Colella
 *   replacement; the proper interface patches come with todo 8;
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
	 * \brief Host-side descriptor of one coarse-fine level interface.
	 *
	 * DECLARED here (todo 7), BUILT by todo 8 (`SimInit` will fill
	 * \ref couplings from the `GEO_AMR_INTERFACE` markings produced by
	 * `markAMRInterface`) and consumed by the coupling launches. The v1
	 * \ref SimUpdate derives the same extents inline from block geometry
	 * (see the file docstring).
	 */
	struct InterLevelCoupling
	{
		int coarse_level = 0;
		int fine_level = 0;
		std::vector<AMR_InterfacePatch<NSE>> patches;
	};

	// all inter-level couplings of the simulation (empty until todo 8)
	std::vector<InterLevelCoupling> couplings;

	// pass-through constructor: State_AMR adds no physical state of its own
	template <typename... ARGS>
	State_AMR(ARGS&&... args)
	: Base(std::forward<ARGS>(args)...)
	{}

	// Berger-Colella subcycling (see the file docstring); delegates to the
	// base driver when `nse.max_level == 0`
	void SimUpdate() override;

	// implementation details of the subcycling stages (kept public for
	// future subclass overrides, same as the base State members)

	// launch `cudaLBMKernel` on the full extent of every block at `level`
	// (null-stream, interior launch configuration) and synchronize
	void launchLBMKernelForLevel(int level, bool compute_macro);
	// fill the ghost layer of every level-`fine_level` block from level
	// `fine_level - 1` via `cudaAMR_CoarseToFine` (inline extent derivation)
	void launchCoarseToFineTransfers(int fine_level);
	// project the filtered fine state of every level-`fine_level` block back
	// onto level `fine_level - 1` via `cudaAMR_FineToCoarse`
	void launchFineToCoarseTransfers(int fine_level);

	// per-level lattice viscosity nu_lb of a block: level-0 blocks use the
	// global lattice (their lat_local is only initialized by the level-aware
	// block constructor); finer levels use their per-level lat_local
	real blockLbmViscosity(const BLOCK_NSE& block) const
	{
		return block.level == 0 ? this->nse.lat.lbmViscosity() : block.lat_local.lbmViscosity();
	}
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
 * \brief Coarse-to-fine ghost-layer fill for one level (step 2a/2d of the
 * subcycling schedule).
 *
 * Inline v1 extent derivation (todo 8 replaces this with \ref couplings):
 * for each fine block at level `fine_level` and each level-(`fine_level-1`)
 * block, the 1-cell ghost shell around the fine interior (global fine
 * coordinates `[fine.offset + {-1}, fine.offset + fine.local + {1})`) is
 * intersected with the coarse block's coverage scaled to fine coordinates
 * (2:1 ratio), partitioned into 6 disjoint face strips, and each non-empty
 * strip is launched as one `cudaAMR_CoarseToFine` call. A face whose
 * exterior holds another fine (same-level) block instead of coarse cells
 * produces no overlap and is skipped - v1 assumes interface ghosts always
 * touch the parent level (same-level fine-fine abutment is todo 8).
 */
template <typename NSE>
void State_AMR<NSE>::launchCoarseToFineTransfers(int fine_level)
{
	const int coarse_level = fine_level - 1;

	for (auto* fine : this->nse.getBlocksAtLevel(fine_level)) {
		// tau = 3*nu_lb + 0.5 per level (see the file docstring)
		const dreal tau_fine = static_cast<dreal>(3 * blockLbmViscosity(*fine) + 0.5);

		// the fine block's interior in fine-level global coordinates
		const idx3d& fo = fine->offset;
		const idx3d& fs = fine->local;

		for (auto* coarse : this->nse.getBlocksAtLevel(coarse_level)) {
			const dreal tau_coarse = static_cast<dreal>(3 * blockLbmViscosity(*coarse) + 0.5);
			// parity of the kernel launch that produced the current coarse
			// data (AA-pattern state; ignored by the kernel for AB)
			const bool coarse_even_iter = coarse->data.even_iter;

			// coarse block's coverage scaled to fine-level global coordinates
			// (consecutive levels always have a 2:1 refinement ratio)
			const idx3d cb{2 * coarse->offset.x(), 2 * coarse->offset.y(), 2 * coarse->offset.z()};
			const idx3d ce{
				2 * (coarse->offset.x() + coarse->local.x()),
				2 * (coarse->offset.y() + coarse->local.y()),
				2 * (coarse->offset.z() + coarse->local.z())
			};

			// bounding box of the 1-cell ghost shell around the fine interior
			const idx3d shell_begin{fo.x() - 1, fo.y() - 1, fo.z() - 1};
			const idx3d shell_end{fo.x() + fs.x() + 1, fo.y() + fs.y() + 1, fo.z() + fs.z() + 1};

			// overlap of the shell with this coarse block's coverage
			const idx3d inter_begin{std::max(shell_begin.x(), cb.x()), std::max(shell_begin.y(), cb.y()), std::max(shell_begin.z(), cb.z())};
			const idx3d inter_end{std::min(shell_end.x(), ce.x()), std::min(shell_end.y(), ce.y()), std::min(shell_end.z(), ce.z())};
			if (inter_begin.x() >= inter_end.x() || inter_begin.y() >= inter_end.y() || inter_begin.z() >= inter_end.z())
				continue;

			// launch one coarse-to-fine kernel on a ghost-extent strip:
			// clip to the overlap box and convert fine-global coordinates
			// into the fine block's indexer coordinates
			const auto fill_strip = [&](const idx3d& strip_begin, const idx3d& strip_end)
			{
				const idx3d begin{
					std::max(strip_begin.x(), inter_begin.x()), std::max(strip_begin.y(), inter_begin.y()), std::max(strip_begin.z(), inter_begin.z())
				};
				const idx3d end{
					std::min(strip_end.x(), inter_end.x()), std::min(strip_end.y(), inter_end.y()), std::min(strip_end.z(), inter_end.z())
				};
				if (begin.x() >= end.x() || begin.y() >= end.y() || begin.z() >= end.z())
					return;

				const idx3d begin_local{begin.x() - fo.x(), begin.y() - fo.y(), begin.z() - fo.z()};
				const idx3d end_local{end.x() - fo.x(), end.y() - fo.y(), end.z() - fo.z()};
				const idx3d size{end_local.x() - begin_local.x(), end_local.y() - begin_local.y(), end_local.z() - begin_local.z()};

				TNL::Backend::LaunchConfiguration launch_config;
				launch_config.blockSize = fine->getCudaBlockSize(size);
				launch_config.gridSize = fine->getCudaGridSize(size, launch_config.blockSize);
				TNL::Backend::launchKernelAsync(
					cudaAMR_CoarseToFine<NSE>, launch_config, fine->data, coarse->data, begin_local, end_local, tau_fine, tau_coarse, coarse_even_iter
				);
			};

			// disjoint partition of the shell around [fo, fo + fs): the
			// x-faces span the full y/z shell, the y-faces the interior
			// x-range, and the z-faces the interior x/y range
			fill_strip({fo.x() - 1, shell_begin.y(), shell_begin.z()}, {fo.x(), shell_end.y(), shell_end.z()});
			fill_strip({fo.x() + fs.x(), shell_begin.y(), shell_begin.z()}, {fo.x() + fs.x() + 1, shell_end.y(), shell_end.z()});
			fill_strip({fo.x(), fo.y() - 1, shell_begin.z()}, {fo.x() + fs.x(), fo.y(), shell_end.z()});
			fill_strip({fo.x(), fo.y() + fs.y(), shell_begin.z()}, {fo.x() + fs.x(), fo.y() + fs.y() + 1, shell_end.z()});
			fill_strip({fo.x(), fo.y(), fo.z() - 1}, {fo.x() + fs.x(), fo.y() + fs.y(), fo.z()});
			fill_strip({fo.x(), fo.y(), fo.z() + fs.z()}, {fo.x() + fs.x(), fo.y() + fs.y(), fo.z() + fs.z() + 1});
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
 * The candidate extent is the fine footprint's 1-cell halo box at the
 * parent level (the cells `markAMRInterface` considers) intersected with
 * each level-(`fine_level-1`) block's range - the same overlap test as
 * \ref launchCoarseToFineTransfers in parent-level coordinates (the fine
 * block's `global_offset` is its parent-level origin and the footprint
 * size is `fine.local / 2`). The extent is additionally clipped to coarse
 * cells whose 2x2x2 fine subcell block is storable in the fine block
 * (including overlaps) - the kernel's documented precondition. With
 * `overlap_width == 1` only footprint-interior cells survive (see the
 * file docstring); the strict-halo cells become reachable with the proper
 * interface patches of todo 8.
 */
template <typename NSE>
void State_AMR<NSE>::launchFineToCoarseTransfers(int fine_level)
{
	const int coarse_level = fine_level - 1;

	for (auto* fine : this->nse.getBlocksAtLevel(fine_level)) {
		const dreal tau_fine = static_cast<dreal>(3 * blockLbmViscosity(*fine) + 0.5);
		// parity of the stored fine data produced by fine substep 2
		// (AA-pattern state; ignored by the kernel for AB)
		const bool fine_even_iter = fine->data.even_iter;
		// parity of the NEXT consuming coarse substep: for level 0 the next
		// launch is the next global iteration's coarse step (the counter was
		// already incremented above); finer source levels start their next
		// subcycling cycle with substep 0 (even_iter == false)
		const bool next_coarse_even_iter = (coarse_level == 0) ? ((this->nse.iterations % 2) == 1) : false;

		// footprint of the fine block in parent-level global coordinates
		// (createAMRBlocks stores the parent-level origin in global_offset;
		// the footprint size is fine.local / 2 with the 2:1 ratio)
		const idx3d& fo = fine->global_offset;
		const idx3d fs{fine->local.x() / 2, fine->local.y() / 2, fine->local.z() / 2};

		// fine-cell subscripts of a coarse cell c are {2c, 2c+1} per axis
		// (see the kernel); keep only coarse cells whose full 2x2x2 subcell
		// block is storable in the fine block, incl. its overlap layer:
		//   2c >= offset - ov  &&  2c+1 < offset + local + ov   per axis,
		// where offset/local are the fine block's fine-level global bounds
		const idx ov = BLOCK_NSE::overlap_width;
		const idx3d storable_begin{(fine->offset.x() - ov + 1) / 2, (fine->offset.y() - ov + 1) / 2, (fine->offset.z() - ov + 1) / 2};
		const idx3d storable_end{
			(fine->offset.x() + fine->local.x() + ov - 2) / 2 + 1,
			(fine->offset.y() + fine->local.y() + ov - 2) / 2 + 1,
			(fine->offset.z() + fine->local.z() + ov - 2) / 2 + 1
		};

		for (auto* coarse : this->nse.getBlocksAtLevel(coarse_level)) {
			const dreal tau_coarse = static_cast<dreal>(3 * blockLbmViscosity(*coarse) + 0.5);

			// 1-cell halo box around the fine footprint, clipped to this
			// coarse block's range (global parent-level coordinates, the
			// same overlap test as markAMRInterface)
			const idx3d begin{
				std::max({coarse->offset.x(), fo.x() - 1, storable_begin.x()}),
				std::max({coarse->offset.y(), fo.y() - 1, storable_begin.y()}),
				std::max({coarse->offset.z(), fo.z() - 1, storable_begin.z()})
			};
			const idx3d end{
				std::min({coarse->offset.x() + coarse->local.x(), fo.x() + fs.x() + 1, storable_end.x()}),
				std::min({coarse->offset.y() + coarse->local.y(), fo.y() + fs.y() + 1, storable_end.y()}),
				std::min({coarse->offset.z() + coarse->local.z(), fo.z() + fs.z() + 1, storable_end.z()})
			};
			if (begin.x() >= end.x() || begin.y() >= end.y() || begin.z() >= end.z())
				continue;

			// convert parent-global coordinates into the coarse block's
			// indexer coordinates
			const idx3d begin_local{begin.x() - coarse->offset.x(), begin.y() - coarse->offset.y(), begin.z() - coarse->offset.z()};
			const idx3d end_local{end.x() - coarse->offset.x(), end.y() - coarse->offset.y(), end.z() - coarse->offset.z()};
			const idx3d size{end_local.x() - begin_local.x(), end_local.y() - begin_local.y(), end_local.z() - begin_local.z()};

			TNL::Backend::LaunchConfiguration launch_config;
			launch_config.blockSize = coarse->getCudaBlockSize(size);
			launch_config.gridSize = coarse->getCudaGridSize(size, launch_config.blockSize);
			TNL::Backend::launchKernelAsync(
				cudaAMR_FineToCoarse<NSE>,
				launch_config,
				coarse->data,
				fine->data,
				begin_local,
				end_local,
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
	// exchange the latest DFs and dmacro on overlaps between blocks;
	// NOTE: synchronizeDFsAndMacroDevice synchronizes ALL blocks - the
	// per-level variant is todo 8 (a no-op for fine blocks in the v1
	// single-rank setup, and skipped entirely for nproc == 1 like the base driver)
	if (this->nse.nproc > 1) {
		this->timer_wait_communication.start();
		this->nse.synchronizeDFsAndMacroDevice(output_df, sync_macro);
		this->timer_wait_communication.stop();
	}
	#endif

	// ---------- Berger-Colella recursion: finer levels ----------
	for (int L = 1; L <= this->nse.max_level; L++) {
		// 1. coarse-to-fine: fill the fine ghost layer from level L-1.
		// v1 fills the PRE-toggle df_cur frame (exact plan ordering); for
		// the AB pattern the toggle below may rotate df_cur to the other
		// physical array - the ghost frame targeting is part of the
		// coupling rework in todo 8 (same v1 caveat as the fine-ghost
		// orientation in the coupling-kernel docstrings)
		launchCoarseToFineTransfers(L);

		// 2. fine substep 1 of 2: toggle the fine level's even_iter parity /
		// DF rotation to substep 0 BEFORE the kernel (CRITICAL: the global
		// updateKernelData() is driven by the coarse clock and must not
		// drive the fine substeps)
		this->nse.updateKernelDataForLevel(L, 0);
		launchLBMKernelForLevel(L, compute_macro);

	#ifdef HAVE_MPI
		if (this->nse.nproc > 1) {
			this->timer_wait_communication.start();
			this->nse.synchronizeDFsAndMacroDevice(output_df, sync_macro);
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
		if (this->nse.nproc > 1) {
			this->timer_wait_communication.start();
			this->nse.synchronizeDFsAndMacroDevice(output_df, sync_macro);
			this->timer_wait_communication.stop();
		}
	#endif

		// 5. fine-to-coarse: project the (Lagrava-filtered) fine state back
		// onto the level L-1 cells within the fine footprint's coarse box
		launchFineToCoarseTransfers(L);
	}

	this->timer_compute.stop();
	this->timer_SimUpdate.stop();
#endif
}
