#pragma once

#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <fmt/core.h>
#include <spdlog/spdlog.h>

#include "lbm.h"

/**
 * \brief Static AMR region declaration, validation and block instantiation.
 *
 * An \ref AMR_Region is a rectangular fine-lattice region specified in
 * coarsest-level (level-0) cell coordinates. With the standard 2:1
 * refinement ratio, each coarse cell of the region becomes 2x2x2 fine cells,
 * so the created fine block has `2 * size_coarse` local cells per axis.
 *
 * The region format parsed by \ref parseAMRConfig is one region per line:
 *
 *     level  origin_x origin_y origin_z  size_x size_y size_z
 *
 * Lines beginning with '#' are comments and blank lines are skipped.
 *
 * v1 scope (static single-hop refinement):
 * - every region must have `level == 1` (level-0 blocks are created only by
 *   the LBM constructors, and multi-level nesting is future work),
 * - a region is stored on a single MPI rank (`lbm.nproc` must be 1),
 * - overlapping regions are not merged.
 *
 * Coupling between levels (Wave 3, `amr_coupling.h`) locates interfaces from
 * `block.global_offset` (parent-level coordinates of the block origin) and
 * `block.lat_local` (per-level lattice parameters scaled by `initLevelLattice`).
 */

template <typename CONFIG>
struct AMR_Region
{
	using TRAITS = typename CONFIG::TRAITS;
	using idx = typename TRAITS::idx;
	using idx3d = typename TRAITS::idx3d;
	idx3d origin_coarse;  // lower-left-front corner in coarse-level coordinates
	idx3d size_coarse;	  // extent in coarse cells (each becomes 2^3 = 8 fine cells)
	int level;			  // refinement level (must be > 0)
};

/**
 * \brief Parse a text configuration of AMR refinement regions.
 *
 * Fails early on the first malformed line with spdlog::error + std::runtime_error
 * including the line number and the raw line content.
 */
template <typename CONFIG>
std::vector<AMR_Region<CONFIG>> parseAMRConfig(const std::string& config)
{
	using idx = typename CONFIG::TRAITS::idx;
	using idx3d = typename CONFIG::TRAITS::idx3d;

	std::vector<AMR_Region<CONFIG>> regions;

	std::istringstream input(config);
	std::string line;
	int line_number = 0;

	while (std::getline(input, line)) {
		line_number++;
		const std::string raw_line = line;

		// '#' starts a comment anywhere on the line
		const std::size_t comment = line.find('#');
		if (comment != std::string::npos)
			line.erase(comment);
		// blank lines are skipped
		if (line.find_first_not_of(" \t\r") == std::string::npos)
			continue;

		// expected format (whitespace-separated): level ox oy oz lx ly lz
		std::istringstream tokens(line);
		int level;
		idx ox, oy, oz, lx, ly, lz;
		std::string trailing;
		if (! (tokens >> level >> ox >> oy >> oz >> lx >> ly >> lz) || (tokens >> trailing)) {
			const std::string message =
				fmt::format("malformed AMR region at line {}: \"{}\" (expected format: level ox oy oz lx ly lz)", line_number, raw_line);
			spdlog::error("{}", message);
			throw std::runtime_error(message);
		}

		AMR_Region<CONFIG> region;
		region.level = level;
		region.origin_coarse = idx3d{ox, oy, oz};
		region.size_coarse = idx3d{lx, ly, lz};
		// keep file order so that validation errors reference stable region indices
		regions.push_back(region);
	}

	return regions;
}

/**
 * \brief Validate AMR regions and create the corresponding fine-level blocks.
 *
 * Strict two-phase behavior: every region is validated (read-only) before any
 * block is created, so a validation failure never mutates `lbm.blocks`.
 * Each created block is allocated on host+device, its interior map is set to
 * `GEO_FLUID`, and its DFs are initialized to equilibrium with zero
 * macroscopic fields (the caller overwrites this with its own initial
 * condition on all levels before the first step).
 */
template <typename CONFIG>
void createAMRBlocks(LBM<CONFIG>& lbm, const std::vector<AMR_Region<CONFIG>>& regions)
{
	using idx = typename CONFIG::TRAITS::idx;
	using idx3d = typename CONFIG::TRAITS::idx3d;
	using point_t = typename LBM<CONFIG>::lat_t::PointType;

	// v1 scope guard: a fine region is stored on a single rank (no cross-rank decomposition)
	if (lbm.nproc > 1) {
		const std::string message = fmt::format("createAMRBlocks: multi-GPU AMR is not supported (nproc = {} > 1)", lbm.nproc);
		spdlog::error("{}", message);
		throw std::runtime_error(message);
	}

	// ---------------------------------------------------------------------------
	// phase 1: validate ALL regions (read-only) before creating ANY block
	// ---------------------------------------------------------------------------
	for (std::size_t i = 0; i < regions.size(); i++) {
		const AMR_Region<CONFIG>& region = regions[i];

		// every rejection logs an spdlog::error and throws - no silent skips
		const auto reject = [&](const std::string& reason)
		{
			const std::string message = fmt::format(
				"createAMRBlocks: invalid region #{} (level {}, origin [{},{},{}], size [{},{},{}]): {}",
				i,
				region.level,
				region.origin_coarse.x(),
				region.origin_coarse.y(),
				region.origin_coarse.z(),
				region.size_coarse.x(),
				region.size_coarse.y(),
				region.size_coarse.z(),
				reason
			);
			spdlog::error("{}", message);
			throw std::runtime_error(message);
		};

		// level-0 blocks are created only by the LBM constructors, not by this function
		if (region.level <= 0)
			reject("refinement level must be > 0");

		// the LBM level structure must have been pre-allocated with the max_level constructor
		if (region.level > lbm.max_level)
			reject(fmt::format("level {} exceeds LBM::max_level = {}", region.level, lbm.max_level));

		// a region is an extent of whole coarse cells
		if (region.size_coarse.x() <= 0 || region.size_coarse.y() <= 0 || region.size_coarse.z() <= 0)
			reject("region size must be positive in all axes");

		// nesting: the region must be fully contained in the coarsest-level global domain
		if (region.origin_coarse.x() < 0 || region.origin_coarse.y() < 0 || region.origin_coarse.z() < 0
			|| region.origin_coarse.x() + region.size_coarse.x() > lbm.lat.global.x()
			|| region.origin_coarse.y() + region.size_coarse.y() > lbm.lat.global.y()
			|| region.origin_coarse.z() + region.size_coarse.z() > lbm.lat.global.z())
			reject(
				fmt::format(
					"region extends beyond the global coarsest-level domain [{},{},{}]", lbm.lat.global.x(), lbm.lat.global.y(), lbm.lat.global.z()
				)
			);

		// 2:1 balance: block boundaries must align with the parent level's coarse cells
		// (trivially satisfied at level 1, but enforced for future multi-level support)
		const idx alignment = idx(1) << (region.level - 1);
		if (region.origin_coarse.x() % alignment != 0 || region.origin_coarse.y() % alignment != 0 || region.origin_coarse.z() % alignment != 0
			|| region.size_coarse.x() % alignment != 0 || region.size_coarse.y() % alignment != 0 || region.size_coarse.z() % alignment != 0)
			reject(fmt::format("region origin and size must be multiples of {} (parent-level coarse cells)", alignment));

		// v1 scope guard: static single-hop refinement only (modernize to multi-level nesting in future)
		if (region.level > 1)
			reject("only a single refinement level (level == 1) is supported");
	}

	// ---------------------------------------------------------------------------
	// phase 2: create blocks (only reached when every region is valid)
	// ---------------------------------------------------------------------------

	// level_block_counts is empty when LBM was built with a non-AMR constructor
	if (static_cast<int>(lbm.level_block_counts.size()) < lbm.max_level + 1)
		lbm.level_block_counts.resize(lbm.max_level + 1, 0);

	// fine blocks have no same-level MPI neighbors in v1 - the ghost layers are
	// filled by the inter-level coupling kernels, not by MPI synchronization
	// (mirrors findNeighbors(..., nproc=1, periodic=false): all directions map to -1)
	std::map<TNL::Containers::SyncDirection, int> neighbors;
	for (auto direction : TNL::Containers::NDArraySyncPatterns::D3Q27)
		neighbors[direction] = -1;

	for (const AMR_Region<CONFIG>& region : regions) {
		// 2^level refinement ratio: with 2:1 refinement the fine block doubles the region in every axis
		const idx ratio = idx(1) << region.level;
		// fine-level extent and origin in fine-level global coordinates
		const idx3d local_fine{ratio * region.size_coarse.x(), ratio * region.size_coarse.y(), ratio * region.size_coarse.z()};
		const idx3d offset_fine{ratio * region.origin_coarse.x(), ratio * region.origin_coarse.y(), ratio * region.origin_coarse.z()};
		// the block lives on the refined lattice, so its `global` is the refined coarsest-level size;
		// block arrays are sized by `global` and must cover [offset, offset + local) in fine coordinates
		const idx3d global_fine{ratio * lbm.lat.global.x(), ratio * lbm.lat.global.y(), ratio * lbm.lat.global.z()};

		// sequential block id for logging and MPI tag bookkeeping (the LBM ctors use rank only)
		const int block_id = static_cast<int>(lbm.blocks.size());
		lbm.blocks.emplace_back(lbm.communicator, global_fine, local_fine, offset_fine, lbm.lat, region.level, block_id);
		auto& block = lbm.blocks.back();

		// AMR bookkeeping: the ctor defaults global_offset to offset (fine coords), but the
		// interface coupling (Wave 3) matches interfaces in PARENT-level coordinates
		block.global_offset = region.origin_coarse;

		// per-region physical origin: the fine block's local origin must coincide with the
		// coarse cell at origin_coarse on the parent's physical lattice
		block.lat_local.physOrigin =
			lbm.lat.physOrigin + point_t(region.origin_coarse.x(), region.origin_coarse.y(), region.origin_coarse.z()) * lbm.lat.physDl;

		// CUDA streams and kernel launch extents (single rank: interior compute only)
		block.setLatticeDecomposition(TNL::Containers::NDArraySyncPatterns::D3Q27, neighbors, neighbors);

		// allocate dfs/dmap/dmacro with the fine block extents (including ghost layers)
		block.allocateHostData();
		block.allocateDeviceData();

		// interior map: all fine cells are regular fluid (the interface ghost cells are
		// overwritten by the coupling kernels; GEO_AMR_INTERFACE is a coarse-cell tag)
		block.resetMap(CONFIG::BC::GEO_FLUID);
		block.copyMapToDevice();

		// initialize fine DFs to equilibrium (setEquilibrium dispatches to
		// CONFIG::COLL::setEquilibriumLat); the caller overwrites its own initial condition
		block.setEquilibrium(0 /* rho */, 0, 0, 0);

		// maintain the multi-level bookkeeping consistent with the LBM constructor invariants
		lbm.level_block_counts[region.level]++;
		lbm.total_blocks += lbm.nproc;
	}
}
