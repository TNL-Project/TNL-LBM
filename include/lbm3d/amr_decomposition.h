#pragma once

#include <algorithm>
#include <cstdint>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <fmt/core.h>
#include <spdlog/spdlog.h>

#include <TNL/Containers/Array.h>

#include "lbm.h"

/**
 * \brief Static AMR region declaration, validation and block instantiation.
 *
 * An \ref AMR_Region is a rectangular fine-lattice region specified in
 * coarsest-level (level-0) cell coordinates. With the standard 2:1
 * refinement ratio, each coarse cell of the region becomes 2x2x2 fine cells.
 * The fine block's interior does NOT span the whole requested footprint:
 * under the re-anchored band registration (Schönherr-ch7 ruling) the
 * outermost requested-footprint row becomes a coarse-authoritative ring row
 * (simulated), the covered, F2C-refilled destination row starts one coarse
 * row inside, and the fine-authoritative full coverage is the footprint
 * inset by 1 coarse row per face -- `offset_fine = 2 * origin_coarse + 1`,
 * `local_fine = 2 * size_coarse - 2` per axis.
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
 *
 * Band registration (Schönherr-ch7 ruling): the outermost requested-footprint
 * row becomes a coarse-authoritative ring row (simulated); the covered,
 * F2C-refilled destination row starts one coarse row inside; fine-authoritative
 * full coverage = footprint inset 1 coarse row per face. Concretely
 * `offset_fine = ratio * origin_coarse + 1` and
 * `local_fine = ratio * size_coarse - 2` per axis, so physical cell positions
 * (a fine-global-coordinate property) and every band row's (home, t) storage
 * parity are invariant under the shift -- the old local index equals the new
 * one plus one. The total extent requested from the allocator is restored to
 * `ratio * size + 2` per axis once the ghost overlap is 2 deep.
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

		// footprint minimum gs >= 3 per axis under the re-anchored band
		// registration: with the interior inset one fine cell per face, a
		// 2-coarse-cell-thin axis would give the outermost (c=0) ring row of
		// one face and the depth-1 (c=1) F2C destination row of the opposite
		// face to the same cell -- a dual-role row the band structure does
		// not admit -- and the fine interior (local 2 for gs = 2) would also
		// underflow the F2C filter's 4-node window
		if (region.size_coarse.x() < 3)
			reject(
				fmt::format(
					"AMR footprint size below the 3-coarse-cell minimum required by the interface band structure (distinct c=0 ring and c=1 "
					"destination rows) on axis X (got {})",
					region.size_coarse.x()
				)
			);
		if (region.size_coarse.y() < 3)
			reject(
				fmt::format(
					"AMR footprint size below the 3-coarse-cell minimum required by the interface band structure (distinct c=0 ring and c=1 "
					"destination rows) on axis Y (got {})",
					region.size_coarse.y()
				)
			);
		if (region.size_coarse.z() < 3)
			reject(
				fmt::format(
					"AMR footprint size below the 3-coarse-cell minimum required by the interface band structure (distinct c=0 ring and c=1 "
					"destination rows) on axis Z (got {})",
					region.size_coarse.z()
				)
			);

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
		// fine-level extent and origin in fine-level global coordinates,
		// re-anchored one fine cell inward per footprint face (Schönherr-ch7
		// band registration ruling): the outermost row of each face is a
		// coarse-authoritative simulated ring row and the F2C-refilled
		// destination rows shift one coarse row inside. Physical cell
		// positions are a fine-global-coordinate property (invariant under
		// the anchor shift: old local index = new local index + 1), and every
		// band row keeps its storage parity.
		const idx3d local_fine{ratio * region.size_coarse.x() - 2, ratio * region.size_coarse.y() - 2, ratio * region.size_coarse.z() - 2};
		const idx3d offset_fine{ratio * region.origin_coarse.x() + 1, ratio * region.origin_coarse.y() + 1, ratio * region.origin_coarse.z() + 1};
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

		// per-region physical origin: initLevelLattice already set
		// lat_local.physOrigin to the correct fine-level global origin
		// (adjusted for the cell-centered convention); add the block's
		// offset in fine coordinates
		block.lat_local.physOrigin += point_t(block.offset.x(), block.offset.y(), block.offset.z()) * block.lat_local.physDl;

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

// D3Q27 lattice velocity vectors c_q = (cx, cy, cz), matching the direction
// enum in lbm3d/defs.h (zzz = 0, pzz = 1, mzz = 2, ...)
inline constexpr int amr_d3q27_directions[27][3] = {
	{0, 0, 0},	   // zzz
	{1, 0, 0},	   // pzz
	{-1, 0, 0},	   // mzz
	{0, 1, 0},	   // zpz
	{0, -1, 0},	   // zmz
	{0, 0, 1},	   // zzp
	{0, 0, -1},	   // zzm
	{1, 1, 0},	   // ppz
	{-1, -1, 0},   // mmz
	{1, -1, 0},	   // pmz
	{-1, 1, 0},	   // mpz
	{1, 0, 1},	   // pzp
	{-1, 0, -1},   // mzm
	{1, 0, -1},	   // pzm
	{-1, 0, 1},	   // mzp
	{0, 1, 1},	   // zpp
	{0, -1, -1},   // zmm
	{0, 1, -1},	   // zpm
	{0, -1, 1},	   // zmp
	{1, 1, 1},	   // ppp
	{-1, -1, -1},  // mmm
	{1, 1, -1},	   // ppm
	{-1, -1, 1},   // mmp
	{1, -1, 1},	   // pmp
	{-1, 1, -1},   // mpm
	{1, -1, -1},   // pmm
	{-1, 1, 1},	   // mpp
};

// Host+device storage backing LBM_Data::dinterface_dir for one block. The
// arrays are addressed by the block's kernel indexer storage index
// (`data.indexer.getStorageIndex(x, y, z)`, flat storage size `data.XYZ`).
template <typename CONFIG>
struct AMR_InterfaceDirStorage
{
	using TRAITS = typename CONFIG::TRAITS;
	TNL::Containers::Array<std::uint32_t, TNL::Devices::Host> host;
	TNL::Containers::Array<std::uint32_t, DeviceType> device;
};

// The storage cannot be an LBM_BLOCK member because blocks are move-constructed
// during `lbm.blocks` growth while `data.dinterface_dir` must keep pointing at
// the same allocation; keys are raw block addresses, stable because
// markAMRInterface is called only after all blocks exist. An entry left behind
// by a destroyed LBM instance at a reused address is harmless: a new block
// starts with dinterface_dir == nullptr and the allocate path overwrites the
// entry.
template <typename CONFIG>
std::map<LBM_BLOCK<CONFIG>*, AMR_InterfaceDirStorage<CONFIG>>& amrInterfaceDirRegistry()
{
	static std::map<LBM_BLOCK<CONFIG>*, AMR_InterfaceDirStorage<CONFIG>> registry;
	return registry;
}

/**
 * \brief Allocate and zero the interface-direction bitmask of `block` and wire
 * it into `block.data.dinterface_dir`.
 *
 * Idempotent (a block with a non-null `data.dinterface_dir` is untouched).
 * Blocks without GEO_AMR_INTERFACE cells never get the array, so the
 * nullptr check in `D3Q27_BC_All::getInterfaceDir` keeps non-AMR runs fast.
 */
template <typename CONFIG>
void allocateInterfaceDirArray(LBM_BLOCK<CONFIG>& block)
{
	auto& registry = amrInterfaceDirRegistry<CONFIG>();
	if (block.data.dinterface_dir == nullptr) {
		auto& arrays = registry[&block];
		arrays.host.setSize(block.data.XYZ);
		arrays.host.setValue(0);
		arrays.device.setSize(block.data.XYZ);
		arrays.device.setValue(0);
		block.data.dinterface_dir = arrays.device.getData();
	}
}

/**
 * \brief Tag coarse cells adjacent to fine blocks as GEO_AMR_INTERFACE and
 * fill their interface-direction bitmasks.
 *
 * Every fine block's footprint is projected onto its parent level as the
 * rectangle [global_offset, global_offset + (local + 2)/2) in parent-level
 * coordinates (global_offset is the parent-level origin set by
 * createAMRBlocks, consecutive levels always have a 2:1 ratio; the +2
 * recovers the inset interior under the re-anchored indexer, local =
 * 2*size - 2). Two populations are tagged:
 *
 * - **Interface ring**: every parent-level cell within Chebyshev distance 1
 *   of the rectangle but outside it (the halo row c=-1), PLUS the
 *   rectangle's own surface shell (the reactivated ring row c=0), is tagged
 *   GEO_AMR_INTERFACE and its bitmask records the D3Q27 directions (bit q
 *   matching the direction enum in defs.h) whose neighbor cell lies inside
 *   the rectangle - the directions pointing INTO the fine region. Both
 *   rows are collision-active (driven by the coarse collide-and-stream
 *   kernel) and serve as the coarse-to-fine source pair {c=-1, c=0} of
 *   the contract band map (docs/AMR-schonherr-ch7-target-contract.md
 *   sec. 2.1); they are never fine-to-coarse written.
 *
 * - **Hidden (frozen) cells**: every parent-level cell INSIDE the rectangle
 *   at depth >= 1 (c >= 1: the c=1 skin destination row and the c>=2
 *   never-read deep core) is tagged GEO_NOTHING. These cells are "hidden"
 *   under the fine footprint — the fine lattice is the authoritative
 *   solution there. They do not stream or collide (GEO_NOTHING:
 *   preCollision/postCollision early return, doCollision returns false).
 *   The skin row's DFs are set exclusively by the interior fine-to-coarse
 *   transfer at the end of each coarse cycle, which injects
 *   Lagrava-filtered fine-averaged DFs into exactly this depth-1 shell
 *   (the deep core is never written). This eliminates the "shadow solve"
 *   (a diverging coarse-evolved state under the footprint that would
 *   corrupt the fine boundary via C2F interpolation — see
 *   docs/AMR-for-LBM-implementation.md §9.2.1).
 *
 * Only GEO_FLUID cells are re-tagged: physical boundary conditions (walls,
 * inflows, ...) survive where a fine region touches the domain boundary, and
 * bits accumulate where two fine regions are adjacent to the same cell.
 *
 * v1 limitations: single MPI rank (same as createAMRBlocks), periodic
 * wrap-around interfaces are not handled. Must be called AFTER all fine
 * blocks exist and after the coarse boundary conditions were set on hmap;
 * uploads the updated hmap and bitmasks to the device before returning.
 */
template <typename CONFIG>
void markAMRInterface(LBM<CONFIG>& lbm)
{
	using BC = typename CONFIG::BC;
	using idx = typename CONFIG::TRAITS::idx;
	using idx3d = typename CONFIG::TRAITS::idx3d;
	using BLOCK = LBM_BLOCK<CONFIG>;

	// blocks whose map and bitmask need a host -> device upload at the end
	std::vector<BLOCK*> dirty_blocks;

	for (auto& fine : lbm.blocks) {
		if (fine.level <= 0)
			continue;
		const idx3d origin = fine.global_offset;
		// fine footprint extent in parent-level cells (2:1 ratio): under the
		// re-anchored indexer the interior is local = 2*size - 2 (inset one
		// fine cell per face), so the footprint is (local + 2)/2, exact
		const idx3d size{(fine.local.x() + 2) / 2, (fine.local.y() + 2) / 2, (fine.local.z() + 2) / 2};
		const int parent_level = fine.level - 1;
		int marked_fine = 0;
		int frozen_fine = 0;

		for (auto& coarse : lbm.blocks) {
			if (coarse.level != parent_level)
				continue;

			// 1-cell halo around the fine footprint, clipped to this coarse
			// block's local range (global parent-level coordinates); a halo
			// spanning multiple coarse blocks is handled by the other
			// (fine, coarse) pair iterations
			const idx x_begin = std::max(coarse.offset.x(), origin.x() - 1);
			const idx x_end = std::min(coarse.offset.x() + coarse.local.x(), origin.x() + size.x() + 1);
			const idx y_begin = std::max(coarse.offset.y(), origin.y() - 1);
			const idx y_end = std::min(coarse.offset.y() + coarse.local.y(), origin.y() + size.y() + 1);
			const idx z_begin = std::max(coarse.offset.z(), origin.z() - 1);
			const idx z_end = std::min(coarse.offset.z() + coarse.local.z(), origin.z() + size.z() + 1);

			int marked = 0;
			int frozen = 0;
			for (idx x = x_begin; x < x_end; x++) {
				for (idx y = y_begin; y < y_end; y++) {
					for (idx z = z_begin; z < z_end; z++) {
						// distance-band tag rule of the contract band map
						// (docs/AMR-schonherr-ch7-target-contract.md sec. 2.1):
						// the footprint surface shell c=0 (inside the rectangle
						// but not inside its 1-cell inset) is the REACTIVATED
						// second ring row, tagged GEO_AMR_INTERFACE like the
						// halo below -- it falls through to the bitmask + guard
						// path, so it stays collision-active and serves as the
						// C2F source line 2; deeper covered cells (c >= 1: the
						// c=1 skin destination row plus the never-read deep
						// core) freeze to GEO_NOTHING (no stream/collide,
						// preventing the diverging shadow solve) and the c=1
						// skin is F2C-injected with fine-averaged DFs each cycle
						const bool inside = x >= origin.x() && x < origin.x() + size.x() && y >= origin.y() && y < origin.y() + size.y()
										 && z >= origin.z() && z < origin.z() + size.z();
						if (inside) {
							const bool surface = x == origin.x() || x == origin.x() + size.x() - 1 || y == origin.y()
											  || y == origin.y() + size.y() - 1 || z == origin.z() || z == origin.z() + size.z() - 1;
							if (! surface) {
								if (coarse.hmap(x, y, z) == BC::GEO_FLUID) {
									coarse.setMap(x, y, z, BC::GEO_NOTHING);
									frozen++;
								}
								continue;
							}
						}

						// directions whose neighbor cell is inside the footprint
						std::uint32_t mask = 0;
						for (int q = 1; q < 27; q++) {
							const idx nx = x + amr_d3q27_directions[q][0];
							const idx ny = y + amr_d3q27_directions[q][1];
							const idx nz = z + amr_d3q27_directions[q][2];
							const bool crosses = nx >= origin.x() && nx < origin.x() + size.x() && ny >= origin.y() && ny < origin.y() + size.y()
											  && nz >= origin.z() && nz < origin.z() + size.z();
							if (crosses)
								mask |= std::uint32_t(1) << q;
						}
						if (mask == 0)
							continue;

						const auto map_value = coarse.hmap(x, y, z);
						const bool already_interface = map_value == BC::GEO_AMR_INTERFACE;
						// do not clobber physical boundary conditions
						if (! already_interface && ! BC::isFluid(map_value))
							continue;

						allocateInterfaceDirArray(coarse);
						auto& arrays = amrInterfaceDirRegistry<CONFIG>().at(&coarse);
						arrays.host[coarse.data.indexer.getStorageIndex(x, y, z)] |= mask;

						if (! already_interface) {
							coarse.setMap(x, y, z, BC::GEO_AMR_INTERFACE);
							marked++;
						}
					}
				}
			}

			if (marked > 0 || frozen > 0) {
				marked_fine += marked;
				frozen_fine += frozen;
				if (std::find(dirty_blocks.begin(), dirty_blocks.end(), &coarse) == dirty_blocks.end())
					dirty_blocks.push_back(&coarse);
			}
		}

		if (marked_fine > 0 || frozen_fine > 0)
			spdlog::info(
				"markAMRInterface: fine block {} (level {}) added {} interface cells, froze {} hidden cells on level {}",
				fine.id,
				fine.level,
				marked_fine,
				frozen_fine,
				parent_level
			);
	}

	for (auto* coarse : dirty_blocks) {
		coarse->copyMapToDevice();
		auto it = amrInterfaceDirRegistry<CONFIG>().find(coarse);
		if (it != amrInterfaceDirRegistry<CONFIG>().end())
			it->second.device = it->second.host;
	}
}
