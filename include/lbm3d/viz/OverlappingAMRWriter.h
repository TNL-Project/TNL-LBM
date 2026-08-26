#pragma once

#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include <hdf5.h>

#include <fmt/format.h>
#include <spdlog/spdlog.h>

#include "../lbm.h"

/**
 * \brief Standalone VTKHDF OverlappingAMR writer (single-file HDF5 output).
 *
 * Writes the block-decomposed AMR lattice of an \ref LBM instance to a single
 * HDF5 file following the canonical VTKHDF OverlappingAMR layout (FLAT
 * per-level groups, no per-block subgroups):
 *
 * ```
 * simulation.vtkhdf (single HDF5 file)
 * └── VTKHDF/                          (group)
 *     ├── Version    (attr, int64[2])  = [2, 8]
 *     ├── Type        (attr, string)   = "OverlappingAMR"
 *     ├── Origin      (attr, double[3])  <- GLOBAL origin = lbm.lat.lbm2physPoint(0,0,0)
 *     ├── Level0/                       (group)
 *     │   ├── Spacing   (attr, double[3]) = nse.lat.physDl
 *     │   ├── AMRBox    (dataset, int32[N_blocks x 6], inclusive [lo,hi] in level-0 lattice)
 *     │   └── CellData/                  (group)
 *     │       ├── rho            (dataset, double[sum of cells])
 *     │       ├── vx             (dataset, double[sum of cells])
 *     │       ├── vy             (dataset, double[sum of cells])
 *     │       ├── vz             (dataset, double[sum of cells])
 *     │       ├── map            (dataset, int32[sum of cells], BC::GEO_* tags)
 *     │       └── f00..f{Q-1}    (optional datasets, double[sum of cells])
 *     │       └── vtkGhostType   (dataset, uint8[sum of cells])  <- REQUIRED, 0=visible / 4=REFINEDCELL
 *     └── Level1/                       (group, same structure)
 * ```
 *
 * Layout notes:
 * - AMRBox rows are INCLUSIVE [lo,hi] boxes in the level's OWN lattice
 *   coordinates, stored as `{lo_x, hi_x, lo_y, hi_y, lo_z, hi_z}` per block.
 *   The box of a refinement-level block spans the interior PLUS the ghost
 *   rows inside the parent footprint coverage `[offset - 1, offset + local + 1)`
 *   (their C2F-filled values are valid), i.e. exactly the parent-level
 *   footprint's fine cells; ghost rows beyond the footprint are dropped.
 *   The reader blanks every coarser cell overlapped by a finer AMRBox, so
 *   an overhanging box would blank the coarse interface ring whose covering
 *   fine rows are HIDDENCELL -- a 0.5-coarse-cell white band around the
 *   patch (see \ref emitted_range).
 * - The reader computes refinement ratios from the per-level Spacing
 *   attributes; there are intentionally NO RefinementRatios, per-block
 *   Origin/Spacing or Dimensions datasets.
 * - Level groups are named "Level0", "Level1", ... (no underscore).
 * - `vtkGhostType` is REQUIRED (the reader does NOT auto-blank): 0 = visible,
 *   4 = REFINEDCELL (covered by a finer level), 8 = HIDDENCELL. This writer
 *   marks coarse cells covered by next-level block footprints as 4; all cells
 *   on the finest level are 0.
 * - CellData arrays are concatenated over all blocks at a level in AMRBox
 *   order; within a block the cell order is `z*ny*nx + y*nx + x` (x fastest,
 *   same as UniformDataWriter.hpp).
 *
 * v1 limitations:
 * - single MPI rank only (throws otherwise),
 * - no temporal `Steps/` metadata (the `time` argument is reserved for v2),
 * - the writer only copies device macros to the host (`copyMacroToHost`);
 *   it does NOT recompute macroscopic quantities. When
 *   `MACRO::compute_in_each_iteration == false` (e.g. D3Q27_MACRO_Default),
 *   the emitted values are whatever is currently stored in `hmacro` -
 *   recomputation before output is the caller's responsibility (Wave 4).
 */
template <typename TRAITS>
class OverlappingAMRWriter
{
public:
	using idx = typename TRAITS::idx;
	using real = typename TRAITS::real;
	using dreal = typename TRAITS::dreal;
	using idx3d = typename TRAITS::idx3d;
	using point_t = typename TRAITS::point_t;

	/**
	 * \brief Write the AMR lattice of \a lbm to the VTKHDF file \a filename.
	 *
	 * \param filename  path of the output file (overwritten if it exists);
	 *                  the caller composes per-step names such as
	 *                  `output_amr_{cycle:04d}.vtkhdf`
	 * \param lbm       the (single-rank) lattice manager holding the blocks
	 * \param time      physical time of the step (currently unused; temporal
	 *                  metadata is deferred to v2)
	 *
	 * \throws std::runtime_error when called with more than one MPI rank or
	 *         when any HDF5 operation fails.
	 */
	template <typename CONFIG>
	static void write(const std::string& filename, const LBM<CONFIG>& lbm, real time, bool write_dfs = false);

private:
	// vtkGhostType cell visibility tags (vtkDataSetAttributes::CellGhostTypes)
	static constexpr std::uint8_t vtk_visible = 0;		 // normal cell
	static constexpr std::uint8_t vtk_refined_cell = 4;	 // REFINEDCELL (refined, covered by a finer level)
	static constexpr std::uint8_t vtk_hidden_cell = 8;	 // HIDDENCELL

	/**
	 * \brief Emitted-cell range of one block in the block's extended storage
	 * coordinates `[0, ext)` with `ext = local + 2 * overlap` per axis.
	 *
	 * The range is the block interior plus the ghost rows that lie inside
	 * the parent-footprint coverage `[offset - 1, offset + local + 1)` in
	 * the block's global lattice coordinates (ghost rows beyond the
	 * footprint are storage-only and carry no coupling-valid data). For a
	 * refinement-level block (storage overlap 2) it drops exactly one
	 * outer row per face; for level-0 blocks (no overlap) it is the full
	 * storage. The resulting fine AMRBox is `[offset - 1, offset + local]`
	 * inclusive, exactly the parent footprint's fine cells, so the
	 * reader's overlap-blanking of the coarser level reproduces the
	 * writer's REFINEDCELL footprint exactly (no interface-ring band).
	 */
	static void emitted_range(const idx3d& local, const idx3d& overlap, idx3d& e_lo, idx3d& e_hi);

	// low-level HDF5 helpers - each throws std::runtime_error on failure
	static void write_attr_i64x2(hid_t loc, const char* name, const std::int64_t value[2]);
	static void write_attr_str(hid_t loc, const char* name, const char* value);
	static void write_attr_f64x3(hid_t loc, const char* name, const double value[3]);
	static void write_dataset_i32(hid_t loc, const char* name, const std::int32_t* data, hsize_t rows, hsize_t cols);
	static void write_dataset_f64(hid_t loc, const char* name, const double* data, hsize_t size);
	static void write_dataset_u8(hid_t loc, const char* name, const std::uint8_t* data, hsize_t size);
};

#include "OverlappingAMRWriter.hpp"
