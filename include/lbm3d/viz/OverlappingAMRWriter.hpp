#pragma once

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <memory>

#include "OverlappingAMRWriter.h"

template <typename TRAITS>
void OverlappingAMRWriter<TRAITS>::write_attr_i64x2(hid_t loc, const char* name, const std::int64_t value[2])
{
	const hsize_t dims[1] = {2};
	hid_t space = H5Screate_simple(1, dims, nullptr);
	if (space < 0)
		throw std::runtime_error(fmt::format("OverlappingAMRWriter: H5Screate_simple failed for attribute '{}'", name));

	hid_t attr = H5Acreate2(loc, name, H5T_NATIVE_INT64, space, H5P_DEFAULT, H5P_DEFAULT);
	if (attr < 0) {
		H5Sclose(space);
		throw std::runtime_error(fmt::format("OverlappingAMRWriter: H5Acreate2 failed for attribute '{}'", name));
	}
	if (H5Awrite(attr, H5T_NATIVE_INT64, value) < 0) {
		H5Aclose(attr);
		H5Sclose(space);
		throw std::runtime_error(fmt::format("OverlappingAMRWriter: H5Awrite failed for attribute '{}'", name));
	}

	H5Aclose(attr);
	H5Sclose(space);
}

template <typename TRAITS>
void OverlappingAMRWriter<TRAITS>::write_attr_str(hid_t loc, const char* name, const char* value)
{
	hid_t str_type = H5Tcopy(H5T_C_S1);
	if (str_type < 0)
		throw std::runtime_error(fmt::format("OverlappingAMRWriter: H5Tcopy failed for attribute '{}'", name));
	if (H5Tset_size(str_type, std::strlen(value) + 1) < 0 || H5Tset_strpad(str_type, H5T_STR_NULLTERM) < 0) {
		H5Tclose(str_type);
		throw std::runtime_error(fmt::format("OverlappingAMRWriter: H5Tset_size failed for attribute '{}'", name));
	}

	hid_t space = H5Screate(H5S_SCALAR);
	if (space < 0) {
		H5Tclose(str_type);
		throw std::runtime_error(fmt::format("OverlappingAMRWriter: H5Screate failed for attribute '{}'", name));
	}

	hid_t attr = H5Acreate2(loc, name, str_type, space, H5P_DEFAULT, H5P_DEFAULT);
	if (attr < 0) {
		H5Sclose(space);
		H5Tclose(str_type);
		throw std::runtime_error(fmt::format("OverlappingAMRWriter: H5Acreate2 failed for attribute '{}'", name));
	}
	if (H5Awrite(attr, str_type, value) < 0) {
		H5Aclose(attr);
		H5Sclose(space);
		H5Tclose(str_type);
		throw std::runtime_error(fmt::format("OverlappingAMRWriter: H5Awrite failed for attribute '{}'", name));
	}

	H5Aclose(attr);
	H5Sclose(space);
	H5Tclose(str_type);
}

template <typename TRAITS>
void OverlappingAMRWriter<TRAITS>::write_attr_f64x3(hid_t loc, const char* name, const double value[3])
{
	const hsize_t dims[1] = {3};
	hid_t space = H5Screate_simple(1, dims, nullptr);
	if (space < 0)
		throw std::runtime_error(fmt::format("OverlappingAMRWriter: H5Screate_simple failed for attribute '{}'", name));

	hid_t attr = H5Acreate2(loc, name, H5T_NATIVE_DOUBLE, space, H5P_DEFAULT, H5P_DEFAULT);
	if (attr < 0) {
		H5Sclose(space);
		throw std::runtime_error(fmt::format("OverlappingAMRWriter: H5Acreate2 failed for attribute '{}'", name));
	}
	if (H5Awrite(attr, H5T_NATIVE_DOUBLE, value) < 0) {
		H5Aclose(attr);
		H5Sclose(space);
		throw std::runtime_error(fmt::format("OverlappingAMRWriter: H5Awrite failed for attribute '{}'", name));
	}

	H5Aclose(attr);
	H5Sclose(space);
}

template <typename TRAITS>
void OverlappingAMRWriter<TRAITS>::write_dataset_i32(hid_t loc, const char* name, const std::int32_t* data, hsize_t rows, hsize_t cols)
{
	const hsize_t dims[2] = {rows, cols};
	hid_t space = H5Screate_simple(2, dims, nullptr);
	if (space < 0)
		throw std::runtime_error(fmt::format("OverlappingAMRWriter: H5Screate_simple failed for dataset '{}'", name));

	hid_t dataset = H5Dcreate2(loc, name, H5T_NATIVE_INT32, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	if (dataset < 0) {
		H5Sclose(space);
		throw std::runtime_error(fmt::format("OverlappingAMRWriter: H5Dcreate2 failed for dataset '{}'", name));
	}
	if (H5Dwrite(dataset, H5T_NATIVE_INT32, H5S_ALL, H5S_ALL, H5P_DEFAULT, data) < 0) {
		H5Dclose(dataset);
		H5Sclose(space);
		throw std::runtime_error(fmt::format("OverlappingAMRWriter: H5Dwrite failed for dataset '{}'", name));
	}

	H5Dclose(dataset);
	H5Sclose(space);
}

template <typename TRAITS>
void OverlappingAMRWriter<TRAITS>::write_dataset_f64(hid_t loc, const char* name, const double* data, hsize_t size)
{
	const hsize_t dims[1] = {size};
	hid_t space = H5Screate_simple(1, dims, nullptr);
	if (space < 0)
		throw std::runtime_error(fmt::format("OverlappingAMRWriter: H5Screate_simple failed for dataset '{}'", name));

	hid_t dataset = H5Dcreate2(loc, name, H5T_NATIVE_DOUBLE, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	if (dataset < 0) {
		H5Sclose(space);
		throw std::runtime_error(fmt::format("OverlappingAMRWriter: H5Dcreate2 failed for dataset '{}'", name));
	}
	if (H5Dwrite(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, data) < 0) {
		H5Dclose(dataset);
		H5Sclose(space);
		throw std::runtime_error(fmt::format("OverlappingAMRWriter: H5Dwrite failed for dataset '{}'", name));
	}

	H5Dclose(dataset);
	H5Sclose(space);
}

template <typename TRAITS>
void OverlappingAMRWriter<TRAITS>::write_dataset_u8(hid_t loc, const char* name, const std::uint8_t* data, hsize_t size)
{
	const hsize_t dims[1] = {size};
	hid_t space = H5Screate_simple(1, dims, nullptr);
	if (space < 0)
		throw std::runtime_error(fmt::format("OverlappingAMRWriter: H5Screate_simple failed for dataset '{}'", name));

	hid_t dataset = H5Dcreate2(loc, name, H5T_NATIVE_UINT8, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	if (dataset < 0) {
		H5Sclose(space);
		throw std::runtime_error(fmt::format("OverlappingAMRWriter: H5Dcreate2 failed for dataset '{}'", name));
	}
	if (H5Dwrite(dataset, H5T_NATIVE_UINT8, H5S_ALL, H5S_ALL, H5P_DEFAULT, data) < 0) {
		H5Dclose(dataset);
		H5Sclose(space);
		throw std::runtime_error(fmt::format("OverlappingAMRWriter: H5Dwrite failed for dataset '{}'", name));
	}

	H5Dclose(dataset);
	H5Sclose(space);
}

template <typename TRAITS>
void OverlappingAMRWriter<TRAITS>::emitted_range(const idx3d& local, const idx3d& overlap, idx3d& e_lo, idx3d& e_hi)
{
	for (int a = 0; a < 3; a++) {
		const idx o = overlap[a];
		const idx ext = local[a] + 2 * o;
		// storage rows beyond the footprint coverage [offset - 1, offset +
		// local + 1) (with overlap 2: exactly one outer row per face) are
		// dropped -- emitting them would overhang the fine AMRBox into the
		// coarse interface ring that the reader then blanks, while the
		// covering fine rows would be HIDDENCELL
		e_lo[a] = o > 1 ? o - 1 : 0;
		e_hi[a] = o > 1 ? o + local[a] + 1 : ext;
	}
}

template <typename TRAITS>
template <typename CONFIG>
void OverlappingAMRWriter<TRAITS>::write(const std::string& filename, const LBM<CONFIG>& lbm, [[maybe_unused]] real time)
{
	using BLOCK = typename LBM<CONFIG>::BLOCK;
	using MACRO = typename LBM<CONFIG>::MACRO;

	// v1: single-rank output only (the blocks of all ranks would have to be
	// gathered otherwise)
	if (lbm.nproc != 1)
		throw std::runtime_error(fmt::format("OverlappingAMRWriter: multi-rank not supported in v1 (nproc={})", lbm.nproc));

	// group blocks by refinement level (std::map iterates levels in ascending order)
	std::map<int, std::vector<const BLOCK*>> levels;
	for (const BLOCK& block : lbm.blocks)
		levels[block.level].push_back(&block);

	// refresh the host mirrors of the macroscopic quantities once per block
	// (device -> host copy is logically const: it does not change the
	// simulation state, only updates the host-side view, hence the const_cast
	// on the const-qualified block references)
	// NOTE (Wave 4): the values are NOT recomputed here - with
	// MACRO::compute_in_each_iteration == false they may be stale
	for (const BLOCK& block : lbm.blocks)
		const_cast<BLOCK&>(block).copyMacroToHost();

	// refresh the host mirror of the geometry map (tiny, always written as
	// the "map" field for masking in ParaView)
	for (const BLOCK& block : lbm.blocks)
		const_cast<BLOCK&>(block).copyMapToHost();

	hid_t file = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
	if (file < 0)
		throw std::runtime_error(fmt::format("OverlappingAMRWriter: H5Fcreate failed for '{}'", filename));

	hid_t root = H5Gcreate2(file, "VTKHDF", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
	if (root < 0) {
		H5Fclose(file);
		throw std::runtime_error(fmt::format("OverlappingAMRWriter: H5Gcreate2 failed for group 'VTKHDF' in '{}'", filename));
	}

	// root attributes (Version [2,8] as of VTK 9.6+)
	const std::int64_t version[2] = {2, 8};
	write_attr_i64x2(root, "Version", version);
	write_attr_str(root, "Type", "OverlappingAMR");
	const point_t phys_origin = lbm.lat.lbm2physPoint(idx(0), idx(0), idx(0));
	const double origin[3] = {static_cast<double>(phys_origin.x()), static_cast<double>(phys_origin.y()), static_cast<double>(phys_origin.z())};
	write_attr_f64x3(root, "Origin", origin);

	// scalar macroscopic quantities emitted per level (in this order)
	const std::pair<std::uint8_t, const char*> variables[] = {
		{MACRO::e_rho, "rho"},
		{MACRO::e_vx, "vx"},
		{MACRO::e_vy, "vy"},
		{MACRO::e_vz, "vz"},
	};

	for (const auto& [level, blocks] : levels) {
		const std::string level_name = "Level" + std::to_string(level);
		hid_t level_group = H5Gcreate2(root, level_name.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
		if (level_group < 0) {
			H5Gclose(root);
			H5Fclose(file);
			throw std::runtime_error(fmt::format("OverlappingAMRWriter: H5Gcreate2 failed for group '{}' in '{}'", level_name, filename));
		}

		// the spatial step halves per refinement level (2:1 ratio)
		const real dl = lbm.lat.physDl / real(1 << level);
		const double spacing[3] = {static_cast<double>(dl), static_cast<double>(dl), static_cast<double>(dl)};
		write_attr_f64x3(level_group, "Spacing", spacing);

		// AMRBox: one inclusive [lo,hi] box per block in the level's own
		// lattice coordinates, row = {lo_x, hi_x, lo_y, hi_y, lo_z, hi_z}.
		// Fine-level blocks include the ghost rows inside the parent
		// footprint coverage (C2F-filled, valid after each cycle) so the
		// box covers exactly the footprint's fine cells -- the reader
		// blanks coarser cells overlapped by this box, which then
		// reproduces the REFINEDCELL footprint with no interface-ring band.
		std::vector<std::int32_t> amr_box;
		amr_box.reserve(blocks.size() * 6);
		for (const BLOCK* block : blocks) {
			const idx3d ovl{
				const_cast<BLOCK*>(block)->df_overlap_X(), const_cast<BLOCK*>(block)->df_overlap_Y(), const_cast<BLOCK*>(block)->df_overlap_Z()
			};
			idx3d e_lo, e_hi;
			emitted_range(block->local, ovl, e_lo, e_hi);
			const idx3d lo = block->offset - ovl + e_lo;
			const idx3d hi = block->offset - ovl + e_hi - 1;
			amr_box.push_back(static_cast<std::int32_t>(lo.x()));
			amr_box.push_back(static_cast<std::int32_t>(hi.x()));
			amr_box.push_back(static_cast<std::int32_t>(lo.y()));
			amr_box.push_back(static_cast<std::int32_t>(hi.y()));
			amr_box.push_back(static_cast<std::int32_t>(lo.z()));
			amr_box.push_back(static_cast<std::int32_t>(hi.z()));
		}
		write_dataset_i32(level_group, "AMRBox", amr_box.data(), blocks.size(), 6);

		hid_t cell_data = H5Gcreate2(level_group, "CellData", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
		if (cell_data < 0) {
			H5Gclose(level_group);
			H5Gclose(root);
			H5Fclose(file);
			throw std::runtime_error(fmt::format("OverlappingAMRWriter: H5Gcreate2 failed for group '{}/CellData' in '{}'", level_name, filename));
		}

		// per-block packing descriptor: the emitted range plus the block's
		// fixed cell offset in the concatenated per-level buffers (see the
		// byte-identity contract of the parallel packing loop below)
		struct EmittedBlock
		{
			const BLOCK* block;
			idx3d ovl, e_lo, e_hi;
			std::size_t cell_offset;
		};
		std::vector<EmittedBlock> emitted_blocks;
		emitted_blocks.reserve(blocks.size());
		std::size_t total_cells = 0;
		for (const BLOCK* block : blocks) {
			const idx3d ovl{
				const_cast<BLOCK*>(block)->df_overlap_X(), const_cast<BLOCK*>(block)->df_overlap_Y(), const_cast<BLOCK*>(block)->df_overlap_Z()
			};
			idx3d e_lo, e_hi;
			emitted_range(block->local, ovl, e_lo, e_hi);
			const idx3d e_ext = e_hi - e_lo;
			emitted_blocks.push_back({block, ovl, e_lo, e_hi, total_cells});
			total_cells += static_cast<std::size_t>(e_ext.x()) * static_cast<std::size_t>(e_ext.y()) * static_cast<std::size_t>(e_ext.z());
		}

		// concatenated per-level packing buffers, one per emitted field
		// (default-initialized: every element is written by the packing
		// loop; the page faults then land on the worker threads)
		const std::unique_ptr<double[]> var_buffers[4] = {
			std::unique_ptr<double[]>(new double[total_cells]),
			std::unique_ptr<double[]>(new double[total_cells]),
			std::unique_ptr<double[]>(new double[total_cells]),
			std::unique_ptr<double[]>(new double[total_cells]),
		};
		const std::unique_ptr<std::int32_t[]> map_buffer(new std::int32_t[total_cells]);
		const std::unique_ptr<std::uint8_t[]> ghost(new std::uint8_t[total_cells]);

		// packing tasks: variants 0..3 are the scalar macroscopic fields,
		// 4 is the geometry map, 5 is vtkGhostType; each (variant, block)
		// pair is split into fixed-size chunks of the block's emitted cell
		// order z*ny*nx + y*nx + x (x fastest)
		struct PackTask
		{
			int variant;
			const EmittedBlock* target;
			std::size_t begin, end;
		};
		std::vector<PackTask> tasks;
		for (int variant = 0; variant < 6; variant++)
			for (const EmittedBlock& emitted : emitted_blocks) {
				const idx3d e_ext = emitted.e_hi - emitted.e_lo;
				const std::size_t block_cells =
					static_cast<std::size_t>(e_ext.x()) * static_cast<std::size_t>(e_ext.y()) * static_cast<std::size_t>(e_ext.z());
				// the chunk size is a compile-time constant so the slice
				// boundaries depend only on the data sizes -- never on the
				// runtime thread count
				for (std::size_t begin = 0; begin < block_cells; begin += pack_chunk_cells)
					tasks.push_back({variant, &emitted, begin, std::min(begin + pack_chunk_cells, block_cells)});
			}

		// BYTE-IDENTITY CONTRACT of the parallel packing: every emitted cell
		// is a pure function of its (variant, block, z, y, x) coordinates --
		// a wholesale read of hmacro/hmap or the ghost-tag classification,
		// whose fine-footprint scan runs in lbm.blocks order per cell with
		// no cross-cell state -- and each task writes a disjoint,
		// data-determined slice of the final per-level buffers at offsets
		// fixed by EmittedBlock::cell_offset + the flat emitted index. The
		// buffer contents therefore cannot depend on the thread count, the
		// schedule, or the task completion order. All HDF5 calls stay OUT of
		// this region on the calling thread (the raw HDF5 C API is not
		// thread-safe) and issue the same sequence of creates/writes with
		// the same shapes and contents as the serial packing did
		const int level_id = level;
#pragma omp parallel for schedule(dynamic, 1) default(none) shared(tasks, var_buffers, map_buffer, ghost, variables, lbm, level_id)
		for (std::ptrdiff_t t = 0; t < static_cast<std::ptrdiff_t>(tasks.size()); t++) {
			const PackTask& task = tasks[t];
			const EmittedBlock& emitted = *task.target;
			const BLOCK& block = *emitted.block;
			const idx3d& ovl = emitted.ovl;
			const idx nx = emitted.e_hi.x() - emitted.e_lo.x();
			const idx ny = emitted.e_hi.y() - emitted.e_lo.y();
			const idx3d storage_ext = block.local + idx3d{2 * ovl.x(), 2 * ovl.y(), 2 * ovl.z()};

			// decode the chunk's begin offset into emitted (z, y, x)
			// coordinates, then walk the cell order with O(1) increments
			idx ez = emitted.e_lo.z() + static_cast<idx>(task.begin) / (ny * nx);
			const idx rem = static_cast<idx>(task.begin) % (ny * nx);
			idx ey = emitted.e_lo.y() + rem / nx;
			idx ex = emitted.e_lo.x() + rem % nx;
			std::size_t i = task.begin;
			while (i < task.end) {
				const std::size_t x_left = static_cast<std::size_t>(emitted.e_hi.x() - ex);
				const std::size_t run = std::min(x_left, task.end - i);
				for (std::size_t r = 0; r < run; r++, ex++, i++) {
					const std::size_t out = emitted.cell_offset + i;
					const idx gx = block.offset.x() - ovl.x() + ex;
					const idx gy = block.offset.y() - ovl.y() + ey;
					const idx gz = block.offset.z() - ovl.z() + ez;
					if (task.variant < 4) {
						var_buffers[task.variant][out] = static_cast<double>(block.hmacro(variables[task.variant].first, gx, gy, gz));
					}
					else if (task.variant == 4) {
						map_buffer[out] = static_cast<std::int32_t>(block.hmap(gx, gy, gz));
					}
					else {
						// vtkGhostType: coarse cells covered by a finer-level
						// block are REFINEDCELL(4); a finer block's footprint
						// in the parent (this level's) lattice is
						// [global_offset, global_offset + (local + 2)/2) -- the
						// re-anchored interior local = 2*size - 2 is inset one
						// fine cell per face, so the +2 restores the full
						// requested footprint
						const idx3d cell{gx, gy, gz};
						std::uint8_t tag = vtk_visible;
						const bool is_ghost =
							(ex < ovl.x() || ex >= storage_ext.x() - ovl.x() || ey < ovl.y() || ey >= storage_ext.y() - ovl.y() || ez < ovl.z()
							 || ez >= storage_ext.z() - ovl.z());
						if (is_ghost) {
							// ghost rows within the coarse footprint [offset-1, offset+local+1)
							// in fine coords are valid C2F data — visible; rows beyond the
							// footprint are no longer emitted (see \ref emitted_range), but
							// keep the classification so the tag stays honest if the emitted
							// range is ever widened
							const idx3d fp_lo = block.offset - idx3d{1, 1, 1};
							const idx3d fp_hi = block.offset + block.local + idx3d{1, 1, 1};
							if (cell.x() >= fp_lo.x() && cell.x() < fp_hi.x() && cell.y() >= fp_lo.y() && cell.y() < fp_hi.y()
								&& cell.z() >= fp_lo.z() && cell.z() < fp_hi.z())
								tag = vtk_visible;
							else
								tag = vtk_hidden_cell;
						}
						for (const BLOCK& fine : lbm.blocks) {
							if (fine.level != level_id + 1)
								continue;
							const idx3d fp_lo = fine.global_offset;
							const idx3d fp_size{(fine.local.x() + 2) / 2, (fine.local.y() + 2) / 2, (fine.local.z() + 2) / 2};
							if (cell.x() >= fp_lo.x() && cell.x() < fp_lo.x() + fp_size.x() && cell.y() >= fp_lo.y()
								&& cell.y() < fp_lo.y() + fp_size.y() && cell.z() >= fp_lo.z() && cell.z() < fp_lo.z() + fp_size.z())
							{
								tag = vtk_refined_cell;
								break;
							}
						}
						ghost[out] = tag;
					}
				}
				ex = emitted.e_lo.x();
				if (++ey == emitted.e_hi.y()) {
					ey = emitted.e_lo.y();
					++ez;
				}
			}
		}

		for (int variant = 0; variant < 4; variant++)
			write_dataset_f64(cell_data, variables[variant].second, var_buffers[variant].get(), total_cells);
		// geometry map (int tags -- the "wall field" for masking in
		// ParaView; values are the BC::GEO_* enum) -- always written
		write_dataset_i32(cell_data, "map", map_buffer.get(), total_cells, 1);
		write_dataset_u8(cell_data, "vtkGhostType", ghost.get(), total_cells);

		H5Gclose(cell_data);
		H5Gclose(level_group);
	}

	H5Gclose(root);
	H5Fclose(file);

	spdlog::info("VTKHDF AMR output written: {}", filename);
}
