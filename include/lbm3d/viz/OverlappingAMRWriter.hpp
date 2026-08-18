#pragma once

#include <cstring>

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
template <typename CONFIG>
void OverlappingAMRWriter<TRAITS>::write(const std::string& filename, const LBM<CONFIG>& lbm, [[maybe_unused]] real time, bool write_dfs)
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

	// refresh the host mirrors of the distribution functions only when the
	// raw-DF debug fields are requested (27 fields per frame per level)
	if (write_dfs)
		for (const BLOCK& block : lbm.blocks)
			const_cast<BLOCK&>(block).copyDFsToHost();

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
		// lattice coordinates, row = {lo_x, hi_x, lo_y, hi_y, lo_z, hi_z}
		std::vector<std::int32_t> amr_box;
		amr_box.reserve(blocks.size() * 6);
		for (const BLOCK* block : blocks) {
			const idx3d lo = block->offset;
			const idx3d hi = block->offset + block->local - 1;
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

		// total number of cells at this level (sum over all blocks)
		std::size_t total_cells = 0;
		for (const BLOCK* block : blocks)
			total_cells +=
				static_cast<std::size_t>(block->local.x()) * static_cast<std::size_t>(block->local.y()) * static_cast<std::size_t>(block->local.z());

		// concatenated scalar fields, cell order z*ny*nx + y*nx + x (x fastest)
		std::vector<double> buffer;
		buffer.reserve(total_cells);
		for (const auto& [quantity, name] : variables) {
			buffer.clear();
			for (const BLOCK* block : blocks)
				for (idx z = 0; z < block->local.z(); z++)
					for (idx y = 0; y < block->local.y(); y++)
						for (idx x = 0; x < block->local.x(); x++)
							buffer.push_back(
								static_cast<double>(block->hmacro(quantity, block->offset.x() + x, block->offset.y() + y, block->offset.z() + z))
							);
			write_dataset_f64(cell_data, name, buffer.data(), buffer.size());
		}

		// geometry map (int tags -- the "wall field" for masking in
		// ParaView; values are the BC::GEO_* enum) -- always written
		std::vector<std::int32_t> map_buffer;
		map_buffer.reserve(total_cells);
		for (const BLOCK* block : blocks)
			for (idx z = 0; z < block->local.z(); z++)
				for (idx y = 0; y < block->local.y(); y++)
					for (idx x = 0; x < block->local.x(); x++)
						map_buffer.push_back(
							static_cast<std::int32_t>(block->hmap(block->offset.x() + x, block->offset.y() + y, block->offset.z() + z))
						);
		write_dataset_i32(cell_data, "map", map_buffer.data(), map_buffer.size(), 1);

		// raw distribution functions (only when requested): one f64
		// dataset per direction, named f00..f{Q-1} after the defs.h
		// enumeration; values are the df_cur buffer in the current
		// streaming orientation (AB: natural; AA: parity-twisted)
		if (write_dfs)
			for (int q = 0; q < CONFIG::Q; q++) {
				buffer.clear();
				for (const BLOCK* block : blocks)
					for (idx z = 0; z < block->local.z(); z++)
						for (idx y = 0; y < block->local.y(); y++)
							for (idx x = 0; x < block->local.x(); x++)
								buffer.push_back(
									static_cast<double>(block->hfs[df_cur](q, block->offset.x() + x, block->offset.y() + y, block->offset.z() + z))
								);
				const std::string name = fmt::format("f{:02d}", q);
				write_dataset_f64(cell_data, name.c_str(), buffer.data(), buffer.size());
			}

		// vtkGhostType: mark coarse cells that are covered by a finer-level
		// block as REFINEDCELL(4); a finer block's footprint in the parent
		// (this level's) lattice is [global_offset, global_offset + local/2)
		std::vector<std::uint8_t> ghost;
		ghost.reserve(total_cells);
		for (const BLOCK* block : blocks) {
			for (idx z = 0; z < block->local.z(); z++)
				for (idx y = 0; y < block->local.y(); y++)
					for (idx x = 0; x < block->local.x(); x++) {
						const idx3d cell{block->offset.x() + x, block->offset.y() + y, block->offset.z() + z};
						std::uint8_t tag = vtk_visible;
						for (const BLOCK& fine : lbm.blocks) {
							if (fine.level != level + 1)
								continue;
							const idx3d fp_lo = fine.global_offset;
							const idx3d fp_size{fine.local.x() / 2, fine.local.y() / 2, fine.local.z() / 2};
							if (cell.x() >= fp_lo.x() && cell.x() < fp_lo.x() + fp_size.x() && cell.y() >= fp_lo.y()
								&& cell.y() < fp_lo.y() + fp_size.y() && cell.z() >= fp_lo.z() && cell.z() < fp_lo.z() + fp_size.z())
							{
								tag = vtk_refined_cell;
								break;
							}
						}
						ghost.push_back(tag);
					}
		}
		write_dataset_u8(cell_data, "vtkGhostType", ghost.data(), ghost.size());

		H5Gclose(cell_data);
		H5Gclose(level_group);
	}

	H5Gclose(root);
	H5Fclose(file);

	spdlog::info("VTKHDF AMR output written: {}", filename);
}
