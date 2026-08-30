// Unit test for the VTKHDF OverlappingAMR single-file writer
// (include/lbm3d/viz/OverlappingAMRWriter.h).
//
// The test builds a minimal two-level AMR setup on a 16^3 coarse lattice
// with one centered level-1 region (coarse footprint [4, 12)^3, i.e. a 14^3
// fine block at fine offset (9, 9, 9) under the re-anchored indexer; the
// writer emits the fine rows covering exactly the coarse footprint, i.e.
// [8, 24)^3 = 16^3 cells), initializes a uniform equilibrium
// state (rho = 1, v = 0) on both levels, writes the VTKHDF file with
// OverlappingAMRWriter and verifies the HDF5 structure against the writer's
// documented layout:
//
// - root group VTKHDF with attributes Version == [2, 8],
//   Type == "OverlappingAMR" and Origin == lat.lbm2physPoint(0, 0, 0),
// - per-level groups Level0/Level1 with attribute Spacing = physDl / 2^level,
//   an AMRBox dataset with one inclusive [lo, hi] row per block in the
//   level's own lattice coordinates, and a CellData group with
//   rho/vx/vy/vz (double) and vtkGhostType (uint8) datasets of the level's
//   block extent (16^3 cells coarse, 16^3 cells fine),
// - rho == 1 and v == 0 (the initialized uniform state) at both levels,
// - vtkGhostType == 4 (REFINEDCELL) exactly on the fine block's coarse
//   footprint at level 0 and == 0 everywhere at level 1.
//
// The verification uses the core HDF5 C API only: the build links
// HDF5::HDF5 (find_package COMPONENTS C), which does not include the H5LT
// high-level lite API (libhdf5_hl), so H5LTfind_group/H5LTfind_attribute
// are replaced by H5Lexists and H5Aopen + H5Aread.
//
// The streaming pattern is selected at compile time (AB_PATTERN/AA_PATTERN);
// this suite is compiled into the consolidated doctest binaries
// test_amr_units_{ab,aa} (tests/unit/CMakeLists.txt), which provide main().
// The writer itself is pattern-agnostic (it reads only the macroscopic
// quantities, which both patterns produce identically from an equilibrium
// state). Everything is single-rank.
//
// NESTING (the amr-nlevel-nesting plan's commit D): a second test pins the
// writer's per-level structure on a 3-level telescoping chain (levels
// 0..3): per-level block grouping, spacing = physDl/2^level, AMRBox extents
// covering exactly each level's own footprint, and the vtkGhostType census
// -- REFINEDCELL on exactly the cells covered by the DIRECT level-below
// footprint at every level (the 2-level idiom generalized). STDOUT
// CONTRACT: the bit-identity harness pins this suite's full normalized
// stdout against the pre-nesting manifest (sec. 7.5), so the nesting
// census runs silent on success (a LogMute keeps the second LBM ctor and
// writer call off the stream) and loud on failure.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include <fmt/core.h>

#include <hdf5.h>

#include <spdlog/spdlog.h>

#include "lbm3d/core.h"
#include "lbm3d/amr_decomposition.h"
#include "lbm3d/viz/OverlappingAMRWriter.h"

// the doctest runner main() lives in doctest_main.cu (MPI initialization)
#include <doctest/doctest.h>

using TRAITS = TraitsSP;
using COLL = D3Q27_CUM<TRAITS, D3Q27_EQ_INV_CUM<TRAITS>>;
using NSE_CONFIG = LBM_CONFIG<
	TRAITS,
	D3Q27_KernelStruct,
	NSE_Data_ConstInflow<TRAITS>,
	COLL,
	typename COLL::EQ,
	D3Q27_STREAMING<TRAITS>,
	D3Q27_BC_All,
	D3Q27_MACRO_Default<TRAITS>>;

using idx = typename TRAITS::idx;
using idx3d = typename TRAITS::idx3d;
using real = typename TRAITS::real;
using point_t = typename TRAITS::point_t;
using lat_t = Lattice<3, real, idx>;

// doctest assertion shim: every legacy report(ok, what) call site becomes
// exactly one doctest assertion, keeping the case running on a failed check
// (same continue-on-fail semantics as the retired g_failures accumulator,
// and nothing is printed on success)
inline void report(bool ok, const std::string& what) { CHECK_MESSAGE(ok, what); }

TEST_SUITE_BEGIN("amr_vtkhdf_writer");

// 16^3 box in physical units (same lattice as test_amr_subcycling.cu)
lat_t makeLattice(int N = 16)
{
	const real LBM_VISCOSITY = 0.005;
	const real PHYS_HEIGHT = 0.41;
	const real PHYS_VISCOSITY = 1.5e-5;
	const real PHYS_DL = PHYS_HEIGHT / N;
	const real PHYS_DT = LBM_VISCOSITY / PHYS_VISCOSITY * PHYS_DL * PHYS_DL;

	lat_t lat;
	lat.global = typename lat_t::CoordinatesType(N, N, N);
	lat.physOrigin = point_t{0., 0., 0.};
	lat.physDl = PHYS_DL;
	lat.physDt = PHYS_DT;
	lat.physViscosity = PHYS_VISCOSITY;
	return lat;
}

// ---------------------------------------------------------------------------
// HDF5 verification helpers (core C API; the hdf5_hl lite API is not linked)
// ---------------------------------------------------------------------------

// read an int64[2] attribute (e.g. the VTKHDF Version); false on failure
bool readAttrI64x2(hid_t loc, const char* name, std::int64_t value[2])
{
	hid_t attr = H5Aopen(loc, name, H5P_DEFAULT);
	if (attr < 0)
		return false;
	const herr_t status = H5Aread(attr, H5T_NATIVE_INT64, value);
	H5Aclose(attr);
	return status >= 0;
}

// read a double[3] attribute (e.g. Origin, Spacing); false on failure
bool readAttrF64x3(hid_t loc, const char* name, double value[3])
{
	hid_t attr = H5Aopen(loc, name, H5P_DEFAULT);
	if (attr < 0)
		return false;
	const herr_t status = H5Aread(attr, H5T_NATIVE_DOUBLE, value);
	H5Aclose(attr);
	return status >= 0;
}

// read a scalar string attribute; false on failure
bool readAttrString(hid_t loc, const char* name, std::string& value)
{
	hid_t attr = H5Aopen(loc, name, H5P_DEFAULT);
	if (attr < 0)
		return false;
	hid_t type = H5Aget_type(attr);
	if (type < 0) {
		H5Aclose(attr);
		return false;
	}
	std::vector<char> buffer(H5Tget_size(type) + 1, '\0');
	const herr_t status = H5Aread(attr, type, buffer.data());
	H5Tclose(type);
	H5Aclose(attr);
	if (status < 0)
		return false;
	value = buffer.data();
	return true;
}

// read a whole dataset together with its dimensions; false on failure
template <typename T>
bool readDataset(hid_t file, const char* path, hid_t mem_type, std::vector<T>& values, std::vector<hsize_t>& dims)
{
	hid_t dataset = H5Dopen2(file, path, H5P_DEFAULT);
	if (dataset < 0)
		return false;
	hid_t space = H5Dget_space(dataset);
	if (space < 0) {
		H5Dclose(dataset);
		return false;
	}
	dims.resize(H5Sget_simple_extent_ndims(space));
	H5Sget_simple_extent_dims(space, dims.data(), nullptr);
	hsize_t size = 1;
	for (hsize_t dim : dims)
		size *= dim;
	values.resize(size);
	const herr_t status = H5Dread(dataset, mem_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, values.data());
	H5Sclose(space);
	H5Dclose(dataset);
	return status >= 0;
}

// expected vtkGhostType of a level-0 cell: REFINEDCELL (4) inside the fine
// block's coarse footprint [4, 12)^3, visible (0) elsewhere
std::uint8_t expectedGhostLevel0(idx x, idx y, idx z)
{
	const bool covered = x >= 4 && x < 12 && y >= 4 && y < 12 && z >= 4 && z < 12;
	return covered ? 4 : 0;
}

// verify one level group: Spacing attribute, AMRBox dataset, and the CellData
// datasets rho/vx/vy/vz (uniform state) and vtkGhostType
void verifyLevel(hid_t file, int level, idx n, real expected_spacing, const std::int32_t expected_box[6], bool finest)
{
	const std::string prefix = fmt::format("/VTKHDF/Level{}", level);
	const idx cells = n * n * n;

	report(H5Lexists(file, prefix.c_str(), H5P_DEFAULT) > 0, fmt::format("group {} exists", prefix));
	report(H5Lexists(file, (prefix + "/CellData").c_str(), H5P_DEFAULT) > 0, fmt::format("group {}/CellData exists", prefix));

	hid_t group = H5Gopen2(file, prefix.c_str(), H5P_DEFAULT);
	if (group < 0) {
		report(false, fmt::format("{}: readable for attribute and dataset checks", prefix));
		return;
	}

	// Spacing: the spatial step halves per refinement level (2:1 ratio)
	double spacing[3] = {0, 0, 0};
	bool spacing_ok = readAttrF64x3(group, "Spacing", spacing);
	for (double s : spacing)
		spacing_ok = spacing_ok && (s == static_cast<double>(expected_spacing));
	report(spacing_ok, fmt::format("{} attribute Spacing == {}", prefix, static_cast<double>(expected_spacing)));
	H5Gclose(group);

	std::vector<hsize_t> dims;

	// AMRBox: one inclusive [lo, hi] row per block in the level's own lattice
	// coordinates, stored as {lo_x, hi_x, lo_y, hi_y, lo_z, hi_z}
	std::vector<std::int32_t> amr_box;
	bool box_ok = readDataset(file, (prefix + "/AMRBox").c_str(), H5T_NATIVE_INT32, amr_box, dims);
	box_ok = box_ok && dims.size() == 2 && dims[0] == 1 && dims[1] == 6 && amr_box.size() == 6;
	if (box_ok)
		for (int i = 0; i < 6; i++)
			box_ok = box_ok && (amr_box[i] == expected_box[i]);
	if (amr_box.size() == 6)
		report(
			box_ok,
			fmt::format(
				"{} AMRBox row == [{}, {}, {}, {}, {}, {}] (actual: [{}, {}, {}, {}, {}, {}])",
				prefix,
				expected_box[0],
				expected_box[1],
				expected_box[2],
				expected_box[3],
				expected_box[4],
				expected_box[5],
				amr_box[0],
				amr_box[1],
				amr_box[2],
				amr_box[3],
				amr_box[4],
				amr_box[5]
			)
		);
	else
		report(false, fmt::format("{} AMRBox readable with shape [1, 6]", prefix));

	// rho == 1 on ALL emitted cells, including the finest-level border rows
	// (footprint-covering ghost rows filled above with the equilibrium macro)
	std::vector<double> rho;
	bool rho_ok = readDataset(file, (prefix + "/CellData/rho").c_str(), H5T_NATIVE_DOUBLE, rho, dims);
	rho_ok = rho_ok && dims.size() == 1 && dims[0] == hsize_t(cells);
	double max_rho_err = 0;
	if (rho_ok)
		for (idx z = 0; z < n; z++)
			for (idx y = 0; y < n; y++)
				for (idx x = 0; x < n; x++) {
					const double v = rho[n * n * z + n * y + x];
					max_rho_err = std::max(max_rho_err, std::abs(v - 1.0));
				}
	rho_ok = rho_ok && max_rho_err <= 1e-5;
	report(rho_ok, fmt::format("{} CellData/rho has {} cells with rho ~ 1 (max |rho - 1| = {:.3e})", prefix, cells, max_rho_err));

	// vx/vy/vz == 0 on all emitted cells (see the rho comment above)
	for (const char* name : {"vx", "vy", "vz"}) {
		std::vector<double> velocity;
		bool vel_ok = readDataset(file, (prefix + "/CellData/" + name).c_str(), H5T_NATIVE_DOUBLE, velocity, dims);
		vel_ok = vel_ok && dims.size() == 1 && dims[0] == hsize_t(cells);
		double max_abs = 0;
		if (vel_ok)
			for (idx z = 0; z < n; z++)
				for (idx y = 0; y < n; y++)
					for (idx x = 0; x < n; x++) {
						const double v = velocity[n * n * z + n * y + x];
						max_abs = std::max(max_abs, std::abs(v));
					}
		vel_ok = vel_ok && max_abs <= 1e-6;
		report(
			vel_ok,
			fmt::format("{} CellData/{} has {} cells with {} ~ 0 (max |{}| = {:.3e})", prefix, name, cells, name, name, max_abs)
		);
	}

	// vtkGhostType: REFINEDCELL (4) on coarse cells covered by the finer
	// level's footprint, visible (0) elsewhere; the finest level is all 0
	// (the emitted rows all cover the coarse footprint)
	std::vector<std::uint8_t> ghost;
	bool ghost_ok = readDataset(file, (prefix + "/CellData/vtkGhostType").c_str(), H5T_NATIVE_UINT8, ghost, dims);
	ghost_ok = ghost_ok && dims.size() == 1 && dims[0] == hsize_t(cells) && ghost.size() == std::size_t(cells);
	idx mismatches = 0;
	idx tagged = 0;
	if (ghost_ok)
		for (idx z = 0; z < n; z++)
			for (idx y = 0; y < n; y++)
				for (idx x = 0; x < n; x++) {
					// dataset cell order: z * ny * nx + y * nx + x (x fastest)
					const std::uint8_t actual = ghost[n * n * z + n * y + x];
					const std::uint8_t expected = finest ? 0 : expectedGhostLevel0(x, y, z);
					if (actual == 4)
						tagged++;
					if (actual != expected) {
						if (mismatches == 0)
							fmt::println(
								"  first mismatch: cell ({}, {}, {}), actual = {}, expected = {}", x, y, z, actual, expected
							);
						mismatches++;
					}
				}
	ghost_ok = ghost_ok && mismatches == 0;
	report(
		ghost_ok,
		fmt::format("{} CellData/vtkGhostType: {} cells with expected tags ({} tagged REFINEDCELL)", prefix, cells, tagged)
	);
}

void test_vtkhdf_structure()
{
	const char* filename = "test_amr.vtkhdf";
	// a stale file from a previous failed run must not pass as a fresh write
	std::remove(filename);

	// minimal two-level AMR setup (the writer is invoked on LBM directly, no
	// State driver is involved)
	lat_t lat = makeLattice();
	LBM<NSE_CONFIG> lbm(MPI_COMM_WORLD, lat, /*periodic_lattice=*/false, /*max_level=*/1);
	lbm.allocateHostData();
	lbm.allocateDeviceData();

	// one centered level-1 region: coarse footprint [4, 12)^3, i.e. a 14^3
	// fine block at fine offset (9, 9, 9) under the re-anchored indexer
	createAMRBlocks(lbm, parseAMRConfig<NSE_CONFIG>("1 4 4 4 8 8 8"));

	report(
		lbm.blocks.size() == 2 && lbm.getBlocksAtLevel(0).size() == 1 && lbm.getBlocksAtLevel(1).size() == 1,
		"setup: one level-0 block and one level-1 block created"
	);

	// uniform equilibrium initial state on both levels; computeInitialMacro
	// is required because D3Q27_MACRO_Default does not recompute macroscopic
	// quantities (compute_in_each_iteration == false) and the writer only
	// copies what is stored in dmacro
	for (auto& block : lbm.blocks) {
		block.setEquilibrium(1, 0, 0, 0);
		block.computeInitialMacro();
		block.copyMacroToHost();
	}

	// fill the FULL stored extent -- including the footprint-covering ghost
	// rows that \ref emitted_range emits -- with the equilibrium macro values
	// (rho = 1, v = 0). The real simulation kernel would compute dmacro on
	// those inner ghost rows in the widened fine substep, but the mock never
	// runs the kernel, so without this fill the writer would read
	// uninitialized macro on the emitted border rows. copyMacroToDevice
	// round-trips the fill through the device array so the writer's later
	// copyMacroToHost (hmacro = dmacro) re-reads rho = 1 / v = 0 everywhere.
	for (auto& block : lbm.blocks) {
		const idx3d ovl{block.df_overlap_X(), block.df_overlap_Y(), block.df_overlap_Z()};
		for (idx gz = block.offset.z() - ovl.z(); gz < block.offset.z() + block.local.z() + ovl.z(); gz++)
			for (idx gy = block.offset.y() - ovl.y(); gy < block.offset.y() + block.local.y() + ovl.y(); gy++)
				for (idx gx = block.offset.x() - ovl.x(); gx < block.offset.x() + block.local.x() + ovl.x(); gx++) {
					block.hmacro(NSE_CONFIG::MACRO::e_rho, gx, gy, gz) = 1;
					block.hmacro(NSE_CONFIG::MACRO::e_vx, gx, gy, gz) = 0;
					block.hmacro(NSE_CONFIG::MACRO::e_vy, gx, gy, gz) = 0;
					block.hmacro(NSE_CONFIG::MACRO::e_vz, gx, gy, gz) = 0;
				}
		block.copyMacroToDevice();
	}

	try {
		OverlappingAMRWriter<TRAITS>::write(filename, lbm, 0.0);
	}
	catch (const std::exception& e) {
		report(false, fmt::format("{}: writer threw an exception: {}", filename, e.what()));
		std::remove(filename);
		return;
	}

	// ---- verify the HDF5 structure ----
	hid_t file = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
	report(file >= 0, fmt::format("{}: H5Fopen succeeded", filename));
	if (file < 0) {
		std::remove(filename);
		return;
	}

	report(H5Lexists(file, "/VTKHDF", H5P_DEFAULT) > 0, "group /VTKHDF exists");
	hid_t root = H5Gopen2(file, "/VTKHDF", H5P_DEFAULT);
	if (root < 0) {
		report(false, "group /VTKHDF readable for attribute checks");
		H5Fclose(file);
		std::remove(filename);
		return;
	}

	// root attributes (Version [2, 8] as of VTK 9.6+)
	std::int64_t version[2] = {0, 0};
	const bool version_read = readAttrI64x2(root, "Version", version);
	report(
		version_read && version[0] == 2 && version[1] == 8,
		fmt::format("attribute Version == [2, 8] (actual: [{}, {}])", version[0], version[1])
	);

	std::string type;
	const bool type_read = readAttrString(root, "Type", type);
	report(
		type_read && type == "OverlappingAMR",
		fmt::format("attribute Type == \"OverlappingAMR\" (actual: \"{}\")", type)
	);

	double origin[3] = {0, 0, 0};
	const point_t expected_origin = lat.lbm2physPoint(0, 0, 0);
	bool origin_ok = readAttrF64x3(root, "Origin", origin);
	origin_ok = origin_ok && origin[0] == static_cast<double>(expected_origin.x()) && origin[1] == static_cast<double>(expected_origin.y())
			 && origin[2] == static_cast<double>(expected_origin.z());
	report(
		origin_ok,
		fmt::format(
			"attribute Origin == lbm2physPoint(0, 0, 0) = ({}, {}, {})",
			static_cast<double>(expected_origin.x()),
			static_cast<double>(expected_origin.y()),
			static_cast<double>(expected_origin.z())
		)
	);

	H5Gclose(root);

	// per-level groups: the coarse block spans the whole 16^3 lattice; the
	// fine block emits the rows covering exactly the coarse footprint:
	// [4, 12)^3 coarse = [8, 24)^3 fine = box [8, 23] inclusive, 16^3 cells
	// (re-anchored indexer: offset 2*origin+1 = 9, local 2*8-2 = 14, and
	// the emitted range adds the footprint-inner ghost row of the 2-deep
	// storage back on each face, see OverlappingAMRWriter::emitted_range)
	const std::int32_t box0[6] = {0, 15, 0, 15, 0, 15};
	const std::int32_t box1[6] = {8, 23, 8, 23, 8, 23};
	verifyLevel(file, 0, 16, lat.physDl, box0, /*finest=*/false);
	verifyLevel(file, 1, 16, lat.physDl / 2, box1, /*finest=*/true);

	H5Fclose(file);
	std::remove(filename);
}

// Nesting census (the amr-nlevel-nesting plan's commit D): the writer on a
// 3-level telescoping chain (levels 0..3, one block per level), pinning per
// level the block grouping, the halved spacing, the AMRBox extents covering
// exactly the level's own footprint and the vtkGhostType census --
// REFINEDCELL on exactly the cells covered by the DIRECT level-below
// footprint at every level (the 2-level idiom generalized; the grandparent
// pairing would leave the intermediate levels' sets empty). Stdout contract
// (file header): silent on success, FAIL + nonzero on failure.

// RAII guard that mutes the default logger for the lifetime of the guard:
// the nesting census constructs a second LBM (whose ctor repeats the
// lattice info lines of the first setup -- pinned in this suite's stdout
// exactly once) and a second writer call
struct LogMute
{
	spdlog::level::level_enum saved;

	LogMute()
	: saved(spdlog::default_logger()->level())
	{
		spdlog::set_level(spdlog::level::off);
	}

	LogMute(const LogMute&) = delete;
	LogMute& operator=(const LogMute&) = delete;

	~LogMute()
	{
		spdlog::set_level(saved);
	}
};

// the 3-level telescoping chain of test_amr_nesting.cu's three_level_chain
// (duplicated so the suites stay independent): every level has exactly one
// block; the REFINEDCELL extents below follow from the same parent-frame
// conversions the nesting binary pins
constexpr const char* nesting_writer_chain = "1 8 1 4 12 30 12\n"
											 "2 38 4 16 16 120 16\n"
											 "3 164 16 64 16 480 16";

// per-level AMRBox extents (inclusive [lo, hi] in the level's own lattice
// coordinates) of the chain on a 32^3 level-0 domain -- hand-computed from
// the re-anchored indexer (offset = 2*(origin >> (level-1)) + 1, local =
// 2*(size >> (level-1)) - 2, emitted box [offset-1, offset+local])
const std::int32_t nesting_boxes[4][6] = {
	{0, 31, 0, 31, 0, 31},		// level 0: the full 32^3 domain
	{16, 39, 2, 61, 8, 31},		// level 1: footprint [16,40)x[2,62)x[8,32)
	{38, 53, 4, 123, 16, 31},	// level 2: footprint [38,54)x[4,124)x[16,32)
	{82, 89, 8, 247, 32, 39},	// level 3: footprint [82,90)x[8,248)x[32,40)
};

// per-level REFINEDCELL extents: the DIRECT level-below footprint in the
// level's own lattice coordinates ([go, go + gs) with go = the child
// block's global_offset and gs = (local + 2)/2); level 3 has no finer
// level and is all-visible
const idx nesting_refined_lo[3][3] = {{8, 1, 4}, {19, 2, 8}, {41, 4, 16}};
const idx nesting_refined_hi[3][3] = {{20, 31, 16}, {27, 62, 16}, {45, 124, 20}};

// expected vtkGhostType of a cell at the given level of the nesting chain
std::uint8_t expectedGhostNested(int level, idx x, idx y, idx z)
{
	if (level >= 3)
		return 0;
	const bool covered = x >= nesting_refined_lo[level][0] && x < nesting_refined_hi[level][0] && y >= nesting_refined_lo[level][1]
					  && y < nesting_refined_hi[level][1] && z >= nesting_refined_lo[level][2] && z < nesting_refined_hi[level][2];
	return covered ? 4 : 0;
}

void test_vtkhdf_nesting_structure()
{
	bool ok = true;
	std::string failure;
	{
		LogMute mute;
		const char* filename = "test_amr_nesting.vtkhdf";
		// a stale file from a previous failed run must not pass as a fresh write
		std::remove(filename);

		lat_t lat = makeLattice(32);
		LBM<NSE_CONFIG> lbm(MPI_COMM_WORLD, lat, /*periodic_lattice=*/false, /*max_level=*/3);
		lbm.allocateHostData();
		lbm.allocateDeviceData();
		createAMRBlocks(lbm, parseAMRConfig<NSE_CONFIG>(nesting_writer_chain));
		if (lbm.blocks.size() != 4) {
			ok = false;
			failure = fmt::format("expected 4 blocks (1 per level), got {}", lbm.blocks.size());
		}
		if (ok) {
			// uniform equilibrium macro state on all blocks, filled over the
			// FULL stored extent including the footprint-covering ghost rows
			// (the mock never runs the kernel; the existing test's idiom)
			for (auto& block : lbm.blocks) {
				block.setEquilibrium(1, 0, 0, 0);
				block.computeInitialMacro();
				block.copyMacroToHost();
			}
			for (auto& block : lbm.blocks) {
				const idx3d ovl{block.df_overlap_X(), block.df_overlap_Y(), block.df_overlap_Z()};
				for (idx gz = block.offset.z() - ovl.z(); gz < block.offset.z() + block.local.z() + ovl.z(); gz++)
					for (idx gy = block.offset.y() - ovl.y(); gy < block.offset.y() + block.local.y() + ovl.y(); gy++)
						for (idx gx = block.offset.x() - ovl.x(); gx < block.offset.x() + block.local.x() + ovl.x(); gx++) {
							block.hmacro(NSE_CONFIG::MACRO::e_rho, gx, gy, gz) = 1;
							block.hmacro(NSE_CONFIG::MACRO::e_vx, gx, gy, gz) = 0;
							block.hmacro(NSE_CONFIG::MACRO::e_vy, gx, gy, gz) = 0;
							block.hmacro(NSE_CONFIG::MACRO::e_vz, gx, gy, gz) = 0;
						}
				block.copyMacroToDevice();
			}
			try {
				OverlappingAMRWriter<TRAITS>::write(filename, lbm, 0.0);
			}
			catch (const std::exception& e) {
				ok = false;
				failure = fmt::format("writer threw an exception: {}", e.what());
			}
		}
		if (ok) {
			hid_t file = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
			if (file < 0) {
				ok = false;
				failure = "H5Fopen failed";
			}
			else {
				const double base_spacing = static_cast<double>(lat.physDl);
				for (int level = 0; level < 4 && ok; level++) {
					const std::string prefix = fmt::format("/VTKHDF/Level{}", level);
					// per-level block grouping: one AMRBox row per level
					std::int32_t expected_box[6];
					for (int i = 0; i < 6; i++)
						expected_box[i] = nesting_boxes[level][i];
					const idx nx = expected_box[1] - expected_box[0] + 1;
					const idx ny = expected_box[3] - expected_box[2] + 1;
					const idx nz = expected_box[5] - expected_box[4] + 1;
					const idx cells = nx * ny * nz;

					hid_t group = H5Gopen2(file, prefix.c_str(), H5P_DEFAULT);
					if (group < 0) {
						ok = false;
						failure = fmt::format("group {} missing", prefix);
						break;
					}
					double spacing[3] = {0, 0, 0};
					double expected_spacing = base_spacing / static_cast<double>(1 << level);
					bool spacing_ok = readAttrF64x3(group, "Spacing", spacing);
					for (double s : spacing)
						spacing_ok = spacing_ok && (s == expected_spacing);
					H5Gclose(group);
					if (! spacing_ok) {
						ok = false;
						failure = fmt::format("{} attribute Spacing != {}", prefix, expected_spacing);
						break;
					}

					std::vector<hsize_t> dims;
					std::vector<std::int32_t> amr_box;
					bool box_ok = readDataset(file, (prefix + "/AMRBox").c_str(), H5T_NATIVE_INT32, amr_box, dims);
					box_ok = box_ok && dims.size() == 2 && dims[0] == 1 && dims[1] == 6 && amr_box.size() == 6;
					if (box_ok)
						for (int i = 0; i < 6; i++)
							box_ok = box_ok && (amr_box[i] == expected_box[i]);
					if (! box_ok) {
						ok = false;
						const std::string actual = amr_box.size() == 6
													 ? fmt::format("[{},{},{},{},{},{}]", amr_box[0], amr_box[1], amr_box[2], amr_box[3], amr_box[4], amr_box[5])
													 : std::string("n/a");
						failure = fmt::format("{} AMRBox (actual size {}, first row {})", prefix, amr_box.size(), actual);
						break;
					}

					// macro content: rho == 1, v == 0 on every emitted cell
					std::vector<double> rho;
					bool rho_ok = readDataset(file, (prefix + "/CellData/rho").c_str(), H5T_NATIVE_DOUBLE, rho, dims);
					rho_ok = rho_ok && dims.size() == 1 && dims[0] == hsize_t(cells);
					double max_rho_err = 0;
					if (rho_ok)
						for (idx i = 0; i < cells; i++)
							max_rho_err = std::max(max_rho_err, std::abs(rho[i] - 1.0));
					if (! rho_ok || max_rho_err > 1e-5) {
						ok = false;
						failure = fmt::format("{} CellData/rho deviates ({} cells, max |rho-1| = {:.3e})", prefix, dims.size() ? dims[0] : 0, max_rho_err);
						break;
					}
					for (const char* name : {"vx", "vy", "vz"}) {
						std::vector<double> velocity;
						bool vel_ok = readDataset(file, (prefix + "/CellData/" + name).c_str(), H5T_NATIVE_DOUBLE, velocity, dims);
						vel_ok = vel_ok && dims.size() == 1 && dims[0] == hsize_t(cells);
						double max_abs = 0;
						if (vel_ok)
							for (idx i = 0; i < cells; i++)
								max_abs = std::max(max_abs, std::abs(velocity[i]));
						if (! vel_ok || max_abs > 1e-6) {
							ok = false;
							failure = fmt::format("{} CellData/{} deviates (max |{}| = {:.3e})", prefix, name, name, max_abs);
							break;
						}
					}
					if (! ok)
						break;

					// the geometry map: present everywhere and all-fluid on
					// this mock (no boundary tagging is performed)
					std::vector<std::int32_t> map_values;
					bool map_ok = readDataset(file, (prefix + "/CellData/map").c_str(), H5T_NATIVE_INT32, map_values, dims);
					// (written as a 2-D [cells, 1] dataset by write_dataset_i32)
					map_ok = map_ok && dims.size() == 2 && dims[0] == hsize_t(cells) && dims[1] == 1;
					idx map_bad = 0;
					if (map_ok)
						for (idx i = 0; i < cells; i++)
							if (map_values[i] != 0) {
								map_bad++;
								break;
							}
					if (! map_ok || map_bad > 0) {
						ok = false;
						failure = fmt::format("{} CellData/map: {} cells, {} non-fluid tags", prefix, dims.size() ? dims[0] : 0, map_bad);
						break;
					}

					// vtkGhostType census: REFINEDCELL (4) on exactly the cells
					// covered by the DIRECT level-below footprint
					std::vector<std::uint8_t> ghost;
					bool ghost_ok = readDataset(file, (prefix + "/CellData/vtkGhostType").c_str(), H5T_NATIVE_UINT8, ghost, dims);
					ghost_ok = ghost_ok && dims.size() == 1 && dims[0] == hsize_t(cells) && ghost.size() == std::size_t(cells);
					idx mismatches = 0;
					idx tagged = 0;
					if (ghost_ok)
						for (idx z = 0; z < nz; z++)
							for (idx y = 0; y < ny; y++)
								for (idx x = 0; x < nx; x++) {
									const std::uint8_t actual = ghost[nx * ny * z + nx * y + x];
									const std::uint8_t expected = expectedGhostNested(level, expected_box[0] + x, expected_box[2] + y, expected_box[4] + z);
									if (actual == 4)
										tagged++;
									if (actual != expected) {
										if (mismatches == 0)
											failure = fmt::format(
												"{} vtkGhostType: first mismatch at cell ({},{},{}), actual = {}, expected = {}",
												prefix, expected_box[0] + x, expected_box[2] + y, expected_box[4] + z, actual, expected
											);
										mismatches++;
									}
								}
					if (! ghost_ok || mismatches > 0) {
						ok = false;
						if (mismatches == 0)
							failure = fmt::format("{} CellData/vtkGhostType unreadable (dims {})", prefix, dims.size() ? dims[0] : 0);
						else
							failure = fmt::format("{}: {} cells", failure, mismatches);
						break;
					}
				}
				H5Fclose(file);
			}
		}
		std::remove(filename);
	}
	if (! ok)
		report(false, fmt::format("nesting vtkhdf census: {}", failure.empty() ? "setup failed" : failure));
}

TEST_CASE("two-level structure") { test_vtkhdf_structure(); }
TEST_CASE("three-level nesting census") { test_vtkhdf_nesting_structure(); }

TEST_SUITE_END();
