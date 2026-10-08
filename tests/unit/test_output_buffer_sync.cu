/*
 * Unit tests for lbm3d::synchronizeOutputBuffer in
 * include/lbm3d/output_buffer_sync.h.
 *
 * The local buffer cells are filled with the analytic value
 *     f(gx, gy, gz) = gx + 1000*gy + 1000000*gz
 * which is unique for every global lattice cell of the test grids, the
 * trailing overlap cells start from a sentinel no analytic value can equal,
 * and after the exchange every cell of the (local + overlap) box is checked
 * against the same function:
 *     buffer[i,j,k] == f(block.offset + (i,j,k)).
 * This single invariant covers the strictly local cells, the face overlaps
 * and the cross-dimension edge/corner cells at once: an overlap cell at
 * global position g is correct if and only if the exchange delivered the
 * value authored on the rank owning g.
 *
 * The forced multi-dimensional decompositions (2x2 at np=4, 2x2x2 at np=8)
 * are what distinguishes the phased implementation from a batched one. The
 * buffer corner cell shared by the X and Y overlaps is owned by the
 * diagonal block and reaches this rank only by forwarding: the Y-face send
 * of the lateral neighbor must pack the X-overlap column it has just
 * received. A batched exchange packs the Y-send before the X-receive is
 * unpacked, so the corner cell keeps the sender's stale content (the
 * sentinel) and the invariant fails. The phased exchange completes and
 * unpacks X before Y is packed (and X+Y before Z), forwarding the correct
 * value.
 */

#include <cstdlib>
#include <vector>

#include <doctest/doctest.h>

#include <TNL/MPI/Comm.h>

#include "lbm3d/lbm_data.h"
#include "lbm3d/d2q9/bc.h"
#include "lbm3d/d2q9/col_srt.h"
#include "lbm3d/d2q9/macro.h"

#include "lbm3d/d2q9/streaming.h"

#include "lbm3d/lbm_block.h"
#include "lbm3d/lattice_decomposition.h"
#include "lbm3d/output_buffer_sync.h"

using TRAITS = Traits<float, double, int>;
using COLL = D2Q9_SRT<TRAITS>;
using CONFIG = LBM_CONFIG<TRAITS, D2Q9_KernelStruct, NSE_Data, COLL, COLL::EQ, D2Q9_STREAMING<TRAITS>, D2Q9_BC_All, D2Q9_MACRO_Default<TRAITS>>;
using BLOCK = LBM_BLOCK<CONFIG>;
using idx = TRAITS::idx;
using idx3d = TRAITS::idx3d;
using bool3d = TRAITS::bool3d;
using dreal = TRAITS::dreal;

// analytic value unique to each global lattice cell of the test grids: any
// value delivered to a wrong position (stale, shifted, or a cross-dimension
// corner mix-up) fails the direct comparison against this function
static dreal cellValue(idx gx, idx gy, idx gz)
{
	return gx + 1000 * gy + 1000000 * gz;
}

// sets (or unsets when spec is null) the forced-decomposition test hook of
// decomposeLattice_D3Q27 for the scope of a test case and unsets it again on
// destruction
struct DecompositionHook
{
	explicit DecompositionHook(const char* spec)
	{
		if (spec == nullptr)
			unsetenv("TNL_LBM_FORCE_DECOMPOSITION");
		else
			setenv("TNL_LBM_FORCE_DECOMPOSITION", spec, 1);
	}
	~DecompositionHook()
	{
		unsetenv("TNL_LBM_FORCE_DECOMPOSITION");
	}
};

static void checkOutputBufferSync(const idx3d& global)
{
	const bool3d periodic{false, false, false};
	BLOCK block = decomposeLattice_D3Q27<CONFIG, idx>(MPI_COMM_WORLD, global, periodic);

	const idx3d local_size = block.local;
	// trailing overlap of one cell along every distributed axis where this
	// rank is not the last block in that axis, zero elsewhere
	idx3d overlap_size{0, 0, 0};
	for (int d = 0; d < 3; d++)
		if (block.is_distributed()[d] && block.offset[d] + block.local[d] < block.global[d])
			overlap_size[d] = 1;

	const idx nx = local_size.x() + overlap_size.x();
	const idx ny = local_size.y() + overlap_size.y();
	const idx nz = local_size.z() + overlap_size.z();

	std::vector<dreal> buffer(std::size_t(nx) * ny * nz, -1);
	for (idx z = 0; z < local_size.z(); z++)
		for (idx y = 0; y < local_size.y(); y++)
			for (idx x = 0; x < local_size.x(); x++)
				buffer[std::size_t(z) * ny * nx + y * nx + x] = cellValue(block.offset.x() + x, block.offset.y() + y, block.offset.z() + z);

	lbm3d::synchronizeOutputBuffer(buffer, block, local_size, overlap_size, block.global);

	// every cell of the (local + overlap) box must carry the analytic value
	// of its global position; the checked-cell count asserts the loop really
	// ran, so a passing case always has assertions
	idx checked = 0;
	for (idx z = 0; z < nz; z++)
		for (idx y = 0; y < ny; y++)
			for (idx x = 0; x < nx; x++) {
				const idx gx = block.offset.x() + x;
				const idx gy = block.offset.y() + y;
				const idx gz = block.offset.z() + z;
				const dreal actual = buffer[std::size_t(z) * ny * nx + y * nx + x];
				checked++;
				if (actual != cellValue(gx, gy, gz))
					FAIL_CHECK(
						fmt::format(
							"rank {}: buffer[{},{},{}] = {}, expected f({},{},{}) = {}", block.rank, x, y, z, actual, gx, gy, gz, cellValue(gx, gy, gz)
						)
					);
			}
	CHECK(checked == nx * ny * nz);
}

TEST_SUITE_BEGIN("outputbuffersync");

TEST_CASE("output buffer sync single-rank")
{
	DecompositionHook hook(nullptr);
	checkOutputBufferSync(idx3d{8, 8, 8});
}

#ifdef HAVE_MPI

TEST_CASE("output buffer sync multi-rank np2")
{
	if (TNL::MPI::GetSize(MPI_COMM_WORLD) != 2)
		return;
	// default split: the interface-optimal decomposer cuts the cube along a
	// single axis, exercising the face overlap of that axis
	DecompositionHook hook(nullptr);
	checkOutputBufferSync(idx3d{8, 8, 8});
}

TEST_CASE("output buffer sync multi-rank 2x2 np4")
{
	if (TNL::MPI::GetSize(MPI_COMM_WORLD) != 4)
		return;
	// forced 2D split: the edge/corner buffer cells of the X and Y overlaps
	// are owned by the diagonal block and reach this rank only through
	// two-dimensional forwarding, which is what separates the phased exchange
	// from a batched one
	DecompositionHook hook("2,2,1");
	checkOutputBufferSync(idx3d{8, 8, 4});
}

TEST_CASE("output buffer sync multi-rank 2x2x2 np8")
{
	if (TNL::MPI::GetSize(MPI_COMM_WORLD) != 8)
		return;
	// forced 3D split: every block is a corner, so all edge and corner
	// buffer cells of the three overlap regions are exercised
	DecompositionHook hook("2,2,2");
	checkOutputBufferSync(idx3d{8, 8, 8});
}

#endif	// HAVE_MPI

TEST_SUITE_END();
