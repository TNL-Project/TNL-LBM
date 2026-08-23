// Unit tests for the AMR inter-level coupling kernels `cudaAMR_CoarseToFine`
// and `cudaAMR_FineToCoarse` (include/lbm3d/d3q27/amr_coupling.h).
//
// The tests are intentionally standalone: each test allocates a pair of
// minimal mock DATA structures on plain NDArray storage (wired exactly like
// LBM_BLOCK::allocateDeviceData), fills them with known fields on the host,
// launches the coupling kernel directly, and verifies the result on the
// host. No State/LBM objects are involved and no simulation is run.
//
// The streaming pattern is selected at compile time: tests/CMakeLists.txt
// compiles this file twice, once with -DAB_PATTERN and once with
// -DAA_PATTERN, producing the test_amr_coupling_{ab,aa} binaries (todo 12,
// test 5). The A-A pattern stores post-collision data in the "twisted"
// orientation (df_cur[opposite(q), site] holds direction q, see
// streaming_AA.h and the kernel docstrings), so the fill/verify helpers
// below are parametrized by the storage parity and cover both A-A states.
//
// F2C STRATEGY SPLIT (commit 14 / plan T15 of
// .omo/plans/schonherr-ch7-conversion.md): this suite pins the
// fine-to-coarse transfer under BOTH compile-time strategies of
// cudaAMR_FineToCoarse:
//   - default build (the TNL_LBM_F2C_STRATEGY cache default
//     F2C_SCHONHERR since commit 15 / T17, passed as the -D define): the
//     Schönherr sec. 7.2 sigma-form compact-moment transfer -- the
//     strategy-split expectations below assert its MEAN-DENSITY transfer
//     semantics (destination density == d0 == the mean of the
//     destination cell's own 8 subcell densities; NO conservation claim,
//     plan T15/T4a-successor);
//   - Lagrava opt-out build (-DTNL_LBM_F2C_STRATEGY=F2C_LAGRAVA): the
//     LAGRAVA 4x4x4 tensor-product filter -- the opt-out authority for
//     the alternative branch, so the Lagrava-only locks below carry an
//     explicit "Lagrava (opt-out) branch" anchor in their comments and
//     report strings.
// Tests whose locks only exist under Lagrava machinery semantics (the
// lo = 0 window-clamp sentinels, Tests 15/18) stay ON the Lagrava branch
// (#ifndef F2C_SCHONHERR) with an explicit deferral report on the arm;
// strategy-independent machinery (storability guard, Defect-2 allowed-GEO
// store guards, frame-orientation stores) is locked once by the shared
// tests and not duplicated per strategy.

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include <fmt/core.h>

#include "lbm3d/core.h"
#include "lbm3d/d3q27/amr_coupling.h"

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

#ifdef AA_PATTERN
constexpr const char* pattern_name = "AA";
#else
constexpr const char* pattern_name = "AB";
#endif

// T15 (commit 14, plan row 15): the F2C strategy this TU pins -- the
// Schönherr transfer is the default since the commit-15 / T17 flip (the
// TNL_LBM_F2C_STRATEGY cache default), so on the default build the locks
// below are the F2C_SCHONHERR arm's expectations; the F2C_LAGRAVA opt-out
// build holds the LAGRAVA (OPT-OUT) authority instead
#ifdef F2C_SCHONHERR
constexpr const char* f2c_strategy_name = "F2C_SCHONHERR arm";
#else
constexpr const char* f2c_strategy_name = "Lagrava (opt-out) branch";
#endif

using idx = typename TRAITS::idx;
using idx3d = typename TRAITS::idx3d;
using dreal = typename TRAITS::dreal;
using DATA = typename NSE_CONFIG::DATA;
using LBM_KS = typename NSE_CONFIG::template KernelStruct<dreal>;

// lattice sizes of the mock blocks (coarse: COARSE_N^3, fine: FINE_N^3 at a
// 2:1 ratio) with a one-cell overlap layer on every side, mirroring
// LBM_BLOCK::overlap_width
constexpr idx COARSE_N = 8;
constexpr idx FINE_N = 16;
constexpr idx OV = 1;

// relaxation times of the two levels for nu_lb_coarse = 0.05 (tau = 3*nu + 0.5,
// nu_lb_fine = 2*nu_lb_coarse with the 2:1 refinement): the non-equilibrium
// rescaling factors tau_f/tau_c = 16/13 and tau_c/tau_f = 13/16 are exercised
// for real (they are not 1)
constexpr dreal TAU_COARSE = 3 * 0.05f + 0.5f;
constexpr dreal TAU_FINE = 3 * 0.10f + 0.5f;

// D3Q27 velocity set c_q matching the direction enum in defs.h
// (zzz, pzz, mzz, ...): p = +1, m = -1, z = 0 per (x, y, z) component
constexpr int VELOCITY[27][3] = {
	{0, 0, 0},	  // zzz
	{1, 0, 0},	  // pzz
	{-1, 0, 0},	  // mzz
	{0, 1, 0},	  // zpz
	{0, -1, 0},	  // zmz
	{0, 0, 1},	  // zzp
	{0, 0, -1},	  // zzm
	{1, 1, 0},	  // ppz
	{-1, -1, 0},  // mmz
	{1, -1, 0},	  // pmz
	{-1, 1, 0},	  // mpz
	{1, 0, 1},	  // pzp
	{-1, 0, -1},  // mzm
	{1, 0, -1},	  // pzm
	{-1, 0, 1},	  // mzp
	{0, 1, 1},	  // zpp
	{0, -1, -1},  // zmm
	{0, 1, -1},	  // zpm
	{0, -1, 1},	  // zmp
	{1, 1, 1},	  // ppp
	{-1, -1, -1}, // mmm
	{1, 1, -1},	  // ppm
	{-1, -1, 1},  // mmp
	{1, -1, 1},	  // pmp
	{-1, 1, -1},  // mpm
	{1, -1, -1},  // pmm
	{-1, 1, 1},	  // mpp
};

int g_failures = 0;

void report(bool ok, const std::string& what)
{
	if (ok)
		fmt::println("PASS: {}", what);
	else {
		fmt::println("FAIL: {}", what);
		g_failures++;
	}
}

// reference equilibrium computed on the host with the same COLL::setEquilibrium
// implementation that the coupling kernels use on the device
std::array<dreal, 27> equilibriumOnHost(dreal rho, dreal vx, dreal vy, dreal vz)
{
	LBM_KS KS;
	KS.rho = rho;
	KS.vx = vx;
	KS.vy = vy;
	KS.vz = vz;
	COLL::setEquilibrium(KS);
	std::array<dreal, 27> eq;
	for (int q = 0; q < 27; q++)
		eq[q] = KS.f[q];
	return eq;
}

// minimal mock of an LBM block's device data: plain (non-distributed) NDArrays
// with a one-cell overlap layer, hand-wired into the kernel-facing DATA
// structure exactly like LBM_BLOCK::allocateDeviceData does
struct MockBlock
{
	DATA data;
	TRAITS::__dmap_array_t dmap;
	TRAITS::__dlat_array_t dfs[DFMAX];
	TRAITS::__dmacro_array_t dmacro;

	TRAITS::__hmap_array_t hmap;
	TRAITS::__hlat_array_t hfs[DFMAX];
	TRAITS::__hmacro_array_t hmacro;

	idx size = 0;
	idx ov = OV;  // overlap width of the allocated arrays (storage [-ov, size+ov))

	void allocate(idx N, idx overlap = OV)
	{
		size = N;
		ov = overlap;

		// plain NDArrays allocate inside setSizes (unlike DistributedNDArray,
		// which additionally needs setDistribution + allocate), and the
		// storage size is computed from the CURRENT overlaps -- hence the
		// overlaps must be set first
		dmap.getOverlaps().template setSize<0>(ov);
		dmap.getOverlaps().template setSize<1>(ov);
		dmap.getOverlaps().template setSize<2>(ov);
		dmap.setSizes(N, N, N);
		hmap.getOverlaps().template setSize<0>(ov);
		hmap.getOverlaps().template setSize<1>(ov);
		hmap.getOverlaps().template setSize<2>(ov);
		hmap.setSizes(N, N, N);

		dmacro.getOverlaps().template setSize<1>(ov);
		dmacro.getOverlaps().template setSize<2>(ov);
		dmacro.getOverlaps().template setSize<3>(ov);
		dmacro.setSizes(NSE_CONFIG::MACRO::N, N, N, N);
		hmacro.getOverlaps().template setSize<1>(ov);
		hmacro.getOverlaps().template setSize<2>(ov);
		hmacro.getOverlaps().template setSize<3>(ov);
		hmacro.setSizes(NSE_CONFIG::MACRO::N, N, N, N);

		for (uint8_t dfty = 0; dfty < DFMAX; dfty++) {
			dfs[dfty].getOverlaps().template setSize<1>(ov);
			dfs[dfty].getOverlaps().template setSize<2>(ov);
			dfs[dfty].getOverlaps().template setSize<3>(ov);
			dfs[dfty].setSizes(NSE_CONFIG::Q, N, N, N);
			hfs[dfty].getOverlaps().template setSize<1>(ov);
			hfs[dfty].getOverlaps().template setSize<2>(ov);
			hfs[dfty].getOverlaps().template setSize<3>(ov);
			hfs[dfty].setSizes(NSE_CONFIG::Q, N, N, N);
		}

		// wire the DATA pointers and the indexer exactly like
		// LBM_BLOCK::allocateDeviceData (the plain-NDArray variant)
		for (uint8_t dfty = 0; dfty < DFMAX; dfty++)
			data.dfs[dfty] = dfs[dfty].getData();
		data.indexer = dmap.getIndexer();
		data.XYZ = data.indexer.getStorageSize();
		data.dmap = dmap.getData();
		data.dmacro = dmacro.getData();

		// GEO_FLUID (0) everywhere: the GEO_AMR_INTERFACE macro-write guard
		// in the coupling kernels stays inactive
		hmap.setValue(TRAITS::map_t(0));
		dmap = hmap;
	}

	void copyToDevice()
	{
		for (uint8_t dfty = 0; dfty < DFMAX; dfty++)
			dfs[dfty] = hfs[dfty];
	}

	void copyToHost()
	{
		for (uint8_t dfty = 0; dfty < DFMAX; dfty++)
			hfs[dfty] = dfs[dfty];
	}
};

// Store the post-collision DF of direction `q` in the slot (and array) that
// the coupling kernel reads back as direction `q`:
// - A-B pattern: the reads use df_out[q] in natural orientation (the parity
//   argument is A-A-only state, ignored by the kernels)
// - A-A pattern with even_iter == true (post-collision, twisted): direction q
//   sits in df_cur[opposite_direction(q)]
// - A-A pattern with even_iter == false (post-stream, natural): direction q
//   sits in df_cur[q]
void storePostCollisionDF(MockBlock& block, bool even_iter, int q, idx x, idx y, idx z, dreal value)
{
#ifdef AB_PATTERN
	static_cast<void>(even_iter);
	block.hfs[df_out](q, x, y, z) = value;
#elif defined(AA_PATTERN)
	block.hfs[df_cur](even_iter ? opposite_direction(q) : q, x, y, z) = value;
#endif
}

// read back the post-collision DF value of direction `q` for parity
// `even_iter` (mirror of storePostCollisionDF)
dreal readPostCollisionDF(const MockBlock& block, bool even_iter, int q, idx x, idx y, idx z)
{
#ifdef AB_PATTERN
	static_cast<void>(even_iter);
	return block.hfs[df_out](q, x, y, z);
#elif defined(AA_PATTERN)
	return block.hfs[df_cur](even_iter ? opposite_direction(q) : q, x, y, z);
#endif
}

// zeroth DF moment of the data stored by fillUniform/fillField with storage
// parity `even_iter`
dreal rhoMomentFilled(const MockBlock& block, bool even_iter, idx x, idx y, idx z)
{
	dreal rho = 0;
	for (int q = 0; q < 27; q++)
		rho += readPostCollisionDF(block, even_iter, q, x, y, z);
	return rho;
}

// Slot in the coarse df_cur array where cudaAMR_FineToCoarse stores the DF of
// direction q (see the kernel docstring): natural for A-B and for A-A when the
// NEXT consuming coarse substep is even, twisted when it is odd.
int coarseWriteSlot(int q, bool next_coarse_even_iter)
{
#ifdef AB_PATTERN
	static_cast<void>(next_coarse_even_iter);
	return q;
#elif defined(AA_PATTERN)
	return next_coarse_even_iter ? q : opposite_direction(q);
#endif
}

// Direction slot in the fine df_cur array where the coarse-to-fine fill
// stores the DF of direction q (see the kernel docstring): natural
// orientation for the A-B pattern (df_cur is pulled same-direction), twisted
// for the A-A pattern (the spatial substep pulls ghost DFs from the
// opposite-direction slot)
int c2fWriteSlot(int q)
{
#ifdef AB_PATTERN
	return q;
#elif defined(AA_PATTERN)
	return opposite_direction(q);
#endif
}

// DF array cudaAMR_FineToCoarse stores into: df_out for the A-B pattern
// (the next global DF rotation turns it into the df_cur the next coarse
// kernel launch reads), df_cur for the A-A pattern (single array)
uint8_t f2cWriteArray()
{
#ifdef AB_PATTERN
	return df_out;
#elif defined(AA_PATTERN)
	return df_cur;
#endif
}

// rho moment of the DFs cudaAMR_FineToCoarse wrote at a coarse cell
dreal f2cWrittenRho(const MockBlock& coarse, bool next_coarse_even_iter, idx x, idx y, idx z)
{
	dreal rho = 0;
	for (int q = 0; q < 27; q++)
		rho += coarse.hfs[f2cWriteArray()](coarseWriteSlot(q, next_coarse_even_iter), x, y, z);
	return rho;
}

// fill the whole stored extent (including the overlap layers) of `block` with
// the equilibrium of a constant macroscopic state
void fillUniform(MockBlock& block, bool even_iter, dreal rho, dreal u0, dreal v0, dreal w0)
{
	const std::array<dreal, 27> eq = equilibriumOnHost(rho, u0, v0, w0);
	for (uint8_t dfty = 0; dfty < DFMAX; dfty++)
		for (int q = 0; q < 27; q++)
			for (idx z = -block.ov; z < block.size + block.ov; z++)
				for (idx y = -block.ov; y < block.size + block.ov; y++)
					for (idx x = -block.ov; x < block.size + block.ov; x++)
						block.hfs[dfty](q, x, y, z) = eq[q];

	// overwrite the array/slot the kernels actually read for this parity
	// (kept separate so an accidental cross-write between the two loops
	// would be caught)
	for (idx z = -block.ov; z < block.size + block.ov; z++)
		for (idx y = -block.ov; y < block.size + block.ov; y++)
			for (idx x = -block.ov; x < block.size + block.ov; x++)
				for (int q = 0; q < 27; q++)
					storePostCollisionDF(block, even_iter, q, x, y, z, eq[q]);
}

// fill the whole stored extent of `block` with the equilibrium of a
// cell-dependent macroscopic field evaluated at each cell's coordinates
template <typename F>
void fillField(MockBlock& block, bool even_iter, F&& macros_at_cell)
{
	for (idx z = -block.ov; z < block.size + block.ov; z++) {
		for (idx y = -block.ov; y < block.size + block.ov; y++) {
			for (idx x = -block.ov; x < block.size + block.ov; x++) {
				dreal rho, vx, vy, vz;
				macros_at_cell(x, y, z, rho, vx, vy, vz);
				const std::array<dreal, 27> eq = equilibriumOnHost(rho, vx, vy, vz);
				for (int q = 0; q < 27; q++)
					storePostCollisionDF(block, even_iter, q, x, y, z, eq[q]);
			}
		}
	}
}

// zeroth moment of the DFs stored in natural orientation in df_cur
dreal rhoMoment(const MockBlock& block, idx x, idx y, idx z)
{
	dreal rho = 0;
	for (int q = 0; q < 27; q++)
		rho += block.hfs[df_cur](q, x, y, z);
	return rho;
}

// launch the coupling kernels with the block-offset parameters the kernels
// map between indexer frames with (fine_off/coarse_off are the two blocks'
// origins in the global coordinates of their level; the fine block's size
// and overlap are taken from the fine mock block)
void launchCoarseToFine(MockBlock& fine, MockBlock& coarse, idx3d begin, idx3d end, idx3d fine_off, idx3d coarse_off, bool coarse_even_iter)
{
	const idx3d size = end - begin;
	TNL::Backend::LaunchConfiguration launch_config;
	launch_config.blockSize = dim3(4, 4, 4);
	launch_config.gridSize = dim3(
		static_cast<unsigned>((size.x() + 3) / 4), static_cast<unsigned>((size.y() + 3) / 4), static_cast<unsigned>((size.z() + 3) / 4)
	);
	TNL::Backend::launchKernelAsync(
		cudaAMR_CoarseToFine<NSE_CONFIG>, launch_config, fine.data, coarse.data, begin, end, TAU_FINE, TAU_COARSE, coarse_even_iter, fine_off, coarse_off
	);
	TNL::Backend::streamSynchronize(0);
}

void launchFineToCoarse(MockBlock& coarse, MockBlock& fine, idx3d begin, idx3d end, idx3d fine_off, idx3d coarse_off, bool fine_even_iter, bool coarse_even_iter)
{
	const idx3d size = end - begin;
	TNL::Backend::LaunchConfiguration launch_config;
	launch_config.blockSize = dim3(4, 4, 4);
	launch_config.gridSize = dim3(
		static_cast<unsigned>((size.x() + 3) / 4), static_cast<unsigned>((size.y() + 3) / 4), static_cast<unsigned>((size.z() + 3) / 4)
	);
	TNL::Backend::launchKernelAsync(
		cudaAMR_FineToCoarse<NSE_CONFIG>,
		launch_config,
		coarse.data,
		fine.data,
		begin,
		end,
		TAU_COARSE,
		TAU_FINE,
		fine_even_iter,
		coarse_even_iter,
		fine_off,
		coarse_off,
		idx3d{fine.size, fine.size, fine.size},
		idx3d{fine.ov, fine.ov, fine.ov}
	);
	TNL::Backend::streamSynchronize(0);
}

// tag a coarse-cell rectangle as GEO_AMR_INTERFACE ring cells and upload the
// map to the device: the production coupling restricts the F2C writes to
// coupling cells (GEO_AMR_INTERFACE ring cells and GEO_NOTHING frozen hidden
// cells, see the allowed-GEO store guard in cudaAMR_FineToCoarse), so mock
// blocks must mark the processed rectangle as ring cells to receive writes
void tagCouplingCells(MockBlock& block, idx3d begin, idx3d end)
{
	for (idx z = begin.z(); z < end.z(); z++)
		for (idx y = begin.y(); y < end.y(); y++)
			for (idx x = begin.x(); x < end.x(); x++)
				block.hmap(x, y, z) = NSE_CONFIG::BC::GEO_AMR_INTERFACE;
	block.dmap = block.hmap;
}

// relative-error comparison with a small absolute floor (rounding-level
// agreement of two float computations on host vs. device)
bool closeEnough(dreal actual, dreal expected, dreal rtol, dreal atol)
{
	return std::abs(actual - expected) <= atol + rtol * std::abs(expected);
}

// Test 1: uniform field, coarse-to-fine -- the fine ghost DFs must equal the
// equilibrium of the uniform field for all 27 directions (both storage
// parities of the source coarse state are covered)
void test_uniform_coarse_to_fine()
{
	const dreal rho0 = 1.0, u0 = 0.05, v0 = -0.03, w0 = 0.02;
	const std::array<dreal, 27> expected = equilibriumOnHost(rho0, u0, v0, w0);
	const std::array<bool, 2> parities = {true, false};

	for (const bool even_iter : parities) {
		MockBlock coarse, fine;
		coarse.allocate(COARSE_N);
		fine.allocate(FINE_N);
		fillUniform(coarse, even_iter, rho0, u0, v0, w0);
		coarse.copyToDevice();

		// fill the whole fine interior plus the outer ghost layer
		launchCoarseToFine(
			fine,
			coarse,
			{0, 0, 0},
			{FINE_N + 1, FINE_N + 1, FINE_N + 1},
			{0, 0, 0},
			{0, 0, 0},
			even_iter
		);
		fine.copyToHost();

		double max_err = 0;
		idx bad = 0;
		for (idx z = 0; z <= FINE_N; z++)
			for (idx y = 0; y <= FINE_N; y++)
				for (idx x = 0; x <= FINE_N; x++)
					for (int q = 0; q < 27; q++) {
						// the fill stores each direction in its pattern-specific slot
						const dreal actual = fine.hfs[df_cur](c2fWriteSlot(q), x, y, z);
						if (! closeEnough(actual, expected[q], 1e-6, 1e-8)) {
							if (bad == 0)
								fmt::println(
									"  first mismatch (even_iter={}): q={}, cell=({},{},{}), actual={:.9e}, expected={:.9e}",
									even_iter, q, x, y, z, actual, expected[q]
								);
							bad++;
						}
						max_err = std::max<double>(max_err, std::abs(actual - expected[q]));
					}
		report(bad == 0, fmt::format("Test 1 uniform coarse-to-fine (even_iter={}): all 27 DFs match equilibrium (max |err| = {:.3e})", even_iter, max_err));
	}
}

// Test 2: uniform field, fine-to-coarse -- the coarse DFs after the F2C
// transfer must equal the equilibrium of the same uniform field (all four
// parity combinations of the stored fine state and the next consuming coarse
// substep are covered). BRANCH-TOLERANT under the T15 strategy split: the
// constant field is exact under BOTH the Lagrava (opt-out) filter (any
// normalized weighted average of a constant is the constant) and the
// F2C_SCHONHERR arm (the subcell mean d0 of a constant density is that
// density and all non-equilibrium moments vanish, both reducing to the same
// equilibrium).
void test_uniform_fine_to_coarse()
{
	const dreal rho0 = 1.0, u0 = 0.04, v0 = 0.01, w0 = -0.02;
	const std::array<dreal, 27> expected = equilibriumOnHost(rho0, u0, v0, w0);
	const std::array<bool, 2> parities = {true, false};

	for (const bool fine_even_iter : parities) {
		for (const bool coarse_even_iter : parities) {
			MockBlock fine, coarse;
			fine.allocate(FINE_N);
			coarse.allocate(COARSE_N);
			fillUniform(fine, fine_even_iter, rho0, u0, v0, w0);
			fine.copyToDevice();
			// F2C writes are restricted to coupling cells: the whole
			// processed block acts as the ring (mock of the production map)
			tagCouplingCells(coarse, {0, 0, 0}, {COARSE_N, COARSE_N, COARSE_N});

			launchFineToCoarse(
				coarse,
				fine,
				{0, 0, 0},
				{COARSE_N, COARSE_N, COARSE_N},
				{0, 0, 0},
				{0, 0, 0},
				fine_even_iter,
				coarse_even_iter
			);
			coarse.copyToHost();

			double max_err = 0;
			idx bad = 0;
			for (idx z = 0; z < COARSE_N; z++)
				for (idx y = 0; y < COARSE_N; y++)
					for (idx x = 0; x < COARSE_N; x++)
						for (int q = 0; q < 27; q++) {
							// the kernel writes the direction-q DF into the
							// parity-dependent slot of the write array
							const dreal actual = coarse.hfs[f2cWriteArray()](coarseWriteSlot(q, coarse_even_iter), x, y, z);
							if (! closeEnough(actual, expected[q], 1e-6, 1e-8)) {
								if (bad == 0)
									fmt::println(
										"  first mismatch (fine_even={}, coarse_even={}): q={}, cell=({},{},{}), actual={:.9e}, expected={:.9e}",
										fine_even_iter, coarse_even_iter, q, x, y, z, actual, expected[q]
									);
								bad++;
							}
							max_err = std::max<double>(max_err, std::abs(actual - expected[q]));
						}
		report(
			bad == 0,
			fmt::format(
				"Test 2 uniform fine-to-coarse (fine_even={}, coarse_even={}) [{}]: all 27 DFs match equilibrium (max |err| = {:.3e})",
				fine_even_iter, coarse_even_iter, f2c_strategy_name, max_err
			)
		);
		}
	}
}

// Test 3: linear gradient field, coarse-to-fine -- the macros of the
// interpolated fine ghost DFs must reproduce the exactly-interpolated linear
// fields (second-order interpolation is exact for linear fields, so only
// rounding-level errors are allowed)
void test_linear_gradient_coarse_to_fine()
{
	// rho(x) = rho0 * (1 + alpha*x), vx(y) = u0 * (1 + beta*y) in coarse cell
	// coordinates; the fine cell center x_f sits at the coarse coordinate
	// x_f/2 - 1/4 (cell-centered 2:1 layout, see the kernel docstring)
	const dreal rho0 = 1.0, u0 = 0.04;
	const dreal alpha = 0.01, beta = 0.02;
	const bool even_iter = false;

	const auto rho_coarse = [&](double x) -> double
	{
		return rho0 * (1 + alpha * x);
	};
	const auto vx_coarse = [&](double y) -> double
	{
		return u0 * (1 + beta * y);
	};

	MockBlock coarse, fine;
	coarse.allocate(COARSE_N);
	fine.allocate(FINE_N);
	fillField(
		coarse,
		even_iter,
		[&](idx x, idx y, idx /*z*/, dreal& rho, dreal& vx, dreal& vy, dreal& vz)
		{
			rho = static_cast<dreal>(rho_coarse(x));
			vx = static_cast<dreal>(vx_coarse(y));
			vy = 0;
			vz = 0;
		}
	);
	coarse.copyToDevice();

	launchCoarseToFine(fine, coarse, {0, 0, 0}, {FINE_N, FINE_N, FINE_N}, {0, 0, 0}, {0, 0, 0}, even_iter);
	fine.copyToHost();

	double max_rel_rho = 0, max_rel_jx = 0, max_abs_jyz = 0;
	idx bad = 0;
	for (idx z = 0; z < FINE_N; z++) {
		for (idx y = 0; y < FINE_N; y++) {
			for (idx x = 0; x < FINE_N; x++) {
				dreal moment[3] = {0, 0, 0};
				for (int q = 0; q < 27; q++) {
					const dreal f = fine.hfs[df_cur](c2fWriteSlot(q), x, y, z);
					for (int d = 0; d < 3; d++)
						moment[d] += VELOCITY[q][d] * f;
				}
				const dreal rho_m = rhoMoment(fine, x, y, z);
				const double rho_e = rho_coarse(x * 0.5 - 0.25);
				const double jx_e = rho_e * vx_coarse(y * 0.5 - 0.25);

				const double rel_rho = std::abs(rho_m - rho_e) / rho_e;
				const double rel_jx = std::abs(moment[0] - jx_e) / jx_e;
				const double abs_jyz = std::max(std::abs(moment[1]), std::abs(moment[2]));
				max_rel_rho = std::max(max_rel_rho, rel_rho);
				max_rel_jx = std::max(max_rel_jx, rel_jx);
				max_abs_jyz = std::max(max_abs_jyz, abs_jyz);
				if (rel_rho > 1e-4 || rel_jx > 1e-4 || abs_jyz > 1e-6) {
					if (bad == 0)
						fmt::println(
							"  first mismatch: cell=({},{},{}), rho={:.9e} (expected {:.9e}), jx={:.9e} (expected {:.9e}), jy={:.3e}, jz={:.3e}",
							x, y, z, rho_m, rho_e, moment[0], jx_e, moment[1], moment[2]
						);
					bad++;
				}
			}
		}
	}
	report(
		bad == 0,
		fmt::format(
			"Test 3 linear gradient coarse-to-fine: ghost moments match exact interpolation "
			"(max rel rho err = {:.3e}, max rel jx err = {:.3e}, max |jy,jz| = {:.3e})",
			max_rel_rho, max_rel_jx, max_abs_jyz
		)
	);
}

// Test 4a: quadratic density field, fine-to-coarse -- strategy-split
// expectations (T15 re-scope, commit 14 / plan row 15):
//   - Lagrava (opt-out) branch (default build): with the 4x4x4 Lagrava
//     filter the coarse cell DF moment after the transfer equals the fine
//     field value at the coarse cell center (the projection is exact for
//     cubic fields, hence for this quadratic one); the Filippova-Hanel
//     non-equilibrium rescaling preserves the zeroth moment by
//     construction. (Polarity fix P0.1: this test previously asserted the
//     1/8 box average of the 8 subcells, which the field's quadratic
//     content deliberately distinguishes from the projection; the Lagrava
//     projection is the intended default.)
//   - F2C_SCHONHERR arm: MEAN-DENSITY TRANSFER (the T4a successor of plan
//     T15) -- the destination density equals d0 == the mean of the
//     destination cell's own 8 subcell densities (here the mean of the two
//     x-subcell values; the sigma-form cumulant reconstruction preserves
//     the zeroth moment of d0 exactly, cf. the L5 lock of
//     tests/test_amr_f2c_schonherr.cu). This pins the subcell-mean reading
//     of the transfer and states NO conservation claim: for nonlinear
//     content the subcell mean and the coarse-center field value differ
//     (by 7.8e-4 for this marker, ~80x above the gate), so exactly one of
//     the two branch expectations can hold per build.
void test_mass_conservation_fine_to_coarse()
{
	const bool fine_even_iter = false, coarse_even_iter = true;

	// non-uniform (quadratic) density field on the fine level: both F2C
	// strategies are globally conservative for ANY field, but only the
	// Lagrava projection reproduces the coarse-center value of a quadratic
	// exactly -- the quadratic content is what discriminates the two
	// strategy-split expectations below
	const auto rho_fine = [](double x) -> double
	{
		const double dx = (x - 7.5) / 4.0;
		return 1.0 + 0.05 * dx * dx;
	};

	MockBlock fine, coarse;
	fine.allocate(FINE_N);
	coarse.allocate(COARSE_N);
	fillField(
		fine,
		fine_even_iter,
		[&](idx x, idx /*y*/, idx /*z*/, dreal& rho, dreal& vx, dreal& vy, dreal& vz)
		{
			rho = static_cast<dreal>(rho_fine(x));
			vx = 0.01f;
			vy = -0.02f;
			vz = 0.03f;
		}
	);
	fine.copyToDevice();
	// F2C writes are restricted to coupling cells: the whole processed
	// block acts as the ring (mock of the production map)
	tagCouplingCells(coarse, {0, 0, 0}, {COARSE_N, COARSE_N, COARSE_N});

	launchFineToCoarse(
		coarse,
		fine,
		{0, 0, 0},
		{COARSE_N, COARSE_N, COARSE_N},
		{0, 0, 0},
		{0, 0, 0},
		fine_even_iter,
		coarse_even_iter
	);
	coarse.copyToHost();

	// exact expectation: strategy-split per the Test-4a block comment --
	// [T15, commit 14]
#ifdef F2C_SCHONHERR
	// mean-density transfer (the T4a successor; NO conservation claim): the
	// destination density is d0 == the subcell mean -- for the x-only
	// quadratic marker, the mean of rho(2x) and rho(2x+1); the L5 lock of
	// tests/test_amr_f2c_schonherr.cu pins the same identity on the
	// dedicated-suite geography (dedupe: one machinery class, two
	// geographies -- this case owns the full-block launch class)
	const auto rho_expected = [&rho_fine](idx x) -> double
	{
		return (rho_fine(2 * x) + rho_fine(2 * x + 1)) / 2;
	};
#else
	// the Lagrava (opt-out) expectation: the Lagrange projection reproduces
	// the quadratic field at the coarse cell center t = 2x + 0.5 (fine
	// indexer coordinates) -- valid on every window, see the kernel
	// docstring
	const auto rho_expected = [&rho_fine](idx x) -> double
	{
		return rho_fine(2 * x + 0.5);
	};
#endif
	// [B.5 re-scope, window-clamp class, mock-matrix.md coupling case 1,
	// unconditional since D.1 retired the ring-F2C path (gate B ruling):
	// the axis_window LOWER bound is 0 (the pre-skin default was -ov), so
	// the coarse cells at x == 0 (fx0 == 0) evaluate on the interior-clamped
	// window {0,1,2,3} instead of the nominal {-1,0,1,2}. The pinned
	// expectation is UNCHANGED: a quadratic is reproduced exactly at the
	// fixed evaluation point by ANY 4-node window, so the clamp manifests
	// only as an fp re-shuffle (measured 1.901e-07 -> 2.783e-07 when it
	// landed), ~36x below the 1e-5 gate. The x == 7 cells' hi-side window
	// {13,14,15,16} still INCLUDES the fine ghost node 16 (the upper bound
	// was never loosened) -- the load-bearing control that the shift sits
	// exactly at the lo site. -- this clamp note is Lagrava-branch
	// machinery (the F2C_SCHONHERR arm has no window), kept as the opt-out
	// authority's documentation per T15]
	double max_rel = 0;
	idx bad = 0;
	for (idx z = 0; z < COARSE_N; z++)
		for (idx y = 0; y < COARSE_N; y++)
			for (idx x = 0; x < COARSE_N; x++) {
				const dreal rho_c = f2cWrittenRho(coarse, coarse_even_iter, x, y, z);
				const double rho_e = rho_expected(x);
				const double rel = std::abs(rho_c - rho_e) / rho_e;
				max_rel = std::max(max_rel, rel);
				if (rel > 1e-5)
					bad++;
			}
	report(
		bad == 0,
		fmt::format(
			"Test 4a quadratic reproduction fine-to-coarse [{}]: {}", f2c_strategy_name,
#ifdef F2C_SCHONHERR
			fmt::format("destination density == subcell mean d0 (mean-density transfer; max rel err = {:.3e})", max_rel)
#else
			fmt::format("coarse rho == fine field at the coarse cell center (max rel err = {:.3e})", max_rel)
#endif
		)
	);
}

// Test 4b: mass conservation, coarse-to-fine -- for a linear density field
// the volume-weighted sum of the 8 fine subcell moments of one coarse cell
// must equal the coarse cell mass (trilinear interpolation with the
// symmetric 3/4-1/4 weights is conservative for linear fields)
void test_mass_conservation_coarse_to_fine()
{
	const dreal rho0 = 1.0;
	const dreal alpha = 0.01;
	const bool even_iter = false;

	const auto rho_coarse = [&](double x) -> double
	{
		return rho0 * (1 + alpha * x);
	};

	MockBlock coarse, fine;
	coarse.allocate(COARSE_N);
	fine.allocate(FINE_N);
	fillField(
		coarse,
		even_iter,
		[&](idx x, idx /*y*/, idx /*z*/, dreal& rho, dreal& vx, dreal& vy, dreal& vz)
		{
			rho = static_cast<dreal>(rho_coarse(x));
			vx = 0.02f;
			vy = -0.01f;
			vz = 0.03f;
		}
	);
	coarse.copyToDevice();

	// fill the fine cells covered by the interior coarse cells [1, COARSE_N)
	// (their fine subcells are [2, FINE_N))
	launchCoarseToFine(fine, coarse, {2, 2, 2}, {FINE_N, FINE_N, FINE_N}, {0, 0, 0}, {0, 0, 0}, even_iter);
	fine.copyToHost();

	double max_rel = 0;
	idx bad = 0;
	for (idx z = 1; z < COARSE_N; z++)
		for (idx y = 1; y < COARSE_N; y++)
			for (idx x = 1; x < COARSE_N; x++) {
				double m = 0;
				for (int bz = 0; bz < 2; bz++)
					for (int by = 0; by < 2; by++)
						for (int bx = 0; bx < 2; bx++)
							m += rhoMoment(fine, 2 * x + bx, 2 * y + by, 2 * z + bz);
				m /= 8.0;
				// volume-accounted conservation: the 8 fine subcells (1/8 of
				// the coarse volume each) must sum to the coarse cell mass
				const double rel = std::abs(m - rho_coarse(x)) / rho_coarse(x);
				max_rel = std::max(max_rel, rel);
				if (rel > 1e-4)
					bad++;
			}
	report(
		bad == 0,
		fmt::format("Test 4b mass conservation coarse-to-fine: 1/8 * fine subcell mass == coarse mass (max rel err = {:.3e})", max_rel)
	);
}

// ---------------------------------------------------------------------------
// Test 5: nested-geometry coupling regression (geographic frame mapping)
//
// Tests 1-4 place both mock blocks at the same indexer origin, so the two
// blocks' indexer frames correspond and an offset-blind cell mapping cannot
// be distinguished from the correct one. This test reproduces the geometry
// the production coupling actually runs in (identical to test_amr_subcycling
// and the mock RCA experiment): a coarse 16^3 block at offset (0,0,0) and a
// fine 16^3 block covering the parent footprint [4,12)^3 = fine global
// [8,24)^3, i.e. fine offset (8,8,8). Overlap is 2 cells deep so that the
// full 2-cell-deep fine ghost ring the coupling needs is storable.
//
// Marker fields with exact binary-fraction moments (rho linear in the global
// x coordinate, constant vx != 0 to break the q <-> opposite(q) equilibrium
// symmetry) make the correct mapping distinguishable on the host from any
// index shift, from a wrong storage array (A-B pattern: the array the kernel
// must not read is poisoned with a large sentinel), and from a wrong
// direction slot (A-A pattern: the fill must be twisted for the spatial
// substep's consumer). All expected values are host replicas of the correct
// global-frame mapping.
// ---------------------------------------------------------------------------

// lattice sizes/offsets of the nested mock blocks (both 16^3 cells, fine
// block at fine-global offset 8 per axis) and the marker parameters
constexpr idx NEST_N = 16;
constexpr idx NEST_OV = 2;
constexpr idx NEST_FINE_OFF = 8;
// marker offset: keeps rho >= 2 everywhere (rho == 0 would make
// computeDensityAndVelocity divide by zero inside the kernel)
constexpr dreal NEST_RHO0 = 4;
// direction-asymmetry marker (binary fraction); also used for the slot
// distinguishability of the A-A twisted fill
constexpr dreal NEST_VX = dreal(0.03125);
#ifdef AB_PATTERN
// wrong-array sentinel added to every DF in the array the coupling must NOT
// read (A-B pattern with two DF arrays only)
constexpr dreal NEST_GARBAGE = 1000;
#endif

// floor division by 2 (valid for negative coordinates, unlike C++ integer
// division which truncates toward zero)
idx floor_div2(idx v)
{
	return v >= 0 ? v / 2 : -((-v + 1) / 2);
}

// host replica of the CORRECT global-frame coarse-to-fine mapping: rho of the
// fine ghost cell with fine indexer coordinate fx (marker is x-only, so the
// y/z trilinear weights sum to 1)
dreal correctC2Frho(idx fx, idx fine_off)
{
	const idx fg = fx + fine_off;  // fine global coordinate
	const idx c0 = floor_div2(fg) - 1 + (fg & 1);
	const dreal w0 = (fg & 1) ? dreal(0.75) : dreal(0.25);
	// coarse marker: rho at coarse global c is c + NEST_RHO0
	return w0 * c0 + (1 - w0) * (c0 + 1) + NEST_RHO0;
}

// host replica of the CORRECT global-frame fine-to-coarse mapping: the
// mean subcell density of the marker rho over the 2 x-subcells of coarse
// cell c (the y/z average preserves the x-only marker). [T15: this mean is
// the STRATEGY-INDEPENDENT expectation -- identical to the Lagrava
// (opt-out) projection at the coarse center for a linear marker, and to
// the F2C_SCHONHERR branch's d0 (subcell mean) by definition, so the Test
// 5/7 expectations hold under both F2C strategies]
dreal correctF2Crho(idx c, idx coarse_off, idx fine_off)
{
	dreal m = 0;
	for (int b = 0; b < 2; b++) {
		const idx f = 2 * (c + coarse_off) - fine_off + b;
		m += dreal(0.5) * (f + fine_off) + NEST_RHO0;
	}
	return m / 2;
}

// fill the whole stored extent (including the 2-cell overlap) of `block` with
// the equilibrium of the marker field. The A-B coupling kernels read the
// source block's df_out in natural orientation; every other array is poisoned
// with the wrong-array sentinel so that a read of the wrong array is visible
// in the prob moments. The A-A kernels read df_cur with a parity-dependent
// slot covered by `source_even_iter`.
void fillMarkerNested(MockBlock& block, bool is_fine, bool source_even_iter)
{
	for (idx z = -block.ov; z < block.size + block.ov; z++) {
		for (idx y = -block.ov; y < block.size + block.ov; y++) {
			for (idx x = -block.ov; x < block.size + block.ov; x++) {
				const dreal rho = (is_fine ? dreal(0.5) * static_cast<dreal>(x + NEST_FINE_OFF) : static_cast<dreal>(x)) + NEST_RHO0;
				const std::array<dreal, 27> eq = equilibriumOnHost(rho, NEST_VX, 0, 0);
#ifdef AB_PATTERN
				static_cast<void>(source_even_iter);  // A-A-only state
				for (uint8_t dfty = 0; dfty < DFMAX; dfty++)
					for (int q = 0; q < 27; q++)
						block.hfs[dfty](q, x, y, z) = eq[q] + NEST_GARBAGE;
				for (int q = 0; q < 27; q++)
					block.hfs[df_out](q, x, y, z) = eq[q];
#elif defined(AA_PATTERN)
				for (int q = 0; q < 27; q++)
					storePostCollisionDF(block, source_even_iter, q, x, y, z, eq[q]);
#endif
			}
		}
	}
}

// rho moment of the DFs the coupling kernel is expected to have WRITTEN at a
// cell (direction-wise compose the physical direction q from its storage
// slot: fine ghost fill for C2F, coarse interface cell for F2C)
dreal nestedFineGhostRho(const MockBlock& fine, idx x, idx y, idx z)
{
	dreal rho = 0;
	for (int q = 0; q < 27; q++)
		rho += fine.hfs[df_cur](c2fWriteSlot(q), x, y, z);
	return rho;
}

void test_nested_geometry_coupling()
{
	// storage parities of the source data (A-A-only state): post-stream
	// natural orientation on both levels; the next consuming coarse substep
	// is a spatial one (twisted F2C write)
	const bool coarse_even_iter = false;
	const bool fine_even_iter = false;
	const bool next_coarse_even_iter = false;

	MockBlock coarse, fine;
	coarse.allocate(NEST_N, NEST_OV);
	fine.allocate(NEST_N, NEST_OV);
	fillMarkerNested(coarse, false, coarse_even_iter);
	fillMarkerNested(fine, true, fine_even_iter);
	coarse.copyToDevice();
	fine.copyToDevice();

	// coarse-to-fine: the production ghost extents of the min-x and max-x
	// faces (fine indexer coordinates, cf. buildCouplings:
	// fine_origin = 2 * coarse_rect_begin - fine.offset)
	const idx3d fine_off{NEST_FINE_OFF, NEST_FINE_OFF, NEST_FINE_OFF};
	const idx3d coarse_off{0, 0, 0};
	launchCoarseToFine(fine, coarse, {-2, -2, -2}, {0, 18, 18}, fine_off, coarse_off, coarse_even_iter);
	launchCoarseToFine(fine, coarse, {16, -2, -2}, {18, 18, 18}, fine_off, coarse_off, coarse_even_iter);

	fine.copyToHost();
	coarse.copyToHost();

	double max_err = 0;
	idx bad = 0;
	bool first_mismatch;

	// ----- C2F min-x and max-x ghost faces: every ghost cell and every DF
	// must hold the marker equilibrium of the correctly mapped macros -----
	for (const idx gxx : {-2, -1, 16, 17}) {
		const dreal rho_e = correctC2Frho(gxx, NEST_FINE_OFF);
		const std::array<dreal, 27> eq = equilibriumOnHost(rho_e, NEST_VX, 0, 0);
		max_err = 0;
		bad = 0;
		first_mismatch = true;
		for (idx z = -NEST_OV; z < NEST_N + NEST_OV; z++) {
			for (idx y = -NEST_OV; y < NEST_N + NEST_OV; y++) {
				for (int q = 0; q < 27; q++) {
					const dreal actual = fine.hfs[df_cur](c2fWriteSlot(q), gxx, y, z);
					if (! closeEnough(actual, eq[q], 1e-4, 1e-5)) {
						if (first_mismatch) {
							fmt::println(
								"  first mismatch: ghost=({}, {}, {}), q={}, actual={:.9e}, expected={:.9e}",
								gxx, y, z, q, actual, eq[q]
							);
							first_mismatch = false;
						}
						bad++;
					}
					max_err = std::max<double>(max_err, std::abs(actual - eq[q]));
				}
			}
		}
		report(
			bad == 0,
			fmt::format("Test 5 nested C2F ghost x={}: all y/z cells and DFs match the correct global-frame mapping (max |err| = {:.3e})", gxx, max_err)
		);
	}

	// rho moments as an independent aggregate check (same expectation values,
	// different roundoff path)
	max_err = 0;
	bad = 0;
	first_mismatch = true;
	for (const idx gxx : {-2, -1, 16, 17}) {
		const dreal rho_e = correctC2Frho(gxx, NEST_FINE_OFF);
		for (idx z = 0; z < NEST_N; z++) {
			for (idx y = 0; y < NEST_N; y++) {
				const dreal rho_m = nestedFineGhostRho(fine, gxx, y, z);
				if (! closeEnough(rho_m, rho_e, 1e-4, 1e-5)) {
					if (first_mismatch) {
						fmt::println("  first moment mismatch: ghost=({}, {}, {}), rho={:.9e}, expected={:.9e}", gxx, y, z, rho_m, rho_e);
						first_mismatch = false;
					}
					bad++;
				}
				max_err = std::max<double>(max_err, std::abs(rho_m - rho_e));
			}
		}
	}
	report(bad == 0, fmt::format("Test 5 nested C2F ghost rho moments: moments match (max |err| = {:.3e})", max_err));

	// re-establish the pristine marker state for the F2C direction: the C2F
	// fill just wrote into the fine ghost layer that the F2C filter reads
	// (the intended production flow) -- for the A-A pattern with a spatial-consumer
	// fill, self-opposite directions (e.g. the rest direction) share the slot
	// the F2C natural read uses, so the F2C input must be re-marked to assert
	// pure filter geography
	fillMarkerNested(coarse, false, coarse_even_iter);
	fillMarkerNested(fine, true, fine_even_iter);
	coarse.copyToDevice();
	fine.copyToDevice();

	// fine-to-coarse: FULL halo face extents extended past both storable and
	// non-storable cells -- the per-cell kernel guard replaces the former
	// launch-extent clip (c = 2 and c = 13 are skipped by the guard as their
	// 2x2x2 fine subcell block is not fully storable). F2C writes are
	// restricted to coupling cells: tag the launch rectangles as ring cells
	// (mock of the production GEO_AMR_INTERFACE ring).
	// [B.5 re-scope, window-clamp class, mock-matrix.md coupling cases 2-3,
	// re-scoped to kernel-machinery coverage in D.1 (gate B ruling):
	// production no longer launches these halo (ring) rectangles AT ALL --
	// the ring-F2C launch was hard-deleted with F2C_SKIN_ONLY. The mock
	// keeps the direct kernel launch as coverage of LIVE machinery that no
	// surviving skin test (14-16/18, small ov=1 fixtures, always-storable
	// subcells) exercises: the per-cell storability guard's SKIP path
	// (c = 2/13 stay at the marker IC), the A-B wrong-array trap below
	// (writes land df_out, df_cur sentinel untouched), and the lo = 0
	// clamped window's shifts at hard NEST edges. The pinned host replica
	// correctF2Crho is the analytic coarse-center value of the linear
	// marker, reproduced exactly by EVERY 4-node window (window-
	// independent), so NO assertion value changes across the clamp; the
	// moving windows are (case 2) the c = 3 column's x-window (fx0 = -2:
	// storage-clamped {-2,-1,0,1} -> interior-clamped {0,1,2,3}) plus the
	// asserted y = 4 / z = 4 rows' tangent windows (fy0 = 0: nominal
	// {-1,0,1,2} -> {0,1,2,3}), and (case 3) ONLY the c = 12 assertions'
	// y = 4 / z = 4 rows -- the c = 12 x-window {14,15,16,17} is hi-clamped
	// INCLUDING the ghost nodes in every configuration and the lo clamp
	// cannot move it (control: the x-window is invariant yet the err moved,
	// naming the y/z rows as the mechanism). Measured re-shuffles
	// 2.384e-07 -> 2.861e-06 (c = 3) and 9.537e-07 -> 2.384e-06 (c = 12)
	// when the clamp landed, ~30x below the 1e-4/1e-5 gates.]
	// [T15: the window-clamp re-shuffles named here are LAGRAVA-BRANCH
	// machinery (the F2C_SCHONHERR arm has no window -- own-8 subcells
	// only); the per-cell storability SKIP path, the A-B wrong-array trap,
	// and the frame-orientation stores pinned by this launch are
	// strategy-independent, and the linear-marker transfer values coincide
	// between branches, so this launch block stays shared under the
	// strategy split]
	tagCouplingCells(coarse, {2, 3, 3}, {4, 13, 13});
	tagCouplingCells(coarse, {12, 3, 3}, {14, 13, 13});
	launchFineToCoarse(coarse, fine, {2, 3, 3}, {4, 13, 13}, fine_off, coarse_off, fine_even_iter, next_coarse_even_iter);
	launchFineToCoarse(coarse, fine, {12, 3, 3}, {14, 13, 13}, fine_off, coarse_off, fine_even_iter, next_coarse_even_iter);

	fine.copyToHost();
	coarse.copyToHost();

	// ----- F2C halo faces: the coarse halo cells must hold the subcell
	// average of the correctly mapped fine subcells [T15: BRANCH-TOLERANT
	// expectation -- correctF2Crho is the 1/8 mean over the own 2
	// x-subcells of an x-only LINEAR marker, so the Lagrava (opt-out)
	// projection value and the F2C_SCHONHERR subcell mean d0 coincide
	// exactly (both reproduce a linear field at the coarse center); the
	// pinned value is therefore identical under both strategies] -----
	for (const idx c : {3, 12}) {
		const dreal rho_e = correctF2Crho(c, 0, NEST_FINE_OFF);
		const std::array<dreal, 27> eq = equilibriumOnHost(rho_e, NEST_VX, 0, 0);
		max_err = 0;
		bad = 0;
		first_mismatch = true;
		for (const idx z : {4, 8, 12}) {
			for (const idx y : {4, 8, 12}) {
				for (int q = 0; q < 27; q++) {
					const dreal actual = coarse.hfs[f2cWriteArray()](coarseWriteSlot(q, next_coarse_even_iter), c, y, z);
					if (! closeEnough(actual, eq[q], 1e-4, 1e-5)) {
						if (first_mismatch) {
							fmt::println(
								"  first mismatch: halo=({}, {}, {}), q={}, actual={:.9e}, expected={:.9e}",
								c, y, z, q, actual, eq[q]
							);
							first_mismatch = false;
						}
						bad++;
					}
					max_err = std::max<double>(max_err, std::abs(actual - eq[q]));
				}
			}
		}
		report(
			bad == 0,
			fmt::format(
				"Test 5 nested F2C halo c={} [{}]: subcell average of the correct fine subcells "
				"(global-frame mapping, per-cell storability; max |err| = {:.3e})",
				c, f2c_strategy_name, max_err
			)
		);
	}

#ifdef AB_PATTERN
	// ----- wrong-array trap: the coarse df_cur array was poisoned with the
	// sentinel; the F2C transfer must write df_out (the frame the next coarse
	// kernel launch reads as df_cur), leaving df_cur untouched -----
	bad = 0;
	max_err = 0;
	first_mismatch = true;
	for (const idx z : {4, 8, 12}) {
		for (const idx y : {4, 8, 12}) {
			const std::array<dreal, 27> eq_marker = equilibriumOnHost(static_cast<dreal>(3) + NEST_RHO0, NEST_VX, 0, 0);
			for (int q = 0; q < 27; q++) {
				const dreal actual = coarse.hfs[df_cur](q, 3, y, z);
				const dreal expected = eq_marker[q] + NEST_GARBAGE;
				if (! closeEnough(actual, expected, 1e-5, 1e-2)) {
					if (first_mismatch) {
						fmt::println("  first mismatch: df_cur at halo=(3, {}, {}), q={}, actual={:.9e}, expected(untouched sentinel)={:.9e}", y, z, q, actual, expected);
						first_mismatch = false;
					}
					bad++;
				}
				max_err = std::max<double>(max_err, std::abs(actual - expected));
			}
		}
	}
	report(
		bad == 0,
		fmt::format("Test 5 nested F2C A-B frame: halo writes landed in df_out (df_cur untouched sentinel, max |err| = {:.3e})", max_err)
	);
#endif

	// ----- storability guard: coarse cells whose 2x2x2 fine subcell block is
	// not fully storable must be left at the marker IC (c = 2 and c = 13 lie
	// at the boundary of storable fine subcells with NEST_OV == 2) -----
	max_err = 0;
	bad = 0;
	for (const idx c : {2, 13}) {
		const dreal rho_marker = static_cast<dreal>(c) + NEST_RHO0;
		const dreal rho_m = f2cWrittenRho(coarse, next_coarse_even_iter, c, 8, 8);
		max_err = std::max<double>(max_err, std::abs(rho_m - rho_marker));
		if (! closeEnough(rho_m, rho_marker, 1e-4, 1e-5))
			bad++;
	}
	report(
		bad == 0,
		fmt::format("Test 5 nested F2C storability guard: non-storable halo cells (c=2,13) remain at the marker IC (max |err| = {:.3e})", max_err)
	);
}

// Test 6: exact cubic reproduction, fine-to-coarse -- strategy-split
// expectations (T15 re-scope, commit 14 / plan row 15):
//   - Lagrava (opt-out) branch (default build): the tensor-product
//     4-node-per-axis Lagrange projection onto the coarse cell center
//     reproduces cubic fields exactly (the 1/8 box average is only
//     linear-exact). A fine field carrying exact cubic content must
//     therefore come through the transfer exactly on a nominal interior
//     window (centered {-1,9,9,-1}/16 per-axis weights); on a box-average
//     build this test fails, pinning the filter polarity (Lagrava default,
//     F2C_BOX_AVERAGE opt-out).
//   - F2C_SCHONHERR arm: the mean-density transfer of the same cubic
//     marker -- the destination DFs equal the equilibrium of (d0, u) with
//     d0 == the mean of the own x-subcell densities (mean of rho(2x) and
//     rho(2x+1); NO conservation claim -- mean and center value differ for
//     nonlinear content, so exactly one branch expectation holds per
//     build). The cubic content keeps the test's fp-class discrimination:
//     the subcell mean is a window-independent analytic value, and a
//     hypothetical box-average-OF-THE-DFs regression would fail the 1e-5
//     gate by the same margin as on the Lagrava branch.
void test_cubic_reproduction_fine_to_coarse()
{
	// rho(x) = rho0 + A*dx^3 with dx = (x - 7.5)/4 in fine indexer
	// coordinates, constant velocities: the DF equilibrium is linear in
	// rho, so the projected DFs are exactly the equilibrium of the cubic
	// field evaluated at the destination density of the strategy split
	// (Lagrava: the coarse cell center t = 2x + 0.5; F2C_SCHONHERR: the
	// subcell mean d0)
	const dreal rho0 = 1.0, A = 0.1;
	const dreal u0 = 0.01, v0 = -0.02, w0 = 0.03;
	const std::array<bool, 2> parities = {true, false};

	const auto rho_fine = [&](double x) -> double
	{
		const double dx = (x - 7.5) / 4.0;
		return rho0 + A * dx * dx * dx;
	};

	for (const bool fine_even_iter : parities) {
		for (const bool coarse_even_iter : parities) {
			MockBlock fine, coarse;
			fine.allocate(FINE_N);
			coarse.allocate(COARSE_N);
			fillField(
				fine,
				fine_even_iter,
				[&](idx x, idx /*y*/, idx /*z*/, dreal& rho, dreal& vx, dreal& vy, dreal& vz)
				{
					rho = static_cast<dreal>(rho_fine(x));
					vx = u0;
					vy = v0;
					vz = w0;
				}
			);
			fine.copyToDevice();
			// F2C writes are restricted to coupling cells: the whole
			// processed block acts as the ring (mock of the production map)
			tagCouplingCells(coarse, {0, 0, 0}, {COARSE_N, COARSE_N, COARSE_N});

			launchFineToCoarse(
				coarse,
				fine,
				{0, 0, 0},
				{COARSE_N, COARSE_N, COARSE_N},
				{0, 0, 0},
				{0, 0, 0},
				fine_even_iter,
				coarse_even_iter
			);
			coarse.copyToHost();

			// nominal interior window only: coarse cells [1, COARSE_N-1)
			// see the centered 4x4x4 stencil -- the Lagrava (opt-out)
			// window class; the F2C_SCHONHERR arm has no window (own-8
			// subcells), the same cells are asserted for uniformity
			// [T15, commit 14]
			double max_err = 0;
			idx bad = 0;
			for (idx z = 1; z < COARSE_N - 1; z++)
				for (idx y = 1; y < COARSE_N - 1; y++)
					for (idx x = 1; x < COARSE_N - 1; x++) {
#ifdef F2C_SCHONHERR
						// mean-density transfer (the T4a successor; NO
						// conservation claim): d0 == subcell mean
						const dreal rho_e = static_cast<dreal>((rho_fine(2 * x) + rho_fine(2 * x + 1)) / 2);
#else
						const dreal rho_e = static_cast<dreal>(rho_fine(2 * x + 0.5));
#endif
						const std::array<dreal, 27> eq = equilibriumOnHost(rho_e, u0, v0, w0);
						for (int q = 0; q < 27; q++) {
							const dreal actual = coarse.hfs[f2cWriteArray()](coarseWriteSlot(q, coarse_even_iter), x, y, z);
							if (! closeEnough(actual, eq[q], 1e-5, 1e-6)) {
								if (bad == 0)
									fmt::println(
										"  first mismatch (fine_even={}, coarse_even={}): cell=({},{},{}), q={}, actual={:.9e}, expected={:.9e}",
										fine_even_iter, coarse_even_iter, x, y, z, q, actual, eq[q]
									);
								bad++;
							}
							max_err = std::max<double>(max_err, std::abs(actual - eq[q]));
						}
					}
			report(
				bad == 0,
				fmt::format(
					"Test 6 cubic reproduction fine-to-coarse (fine_even={}, coarse_even={}) [{}]: "
					"interior coarse DFs match {} (max |err| = {:.3e})",
					fine_even_iter,
					coarse_even_iter,
					f2c_strategy_name,
#ifdef F2C_SCHONHERR
					"the mean-density transfer of the cubic marker (destination density == subcell mean d0)",
#else
					"the exactly-projected cubic field",
#endif
					max_err
				)
			);
		}
	}
}

// Test 7: Defect-2 DF-store map guard (NEST lock) -- the coarse DF store of
// cudaAMR_FineToCoarse must be guarded by the SAME allowed-GEO predicate as
// the macro store: only GEO_AMR_INTERFACE ring cells and GEO_NOTHING frozen
// hidden cells are coupling storage; cells tagged with any other GEO (a
// boundary-condition tag such as GEO_WALL, but also plain GEO_FLUID) own
// their DFs and macros and must not be clobbered when a coupling rectangle
// covers them. The geometry is the Test 5 NEST setup with one processed
// column (c = 3) holding one cell per map class.
// [B.5 re-scope, window-clamp class, mock-matrix.md coupling case 4,
// unconditional since D.1 retired the ring path (gate B ruling): the
// processed column c = 3 (fx0 = -2) is the same lo-clamp site as case 2 --
// with the lo = 0 lower bound its x-window moves from the storage-clamped
// {-2,-1,0,1} to {0,1,2,3}; the asserted class cells' y/z windows (y in
// {8,9,10,11}, z = 8, tangent window starts >= 7) cannot engage a
// LOWER-bound clamp, so the x-window is the SOLE moving site for the
// asserted cells (case 2's y/z = 4-row mechanism does not appear among
// Test 7's asserted cells). The expectations (correctF2Crho replica,
// marker IC) are window-independent and UNCHANGED; measured re-shuffles
// DF 2.384e-07 -> 2.146e-06 and macros 9.537e-07 -> 7.629e-06 stay ~100x
// below the gates, and the protected-class assertions (GEO_WALL/GEO_FLUID
// kept, max |err| = 0.000e+00) are clamp-independent -- the Defect-2
// 4-class guard this test locks is untouched by the window clamp. The
// Test 5 halo-geography note applies: production never F2C-launches halo
// (ring) cells (the ring path is deleted); the mock pins the kernel
// directly.]
// [T15: STRATEGY-INDEPENDENT -- the allowed-GEO store predicate guarded
// here is verbatim-shared by both F2C branches of cudaAMR_FineToCoarse
// (amr_coupling.h is_coupling_cell), and the receiving-cell expectations
// use the branch-tolerant linear-marker value (see correctF2Crho), so
// this single lock covers both strategies; no duplicated guard case for
// the F2C_SCHONHERR arm is needed (dedupe audit, commit 14)]
void test_f2c_df_store_map_guard()
{
	// post-stream natural orientation on the fine level, spatial (twisted)
	// coarse consumer -- same orientation state as the Test 5 F2C phase
	const bool fine_even_iter = false;
	const bool next_coarse_even_iter = false;

	// processed halo column and the exercised map classes in it
	constexpr idx CX = 3, CZ = 8;
	constexpr idx Y_WALL = 8, Y_FLUID = 9, Y_NOTHING = 10, Y_INTERFACE = 11;

	MockBlock coarse, fine;
	coarse.allocate(NEST_N, NEST_OV);
	fine.allocate(NEST_N, NEST_OV);
	fillMarkerNested(coarse, false, false);
	fillMarkerNested(fine, true, fine_even_iter);

	// one cell per map class in the processed column
	coarse.hmap(CX, Y_WALL, CZ) = NSE_CONFIG::BC::GEO_WALL;
	coarse.hmap(CX, Y_FLUID, CZ) = NSE_CONFIG::BC::GEO_FLUID;
	coarse.hmap(CX, Y_NOTHING, CZ) = NSE_CONFIG::BC::GEO_NOTHING;
	coarse.hmap(CX, Y_INTERFACE, CZ) = NSE_CONFIG::BC::GEO_AMR_INTERFACE;
	coarse.dmap = coarse.hmap;

	// poison the macro array so untouched macros are recognizable after
	// the launch
	constexpr dreal MACRO_SENTINEL = -123;
	coarse.hmacro.setValue(MACRO_SENTINEL);
	coarse.dmacro = coarse.hmacro;

	coarse.copyToDevice();
	fine.copyToDevice();

	const idx3d fine_off{NEST_FINE_OFF, NEST_FINE_OFF, NEST_FINE_OFF};
	const idx3d coarse_off{0, 0, 0};
	launchFineToCoarse(coarse, fine, {2, 3, 3}, {4, 13, 13}, fine_off, coarse_off, fine_even_iter, next_coarse_even_iter);

	coarse.copyToHost();
	coarse.hmacro = coarse.dmacro;

	// marker IC of the coarse column (rho of coarse cell c = c + NEST_RHO0)
	// and the exact F2C transfer result (host replica from Test 5)
	const std::array<dreal, 27> eq_marker = equilibriumOnHost(static_cast<dreal>(CX) + NEST_RHO0, NEST_VX, 0, 0);
	const dreal rho_transfer = correctF2Crho(CX, 0, NEST_FINE_OFF);
	const std::array<dreal, 27> eq_transfer = equilibriumOnHost(rho_transfer, NEST_VX, 0, 0);

	// DF check per map class: protected cells must still hold the marker
	// IC in the kernel's write slot of every direction, coupling cells
	// must hold the transfer result there
	const idx case_ys[4] = {Y_WALL, Y_FLUID, Y_NOTHING, Y_INTERFACE};
	const bool case_write[4] = {false, false, true, true};
	const char* case_names[4] = {"GEO_WALL", "GEO_FLUID", "GEO_NOTHING", "GEO_AMR_INTERFACE"};
	for (int cse = 0; cse < 4; cse++) {
		const idx y = case_ys[cse];
		const bool expect_write = case_write[cse];
		double max_err = 0;
		idx bad = 0;
		bool first_mismatch = true;
		for (int q = 0; q < 27; q++) {
			// the kernel's write slot of direction q; the marker IC stored
			// eq of direction `slot` in that slot (natural fill)
			const int slot = coarseWriteSlot(q, next_coarse_even_iter);
			const dreal actual = coarse.hfs[f2cWriteArray()](slot, CX, y, CZ);
			const dreal expected = expect_write ? eq_transfer[q] : eq_marker[slot];
			if (! closeEnough(actual, expected, 1e-4, 1e-5)) {
				if (first_mismatch) {
					fmt::println("  first mismatch: cell=({}, {}, {}), q={}, actual={:.9e}, expected={:.9e}", CX, y, CZ, q, actual, expected);
					first_mismatch = false;
				}
				bad++;
			}
			max_err = std::max<double>(max_err, std::abs(actual - expected));
		}
		report(
			bad == 0,
			fmt::format(
				"Test 7 F2C DF-store map guard: {} cell {} (max |err| = {:.3e})",
				case_names[cse],
				expect_write ? "received the transfer" : "kept its DFs (not overwritten)",
				max_err
			)
		);
	}

	// macro check under the same predicate: protected cells keep the
	// sentinel, coupling cells hold the transfer macros
	const int macro_ids[4] = {NSE_CONFIG::MACRO::e_rho, NSE_CONFIG::MACRO::e_vx, NSE_CONFIG::MACRO::e_vy, NSE_CONFIG::MACRO::e_vz};
	for (int cse = 0; cse < 4; cse++) {
		const idx y = case_ys[cse];
		const bool expect_write = case_write[cse];
		const std::array<dreal, 4> expected =
			expect_write ? std::array<dreal, 4>{rho_transfer, NEST_VX, 0, 0} :
							std::array<dreal, 4>{MACRO_SENTINEL, MACRO_SENTINEL, MACRO_SENTINEL, MACRO_SENTINEL};
		double max_err = 0;
		idx bad = 0;
		bool first_mismatch = true;
		for (int m = 0; m < 4; m++) {
			const dreal actual = coarse.hmacro(macro_ids[m], CX, y, CZ);
			if (! closeEnough(actual, expected[m], 1e-4, 1e-5)) {
				if (first_mismatch) {
					fmt::println(
						"  first macro mismatch: cell=({}, {}, {}), id={}, actual={:.9e}, expected={:.9e}", CX, y, CZ, macro_ids[m], actual, expected[m]
					);
					first_mismatch = false;
				}
				bad++;
			}
			max_err = std::max<double>(max_err, std::abs(actual - expected[m]));
		}
		report(
			bad == 0,
			fmt::format(
				"Test 7 F2C macro-store map guard: {} macros {} (max |err| = {:.3e})",
				case_names[cse],
				expect_write ? "written by the transfer" : "kept the sentinel",
				max_err
			)
		);
	}
}

// Tests 8/9: compact-moment (CM; the default C2F branch since the
// 2026-08-18 flip) exactness -- Experiment A item 2 (adjudicator of the A.2
// TG smoke blow-up: synthetic-field exactness separates kernel-math bugs
// from coupling physics). (Pre-flip note: before the 2026-08-18 default
// flip, the default build compiled this whole block out; the mirror
// conditions below now include it whenever the production branch is CM --
// including the default build.) The former carve exactness tests (Tests
// 10-13/17) were removed on 2026-08-23 together with the kernel's carve
// pre-pass: under the ch7 band registration every nominal source window of
// a valid coupling straddles live GEO_AMR_INTERFACE cells only (asserted
// statically by checkCouplingMapPattern at SimInit), so a covered-source
// map is an invalid registration, not a runtime case.
//
// EXACTNESS CLASS (from the CM machinery, amr_coupling.h:321-334): the
// 8-coefficient density polynomial is the plain trilinear fit of the 8 nodal
// values (no moment corrections), so it is exact only for LINEAR density.
// The three 11-coefficient velocity polynomials additionally carry the
// second-order non-equilibrium moments (the k values = strain combos when
// the source DFs carry the Chapman-Enskog non-equilibrium), which supply
// the curvature information that the 2x2x2 nodal values alone cannot
// distinguish from a constant: velocity is exact for LINEAR and PURE
// QUADRATIC fields (quadratic in the separate coordinates, no cross terms)
// under CE-consistent source states. A pure quadratic DENSITY is outside
// the exactness class by construction (the trilinear fit absorbs nodal
// quadratics into the constant), so quadratic-rho exactness is NOT asserted
// anywhere below -- density stays linear or constant in every test field.
//
// CE-CONSISTENT SOURCE FILL: each coarse cell carries
//   f_q = eq_q(rho,u) - [rho / (3*omega_s)] * [w_q / (2*cs^4)] * Q_ab(q)*G_ab,
// with Q_ab = c_qa*c_qb - cs^2*delta_ab, cs^2 = 1/3, omega_s = 1/tau_coarse,
// G_ab = du_b/dx_a + du_a/dx_b evaluated at the coarse cell center. The
// f_neq term has zero zeroth and first moments (isotropy of the D3Q27
// weights), so the macros of the fill are exactly (rho,u), and its second
// moment equals -(rho/(3*omega_s))*G_ab, i.e. the kernel's k_xy/k_xx_yy/...
// combinations evaluate to the strain field derivatives the polynomial fit
// expects (verified by hand against Steps B-D of the CM branch: for linear
// fields a_0/a_x/a_y/... all coincide with the analytic coefficients, for
// pure quadratic velocity fields additionally a_xx/a_yy/a_xx = the
// curvature).
//
// GEOMETRY (Tests 8/9): coarse 8^3 with 1-cell overlap, storage [-1,9);
// fine 16^3; offsets zero; fine cell center = fg*0.5-0.25 in coarse
// indexer coords. Launch fg in [2,15)^3 -> nominal windows subset of
// [0,8], never clamped, |t_rel| = 0.25 everywhere.
// Covered cells' DFs are poisoned with NaN in every array/slot: any read of
// a covered cell contaminates the output with NaN and fails the finiteness
// assertion ("no covered data contaminates the output").
//
// EXISTING TESTS: untouched; they pin the default Lagrange path and are
// compiled identically under the defines (Tests 1-7 pass unchanged in the
// a1/a2 builds, verified by the run logs).
// B.5 hoist: the six helpers below (d3q27Weight, fillFieldCE,
// poisonCellDFs, fineGhostMacros, checkFineMacrosExact, tagNothingCells)
// are shared by the CM exactness tests (Tests 8/9, define-gated below)
// and the F2C skin tests (Tests 14-16 + 18). Unconditional since D.1
// retired F2C_SKIN_ONLY (gate B ruling): the skin tests now compile in
// every build.

// D3Q27 lattice weight of direction q (product weights: 8/27 rest, 2/27
// axis, 1/54 face diagonal, 1/216 corner)
dreal d3q27Weight(int q)
{
	const int l2 = VELOCITY[q][0] * VELOCITY[q][0] + VELOCITY[q][1] * VELOCITY[q][1] + VELOCITY[q][2] * VELOCITY[q][2];
	if (l2 == 0)
		return dreal(8) / dreal(27);
	if (l2 == 1)
		return dreal(2) / dreal(27);
	if (l2 == 2)
		return dreal(1) / dreal(54);
	return dreal(1) / dreal(216);
}

// fill the whole stored extent of the coarse mock block with the
// CE-consistent DF state f_eq + f_neq of the field returned by
// `FIELD::fill(x,y,z) = {rho, vx, vy, vz, Gxx, Gyy, Gzz, Gxy, Gxz, Gyz}`
// (see the block comment for the f_neq construction)
template <typename FIELD>
void fillFieldCE(MockBlock& block, bool even_iter, const FIELD& field)
{
	constexpr dreal cs2 = dreal(1) / dreal(3);
	const dreal omega_s = dreal(1) / TAU_COARSE;
	for (idx z = -block.ov; z < block.size + block.ov; z++) {
		for (idx y = -block.ov; y < block.size + block.ov; y++) {
			for (idx x = -block.ov; x < block.size + block.ov; x++) {
				const std::array<dreal, 10> fld = field.fill(x, y, z);
				const dreal rho = fld[0];
				const std::array<dreal, 27> eq = equilibriumOnHost(rho, fld[1], fld[2], fld[3]);
				for (int q = 0; q < 27; q++) {
					const dreal qx = VELOCITY[q][0], qy = VELOCITY[q][1], qz = VELOCITY[q][2];
					const dreal QG = (qx * qx - cs2) * fld[4] + (qy * qy - cs2) * fld[5] + (qz * qz - cs2) * fld[6]
								   + 2 * qx * qy * fld[7] + 2 * qx * qz * fld[8] + 2 * qy * qz * fld[9];
					const dreal f_neq = -(rho / (3 * omega_s)) * (d3q27Weight(q) / (2 * cs2 * cs2)) * QG;
					storePostCollisionDF(block, even_iter, q, x, y, z, eq[q] + f_neq);
				}
			}
		}
	}
}

// poison one coarse cell's DFs with NaN in every array/slot (covered-cell
// contamination trap: any kernel read of this cell lands NaN in the output)
void poisonCellDFs(MockBlock& block, idx x, idx y, idx z)
{
	const dreal nan = std::numeric_limits<dreal>::quiet_NaN();
	for (uint8_t dfty = 0; dfty < DFMAX; dfty++)
		for (int q = 0; q < 27; q++)
			block.hfs[dfty](q, x, y, z) = nan;
}

// macros (rho, u, v, w) of the DF state the C2F fill wrote at a fine cell
void fineGhostMacros(const MockBlock& fine, idx x, idx y, idx z, dreal& rho, dreal& u, dreal& v, dreal& w)
{
	rho = 0;
	dreal jx = 0, jy = 0, jz = 0;
	for (int q = 0; q < 27; q++) {
		const dreal f = fine.hfs[df_cur](c2fWriteSlot(q), x, y, z);
		rho += f;
		jx += VELOCITY[q][0] * f;
		jy += VELOCITY[q][1] * f;
		jz += VELOCITY[q][2] * f;
	}
	u = jx / rho;
	v = jy / rho;
	w = jz / rho;
}

// assert that the reconstructed macros of every fine cell in
// [begin,end) equal the analytic field `FIELD::exact(X,Y,Z) = {rho,vx,vy,vz}`
// (evaluated at the fine cell center in coarse indexer coords,
// X = fx*0.5-0.25), and that every value is finite (NaN-poison guard)
template <typename FIELD>
bool checkFineMacrosExact(const MockBlock& fine, idx3d begin, idx3d end, const FIELD& field, dreal rtol, dreal atol, const char* what)
{
	double max_rel_rho = 0, max_abs_u = 0;
	idx bad = 0;
	bool first_mismatch = true;
	for (idx z = begin.z(); z < end.z(); z++) {
		for (idx y = begin.y(); y < end.y(); y++) {
			for (idx x = begin.x(); x < end.x(); x++) {
				dreal rho_m, u_m, v_m, w_m;
				fineGhostMacros(fine, x, y, z, rho_m, u_m, v_m, w_m);
				const std::array<double, 4> e = field.exact(x * 0.5 - 0.25, y * 0.5 - 0.25, z * 0.5 - 0.25);
				const bool finite = std::isfinite(rho_m) && std::isfinite(u_m) && std::isfinite(v_m) && std::isfinite(w_m);
				const double rel_rho = std::abs(rho_m - e[0]) / e[0];
				const double abs_du = std::max({std::abs(u_m - e[1]), std::abs(v_m - e[2]), std::abs(w_m - e[3])});
				max_rel_rho = std::max(max_rel_rho, rel_rho);
				max_abs_u = std::max(max_abs_u, abs_du);
				const bool ok = finite && closeEnough(rho_m, static_cast<dreal>(e[0]), rtol, atol) && abs_du <= atol + rtol * 0.03;
				if (! ok) {
					if (first_mismatch) {
						fmt::println(
							"  first mismatch: cell=({},{},{}), finite={}, rho={:.9e} (expected {:.9e}), "
							"u=({:.9e},{:.9e},{:.9e}) (expected {:.9e},{:.9e},{:.9e})",
							x, y, z, finite, rho_m, e[0], u_m, v_m, w_m, e[1], e[2], e[3]
						);
						first_mismatch = false;
					}
					bad++;
				}
			}
		}
	}
	report(bad == 0, fmt::format("{}: reconstructed fine macros match the analytic field (max rel rho err = {:.3e}, max abs vel err = {:.3e})", what, max_rel_rho, max_abs_u));
	return bad == 0;
}

// tag a coarse-cell list as GEO_NOTHING and upload the map (MockBlock maps
// default to GEO_FLUID, see tagCouplingCells): a tagged cell is NaN-poison
// proving the C2F window solve never reads outside the nominal window
// (Tests 8/9), and the F2C allowed-GEO store guard lets GEO_NOTHING cells
// RECEIVE skin writes (the Tests 7/16 guard-matrix class cells)
void tagNothingCells(MockBlock& block, const std::vector<idx3d>& cells)
{
	for (const idx3d& c : cells)
		block.hmap(c.x(), c.y(), c.z()) = NSE_CONFIG::BC::GEO_NOTHING;
	block.dmap = block.hmap;
}

#if defined(C2F_COMPACT_MOMENT) || (!defined(C2F_LAGRANGE) && !defined(C2F_TRILINEAR) && !defined(C2F_LINEAR_EXPLOSION) && !defined(C2F_UNIFORM_EXPLOSION))

// analytic fields (see the block comment for the exactness class): every
// struct provides `fill(x,y,z) -> {rho,vx,vy,vz,Gxx,Gyy,Gzz,Gxy,Gxz,Gyz}`
// in dreal (coarse cell centers = integer indexer coords) and
// `exact(X,Y,Z) -> {rho,vx,vy,vz}` in double at arbitrary coordinates.

// Test 8 field: everything linear in all three coordinates
struct CMLinearField {
	std::array<dreal, 10> fill(idx x, idx y, idx z) const
	{
		const std::array<double, 4> e = exact(x, y, z);
		// constant strain: Gij from the linear velocity gradients
		return {static_cast<dreal>(e[0]), static_cast<dreal>(e[1]), static_cast<dreal>(e[2]), static_cast<dreal>(e[3]),
			dreal(2 * 0.002), dreal(2 * 0.0018), dreal(2 * 0.0014),										// Gxx, Gyy, Gzz
			dreal(0.001 + (-0.0015)), dreal(-0.0009 + 0.001), dreal(0.0011 + (-0.0012))};					// Gxy, Gxz, Gyz
	}
	std::array<double, 4> exact(double X, double Y, double Z) const
	{
		return {
			1.0 + 0.01 * X - 0.008 * Y + 0.006 * Z,
			0.03 + 0.002 * X - 0.0015 * Y + 0.001 * Z,
			-0.02 + 0.001 * X + 0.0018 * Y - 0.0012 * Z,
			0.015 - 0.0009 * X + 0.0011 * Y + 0.0014 * Z,
		};
	}
};

// Test 9 field: linear density, velocities linear + pure quadratic in each
// coordinate (no cross terms, inside the CM exactness class)
struct CMQuadraticField {
	std::array<dreal, 10> fill(idx x, idx y, idx z) const
	{
		const std::array<double, 4> e = exact(x, y, z);
		const double dudx = 0.002 + 2 * 0.0008 * x, dudy = -0.0015 - 2 * 0.0006 * y, dudz = 0.001 + 2 * 0.0004 * z;
		const double dvdx = 0.001 - 2 * 0.0007 * x, dvdy = 0.0018 + 2 * 0.0005 * y, dvdz = -0.0012 + 2 * 0.0003 * z;
		const double dwdx = -0.0009 + 2 * 0.0006 * x, dwdy = 0.0011 - 2 * 0.0005 * y, dwdz = 0.0014 + 2 * 0.00045 * z;
		return {static_cast<dreal>(e[0]), static_cast<dreal>(e[1]), static_cast<dreal>(e[2]), static_cast<dreal>(e[3]),
			static_cast<dreal>(2 * dudx), static_cast<dreal>(2 * dvdy), static_cast<dreal>(2 * dwdz),
			static_cast<dreal>(dvdx + dudy), static_cast<dreal>(dwdx + dudz), static_cast<dreal>(dwdy + dvdz)};
	}
	std::array<double, 4> exact(double X, double Y, double Z) const
	{
		return {
			1.0 + 0.01 * X - 0.008 * Y + 0.006 * Z,
			0.03 + 0.002 * X - 0.0015 * Y + 0.001 * Z + 0.0008 * X * X - 0.0006 * Y * Y + 0.0004 * Z * Z,
			-0.02 + 0.001 * X + 0.0018 * Y - 0.0012 * Z - 0.0007 * X * X + 0.0005 * Y * Y + 0.0003 * Z * Z,
			0.015 - 0.0009 * X + 0.0011 * Y + 0.0014 * Z + 0.0006 * X * X - 0.0005 * Y * Y + 0.00045 * Z * Z,
		};
	}
};


// covered-plane helper for Tests 8/9 (far plane x=-1): tag + NaN-poison a
// y-z plane of coarse cells
void coverPlaneYZ(MockBlock& coarse, idx cx, idx lo, idx hi)
{
	std::vector<idx3d> cells;
	for (idx z = lo; z <= hi; z++)
		for (idx y = lo; y <= hi; y++)
			cells.push_back({cx, y, z});
	for (const idx3d& c : cells)
		poisonCellDFs(coarse, c.x(), c.y(), c.z());
	tagNothingCells(coarse, cells);
}

// Tests 8 and 9: nominal-window CM exactness -- CE-consistent linear (8) and
// linear-rho + pure-quadratic-velocity (9) fields, launch over fg in
// [2,15)^3: every nominal window is inside [0,8], never clamped (|t_rel| =
// 0.25 everywhere). A GEO_NOTHING plane at x = -1 (outside every candidate
// window, NaN-poisoned) proves the window solve never reads outside the
// nominal window. The reconstructed macros must match the analytic field at
// every fine cell center to fp tolerance.
void test_cm_exactness_nominal()
{
	const bool even_iter = false;

	struct Case
	{
		const char* name;
		bool quadratic;
	};
	const Case cases[2] = {{"linear field", false}, {"linear rho + pure quadratic velocity field", true}};

	for (const Case& cse : cases) {
		MockBlock coarse, fine;
		coarse.allocate(COARSE_N);
		fine.allocate(FINE_N);
		if (cse.quadratic)
			fillFieldCE(coarse, even_iter, CMQuadraticField{});
		else
			fillFieldCE(coarse, even_iter, CMLinearField{});
		// far-away covered plane: inert for every candidate window
		coverPlaneYZ(coarse, -1, -1, COARSE_N);
		coarse.copyToDevice();

		launchCoarseToFine(fine, coarse, {2, 2, 2}, {15, 15, 15}, {0, 0, 0}, {0, 0, 0}, even_iter);
		fine.copyToHost();

		if (cse.quadratic)
			checkFineMacrosExact(
				fine, {2, 2, 2}, {15, 15, 15}, CMQuadraticField{}, 1e-4, 1e-6, fmt::format("Test 9 CM nominal-window exactness ({}; far-away GEO_NOTHING inert)", cse.name).c_str()
			);
		else
			checkFineMacrosExact(
				fine, {2, 2, 2}, {15, 15, 15}, CMLinearField{}, 1e-4, 1e-6, fmt::format("Test 8 CM nominal-window exactness ({}; far-away GEO_NOTHING inert)", cse.name).c_str()
			);
	}
}

#endif	// CM semantics active

// Tests 14-16 + 18: F2C skin-launch coverage (changes 2+3 of the AMR
// interface redesign, unconditional since D.1 retired the ring path) --
// Experiment B item 5 (B.5, D.2). Production launches
// cudaAMR_FineToCoarse over the 6 disjoint inset-face SKIN rectangles of
// each fine footprint (the depth-1 shell of the frozen GEO_NOTHING
// region one coarse row inside the reactivated c=0 ring row,
// amr_state.h buildCouplings); the pre-B.5 mock suite had NO
// interior-F2C geography coverage at all (the NEST cases cover C2F ghost
// faces and the former ring-F2C halo faces only). These tests hand-emulate
// production skin launches with the basic mock fixture (coarse 8^3, fine
// 16^3, OV = 1, zero offsets as Tests 1-4/6): the footprint is the coarse
// block interior [0,8)^3 exactly covered by the fine block, and the
// production fine block covering a footprint at go = (0,0,0) has fine
// offset 2*go = (0,0,0), so the zero-offset mock reproduces the production
// skin window positions EXACTLY (fx0 = 2x per axis; depth-1 face cells sit
// at fx0 = 2).
//
// [commit-7 position re-anchor (schonherr-ch7 conversion, plan T5): the
// launch rectangles moved from the footprint-SURFACE row (x = 0) to the
// depth-1 skin row (x = 1) to mirror the production skin of
// amr_state.h buildCouplings, and the destination rows are tagged
// GEO_NOTHING (their production band class at depth 1; the F2C
// allowed-GEO store guard admits both coupling classes). The normal-axis
// window at depth 1 is NOMINAL (fx0 = 2 -> {1,2,3,4}, no clamp possible
// in production): the lo = 0 clamp's coverage now lives exclusively on
// the tangent edges of the launch rectangles (the Tests 15/18 probes
// rewrite, below). All value-bitwise expectations are unchanged-cell
// analytic (any consistent 4-node window projects the cubic exactly at
// the fixed evaluation point).]
//
// WINDOW/WEIGHT TABLE (x-min skin face launch {1}x{0..8}x{0..8}, the
// production x-min depth-1 rectangle of this footprint):
//   every cell: x-window nominal {1,2,3,4} (fx0 = 2) at the FIXED
//     evaluation point t = fx0 + 0.5 = 2.5 -- the skin sits one coarse
//     row inside the reactivated c=0 ring, so the face-normal lower-
//     bound clamp can no longer engage in production;
//   tangent cells with y,z in {1..6}: y/z windows nominal, start >= 1 --
//     a LOWER-bound-only clamp cannot engage ("the clamp changes nothing
//     away from edges"; the nominal code path, sharing Test 6's interior
//     fp class 1.192e-07);
//   tangent cells with y or z == 0: that axis's window clamps to {0,1,2,3};
//   tangent cells with y or z == 7: hi-side window {13,14,15,16} INCLUDING
//     the fine ghost node 16 (the upper bound was never loosened) -- the
//     lo-only clamp's documented upper-bound asymmetry (max-side windows
//     still read the C2F-filled ghost; analytically filled here, so exact).
//   Test 15's y-min/z-min launches mirror the same table per axis.
//
// TOLERANCE CLASS: additive-separable cubic density (any 4-node window
// reproduces it exactly at the fixed evaluation point), constant
// velocities, zero strain (the CE fill reduces to the equilibrium). The
// expectations are ANALYTIC: the strategy-split destination density of
// skinExpectedRho [T15, commit 14] -- on the Lagrava (opt-out) branch the
// coarse-center field value, on the F2C_SCHONHERR arm the subcell mean
// d0 (mean-density transfer, NO conservation claim); the branches' values
// disagree by O(1e-3) on this marker, so exactly one holds per build.
// Gates rtol = 1e-5 / atol = 1e-6 separate the fp-exact class (Test 6
// measured 1.192e-07) from the box average (measured 1.7e-03 there) and
// from any shortened or otherwise mismachined window (~1e-03 for this
// marker) by ~2 decades. Test 15 additionally makes a lower-bound
// REGRESSION positively detectable via sentinel-poisoned ghost planes.
//
// [T15 strategy split (commit 14 / plan row 15): the WINDOW/WEIGHT TABLE
// above is LAGRAVA machinery (the F2C_SCHONHERR arm has no window -- it
// reads the destination cell's own 8 subcells, so every launch-rectangle
// cell is treated uniformly and the clamp classes below do not exist
// there). Tests 14/16 stay live on BOTH branches via the strategy-split
// expectation; the lo = 0 clamp sentinels (Tests 15/18) are pinned ON
// the Lagrava (opt-out) branch as its authority -- #ifndef-gated with an
// explicit deferral report on the arm, retire nothing silently.]

// skin-test field: rho = 1 + 0.1*(dx^3 + dy^3 + dz^3) with dx = (x - 7.5)/4
// in fine indexer coordinates (Test 6's cubic, extended separably to all
// axes so every axis's window machinery is load-bearing for exactness);
// constant velocities, zero velocity gradient (G == 0 -> f_neq == 0, so the
// projected DFs are exactly the equilibrium of the projected macros)
struct SkinCubicField {
	std::array<dreal, 10> fill(idx x, idx y, idx z) const
	{
		return {
			static_cast<dreal>(rhoAt(x, y, z)), U0, V0, W0, dreal(0), dreal(0), dreal(0), dreal(0), dreal(0), dreal(0)
		};
	}

	std::array<double, 4> exact(double X, double Y, double Z) const
	{
		return {rhoAt(X, Y, Z), U0, V0, W0};
	}

	static double rhoAt(double x, double y, double z)
	{
		const double dx = (x - 7.5) / 4.0, dy = (y - 7.5) / 4.0, dz = (z - 7.5) / 4.0;
		return 1.0 + 0.1 * (dx * dx * dx + dy * dy * dy + dz * dz * dz);
	}

	static constexpr dreal U0 = dreal(0.01);
	static constexpr dreal V0 = dreal(-0.02);
	static constexpr dreal W0 = dreal(0.03);
};

// overwrite one negative fine ghost plane (axis 0/1/2 at index -1) with a
// large sentinel on every DF array/slot: under the lo = 0 clamp no skin
// window may read these nodes, so a lower-bound REGRESSION is positively
// detected -- a read shifts the transfer result by O(10..100), decades
// above the fp-exactness gates (fillMarkerNested's wrong-array sentinel
// idiom from Test 5)
void sentinelFineGhostPlane(MockBlock& fine, int axis)
{
	constexpr dreal SENTINEL = 1000;
	for (uint8_t dfty = 0; dfty < DFMAX; dfty++)
		for (int q = 0; q < 27; q++)
			for (idx b = -fine.ov; b < fine.size + fine.ov; b++)
				for (idx a = -fine.ov; a < fine.size + fine.ov; a++) {
					if (axis == 0)
						fine.hfs[dfty](q, -1, a, b) = SENTINEL;
					else if (axis == 1)
						fine.hfs[dfty](q, a, -1, b) = SENTINEL;
					else
						fine.hfs[dfty](q, a, b, -1) = SENTINEL;
				}
}

// T15 (commit 14, plan row 15): strategy-split expected destination
// density at a skin-launch cell for the SkinCubicField marker --
//   - F2C_SCHONHERR arm: MEAN-DENSITY TRANSFER (the T4a successor; NO
//     conservation claim) -- d0 == the mean of the destination cell's own
//     8 subcell densities (the t = (0,0,0) evaluation of the sec. 7.2
//     F2C; the constant velocities carry through exactly as a0/b0/c0 and
//     the zero-strain fill gives vanishing non-equilibrium moments, so
//     the destination DF state is the equilibrium of (d0, U0, V0, W0);
//     the L5 lock of tests/test_amr_f2c_schonherr.cu pins the sum-DF
//     identity on the dedicated-suite geography -- dedupe: same machinery
//     class, this helper owns the production skin-launch geography).
//   - Lagrava (opt-out) branch: the analytically projected coarse-center
//     value of the cubic marker (window-independent in exact arithmetic).
dreal skinExpectedRho(const idx3d& c)
{
#ifdef F2C_SCHONHERR
	dreal mean = 0;
	for (int bz = 0; bz < 2; bz++)
		for (int by = 0; by < 2; by++)
			for (int bx = 0; bx < 2; bx++)
				mean += static_cast<dreal>(SkinCubicField::rhoAt(2 * c.x() + bx, 2 * c.y() + by, 2 * c.z() + bz));
	return mean / 8;
#else
	return static_cast<dreal>(SkinCubicField::rhoAt(2 * c.x() + 0.5, 2 * c.y() + 0.5, 2 * c.z() + 0.5));
#endif
}

// Test-6-style per-cell assertion of the DFs the F2C transfer wrote in its
// parity-dependent write slot against the equilibrium of the
// strategy-split expected skin-test field (skinExpectedRho; see the block
// comment)
bool checkCoarseTransferExact(const MockBlock& coarse, const std::vector<idx3d>& cells, bool coarse_even_iter, const char* what)
{
	double max_err = 0;
	idx bad = 0;
	bool first_mismatch = true;
	for (const idx3d& c : cells) {
		const std::array<dreal, 27> eq = equilibriumOnHost(skinExpectedRho(c), SkinCubicField::U0, SkinCubicField::V0, SkinCubicField::W0);
		for (int q = 0; q < 27; q++) {
			const dreal actual = coarse.hfs[f2cWriteArray()](coarseWriteSlot(q, coarse_even_iter), c.x(), c.y(), c.z());
			if (! (std::isfinite(actual) && closeEnough(actual, eq[q], 1e-5, 1e-6))) {
				if (first_mismatch) {
					fmt::println("  first mismatch: cell=({},{},{}), q={}, actual={:.9e}, expected={:.9e}", c.x(), c.y(), c.z(), q, actual, eq[q]);
					first_mismatch = false;
				}
				bad++;
			}
			max_err = std::max<double>(max_err, std::abs(actual - eq[q]));
		}
	}
	report(bad == 0, fmt::format("{}: written DFs match the analytically projected cubic field (max |err| = {:.3e})", what, max_err));
	return bad == 0;
}

// Test 14: skin-launch exactness, interior -- the x-min launch rectangle
// {1}x{0..8}x{0..8} over the depth-1 skin destination row tagged
// GEO_NOTHING (its production band class after the commit-7 position
// re-anchor, see the Tests 14-16/18 block comment for the window table).
// Asserts cubic exactness on all 64 face cells under the T15 strategy
// split (skinExpectedRho): on the Lagrava (opt-out) branch the normal
// window is nominal {1,2,3,4} at depth 1, tangent-interior axes share the
// nominal window semantics (a lower-bound clamp cannot engage away from
// edges -- mock-matrix.md's coupling case-1 class made a POSITIVE skin
// test), and the lo-/hi-edge tangent cells stay exact on their shifted
// windows; on the F2C_SCHONHERR arm every launched cell instead satisfies
// the mean-density transfer (destination density == the own-8 subcell
// mean d0; NO conservation claim) with uniform window-free treatment. The
// GEO_NOTHING tagging keeps the mock's destination row in the production
// class of its band position (the allowed-GEO guard admits it; the
// protected-class corners live in the Tests 7/16 class cells) -- the
// coarse block is pre-filled with the (rho = 1, v = 0) placeholder, so a
// skipped write fails the gate.
void test_f2c_skin_exactness_interior()
{
	const std::array<bool, 2> parities = {true, false};

	for (const bool fine_even_iter : parities) {
		for (const bool coarse_even_iter : parities) {
			MockBlock coarse, fine;
			coarse.allocate(COARSE_N);
			fine.allocate(FINE_N);
			fillFieldCE(fine, fine_even_iter, SkinCubicField{});
			fine.copyToDevice();
			fillUniform(coarse, true, 1.0, 0.0, 0.0, 0.0);

			// the depth-1 skin destination row of the production x-min
			// face (commit-7 position re-anchor): GEO_NOTHING at
			// surface-depth 1 -- the F2C allowed-GEO store guard admits
			// frozen coupling cells to receive the skin writes
			std::vector<idx3d> face;
			for (idx z = 0; z < COARSE_N; z++)
				for (idx y = 0; y < COARSE_N; y++) {
					face.push_back({1, y, z});
					coarse.hmap(1, y, z) = NSE_CONFIG::BC::GEO_NOTHING;
				}
			coarse.dmap = coarse.hmap;
			coarse.copyToDevice();

			launchFineToCoarse(
				coarse, fine, {1, 0, 0}, {2, COARSE_N, COARSE_N}, {0, 0, 0}, {0, 0, 0}, fine_even_iter, coarse_even_iter
			);
			coarse.copyToHost();

			checkCoarseTransferExact(
				coarse,
				face,
				coarse_even_iter,
				fmt::format(
					"Test 14 skin x-min face F2C exactness (fine_even={}, coarse_even={}) [{}]: all 64 GEO_NOTHING depth-1 cells received the transfer",
					fine_even_iter,
					coarse_even_iter,
					f2c_strategy_name
				)
					.c_str()
			);
		}
	}
}

// Test 15: footprint-lo-EDGE clamp exactness with a POSITIVE ghost-read
// detector [LAGRAVA (OPT-OUT) BRANCH ONLY -- T15, commit 14: the lo = 0
// clamp is machinery of the Lagrava axis_window and has NO counterpart on
// the F2C_SCHONHERR arm, which reads only the destination cell's own 8
// subcells (there is no window to shift, so nothing to clamp). This lock
// is the opt-out authority for the clamp and stays ON the Lagrava branch
// -- retire nothing silently; the arm build reports an explicit deferral
// line below. The storability guard that bounds subcell reads on the arm
// is strategy-independent and stays locked by Tests 5/14/16.] -- the
// three min-face launch rectangles of the depth-1 skin row (commit-7
// position re-anchor: x-min {1} over the full y/z range, y-min {1} over
// the launch x extent, z-min {1} over the launch x/y extent, tagged
// GEO_NOTHING -- their production band class at depth 1; production's
// corner ownership by the x-faces is immaterial here since a re-write
// would be idempotent), are launched
// while the negative fine ghost planes x = y = z = -1 carry the +1000
// sentinel on every DF array/slot. Under the lo = 0 clamp every nominal
// window
// {f0-1,...,f0+2} with f0 == 0 becomes {0,1,2,3} and never touches the
// sentinel planes; a lower-bound REGRESSION reads a sentinel node and
// shifts the transfer by O(10..100), decades above the gates. Exactness at
// the probes therefore proves the clamp CHOSE the shifted window with the
// shared weight machinery (cubic-exact at the fixed evaluation point --
// same machinery class as Test 6), not a degenerate shorten and not the
// box average. At depth 1 the face-normal window is nominal (fx0 = 2)
// everywhere, so the clamp's coverage lives on the TANGENT edges of the
// launch rectangles (probes rewritten accordingly); the surface era's
// three-clamped corner class is extinct (one of the four old probes is
// replaced by the two-clamped maximum now possible).
void test_f2c_skin_edge_clamp_exactness()
{
#ifdef F2C_SCHONHERR
	// T15: no clamp machinery on the F2C_SCHONHERR arm -- this lock is the
	// Lagrava (opt-out) branch's authority and is not duplicated (dedupe
	// audit); the explicit deferral line keeps the arm's report truthful
	report(
		true,
		"Test 15 depth-1 lo-edge clamp exactness [F2C_SCHONHERR arm]: N/A -- clamp is Lagrava-only window machinery; authority lives on the Lagrava (opt-out) branch (retired nothing)"
	);
#else
	const bool coarse_even_iter = false;

	for (const bool fine_even_iter : {true, false}) {
		MockBlock coarse, fine;
		coarse.allocate(COARSE_N);
		fine.allocate(FINE_N);
		fillFieldCE(fine, fine_even_iter, SkinCubicField{});
		sentinelFineGhostPlane(fine, 0);
		sentinelFineGhostPlane(fine, 1);
		sentinelFineGhostPlane(fine, 2);
		fine.copyToDevice();
		fillUniform(coarse, true, 1.0, 0.0, 0.0, 0.0);

		// the three min-face depth-1 skin destination rows (commit-7
		// position re-anchor), tagged GEO_NOTHING (the F2C allowed-GEO
		// store guard admits frozen coupling cells to receive the writes)
		for (idx z = 0; z < COARSE_N; z++)
			for (idx y = 0; y < COARSE_N; y++)
				coarse.hmap(1, y, z) = NSE_CONFIG::BC::GEO_NOTHING;  // x-min face (full y/z)
		for (idx z = 0; z < COARSE_N; z++)
			for (idx x = 1; x < COARSE_N; x++)
				coarse.hmap(x, 1, z) = NSE_CONFIG::BC::GEO_NOTHING;  // y-min face (launch x extent)
		for (idx y = 1; y < COARSE_N; y++)
			for (idx x = 1; x < COARSE_N; x++)
				coarse.hmap(x, y, 1) = NSE_CONFIG::BC::GEO_NOTHING;  // z-min face (launch x/y extent)
		coarse.dmap = coarse.hmap;
		coarse.copyToDevice();

		launchFineToCoarse(coarse, fine, {1, 0, 0}, {2, COARSE_N, COARSE_N}, {0, 0, 0}, {0, 0, 0}, fine_even_iter, coarse_even_iter);
		launchFineToCoarse(coarse, fine, {1, 1, 0}, {COARSE_N, 2, COARSE_N}, {0, 0, 0}, {0, 0, 0}, fine_even_iter, coarse_even_iter);
		launchFineToCoarse(coarse, fine, {1, 1, 1}, {COARSE_N, COARSE_N, 2}, {0, 0, 0}, {0, 0, 0}, fine_even_iter, coarse_even_iter);
		coarse.copyToHost();

		// per-rectangle single-clamp probes plus the doubly-clamped corner:
		// each probe carries at least one clamped {0,1,2,3} tangent window
		// (the lo edges of the launch rectangles) that would read a
		// sentinel ghost plane on a clamp regression; (1,0,0)'s two clamps
		// are the depth-1 maximum (the normal axis is nominal everywhere)
		const std::vector<idx3d> probes = {{1, 0, 4}, {1, 4, 0}, {4, 1, 0}, {1, 0, 0}};
		checkCoarseTransferExact(
			coarse,
			probes,
			coarse_even_iter,
			fmt::format(
				"Test 15 depth-1 lo-edge clamp exactness (fine_even={}) [Lagrava (opt-out) branch]: probes (1,0,4) / (1,4,0) / (4,1,0) single-clamped tangent windows, (1,0,0) doubly clamped, sentinel-guarded",
				fine_even_iter
			)
				.c_str()
		);
	}
#endif	// F2C_SCHONHERR
}

// Test 16: skin-launch Defect-2 DF/macro-store map guard (the Test 7
// 4-class lock, skin variant) -- the x-min launch rectangle
// {1}x{0..8}x{0..8} tagged GEO_NOTHING throughout (the depth-1 skin
// destination row's production band class after the commit-7 position
// re-anchor, see the block comment) with one cell per map class inside
// it: GEO_WALL and GEO_FLUID are PROTECTED (their DFs and macros were
// NaN-poisoned before the launch, so ANY forbidden write trips the isnan
// assertion), GEO_NOTHING (the depth-1 skin destination class itself) and
// GEO_AMR_INTERFACE RECEIVE the transfer. The allowed-GEO predicate
// itself (Phase 0.4, amr_coupling.h) is unaffected by the ring-path
// removal -- this test pins it on the launch-row geography.
// [T15: the guard predicate is verbatim-shared by both F2C branches
// (strategy-INDEPENDENT machinery, the dedupe-audit reason no second map
// guard exists for the arm); the RECEIVING cells' transfer values follow
// the strategy split via skinExpectedRho (Lagrava: coarse-center
// projection; F2C_SCHONHERR: subcell mean d0, mean-density transfer).]
void test_f2c_skin_df_store_map_guard()
{
	// post-stream natural orientation on the fine level, spatial (twisted)
	// coarse consumer -- same orientation state as Tests 5 and 7
	const bool fine_even_iter = false;
	const bool next_coarse_even_iter = false;

	// one cell per map class in the launched x-min face (z == 3 row); the
	// receiving cells' tangent windows are nominal (fy0 in {10,12}), the
	// class assertion is window-independent (cubic marker)
	constexpr idx Y_WALL = 3, Y_FLUID = 4, Y_NOTHING = 5, Y_INTERFACE = 6, CZ = 3;

	MockBlock coarse, fine;
	coarse.allocate(COARSE_N);
	fine.allocate(FINE_N);
	fillFieldCE(fine, fine_even_iter, SkinCubicField{});
	fine.copyToDevice();
	fillUniform(coarse, true, 1.0, 0.0, 0.0, 0.0);

	// launch rectangle tagged with its band-position class after the
	// commit-7 position re-anchor (GEO_NOTHING, the depth-1 skin
	// destination class); the four class cells overwrite their map tags,
	// including the explicit GEO_AMR_INTERFACE row that keeps the
	// ring-class corner of the guard matrix
	for (idx z = 0; z < COARSE_N; z++)
		for (idx y = 0; y < COARSE_N; y++)
			coarse.hmap(1, y, z) = NSE_CONFIG::BC::GEO_NOTHING;
	coarse.hmap(1, Y_WALL, CZ) = NSE_CONFIG::BC::GEO_WALL;
	coarse.hmap(1, Y_FLUID, CZ) = NSE_CONFIG::BC::GEO_FLUID;
	coarse.hmap(1, Y_NOTHING, CZ) = NSE_CONFIG::BC::GEO_NOTHING;
	coarse.hmap(1, Y_INTERFACE, CZ) = NSE_CONFIG::BC::GEO_AMR_INTERFACE;
	coarse.dmap = coarse.hmap;

	// NaN-poison the protected cells' DFs (every array/slot) and macros:
	// the guard keeps them NaN; any forbidden write lands a finite value
	// (the kernel's inputs are finite) and trips the finiteness check
	const dreal nan = std::numeric_limits<dreal>::quiet_NaN();
	poisonCellDFs(coarse, 1, Y_WALL, CZ);
	poisonCellDFs(coarse, 1, Y_FLUID, CZ);
	for (int m = 0; m < NSE_CONFIG::MACRO::N; m++) {
		coarse.hmacro(m, 1, Y_WALL, CZ) = nan;
		coarse.hmacro(m, 1, Y_FLUID, CZ) = nan;
	}
	coarse.dmacro = coarse.hmacro;
	coarse.copyToDevice();

	launchFineToCoarse(coarse, fine, {1, 0, 0}, {2, COARSE_N, COARSE_N}, {0, 0, 0}, {0, 0, 0}, fine_even_iter, next_coarse_even_iter);

	coarse.copyToHost();
	coarse.hmacro = coarse.dmacro;

	// DF check per map class: protected cells must still hold the NaN
	// poison in the kernel's write slot of every direction, coupling cells
	// must hold the analytic transfer result there (Test 7's structure) --
	// the transfer value is the T15 strategy-split expectation
	// (skinExpectedRho at the receiving cell)
	const idx case_ys[4] = {Y_WALL, Y_FLUID, Y_NOTHING, Y_INTERFACE};
	const bool case_write[4] = {false, false, true, true};
	const char* case_names[4] = {"GEO_WALL", "GEO_FLUID", "GEO_NOTHING", "GEO_AMR_INTERFACE"};
	for (int cse = 0; cse < 4; cse++) {
		const idx y = case_ys[cse];
		const bool expect_write = case_write[cse];
		const std::array<dreal, 27> eq_transfer =
			equilibriumOnHost(skinExpectedRho({1, y, CZ}), SkinCubicField::U0, SkinCubicField::V0, SkinCubicField::W0);
		double max_err = 0;
		idx bad = 0;
		bool first_mismatch = true;
		for (int q = 0; q < 27; q++) {
			const int slot = coarseWriteSlot(q, next_coarse_even_iter);
			const dreal actual = coarse.hfs[f2cWriteArray()](slot, 1, y, CZ);
			const bool ok = expect_write ? (std::isfinite(actual) && closeEnough(actual, eq_transfer[q], 1e-4, 1e-5))
										 : static_cast<bool>(std::isnan(actual));
			if (! ok) {
				if (first_mismatch) {
					fmt::println(
						"  first mismatch: cell=(1, {}, {}), q={}, actual={:.9e} (expected {})",
						y,
						CZ,
						q,
						actual,
						expect_write ? "transfer value" : "NaN poison"
					);
					first_mismatch = false;
				}
				bad++;
			}
			if (expect_write)
				max_err = std::max<double>(max_err, std::abs(actual - eq_transfer[q]));
		}
		report(
			bad == 0,
			fmt::format(
				"Test 16 skin F2C DF-store map guard [{}]: {} cell {} (max |err| = {:.3e})",
				f2c_strategy_name,
				case_names[cse],
				expect_write ? "received the transfer (finite)" : "kept the NaN poison (not overwritten)",
				max_err
			)
		);
	}

	// macro store under the same predicate: protected macros stay NaN,
	// receiving macros hold the transfer macros (the strategy-split
	// expectation of skinExpectedRho -- T15)
	const int macro_ids[4] = {NSE_CONFIG::MACRO::e_rho, NSE_CONFIG::MACRO::e_vx, NSE_CONFIG::MACRO::e_vy, NSE_CONFIG::MACRO::e_vz};
	for (int cse = 0; cse < 4; cse++) {
		const idx y = case_ys[cse];
		const bool expect_write = case_write[cse];
		const std::array<dreal, 4> expected = {skinExpectedRho({1, y, CZ}), SkinCubicField::U0, SkinCubicField::V0, SkinCubicField::W0};
		double max_err = 0;
		idx bad = 0;
		bool first_mismatch = true;
		for (int m = 0; m < 4; m++) {
			const dreal actual = coarse.hmacro(macro_ids[m], 1, y, CZ);
			const bool ok = expect_write ? (std::isfinite(actual) && closeEnough(actual, expected[m], 1e-4, 1e-5))
										 : static_cast<bool>(std::isnan(actual));
			if (! ok) {
				if (first_mismatch) {
					fmt::println(
						"  first macro mismatch: cell=(1, {}, {}), id={}, actual={:.9e} (expected {})",
						y,
						CZ,
						macro_ids[m],
						actual,
						expect_write ? "transfer macro" : "NaN poison"
					);
					first_mismatch = false;
				}
				bad++;
			}
			if (expect_write)
				max_err = std::max<double>(max_err, std::abs(actual - expected[m]));
		}
		report(
			bad == 0,
			fmt::format(
				"Test 16 skin F2C macro-store map guard [{}]: {} macros {} (max |err| = {:.3e})",
				f2c_strategy_name,
				case_names[cse],
				expect_write ? "written by the transfer" : "kept the NaN poison",
				max_err
			)
		);
	}
}

// Test 18: footprint lo-lo EDGE clamp exactness (2-edge-adjacent probes),
// sentinel-guarded [LAGRAVA (OPT-OUT) BRANCH ONLY -- T15, commit 14: same
// disposition as Test 15; the two-clamp multiplicity class is Lagrava
// window machinery with no F2C_SCHONHERR counterpart (no window on the
// arm). The lock is the opt-out authority and stays ON the Lagrava
// branch -- retire nothing silently; the arm build reports an explicit
// deferral line below] -- Test 15's machinery (same +1000 sentinel
// planes, same gates) with the probe set swapped to the depth-1 lo-lo
// tangent edges: (1,0,0) lands in the x-min launch (y,z windows clamped),
// (0,1,0) in the y-min launch (x,z windows clamped), (0,0,1) in the z-min
// launch (x,y windows clamped). Each probe has exactly TWO windows
// clamped to {0,1,2,3} and the third window nominal -- a
// clamp-multiplicity class no sibling positively locks: at depth 1 the
// face-normal window is nominal (fx0 = 2) everywhere, so TWO clamps per
// cell is the maximum possible multiplicity (the surface era's
// three-clamped corner class is extinct; the launch tangent extents are
// the full rows so the edges reach the f0 == 0 clamp on both tangent
// axes). A lower-bound regression on either clamped axis reads a sentinel
// node and shifts the transfer by O(10..100), decades above the gates.
void test_f2c_skin_edge2pair_clamp_exactness()
{
#ifdef F2C_SCHONHERR
	// T15: no clamp machinery on the F2C_SCHONHERR arm -- this lock is the
	// Lagrava (opt-out) branch's authority and is not duplicated (dedupe
	// audit); the explicit deferral line keeps the arm's report truthful
	report(
		true,
		"Test 18 depth-1 lo-lo edge clamp exactness [F2C_SCHONHERR arm]: N/A -- clamp is Lagrava-only window machinery; authority lives on the Lagrava (opt-out) branch (retired nothing)"
	);
#else
	const bool coarse_even_iter = false;

	for (const bool fine_even_iter : {true, false}) {
		MockBlock coarse, fine;
		coarse.allocate(COARSE_N);
		fine.allocate(FINE_N);
		fillFieldCE(fine, fine_even_iter, SkinCubicField{});
		sentinelFineGhostPlane(fine, 0);
		sentinelFineGhostPlane(fine, 1);
		sentinelFineGhostPlane(fine, 2);
		fine.copyToDevice();
		fillUniform(coarse, true, 1.0, 0.0, 0.0, 0.0);

		// the three min-face depth-1 skin destination rows (commit-7
		// position re-anchor), tagged GEO_NOTHING (their production band
		// class at depth 1; the F2C allowed-GEO store guard admits frozen
		// coupling cells). The tangent extents are the FULL rows here (the
		// probes need the lo tangent edges of the y-/z-min launches; any
		// corner re-write is idempotent)
		for (idx z = 0; z < COARSE_N; z++)
			for (idx y = 0; y < COARSE_N; y++)
				coarse.hmap(1, y, z) = NSE_CONFIG::BC::GEO_NOTHING;  // x-min face (full y/z)
		for (idx z = 0; z < COARSE_N; z++)
			for (idx x = 0; x < COARSE_N; x++)
				coarse.hmap(x, 1, z) = NSE_CONFIG::BC::GEO_NOTHING;  // y-min face (full x/z)
		for (idx y = 0; y < COARSE_N; y++)
			for (idx x = 0; x < COARSE_N; x++)
				coarse.hmap(x, y, 1) = NSE_CONFIG::BC::GEO_NOTHING;  // z-min face (full x/y)
		coarse.dmap = coarse.hmap;
		coarse.copyToDevice();

		launchFineToCoarse(coarse, fine, {1, 0, 0}, {2, COARSE_N, COARSE_N}, {0, 0, 0}, {0, 0, 0}, fine_even_iter, coarse_even_iter);
		launchFineToCoarse(coarse, fine, {0, 1, 0}, {COARSE_N, 2, COARSE_N}, {0, 0, 0}, {0, 0, 0}, fine_even_iter, coarse_even_iter);
		launchFineToCoarse(coarse, fine, {0, 0, 1}, {COARSE_N, COARSE_N, 2}, {0, 0, 0}, {0, 0, 0}, fine_even_iter, coarse_even_iter);
		coarse.copyToHost();

		// two-clamped-axes probes (the depth-1 maximum multiplicity):
		// (1,0,0) x-min [y,z clamped], (0,1,0) y-min [x,z clamped],
		// (0,0,1) z-min [x,y clamped]; each ALSO positively guards the
		// clamp on both edges via the sentinel ghost planes
		const std::vector<idx3d> probes = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
		checkCoarseTransferExact(
			coarse,
			probes,
			coarse_even_iter,
			fmt::format(
				"Test 18 depth-1 lo-lo edge clamp exactness (fine_even={}) [Lagrava (opt-out) branch]: probes (1,0,0) x-min / (0,1,0) y-min / (0,0,1) z-min, two clamped {{0,1,2,3}} windows + one nominal each, sentinel-guarded",
				fine_even_iter
			)
				.c_str()
		);
	}
#endif	// F2C_SCHONHERR
}

int main()
{
	fmt::println("AMR coupling kernel unit tests (streaming pattern: {})", pattern_name);

	test_uniform_coarse_to_fine();
	test_uniform_fine_to_coarse();
	test_linear_gradient_coarse_to_fine();
	test_mass_conservation_fine_to_coarse();
	test_mass_conservation_coarse_to_fine();
	test_nested_geometry_coupling();
	test_cubic_reproduction_fine_to_coarse();
	test_f2c_df_store_map_guard();

	// skin F2C path coverage (production's only F2C channel since the ring
	// path was removed in D.1): exactness, lo-edge clamp, Defect-2 guard on
	// skin geography, lo-lo edge clamp probes -- strategy-split per T15:
	// Tests 14/16 assert BOTH branches (strategy-split expectation), the
	// lo = 0 clamp locks (Tests 15/18) are Lagrava (opt-out) authorities
	// and defer explicitly on the F2C_SCHONHERR arm
	test_f2c_skin_exactness_interior();
	test_f2c_skin_edge_clamp_exactness();
	test_f2c_skin_df_store_map_guard();
	test_f2c_skin_edge2pair_clamp_exactness();

#if defined(C2F_COMPACT_MOMENT) || (!defined(C2F_LAGRANGE) && !defined(C2F_TRILINEAR) && !defined(C2F_LINEAR_EXPLOSION) && !defined(C2F_UNIFORM_EXPLOSION))
	// production CM semantics (default since the 2026-08-18 flip, user ruling)
	test_cm_exactness_nominal();
#endif	// CM semantics active

	if (g_failures == 0) {
		fmt::println("RESULT: all AMR coupling tests passed");
		return 0;
	}
	fmt::println("RESULT: {} AMR coupling check(s) FAILED", g_failures);
	return 1;
}
