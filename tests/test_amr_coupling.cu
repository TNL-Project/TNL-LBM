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

#include <algorithm>
#include <array>
#include <cmath>
#include <string>

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

	idx size;

	void allocate(idx N)
	{
		size = N;

		// plain NDArrays allocate inside setSizes (unlike DistributedNDArray,
		// which additionally needs setDistribution + allocate), and the
		// storage size is computed from the CURRENT overlaps -- hence the
		// overlaps must be set first
		dmap.getOverlaps().template setSize<0>(OV);
		dmap.getOverlaps().template setSize<1>(OV);
		dmap.getOverlaps().template setSize<2>(OV);
		dmap.setSizes(N, N, N);
		hmap.getOverlaps().template setSize<0>(OV);
		hmap.getOverlaps().template setSize<1>(OV);
		hmap.getOverlaps().template setSize<2>(OV);
		hmap.setSizes(N, N, N);

		dmacro.getOverlaps().template setSize<1>(OV);
		dmacro.getOverlaps().template setSize<2>(OV);
		dmacro.getOverlaps().template setSize<3>(OV);
		dmacro.setSizes(NSE_CONFIG::MACRO::N, N, N, N);
		hmacro.getOverlaps().template setSize<1>(OV);
		hmacro.getOverlaps().template setSize<2>(OV);
		hmacro.getOverlaps().template setSize<3>(OV);
		hmacro.setSizes(NSE_CONFIG::MACRO::N, N, N, N);

		for (uint8_t dfty = 0; dfty < DFMAX; dfty++) {
			dfs[dfty].getOverlaps().template setSize<1>(OV);
			dfs[dfty].getOverlaps().template setSize<2>(OV);
			dfs[dfty].getOverlaps().template setSize<3>(OV);
			dfs[dfty].setSizes(NSE_CONFIG::Q, N, N, N);
			hfs[dfty].getOverlaps().template setSize<1>(OV);
			hfs[dfty].getOverlaps().template setSize<2>(OV);
			hfs[dfty].getOverlaps().template setSize<3>(OV);
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

// fill the whole stored extent (including the overlap layers) of `block` with
// the equilibrium of a constant macroscopic state
void fillUniform(MockBlock& block, bool even_iter, dreal rho, dreal u0, dreal v0, dreal w0)
{
	const std::array<dreal, 27> eq = equilibriumOnHost(rho, u0, v0, w0);
	for (uint8_t dfty = 0; dfty < DFMAX; dfty++)
		for (int q = 0; q < 27; q++)
			for (idx z = -OV; z < block.size + OV; z++)
				for (idx y = -OV; y < block.size + OV; y++)
					for (idx x = -OV; x < block.size + OV; x++)
						block.hfs[dfty](q, x, y, z) = eq[q];

	// overwrite the array/slot the kernels actually read for this parity
	// (kept separate so an accidental cross-write between the two loops
	// would be caught)
	for (idx z = -OV; z < block.size + OV; z++)
		for (idx y = -OV; y < block.size + OV; y++)
			for (idx x = -OV; x < block.size + OV; x++)
				for (int q = 0; q < 27; q++)
					storePostCollisionDF(block, even_iter, q, x, y, z, eq[q]);
}

// fill the whole stored extent of `block` with the equilibrium of a
// cell-dependent macroscopic field evaluated at each cell's coordinates
template <typename F>
void fillField(MockBlock& block, bool even_iter, F&& macros_at_cell)
{
	for (idx z = -OV; z < block.size + OV; z++) {
		for (idx y = -OV; y < block.size + OV; y++) {
			for (idx x = -OV; x < block.size + OV; x++) {
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

void launchCoarseToFine(MockBlock& fine, MockBlock& coarse, idx3d begin, idx3d end, bool coarse_even_iter)
{
	const idx3d size = end - begin;
	TNL::Backend::LaunchConfiguration launch_config;
	launch_config.blockSize = dim3(4, 4, 4);
	launch_config.gridSize = dim3(
		static_cast<unsigned>((size.x() + 3) / 4), static_cast<unsigned>((size.y() + 3) / 4), static_cast<unsigned>((size.z() + 3) / 4)
	);
	TNL::Backend::launchKernelAsync(cudaAMR_CoarseToFine<NSE_CONFIG>, launch_config, fine.data, coarse.data, begin, end, TAU_FINE, TAU_COARSE, coarse_even_iter);
	TNL::Backend::streamSynchronize(0);
}

void launchFineToCoarse(MockBlock& coarse, MockBlock& fine, idx3d begin, idx3d end, bool fine_even_iter, bool coarse_even_iter)
{
	const idx3d size = end - begin;
	TNL::Backend::LaunchConfiguration launch_config;
	launch_config.blockSize = dim3(4, 4, 4);
	launch_config.gridSize = dim3(
		static_cast<unsigned>((size.x() + 3) / 4), static_cast<unsigned>((size.y() + 3) / 4), static_cast<unsigned>((size.z() + 3) / 4)
	);
	TNL::Backend::launchKernelAsync(
		cudaAMR_FineToCoarse<NSE_CONFIG>, launch_config, coarse.data, fine.data, begin, end, TAU_COARSE, TAU_FINE, fine_even_iter, coarse_even_iter
	);
	TNL::Backend::streamSynchronize(0);
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
		launchCoarseToFine(fine, coarse, {0, 0, 0}, {FINE_N + 1, FINE_N + 1, FINE_N + 1}, even_iter);
		fine.copyToHost();

		double max_err = 0;
		idx bad = 0;
		for (idx z = 0; z <= FINE_N; z++)
			for (idx y = 0; y <= FINE_N; y++)
				for (idx x = 0; x <= FINE_N; x++)
					for (int q = 0; q < 27; q++) {
						const dreal actual = fine.hfs[df_cur](q, x, y, z);
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

// Test 2: uniform field, fine-to-coarse -- the coarse DFs after the Lagrava
// 1/8 averaging must equal the equilibrium of the same uniform field (all
// four parity combinations of the stored fine state and the next consuming
// coarse substep are covered)
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

			launchFineToCoarse(coarse, fine, {0, 0, 0}, {COARSE_N, COARSE_N, COARSE_N}, fine_even_iter, coarse_even_iter);
			coarse.copyToHost();

			double max_err = 0;
			idx bad = 0;
			for (idx z = 0; z < COARSE_N; z++)
				for (idx y = 0; y < COARSE_N; y++)
					for (idx x = 0; x < COARSE_N; x++)
						for (int q = 0; q < 27; q++) {
							// the kernel writes the direction-q DF into the
							// parity-dependent df_cur slot
							const dreal actual = coarse.hfs[df_cur](coarseWriteSlot(q, coarse_even_iter), x, y, z);
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
					"Test 2 uniform fine-to-coarse (fine_even={}, coarse_even={}): all 27 DFs match equilibrium (max |err| = {:.3e})",
					fine_even_iter, coarse_even_iter, max_err
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

	launchCoarseToFine(fine, coarse, {0, 0, 0}, {FINE_N, FINE_N, FINE_N}, even_iter);
	fine.copyToHost();

	double max_rel_rho = 0, max_rel_jx = 0, max_abs_jyz = 0;
	idx bad = 0;
	for (idx z = 0; z < FINE_N; z++) {
		for (idx y = 0; y < FINE_N; y++) {
			for (idx x = 0; x < FINE_N; x++) {
				dreal moment[3] = {0, 0, 0};
				for (int q = 0; q < 27; q++) {
					const dreal f = fine.hfs[df_cur](q, x, y, z);
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

// Test 4a: mass conservation, fine-to-coarse -- the coarse cell DF moment
// after the transfer must equal the volume-weighted (1/8) sum of the fine
// subcell DF moments before the transfer, for every coarse cell; the
// non-equilibrium rescaling preserves the zeroth moment by construction
void test_mass_conservation_fine_to_coarse()
{
	const bool fine_even_iter = false, coarse_even_iter = true;

	// non-uniform (quadratic) density field on the fine level: the Lagrava
	// averaging is conservative for ANY field, unlike interpolation
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

	// volume-weighted fine mass per coarse cell BEFORE the transfer (from the
	// host copy, which the kernel does not modify)
	double fine_mass[COARSE_N][COARSE_N][COARSE_N];
	for (idx z = 0; z < COARSE_N; z++)
		for (idx y = 0; y < COARSE_N; y++)
			for (idx x = 0; x < COARSE_N; x++) {
				double m = 0;
				for (int bz = 0; bz < 2; bz++)
					for (int by = 0; by < 2; by++)
						for (int bx = 0; bx < 2; bx++)
							m += rhoMomentFilled(fine, fine_even_iter, 2 * x + bx, 2 * y + by, 2 * z + bz);
				fine_mass[z][y][x] = m / 8.0;
			}

	launchFineToCoarse(coarse, fine, {0, 0, 0}, {COARSE_N, COARSE_N, COARSE_N}, fine_even_iter, coarse_even_iter);
	coarse.copyToHost();

	double max_rel = 0;
	idx bad = 0;
	for (idx z = 0; z < COARSE_N; z++)
		for (idx y = 0; y < COARSE_N; y++)
			for (idx x = 0; x < COARSE_N; x++) {
				const dreal rho_c = rhoMoment(coarse, x, y, z);
				const double rel = std::abs(rho_c - fine_mass[z][y][x]) / fine_mass[z][y][x];
				max_rel = std::max(max_rel, rel);
				if (rel > 1e-5)
					bad++;
			}
	report(
		bad == 0,
		fmt::format("Test 4a mass conservation fine-to-coarse: coarse mass == 1/8 * fine subcell mass (max rel err = {:.3e})", max_rel)
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
	launchCoarseToFine(fine, coarse, {2, 2, 2}, {FINE_N, FINE_N, FINE_N}, even_iter);
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

int main()
{
	fmt::println("AMR coupling kernel unit tests (streaming pattern: {})", pattern_name);

	test_uniform_coarse_to_fine();
	test_uniform_fine_to_coarse();
	test_linear_gradient_coarse_to_fine();
	test_mass_conservation_fine_to_coarse();
	test_mass_conservation_coarse_to_fine();

	if (g_failures == 0) {
		fmt::println("RESULT: all AMR coupling tests passed");
		return 0;
	}
	fmt::println("RESULT: {} AMR coupling check(s) FAILED", g_failures);
	return 1;
}
