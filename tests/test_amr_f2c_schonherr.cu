// Compile-and-run exactness locks for the F2C_SCHONHERR fine-to-coarse
// branch of cudaAMR_FineToCoarse (include/lbm3d/d3q27/amr_coupling.h;
// plan T14 / commit 13 of .omo/plans/schonherr-ch7-conversion.md).
//
// The F2C_SCHONHERR define is a per-TU compile-time switch selecting the
// thesis sec. 7.2 sigma-form compact-moment transfer (sigma_{f->c} = 2)
// inside the F2C kernel and cannot share a binary with the default
// Lagrava-filter build (ODR hazard on the kernel template symbol), so it
// locks as standalone per-pattern binaries: tests/CMakeLists.txt compiles
// this source once per streaming pattern with the define hardcoded (the
// same idiom as the test_amr_c2f_smoke_* seam-investigation binaries), and
// tests/unit/test_amr_f2c_schonherr.py drives them. The default build
// (tests/test_amr_coupling.cu + tests/run-amr-tests.sh) keeps the Lagrava
// path pinned separately, so the pair of batteries is green under BOTH
// strategies.
//
// Branch semantics under test (Schönherr 2015 thesis sec. 7.2 + contract
// doc docs/AMR-schonherr-ch7-target-contract.md appendix A.2.3): sources
// are the destination cell's OWN 8 fine subcells; per-source k-moments are
// formed at the source (fine) grid rate omega_s = 1/TAU_FINE; the
// destination is evaluated at the window center t = (0,0,0), where every
// correction aggregate A011/A101/A110/corr_B/corr_C vanishes identically
// (the avk gradient corrections retract exactly onto the sk_* means), so
// the recovered non-equilibrium pressure tensor of the written coarse DF
// state (rec_ab = sum_q c_a c_b (f_q - eq_q(rho_f, u_f))) is analytic in
// the strain data of a CE-consistent fill:
//   rec_xy/xz/yz = -(sigma rho_f / (3 omega_d)) * G_{ab}
//   rec_xx       = -(2 sigma rho_f / (9 omega_d)) * (2 Gxx - Gyy - Gzz) / 2
//   rec_yy       = -(2 sigma rho_f / (9 omega_d)) * (-Gxx + 2 Gyy - Gzz) / 2
//   rec_zz       = -(2 sigma rho_f / (9 omega_d)) * (-Gxx - Gyy + 2 Gzz) / 2
// with sigma = 2 and omega_d = 1/TAU_COARSE (the F2C sigma-form of Eqs.
// 7.38-7.48; rec_ab = C_ab - delta_ab rho_f/3 at the same first moments,
// cf. the S2 lock of tests/test_amr_c2f_debug_smoke.cu).
//
// Checks per binary (each x all four combos of fine read parity and coarse
// store parity):
//   L1 constant exact: uniform fill -> every destination DF == the
//       equilibrium of the same uniform field (rtol 1e-6 / atol 1e-8, the
//       Tests-1/2 class).
//   L2 linear velocity exact: linear rho + linear velocity with
//       CE-consistent constant strain -> destination macros == analytic
//       field at the coarse cell center (rtol 1e-4 / atol 1e-6, the
//       Tests-8/9 class; the full {0..8}^3 launch covers every extent
//       CORNER -- there is no window to shift, corners get the identical
//       uniform treatment).
//   L3 quadratic-velocity + linear-density exact at t = (0,0,0): the
//       CMQuadraticField exactness class of T10c, re-anchored to the F2C
//       destination position (the k-corrected Eqs. 7.18/7.23/7.24 velocity
//       family; the print family would sit ~1e-5 off here as well).
//   L4 CE-consistent strain round-trip at sigma = 2: recovered Pi^neq ==
//       the reduced-cumulant targets above (atol 1e-6 + rtol 1e-4, the S2
//       class of test_amr_c2f_debug_smoke.cu).
//   L5 mass identity: the sum of the 27 destination DFs equals d_0 == the
//       mean of the 8 subcell densities exactly (the transfer is a
//       mean-density transfer with no conservation claim -- T15's T4a
//       successor), reported against the gate rtol 1e-6 / atol 1e-7.
//
// The binary prints measured maxima and exits nonzero on any failed check
// (fmt REPORT idiom of tests/test_amr_coupling.cu).

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

#ifndef F2C_SCHONHERR
	#error "test_amr_f2c_schonherr must be compiled with -DF2C_SCHONHERR (see tests/CMakeLists.txt)"
#endif

using idx = typename TRAITS::idx;
using idx3d = typename TRAITS::idx3d;
using dreal = typename TRAITS::dreal;
using DATA = typename NSE_CONFIG::DATA;
using LBM_KS = typename NSE_CONFIG::template KernelStruct<dreal>;

constexpr idx COARSE_N = 8;
constexpr idx FINE_N = 16;
constexpr idx OV = 1;
constexpr dreal TAU_COARSE = 3 * 0.05f + 0.5f;
constexpr dreal TAU_FINE = 3 * 0.10f + 0.5f;

// D3Q27 velocity set c_q matching the direction enum in defs.h
constexpr int VELOCITY[27][3] = {
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

// minimal mock of an LBM block's device data (the tests/test_amr_coupling.cu
// idiom, as specialized in tests/test_amr_c2f_debug_smoke.cu)
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
	idx ov = OV;

	void allocate(idx N, idx overlap = OV)
	{
		size = N;
		ov = overlap;

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

		for (uint8_t dfty = 0; dfty < DFMAX; dfty++)
			data.dfs[dfty] = dfs[dfty].getData();
		data.indexer = dmap.getIndexer();
		data.XYZ = data.indexer.getStorageSize();
		data.dmap = dmap.getData();
		data.dmacro = dmacro.getData();

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

	void copyMacrosToHost()
	{
		hmacro = dmacro;
	}
};

void storePostCollisionDF(MockBlock& block, bool even_iter, int q, idx x, idx y, idx z, dreal value)
{
#ifdef AB_PATTERN
	static_cast<void>(even_iter);
	block.hfs[df_out](q, x, y, z) = value;
#elif defined(AA_PATTERN)
	block.hfs[df_cur](even_iter ? opposite_direction(q) : q, x, y, z) = value;
#endif
}

// host-side readback of a destination DF written by the F2C kernel: AB
// writes the logical df_out in natural orientation; AA writes df_cur natural
// when the next substep is even ("reflect") and twisted when odd
dreal readCoarseDF(const MockBlock& block, bool coarse_even_iter, int q, idx x, idx y, idx z)
{
#ifdef AB_PATTERN
	static_cast<void>(coarse_even_iter);
	return block.hfs[df_out](q, x, y, z);
#elif defined(AA_PATTERN)
	return block.hfs[df_cur](coarse_even_iter ? q : opposite_direction(q), x, y, z);
#endif
}

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

// L2/L4 field: rho and velocities linear in all three coordinates (the
// Tests-8 coefficients of tests/test_amr_coupling.cu)
struct FLinear
{
	std::array<dreal, 10> fill(idx x, idx y, idx z) const
	{
		const std::array<double, 4> e = exact(x, y, z);
		return {
			static_cast<dreal>(e[0]),
			static_cast<dreal>(e[1]),
			static_cast<dreal>(e[2]),
			static_cast<dreal>(e[3]),
			dreal(2 * 0.002),
			dreal(2 * 0.0018),
			dreal(2 * 0.0014),
			dreal(0.001 + (-0.0015)),
			dreal(-0.0009 + 0.001),
			dreal(0.0011 + (-0.0012))
		};
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

// L3 field: linear density, velocities linear + pure quadratic in each
// coordinate (the Tests-9 coefficients; the T10c R1 exactness class)
struct FQuadratic
{
	std::array<dreal, 10> fill(idx x, idx y, idx z) const
	{
		const std::array<double, 4> e = exact(x, y, z);
		const double dudx = 0.002 + 2 * 0.0008 * x, dudy = -0.0015 - 2 * 0.0006 * y, dudz = 0.001 + 2 * 0.0004 * z;
		const double dvdx = 0.001 - 2 * 0.0007 * x, dvdy = 0.0018 + 2 * 0.0005 * y, dvdz = -0.0012 + 2 * 0.0003 * z;
		const double dwdx = -0.0009 + 2 * 0.0006 * x, dwdy = 0.0011 - 2 * 0.0005 * y, dwdz = 0.0014 + 2 * 0.00045 * z;
		return {
			static_cast<dreal>(e[0]),
			static_cast<dreal>(e[1]),
			static_cast<dreal>(e[2]),
			static_cast<dreal>(e[3]),
			static_cast<dreal>(2 * dudx),
			static_cast<dreal>(2 * dvdy),
			static_cast<dreal>(2 * dwdz),
			static_cast<dreal>(dvdx + dudy),
			static_cast<dreal>(dwdx + dudz),
			static_cast<dreal>(dwdy + dvdz)
		};
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

// fill a fine block with the CE-consistent state of `field`: the
// non-equilibrium carries the strain G at the SOURCE grid rate -- the fine
// level here, so omega_s = 1/TAU_FINE (the F2C mirror of the C2F-side
// fillFieldCE, which uses the coarse rate)
template <typename FIELD>
void fillFieldCE_fine(MockBlock& block, bool even_iter, const FIELD& field)
{
	constexpr dreal cs2 = dreal(1) / dreal(3);
	const dreal omega_s = dreal(1) / TAU_FINE;
	for (idx z = -block.ov; z < block.size + block.ov; z++) {
		for (idx y = -block.ov; y < block.size + block.ov; y++) {
			for (idx x = -block.ov; x < block.size + block.ov; x++) {
				const std::array<dreal, 10> fld = field.fill(x, y, z);
				const dreal rho = fld[0];
				const std::array<dreal, 27> eq = equilibriumOnHost(rho, fld[1], fld[2], fld[3]);
				for (int q = 0; q < 27; q++) {
					const dreal qx = VELOCITY[q][0], qy = VELOCITY[q][1], qz = VELOCITY[q][2];
					const dreal QG = (qx * qx - cs2) * fld[4] + (qy * qy - cs2) * fld[5] + (qz * qz - cs2) * fld[6] + 2 * qx * qy * fld[7]
								   + 2 * qx * qz * fld[8] + 2 * qy * qz * fld[9];
					const dreal f_neq = -(rho / (3 * omega_s)) * (d3q27Weight(q) / (2 * cs2 * cs2)) * QG;
					storePostCollisionDF(block, even_iter, q, x, y, z, eq[q] + f_neq);
				}
			}
		}
	}
}

void fillUniform(MockBlock& block, bool even_iter, dreal rho0, dreal u0, dreal v0, dreal w0)
{
	const std::array<dreal, 27> eq = equilibriumOnHost(rho0, u0, v0, w0);
	for (idx z = -block.ov; z < block.size + block.ov; z++)
		for (idx y = -block.ov; y < block.size + block.ov; y++)
			for (idx x = -block.ov; x < block.size + block.ov; x++)
				for (int q = 0; q < 27; q++)
					storePostCollisionDF(block, even_iter, q, x, y, z, eq[q]);
}

// tag the destination rectangle as GEO_AMR_INTERFACE ring cells: the
// production coupling restricts the F2C writes to coupling cells, so mock
// blocks must mark the processed rectangle to receive writes
void tagCouplingCells(MockBlock& block, idx3d begin, idx3d end)
{
	for (idx z = begin.z(); z < end.z(); z++)
		for (idx y = begin.y(); y < end.y(); y++)
			for (idx x = begin.x(); x < end.x(); x++)
				block.hmap(x, y, z) = NSE_CONFIG::BC::GEO_AMR_INTERFACE;
	block.dmap = block.hmap;
}

void launchFineToCoarse(MockBlock& coarse, MockBlock& fine, idx3d begin, idx3d end, bool fine_even_iter, bool coarse_even_iter)
{
	const idx3d size = end - begin;
	TNL::Backend::LaunchConfiguration launch_config;
	launch_config.blockSize = dim3(4, 4, 4);
	launch_config.gridSize =
		dim3(static_cast<unsigned>((size.x() + 3) / 4), static_cast<unsigned>((size.y() + 3) / 4), static_cast<unsigned>((size.z() + 3) / 4));
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
		idx3d{0, 0, 0},
		idx3d{0, 0, 0},
		idx3d{fine.size, fine.size, fine.size},
		idx3d{fine.ov, fine.ov, fine.ov}
	);
	TNL::Backend::streamSynchronize(0);
}

// macros (rho, u, v, w) of the coarse DF state the F2C transfer wrote at a
// coupling cell, read back in the store parity
void coarseDestMacros(const MockBlock& coarse, bool coarse_even_iter, idx x, idx y, idx z, dreal& rho, dreal& u, dreal& v, dreal& w)
{
	rho = 0;
	dreal jx = 0, jy = 0, jz = 0;
	for (int q = 0; q < 27; q++) {
		const dreal f = readCoarseDF(coarse, coarse_even_iter, q, x, y, z);
		rho += f;
		jx += VELOCITY[q][0] * f;
		jy += VELOCITY[q][1] * f;
		jz += VELOCITY[q][2] * f;
	}
	u = jx / rho;
	v = jy / rho;
	w = jz / rho;
}

bool closeEnough(dreal actual, double expected, dreal rtol, dreal atol)
{
	return std::abs(actual - expected) <= atol + rtol * std::abs(expected);
}

// L1: uniform field -> destination DFs are the equilibrium of the same
// field (both read and store parities)
void test_constant_exact()
{
	const dreal rho0 = 1.0, u0 = 0.04, v0 = 0.01, w0 = -0.02;
	const std::array<dreal, 27> expected = equilibriumOnHost(rho0, u0, v0, w0);
	const std::array<bool, 2> parities = {false, true};
	for (const bool fe : parities) {
		for (const bool ce : parities) {
			MockBlock fine, coarse;
			fine.allocate(FINE_N);
			coarse.allocate(COARSE_N);
			fillUniform(fine, fe, rho0, u0, v0, w0);
			fine.copyToDevice();
			tagCouplingCells(coarse, {0, 0, 0}, {COARSE_N, COARSE_N, COARSE_N});
			launchFineToCoarse(coarse, fine, {0, 0, 0}, {COARSE_N, COARSE_N, COARSE_N}, fe, ce);
			coarse.copyToHost();

			double max_err = 0;
			idx bad = 0;
			for (idx z = 0; z < COARSE_N; z++)
				for (idx y = 0; y < COARSE_N; y++)
					for (idx x = 0; x < COARSE_N; x++)
						for (int q = 0; q < 27; q++) {
							const dreal actual = readCoarseDF(coarse, ce, q, x, y, z);
							if (! closeEnough(actual, expected[q], 1e-6, 1e-8))
								bad++;
							max_err = std::max<double>(max_err, std::abs(actual - expected[q]));
						}
			report(
				bad == 0,
				fmt::format("L1 constant exact (fine_even={}, coarse_even={}): all 27 DFs match equilibrium (max |err| = {:.3e})", fe, ce, max_err)
			);
		}
	}
}

// L2/L3 (+L5): destination macros == analytic field at the coarse cell
// center (centroid of the 2x2x2 subcell block); the launch covers the full
// destination block, corners included (no window machinery in this branch)
template <typename FIELD>
void test_macros_exact(const FIELD& field, const char* name)
{
	const std::array<bool, 2> parities = {false, true};
	for (const bool fe : parities) {
		for (const bool ce : parities) {
			MockBlock fine, coarse;
			fine.allocate(FINE_N);
			coarse.allocate(COARSE_N);
			fillFieldCE_fine(fine, fe, field);
			fine.copyToDevice();
			tagCouplingCells(coarse, {0, 0, 0}, {COARSE_N, COARSE_N, COARSE_N});
			launchFineToCoarse(coarse, fine, {0, 0, 0}, {COARSE_N, COARSE_N, COARSE_N}, fe, ce);
			coarse.copyToHost();
			coarse.copyMacrosToHost();

			double max_rel_rho = 0, max_abs_u = 0, max_abs_macro = 0, max_abs_d0 = 0;
			idx bad_macros = 0, bad_d0 = 0;
			for (idx z = 0; z < COARSE_N; z++)
				for (idx y = 0; y < COARSE_N; y++)
					for (idx x = 0; x < COARSE_N; x++) {
						dreal rho_m, u_m, v_m, w_m;
						coarseDestMacros(coarse, ce, x, y, z, rho_m, u_m, v_m, w_m);
						const std::array<double, 4> e = field.exact(2 * x + 0.5, 2 * y + 0.5, 2 * z + 0.5);
						const bool finite = std::isfinite(rho_m) && std::isfinite(u_m) && std::isfinite(v_m) && std::isfinite(w_m);
						const double rel_rho = std::abs(rho_m - e[0]) / e[0];
						const double abs_du = std::max({std::abs(u_m - e[1]), std::abs(v_m - e[2]), std::abs(w_m - e[3])});
						max_rel_rho = std::max(max_rel_rho, rel_rho);
						max_abs_u = std::max(max_abs_u, abs_du);
						if (! finite || ! closeEnough(rho_m, e[0], 1e-4, 1e-6) || abs_du > 1e-6 + 1e-4 * 0.03)
							bad_macros++;
						// the written output macros of the branch (d_0/a_0/b_0/c_0)
						const dreal mm[4] = {rho_m, u_m, v_m, w_m};
						const int macro_ids[4] = {
							NSE_CONFIG::MACRO::e_rho, NSE_CONFIG::MACRO::e_vx, NSE_CONFIG::MACRO::e_vy, NSE_CONFIG::MACRO::e_vz
						};
						for (int m = 0; m < 4; m++) {
							const double err = std::abs(coarse.hmacro(macro_ids[m], x, y, z) - static_cast<double>(mm[m]));
							max_abs_macro = std::max(max_abs_macro, err);
						}
						// L5: the destination DF sum is d_0 == the mean of
						// the 8 subcell densities (mean-density transfer)
						double d0_mean = 0;
						for (int bz = 0; bz < 2; bz++)
							for (int by = 0; by < 2; by++)
								for (int bx = 0; bx < 2; bx++)
									d0_mean += field.exact(2 * x + bx, 2 * y + by, 2 * z + bz)[0];
						d0_mean /= 8;
						max_abs_d0 = std::max(max_abs_d0, std::abs(rho_m - d0_mean));
						if (! closeEnough(rho_m, d0_mean, 1e-6, 1e-7))
							bad_d0++;
					}
			report(
				bad_macros == 0,
				fmt::format(
					"macros exact ({}; fine_even={}, coarse_even={}): destination macros match the analytic center field incl. extent corners (max "
					"rel rho err = {:.3e}, max abs vel err = {:.3e})",
					name,
					fe,
					ce,
					max_rel_rho,
					max_abs_u
				)
			);
			report(
				max_abs_macro <= 1e-6,
				fmt::format(
					"written macros ({}; fine_even={}, coarse_even={}): dmacro carries d0/a0/b0/c0 of the DF state (max |err| = {:.3e})",
					name,
					fe,
					ce,
					max_abs_macro
				)
			);
			report(
				bad_d0 == 0,
				fmt::format(
					"L5 mass identity ({}; fine_even={}, coarse_even={}): sum of destination DFs == mean subcell density d0 exactly (max |rho_sum - "
					"d0| = {:.3e})",
					name,
					fe,
					ce,
					max_abs_d0
				)
			);
		}
	}
}

// L4: recovered Pi^neq of the destination DF state == the reduced-cumulant
// targets at sigma = 2, omega_d = 1/TAU_COARSE (linear field, CE-consistent
// constant strain)
void test_strain_roundtrip()
{
	// constant strain of the FLinear fill (Gxx = 2 du/dx, ...)
	const std::array<double, 6> G = {2 * 0.002, 2 * 0.0018, 2 * 0.0014, 0.001 + (-0.0015), -0.0009 + 0.001, 0.0011 + (-0.0012)};
	const FLinear field{};
	const std::array<bool, 2> parities = {false, true};
	for (const bool fe : parities) {
		for (const bool ce : parities) {
			MockBlock fine, coarse;
			fine.allocate(FINE_N);
			coarse.allocate(COARSE_N);
			fillFieldCE_fine(fine, fe, field);
			fine.copyToDevice();
			tagCouplingCells(coarse, {0, 0, 0}, {COARSE_N, COARSE_N, COARSE_N});
			launchFineToCoarse(coarse, fine, {0, 0, 0}, {COARSE_N, COARSE_N, COARSE_N}, fe, ce);
			coarse.copyToHost();

			const dreal sigma = dreal(2);
			const dreal omega_d = dreal(1) / TAU_COARSE;
			double max_abs_err = 0, max_rel_err = 0;
			idx bad = 0;
			for (idx z = 0; z < COARSE_N; z++)
				for (idx y = 0; y < COARSE_N; y++)
					for (idx x = 0; x < COARSE_N; x++) {
						dreal rho, u, v, w;
						coarseDestMacros(coarse, ce, x, y, z, rho, u, v, w);
						const double rho_f = rho;
						const dreal off = static_cast<dreal>(sigma * rho_f / (3 * omega_d));
						const dreal diag = static_cast<dreal>(2 * sigma * rho_f / (9 * omega_d));
						const double targets[6] = {
							-static_cast<double>(off) * G[3],		// xy
							-static_cast<double>(off) * G[4],		// xz
							-static_cast<double>(off) * G[5],		// yz
							-diag * (2 * G[0] - G[1] - G[2]) / 2,	// xx
							-diag * (-G[0] + 2 * G[1] - G[2]) / 2,	// yy
							-diag * (-G[0] - G[1] + 2 * G[2]) / 2,	// zz
						};
						const std::array<dreal, 27> eq = equilibriumOnHost(rho, u, v, w);
						double rec[6] = {0, 0, 0, 0, 0, 0};
						for (int q = 0; q < 27; q++) {
							const dreal fn = readCoarseDF(coarse, ce, q, x, y, z) - eq[q];
							const dreal cx = VELOCITY[q][0], cy = VELOCITY[q][1], cz = VELOCITY[q][2];
							rec[0] += cx * cy * fn;
							rec[1] += cx * cz * fn;
							rec[2] += cy * cz * fn;
							// diagonals: sum qa*qa*(f - eq(rec)) = C2nn - rho/3
							// directly (the equilibrium of the recovered macros
							// carries the isotropic cs^2 part itself)
							rec[3] += cx * cx * fn;
							rec[4] += cy * cy * fn;
							rec[5] += cz * cz * fn;
						}
						for (int k = 0; k < 6; k++) {
							const double err = std::abs(rec[k] - targets[k]);
							const double rel = err / std::max(1e-12, std::abs(targets[k]));
							max_abs_err = std::max(max_abs_err, err);
							max_rel_err = std::max(max_rel_err, rel);
							if (! std::isfinite(rec[k]) || err > 1e-6 + 1e-4 * std::abs(targets[k]))
								bad++;
						}
					}
			report(
				bad == 0,
				fmt::format(
					"L4 strain round-trip at sigma = 2 (fine_even={}, coarse_even={}): recovered Pi^neq == reduced-cumulant targets (max abs err = "
					"{:.3e}, max rel err = {:.3e})",
					fe,
					ce,
					max_abs_err,
					max_rel_err
				)
			);
		}
	}
}

int main()
{
	fmt::println("AMR F2C Schönherr exactness locks (define: F2C_SCHONHERR, pattern: {})", pattern_name);

	test_constant_exact();
	test_macros_exact(FLinear{}, "linear field");
	test_macros_exact(FQuadratic{}, "linear rho + pure quadratic velocity field");
	test_strain_roundtrip();

	if (g_failures == 0) {
		fmt::println("RESULT: all AMR F2C Schönherr exactness locks passed");
		return 0;
	}
	fmt::println("RESULT: {} AMR F2C Schönherr exactness lock(s) FAILED", g_failures);
	return 1;
}
