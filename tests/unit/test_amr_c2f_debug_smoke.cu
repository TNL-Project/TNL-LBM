// Compile-and-run smoke locks for the four C2F_*_ONLY seam-investigation
// debug defines of the §7.2 compact-moment C2F branch (C2F_EQ_ONLY /
// C2F_DEV_ONLY / C2F_NORM_ONLY / C2F_SHEAR_ONLY; case T10g of plan row 11,
// commit 10 of .omo/plans/schonherr-ch7-conversion.md).
//
// The defines are per-TU compile-time switches INSIDE the default
// compact-moment branch of cudaAMR_CoarseToFine (include/lbm3d/d3q27/
// amr_coupling.h :588-637) and cannot share a binary with the default build
// (ODR hazard on the kernel template symbol), so they smoke as standalone
// per-define doctest binaries: tests/unit/CMakeLists.txt compiles
// doctest_main.cu + this source once per define per streaming pattern,
// registering the "amr_c2f_smoke" TEST_SUITE as the
// test_amr_c2f_smoke_{eq,dev,norm,shear}_{ab,aa} targets, and
// tests/unit/test_amr_c2f_debug_smoke.py drives them.
// Semantics of the defines (each suppresses part of the non-equilibrium
// pressure tensor Pi of f_neq before the k-moment construction, Eqs.
// 7.5-7.9):
// - C2F_EQ_ONLY:   Pi == 0 outright -> pure-equilibrium fill.
// - C2F_DEV_ONLY:  trace of Pi subtracted from the diagonals. Because the
//                  reconstruction consumes Pi only through the diagonal
//                  DIFFERENCES (k_xx_yy, k_xx_zz), the trace cancels
//                  identically -- the define is an algebraic no-op for the
//                  CM branch (asserted here against the FULL targets).
// - C2F_NORM_ONLY: off-diagonals (shear) zeroed; diagonal content survives
//                  (same trace-in-differences cancellation as DEV_ONLY).
// - C2F_SHEAR_ONLY: diagonals zeroed; shear content survives.
//
// For a LINEAR velocity field with CE-consistent constant strain
// (fillFieldCE below), the full Steps-E/F chain of Eqs. 7.29-7.33 and
// 7.38-7.48 collapses analytically: the fitted first-order coefficients
// equal the velocity gradients of the field, the avg_k correction cancels
// exactly them (e.g. C110 = -off*(a_y + b_x + avg_k_xy + A110) with
// avg_k_xy = mean(k_xy) - (a_y + b_x) reduces to -off*mean(k_xy)), the
// A/B/C aggregates vanish (a linear field carries no quadratic
// coefficients), and the k-moments computed from the CE fill are analytic
// in the strain data: with Pi_ab(f_neq) = -(rho/(3 omega_s)) G_ab for the
// CE construction, k_xy = Gxy (Eqs. 7.5-7.7 carry the prefactor 3), while
// the diagonal differences carry HALF the component differences,
// k_xx_yy = (Gxx - Gyy)/2 and k_xx_zz = (Gxx - Gzz)/2 (Eqs. 7.8/7.9 carry
// 3/2 -- equivalently k_xx_yy = S_xx - S_yy with S the plain velocity
// gradient and Gxx = 2 S_xx). The recovered non-equilibrium pressure
// tensor of the written DF state (rec_ab = sum_q c_a c_b (f_q -
// eq_q(rho_f, u_f))) is therefore analytic in the FILTERED strain G' (the
// define's action):
//   rec_xy/xz/yz = -(sigma rho_f / (3 omega_d)) * G'_{ab}
//   rec_xx       = -(2 sigma rho_f / (9 omega_d)) * (2 G'xx - G'yy - G'zz) / 2
//   rec_yy       = -(2 sigma rho_f / (9 omega_d)) * (-G'xx + 2 G'yy - G'zz) / 2
//   rec_zz       = -(2 sigma rho_f / (9 omega_d)) * (-G'xx - G'yy + 2 G'zz) / 2
// with sigma = 1/2 and omega_d = 1/TAU_FINE (the C2F sigma-form of Eqs.
// 7.38-7.48, omega_d at the destination grid; rec_ab = C_ab - delta_ab
// rho_f/3 via the central/raw moment identity at the same first moments).
// These are the per-define locks compiled in below.
//
// Checks per binary:
//   S1 (shared rail): reconstructed macros == the analytic linear field,
//       rtol 1e-4 / atol 1e-6 (the Tests-8/9 tolerance class; linear fits
//       are exact under every define since the strain moments only feed
//       the curvature corrections).
//   S2 (per-define): recovered Pi^neq == the filtered-strain targets
//       above, atol 1e-6 + rtol 1e-4 -- float32 eps = 1.19e-7, the
//       measured pipeline floor on these quantities is <= ~1e-7 absolute,
//       and the define PARTITION separation (a zeroed vs live component,
//       ~5e-5) sits ~1.5 decades above the gate; beyond-eq-tolerance of
//       the recovered state under C2F_EQ_ONLY is covered by the same rail
//       (targets identically zero).
//   S3: every written DF finite (the all-fluid map keeps the carve inert).
//
// Every check is one plain CHECK_MESSAGE() doctest assertion: the
// per-define measured maxima travel in the assertion messages, and the
// doctest runner exits nonzero on any failed check.

#include <algorithm>
#include <array>
#include <cmath>
#include <string>

#include <fmt/core.h>

#include "lbm3d/core.h"
#include "lbm3d/d3q27/amr_coupling.h"

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

#if defined(C2F_EQ_ONLY)
constexpr const char* define_name = "C2F_EQ_ONLY";
#elif defined(C2F_DEV_ONLY)
constexpr const char* define_name = "C2F_DEV_ONLY";
#elif defined(C2F_NORM_ONLY)
constexpr const char* define_name = "C2F_NORM_ONLY";
#elif defined(C2F_SHEAR_ONLY)
constexpr const char* define_name = "C2F_SHEAR_ONLY";
#else
constexpr const char* define_name = "FULL (no C2F_*_ONLY define)";
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

TEST_SUITE_BEGIN("amr_c2f_smoke");

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

// minimal mock of an LBM block's device data (the tests/unit/test_amr_coupling.cu
// idiom)
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

int c2fWriteSlot(int q)
{
#ifdef AB_PATTERN
	return q;
#elif defined(AA_PATTERN)
	return opposite_direction(q);
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

// the smoke field: rho and velocities linear in all three coordinates (the
// tests/unit/test_amr_coupling.cu CMLinearField coefficients)
std::array<double, 4> exact_field(double X, double Y, double Z)
{
	return {
		1.0 + 0.01 * X - 0.008 * Y + 0.006 * Z,
		0.03 + 0.002 * X - 0.0015 * Y + 0.001 * Z,
		-0.02 + 0.001 * X + 0.0018 * Y - 0.0012 * Z,
		0.015 - 0.0009 * X + 0.0011 * Y + 0.0014 * Z,
	};
}

// analytic strain of the smoke field (constant); Gxy = dvdx + dudy etc.
std::array<double, 6> strain()
{
	return {2 * 0.002, 2 * 0.0018, 2 * 0.0014, 0.001 + (-0.0015), -0.0009 + 0.001, 0.0011 + (-0.0012)};
}

void fillFieldCE(MockBlock& block, bool even_iter)
{
	constexpr dreal cs2 = dreal(1) / dreal(3);
	const dreal omega_s = dreal(1) / TAU_COARSE;
	const std::array<double, 6> G = strain();
	for (idx z = -block.ov; z < block.size + block.ov; z++) {
		for (idx y = -block.ov; y < block.size + block.ov; y++) {
			for (idx x = -block.ov; x < block.size + block.ov; x++) {
				const std::array<double, 4> e = exact_field(x, y, z);
				const dreal rho = static_cast<dreal>(e[0]);
				const std::array<dreal, 27> eq = equilibriumOnHost(rho, static_cast<dreal>(e[1]), static_cast<dreal>(e[2]), static_cast<dreal>(e[3]));
				for (int q = 0; q < 27; q++) {
					const dreal qx = VELOCITY[q][0], qy = VELOCITY[q][1], qz = VELOCITY[q][2];
					const dreal QG = static_cast<dreal>(
						(qx * qx - cs2) * G[0] + (qy * qy - cs2) * G[1] + (qz * qz - cs2) * G[2] + 2 * qx * qy * G[3]
						+ 2 * qx * qz * G[4] + 2 * qy * qz * G[5]
					);
					const dreal f_neq = -(rho / (3 * omega_s)) * (d3q27Weight(q) / (2 * cs2 * cs2)) * QG;
					storePostCollisionDF(block, even_iter, q, x, y, z, eq[q] + f_neq);
				}
			}
		}
	}
}

void launchCoarseToFine(MockBlock& fine, MockBlock& coarse, idx3d begin, idx3d end, bool coarse_even_iter)
{
	const idx3d size = end - begin;
	TNL::Backend::LaunchConfiguration launch_config;
	launch_config.blockSize = dim3(4, 4, 4);
	launch_config.gridSize = dim3(
		static_cast<unsigned>((size.x() + 3) / 4), static_cast<unsigned>((size.y() + 3) / 4), static_cast<unsigned>((size.z() + 3) / 4)
	);
	TNL::Backend::launchKernelAsync(
		cudaAMR_CoarseToFine<NSE_CONFIG>, launch_config, fine.data, coarse.data, begin, end, TAU_FINE, TAU_COARSE, coarse_even_iter, idx3d{0, 0, 0}, idx3d{0, 0, 0}
	);
	TNL::Backend::streamSynchronize(0);
}

// the define-filtered strain (the define's action on Pi maps 1:1 onto G;
// DEV_ONLY does NOT change the reconstruction -- the trace cancels in the
// diagonal differences, so its filtered strain is the FULL one)
std::array<double, 6> filtered_strain()
{
	const auto [Gxx, Gyy, Gzz, Gxy, Gxz, Gyz] = strain();
#if defined(C2F_EQ_ONLY)
	return {0., 0., 0., 0., 0., 0.};
#elif defined(C2F_DEV_ONLY)
	return {Gxx, Gyy, Gzz, Gxy, Gxz, Gyz};	// algebraic no-op for the CM branch
#elif defined(C2F_NORM_ONLY)
	return {Gxx, Gyy, Gzz, 0., 0., 0.};
#elif defined(C2F_SHEAR_ONLY)
	return {0., 0., 0., Gxy, Gxz, Gyz};
#else
	return {Gxx, Gyy, Gzz, Gxy, Gxz, Gyz};
#endif
}

bool closeEnough(dreal actual, double expected, dreal rtol, dreal atol)
{
	return std::abs(actual - expected) <= atol + rtol * std::abs(expected);
}

// shared setup of the two smoke cases (the retired main()'s setup block,
// computations byte-identical): allocate the coarse/fine mock pair and fill
// the coarse block with the CE-consistent linear field; each case performs
// its own kernel launch and readback on top
void setupSmokeBlocks(MockBlock& coarse, MockBlock& fine, bool even_iter)
{
	coarse.allocate(COARSE_N);
	fine.allocate(FINE_N);
	fillFieldCE(coarse, even_iter);
	coarse.copyToDevice();
}

TEST_CASE("S1 field rail")
{
	const bool even_iter = false;

	MockBlock coarse, fine;
	setupSmokeBlocks(coarse, fine, even_iter);
	launchCoarseToFine(fine, coarse, {2, 2, 2}, {15, 15, 15}, even_iter);
	fine.copyToHost();

	// S1: reconstructed macros == analytic linear field (the shared rail)
	{
		double max_rel_rho = 0, max_abs_u = 0;
		idx bad = 0;
		bool finite_bad = false;
		for (idx z = 2; z < 15; z++)
			for (idx y = 2; y < 15; y++)
				for (idx x = 2; x < 15; x++) {
					dreal rho = 0, jx = 0, jy = 0, jz = 0;
					for (int q = 0; q < 27; q++) {
						const dreal f = fine.hfs[df_cur](c2fWriteSlot(q), x, y, z);
						rho += f;
						jx += VELOCITY[q][0] * f;
						jy += VELOCITY[q][1] * f;
						jz += VELOCITY[q][2] * f;
					}
					const dreal u = jx / rho, v = jy / rho, w = jz / rho;
					const auto e = exact_field(x * 0.5 - 0.25, y * 0.5 - 0.25, z * 0.5 - 0.25);
					const bool finite = std::isfinite(rho) && std::isfinite(u) && std::isfinite(v) && std::isfinite(w);
					finite_bad |= ! finite;
					const double rel_rho = std::abs(rho - e[0]) / e[0];
					const double abs_du = std::max({std::abs(u - e[1]), std::abs(v - e[2]), std::abs(w - e[3])});
					max_rel_rho = std::max(max_rel_rho, rel_rho);
					max_abs_u = std::max(max_abs_u, abs_du);
					if (! finite || ! closeEnough(rho, e[0], 1e-4f, 1e-6f) || abs_du > 1e-6 + 1e-4 * 0.03)
						bad++;
				}
	CHECK_MESSAGE(
		(bad == 0 && ! finite_bad),
		fmt::format("S1 macros == analytic linear field (max rel rho err = {:.3e}, max abs vel err = {:.3e})", max_rel_rho, max_abs_u)
	);
	}
}

TEST_CASE("S2 strain targets")
{
	const bool even_iter = false;

	MockBlock coarse, fine;
	setupSmokeBlocks(coarse, fine, even_iter);
	launchCoarseToFine(fine, coarse, {2, 2, 2}, {15, 15, 15}, even_iter);
	fine.copyToHost();

	// S2: recovered Pi^neq == the filtered-strain analytic targets
	{
		const auto [Gxx, Gyy, Gzz, Gxy, Gxz, Gyz] = filtered_strain();
		const dreal sigma = 0.5f;
		const dreal omega_d = dreal(1) / TAU_FINE;
		double max_abs_err = 0, max_rel_err = 0;
		idx bad = 0;
		for (idx z = 2; z < 15; z++)
			for (idx y = 2; y < 15; y++)
				for (idx x = 2; x < 15; x++) {
					dreal rho = 0, jx = 0, jy = 0, jz = 0;
					for (int q = 0; q < 27; q++) {
						const dreal f = fine.hfs[df_cur](c2fWriteSlot(q), x, y, z);
						rho += f;
						jx += VELOCITY[q][0] * f;
						jy += VELOCITY[q][1] * f;
						jz += VELOCITY[q][2] * f;
					}
					const dreal u = jx / rho, v = jy / rho, w = jz / rho;
					const auto e = exact_field(x * 0.5 - 0.25, y * 0.5 - 0.25, z * 0.5 - 0.25);
					const double rho_f = e[0];
					const dreal off = static_cast<dreal>(sigma * rho_f / (3 * omega_d));
					const dreal diag = static_cast<dreal>(2 * sigma * rho_f / (9 * omega_d));
					const double targets[6] = {
						-static_cast<double>(off) * Gxy,			// xy
						-static_cast<double>(off) * Gxz,			// xz
						-static_cast<double>(off) * Gyz,			// yz
						// diagonal k-moments carry HALF the component
						// differences (the 3/2 prefactor of Eqs. 7.8/7.9
						// against the 3 of 7.5-7.7; see the header)
						-diag * (2 * Gxx - Gyy - Gzz) / 2,			// xx
						-diag * (-Gxx + 2 * Gyy - Gzz) / 2,			// yy
						-diag * (-Gxx - Gyy + 2 * Gzz) / 2,			// zz
					};
					const std::array<dreal, 27> eq = equilibriumOnHost(rho, u, v, w);
					double rec[6] = {0, 0, 0, 0, 0, 0};
					for (int q = 0; q < 27; q++) {
						const dreal fn = fine.hfs[df_cur](c2fWriteSlot(q), x, y, z) - eq[q];
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
		CHECK_MESSAGE(
			bad == 0,
			fmt::format(
				"S2 [{}] recovered Pi^neq == filtered-strain targets (max abs err = {:.3e}, max rel err = {:.3e})",
				define_name, max_abs_err, max_rel_err
			)
		);
	}
}

TEST_SUITE_END();
