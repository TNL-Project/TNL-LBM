/*
 * Schönherr-2015 ch.7 AMR conversion -- §7.2 compact-moment (C2F CM) exactness
 * locks (commit 10 / plan row 11, stage T10 of
 * .omo/plans/schonherr-ch7-conversion.md).
 *
 * NORMATIVE REFERENCE: the CODE family of the CM branch
 * (include/lbm3d/d3q27/amr_coupling.h, the default branch). The T9 §7.2
 * equation audit (.omo/evidence/schonherr_conversion/eq-audit-draft.md, plan
 * Appendix A ¶B) verdict: 45/56 formulas match the printed §7.2, 0 code bugs;
 * the only code<->print deviations are Eqs. 7.18/7.23/7.24 (+ cyclic b/c
 * analogs), where the CODE family is the nodal-consistent one (proof in the
 * audit's §A.3.3-note) and the print carries suspected errata (concern R1).
 * Per the adopted disposition, these locks encode the CODE family -- a
 * print-aligned implementation of 7.18/7.23/7.24 must FAIL case (T10c) here,
 * never the reverse.
 *
 * Cases (the task letters refer to plan row 11 / T10):
 * - (T10a) constant-field destination fills: bitwise cross-cell uniformity,
 *   determinism across launches, DF value lock at the Test-1 tolerance class.
 * - (T10b) linear rho + linear velocity exactness on BOTH destination rows
 *   of all six faces at the production (post-re-anchor) patch geometry.
 * - (T10c) linear rho + pure-quadratic velocity + CE-consistent strain (the
 *   R1 exactness class) on the same production geometry.
 * - (T10d) avk-cancellation identity at t=(0,0,0): the full Step-E/F forms
 *   reduce to the §A.2.3 sk* means; A/B/C aggregates vanish bitwise.
 * - (T10e) two-row window census per face on the 16^3 fixture: both
 *   destination rows read the {c=-1, c=0} / {c=gs-1, c=gs} nominal pair at
 *   |t_rel| = 0.25 (the same fdiv2/window arithmetic the SimInit
 *   registration assertion uses).
 * - (T10f) corner destination census on the 3088-cell fixture: every
 *   destination cell of the disjoint patch partition is written exactly
 *   once per fill launch (coverage measured; disjointness is the row-3
 *   registration suite's structural lock, cited in the case).
 * (T10g, the C2F_EQ_ONLY/DEV_ONLY/NORM_ONLY/SHEAR_ONLY compile-and-run
 * smokes, lives in tests/test_amr_c2f_debug_smoke.cu -- the defines are
 * per-TU compile-time switches of the kernel semantics and cannot share a
 * doctest binary with the default-branch build (ODR hazard on the kernel
 * template symbol); see the tests/CMakeLists.txt block.)
 *
 * Mock geometry for (T10a)-(T10c) mirrors the post-re-anchor production
 * band of the contract doc (docs/AMR-schonherr-ch7-target-contract.md
 * sections 1.1/1.2): fine block offset 9, local 14, overlap 2 (the "1 4 4 4
 * 8 8 8" fixture's level-1 block; destination rows = fine locals {-2, -1}
 * on min faces and {14, 15} on max faces), coarse block 16^3, overlap 2,
 * offset 0. The six launch rectangles are copied verbatim from the row-3
 * halo-fingerprint lock (tests/unit/test_amr_schonherr_registration.cu's
 * expected patch table), so the fills write exactly the production
 * destination cells. Coarse cells with any coordinate outside
 * [GO-1, GO+GS] = [3, 12] are NaN-poisoned: valid nominal windows read only
 * c in [-1, gs] per axis (the vertex-straddling band), so a stale or
 * misregistered window read contaminates the output with NaN and fails the
 * finiteness rail. The map is all-fluid: a carve shift would move a window
 * onto a poisoned cell, so the carve pre-pass is pinned provably inert on
 * valid faces (the carve lanes are Tests 10-13/17 of
 * tests/test_amr_coupling.cu, untouched by this suite).
 *
 * TOLERANCE DOCUMENTATION (plan row 11 MUST-DO: each numeric tolerance is
 * derived from first principles; bitwise only where structurally sound):
 * - BITWISE (sound): cross-cell uniformity and re-launch determinism of a
 *   constant-field fill (cell-independent inputs + a deterministic kernel
 *   => identical outputs); the t=0 vanishing of the A/B/C aggregates (every
 *   term carries a coordinate factor, and coeff * +0.0 = +0.0 for finite
 *   coeff); |t_rel| == 0.25 (dyadic arithmetic, exact in float); integer
 *   window/census identities (both-frames isolation and the 3088 counts).
 * - NOT bitwise (unsound): filled DFs vs COLL::setEquilibrium on the host,
 *   and recovered fill macros vs the analytic field. The reconstruction
 *   evaluates the cumulant back-transformation (amr_coupling.h Step G/H)
 *   while setEquilibrium evaluates the equilibrium polynomials directly;
 *   the two paths are algebraically equal but arithmetically distinct
 *   (different rounding sequences), so last-ulp differences are legitimate
 *   and the constant-field vs-equilibrium equality is locked at the Test-1
 *   tolerance class (rtol 1e-6 / atol 1e-8), the analytically-seeded fields
 *   at the Tests-8/9 tolerance class (rtol 1e-4 / atol 1e-6). The
 *   separation argument: the measured fp floor of the full pipeline on
 *   these fields is <= ~7e-7 relative (rho) and <= ~1.2e-7 absolute
 *   (velocities) -- float32, eps = 1.19e-7, a few hundred flops at O(1)
 *   magnitudes -- while a PRINT-FAMILY (7.18/7.23/7.24 singleton-
 *   coefficient) implementation injects a systematic fit error of order
 *   |q| * (1/4)^2 / 2 ~ 1e-5 .. 1e-4 on the (T10c) field (the audit's
 *   §A.3.3-note u = qy^2 example: a_yy comes out q/2 instead of the
 *   nodal-consistent q) -- at least a decade above the gate and two above
 *   the floor.
 * - (T10d) cancellation tolerance: the avk cancellation is algebraic, not
 *   associativity-safe: with s = fl(a_y + b_x) and m = fl(n1o8 * sk) (the
 *   1/8 scaling is exact -- a power of two), the full form evaluates
 *   fl(s + fl(m - s)) whose forward error against m is bounded by
 *   eps*(|m-s|+|m|) <= eps*(2|m|+|s|); the diagonal X/Y forms have the same
 *   structure. The test locks 8*eps*factor*(|s|+|m|) per quantity (>= 4x
 *   margin over the derivation, absorbing the final scaling multiply), over
 *   8192 deterministic pseudo-random states with coefficient magnitudes
 *   spanning +/-8.
 */

#include <algorithm>
#include <array>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <limits>
#include <string>
#include <vector>

#include <fmt/core.h>

#include <doctest/doctest.h>

#include "lbm3d/core.h"
#include "lbm3d/amr_state.h"
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

using idx = typename TRAITS::idx;
using idx3d = typename TRAITS::idx3d;
using dreal = typename TRAITS::dreal;
using lat_t = Lattice<3, typename TRAITS::real, idx>;
using DATA = typename NSE_CONFIG::DATA;
using LBM_KS = typename NSE_CONFIG::template KernelStruct<dreal>;
using BLOCK = LBM_BLOCK<NSE_CONFIG>;
using SyncDirection = TNL::Containers::SyncDirection;

namespace {

// mock lattice sizes (coarse 16^3 real cells = the level-0 fixture block of
// the "1 4 4 4 8 8 8" configuration, fine 14^3 real cells = the re-anchored
// level-1 block; both with a 2-deep overlap and production offsets)
constexpr idx COARSE_N = 16;
constexpr idx FINE_N = 14;
constexpr idx OV = 2;
constexpr idx FINE_OFF = 9;  // 2*go + 1 with go = 4 (the re-anchored indexer)
constexpr idx COARSE_OFF = 0;
constexpr idx GO = 4;  // footprint origin (coarse)
constexpr idx GS = 8;  // footprint size (coarse)

// relaxation times of the two levels for nu_lb_coarse = 0.05 (nu_lb_fine
// double at the 2:1 refinement; same values as tests/test_amr_coupling.cu so
// the sigma/omega ratios are exercised for real)
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

// reference equilibrium computed on the host with the same
// COLL::setEquilibrium implementation that the coupling kernel uses on the
// device (same helper as tests/test_amr_coupling.cu)
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

// minimal mock of an LBM block's device data (plain NDArrays with a 2-cell
// overlap layer, hand-wired into the kernel-facing DATA structure exactly
// like LBM_BLOCK::allocateDeviceData does -- trimmed from
// tests/test_amr_coupling.cu's MockBlock to the C2F surface)
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

		// GEO_FLUID (0) everywhere: the C2F carve pre-pass never fires
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

// Store the post-collision DF of direction `q` in the slot (and array) the
// coupling kernel reads back as direction `q` (pattern-parametrized idiom
// from tests/test_amr_coupling.cu)
void storePostCollisionDF(MockBlock& block, bool even_iter, int q, idx x, idx y, idx z, dreal value)
{
#ifdef AB_PATTERN
	static_cast<void>(even_iter);
	block.hfs[df_out](q, x, y, z) = value;
#elif defined(AA_PATTERN)
	block.hfs[df_cur](even_iter ? opposite_direction(q) : q, x, y, z) = value;
#endif
}

// Direction slot in the fine df_cur array where the coarse-to-fine fill
// stores the DF of direction q
int c2fWriteSlot(int q)
{
#ifdef AB_PATTERN
	return q;
#elif defined(AA_PATTERN)
	return opposite_direction(q);
#endif
}

// D3Q27 lattice weight of direction q (product weights)
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
// `FIELD::fill(x,y,z) = {rho, vx, vy, vz, Gxx, Gyy, Gzz, Gxy, Gxz, Gyz}` --
// the CE-consistent strain construction (same as tests/test_amr_coupling.cu's
// fillFieldCE): the partner non-equilibrium is seeded from the analytic
// symmetric strain G of the velocity field via the second-order Hermite
// moment
//   f_neq(q) = -(rho / (3 omega_s)) * (w_q / (2 cs^4)) * (c_q c_q - cs^2 I) : G
// so that the source cells' raw non-equilibrium pressure tensor satisfies
// Pi_ab(f_neq) = -(rho / (3 omega_s)) * G_ab and hence the code's
// reconstructed k-moments are analytic in the strain data: k_xy = Gxy
// (Eqs. 7.5-7.7 carry the prefactor 3), while the diagonal differences
// carry HALF the component differences, k_xx_yy = (Gxx - Gyy)/2 and
// k_xx_zz = (Gxx - Gzz)/2 (Eqs. 7.8/7.9 carry 3/2). This construction is
// what makes the CE-consistent exactness class well-posed: the fitted
// gradients (Eqs. 7.19-7.21 class) and the strain moments (sk*) carry the
// SAME analytic derivative information.
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

// fill the whole stored extent of the block with the equilibrium of a
// constant macroscopic state (one shared array value per direction, so the
// input state is cell-independent bitwise -- the premise of the (T10a)
// uniformity lock)
void fillUniform(MockBlock& block, bool even_iter, dreal rho, dreal u0, dreal v0, dreal w0)
{
	const std::array<dreal, 27> eq = equilibriumOnHost(rho, u0, v0, w0);
	for (uint8_t dfty = 0; dfty < DFMAX; dfty++)
		for (int q = 0; q < 27; q++)
			for (idx z = -block.ov; z < block.size + block.ov; z++)
				for (idx y = -block.ov; y < block.size + block.ov; y++)
					for (idx x = -block.ov; x < block.size + block.ov; x++)
						block.hfs[dfty](q, x, y, z) = eq[q];
	for (idx z = -block.ov; z < block.size + block.ov; z++)
		for (idx y = -block.ov; y < block.size + block.ov; y++)
			for (idx x = -block.ov; x < block.size + block.ov; x++)
				for (int q = 0; q < 27; q++)
					storePostCollisionDF(block, even_iter, q, x, y, z, eq[q]);
}

// poison one coarse cell's DFs with NaN in every array/slot (contamination
// trap: any kernel read of this cell lands NaN in the output)
void poisonCellDFs(MockBlock& block, idx x, idx y, idx z)
{
	const dreal nan = std::numeric_limits<dreal>::quiet_NaN();
	for (uint8_t dfty = 0; dfty < DFMAX; dfty++)
		for (int q = 0; q < 27; q++)
			block.hfs[dfty](q, x, y, z) = nan;
}

// NaN-poison every coarse cell with any coordinate outside the nominal-read
// band [GO-1, GO+GS] = [3, 12]. All nominal windows of the production
// destination rows (the face-normal ring pairs {c=-1, c=0} / {c=gs-1, c=gs}
// and the tangent trilinear pairs) read cells inside this band.
void poisonOutsideNominalBand(MockBlock& coarse)
{
	for (idx z = -coarse.ov; z < coarse.size + coarse.ov; z++)
		for (idx y = -coarse.ov; y < coarse.size + coarse.ov; y++)
			for (idx x = -coarse.ov; x < coarse.size + coarse.ov; x++) {
				const bool in_band = x >= GO - 1 && x <= GO + GS && y >= GO - 1 && y <= GO + GS && z >= GO - 1 && z <= GO + GS;
				if (! in_band)
					poisonCellDFs(coarse, x, y, z);
			}
}

// launch the C2F kernel over [begin, end) in fine local coordinates (the
// block-offset parameters map between the indexer frames)
void launchCoarseToFine(MockBlock& fine, MockBlock& coarse, idx3d begin, idx3d end, bool coarse_even_iter)
{
	const idx3d size = end - begin;
	TNL::Backend::LaunchConfiguration launch_config;
	launch_config.blockSize = dim3(4, 4, 4);
	launch_config.gridSize = dim3(
		static_cast<unsigned>((size.x() + 3) / 4), static_cast<unsigned>((size.y() + 3) / 4), static_cast<unsigned>((size.z() + 3) / 4)
	);
	TNL::Backend::launchKernelAsync(
		cudaAMR_CoarseToFine<NSE_CONFIG>,
		launch_config,
		fine.data,
		coarse.data,
		begin,
		end,
		TAU_FINE,
		TAU_COARSE,
		coarse_even_iter,
		idx3d{FINE_OFF, FINE_OFF, FINE_OFF},
		idx3d{COARSE_OFF, COARSE_OFF, COARSE_OFF}
	);
	TNL::Backend::streamSynchronize(0);
}

// the six production C2F destination rectangles of the post-re-anchor band
// (fine local coordinates; verbatim from the row-3 halo-fingerprint lock of
// tests/unit/test_amr_schonherr_registration.cu, which asserts the runtime
// buildCouplings pushes exactly these). Union = 3088 = 18^3 - 14^3 cells,
// a disjoint partition, two destination rows per face.
struct DestRect
{
	SyncDirection face;
	const char* name;
	idx3d origin;
	idx3d size;
	int axis;		// face-normal axis (0/1/2)
	bool max_face;	// max faces: destination rows are {local, local+1}
};

const std::array<DestRect, 6> DEST_RECTS = {{
	{SyncDirection::Left, "x-min", {-2, -2, -2}, {2, 18, 18}, 0, false},
	{SyncDirection::Right, "x-max", {14, -2, -2}, {2, 18, 18}, 0, true},
	{SyncDirection::Bottom, "y-min", {0, -2, -2}, {14, 2, 18}, 1, false},
	{SyncDirection::Top, "y-max", {0, 14, -2}, {14, 2, 18}, 1, true},
	{SyncDirection::Back, "z-min", {0, 0, -2}, {14, 14, 2}, 2, false},
	{SyncDirection::Front, "z-max", {0, 0, 14}, {14, 14, 2}, 2, true},
}};

// fine cell center in coarse cell-center coordinates (home +- 1/4), with the
// production fine offset: X = fg * 0.5 - 0.25, fg = x + FINE_OFF
std::array<double, 3> fineCellCenter(idx x, idx y, idx z)
{
	return {(x + FINE_OFF) * 0.5 - 0.25, (y + FINE_OFF) * 0.5 - 0.25, (z + FINE_OFF) * 0.5 - 0.25};
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

bool closeEnough(dreal actual, double expected, dreal rtol, dreal atol)
{
	return std::abs(actual - expected) <= atol + rtol * std::abs(expected);
}

// per-(face, destination-row) mismatch and max-error statistics of the
// reconstructed fill macros against the analytic field
struct RowStats
{
	long bad = 0;
	long cells = 0;
	double max_rel_rho = 0;
	double max_abs_u = 0;
};

template <typename FIELD>
RowStats checkDestRectExact(const MockBlock& fine, const DestRect& rect, int outer_row, const FIELD& field, dreal rtol, dreal atol)
{
	RowStats stats;
	for (idx z = rect.origin.z(); z < rect.origin.z() + rect.size.z(); z++)
		for (idx y = rect.origin.y(); y < rect.origin.y() + rect.size.y(); y++)
			for (idx x = rect.origin.x(); x < rect.origin.x() + rect.size.x(); x++) {
				const idx local_n[3] = {x, y, z};
				const bool row_is_outer = rect.max_face ? (local_n[rect.axis] == FINE_N + 1) : (local_n[rect.axis] == -2);
				if (row_is_outer != (outer_row != 0))
					continue;
				stats.cells++;
				dreal rho_m, u_m, v_m, w_m;
				fineGhostMacros(fine, x, y, z, rho_m, u_m, v_m, w_m);
				const std::array<double, 3> X = fineCellCenter(x, y, z);
				const std::array<double, 4> e = field.exact(X[0], X[1], X[2]);
				const bool finite = std::isfinite(rho_m) && std::isfinite(u_m) && std::isfinite(v_m) && std::isfinite(w_m);
				const double rel_rho = std::abs(rho_m - e[0]) / e[0];
				const double abs_du = std::max({std::abs(u_m - e[1]), std::abs(v_m - e[2]), std::abs(w_m - e[3])});
				stats.max_rel_rho = std::max(stats.max_rel_rho, rel_rho);
				stats.max_abs_u = std::max(stats.max_abs_u, abs_du);
				const bool ok = finite && closeEnough(rho_m, e[0], rtol, atol) && abs_du <= atol + rtol * 0.03;
				if (! ok) {
					if (stats.bad == 0)
						INFO(fmt::format(
							"  first mismatch on {} row {}: cell=({},{},{}), finite={}, rho={:.9e} (expected {:.9e}), "
							"u=({:.9e},{:.9e},{:.9e}) (expected {:.9e},{:.9e},{:.9e})\n",
							rect.name, outer_row, x, y, z, finite, rho_m, e[0], u_m, v_m, w_m, e[1], e[2], e[3]
						));
					stats.bad++;
				}
			}
	return stats;
}

// (T10b) field, verbatim coefficients of tests/test_amr_coupling.cu's
// CMLinearField: rho and (u,v,w) linear in all three coordinates with a
// nonzero constant strain (the CE-consistent partner of fillFieldCE)
struct ExactLinearField
{
	std::array<dreal, 10> fill(idx x, idx y, idx z) const
	{
		const std::array<double, 4> e = exact(x, y, z);
		// constant strain: Gij from the linear velocity gradients
		return {
			static_cast<dreal>(e[0]), static_cast<dreal>(e[1]), static_cast<dreal>(e[2]), static_cast<dreal>(e[3]),
			dreal(2 * 0.002), dreal(2 * 0.0018), dreal(2 * 0.0014),						 // Gxx, Gyy, Gzz
			dreal(0.001 + (-0.0015)), dreal(-0.0009 + 0.001), dreal(0.0011 + (-0.0012))	 // Gxy, Gxz, Gyz
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

// (T10c) field, verbatim coefficients of tests/test_amr_coupling.cu's
// CMQuadraticField: linear rho, velocities linear + per-axis PURE quadratic
// terms (no cross terms, inside the CM exactness class). The field
// deliberately spans both R1-discriminating classes of the audit
// (§A.3.3-note / §A.5-U4): the diagonal quadratic coefficients (+0.0008 X^2
// in u, -0.0007 X^2 in v, +0.0006 X^2 in w) make the strain COMPRESSIBLE
// (k_xx_yy, k_xx_zz != 0 -- the 7.18-family discriminator: the print's
// singleton k_xx_yy in a_0 cannot reproduce these for ANY strain carrier),
// and the cross-axis quadratic coefficients (u contains y^2, z^2; v
// contains y^2, z^2; w contains y^2, z^2) drive the 7.23/7.24-family
// factor-2 discriminator (the print yields a_yy = q/2 where the
// nodal-consistent value is q). The strain partner seeded by fillFieldCE is
// the analytic sym(grad u) of this velocity -- the CE-consistent
// construction, witnessing the "CE-consistent strain partner velocity"
// reference class of the audit's §A.4-R1 (the divergence-free quadratic
// pair u = qy^2, v = w = 0 is the special case with the diagonal quadratic
// coefficients zeroed out here).
struct ExactQuadraticField
{
	std::array<dreal, 10> fill(idx x, idx y, idx z) const
	{
		const std::array<double, 4> e = exact(x, y, z);
		const double dudx = 0.002 + 2 * 0.0008 * x, dudy = -0.0015 - 2 * 0.0006 * y, dudz = 0.001 + 2 * 0.0004 * z;
		const double dvdx = 0.001 - 2 * 0.0007 * x, dvdy = 0.0018 + 2 * 0.0005 * y, dvdz = -0.0012 + 2 * 0.0003 * z;
		const double dwdx = -0.0009 + 2 * 0.0006 * x, dwdy = 0.0011 - 2 * 0.0005 * y, dwdz = 0.0014 + 2 * 0.00045 * z;
		return {
			static_cast<dreal>(e[0]), static_cast<dreal>(e[1]), static_cast<dreal>(e[2]), static_cast<dreal>(e[3]),
			static_cast<dreal>(2 * dudx), static_cast<dreal>(2 * dvdy), static_cast<dreal>(2 * dwdz),
			static_cast<dreal>(dvdx + dudy), static_cast<dreal>(dwdx + dudz), static_cast<dreal>(dwdy + dvdz)
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

// launch the C2F fill over all six production destination rectangles
void launchAllDestRects(MockBlock& fine, MockBlock& coarse, bool coarse_even_iter)
{
	for (const DestRect& rect : DEST_RECTS)
		launchCoarseToFine(fine, coarse, rect.origin, rect.origin + rect.size, coarse_even_iter);
}

// sentinel-fill every destination cell in df_cur of the fine mock
void poisonDestRectsFine(MockBlock& fine, dreal sentinel)
{
	for (const DestRect& rect : DEST_RECTS)
		for (idx z = rect.origin.z(); z < rect.origin.z() + rect.size.z(); z++)
			for (idx y = rect.origin.y(); y < rect.origin.y() + rect.size.y(); y++)
				for (idx x = rect.origin.x(); x < rect.origin.x() + rect.size.x(); x++)
					for (int q = 0; q < 27; q++)
						fine.hfs[df_cur](q, x, y, z) = sentinel;
}

// the same 16^3 periodic box in physical units as the row-3 fixture
// (tests/test_amr_subcycling.cu idiom: nu_lb coarse 0.005)
lat_t makeLattice()
{
	const int N = 16;
	const typename TRAITS::real LBM_VISCOSITY = 0.005;
	const typename TRAITS::real PHYS_HEIGHT = 0.41;
	const typename TRAITS::real PHYS_VISCOSITY = 1.5e-5;
	const typename TRAITS::real PHYS_DL = PHYS_HEIGHT / N;
	const typename TRAITS::real PHYS_DT = LBM_VISCOSITY / PHYS_VISCOSITY * PHYS_DL * PHYS_DL;

	lat_t lat;
	lat.global = typename lat_t::CoordinatesType(N, N, N);
	lat.physOrigin = typename TRAITS::point_t{0., 0., 0.};
	lat.physDl = PHYS_DL;
	lat.physDt = PHYS_DT;
	lat.physViscosity = PHYS_VISCOSITY;
	return lat;
}

// adios2.xml from the project root, anchored at this source file (the
// pytest wrapper runs the binary in a scratch CWD)
std::string adiosConfigPath()
{
	const std::filesystem::path root = std::filesystem::path(__FILE__).parent_path().parent_path().parent_path();
	return (root / "adios2.xml").string();
}

// minimal State_AMR subclass (the row-3 suite's idiom)
template <typename NSE>
struct StateLock_AMR : State_AMR<NSE>
{
	template <typename... ARGS>
	StateLock_AMR(ARGS&&... args)
	: State_AMR<NSE>(std::forward<ARGS>(args)...)
	{}

	[[nodiscard]] std::vector<std::string> getOutputDataNames() const override
	{
		return {};
	}
};

// per-case fresh state with the shared fixture (each case scopes its own
// instance: the State constructor registers a global spdlog logger per
// instance, so two states must never be alive at the same time)
StateLock_AMR<NSE_CONFIG> makeState(const std::string& id)
{
	return StateLock_AMR<NSE_CONFIG>(
		id, MPI_COMM_WORLD, makeLattice(), adiosConfigPath(), /*periodic=*/TRAITS::bool3d{true, true, true}, /*max_level=*/1
	);
}

// create the shared level-1 fixture and run SimInit (allocation, boundary
// setup, markAMRInterface tagging, buildCouplings, initial both-frames fill)
template <typename STATE>
void initFixture(STATE& state)
{
	REQUIRE(state.canCompute());
	createAMRBlocks(state.nse, parseAMRConfig<NSE_CONFIG>("1 4 4 4 8 8 8"));
	state.SimInit();
	REQUIRE(! state.nse.terminate);
	REQUIRE(state.couplings.size() == 1);
}

// (T10d) device harness: at t = (0,0,0) (1) the five A/B/C aggregates of
// Eqs. 7.44-7.48 evaluate to +0.0 bitwise (every term carries a coordinate
// factor), and (2) the full Step-F averages/cumulants of amr_coupling.h
// (:772-776, :786-798, code-verbatim) coincide with the reduced §A.2.3
// forms within the documented cancellation bound (file header). Both paths
// are evaluated on the device so the two arithmetics compare under identical
// compilation. Results per lane: max |full - reduced| error, the number of
// tolerance violations, and the aggregate-nonzero flag.
struct AvkLane
{
	dreal max_err;
	int violations;
	int agg_nonzero;
};

// deterministic LCG for lane inputs (no <random> on device): uniform floats
// in [-0.5, 0.5) scaled by `scale`
__device__ dreal drawFloat(uint32_t& state, dreal scale)
{
	state = state * 1664525u + 1013904223u;
	const dreal u01 = static_cast<dreal>((state >> 8) & 0xFFFFFFu) * (dreal(1) / dreal(16777216));
	return (u01 - dreal(0.5)) * scale;
}

__device__ dreal dabs(dreal v)
{
	return v < 0 ? -v : v;
}

__global__ void avkCancellationKernel(AvkLane* lanes, int n)
{
	const int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= n)
		return;
	uint32_t rng = 0x9E3779B9u + static_cast<uint32_t>(i) * 0x85EBCA6Bu;

	// first-order velocity coefficients (the free variables of the identity)
	const dreal a_x = drawFloat(rng, 8), a_y = drawFloat(rng, 8), a_z = drawFloat(rng, 8);
	const dreal b_x = drawFloat(rng, 8), b_y = drawFloat(rng, 8), b_z = drawFloat(rng, 8);
	const dreal c_x = drawFloat(rng, 8), c_y = drawFloat(rng, 8), c_z = drawFloat(rng, 8);
	// quadratic coefficients (only enter the A/B/C vanishing rail)
	const dreal a_xx = drawFloat(rng, 8), a_yy = drawFloat(rng, 8), a_zz = drawFloat(rng, 8);
	const dreal a_xy = drawFloat(rng, 8), a_xz = drawFloat(rng, 8), a_yz = drawFloat(rng, 8);
	const dreal a_xyz = drawFloat(rng, 8);
	const dreal b_xx = drawFloat(rng, 8), b_yy = drawFloat(rng, 8), b_zz = drawFloat(rng, 8);
	const dreal b_xy = drawFloat(rng, 8), b_xz = drawFloat(rng, 8), b_yz = drawFloat(rng, 8);
	const dreal b_xyz = drawFloat(rng, 8);
	const dreal c_xx = drawFloat(rng, 8), c_yy = drawFloat(rng, 8), c_zz = drawFloat(rng, 8);
	const dreal c_xy = drawFloat(rng, 8), c_xz = drawFloat(rng, 8), c_yz = drawFloat(rng, 8);
	const dreal c_xyz = drawFloat(rng, 8);
	// 8-node k-moment sums and the destination state
	const dreal sk_yz = drawFloat(rng, 4), sk_xz = drawFloat(rng, 4), sk_xy = drawFloat(rng, 4);
	const dreal sk_xx_yy = drawFloat(rng, 4), sk_xx_zz = drawFloat(rng, 4);
	const dreal rho = dreal(1) + drawFloat(rng, 2);		  // (0, 2)
	const dreal omega_d = dreal(1.25) + drawFloat(rng, 1);  // (0.25, 2.25)

	const dreal tx = 0, ty = 0, tz = 0;

	// (1) the aggregates at t = 0 (verbatim from amr_coupling.h :795-799)
	const dreal corr_B = no2 * a_xx * tx - b_xy * tx + a_xy * ty - no2 * b_yy * ty + a_xz * tz - b_yz * tz - b_xyz * tx * tz + a_xyz * ty * tz;
	const dreal corr_C = no2 * a_xx * tx - c_xz * tx + a_xy * ty - c_yz * ty - c_xyz * tx * ty + a_xz * tz - no2 * c_zz * tz + a_xyz * ty * tz;
	const dreal A011 = b_xz * tx + c_xy * tx + b_yz * ty + no2 * c_yy * ty + b_xyz * tx * ty + no2 * b_zz * tz + c_yz * tz + c_xyz * tx * tz;
	const dreal A101 = a_xz * tx + no2 * c_xx * tx + a_yz * ty + c_xy * ty + a_xyz * tx * ty + no2 * a_zz * tz + c_xz * tz + c_xyz * ty * tz;
	const dreal A110 = a_xy * tx + no2 * b_xx * tx + no2 * a_yy * ty + b_xy * ty + a_yz * tz + b_xz * tz + a_xyz * tx * tz + b_xyz * ty * tz;
	int agg_nonzero = 0;
	if (corr_B != dreal(0) || corr_C != dreal(0) || A011 != dreal(0) || A101 != dreal(0) || A110 != dreal(0))
		agg_nonzero = 1;

	// (2) the avg_k cancellation (averages verbatim from :772-776, cumulant
	// structure from :786-798) vs the reduced forms of the audit's §A.2.3
	const dreal sigma = n1o2;
	const dreal off_factor = sigma * rho / (no3 * omega_d);

	const dreal avg_k_yz = n1o8 * sk_yz - (b_z + c_y);
	const dreal avg_k_xz = n1o8 * sk_xz - (a_z + c_x);
	const dreal avg_k_xy = n1o8 * sk_xy - (a_y + b_x);
	const dreal avg_k_xx_yy = n1o8 * sk_xx_yy - (a_x - b_y);
	const dreal avg_k_xx_zz = n1o8 * sk_xx_zz - (a_x - c_z);

	const dreal C011_full = -off_factor * (b_z + c_y + avg_k_yz + A011);
	const dreal C101_full = -off_factor * (a_z + c_x + avg_k_xz + A101);
	const dreal C110_full = -off_factor * (a_y + b_x + avg_k_xy + A110);
	const dreal X_full = a_x - b_y + avg_k_xx_yy + corr_B;
	const dreal Y_full = a_x - c_z + avg_k_xx_zz + corr_C;

	const dreal C011_red = -off_factor * (n1o8 * sk_yz);
	const dreal C101_red = -off_factor * (n1o8 * sk_xz);
	const dreal C110_red = -off_factor * (n1o8 * sk_xy);
	const dreal X_red = n1o8 * sk_xx_yy;
	const dreal Y_red = n1o8 * sk_xx_zz;

	// per-quantity locks: |full - reduced| <= 8 eps factor (|s| + |m|)
	// (header derivation; the diagonal cumulants C200/C020/C002 are linear
	// in X/Y with diag_eq identical on both paths, so locking X and Y locks
	// all three)
	dreal max_err = 0;
	int violations = 0;
	struct AvkCheck
	{
		dreal full, red, s, m, factor;
	};
	const AvkCheck checks[5] = {
		{C011_full, C011_red, b_z + c_y, n1o8 * sk_yz, off_factor},
		{C101_full, C101_red, a_z + c_x, n1o8 * sk_xz, off_factor},
		{C110_full, C110_red, a_y + b_x, n1o8 * sk_xy, off_factor},
		{X_full, X_red, a_x - b_y, n1o8 * sk_xx_yy, 1},
		{Y_full, Y_red, a_x - c_z, n1o8 * sk_xx_zz, 1},
	};
	for (int k = 0; k < 5; k++) {
		const dreal err = dabs(checks[k].full - checks[k].red);
		const dreal tol = dreal(8) * static_cast<dreal>(FLT_EPSILON) * checks[k].factor * (dabs(checks[k].s) + dabs(checks[k].m));
		max_err = max_err < err ? err : max_err;
		if (err > tol)
			violations++;
	}

	lanes[i].max_err = max_err;
	lanes[i].violations = violations;
	lanes[i].agg_nonzero = agg_nonzero;
}

} // anonymous namespace

TEST_SUITE_BEGIN("amr_schonherr_exactness");

// (T10a) constant field exact on both destination rows per face: for a
// constant input state every polynomial coefficient except d_0 = rho and
// a_0/b_0/c_0 = (u,v,w) vanishes EXACTLY in float (symmetric +-1/2 nodal
// sums cancel pairwise), so rho_f = rho, u_f = u, ... are bitwise the
// constants and the fill is cell-independent. Structurally-sound BITWISE
// locks: (i) cross-cell uniformity -- every written destination cell of
// all six rects carries the identical 27 DF values (cell-independent
// inputs + a deterministic kernel => identical outputs, across BOTH
// destination rows whose evaluation points differ only in terms that are
// exactly zero); (ii) re-launch determinism -- a second fill over freshly
// poisoned destinations reproduces the first fill bitwise. The DF-vs-
// equilibrium lock is at the Test-1 tolerance class (two arithmetically
// distinct but algebraically equal reconstruction paths, see the header).
TEST_CASE("T10a constant-field fill bitwise uniformity + determinism (production destination rows)")
{
	const dreal rho0 = 1.0, u0 = 0.05, v0 = -0.03, w0 = 0.02;
	const std::array<dreal, 27> expected = equilibriumOnHost(rho0, u0, v0, w0);
	constexpr dreal SENTINEL = -777;

	// first fill over poisoned destinations
	std::array<dreal, 27> ref;
	{
		MockBlock coarse, fine;
		coarse.allocate(COARSE_N);
		fine.allocate(FINE_N);
		fillUniform(coarse, /*even_iter=*/false, rho0, u0, v0, w0);
		poisonOutsideNominalBand(coarse);
		coarse.copyToDevice();
		poisonDestRectsFine(fine, SENTINEL);
		fine.copyToDevice();

		launchAllDestRects(fine, coarse, false);
		fine.copyToHost();

		// reference cell = the corner cell (-2,-2,-2) of the x-min rect
		for (int q = 0; q < 27; q++)
			ref[q] = fine.hfs[df_cur](c2fWriteSlot(q), -2, -2, -2);

		// (i) cross-cell bitwise uniformity + DF-vs-equilibrium (Test-1 class)
		long cells = 0, non_uniform = 0, beyond_eq_tol = 0, sentinel_left = 0;
		double max_eq_err = 0;
		for (const DestRect& rect : DEST_RECTS)
			for (idx z = rect.origin.z(); z < rect.origin.z() + rect.size.z(); z++)
				for (idx y = rect.origin.y(); y < rect.origin.y() + rect.size.y(); y++)
					for (idx x = rect.origin.x(); x < rect.origin.x() + rect.size.x(); x++) {
						cells++;
						for (int q = 0; q < 27; q++) {
							const dreal actual = fine.hfs[df_cur](c2fWriteSlot(q), x, y, z);
							if (actual == SENTINEL)
								sentinel_left++;
							if (std::memcmp(&actual, &ref[q], sizeof(dreal)) != 0)
								non_uniform++;
							if (! closeEnough(actual, expected[q], 1e-6f, 1e-8f))
								beyond_eq_tol++;
							max_eq_err = std::max(max_eq_err, std::abs(static_cast<double>(actual) - expected[q]));
						}
					}
		INFO(fmt::format("T10a: {} destination cells, max |DF - eq| = {:.3e}\n", cells, max_eq_err));
		CHECK(cells == 3088);
		CHECK(sentinel_left == 0);
		CHECK(non_uniform == 0);
		CHECK(beyond_eq_tol == 0);
	}

	// (ii) re-launch determinism: freshly poisoned destinations refill
	// bitwise-identically (same inputs, deterministic kernel)
	{
		MockBlock coarse, fine;
		coarse.allocate(COARSE_N);
		fine.allocate(FINE_N);
		fillUniform(coarse, /*even_iter=*/false, rho0, u0, v0, w0);
		poisonOutsideNominalBand(coarse);
		coarse.copyToDevice();
		poisonDestRectsFine(fine, SENTINEL);
		fine.copyToDevice();

		launchAllDestRects(fine, coarse, false);
		fine.copyToHost();

		long different = 0;
		for (const DestRect& rect : DEST_RECTS)
			for (idx z = rect.origin.z(); z < rect.origin.z() + rect.size.z(); z++)
				for (idx y = rect.origin.y(); y < rect.origin.y() + rect.size.y(); y++)
					for (idx x = rect.origin.x(); x < rect.origin.x() + rect.size.x(); x++)
						for (int q = 0; q < 27; q++)
							if (std::memcmp(&fine.hfs[df_cur](c2fWriteSlot(q), x, y, z), &ref[q], sizeof(dreal)) != 0)
								different++;
		CHECK(different == 0);
	}
}

// shared body of (T10b) and (T10c): launch the CE-filled field over the six
// production destination rectangles and assert the reconstructed macros
// against the analytic field per (face, destination row). Tolerance
// (header): rtol = 1e-4, atol = 1e-6 -- the Tests-8/9 gate class, decades
// above the measured fp floor and below the print-family systematic.
template <typename FIELD>
void runProductionExactness(const FIELD& field, const char* field_name)
{
	MockBlock coarse, fine;
	coarse.allocate(COARSE_N);
	fine.allocate(FINE_N);
	fillFieldCE(coarse, /*even_iter=*/false, field);
	poisonOutsideNominalBand(coarse);
	coarse.copyToDevice();

	launchAllDestRects(fine, coarse, false);
	fine.copyToHost();

	constexpr dreal rtol = 1e-4f, atol = 1e-6f;
	for (const DestRect& rect : DEST_RECTS) {
		for (int row = 0; row < 2; row++) {
			const RowStats stats = checkDestRectExact(fine, rect, row, field, rtol, atol);
			INFO(fmt::format(
				"{} {} row {}: {} cells, max rel rho err = {:.3e}, max abs vel err = {:.3e}, bad = {}\n",
				field_name, rect.name, row, stats.cells, stats.max_rel_rho, stats.max_abs_u, stats.bad
			));
			CHECK(stats.cells > 0);
			CHECK(stats.bad == 0);
		}
	}
}

// (T10b) linear rho + linear velocity exact across BOTH destination rows of
// every face (the re-paired post-re-anchor rows; the analytic field is
// seeded by the global (i,j,k) of the coarse source cells). This is the
// production-geometry pin of the Test-8 linear exactness class: the
// zero-offset box mock there cannot see the destination registration
// (offset 9 / rows {-2,-1} and {14,15} / ring-pair windows {c-1, c0}).
TEST_CASE("T10b linear-field exactness on both destination rows per face (production registration)")
{
	runProductionExactness(ExactLinearField{}, "T10b linear");
}

// (T10c) pure-quadratic VELOCITY + CE-consistent strain exact on the same
// production rows -- the R1 exactness lock: the CODE family of
// Eqs. 7.18/7.23/7.24 is the reference (never the print). A print-aligned
// implementation fails this case systematically at O(|q|/32) ~ 1e-5..1e-4
// on this field while the code family's measured floor is <= ~2e-7; the
// gate sits between (see the header tolerance note). Tests 9/10/11/17 of
// tests/test_amr_coupling.cu pin this field on the zero-offset mock box;
// this case pins it on the production destination rows.
TEST_CASE("T10c quadratic-velocity CE-strain exactness on both destination rows per face (R1 code family)")
{
	runProductionExactness(ExactQuadraticField{}, "T10c quadratic-CE");
}

// (T10d) avk-cancellation identity at t = (0,0,0) (the F2C evaluation
// point; eq-audit §A.2.3 reduced forms): (1) all five A/B/C aggregates
// vanish bitwise; (2) the full Steps-E/F averages/cumulants of
// amr_coupling.h (:772-776, :786-798) coincide with the reduced sk* means
// within the documented cancellation bound (header). The harness evaluates
// BOTH paths verbatim on the device over 8192 deterministic pseudo-random
// states -- the C2F kernel itself never evaluates t = 0 (its destinations
// sit at |t_rel| = 0.25 nominal / 0.75 carved), so the identity cannot go
// through the production launcher; the F2C arm of T14 (commit 13) will
// consume this identity, and the audit requires it locked before that.
TEST_CASE("T10d avk cancellation at t=0 reduces cumulants to the sk means")
{
	constexpr int N = 8192;
	TNL::Containers::Array<AvkLane, TNL::Devices::Cuda> d_lanes(N);
	TNL::Containers::Array<AvkLane, TNL::Devices::Host> lanes(N);
	AvkLane* d_ptr = d_lanes.getData();

	TNL::Backend::LaunchConfiguration launch_config;
	launch_config.blockSize = dim3(128, 1, 1);
	launch_config.gridSize = dim3((N + 127) / 128, 1, 1);
	TNL::Backend::launchKernelAsync(avkCancellationKernel, launch_config, d_ptr, N);
	TNL::Backend::streamSynchronize(0);
	lanes = d_lanes;

	long violations = 0, agg_nonzero = 0;
	double max_err = 0;
	for (int i = 0; i < N; i++) {
		violations += lanes[i].violations;
		agg_nonzero += lanes[i].agg_nonzero;
		max_err = std::max(max_err, static_cast<double>(lanes[i].max_err));
	}
	INFO(fmt::format("T10d: {} lanes, max |full - reduced| = {:.3e}\n", N, max_err));
	CHECK(agg_nonzero == 0);
	CHECK(violations == 0);
}

// (T10e) two-row window census per face on the real 16^3 fixture: for every
// C2F destination cell, replicate the kernel's fdiv2/axis_window nominal
// arithmetic (the same rules the SimInit registration assertion
// checkCouplingMapPattern uses) and assert BOTH destination rows of every
// face read exactly the nominal ring pair -- {c=-1, c=0} on min faces /
// {c=gs-1, c=gs} on max faces -- at |t_rel| = 0.25 exactly (dyadic
// arithmetic, the comparison is exact in float). A 1-cell registration
// error in either direction (patch origin or indexer) breaks the pair
// identity here.
TEST_CASE("T10e destination-row window census: both rows read the ring pair")
{
	StateLock_AMR<NSE_CONFIG> state = makeState("schonherr_exactness_windows");
	initFixture(state);

	const auto& coupling = state.couplings.front();
	REQUIRE(coupling.patches.size() == 6);

	BLOCK* coarse = state.nse.getBlocksAtLevel(0).front();
	REQUIRE(coarse->offset == idx3d{0, 0, 0});
	BLOCK* fine = state.nse.getBlocksAtLevel(1).front();
	REQUIRE(fine->offset == idx3d{2 * GO + 1, 2 * GO + 1, 2 * GO + 1});

	// kernel fdiv2 (true floor division for negative fg)
	const auto fdiv2 = [](idx v) -> idx
	{
		return v >= 0 ? v / 2 : -((-v + 1) / 2);
	};

	long total = 0;
	for (std::size_t pi = 0; pi < coupling.patches.size(); pi++) {
		const auto& patch = coupling.patches[pi];
		const int axis = (patch.face == SyncDirection::Left || patch.face == SyncDirection::Right) ? 0
					   : (patch.face == SyncDirection::Bottom || patch.face == SyncDirection::Top) ? 1
																								   : 2;
		const bool max_face = (patch.face == SyncDirection::Right || patch.face == SyncDirection::Top || patch.face == SyncDirection::Front);

		// expected nominal source pair (coarse indexer == global frame; the
		// fixture is cubic: go = 4, gs = 8 on every axis)
		const idx want0 = max_face ? GO + GS - 1 : GO - 1;
		const idx want1 = max_face ? GO + GS : GO;

		const idx fine_off_a = fine->offset[axis];
		long rows_seen[2] = {0, 0};
		long bad_pair = 0, bad_trel = 0, bad_row = 0;
		for (idx x = patch.fine_origin.x(); x < patch.fine_origin.x() + patch.fine_size.x(); x++)
			for (idx y = patch.fine_origin.y(); y < patch.fine_origin.y() + patch.fine_size.y(); y++)
				for (idx z = patch.fine_origin.z(); z < patch.fine_origin.z() + patch.fine_size.z(); z++) {
					const idx lc[3] = {x, y, z};
					const idx fg = fine_off_a + lc[axis];
					const idx home = fdiv2(fg);
					const idx p = fg & 1;
					// nominal per-axis window of axis_window (coarse_off = 0;
					// the storability clamp can never engage here: start in
					// [3, 11] inside [0, 16 - 1])
					const idx start = home - 1 + p;
					if (start != want0 || start + 1 != want1)
						bad_pair++;
					// evaluation point relative to the window center
					const double t = static_cast<double>(home) + (p ? 0.25 : -0.25);
					const double t_rel = t - (static_cast<double>(start) + 0.5);
					if (t_rel * t_rel != 0.0625)
						bad_trel++;
					// destination-row identity on the face normal: the patch
					// rows on this axis must be exactly the overlap rows
					// {-2,-1} (min face) or {local, local+1} (max face)
					const idx row = lc[axis];
					if (max_face ? (row == fine->local[axis] + 1) : (row == -2))
						rows_seen[0]++;
					else if (max_face ? (row == fine->local[axis]) : (row == -1))
						rows_seen[1]++;
					else
						bad_row++;
				}
		total += rows_seen[0] + rows_seen[1];
		INFO(fmt::format(
			"T10e patch {} (axis {}, {} face): rows outer/inner = {}/{}, bad_pair = {}, bad_trel = {}, bad_row = {}\n",
			pi, axis, max_face ? "max" : "min", rows_seen[0], rows_seen[1], bad_pair, bad_trel, bad_row
		));
		CHECK(rows_seen[0] == rows_seen[1]);  // both destination rows, same census
		CHECK(bad_pair == 0);
		CHECK(bad_trel == 0);
		CHECK(bad_row == 0);
	}
	CHECK(total == 3088);  // the full destination census of the fixture
}

// (T10f) corner destination census on the 3088 fixture: every destination
// cell of the disjoint patch partition is written EXACTLY ONCE per fill
// launch. >= 1 (coverage) is measured here: the destinations are
// sentinel-poisoned on both AB frames and the fill launches are replayed
// under both rotations (updateKernelDataForLevel +
// launchCoarseToFineTransfers, the straight SimUpdate call sequence; the
// production cycle fills frame 0 only, the frame-1 replay pins the
// machinery's frame isolation); every
// destination cell must hold a non-sentinel finite DF after its frame's
// fill while the OTHER frame and every non-destination cell stay bitwise
// untouched (frame isolation + write-closure of the launcher). <= 1 (no
// double writes) is the disjoint-partition structural lock of the row-3
// registration suite ("C2F patch destination census + splits": pushed
// volume == union size == 3088). Coverage + disjointness = exactly once.
// The 64 corner cells (8 boxes of 2^3, the overlap-prone class whose
// bounding volumes meet three patch families) are enumerated separately.
TEST_CASE("T10f fill census: every destination cell written exactly once per fill launch")
{
	constexpr dreal SENTINEL = -777;

	StateLock_AMR<NSE_CONFIG> state = makeState("schonherr_exactness_fill_census");
	initFixture(state);

	BLOCK* fine = state.nse.getBlocksAtLevel(1).front();
	const idx3d off = fine->offset;
	const idx3d ov{fine->df_overlap_X(), fine->df_overlap_Y(), fine->df_overlap_Z()};
	REQUIRE(ov == idx3d{2, 2, 2});
	const auto& coupling = state.couplings.front();
	REQUIRE(coupling.patches.size() == 6);

	// deep snapshot of both AB DF arrays over the full stored extent (host
	// mirrors take (offset + local) coordinates, valid window
	// [offset - ov, offset + local + ov))
	fine->copyDFsToHost();
	const idx3d lo = off - ov;
	const idx3d hi = off + fine->local + ov;
	const long storage_cells = static_cast<long>(hi.x() - lo.x()) * (hi.y() - lo.y()) * (hi.z() - lo.z());
	REQUIRE(storage_cells == 18 * 18 * 18);
	std::vector<dreal> snap0, snap1;
	snap0.reserve(27 * storage_cells);
	snap1.reserve(27 * storage_cells);
	for (int q = 0; q < 27; q++)
		for (idx z = lo.z(); z < hi.z(); z++)
			for (idx y = lo.y(); y < hi.y(); y++)
				for (idx x = lo.x(); x < hi.x(); x++) {
					snap0.push_back(fine->hfs[0](q, x, y, z));
					snap1.push_back(fine->hfs[1](q, x, y, z));
				}

	// destination-union membership predicate (fine LOCAL coordinates)
	const auto in_dest = [&coupling](idx lx, idx ly, idx lz) -> bool
	{
		for (const auto& patch : coupling.patches)
			if (lx >= patch.fine_origin.x() && lx < patch.fine_origin.x() + patch.fine_size.x() && ly >= patch.fine_origin.y()
				&& ly < patch.fine_origin.y() + patch.fine_size.y() && lz >= patch.fine_origin.z()
				&& lz < patch.fine_origin.z() + patch.fine_size.z())
				return true;
		return false;
	};

	// sentinel-poison every destination cell on both frames
	long poisoned = 0;
	for (const auto& patch : coupling.patches)
		for (idx z = patch.fine_origin.z(); z < patch.fine_origin.z() + patch.fine_size.z(); z++)
			for (idx y = patch.fine_origin.y(); y < patch.fine_origin.y() + patch.fine_size.y(); y++)
				for (idx x = patch.fine_origin.x(); x < patch.fine_origin.x() + patch.fine_size.x(); x++) {
					for (int q = 0; q < 27; q++) {
						fine->hfs[0](q, off.x() + x, off.y() + y, off.z() + z) = SENTINEL;
						fine->hfs[1](q, off.x() + x, off.y() + y, off.z() + z) = SENTINEL;
					}
					poisoned++;
				}
	REQUIRE(poisoned == 3088);
	fine->copyDFsToDevice();

	// replay the production-cycle fill (its single fill targets frame 0
	// under the substep-0 rotation)
	state.nse.updateKernelDataForLevel(1, 0);
	state.launchCoarseToFineTransfers(1);
	fine->copyDFsToHost();

	// census on frame 0's physical array + frame isolation; capture the
	// frame-0 fill values for the later cross-frame bitwise check
	std::vector<dreal> post_f0;
	post_f0.reserve(27 * 3088);
	long still_sentinel_f0 = 0, non_finite_f0 = 0, other_frame_touched = 0;
	long corner_total = 0, corner_written = 0;
	for (const auto& patch : coupling.patches)
		for (idx z = patch.fine_origin.z(); z < patch.fine_origin.z() + patch.fine_size.z(); z++)
			for (idx y = patch.fine_origin.y(); y < patch.fine_origin.y() + patch.fine_size.y(); y++)
				for (idx x = patch.fine_origin.x(); x < patch.fine_origin.x() + patch.fine_size.x(); x++) {
					const int n_out = static_cast<int>(x < 0 || x >= fine->local.x()) + static_cast<int>(y < 0 || y >= fine->local.y())
									+ static_cast<int>(z < 0 || z >= fine->local.z());
					bool cell_written = true;
					for (int q = 0; q < 27; q++) {
						const dreal f0 = fine->hfs[0](q, off.x() + x, off.y() + y, off.z() + z);
						const dreal f1 = fine->hfs[1](q, off.x() + x, off.y() + y, off.z() + z);
						post_f0.push_back(f0);
						if (f0 == SENTINEL) {
							still_sentinel_f0++;
							cell_written = false;
						}
						if (! std::isfinite(f0))
							non_finite_f0++;
						if (f1 != SENTINEL)
							other_frame_touched++;
					}
					if (n_out == 3) {
						corner_total++;
						if (cell_written)
							corner_written++;
					}
				}
	CHECK(still_sentinel_f0 == 0);
	CHECK(non_finite_f0 == 0);
	CHECK(other_frame_touched == 0);  // the fill writes exactly ONE frame per launch
	CHECK(corner_total == 64);
	CHECK(corner_written == 64);

	// write-closure on the non-destination storage (both frames bitwise
	// untouched everywhere outside the destination union)
	long spill = 0;
	{
		std::size_t p = 0;
		for (int q = 0; q < 27; q++)
			for (idx z = lo.z(); z < hi.z(); z++)
				for (idx y = lo.y(); y < hi.y(); y++)
					for (idx x = lo.x(); x < hi.x(); x++, p++) {
						if (in_dest(x - off.x(), y - off.y(), z - off.z()))
							continue;
						if (std::memcmp(&fine->hfs[0](q, x, y, z), &snap0[p], sizeof(dreal)) != 0
							|| std::memcmp(&fine->hfs[1](q, x, y, z), &snap1[p], sizeof(dreal)) != 0)
							spill++;
					}
	}
	CHECK(spill == 0);

	// replay the same fill machinery under the substep-1 rotation
	// (production uses no frame-1 fill under the simulated band; this
	// replay pins the machinery's frame isolation); the destination
	// cells of frame 1 must now be written, and frame 0's values must be
	// bitwise at the post-frame-0 state.
	state.nse.updateKernelDataForLevel(1, 1);
	state.launchCoarseToFineTransfers(1);
	fine->copyDFsToHost();

	long still_sentinel_f1 = 0, non_finite_f1 = 0, frame0_rewritten = 0;
	long corner_total_f1 = 0, corner_written_f1 = 0;
	std::size_t p = 0;
	for (const auto& patch : coupling.patches)
		for (idx z = patch.fine_origin.z(); z < patch.fine_origin.z() + patch.fine_size.z(); z++)
			for (idx y = patch.fine_origin.y(); y < patch.fine_origin.y() + patch.fine_size.y(); y++)
				for (idx x = patch.fine_origin.x(); x < patch.fine_origin.x() + patch.fine_size.x(); x++) {
					const int n_out = static_cast<int>(x < 0 || x >= fine->local.x()) + static_cast<int>(y < 0 || y >= fine->local.y())
									+ static_cast<int>(z < 0 || z >= fine->local.z());
					bool cell_written = true;
					for (int q = 0; q < 27; q++, p++) {
						const dreal f0 = fine->hfs[0](q, off.x() + x, off.y() + y, off.z() + z);
						const dreal f1 = fine->hfs[1](q, off.x() + x, off.y() + y, off.z() + z);
						if (f1 == SENTINEL) {
							still_sentinel_f1++;
							cell_written = false;
						}
						if (! std::isfinite(f1))
							non_finite_f1++;
						if (std::memcmp(&f0, &post_f0[p], sizeof(dreal)) != 0)
							frame0_rewritten++;
					}
					if (n_out == 3) {
						corner_total_f1++;
						if (cell_written)
							corner_written_f1++;
					}
				}
	CHECK(still_sentinel_f1 == 0);
	CHECK(non_finite_f1 == 0);
	CHECK(frame0_rewritten == 0);  // frame-1 fill is frame-isolated as well
	CHECK(corner_total_f1 == 64);
	CHECK(corner_written_f1 == 64);

	// write-closure after both fills: spill0 counts non-destination cells
	// that do NOT match the snapshot on frame 0; spill1 likewise frame 1
	long spill0 = 0, spill1 = 0;
	{
		std::size_t p2 = 0;
		for (int q = 0; q < 27; q++)
			for (idx z = lo.z(); z < hi.z(); z++)
				for (idx y = lo.y(); y < hi.y(); y++)
					for (idx x = lo.x(); x < hi.x(); x++, p2++) {
						if (in_dest(x - off.x(), y - off.y(), z - off.z()))
							continue;
						if (std::memcmp(&fine->hfs[0](q, x, y, z), &snap0[p2], sizeof(dreal)) != 0)
							spill0++;
						if (std::memcmp(&fine->hfs[1](q, x, y, z), &snap1[p2], sizeof(dreal)) != 0)
							spill1++;
					}
	}
	CHECK(spill0 == 0);
	CHECK(spill1 == 0);
}

TEST_SUITE_END();
