#pragma once

#if ! defined(AA_PATTERN) && ! defined(AB_PATTERN)
	#error "amr_coupling.h requires either AA_PATTERN or AB_PATTERN to be defined before inclusion"
#endif

#include "lbm3d/defs.h"

/**
 * \brief AMR inter-level coupling: coarse-to-fine volumetric DF rescaling.
 *
 * Fills a rectangular extent of fine-level ghost cells at a refinement
 * interface with distribution functions (DFs) reconstructed from the coarse
 * level (Chen et al. 1998, Rohde et al. 2006, Guzik et al. 2014):
 *
 * 1. The macroscopic quantities (rho, u) of the 4x4x4 coarse cells whose
 *    centers surround the fine cell center are interpolated to the fine
 *    cell with tensor-product 3rd-order Lagrange interpolation
 *    (cell-centered layout at a 2:1 ratio; trilinear 2nd-order
 *    interpolation is available as a fallback with -DC2F_TRILINEAR).
 * 2. The equilibrium part of each fine DF is re-evaluated at the
 *    interpolated macros (one `EQ::eq_*` call per direction, never shared).
 * 3. The non-equilibrium part is interpolated per direction from the same
 *    coarse cells with the same weights and rescaled by
 *    `tau_fine / tau_coarse`.
 *
 * With 2:1 refinement `nu_lb_f = 2 * nu_lb_c`, so `tau_fine != tau_coarse`
 * (tau = 3 * nu_lb + 0.5 per level): the non-equilibrium rescaling is NOT a
 * no-op. The caller computes both tau values from the per-level `lbmViscosity`
 * and passes them as kernel parameters. NO additional volume factor is
 * applied: each coarse-to-fine DF is a point density (the 1/8 volume
 * averaging belongs to the fine-to-coarse direction in todo 6).
 *
 * Cell-centered coordinate mapping (per axis): the two blocks' indexer
 * origins do NOT correspond in general (a nested fine block starts wherever
 * its footprint places it), so the mapping is computed in the GLOBAL frame.
 * For a fine indexer coordinate `x_f` the fine global coordinate is
 * `fg = x_f + fine_off`, the home coarse cell (global) is
 * `home = floor(fg / 2)` (true floor division -- valid for negative fine
 * global coordinates, unlike the truncating integer division of C++, so the
 * x_f = -1 ghost maps to the correct home cell), and the fine cell center
 * sits at +-1/4 of the home cell's width from the home cell center. The
 * 3rd-order per-axis stencil uses the 4 coarse cell centers (global)
 * {home-2, home-1, home, home+1} for even `fg` and
 * {home-1, home, home+1, home+2} for odd `fg`, unified as
 * `home - 2 + (fg & 1) + {0,1,2,3}` and converted back to the coarse
 * block's indexer frame via `- coarse_off`, with Lagrange interpolation
 * weights evaluated at the fine cell center (offset -1/4 resp. +1/4 of a
 * coarse cell width from the home cell center). For the centered stencil
 * the weights are the exact dyadic rationals {-5, 35, 105, -7}/128
 * (even `fg`) and {-7, 105, 35, -5}/128 (odd `fg`), which sum to 1 so
 * constant fields are preserved exactly; the interpolation reproduces
 * cubic fields exactly on any 4 distinct nodes. `fine_off` and
 * `coarse_off` are the blocks' indexer origins in the global coordinates
 * of their own level (`LBM_BLOCK::offset`).
 *
 * Storability guard: the nominal 4-cell stencil extends up to 2 coarse
 * cells on each side of the fine cell's home cell, so it can overreach the
 * coarse block's stored extent near block boundaries. The kernel queries
 * the coarse storage extent (sizes and overlap) from `coarse_SD.indexer`
 * and SHIFTS the per-axis stencil window so that all of its cells are
 * valid storage indices (shortening it if the per-axis extent is smaller
 * than the stencil), and evaluates the Lagrange weights of the actual
 * window at runtime in double precision, normalized so they sum to one.
 * Full nominal accuracy requires the interface ghosts to be positioned at
 * least two coarse cells inside the coarse block's stored extent; boundary
 * cells receive a shifted-stencil interpolation that is still exact for
 * cubic fields but evaluated from a non-centered window.
 *
 * Explosion strategies (Eitel-Amor et al. 2025, Fluids 10(2):31): instead
 * of interpolating from neighboring coarse cells, each fine ghost cell
 * reads ONLY its home coarse cell, which "explodes" into its 8 fine
 * subcells independently. No stencil-neighbor coupling exists at the
 * interface, so the interpolation stencil cannot inject spurious noise
 * (e.g. from superseded "inside-hidden" coarse cells):
 * - `C2F_UNIFORM_EXPLOSION`: the home cell's DFs are duplicated to every
 *   fine subcell unchanged (zeroth order; exactly preserves the home
 *   cell's moments, no equilibrium re-evaluation, no rescaling).
 * - `C2F_LINEAR_EXPLOSION`: the home cell's macros (rho, u) are taken
 *   over directly and the equilibrium is re-evaluated at those macros;
 *   the non-equilibrium part is zeroed (pure equilibrium explosion --
 *   the simplest and most stable variant).
 * With an explosion define the home cell index is clamped per axis into
 * the coarse storage extent (the analog of the storability guard; valid
 * geometries never clamp).
 *
 * Compile-time switches (C2F reconstruction strategy; the 3rd-order
 * Lagrange scheme above is the default):
 * - `C2F_TRILINEAR`: reverts the interpolation to the original 2nd-order
 *   trilinear scheme (2-point per-axis stencil {home-1+(fg&1),
 *   home+(fg&1)} with 3/4:1/4 weights, reproduced by the same runtime
 *   Lagrange machinery with a 2-node window).
 * - `C2F_LINEAR_EXPLOSION` / `C2F_UNIFORM_EXPLOSION`: the explosion
 *   strategies described above (linear takes precedence if both are
 *   defined).
 *
 * Ghost cells: the kernel fills EVERY cell in
 * [ghost_begin_fine, ghost_end_fine) -- the caller passes exactly the
 * interface ghost extent in fine block-indexer coordinates. Ghost cells have
 * no valid map/classification of their own (fine cells are GEO_FLUID in v1),
 * so no map filtering is done beyond the single `dmacro` guard below.
 *
 * Precondition: the coarse cells of every fine ghost cell's interpolation
 * stencil must be valid storage indices in `coarse_SD` (including
 * overlaps). The kernel enforces this itself by shifting/shortening each
 * per-axis stencil window into the coarse storage extent (see the
 * storability guard above); no coarse storage index is ever accessed out
 * of bounds. For the full unshifted 3rd-order stencil the caller should
 * position interface ghosts at least two coarse cells inside the coarse
 * block's stored extent.
 *
 * Streaming-pattern handling (DF reads are the ONLY pattern-dependent part):
 * - AB pattern: `postCollisionStreaming` stores the post-collision DF of each
 *   direction at the same site in natural orientation into `df_out`, which is
 *   what the coarse side reads here. The caller passes `coarse_SD` in the
 *   rotation state of the kernel launch that just produced the coarse data.
 * - AA pattern: if `coarse_even_iter == true` the coarse state is twisted
 *   post-collision (`df_cur[opposite(q), site]` holds direction q), otherwise
 *   it is the post-stream natural state. Caveat: the post-stream state is NOT
 *   the post-collision value at the site -- it is the working state the next
 *   coarse substep will collide with, which is the physics-correct input for
 *   mirroring the coarse level onto the fine ghosts.
 *
 * Writes: fine ghost DFs go to `fine_SD.df(df_cur, ...)` for both patterns
 * (the caller sets the fine level's DF rotation for the upcoming substep
 * BEFORE the fill, so `df_cur` is exactly the array the next fine-level
 * streaming step pulls from; ghosts are re-filled every substep). The
 * direction slot is pattern-dependent: A-B pulls ghost DFs in natural
 * orientation, so direction q is stored in slot q; the A-A spatial
 * ("odd") substep pulls the DF streaming out of a ghost cell in direction q
 * from the opposite-direction slot (see D3Q27_STREAMING::streaming in
 * streaming_AA.h), so direction q is stored twisted in
 * `opposite_direction(q)`.
 *
 * Macroscopic output: for cells tagged `GEO_AMR_INTERFACE` the interpolated
 * macros are written to `dmacro` so visualization shows coupling-produced
 * values (the main kernel also recomputes macros for these
 * collision-active cells every step; this write would be the
 * output-relevant one for any fine cell carrying the tag). In v1 fine
 * ghosts are never GEO_AMR_INTERFACE, so the guard is a no-op kept for
 * safety.
 */

/**
 * \brief Host-side descriptor of one fine-level ghost region being filled.
 *
 * The Wave-4/5 caller builds one patch per refinement-interface face and
 * derives the kernel range as `ghost_begin_fine = fine_origin`,
 * `ghost_end_fine = fine_origin + fine_size`. The coarse rectangle (parent
 * coordinates) documents which coarse cells the patch couples to, and `face`
 * (pattern reused from `TNL::Containers::SyncDirection`, the same taxonomy
 * as the MPI synchronizer and `AMR_Region`) is the interface normal on the
 * coarse side.
 */
template <typename CONFIG>
struct AMR_InterfacePatch
{
	using TRAITS = typename CONFIG::TRAITS;
	using idx = typename TRAITS::idx;
	using idx3d = typename TRAITS::idx3d;

	idx3d coarse_origin;				  // origin of the parent coarse region (coarse coords)
	idx3d coarse_size;					  // extent in coarse cells
	idx3d fine_origin;					  // origin of the fine ghost extent (fine coords)
	idx3d fine_size;					  // extent in fine cells
	TNL::Containers::SyncDirection face;  // interface normal direction
};

/**
 * \brief Fill one fine-level ghost extent with DFs rescaled from the coarse
 * level (see the file docstring for the algorithm and its assumptions).
 */
template <typename CONFIG>
__global__ void cudaAMR_CoarseToFine(
	typename CONFIG::DATA fine_SD,
	typename CONFIG::DATA coarse_SD,
	typename CONFIG::TRAITS::idx3d ghost_begin_fine,
	typename CONFIG::TRAITS::idx3d ghost_end_fine,
	typename CONFIG::TRAITS::dreal tau_fine,
	typename CONFIG::TRAITS::dreal tau_coarse,
	bool coarse_even_iter,
	typename CONFIG::TRAITS::idx3d fine_off,
	typename CONFIG::TRAITS::idx3d coarse_off
)
{
	using TRAITS = typename CONFIG::TRAITS;
	using COLL = typename CONFIG::COLL;
	using BC = typename CONFIG::BC;
	using MACRO = typename CONFIG::MACRO;

	using idx = typename TRAITS::idx;
	using dreal = typename TRAITS::dreal;
	using LBM_KS = typename CONFIG::template KernelStruct<dreal>;

	const idx x = threadIdx.x + blockIdx.x * blockDim.x + ghost_begin_fine.x();
	const idx y = threadIdx.y + blockIdx.y * blockDim.y + ghost_begin_fine.y();
	const idx z = threadIdx.z + blockIdx.z * blockDim.z + ghost_begin_fine.z();

	if (x >= ghost_end_fine.x() || y >= ghost_end_fine.y() || z >= ghost_end_fine.z())
		return;

	// coarse-side DF read in the orientation produced by the last coarse
	// kernel launch -- the ONLY streaming-pattern-dependent code in the kernel
	const auto read_coarse_df = [&coarse_SD, coarse_even_iter](int q, idx cx, idx cy, idx cz) -> dreal
	{
#ifdef AB_PATTERN
		// AB: post-collision DF of direction q at the same site, natural
		// orientation, is stored in df_out (coarse_even_iter is AA-only state)
		static_cast<void>(coarse_even_iter);
		return coarse_SD.df(df_out, q, cx, cy, cz);
#elif defined(AA_PATTERN)
		if (coarse_even_iter)
			// AA post-collision state (twisted): the post-collision DF of
			// direction q at (cx,cy,cz) sits in the opposite-direction slot
			return coarse_SD.df(df_cur, opposite_direction(q), cx, cy, cz);
		// AA post-stream state (natural): the streamed-in DF of direction q --
		// the working state the next coarse substep will collide with
		return coarse_SD.df(df_cur, q, cx, cy, cz);
#endif
	};

	// true floor division by 2 (valid for negative fine global coordinates,
	// unlike the truncating integer division of C++ which truncates toward
	// zero and would map the x = -1 ghost to the wrong home cell)
	const auto fdiv2 = [](idx v) -> idx
	{
		return v >= 0 ? v / 2 : -((-v + 1) / 2);
	};

	// fine-DF write in the orientation the next fine substep pulls (see the
	// file docstring): A-B pulls ghost DFs in natural orientation, so
	// direction q goes to slot q; the A-A spatial ("odd") substep pulls the
	// DF streaming out of a ghost cell in direction q from the
	// opposite-direction slot, so direction q is stored twisted
	const auto store_fine_df = [&fine_SD, x, y, z](int q, dreal f) -> void
	{
#ifdef AB_PATTERN
		fine_SD.df(df_cur, q, x, y, z) = f;
#elif defined(AA_PATTERN)
		fine_SD.df(df_cur, opposite_direction(q), x, y, z) = f;
#endif
	};

	// macros supplied to the fine ghost cell (interpolated or the home
	// coarse cell's), also used for the dmacro write at the end
	dreal rho_f = 0, vx_f = 0, vy_f = 0, vz_f = 0;

#if defined(C2F_LINEAR_EXPLOSION) || defined(C2F_UNIFORM_EXPLOSION)
	// ---- Explosion strategies (Eitel-Amor et al. 2025): the fine ghost
	// cell reads ONLY its home coarse cell, which explodes into its 8 fine
	// subcells independently -- no stencil-neighbor coupling at the
	// interface (see the file docstring) ----
	static_cast<void>(tau_fine);
	static_cast<void>(tau_coarse);

	// home coarse cell in the coarse indexer frame, clamped per axis into
	// the stored extent (the explosion analog of the storability guard;
	// valid geometries never clamp because the fine ghost ring's home
	// cells lie inside the coarse block's stored footprint)
	const auto clamped_home = [&coarse_SD, &fdiv2](idx fg, idx coarse_off_a, idx size_a, idx ov_a) -> idx
	{
		const idx h = fdiv2(fg) - coarse_off_a;
		const idx lo = -ov_a;
		const idx hi = size_a - 1 + ov_a;
		return h < lo ? lo : (h > hi ? hi : h);
	};
	const idx cx = clamped_home(x + fine_off.x(), coarse_off.x(), coarse_SD.X(), coarse_SD.indexer.template getOverlap<0>());
	const idx cy = clamped_home(y + fine_off.y(), coarse_off.y(), coarse_SD.Y(), coarse_SD.indexer.template getOverlap<1>());
	const idx cz = clamped_home(z + fine_off.z(), coarse_off.z(), coarse_SD.Z(), coarse_SD.indexer.template getOverlap<2>());

	// the home coarse cell's DF state and its macros
	LBM_KS KS_H;
	for (int q = 0; q < CONFIG::Q; q++)
		KS_H.f[q] = read_coarse_df(q, cx, cy, cz);
	COLL::computeDensityAndVelocity(KS_H);
	rho_f = KS_H.rho;
	vx_f = KS_H.vx;
	vy_f = KS_H.vy;
	vz_f = KS_H.vz;

	#ifdef C2F_LINEAR_EXPLOSION
	// linear explosion: the home cell's macros (rho, u) are distributed to
	// the fine subcells -- the equilibrium is re-evaluated at those macros
	// (one `EQ::eq_*` call per direction, never shared) and the
	// non-equilibrium part is zeroed (pure equilibrium explosion)
	LBM_KS KS_EQ;
	KS_EQ.rho = rho_f;
	KS_EQ.vx = vx_f;
	KS_EQ.vy = vy_f;
	KS_EQ.vz = vz_f;
	COLL::setEquilibrium(KS_EQ);
	for (int q = 0; q < CONFIG::Q; q++)
		store_fine_df(q, KS_EQ.f[q]);
	#else
	// uniform explosion: the home cell's DFs are duplicated to every fine
	// subcell unchanged (zeroth order; no equilibrium re-evaluation, no
	// rescaling)
	for (int q = 0; q < CONFIG::Q; q++)
		store_fine_df(q, KS_H.f[q]);
	#endif

#else
	// ---- Interpolation strategies (default: 3rd-order Lagrange) ----
	// Per-axis interpolation stencils and weights in the GLOBAL frame (see
	// the file docstring): fine global coordinate fg = coord + fine_off; the
	// home coarse cell is floor(fg/2) and the fine cell center sits at
	// +-1/4 of the home cell's width from its center. The nominal per-axis
	// stencil covers the C2F_STENCIL coarse cell centers
	// `home - C2F_STENCIL/2 + (fg&1) + {0..C2F_STENCIL-1}` and is shifted
	// (and shortened if the per-axis storage extent is smaller) so that all
	// nodes are valid coarse storage indices -- the storability guard. The
	// Lagrange weights are evaluated at runtime in double precision and
	// normalized to sum to one; for the centered 4-node windows they round
	// to the exact dyadic rationals {-5,35,105,-7}/128 (even fg) and
	// {-7,105,35,-5}/128 (odd fg).
	#ifdef C2F_TRILINEAR
	// 2nd-order trilinear fallback (the original scheme): 2-point per-axis
	// stencil {home-1+(fg&1), home+(fg&1)} with 3/4:1/4 weights
	constexpr int C2F_STENCIL = 2;
	#else
	// 3rd-order interpolation: 4-point per-axis Lagrange stencil
	constexpr int C2F_STENCIL = 4;
	#endif
	const auto axis_stencil = [&fdiv2](idx fg, idx coarse_off_a, idx size_a, idx ov_a, idx* nodes, dreal* weights) -> int
	{
		const idx home = fdiv2(fg);
		const idx p = fg & 1;
		// evaluation point (fine cell center) in the coarse indexer frame
		const double t = static_cast<double>(home - coarse_off_a) + (p ? 0.25 : -0.25);
		// limit the stencil to the available storage extent and shift the
		// window so all nodes are valid storage indices (storability guard)
		const int extent = static_cast<int>(size_a + 2 * ov_a);
		const int n = C2F_STENCIL < extent ? C2F_STENCIL : extent;
		const idx lo = -ov_a;
		const idx hi = size_a - 1 + ov_a - (n - 1);
		idx start = home - coarse_off_a - C2F_STENCIL / 2 + p;
		start = start < lo ? lo : (start > hi ? hi : start);
		// Lagrange weights of the n consecutive nodes {start..start+n-1} at t
		// (nodes are consecutive integers, so the denominators are i - j)
		double w[C2F_STENCIL], wsum = 0;
		for (int i = 0; i < n; i++) {
			double wi = 1;
			for (int j = 0; j < n; j++)
				if (j != i)
					wi *= (t - static_cast<double>(start + j)) / static_cast<double>(i - j);
			w[i] = wi;
			wsum += wi;
		}
		for (int i = 0; i < n; i++) {
			nodes[i] = start + i;
			weights[i] = static_cast<dreal>(w[i] / wsum);
		}
		return n;
	};
	idx cnx[C2F_STENCIL], cny[C2F_STENCIL], cnz[C2F_STENCIL];
	dreal cwx[C2F_STENCIL], cwy[C2F_STENCIL], cwz[C2F_STENCIL];
	const int nnx = axis_stencil(x + fine_off.x(), coarse_off.x(), coarse_SD.X(), coarse_SD.indexer.template getOverlap<0>(), cnx, cwx);
	const int nny = axis_stencil(y + fine_off.y(), coarse_off.y(), coarse_SD.Y(), coarse_SD.indexer.template getOverlap<1>(), cny, cwy);
	const int nnz = axis_stencil(z + fine_off.z(), coarse_off.z(), coarse_SD.Z(), coarse_SD.indexer.template getOverlap<2>(), cnz, cwz);

	// per-direction non-equilibrium sums (the interpolated macros accumulate
	// into the rho_f/vx_f/vy_f/vz_f declared above)
	dreal f_neq[CONFIG::Q] = {};

	// visit the coarse cells of the interpolation stencil (up to 4x4x4 cells
	// whose centers surround the fine cell center)
	for (int bz = 0; bz < nnz; bz++) {
		for (int by = 0; by < nny; by++) {
			for (int bx = 0; bx < nnx; bx++) {
				const dreal w = cwx[bx] * cwy[by] * cwz[bz];

				// coarse-cell DF state and its macroscopic quantities
				LBM_KS KS;
				for (int q = 0; q < CONFIG::Q; q++)
					KS.f[q] = read_coarse_df(q, cnx[bx], cny[by], cnz[bz]);
				COLL::computeDensityAndVelocity(KS);
				rho_f += w * KS.rho;
				vx_f += w * KS.vx;
				vy_f += w * KS.vy;
				vz_f += w * KS.vz;

				// non-equilibrium at the coarse cell: f_neq[q] = f[q] - eq_q(rho_c, u_c)
				LBM_KS KS_EQ;
				KS_EQ.rho = KS.rho;
				KS_EQ.vx = KS.vx;
				KS_EQ.vy = KS.vy;
				KS_EQ.vz = KS.vz;
				COLL::setEquilibrium(KS_EQ);
				for (int q = 0; q < CONFIG::Q; q++)
					f_neq[q] += w * (KS.f[q] - KS_EQ.f[q]);
			}
		}
	}

	// equilibrium at the interpolated macros
	LBM_KS KS_F;
	KS_F.rho = rho_f;
	KS_F.vx = vx_f;
	KS_F.vy = vy_f;
	KS_F.vz = vz_f;
	COLL::setEquilibrium(KS_F);

	// volumetric rescaling, f_fine[q] = eq_q(rho_f,u_f) + (tau_f/tau_c)*f_neq[q]
	const dreal neq_scale = tau_fine / tau_coarse;
	for (int q = 0; q < CONFIG::Q; q++)
		store_fine_df(q, KS_F.f[q] + neq_scale * f_neq[q]);
#endif

	// macros for GEO_AMR_INTERFACE cells (no-op in v1, see the file docstring)
	if (fine_SD.map(x, y, z) == BC::GEO_AMR_INTERFACE) {
		fine_SD.macro(MACRO::e_rho, x, y, z) = rho_f;
		fine_SD.macro(MACRO::e_vx, x, y, z) = vx_f;
		fine_SD.macro(MACRO::e_vy, x, y, z) = vy_f;
		fine_SD.macro(MACRO::e_vz, x, y, z) = vz_f;
	}
}

/**
 * \brief Fill one coarse-level interface extent with DFs rescaled from the
 * fine level -- the reverse direction of `cudaAMR_CoarseToFine` (Lagrava et
 * al. 2012, JCP 231, merged with the volumetric formulation of Rohde et al.
 * 2006 / Chen et al. 1998).
 *
 * Cell-centered coordinate mapping (exact inverse of the coarse-to-fine
 * kernel, computed in the GLOBAL frame for the same reason): coarse cell
 * `(x,y,z)` (coarse indexer coordinates) covers the 8 fine subcells
 * `2*(x+coarse_off)-fine_off+{0,1}` per axis (fine indexer coordinates),
 * where `coarse_off`/`fine_off` are the blocks' indexer origins in the
 * global coordinates of their level (`LBM_BLOCK::offset`). One thread per
 * coarse interface cell in [coarse_begin, coarse_end); the caller derives
 * `coarse_end = coarse_begin + coarse_size` from the patch (same
 * begin/end range handling as `cudaAMR_CoarseToFine`).
 *
 * Storability guard (per cell): all 8 fine subcells of a processed coarse
 * cell must be valid fine STORAGE indices, i.e. within the per-axis
 * overlap-extended range `[-ov_i, fine_local_i + ov_i)`, where `ov` is the
 * overlap depth the caller allocated on the fine block's indexer (2 on
 * refinement-level blocks, see `LBM_BLOCK::storage_overlap`); cells failing
 * the test are skipped individually. This per-cell guard replaces the
 * former launch-extent clip, which evaluated the storability condition in
 * the wrong (origin-aligned) frame and silently dropped the max-side faces
 * and half of the patch extents of nested geometries. Note the fine overlap
 * storage must actually cover the 2-cell-deep ghost ring of the block's
 * footprint, otherwise every interface cell is skipped and keeps the
 * `preCollision` placeholder (`LBM_BLOCK` allocates it; the mock tests
 * allocate `ov` explicitly).
 *
 * Algorithm per coarse cell:
 * 1. Read the post-kernel fine DFs of all 8 subcells (orientation below).
 * 2. FILTER (MANDATORY): the per-direction arithmetic average
 *    `f_avg[q] = (1/8) * sum_k f_fine[q]_k` suppresses unresolved
 *    high-frequency fine-grid modes before projection onto the coarse
 *    grid -- without it the projection aliases and the coarse solution
 *    goes unstable (Lagrava et al. 2012). The 1/8 averaging IS the
 *    volumetric correction: DFs are point densities and each fine cell
 *    holds 1/8 of the coarse cell volume, so NO additional volume factor
 *    is applied anywhere.
 * 3. Compute the coarse macros (rho_c, u_c) as the moments of f_avg and
 *    the equilibrium f_eq[q] at those macros (EQ::eq_* per direction).
 * 4. Rescale the non-equilibrium part by tau_coarse / tau_fine -- the
 *    reciprocal of the coarse-to-fine factor, NOT a no-op (tau_fine !=
 *    tau_coarse with 2:1 refinement; both taus are computed by the caller
 *    from the per-level `lbmViscosity`). The coarse DF is
 *    `f_coarse[q] = f_eq[q] + (tau_c/tau_f) * (f_avg[q] - f_eq[q])`.
 *
 * Streaming-pattern handling (DF reads and writes are the ONLY
 * pattern-dependent parts):
 * - Reads from the fine level mirror `cudaAMR_CoarseToFine`: AB reads the
 *   post-collision DFs from `df_out` in natural orientation; AA reads
 *   `df_cur` from the opposite-direction slot when `fine_even_iter` is
 *   true (twisted post-collision state) or from the natural slot when
 *   false (post-stream state).
 * - Writes go to `coarse_SD.df(df_cur, ...)` for BOTH patterns (a single
 *   site), but the DIRECTION slot depends on which substep consumes the
 *   data next (convention in `streaming_AA.h` lines 31-90, against which
 *   this v1 decision was reviewed):
 *   - AB: the write goes to logical `df_out` in natural orientation. The
 *     caller passes `coarse_SD` in the rotation state of the LAST coarse
 *     kernel launch (which read df_cur and wrote df_out); the next global
 *     `updateKernelData()` rotates the coarse frames, so the physical
 *     array written here becomes exactly the `df_cur` the next coarse
 *     kernel launch pulls from. `coarse_even_iter` is AA-only state.
 *   - AA with `coarse_even_iter == true`: the NEXT coarse substep is the
 *     even ("reflect") substep, whose streaming reads the same site,
 *     same direction -- store natural (the post-stream state the even
 *     substep consumes).
 *   - AA with `coarse_even_iter == false`: the next coarse substep is the
 *     odd ("spatial") substep, whose pull reads
 *     `KS.f[q](P+vel(q)) = df_cur[opposite_direction(q), P]`, i.e. the DF
 *     streaming out of P in direction q sits in the opposite-direction
 *     slot at P -- store twisted. This reproduces exactly the state the
 *     previous coarse even substep's postCollisionStreaming would have
 *     left at the interface cells, so the odd substep consumes it with no
 *     additional handling.
 *   Note the parameter asymmetry: on the READ side `fine_even_iter`
 *   describes the parity of the stored fine data, on the WRITE side
 *   `coarse_even_iter` describes the parity of the NEXT consuming coarse
 *   substep.
 *
 * Macros: for cells tagged `GEO_AMR_INTERFACE` the filtered macros are
 * written to `dmacro` at the end of each coarse step; the main coarse
 * kernel also recomputes macros for these collision-active cells every
 * step, so both writers produce real values -- the transfer's write is the
 * output-relevant one until the next coarse step recomputes it.
 */
template <typename CONFIG>
__global__ void cudaAMR_FineToCoarse(
	typename CONFIG::DATA coarse_SD,
	typename CONFIG::DATA fine_SD,
	typename CONFIG::TRAITS::idx3d coarse_begin,
	typename CONFIG::TRAITS::idx3d coarse_end,
	typename CONFIG::TRAITS::dreal tau_coarse,
	typename CONFIG::TRAITS::dreal tau_fine,
	bool fine_even_iter,
	bool coarse_even_iter,
	typename CONFIG::TRAITS::idx3d fine_off,
	typename CONFIG::TRAITS::idx3d coarse_off,
	typename CONFIG::TRAITS::idx3d fine_local,
	typename CONFIG::TRAITS::idx3d ov
)
{
	using TRAITS = typename CONFIG::TRAITS;
	using COLL = typename CONFIG::COLL;
	using BC = typename CONFIG::BC;
	using MACRO = typename CONFIG::MACRO;

	using idx = typename TRAITS::idx;
	using dreal = typename TRAITS::dreal;
	using LBM_KS = typename CONFIG::template KernelStruct<dreal>;

	const idx x = threadIdx.x + blockIdx.x * blockDim.x + coarse_begin.x();
	const idx y = threadIdx.y + blockIdx.y * blockDim.y + coarse_begin.y();
	const idx z = threadIdx.z + blockIdx.z * blockDim.z + coarse_begin.z();

	if (x >= coarse_end.x() || y >= coarse_end.y() || z >= coarse_end.z())
		return;

	// fine subcells of this coarse cell in the fine block's indexer frame
	// (global-frame mapping, see the file docstring)
	const idx fx0 = 2 * (x + coarse_off.x()) - fine_off.x();
	const idx fy0 = 2 * (y + coarse_off.y()) - fine_off.y();
	const idx fz0 = 2 * (z + coarse_off.z()) - fine_off.z();

	// per-cell storability guard: all 8 subcells must be valid fine storage
	// indices (see the file docstring)
	for (int b = 0; b < 2; b++) {
		if (fx0 + b < -ov.x() || fx0 + b >= fine_local.x() + ov.x())
			return;
		if (fy0 + b < -ov.y() || fy0 + b >= fine_local.y() + ov.y())
			return;
		if (fz0 + b < -ov.z() || fz0 + b >= fine_local.z() + ov.z())
			return;
	}

	// fine-side DF read in the orientation produced by the last fine kernel
	// launch -- one of only TWO streaming-pattern-dependent sites
	const auto read_fine_df = [&fine_SD, fine_even_iter](int q, idx fx, idx fy, idx fz) -> dreal
	{
#ifdef AB_PATTERN
		// AB: post-collision DF of direction q at the same site, natural
		// orientation, is stored in df_out (fine_even_iter is AA-only state)
		static_cast<void>(fine_even_iter);
		return fine_SD.df(df_out, q, fx, fy, fz);
#elif defined(AA_PATTERN)
		if (fine_even_iter)
			// AA post-collision state (twisted): the post-collision DF of
			// direction q at (fx,fy,fz) sits in the opposite-direction slot
			return fine_SD.df(df_cur, opposite_direction(q), fx, fy, fz);
		// AA post-stream state (natural): the streamed-in DF of direction q
		return fine_SD.df(df_cur, q, fx, fy, fz);
#endif
	};

	// Lagrava spatial filter: per-direction arithmetic average of the 8 fine
	// subcells covered by this coarse cell. The (1/8) factor IS the
	// volumetric fine-to-coarse conversion -- no other volume factor.
	dreal f_avg[CONFIG::Q] = {};
	for (int bz = 0; bz < 2; bz++) {
		for (int by = 0; by < 2; by++) {
			for (int bx = 0; bx < 2; bx++) {
				for (int q = 0; q < CONFIG::Q; q++)
					f_avg[q] += read_fine_df(q, fx0 + bx, fy0 + by, fz0 + bz);
			}
		}
	}
	for (int q = 0; q < CONFIG::Q; q++)
		f_avg[q] *= dreal(0.125);

	// coarse macros from the filtered DFs
	LBM_KS KS;
	for (int q = 0; q < CONFIG::Q; q++)
		KS.f[q] = f_avg[q];
	COLL::computeDensityAndVelocity(KS);

	// equilibrium at the filtered macros
	LBM_KS KS_EQ;
	KS_EQ.rho = KS.rho;
	KS_EQ.vx = KS.vx;
	KS_EQ.vy = KS.vy;
	KS_EQ.vz = KS.vz;
	COLL::setEquilibrium(KS_EQ);

	// volumetric rescaling, f_coarse[q] = eq_q(rho_c,u_c) + (tau_c/tau_f)*f_neq[q]
	const dreal neq_scale = tau_coarse / tau_fine;
	for (int q = 0; q < CONFIG::Q; q++) {
		const dreal f_coarse = KS_EQ.f[q] + neq_scale * (f_avg[q] - KS_EQ.f[q]);

		// coarse DF write in the orientation the NEXT coarse substep will read
		// (see the kernel docstring) -- the other pattern-dependent site
#ifdef AB_PATTERN
		// AB: write to logical df_out, natural orientation -- the next global
		// updateKernelData() rotates the coarse frames, so this physical
		// array is the df_cur the next coarse kernel launch pulls from
		// (coarse_even_iter is AA-only state)
		static_cast<void>(coarse_even_iter);
		coarse_SD.df(df_out, q, x, y, z) = f_coarse;
#elif defined(AA_PATTERN)
		if (coarse_even_iter)
			// next substep is even ("reflect"): reads the same site, same
			// direction -- store natural
			coarse_SD.df(df_cur, q, x, y, z) = f_coarse;
		else
			// next substep is odd ("spatial"): the DF streaming out of this
			// cell in direction q is pulled from the opposite-direction slot
			// -- store twisted
			coarse_SD.df(df_cur, opposite_direction(q), x, y, z) = f_coarse;
#endif
	}

	// macros for coupling cells (GEO_AMR_INTERFACE ring or GEO_NOTHING
	// frozen hidden cells): authoritative coupling value for output
	const auto map_val = coarse_SD.map(x, y, z);
	if (map_val == BC::GEO_AMR_INTERFACE || map_val == BC::GEO_NOTHING) {
		coarse_SD.macro(MACRO::e_rho, x, y, z) = KS.rho;
		coarse_SD.macro(MACRO::e_vx, x, y, z) = KS.vx;
		coarse_SD.macro(MACRO::e_vy, x, y, z) = KS.vy;
		coarse_SD.macro(MACRO::e_vz, x, y, z) = KS.vz;
	}
}
