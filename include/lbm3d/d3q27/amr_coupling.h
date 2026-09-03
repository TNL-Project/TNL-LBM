#pragma once

#if ! defined(AA_PATTERN) && ! defined(AB_PATTERN)
	#error "amr_coupling.h requires either AA_PATTERN or AB_PATTERN to be defined before inclusion"
#endif

#include "lbm3d/defs.h"
#include "lbm_common/ciselnik.h"

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
 * Wall guard (Schönherr 2015 thesis, Sec. 7.3, "refinement at the wall"):
 * when a footprint face coincides with a physical boundary plane, the
 * nominal compact-moment source window covers a non-fluid cell (the
 * bounce-back wall row, or any other physical-BC-tagged row) -- an
 * inappropriate interpolation source in the thesis's words. Following the
 * thesis remedy (extrapolate from the nearest complete source cell), the
 * kernel's per-axis window guard detects a physically-tagged cell at
 * exactly one end of the destination's 2x2x2 nominal window tuple and
 * shifts the window one cell away from the tainted end; the evaluation
 * point follows the window center, which is the algebraic equivalent of
 * the thesis's offset + hat-coefficient transforms (Eqs. 7.49-7.57) for
 * |offset| <= 1 per axis (verified equivalent in the Sec. A.3.8 audit).
 * A source cell counts as live iff its map tag is GEO_FLUID,
 * GEO_AMR_INTERFACE or GEO_NOTHING: at a valid wall-attached registration
 * the shifted window reads the {c = 0 ring, c = 1 frozen skin} pair whose
 * content is current at the C2F fill point of the cycle (the skin row is
 * F2C-refilled one step earlier; cycle 0 reads the initial condition).
 * The guard never fires at an interior registration (bitwise identity),
 * and a residual shifted-window taint (a physical BC thicker than one
 * cell, or a mid-window straddle) collapses to the mirrored home cell --
 * checkCouplingMapPattern statically rejects those degenerate cases at
 * SimInit. The same end-tag rule guards the fine-to-coarse filter windows
 * (inert on the all-GEO_FLUID fine blocks of v1). Streaming-pattern
 * caveat: unlike an interior face, the shifted wall-window reads a frozen
 * row that is never rewritten by the coarse kernel, so the read-vs-write
 * orientation conventions can diverge there -- under AA_PATTERN this is
 * exactly the frozen-cell read mismatch catalogued as Defect-1 in
 * docs/AMR-for-LBM-implementation.md (the wall refinement lane, like all
 * frozen-cell coupling reads, is A-B-pattern-only until Defect-1 is
 * fixed).
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
 * Compile-time switches (C2F reconstruction strategy; DEFAULT since the
 * 2026-08-18 flip, user ruling, is the compact-moment scheme: the
 * interpolation reads NO covered (`GEO_NOTHING`) coarse cells at a valid
 * registration -- the ch7 band map statically rejects covered sources at
 * SimInit, see checkCouplingMapPattern):
 * - `C2F_COMPACT_MOMENT` [DEFAULT]: moment-based compact
 *   interpolation (Schönherr 2015 thesis, Sec. 7.2, Eqs. 7.10-7.48; the
 *   production scheme of the Musubi code) from the 2x2x2-cell home window
 *   (the C2F_TRILINEAR stencil): the five independent second-order
 *   non-equilibrium moments (strain rates) are computed per source cell
 *   from f_neq = f - f_eq, the 8-coefficient density and three
 *   11-coefficient velocity polynomials are fitted (exact for linear
 *   fields, and for pure quadratic VELOCITY fields; pure quadratic
 *   DENSITY is not -- the 8-coefficient density fit is trilinear (D.5,
 *   2026-08-16)), the averaged moments are corrected by the fitted
 *   gradients, and the fine DFs are reconstructed from the six
 *   second-order cumulants with the cumulant back-transformation of
 *   col_cum.h, with third-order and higher central moments zeroed
 *   (projects the non-hydrodynamic modes out of the interface instead of
 *   interpolating them per direction). Under USE_GEIER_CUM_2017 the mode
 *   filter is aligned to the collision's own persistent modes instead:
 *   the seven third-order cumulants the 2017 operator relaxes at its
 *   limiter-adapted finite rates (not to zero) are transferred with the
 *   trilinear nodal fit and reconstructed by AMR_CM_BACKTRANSFORM_GEIER
 *   (see the docstrings of AMR_CM_THIRD_MOMENTS and the back-transform);
 *   with the macro undefined the filter is unchanged. The former carve pre-pass (one-cell
 *   window shift away from covered sources, off-center evaluation up to
 *   |t_rel| = 0.75) was HARD-REMOVED on 2026-08-23: under the ch7 band
 *   registration every nominal window of a valid coupling straddles live
 *   GEO_AMR_INTERFACE cells only -- asserted statically by
 *   checkCouplingMapPattern at SimInit -- so a covered window is an
 *   invalid registration, not a runtime case, and the pre-pass could never
 *   fire; `C2F_CARVE`/`C2F_NO_CARVE` warn at configure and gate no code.
 *   The define `C2F_COMPACT_MOMENT` itself is a no-op kept as an explicit
 *   selector for readability of old configurations; it remains the default
 *   under USE_GEIER_CUM_2017 as well (user ruling 2026-09-03: the
 *   per-direction fills keep the weakly-damped residual modes alive and
 *   show a checkerboard-like vx artifact in the fine box), with the
 *   compact-moment branch mode-aligned to the collision via
 *   AMR_CM_THIRD_MOMENTS/AMR_CM_BACKTRANSFORM_GEIER (see above); defining
 *   `C2F_LAGRANGE`/`C2F_TRILINEAR` explicitly still reaches the
 *   per-direction interpolation branches.
 * - `C2F_LAGRANGE`: opts out to the 3rd-order tensor-product Lagrange
 *   scheme described above (the pre-flip default; its 4-node window can
 *   read covered coarse cells).
 * - `C2F_TRILINEAR`: reverts the interpolation to the original 2nd-order
 *   trilinear scheme (2-point per-axis stencil {home-1+(fg&1),
 *   home+(fg&1)} with 3/4:1/4 weights, reproduced by the same runtime
 *   Lagrange machinery with a 2-node window).
 * - `C2F_LINEAR_EXPLOSION` / `C2F_UNIFORM_EXPLOSION`: the explosion
 *   strategies described above (linear takes precedence if both are
 *   defined).
 *   Precedence if several defines are given: explosion > Lagrange-family
 *   opt-outs (`C2F_LAGRANGE`, `C2F_TRILINEAR`) > compact-moment default.
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

	// launch-configuration cache (2026-08-18): each transfer launcher
	// computes the block/grid sizes through the logging optimizer ONCE on
	// first use; the sentinel zero marks "not computed" (the optimizer
	// logs per call -- see LBM_BLOCK::getCudaBlockSize)
	mutable dim3 cached_block_size{0, 0, 0};
	mutable dim3 cached_grid_size{0, 0, 0};
};

/**
 * \brief Compact-moment (CM) reconstruction -- the shared machinery of
 * Schönherr 2015 thesis, Sec. 7.2, packaged as preprocessor blocks so that
 * every kernel performing the reconstruction expands the SAME statements
 * (the coarse-to-fine branch of `cudaAMR_CoarseToFine` below; T14's
 * `F2C_SCHONHERR` fine-to-coarse branch is the second intended consumer).
 *
 * Why macros and not `__device__` helper functions (2026-08-21, plan T13):
 * on nvcc 13.3 / sm_120 ANY helper call from the CM kernel body -- tested
 * forms: struct return, reference out-params, `static` linkage, function
 * definitions placed after both kernels, and a single once-called tail
 * helper -- makes NVVM re-schedule the caller kernel and perturbs the
 * generated SASS (75-126 instruction deltas around the per-source-cell
 * division cluster, seeding ulp differences amplified by the TGV wake;
 * forensics: git notes of this commit). The macro form expands to token
 * streams identical to the former inline code, preserving codegen and run
 * values bitwise by construction (SASS of `cudaAMR_CoarseToFine`
 * reproduced exactly, 10-iter TGV frame series value-exact vs the
 * stage-2-cycle capture).
 *
 * Expansion-site contract (both consumers): invoke each macro as a plain
 * STATEMENT inside a braced block where the aliases TRAITS, COLL, idx,
 * dreal, LBM_KS (= CONFIG::KernelStruct<dreal>) and CONFIG::Q are in
 * scope. Declarations are emitted into the invoking scope under the
 * canonical names of the former inline code (rho_n, u, v, w, the Pi_*
 * tensor, k_xy .. k_xx_zz, the 45 sd, sa, sb, sc and sk accumulators
 * expected in scope, the d_0 .. c_xyz coefficients, rho_f .. vz_f, the
 * E/F intermediates and C200 .. C110). Macros carry no internal comments:
 * `//` lines cannot live inside a backslash-continued #define body, so
 * all documentation sits immediately above each definition.
 */

/**
 * \brief CM per-source-cell state -- lead block (Eqs. 7.1-7.4 and the Pi
 * tensor setup): read the source cell's DFs through `read_df`, compute its
 * macros, set the equilibrium at those macros and declare the Pi tensor
 * accumulators (zeroed; filled by AMR_CM_PI_NEQ unless the EQ_ONLY debug
 * arm is active).
 *
 * (the coupling velocity uses the force-free first moment
 * -- KS_C carries zero force terms -- consistently with
 * thesis sec. 7.2, which defines no forcing; if volume
 * forcing ever reaches the band, revisit the Guo
 * half-offset here -- audit A.4-R3)
 *
 * non-equilibrium at the coarse cell and its pressure
 * tensor, Pi_ab = sum_q c_qa * c_qb * (f[q] - f_eq[q])
 */
#define AMR_CM_MACROS_AND_KMOMENTS(read_df, cx, cy, cz) \
	LBM_KS KS_C;                                        \
	for (int q = 0; q < CONFIG::Q; q++)                 \
		KS_C.f[q] = read_df(q, cx, cy, cz);             \
	COLL::computeDensityAndVelocity(KS_C);              \
	const dreal rho_n = KS_C.rho;                       \
	const dreal u = KS_C.vx;                            \
	const dreal v = KS_C.vy;                            \
	const dreal w = KS_C.vz;                            \
                                                        \
	LBM_KS KS_E;                                        \
	KS_E.rho = rho_n;                                   \
	KS_E.vx = u;                                        \
	KS_E.vy = v;                                        \
	KS_E.vz = w;                                        \
	COLL::setEquilibrium(KS_E);                         \
	dreal Pi_xx = 0, Pi_yy = 0, Pi_zz = 0, Pi_xy = 0, Pi_xz = 0, Pi_yz = 0

/**
 * \brief CM Pi tensor accumulation over the 27 directions (the
 * non-equilibrium part of Eqs. 7.5-7.9): invoke under the kernel's own
 * `#ifndef C2F_EQ_ONLY` bracket (the EQ_ONLY debug experiment zeroes the
 * strain-rate content so the reconstruction becomes pure equilibrium).
 * Declares the D3Q27 lattice-velocity component tables it reads
 * (enumeration from defs.h: the p/m/z letters map to +1/-1/0 in x/y/z
 * order).
 */
#define AMR_CM_PI_NEQ                                                                                                              \
	constexpr signed char vel_cx[27] = {0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1}; \
	constexpr signed char vel_cy[27] = {0, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, -1, 1, -1, 1}; \
	constexpr signed char vel_cz[27] = {0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1}; \
	for (int q = 0; q < CONFIG::Q; q++) {                                                                                          \
		const dreal f_neq = KS_C.f[q] - KS_E.f[q];                                                                                 \
		const dreal cqx = static_cast<dreal>(vel_cx[q]);                                                                           \
		const dreal cqy = static_cast<dreal>(vel_cy[q]);                                                                           \
		const dreal cqz = static_cast<dreal>(vel_cz[q]);                                                                           \
		Pi_xx += cqx * cqx * f_neq;                                                                                                \
		Pi_yy += cqy * cqy * f_neq;                                                                                                \
		Pi_zz += cqz * cqz * f_neq;                                                                                                \
		Pi_xy += cqx * cqy * f_neq;                                                                                                \
		Pi_xz += cqx * cqz * f_neq;                                                                                                \
		Pi_yz += cqy * cqz * f_neq;                                                                                                \
	}

/**
 * \brief Pi debug arms (seam investigation, 2026-08-19): invoke under the
 * kernel's own C2F_DEV_ONLY / C2F_NORM_ONLY / C2F_SHEAR_ONLY brackets.
 * DEV: subtract the trace of the non-equilibrium pressure tensor from the
 * diagonals before the omega scaling -- deviatoric only, since the
 * tau-rescaled compressional part is the suspected density-pump term.
 * NORM: keep only the diagonal deviatoric part and zero the off-diagonal
 * shear, so that FULL = NORM + SHEAR + trace partitions bit-exactly (the
 * trace itself is zero-tested by C2F_DEV_ONLY). SHEAR: keep only the
 * off-diagonal shear part, zero the diagonals; NORM + SHEAR partition the
 * full tensor's deviatoric content (trace is zero at the TGV fill sites,
 * see C2F_DEV_ONLY).
 */
#define AMR_CM_PI_DEV                                \
	const dreal Pi_tr = (Pi_xx + Pi_yy + Pi_zz) / 3; \
	Pi_xx -= Pi_tr;                                  \
	Pi_yy -= Pi_tr;                                  \
	Pi_zz -= Pi_tr

#define AMR_CM_PI_NORM                                \
	const dreal Pi_tr2 = (Pi_xx + Pi_yy + Pi_zz) / 3; \
	Pi_xx -= Pi_tr2;                                  \
	Pi_yy -= Pi_tr2;                                  \
	Pi_zz -= Pi_tr2;                                  \
	Pi_xy = 0;                                        \
	Pi_xz = 0;                                        \
	Pi_yz = 0

#define AMR_CM_PI_SHEAR \
	Pi_xx = 0;          \
	Pi_yy = 0;          \
	Pi_zz = 0

#ifdef USE_GEIER_CUM_2017
	/**
	 * \brief CM third-order cumulants of one source cell (Geier 2017 mode
	 * consistency): the seven persistent third-order cumulants of the
	 * USE_GEIER_CUM_2017 collision (k_120, k_210, k_201, k_102, k_012, k_021,
	 * k_111), evaluated as central moments of this cell's non-equilibrium part
	 * at its own macros (KS_C/KS_E/u/v/w in scope from
	 * AMR_CM_MACROS_AND_KMOMENTS).
	 *
	 * For this index family the cumulants equal the central moments exactly
	 * (every third-order cumulant subtraction carries a first central moment,
	 * which vanishes in the central frame; col_cum.h aliases C_abc = k_abc for
	 * all seven), and the factorized polynomial equilibrium contributes zero
	 * (the 3D product splits per axis and every member of the family holds a
	 * first central moment in at least one axis), so the third-order cumulants
	 * of the full state equal those of f_neq -- the same well-conditioned
	 * source the Pi tensor is read from. In convective 2:1 scaling both levels
	 * carry identical lattice density and velocity for the same physical
	 * state, so the third-order cumulants transfer IDENTITY-wise (no
	 * relaxation-rate rescaling: they are mode state, not strain encodings).
	 */
	#define AMR_CM_THIRD_MOMENTS                                                                                                    \
		constexpr signed char v3x[27] = {0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1}; \
		constexpr signed char v3y[27] = {0, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, -1, 1, -1, 1}; \
		constexpr signed char v3z[27] = {0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1, 1, -1, -1, 1}; \
		dreal k_120 = 0, k_210 = 0, k_201 = 0, k_102 = 0, k_012 = 0, k_021 = 0, k_111 = 0;                                          \
		for (int q = 0; q < CONFIG::Q; q++) {                                                                                       \
			const dreal f_neq = KS_C.f[q] - KS_E.f[q];                                                                              \
			const dreal gx = static_cast<dreal>(v3x[q]) - u;                                                                        \
			const dreal gy = static_cast<dreal>(v3y[q]) - v;                                                                        \
			const dreal gz = static_cast<dreal>(v3z[q]) - w;                                                                        \
			k_120 += gx * gy * gy * f_neq;                                                                                          \
			k_210 += gy * gx * gx * f_neq;                                                                                          \
			k_201 += gz * gx * gx * f_neq;                                                                                          \
			k_102 += gx * gz * gz * f_neq;                                                                                          \
			k_012 += gy * gz * gz * f_neq;                                                                                          \
			k_021 += gz * gy * gy * f_neq;                                                                                          \
			k_111 += gx * gy * gz * f_neq;                                                                                          \
		}
#endif

/**
 * \brief CM second-order k-moments from f_neq (Eqs. 7.5-7.9): the
 * off-diagonals carry -3*omega_s, the diagonal differences -(3/2)*omega_s,
 * where omega_s is the SOURCE (coarse) grid relaxation rate -- the
 * argument `oms`, named generically so the F2C direction can pass its own
 * source rate.
 */
#define AMR_CM_KMOMENTS(oms)                                \
	const dreal om_rho = oms / rho_n;                       \
	const dreal k_xy = -no3 * om_rho * Pi_xy;               \
	const dreal k_yz = -no3 * om_rho * Pi_yz;               \
	const dreal k_xz = -no3 * om_rho * Pi_xz;               \
	const dreal k_xx_yy = -n3o2 * om_rho * (Pi_xx - Pi_yy); \
	const dreal k_xx_zz = -n3o2 * om_rho * (Pi_xx - Pi_zz)

/**
 * \brief CM coefficient-sum accumulation (Eqs. 7.10-7.28, one source
 * cell): the diagonal-moment combinations of the velocity families
 * (K_b = k_yy_xx + k_yy_zz, K_c = k_zz_xx + k_zz_yy) and the donation of
 * this cell's macros and k-moments into the 45 canonical accumulators.
 * Invoke once per source cell with the loop-local coordinates xn/yn/zn
 * in {+-1/2}^3 in scope.
 *
 * Schönherr ch7 conversion, 2026-08-21 (T11 errata record):
 * the velocity-coefficient fits below carry the closed,
 * cyclically complete family of Eqs. 7.18/7.23/7.24 (and of
 * their unprinted cyclic b/c analogs): a_0 subtracts the full
 * k-sum -x*(k_xx-yy + k_xx-zz), and the a_yy/a_zz rows carry
 * doubled inner terms (1/8) sum [2y k_xy - 8xy v] etc. The
 * thesis print deviates from this family: print-7.18 carries
 * the singleton term -x*k_xx-yy in a_0 (while its a_xx,
 * print-7.22, correctly carries the full k-sum, making the
 * singleton internally asymmetric in the print), and
 * print-7.23/7.24 carry half of the inner prefactors shown
 * here. The printed singleton set is internally inconsistent
 * under the nodal-consistency identity
 *   a_0 = (1/8) sum(u) - (a_xx + a_yy + a_zz) / 4,
 * which it satisfies for no strain carrier, while the
 * implemented family satisfies it for any (quadratic exactness
 * follows). Suspected thesis-print errata -- the print carries
 * verified typos of the same class (7.54 missing the
 * y-subscript on d_y; 7.56/7.57 printing dhat_x on the LHS for
 * dhat_y, dhat_z; 7.49/7.50 dual tags on one formula) -- the
 * implemented family is the nodal-consistent one (derivation:
 * docs/AMR-schonherr-ch7-target-contract.md, appendix
 * "Sec. 7.2 equation audit", A.3.3-note; external check
 * record: appendix A.5-U1. Executable lock:
 * tests/unit/test_amr_schonherr_exactness.cu case T10c
 * discriminates the two families, code family green).
 */
#define AMR_CM_FIT_ACCUMULATE                                                                               \
	const dreal K_a = k_xx_yy + k_xx_zz;                                                                    \
	const dreal K_b = k_xx_zz - no2 * k_xx_yy;                                                              \
	const dreal K_c = k_xx_yy - no2 * k_xx_zz;                                                              \
                                                                                                            \
	sd0 += rho_n;                                                                                           \
	sdx += xn * rho_n;                                                                                      \
	sdy += yn * rho_n;                                                                                      \
	sdz += zn * rho_n;                                                                                      \
	sdxy += xn * yn * rho_n;                                                                                \
	sdyz += yn * zn * rho_n;                                                                                \
	sdxz += xn * zn * rho_n;                                                                                \
	sdxyz += xn * yn * zn * rho_n;                                                                          \
                                                                                                            \
	sa0 += -xn * K_a - no2 * yn * k_xy - no2 * zn * k_xz + no4 * u + no4 * xn * yn * v + no4 * xn * zn * w; \
	sax += xn * u;                                                                                          \
	say += yn * u;                                                                                          \
	saz += zn * u;                                                                                          \
	saxx += xn * K_a + no4 * xn * yn * v + no4 * xn * zn * w;                                               \
	sayy += no2 * yn * k_xy - no8 * xn * yn * v;                                                            \
	sazz += no2 * zn * k_xz - no8 * xn * zn * w;                                                            \
	saxy += xn * yn * u;                                                                                    \
	sayz += yn * zn * u;                                                                                    \
	saxz += xn * zn * u;                                                                                    \
	saxyz += xn * yn * zn * u;                                                                              \
                                                                                                            \
	sb0 += -yn * K_b - no2 * xn * k_xy - no2 * zn * k_yz + no4 * v + no4 * xn * yn * u + no4 * yn * zn * w; \
	sbx += xn * v;                                                                                          \
	sby += yn * v;                                                                                          \
	sbz += zn * v;                                                                                          \
	sbxx += no2 * xn * k_xy - no8 * xn * yn * u;                                                            \
	sbyy += yn * K_b + no4 * xn * yn * u + no4 * yn * zn * w;                                               \
	sbzz += no2 * zn * k_yz - no8 * yn * zn * w;                                                            \
	sbxy += xn * yn * v;                                                                                    \
	sbxz += xn * zn * v;                                                                                    \
	sbyz += yn * zn * v;                                                                                    \
	sbxyz += xn * yn * zn * v;                                                                              \
                                                                                                            \
	sc0 += -zn * K_c - no2 * xn * k_xz - no2 * yn * k_yz + no4 * w + no4 * xn * zn * u + no4 * yn * zn * v; \
	scx += xn * w;                                                                                          \
	scy += yn * w;                                                                                          \
	scz += zn * w;                                                                                          \
	scxx += no2 * xn * k_xz - no8 * xn * zn * u;                                                            \
	scyy += no2 * yn * k_yz - no8 * yn * zn * v;                                                            \
	sczz += zn * K_c + no4 * xn * zn * u + no4 * yn * zn * v;                                               \
	scxy += xn * yn * w;                                                                                    \
	scyz += yn * zn * w;                                                                                    \
	scxz += xn * zn * w;                                                                                    \
	scxyz += xn * yn * zn * w;                                                                              \
                                                                                                            \
	sk_xy += k_xy;                                                                                          \
	sk_yz += k_yz;                                                                                          \
	sk_xz += k_xz;                                                                                          \
	sk_xx_yy += k_xx_yy;                                                                                    \
	sk_xx_zz += k_xx_zz

/**
 * \brief CM polynomial coefficient fits (s*-sums -> d/a/b/c
 * coefficients) and the destination density at the Step C position: the
 * density family d_0 .. d_xyz (Eqs. 7.10-7.17) and the density polynomial
 * evaluated at the destination (Eq. 7.37, written into the canonical
 * rho_f lvalue), then the velocity families a_0 .. a_xyz, b_0 .. b_xyz,
 * c_0 .. c_xyz (Eqs. 7.18-7.28 and the cyclic permutations), folded from
 * the canonical post-loop accumulators.
 */
#define AMR_CM_FIT_COEFFICIENTS(tx, ty, tz)                                                                                 \
	const dreal d_0 = n1o8 * sd0;                                                                                           \
	const dreal d_x = n1o2 * sdx;                                                                                           \
	const dreal d_y = n1o2 * sdy;                                                                                           \
	const dreal d_z = n1o2 * sdz;                                                                                           \
	const dreal d_xy = no2 * sdxy;                                                                                          \
	const dreal d_yz = no2 * sdyz;                                                                                          \
	const dreal d_xz = no2 * sdxz;                                                                                          \
	const dreal d_xyz = no8 * sdxyz;                                                                                        \
	rho_f = d_0 + d_x * tx + d_y * ty + d_z * tz + d_xy * tx * ty + d_xz * tx * tz + d_yz * ty * tz + d_xyz * tx * ty * tz; \
                                                                                                                            \
	const dreal n1o32 = n1o8 * n1o4;                                                                                        \
	const dreal a_0 = n1o32 * sa0;                                                                                          \
	const dreal a_x = n1o2 * sax;                                                                                           \
	const dreal a_y = n1o2 * say;                                                                                           \
	const dreal a_z = n1o2 * saz;                                                                                           \
	const dreal a_xx = n1o8 * saxx;                                                                                         \
	const dreal a_yy = n1o8 * sayy;                                                                                         \
	const dreal a_zz = n1o8 * sazz;                                                                                         \
	const dreal a_xy = no2 * saxy;                                                                                          \
	const dreal a_yz = no2 * sayz;                                                                                          \
	const dreal a_xz = no2 * saxz;                                                                                          \
	const dreal a_xyz = no8 * saxyz;                                                                                        \
	const dreal b_0 = n1o32 * sb0;                                                                                          \
	const dreal b_x = n1o2 * sbx;                                                                                           \
	const dreal b_y = n1o2 * sby;                                                                                           \
	const dreal b_z = n1o2 * sbz;                                                                                           \
	const dreal b_xx = n1o8 * sbxx;                                                                                         \
	const dreal b_yy = n1o8 * sbyy;                                                                                         \
	const dreal b_zz = n1o8 * sbzz;                                                                                         \
	const dreal b_xy = no2 * sbxy;                                                                                          \
	const dreal b_yz = no2 * sbyz;                                                                                          \
	const dreal b_xz = no2 * sbxz;                                                                                          \
	const dreal b_xyz = no8 * sbxyz;                                                                                        \
	const dreal c_0 = n1o32 * sc0;                                                                                          \
	const dreal c_x = n1o2 * scx;                                                                                           \
	const dreal c_y = n1o2 * scy;                                                                                           \
	const dreal c_z = n1o2 * scz;                                                                                           \
	const dreal c_xx = n1o8 * scxx;                                                                                         \
	const dreal c_yy = n1o8 * scyy;                                                                                         \
	const dreal c_zz = n1o8 * sczz;                                                                                         \
	const dreal c_xy = no2 * scxy;                                                                                          \
	const dreal c_yz = no2 * scyz;                                                                                          \
	const dreal c_xz = no2 * scxz;                                                                                          \
	const dreal c_xyz = no8 * scxyz

/**
 * \brief CM destination velocities at (tx,ty,tz): the velocity
 * polynomials (Eqs. 7.34-7.36) evaluated at the destination cell's
 * window-local coordinates, written into the canonical vx_f .. vz_f
 * lvalues of the invoking scope (the destination density is evaluated at
 * its Step C position by AMR_CM_FIT_COEFFICIENTS).
 */
#define AMR_CM_EVALUATE(tx, ty, tz)                                                                                                       \
	vx_f = a_0 + tx * (a_x + tx * a_xx + ty * a_xy + tz * a_xz + ty * tz * a_xyz) + ty * a_y + tz * a_z + ty * ty * a_yy + tz * tz * a_zz \
		 + ty * tz * a_yz;                                                                                                                \
	vy_f = b_0 + tx * (b_x + tx * b_xx + ty * b_xy + tz * b_xz + ty * tz * b_xyz) + ty * b_y + tz * b_z + ty * ty * b_yy + tz * tz * b_zz \
		 + ty * tz * b_yz;                                                                                                                \
	vz_f = c_0 + tx * (c_x + tx * c_xx + ty * c_xy + tz * c_xz + ty * tz * c_xyz) + ty * c_y + tz * c_z + ty * ty * c_yy + tz * tz * c_zz \
		 + ty * tz * c_yz

/**
 * \brief CM averaged k-moments with the velocity-gradient corrections
 * (Eqs. 7.29-7.33) and the six corrected second-order cumulants (Eqs.
 * 7.38-7.48). Reads the canonical coefficients, the sk_* sums, rho_f and
 * the canonical omega_d lvalue of the invoking scope; the argument `sig`
 * is the source->destination grid ratio, substituted verbatim into the
 * sigma declaration (C2F: sigma_{c->f} = 1/2, invoked as
 * AMR_CM_CORRECTED_CUMULANTS(n1o2); the F2C direction passes sig = 2 with
 * its own destination rate). The diagonal cumulants carry the rho/3
 * equilibrium term; the non-equilibrium corrections are trace-free.
 */
#define AMR_CM_CORRECTED_CUMULANTS(sig)                                                                                                         \
	const dreal avg_k_xy = n1o8 * sk_xy - (a_y + b_x);                                                                                          \
	const dreal avg_k_yz = n1o8 * sk_yz - (b_z + c_y);                                                                                          \
	const dreal avg_k_xz = n1o8 * sk_xz - (a_z + c_x);                                                                                          \
	const dreal avg_k_xx_yy = n1o8 * sk_xx_yy - (a_x - b_y);                                                                                    \
	const dreal avg_k_xx_zz = n1o8 * sk_xx_zz - (a_x - c_z);                                                                                    \
                                                                                                                                                \
	const dreal sigma = sig;                                                                                                                    \
	const dreal corr_B = no2 * a_xx * tx - b_xy * tx + a_xy * ty - no2 * b_yy * ty + a_xz * tz - b_yz * tz - b_xyz * tx * tz + a_xyz * ty * tz; \
	const dreal corr_C = no2 * a_xx * tx - c_xz * tx + a_xy * ty - c_yz * ty - c_xyz * tx * ty + a_xz * tz - no2 * c_zz * tz + a_xyz * ty * tz; \
	const dreal A011 = b_xz * tx + c_xy * tx + b_yz * ty + no2 * c_yy * ty + b_xyz * tx * ty + no2 * b_zz * tz + c_yz * tz + c_xyz * tx * tz;   \
	const dreal A101 = a_xz * tx + no2 * c_xx * tx + a_yz * ty + c_xy * ty + a_xyz * tx * ty + no2 * a_zz * tz + c_xz * tz + c_xyz * ty * tz;   \
	const dreal A110 = a_xy * tx + no2 * b_xx * tx + no2 * a_yy * ty + b_xy * ty + a_yz * tz + b_xz * tz + a_xyz * tx * tz + b_xyz * ty * tz;   \
	const dreal off_factor = sigma * rho_f / (no3 * omega_d);                                                                                   \
	const dreal C011 = -off_factor * (b_z + c_y + avg_k_yz + A011);                                                                             \
	const dreal C101 = -off_factor * (a_z + c_x + avg_k_xz + A101);                                                                             \
	const dreal C110 = -off_factor * (a_y + b_x + avg_k_xy + A110);                                                                             \
	const dreal X = a_x - b_y + avg_k_xx_yy + corr_B;                                                                                           \
	const dreal Y = a_x - c_z + avg_k_xx_zz + corr_C;                                                                                           \
	const dreal diag_factor = no2 * sigma * rho_f / (no9 * omega_d);                                                                            \
	const dreal diag_eq = rho_f * n1o3;                                                                                                         \
	const dreal C200 = diag_eq - diag_factor * (X + Y);                                                                                         \
	const dreal C020 = diag_eq - diag_factor * (-no2 * X + Y);                                                                                  \
	const dreal C002 = diag_eq - diag_factor * (X - no2 * Y)

/**
 * \brief CM back-transformation (central moments -> DFs of the
 * destination), copied from col_cum.h (the non-USE_GEIER_CUM_2017 path,
 * Geier 2015 Eqs. 81-96) with the #define aliases replaced by explicit
 * variables and KS.f[...] replaced by `store_df(...)`. There is no
 * collision: the post-collision cumulants equal the pre-collision ones.
 * The cumulant/central-moment state: the zeroth cumulant is the
 * interpolated density, the first-order central moments vanish by
 * construction (central frame), the second-order central moments equal
 * the cumulants of AMR_CM_CORRECTED_CUMULANTS (the diagonals carry the
 * rho/3 equilibrium term, i.e. they are TOTAL second-order central
 * moments of the reconstructed distribution), and ALL third-order and
 * higher central moments are zero (the mode filter, cf. col_cum.h's
 * simplified non-USE_GEIER_CUM_2017 path).
 *
 * Well-conditioned storage (COLL::is_well_conditioned): the emitted DFs
 * are fhat = f - w_q directly, no write-time COLL::toStorage conversion.
 * The dispatcher below selects between two TEXTUALLY SEPARATE expansions
 * (AMR_CM_BACKTRANSFORM_PLAIN / _WELL) rather than folding the K
 * corrections away in one macro: a folded zero term (e.g. `+ 0 * x`)
 * still changes NVVM's FMA-contraction candidates in the coupling
 * kernels and re-seeds ulp-level divergence against the reference, so
 * the production path must see the pre-gating statements verbatim.
 *
 * The _WELL chain is the K-corrected backward transform of
 * col_cum_well.h (Geier 2017 Eqs. 53-65) under the same mode filter. The
 * derivation: the weights w_q factorize per axis (1/6, 2/3, 1/6), so
 * the forward transform of fhat = f - w reports the cumulants of f from
 * the raw moments of w, i.e. C_000(fhat) = rho - 1,
 * C_200(fhat) = k_200(f) - 1/3 (resp. 020/002), C_110/101/011 unchanged
 * (raw off-diagonals of w vanish), third+ cumulants unchanged (the +K
 * terms in the forward cumulant definitions recover the physical
 * cumulants of f exactly), and the density staying PHYSICAL
 * (COMMON_WELL's +1 lives in the ks_000 shift, so rho_inv = 1/rho_f in
 * both conventions -- never 1/(rho-1)). The AMR input state therefore
 * maps as ks_000 = rho_f - 1, ks_200 = C200 - 1/3 (resp. 020/002),
 * third+ cumulants zero as before, and the K constants of col_cum_well.h
 * (directional weight-column sums) shift the sweep/store polynomials to
 * emit fhat. Verified at equilibrium (v=0, C-diag=rho/3): _PLAIN gives
 * w_q*rho, _WELL gives (rho-1)*w_q (zero at rho=1).
 */
#define AMR_CM_BACKTRANSFORM(store_df)         \
	if constexpr (COLL::is_well_conditioned) { \
		AMR_CM_BACKTRANSFORM_WELL(store_df);   \
	}                                          \
	else {                                     \
		AMR_CM_BACKTRANSFORM_PLAIN(store_df);  \
	}

// plain storage convention: the verbatim Geier 2015 Eqs. 81-96 chain,
// kept textually identical to the pre-gating macro -- an inline zero
// term (e.g. `+ 0 * x` from a folded K) still changes NVVM's FMA
// contraction candidates and re-seeds ulp-level divergence in the
// coupling kernels, so the production path must see the same statements
#define AMR_CM_BACKTRANSFORM_PLAIN(store_df)                                                                                          \
	const dreal ks_000 = rho_f;                                                                                                       \
	const dreal ks_100 = 0;                                                                                                           \
	const dreal ks_010 = 0;                                                                                                           \
	const dreal ks_001 = 0;                                                                                                           \
	const dreal ks_200 = C200;                                                                                                        \
	const dreal ks_020 = C020;                                                                                                        \
	const dreal ks_002 = C002;                                                                                                        \
	const dreal ks_110 = C110;                                                                                                        \
	const dreal ks_101 = C101;                                                                                                        \
	const dreal ks_011 = C011;                                                                                                        \
	const dreal ks_210 = 0;                                                                                                           \
	const dreal ks_120 = 0;                                                                                                           \
	const dreal ks_201 = 0;                                                                                                           \
	const dreal ks_102 = 0;                                                                                                           \
	const dreal ks_021 = 0;                                                                                                           \
	const dreal ks_012 = 0;                                                                                                           \
	const dreal ks_111 = 0;                                                                                                           \
                                                                                                                                      \
	const dreal rho_inv = no1 / rho_f;                                                                                                \
	const dreal vx_sqr = vx_f * vx_f;                                                                                                 \
	const dreal vy_sqr = vy_f * vy_f;                                                                                                 \
	const dreal vz_sqr = vz_f * vz_f;                                                                                                 \
                                                                                                                                      \
	const dreal ks_211 = (ks_200 * ks_011 + no2 * ks_101 * ks_110) * rho_inv;                                                         \
	const dreal ks_121 = (ks_020 * ks_101 + no2 * ks_110 * ks_011) * rho_inv;                                                         \
	const dreal ks_112 = (ks_002 * ks_110 + no2 * ks_011 * ks_101) * rho_inv;                                                         \
	const dreal ks_220 = (ks_020 * ks_200 + no2 * ks_110 * ks_110) * rho_inv;                                                         \
	const dreal ks_022 = (ks_002 * ks_020 + no2 * ks_011 * ks_011) * rho_inv;                                                         \
	const dreal ks_202 = (ks_200 * ks_002 + no2 * ks_101 * ks_101) * rho_inv;                                                         \
                                                                                                                                      \
	const dreal ks_122 = 0;                                                                                                           \
	const dreal ks_212 = 0;                                                                                                           \
	const dreal ks_221 = 0;                                                                                                           \
                                                                                                                                      \
	const dreal ks_222 =                                                                                                              \
		(ks_200 * ks_022 + ks_020 * ks_202 + ks_002 * ks_220 + no4 * (ks_011 * ks_211 + ks_101 * ks_121 + ks_110 * ks_112)) * rho_inv \
		- (no16 * ks_110 * ks_101 * ks_011 + no4 * (ks_101 * ks_101 * ks_020 + ks_011 * ks_011 * ks_200 + ks_110 * ks_110 * ks_002)   \
		   + no2 * ks_200 * ks_020 * ks_002)                                                                                          \
			  * rho_inv * rho_inv;                                                                                                    \
                                                                                                                                      \
	const dreal ks_z00 = ks_000 * (no1 - vx_sqr) - no2 * vx_f * ks_100 - ks_200;                                                      \
	const dreal ks_z01 = ks_001 * (no1 - vx_sqr) - no2 * vx_f * ks_101 - ks_201;                                                      \
	const dreal ks_z02 = ks_002 * (no1 - vx_sqr) - no2 * vx_f * ks_102 - ks_202;                                                      \
	const dreal ks_z10 = ks_010 * (no1 - vx_sqr) - no2 * vx_f * ks_110 - ks_210;                                                      \
	const dreal ks_z11 = ks_011 * (no1 - vx_sqr) - no2 * vx_f * ks_111 - ks_211;                                                      \
	const dreal ks_z12 = ks_012 * (no1 - vx_sqr) - no2 * vx_f * ks_112 - ks_212;                                                      \
	const dreal ks_z20 = ks_020 * (no1 - vx_sqr) - no2 * vx_f * ks_120 - ks_220;                                                      \
	const dreal ks_z21 = ks_021 * (no1 - vx_sqr) - no2 * vx_f * ks_121 - ks_221;                                                      \
	const dreal ks_z22 = ks_022 * (no1 - vx_sqr) - no2 * vx_f * ks_122 - ks_222;                                                      \
                                                                                                                                      \
	const dreal ks_m00 = (ks_000 * (vx_sqr - vx_f) + ks_100 * (no2 * vx_f - no1) + ks_200) * n1o2;                                    \
	const dreal ks_m01 = (ks_001 * (vx_sqr - vx_f) + ks_101 * (no2 * vx_f - no1) + ks_201) * n1o2;                                    \
	const dreal ks_m02 = (ks_002 * (vx_sqr - vx_f) + ks_102 * (no2 * vx_f - no1) + ks_202) * n1o2;                                    \
	const dreal ks_m10 = (ks_010 * (vx_sqr - vx_f) + ks_110 * (no2 * vx_f - no1) + ks_210) * n1o2;                                    \
	const dreal ks_m11 = (ks_011 * (vx_sqr - vx_f) + ks_111 * (no2 * vx_f - no1) + ks_211) * n1o2;                                    \
	const dreal ks_m12 = (ks_012 * (vx_sqr - vx_f) + ks_112 * (no2 * vx_f - no1) + ks_212) * n1o2;                                    \
	const dreal ks_m20 = (ks_020 * (vx_sqr - vx_f) + ks_120 * (no2 * vx_f - no1) + ks_220) * n1o2;                                    \
	const dreal ks_m21 = (ks_021 * (vx_sqr - vx_f) + ks_121 * (no2 * vx_f - no1) + ks_221) * n1o2;                                    \
	const dreal ks_m22 = (ks_022 * (vx_sqr - vx_f) + ks_122 * (no2 * vx_f - no1) + ks_222) * n1o2;                                    \
                                                                                                                                      \
	const dreal ks_p00 = (ks_000 * (vx_sqr + vx_f) + ks_100 * (no2 * vx_f + no1) + ks_200) * n1o2;                                    \
	const dreal ks_p01 = (ks_001 * (vx_sqr + vx_f) + ks_101 * (no2 * vx_f + no1) + ks_201) * n1o2;                                    \
	const dreal ks_p02 = (ks_002 * (vx_sqr + vx_f) + ks_102 * (no2 * vx_f + no1) + ks_202) * n1o2;                                    \
	const dreal ks_p10 = (ks_010 * (vx_sqr + vx_f) + ks_110 * (no2 * vx_f + no1) + ks_210) * n1o2;                                    \
	const dreal ks_p11 = (ks_011 * (vx_sqr + vx_f) + ks_111 * (no2 * vx_f + no1) + ks_211) * n1o2;                                    \
	const dreal ks_p12 = (ks_012 * (vx_sqr + vx_f) + ks_112 * (no2 * vx_f + no1) + ks_212) * n1o2;                                    \
	const dreal ks_p20 = (ks_020 * (vx_sqr + vx_f) + ks_120 * (no2 * vx_f + no1) + ks_220) * n1o2;                                    \
	const dreal ks_p21 = (ks_021 * (vx_sqr + vx_f) + ks_121 * (no2 * vx_f + no1) + ks_221) * n1o2;                                    \
	const dreal ks_p22 = (ks_022 * (vx_sqr + vx_f) + ks_122 * (no2 * vx_f + no1) + ks_222) * n1o2;                                    \
                                                                                                                                      \
	const dreal ks_mz0 = ks_m00 * (no1 - vy_sqr) - no2 * vy_f * ks_m10 - ks_m20;                                                      \
	const dreal ks_mz1 = ks_m01 * (no1 - vy_sqr) - no2 * vy_f * ks_m11 - ks_m21;                                                      \
	const dreal ks_mz2 = ks_m02 * (no1 - vy_sqr) - no2 * vy_f * ks_m12 - ks_m22;                                                      \
	const dreal ks_zz0 = ks_z00 * (no1 - vy_sqr) - no2 * vy_f * ks_z10 - ks_z20;                                                      \
	const dreal ks_zz1 = ks_z01 * (no1 - vy_sqr) - no2 * vy_f * ks_z11 - ks_z21;                                                      \
	const dreal ks_zz2 = ks_z02 * (no1 - vy_sqr) - no2 * vy_f * ks_z12 - ks_z22;                                                      \
	const dreal ks_pz0 = ks_p00 * (no1 - vy_sqr) - no2 * vy_f * ks_p10 - ks_p20;                                                      \
	const dreal ks_pz1 = ks_p01 * (no1 - vy_sqr) - no2 * vy_f * ks_p11 - ks_p21;                                                      \
	const dreal ks_pz2 = ks_p02 * (no1 - vy_sqr) - no2 * vy_f * ks_p12 - ks_p22;                                                      \
                                                                                                                                      \
	const dreal ks_mm0 = (ks_m00 * (vy_sqr - vy_f) + ks_m10 * (no2 * vy_f - no1) + ks_m20) * n1o2;                                    \
	const dreal ks_mm1 = (ks_m01 * (vy_sqr - vy_f) + ks_m11 * (no2 * vy_f - no1) + ks_m21) * n1o2;                                    \
	const dreal ks_mm2 = (ks_m02 * (vy_sqr - vy_f) + ks_m12 * (no2 * vy_f - no1) + ks_m22) * n1o2;                                    \
	const dreal ks_zm0 = (ks_z00 * (vy_sqr - vy_f) + ks_z10 * (no2 * vy_f - no1) + ks_z20) * n1o2;                                    \
	const dreal ks_zm1 = (ks_z01 * (vy_sqr - vy_f) + ks_z11 * (no2 * vy_f - no1) + ks_z21) * n1o2;                                    \
	const dreal ks_zm2 = (ks_z02 * (vy_sqr - vy_f) + ks_z12 * (no2 * vy_f - no1) + ks_z22) * n1o2;                                    \
	const dreal ks_pm0 = (ks_p00 * (vy_sqr - vy_f) + ks_p10 * (no2 * vy_f - no1) + ks_p20) * n1o2;                                    \
	const dreal ks_pm1 = (ks_p01 * (vy_sqr - vy_f) + ks_p11 * (no2 * vy_f - no1) + ks_p21) * n1o2;                                    \
	const dreal ks_pm2 = (ks_p02 * (vy_sqr - vy_f) + ks_p12 * (no2 * vy_f - no1) + ks_p22) * n1o2;                                    \
                                                                                                                                      \
	const dreal ks_mp0 = (ks_m00 * (vy_sqr + vy_f) + ks_m10 * (no2 * vy_f + no1) + ks_m20) * n1o2;                                    \
	const dreal ks_mp1 = (ks_m01 * (vy_sqr + vy_f) + ks_m11 * (no2 * vy_f + no1) + ks_m21) * n1o2;                                    \
	const dreal ks_mp2 = (ks_m02 * (vy_sqr + vy_f) + ks_m12 * (no2 * vy_f + no1) + ks_m22) * n1o2;                                    \
	const dreal ks_zp0 = (ks_z00 * (vy_sqr + vy_f) + ks_z10 * (no2 * vy_f + no1) + ks_z20) * n1o2;                                    \
	const dreal ks_zp1 = (ks_z01 * (vy_sqr + vy_f) + ks_z11 * (no2 * vy_f + no1) + ks_z21) * n1o2;                                    \
	const dreal ks_zp2 = (ks_z02 * (vy_sqr + vy_f) + ks_z12 * (no2 * vy_f + no1) + ks_z22) * n1o2;                                    \
	const dreal ks_pp0 = (ks_p00 * (vy_sqr + vy_f) + ks_p10 * (no2 * vy_f + no1) + ks_p20) * n1o2;                                    \
	const dreal ks_pp1 = (ks_p01 * (vy_sqr + vy_f) + ks_p11 * (no2 * vy_f + no1) + ks_p21) * n1o2;                                    \
	const dreal ks_pp2 = (ks_p02 * (vy_sqr + vy_f) + ks_p12 * (no2 * vy_f + no1) + ks_p22) * n1o2;                                    \
                                                                                                                                      \
	store_df(mmz, ks_mm0*(no1 - vz_sqr) - no2 * vz_f * ks_mm1 - ks_mm2);                                                              \
	store_df(mzz, ks_mz0*(no1 - vz_sqr) - no2 * vz_f * ks_mz1 - ks_mz2);                                                              \
	store_df(mpz, ks_mp0*(no1 - vz_sqr) - no2 * vz_f * ks_mp1 - ks_mp2);                                                              \
	store_df(zmz, ks_zm0*(no1 - vz_sqr) - no2 * vz_f * ks_zm1 - ks_zm2);                                                              \
	store_df(zzz, ks_zz0*(no1 - vz_sqr) - no2 * vz_f * ks_zz1 - ks_zz2);                                                              \
	store_df(zpz, ks_zp0*(no1 - vz_sqr) - no2 * vz_f * ks_zp1 - ks_zp2);                                                              \
	store_df(pmz, ks_pm0*(no1 - vz_sqr) - no2 * vz_f * ks_pm1 - ks_pm2);                                                              \
	store_df(pzz, ks_pz0*(no1 - vz_sqr) - no2 * vz_f * ks_pz1 - ks_pz2);                                                              \
	store_df(ppz, ks_pp0*(no1 - vz_sqr) - no2 * vz_f * ks_pp1 - ks_pp2);                                                              \
                                                                                                                                      \
	store_df(mmm, (ks_mm0 * (vz_sqr - vz_f) + ks_mm1 * (no2 * vz_f - no1) + ks_mm2) * n1o2);                                          \
	store_df(mzm, (ks_mz0 * (vz_sqr - vz_f) + ks_mz1 * (no2 * vz_f - no1) + ks_mz2) * n1o2);                                          \
	store_df(mpm, (ks_mp0 * (vz_sqr - vz_f) + ks_mp1 * (no2 * vz_f - no1) + ks_mp2) * n1o2);                                          \
	store_df(zmm, (ks_zm0 * (vz_sqr - vz_f) + ks_zm1 * (no2 * vz_f - no1) + ks_zm2) * n1o2);                                          \
	store_df(zzm, (ks_zz0 * (vz_sqr - vz_f) + ks_zz1 * (no2 * vz_f - no1) + ks_zz2) * n1o2);                                          \
	store_df(zpm, (ks_zp0 * (vz_sqr - vz_f) + ks_zp1 * (no2 * vz_f - no1) + ks_zp2) * n1o2);                                          \
	store_df(pmm, (ks_pm0 * (vz_sqr - vz_f) + ks_pm1 * (no2 * vz_f - no1) + ks_pm2) * n1o2);                                          \
	store_df(pzm, (ks_pz0 * (vz_sqr - vz_f) + ks_pz1 * (no2 * vz_f - no1) + ks_pz2) * n1o2);                                          \
	store_df(ppm, (ks_pp0 * (vz_sqr - vz_f) + ks_pp1 * (no2 * vz_f - no1) + ks_pp2) * n1o2);                                          \
                                                                                                                                      \
	store_df(mmp, (ks_mm0 * (vz_sqr + vz_f) + ks_mm1 * (no2 * vz_f + no1) + ks_mm2) * n1o2);                                          \
	store_df(mzp, (ks_mz0 * (vz_sqr + vz_f) + ks_mz1 * (no2 * vz_f + no1) + ks_mz2) * n1o2);                                          \
	store_df(mpp, (ks_mp0 * (vz_sqr + vz_f) + ks_mp1 * (no2 * vz_f + no1) + ks_mp2) * n1o2);                                          \
	store_df(zmp, (ks_zm0 * (vz_sqr + vz_f) + ks_zm1 * (no2 * vz_f + no1) + ks_zm2) * n1o2);                                          \
	store_df(zzp, (ks_zz0 * (vz_sqr + vz_f) + ks_zz1 * (no2 * vz_f + no1) + ks_zz2) * n1o2);                                          \
	store_df(zpp, (ks_zp0 * (vz_sqr + vz_f) + ks_zp1 * (no2 * vz_f + no1) + ks_zp2) * n1o2);                                          \
	store_df(pmp, (ks_pm0 * (vz_sqr + vz_f) + ks_pm1 * (no2 * vz_f + no1) + ks_pm2) * n1o2);                                          \
	store_df(pzp, (ks_pz0 * (vz_sqr + vz_f) + ks_pz1 * (no2 * vz_f + no1) + ks_pz2) * n1o2);                                          \
	store_df(ppp, (ks_pp0 * (vz_sqr + vz_f) + ks_pp1 * (no2 * vz_f + no1) + ks_pp2) * n1o2)

// well-conditioned storage (invoked only from AMR_CM_BACKTRANSFORM when
// COLL::is_well_conditioned holds): produces fhat = f - w_q with the
// K-corrected backward chain of col_cum_well.h under the mode filter --
// KWC_3/KWC_9 are the second/fourth-order weight moments, KC_* the
// directional column sums of the D3Q27 weights (K constants of
// col_cum_well.h, K_... = 0 coefficients omitted as in Eqs. 57-65)
#define AMR_CM_BACKTRANSFORM_WELL(store_df)                                                                                                      \
	constexpr dreal KWC = no1;                                                                                                                   \
	constexpr dreal KWC_3 = n1o3;                                                                                                                \
	constexpr dreal KWC_9 = n1o9;                                                                                                                \
	constexpr dreal KC_m00 = n1o6;                                                                                                               \
	constexpr dreal KC_z00 = n2o3;                                                                                                               \
	constexpr dreal KC_p00 = n1o6;                                                                                                               \
	constexpr dreal KC_m02 = n1o18;                                                                                                              \
	constexpr dreal KC_z02 = n2o9;                                                                                                               \
	constexpr dreal KC_p02 = n1o18;                                                                                                              \
	constexpr dreal KC_mm0 = n1o36;                                                                                                              \
	constexpr dreal KC_zm0 = n1o9;                                                                                                               \
	constexpr dreal KC_pm0 = n1o36;                                                                                                              \
	constexpr dreal KC_mz0 = n1o9;                                                                                                               \
	constexpr dreal KC_zz0 = n4o9;                                                                                                               \
	constexpr dreal KC_pz0 = n1o9;                                                                                                               \
	constexpr dreal KC_mp0 = n1o36;                                                                                                              \
	constexpr dreal KC_zp0 = n1o9;                                                                                                               \
	constexpr dreal KC_pp0 = n1o36;                                                                                                              \
                                                                                                                                                 \
	const dreal ks_000 = rho_f - KWC;                                                                                                            \
	const dreal ks_100 = 0;                                                                                                                      \
	const dreal ks_010 = 0;                                                                                                                      \
	const dreal ks_001 = 0;                                                                                                                      \
	const dreal ks_200 = C200 - KWC_3;                                                                                                           \
	const dreal ks_020 = C020 - KWC_3;                                                                                                           \
	const dreal ks_002 = C002 - KWC_3;                                                                                                           \
	const dreal ks_110 = C110;                                                                                                                   \
	const dreal ks_101 = C101;                                                                                                                   \
	const dreal ks_011 = C011;                                                                                                                   \
	const dreal ks_210 = 0;                                                                                                                      \
	const dreal ks_120 = 0;                                                                                                                      \
	const dreal ks_201 = 0;                                                                                                                      \
	const dreal ks_102 = 0;                                                                                                                      \
	const dreal ks_021 = 0;                                                                                                                      \
	const dreal ks_012 = 0;                                                                                                                      \
	const dreal ks_111 = 0;                                                                                                                      \
                                                                                                                                                 \
	const dreal rho_inv = no1 / rho_f; /* PHYSICAL destination density in both conventions (see the docstring above) */                          \
	const dreal vx_sqr = vx_f * vx_f;                                                                                                            \
	const dreal vy_sqr = vy_f * vy_f;                                                                                                            \
	const dreal vz_sqr = vz_f * vz_f;                                                                                                            \
                                                                                                                                                 \
	const dreal ks_211 = ((ks_200 + KWC_3) * ks_011 + no2 * ks_101 * ks_110) * rho_inv;                                                          \
	const dreal ks_121 = ((ks_020 + KWC_3) * ks_101 + no2 * ks_110 * ks_011) * rho_inv;                                                          \
	const dreal ks_112 = ((ks_002 + KWC_3) * ks_110 + no2 * ks_011 * ks_101) * rho_inv;                                                          \
	const dreal ks_220 = ((ks_020 * ks_200 + no2 * ks_110 * ks_110) + (ks_020 + ks_200) * KWC_3 - KWC_9 * ks_000) * rho_inv;                     \
	const dreal ks_022 = ((ks_002 * ks_020 + no2 * ks_011 * ks_011) + (ks_002 + ks_020) * KWC_3 - KWC_9 * ks_000) * rho_inv;                     \
	const dreal ks_202 = ((ks_200 * ks_002 + no2 * ks_101 * ks_101) + (ks_200 + ks_002) * KWC_3 - KWC_9 * ks_000) * rho_inv;                     \
                                                                                                                                                 \
	const dreal ks_122 = 0;                                                                                                                      \
	const dreal ks_212 = 0;                                                                                                                      \
	const dreal ks_221 = 0;                                                                                                                      \
                                                                                                                                                 \
	dreal ks_222 = (ks_200 * ks_022 + ks_020 * ks_202 + ks_002 * ks_220 + no4 * (ks_011 * ks_211 + ks_101 * ks_121 + ks_110 * ks_112)) * rho_inv \
				 - (no16 * ks_110 * ks_101 * ks_011 + no4 * (ks_101 * ks_101 * ks_020 + ks_011 * ks_011 * ks_200 + ks_110 * ks_110 * ks_002)     \
					+ no2 * ks_200 * ks_020 * ks_002)                                                                                            \
					   * rho_inv * rho_inv;                                                                                                      \
	if constexpr (COLL::is_well_conditioned) {                                                                                                   \
		ks_222 += (no3 * (ks_022 + ks_202 + ks_220) + (ks_200 + ks_020 + ks_002)) * n1o9 * rho_inv                                               \
				- n2o3                                                                                                                           \
					  * (no2 * (ks_101 * ks_101 + ks_011 * ks_011 + ks_110 * ks_110) + (ks_002 * ks_020 + ks_002 * ks_200 + ks_020 * ks_200)     \
						 + (ks_002 + ks_020 + ks_200) * n1o3)                                                                                    \
					  * rho_inv * rho_inv                                                                                                        \
				- (ks_000 * ks_000 - ks_000) * n1o27 * rho_inv * rho_inv;                                                                        \
	}                                                                                                                                            \
                                                                                                                                                 \
	const dreal ks_z00 = ks_000 * (no1 - vx_sqr) - no2 * vx_f * ks_100 - ks_200 - KWC * vx_sqr;                                                  \
	const dreal ks_z01 = ks_001 * (no1 - vx_sqr) - no2 * vx_f * ks_101 - ks_201;                                                                 \
	const dreal ks_z02 = ks_002 * (no1 - vx_sqr) - no2 * vx_f * ks_102 - ks_202 - KWC_3 * vx_sqr;                                                \
	const dreal ks_z10 = ks_010 * (no1 - vx_sqr) - no2 * vx_f * ks_110 - ks_210;                                                                 \
	const dreal ks_z11 = ks_011 * (no1 - vx_sqr) - no2 * vx_f * ks_111 - ks_211;                                                                 \
	const dreal ks_z12 = ks_012 * (no1 - vx_sqr) - no2 * vx_f * ks_112 - ks_212;                                                                 \
	const dreal ks_z20 = ks_020 * (no1 - vx_sqr) - no2 * vx_f * ks_120 - ks_220 - KWC_3 * vx_sqr;                                                \
	const dreal ks_z21 = ks_021 * (no1 - vx_sqr) - no2 * vx_f * ks_121 - ks_221;                                                                 \
	const dreal ks_z22 = ks_022 * (no1 - vx_sqr) - no2 * vx_f * ks_122 - ks_222 - KWC_9 * vx_sqr;                                                \
                                                                                                                                                 \
	const dreal ks_m00 = ((ks_000 + KWC) * (vx_sqr - vx_f) + ks_100 * (no2 * vx_f - no1) + ks_200) * n1o2;                                       \
	const dreal ks_m01 = (ks_001 * (vx_sqr - vx_f) + ks_101 * (no2 * vx_f - no1) + ks_201) * n1o2;                                               \
	const dreal ks_m02 = ((ks_002 + KWC_3) * (vx_sqr - vx_f) + ks_102 * (no2 * vx_f - no1) + ks_202) * n1o2;                                     \
	const dreal ks_m10 = (ks_010 * (vx_sqr - vx_f) + ks_110 * (no2 * vx_f - no1) + ks_210) * n1o2;                                               \
	const dreal ks_m11 = (ks_011 * (vx_sqr - vx_f) + ks_111 * (no2 * vx_f - no1) + ks_211) * n1o2;                                               \
	const dreal ks_m12 = (ks_012 * (vx_sqr - vx_f) + ks_112 * (no2 * vx_f - no1) + ks_212) * n1o2;                                               \
	const dreal ks_m20 = ((ks_020 + KWC_3) * (vx_sqr - vx_f) + ks_120 * (no2 * vx_f - no1) + ks_220) * n1o2;                                     \
	const dreal ks_m21 = (ks_021 * (vx_sqr - vx_f) + ks_121 * (no2 * vx_f - no1) + ks_221) * n1o2;                                               \
	const dreal ks_m22 = ((ks_022 + KWC_9) * (vx_sqr - vx_f) + ks_122 * (no2 * vx_f - no1) + ks_222) * n1o2;                                     \
                                                                                                                                                 \
	const dreal ks_p00 = ((ks_000 + KWC) * (vx_sqr + vx_f) + ks_100 * (no2 * vx_f + no1) + ks_200) * n1o2;                                       \
	const dreal ks_p01 = (ks_001 * (vx_sqr + vx_f) + ks_101 * (no2 * vx_f + no1) + ks_201) * n1o2;                                               \
	const dreal ks_p02 = ((ks_002 + KWC_3) * (vx_sqr + vx_f) + ks_102 * (no2 * vx_f + no1) + ks_202) * n1o2;                                     \
	const dreal ks_p10 = (ks_010 * (vx_sqr + vx_f) + ks_110 * (no2 * vx_f + no1) + ks_210) * n1o2;                                               \
	const dreal ks_p11 = (ks_011 * (vx_sqr + vx_f) + ks_111 * (no2 * vx_f + no1) + ks_211) * n1o2;                                               \
	const dreal ks_p12 = (ks_012 * (vx_sqr + vx_f) + ks_112 * (no2 * vx_f + no1) + ks_212) * n1o2;                                               \
	const dreal ks_p20 = ((ks_020 + KWC_3) * (vx_sqr + vx_f) + ks_120 * (no2 * vx_f + no1) + ks_220) * n1o2;                                     \
	const dreal ks_p21 = (ks_021 * (vx_sqr + vx_f) + ks_121 * (no2 * vx_f + no1) + ks_221) * n1o2;                                               \
	const dreal ks_p22 = ((ks_022 + KWC_9) * (vx_sqr + vx_f) + ks_122 * (no2 * vx_f + no1) + ks_222) * n1o2;                                     \
                                                                                                                                                 \
	const dreal ks_mz0 = ks_m00 * (no1 - vy_sqr) - no2 * vy_f * ks_m10 - ks_m20 - KC_m00 * vy_sqr;                                               \
	const dreal ks_mz1 = ks_m01 * (no1 - vy_sqr) - no2 * vy_f * ks_m11 - ks_m21;                                                                 \
	const dreal ks_mz2 = ks_m02 * (no1 - vy_sqr) - no2 * vy_f * ks_m12 - ks_m22 - KC_m02 * vy_sqr;                                               \
	const dreal ks_zz0 = ks_z00 * (no1 - vy_sqr) - no2 * vy_f * ks_z10 - ks_z20 - KC_z00 * vy_sqr;                                               \
	const dreal ks_zz1 = ks_z01 * (no1 - vy_sqr) - no2 * vy_f * ks_z11 - ks_z21;                                                                 \
	const dreal ks_zz2 = ks_z02 * (no1 - vy_sqr) - no2 * vy_f * ks_z12 - ks_z22 - KC_z02 * vy_sqr;                                               \
	const dreal ks_pz0 = ks_p00 * (no1 - vy_sqr) - no2 * vy_f * ks_p10 - ks_p20 - KC_p00 * vy_sqr;                                               \
	const dreal ks_pz1 = ks_p01 * (no1 - vy_sqr) - no2 * vy_f * ks_p11 - ks_p21;                                                                 \
	const dreal ks_pz2 = ks_p02 * (no1 - vy_sqr) - no2 * vy_f * ks_p12 - ks_p22 - KC_p02 * vy_sqr;                                               \
                                                                                                                                                 \
	const dreal ks_mm0 = ((ks_m00 + KC_m00) * (vy_sqr - vy_f) + ks_m10 * (no2 * vy_f - no1) + ks_m20) * n1o2;                                    \
	const dreal ks_mm1 = (ks_m01 * (vy_sqr - vy_f) + ks_m11 * (no2 * vy_f - no1) + ks_m21) * n1o2;                                               \
	const dreal ks_mm2 = ((ks_m02 + KC_m02) * (vy_sqr - vy_f) + ks_m12 * (no2 * vy_f - no1) + ks_m22) * n1o2;                                    \
	const dreal ks_zm0 = ((ks_z00 + KC_z00) * (vy_sqr - vy_f) + ks_z10 * (no2 * vy_f - no1) + ks_z20) * n1o2;                                    \
	const dreal ks_zm1 = (ks_z01 * (vy_sqr - vy_f) + ks_z11 * (no2 * vy_f - no1) + ks_z21) * n1o2;                                               \
	const dreal ks_zm2 = ((ks_z02 + KC_z02) * (vy_sqr - vy_f) + ks_z12 * (no2 * vy_f - no1) + ks_z22) * n1o2;                                    \
	const dreal ks_pm0 = ((ks_p00 + KC_p00) * (vy_sqr - vy_f) + ks_p10 * (no2 * vy_f - no1) + ks_p20) * n1o2;                                    \
	const dreal ks_pm1 = (ks_p01 * (vy_sqr - vy_f) + ks_p11 * (no2 * vy_f - no1) + ks_p21) * n1o2;                                               \
	const dreal ks_pm2 = ((ks_p02 + KC_p02) * (vy_sqr - vy_f) + ks_p12 * (no2 * vy_f - no1) + ks_p22) * n1o2;                                    \
                                                                                                                                                 \
	const dreal ks_mp0 = ((ks_m00 + KC_m00) * (vy_sqr + vy_f) + ks_m10 * (no2 * vy_f + no1) + ks_m20) * n1o2;                                    \
	const dreal ks_mp1 = (ks_m01 * (vy_sqr + vy_f) + ks_m11 * (no2 * vy_f + no1) + ks_m21) * n1o2;                                               \
	const dreal ks_mp2 = ((ks_m02 + KC_m02) * (vy_sqr + vy_f) + ks_m12 * (no2 * vy_f + no1) + ks_m22) * n1o2;                                    \
	const dreal ks_zp0 = ((ks_z00 + KC_z00) * (vy_sqr + vy_f) + ks_z10 * (no2 * vy_f + no1) + ks_z20) * n1o2;                                    \
	const dreal ks_zp1 = (ks_z01 * (vy_sqr + vy_f) + ks_z11 * (no2 * vy_f + no1) + ks_z21) * n1o2;                                               \
	const dreal ks_zp2 = ((ks_z02 + KC_z02) * (vy_sqr + vy_f) + ks_z12 * (no2 * vy_f + no1) + ks_z22) * n1o2;                                    \
	const dreal ks_pp0 = ((ks_p00 + KC_p00) * (vy_sqr + vy_f) + ks_p10 * (no2 * vy_f + no1) + ks_p20) * n1o2;                                    \
	const dreal ks_pp1 = (ks_p01 * (vy_sqr + vy_f) + ks_p11 * (no2 * vy_f + no1) + ks_p21) * n1o2;                                               \
	const dreal ks_pp2 = ((ks_p02 + KC_p02) * (vy_sqr + vy_f) + ks_p12 * (no2 * vy_f + no1) + ks_p22) * n1o2;                                    \
                                                                                                                                                 \
	store_df(mmz, ks_mm0*(no1 - vz_sqr) - no2 * vz_f * ks_mm1 - ks_mm2 - KC_mm0 * vz_sqr);                                                       \
	store_df(mzz, ks_mz0*(no1 - vz_sqr) - no2 * vz_f * ks_mz1 - ks_mz2 - KC_mz0 * vz_sqr);                                                       \
	store_df(mpz, ks_mp0*(no1 - vz_sqr) - no2 * vz_f * ks_mp1 - ks_mp2 - KC_mp0 * vz_sqr);                                                       \
	store_df(zmz, ks_zm0*(no1 - vz_sqr) - no2 * vz_f * ks_zm1 - ks_zm2 - KC_zm0 * vz_sqr);                                                       \
	store_df(zzz, ks_zz0*(no1 - vz_sqr) - no2 * vz_f * ks_zz1 - ks_zz2 - KC_zz0 * vz_sqr);                                                       \
	store_df(zpz, ks_zp0*(no1 - vz_sqr) - no2 * vz_f * ks_zp1 - ks_zp2 - KC_zp0 * vz_sqr);                                                       \
	store_df(pmz, ks_pm0*(no1 - vz_sqr) - no2 * vz_f * ks_pm1 - ks_pm2 - KC_pm0 * vz_sqr);                                                       \
	store_df(pzz, ks_pz0*(no1 - vz_sqr) - no2 * vz_f * ks_pz1 - ks_pz2 - KC_pz0 * vz_sqr);                                                       \
	store_df(ppz, ks_pp0*(no1 - vz_sqr) - no2 * vz_f * ks_pp1 - ks_pp2 - KC_pp0 * vz_sqr);                                                       \
                                                                                                                                                 \
	store_df(mmm, ((ks_mm0 + KC_mm0) * (vz_sqr - vz_f) + ks_mm1 * (no2 * vz_f - no1) + ks_mm2) * n1o2);                                          \
	store_df(mzm, ((ks_mz0 + KC_mz0) * (vz_sqr - vz_f) + ks_mz1 * (no2 * vz_f - no1) + ks_mz2) * n1o2);                                          \
	store_df(mpm, ((ks_mp0 + KC_mp0) * (vz_sqr - vz_f) + ks_mp1 * (no2 * vz_f - no1) + ks_mp2) * n1o2);                                          \
	store_df(zmm, ((ks_zm0 + KC_zm0) * (vz_sqr - vz_f) + ks_zm1 * (no2 * vz_f - no1) + ks_zm2) * n1o2);                                          \
	store_df(zzm, ((ks_zz0 + KC_zz0) * (vz_sqr - vz_f) + ks_zz1 * (no2 * vz_f - no1) + ks_zz2) * n1o2);                                          \
	store_df(zpm, ((ks_zp0 + KC_zp0) * (vz_sqr - vz_f) + ks_zp1 * (no2 * vz_f - no1) + ks_zp2) * n1o2);                                          \
	store_df(pmm, ((ks_pm0 + KC_pm0) * (vz_sqr - vz_f) + ks_pm1 * (no2 * vz_f - no1) + ks_pm2) * n1o2);                                          \
	store_df(pzm, ((ks_pz0 + KC_pz0) * (vz_sqr - vz_f) + ks_pz1 * (no2 * vz_f - no1) + ks_pz2) * n1o2);                                          \
	store_df(ppm, ((ks_pp0 + KC_pp0) * (vz_sqr - vz_f) + ks_pp1 * (no2 * vz_f - no1) + ks_pp2) * n1o2);                                          \
                                                                                                                                                 \
	store_df(mmp, ((ks_mm0 + KC_mm0) * (vz_sqr + vz_f) + ks_mm1 * (no2 * vz_f + no1) + ks_mm2) * n1o2);                                          \
	store_df(mzp, ((ks_mz0 + KC_mz0) * (vz_sqr + vz_f) + ks_mz1 * (no2 * vz_f + no1) + ks_mz2) * n1o2);                                          \
	store_df(mpp, ((ks_mp0 + KC_mp0) * (vz_sqr + vz_f) + ks_mp1 * (no2 * vz_f + no1) + ks_mp2) * n1o2);                                          \
	store_df(zmp, ((ks_zm0 + KC_zm0) * (vz_sqr + vz_f) + ks_zm1 * (no2 * vz_f + no1) + ks_zm2) * n1o2);                                          \
	store_df(zzp, ((ks_zz0 + KC_zz0) * (vz_sqr + vz_f) + ks_zz1 * (no2 * vz_f + no1) + ks_zz2) * n1o2);                                          \
	store_df(zpp, ((ks_zp0 + KC_zp0) * (vz_sqr + vz_f) + ks_zp1 * (no2 * vz_f + no1) + ks_zp2) * n1o2);                                          \
	store_df(pmp, ((ks_pm0 + KC_pm0) * (vz_sqr + vz_f) + ks_pm1 * (no2 * vz_f + no1) + ks_pm2) * n1o2);                                          \
	store_df(pzp, ((ks_pz0 + KC_pz0) * (vz_sqr + vz_f) + ks_pz1 * (no2 * vz_f + no1) + ks_pz2) * n1o2);                                          \
	store_df(ppp, ((ks_pp0 + KC_pp0) * (vz_sqr + vz_f) + ks_pp1 * (no2 * vz_f + no1) + ks_pp2) * n1o2)

#ifdef USE_GEIER_CUM_2017
	/**
	 * \brief Geier-2017-consistent variant of the CM back-transformation
	 * (USE_GEIER_CUM_2017 mode consistency, 2026-09-03).
	 *
	 * The plain-storage chain of AMR_CM_BACKTRANSFORM_PLAIN with the mode
	 * filter aligned to the collision's own persistent modes: the seven
	 * third-order cumulants are the TRANSFERRED destination values (the
	 * interpolation product of AMR_CM_THIRD_MOMENTS, expected in scope as
	 * k120_f, k210_f, k201_f, k102_f, k012_f, k021_f, k111_f) instead of
	 * being zeroed, and the fourth-order central moments take the full
	 * Eq. 83/84 forms of the Geier-2015 backward transformation (the same
	 * forms the USE_GEIER_CUM_2017 branch of col_cum.h lines 302-328 uses):
	 * Cs_122 = Cs_212 = Cs_221 = 0 and Cs_222 = 0 because the collision
	 * relaxes those cumulants at omega9 = omega10 = 1 every step, so the
	 * factorized-from-lower-moments expressions below are exactly the
	 * collision's own post-collision central moments. Cs_211/121/112 and
	 * Cs_220/202/022 keep the factorized forms of the plain chain: under the
	 * macro they carry the collision's A,B correction terms of O(nu * strain
	 * * rho) (~1e-6 at the AMR sim parameters), orders of magnitude below
	 * the third-order content this variant restores -- a documented
	 * residual, not transferred. The third-order cumulants themselves
	 * receive no relaxation-rate rescaling (mode state, not strain; see the
	 * AMR_CM_THIRD_MOMENTS docstring). Well-conditioned storage dispatches
	 * to the plain mode-filtered chain (the K-corrected Eqs. 83/84 are not
	 * derived; the well collision under the macro is out of scope and keeps
	 * its previous fill).
	 */
	#define AMR_CM_BACKTRANSFORM_GEIER(store_df)        \
		if constexpr (COLL::is_well_conditioned) {      \
			AMR_CM_BACKTRANSFORM_WELL(store_df);        \
		}                                               \
		else {                                          \
			AMR_CM_BACKTRANSFORM_GEIER_PLAIN(store_df); \
		}

	#define AMR_CM_BACKTRANSFORM_GEIER_PLAIN(store_df)                                                                                          \
		const dreal ks_000 = rho_f;                                                                                                             \
		const dreal ks_100 = 0;                                                                                                                 \
		const dreal ks_010 = 0;                                                                                                                 \
		const dreal ks_001 = 0;                                                                                                                 \
		const dreal ks_200 = C200;                                                                                                              \
		const dreal ks_020 = C020;                                                                                                              \
		const dreal ks_002 = C002;                                                                                                              \
		const dreal ks_110 = C110;                                                                                                              \
		const dreal ks_101 = C101;                                                                                                              \
		const dreal ks_011 = C011;                                                                                                              \
		const dreal ks_210 = k210_f;                                                                                                            \
		const dreal ks_120 = k120_f;                                                                                                            \
		const dreal ks_201 = k201_f;                                                                                                            \
		const dreal ks_102 = k102_f;                                                                                                            \
		const dreal ks_021 = k021_f;                                                                                                            \
		const dreal ks_012 = k012_f;                                                                                                            \
		const dreal ks_111 = k111_f;                                                                                                            \
                                                                                                                                                \
		const dreal rho_inv = no1 / rho_f;                                                                                                      \
		const dreal vx_sqr = vx_f * vx_f;                                                                                                       \
		const dreal vy_sqr = vy_f * vy_f;                                                                                                       \
		const dreal vz_sqr = vz_f * vz_f;                                                                                                       \
                                                                                                                                                \
		const dreal ks_211 = (ks_200 * ks_011 + no2 * ks_101 * ks_110) * rho_inv;                                                               \
		const dreal ks_121 = (ks_020 * ks_101 + no2 * ks_110 * ks_011) * rho_inv;                                                               \
		const dreal ks_112 = (ks_002 * ks_110 + no2 * ks_011 * ks_101) * rho_inv;                                                               \
		const dreal ks_220 = (ks_020 * ks_200 + no2 * ks_110 * ks_110) * rho_inv;                                                               \
		const dreal ks_022 = (ks_002 * ks_020 + no2 * ks_011 * ks_011) * rho_inv;                                                               \
		const dreal ks_202 = (ks_200 * ks_002 + no2 * ks_101 * ks_101) * rho_inv;                                                               \
                                                                                                                                                \
		const dreal ks_122 = (ks_020 * ks_102 + ks_002 * ks_120 + no4 * ks_011 * ks_111 + no2 * (ks_110 * ks_012 + ks_101 * ks_021)) * rho_inv; \
		const dreal ks_212 = (ks_002 * ks_210 + ks_200 * ks_012 + no4 * ks_101 * ks_111 + no2 * (ks_011 * ks_201 + ks_110 * ks_102)) * rho_inv; \
		const dreal ks_221 = (ks_200 * ks_021 + ks_020 * ks_201 + no4 * ks_110 * ks_111 + no2 * (ks_101 * ks_120 + ks_011 * ks_210)) * rho_inv; \
                                                                                                                                                \
		const dreal ks_222 =                                                                                                                    \
			(ks_200 * ks_022 + ks_020 * ks_202 + ks_002 * ks_220 + no4 * (ks_011 * ks_211 + ks_101 * ks_121 + ks_110 * ks_112)                  \
			 + no2 * (ks_120 * ks_102 + ks_210 * ks_012 + ks_201 * ks_021) + no4 * ks_111 * ks_111)                                             \
				* rho_inv                                                                                                                       \
			- (no16 * ks_110 * ks_101 * ks_011 + no4 * (ks_101 * ks_101 * ks_020 + ks_011 * ks_011 * ks_200 + ks_110 * ks_110 * ks_002)         \
			   + no2 * ks_200 * ks_020 * ks_002)                                                                                                \
				  * rho_inv * rho_inv;                                                                                                          \
                                                                                                                                                \
		const dreal ks_z00 = ks_000 * (no1 - vx_sqr) - no2 * vx_f * ks_100 - ks_200;                                                            \
		const dreal ks_z01 = ks_001 * (no1 - vx_sqr) - no2 * vx_f * ks_101 - ks_201;                                                            \
		const dreal ks_z02 = ks_002 * (no1 - vx_sqr) - no2 * vx_f * ks_102 - ks_202;                                                            \
		const dreal ks_z10 = ks_010 * (no1 - vx_sqr) - no2 * vx_f * ks_110 - ks_210;                                                            \
		const dreal ks_z11 = ks_011 * (no1 - vx_sqr) - no2 * vx_f * ks_111 - ks_211;                                                            \
		const dreal ks_z12 = ks_012 * (no1 - vx_sqr) - no2 * vx_f * ks_112 - ks_212;                                                            \
		const dreal ks_z20 = ks_020 * (no1 - vx_sqr) - no2 * vx_f * ks_120 - ks_220;                                                            \
		const dreal ks_z21 = ks_021 * (no1 - vx_sqr) - no2 * vx_f * ks_121 - ks_221;                                                            \
		const dreal ks_z22 = ks_022 * (no1 - vx_sqr) - no2 * vx_f * ks_122 - ks_222;                                                            \
                                                                                                                                                \
		const dreal ks_m00 = (ks_000 * (vx_sqr - vx_f) + ks_100 * (no2 * vx_f - no1) + ks_200) * n1o2;                                          \
		const dreal ks_m01 = (ks_001 * (vx_sqr - vx_f) + ks_101 * (no2 * vx_f - no1) + ks_201) * n1o2;                                          \
		const dreal ks_m02 = (ks_002 * (vx_sqr - vx_f) + ks_102 * (no2 * vx_f - no1) + ks_202) * n1o2;                                          \
		const dreal ks_m10 = (ks_010 * (vx_sqr - vx_f) + ks_110 * (no2 * vx_f - no1) + ks_210) * n1o2;                                          \
		const dreal ks_m11 = (ks_011 * (vx_sqr - vx_f) + ks_111 * (no2 * vx_f - no1) + ks_211) * n1o2;                                          \
		const dreal ks_m12 = (ks_012 * (vx_sqr - vx_f) + ks_112 * (no2 * vx_f - no1) + ks_212) * n1o2;                                          \
		const dreal ks_m20 = (ks_020 * (vx_sqr - vx_f) + ks_120 * (no2 * vx_f - no1) + ks_220) * n1o2;                                          \
		const dreal ks_m21 = (ks_021 * (vx_sqr - vx_f) + ks_121 * (no2 * vx_f - no1) + ks_221) * n1o2;                                          \
		const dreal ks_m22 = (ks_022 * (vx_sqr - vx_f) + ks_122 * (no2 * vx_f - no1) + ks_222) * n1o2;                                          \
                                                                                                                                                \
		const dreal ks_p00 = (ks_000 * (vx_sqr + vx_f) + ks_100 * (no2 * vx_f + no1) + ks_200) * n1o2;                                          \
		const dreal ks_p01 = (ks_001 * (vx_sqr + vx_f) + ks_101 * (no2 * vx_f + no1) + ks_201) * n1o2;                                          \
		const dreal ks_p02 = (ks_002 * (vx_sqr + vx_f) + ks_102 * (no2 * vx_f + no1) + ks_202) * n1o2;                                          \
		const dreal ks_p10 = (ks_010 * (vx_sqr + vx_f) + ks_110 * (no2 * vx_f + no1) + ks_210) * n1o2;                                          \
		const dreal ks_p11 = (ks_011 * (vx_sqr + vx_f) + ks_111 * (no2 * vx_f + no1) + ks_211) * n1o2;                                          \
		const dreal ks_p12 = (ks_012 * (vx_sqr + vx_f) + ks_112 * (no2 * vx_f + no1) + ks_212) * n1o2;                                          \
		const dreal ks_p20 = (ks_020 * (vx_sqr + vx_f) + ks_120 * (no2 * vx_f + no1) + ks_220) * n1o2;                                          \
		const dreal ks_p21 = (ks_021 * (vx_sqr + vx_f) + ks_121 * (no2 * vx_f + no1) + ks_221) * n1o2;                                          \
		const dreal ks_p22 = (ks_022 * (vx_sqr + vx_f) + ks_122 * (no2 * vx_f + no1) + ks_222) * n1o2;                                          \
                                                                                                                                                \
		const dreal ks_mz0 = ks_m00 * (no1 - vy_sqr) - no2 * vy_f * ks_m10 - ks_m20;                                                            \
		const dreal ks_mz1 = ks_m01 * (no1 - vy_sqr) - no2 * vy_f * ks_m11 - ks_m21;                                                            \
		const dreal ks_mz2 = ks_m02 * (no1 - vy_sqr) - no2 * vy_f * ks_m12 - ks_m22;                                                            \
		const dreal ks_zz0 = ks_z00 * (no1 - vy_sqr) - no2 * vy_f * ks_z10 - ks_z20;                                                            \
		const dreal ks_zz1 = ks_z01 * (no1 - vy_sqr) - no2 * vy_f * ks_z11 - ks_z21;                                                            \
		const dreal ks_zz2 = ks_z02 * (no1 - vy_sqr) - no2 * vy_f * ks_z12 - ks_z22;                                                            \
		const dreal ks_pz0 = ks_p00 * (no1 - vy_sqr) - no2 * vy_f * ks_p10 - ks_p20;                                                            \
		const dreal ks_pz1 = ks_p01 * (no1 - vy_sqr) - no2 * vy_f * ks_p11 - ks_p21;                                                            \
		const dreal ks_pz2 = ks_p02 * (no1 - vy_sqr) - no2 * vy_f * ks_p12 - ks_p22;                                                            \
                                                                                                                                                \
		const dreal ks_mm0 = (ks_m00 * (vy_sqr - vy_f) + ks_m10 * (no2 * vy_f - no1) + ks_m20) * n1o2;                                          \
		const dreal ks_mm1 = (ks_m01 * (vy_sqr - vy_f) + ks_m11 * (no2 * vy_f - no1) + ks_m21) * n1o2;                                          \
		const dreal ks_mm2 = (ks_m02 * (vy_sqr - vy_f) + ks_m12 * (no2 * vy_f - no1) + ks_m22) * n1o2;                                          \
		const dreal ks_zm0 = (ks_z00 * (vy_sqr - vy_f) + ks_z10 * (no2 * vy_f - no1) + ks_z20) * n1o2;                                          \
		const dreal ks_zm1 = (ks_z01 * (vy_sqr - vy_f) + ks_z11 * (no2 * vy_f - no1) + ks_z21) * n1o2;                                          \
		const dreal ks_zm2 = (ks_z02 * (vy_sqr - vy_f) + ks_z12 * (no2 * vy_f - no1) + ks_z22) * n1o2;                                          \
		const dreal ks_pm0 = (ks_p00 * (vy_sqr - vy_f) + ks_p10 * (no2 * vy_f - no1) + ks_p20) * n1o2;                                          \
		const dreal ks_pm1 = (ks_p01 * (vy_sqr - vy_f) + ks_p11 * (no2 * vy_f - no1) + ks_p21) * n1o2;                                          \
		const dreal ks_pm2 = (ks_p02 * (vy_sqr - vy_f) + ks_p12 * (no2 * vy_f - no1) + ks_p22) * n1o2;                                          \
                                                                                                                                                \
		const dreal ks_mp0 = (ks_m00 * (vy_sqr + vy_f) + ks_m10 * (no2 * vy_f + no1) + ks_m20) * n1o2;                                          \
		const dreal ks_mp1 = (ks_m01 * (vy_sqr + vy_f) + ks_m11 * (no2 * vy_f + no1) + ks_m21) * n1o2;                                          \
		const dreal ks_mp2 = (ks_m02 * (vy_sqr + vy_f) + ks_m12 * (no2 * vy_f + no1) + ks_m22) * n1o2;                                          \
		const dreal ks_zp0 = (ks_z00 * (vy_sqr + vy_f) + ks_z10 * (no2 * vy_f + no1) + ks_z20) * n1o2;                                          \
		const dreal ks_zp1 = (ks_z01 * (vy_sqr + vy_f) + ks_z11 * (no2 * vy_f + no1) + ks_z21) * n1o2;                                          \
		const dreal ks_zp2 = (ks_z02 * (vy_sqr + vy_f) + ks_z12 * (no2 * vy_f + no1) + ks_z22) * n1o2;                                          \
		const dreal ks_pp0 = (ks_p00 * (vy_sqr + vy_f) + ks_p10 * (no2 * vy_f + no1) + ks_p20) * n1o2;                                          \
		const dreal ks_pp1 = (ks_p01 * (vy_sqr + vy_f) + ks_p11 * (no2 * vy_f + no1) + ks_p21) * n1o2;                                          \
		const dreal ks_pp2 = (ks_p02 * (vy_sqr + vy_f) + ks_p12 * (no2 * vy_f + no1) + ks_p22) * n1o2;                                          \
                                                                                                                                                \
		store_df(mmz, ks_mm0*(no1 - vz_sqr) - no2 * vz_f * ks_mm1 - ks_mm2);                                                                    \
		store_df(mzz, ks_mz0*(no1 - vz_sqr) - no2 * vz_f * ks_mz1 - ks_mz2);                                                                    \
		store_df(mpz, ks_mp0*(no1 - vz_sqr) - no2 * vz_f * ks_mp1 - ks_mp2);                                                                    \
		store_df(zmz, ks_zm0*(no1 - vz_sqr) - no2 * vz_f * ks_zm1 - ks_zm2);                                                                    \
		store_df(zzz, ks_zz0*(no1 - vz_sqr) - no2 * vz_f * ks_zz1 - ks_zz2);                                                                    \
		store_df(zpz, ks_zp0*(no1 - vz_sqr) - no2 * vz_f * ks_zp1 - ks_zp2);                                                                    \
		store_df(pmz, ks_pm0*(no1 - vz_sqr) - no2 * vz_f * ks_pm1 - ks_pm2);                                                                    \
		store_df(pzz, ks_pz0*(no1 - vz_sqr) - no2 * vz_f * ks_pz1 - ks_pz2);                                                                    \
		store_df(ppz, ks_pp0*(no1 - vz_sqr) - no2 * vz_f * ks_pp1 - ks_pp2);                                                                    \
                                                                                                                                                \
		store_df(mmm, (ks_mm0 * (vz_sqr - vz_f) + ks_mm1 * (no2 * vz_f - no1) + ks_mm2) * n1o2);                                                \
		store_df(mzm, (ks_mz0 * (vz_sqr - vz_f) + ks_mz1 * (no2 * vz_f - no1) + ks_mz2) * n1o2);                                                \
		store_df(mpm, (ks_mp0 * (vz_sqr - vz_f) + ks_mp1 * (no2 * vz_f - no1) + ks_mp2) * n1o2);                                                \
		store_df(zmm, (ks_zm0 * (vz_sqr - vz_f) + ks_zm1 * (no2 * vz_f - no1) + ks_zm2) * n1o2);                                                \
		store_df(zzm, (ks_zz0 * (vz_sqr - vz_f) + ks_zz1 * (no2 * vz_f - no1) + ks_zz2) * n1o2);                                                \
		store_df(zpm, (ks_zp0 * (vz_sqr - vz_f) + ks_zp1 * (no2 * vz_f - no1) + ks_zp2) * n1o2);                                                \
		store_df(pmm, (ks_pm0 * (vz_sqr - vz_f) + ks_pm1 * (no2 * vz_f - no1) + ks_pm2) * n1o2);                                                \
		store_df(pzm, (ks_pz0 * (vz_sqr - vz_f) + ks_pz1 * (no2 * vz_f - no1) + ks_pz2) * n1o2);                                                \
		store_df(ppm, (ks_pp0 * (vz_sqr - vz_f) + ks_pp1 * (no2 * vz_f - no1) + ks_pp2) * n1o2);                                                \
                                                                                                                                                \
		store_df(mmp, (ks_mm0 * (vz_sqr + vz_f) + ks_mm1 * (no2 * vz_f + no1) + ks_mm2) * n1o2);                                                \
		store_df(mzp, (ks_mz0 * (vz_sqr + vz_f) + ks_mz1 * (no2 * vz_f + no1) + ks_mz2) * n1o2);                                                \
		store_df(mpp, (ks_mp0 * (vz_sqr + vz_f) + ks_mp1 * (no2 * vz_f + no1) + ks_mp2) * n1o2);                                                \
		store_df(zmp, (ks_zm0 * (vz_sqr + vz_f) + ks_zm1 * (no2 * vz_f + no1) + ks_zm2) * n1o2);                                                \
		store_df(zzp, (ks_zz0 * (vz_sqr + vz_f) + ks_zz1 * (no2 * vz_f + no1) + ks_zz2) * n1o2);                                                \
		store_df(zpp, (ks_zp0 * (vz_sqr + vz_f) + ks_zp1 * (no2 * vz_f + no1) + ks_zp2) * n1o2);                                                \
		store_df(pmp, (ks_pm0 * (vz_sqr + vz_f) + ks_pm1 * (no2 * vz_f + no1) + ks_pm2) * n1o2);                                                \
		store_df(pzp, (ks_pz0 * (vz_sqr + vz_f) + ks_pz1 * (no2 * vz_f + no1) + ks_pz2) * n1o2);                                                \
		store_df(ppp, (ks_pp0 * (vz_sqr + vz_f) + ks_pp1 * (no2 * vz_f + no1) + ks_pp2) * n1o2)
#endif

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

	// Schönherr group mapping (per-window organization of the thesis Sec.
	// 7.2 reference implementation): one thread computes the up-to-8
	// destination cells of a 2x2x2 destination group rather than a single
	// destination cell, so the source work shared by neighbouring
	// destinations is computed once; a thread's group origin is begin +
	// 2*tid per axis, so a launch grid spanning ceil-div-2 per axis is the
	// tight sizing -- today's callers grid per-cell, which is equally
	// correct: the group mapping is idempotent per cell and the excess
	// threads early-return (perf-only deferred trim)
	const idx gx0 = ghost_begin_fine.x() + 2 * (threadIdx.x + blockIdx.x * blockDim.x);
	const idx gy0 = ghost_begin_fine.y() + 2 * (threadIdx.y + blockIdx.y * blockDim.y);
	const idx gz0 = ghost_begin_fine.z() + 2 * (threadIdx.z + blockIdx.z * blockDim.z);
	if (gx0 >= ghost_end_fine.x() || gy0 >= ghost_end_fine.y() || gz0 >= ghost_end_fine.z())
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
	const auto store_fine_df = [&fine_SD](int q, idx x, idx y, idx z, dreal f) -> void
	{
#ifdef AB_PATTERN
		fine_SD.df(df_cur, q, x, y, z) = f;
#elif defined(AA_PATTERN)
		fine_SD.df(df_cur, opposite_direction(q), x, y, z) = f;
#endif
	};

	// per-destination macro write for GEO_AMR_INTERFACE cells (no-op in v1,
	// see the file docstring)
	const auto write_fine_macro = [&fine_SD](idx x, idx y, idx z, dreal rho_f, dreal vx_f, dreal vy_f, dreal vz_f) -> void
	{
		if (fine_SD.map(x, y, z) == BC::GEO_AMR_INTERFACE) {
			fine_SD.macro(MACRO::e_rho, x, y, z) = rho_f;
			fine_SD.macro(MACRO::e_vx, x, y, z) = vx_f;
			fine_SD.macro(MACRO::e_vy, x, y, z) = vy_f;
			fine_SD.macro(MACRO::e_vz, x, y, z) = vz_f;
		}
	};

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

	// destination-group loop (up to 8 cells per thread, bounds-guarded)
	for (int idz = 0; idz < 2; idz++) {
		const idx z = gz0 + idz;
		if (z >= ghost_end_fine.z())
			continue;
		for (int idy = 0; idy < 2; idy++) {
			const idx y = gy0 + idy;
			if (y >= ghost_end_fine.y())
				continue;
			for (int idx_ = 0; idx_ < 2; idx_++) {
				const idx x = gx0 + idx_;
				if (x >= ghost_end_fine.x())
					continue;

				const idx cx = clamped_home(x + fine_off.x(), coarse_off.x(), coarse_SD.X(), coarse_SD.indexer.template getOverlap<0>());
				const idx cy = clamped_home(y + fine_off.y(), coarse_off.y(), coarse_SD.Y(), coarse_SD.indexer.template getOverlap<1>());
				const idx cz = clamped_home(z + fine_off.z(), coarse_off.z(), coarse_SD.Z(), coarse_SD.indexer.template getOverlap<2>());

				// the home coarse cell's DF state and its macros
				LBM_KS KS_H;
				for (int q = 0; q < CONFIG::Q; q++)
					KS_H.f[q] = read_coarse_df(q, cx, cy, cz);
				COLL::computeDensityAndVelocity(KS_H);
				const dreal rho_f = KS_H.rho;
				const dreal vx_f = KS_H.vx;
				const dreal vy_f = KS_H.vy;
				const dreal vz_f = KS_H.vz;

	#ifdef C2F_LINEAR_EXPLOSION
				// linear explosion: the home cell's macros (rho, u) are
				// distributed to the fine subcells -- the equilibrium is
				// re-evaluated at those macros and the non-equilibrium part
				// is τ-rescaled from the home cell (preserves stress
				// information without neighbor-cell reads)
				LBM_KS KS_EQ;
				KS_EQ.rho = rho_f;
				KS_EQ.vx = vx_f;
				KS_EQ.vy = vy_f;
				KS_EQ.vz = vz_f;
				COLL::setEquilibrium(KS_EQ);
				const dreal neq_scale = tau_fine / tau_coarse;
				for (int q = 0; q < CONFIG::Q; q++)
					store_fine_df(q, x, y, z, KS_EQ.f[q] + neq_scale * (KS_H.f[q] - KS_EQ.f[q]));
	#else
				// uniform explosion: the home cell's DFs are duplicated to
				// every fine subcell unchanged (zeroth order; no
				// equilibrium re-evaluation, no rescaling)
				for (int q = 0; q < CONFIG::Q; q++)
					store_fine_df(q, x, y, z, KS_H.f[q]);
	#endif

				write_fine_macro(x, y, z, rho_f, vx_f, vy_f, vz_f);
			}
		}
	}

#elif ! defined(C2F_LAGRANGE) && ! defined(C2F_TRILINEAR)
	// (default branch -- C2F_COMPACT_MOMENT is accepted as an explicit selector
	// of this branch; 2026-08-18 default flip, user ruling; also under
	// USE_GEIER_CUM_2017 this stays the default: the per-direction
	// Lagrange/trilinear fills preserve the weakly-damped non-hydrodynamic
	// residual modes the 2017 operator keeps alive, which show up as a
	// checkerboard-like vx artifact -- user ruling 2026-09-03. This branch
	// is mode-consistent under the macro instead: it transfers the seven
	// persistent third-order cumulants and reconstructs with
	// AMR_CM_BACKTRANSFORM_GEIER; the measured seam/noise floor stays
	// meaningful (see the file docstring).)
	// ---- Compact moment-based interpolation (Schönherr 2015 thesis,
	// Sec. 7.2, Eqs. 7.10-7.48; see the file docstring for the outline):
	// the fine ghost cell brackets its home window of 2x2x2 coarse source
	// cells (the C2F_TRILINEAR stencil), computes the five independent
	// second-order non-equilibrium moments per source cell from
	// f_neq = f - f_eq (strain-rate information), fits the 8-coefficient
	// density polynomial and the three 11-coefficient velocity polynomials
	// (exact for linear fields; pure-quadratic exactness holds for the
	// k-corrected 11-coefficient VELOCITY fits only -- the 8-coefficient
	// density fit is the plain trilinear nodal fit, reproducing only
	// linear/constant densities exactly; D.5, 2026-08-16, per the A.2
	// symbolic/numeric check validated by Tests 8/9 of
	// tests/unit/test_amr_coupling.cu), corrects the averaged
	// moments by the fitted gradients, and reconstructs the fine DFs from
	// the six second-order cumulants via the cumulant back-transformation
	// of col_cum.h; third-order and higher central moments are set to
	// zero, so the non-hydrodynamic modes of D3Q27 are projected out at
	// the interface instead of being interpolated per direction.

	// relaxation rates of the source (coarse) and destination (fine) grids
	const dreal omega_s = no1 / tau_coarse;
	const dreal omega_d = no1 / tau_fine;

	// Per-destination axis window (nodes only; the Lagrange weights of the
	// default branch are not needed): the nominal window is
	// {home-1, home} for even fg and {home, home+1} for odd fg, shifted
	// into the coarse storage extent by the storability guard; the nodal
	// local coordinates are +-1/2 (bx=0 -> -1/2) and the fine cell center
	// evaluates at +-1/4 per axis. On a degenerate axis (storage extent 1)
	// the single cell is mirrored at both nodes: all coefficients of
	// monomials containing that axis then evaluate to zero, so the
	// polynomials reduce correctly.
	const auto axis_window = [&fdiv2](idx fg, idx coarse_off_a, idx size_a, idx ov_a, idx* nodes) -> dreal
	{
		const idx home = fdiv2(fg);
		const idx p = fg & 1;
		// fine cell center (evaluation point) in the coarse indexer frame
		const double t = static_cast<double>(home - coarse_off_a) + (p ? 0.25 : -0.25);
		// nominal 2-cell window home - 1 + (fg&1) + {0,1}, shifted into
		// the storage extent (storability guard)
		const int extent = static_cast<int>(size_a + 2 * ov_a);
		const int n = 2 < extent ? 2 : extent;
		const idx lo = -ov_a;
		const idx hi = size_a - 1 + ov_a - (n - 1);
		idx start = home - coarse_off_a - 1 + p;
		start = start < lo ? lo : (start > hi ? hi : start);
		nodes[0] = start;
		nodes[1] = start + (n - 1);
		// evaluation point relative to the window center (start + 1/2)
		return static_cast<dreal>(t - (static_cast<double>(start) + 0.5));
	};

	// ---- Phase 1: per-destination window solve for the group ----
	// Each of the up-to-8 destinations of the group carries the storage-
	// clamped window pair of its own fdiv2/axis_window solve (verbatim the
	// former single-destination behaviour); the union span of all in-range
	// destinations' windows bounds the shared source staging below.
	// Schönherr 2015 thesis, Sec. 7.3 (refinement at the wall): after the
	// nominal solve, a per-destination guard shifts any per-axis window
	// whose 2x2x2 source tuple holds a physically-tagged (non-live)
	// cell at exactly one of its ends one cell away from that end -- the
	// |offset| <= 1 wall extrapolation of the thesis, expressed as a window
	// shift plus relative evaluation rather than the hat-coefficient
	// transform (the algebraic equivalent, Sec. A.3.8). Live tags are
	// GEO_FLUID, GEO_AMR_INTERFACE and GEO_NOTHING (the frozen skin of the
	// ch7 band is F2C-refilled one cycle step before the C2F fill, so its
	// content is current); interior registrations never trip the guard.
	const auto shift_off_bc = [](idx* nodes, dreal& t_rel, bool taint_lo, bool taint_hi, idx lo, idx hi, idx home) -> void
	{
		if (nodes[0] == nodes[1])
			return;	 // storage-shortened axis: nothing left to shift
		if (taint_lo == taint_hi)
			return;	 // clean window, or both ends tainted (a residual case: no valid one-cell direction)
		const idx start = taint_hi ? nodes[0] - 1 : nodes[0] + 1;  // shift one cell away from the tainted end
		if (start < lo || start + 1 > hi) {
			// storage edge within one cell of the face: degenerate collapse
			// to the mirrored home cell (rejected at SimInit)
			t_rel += static_cast<dreal>(nodes[0] - home);
			nodes[0] = home;
			nodes[1] = home;
			return;
		}
		// evaluation point moves with the window center (the thesis offset)
		t_rel += static_cast<dreal>(nodes[0] - start);
		nodes[0] = start;
		nodes[1] = start + 1;
	};
	bool dst_in[8];
	idx dst_n[8][3][2];	 // per-destination per-axis window nodes {lo, hi}
	dreal dst_t[8][3];
	idx dst_home[8][3];	 // per-destination per-axis home cell (the degenerate-collapse target)
	idx umin[3], umax[3];
	bool u_init = false;
	for (int d = 0; d < 8; d++) {
		const idx fx = gx0 + (d & 1);
		const idx fy = gy0 + ((d >> 1) & 1);
		const idx fz = gz0 + ((d >> 2) & 1);
		dst_in[d] = fx < ghost_end_fine.x() && fy < ghost_end_fine.y() && fz < ghost_end_fine.z();
		if (! dst_in[d])
			continue;
		dst_t[d][0] = axis_window(fx + fine_off.x(), coarse_off.x(), coarse_SD.X(), coarse_SD.indexer.template getOverlap<0>(), dst_n[d][0]);
		dst_t[d][1] = axis_window(fy + fine_off.y(), coarse_off.y(), coarse_SD.Y(), coarse_SD.indexer.template getOverlap<1>(), dst_n[d][1]);
		dst_t[d][2] = axis_window(fz + fine_off.z(), coarse_off.z(), coarse_SD.Z(), coarse_SD.indexer.template getOverlap<2>(), dst_n[d][2]);
		dst_home[d][0] = fdiv2(fx + fine_off.x()) - coarse_off.x();
		dst_home[d][1] = fdiv2(fy + fine_off.y()) - coarse_off.y();
		dst_home[d][2] = fdiv2(fz + fine_off.z()) - coarse_off.z();

		// taint scan of the nominal 2x2x2 source tuple
		bool inv[2][2][2];
		bool tainted = false;
		for (int ibz = 0; ibz < 2; ibz++)
			for (int iby = 0; iby < 2; iby++)
				for (int ibx = 0; ibx < 2; ibx++) {
					const auto mapgi = coarse_SD.map(dst_n[d][0][ibx], dst_n[d][1][iby], dst_n[d][2][ibz]);
					inv[ibx][iby][ibz] = ! (BC::isFluid(mapgi) || mapgi == BC::GEO_AMR_INTERFACE || mapgi == BC::GEO_NOTHING);
					tainted = tainted || inv[ibx][iby][ibz];
				}
		if (tainted) {
			// end taints per axis (union over the two tangent axes)
			shift_off_bc(
				dst_n[d][0],
				dst_t[d][0],
				inv[0][0][0] || inv[0][0][1] || inv[0][1][0] || inv[0][1][1],
				inv[1][0][0] || inv[1][0][1] || inv[1][1][0] || inv[1][1][1],
				-coarse_SD.indexer.template getOverlap<0>(),
				coarse_SD.X() - 1 + coarse_SD.indexer.template getOverlap<0>(),
				dst_home[d][0]
			);
			shift_off_bc(
				dst_n[d][1],
				dst_t[d][1],
				inv[0][0][0] || inv[0][0][1] || inv[1][0][0] || inv[1][0][1],
				inv[0][1][0] || inv[0][1][1] || inv[1][1][0] || inv[1][1][1],
				-coarse_SD.indexer.template getOverlap<1>(),
				coarse_SD.Y() - 1 + coarse_SD.indexer.template getOverlap<1>(),
				dst_home[d][1]
			);
			shift_off_bc(
				dst_n[d][2],
				dst_t[d][2],
				inv[0][0][0] || inv[0][1][0] || inv[1][0][0] || inv[1][1][0],
				inv[0][0][1] || inv[0][1][1] || inv[1][0][1] || inv[1][1][1],
				-coarse_SD.indexer.template getOverlap<2>(),
				coarse_SD.Z() - 1 + coarse_SD.indexer.template getOverlap<2>(),
				dst_home[d][2]
			);

			// residual scan: a shifted tuple still carrying a non-live cell
			// (a physical BC thicker than one cell, or a mid-window
			// straddle -- both invalid registrations, rejected statically
			// by checkCouplingMapPattern at SimInit) collapses to the
			// destination's mirrored home cell
			bool residual = false;
			for (int ibz = 0; ibz < 2 && ! residual; ibz++)
				for (int iby = 0; iby < 2 && ! residual; iby++)
					for (int ibx = 0; ibx < 2 && ! residual; ibx++) {
						const auto mapgi = coarse_SD.map(dst_n[d][0][ibx], dst_n[d][1][iby], dst_n[d][2][ibz]);
						residual = ! (BC::isFluid(mapgi) || mapgi == BC::GEO_AMR_INTERFACE || mapgi == BC::GEO_NOTHING);
					}
			if (residual) {
				for (int a = 0; a < 3; a++) {
					dst_n[d][a][0] = dst_home[d][a];
					dst_n[d][a][1] = dst_home[d][a];
					dst_t[d][a] = 0;
				}
			}
		}

		if (! u_init) {
			u_init = true;
			umin[0] = umax[0] = dst_n[d][0][0];
			umin[1] = umax[1] = dst_n[d][1][0];
			umin[2] = umax[2] = dst_n[d][2][0];
		}
		umin[0] = std::min(std::min(dst_n[d][0][0], dst_n[d][0][1]), umin[0]);
		umax[0] = std::max(std::max(dst_n[d][0][0], dst_n[d][0][1]), umax[0]);
		umin[1] = std::min(std::min(dst_n[d][1][0], dst_n[d][1][1]), umin[1]);
		umax[1] = std::max(std::max(dst_n[d][1][0], dst_n[d][1][1]), umax[1]);
		umin[2] = std::min(std::min(dst_n[d][2][0], dst_n[d][2][1]), umin[2]);
		umax[2] = std::max(std::max(dst_n[d][2][0], dst_n[d][2][1]), umax[2]);
	}
	// group-span safety for the shared staging below: sibling destinations
	// whose windows took opposite wall shifts (a physical BC cutting the
	// group mid-window, the straddle class rejected at SimInit) can push a
	// per-axis union span past the 3-cell staging capacity -- never fires at
	// a valid registration (half-space BCs shift the whole face uniformly)
	for (int a = 0; a < 3; a++)
		if (umax[a] - umin[a] > 2) {
			bool a_init = true;
			for (int d = 0; d < 8; d++) {
				if (! dst_in[d])
					continue;
				dst_n[d][a][0] = dst_home[d][a];
				dst_n[d][a][1] = dst_home[d][a];
				dst_t[d][a] = 0;
				if (a_init) {
					a_init = false;
					umin[a] = umax[a] = dst_home[d][a];
					continue;
				}
				umin[a] = dst_home[d][a] < umin[a] ? dst_home[d][a] : umin[a];
				umax[a] = dst_home[d][a] > umax[a] ? dst_home[d][a] : umax[a];
			}
		}

	// ---- Phase 2: shared source staging ----
	// Read each coarse source cell of the group's union window ONCE (DF
	// loads, macros, the five second-order non-equilibrium moments) and
	// stage the rows every destination's donation chain consumes;
	// per-axis span is at most 3 (two parities' 2-cell windows)
	dreal su_rho[3][3][3], su_vx[3][3][3], su_vy[3][3][3], su_vz[3][3][3];
	dreal su_kxy[3][3][3], su_kyz[3][3][3], su_kxz[3][3][3], su_kxxyy[3][3][3], su_kxxzz[3][3][3];
	#if defined(USE_GEIER_CUM_2017) && ! defined(C2F_EQ_ONLY)
	// Geier 2017 mode consistency: the seven persistent third-order
	// cumulants per staged source cell (AMR_CM_THIRD_MOMENTS docstring)
	dreal su_c120[3][3][3], su_c210[3][3][3], su_c201[3][3][3], su_c102[3][3][3], su_c012[3][3][3], su_c021[3][3][3], su_c111[3][3][3];
	#endif
	for (int ju_z = 0; ju_z <= umax[2] - umin[2]; ju_z++) {
		const idx cz = umin[2] + ju_z;
		for (int ju_y = 0; ju_y <= umax[1] - umin[1]; ju_y++) {
			const idx cy = umin[1] + ju_y;
			for (int ju_x = 0; ju_x <= umax[0] - umin[0]; ju_x++) {
				const idx cx = umin[0] + ju_x;
				AMR_CM_MACROS_AND_KMOMENTS(read_coarse_df, cx, cy, cz);
	#ifndef C2F_EQ_ONLY
				// (skipped under the C2F_EQ_ONLY debug experiment, as in
				// the single-destination branch)
				AMR_CM_PI_NEQ;
	#endif
	#ifdef C2F_DEV_ONLY
				AMR_CM_PI_DEV;
	#endif
	#ifdef C2F_NORM_ONLY
				AMR_CM_PI_NORM;
	#endif
	#ifdef C2F_SHEAR_ONLY
				AMR_CM_PI_SHEAR;
	#endif
				AMR_CM_KMOMENTS(omega_s);
				su_rho[ju_z][ju_y][ju_x] = rho_n;
				su_vx[ju_z][ju_y][ju_x] = u;
				su_vy[ju_z][ju_y][ju_x] = v;
				su_vz[ju_z][ju_y][ju_x] = w;
				su_kxy[ju_z][ju_y][ju_x] = k_xy;
				su_kyz[ju_z][ju_y][ju_x] = k_yz;
				su_kxz[ju_z][ju_y][ju_x] = k_xz;
				su_kxxyy[ju_z][ju_y][ju_x] = k_xx_yy;
				su_kxxzz[ju_z][ju_y][ju_x] = k_xx_zz;
	#if defined(USE_GEIER_CUM_2017) && ! defined(C2F_EQ_ONLY)
				AMR_CM_THIRD_MOMENTS;
				su_c120[ju_z][ju_y][ju_x] = k_120;
				su_c210[ju_z][ju_y][ju_x] = k_210;
				su_c201[ju_z][ju_y][ju_x] = k_201;
				su_c102[ju_z][ju_y][ju_x] = k_102;
				su_c012[ju_z][ju_y][ju_x] = k_012;
				su_c021[ju_z][ju_y][ju_x] = k_021;
				su_c111[ju_z][ju_y][ju_x] = k_111;
	#endif
			}
		}
	}

	// (the former map-based carve pre-pass was removed on 2026-08-23 after
	// the simulated-band conversion: under the ch7 band map every nominal
	// source window of a valid registration straddles live GEO_AMR_INTERFACE
	// cells only, so the pre-pass could never fire; the frozen-window
	// invariant is asserted statically by checkCouplingMapPattern at
	// SimInit, and degenerate multi-footprint maps take the
	// storability-guard shorten of axis_window above)
	// ---- Phase 3: per-destination downstream for the group ----
	// Every destination of the group carries the full single-destination
	// synthesis with the donations taken from the shared staged union
	// rows (Steps B-D accumulators, the coefficient fits, the corrected
	// cumulants and the back-transformation); the donation slot order is
	// the destination's own window order
	for (int d = 0; d < 8; d++) {
		if (! dst_in[d])
			continue;
		const idx fx = gx0 + (d & 1);
		const idx fy = gy0 + ((d >> 1) & 1);
		const idx fz = gz0 + ((d >> 2) & 1);

		dreal rho_f = 0, vx_f = 0, vy_f = 0, vz_f = 0;
		dreal sd0 = 0, sdx = 0, sdy = 0, sdz = 0, sdxy = 0, sdyz = 0, sdxz = 0, sdxyz = 0;
		dreal sa0 = 0, sax = 0, say = 0, saz = 0, saxx = 0, sayy = 0, sazz = 0, saxy = 0, sayz = 0, saxz = 0, saxyz = 0;
		dreal sb0 = 0, sbx = 0, sby = 0, sbz = 0, sbxx = 0, sbyy = 0, sbzz = 0, sbxy = 0, sbxz = 0, sbyz = 0, sbxyz = 0;
		dreal sc0 = 0, scx = 0, scy = 0, scz = 0, scxx = 0, scyy = 0, sczz = 0, scxy = 0, scyz = 0, scxz = 0, scxyz = 0;
		dreal sk_xy = 0, sk_yz = 0, sk_xz = 0, sk_xx_yy = 0, sk_xx_zz = 0;
		for (int ibz = 0; ibz < 2; ibz++) {
			const dreal zn = static_cast<dreal>(ibz) - n1o2;
			const idx ju_z = dst_n[d][2][ibz] - umin[2];
			for (int iby = 0; iby < 2; iby++) {
				const dreal yn = static_cast<dreal>(iby) - n1o2;
				const idx ju_y = dst_n[d][1][iby] - umin[1];
				for (int ibx = 0; ibx < 2; ibx++) {
					const dreal xn = static_cast<dreal>(ibx) - n1o2;
					const idx ju_x = dst_n[d][0][ibx] - umin[0];

					// canonical donation scope of AMR_CM_FIT_ACCUMULATE,
					// served from the staged union rows (identical values
					// and accumulation order to the single-destination
					// branch's per-source recomputation)
					const dreal rho_n = su_rho[ju_z][ju_y][ju_x];
					const dreal u = su_vx[ju_z][ju_y][ju_x];
					const dreal v = su_vy[ju_z][ju_y][ju_x];
					const dreal w = su_vz[ju_z][ju_y][ju_x];
					const dreal k_xy = su_kxy[ju_z][ju_y][ju_x];
					const dreal k_yz = su_kyz[ju_z][ju_y][ju_x];
					const dreal k_xz = su_kxz[ju_z][ju_y][ju_x];
					const dreal k_xx_yy = su_kxxyy[ju_z][ju_y][ju_x];
					const dreal k_xx_zz = su_kxxzz[ju_z][ju_y][ju_x];
					AMR_CM_FIT_ACCUMULATE;
				}
			}
		}

		// Step C: density coefficients (Eqs. 7.10-7.17) + destination
		// density (Eq. 7.37); Step D: velocity coefficient families (Eqs.
		// 7.18-7.28 and the cyclic permutations)
		const dreal tx = dst_t[d][0];
		const dreal ty = dst_t[d][1];
		const dreal tz = dst_t[d][2];
		AMR_CM_FIT_COEFFICIENTS(tx, ty, tz);

		// velocities at the destination (Eqs. 7.34-7.36)
		AMR_CM_EVALUATE(tx, ty, tz);

		// Step E: averaged second-order moments with velocity-gradient
		// corrections (Eqs. 7.29-7.33) and Step F: second-order cumulants
		// at the destination (Eqs. 7.38-7.48); sigma_{c->f} = 1/2
		AMR_CM_CORRECTED_CUMULANTS(n1o2);

	#if defined(USE_GEIER_CUM_2017) && ! defined(C2F_EQ_ONLY)
		// Geier 2017 mode consistency: the seven persistent third-order
		// cumulants at the destination, evaluated with the same trilinear
		// nodal fit as the density (Eqs. 7.10-7.17 at (tx,ty,tz) -- the
		// d-coefficient evaluation is exactly this nodal-basis sum);
		// identity transfer, no relaxation-rate rescaling (mode state,
		// not strain -- AMR_CM_THIRD_MOMENTS docstring)
		dreal k120_f = 0, k210_f = 0, k201_f = 0, k102_f = 0, k012_f = 0, k021_f = 0, k111_f = 0;
		for (int i3z = 0; i3z < 2; i3z++) {
			const dreal w3z = n1o2 + no2 * (static_cast<dreal>(i3z) - n1o2) * tz;
			const idx j3z = dst_n[d][2][i3z] - umin[2];
			for (int i3y = 0; i3y < 2; i3y++) {
				const dreal w3y = n1o2 + no2 * (static_cast<dreal>(i3y) - n1o2) * ty;
				const idx j3y = dst_n[d][1][i3y] - umin[1];
				const dreal w3yz = w3y * w3z;
				for (int i3x = 0; i3x < 2; i3x++) {
					const dreal w3 = (n1o2 + no2 * (static_cast<dreal>(i3x) - n1o2) * tx) * w3yz;
					const idx j3x = dst_n[d][0][i3x] - umin[0];
					k120_f += w3 * su_c120[j3z][j3y][j3x];
					k210_f += w3 * su_c210[j3z][j3y][j3x];
					k201_f += w3 * su_c201[j3z][j3y][j3x];
					k102_f += w3 * su_c102[j3z][j3y][j3x];
					k012_f += w3 * su_c012[j3z][j3y][j3x];
					k021_f += w3 * su_c021[j3z][j3y][j3x];
					k111_f += w3 * su_c111[j3z][j3y][j3x];
				}
			}
		}
	#endif

		// Steps G-H: cumulant/central-moment state and the cumulant
		// back-transformation into the destination DFs (Geier 2015 Eqs.
		// 81-96)
		const auto store_df_cell = [&store_fine_df, fx, fy, fz](int q, dreal f) -> void
		{
			// the back-transformation emits STORAGE-convention values
			// directly: physical DFs on D3Q27_COMMON, fhat = f - w_q on
			// D3Q27_COMMON_WELL (the KWC/KC terms of AMR_CM_BACKTRANSFORM)
			store_fine_df(q, fx, fy, fz, f);
		};
	#if defined(USE_GEIER_CUM_2017) && ! defined(C2F_EQ_ONLY)
		AMR_CM_BACKTRANSFORM_GEIER(store_df_cell);
	#else
		AMR_CM_BACKTRANSFORM(store_df_cell);
	#endif

		write_fine_macro(fx, fy, fz, rho_f, vx_f, vy_f, vz_f);
	}

#else  // (C2F_LAGRANGE || C2F_TRILINEAR)
	// ---- Interpolation strategies (opt-in since the 2026-08-18 flip:
	// 3rd-order Lagrange; 2nd-order trilinear under C2F_TRILINEAR) ----
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

	// destination-group loop (up to 8 cells per thread, bounds-guarded)
	for (int idz = 0; idz < 2; idz++) {
		const idx z = gz0 + idz;
		if (z >= ghost_end_fine.z())
			continue;
		for (int idy = 0; idy < 2; idy++) {
			const idx y = gy0 + idy;
			if (y >= ghost_end_fine.y())
				continue;
			for (int idx_ = 0; idx_ < 2; idx_++) {
				const idx x = gx0 + idx_;
				if (x >= ghost_end_fine.x())
					continue;

				idx cnx[C2F_STENCIL], cny[C2F_STENCIL], cnz[C2F_STENCIL];
				dreal cwx[C2F_STENCIL], cwy[C2F_STENCIL], cwz[C2F_STENCIL];
				const int nnx = axis_stencil(x + fine_off.x(), coarse_off.x(), coarse_SD.X(), coarse_SD.indexer.template getOverlap<0>(), cnx, cwx);
				const int nny = axis_stencil(y + fine_off.y(), coarse_off.y(), coarse_SD.Y(), coarse_SD.indexer.template getOverlap<1>(), cny, cwy);
				const int nnz = axis_stencil(z + fine_off.z(), coarse_off.z(), coarse_SD.Z(), coarse_SD.indexer.template getOverlap<2>(), cnz, cwz);

				// interpolated macros and per-direction non-equilibrium sums
				dreal rho_f = 0, vx_f = 0, vy_f = 0, vz_f = 0;
				dreal f_neq[CONFIG::Q] = {};

				// visit the coarse cells of the interpolation stencil (up to
				// 4x4x4 cells whose centers surround the fine cell center)
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

	#ifdef C2F_EQ_ONLY
				// debug experiment: equilibrium-only fill (no non-equilibrium content)
				static_cast<void>(tau_fine);
				static_cast<void>(tau_coarse);
				for (int q = 0; q < CONFIG::Q; q++)
					store_fine_df(q, x, y, z, KS_F.f[q]);
	#else
				// volumetric rescaling, f_fine[q] = eq_q(rho_f,u_f) + (tau_f/tau_c)*f_neq[q]
				const dreal neq_scale = tau_fine / tau_coarse;
				for (int q = 0; q < CONFIG::Q; q++)
					store_fine_df(q, x, y, z, KS_F.f[q] + neq_scale * f_neq[q]);
	#endif

				write_fine_macro(x, y, z, rho_f, vx_f, vy_f, vz_f);
			}
		}
	}
#endif
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
 * overlap depth the caller allocated on the fine block's indexer (1 on
 * refinement-level blocks, see `LBM_BLOCK::storage_overlap`); cells failing
 * the test are skipped individually. This per-cell guard replaces the
 * former launch-extent clip, which evaluated the storability condition in
 * the wrong (origin-aligned) frame and silently dropped the max-side faces
 * and half of the patch extents of nested geometries. Production launches
 * cover the skin rectangles of each fine footprint (the ring launch was
 * removed, D.1): the min-side skin windows are clamped to the fine
 * interior (lo = 0, see below), so they never read ghost storage, while
 * the max-side windows include exactly the one ghost layer the
 * coarse-to-fine fill maintains. The filter's WIDER stencil (one extra
 * fine cell on each side of the subcell block per axis, see below) is
 * handled by the same shifted-window machinery as the coarse-to-fine
 * kernel: each per-axis window is shifted into the storable extent
 * (shortened if the extent is smaller than the window) and its Lagrange
 * weights are re-evaluated at runtime, so coupling cells adjacent to the
 * fine block boundary are still written (never skipped) as long as
 * their 8 subcells are storable.
 *
 * Algorithm per coarse cell:
 * 1. Read the post-kernel fine DFs of the filter stencil (orientation
 *    below).
 * 2. FILTER (MANDATORY, Lagrava et al. 2012): unresolved high-frequency
 *    fine-grid modes alias onto the coarse grid without a spatial filter,
 *    destabilizing the coarse solution. The default filter is the
 *    tensor-product 4-node-per-axis Lagrange projection of the fine DFs
 *    onto the coarse cell center `t = fx0 + 0.5` (fine indexer
 *    coordinates, same global-frame mapping as the subcell mapping): the
 *    nominal per-axis window `{fx0-1, ..., fx0+2}` covers the 2x2x2
 *    subcell block extended by one fine cell on each side (4x4x4 = 64
 *    fine cells in 3D), and centered windows yield the exact dyadic
 *    rationals {-1, 9, 9, -1}/16 per axis. The projection reproduces
 *    cubic fields at the coarse center exactly (the plain 1/8 box average
 *    only reproduces linear ones) and exactly preserves constant and
 *    linear fields on shifted windows as well, which is what the
 *    boundary-adjacent interface cells see; the highest fine-resolvable
 *    modes (odd/even checkerboard of the two subcell parities) are
 *    annihilated. Normalization: the per-axis Lagrange weights sum to
 *    one, so the tensor weight over the full 4x4x4 stencil sums to one
 *    and NO additional volume factor is applied anywhere (the sum-to-one
 *    weighted average IS the volumetric fine-to-coarse conversion, the
 *    same role the 1/8 factor plays in the box average). Global mass is
 *    conserved exactly: on a translation-invariant extent each fine cell
 *    contributes to the coarse values with total weight 1/2 per axis
 *    (1/8 in 3D), the same total as the box average. The original 1/8
 *    box average of the 8 subcells remains available as a compile-time
 *    fallback with `-DF2C_BOX_AVERAGE`.
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
 * Macros: the filtered macros are written to `dmacro` for every coupling
 * cell the launch covers (GEO_NOTHING frozen cells in production -- the
 * skin rectangles; the Defect-2 predicate below also admits
 * GEO_AMR_INTERFACE ring cells, exercised by the mock tests). The
 * collision-active ring cells' macros are computed by the main coarse
 * kernel only: since the ring fine-to-coarse launch was removed (gate B
 * ruling, D.1 hard-delete), no production launch covers ring cells
 * anymore.
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

#ifdef F2C_SCHONHERR
	// ---- Schönherr compact-moment transfer (Schönherr 2015 thesis, Sec.
	// 7.2, the sigma-form of the fine-to-coarse coupling; plan T14,
	// introduced as the TNL_LBM_F2C_STRATEGY=F2C_SCHONHERR opt-in at
	// commit 13 and the DEFAULT strategy since commit 15 / T17 -- the
	// Lagrava filter below is the F2C_LAGRAVA opt-out) ----
	// Sources: the destination cell's OWN 8 fine subcells (the cell-centered
	// mapping of this kernel's docstring; plan registration
	// F2C_SRC_ROW_OFFSET = 0, contract doc sec. 2 row (h)) -- no 4x4x4
	// filter window, no lo = 0 window clamp, no box/Lagrava averaging, and
	// NO Filippova-Hänel tau-rescale anywhere in this branch (the neq_scale
	// of the Lagrava opt-out branch below). Per source cell the five independent
	// second-order non-equilibrium moments are formed at the SOURCE (fine)
	// grid rate omega_s = 1/tau_fine and donated to the Eqs. 7.10-7.28
	// coefficient sums; the coarse destination sits at the window center
	// t = (0,0,0), so only the polynomial constants d_0/a_0/b_0/c_0
	// survive as the coarse macros (contract doc
	// docs/AMR-schonherr-ch7-target-contract.md, appendix A.2.3 census). The
	// velocity constants still inherit the Eq. 7.18/7.23/7.24 k-corrections
	// through the full code family of the velocity fits (the A.4-R1
	// decision, locked at commit 10 -- the print-family implementation is
	// rejected by the T10c lock). The averaged k-moments keep the
	// velocity-gradient corrections of Eqs. 7.29-7.33 (which cancel the
	// fitted-gradient summands of the Step-F forms exactly at t = 0 -- the
	// "avk retained" row), and the six second-order cumulants use the F2C
	// sigma-form sigma_{f->c} = 2 with the DESTINATION (coarse) grid rate
	// omega_d = 1/tau_coarse; third and higher central moments are zero
	// (the CM mode filter, cf. col_cum.h), and the coarse DFs are
	// reconstructed by the same cumulant back-transformation as the
	// coarse-to-fine branch. Under USE_GEIER_CUM_2017 the mode filter is
	// aligned to the collision's persistent modes as in the coarse-to-fine
	// branch: the own-subcell third-order cumulant averages are
	// transferred (the window-center trilinear fit at t = 0) and the
	// reconstruction dispatches to AMR_CM_BACKTRANSFORM_GEIER.

	// relaxation rates of the source (fine) and destination (coarse) grids
	const dreal omega_s = no1 / tau_fine;
	const dreal omega_d = no1 / tau_coarse;

	// Steps B-D (mirroring the coarse-to-fine branch): visit the own 8
	// subcells and accumulate the polynomial coefficient sums; the 2x2x2
	// subcell block is centered on the coarse destination, so the source
	// local coordinates are (xn,yn,zn) in {+-1/2}^3
	dreal sd0 = 0, sdx = 0, sdy = 0, sdz = 0, sdxy = 0, sdyz = 0, sdxz = 0, sdxyz = 0;
	dreal sa0 = 0, sax = 0, say = 0, saz = 0, saxx = 0, sayy = 0, sazz = 0, saxy = 0, sayz = 0, saxz = 0, saxyz = 0;
	dreal sb0 = 0, sbx = 0, sby = 0, sbz = 0, sbxx = 0, sbyy = 0, sbzz = 0, sbxy = 0, sbxz = 0, sbyz = 0, sbxyz = 0;
	dreal sc0 = 0, scx = 0, scy = 0, scz = 0, scxx = 0, scyy = 0, sczz = 0, scxy = 0, scyz = 0, scxz = 0, scxyz = 0;
	dreal sk_xy = 0, sk_yz = 0, sk_xz = 0, sk_xx_yy = 0, sk_xx_zz = 0;
	#ifdef USE_GEIER_CUM_2017
	// Geier 2017 mode consistency: the seven persistent third-order
	// cumulants summed over the own 8 subcells (AMR_CM_THIRD_MOMENTS
	// docstring; the n1o8 window-center average is applied below)
	dreal s3_120 = 0, s3_210 = 0, s3_201 = 0, s3_102 = 0, s3_012 = 0, s3_021 = 0, s3_111 = 0;
	#endif
	for (int ibz = 0; ibz < 2; ibz++) {
		const dreal zn = static_cast<dreal>(ibz) - n1o2;
		const idx fz = fz0 + ibz;
		for (int iby = 0; iby < 2; iby++) {
			const dreal yn = static_cast<dreal>(iby) - n1o2;
			const idx fy = fy0 + iby;
			for (int ibx = 0; ibx < 2; ibx++) {
				const dreal xn = static_cast<dreal>(ibx) - n1o2;
				const idx fx = fx0 + ibx;

				AMR_CM_MACROS_AND_KMOMENTS(read_fine_df, fx, fy, fz);

				AMR_CM_PI_NEQ;

				AMR_CM_KMOMENTS(omega_s);

				AMR_CM_FIT_ACCUMULATE;

	#ifdef USE_GEIER_CUM_2017
				AMR_CM_THIRD_MOMENTS;
				s3_120 += k_120;
				s3_210 += k_210;
				s3_201 += k_201;
				s3_102 += k_102;
				s3_012 += k_012;
				s3_021 += k_021;
				s3_111 += k_111;
	#endif
			}
		}
	}

	// Steps C-D: density and velocity polynomial coefficients and the
	// destination macros at t = (0,0,0) -- rho_f = d_0, vx_f = a_0,
	// vy_f = b_0, vz_f = c_0 (AMR_CM_FIT_COEFFICIENTS folds rho_f directly;
	// the tx/ty/tz lvalues exist for the Step E cross terms below, all of
	// which carry at least one of them and vanish)
	dreal rho_f = 0, vx_f = 0, vy_f = 0, vz_f = 0;
	const dreal tx = 0, ty = 0, tz = 0;
	AMR_CM_FIT_COEFFICIENTS(tx, ty, tz);
	AMR_CM_EVALUATE(tx, ty, tz);

	// Steps E-F: averaged second-order moments with the velocity-gradient
	// corrections (Eqs. 7.29-7.33) and the second-order cumulants at the
	// coarse destination (Eqs. 7.38-7.48); sigma_{f->c} = 2, omega_d is the
	// destination (coarse) grid rate -- AMR_CM_CORRECTED_CUMULANTS
	AMR_CM_CORRECTED_CUMULANTS(no2);

	#ifdef USE_GEIER_CUM_2017
	// Geier 2017 mode consistency: the seven persistent third-order
	// cumulants at the coarse destination = the window-center (1/8)
	// subcell averages -- the F2C trilinear nodal fit at t = 0, the same
	// reduction the density and velocity constants take here
	const dreal k120_f = n1o8 * s3_120;
	const dreal k210_f = n1o8 * s3_210;
	const dreal k201_f = n1o8 * s3_201;
	const dreal k102_f = n1o8 * s3_102;
	const dreal k012_f = n1o8 * s3_012;
	const dreal k021_f = n1o8 * s3_021;
	const dreal k111_f = n1o8 * s3_111;
	#endif

	// allowed-GEO predicate for the coarse-cell writes of this kernel
	// (Defect-2 fix), same semantics as the default branch below
	const auto map_val = coarse_SD.map(x, y, z);
	const bool is_coupling_cell = (map_val == BC::GEO_AMR_INTERFACE || map_val == BC::GEO_NOTHING);

	// Steps G-H: cumulant back-transformation into the coarse DFs (Geier
	// 2015 Eqs. 81-96) with the pattern-dependent store orientation of the
	// default branch (AB writes logical df_out natural; AA stores natural
	// for an even next substep, twisted for an odd one) --
	// AMR_CM_BACKTRANSFORM
	if (is_coupling_cell) {
		const auto store_coarse_df = [&coarse_SD, coarse_even_iter, x, y, z](int q, dreal f) -> void
		{
		// the back-transformation emits STORAGE-convention values directly:
		// physical DFs on D3Q27_COMMON, fhat = f - w_q on D3Q27_COMMON_WELL
	#ifdef AB_PATTERN
			static_cast<void>(coarse_even_iter);
			coarse_SD.df(df_out, q, x, y, z) = f;
	#elif defined(AA_PATTERN)
			if (coarse_even_iter)
				coarse_SD.df(df_cur, q, x, y, z) = f;
			else
				coarse_SD.df(df_cur, opposite_direction(q), x, y, z) = f;
	#endif
		};

	#ifdef USE_GEIER_CUM_2017
		AMR_CM_BACKTRANSFORM_GEIER(store_coarse_df);
	#else
		AMR_CM_BACKTRANSFORM(store_coarse_df);
	#endif

		// macros for coupling cells (GEO_AMR_INTERFACE ring or GEO_NOTHING
		// frozen hidden cells): authoritative coupling value for output
		coarse_SD.macro(MACRO::e_rho, x, y, z) = rho_f;
		coarse_SD.macro(MACRO::e_vx, x, y, z) = vx_f;
		coarse_SD.macro(MACRO::e_vy, x, y, z) = vy_f;
		coarse_SD.macro(MACRO::e_vz, x, y, z) = vz_f;
	}
#else
	// Lagrava spatial filter (see the kernel docstring) -- the
	// F2C_LAGRAVA opt-out branch (the default until the commit-15 / T17
	// flip); defining F2C_BOX_AVERAGE selects the original 1/8 box
	// average of the 8 subcells inside this branch
	#ifndef F2C_BOX_AVERAGE
	// tensor-product 4-node-per-axis Lagrange projection onto the coarse
	// cell center t = fx0 + 0.5 (fine indexer coordinates): the nominal
	// per-axis window {fx0-1, ..., fx0+2} covers the 2x2x2 subcell block
	// extended by one fine cell on each side (4x4x4 = 64 fine cells).
	// Near fine-block boundaries the window is shifted (and shortened if
	// the per-axis storage extent is smaller) so that all of its nodes are
	// valid storage indices -- the wider-stencil extension of the
	// storability guard, mirrored from the coarse-to-fine kernel; the
	// evaluation point stays FIXED at the coarse cell center, so the
	// shifted-window weights still reproduce cubic fields exactly (and
	// constant/linear fields on every window). The Lagrange weights are
	// evaluated at runtime in double precision and normalized to sum to
	// one; centered windows round to the exact dyadic rationals
	// {-1, 9, 9, -1}/16 per axis.
	constexpr int F2C_STENCIL = 4;
	const auto axis_weights = [](idx start, double t, int n, idx* nodes, dreal* weights) -> void
	{
		double w[F2C_STENCIL], wsum = 0;
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
	};
	const auto axis_window = [&axis_weights](idx f0, idx size_a, idx ov_a, idx* nodes, dreal* weights) -> int
	{
		const double t = static_cast<double>(f0) + 0.5;
		const int extent = static_cast<int>(size_a + 2 * ov_a);
		const int n = F2C_STENCIL < extent ? F2C_STENCIL : extent;
		// fine-interior lower bound (changes 2+3 of the redesign,
		// unconditional since the ring launch was removed in D.1): the only
		// F2C launches left are the skin launches, and those never read the
		// C2F-filled ghost — at footprint-edge cells the nominal window
		// {f0-1, ..., f0+2} shifts to start at 0 and the SAME shifted-window
		// machinery below re-evaluates the Lagrange weights at the fixed
		// evaluation point t = f0 + 0.5 (still cubic-exact there; the
		// shifted-edge conservation caveat of proposal §3 applies). The
		// upper bound keeps the overlap term, so a max-side edge window can
		// still include ghost nodes — this clamp is intentionally the LOWER
		// bound only.
		const idx lo = 0;
		const idx hi = size_a - 1 + ov_a - (n - 1);
		idx start = f0 - 1;
		start = start < lo ? lo : (start > hi ? hi : start);
		axis_weights(start, t, n, nodes, weights);
		return n;
	};
	idx fnx[F2C_STENCIL], fny[F2C_STENCIL], fnz[F2C_STENCIL];
	dreal fwx[F2C_STENCIL], fwy[F2C_STENCIL], fwz[F2C_STENCIL];
	const int nnx = axis_window(fx0, fine_SD.X(), ov.x(), fnx, fwx);
	const int nny = axis_window(fy0, fine_SD.Y(), ov.y(), fny, fwy);
	const int nnz = axis_window(fz0, fine_SD.Z(), ov.z(), fnz, fwz);

	// thesis Sec. 7.3 wall guard, the fine-grid mirror of the C2F guard
	// (Schönherr is silent on F2C-at-wall; applied symmetrically): a
	// per-axis filter window holding a physically-tagged (non-live) fine
	// cell at exactly one of its ends is shifted one cell away from that
	// end, with the Lagrange weights re-evaluated at the FIXED evaluation
	// point; live tags are GEO_FLUID, GEO_AMR_INTERFACE and GEO_NOTHING
	// (fine blocks carry no BC rows in v1 -- createAMRBlocks resets the
	// fine interior map to GEO_FLUID -- so the guard is inert on every
	// registration the framework builds today and covers future fine-level
	// BC tagging)
	bool t_lo[3] = {false, false, false}, t_hi[3] = {false, false, false};
	for (int bz = 0; bz < nnz; bz++)
		for (int by = 0; by < nny; by++)
			for (int bx = 0; bx < nnx; bx++) {
				const auto mapgi = fine_SD.map(fnx[bx], fny[by], fnz[bz]);
				if (BC::isFluid(mapgi) || mapgi == BC::GEO_AMR_INTERFACE || mapgi == BC::GEO_NOTHING)
					continue;
				t_lo[0] = t_lo[0] || bx == 0;
				t_hi[0] = t_hi[0] || bx == nnx - 1;
				t_lo[1] = t_lo[1] || by == 0;
				t_hi[1] = t_hi[1] || by == nny - 1;
				t_lo[2] = t_lo[2] || bz == 0;
				t_hi[2] = t_hi[2] || bz == nnz - 1;
			}
	const auto shift_off_bc = [&axis_weights](idx* nodes, dreal* weights, int n, double t, bool taint_lo, bool taint_hi, idx hi_start) -> void
	{
		if (n < 2 || taint_lo == taint_hi)
			return;	 // clean axis, degenerate window, or both ends tainted (no valid one-cell direction)
		idx start = taint_hi ? nodes[0] - 1 : nodes[0] + 1;
		start = start < 0 ? 0 : (start > hi_start ? hi_start : start);
		axis_weights(start, t, n, nodes, weights);
	};
	shift_off_bc(fnx, fwx, nnx, static_cast<double>(fx0) + 0.5, t_lo[0], t_hi[0], fine_SD.X() - 1 + ov.x() - (nnx - 1));
	shift_off_bc(fny, fwy, nny, static_cast<double>(fy0) + 0.5, t_lo[1], t_hi[1], fine_SD.Y() - 1 + ov.y() - (nny - 1));
	shift_off_bc(fnz, fwz, nnz, static_cast<double>(fz0) + 0.5, t_lo[2], t_hi[2], fine_SD.Z() - 1 + ov.z() - (nnz - 1));

	// per-direction weighted sums (the weights sum to one over the full
	// stencil, so no normalization factor is needed afterwards)
	dreal f_avg[CONFIG::Q] = {};
	for (int bz = 0; bz < nnz; bz++) {
		for (int by = 0; by < nny; by++) {
			for (int bx = 0; bx < nnx; bx++) {
				const dreal w = fwx[bx] * fwy[by] * fwz[bz];
				for (int q = 0; q < CONFIG::Q; q++)
					f_avg[q] += w * read_fine_df(q, fnx[bx], fny[by], fnz[bz]);
			}
		}
	}
	#else
	// plain per-direction arithmetic average of the 8 fine subcells covered
	// by this coarse cell. The (1/8) factor IS the volumetric
	// fine-to-coarse conversion -- no other volume factor.
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
	#endif

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

	// allowed-GEO predicate for the coarse-cell writes of this kernel
	// (Defect-2 fix): only GEO_AMR_INTERFACE ring cells and GEO_NOTHING
	// frozen hidden cells are coupling-owned storage; cells tagged with
	// any other GEO (boundary-condition tags, but also plain GEO_FLUID)
	// own their DFs and macros and must NOT be overwritten when a coupling
	// rectangle covers them. The map is read once before the DF-store loop
	// and the single predicate guards both the DF store and the macro
	// store below.
	const auto map_val = coarse_SD.map(x, y, z);
	const bool is_coupling_cell = (map_val == BC::GEO_AMR_INTERFACE || map_val == BC::GEO_NOTHING);

	// volumetric rescaling, f_coarse[q] = eq_q(rho_c,u_c) + (tau_c/tau_f)*f_neq[q]
	const dreal neq_scale = tau_coarse / tau_fine;
	if (is_coupling_cell) {
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
		coarse_SD.macro(MACRO::e_rho, x, y, z) = KS.rho;
		coarse_SD.macro(MACRO::e_vx, x, y, z) = KS.vx;
		coarse_SD.macro(MACRO::e_vy, x, y, z) = KS.vy;
		coarse_SD.macro(MACRO::e_vz, x, y, z) = KS.vz;
	}
#endif	// F2C_SCHONHERR
}
