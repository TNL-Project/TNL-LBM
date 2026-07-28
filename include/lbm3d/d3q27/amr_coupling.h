#pragma once

#include "lbm3d/defs.h"

/**
 * \brief AMR inter-level coupling: coarse-to-fine volumetric DF rescaling.
 *
 * Fills a rectangular extent of fine-level ghost cells at a refinement
 * interface with distribution functions (DFs) reconstructed from the coarse
 * level (Chen et al. 1998, Rohde et al. 2006, Guzik et al. 2014):
 *
 * 1. The macroscopic quantities (rho, u) of the 2x2x2 coarse cells whose
 *    centers bracket the fine cell center are trilinearly interpolated to
 *    the fine cell (cell-centered layout at a 2:1 ratio).
 * 2. The equilibrium part of each fine DF is re-evaluated at the
 *    interpolated macros (one `EQ::eq_*` call per direction, never shared).
 * 3. The non-equilibrium part is trilinearly interpolated per direction from
 *    the 8 coarse corner cells and rescaled by `tau_fine / tau_coarse`.
 *
 * With 2:1 refinement `nu_lb_f = 2 * nu_lb_c`, so `tau_fine != tau_coarse`
 * (tau = 3 * nu_lb + 0.5 per level): the non-equilibrium rescaling is NOT a
 * no-op. The caller computes both tau values from the per-level `lbmViscosity`
 * and passes them as kernel parameters. NO additional volume factor is
 * applied: each coarse-to-fine DF is a point density (the 1/8 volume
 * averaging belongs to the fine-to-coarse direction in todo 6).
 *
 * Cell-centered coordinate mapping (per axis, fine coord `x_f`):
 * the home coarse cell is `x_f / 2` (non-negative block-indexer coords, so
 * integer division floors). The fine cell center sits at +-1/4 of the home
 * cell's width from the home cell center, hence the two bracketing coarse
 * cell centers are {home-1, home} for even `x_f` and {home, home+1} for odd
 * `x_f`, unified as `home - 1 + (x_f & 1)`. The trilinear weight is 3/4 on
 * the home side and 1/4 on the far side of each axis (exact for linear
 * fields; all weights are binary fractions so constant fields are preserved
 * exactly).
 *
 * Ghost cells: the kernel fills EVERY cell in
 * [ghost_begin_fine, ghost_end_fine) -- the caller passes exactly the
 * interface ghost extent in fine block-indexer coordinates. Ghost cells have
 * no valid map/classification of their own (fine cells are GEO_FLUID in v1),
 * so no map filtering is done beyond the single `dmacro` guard below.
 *
 * Precondition: the 2x2x2 bracketing coarse cells of every fine ghost cell
 * must be valid storage indices in `coarse_SD` (including overlaps) -- the
 * caller must position interface ghosts at least one coarse cell inside the
 * coarse block's stored extent.
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
 * Writes: fine ghost DFs go to `fine_SD.df(df_cur, ...)` for BOTH patterns
 * (a single site, so no #ifdef here): `df_cur` is the array the next
 * fine-level streaming step pulls from, and ghosts are re-filled every
 * substep. For the AA pattern the values are written in natural orientation
 * (v1: fine ghosts are consumed as streamed-in data; the twisted-phase fine
 * handling is Wave 4/5 work).
 *
 * Macroscopic output: for cells tagged `GEO_AMR_INTERFACE` the interpolated
 * macros are written to `dmacro` (the main kernel's `outputMacro` would
 * otherwise read garbage KS there). In v1 fine ghosts are never
 * GEO_AMR_INTERFACE, so the guard is a no-op kept for safety.
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
	bool coarse_even_iter
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

	// bracketing coarse corners and trilinear weights (see the file docstring)
	const idx cx0 = x / 2 - 1 + (x & 1);
	const idx cy0 = y / 2 - 1 + (y & 1);
	const idx cz0 = z / 2 - 1 + (z & 1);
	const dreal wx0 = (x & 1) ? dreal(0.75) : dreal(0.25);
	const dreal wy0 = (y & 1) ? dreal(0.75) : dreal(0.25);
	const dreal wz0 = (z & 1) ? dreal(0.75) : dreal(0.25);

	// interpolated fine-level macros and per-direction non-equilibrium sums
	dreal rho_f = 0, vx_f = 0, vy_f = 0, vz_f = 0;
	dreal f_neq[CONFIG::Q] = {};

	// visit the 2x2x2 coarse cells whose centers bracket the fine cell center
	for (int bz = 0; bz < 2; bz++) {
		for (int by = 0; by < 2; by++) {
			for (int bx = 0; bx < 2; bx++) {
				const dreal w = (bx ? 1 - wx0 : wx0) * (by ? 1 - wy0 : wy0) * (bz ? 1 - wz0 : wz0);

				// corner DF state and its macroscopic quantities
				LBM_KS KS;
				for (int q = 0; q < CONFIG::Q; q++)
					KS.f[q] = read_coarse_df(q, cx0 + bx, cy0 + by, cz0 + bz);
				COLL::computeDensityAndVelocity(KS);
				rho_f += w * KS.rho;
				vx_f += w * KS.vx;
				vy_f += w * KS.vy;
				vz_f += w * KS.vz;

				// non-equilibrium at the corner: f_neq[q] = f[q] - eq_q(rho_c, u_c)
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
		fine_SD.df(df_cur, q, x, y, z) = KS_F.f[q] + neq_scale * f_neq[q];

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
 * kernel): coarse cell `(x,y,z)` covers the 8 fine cells
 * `(2*x+{0,1}, 2*y+{0,1}, 2*z+{0,1})`. One thread per coarse interface cell
 * in [coarse_begin, coarse_end); the caller derives
 * `coarse_end = coarse_begin + coarse_size` from the patch (same
 * begin/end range handling as `cudaAMR_CoarseToFine`) and must guarantee
 * that all 8 fine subcells of every coarse cell in the extent are valid
 * storage indices of `fine_SD` (the fine block must store the region
 * overlapping the coarse interface extent, and the two blocks' indexer
 * origins must correspond -- the same alignment assumption as todo 5).
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
 *   - AB: natural orientation -- the next coarse `streaming()` pull reads
 *     `df_cur[q]` at the same site, and AB always pulls in natural
 *     orientation. The caller passes `coarse_SD` in the rotation state
 *     the next coarse kernel launch will read (same caveat as the coarse
 *     read side in todo 5); `coarse_even_iter` is AA-only state.
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
 * written to `dmacro` -- the main coarse kernel skips collision on these
 * cells, so its `outputMacro` would otherwise write garbage KS there.
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
	bool coarse_even_iter
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
					f_avg[q] += read_fine_df(q, 2 * x + bx, 2 * y + by, 2 * z + bz);
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
		// AB: the next streaming() pulls df_cur[q] at the same site, natural
		// orientation (coarse_even_iter is AA-only state)
		static_cast<void>(coarse_even_iter);
		coarse_SD.df(df_cur, q, x, y, z) = f_coarse;
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

	// macros for GEO_AMR_INTERFACE cells (the main kernel's outputMacro must
	// not read garbage KS there)
	if (coarse_SD.map(x, y, z) == BC::GEO_AMR_INTERFACE) {
		coarse_SD.macro(MACRO::e_rho, x, y, z) = KS.rho;
		coarse_SD.macro(MACRO::e_vx, x, y, z) = KS.vx;
		coarse_SD.macro(MACRO::e_vy, x, y, z) = KS.vy;
		coarse_SD.macro(MACRO::e_vz, x, y, z) = KS.vz;
	}
}
