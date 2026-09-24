#define USE_GEIER_CUM_2017		 // use Geier 2017 Cummulant improvement A,B terms
#define USE_GEIER_CUM_ANTIALIAS	 // use antialiasing Dxu, Dyv, Dzw from Geier 2015/2017

#include <argparse/argparse.hpp>
#include <array>
#include <cmath>
#include <fstream>
#include <utility>

#include "lbm3d/core.h"
#include "lbm3d/amr_state.h"
#include "lbm3d/obstacles_lbm.h"

// AMR ball-in-channel simulation: a port of sim_NSE/sim_3.cu ("LBM simulation
// with ball in 3D") onto the State_AMR Berger-Colella subcycling driver of
// sim_AMR/sim_AMR_channel.cu. Same streaming/macro configuration and CLI
// surface as the other two sim_AMR binaries; the physics constants, boundary
// conditions and the level-0 obstacle stamping are sim_3's verbatim. The
// collision operator is the well-conditioned cumulant variant
// (D3Q27_CUM_WELL with D3Q27_EQ_INV_CUM_WELL: stored DFs are f - w with
// sum(w) = 1, see d3q27/common.h) instead of sim_3's plain D3Q27_CUM; the
// Geier 2017 A,B-term and antialiasing improvements are honored by both.
//
// Geometry (coarse level-0 cells, R = --resolution; sim_3's formulas
// byte-for-byte): domain H = 11*ball_diameter tall/deep (LBM_Y = LBM_Z =
// 32*R, PHYS_DL = H/(32R-2)), L = 2*H long (LBM_X = floor(L/PHYS_DL) + 2,
// 61 cells at R = 1), ball of diameter D centered at (2D, H/2, H/2) -- at
// R = 1 the stamped ball occupies the coarse cells x in {4,5,6}, y/z in
// {14,15,16} (lbmDrawSphere's truncating phys2lbmPoint + l2Norm test on a
// 1.364-cell radius). BCs are sim_3's: inflow x = 1 (GEO_INFLOW_MOMENT, formerly GEO_INFLOW_LEFT,
// constant profile via NSE_Data_ConstInflow), outflow x = X-2
// (GEO_OUTFLOW_RIGHT_INTERP), symmetry planes y/z at 1 and N-2, GEO_NOTHING
// on every edge plane (the A-A extra-layer idiom sim_3 already carries).
//
// Level-1 footprint "1 2R 12R 12R 21R 8R 8R" (coarse cells; R = 2:
// [4,46) x [24,40) x [24,40), i.e. the 12R cells below/band/above layout
// of the 32R-cell cross-section): y/z-centered on the ball, x-aligned
// with its near wake. The ball's continuous center (0.55 m) sits exactly
// on a level-0 cell face (y/z face index 16*R - 1), so the stamped ball
// column -- and everything the renderer draws from it -- falls on the
// side cells and renders half a coarse cell above the continuous center;
// the 8R-cell band [(12R), (20R)) has center (16R - 0.5) * DL, dead on
// the stamped column: in the map/render views the ball sits exactly in
// the band's center. The true band-ball mismatch is the remaining half
// coarse cell (one finest-level cell at level 1, inside the tolerated
// one-finest-cell bound), and the telescoping chain below shares the same
// inset on every face, so all derived levels share it. The stamped ball
// column (computed and logged at SimInit; 12 cells at R = 1: x in
// {5,6,7}, y/z in {15,16} by lbmDrawSphere's truncating phys2lbmPoint +
// cell-center l2Norm test) keeps depth >= 2 from every face -- the depth-2
// floor of the Schonherr-ch7 band registration (the depth-1 skin row is
// the F2C destination band and must stay frozen GEO_NOTHING).
//
// The x extent is sim's original ball-plus-near-wake slab: from the
// maximum upwind margin (origin 2, halo on the inflow plane at R = 1 --
// see below) past the ball (surface x in [4.09, 6.91]) to coarse x = 22,
// about 5.5 D behind the ball's back face. All six ring faces sit on
// collision-active fluid except the x-min face at R = 1 (see below).
//
// The x-min face is INFLOW-ADJACENT at R = 1: origin 2 puts the halo row on
// the inflow plane x = 1 (origin >= 2 is forced by the depth-2 ball
// constraint above and origin <= 2 by the halo staying inside the domain).
// markAMRInterface does not re-tag the BC plane, so that face couples
// through the coupling kernel's Sec. 7.3 wall guard: the nominal C2F
// source pair {c=-1, c=0} covers the inflow row and the guard steers the
// window onto the {ring c=0, skin c=1} pair exactly as on the channel's
// wall-attached z-min face (the SimInit map-pattern assertion exercises
// this path; the inflow BC itself is untouched). At R >= 2 the same
// R-scaled footprint leaves the halo on plain fluid (x = 2R-1 >= 3) and
// the face is an ordinary interior face.
//
// Ball stamping policy: level 0 carries the ball via sim_3's own call
// (lbmDrawSphere on nse; the coarse solve then computes the sim_3-scale
// ball-wake solution in the non-refined exterior, and the footprint's
// downstream ring bands are ball-wake-informed). The FINEST level carries
// the resolved ball via stampBallOnFineBlocks, a per-block replica of
// lbmDrawSphere's cell test under block.lat_local (obstacles_lbm.h binds
// the level-0 LBM lattice/map, so it cannot address fine blocks). Every
// intermediate parent treats the ball columns as plain fluid -- the
// windbreak precedent (locked item 1): parent's coupling bands and hidden
// frozen cores then never carry obstacle tags, which the map-pattern
// assertion and buildFineWallMasks require by construction.
//
// Nested mode (opt-in via --max-level 2..4; the default --max-level 1 shape
// above is unchanged): levels 2..max_level telescope inside the level-1
// anchor with inset = 3 parent-level cells on every face per hop (all six
// faces interior -- no wall-shared faces anywhere, so the whole chain sits
// in the V-suite's no-warning tier), derived by deriveAMRBallChain below
// (the same integer parent-cell rect arithmetic as amr_chain_solver.h; the
// derived spec is logged). The telescoping closes in symmetrically, so
// every derived level keeps the box centered on the ball in y/z; the chain
// budget exhausts at low R: the derivation hard-fails when a derived level
// no longer contains the stamped ball column with >= 2 of its own cells of
// margin per face (with the 8R-cell band the y/z margins are 6 own cells
// on every face at every hop at R = 1 and only grow with R, so R = 1
// admits levels 0..4 -- the full five-level chain; the SimInit V-suite
// and map-pattern guards remain the authoritative gate and the derivation
// names the failing face when the budget exhausts at even deeper
// levels). --max-level 0 is the uniform sim_3-equivalent reference run
// (all AMR machinery off, write3D_AMR no-ops).
//
// Physics (sim_3 verbatim): PHYS_VISCOSITY = 0.001 m^2/s, LBM_VISCOSITY =
// 0.001 (--lattice-viscosity overrides), PHYS_VELOCITY = Re * nu / D with
// --Re = 100 default (lbm inflow velocity ~0.0367 at R = 1), PHYS_DT =
// (nu_lb/nu_phys) * DL^2. Fine levels scale diffusively (nu doubles per
// level, see initLevelLattice); the uniform inflow drives the level-0
// GEO_INFLOW_MOMENT plane only. Default final time is sim_3's 30 s;
// --phys-final-time overrides. Deliberately NOT ported from sim_3: the 2D
// cuts (cut_X/cut_Y/cut_Z) and the OUT2D cadence (the base 2D-cut pipeline
// is level-0-only and the two existing AMR sims write no 2D output), and
// the custom BP5 output fields (density fluctuation, physical velocities)
// -- the AMR 3D output is the base class' VTKHDF OverlappingAMR writer
// (map + macros per level at the OUT3D cadence), the sim returns the
// AMR-house empty outputData pair. probe1 keeps sim_3's Reynolds/velocity
// log line at sim_3's PROBE1 period; PRINT/OUT3D use the sim_AMR house
// cadences and --out3d-iter-period mirrors the channel sim.

// Chain-mode selection (--chain {band,tight}; band is the default and the
// historical behavior, kept bit-identical): the band mode telescopes the
// fixed level-1 band by 3 parent cells per face per hop, so its footprint
// shrinks by only 6/(42R) of the band per hop while the cell count doubles
// per axis -- the budget grows ~ x7.7 per level for a x2 gain in points
// across D (the drag-crisis report's sec. 9 accounting). The tight mode
// (close-the-gap item #1, deriveAMRBallChainTight below) instead hugs the
// ball: level 1 keeps the same wake-aligned band anchor (the wake physics
// is unchanged), and each level L >= 2 is the stamped ball column inflated
// by the mandatory 2-own-cell ball clearance PLUS a tuning pad
// (--tight-pad, a fraction of the local ball diameter D_L, default 0.3),
// trimmed per axis by (i) the <= 2 D box-side cap measured in physical
// units INCLUDING the pad, (ii) the V7 telescoping gap against the parent
// rect -- the V-suite's own hard floor of 2 parent-level cells is enforced
// per axis, with the recommended 3 preferred and a per-axis relaxation
// logged when 3 is infeasible (the R = 1 x-faces at max_level 4 land in
// the accepted-with-warning 2-cell tier), and (iii) the V4 even-alignment
// (rect components even at every level). The intermediate-parents
// fluid-column rule is unchanged: only the finest level carries the
// stamped ball. The tight chain's per-level cost is (span - 2)^3 fine
// substeps with span <= 2 D_L, so the budget inverts the band mode's
// ratio: resolution gains cost ~ x16 per level but start from a ~ 1.4 D
// box instead of a 3.9-5.4 D band.
//
// Nested-footprint chain derivation of the --max-level 2..4 mode (mirrors
// amr_chain_solver.h's integer rect arithmetic, with an all-interior
// chain): rect_L = the footprint rectangle in level-L cells; rect_1 is the
// level-1 anchor doubled into level-1 cells; each hop insets the parent
// rect by `inset` parent-level cells on EVERY face and doubles into the
// child frame (rect_L = 2 * inset(rect_{L-1})). The region-file lines
// follow from createAMRBlocks' parent-frame conversion
// (amrParentFrameOrigin):
//
//     origin = (rect_L.lo / 2) * 2^(L-1),  size = (rect_L.span / 2) * 2^(L-1)
//
// per component (rect components are even at every level, so the /2 is
// exact and the product is a multiple of 2^(L-1): the V4 alignment holds
// automatically). The solver's own checks are the span floor (>= 3
// parent-level cells per axis) and the ball containment of every derived
// level; the authoritative guard remains createAMRBlocks' full V-suite at
// SimInit, which throws on any violation.
struct AMRBallChainLevel
{
	int level = 0;
	std::array<int, 3> origin{};  // region-file coordinates (the level-0 convention of AMRChainLevelGeometry)
	std::array<int, 3> size{};	  // region-file footprint size
};

struct AMRBallChain
{
	std::vector<AMRBallChainLevel> levels;	// one entry per level 1..max_level
	std::string region_config;				// parseAMRConfig-ready region spec
};

inline AMRBallChain deriveAMRBallChain(
	int R,
	int max_level,
	const std::array<double, 3>& ball_col_min_l0,  // inclusive lower cell of the stamped ball column on the level-0 lattice
	const std::array<double, 3>& ball_col_max_l0   // inclusive upper cell
)
{
	// telescoping inset per face per hop, in parent-level cells (3 = the
	// no-warning tier of the V7/V8 gap rules; no wall-shared faces exist in
	// this sim, so every face carries the inset)
	constexpr int inset = 3;
	// own-cell clearance demanded between the ball surface and every derived
	// footprint face: keeps the stamped ball out of the 2-row coupling band
	constexpr int ball_margin = 2;

	if (R < 1) {
		const std::string message = fmt::format("AMR ball chain: resolution R = {} is below 1", R);
		spdlog::error("{}", message);
		throw std::runtime_error(message);
	}
	if (max_level < 1 || max_level > 4) {
		const std::string message =
			fmt::format("AMR ball chain: max_level = {} is outside the supported range 1..4 (0 selects the uniform reference run)", max_level);
		spdlog::error("{}", message);
		throw std::runtime_error(message);
	}

	struct Rect
	{
		std::array<int, 3> lo;
		std::array<int, 3> span;
	};

	// the anchor: x is the original ball-wake slab doubled into level-1
	// cells; y/z are "1 2R 12R 12R 21R 8R 8R" doubled into level-1 cells:
	// an 8R-cell band whose center (16R - 0.5) * DL sits half a coarse
	// cell above the ball's continuous center -- dead on the stamped ball
	// column, which is quantized onto the cell-face boundary next to the
	// continuous center (see the file header for the exact arithmetic)
	std::vector<Rect> rects;
	rects.push_back(Rect{{0, 0, 0}, {0, 0, 0}});  // level 0 unused (indexing by level)
	rects.push_back(Rect{{4 * R, 24 * R, 24 * R}, {42 * R, 16 * R, 16 * R}});

	for (int L = 2; L <= max_level; L++) {
		const Rect& parent = rects[L - 1];
		Rect child;
		for (int a = 0; a < 3; a++) {
			child.lo[a] = parent.lo[a] + inset;
			child.span[a] = parent.span[a] - 2 * inset;
			if (child.span[a] < 3) {
				const std::string message = fmt::format(
					"AMR ball chain: level-{} footprint span below the 3-parent-cell minimum on axis {} ({} < 3 at R = {}): "
					"the telescoping budget is exhausted; do not nest this deep",
					L,
					char('x' + a),
					child.span[a],
					R
				);
				spdlog::error("{}", message);
				throw std::runtime_error(message);
			}
		}
		// double the child rect into level-L cells
		for (int a = 0; a < 3; a++) {
			child.lo[a] *= 2;
			child.span[a] *= 2;
		}
		rects.push_back(child);
	}

	// ball containment: every derived level must keep the STAMPED ball
	// column at least `ball_margin` of its own cells inside the rect on
	// every face (stamped cells, not the continuous surface -- the
	// registration map-pattern gate binds MAP cells, which sit up to a
	// cell outside the continuous surface AABB on the truncating
	// rounding side). Stamped level-0 cell i covers [i, i+1) of the
	// level-0 lattice, i.e. [i, i+1) * scale level-L cells
	for (int L = 1; L <= max_level; L++) {
		const Rect& rect = rects[L];
		const double scale = 1 << L;  // level-L cells per level-0 cell
		for (int a = 0; a < 3; a++) {
			const double col_min = ball_col_min_l0[a] * scale;
			const double col_max = (ball_col_max_l0[a] + 1.0) * scale;
			if (rect.lo[a] > col_min - ball_margin || rect.lo[a] + rect.span[a] < col_max + ball_margin) {
				const std::string message = fmt::format(
					"AMR ball chain: level-{} footprint [{},{},{}] + [{},{},{}] no longer contains the stamped ball column (cells "
					"[{}..{}] on axis {}) with the {}-cell clearance at R = {}: the telescoping budget is exhausted; raise "
					"--resolution or lower --max-level",
					L,
					rect.lo[0],
					rect.lo[1],
					rect.lo[2],
					rect.span[0],
					rect.span[1],
					rect.span[2],
					col_min,
					col_max,
					char('x' + a),
					ball_margin,
					R
				);
				spdlog::error("{}", message);
				throw std::runtime_error(message);
			}
		}
	}

	AMRBallChain chain;
	for (int L = 1; L <= max_level; L++) {
		const Rect& rect = rects[L];
		AMRBallChainLevel geometry;
		geometry.level = L;
		for (int a = 0; a < 3; a++) {
			// createAMRBlocks' parent-frame conversion inverted: footprint
			// rect in level-L cells is 2 * (value >> (L-1)) per component
			geometry.origin[a] = (rect.lo[a] / 2) << (L - 1);
			geometry.size[a] = (rect.span[a] / 2) << (L - 1);
		}
		chain.levels.push_back(geometry);
		chain.region_config += fmt::format(
			"{} {} {} {} {} {} {}",
			L,
			geometry.origin[0],
			geometry.origin[1],
			geometry.origin[2],
			geometry.size[0],
			geometry.size[1],
			geometry.size[2]
		);
		if (L < max_level)
			chain.region_config += "\n";
	}

	spdlog::info(
		"AMR ball chain: derived {} level(s) on top of the y/z-centered ball-wake anchor (R = {}, inset = {} parent-level cells on every "
		"face, no wall-shared faces; ball contained with >= {} own-cell margins)",
		max_level,
		R,
		inset,
		ball_margin
	);
	for (const AMRBallChainLevel& geometry : chain.levels)
		spdlog::info(
			"AMR ball chain: level {} origin [{},{},{}] size [{},{},{}] (level-0 coordinates)",
			geometry.level,
			geometry.origin[0],
			geometry.origin[1],
			geometry.origin[2],
			geometry.size[0],
			geometry.size[1],
			geometry.size[2]
		);
	spdlog::info("AMR ball chain region spec (reproduces this configuration):\n{}", chain.region_config);

	return chain;
}

// Tight-nested-patches chain derivation (--chain tight; close-the-gap item
// #1 of the drag-crisis report's sec. 10): level 1 keeps the band mode's
// y/z-centered ball-wake anchor unchanged, levels 2..max_level hug the
// ball. Per axis and level the box is the stamped ball column inflated by
// (ball_margin + pad_L) own cells, with pad_L = round(tight_pad_fraction *
// D_L) and D_L the ball diameter in level-L cells, then trimmed by a
// walk-down over the integer margin m until all of the following hold
// (mirroring createAMRBlocks' own tiering, enforced HERE so violations are
// attributable to the derivation instead of a SimInit throw):
//   - containment: the stamped column keeps ball_margin = 2 own cells of
//     clearance on every face (the depth-1 skin row is the F2C destination
//     band and must stay frozen GEO_NOTHING -- the file header's rule);
//   - 2 D box-side cap: span <= 2 * D_L (the tight-patch target, measured
//     in physical units INCLUDING the pad);
//   - V7 telescoping gap against the parent rect >= 2 parent-level cells
//     on every face (the V-suite hard floor); the recommended 3 is tried
//     first per axis and a relaxation to 2 is logged on the axis that
//     needs it (gap 3 infeasible under the 2 D cap at R = 1 x-faces);
//   - even lo/span (the V4-multiple alignment of the emitted region line);
//   - footprint span >= 3 parent cells (the V-suite gs floor, always
//     comfortably satisfied at this geometry but checked for completeness).
// A margin that walks below ball_margin hard-fails with the axis and the
// binding constraint named. The greedy per-level maximum margin is
// downstream-optimal: the child gap is m_parent - m/2, so a larger parent
// margin only widens the child's feasible range.
struct AMRBallChainRect
{
	std::array<int, 3> lo;
	std::array<int, 3> span;
};

inline AMRBallChain deriveAMRBallChainTight(
	int R,
	int max_level,
	const std::array<double, 3>& ball_col_min_l0,  // inclusive lower cell of the stamped ball column on the level-0 lattice
	const std::array<double, 3>& ball_col_max_l0,  // inclusive upper cell
	double ball_diameter_l0_cells,				   // ball diameter in level-0 cells (ball_diameter / PHYS_DL)
	double tight_pad_fraction					   // pad as a fraction of the local ball diameter D_L
)
{
	// own-cell clearance demanded between the ball surface and every derived
	// footprint face (identical to the band mode's rule)
	constexpr int ball_margin = 2;
	// face names in the same (min, max) per-axis convention as the V-suite messages
	static constexpr const char* face_names[6] = {"x-min", "x-max", "y-min", "y-max", "z-min", "z-max"};

	if (R < 1) {
		const std::string message = fmt::format("AMR ball tight chain: resolution R = {} is below 1", R);
		spdlog::error("{}", message);
		throw std::runtime_error(message);
	}
	if (max_level < 1 || max_level > 4) {
		const std::string message = fmt::format(
			"AMR ball tight chain: max_level = {} is outside the supported range 1..4 (0 selects the uniform reference run)", max_level
		);
		spdlog::error("{}", message);
		throw std::runtime_error(message);
	}

	std::vector<AMRBallChainRect> rects;
	rects.push_back(AMRBallChainRect{{0, 0, 0}, {0, 0, 0}});  // level 0 unused (indexing by level)
	// the level-1 anchor is the band mode's rect, bit-identical: the wake
	// physics of the ball-plus-near-wake slab is unchanged in tight mode
	rects.push_back(AMRBallChainRect{{4 * R, 24 * R, 24 * R}, {42 * R, 16 * R, 16 * R}});

	// per-level derived geometry for logging (one entry per level >= 2)
	struct TightLevelInfo
	{
		std::array<int, 3> margin{};	// own cells per face beyond the stamped column
		std::array<int, 6> gap{};		// telescoping gap to the parent rect, per face, parent-level cells
		std::array<int, 3> gap_floor{}; // the floor the axis enforced (3 recommended, 2 = the V-suite's warn tier)
	};
	std::vector<TightLevelInfo> infos(max_level + 1);

	if (max_level >= 2) {
		rects.resize(max_level + 1);
		for (int a = 0; a < 3; a++) {
			// per-axis telescoping floor: try the recommended 3 parent
			// cells first; relax to the V-suite's hard floor 2 with a warn
			// line when the axis cannot keep the ball clearance under 3
			bool axis_done = false;
			std::string axis_bind;
			for (int floor_gap = 3; floor_gap >= 2 && ! axis_done; floor_gap--) {
				// parent rect of the current walk (the anchor for L = 2),
				// doubles in the level-L frame per hop afterwards
				int parent_lo = rects[1].lo[a];
				int parent_span = rects[1].span[a];
				bool feasible = true;
				std::vector<TightLevelInfo> trial_infos(max_level + 1);
				std::vector<AMRBallChainRect> trial_rects(max_level + 1);
				for (int L = 2; L <= max_level && feasible; L++) {
					const int scale = 1 << L;
					const int col_lo = (int) ball_col_min_l0[a] * scale;
					const int col_hi = ((int) ball_col_max_l0[a] + 1) * scale;	// exclusive
					const double D_L = ball_diameter_l0_cells * scale;
					// desired margin: the mandatory clearance plus the pad
					// (a fraction of the local ball diameter)
					bool ok = false;
					for (int m = ball_margin + (int) std::lround(tight_pad_fraction * D_L); m >= ball_margin && ! ok; m--) {
						// integerize: even lo/span (V4); parity expands the
						// box outward, then the same checks re-run
						int lo = col_lo - m;
						if (lo & 1)
							lo -= 1;
						int hi = col_hi + m;
						if (hi & 1)
							hi += 1;
						const int span = hi - lo;
						const int gap_min = lo / 2 - parent_lo;
						const int gap_max = parent_lo + parent_span - (lo + span) / 2;
						const char* violated = nullptr;
						int violated_face = -1;
						if (span < 6)
							violated = "the 3-parent-cell footprint floor";
						else if (span > 2.0 * D_L)
							violated = "the 2-D box-side cap";
						else if (gap_min < floor_gap) {
							violated = "the V7 telescoping floor";
							violated_face = 2 * a;
						}
						else if (gap_max < floor_gap) {
							violated = "the V7 telescoping floor";
							violated_face = 2 * a + 1;
						}
						if (violated != nullptr) {
							axis_bind = fmt::format(
								"level-{} margin below the {}-cell ball clearance on axis {} under {}{} at R = {}",
								L,
								ball_margin,
								char('x' + a),
								violated,
								violated_face >= 0 ? fmt::format(" ({} face)", face_names[violated_face]) : std::string{},
								R
							);
							continue;
						}
						trial_infos[L].margin[a] = m;
						trial_infos[L].gap[2 * a] = gap_min;
						trial_infos[L].gap[2 * a + 1] = gap_max;
						trial_infos[L].gap_floor[a] = floor_gap;
						trial_rects[L].lo[a] = lo;
						trial_rects[L].span[a] = span;
						// the derived rect is in level-L cells, which IS
						// the parent frame of the next hop (the gap check
						// above still halves the child into this frame)
						parent_lo = lo;
						parent_span = span;
						ok = true;
					}
					feasible = ok;
				}
				if (feasible) {
					for (int L = 2; L <= max_level; L++) {
						rects[L].lo[a] = trial_rects[L].lo[a];
						rects[L].span[a] = trial_rects[L].span[a];
						infos[L].margin[a] = trial_infos[L].margin[a];
						infos[L].gap[2 * a] = trial_infos[L].gap[2 * a];
						infos[L].gap[2 * a + 1] = trial_infos[L].gap[2 * a + 1];
						infos[L].gap_floor[a] = floor_gap;
					}
					if (floor_gap == 2)
						spdlog::warn(
							"AMR ball tight chain: axis {} relaxed to the V7 hard floor (2 parent cells, the accepted-with-warning "
							"tier: the parent's fine-to-coarse transfer windows will read coupling-authored ring/skin cells; the "
							"recommended 3 cannot keep the {}-cell ball clearance under the 2-D cap at this R/depth)",
							char('x' + a),
							ball_margin
						);
					axis_done = true;
				}
			}
			if (! axis_done) {
				const std::string message = fmt::format(
					"AMR ball tight chain: {}: the tight-box budget is exhausted; raise --resolution, lower --max-level, or reduce "
					"--tight-pad",
					axis_bind
				);
				spdlog::error("{}", message);
				throw std::runtime_error(message);
			}
		}
	}

	// ball containment guard of the final integer rects (the host replica's
	// stamped column; stamped cells, not the continuous surface -- the
	// map-pattern gate binds MAP cells), mirroring the band mode's check
	for (int L = 1; L <= max_level; L++) {
		const AMRBallChainRect& rect = rects[L];
		const double scale = 1 << L;  // level-L cells per level-0 cell
		for (int a = 0; a < 3; a++) {
			const double col_min = ball_col_min_l0[a] * scale;
			const double col_max = (ball_col_max_l0[a] + 1.0) * scale;
			if (rect.lo[a] > col_min - ball_margin || rect.lo[a] + rect.span[a] < col_max + ball_margin) {
				const std::string message = fmt::format(
					"AMR ball tight chain: level-{} footprint [{},{},{}] + [{},{},{}] no longer contains the stamped ball column (cells "
					"[{}..{}] on axis {}) with the {}-cell clearance at R = {}: the tight-box budget is exhausted; raise "
					"--resolution, lower --max-level, or reduce --tight-pad",
					L,
					rect.lo[0],
					rect.lo[1],
					rect.lo[2],
					rect.span[0],
					rect.span[1],
					rect.span[2],
					col_min,
					col_max,
					char('x' + a),
					ball_margin,
					R
				);
				spdlog::error("{}", message);
				throw std::runtime_error(message);
			}
		}
	}

	AMRBallChain chain;
	for (int L = 1; L <= max_level; L++) {
		const AMRBallChainRect& rect = rects[L];
		AMRBallChainLevel geometry;
		geometry.level = L;
		for (int a = 0; a < 3; a++) {
			// createAMRBlocks' parent-frame conversion inverted: footprint
			// rect in level-L cells is 2 * (value >> (L-1)) per component
			// (rect components are even at every level by construction)
			geometry.origin[a] = (rect.lo[a] / 2) << (L - 1);
			geometry.size[a] = (rect.span[a] / 2) << (L - 1);
		}
		chain.levels.push_back(geometry);
		chain.region_config += fmt::format(
			"{} {} {} {} {} {} {}",
			L,
			geometry.origin[0],
			geometry.origin[1],
			geometry.origin[2],
			geometry.size[0],
			geometry.size[1],
			geometry.size[2]
		);
		if (L < max_level)
			chain.region_config += "\n";
	}

	spdlog::info(
		"AMR ball tight chain: derived {} level(s) on top of the y/z-centered ball-wake anchor (R = {}, pad {:.2f} x D_L beyond the "
		"{}-cell clearance, <= 2 D box sides, all-interior faces)",
		max_level,
		R,
		tight_pad_fraction,
		ball_margin
	);
	// per-level cost accounting: a level's interior holds span - 2 cells per
	// axis (the re-anchored inset) and runs 2^L substeps per coarse step
	double total_updates = 0;
	long total_cells = 0;
	for (int L = 1; L <= max_level; L++) {
		const AMRBallChainRect& rect = rects[L];
		long cells = 1;
		for (int a = 0; a < 3; a++)
			cells *= (long) rect.span[a] - 2;
		const double updates = (double) cells * (1 << L);
		total_updates += updates;
		total_cells += cells;
		if (L >= 2) {
			const double D_L = ball_diameter_l0_cells * (1 << L);
			spdlog::info(
				"AMR ball tight chain: level {} rect [{},{},{}] + [{},{},{}] (spans {:.2f} x {:.2f} x {:.2f} D, margins [{},{},{}] own "
				"cells, gaps [{},{},{},{},{},{}] parent cells, floors [{},{},{}])",
				L,
				rect.lo[0],
				rect.lo[1],
				rect.lo[2],
				rect.span[0],
				rect.span[1],
				rect.span[2],
				rect.span[0] / D_L,
				rect.span[1] / D_L,
				rect.span[2] / D_L,
				infos[L].margin[0],
				infos[L].margin[1],
				infos[L].margin[2],
				infos[L].gap[0],
				infos[L].gap[1],
				infos[L].gap[2],
				infos[L].gap[3],
				infos[L].gap[4],
				infos[L].gap[5],
				infos[L].gap_floor[0],
				infos[L].gap_floor[1],
				infos[L].gap_floor[2]
			);
		}
		spdlog::info(
			"AMR ball tight chain: level {} interior {} cells x {} substeps = {:.4g} cell updates per coarse step",
			L,
			cells,
			1 << L,
			updates
		);
	}
	spdlog::info(
		"AMR ball tight chain: fine-level total {} cells, {:.4g} cell updates per coarse step (level-0 block excluded)",
		total_cells,
		total_updates
	);
	for (const AMRBallChainLevel& geometry : chain.levels)
		spdlog::info(
			"AMR ball tight chain: level {} origin [{},{},{}] size [{},{},{}] (level-0 coordinates)",
			geometry.level,
			geometry.origin[0],
			geometry.origin[1],
			geometry.origin[2],
			geometry.size[0],
			geometry.size[1],
			geometry.size[2]
		);
	spdlog::info("AMR ball tight chain region spec (reproduces this configuration):\n{}", chain.region_config);

	return chain;
}

// ---------------------------------------------------------------------------
// Momentum-exchange drag probe on the staircased ball wall (uncommitted study
// diagnostic). The wall BC is on-node full-way bounce-back (GEO_WALL cells swap
// opposite DF pairs after pulling), so the momentum deposited on the solid per
// iteration equals, per fluid cell x and direction q whose neighbor x + c_q is
// wall, twice e_q * f^*_q(x), where f^*_q(x) is the post-collision population
// of x living in the pattern's output frame (own-site store in A-B pull). With
// the well-conditioned storage f^*_q - w_q equals the stored value directly,
// which makes the probe exactly zero at rest; identity storage would carry the
// closed-link-set equilibrium bias, hence the explicit weight subtraction in
// that branch. The kernel below sums the FLUID-side deviations over a window
// around the ball (channel y/z walls of uniform max_level == 0 runs are cut
// away by the window); probe2() then evaluates
//
//     Cd = 2 * sum_dev / (0.5 * u_lb^2 * A_lb),   A_lb = pi/4 * (D/dx_L)^2
//
// with u_lb/A_lb in the units of the level the probe runs on (diffusive
// scaling: u_lb is level-independent). probe2 fires at the cnt[PROBE2]
// cadence, i.e. once per coarse iteration block; each sample is the per-step
// momentum transfer of one finest-level state.
struct BallForceAccumulator
{
	double fx = 0;
	double fy = 0;
	double fz = 0;
	double links = 0;
};

template <typename NSE>
__global__ void
cudaBallForceKernel(typename NSE::DATA SD, typename NSE::TRAITS::idx3d begin, typename NSE::TRAITS::idx3d end, BallForceAccumulator* out)
{
	using idx = typename NSE::TRAITS::idx;
	using dreal = typename NSE::TRAITS::dreal;

	const idx x = begin.x() + blockIdx.x * blockDim.x + threadIdx.x;
	const idx y = begin.y() + blockIdx.y * blockDim.y + threadIdx.y;
	const idx z = begin.z() + blockIdx.z * blockDim.z + threadIdx.z;
	if (x >= end.x() || y >= end.y() || z >= end.z())
		return;
	if (SD.map(x, y, z) != NSE::BC::GEO_FLUID)
		return;

	double fx = 0, fy = 0, fz = 0, links = 0;
#pragma unroll
	for (int q = 1; q < 27; q++) {	// skip zzz (zero velocity)
		const int cx = dir27_cx(q);
		const int cy = dir27_cy(q);
		const int cz = dir27_cz(q);
		if (SD.map(x + cx, y + cy, z + cz) != NSE::BC::GEO_WALL)
			continue;
		dreal dev = SD.df(NSE::STREAMING::output_df, q, x, y, z);
		if constexpr (! NSE::COLL::is_well_conditioned)
			dev -= NSE::COLL::weight(q);
		fx += cx * dev;
		fy += cy * dev;
		fz += cz * dev;
		links += 1;
	}
	if (links > 0) {
		atomicAdd(&out->fx, fx);
		atomicAdd(&out->fy, fy);
		atomicAdd(&out->fz, fz);
		atomicAdd(&out->links, links);
	}
}

template <typename NSE>
struct StateLocal_AMR_Ball : State_AMR<NSE>
{
	using TRAITS = typename NSE::TRAITS;
	using BC = typename NSE::BC;
	using BLOCK = LBM_BLOCK<NSE>;

	using State<NSE>::nse;

	using idx = typename TRAITS::idx;
	using idx3d = typename TRAITS::idx3d;
	using real = typename TRAITS::real;
	using dreal = typename TRAITS::dreal;
	using point_t = typename TRAITS::point_t;
	using bool3d = typename TRAITS::bool3d;
	using lat_t = Lattice<3, real, idx>;

	// problem parameters (set before execute(), consumed by the init/BC hooks)
	dreal phys_inflow_velocity = 0;	 // [m/s] uniform inflow velocity
	dreal lbm_inflow_vx = 0;		 // level-0 lattice inflow velocity (probe log)
	real ball_diameter = 0;			 // [m]
	point_t ball_c;					 // [m]

	// drag probe (uncommitted study diagnostic; see above): one
	// momentum-exchange sample per cnt[PROBE2] tick, appended to a CSV
	double drag_probe_period = -1;	// [s] <= 0: disabled
	std::string drag_csv_path = "drag_probe.csv";
	BallForceAccumulator* d_force = nullptr;
	std::ofstream drag_csv;
	long drag_sample_count = 0;
	double drag_cd_ema = 0;
	double drag_cd_min = 0;
	double drag_cd_max = 0;
	int drag_nan_events = 0;

	StateLocal_AMR_Ball(const std::string& id, const TNL::MPI::Comm& communicator, lat_t lat, const std::string& adiosConfigPath, int max_level = 1)
	: State_AMR<NSE>(
		  id,
		  communicator,
		  std::move(lat),
		  adiosConfigPath,
		  // channel around the ball: non-periodic everywhere -- the inflow/
		  // outflow planes and the symmetry planes are explicit BC cells set
		  // in setupBoundaries() below (sim_3's arrangement)
		  bool3d{false, false, false},
		  max_level
	  )
	{}

	// Per-level replica of lbmDrawSphere's cell test (obstacles_lbm.h binds
	// the level-0 LBM object's lattice and map, so it cannot address fine
	// blocks). lat_local.phys2lbmPoint yields the block's LOCAL indexer
	// coordinates (createAMRBlocks shifted lat_local.physOrigin by the
	// block offset); the block's hmap is global-indexed, so the stamp hits
	// offset + local, clipped to the interior -- the ball sits deep inside
	// the footprint by construction and must never touch the ghost band.
	void stampBallOnFineBlocks()
	{
		for (auto& block : nse.blocks) {
			if (block.level != nse.max_level)
				continue;
			const idx3d c = block.lat_local.phys2lbmPoint(ball_c);
			const real r = ball_diameter * 0.5 / block.lat_local.physDl;
			const idx range = ceil(r) + 1;
			idx n_ball = 0;
			for (idx py = c.y() - range; py <= c.y() + range; py++)
				for (idx pz = c.z() - range; pz <= c.z() + range; pz++)
					for (idx px = c.x() - range; px <= c.x() + range; px++) {
						const idx3d p{px, py, pz};
						if (TNL::l2Norm(p - c) >= r)
							continue;
						if (px < 0 || py < 0 || pz < 0 || px >= block.local.x() || py >= block.local.y() || pz >= block.local.z())
							continue;
						block.hmap(block.offset.x() + px, block.offset.y() + py, block.offset.z() + pz) = BC::GEO_WALL;
						n_ball++;
					}
			spdlog::info("fine block {} (level {}): stamped {} ball cells", block.id, block.level, n_ball);
		}
	}

	// boundary map on the level-0 lattice: sim_NSE/sim_3.cu's
	// setupBoundaries verbatim (the BC planes sit at the fixed indices 1 /
	// N-2 at every R), then the ball on level 0 via the same stamping call
	// as sim_3, then the resolved ball on the finest level (the header
	// comment). markAMRInterface runs later and only re-tags GEO_FLUID
	// cells, so the ball walls survive under the footprint
	void setupBoundaries() override
	{
		nse.setBoundaryX(1, BC::GEO_INFLOW_MOMENT);								 // left
		nse.setBoundaryX(nse.lat.global.x() - 2, BC::GEO_OUTFLOW_RIGHT_INTERP);	 // right

		//nse.setBoundaryY(1, BC::GEO_SYMMETRY);						 // front
		//nse.setBoundaryY(nse.lat.global.y() - 2, BC::GEO_SYMMETRY);	 // back
		//nse.setBoundaryZ(1, BC::GEO_SYMMETRY);						 // bottom
		//nse.setBoundaryZ(nse.lat.global.z() - 2, BC::GEO_SYMMETRY);	 // top

		nse.setBoundaryY(1, BC::GEO_WALL);						 // front
		nse.setBoundaryY(nse.lat.global.y() - 2, BC::GEO_WALL);	 // back
		nse.setBoundaryZ(1, BC::GEO_WALL);						 // bottom
		nse.setBoundaryZ(nse.lat.global.z() - 2, BC::GEO_WALL);	 // top

		// extra layer needed due to A-A pattern
		nse.setBoundaryX(0, BC::GEO_NOTHING);						// left
		nse.setBoundaryX(nse.lat.global.x() - 1, BC::GEO_NOTHING);	// right
		nse.setBoundaryZ(0, BC::GEO_NOTHING);						// bottom
		nse.setBoundaryZ(nse.lat.global.z() - 1, BC::GEO_NOTHING);	// top
		nse.setBoundaryY(0, BC::GEO_NOTHING);						// front
		nse.setBoundaryY(nse.lat.global.y() - 1, BC::GEO_NOTHING);	// back

		lbmDrawSphere(nse, BC::GEO_WALL, ball_c, ball_diameter * 0.5);

		stampBallOnFineBlocks();
	}

	// uniform-flow initial condition at rest: rho = 1, u = 0 on all blocks;
	// the inflow BC then develops the flow around the ball from t = 0 (the
	// same developing regime sim_3 runs). The engine initializes the FULL
	// stored extent (including the ghost band) so that the ghost rows hold
	// a valid state from the start (sim_AMR_channel's idiom); level-0
	// blocks have no DF overlaps, so only their interior is authored (their
	// ghost rows are managed by the exterior boundary conditions)
	void setInitialCondition()
	{
		for (auto& block : nse.blocks)
			block.setEquilibrium(1, 0, 0, 0);

		nse.copyDFsToHost();
	}

	void resetDFs() override
	{
		spdlog::info("Computing uniform-at-rest initial condition (ball in channel)");
		setInitialCondition();
	}

	// per-block (per-level) lattice conversion of the physical inflow
	// velocity: with the 2:1 diffusive scaling the lattice velocity is the
	// same on both levels, but converting per block keeps the hook correct
	// per level by construction (mirrors sim_AMR_channel's idiom)
	void updateKernelVelocities() override
	{
		for (auto& block : nse.blocks) {
			const lat_t lat_local = (block.level == 0) ? nse.lat : block.lat_local;
			block.data.inflow_vx = lat_local.phys2lbmVelocity(phys_inflow_velocity);
			block.data.inflow_vy = 0;
			block.data.inflow_vz = 0;
		}
	}

	// sim_3's probe: the Reynolds number reconstructed from the level-0
	// lattice inflow velocity and the ball diameter
	void probe1() override
	{
		spdlog::info(
			"Reynolds = {:f} lbmvel {:f} physvel {:f}",
			lbm_inflow_vx * ball_diameter / nse.lat.physDl / nse.lat.lbmViscosity(),
			lbm_inflow_vx,
			nse.lat.lbm2physVelocity(lbm_inflow_vx)
		);
	}

	// momentum-exchange drag sample on the staircased ball wall of the
	// FINEST level (see the probe header above): runs a small reduction
	// kernel over a window around the ball, appends one CSV row, and runs a
	// compact EMA log every 25 samples. Non-finite force values mark the
	// run unstable: logged, counted, and the run terminates gracefully
	void probe2() override
	{
		if (drag_probe_period <= 0)
			return;

		const int probe_level = nse.max_level;	// 0 selects the uniform reference run
		if (d_force == nullptr && cudaMalloc(&d_force, sizeof(BallForceAccumulator)) != cudaSuccess) {
			spdlog::error("drag probe: device accumulator allocation failed; probe disabled");
			drag_probe_period = -1;
			return;
		}

		BallForceAccumulator total;
		for (auto& block : nse.blocks) {
			if ((int) block.level != probe_level)
				continue;
			// level 0 has no lat_local (valid only on fine blocks; the
			// updateKernelVelocities idiom)
			const lat_t lat_local = (block.level == 0) ? nse.lat : block.lat_local;
			const idx3d c = lat_local.phys2lbmPoint(ball_c);
			const idx half = (idx) ceil(0.55 * ball_diameter / lat_local.physDl) + 2;
			idx3d begin, end;
			for (int a = 0; a < 3; a++) {
				begin[a] = TNL::max((idx) 0, c[a] - half);
				end[a] = TNL::min(block.local[a], c[a] + half + 1);
			}
			cudaMemsetAsync(d_force, 0, sizeof(BallForceAccumulator));
			const auto direction = TNL::Containers::SyncDirection::None;
			TNL::Backend::LaunchConfiguration launch_config;
			launch_config.blockSize = block.computeData.at(direction).blockSize;
			launch_config.gridSize = block.getCudaGridSize(end - begin, launch_config.blockSize);
			TNL::Backend::launchKernelAsync(cudaBallForceKernel<NSE>, launch_config, block.data, begin, end, d_force);
			TNL::Backend::streamSynchronize(0);
			BallForceAccumulator block_acc;
			cudaMemcpy(&block_acc, d_force, sizeof(BallForceAccumulator), cudaMemcpyDeviceToHost);
			total.fx += block_acc.fx;
			total.fy += block_acc.fy;
			total.fz += block_acc.fz;
			total.links += block_acc.links;
		}

		// per-level conversion factors (blocks of one level share lat_local;
		// diffusive scaling keeps u_lb level-independent)
		const double dxL = nse.lat.physDl / (double) (1 << probe_level);
		const double u_lb = phys_inflow_velocity * (nse.lat.physDt / (double) (1 << probe_level)) / dxL;
		const double A_lb = 3.14159265358979323846 * 0.25 * (ball_diameter / dxL) * (ball_diameter / dxL);
		const double scale = 2.0 / (0.5 * u_lb * u_lb * A_lb);	// 2 = the full-way bounce factor
		const double cd_x = total.fx * scale;
		const double cd_y = total.fy * scale;
		const double cd_z = total.fz * scale;

		const double t = nse.physTime();
		const bool finite = std::isfinite(cd_x) && std::isfinite(cd_y) && std::isfinite(cd_z);

		if (! drag_csv.is_open()) {
			drag_csv.open(drag_csv_path, std::ios::out | std::ios::app);
			if (drag_csv.tellp() == 0) {
				drag_csv << "# sim_AMR_ball momentum-exchange drag probe on the staircased ball wall\n";
				drag_csv << "# probe_level = " << probe_level << ", dx_fine = " << dxL << ", D_cells_fine = " << ball_diameter / dxL
						 << ", u_lb = " << u_lb << ", A_lb = " << A_lb << "\n";
				drag_csv << "time_phys,iterations,cd_x,cd_y,cd_z,fx_lb,fy_lb,fz_lb,links\n";
			}
		}
		if (drag_csv.is_open()) {
			drag_csv << fmt::format(
				"{:.10e},{},{:.8e},{:.8e},{:.8e},{:.8e},{:.8e},{:.8e},{}\n",
				t,
				nse.iterations,
				cd_x,
				cd_y,
				cd_z,
				total.fx,
				total.fy,
				total.fz,
				(long) total.links
			);
			drag_csv.flush();
		}

		drag_sample_count++;
		if (finite) {
			if (drag_sample_count == 1) {
				drag_cd_ema = cd_x;
				drag_cd_min = drag_cd_max = cd_x;
			}
			else {
				drag_cd_ema = 0.9 * drag_cd_ema + 0.1 * cd_x;
				drag_cd_min = TNL::min(drag_cd_min, cd_x);
				drag_cd_max = TNL::max(drag_cd_max, cd_x);
			}
		}
		else {
			drag_nan_events++;
			spdlog::error("drag probe: non-finite force sample at t = {:.6e} s (event {}); marking the run unstable and terminating", t, drag_nan_events);
			nse.terminate = 1;
		}
		if (drag_sample_count % 25 == 0)
			spdlog::info(
				"drag probe: t = {:.4e} s, Cd = {:.4f} (EMA {:.4f}, min {:.4f}, max {:.4f}, links {:.0f})",
				t,
				cd_x,
				drag_cd_ema,
				drag_cd_min,
				drag_cd_max,
				total.links
			);
	}

	[[nodiscard]] std::vector<std::string> getOutputDataNames() const override
	{
		return {};
	}

	void outputData(UniformDataWriter<TRAITS>&, const BLOCK&, const idx3d&, const idx3d&) override {}
};

template <typename NSE>
void
sim(const std::string& adios_config = "adios2.xml",
	int RESOLUTION = 1,
	int max_level = 1,
	double lattice_viscosity_override = -1.0,
	double phys_final_time = -1.0,
	int out3d_iter_period = 0,
	double Re = 100.0,
	double ball_diameter_phys = 0.10,
	double drag_probe_period = -1.0,
	const std::string& chain_mode = "band",
	double tight_pad_fraction = 0.3)
{
	using idx = typename NSE::TRAITS::idx;
	using real = typename NSE::TRAITS::real;
	using point_t = typename NSE::TRAITS::point_t;
	using lat_t = Lattice<3, real, idx>;

	// sim_3's geometry and unit derivations byte-for-byte (block_size,
	// domain extents, PHYS_DL/DL_X, viscosity/velocity/dt formulas)
	const int R = RESOLUTION;
	const int block_size = 32;
	const real ball_diameter = ball_diameter_phys;			 // [m]
	const real real_domain_height = 11 * ball_diameter;		 // [m]
	const real real_domain_length = 2 * real_domain_height;	 // [m]
	const idx LBM_Y = R * block_size;
	const idx LBM_Z = LBM_Y;
	const real PHYS_DL = real_domain_height / ((real) LBM_Y - 2.0);
	const idx LBM_X = (int) (real_domain_length / PHYS_DL) + 2;
	point_t PHYS_ORIGIN = {0., 0., 0.};

	const real PHYS_VISCOSITY = 0.001;	// [m^2/s]
	const real PHYS_VELOCITY = Re * PHYS_VISCOSITY / ball_diameter;

	const real LBM_VISCOSITY = (lattice_viscosity_override > 0) ? lattice_viscosity_override : 0.001;  // [Δx^2/Δt]
	const real PHYS_DT = LBM_VISCOSITY / PHYS_VISCOSITY * PHYS_DL * PHYS_DL;						   // [s]

	// initialize the lattice
	lat_t lat;
	lat.global = typename lat_t::CoordinatesType(LBM_X, LBM_Y, LBM_Z);
	lat.physOrigin = PHYS_ORIGIN;
	lat.physDl = PHYS_DL;
	lat.physDt = PHYS_DT;
	lat.physViscosity = PHYS_VISCOSITY;

	const std::string state_id = fmt::format("sim_AMR_ball_res{:03d}_np{:03d}", RESOLUTION, TNL::MPI::GetSize(MPI_COMM_WORLD));
	StateLocal_AMR_Ball<NSE> state(state_id, MPI_COMM_WORLD, lat, adios_config, max_level);

	if (! state.canCompute())
		return;

	// set problem parameters (sim_3's placement: 2 D from the origin, mid-span)
	state.ball_c[0] = 2 * ball_diameter;		 // [m]
	state.ball_c[1] = 0.5 * real_domain_height;	 // [m]
	state.ball_c[2] = 0.5 * real_domain_height;	 // [m]
	state.ball_diameter = ball_diameter;		 // [m]
	state.nse.physCharLength = ball_diameter;	 // [m]
	state.phys_inflow_velocity = PHYS_VELOCITY;
	state.lbm_inflow_vx = state.nse.lat.phys2lbmVelocity(PHYS_VELOCITY);

	spdlog::info("Reynolds = {:f} lbmvel {:f} physvel {:f}", Re, state.lbm_inflow_vx, PHYS_VELOCITY);

	// sim_3's default final time (30 s); override with --phys-final-time
	state.nse.physFinalTime = (phys_final_time > 0.0) ? phys_final_time : 30.0;	 // [s]
	state.cnt[PRINT].period = 0.01;
	state.cnt[OUT3D].period = 0.05;
	state.cnt[PROBE1].period = 0.1;	 // sim_3's probe cadence

	// momentum-exchange drag probe on the ball wall of the finest level:
	// default cadence of 200 coarse steps (fine substep cadence of 200/2^L);
	// 0 disables. The CSV goes to the current working directory (the study
	// runner uses a fresh workdir per case)
	state.drag_probe_period = (drag_probe_period >= 0.0) ? drag_probe_period : 200.0 * PHYS_DT;
	state.drag_csv_path = fmt::format("drag_probe_{}.csv", state_id);
	if (state.drag_probe_period > 0.0)
		state.cnt[PROBE2].period = state.drag_probe_period;

	// per-iteration frame cadence (--out3d-iter-period N): write the OUT3D
	// macroscopic frame every N fine iterations, independent of the
	// time-based cadence above; mirrors sim_AMR_channel. The fine (level-1)
	// timestep is PHYS_DT / 2 (2:1 subcycling) and the OUT3D hook in
	// State_AMR::AfterSimUpdate fires at most once per coarse step, so N = 1
	// and N = 2 both write every coarse step
	if (out3d_iter_period > 0)
		state.cnt[OUT3D].period = out3d_iter_period * PHYS_DT / 2;

	// AMR setup before execute: allocate, create the fine blocks, initialize all levels.
	// There is NO sim-level markAMRInterface call: State::SimInit->reset() first clears
	// every map (resetMap) and only then runs setupBoundaries(),
	// so an interface tagging issued here runs before any boundary exists.
	// State_AMR::SimInit's own markAMRInterface call re-derives the correct set afterwards
	// (sim_AMR_channel's ruling; the re-invocation is idempotent by construction)
	if (max_level > 0) {
		// stamped ball column on the level-0 lattice: a host-side replica of
		// lbmDrawSphere's truncating phys2lbmPoint + cell-center l2Norm test
		// (obstacles_lbm.h). The nested-chain containment margin must clear
		// stamped MAP cells, which sit up to a cell outside the continuous
		// surface AABB on the truncating rounding side (see the file header)
		const typename NSE::TRAITS::idx3d ball_c_l0{
			(idx) (state.ball_c[0] / PHYS_DL + 0.5),
			(idx) (state.ball_c[1] / PHYS_DL + 0.5),
			(idx) (state.ball_c[2] / PHYS_DL + 0.5)};
		const real lbm_radius = 0.5 * state.ball_diameter / PHYS_DL;
		const idx range = (idx) ceil(lbm_radius) + 1;
		idx col_min[3]{LBM_X, LBM_Y, LBM_Z};
		idx col_max[3]{0, 0, 0};
		for (idx py = ball_c_l0.y() - range; py <= ball_c_l0.y() + range; py++)
			for (idx pz = ball_c_l0.z() - range; pz <= ball_c_l0.z() + range; pz++)
				for (idx px = ball_c_l0.x() - range; px <= ball_c_l0.x() + range; px++) {
					const point_t p{(px - 0.5) * PHYS_DL, (py - 0.5) * PHYS_DL, (pz - 0.5) * PHYS_DL};
					if (TNL::l2Norm(p - state.ball_c) >= 0.5 * state.ball_diameter)
						continue;
					if (px < col_min[0])
						col_min[0] = px;
					if (px > col_max[0])
						col_max[0] = px;
					if (py < col_min[1])
						col_min[1] = py;
					if (py > col_max[1])
						col_max[1] = py;
					if (pz < col_min[2])
						col_min[2] = pz;
					if (pz > col_max[2])
						col_max[2] = pz;
				}
		const std::array<double, 3> ball_col_min_l0{(double) col_min[0], (double) col_min[1], (double) col_min[2]};
		const std::array<double, 3> ball_col_max_l0{(double) col_max[0], (double) col_max[1], (double) col_max[2]};
		spdlog::info(
			"stamped ball column on level 0: x [{}..{}], y [{}..{}], z [{}..{}]",
			col_min[0],
			col_max[0],
			col_min[1],
			col_max[1],
			col_min[2],
			col_max[2]
		);
		// chain-mode selection: "band" is the historical telescoping chain
		// (default, behavior unchanged); "tight" hugs the ball below the
		// same level-1 anchor (the file header's close-the-gap item #1)
		const std::string amr_config =
			(chain_mode == "tight")
				? deriveAMRBallChainTight(
					  R, max_level, ball_col_min_l0, ball_col_max_l0, (double) state.ball_diameter / PHYS_DL, tight_pad_fraction
				  )
					  .region_config
				: deriveAMRBallChain(R, max_level, ball_col_min_l0, ball_col_max_l0).region_config;

		state.nse.allocateHostData();
		state.nse.allocateDeviceData();
		state.nse.iterations = 0;
		createAMRBlocks(state.nse, parseAMRConfig<NSE>(amr_config));

		// (2026-09-22 drag-study fix: setInitialCondition before execute()
		// requires allocated device data, so it must stay inside the
		// max_level > 0 block -- on the max_level == 0 uniform path the
		// execute()->SimInit->reset()->resetDFs() chain runs it after
		// allocation; the pre-execute call on unallocated blocks hit the
		// TNL zero-sized-NDArray bounds assertion)
		state.setInitialCondition();
	}

	execute(state);
}

template <typename TRAITS = TraitsSP>
void
run(const std::string& adios_config,
	int resolution,
	int max_level = 1,
	double lattice_viscosity = -1.0,
	double phys_final_time = -1.0,
	int out3d_iter_period = 0,
	double Re = 100.0,
	double ball_diameter = 0.10,
	double drag_probe_period = -1.0,
	const std::string& chain_mode = "band",
	double tight_pad_fraction = 0.3)
{
	//using COLL = D3Q27_CUM<TRAITS, D3Q27_EQ_INV_CUM<TRAITS>>;
	using COLL = D3Q27_CUM_WELL<TRAITS, D3Q27_EQ_INV_CUM_WELL<TRAITS>>;

	using NSE_CONFIG = LBM_CONFIG<
		TRAITS,
		D3Q27_KernelStruct,
		NSE_Data_ConstInflow,
		COLL,
		typename COLL::EQ,
		D3Q27_STREAMING<TRAITS>,
		D3Q27_BC_All,
		D3Q27_MACRO_Default<TRAITS>>;

	sim<NSE_CONFIG>(
		adios_config,
		resolution,
		max_level,
		lattice_viscosity,
		phys_final_time,
		out3d_iter_period,
		Re,
		ball_diameter,
		drag_probe_period,
		chain_mode,
		tight_pad_fraction
	);
}

int main(int argc, char** argv)
{
	TNLMPI_INIT mpi(argc, argv);

	argparse::ArgumentParser program("sim_AMR_ball");
	program.add_description(
		"AMR ball-in-channel simulation (a sim_3 port): Dirichlet inflow/outflow channel, y/z-centered refinement boxes around the ball and its near wake."
	);
	program.add_argument("--adios-config").help("path to ADIOS2 configuration file").default_value(std::string("adios2.xml")).nargs(1);
	program.add_argument("--resolution").help("resolution of the lattice").scan<'i', int>().default_value(1).nargs(1);
	program.add_argument("--max-level")
		.help(
			"maximum AMR refinement level: 0 = uniform sim_3-equivalent reference, 1 = the default 2-level ball-wake "
			"footprint, 2..4 = the derived nested interior chain of that depth (subject to the ball-containment "
			"budget at this resolution: R = 1 admits levels 0..4 with the 8R-cell y/z band)"
		)
		.scan<'i', int>()
		.default_value(1)
		.nargs(1);
	program.add_argument("--lattice-viscosity")
		.help("override lattice viscosity [dx^2/dt] (for uniform reference runs; default 0.001, sim_3's value)")
		.scan<'g', double>()
		.default_value(-1.0)
		.nargs(1);
	program.add_argument("--phys-final-time")
		.help("physical final time [s] (default: 30.0, sim_3's value)")
		.scan<'g', double>()
		.default_value(-1.0)
		.nargs(1);
	program.add_argument("--out3d-iter-period")
		.help(
			"write the OUT3D macroscopic frame every N fine iterations, independent of the time-based cadence "
			"(fine dt = coarse dt/2; the write hook fires at most once per coarse step, so N = 1 and N = 2 "
			"both write every coarse step; 0 = off)"
		)
		.scan<'i', int>()
		.default_value(0)
		.nargs(1);
	program.add_argument("--Re").help("desired Reynolds number (affects the inflow velocity)").scan<'g', double>().default_value(100.0).nargs(1);
	program.add_argument("--ball-diameter")
		.help("ball diameter [m] (sets the domain extents 11 D x 11 D x 22 D and the inflow velocity via Re)")
		.scan<'g', double>()
		.default_value(0.10)
		.nargs(1);
	program.add_argument("--drag-probe-period")
		.help(
			"period [s] of the momentum-exchange drag probe on the ball wall of the finest level "
			"(appends drag_probe_<id>.csv in the working directory; default: 200 coarse steps; 0 disables)"
		)
		.scan<'g', double>()
		.default_value(-1.0)
		.nargs(1);
	program.add_argument("--chain")
		.help(
			"refinement chain mode: 'band' telescopes the fixed level-1 band by 3 parent cells per face per hop (default, the "
			"historical behavior); 'tight' hugs the ball below the same level-1 anchor with per-level boxes <= 2 D across "
			"derived from the stamped ball column (see the file header; only affects --max-level 2..4)"
		)
		.default_value(std::string("band"))
		.nargs(1);
	program.add_argument("--tight-pad")
		.help(
			"tight-chain pad beyond the mandatory 2-own-cell ball clearance, as a fraction of the local ball diameter D_L "
			"(each level's box is trimmed to <= 2 D across and to the V7 telescoping floor; only used with --chain tight)"
		)
		.scan<'g', double>()
		.default_value(0.3)
		.nargs(1);

	try {
		program.parse_args(argc, argv);
	}
	catch (const std::exception& err) {
		std::cerr << err.what() << '\n';
		std::cerr << program;
		return 1;
	}

	const auto adios_config = program.get<std::string>("--adios-config");
	const auto resolution = program.get<int>("--resolution");
	const auto max_level = program.get<int>("--max-level");
	const auto lattice_viscosity = program.get<double>("--lattice-viscosity");
	const auto phys_final_time = program.get<double>("--phys-final-time");
	const auto out3d_iter_period = program.get<int>("--out3d-iter-period");
	const auto Re = program.get<double>("--Re");
	const auto ball_diameter = program.get<double>("--ball-diameter");
	const auto drag_probe_period = program.get<double>("--drag-probe-period");
	const auto chain_mode = program.get<std::string>("--chain");
	const auto tight_pad_fraction = program.get<double>("--tight-pad");

	if (resolution < 1) {
		fmt::println(stderr, "CLI error: resolution must be at least 1");
		return 1;
	}
	if (max_level < 0 || max_level > 4) {
		fmt::println(stderr, "CLI error: max-level must be in 0..4 (the nested interior chain is derived up to five lattice levels 0..4)");
		return 1;
	}
	if (out3d_iter_period < 0) {
		fmt::println(stderr, "CLI error: out3d-iter-period must be non-negative");
		return 1;
	}
	if (Re < 1) {
		fmt::println(stderr, "CLI error: Re must be at least 1");
		return 1;
	}
	if (ball_diameter <= 0) {
		fmt::println(stderr, "CLI error: ball-diameter must be positive");
		return 1;
	}
	if (chain_mode != "band" && chain_mode != "tight") {
		fmt::println(stderr, "CLI error: chain must be 'band' or 'tight'");
		return 1;
	}
	if (tight_pad_fraction < 0) {
		fmt::println(stderr, "CLI error: tight-pad must be non-negative");
		return 1;
	}

	// SP only (2026-08-18): the DP branch doubled the device-code
	// instantiation cost of this TU (build-time investigation)
	run<TraitsSP>(
		adios_config,
		resolution,
		max_level,
		lattice_viscosity,
		phys_final_time,
		out3d_iter_period,
		Re,
		ball_diameter,
		drag_probe_period,
		chain_mode,
		tight_pad_fraction
	);

	return 0;
}
