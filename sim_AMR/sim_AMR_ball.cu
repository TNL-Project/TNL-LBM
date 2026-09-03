#define USE_GEIER_CUM_2017		 // use Geier 2017 Cummulant improvement A,B terms
#define USE_GEIER_CUM_ANTIALIAS	 // use antialiasing Dxu, Dyv, Dzw from Geier 2015/2017

#include <argparse/argparse.hpp>
#include <array>
#include <utility>

#include "lbm3d/core.h"
#include "lbm3d/amr_state.h"
#include "lbm3d/obstacles_lbm.h"

// AMR ball-in-channel simulation: a port of sim_NSE/sim_3.cu ("LBM simulation
// with ball in 3D") onto the State_AMR Berger-Colella subcycling driver of
// sim_AMR/sim_AMR_channel.cu. Same collision/streaming/macro configuration
// and CLI surface as the other two sim_AMR binaries; the physics constants,
// boundary conditions and the level-0 obstacle stamping are sim_3's verbatim.
//
// Geometry (coarse level-0 cells, R = --resolution; sim_3's formulas
// byte-for-byte): domain H = 11*ball_diameter tall/deep (LBM_Y = LBM_Z =
// 32*R, PHYS_DL = H/(32R-2)), L = 2*H long (LBM_X = floor(L/PHYS_DL) + 2,
// 61 cells at R = 1), ball of diameter D centered at (2D, H/2, H/2) -- at
// R = 1 the stamped ball occupies the coarse cells x in {4,5,6}, y/z in
// {14,15,16} (lbmDrawSphere's truncating phys2lbmPoint + l2Norm test on a
// 1.364-cell radius). BCs are sim_3's: inflow x = 1 (GEO_INFLOW_LEFT,
// constant profile via NSE_Data_ConstInflow), outflow x = X-2
// (GEO_OUTFLOW_RIGHT_INTERP), symmetry planes y/z at 1 and N-2, GEO_NOTHING
// on every edge plane (the A-A extra-layer idiom sim_3 already carries).
//
// Level-1 footprint "1 2R 11R 11R 21R 8R 8R" (coarse cells; R = 1:
// [2,23) x [11,19) x [11,19)): the ball plus the beginning of its wake --
// the analogue of the channel sim's developing-region slab. The wake side
// reaches coarse x = 22, about 5.5 D behind the ball's back face; both y/z
// faces keep ~2.6 coarse cells (~1 D) of margin from the ball surface; all
// six ring faces sit on collision-active fluid except the x-min face at
// R = 1 (see below). The box also satisfies the Schonherr-ch7 band
// registration w.r.t. the level-0 ball: every stamped ball cell sits at
// footprint depth >= 2 (the depth-1 skin row is the F2C destination band
// and must stay frozen GEO_NOTHING; the ball surface stops exactly one
// coarse cell above the floor at x = 4 vs. origin 2).
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
// derived spec is logged). The telescoping x-min face chases the fixed ball
// position, so the chain budget exhausts at low R: the derivation
// hard-fails when a derived level no longer contains the ball surface with
// >= 2 of its own cells of margin per face (R = 1 supports levels 0..2 --
// at level 2 the ball-front margin thins to ~2.4 finest cells, the
// resolution floor of sim_3's ball-at-2D on a 32-cell cross-section;
// levels 3..4 need R >= 2 / R >= 3 respectively, the guard names the
// failing face). --max-level 0 is the uniform sim_3-equivalent reference
// run (all AMR machinery off, write3D_AMR no-ops).
//
// Physics (sim_3 verbatim): PHYS_VISCOSITY = 0.001 m^2/s, LBM_VISCOSITY =
// 0.001 (--lattice-viscosity overrides), PHYS_VELOCITY = Re * nu / D with
// --Re = 100 default (lbm inflow velocity ~0.0367 at R = 1), PHYS_DT =
// (nu_lb/nu_phys) * DL^2. Fine levels scale diffusively (nu doubles per
// level, see initLevelLattice); the uniform inflow drives the level-0
// GEO_INFLOW_LEFT plane only. Default final time is sim_3's 30 s;
// --phys-final-time overrides. Deliberately NOT ported from sim_3: the 2D
// cuts (cut_X/cut_Y/cut_Z) and the OUT2D cadence (the base 2D-cut pipeline
// is level-0-only and the two existing AMR sims write no 2D output), and
// the custom BP5 output fields (density fluctuation, physical velocities)
// -- the AMR 3D output is the base class' VTKHDF OverlappingAMR writer
// (map + macros per level at the OUT3D cadence), the sim returns the
// AMR-house empty outputData pair. probe1 keeps sim_3's Reynolds/velocity
// log line at sim_3's PROBE1 period; PRINT/OUT3D use the sim_AMR house
// cadences and --out3d-iter-period mirrors the channel sim.

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
	const std::array<double, 3>& ball_surf_min_l0,	// ball surface AABB lower corner in continuous level-0 cell units
	const std::array<double, 3>& ball_surf_max_l0	// upper corner
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

	// the anchor: the level-1 footprint "1 2R 11R 11R 21R 8R 8R" doubled
	// into level-1 cells (emission below reproduces that string exactly)
	std::vector<Rect> rects;
	rects.push_back(Rect{{0, 0, 0}, {0, 0, 0}});  // level 0 unused (indexing by level)
	rects.push_back(Rect{{4 * R, 22 * R, 22 * R}, {42 * R, 16 * R, 16 * R}});

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

	// ball containment: every derived level must keep the ball surface at
	// least `ball_margin` of its own cells inside the rect on every face
	// (the conservative continuous-surface floor -- the stamped cells sit
	// up to half a cell deeper inside); the x-min face is the binding one
	// at low R (the file header comment)
	for (int L = 1; L <= max_level; L++) {
		const Rect& rect = rects[L];
		const double scale = 1 << L;  // level-L cells per level-0 cell
		for (int a = 0; a < 3; a++) {
			const double surf_min = ball_surf_min_l0[a] * scale;
			const double surf_max = ball_surf_max_l0[a] * scale;
			if (rect.lo[a] > surf_min - ball_margin || rect.lo[a] + rect.span[a] < surf_max + ball_margin) {
				const std::string message = fmt::format(
					"AMR ball chain: level-{} footprint [{},{},{}] + [{},{},{}] no longer contains the ball surface ([{}..{}] on "
					"axis {}) with the {}-cell clearance at R = {}: the telescoping budget is exhausted; raise --resolution or "
					"lower --max-level",
					L,
					rect.lo[0],
					rect.lo[1],
					rect.lo[2],
					rect.span[0],
					rect.span[1],
					rect.span[2],
					surf_min,
					surf_max,
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
		"AMR ball chain: derived {} level(s) on top of the ball-wake anchor (R = {}, inset = {} parent-level cells on every "
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
		nse.setBoundaryX(1, BC::GEO_INFLOW_LEFT);								 // left
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
	// same developing regime sim_3 runs). Fine blocks initialize the FULL
	// stored extent (including the ghost band) so that the ghost rows hold
	// a valid state from the start (sim_AMR_channel's idiom); level-0
	// blocks keep the interior-only loop (their ghost rows are managed by
	// the exterior boundary conditions)
	void setInitialCondition()
	{
		for (auto& block : nse.blocks) {
#ifdef HAVE_MPI
			auto local_df = block.dfs[0].getLocalView();
#else
			auto local_df = block.dfs[0].getView();
#endif
			const int ov_x = block.level == 0 ? 0 : local_df.template getOverlap<1>();
			const int ov_y = block.level == 0 ? 0 : local_df.template getOverlap<2>();
			const int ov_z = block.level == 0 ? 0 : local_df.template getOverlap<3>();
			const idx3d begin = {-ov_y, -ov_z, -ov_x};
			const idx3d end = {block.local.y() + ov_y, block.local.z() + ov_z, block.local.x() + ov_x};
			TNL::Algorithms::parallelFor<DeviceType>(
				begin,
				end,
				[local_df] __cuda_callable__(const idx3d& yzx) mutable
				{
					const auto& [y, z, x] = yzx;
					NSE::COLL::setEquilibriumLat(local_df, x, y, z, 1, 0, 0, 0);
				}
			);

			// copy the initialized DFs so that they are not overridden
			for (uint8_t dftype = 1; dftype < DFMAX; dftype++)
				block.dfs[dftype] = block.dfs[0];
		}

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
	double ball_diameter_phys = 0.10)
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
		// ball surface AABB in continuous level-0 cell units:
		// the nested chain derivation's containment floor (see the file header)
		const std::array<double, 3> ball_surf_min{
			(state.ball_c[0] - 0.5 * state.ball_diameter) / PHYS_DL,
			(state.ball_c[1] - 0.5 * state.ball_diameter) / PHYS_DL,
			(state.ball_c[2] - 0.5 * state.ball_diameter) / PHYS_DL
		};
		const std::array<double, 3> ball_surf_max{
			(state.ball_c[0] + 0.5 * state.ball_diameter) / PHYS_DL,
			(state.ball_c[1] + 0.5 * state.ball_diameter) / PHYS_DL,
			(state.ball_c[2] + 0.5 * state.ball_diameter) / PHYS_DL
		};
		const std::string amr_config = deriveAMRBallChain(R, max_level, ball_surf_min, ball_surf_max).region_config;

		state.nse.allocateHostData();
		state.nse.allocateDeviceData();
		state.nse.iterations = 0;
		createAMRBlocks(state.nse, parseAMRConfig<NSE>(amr_config));
	}

	state.setInitialCondition();

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
	double ball_diameter = 0.10)
{
	using COLL = D3Q27_CUM<TRAITS, D3Q27_EQ_INV_CUM<TRAITS>>;
	//using COLL = D3Q27_CUM_WELL<TRAITS, D3Q27_EQ_INV_CUM_WELL<TRAITS>>;

	using NSE_CONFIG = LBM_CONFIG<
		TRAITS,
		D3Q27_KernelStruct,
		NSE_Data_ConstInflow<TRAITS>,
		COLL,
		typename COLL::EQ,
		D3Q27_STREAMING<TRAITS>,
		D3Q27_BC_All,
		D3Q27_MACRO_Default<TRAITS>>;

	sim<NSE_CONFIG>(adios_config, resolution, max_level, lattice_viscosity, phys_final_time, out3d_iter_period, Re, ball_diameter);
}

int main(int argc, char** argv)
{
	TNLMPI_INIT mpi(argc, argv);

	argparse::ArgumentParser program("sim_AMR_ball");
	program.add_description(
		"AMR ball-in-channel simulation (a sim_3 port): Dirichlet inflow/outflow channel, refinement box around the ball and its near wake."
	);
	program.add_argument("--adios-config").help("path to ADIOS2 configuration file").default_value(std::string("adios2.xml")).nargs(1);
	program.add_argument("--resolution").help("resolution of the lattice").scan<'i', int>().default_value(1).nargs(1);
	program.add_argument("--max-level")
		.help(
			"maximum AMR refinement level: 0 = uniform sim_3-equivalent reference, 1 = the default 2-level ball-wake "
			"footprint, 2..4 = the derived nested interior chain of that depth (subject to the ball-containment "
			"budget at this resolution: R = 1 admits levels 0..2)"
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

	// SP only (2026-08-18): the DP branch doubled the device-code
	// instantiation cost of this TU (build-time investigation)
	run<TraitsSP>(adios_config, resolution, max_level, lattice_viscosity, phys_final_time, out3d_iter_period, Re, ball_diameter);

	return 0;
}
