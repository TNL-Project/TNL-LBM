#include <argparse/argparse.hpp>
#include <utility>

#include "lbm3d/core.h"
#include "lbm3d/amr_state.h"

// 2-level AMR developing-channel diagnostic (Experiment B, task B.7 artifact):
// Dirichlet inflow/outflow channel with the refinement patch placed in the
// developing-flow region so that the mean flux passes through the patch's
// skin/interface region — a regime the decaying (periodic) Taylor-Green
// benchmark of sim_AMR.cu cannot probe. This file is the B.7 "Decision 0
// cousin" variant of sim_AMR/sim_AMR.cu: same collision/streaming/macro
// configuration, same CLI surface, same State_AMR Berger-Colella driver.
//
// Geometry (coarse level-0 cells, R = --resolution):
// - domain X=64R x Y=16R x Z=16R, planar channel along x (periodic tangent y),
// - inflow: x=1 GEO_INFLOW_LEFT with a uniform (constant) velocity profile
//   (NSE_Data_ConstInflow), outflow: x=X-2 GEO_OUTFLOW_RIGHT_INTERP,
// - walls: z=R and z=13R+1 GEO_WALL (bounce-back planes at 1.0 and 13.0 coarse
//   dx — resolution-independent physical locations, interior fluid z-depth is
//   exactly 12 coarse cells at every R), outer layers z<R and z>13R+1
//   GEO_NOTHING (the BC planes never sit on the array edge; mirrors sim_1's
//   extra-layer idiom generalized to R>1),
// - x edge layers x=0 and x=X-1 GEO_NOTHING,
// - one level-1 footprint "1 24R 4R 4R 16R 8R 8R" (coarse cells): a slab
//   crossing the channel's mid-x developing region, strictly interior on all
//   axes (ring cells at x=23/40, y=3/12, z=3/12 are all collision-active
//   fluid, 2 cells clear of the walls), so the mean flux passes through the
//   footprint's x-min/x-max skin faces (B-off: ring-F2C + full-footprint
//   interior; B-on: skin rect faces only).
//
// Physics: lattice inflow velocity U_lb = 0.1 (uniform inflow), lattice
// viscosity 0.005 on the coarse level (doubles to 0.01 on the fine level by
// the standard diffusive 2:1 scaling, see initLevelLattice), PHYS_VISCOSITY
// = 1.5e-5 m^2/s, channel height 0.041 m (thin-channel regime: 12 coarse /
// 24 fine interior cells), Re_H ~ 320 (laminar, development length much
// longer than the domain: the whole channel including the patch stays in the
// developing regime at the final time). Run length: 640R coarse iterations
// (convects the inflow front across the full domain once).
//
// Reference path: uniform fine-resolution run of THE SAME variant with
// --max-level 0 --resolution 2 --lattice-viscosity 0.01 (uniform 2R grid with
// nu_lb matching the AMR fine level). The production write3D_AMR no-ops for
// max_level == 0 (see .omo/notepads issues.md), so the reference VTKHDF
// series is emitted by the untracked helper driver
// .omo/b67/sim_channel_ref_vtkhdf.cu (byte-copy + the P0.3-established two
// deltas), NOT by editing the production writer guard.

template <typename NSE>
struct StateLocal_AMR_Channel : State_AMR<NSE>
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
	idx res = 1;					 // resolution factor R (coarse cells per 0.041m/16)

	StateLocal_AMR_Channel(
		const std::string& id,
		const TNL::MPI::Comm& communicator,
		lat_t lat,
		const std::string& adiosConfigPath,
		int max_level = 1
	)
	: State_AMR<NSE>(
		  id,
		  communicator,
		  std::move(lat),
		  adiosConfigPath,
		  // planar channel: non-periodic x (inflow/outflow), periodic tangent
		  // y, non-periodic z (walls) -- boundary planes are set in
		  // setupBoundaries() below
		  bool3d{false, true, false},
		  max_level
	  )
	{}

	// channel boundary map on the level-0 lattice (fine blocks are strictly
	// interior, so no plane reaches them; markAMRInterface runs later and
	// only re-tags GEO_FLUID cells, leaving these tags intact)
	void setupBoundaries() override
	{
		nse.setBoundaryX(1, BC::GEO_INFLOW_LEFT);								 // inflow (constant profile)
		nse.setBoundaryX(nse.lat.global.x() - 2, BC::GEO_OUTFLOW_RIGHT_INTERP);	 // outflow

		// bounce-back wall cells such that the wall link-planes sit at 1.0 and
		// 13.0 coarse dx at any R: bottom wall cell center is 0.5*dl below the
		// 1.0-dx plane (z = 0.5 + (1.0*R*dl - 0.5*dl)/dl = R), top analogously
		nse.setBoundaryZ(res, BC::GEO_WALL);		 // bottom wall
		nse.setBoundaryZ(13 * res + 1, BC::GEO_WALL);	 // top wall

		// GEO_NOTHING edge/dead layers (BC planes never sit on the array edge;
		// at R>1 the cells below/above the wall planes are unreachable dead
		// zones and are tagged NOTHING as well)
		for (idx d = 0; d < res; d++) {
			nse.setBoundaryZ(d, BC::GEO_NOTHING);						   // below bottom wall
			nse.setBoundaryZ(nse.lat.global.z() - 1 - d, BC::GEO_NOTHING);	   // above top wall
		}
		// x: ONLY the true outermost layers — the inflow/outflow planes sit at
		// the fixed indices 1 and X-2 at every R, so tagging deeper x layers
		// would clobber them at R > 1 (caught by the R=2 reference run: the
		// inflow plane was overwritten and the field never developed)
		nse.setBoundaryX(0, BC::GEO_NOTHING);						   // left edge
		nse.setBoundaryX(nse.lat.global.x() - 1, BC::GEO_NOTHING);	   // right edge
		// y planes are intentionally untouched: periodic tangent direction
	}

	// uniform-flow initial condition at rest: rho = 1, u = 0 on all blocks;
	// the inflow BC then develops the channel flow from t = 0
	void setInitialCondition()
	{
		for (auto& block : nse.blocks) {
#ifdef HAVE_MPI
			auto local_df = block.dfs[0].getLocalView();
#else
			auto local_df = block.dfs[0].getView();
#endif
			const idx3d begin = {0, 0, 0};
			const idx3d end = {block.local.y(), block.local.z(), block.local.x()};
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
		spdlog::info("Computing uniform-at-rest initial condition (developing channel)");
		setInitialCondition();
	}

	// per-block (per-level) lattice conversion of the physical inflow
	// velocity: with the 2:1 diffusive scaling the lattice velocity is the
	// same on both levels, but converting per block keeps the hook correct
	// per level by construction (mirrors sim_AMR's Taylor-Green IC idiom)
	void updateKernelVelocities() override
	{
		for (auto& block : nse.blocks) {
			const lat_t lat_local = (block.level == 0) ? nse.lat : block.lat_local;
			block.data.inflow_vx = lat_local.phys2lbmVelocity(phys_inflow_velocity);
			block.data.inflow_vy = 0;
			block.data.inflow_vz = 0;
		}
	}

	[[nodiscard]] std::vector<std::string> getOutputDataNames() const override
	{
		return {};
	}

	void outputData(UniformDataWriter<TRAITS>&, const BLOCK&, const idx3d&, const idx3d&) override {}
};

template <typename NSE>
void sim(const std::string& adios_config = "adios2.xml", int RESOLUTION = 1, int max_level = 1, float lattice_viscosity_override = -1.0f, float phys_final_time = -1.0f, bool write_dfs = false, int out3d_iter_period = 0)
{
	using idx = typename NSE::TRAITS::idx;
	using real = typename NSE::TRAITS::real;
	using point_t = typename NSE::TRAITS::point_t;
	using lat_t = Lattice<3, real, idx>;

	const int R = RESOLUTION;
	const int X = 64 * R;  // streamwise extent
	const int Y = 16 * R;  // periodic tangent extent
	const int Z = 16 * R;  // wall-normal extent
	const real LBM_VISCOSITY = (lattice_viscosity_override > 0) ? lattice_viscosity_override : 0.005f;	// [Δx^2/Δt]
	const real PHYS_HEIGHT = 0.041;			// [m] channel height (z extent)
	const real PHYS_VISCOSITY = 1.5e-5;		// [m^2/s]
	const real LBM_INFLOW = 0.1f;			// [Δx/Δt] uniform inflow lattice velocity
	const real PHYS_DL = PHYS_HEIGHT / Z;	// [m]
	const real PHYS_DT = LBM_VISCOSITY / PHYS_VISCOSITY * PHYS_DL * PHYS_DL;  // [s]
	const real PHYS_INFLOW = LBM_INFLOW * PHYS_DL / PHYS_DT;				  // [m/s]
	const real REYNOLDS = PHYS_INFLOW * PHYS_HEIGHT / PHYS_VISCOSITY;
	point_t PHYS_ORIGIN = {0., 0., 0.};

	lat_t lat;
	lat.global = typename lat_t::CoordinatesType(X, Y, Z);
	lat.physOrigin = PHYS_ORIGIN;
	lat.physDl = PHYS_DL;
	lat.physDt = PHYS_DT;
	lat.physViscosity = PHYS_VISCOSITY;

	// one level-1 slab in the mid-channel developing region (coarse cells)
	const std::string amr_config = fmt::format("1 {} {} {} {} {} {}", 24 * R, 4 * R, 4 * R, 16 * R, 8 * R, 8 * R);

	const std::string state_id = fmt::format("sim_AMR_channel_res{:02d}_np{:03d}", RESOLUTION, TNL::MPI::GetSize(MPI_COMM_WORLD));
	StateLocal_AMR_Channel<NSE> state(state_id, MPI_COMM_WORLD, lat, adios_config, max_level);
	state.amr_write_dfs = write_dfs;

	if (! state.canCompute())
		return;

	state.phys_inflow_velocity = PHYS_INFLOW;
	state.res = R;

	spdlog::info("developing channel: U_lb = {:e}, Re_H = {:e}, PHYS_DT = {:e}", LBM_INFLOW, REYNOLDS, PHYS_DT);

	// one full convective pass by default (640R coarse iterations); override
	// with --phys-final-time for long-horizon development/acoustic-decay runs
	state.nse.physFinalTime = (phys_final_time > 0.0f) ? phys_final_time : 640.0f * R * PHYS_DT;	 // [s]
	state.cnt[PRINT].period = 0.01;
	state.cnt[OUT3D].period = 0.05;

	// per-iteration frame cadence (--out3d-iter-period N): write the OUT3D
	// macroscopic frame every N fine iterations, independent of the
	// time-based cadence above; mirrors sim_AMR.cu and replaces the
	// uncommitted "cnt[OUT3D].period = PHYS_DT" probe hack. The fine
	// (level-1) timestep is PHYS_DT / 2 (2:1 subcycling) and the OUT3D
	// hook in State_AMR::AfterSimUpdate fires at most once per coarse
	// step, so N = 1 and N = 2 both write every coarse step
	if (out3d_iter_period > 0)
		state.cnt[OUT3D].period = out3d_iter_period * PHYS_DT / 2;

	// AMR setup before execute: allocate, set the coarse boundary map, create
	// the fine block, tag the interface cells, initialize all levels;
	// State_AMR::SimInit re-applies the map/interface setup internally
	state.nse.allocateHostData();
	state.nse.allocateDeviceData();
	state.nse.iterations = 0;
	if (max_level > 0) {
		createAMRBlocks(state.nse, parseAMRConfig<NSE>(amr_config));
		markAMRInterface(state.nse);
	}
	state.setInitialCondition();

	execute(state);
}

template <typename TRAITS = TraitsSP>
void run(const std::string& adios_config, int resolution, int max_level = 1, float lattice_viscosity = -1.0f, float phys_final_time = -1.0f, bool write_dfs = false, int out3d_iter_period = 0)
{
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

	sim<NSE_CONFIG>(adios_config, resolution, max_level, lattice_viscosity, phys_final_time, write_dfs, out3d_iter_period);
}

int main(int argc, char** argv)
{
	TNLMPI_INIT mpi(argc, argv);

	argparse::ArgumentParser program("sim_AMR_channel");
	program.add_description("2-level AMR developing-channel diagnostic (B.7): inflow/outflow channel, refinement slab in the developing region.");
	program.add_argument("--adios-config").help("path to ADIOS2 configuration file").default_value(std::string("adios2.xml")).nargs(1);
	program.add_argument("--resolution").help("resolution of the lattice").scan<'i', int>().default_value(1).nargs(1);
	program.add_argument("--max-level").help("maximum AMR refinement level (0 = uniform)").scan<'i', int>().default_value(1).nargs(1);
	program.add_argument("--lattice-viscosity")
		.help("override lattice viscosity [dx^2/dt] (for uniform reference runs)")
		.scan<'f', float>()
		.default_value(-1.0f)
		.nargs(1);
	program.add_argument("--phys-final-time").help("physical final time [s] (default: one full convective pass, 640R coarse iterations)").scan<'f', float>().default_value(-1.0f).nargs(1);
	program.add_argument("--write-dfs").help("write raw df_cur fields f00..f{Q-1} into the VTKHDF frames (debug)").default_value(false).implicit_value(true);
	program.add_argument("--out3d-iter-period")
		.help(
			"write the OUT3D macroscopic frame every N fine iterations, independent of the time-based cadence "
			"(fine dt = coarse dt/2; the write hook fires at most once per coarse step, so N = 1 and N = 2 "
			"both write every coarse step; 0 = off)"
		)
		.scan<'i', int>()
		.default_value(0)
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
	const auto lattice_viscosity = program.get<float>("--lattice-viscosity");
	const auto phys_final_time = program.get<float>("--phys-final-time");
	const auto write_dfs = program.get<bool>("--write-dfs");
	const auto out3d_iter_period = program.get<int>("--out3d-iter-period");

	if (resolution < 1) {
		fmt::println(stderr, "CLI error: resolution must be at least 1");
		return 1;
	}
	if (out3d_iter_period < 0) {
		fmt::println(stderr, "CLI error: out3d-iter-period must be non-negative");
		return 1;
	}

	// SP only (2026-08-18): the DP branch doubled the device-code
	// instantiation cost of this TU (build-time investigation)
	run<TraitsSP>(adios_config, resolution, max_level, lattice_viscosity, phys_final_time, write_dfs, out3d_iter_period);

	return 0;
}
