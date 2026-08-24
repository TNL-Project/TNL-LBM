#include <argparse/argparse.hpp>
#include <utility>

#include "lbm3d/core.h"
#include "lbm3d/amr_state.h"

// 2-level AMR Taylor-Green vortex benchmark: periodic cube with one
// level-1 refined region in the center of the domain

template <typename NSE>
struct StateLocal_AMR : State_AMR<NSE>
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

	// problem parameters
	dreal V_0 = 0;	// [m/s] velocity amplitude
	dreal k = 0;	// [1/m] wave number

	StateLocal_AMR(const std::string& id, const TNL::MPI::Comm& communicator, lat_t lat, const std::string& adiosConfigPath, int max_level = 1)
	: State_AMR<NSE>(
		  id,
		  communicator,
		  std::move(lat),
		  adiosConfigPath,
		  // fully periodic domain, so no setupBoundaries() override
		  bool3d{true, true, true},
		  max_level
	  )
	{}

	// Taylor-Green initial condition on all blocks; per-level lattice
	// parameters are taken from block.lat_local (level 0 uses nse.lat, see
	// State_AMR::blockLbmViscosity); lat_local.physOrigin already accounts
	// for the fine block offset, so local coordinates are passed directly
	void setInitialCondition()
	{
		for (auto& block : nse.blocks) {
#ifdef HAVE_MPI
			auto local_df = block.dfs[0].getLocalView();
#else
			auto local_df = block.dfs[0].getView();
#endif
			const lat_t lat_local = (block.level == 0) ? nse.lat : block.lat_local;
			const dreal V_0 = lat_local.phys2lbmVelocity(this->V_0);
			const dreal k = this->k;

			const idx3d begin = {0, 0, 0};
			const idx3d end = {block.local.y(), block.local.z(), block.local.x()};
			TNL::Algorithms::parallelFor<DeviceType>(
				begin,
				end,
				[local_df, lat_local, V_0, k] __cuda_callable__(const idx3d& yzx) mutable
				{
					const auto& [y, z, x] = yzx;
					const point_t phys = lat_local.lbm2physPoint(x, y, z);
					const dreal u = V_0 * TNL::sin(k * phys.x()) * TNL::cos(k * phys.y()) * TNL::cos(k * phys.z());
					const dreal v = -V_0 * TNL::cos(k * phys.x()) * TNL::sin(k * phys.y()) * TNL::cos(k * phys.z());
					const dreal w = 0;
					const dreal rho =
						1 + 3 * (V_0 * V_0 / 16) * (TNL::cos(2 * k * phys.x()) + TNL::cos(2 * k * phys.y())) * (TNL::cos(2 * k * phys.z()) + 2);
					NSE::COLL::setEquilibriumLat(local_df, x, y, z, rho, u, v, w);
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
		spdlog::info("Computing Taylor-Green initial condition");
		setInitialCondition();
	}

	[[nodiscard]] std::vector<std::string> getOutputDataNames() const override
	{
		return {};
	}

	void outputData(UniformDataWriter<TRAITS>&, const BLOCK&, const idx3d&, const idx3d&) override {}
};

template <typename NSE>
void sim(
	const std::string& adios_config = "adios2.xml",
	int RESOLUTION = 1,
	int max_level = 1,
	float lattice_viscosity_override = -1.0f,
	float phys_final_time = 0.5f,
	float convective_times = 0.0f,
	bool write_dfs = false,
	int out3d_iter_period = 0,
	bool rest_ic = false
)
{
	using idx = typename NSE::TRAITS::idx;
	using real = typename NSE::TRAITS::real;
	using point_t = typename NSE::TRAITS::point_t;
	using lat_t = Lattice<3, real, idx>;

	const int N = 64 * RESOLUTION;
	const real LBM_VISCOSITY = (lattice_viscosity_override > 0) ? lattice_viscosity_override : 0.005f;	// [Δx^2/Δt] lattice viscosity
	const real PHYS_HEIGHT = 0.41;																		// [m] domain extent (periodic cube)
	const real PHYS_VISCOSITY = 1.5e-5;																	// [m^2/s]
	const real REYNOLDS = 100;																			// [-] Re = V_0 * L / nu
	const real PHYS_VELOCITY = REYNOLDS * PHYS_VISCOSITY / PHYS_HEIGHT;									// [m/s]
	const real PHYS_DL = PHYS_HEIGHT / N;
	const real PHYS_DT = LBM_VISCOSITY / PHYS_VISCOSITY * PHYS_DL * PHYS_DL;  // [s]
	point_t PHYS_ORIGIN = {0., 0., 0.};

	lat_t lat;
	lat.global = typename lat_t::CoordinatesType(N, N, N);
	lat.physOrigin = PHYS_ORIGIN;
	lat.physDl = PHYS_DL;
	lat.physDt = PHYS_DT;
	lat.physViscosity = PHYS_VISCOSITY;

	// one level-1 region centered in the coarse domain: "level ox oy oz lx ly lz" in coarse cells
	const std::string amr_config = "1 16 16 16 32 32 32";

	const std::string state_id = fmt::format("sim_AMR_res{:02d}_np{:03d}", RESOLUTION, TNL::MPI::GetSize(MPI_COMM_WORLD));
	StateLocal_AMR<NSE> state(state_id, MPI_COMM_WORLD, lat, adios_config, max_level);
	state.amr_write_dfs = write_dfs;

	if (! state.canCompute())
		return;

	// rest initial condition (--rest-ic): V_0 = 0 collapses the Taylor-Green
	// field to exactly rho = 1, u = v = w = 0 bitwise (the density term is
	// V_0^2-weighted and all velocity amplitudes are V_0-weighted); used for
	// coupling null-case experiments (constant macros across the interface)
	if (! rest_ic) {
		state.V_0 = PHYS_VELOCITY;
		state.k = 2 * TNL::pi / (N * PHYS_DL);
	}

	state.nse.physFinalTime = phys_final_time;	// [s]
	state.cnt[PRINT].period = 0.01;
	state.cnt[OUT3D].period = 0.05;

	// convective-time mode (the sim_4 convention, 2026-08-18): with
	// --convective-times N the run lasts N * L/V_0 seconds and the cadence
	// scales with the final time (PRINT at final/1000, OUT3D at final/10)
	if (convective_times > 0) {
		const real convective_time = PHYS_HEIGHT / PHYS_VELOCITY;	// [s]
		state.nse.physFinalTime = convective_times * convective_time;
		state.cnt[PRINT].period = state.nse.physFinalTime / 1000;
		state.cnt[OUT3D].period = state.nse.physFinalTime / 10;
	}

	// per-iteration frame cadence (--out3d-iter-period N): write the OUT3D
	// macroscopic frame every N fine iterations, independent of the
	// physical/convective-time cadence above; replaces the uncommitted
	// "cnt[OUT3D].period = PHYS_DT" probe hack. The fine (level-1) timestep
	// is PHYS_DT / 2 (2:1 subcycling) and the OUT3D hook in
	// State_AMR::AfterSimUpdate fires at most once per coarse step, so
	// N = 1 and N = 2 both write every coarse step
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
void run(
	const std::string& adios_config,
	int resolution,
	int max_level = 1,
	float lattice_viscosity = -1.0f,
	float phys_final_time = 0.5f,
	float convective_times = 0.0f,
	bool write_dfs = false,
	int out3d_iter_period = 0,
	bool rest_ic = false
)
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

	sim<NSE_CONFIG>(adios_config, resolution, max_level, lattice_viscosity, phys_final_time, convective_times, write_dfs, out3d_iter_period, rest_ic);
}

int main(int argc, char** argv)
{
	TNLMPI_INIT mpi(argc, argv);

	argparse::ArgumentParser program("sim_AMR");
	program.add_description("2-level AMR Taylor-Green vortex simulation using incompressible Navier-Stokes equations.");
	program.add_argument("--adios-config").help("path to ADIOS2 configuration file").default_value(std::string("adios2.xml")).nargs(1);
	program.add_argument("--resolution").help("resolution of the lattice").scan<'i', int>().default_value(1).nargs(1);
	program.add_argument("--max-level").help("maximum AMR refinement level (0 = uniform)").scan<'i', int>().default_value(1).nargs(1);
	program.add_argument("--lattice-viscosity")
		.help("override lattice viscosity [dx^2/dt] (for uniform reference runs)")
		.scan<'f', float>()
		.default_value(-1.0f)
		.nargs(1);
	program.add_argument("--phys-final-time").help("physical final time [s]").scan<'f', float>().default_value(0.5f).nargs(1);
	program.add_argument("--convective-times")
		.help("run N convective times L/V_0 with the sim_4 output cadence (overrides --phys-final-time; 0 = off)")
		.scan<'f', float>()
		.default_value(0.0f)
		.nargs(1);
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
	program.add_argument("--rest-ic").help("rest initial condition (V_0 = 0: rho = 1, u = 0 everywhere) for coupling null-case experiments").default_value(false).implicit_value(true);

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
	const auto convective_times = program.get<float>("--convective-times");
	const auto write_dfs = program.get<bool>("--write-dfs");
	const auto out3d_iter_period = program.get<int>("--out3d-iter-period");
	const auto rest_ic = program.get<bool>("--rest-ic");

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
	run<TraitsSP>(adios_config, resolution, max_level, lattice_viscosity, phys_final_time, convective_times, write_dfs, out3d_iter_period, rest_ic);

	return 0;
}
