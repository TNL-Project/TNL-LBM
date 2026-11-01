#include <argparse/argparse.hpp>

// umbrella include for the default D3Q27 model (State, LBM_CONFIG, streaming patterns, BCs)
#include "lbm3d/core.h"	 // IWYU pragma: keep

// Minimal simulation used to test checkpointing and simulation restart.
//
// The simulation loop is driven explicitly (instead of calling execute(state))
// so that the checkpoint can be requested at a fixed iteration rather than at
// a wall-clock time. Wall-clock checkpointing is not reproducible, which would
// make the regression test flaky.
template <typename NSE>
struct StateLocal : State<NSE>
{
	using TRAITS = typename NSE::TRAITS;
	using BC = typename NSE::BC;
	using MACRO = typename NSE::MACRO;
	using BLOCK = LBM_BLOCK<NSE>;

	using State<NSE>::nse;

	using idx = typename TRAITS::idx;
	using idx3d = typename TRAITS::idx3d;
	using real = typename TRAITS::real;
	using dreal = typename TRAITS::dreal;
	using lat_t = Lattice<3, real, idx>;

	StateLocal(const std::string& id, const TNL::MPI::Comm& communicator, lat_t lat, const std::string& adiosConfigPath = "adios2.xml")
	: State<NSE>(id, communicator, std::move(lat), adiosConfigPath)
	{}

	void setupBoundaries() override
	{
		nse.setBoundaryX(1, BC::GEO_INFLOW_MOMENT);						  // inflow
		nse.setBoundaryX(nse.lat.global.x() - 2, BC::GEO_OUTFLOW_RIGHT);  // outflow
		nse.setBoundaryY(1, BC::GEO_WALL);								  // bottom
		nse.setBoundaryY(nse.lat.global.y() - 2, BC::GEO_WALL);			  // top
		nse.setBoundaryZ(1, BC::GEO_WALL);								  // front
		nse.setBoundaryZ(nse.lat.global.z() - 2, BC::GEO_WALL);			  // back

		// extra layer needed due to the ghost-layer idiom
		nse.setBoundaryX(0, BC::GEO_NOTHING);
		nse.setBoundaryX(nse.lat.global.x() - 1, BC::GEO_NOTHING);
		nse.setBoundaryY(0, BC::GEO_NOTHING);
		nse.setBoundaryY(nse.lat.global.y() - 1, BC::GEO_NOTHING);
		nse.setBoundaryZ(0, BC::GEO_NOTHING);
		nse.setBoundaryZ(nse.lat.global.z() - 1, BC::GEO_NOTHING);
	}

	[[nodiscard]] std::vector<std::string> getOutputDataNames() const override
	{
		return {"lbm_density", "velocity_x", "velocity_y", "velocity_z"};
	}

	void outputData(UniformDataWriter<TRAITS>& writer, const BLOCK& block, const idx3d& begin, const idx3d& end) override
	{
		writer.write("lbm_density", getMacroView<TRAITS>(block.hmacro, MACRO::e_rho), begin, end);
		writer.write(
			"velocity_x",
			[&](idx x, idx y, idx z) -> dreal
			{
				return nse.lat.lbm2physVelocity(block.hmacro(MACRO::e_vx, x, y, z));
			},
			begin,
			end
		);
		writer.write(
			"velocity_y",
			[&](idx x, idx y, idx z) -> dreal
			{
				return nse.lat.lbm2physVelocity(block.hmacro(MACRO::e_vy, x, y, z));
			},
			begin,
			end
		);
		writer.write(
			"velocity_z",
			[&](idx x, idx y, idx z) -> dreal
			{
				return nse.lat.lbm2physVelocity(block.hmacro(MACRO::e_vz, x, y, z));
			},
			begin,
			end
		);
	}

	void updateKernelVelocities() override
	{
		for (auto& block : nse.blocks) {
			block.data.inflow_vx = nse.lat.phys2lbmVelocity(0.1);
			block.data.inflow_vy = 0;
			block.data.inflow_vz = 0;
		}
	}
};

// Runs the simulation for `iterations` steps. When `checkpoint_after` is a
// positive number smaller than `iterations`, the run stops right after that
// many steps, saves a checkpoint and exits without creating the final output
// (mimicking an interrupted/wall-time-limited run). The subsequent run with
// the same state id resumes from the checkpoint and finishes the simulation.
template <typename TRAITS = TraitsSP>
void run(const std::string& adios_config, int resolution, int iterations, int checkpoint_after)
{
	using COLL = D3Q27_CUM<TRAITS, D3Q27_EQ_INV_CUM<TRAITS>>;

	using NSE_CONFIG = LBM_CONFIG<
		TRAITS,
		D3Q27_KernelStruct,
		NSE_Data_ConstInflow,
		COLL,
		typename COLL::EQ,
		D3Q27_STREAMING<TRAITS>,
		D3Q27_BC_All,
		D3Q27_MACRO_Default<TRAITS>>;

	using idx = typename TRAITS::idx;
	using real = typename TRAITS::real;
	using point_t = typename TRAITS::point_t;
	using lat_t = Lattice<3, real, idx>;

	const int block_size = 8;
	const int X = 2 * block_size * resolution;
	const int Y = block_size * resolution;
	const int Z = block_size * resolution;
	const real LBM_VISCOSITY = 0.01;
	const real PHYS_VISCOSITY = 1.5e-5;
	const real PHYS_HEIGHT = 0.01;
	const real PHYS_DL = PHYS_HEIGHT / ((real) Y - 2);
	const real PHYS_DT = LBM_VISCOSITY / PHYS_VISCOSITY * PHYS_DL * PHYS_DL;
	const point_t PHYS_ORIGIN = {0., 0., 0.};

	lat_t lat;
	lat.global = typename lat_t::CoordinatesType(X, Y, Z);
	lat.physOrigin = PHYS_ORIGIN;
	lat.physDl = PHYS_DL;
	lat.physDt = PHYS_DT;
	lat.physViscosity = PHYS_VISCOSITY;

	const std::string state_id = fmt::format("test_checkpoint_res{:02d}_np{:03d}", resolution, TNL::MPI::GetSize(MPI_COMM_WORLD));
	StateLocal<NSE_CONFIG> state(state_id, MPI_COMM_WORLD, lat, adios_config);

	// Returns false if the results directory is locked or already finished
	// (e.g. when the resumed run has already completed the simulation).
	if (! state.canCompute())
		return;

	// initialize the simulation (this loads the checkpoint when flag.loadstate exists)
	state.SimInit();

	// make snapshot for the initial condition, as in execute(state)
	state.AfterSimUpdate();

	const int stop_iteration = (checkpoint_after > 0 && checkpoint_after < iterations) ? checkpoint_after : iterations;
	while (state.nse.iterations < stop_iteration) {
		state.updateKernelData();
		state.updateKernelVelocities();
		state.SimUpdate();
		state.AfterSimUpdate();
		if (state.nse.terminate)
			throw std::runtime_error(fmt::format("simulation terminated at iteration {}", state.nse.iterations));
	}

	state.waitForPendingIO();
	state.copyAllToHost();

	if (stop_iteration < iterations) {
		// save the checkpoint and exit, leaving flag.loadstate for the next run
		state.saveState();
		spdlog::info("CHECKPOINT_SAVED iteration={}", state.nse.iterations);
	}
	else {
		// write the final state so that the tests can compare the fields of a
		// continuously running simulation with a restarted one
		state.write3D(state.nse.physTime(), 0);
		state.flagCreate("finished");
		state.flagDelete("loadstate");
		spdlog::info("SIMULATION_FINISHED iteration={} time={}", state.nse.iterations, state.nse.physTime());
	}
}

int main(int argc, char** argv)
{
	TNLMPI_INIT mpi(argc, argv);

	argparse::ArgumentParser program("test_checkpoint");
	program.add_description("Minimal simulation for checkpointing/restart regression testing.");
	program.add_argument("--adios-config").help("path to ADIOS2 configuration file").default_value(std::string("adios2.xml")).nargs(1);
	program.add_argument("--resolution").help("resolution of the lattice").scan<'i', int>().default_value(1).nargs(1);
	program.add_argument("--iterations").help("total number of LBM iterations").scan<'i', int>().default_value(200).nargs(1);
	program.add_argument("--checkpoint-after")
		.help("stop after this many iterations, save a checkpoint and exit (0 disables)")
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
	const auto iterations = program.get<int>("--iterations");
	const auto checkpoint_after = program.get<int>("--checkpoint-after");

	if (resolution < 1) {
		fmt::println(stderr, "CLI error: resolution must be at least 1");
		return 1;
	}
	if (iterations < 1) {
		fmt::println(stderr, "CLI error: iterations must be at least 1");
		return 1;
	}
	if (checkpoint_after < 0 || checkpoint_after >= iterations) {
		fmt::println(stderr, "CLI error: checkpoint-after must be in the range [0, iterations)");
		return 1;
	}

	run(adios_config, resolution, iterations, checkpoint_after);

	return 0;
}
