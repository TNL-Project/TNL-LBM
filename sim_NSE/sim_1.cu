#include <argparse/argparse.hpp>
#include <utility>

#include "lbm3d/core.h"

template <typename NSE>
struct StateLocal : State<NSE>
{
	using TRAITS = typename NSE::TRAITS;
	using BC = typename NSE::BC;
	using MACRO = typename NSE::MACRO;
	using BLOCK = LBM_BLOCK<NSE>;

	using State<NSE>::checkpoint;
	using State<NSE>::nse;

	using idx = typename TRAITS::idx;
	using real = typename TRAITS::real;
	using dreal = typename TRAITS::dreal;
	using point_t = typename TRAITS::point_t;
	using lat_t = Lattice<3, real, idx>;

	real lbm_inflow_vx = 0;

	// Override checkpointStateLocal to save/load additional state data
	void checkpointStateLocal(adios2::Mode mode) override
	{
		// Save/load the inflow velocity
		checkpoint.saveLoadAttribute("lbm_inflow_vx", lbm_inflow_vx);

		// You can add any additional state data that needs to be saved/loaded here

		if (mode == adios2::Mode::Read)
			spdlog::info("Checkpoint loaded local state (mode: Read)");
		else
			spdlog::info("Checkpoint saved local state (mode: Write)");
	}

	void setupBoundaries() override
	{
		nse.setBoundaryX(0, BC::GEO_INFLOW_LEFT);						  // left
		nse.setBoundaryX(nse.lat.global.x() - 1, BC::GEO_OUTFLOW_RIGHT);  // right

		nse.setBoundaryZ(1, BC::GEO_WALL);						 // top
		nse.setBoundaryZ(nse.lat.global.z() - 2, BC::GEO_WALL);	 // bottom
		nse.setBoundaryY(1, BC::GEO_WALL);						 // back
		nse.setBoundaryY(nse.lat.global.y() - 2, BC::GEO_WALL);	 // front

		// extra layer needed due to A-A pattern
		nse.setBoundaryZ(0, BC::GEO_NOTHING);						// top
		nse.setBoundaryZ(nse.lat.global.z() - 1, BC::GEO_NOTHING);	// bottom
		nse.setBoundaryY(0, BC::GEO_NOTHING);						// back
		nse.setBoundaryY(nse.lat.global.y() - 1, BC::GEO_NOTHING);	// front

		// draw a wall with a hole
		int cx = floor(0.20 / nse.lat.physDl);
		int width = nse.lat.global.z() / 10;
		for (int px = cx; px <= cx + width; px++)
			for (int pz = 1; pz <= nse.lat.global.z() - 2; pz++)
				for (int py = 1; py <= nse.lat.global.y() - 2; py++)
					if (! (pz >= nse.lat.global.z() * 4 / 10 && pz <= nse.lat.global.z() * 6 / 10 && py >= nse.lat.global.y() * 4 / 10
						   && py <= nse.lat.global.y() * 6 / 10))
					{
						nse.setMap(px, py, pz, BC::GEO_WALL);
					}
	}

	bool outputData(const BLOCK& block, int index, int dof, idx x, idx y, idx z, OutputDataDescriptor<dreal>& desc) override
	{
		int k = 0;
		if (index == k++)
			return desc.set("lbm_density", block.hmacro(MACRO::e_rho, x, y, z), 1);
		if (index == k++) {
			switch (dof) {
				case 0:
					return desc.set("velocity", block.hmacro(MACRO::e_vx, x, y, z), 3);
				case 1:
					return desc.set("velocity", block.hmacro(MACRO::e_vy, x, y, z), 3);
				case 2:
					return desc.set("velocity", block.hmacro(MACRO::e_vz, x, y, z), 3);
			}
		}
		return false;
	}

	void updateKernelVelocities() override
	{
		for (auto& block : nse.blocks) {
			block.data.inflow_vx = lbm_inflow_vx;
			block.data.inflow_vy = 0;
			block.data.inflow_vz = 0;
		}
	}

	StateLocal(const std::string& id, const TNL::MPI::Comm& communicator, lat_t lat, const std::string& adiosConfigPath = "adios2.xml")
	: State<NSE>(id, communicator, std::move(lat), adiosConfigPath)
	{}
};

template <typename NSE>
int sim(int RESOLUTION = 2, const std::string& adiosConfigPath = "adios2.xml")
{
	using idx = typename NSE::TRAITS::idx;
	using real = typename NSE::TRAITS::real;
	using point_t = typename NSE::TRAITS::point_t;
	using lat_t = Lattice<3, real, idx>;

	int block_size = 32;
	int X = 128 * RESOLUTION;		  // width in pixels
	int Y = block_size * RESOLUTION;  // height in pixels --- top and bottom walls 1px
	int Z = Y;						  // height in pixels --- top and bottom walls 1px
	real LBM_VISCOSITY = 0.00001;	  //1.0/6.0; /// GIVEN: optimal is 1/6
	real PHYS_HEIGHT = 0.41;		  // [m] domain height (physical)
	real PHYS_VISCOSITY = 1.5e-5;	  // [m^2/s] fluid viscosity .... blood?
	real PHYS_VELOCITY = 1.0;
	real PHYS_DL = PHYS_HEIGHT / ((real) Y - 2);
	real PHYS_DT = LBM_VISCOSITY / PHYS_VISCOSITY * PHYS_DL * PHYS_DL;
	point_t PHYS_ORIGIN = {0., 0., 0.};

	// Initialize the lattice
	lat_t lat;
	lat.global = typename lat_t::CoordinatesType(X, Y, Z);
	lat.physOrigin = PHYS_ORIGIN;
	lat.physDl = PHYS_DL;
	lat.physDt = PHYS_DT;
	lat.physViscosity = PHYS_VISCOSITY;

	const std::string state_id = fmt::format("sim_1_res{:02d}_np{:03d}", RESOLUTION, TNL::MPI::GetSize(MPI_COMM_WORLD));
	StateLocal<NSE> state(state_id, MPI_COMM_WORLD, lat, adiosConfigPath);

	if (! state.canCompute())
		return 0;

	// Problem parameters
	state.lbm_inflow_vx = lat.phys2lbmVelocity(PHYS_VELOCITY);

	// Set up simulation parameters
	state.nse.physFinalTime = 1.0;	  // Final physical time of simulation
	state.cnt[PRINT].period = 0.001;  // Print info every 0.001 physical time units

	// Enable checkpointing - create a checkpoint every 10 seconds of wall time
	// state.cnt[SAVESTATE].period = 10;
	// state.wallTime = 600;

	// Add visualization cuts
	state.cnt[VTK2D].period = 0.001;
	state.add2Dcut_X(X / 2, "cutsX/cut_X");
	state.add2Dcut_X(X / 4, "cutsX/cut_X4");
	state.add2Dcut_Y(Y / 2, "cutsY/cut_Y");
	state.add2Dcut_Z(Z / 2, "cutsZ/cut_Z");

	state.cnt[VTK3D].period = 0.001;
	state.cnt[VTK3DCUT].period = 0.001;
	state.add3Dcut(X / 4, Y / 4, Z / 4, X / 2, Y / 2, Z / 2, 2, "box");

	// Execute the simulation
	spdlog::info("Starting simulation with checkpointing. Wall time limit: {} seconds", state.wallTime);
	spdlog::info("Creating checkpoints every {} seconds of wall time", state.cnt[SAVESTATE].period);

	execute(state);

	return 0;
}

template <typename TRAITS = TraitsSP>
void run(int RES, const std::string& adiosConfigPath)
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

	sim<NSE_CONFIG>(RES, adiosConfigPath);
}

int main(int argc, char** argv)
{
	TNLMPI_INIT mpi(argc, argv);

	argparse::ArgumentParser program("sim_1");
	program.add_description("Simple incompressible Navier-Stokes simulation example.");
	program.add_argument("resolution").help("resolution of the lattice").scan<'i', int>().default_value(1);

	program.add_argument("--adios-config").help("path to adios2.xml configuration file").default_value(std::string("adios2.xml"));

	try {
		program.parse_args(argc, argv);
	}
	catch (const std::exception& err) {
		std::cerr << err.what() << '\n';
		std::cerr << program;
		return 1;
	}

	const auto resolution = program.get<int>("resolution");
	if (resolution < 1) {
		fmt::println(stderr, "CLI error: resolution must be at least 1");
		return 1;
	}

	const auto adiosConfigPath = program.get<std::string>("adios-config");
	run(resolution, adiosConfigPath);

	return 0;
}
