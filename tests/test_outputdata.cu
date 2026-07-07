#include <argparse/argparse.hpp>

#include "lbm3d/core.h"

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
		nse.setBoundaryX(0, BC::GEO_INFLOW_LEFT);
		nse.setBoundaryX(nse.lat.global.x() - 1, BC::GEO_OUTFLOW_RIGHT);
		nse.setBoundaryY(0, BC::GEO_PERIODIC);
		nse.setBoundaryY(nse.lat.global.y() - 1, BC::GEO_PERIODIC);
		nse.setBoundaryZ(0, BC::GEO_PERIODIC);
		nse.setBoundaryZ(nse.lat.global.z() - 1, BC::GEO_PERIODIC);
	}

	[[nodiscard]] std::vector<std::string> getOutputDataNames() const override
	{
		return {"lbm_density", "lbm_density_fluctuation", "velocity_x", "velocity_y", "velocity_z"};
	}

	void outputData(UniformDataWriter<TRAITS>& writer, const BLOCK& block, const idx3d& begin, const idx3d& end) override
	{
		writer.write("lbm_density", getMacroView<TRAITS>(block.hmacro, MACRO::e_rho), begin, end);
		writer.write(
			"lbm_density_fluctuation",
			[&](idx x, idx y, idx z) -> dreal
			{
				return block.hmacro(MACRO::e_rho, x, y, z) - 1.0;
			},
			begin,
			end
		);
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

template <typename TRAITS = TraitsSP>
void run(const std::string& adios_config, int resolution, const std::string& output_kind)
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

	using idx = typename TRAITS::idx;
	using real = typename TRAITS::real;
	using point_t = typename TRAITS::point_t;
	using lat_t = Lattice<3, real, idx>;

	int block_size = 8;
	int X = block_size * resolution;
	int Y = block_size * resolution;
	int Z = block_size * resolution;
	real LBM_VISCOSITY = 0.01;
	real PHYS_VISCOSITY = 1.5e-5;
	real PHYS_HEIGHT = 0.01;
	real PHYS_DL = PHYS_HEIGHT / ((real) Y - 2);
	real PHYS_DT = LBM_VISCOSITY / PHYS_VISCOSITY * PHYS_DL * PHYS_DL;
	point_t PHYS_ORIGIN = {0., 0., 0.};

	lat_t lat;
	lat.global = typename lat_t::CoordinatesType(X, Y, Z);
	lat.physOrigin = PHYS_ORIGIN;
	lat.physDl = PHYS_DL;
	lat.physDt = PHYS_DT;
	lat.physViscosity = PHYS_VISCOSITY;

	const std::string state_id = fmt::format("test_outputdata_res{:02d}_np{:03d}", resolution, TNL::MPI::GetSize(MPI_COMM_WORLD));
	StateLocal<NSE_CONFIG> state(state_id, MPI_COMM_WORLD, lat, adios_config);

	if (! state.canCompute())
		return;

	state.nse.physFinalTime = 1e-5;
	state.cnt[PRINT].period = 1e-5;

	const bool write_3d = output_kind == "all" || output_kind == "3d";
	const bool write_3dcut = output_kind == "all" || output_kind == "3dcut";
	const bool write_2d = output_kind == "all" || output_kind == "2d";

	if (write_2d) {
		state.cnt[OUT2D].period = 5e-6;
		state.add2Dcut_X(X / 2, "cut_X");
		state.add2Dcut_Y(Y / 2, "cut_Y");
		state.add2Dcut_Z(Z / 2, "cut_Z");
	}

	if (write_3d) {
		state.cnt[OUT3D].period = 5e-6;
	}

	if (write_3dcut) {
		state.cnt[OUT3DCUT].period = 5e-6;
		state.add3Dcut(X / 4, Y / 4, Z / 4, X / 2, Y / 2, Z / 2, "box");
	}

	execute(state);
}

int main(int argc, char** argv)
{
	TNLMPI_INIT mpi(argc, argv);

	argparse::ArgumentParser program("test_outputdata");
	program.add_description("Minimal simulation for end-to-end output-data regression testing.");
	program.add_argument("--adios-config").help("path to ADIOS2 configuration file").default_value(std::string("adios2.xml")).nargs(1);
	program.add_argument("--resolution").help("resolution of the lattice").scan<'i', int>().default_value(1).nargs(1);
	program.add_argument("--output-kind")
		.help("which outputs to write: all, 3d, 3dcut, 2d")
		.default_value(std::string("all"))
		.choices("all", "3d", "3dcut", "2d")
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
	const auto output_kind = program.get<std::string>("--output-kind");

	if (resolution < 1) {
		fmt::println(stderr, "CLI error: resolution must be at least 1");
		return 1;
	}

	run(adios_config, resolution, output_kind);

	return 0;
}
