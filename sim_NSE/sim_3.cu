#include <argparse/argparse.hpp>
#include <utility>

#include "lbm3d/core.h"
#include "lbm3d/obstacles_lbm.h"

// ball in 3D

template <typename NSE>
struct StateLocal : State<NSE>
{
	using TRAITS = typename NSE::TRAITS;
	using BC = typename NSE::BC;
	using MACRO = typename NSE::MACRO;
	using BLOCK = LBM_BLOCK<NSE>;

	using State<NSE>::nse;
	using State<NSE>::id;

	using idx = typename TRAITS::idx;
	using real = typename TRAITS::real;
	using dreal = typename TRAITS::dreal;
	using point_t = typename TRAITS::point_t;
	using lat_t = Lattice<3, real, idx>;

	dreal lbm_inflow_vx = 0;
	real ball_diameter = 0;
	point_t ball_c;

	void setupBoundaries() override
	{
		nse.setBoundaryX(0, BC::GEO_INFLOW_LEFT);								 // left
		nse.setBoundaryX(nse.lat.global.x() - 1, BC::GEO_OUTFLOW_RIGHT_INTERP);	 // right
		nse.setBoundaryY(0, BC::GEO_INFLOW_LEFT);								 // back
		nse.setBoundaryY(nse.lat.global.y() - 1, BC::GEO_INFLOW_LEFT);			 // front
		nse.setBoundaryZ(0, BC::GEO_INFLOW_LEFT);								 // top
		nse.setBoundaryZ(nse.lat.global.z() - 1, BC::GEO_INFLOW_LEFT);			 // bottom

		lbmDrawSphere(nse, BC::GEO_WALL, ball_c, ball_diameter * 0.5);
	}

	bool outputData(const BLOCK& block, int index, int dof, idx x, idx y, idx z, OutputDataDescriptor<dreal>& desc) override
	{
		int k = 0;
		if (index == k++)
			return desc.set("lbm_density", block.hmacro(MACRO::e_rho, x, y, z), 1);
		if (index == k++)
			return desc.set("lbm_density_fluctuation", block.hmacro(MACRO::e_rho, x, y, z) - 1.0, 1);
		if (index == k++) {
			switch (dof) {
				case 0:
					return desc.set("lbm_velocity", block.hmacro(MACRO::e_vx, x, y, z), 3);
				case 1:
					return desc.set("lbm_velocity", block.hmacro(MACRO::e_vy, x, y, z), 3);
				case 2:
					return desc.set("lbm_velocity", block.hmacro(MACRO::e_vz, x, y, z), 3);
			}
		}
		if (index == k++) {
			switch (dof) {
				case 0:
					return desc.set("velocity", nse.lat.lbm2physVelocity(block.hmacro(MACRO::e_vx, x, y, z)), 3);
				case 1:
					return desc.set("velocity", nse.lat.lbm2physVelocity(block.hmacro(MACRO::e_vy, x, y, z)), 3);
				case 2:
					return desc.set("velocity", nse.lat.lbm2physVelocity(block.hmacro(MACRO::e_vz, x, y, z)), 3);
			}
		}
		return false;
	}

	void probe1() override
	{
		spdlog::info(
			"Reynolds = {:f} lbmvel {:f} physvel {:f}",
			lbm_inflow_vx * ball_diameter / nse.lat.physDl / nse.lat.lbmViscosity(),
			lbm_inflow_vx,
			nse.lat.lbm2physVelocity(lbm_inflow_vx)
		);
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
int sim(int RES, double Re, const std::string& adiosConfigPath)
{
	using idx = typename NSE::TRAITS::idx;
	using real = typename NSE::TRAITS::real;
	using point_t = typename NSE::TRAITS::point_t;
	using lat_t = Lattice<3, real, idx>;

	int block_size = 32;
	real ball_diameter = 0.10;						   // [m]
	real real_domain_height = 11 * ball_diameter;	   // [m]
	real real_domain_length = 2 * real_domain_height;  // [m]
	idx LBM_Y = RES * block_size;
	idx LBM_Z = LBM_Y;
	real PHYS_DL = real_domain_height / ((real) LBM_Y - 2.0);
	idx LBM_X = (int) (real_domain_length / PHYS_DL) + 2;
	point_t PHYS_ORIGIN = {0., 0., 0.};

	real PHYS_VISCOSITY = 0.001;  // [m^2/s]
	real PHYS_VELOCITY = Re * PHYS_VISCOSITY / ball_diameter;

	real LBM_VISCOSITY = 0.001;

	//fmt::print("input phys velocity {:f}\ninput lbm velocity {:f}\nRe {:f}\nlbm viscosity {:f}\nphys viscosity {:f}\n", i_PHYS_VELOCITY,
	//	i_LBM_VELOCITY, i_Re, i_LBM_VISCOSITY, i_PHYS_VISCOSITY);
	real PHYS_DT = LBM_VISCOSITY / PHYS_VISCOSITY * PHYS_DL * PHYS_DL;

	// initialize the lattice
	lat_t lat;
	lat.global = typename lat_t::CoordinatesType(LBM_X, LBM_Y, LBM_Z);
	lat.physOrigin = PHYS_ORIGIN;
	lat.physDl = PHYS_DL;
	lat.physDt = PHYS_DT;
	lat.physViscosity = PHYS_VISCOSITY;

	const std::string state_id = fmt::format("sim_3_{}_res_{}_Re_{}", NSE::COLL::id, RES, Re);
	StateLocal<NSE> state(state_id, MPI_COMM_WORLD, lat, adiosConfigPath);

	if (! state.canCompute())
		return 0;

	// set problem parameters
	state.ball_c[0] = 2 * ball_diameter;		 // [m]
	state.ball_c[1] = 0.5 * real_domain_height;	 // [m]
	state.ball_c[2] = 0.5 * real_domain_height;	 // [m]
	state.ball_diameter = ball_diameter;		 // [m]
	state.nse.physCharLength = ball_diameter;	 // [m]
	state.lbm_inflow_vx = state.nse.lat.phys2lbmVelocity(PHYS_VELOCITY);
	//state.nse.physFluidDensity = 1000.0; // [kg/m^3]

	spdlog::info("Reynolds = {:f} lbmvel {:f} physvel {:f}", Re, state.lbm_inflow_vx, PHYS_VELOCITY);

	state.cnt[PRINT].period = 0.1;
	state.nse.physFinalTime = 30.0;

	//state.cnt[VTK3D].period = 1.0;
	state.cnt[VTK2D].period = 0.1;
	state.cnt[PROBE1].period = 0.1;

	// add cuts
	state.add2Dcut_X(LBM_X / 2, "cut_X");
	//state.add2Dcut_X(2*ball_diameter/PHYS_DL,"cut_Xball");
	state.add2Dcut_Y(LBM_Y / 2, "cut_Y");
	state.add2Dcut_Z(LBM_Z / 2, "cut_Z");

	execute(state);

	return 0;
}

template <typename TRAITS = TraitsSP>
void run(int res, double Re, const std::string& adiosConfigPath)
{
	using COLL = D3Q27_CUM<TRAITS>;
	using NSE_CONFIG = LBM_CONFIG<
		TRAITS,
		D3Q27_KernelStruct,
		NSE_Data_ConstInflow<TRAITS>,
		COLL,
		typename COLL::EQ,
		D3Q27_STREAMING<TRAITS>,
		D3Q27_BC_All,
		D3Q27_MACRO_Default<TRAITS>>;

	sim<NSE_CONFIG>(res, Re, adiosConfigPath);
}

int main(int argc, char** argv)
{
	TNLMPI_INIT mpi(argc, argv);

	argparse::ArgumentParser program("sim_3");
	program.add_description("LBM simulation with ball in 3D.");
	program.add_argument("--resolution").help("resolution of the lattice").scan<'i', int>().default_value(1).nargs(1);
	program.add_argument("--Re").help("desired Reynolds number (affects the inflow velocity)").scan<'g', double>().default_value(100.0).nargs(1);
	program.add_argument("--adios-config").help("path to adios2 configuration file").default_value(std::string("adios2.xml")).nargs(1);

	try {
		program.parse_args(argc, argv);
	}
	catch (const std::exception& err) {
		std::cerr << err.what() << '\n';
		std::cerr << program;
		return 1;
	}

	const auto resolution = program.get<int>("--resolution");
	const auto Re = program.get<double>("--Re");
	const auto adiosConfigPath = program.get<std::string>("--adios-config");

	if (resolution < 1) {
		fmt::println(stderr, "CLI error: resolution must be at least 1");
		return 1;
	}
	if (Re < 1) {
		fmt::println(stderr, "CLI error: Re must be at least 1");
		return 1;
	}

	run(resolution, Re, adiosConfigPath);

	return 0;
}
