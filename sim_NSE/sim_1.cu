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

	using State<NSE>::nse;
	using State<NSE>::vtk_helper;

	using idx = typename TRAITS::idx;
	using real = typename TRAITS::real;
	using dreal = typename TRAITS::dreal;
	using point_t = typename TRAITS::point_t;
	using lat_t = Lattice<3, real, idx>;

	real lbm_inflow_vx = 0;

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

	bool outputData(const BLOCK& block, int index, int dof, char* desc, idx x, idx y, idx z, real& value, int& dofs) override
	{
		int k = 0;
		if (index == k++)
			return vtk_helper("lbm_density", block.hmacro(MACRO::e_rho, x, y, z), 1, desc, value, dofs);
		if (index == k++) {
			switch (dof) {
				case 0:
					return vtk_helper("velocity", block.hmacro(MACRO::e_vx, x, y, z), 3, desc, value, dofs);
				case 1:
					return vtk_helper("velocity", block.hmacro(MACRO::e_vy, x, y, z), 3, desc, value, dofs);
				case 2:
					return vtk_helper("velocity", block.hmacro(MACRO::e_vz, x, y, z), 3, desc, value, dofs);
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

	StateLocal(const std::string& id, const TNL::MPI::Comm& communicator, lat_t lat)
	: State<NSE>(id, communicator, std::move(lat))
	{}
};

template <typename NSE>
int sim(int RESOLUTION = 2)
{
	using idx = typename NSE::TRAITS::idx;
	using real = typename NSE::TRAITS::real;
	using point_t = typename NSE::TRAITS::point_t;
	using lat_t = Lattice<3, real, idx>;

	int block_size = 32;
	int X = 128 * RESOLUTION;  // width in pixels
	//int Y = 41*RESOLUTION;// height in pixels --- top and bottom walls 1px
	//int Z = 41*RESOLUTION;// height in pixels --- top and bottom walls 1px
	int Y = block_size * RESOLUTION;  // height in pixels --- top and bottom walls 1px
	int Z = Y;						  // height in pixels --- top and bottom walls 1px
	real LBM_VISCOSITY = 0.00001;	  //1.0/6.0; /// GIVEN: optimal is 1/6
	real PHYS_HEIGHT = 0.41;		  // [m] domain height (physical)
	real PHYS_VISCOSITY = 1.5e-5;	  // [m^2/s] fluid viscosity .... blood?
	//real PHYS_VELOCITY = 2.25; // m/s ... will be multip
	real PHYS_VELOCITY = 1.0;
	real PHYS_DL = PHYS_HEIGHT / ((real) Y - 2);
	real PHYS_DT = LBM_VISCOSITY / PHYS_VISCOSITY * PHYS_DL * PHYS_DL;	//PHYS_HEIGHT/(real)LBM_HEIGHT;
	point_t PHYS_ORIGIN = {0., 0., 0.};

	// initialize the lattice
	lat_t lat;
	lat.global = typename lat_t::CoordinatesType(X, Y, Z);
	lat.physOrigin = PHYS_ORIGIN;
	lat.physDl = PHYS_DL;
	lat.physDt = PHYS_DT;
	lat.physViscosity = PHYS_VISCOSITY;

	const std::string state_id = fmt::format("sim_1_res{:02d}_np{:03d}", RESOLUTION, TNL::MPI::GetSize(MPI_COMM_WORLD));
	StateLocal<NSE> state(state_id, MPI_COMM_WORLD, lat);

	if (! state.canCompute())
		return 0;

	// problem parameters
	state.lbm_inflow_vx = lat.phys2lbmVelocity(PHYS_VELOCITY);

	state.nse.physFinalTime = 1.0;
	state.cnt[PRINT].period = 0.001;
	// test
	//state.cnt[PRINT].period = 100*PHYS_DT;
	//state.nse.physFinalTime = 1000*PHYS_DT;
	//state.cnt[VTK3D].period = 1000*PHYS_DT;
	//state.cnt[SAVESTATE].period = 600;  // save state every [period] of wall time
	//state.wallTime = 60;
	// RCI
	//state.nse.physFinalTime = 0.5;
	//state.cnt[VTK3D].period = 0.5;
	//state.cnt[SAVESTATE].period = 3600;  // save state every [period] of wall time
	//state.wallTime = 3600 * 23.5;

	// add cuts
	state.cnt[VTK2D].period = 0.001;
	state.add2Dcut_X(X / 2, "cutsX/cut_X");
	state.add2Dcut_X(X / 4, "cutsX/cut_X4");
	state.add2Dcut_Y(Y / 2, "cutsY/cut_Y");
	state.add2Dcut_Z(Z / 2, "cutsZ/cut_Z");

	state.cnt[VTK3D].period = 0.001;
	state.cnt[VTK3DCUT].period = 0.001;
	state.add3Dcut(X / 4, Y / 4, Z / 4, X / 2, Y / 2, Z / 2, 2, "box");

	execute(state);

	return 0;
}

template <typename TRAITS = TraitsSP>
void run(int RES)
{
	//	using COLL = D3Q27_CUM< TRAITS >;
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

	sim<NSE_CONFIG>(RES);
}

int main(int argc, char** argv)
{
	TNLMPI_INIT mpi(argc, argv);

	argparse::ArgumentParser program("sim_1");
	program.add_description("Simple incompressible Navier-Stokes simulation example.");
	program.add_argument("resolution").help("resolution of the lattice").scan<'i', int>().default_value(1);

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

	run(resolution);

	return 0;
}
