#include <argparse/argparse.hpp>
#include <utility>

// As of now, enum and sync direction are specific for different models and need to be included before core!!!
#include "lbm3d/d3q27/defs.h"
//#include "lbm3d/d3q343/defs.h"
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
	using State<NSE>::id;

	using idx = typename TRAITS::idx;
	using real = typename TRAITS::real;
	using dreal = typename TRAITS::dreal;
	using point_t = typename TRAITS::point_t;
	using lat_t = Lattice<3, real, idx>;

	real lbm_inflow_vx = 0;
	real inflow_g = 0;
	bool NoDV = 1;

	void setupBoundaries() override
	{
		nse.setBoundaryX(0,                      BC::GEO_PERIODIC);						  // left
		nse.setBoundaryX(1,                      BC::GEO_PERIODIC);						  // left
		nse.setBoundaryX(2,                      BC::GEO_PERIODIC);						  // left
		nse.setBoundaryX(nse.lat.global.x() - 1, BC::GEO_PERIODIC);  // right
		nse.setBoundaryX(nse.lat.global.x() - 2, BC::GEO_PERIODIC);  // right
		nse.setBoundaryX(nse.lat.global.x() - 3, BC::GEO_PERIODIC);  // right

		const int margin = 0;
		nse.setBoundaryZ(1-1+margin,                      BC::GEO_WALL);						 // top
		nse.setBoundaryZ(2-1+margin,                      BC::GEO_WALL);						 // top
		nse.setBoundaryZ(3-1+margin,                      BC::GEO_WALL);						 // top
		nse.setBoundaryZ(nse.lat.global.z() - 2+1-margin, BC::GEO_WALL);	 // bottom
		nse.setBoundaryZ(nse.lat.global.z() - 3+1-margin, BC::GEO_WALL);	 // bottom
		nse.setBoundaryZ(nse.lat.global.z() - 4+1-margin, BC::GEO_WALL);	 // bottom
		nse.setBoundaryY(1-1+margin, 					  BC::GEO_WALL);						 // back
		nse.setBoundaryY(2-1+margin, 					  BC::GEO_WALL);						 // back
		nse.setBoundaryY(3-1+margin, 					  BC::GEO_WALL);						 // back
		nse.setBoundaryY(nse.lat.global.y() - 2+1-margin, BC::GEO_WALL);	 // front
		nse.setBoundaryY(nse.lat.global.y() - 3+1-margin, BC::GEO_WALL);	 // front
		nse.setBoundaryY(nse.lat.global.y() - 4+1-margin, BC::GEO_WALL);	 // front
	}
	void probe1() override {

  	}


	bool outputData(const BLOCK& block, int index, int dof, char* desc, idx x, idx y, idx z, real& value, int& dofs) override
	{
		int k = 0;
		if (index == k++)
			return vtk_helper("lbm_density", block.hmacro(MACRO::e_rho, x, y, z), 1, desc, value, dofs);
		if (index == k++) {
			switch (dof) {
				case 0:
					return vtk_helper("velocity", nse.lat.lbm2physVelocity(block.hmacro(MACRO::e_vx, x, y, z)), 3, desc, value, dofs);
				case 1:
					return vtk_helper("velocity", nse.lat.lbm2physVelocity(block.hmacro(MACRO::e_vy, x, y, z)), 3, desc, value, dofs);
				case 2:
					return vtk_helper("velocity", nse.lat.lbm2physVelocity(block.hmacro(MACRO::e_vz, x, y, z)), 3, desc, value, dofs);
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
			block.data.InitPoint = nse.lat.phys2lbmPoint(nse.lat.physOrigin);
			block.data.inflow_g = inflow_g;
			block.data.inflow_y = nse.lat.global.y();
			block.data.inflow_z = nse.lat.global.z();
			if(NSE::LBM_KS::NoDV == 3){
				block.data.no1oT0 = 1./0.6979533220196830882384091; // FOR D2Q49
			}
			
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
	real PHYS_LENGTH = 2.2; // domain length
	real PHYS_HEIGHT = 0.41;		  // domain height (physical)
	real PHYS_DEPTH = 0.41;		  // domain depth (physical)
	// TODO: solve the rounding of pixels to have it precise
	int X = floor(PHYS_LENGTH * RESOLUTION * block_size);  // width in pixels
	int Y = floor(PHYS_DEPTH  * RESOLUTION * block_size);  // height in pixels --- top and bottom walls NoDV px
	int Z = floor(PHYS_HEIGHT * RESOLUTION * block_size); // depth in pixels --- top and bottom walls  NoDV px
	real LBM_VISCOSITY = 0.001;
	real PHYS_VISCOSITY = 0.001;
	real PHYS_VELOCITY = 0.3;
	real PHYS_DL = PHYS_HEIGHT / ((real) Z - 6); // naive fullway bounce-back
	real PHYS_DT = LBM_VISCOSITY / PHYS_VISCOSITY * PHYS_DL * PHYS_DL;	//PHYS_HEIGHT/(real)LBM_HEIGHT;
	point_t PHYS_ORIGIN = {0., -5./2*PHYS_DL, -5./2*PHYS_DL};

	real g = PHYS_VISCOSITY*PHYS_VELOCITY/(PHYS_HEIGHT*PHYS_HEIGHT*0.25*0.5);

	// initialize the lattice
	lat_t lat;
	lat.global = typename lat_t::CoordinatesType(X, Y, Z);
	lat.physOrigin = PHYS_ORIGIN;
	lat.physDl = PHYS_DL;
	lat.physDt = PHYS_DT;
	lat.physViscosity = PHYS_VISCOSITY;

	const std::string state_id = fmt::format("sim_schafer_turek_res{:02d}_np{:03d}", RESOLUTION, TNL::MPI::GetSize(MPI_COMM_WORLD));
	StateLocal<NSE> state(state_id, MPI_COMM_WORLD, lat);

	// problem parameters
	state.lbm_inflow_vx = lat.phys2lbmVelocity(PHYS_VELOCITY);
	state.inflow_g = lat.phys2lbmForce(g);

	state.nse.physFinalTime = 20;
	state.cnt[PRINT].period = 1;

	state.loadState();
	state.wallTime = 10000;
	// add cuts
	state.cnt[VTK2D].period = 1;
	state.add2Dcut_X(X / 2, "cutsX/cut_X");
	state.add2Dcut_Y(Y / 2, "cutsY/cut_Y");
	state.add2Dcut_Z(Z / 2, "cutsZ/cut_Z");

	state.cnt[VTK3D].period = 1;
	state.cnt[VTK3DCUT].period = 1;
	state.add3Dcut(X / 4, Y / 4, Z / 4, X / 2, Y / 2, Z / 2, 2, "box");

	state.cnt[PROBE1].period = 0.1;

	state.updateKernelData();
	state.updateKernelVelocities();
	typename NSE::LBM_KS KS;

	// Debug output of the inflow
	//const std::string inflow_profile_output = fmt::format("{}/velocities.csv",state_id);
	//FILE *fp = fopen(inflow_profile_output.c_str(), "w");
	FILE *fp = fopen("velocities.csv", "w");
	fprintf(fp, "y,z,vx\n");
	for (int y = 0; y < Y; y++) {
	    for (int z = 0; z < Z; z++) {
	        state.nse.blocks[0].data.inflow(KS, 0, y, z);
	        fprintf(fp, "%d,%d,%e\n", y, z, KS.vx);
	    }
	}
	fclose(fp);


	execute(state);

	return 0;
}

template <typename TRAITS = TraitsDP>
void run(int RES)
{
	// D3Q27
	using COLL = D3Q27_CUM<TRAITS, D3Q27_EQ_INV_CUM<TRAITS>>;
	using NSE_CONFIG = LBM_CONFIG<
		TRAITS,
		D3Q27_KernelStruct,
		//NSE_Data_Parabolic_yconst<TRAITS>,
		NSE_Data_DoubleParabolic<TRAITS>,
		COLL,
		typename COLL::EQ,
		D3Q27_STREAMING<TRAITS>,
		D3Q27_BC_All,
		D3Q27_MACRO_Default<TRAITS>>;

	// D3Q343
	//using COLL = D3Q343_ELBM<TRAITS, D3Q343_EQ<TRAITS>>;
	//using NSE_CONFIG = LBM_CONFIG<
	//	TRAITS,
	//	D3Q343_KernelStruct,
	//	NSE_Data_DoubleParabolic<TRAITS>,
	//	COLL,
	//	typename COLL::EQ,
	//	D3Q343_STREAMING<TRAITS>,
	//	D3Q343_BC_All,
	//	D3Q343_MACRO_Default<TRAITS>>;

	sim<NSE_CONFIG>(RES);
}

int main(int argc, char** argv)
{
	TNLMPI_INIT mpi(argc, argv);

	argparse::ArgumentParser program("sim_bump_NASA");
	program.add_description("Simulation of a bump in a channel.");
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
