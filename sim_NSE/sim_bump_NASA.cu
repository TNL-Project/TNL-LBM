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

	using idx = typename TRAITS::idx;
	using real = typename TRAITS::real;
	using dreal = typename TRAITS::dreal;
	using point_t = typename TRAITS::point_t;
	using lat_t = Lattice<3, real, idx>;

	real lbm_inflow_vx = 0;

	void setupBoundaries() override
	{
		nse.setBoundaryX(0, BC::GEO_INFLOW);						  // left
		nse.setBoundaryX(1, BC::GEO_INFLOW);						  // left
		nse.setBoundaryX(2, BC::GEO_INFLOW);						  // left
		nse.setBoundaryX(nse.lat.global.x() - 1, BC::GEO_OUTFLOW_RIGHT);  // right
		nse.setBoundaryX(nse.lat.global.x() - 2, BC::GEO_OUTFLOW_RIGHT);  // right
		nse.setBoundaryX(nse.lat.global.x() - 3, BC::GEO_OUTFLOW_RIGHT);  // right

		//nse.setBoundaryX(0,                      BC::GEO_PERIODIC);						  // left
		//nse.setBoundaryX(1,                      BC::GEO_PERIODIC);						  // left
		//nse.setBoundaryX(2,                      BC::GEO_PERIODIC);						  // left
		//nse.setBoundaryX(nse.lat.global.x() - 1, BC::GEO_PERIODIC);  // right
		//nse.setBoundaryX(nse.lat.global.x() - 2, BC::GEO_PERIODIC);  // right
		//nse.setBoundaryX(nse.lat.global.x() - 3, BC::GEO_PERIODIC);  // right

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

		for (int px = 0; px <= nse.lat.global.x(); px++){
		for (int py = 0; py <= nse.lat.global.y(); py++){
		for (int pz = 0; pz <= nse.lat.global.z(); pz++){
			float x = nse.lat.lbm2physX(px);
			float y = nse.lat.lbm2physY(py);
			float z = nse.lat.lbm2physZ(pz);
			// Shift the x for non-symmetric bump
			float xshift = x + 0.3*pow(sin(PI*y),4);

			// Bump area
			if(xshift > 0.3 && xshift  < 1.2){
				float fxy = 1.*pow(sin(PI*xshift/0.9 - PI/3.),4); // Wall position
				if(fxy > z){
					nse.setMap(px, py, pz, BC::GEO_WALL);
				}
			}
		}}}
		// TODO: set the symmetric walls
	}



	double get_drag(const double normalDerivativeCoefficient = 2., const bool dynamicViscosity = false){
		double C_drag = 0.;
		const double visc = (double)nse.lat.physViscosity;
		const double Uoverline = (double)nse.lat.lbm2physVelocity(nse.lat.data.inflow_vx)*2./3.;
		const double D = (double)0.1;
		const double delta_x = (double)nse.lat.physDl;
		const int NoDV = NSE::LBM_KS::NoDV;
		const double T0 = NoDV == 3  ? (double)0.6979533220196830882384091 : (double)1./3;

		for (int x=NoDV; x<nse.lat.global.x()-NoDV; x++){
		for (int y=NoDV; y<nse.lat.global.y()-NoDV; y++){
		for (int z=NoDV; y<nse.lat.global.z()-NoDV; z++){
			//int gi = POS(x, y, nse.lat.X, nse.lat.Y);
			if(nse.data.map(x, y, z) == BC::GEO_WALL){
				// n = (-1,0,0)
		 		if(nse.data.map(x-1, y, z) != BC::GEO_WALL){
					double rho = (double)nse.data.macro(MACRO::e_rho,x-1,y,z);
					double vy = (double)nse.lat.lbm2physVelocity(nse.data.macro(MACRO::e_vy,x-1,y,z));
					double vx = (double)nse.lat.lbm2physVelocity(nse.data.macro(MACRO::e_vx,x-1,y,z));
					// pressure
					C_drag += nse.lat.lbm2physVelocity(nse.lat.lbm2physVelocity(T0*(rho-1)));
					// -T_11
					C_drag += rho*normalDerivativeCoefficient*visc*vx/(delta_x/2);
				}
				//// n = (1,0,0)
				//if(nse.lat.map(gi) != BC::GEO_WALL){
				//	double rho = (double)nse.lat.macro(MACRO::e_rho,x+1,y);
				//	double vy = (double)nse.lat.lbm2physVelocity(block.hmacro(MACRO::e_vy,x+1,y));
				//	double vx = (double)nse.lat.lbm2physVelocity(block.hmacro(MACRO::e_vx,x+1,y));
				//	// pressure
				//	C_drag -= nse.lat.lbm2physVelocity(nse.lat.lbm2physVelocity(T0*(rho-1)));
				//	// T_11
				//	C_drag += rho*normalDerivativeCoefficient*visc*vx/(delta_x/2);
				//}
				//// n = (0,-1,0)
				//if(nse.lat.map(gi) != BC::GEO_WALL){
				//	double rho = (double)nse.lat.macro(MACRO::e_rho,x,y-1);
				//	double vx = (double)nse.lat.lbm2physVelocity(nse.lat.macro(MACRO::e_vx,x,y-1));
				//	// -T_21
				//	C_drag += rho*visc*vx/(delta_x/2);
				//}
				//// n = (0,1,0)
				//if(nse.lat.map(gi) != BC::GEO_WALL){
				//	double rho = (double)nse.lat.macro(MACRO::e_rho,x,y+1);
				//	double vx = (double)nse.lat.lbm2physVelocity(nse.lat.macro(MACRO::e_vx,x,y+1));
				//	// T_21
				//	C_drag += rho*visc*vx/(delta_x/2);
				//}
			}
    	}}}
		return 2*delta_x*C_drag/(Uoverline*Uoverline)/D;
	}

	void probe1() override {
		// testing log
		//spdlog::info(
		//	"Reynolds = {:f} lbmvel {:f} physvel {:f}",
		//	get_drag()
		//);
		real local_drag = 0;
		//real local_la1sum=0;
		//real local_la2sum=0;
		for (int x = nse.blocks.front().offset.x() + 1; x < nse.blocks.front().offset.x() + nse.blocks.front().local.x() - 1; x++) {
		for (int y = nse.blocks.front().offset.y() + 1; y < nse.blocks.front().offset.y() + nse.blocks.front().local.y() - 1; y++) {
		for (int z = nse.blocks.front().offset.z() + 1; z < nse.blocks.front().offset.z() + nse.blocks.front().local.z() - 1; z++) {
			//if(nse.blocks.front().map(x, y, z) == BC::GEO_WALL){
			//	// n = (-1,0,0)
		 	//	if(nse.blocks.front().map(x-1, y, z) != BC::GEO_WALL){
					double rho = (double)nse.blocks.front().hmacro(MACRO::e_rho,x-1,y,z);
					double vy = (double)nse.lat.lbm2physVelocity(nse.blocks.front().hmacro(MACRO::e_vy,x-1,y,z));
					double vx = (double)nse.lat.lbm2physVelocity(nse.blocks.front().hmacro(MACRO::e_vx,x-1,y,z));
					// pressure
					//C_drag += nse.lat.lbm2physVelocity(nse.lat.lbm2physVelocity(T0*(rho-1)));
					// -T_11
					//local_drag += rho*normalDerivativeCoefficient*visc*vx/(delta_x/2);
					local_drag += rho;
			//	}
			//}
		}}}

		real drag = TNL::MPI::reduce(local_drag, MPI_SUM, MPI_COMM_WORLD);
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
	real PHYS_LENGTH = 10.; // length in some units (NASA does not specify)
	real PHYS_HEIGHT = 5.;		  // domain height (physical)
	real PHYS_DEPTH = 0.5;		  // domain depth (physical)
	// TODO: solve the rounding of pixels to have it precise
	int X = floor(PHYS_LENGTH * RESOLUTION * block_size);  // width in pixels
	int Y = floor(PHYS_DEPTH  * RESOLUTION * block_size);  // height in pixels --- top and bottom walls NoDV px
	int Z = floor(PHYS_HEIGHT * RESOLUTION * block_size); // depth in pixels --- top and bottom walls  NoDV px
	real LBM_VISCOSITY = 0.00001;
	real PHYS_VISCOSITY = 1.5e-5;
	real PHYS_VELOCITY = 1.0;
	real PHYS_DL = PHYS_HEIGHT / ((real) Z - 6); // naive fullway bounce-back
	real PHYS_DT = LBM_VISCOSITY / PHYS_VISCOSITY * PHYS_DL * PHYS_DL;	//PHYS_HEIGHT/(real)LBM_HEIGHT;
	point_t PHYS_ORIGIN = {-PHYS_LENGTH/2., -PHYS_DEPTH, 0.};

	// initialize the lattice
	lat_t lat;
	lat.global = typename lat_t::CoordinatesType(X, Y, Z);
	lat.physOrigin = PHYS_ORIGIN;
	lat.physDl = PHYS_DL;
	lat.physDt = PHYS_DT;
	lat.physViscosity = PHYS_VISCOSITY;

	const std::string state_id = fmt::format("sim_bump_NASA_res{:02d}_np{:03d}", RESOLUTION, TNL::MPI::GetSize(MPI_COMM_WORLD));
	StateLocal<NSE> state(state_id, MPI_COMM_WORLD, lat);

	// problem parameters
	state.lbm_inflow_vx = lat.phys2lbmVelocity(PHYS_VELOCITY);

	state.nse.physFinalTime = 1;
	state.cnt[PRINT].period = 0.001;

	// add cuts
	state.cnt[VTK2D].period = 0.001;
	state.add2Dcut_X(X / 2, "cutsX/cut_X");
	state.add2Dcut_Y(Y / 2, "cutsY/cut_Y");
	state.add2Dcut_Z(Z / 2, "cutsZ/cut_Z");

	state.cnt[VTK3D].period = 0.1;
	state.cnt[VTK3DCUT].period = 0.1;
	state.add3Dcut(X / 4, Y / 4, Z / 4, X / 2, Y / 2, Z / 2, 2, "box");

	state.cnt[PROBE1].period = 0.001;

	execute(state);

	return 0;
}

template <typename TRAITS = TraitsSP>
void run(int RES)
{
	// D3Q27
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

	// D3Q343
	//using COLL = D3Q343_SRT<TRAITS, D3Q343_EQ<TRAITS>>;
	//using NSE_CONFIG = LBM_CONFIG<
	//	TRAITS,
	//	D3Q343_KernelStruct,
	//	NSE_Data_ConstInflow<TRAITS>,
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
