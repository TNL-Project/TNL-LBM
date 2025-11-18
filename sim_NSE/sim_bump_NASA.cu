#include <argparse/argparse.hpp>
#include <cstdio>
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
		nse.setBoundaryZ(1-1+margin,                      BC::GEO_SYM_TOP);						 // top
		nse.setBoundaryZ(2-1+margin,                      BC::GEO_SYM_TOP);						 // top
		nse.setBoundaryZ(3-1+margin,                      BC::GEO_SYM_TOP);						 // top
		nse.setBoundaryZ(nse.lat.global.z() - 2+1-margin, BC::GEO_SYM_BOTTOM);	 // bottom
		nse.setBoundaryZ(nse.lat.global.z() - 3+1-margin, BC::GEO_SYM_BOTTOM);	 // bottom
		nse.setBoundaryZ(nse.lat.global.z() - 4+1-margin, BC::GEO_SYM_BOTTOM);	 // bottom
		nse.setBoundaryY(1-1+margin, 					  BC::GEO_SYM_BACK);						 // back
		nse.setBoundaryY(2-1+margin, 					  BC::GEO_SYM_BACK);						 // back
		nse.setBoundaryY(3-1+margin, 					  BC::GEO_SYM_BACK);						 // back
		nse.setBoundaryY(nse.lat.global.y() - 2+1-margin, BC::GEO_SYM_FRONT);	 // front
		nse.setBoundaryY(nse.lat.global.y() - 3+1-margin, BC::GEO_SYM_FRONT);	 // front
		nse.setBoundaryY(nse.lat.global.y() - 4+1-margin, BC::GEO_SYM_FRONT);	 // front

		for (int px = 0; px <= nse.lat.global.x(); px++){
		for (int py = 0; py <= nse.lat.global.y(); py++){
		for (int pz = 0; pz <= nse.lat.global.z(); pz++){
			if(isObject(px,py,pz)){
				nse.setMap(px, py, pz, BC::GEO_WALL);
			}
		}}}
		// TODO: set the symmetric walls
	}

	bool isObject(int ix, int iy, int iz){
		const float x = nse.lat.lbm2physX(ix);
		const float y = nse.lat.lbm2physY(iy);
		const float z = nse.lat.lbm2physZ(iz);
		// Shift the x for non-symmetric bump
		float xshift = x + 0.3*pow(sin(PI*y),4);
		// Bump area
		if(xshift > 0.3 && xshift  < 1.2){
			float fxy = 1.*pow(sin(PI*xshift/0.9 - PI/3.),4); // Wall position
			if(fxy > z){
				return true;
			}
		}
		return false;
	}

	template<typename Filter>
	double integrate_stress_tensor_general(Filter filter, int dir, const double ndc = 2.,const bool dynamicViscosity = true){
		// filter ... which nodes to check (set to desired object)
		// dir ... in which direction to evaluate
		// ndc ... normal derivative coefficient (2 for incompressible, 4/3 for compressible)
		// dynVisc .. dynamic viscosity - whether to use \nu or \nu * rho  = \mu <- dynamic viscosity \mu

		// access lattice parameters
		const double visc = (double)nse.lat.physViscosity;
		const double delta_x = (double)nse.lat.physDl;
		// get LBM reference temperature (it is speed of sound)
		//const double T0 = NoDV == 3  ? (double)0.6979533220196830882384091 : (double)1./3;
		const double T0 = 1./3;
		real local_drag = 0;
		// precalculate which macro to use and in which direction to add pressure
		const int dirMacro = (dir==0) ? MACRO::e_vx : (dir==1) ? MACRO::e_vy : MACRO::e_vz;
		const int dirx = int(dir==0);
		const int diry = int(dir==1);
		const int dirz = int(dir==2);


		for (int x = nse.blocks.front().offset.x() + 1; x < nse.blocks.front().offset.x() + nse.blocks.front().local.x() - 1; x++) {
		for (int y = nse.blocks.front().offset.y() + 1; y < nse.blocks.front().offset.y() + nse.blocks.front().local.y() - 1; y++) {
		for (int z = nse.blocks.front().offset.z() + 1; z < nse.blocks.front().offset.z() + nse.blocks.front().local.z() - 1; z++) {
			if(nse.blocks.front().hmap(x, y, z) == BC::GEO_WALL && filter(x, y, z)){
				// N_1 = (1,0,0)
		 		if(nse.blocks.front().hmap(x+1, y, z) == BC::GEO_FLUID){
					const double rho_lbm = (double)nse.blocks.front().hmacro(MACRO::e_rho,x+1,y,z);
					const double v = (double)nse.lat.lbm2physVelocity(nse.blocks.front().hmacro(dirMacro,x+1,y,z));
					const double pressure =   nse.lat.lbm2physVelocity(nse.lat.lbm2physVelocity(T0*(rho_lbm-1)));
					// pressure
					local_drag -= dirx*pressure;
					// +T_(dir+1)1
					const double dv = v/(delta_x/2);
					if(dynamicViscosity){
						local_drag += rho_lbm*ndc*visc*dv;
					}else{
						local_drag += ndc*visc*dv;
					}
				}
				// N_2 = (-1,0,0)
		 		if(nse.blocks.front().hmap(x-1, y, z) == BC::GEO_FLUID){
					const double rho_lbm = (double)nse.blocks.front().hmacro(MACRO::e_rho,x-1,y,z);
					const double v = (double)nse.lat.lbm2physVelocity(nse.blocks.front().hmacro(dirMacro,x-1,y,z));
					const double pressure = nse.lat.lbm2physVelocity(nse.lat.lbm2physVelocity(T0*(rho_lbm-1)));
					// pressure
					local_drag += dirx*pressure;
					// -T_(dir+1)1
					const double dv = v/(delta_x/2);
					if(dynamicViscosity){
						local_drag += rho_lbm*ndc*visc*dv;
					}else{
						local_drag += ndc*visc*dv;
					}
				}
				// N_3 = (0,1,0)
		 		if(nse.blocks.front().hmap(x, y+1, z) == BC::GEO_FLUID){
					const double rho_lbm = (double)nse.blocks.front().hmacro(MACRO::e_rho,x,y+1,z);
					const double v = (double)nse.lat.lbm2physVelocity(nse.blocks.front().hmacro(dirMacro,x,y+1,z));
					const double pressure = nse.lat.lbm2physVelocity(nse.lat.lbm2physVelocity(T0*(rho_lbm-1)));
					// pressure
					local_drag -= diry*pressure;
					// T_(dir+1)2
					const double dv = v/(delta_x/2);
					if(dynamicViscosity){
						local_drag += rho_lbm*ndc*visc*dv;
					}else{
						local_drag += ndc*visc*dv;
					}
				}
				// N_4 = (0,-1,0)
		 		if(nse.blocks.front().hmap(x, y-1, z) == BC::GEO_FLUID){
					const double rho_lbm = (double)nse.blocks.front().hmacro(MACRO::e_rho,x,y-1,z);
					const double v = (double)nse.lat.lbm2physVelocity(nse.blocks.front().hmacro(dirMacro,x,y-1,z));
					const double pressure = nse.lat.lbm2physVelocity(nse.lat.lbm2physVelocity(T0*(rho_lbm-1)));
					// pressure
					local_drag += diry*pressure;
					// -T_(dir+1)2
					const double dv = v/(delta_x/2);
					if(dynamicViscosity){
						local_drag += rho_lbm*ndc*visc*dv;
					}else{
						local_drag += ndc*visc*dv;
					}
				}
				// N_5 = (0,0,1)
		 		if(nse.blocks.front().hmap(x, y, z+1) == BC::GEO_FLUID){
					const double rho_lbm = (double)nse.blocks.front().hmacro(MACRO::e_rho,x,y,z+1);
					const double v = (double)nse.lat.lbm2physVelocity(nse.blocks.front().hmacro(dirMacro,x,y,z+1));
					const double pressure = nse.lat.lbm2physVelocity(nse.lat.lbm2physVelocity(T0*(rho_lbm-1)));
					// pressure
					local_drag -= dirz*pressure;
					// T_(dir+1)3
					const double dv = v/(delta_x/2);
					if(dynamicViscosity){
						local_drag += rho_lbm*ndc*visc*dv;
					}else{
						local_drag += ndc*visc*dv;
					}
				}
				// N_6 = (0,0,-1)
		 		if(nse.blocks.front().hmap(x, y, z-1) == BC::GEO_FLUID){
					const double rho_lbm = (double)nse.blocks.front().hmacro(MACRO::e_rho,x,y,z);
					const double v = (double)nse.lat.lbm2physVelocity(nse.blocks.front().hmacro(dirMacro,x,y,z-1));
					const double pressure = nse.lat.lbm2physVelocity(nse.lat.lbm2physVelocity(T0*(rho_lbm-1)));
					// pressure
					local_drag += dirz*pressure;
					// -T_(dir+1)3
					const double dv = v/(delta_x/2);
					if(dynamicViscosity){
						local_drag += rho_lbm*ndc*visc*dv;
					}else{
						local_drag += ndc*visc*dv;
					}
				}
			}
		}}}

		real drag = TNL::MPI::reduce(local_drag, MPI_SUM, MPI_COMM_WORLD);
		return delta_x*delta_x*drag; // multiply by lattice square size (here in 3D)
	}


	bool firstrun = true;

	void dragshiftlift() {
		const double H = 1.; // height of bump, 0.05 in origin
		const double L = 1.5; // length of bump
		const double W = 0.5; // width of bump
		const double Uoverline = 1; // average inflow velocity
		real C_D = 2.*integrate_stress_tensor_general([this](int ix,int iy,int iz){ return this->isObject(ix, iy, iz);},0)/(Uoverline*Uoverline)/(H*W);
		real C_S = 2.*integrate_stress_tensor_general([this](int ix,int iy,int iz){ return this->isObject(ix, iy, iz);},1)/(Uoverline*Uoverline)/(L*H);
		real C_L = 2.*integrate_stress_tensor_general([this](int ix,int iy,int iz){ return this->isObject(ix, iy, iz);},2)/(Uoverline*Uoverline)/(L*W);

		if (nse.rank == 0){
			// empty files
			const char* iotype = (firstrun) ? "wt" : "at";
			firstrun = false;
			// output
			FILE* f;
			const std::string dir = fmt::format("results_{}/probes", id);
			mkdir_p(dir.c_str(), 0755);

			std::string str = fmt::format("{}/probe_cd", dir);
			f = fopen(str.c_str(), iotype);
			fprintf(f, "%e\t%e\n", nse.physTime(), C_D);
			fclose(f);

			str = fmt::format("{}/probe_cs", dir);
			f = fopen(str.c_str(), iotype);
			fprintf(f, "%e\t%e\n", nse.physTime(), C_S);
			fclose(f);

			str = fmt::format("{}/probe_cl", dir);
			f = fopen(str.c_str(), iotype);
			fprintf(f, "%e\t%e\n", nse.physTime(), C_L);
			fclose(f);

			spdlog::info(
				"at t={:1.2f}s, iterations={:d} drag={:e} shift?={:e} lift={:e}",
				nse.physTime(),
				nse.iterations,
				C_D,
				C_S,
				C_L
			);
		}
	}
	bool firstrunProfile = true;
	void dragprofile(){
		const double H = 1.; // height of bump, 0.05 in origin
		//const double L = 1.5; // length of bump
		//const double W = 0.5; // width of bump
		const double Uoverline = 1; // average inflow velocity
		const double delta_x = (double)nse.lat.physDl;


		if (nse.rank == 0){
			// empty file
			const char* iotype = (firstrunProfile) ? "wt" : "at";
			firstrunProfile = false;
			FILE* f;
			const std::string dir = fmt::format("results_{}/probes", id);
			mkdir_p(dir.c_str(), 0755);

			// write nothing to delete them?
			std::string str = fmt::format("{}/probe_drag_profile", dir);
			f = fopen(str.c_str(), iotype);
			//fprintf(f, "");
			fclose(f);
		}
		double values[nse.lat.global.x()];

		for(int i = 0; i < nse.lat.global.y(); i++){
			real C_D = 2.*integrate_stress_tensor_general([this,i](int ix,int iy,int iz){ return iy==i && this->isObject(ix, iy, iz);},0)/(Uoverline*Uoverline)/(H*delta_x);
			if(nse.rank == 0){
				values[i] = C_D;
			}
		}

		if (nse.rank == 0){
			FILE* f;
			const std::string dir = fmt::format("results_{}/probes", id);
			std::string str = fmt::format("{}/probe_drag_profile", dir);
			f = fopen(str.c_str(), "at");// always append
			for(int i = 0; i < nse.lat.global.y(); i++){
				fprintf(f, "%e", values[i]);
				if(i != nse.lat.global.y()){
					fprintf(f, "\t");
				}
			}
			fprintf(f, "\n");
			fclose(f);
		}
	}


	void probe1() override {
		dragshiftlift();
		dragprofile();
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
	real LBM_VISCOSITY = 0.001;
	real PHYS_VISCOSITY = 1.5e-3;
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

	state.nse.physFinalTime = 10;
	state.cnt[PRINT].period = 0.1;

	// add cuts
	state.cnt[VTK2D].period = 0.1;
	state.add2Dcut_X(X / 2, "cutsX/cut_X");
	state.add2Dcut_Y(Y / 2, "cutsY/cut_Y");
	state.add2Dcut_Z(Z / 2, "cutsZ/cut_Z");

	state.cnt[VTK3D].period = 1.;
	state.cnt[VTK3DCUT].period = 1.;
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
