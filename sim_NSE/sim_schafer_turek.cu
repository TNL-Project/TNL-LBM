#include <argparse/argparse.hpp>
#include <utility>

// As of now, enum and sync direction are specific for different models and need to be included before core!!!
//#include "lbm3d/d3q27/defs.h"
#include "lbm3d/d3q53/defs.h"
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
	using State<NSE>::checkpoint;
	using State<NSE>::id;

	using idx = typename TRAITS::idx;
	using idx3d = typename TRAITS::idx3d;
	using real = typename TRAITS::real;
	using dreal = typename TRAITS::dreal;
	using point_t = typename TRAITS::point_t;
	using lat_t = Lattice<3, real, idx>;

	real lbm_inflow_vx = 0;
	real inflow_g = 0;
	bool NoDV = 1;

	void checkpointStateLocal(adios2::Mode mode) override
	{
		// Save/load the inflow velocity
		checkpoint.saveLoadAttribute("lbm_inflow_vx", lbm_inflow_vx);
		checkpoint.saveLoadAttribute("lbm_inflow_vx", inflow_g);

		// You can add any additional state data that needs to be saved/loaded here

		if (mode == adios2::Mode::Read)
			spdlog::info("Checkpoint loaded local state (mode: Read)");
		else
			spdlog::info("Checkpoint saved local state (mode: Write)");
	}

	void setupBoundaries() override
	{
		nse.setBoundaryX(0, BC::GEO_INFLOW_LEFT_PRESSURE);						  // left
		nse.setBoundaryX(1, BC::GEO_INFLOW_LEFT_PRESSURE);						  // left
		nse.setBoundaryX(2, BC::GEO_INFLOW_LEFT_PRESSURE);						  // left
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
			if(isObject(px,py,pz)){
				nse.setMap(px, py, pz, BC::GEO_WALL);
			}
		}}}
	}

	bool isObject(int ix, int iy, int iz){
		const float x = nse.lat.lbm2physX(ix);
		const float y = nse.lat.lbm2physY(iy);
		const float z = nse.lat.lbm2physZ(iz);
		const float cx = 0.2;
		const float cz = 0.2;
		const float R = 0.05;
		if(pow(x-cx,2)+pow(z-cz,2)<pow(R,2)){
			return true;
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
		const double T0 = NSE::LBM_KS::NoDV == 3  ? (double)0.6979533220196830882384091 : (double)1./3;
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
					const double rho_lbm = (double)nse.blocks.front().hmacro(MACRO::e_rho,x,y,z-1);
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
		const double H = 0.1; // height of cylinder
		const double L = 0.1; // length of cylinder
		const double W = 0.41; // width of cylinder
		const double Uoverline = 4./9*0.45; // average inflow velocity
		// DIFFERENT AXIS ORIENTATION
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
		const double H = 0.1;
		const double Uoverline = 4./9.*0.45; // average inflow velocity
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

	[[nodiscard]] std::vector<std::string> getOutputDataNames() const override
	{
		// return all quantity names used in outputData
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
			block.data.inflow_vx = lbm_inflow_vx;
			block.data.inflow_vy = 0;
			block.data.inflow_vz = 0;
			block.data.InitPoint = nse.lat.phys2lbmPoint(nse.lat.physOrigin);
			block.data.inflow_g = inflow_g;
			block.data.inflow_y = nse.lat.global.y()-6;
			block.data.inflow_z = nse.lat.global.z()-6;
			if(NSE::LBM_KS::NoDV == 3){
				block.data.no1oT0 = 1./0.6979533220196830882384091; // FOR D3Q343
			}
		}
	}

	StateLocal(const std::string& id, const TNL::MPI::Comm& communicator, lat_t lat, const std::string& adiosConfigPath = "adios2.xml")
	: State<NSE>(id, communicator, std::move(lat), adiosConfigPath)
	{}
};

template <typename NSE>
int sim(const std::string& adios_config = "adios2.xml", int RESOLUTION = 2)
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
	int Y = block_size*RESOLUTION; // Number of points in Y direction
	real PHYS_DL = PHYS_HEIGHT / ((real) Y - 6); // naive fullway bounce-back
	int X = floor(PHYS_LENGTH/PHYS_DL);
	int Z = Y;
	real LBM_VISCOSITY = 0.001;
	real PHYS_VISCOSITY = 0.001;
	real PHYS_VELOCITY = 0.45;
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
	StateLocal<NSE> state(state_id, MPI_COMM_WORLD, lat, adios_config);

	// problem parameters
	state.lbm_inflow_vx = lat.phys2lbmVelocity(PHYS_VELOCITY);
	state.inflow_g = lat.phys2lbmForce(g);

	state.nse.physFinalTime = 20;
	state.cnt[PRINT].period = 1;

	state.wallTime = 10000;
	// add cuts
	state.cnt[OUT2D].period = 1;
	state.add2Dcut_X(X / 2, "cutsX/cut_X");
	state.add2Dcut_Y(Y / 2, "cutsY/cut_Y");
	state.add2Dcut_Z(Z / 2, "cutsZ/cut_Z");

	state.cnt[OUT3D].period = 1;
	state.cnt[OUT3DCUT].period = 1;
	state.add3Dcut(X / 4, Y / 4, Z / 4, X / 2, Y / 2, Z / 2, "box");

	state.cnt[PROBE1].period = 10.;

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
	        fprintf(fp, "%d,%d,%e\n", y, z, state.nse.lat.lbm2physVelocity(KS.vx));
	    }
	}
	fclose(fp);


	execute(state);

	return 0;
}

template <typename TRAITS = TraitsSP>
void run(const std::string& adios_config, int resolution)
{
	// D3Q27
	//using COLL = D3Q27_CUM<TRAITS, D3Q27_EQ_INV_CUM<TRAITS>>;
	//using NSE_CONFIG = LBM_CONFIG<
	//	TRAITS,
	//	D3Q27_KernelStruct,
	//	NSE_Data_DoubleParabolic<TRAITS>,
	//	COLL,
	//	typename COLL::EQ,
	//	D3Q27_STREAMING<TRAITS>,
	//	D3Q27_BC_All,
	//	D3Q27_MACRO_Default<TRAITS>>;

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

	// D3Q53
	using COLL = D3Q53_SRT<TRAITS, D3Q53_EQ<TRAITS>>;
	using NSE_CONFIG = LBM_CONFIG<
		TRAITS,
		D3Q53_KernelStruct,
		NSE_Data_DoubleParabolic<TRAITS>,
		COLL,
		typename COLL::EQ,
		D3Q53_STREAMING<TRAITS>,
		D3Q53_BC_All,
		D3Q53_MACRO_Default<TRAITS>>;

	sim<NSE_CONFIG>(adios_config, resolution);
}

int main(int argc, char** argv)
{
	TNLMPI_INIT mpi(argc, argv);

	argparse::ArgumentParser program("sim_schafer_turek");
	program.add_description("Schafer Turek simulation in 3D");
	program.add_argument("--adios-config").help("path to ADIOS2 configuration file").default_value(std::string("adios2.xml")).nargs(1);
	program.add_argument("--resolution").help("resolution of the lattice").scan<'i', int>().default_value(1).nargs(1);

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

	if (resolution < 1) {
		fmt::println(stderr, "CLI error: resolution must be at least 1");
		return 1;
	}

	run(adios_config, resolution);

	return 0;
}

