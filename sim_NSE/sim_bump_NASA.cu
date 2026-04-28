#include <argparse/argparse.hpp>
#include <cstdio>
#include <utility>

// Change model: change this, and precision at the end
#define ELBM_D3Q53
// #define D3Q53
// #define ELBM_D3Q27
// #define CLBM_D3Q27



// #if defined(ELBM_D3Q53) || defined(D3Q53)
// #define USE_DFMAX3
// #endif
//#define OSCILLATION_ANALYSIS
#define STRESS_TENSOR_FROM_MEAN

// As of now, enum and sync direction are specific for different models and need to be included before core!!!
#if defined(ELBM_D3Q27) || defined(CLBM_D3Q27)
#include "lbm3d/d3q27/defs.h"
#endif
#if defined(ELBM_D3Q53) || defined(D3Q53)
#include "lbm3d/d3q53/defs.h"
#endif
//#include "lbm3d/d3q343/defs.h"
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
	using State<NSE>::id;

	using idx = typename TRAITS::idx;
	using idx3d = typename TRAITS::idx3d;
	using real = typename TRAITS::real;
	using dreal = typename TRAITS::dreal;
	using point_t = typename TRAITS::point_t;
	using lat_t = Lattice<3, real, idx>;

	real lbm_inflow_vx = 0;
	real rise_up_time = 0.;
	real average_inflow = 0;
	real inflow_g = 0;
	real bump_height = 0.05;
	int probeCount = 0;
	int probeCountProfile1 = 0;
	int probeCountProfile2 = 0;
	int probeCountProfile3 = 0;
	int avg_start_iteration = 0;

	// Refernce values for drag and lift
	double H,L,W; // bump dimensions

	// Override checkpointStateLocal to save/load additional state data
	void checkpointStateLocal(adios2::Mode mode) override
	{
		// Save/load the inflow velocity
		checkpoint.saveLoadAttribute("lbm_inflow_vx", lbm_inflow_vx);
		checkpoint.saveLoadAttribute("rise_up_time", rise_up_time);

		// You can add any additional state data that needs to be saved/loaded here
		checkpoint.saveLoadAttribute("average_inflow", average_inflow);
		checkpoint.saveLoadAttribute("inflow_g", inflow_g);
		checkpoint.saveLoadAttribute("bump_height", bump_height);
		checkpoint.saveLoadAttribute("probeCount",probeCount);
		checkpoint.saveLoadAttribute("probeCountProfile1",probeCountProfile1);
		checkpoint.saveLoadAttribute("probeCountProfile2",probeCountProfile2);
		checkpoint.saveLoadAttribute("probeCountProfile3",probeCountProfile3);
		checkpoint.saveLoadAttribute("avg_start_iteration",avg_start_iteration);

		if (mode == adios2::Mode::Read)
			spdlog::info("Checkpoint loaded local state (mode: Read)");
		else
			spdlog::info("Checkpoint saved local state (mode: Write)");
	}

	void setupBoundaries() override
	{
		// 1) Single-speed setup
		// nse.setBoundaryX(0, BC::GEO_INFLOW_LEFT_PRESSURE);			    // left
		// nse.setBoundaryX(nse.lat.global.x() - 1, BC::GEO_OUTFLOW_RIGHT); // right

		// nse.setBoundaryZ(0,                      BC::GEO_SYM_TOP   ); // top
		// nse.setBoundaryZ(nse.lat.global.z() - 1, BC::GEO_SYM_BOTTOM); // bottom
		// nse.setBoundaryY(0, 					    BC::GEO_SYM_BACK  ); // back
		// nse.setBoundaryY(nse.lat.global.y() - 1, BC::GEO_SYM_FRONT ); // front
		// for(int y = 0; y < 1; y++){
		// for(int z = 0; z < 1; z++){
		// 	nse.setBoundaryYZ(y,z,BC::GEO_SYM_TOP_BACK);
		// 	nse.setBoundaryYZ(y,nse.lat.global.z()-1-z,BC::GEO_SYM_TOP_FRONT);
		// 	nse.setBoundaryYZ(nse.lat.global.y()-1-y,z,BC::GEO_SYM_BOTTOM_BACK);
		// 	nse.setBoundaryYZ(nse.lat.global.y()-1-y,nse.lat.global.z()-1-z,BC::GEO_SYM_BOTTOM_FRONT);
		// }
		// }
		// 2) Multi-speed setup


		nse.setBoundaryX(0, BC::GEO_INFLOW_LEFT_PRESSURE);			      // left
		nse.setBoundaryX(1, BC::GEO_INFLOW_LEFT_PRESSURE);			      // left
		nse.setBoundaryX(2, BC::GEO_INFLOW_LEFT_PRESSURE);			      // left
		nse.setBoundaryX(nse.lat.global.x() - 1, BC::GEO_OUTFLOW_RIGHT_INTERP);  // right
		nse.setBoundaryX(nse.lat.global.x() - 2, BC::GEO_OUTFLOW_RIGHT_INTERP);  // right
		nse.setBoundaryX(nse.lat.global.x() - 3, BC::GEO_OUTFLOW_RIGHT_INTERP);  // right

		// 2a) wall boundaries on sides
		// nse.setBoundaryZ(0,                      BC::GEO_WALL);	 // top
		// nse.setBoundaryZ(1,                      BC::GEO_WALL);	 // top
		// nse.setBoundaryZ(2,                      BC::GEO_WALL);	 // top
		// nse.setBoundaryZ(nse.lat.global.z() - 1, BC::GEO_WALL);	 // bottom
		// nse.setBoundaryZ(nse.lat.global.z() - 2, BC::GEO_WALL);	 // bottom
		// nse.setBoundaryZ(nse.lat.global.z() - 3, BC::GEO_WALL);	 // bottom
		// nse.setBoundaryY(0, 					    BC::GEO_WALL);	 // back
		// nse.setBoundaryY(1, 					    BC::GEO_WALL);	 // back
		// nse.setBoundaryY(2, 					    BC::GEO_WALL);	 // back
		// nse.setBoundaryY(nse.lat.global.y() - 1, BC::GEO_WALL);	 // front
		// nse.setBoundaryY(nse.lat.global.y() - 2, BC::GEO_WALL);	 // front
		// nse.setBoundaryY(nse.lat.global.y() - 3, BC::GEO_WALL);	 // front

		// 2b) symmetric condition
		nse.setBoundaryZ(0,                      BC::GEO_SYM_TOP);		// top
		nse.setBoundaryZ(1,                      BC::GEO_SYM_TOP);		// top
		nse.setBoundaryZ(2,                      BC::GEO_SYM_TOP);		// top
		nse.setBoundaryZ(nse.lat.global.z() - 1, BC::GEO_SYM_BOTTOM);	 // bottom
		nse.setBoundaryZ(nse.lat.global.z() - 2, BC::GEO_SYM_BOTTOM);	 // bottom
		nse.setBoundaryZ(nse.lat.global.z() - 3, BC::GEO_SYM_BOTTOM);	 // bottom
		nse.setBoundaryY(0, 					 BC::GEO_SYM_BACK);		// back
		nse.setBoundaryY(1, 					 BC::GEO_SYM_BACK);		// back
		nse.setBoundaryY(2, 					 BC::GEO_SYM_BACK);		// back
		nse.setBoundaryY(nse.lat.global.y() - 1, BC::GEO_SYM_FRONT);	 // front
		nse.setBoundaryY(nse.lat.global.y() - 2, BC::GEO_SYM_FRONT);	 // front
		nse.setBoundaryY(nse.lat.global.y() - 3, BC::GEO_SYM_FRONT);	 // front
		// Corner edges
		for(int y = 0; y < 3; y++){
		for(int z = 0; z < 3; z++){
			nse.setBoundaryYZ(y,z,BC::GEO_SYM_TOP_BACK);
			nse.setBoundaryYZ(y,nse.lat.global.z()-1-z,BC::GEO_SYM_TOP_FRONT);
			nse.setBoundaryYZ(nse.lat.global.y()-1-y,z,BC::GEO_SYM_BOTTOM_BACK);
			nse.setBoundaryYZ(nse.lat.global.y()-1-y,nse.lat.global.z()-1-z,BC::GEO_SYM_BOTTOM_FRONT);
		}
		}


		// // 3) BUMP
		for (int px = 0; px <= nse.lat.global.x(); px++){
		for (int py = 0; py <= nse.lat.global.y(); py++){
		for (int pz = 0; pz <= nse.lat.global.z(); pz++){
			if(isObject(px,py,pz)){
				nse.setMap(px, py, pz, BC::GEO_WALL);
			}
			//if(isThatLine(px,py,pz)){
			//	nse.setMap(px, py, pz, BC::LINE_DEBUG);
			//}
		}}}




		// 4) (Optional) Mark walls next to bump to performed third-array full-way bounce-back
		// ! turn off for single-speed simulations
		// does not work with MPI - crashes before simulation start
		#ifdef USE_DFMAX3
		mark_next_to_wall_mpi();
		#endif
	}

	bool isObject(int ix, int iy, int iz){
		const float x = nse.lat.physOrigin.x() + (ix) * nse.lat.physDl;
		const float y = nse.lat.physOrigin.y() + (iy) * nse.lat.physDl;
		const float z = nse.lat.physOrigin.z() + (iz) * nse.lat.physDl;
		// Shift the x for non-symmetric bump
		float xshift = x + 0.3*pow(sin(PI*z),4);
		// Bump area
		if(xshift > 0.3 && xshift  < 1.2){
			float fxy = bump_height*pow(sin(PI*xshift/0.9 - PI/3.),4); // Wall position
			if(fxy > y){
				return true;
			}
		}
		return false;
	}

	bool isThatLine(int ix, int iy, int iz){
		const double z = nse.lat.physOrigin.z() + (iz) * nse.lat.physDl;
		const double x = 0.690420848175 + 0.3*(pow(sin(PI*z),4));
		const int ixneeded = floor((x-nse.lat.physOrigin.x())/nse.lat.physDl);
		if(ixneeded == ix){
			return true;
		}
		return false;
	}

	void mark_next_to_wall_mpi(){
		for (auto& block : nse.blocks) {
			for (int x = NSE::LBM_KS::NoDV; x < nse.lat.global.x()-NSE::LBM_KS::NoDV; x++){
				// TODO: optimize
				for (int y = NSE::LBM_KS::NoDV; y < nse.lat.global.y()-NSE::LBM_KS::NoDV; y++){
			for (int z = NSE::LBM_KS::NoDV; z < nse.lat.global.z()-NSE::LBM_KS::NoDV; z++){
				if (!block.isLocalIndex(x, y, z)) {continue;}


				if(block.hmap(x,y,z)!=BC::GEO_FLUID){continue;}
				bool done = false;
				for(int dx = - NSE::LBM_KS::NoDV; dx <= NSE::LBM_KS::NoDV;dx ++){
				for(int dy = - NSE::LBM_KS::NoDV; dy <= NSE::LBM_KS::NoDV;dy ++){
				for(int dz = - NSE::LBM_KS::NoDV; dz <= NSE::LBM_KS::NoDV;dz ++){
					if (!block.isLocalIndex(x+dx, y+dy, z+dz)) {continue;}
					if(block.hmap(x+dx,y+dy,z+dz) == BC::GEO_WALL){
						nse.setMap(x,y,z,BC::GEO_NEXT_TO_WALL);
						done = true;
					}
				}
				if(done){break;}
				}
				if(done){break;}
				}
			}}}
		}
		printf("Initialization of next to wall was successful");
	}

	template<typename Filter>
	double integrate_stress_tensor_general(Filter filter, int dir, const double ndc = 4./3.,const bool dynamicViscosity = false){
		// filter ... which nodes to check (set to desired object)
		// dir ... in which direction to evaluate
		// ndc ... normal derivative coefficient (2 for incompressible, 4/3 for compressible)
		// dynVisc .. dynamic viscosity - whether to use \nu or \nu * rho  = \mu <- dynamic viscosity \mu

		// access lattice parameters
		const double visc = (double)nse.lat.physViscosity;
		const double delta_x = (double)nse.lat.physDl;
		// get LBM reference temperature (it is speed of sound)
		const double T0 = NSE::LBM_KS::T0;
		real local_drag = 0;
		// precalculate which macro to use and in which direction to add pressure
		#ifdef STRESS_TENSOR_FROM_MEAN
		const int dirMacro = (dir==0) ? MACRO::e_vm_x : (dir==1) ? MACRO::e_vm_y : MACRO::e_vm_z;
		#else
		const int dirMacro = (dir==0) ? MACRO::e_vx : (dir==1) ? MACRO::e_vy : MACRO::e_vz;
		#endif
		const int dirx = int(dir==0);
		const int diry = int(dir==1);
		const int dirz = int(dir==2);

		auto& block = nse.blocks.front();
		auto& offset = block.offset;
		auto& local = block.local;


		for (int x = offset.x() + 1; x < offset.x() + local.x() - 1; x++) {
		for (int y = offset.y() + 1; y < offset.y() + local.y() - 1; y++) {
		for (int z = offset.z() + 1; z < offset.z() + local.z() - 1; z++) {
			if(filter(x, y, z)){//block.hmap(x, y, z) == BC::GEO_WALL && filter(x, y, z)
				// N_1 = (1,0,0)
		 		if(BC::isFluid(block.hmap(x+1, y, z))){
					#ifdef STRESS_TENSOR_FROM_MEAN
					const double rho_lbm = (double)block.hmacro(MACRO::e_rhom,x+1,y,z);
					#else
					const double rho_lbm = (double)block.hmacro(MACRO::e_rho,x+1,y,z);
					#endif
					const double v = (double)nse.lat.lbm2physVelocity(block.hmacro(dirMacro,x+1,y,z));
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
		 		if(BC::isFluid(block.hmap(x-1, y, z))){
					#ifdef STRESS_TENSOR_FROM_MEAN
					const double rho_lbm = (double)block.hmacro(MACRO::e_rhom,x-1,y,z);
					#else
					const double rho_lbm = (double)block.hmacro(MACRO::e_rho,x-1,y,z);
					#endif
					const double v = (double)nse.lat.lbm2physVelocity(block.hmacro(dirMacro,x-1,y,z));
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
		 		if(BC::isFluid(block.hmap(x, y+1, z))){
					#ifdef STRESS_TENSOR_FROM_MEAN
					const double rho_lbm = (double)block.hmacro(MACRO::e_rhom,x,y+1,z);
					#else
					const double rho_lbm = (double)block.hmacro(MACRO::e_rho,x,y+1,z);
					#endif
					const double v = (double)nse.lat.lbm2physVelocity(block.hmacro(dirMacro,x,y+1,z));
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
		 		if(BC::isFluid(block.hmap(x, y-1, z))){
					#ifdef STRESS_TENSOR_FROM_MEAN
					const double rho_lbm = (double)block.hmacro(MACRO::e_rhom,x,y-1,z);
					#else
					const double rho_lbm = (double)block.hmacro(MACRO::e_rho,x,y-1,z);
					#endif
					const double v = (double)nse.lat.lbm2physVelocity(block.hmacro(dirMacro,x,y-1,z));
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
		 		if(BC::isFluid(block.hmap(x, y-1, z+1))){
					#ifdef STRESS_TENSOR_FROM_MEAN
					const double rho_lbm = (double)block.hmacro(MACRO::e_rhom,x,y,z+1);
					#else
					const double rho_lbm = (double)block.hmacro(MACRO::e_rho,x,y,z+1);
					#endif
					const double v = (double)nse.lat.lbm2physVelocity(block.hmacro(dirMacro,x,y,z+1));
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
		 		if(block.hmap(x, y, z-1) == BC::GEO_FLUID){
					#ifdef STRESS_TENSOR_FROM_MEAN
					const double rho_lbm = (double)block.hmacro(MACRO::e_rhom,x,y,z-1);
					#else
					const double rho_lbm = (double)block.hmacro(MACRO::e_rho,x,y,z-1);
					#endif
					const double v = (double)nse.lat.lbm2physVelocity(block.hmacro(dirMacro,x,y,z-1));
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

	template<typename Filter>
	double integrate_stress_tensor_general_only_pressure(Filter filter, int dir){
		// filter ... which nodes to check (set to desired object)
		// dir ... in which direction to evaluate

		// access lattice parameters
		const double visc = (double)nse.lat.physViscosity;
		const double delta_x = (double)nse.lat.physDl;
		// get LBM reference temperature (it is speed of sound)
		const double T0 = NSE::LBM_KS::T0;
		real local_drag = 0;
		// precalculate which macro to use and in which direction to add pressure
		#ifdef STRESS_TENSOR_FROM_MEAN
		const int dirMacro = (dir==0) ? MACRO::e_vm_x : (dir==1) ? MACRO::e_vm_y : MACRO::e_vm_z;
		#else
		const int dirMacro = (dir==0) ? MACRO::e_vx : (dir==1) ? MACRO::e_vy : MACRO::e_vz;
		#endif
		const int dirx = int(dir==0);
		const int diry = int(dir==1);
		const int dirz = int(dir==2);

		auto& block = nse.blocks.front();
		auto& offset = block.offset;
		auto& local = block.local;


		for (int x = offset.x() + 1; x < offset.x() + local.x() - 1; x++) {
		for (int y = offset.y() + 1; y < offset.y() + local.y() - 1; y++) {
		for (int z = offset.z() + 1; z < offset.z() + local.z() - 1; z++) {
			if(filter(x, y, z)){
				// N_1 = (1,0,0)
		 		if(BC::isFluid(block.hmap(x+1, y, z))){
					#ifdef STRESS_TENSOR_FROM_MEAN
					const double rho_lbm = (double)block.hmacro(MACRO::e_rhom,x+1,y,z);
					#else
					const double rho_lbm = (double)block.hmacro(MACRO::e_rho,x+1,y,z);
					#endif
					const double pressure =   nse.lat.lbm2physVelocity(nse.lat.lbm2physVelocity(T0*(rho_lbm-1)));
					// pressure
					local_drag -= dirx*pressure;
				}
				// N_2 = (-1,0,0)
		 		if(BC::isFluid(block.hmap(x-1, y, z))){
					#ifdef STRESS_TENSOR_FROM_MEAN
					const double rho_lbm = (double)block.hmacro(MACRO::e_rhom,x-1,y,z);
					#else
					const double rho_lbm = (double)block.hmacro(MACRO::e_rho,x-1,y,z);
					#endif
					const double pressure = nse.lat.lbm2physVelocity(nse.lat.lbm2physVelocity(T0*(rho_lbm-1)));
					// pressure
					local_drag += dirx*pressure;
				}
				// N_3 = (0,1,0)
		 		if(BC::isFluid(block.hmap(x, y+1, z))){
					#ifdef STRESS_TENSOR_FROM_MEAN
					const double rho_lbm = (double)block.hmacro(MACRO::e_rhom,x,y+1,z);
					#else
					const double rho_lbm = (double)block.hmacro(MACRO::e_rho,x,y+1,z);
					#endif
					const double pressure = nse.lat.lbm2physVelocity(nse.lat.lbm2physVelocity(T0*(rho_lbm-1)));
					// pressure
					local_drag -= diry*pressure;
				}
				// N_4 = (0,-1,0)
		 		if(BC::isFluid(block.hmap(x, y-1, z))){
					#ifdef STRESS_TENSOR_FROM_MEAN
					const double rho_lbm = (double)block.hmacro(MACRO::e_rhom,x,y-1,z);
					#else
					const double rho_lbm = (double)block.hmacro(MACRO::e_rho,x,y-1,z);
					#endif
					const double pressure = nse.lat.lbm2physVelocity(nse.lat.lbm2physVelocity(T0*(rho_lbm-1)));
					// pressure
					local_drag += diry*pressure;
				}
				// N_5 = (0,0,1)
		 		if(BC::isFluid(block.hmap(x, y-1, z+1))){
					#ifdef STRESS_TENSOR_FROM_MEAN
					const double rho_lbm = (double)block.hmacro(MACRO::e_rhom,x,y,z+1);
					#else
					const double rho_lbm = (double)block.hmacro(MACRO::e_rho,x,y,z+1);
					#endif
					const double pressure = nse.lat.lbm2physVelocity(nse.lat.lbm2physVelocity(T0*(rho_lbm-1)));
					// pressure
					local_drag -= dirz*pressure;
				}
				// N_6 = (0,0,-1)
		 		if(block.hmap(x, y, z-1) == BC::GEO_FLUID){
					#ifdef STRESS_TENSOR_FROM_MEAN
					const double rho_lbm = (double)block.hmacro(MACRO::e_rhom,x,y,z-1);
					#else
					const double rho_lbm = (double)block.hmacro(MACRO::e_rho,x,y,z-1);
					#endif
					const double pressure = nse.lat.lbm2physVelocity(nse.lat.lbm2physVelocity(T0*(rho_lbm-1)));
					// pressure
					local_drag += dirz*pressure;
				}
			}
		}}}

		real drag = TNL::MPI::reduce(local_drag, MPI_SUM, MPI_COMM_WORLD);
		return delta_x*delta_x*drag; // multiply by lattice square size (here in 3D)
	}

	template<typename Filter>
	double integrate_stress_tensor_general_only_viscous(Filter filter, int dir, const double ndc = 4./3.,const bool dynamicViscosity = false){
		// filter ... which nodes to check (set to desired object)
		// dir ... in which direction to evaluate
		// ndc ... normal derivative coefficient (2 for incompressible, 4/3 for compressible)
		// dynVisc .. dynamic viscosity - whether to use \nu or \nu * rho  = \mu <- dynamic viscosity \mu

		// access lattice parameters
		const double visc = (double)nse.lat.physViscosity;
		const double delta_x = (double)nse.lat.physDl;
		// get LBM reference temperature (it is speed of sound)
		const double T0 = NSE::LBM_KS::T0;
		real local_drag = 0;
		// precalculate which macro to use and in which direction to add pressure
		const int dirMacro = (dir==0) ? MACRO::e_vx : (dir==1) ? MACRO::e_vy : MACRO::e_vz;

		auto& block = nse.blocks.front();
		auto& offset = block.offset;
		auto& local = block.local;


		for (int x = offset.x() + 1; x < offset.x() + local.x() - 1; x++) {
		for (int y = offset.y() + 1; y < offset.y() + local.y() - 1; y++) {
		for (int z = offset.z() + 1; z < offset.z() + local.z() - 1; z++) {
			if(filter(x, y, z)){
				// N_1 = (1,0,0)
		 		if(BC::isFluid(block.hmap(x+1, y, z))){
					const double rho_lbm = (double)block.hmacro(MACRO::e_rho,x+1,y,z);
					const double v = (double)nse.lat.lbm2physVelocity(block.hmacro(dirMacro,x+1,y,z));
					// +T_(dir+1)1
					const double dv = v/(delta_x/2);
					if(dynamicViscosity){
						local_drag += rho_lbm*ndc*visc*dv;
					}else{
						local_drag += ndc*visc*dv;
					}
				}
				// N_2 = (-1,0,0)
		 		if(BC::isFluid(block.hmap(x-1, y, z))){
					const double rho_lbm = (double)block.hmacro(MACRO::e_rho,x-1,y,z);
					const double v = (double)nse.lat.lbm2physVelocity(block.hmacro(dirMacro,x-1,y,z));
					// -T_(dir+1)1
					const double dv = v/(delta_x/2);
					if(dynamicViscosity){
						local_drag += rho_lbm*ndc*visc*dv;
					}else{
						local_drag += ndc*visc*dv;
					}
				}
				// N_3 = (0,1,0)
		 		if(BC::isFluid(block.hmap(x, y+1, z))){
					const double rho_lbm = (double)block.hmacro(MACRO::e_rho,x,y+1,z);
					const double v = (double)nse.lat.lbm2physVelocity(block.hmacro(dirMacro,x,y+1,z));
					// T_(dir+1)2
					const double dv = v/(delta_x/2);
					if(dynamicViscosity){
						local_drag += rho_lbm*ndc*visc*dv;
					}else{
						local_drag += ndc*visc*dv;
					}
				}
				// N_4 = (0,-1,0)
		 		if(BC::isFluid(block.hmap(x, y-1, z))){
					const double rho_lbm = (double)block.hmacro(MACRO::e_rho,x,y-1,z);
					const double v = (double)nse.lat.lbm2physVelocity(block.hmacro(dirMacro,x,y-1,z));
					// -T_(dir+1)2
					const double dv = v/(delta_x/2);
					if(dynamicViscosity){
						local_drag += rho_lbm*ndc*visc*dv;
					}else{
						local_drag += ndc*visc*dv;
					}
				}
				// N_5 = (0,0,1)
		 		if(BC::isFluid(block.hmap(x, y-1, z+1))){
					const double rho_lbm = (double)block.hmacro(MACRO::e_rho,x,y,z+1);
					const double v = (double)nse.lat.lbm2physVelocity(block.hmacro(dirMacro,x,y,z+1));
					// T_(dir+1)3
					const double dv = v/(delta_x/2);
					if(dynamicViscosity){
						local_drag += rho_lbm*ndc*visc*dv;
					}else{
						local_drag += ndc*visc*dv;
					}
				}
				// N_6 = (0,0,-1)
		 		if(block.hmap(x, y, z-1) == BC::GEO_FLUID){
					const double rho_lbm = (double)block.hmacro(MACRO::e_rho,x,y,z-1);
					const double v = (double)nse.lat.lbm2physVelocity(block.hmacro(dirMacro,x,y,z-1));
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

	void dragshiftlift() {
		const double H = 1; //bump_height; // taken as 1 in article !!  // height of bump, 0.05 in origin
		const double L = 1.5; // length of bump
		const double W = 0.5; // width of bump
		const double Uoverline = average_inflow; // average inflow velocity
		real C_D = 2.*integrate_stress_tensor_general([this](int ix,int iy,int iz){ return this->isObject(ix, iy, iz);},0)/(Uoverline*Uoverline)/(H*W);
		real C_D_P = 2.*integrate_stress_tensor_general_only_pressure([this](int ix,int iy,int iz){ return this->isObject(ix, iy, iz);},0)/(Uoverline*Uoverline)/(H*W);
		real C_D_nu = 2.*integrate_stress_tensor_general_only_viscous([this](int ix,int iy,int iz){ return this->isObject(ix, iy, iz);},0)/(Uoverline*Uoverline)/(H*W);
		real C_S = 2.*integrate_stress_tensor_general([this](int ix,int iy,int iz){ return this->isObject(ix, iy, iz);},2)/(Uoverline*Uoverline)/(L*H);
		real C_L = 2.*integrate_stress_tensor_general([this](int ix,int iy,int iz){ return this->isObject(ix, iy, iz);},1)/(Uoverline*Uoverline)/(L*W);

		if (nse.rank == 0){
			// empty files
			const char* iotype = (probeCount == 0) ? "wt" : "at";
			probeCount += 1;
			// output
			FILE* f;
			const std::string dir = fmt::format("results_{}/probes", id);
			mkdir_p(dir.c_str(), 0755);

			std::string str = fmt::format("{}/probe_cd", dir);
			f = fopen(str.c_str(), iotype);
			fprintf(f, "%e\t%e\n", nse.physTime(), C_D);
			fclose(f);
			str = fmt::format("{}/probe_cdP", dir);
			f = fopen(str.c_str(), iotype);
			fprintf(f, "%e\t%e\n", nse.physTime(), C_D_P);
			fclose(f);
			str = fmt::format("{}/probe_cdnu", dir);
			f = fopen(str.c_str(), iotype);
			fprintf(f, "%e\t%e\n", nse.physTime(), C_D_nu);
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
				"at t={:1.2f}s, iterations={:d} drag={:e} lift={:e} shift?={:e}",
				nse.physTime(),
				nse.iterations,
				C_D,
				C_L,
				C_S
			);
		}
	}

	void dragprofile(){
		const double Uoverline = average_inflow; // average inflow velocity
		const double delta_x = (double)nse.lat.physDl;

		const int SIZE = nse.lat.global.x();

		if (nse.rank == 0){
			// empty file
			const char* iotype = (probeCountProfile1 == 0) ? "wt" : "at";

			FILE* f;
			const std::string dir = fmt::format("results_{}/probes", id);
			mkdir_p(dir.c_str(), 0755);

			// write nothing to delete them?
			std::string str = fmt::format("{}/probe_drag_profile_x_cd", dir);
			f = fopen(str.c_str(), iotype);
			if(probeCountProfile1 == 0){
				for(int ix = 0; ix < SIZE; ix++){
					fprintf(f, "%e", nse.lat.physOrigin.x() + (ix) * nse.lat.physDl);
					if(ix != SIZE-1){
						fprintf(f, ",");
					}
				}
				fprintf(f, "\n");
			}
			fclose(f);
			// CdP
			str = fmt::format("{}/probe_drag_profile_x_cdP", dir);
			f = fopen(str.c_str(), iotype);
			if(probeCountProfile1 == 0){
				for(int ix = 0; ix < SIZE; ix++){
					fprintf(f, "%e", nse.lat.physOrigin.x() + (ix) * nse.lat.physDl);
					if(ix != SIZE-1){
						fprintf(f, ",");
					}
				}
				fprintf(f, "\n");
			}
			fclose(f);
			// Cdnu
			str = fmt::format("{}/probe_drag_profile_x_cdnu", dir);
			f = fopen(str.c_str(), iotype);
			if(probeCountProfile1 == 0){
				for(int ix = 0; ix < SIZE; ix++){
					fprintf(f, "%e", nse.lat.physOrigin.x() + (ix) * nse.lat.physDl);
					if(ix != SIZE-1){
						fprintf(f, ",");
					}
				}
				fprintf(f, "\n");
			}
			fclose(f);
			probeCountProfile1 += 1;
		}
		double values_cd[SIZE];
		double values_cdP[SIZE];
		double values_cdnu[SIZE];

		for(int ix = 0; ix < SIZE; ix++){
			const int ixneeded = ix;
			const real C_D = 2.*integrate_stress_tensor_general([this,ixneeded](int ix,int iy,int iz){ return ix==ixneeded && this->isObject(ix, iy, iz);},0)
						/(Uoverline*Uoverline)/(1.*delta_x);
			const real C_DP = 2.*integrate_stress_tensor_general_only_pressure([this,ixneeded](int ix,int iy,int iz){ return ix==ixneeded && this->isObject(ix, iy, iz);},0)
			                 /(Uoverline*Uoverline)/(1.*delta_x);
			const real C_Dnu = 2.*integrate_stress_tensor_general_only_viscous([this,ixneeded](int ix,int iy,int iz){ return ix==ixneeded && this->isObject(ix, iy, iz);},0)
			                 /(Uoverline*Uoverline)/(1.*delta_x);
			if(nse.rank == 0){
				values_cd[ix] = C_D;
				values_cdP[ix] = C_DP;
				values_cdnu[ix] = C_Dnu;
			}
		}

		if (nse.rank == 0){
			FILE* f;
			const std::string dir = fmt::format("results_{}/probes", id);

			std::string str = fmt::format("{}/probe_drag_profile_x_cd", dir);
			f = fopen(str.c_str(), "at");// always append
			for(int i = 0; i < SIZE; i++){
				fprintf(f, "%e", values_cd[i]);
				if(i != SIZE-1){
					fprintf(f, ",");
				}
			}
			fprintf(f, "\n");
			fclose(f);

			str = fmt::format("{}/probe_drag_profile_x_cdP", dir);
			f = fopen(str.c_str(), "at");// always append
			for(int i = 0; i < SIZE; i++){
				fprintf(f, "%e", values_cdP[i]);
				if(i != SIZE-1){
					fprintf(f, ",");
				}
			}
			fprintf(f, "\n");
			fclose(f);

			str = fmt::format("{}/probe_drag_profile_x_cdnu", dir);
			f = fopen(str.c_str(), "at");// always append
			for(int i = 0; i < SIZE; i++){
				fprintf(f, "%e", values_cdnu[i]);
				if(i != SIZE-1){
					fprintf(f, ",");
				}
			}
			fprintf(f, "\n");
			fclose(f);
		}
	}
	void dragprofile_onaline(){
		const double H = bump_height; // taken as 1 in article !! // height of bump, 0.05 in origin
		//const double L = 1.5; // length of bump
		//const double W = 0.5; // width of bump
		const double Uoverline = average_inflow; // average inflow velocity
		const double delta_x = (double)nse.lat.physDl;

		const int SIZE = nse.lat.global.z();

		if (nse.rank == 0){
			// empty file
			const char* iotype = (probeCountProfile2 == 0) ? "wt" : "at";

			FILE* f;
			const std::string dir = fmt::format("results_{}/probes", id);
			mkdir_p(dir.c_str(), 0755);

			// write nothing to delete them?
			std::string str = fmt::format("{}/probe_drag_profile_line", dir);
			f = fopen(str.c_str(), iotype);

			if(probeCountProfile2 == 0){
				for(int iz = 0; iz < SIZE; iz++){
					fprintf(f, "%e", nse.lat.physOrigin.z() + (iz) * nse.lat.physDl);
					if(iz != SIZE-1){
						fprintf(f, ",");
					}
				}
				fprintf(f, "\n");
			}

			//fprintf(f, "");
			fclose(f);
			probeCountProfile2 += 1;
		}
		double values[SIZE];

		for(int iz = 0; iz < SIZE; iz++){
			const double z = nse.lat.physOrigin.z() + (iz) * nse.lat.physDl;
			const double x=0.690420848175 + 0.3*(pow(sin(3.1415926*(double)z),4));
			const int ixneeded = floor((x-nse.lat.physOrigin.x())/nse.lat.physDl);
			const int izneeded = iz;
			// CHANGED TO ONLY PRESSURE AS IN BUMP-IN-CHANNEL
			real C_D = 2.*integrate_stress_tensor_general_only_pressure([this,izneeded,ixneeded](int ix,int iy,int iz){ return ix==ixneeded && iz==izneeded && this->isObject(ix, iy, iz);},0)
						/(Uoverline*Uoverline)/(delta_x*delta_x);
			if(nse.rank == 0){
				values[iz] = C_D;
			}
		}

		if (nse.rank == 0){
			FILE* f;
			const std::string dir = fmt::format("results_{}/probes", id);
			std::string str = fmt::format("{}/probe_drag_profile_line", dir);
			f = fopen(str.c_str(), "at");// always append
			for(int i = 0; i < SIZE; i++){
				fprintf(f, "%e", values[i]);
				if(i != SIZE-1){
					fprintf(f, ",");
				}
			}
			fprintf(f, "\n");
			fclose(f);
		}
	}

	void dragprofile_verticalprofile(){
		const double H = bump_height; // taken as 1 in article !! // height of bump, 0.05 in origin
		//const double L = 1.5; // length of bump
		//const double W = 0.5; // width of bump
		const double Uoverline = average_inflow; // average inflow velocity
		const double delta_x = (double)nse.lat.physDl;

		const int SIZE = nse.lat.global.y(); // vertical => y (this axis is swapped with z)

		if (nse.rank == 0){
			// empty file
			const char* iotype = (probeCountProfile3 == 0) ? "wt" : "at";

			FILE* f;
			const std::string dir = fmt::format("results_{}/probes", id);
			mkdir_p(dir.c_str(), 0755);

			// write nothing to delete them?
			std::string str = fmt::format("{}/probe_drag_profile_vertical_cd", dir);
			f = fopen(str.c_str(), iotype);
			if(probeCountProfile3 == 0){
				for(int iy = 0; iy < SIZE; iy++){
					fprintf(f, "%e", nse.lat.physOrigin.y() + (iy) * nse.lat.physDl);
					if(iy != SIZE-1){
						fprintf(f, ",");
					}
				}
				fprintf(f, "\n");
			}
			fclose(f);
			//
			str = fmt::format("{}/probe_drag_profile_vertical_cdp", dir);
			f = fopen(str.c_str(), iotype);
			if(probeCountProfile3 == 0){
				for(int iy = 0; iy < SIZE; iy++){
					fprintf(f, "%e", nse.lat.physOrigin.y() + (iy) * nse.lat.physDl);
					if(iy != SIZE-1){
						fprintf(f, ",");
					}
				}
				fprintf(f, "\n");
			}
			fclose(f);

			probeCountProfile3 += 1;
		}
		double values[SIZE];
		double values_cdp[SIZE];
		const double x = 1.207912207;
		const double z = -0.125;
		const int ixneeded = floor((x-nse.lat.physOrigin.x())/nse.lat.physDl);
		const int izneeded = floor((z-nse.lat.physOrigin.z())/nse.lat.physDl);

		for(int iy = 0; iy < SIZE; iy++){
			const int iyneeded = iy;
			const real C_D = 2.*integrate_stress_tensor_general([this,iyneeded,ixneeded,izneeded](int ix,int iy,int iz){ return ix==ixneeded && iz==izneeded && iy==iyneeded;},0)
			/(Uoverline*Uoverline)/(delta_x*delta_x);
			const real C_DP = 2.*integrate_stress_tensor_general_only_pressure([this,iyneeded,ixneeded,izneeded](int ix,int iy,int iz){ return ix==ixneeded && iz==izneeded && iy==iyneeded;},0)
				/(Uoverline*Uoverline)/(delta_x*delta_x);
			if(nse.rank == 0){
				values[iy] = C_D;
				values_cdp[iy] = C_DP;
			}
		}

		if (nse.rank == 0){
			FILE* f;
			const std::string dir = fmt::format("results_{}/probes", id);

			std::string str = fmt::format("{}/probe_drag_profile_vertical_cd", dir);
			f = fopen(str.c_str(), "at");// always append
			for(int i = 0; i < SIZE; i++){
				fprintf(f, "%e", values[i]);
				if(i != SIZE-1){
					fprintf(f, ",");
				}
			}
			fprintf(f, "\n");
			fclose(f);

			str = fmt::format("{}/probe_drag_profile_vertical_cdp", dir);
			f = fopen(str.c_str(), "at");// always append
			for(int i = 0; i < SIZE; i++){
				fprintf(f, "%e", values_cdp[i]);
				if(i != SIZE-1){
					fprintf(f, ",");
				}
			}
			fprintf(f, "\n");
			fclose(f);
		}
	}


	void probe1() override {
		dragshiftlift();
	}
	void probe2() override {
		dragprofile();
		dragprofile_onaline();
		dragprofile_verticalprofile();
  	}

	[[nodiscard]] std::vector<std::string> getOutputDataNames() const override
	{
		// return all quantity names used in outputData
		return {"lbm_density", "lbm_density_fluctuation", "velocity_x", "velocity_y", "velocity_z","lbm_density_mean","velocity_x_mean", "velocity_y_mean", "velocity_z_mean","mywall"};
	}

	void outputData(UniformDataWriter<TRAITS>& writer, const BLOCK& block, const idx3d& begin, const idx3d& end) override
	{
		writer.write("lbm_density", [&](idx x, idx y, idx z) -> dreal
			{
				return block.hmacro(MACRO::e_rho, x, y, z);
			},
			begin, end);
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
		writer.write(
			"lbm_density_mean",
			[&](idx x, idx y, idx z) -> dreal
			{
				return block.hmacro(MACRO::e_rhom, x, y, z);
			},
			begin,
			end
		);
		writer.write(
			"velocity_x_mean",
			[&](idx x, idx y, idx z) -> dreal
			{
				return nse.lat.lbm2physVelocity(block.hmacro(MACRO::e_vm_x, x, y, z));
			},
			begin,
			end
		);
		writer.write(
			"velocity_y_mean",
			[&](idx x, idx y, idx z) -> dreal
			{
				return nse.lat.lbm2physVelocity(block.hmacro(MACRO::e_vm_y, x, y, z));
			},
			begin,
			end
		);
		writer.write(
			"velocity_z_mean",
			[&](idx x, idx y, idx z) -> dreal
			{
				return nse.lat.lbm2physVelocity(block.hmacro(MACRO::e_vm_z, x, y, z));
			},
			begin,
			end
		);
		writer.write(
			"mywall",
			[&](idx x, idx y, idx z) -> dreal
			{
				return (dreal)(block.hmap(x, y, z));
			},
			begin,
			end
		);
	}

	void updateKernelVelocities() override
	{
		for (auto& block : nse.blocks) {
			real riseUpCoefficient = nse.physTime()/rise_up_time;
			if(riseUpCoefficient > 1.){	riseUpCoefficient = 1.;}
			riseUpCoefficient = riseUpCoefficient*exp(1-riseUpCoefficient)*0.99+0.01; // better interpolation
			block.data.inflow_vx = riseUpCoefficient*lbm_inflow_vx;
			block.data.inflow_vy = 0;
			block.data.inflow_vz = 0;
			block.data.inflow_g = inflow_g;
			block.data.no1oT0 = 1./NSE::LBM_KS::T0;

			// TEST
			block.data.stat_counter = nse.iterations - avg_start_iteration;
		}
	}

	StateLocal(const std::string& id, const TNL::MPI::Comm& communicator, lat_t lat, const std::string& adiosConfigPath = "adios2.xml")
	: State<NSE>(id, communicator, std::move(lat), adiosConfigPath)
	{}
};



template <typename NSE>
int sim(const std::string& adios_config = "adios2.xml", int RESOLUTION = 2, double viscosity = 1e-5, const std::string& suffix = "")
{
	using idx = typename NSE::TRAITS::idx;
	using real = typename NSE::TRAITS::real;
	using point_t = typename NSE::TRAITS::point_t;
	using lat_t = Lattice<3, real, idx>;

	int block_size = 32;
	real PHYS_LENGTH = 3.; // length in some units (NASA does not specify)
	real PHYS_HEIGHT = 0.2;		  // domain height (physical)
	real PHYS_DEPTH = 0.5;		  // domain depth (physical) FIXED for sine to work correctly


	int Y = floor(RESOLUTION * block_size); // height in pixels --- top and bottom walls  NoDV px
	int wallSize = 3;
	real PHYS_DL = PHYS_HEIGHT / ((real) Y - 2*wallSize); // naive fullway bounce-back but everything is part of the domain

	int X = floor(PHYS_LENGTH / PHYS_DL);  // width in pixels
	int Z = floor(PHYS_DEPTH  / PHYS_DL);  // depth in pixels --- top and bottom walls NoDV px
	real PHYS_VISCOSITY = viscosity; // viscosity as input to analyze when oscillations happen
	real PHYS_VELOCITY = 1.;



	// Diffusive scaling
	//real LBM_VISCOSITY = 0.001;
	//real PHYS_DT = LBM_VISCOSITY / PHYS_VISCOSITY * PHYS_DL * PHYS_DL;	//PHYS_HEIGHT/(real)LBM_HEIGHT;
	// Acoustic scaling
	real LBM_VELOCITY = 0.1;
	real PHYS_DT = PHYS_DL * LBM_VELOCITY/PHYS_VELOCITY;
	real LBM_VISCOSITY = PHYS_VISCOSITY * PHYS_DT / PHYS_DL /PHYS_DL;

	//
	real XSHIFT_RATIO = 0.05;

	point_t PHYS_ORIGIN = {-PHYS_LENGTH*XSHIFT_RATIO, -PHYS_DL*(2.*wallSize-1)/2., -PHYS_DEPTH};

	// initialize the lattice
	lat_t lat;
	lat.global = typename lat_t::CoordinatesType(X, Y, Z);
	lat.physOrigin = PHYS_ORIGIN;
	lat.physDl = PHYS_DL;
	lat.physDt = PHYS_DT;
	lat.physViscosity = PHYS_VISCOSITY;

	#ifdef OSCILLATION_ANALYSIS
	const std::string state_id = fmt::format("sim_bump_NASA_res{:02d}_visc{:07e}_np{:03d}", RESOLUTION, viscosity, TNL::MPI::GetSize(MPI_COMM_WORLD));
	#else
	const std::string state_id = fmt::format("sim_bump_NASA_res{:02d}_np{:03d}", RESOLUTION, TNL::MPI::GetSize(MPI_COMM_WORLD));
	#endif
	StateLocal<NSE> state(state_id, MPI_COMM_WORLD, lat,adios_config);
	if (! state.canCompute())
		return 0;


	state.wallTime = 12*3600;
	state.avg_start_iteration = (int)(10./PHYS_DT); // Start averaging after 10 seconds

	// problem parameters
	state.lbm_inflow_vx = lat.phys2lbmVelocity(PHYS_VELOCITY);
	state.inflow_g = 0.0005656; // already in nondim
	state.average_inflow = PHYS_VELOCITY;
	state.bump_height = 0.05;

	std::cout << "Reynolds number: " << PHYS_VELOCITY*state.bump_height/PHYS_VISCOSITY << std::endl;
	std::cout << "Reynolds based on unit length: " << PHYS_VELOCITY*1./PHYS_VISCOSITY << std::endl;

	state.nse.physFinalTime = 100.;
	state.rise_up_time = 1.;
	state.cnt[PRINT].period = 0.1;

	// add cuts
	state.cnt[OUT2D].period = 1.;
	//state.add2Dcut_X(X / 2, "cutsX/cut_X"); // can be used due to bug with MPI
	//state.add2Dcut_Y(Y / 2, "cutsY/cut_Y");
	state.add2Dcut_Z(Z / 2, "cutsZ/cut_Z");

	state.cnt[OUT3D].period = 5.;
	//state.cnt[OUT3DCUT].period = 100.;
	//state.add3Dcut(X / 4, Y / 4, Z / 4, X / 2, Y / 2, Z / 2, "box");

	state.cnt[PROBE1].period = 1.;
	state.cnt[PROBE2].period = 1.;

	spdlog::info("Starting simulation with checkpointing. Wall time limit: {} seconds", state.wallTime);
	spdlog::info("Creating checkpoints every {} seconds of wall time", state.cnt[SAVESTATE].period);


	execute(state);

	return 0;
}

template <typename TRAITS = TraitsDP> // Change to TraitsDP for ELBM multi-speed
void run(const std::string& adios_config, int resolution, double viscosity)
{
	// D3Q27
	#ifdef CLBM_D3Q27
	using COLL = D3Q27_CUM<TRAITS, D3Q27_EQ_INV_CUM<TRAITS>>;
	using NSE_CONFIG = LBM_CONFIG<
		TRAITS,
		D3Q27_KernelStruct,
		NSE_Data_ConstInflow_PressureGradient<TRAITS>,
		COLL,
		typename COLL::EQ,
		D3Q27_STREAMING<TRAITS>,
		D3Q27_BC_All,
		D3Q27_MACRO_Mean<TRAITS>>;
	#endif
	//using COLL = ;
	#ifdef ELBM_D3Q27
	using NSE_CONFIG = LBM_CONFIG<
		TRAITS,
		D3Q27_KernelStruct,
		NSE_Data_ConstInflow_PressureGradient<TRAITS>,
		D3Q27_GENERAL_SRT<TRAITS, D3Q27_EQ_ENTROPIC2<TRAITS>>,
		D3Q27_EQ_ENTROPIC2<TRAITS>,
		D3Q27_STREAMING<TRAITS>,
		D3Q27_BC_All,
		D3Q27_MACRO_Mean<TRAITS>>;
	#endif

	// D3Q53

	#ifdef D3Q53
	using COLL = D3Q53_SRT<TRAITS, D3Q53_EQ<TRAITS>>;
	using NSE_CONFIG = LBM_CONFIG<
		TRAITS,
		D3Q53_LOOKUP_KernelStruct,
		NSE_Data_ConstInflow_PressureGradient<TRAITS>,
		COLL,
		typename COLL::EQ,
		#ifdef USE_DFMAX3
		D3Q53_STREAMING_THIRD_ARRAY<TRAITS>,
		#else
		D3Q53_STREAMING<TRAITS>,
		#endif
		D3Q53_BC_All,
		D3Q27_MACRO_Mean<TRAITS>>;
	#endif

	#ifdef ELBM_D3Q53
	using COLL = D3Q53_ELBM<TRAITS, D3Q53_EQ<TRAITS>>;
	using NSE_CONFIG = LBM_CONFIG<
		TRAITS,
		D3Q53_LOOKUP_KernelStruct,
		NSE_Data_ConstInflow_PressureGradient<TRAITS>,
		COLL,
		typename COLL::EQ,
		#ifdef USE_DFMAX3
		D3Q53_STREAMING_THIRD_ARRAY<TRAITS>,
		#else
		D3Q53_STREAMING<TRAITS>,
		#endif
		D3Q53_BC_All,
		D3Q27_MACRO_Mean<TRAITS>>;
	#endif


	sim<NSE_CONFIG>(adios_config,resolution,viscosity,"test");
}

int main(int argc, char** argv)
{
	TNLMPI_INIT mpi(argc, argv);

	argparse::ArgumentParser program("sim_bump_NASA");
	program.add_description("3D bump NASA modified simulation.");
	program.add_argument("--adios-config").help("path to ADIOS2 configuration file").default_value(std::string("adios2.xml")).nargs(1);
	program.add_argument("--resolution").help("resolution of the lattice").scan<'i', int>().default_value(1).nargs(1);
	program.add_argument("--viscosity").help("the physical viscosity of the fluid").scan<'g', double>().default_value(1e-4).nargs(1);

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
	const auto viscosity = program.get<double>("--viscosity");

	if (resolution < 1) {
		fmt::println(stderr, "CLI error: resolution must be at least 1");
		return 1;
	}

	run(adios_config, resolution, viscosity);

	return 0;
}
