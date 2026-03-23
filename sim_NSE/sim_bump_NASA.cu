#include <argparse/argparse.hpp>
#include <cstdio>
#include <utility>

#define USE_DFMAX3

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
	int probeCountProfile = 0;

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
		checkpoint.saveLoadAttribute("probeCountProfile",probeCountProfile);

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
		nse.setBoundaryX(nse.lat.global.x() - 1, BC::GEO_OUTFLOW_RIGHT);  // right
		nse.setBoundaryX(nse.lat.global.x() - 2, BC::GEO_OUTFLOW_RIGHT);  // right
		nse.setBoundaryX(nse.lat.global.x() - 3, BC::GEO_OUTFLOW_RIGHT);  // right


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


		// 3) BUMP
		for (int px = 0; px <= nse.lat.global.x(); px++){
		for (int py = 0; py <= nse.lat.global.y(); py++){
		for (int pz = 0; pz <= nse.lat.global.z(); pz++){
			if(isObject(px,py,pz)){
				nse.setMap(px, py, pz, BC::GEO_WALL);
			}
		}}}


		// 4) (Optional) Mark walls next to bump to performed third-array full-way bounce-back
		// ! turn off for single-speed simulations
		mark_next_to_wall();
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

	void mark_next_to_wall(){
		for (int x = NSE::LBM_KS::NoDV; x <= nse.lat.global.x()-NSE::LBM_KS::NoDV; x++){
		for (int y = NSE::LBM_KS::NoDV; y <= nse.lat.global.y()-NSE::LBM_KS::NoDV; y++){
		for (int z = NSE::LBM_KS::NoDV; z <= nse.lat.global.z()-NSE::LBM_KS::NoDV; z++){
			if(nse.blocks.front().hmap(x,y,z)!=BC::GEO_FLUID){continue;}
			bool done = false;
			for(int dx = - NSE::LBM_KS::NoDV; dx < NSE::LBM_KS::NoDV;dx ++){
			for(int dy = - NSE::LBM_KS::NoDV; dy < NSE::LBM_KS::NoDV;dy ++){
			for(int dz = - NSE::LBM_KS::NoDV; dz < NSE::LBM_KS::NoDV;dz ++){
				if(nse.blocks.front().hmap(x+dx,y+dy,z+dz) == BC::GEO_WALL){
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
		const int dirMacro = (dir==0) ? MACRO::e_vx : (dir==1) ? MACRO::e_vy : MACRO::e_vz;
		const int dirx = int(dir==0);
		const int diry = int(dir==1);
		const int dirz = int(dir==2);

		auto& block = nse.blocks.front();
		auto& offset = block.offset;
		auto& local = block.local;


		for (int x = offset.x() + 1; x < offset.x() + local.x() - 1; x++) {
		for (int y = offset.y() + 1; y < offset.y() + local.y() - 1; y++) {
		for (int z = offset.z() + 1; z < offset.z() + local.z() - 1; z++) {
			if(block.hmap(x, y, z) == BC::GEO_WALL && filter(x, y, z)){
				// N_1 = (1,0,0)
		 		if(BC::isFluid(block.hmap(x+1, y, z))){
					const double rho_lbm = (double)block.hmacro(MACRO::e_rho,x+1,y,z);
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
					const double rho_lbm = (double)block.hmacro(MACRO::e_rho,x-1,y,z);
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
					const double rho_lbm = (double)block.hmacro(MACRO::e_rho,x,y+1,z);
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
					const double rho_lbm = (double)block.hmacro(MACRO::e_rho,x,y-1,z);
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
					const double rho_lbm = (double)block.hmacro(MACRO::e_rho,x,y,z+1);
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
					const double rho_lbm = (double)block.hmacro(MACRO::e_rho,x,y,z-1);
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
		const int dirMacro = (dir==0) ? MACRO::e_vx : (dir==1) ? MACRO::e_vy : MACRO::e_vz;
		const int dirx = int(dir==0);
		const int diry = int(dir==1);
		const int dirz = int(dir==2);

		auto& block = nse.blocks.front();
		auto& offset = block.offset;
		auto& local = block.local;


		for (int x = offset.x() + 1; x < offset.x() + local.x() - 1; x++) {
		for (int y = offset.y() + 1; y < offset.y() + local.y() - 1; y++) {
		for (int z = offset.z() + 1; z < offset.z() + local.z() - 1; z++) {
			if(block.hmap(x, y, z) == BC::GEO_WALL && filter(x, y, z)){
				// N_1 = (1,0,0)
		 		if(BC::isFluid(block.hmap(x+1, y, z))){
					const double rho_lbm = (double)block.hmacro(MACRO::e_rho,x+1,y,z);
					const double pressure =   nse.lat.lbm2physVelocity(nse.lat.lbm2physVelocity(T0*(rho_lbm-1)));
					// pressure
					local_drag -= dirx*pressure;
				}
				// N_2 = (-1,0,0)
		 		if(BC::isFluid(block.hmap(x-1, y, z))){
					const double rho_lbm = (double)block.hmacro(MACRO::e_rho,x-1,y,z);
					const double pressure = nse.lat.lbm2physVelocity(nse.lat.lbm2physVelocity(T0*(rho_lbm-1)));
					// pressure
					local_drag += dirx*pressure;
				}
				// N_3 = (0,1,0)
		 		if(BC::isFluid(block.hmap(x, y+1, z))){
					const double rho_lbm = (double)block.hmacro(MACRO::e_rho,x,y+1,z);
					const double pressure = nse.lat.lbm2physVelocity(nse.lat.lbm2physVelocity(T0*(rho_lbm-1)));
					// pressure
					local_drag -= diry*pressure;
				}
				// N_4 = (0,-1,0)
		 		if(BC::isFluid(block.hmap(x, y-1, z))){
					const double rho_lbm = (double)block.hmacro(MACRO::e_rho,x,y-1,z);
					const double pressure = nse.lat.lbm2physVelocity(nse.lat.lbm2physVelocity(T0*(rho_lbm-1)));
					// pressure
					local_drag += diry*pressure;
				}
				// N_5 = (0,0,1)
		 		if(BC::isFluid(block.hmap(x, y-1, z+1))){
					const double rho_lbm = (double)block.hmacro(MACRO::e_rho,x,y,z+1);
					const double pressure = nse.lat.lbm2physVelocity(nse.lat.lbm2physVelocity(T0*(rho_lbm-1)));
					// pressure
					local_drag -= dirz*pressure;
				}
				// N_6 = (0,0,-1)
		 		if(block.hmap(x, y, z-1) == BC::GEO_FLUID){
					const double rho_lbm = (double)block.hmacro(MACRO::e_rho,x,y,z-1);
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
			if(block.hmap(x, y, z) == BC::GEO_WALL && filter(x, y, z)){
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
		const double H = bump_height; // height of bump, 0.05 in origin
		const double L = 1.5; // length of bump
		const double W = 0.5; // width of bump
		const double Uoverline = average_inflow; // average inflow velocity
		real C_D = 2.*integrate_stress_tensor_general([this](int ix,int iy,int iz){ return this->isObject(ix, iy, iz);},0)/(Uoverline*Uoverline)/(H*W);
		real C_D_P = 2.*integrate_stress_tensor_general_only_pressure([this](int ix,int iy,int iz){ return this->isObject(ix, iy, iz);},0)/(Uoverline*Uoverline)/(H*W);
		real C_D_nu = 2.*integrate_stress_tensor_general_only_viscous([this](int ix,int iy,int iz){ return this->isObject(ix, iy, iz);},0)/(Uoverline*Uoverline)/(H*W);
		real C_S = 2.*integrate_stress_tensor_general([this](int ix,int iy,int iz){ return this->isObject(ix, iy, iz);},1)/(Uoverline*Uoverline)/(L*H);
		real C_L = 2.*integrate_stress_tensor_general([this](int ix,int iy,int iz){ return this->isObject(ix, iy, iz);},2)/(Uoverline*Uoverline)/(L*W);

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
				C_S,
				C_L
			);
		}
	}
	void dragprofile(){
		const double H = bump_height; // height of bump, 0.05 in origin
		//const double L = 1.5; // length of bump
		//const double W = 0.5; // width of bump
		const double Uoverline = average_inflow; // average inflow velocity
		const double delta_x = (double)nse.lat.physDl;


		if (nse.rank == 0){
			// empty file
			const char* iotype = (probeCountProfile == 0) ? "wt" : "at";
			probeCountProfile += 1;
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
	}
	void probe2() override {
		dragprofile();
  	}

	[[nodiscard]] std::vector<std::string> getOutputDataNames() const override
	{
		// return all quantity names used in outputData
		return {"lbm_density", "lbm_density_fluctuation", "velocity_x", "velocity_y", "velocity_z","mywall"};
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
			real rise_up_lbm_inflow_vx = nse.physTime()/rise_up_time;
			if(rise_up_lbm_inflow_vx > 1.){
				block.data.inflow_vx = lbm_inflow_vx;
			}else{
				block.data.inflow_vx = rise_up_lbm_inflow_vx*lbm_inflow_vx;
			}
			block.data.inflow_vy = 0;
			block.data.inflow_vz = 0;
			block.data.inflow_g = inflow_g;
			block.data.no1oT0 = 1./NSE::LBM_KS::T0;
		}
	}

	StateLocal(const std::string& id, const TNL::MPI::Comm& communicator, lat_t lat, const std::string& adiosConfigPath = "adios2.xml")
	: State<NSE>(id, communicator, std::move(lat), adiosConfigPath)
	{}
};

#define OSCILLATION_ANALYSIS

template <typename NSE>
int sim(const std::string& adios_config = "adios2.xml", int RESOLUTION = 2, double viscosity = 1e-4)
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
	real XSHIFT_RATIO = 0.2;

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
	state.wallTime = 12*3600;

	// problem parameters
	state.lbm_inflow_vx = lat.phys2lbmVelocity(PHYS_VELOCITY);
	state.inflow_g = 0.0005656; // already in nondim
	state.average_inflow = PHYS_VELOCITY;
	state.bump_height = 0.05;

	std::cout << "Reynolds number: " << PHYS_VELOCITY*state.bump_height/PHYS_VISCOSITY << std::endl;
	std::cout << "Reynolds based on unit length: " << PHYS_VELOCITY*1./PHYS_VISCOSITY << std::endl;

	state.nse.physFinalTime = 50.;
	state.rise_up_time = 1.;
	state.cnt[PRINT].period = 0.1;

	// add cuts
	state.cnt[OUT2D].period = 1.;
	//state.add2Dcut_X(X / 2, "cutsX/cut_X"); // can be used due to bug with MPI
	//state.add2Dcut_Y(Y / 2, "cutsY/cut_Y");
	state.add2Dcut_Z(Z / 2, "cutsZ/cut_Z");

	state.cnt[OUT3D].period = 10.;
	state.cnt[OUT3DCUT].period = 100.;
	state.add3Dcut(X / 4, Y / 4, Z / 4, X / 2, Y / 2, Z / 2, "box");

	state.cnt[PROBE1].period = 1.;
	state.cnt[PROBE2].period = -10.;

	spdlog::info("Starting simulation with checkpointing. Wall time limit: {} seconds", state.wallTime);
	spdlog::info("Creating checkpoints every {} seconds of wall time", state.cnt[SAVESTATE].period);


	execute(state);

	return 0;
}

template <typename TRAITS = TraitsSP> // Change to TraitsDP for ELBM multi-speed
void run(const std::string& adios_config, int resolution, double viscosity)
{
	// D3Q27
	// using COLL = D3Q27_CUM<TRAITS, D3Q27_EQ_INV_CUM<TRAITS>>;
	// using NSE_CONFIG = LBM_CONFIG<
	// 	TRAITS,
	// 	D3Q27_KernelStruct,
	// 	NSE_Data_ConstInflow_PressureGradient<TRAITS>,
	// 	COLL,
	// 	typename COLL::EQ,
	// 	D3Q27_STREAMING<TRAITS>,
	// 	D3Q27_BC_All,
	// 	D3Q27_MACRO_Default<TRAITS>>;
	// using COLL = D3Q27_GENERAL_SRT<TRAITS, D3Q27_EQ_ENTROPIC2<TRAITS>>;
	// using NSE_CONFIG = LBM_CONFIG<
	// 	TRAITS,
	// 	D3Q27_KernelStruct,
	// 	NSE_Data_DoubleParabolic<TRAITS>,
	// 	COLL,
	// 	typename COLL::EQ,
	// 	D3Q27_STREAMING<TRAITS>,
	// 	D3Q27_BC_All,
	// 	D3Q27_MACRO_Default<TRAITS>>;

	// D3Q53
	using COLL = D3Q53_SRT<TRAITS, D3Q53_EQ<TRAITS>>;
	using NSE_CONFIG = LBM_CONFIG<
		TRAITS,
		D3Q53_LOOKUP_KernelStruct,
		NSE_Data_ConstInflow_PressureGradient<TRAITS>,
		COLL,
		typename COLL::EQ,
		D3Q53_STREAMING_THIRD_ARRAY<TRAITS>,
		D3Q53_BC_All,
		D3Q53_MACRO_Default<TRAITS>>;

	// using COLL = D3Q53_ELBM<TRAITS, D3Q53_EQ<TRAITS>>;
	// using NSE_CONFIG = LBM_CONFIG<
	// 	TRAITS,
	// 	D3Q53_LOOKUP_KernelStruct,
	// 	NSE_Data_ConstInflow_PressureGradient<TRAITS>,
	// 	COLL,
	// 	typename COLL::EQ,
	// 	D3Q53_STREAMING_THIRD_ARRAY<TRAITS>,
	// 	D3Q53_BC_All,
	// 	D3Q53_MACRO_Default<TRAITS>>;


	sim<NSE_CONFIG>(adios_config,resolution,viscosity);
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
