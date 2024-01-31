#include "lbm3d/core.h"

// 3D test problem: forcing/input velocity
// analytical solution for rectangular duct: forcing accelerated

enum Scaling
{
	STRONG_SCALING,
	WEAK_SCALING_1D,
	WEAK_SCALING_3D,
};

template < typename TRAITS >
struct NSE_Data_XProfileInflow : NSE_Data < TRAITS >
{
	using idx = typename TRAITS::idx;
	using dreal = typename TRAITS::dreal;

//	dreal inflow_rho=no1;
	dreal* vx_profile=NULL;
	idx size_y=0;

	template< typename LBM_KS >
	CUDA_HOSTDEV void inflow(LBM_KS &KS, idx x, idx y, idx z)
	{
//		KS.rho = inflow_rho;
		KS.vx  = vx_profile[y+z*size_y];
		KS.vy  = 0;
		KS.vz  = 0;
	}
};

// uloha: periodic, forcing accelerated
template < typename NSE >
struct StateLocal : State<NSE>
{
	using TRAITS = typename NSE::TRAITS;
	using BC = typename NSE::BC;
	using MACRO = typename NSE::MACRO;
	using BLOCK = LBM_BLOCK< NSE >;

	using State<NSE>::nse;
	using State<NSE>::vtk_helper;

	using idx = typename TRAITS::idx;
	using real = typename TRAITS::real;
	using dreal = typename TRAITS::dreal;
	using point_t = typename TRAITS::point_t;
	using lat_t = Lattice<3, real, idx>;

#ifdef HAVE_MPI
	TNL::Containers::DistributedNDArray< typename TRAITS::template array3d<real, TNL::Devices::Host> > an_cache;
#else
	typename TRAITS::template array3d<real, TNL::Devices::Host> an_cache;
#endif
	int an_n=50;

	int errors_count;
	real* l1errors;
	int error_idx = 0;

	real raw_analytical_ux(int n, idx lbm_y, idx lbm_z)
	{
		if (lbm_y==0 || lbm_y==nse.lat.global.y()-1 || lbm_z==0 || lbm_z==nse.lat.global.z()-1)
			return 0;

		real a = nse.lat.global.y()/2.0 - 1.0;
		real b = nse.lat.global.z()/2.0 - 1.0;
		real y = ((real)lbm_y + 0.5 - nse.lat.global.y()/2.)/a;
		real z = ((real)lbm_z + 0.5 - nse.lat.global.z()/2.)/a;
		real b_ku_a = b/a;
		real sum=0;
		real minusonek=1.0;
		real kkk;
		real omega=PI/2.0;
		for (int k=0;k<=n;k++)
		{
			kkk=2.0*k+1.;
			sum += minusonek*(1.0 - exp( omega*kkk*(z-b_ku_a) ) * (1.0 + exp( -omega*2.0*kkk*z ))/(1.0 + exp(-omega*2.0*kkk*b_ku_a) ))*cos( omega*kkk*y )/kkk/kkk/kkk;
			minusonek *= -1.0;
		}

//		real coef = (nse.blocks.front().data.fx != 0) ? nse.blocks.front().data.fx : nse.blocks.front().data.inflow_vx;
		real coef = nse.blocks.front().data.fx;
		return coef * 16.0*a*a/PI/PI/PI*sum/nse.lbmViscosity();
	}

	real analytical_ux(idx lbm_y, idx lbm_z)
	{
		if (an_cache.getData() == nullptr) {
			cache_analytical();
		}

		return an_cache(0, lbm_y, lbm_z);
	}

	void cache_analytical()
	{
		const auto& block = nse.blocks.front();
		an_cache.setSizes(1, block.global.y(), block.global.z());
		#ifdef HAVE_MPI
		an_cache.template setDistribution< 1 >(block.offset.y(), block.offset.y() + block.local.y(), block.communicator);
		an_cache.template setDistribution< 2 >(block.offset.z(), block.offset.z() + block.local.z(), block.communicator);
		an_cache.allocate();
		#endif

		#pragma omp parallel for schedule(static) collapse(2)
		for (idx z = block.offset.z(); z < block.offset.z() + block.local.z(); z++)
		for (idx y = block.offset.y(); y < block.offset.y() + block.local.y(); y++)
			an_cache(0, y, z) = raw_analytical_ux(an_n, y, z);
	}

	virtual void setupBoundaries()
	{
//		if (nse.blocks.front().data.inflow_vx != 0)
		if (nse.blocks.front().data.vx_profile)
		{
			nse.setBoundaryX(0, BC::GEO_INFLOW); 		// left
//			nse.setBoundaryX(0, BC::GEO_INFLOW_FREE_RHO); 		// left
//			nse.setBoundaryX(nse.lat.global.x()-1, BC::GEO_OUTFLOW_EQ);		// right
			nse.setBoundaryX(nse.lat.global.x()-1, BC::GEO_OUTFLOW_RIGHT);		// right
//			nse.setBoundaryX(nse.lat.global.x()-1, BC::GEO_OUTFLOW_RIGHT_INTERP);		// right
		} else
		{
			nse.setBoundaryX(0, BC::GEO_PERIODIC); 		// left
			nse.setBoundaryX(nse.lat.global.x()-1, BC::GEO_PERIODIC);		// right
		}

		nse.setBoundaryZ(1, BC::GEO_WALL);		// top
		nse.setBoundaryZ(nse.lat.global.z()-2, BC::GEO_WALL);	// bottom
		nse.setBoundaryY(1, BC::GEO_WALL); 		// back
		nse.setBoundaryY(nse.lat.global.y()-2, BC::GEO_WALL);		// front

		// extra layer needed due to A-A pattern
		nse.setBoundaryZ(0, BC::GEO_NOTHING);		// top
		nse.setBoundaryZ(nse.lat.global.z()-1, BC::GEO_NOTHING);	// bottom
		nse.setBoundaryY(0, BC::GEO_NOTHING); 		// back
		nse.setBoundaryY(nse.lat.global.y()-1, BC::GEO_NOTHING);		// front
	}

	virtual bool outputData(const BLOCK& block, int index, int dof, char *desc, idx x, idx y, idx z, real &value, int &dofs)
	{
		int k=0;
		if (index==k++) return vtk_helper("lbm_density", block.hmacro(MACRO::e_rho,x,y,z), 1, desc, value, dofs);
		if (index==k++) return vtk_helper("lbm_delta_density", block.hmacro(MACRO::e_rho,x,y,z) - 1.0, 1, desc, value, dofs);
		if (index==k++)
		{
			switch (dof)
			{
				case 0: return vtk_helper("lbm_velocity", block.hmacro(MACRO::e_vx,x,y,z), 3, desc, value, dofs);
				case 1: return vtk_helper("lbm_velocity", block.hmacro(MACRO::e_vy,x,y,z), 3, desc, value, dofs);
				case 2: return vtk_helper("lbm_velocity", block.hmacro(MACRO::e_vz,x,y,z), 3, desc, value, dofs);
			}
		}
		if (index==k++) return vtk_helper("lbm_analytical_ux", analytical_ux(y, z), 1, desc, value, dofs);
		if (index==k++) return vtk_helper("lbm_ux", block.hmacro(MACRO::e_vx,x,y,z), 1, desc, value, dofs);
		if (index==k++) return vtk_helper("lbm_uy", block.hmacro(MACRO::e_vy,x,y,z), 1, desc, value, dofs);
		if (index==k++) return vtk_helper("lbm_uz", block.hmacro(MACRO::e_vz,x,y,z), 1, desc, value, dofs);
		if (index==k++) return vtk_helper("lbm_error_ux", fabs(block.hmacro(MACRO::e_vx,x,y,z) - analytical_ux(y,z)), 1, desc, value, dofs);
		if (index==k++)
		{
			switch (dof)
			{
				case 0: return vtk_helper("velocity", nse.lbm2physVelocity(block.hmacro(MACRO::e_vx,x,y,z)), 3, desc, value, dofs);
				case 1: return vtk_helper("velocity", nse.lbm2physVelocity(block.hmacro(MACRO::e_vy,x,y,z)), 3, desc, value, dofs);
				case 2: return vtk_helper("velocity", nse.lbm2physVelocity(block.hmacro(MACRO::e_vz,x,y,z)), 3, desc, value, dofs);
			}
		}
		if (index==k++) return vtk_helper("lbm_analytical_ux", nse.lbm2physVelocity(analytical_ux(y, z)), 1, desc, value, dofs);
		if (index==k++) return vtk_helper("lbm_ux", nse.lbm2physVelocity(block.hmacro(MACRO::e_vx,x,y,z)), 1, desc, value, dofs);
		if (index==k++) return vtk_helper("lbm_uy", nse.lbm2physVelocity(block.hmacro(MACRO::e_vy,x,y,z)), 1, desc, value, dofs);
		if (index==k++) return vtk_helper("lbm_uz", nse.lbm2physVelocity(block.hmacro(MACRO::e_vz,x,y,z)), 1, desc, value, dofs);
		if (index==k++) return vtk_helper("lbm_error_ux", nse.lbm2physVelocity(fabs(block.hmacro(MACRO::e_vx,x,y,z) - analytical_ux(y,z))), 1, desc, value, dofs);
		return false;
	}

/*
	// clear Lattice + boundary setup
	virtual void resetLattice(real irho, real ivx, real ivy, real ivz)
	{
		for (int x = 0; x < nse.X;x++)
		for (int y = 0; y < nse.Y;y++)
		for (int z = 0; z < nse.Z;z++)
		{
			dreal rho= no1;
			dreal vx= 0;//(lbmInputVelocityX!=0) ? (dreal)analytical_ux(y,z) : 0;
			dreal vy= 0;
			dreal vz= 0;
			nse.setEqLat(x,y,z,rho,vx,vy,vz);
		}

	}
*/
	virtual void probe1()
	{
		// compute exact error
		// uy,uz should be zero
		// use openmp
		// warning: ux,uy,uz are in LBM units ... measure that in phys units
		// analytical ux has no dpdx --- add
		// compute error
		real local_l1sum=0;
		real local_l2sum=0;
		//real local_la1sum=0;
		//real local_la2sum=0;
		for (int i = nse.blocks.front().offset.x() + 1; i < nse.blocks.front().offset.x() + nse.blocks.front().local.x() - 1; i++)
		for (int j = nse.blocks.front().offset.y() + 1; j < nse.blocks.front().offset.y() + nse.blocks.front().local.y() - 1; j++)
		for (int k = nse.blocks.front().offset.z() + 1; k < nse.blocks.front().offset.z() + nse.blocks.front().local.z() - 1; k++)
		{
			real an = analytical_ux(j,k);
			real diff = fabs(nse.blocks.front().hmacro(MACRO::e_vx,i,j,k) - an);
			//local_la1sum += an;
			//local_la2sum += SQ(an);
			local_l1sum += diff;
			local_l2sum += SQ(diff);
		}

		// MPI reduction of the local results
		real l1sum=TNL::MPI::reduce(local_l1sum, MPI_SUM, MPI_COMM_WORLD);
		real l2sum=TNL::MPI::reduce(local_l2sum, MPI_SUM, MPI_COMM_WORLD);
		//real la1sum=TNL::MPI::reduce(local_la1sum, MPI_SUM, MPI_COMM_WORLD);
		//real la2sum=TNL::MPI::reduce(local_la2sum, MPI_SUM, MPI_COMM_WORLD);

		// considering PHYS_DL, converting to physical units
		real l1error_phys = l1sum*nse.lat.physDl*nse.lat.physDl*nse.lat.physDl;
		real l2error_phys = l2sum*nse.lat.physDl*nse.lat.physDl*nse.lat.physDl;
		l2error_phys = sqrt(l2error_phys);
		l1error_phys = nse.lbm2physVelocity(l1error_phys);
		l2error_phys = nse.lbm2physVelocity(l2error_phys);

		// dynamic stopping criterion
		real threshold = 1e-4;
		real threshold_stddev = 1e-3;
		real l1prev = 0.0;
		for (int i = 0; i < errors_count; i++) l1prev += l1errors[i];
		l1prev /= errors_count;
		real stddev = 0.0;
		for (int i = 0; i < errors_count; i++) stddev += SQ(l1errors[i] - l1prev);
		stddev /= (errors_count-1);
		stddev = sqrt(stddev);
		real stopping = abs(l1prev - l1error_phys) / l1error_phys;
		if( stopping < threshold && stddev < threshold_stddev ) nse.terminate = true;

		error_idx = (error_idx + 1) % errors_count;
		l1errors[error_idx] = l1error_phys;

		if (nse.rank == 0)
			spdlog::info("at t={:1.2f}s, iterations={:d} l1error_phys={:e} l2error_phys={:e} stopping={:e}",
				nse.physTime(), nse.iterations, l1error_phys, l2error_phys, stopping);
	}


	StateLocal(const std::string& id, const TNL::MPI::Comm& communicator, lat_t ilat, real iphysViscosity, real iphysDt, bool periodic_lattice, int RES)
		: State<NSE>(id, communicator, ilat, iphysViscosity, iphysDt, periodic_lattice)
	{
		errors_count = 10;
		l1errors = new real[errors_count];
		for (int i = 0; i < errors_count; i++) l1errors[i] = 1;
	}

	~StateLocal()
	{
		delete[] l1errors;
	}
};

template < typename NSE >
int sim02(int RES=1, bool use_forcing=true, Scaling scaling=STRONG_SCALING)
{
	using idx = typename NSE::TRAITS::idx;
	using real = typename NSE::TRAITS::real;
	using dreal = typename NSE::TRAITS::dreal;
	using point_t = typename NSE::TRAITS::point_t;
	using lat_t = Lattice<3, real, idx>;

	int block_size=32;
	int LBM_X = block_size;
	if (!use_forcing) LBM_X *= RES;
	int LBM_Y = RES*block_size;
	int LBM_Z = RES*block_size;
	if (scaling == WEAK_SCALING_1D)
		LBM_X *= TNL::MPI::GetSize(MPI_COMM_WORLD);
	else if (scaling == WEAK_SCALING_3D) {
		// NOTE: scale volume by nproc, preserve the proportions of the domain
		const real factor = std::cbrt(TNL::MPI::GetSize(MPI_COMM_WORLD));
		LBM_X = std::round(LBM_X * factor);
		LBM_Y = std::round(LBM_Y * factor);
		LBM_Z = std::round(LBM_Z * factor);
	}
	// NOTE: LBM_VISCOSITY must be less than 1/6
//	real LBM_VISCOSITY = 0.01*RES;
	real LBM_VISCOSITY = std::min(0.1, 0.01*RES);
	real PHYS_VISCOSITY = 1.5e-5;// [m^2/s] fluid viscosity air: 1.81e-5
	real PHYS_HEIGHT = 0.25;
	real PHYS_DL = PHYS_HEIGHT / real(LBM_Z - 2);
	real PHYS_DT = LBM_VISCOSITY / PHYS_VISCOSITY * PHYS_DL * PHYS_DL;
	point_t PHYS_ORIGIN = {0., 0., 0.};

	// initialize the lattice
	lat_t lat;
	lat.global = typename lat_t::CoordinatesType( LBM_X, LBM_Y, LBM_Z );
	lat.physOrigin = PHYS_ORIGIN;
	lat.physDl = PHYS_DL;

	const char* prec = (std::is_same<dreal,float>::value) ? "float" : "double";
	const std::string state_id = fmt::format("sim_2_{}_{}_{}_res_{}_np_{}", NSE::COLL::id, prec, (use_forcing)?"forcing":"velocity", RES, TNL::MPI::GetSize(MPI_COMM_WORLD));
	StateLocal<NSE> state(state_id, MPI_COMM_WORLD, lat, PHYS_VISCOSITY, PHYS_DT, use_forcing, RES);

	if (state.isMark())
		return 0;

	if (state.nse.blocks.front().local.x() <= 2) {
		std::cout << "Local block size " << state.nse.blocks.front().local.x() << " is too small, skipping this resolution." << std::endl;
		return 0;
	}

// NOTE: this is for NSE_Data_ConstInflow
//	if (use_forcing)
//	{
//		state.nse.blocks.front().data.fx = state.nse.phys2lbmForce(1e-4);
//		state.nse.blocks.front().data.fy = 0;
//		state.nse.blocks.front().data.fz = 0;
//		state.nse.blocks.front().data.inflow_vx = 0;
//		state.nse.blocks.front().data.inflow_vy = 0;
//		state.nse.blocks.front().data.inflow_vz = 0;
//	} else
//	{
//		state.nse.blocks.front().data.fx = 0;
//		state.nse.blocks.front().data.fy = 0;
//		state.nse.blocks.front().data.fz = 0;
//		state.nse.blocks.front().data.inflow_vx = state.nse.phys2lbmVelocity(2e-6);
//		state.nse.blocks.front().data.inflow_vy = 0;
//		state.nse.blocks.front().data.inflow_vz = 0;
//	}

// NOTE: this is for NSE_Data_XProfileInflow
	dreal force = 1e-4;
	if (use_forcing)
	{
		state.nse.blocks.front().data.fx = state.nse.phys2lbmForce(force);
		state.nse.blocks.front().data.fy = 0;
		state.nse.blocks.front().data.fz = 0;
		state.nse.blocks.front().data.vx_profile = NULL;
	} else
	{
		// calculate analytical solution using forcing just like above
		state.nse.blocks.front().data.fx = state.nse.phys2lbmForce(force);
		state.nse.blocks.front().data.fy = 0;
		state.nse.blocks.front().data.fz = 0;
		state.cache_analytical();
		// reset the forcing for the LBM simulation
		state.nse.blocks.front().data.fx = 0;

		// allocate array for the inflow profile
		#ifdef USE_CUDA
			cudaMalloc((void**)&state.nse.blocks.front().data.vx_profile, state.nse.blocks.front().local.y()*state.nse.blocks.front().local.z()*sizeof(dreal));
		#else
			state.nse.blocks.front().data.vx_profile = new dreal[state.nse.blocks.front().local.y()*state.nse.blocks.front().local.z()];
		#endif

		#ifdef USE_CUDA
			// convert analytical solution from double to float
			std::unique_ptr< dreal[] > analytical{ new dreal[ state.nse.blocks.front().local.y()*state.nse.blocks.front().local.z() ] };
			for (int j = 0; j < state.nse.blocks.front().local.y(); j++)
			for (int k = 0; k < state.nse.blocks.front().local.z(); k++)
				analytical[k*state.nse.blocks.front().local.y()+j] = state.analytical_ux(state.nse.blocks.front().offset.y() + j, state.nse.blocks.front().offset.z() + k);
			// copy the analytical profile to the GPU
			TNL::Backend::memcpy(state.nse.blocks.front().data.vx_profile, analytical.get(), state.nse.blocks.front().local.y()*state.nse.blocks.front().local.z()*sizeof(dreal), TNL::Backend::MemcpyHostToDevice);
		#else
			for (int j = 0; j < state.nse.blocks.front().local.y(); j++)
			for (int k = 0; k < state.nse.blocks.front().local.z(); k++)
				state.nse.blocks.front().data.vx_profile[k*state.nse.blocks.front().local.y()+j] = state.analytical_ux(state.nse.blocks.front().offset.y() + j, state.nse.blocks.front().offset.z() + k);
		#endif
		state.nse.blocks.front().data.size_y = state.nse.blocks.front().local.y();
	}

	state.cnt[PRINT].period = 10.0;
	state.cnt[PROBE1].period = 10.0;// / RES;
//	state.nse.physFinalTime = PHYS_DT * 1e7;
	state.nse.physFinalTime = 100; //5000;
//	state.cnt[VTK2D].period = 1.0;

	if (scaling == WEAK_SCALING_3D) {
		// TRICK to keep the benchmark fast: decrease the periods and physFinalTime to keep the compute time (more or less) constant
		const real factor = (LBM_Y - 2) / real(block_size * RES - 2) * RES / 2;
		state.cnt[PRINT].period /= factor;
		state.cnt[PROBE1].period /= factor;
		state.nse.physFinalTime /= factor;
	}

	spdlog::info("PHYS_DL = {:e}", PHYS_DL);
//	spdlog::info("in lbm units: forcing={:e} velocity={:e}", state.nse.blocks.front().data.fx, state.nse.blocks.front().data.inflow_vx);
	spdlog::info("in lbm units: forcing={:e}", force);

	// add cuts
//	state.add2Dcut_X(LBM_X/2,"cut_X");
//	state.add2Dcut_Y(LBM_Y/2,"cut_Y");
//	state.add2Dcut_Z(LBM_Z/2,"cut_Z");
//	state.add1Dcut_Z(LBM_X/2*PHYS_DL, LBM_Y/2*PHYS_DL, "cut_Z");

	execute(state);

	// deallocate inflow data
	if (state.nse.blocks.front().data.vx_profile)
	{
		#ifdef USE_CUDA
			cudaFree(state.nse.blocks.front().data.vx_profile);
		#else
			delete[] state.nse.blocks.front().data.vx_profile;
		#endif
	}

	return 0;
}

template < typename TRAITS=TraitsSP >
void run()
{
	using COLL = D3Q27_CUM<TRAITS, D3Q27_EQ_INV_CUM<TRAITS> >;
//	using COLL = D3Q27_FCLBM<TRAITS>;
//	using COLL = D3Q27_SRT<TRAITS>;
//	using COLL = D3Q27_SRT_WELL<TRAITS>;
//	using COLL = D3Q27_SRT_MODIF_FORCE<TRAITS>;
//	using COLL = D3Q27_BGK<TRAITS>;
//	using COLL = D3Q27_KBC_N1<TRAITS>;
//	using COLL = D3Q27_CUM<TRAITS>;
//	using COLL = D3Q27_CLBM<TRAITS>;
//	using COLL = D3Q27_CLBM_WELL<TRAITS>;
//	using COLL = D3Q27_CUM_SGS<TRAITS>;
//	using COLL = D3Q27_CUM_FIX<TRAITS>;
//	using COLL = D3Q27_CUM_WELL<TRAITS>;

	using NSE_CONFIG = LBM_CONFIG<
				TRAITS,
				D3Q27_KernelStruct,
				NSE_Data_XProfileInflow< TRAITS >,
				COLL,
				typename COLL::EQ,
				D3Q27_STREAMING< TRAITS >,
				D3Q27_BC_All,
				D3Q27_MACRO_Default< TRAITS >,
				D3Q27_MACRO_Void< TRAITS >
			>;

	bool use_forcing = false;
	Scaling scaling = STRONG_SCALING;
//	Scaling scaling = WEAK_SCALING;
//	Scaling scaling = WEAK_SCALING_3D;
	for (int i = 2; i <= 4; i++)
	{
//		int res=4;
		int res = pow(2, i);
		sim02<NSE_CONFIG>(res, use_forcing, scaling);
	}
}

int main(int argc, char **argv)
{
	TNLMPI_INIT mpi(argc, argv);
	run();
	return 0;
}
