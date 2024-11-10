#include "lbm3d/core.h"
#include "lbm3d/lagrange_3D.h"
#include "lbm3d/obstacles_ibm.h"

// bouncing ball in 3D
// IBM-LBM

template < typename TRAITS >
struct MacroLocal : D3Q27_MACRO_Base< TRAITS >
{
	using dreal = typename TRAITS::dreal;
	using idx = typename TRAITS::idx;

	enum { e_fx, e_fy, e_fz, e_vx, e_vy, e_vz, e_rho, N };

	template < typename LBM_DATA, typename LBM_KS >
	CUDA_HOSTDEV static void outputMacro(LBM_DATA &SD, LBM_KS &KS, idx x, idx y, idx z)
	{
		SD.macro(e_rho, x, y, z) = KS.rho;
		SD.macro(e_vx, x, y, z)  = KS.vx;
		SD.macro(e_vy, x, y, z)  = KS.vy;
		SD.macro(e_vz, x, y, z)  = KS.vz;
	}

	template < typename LBM_DATA >
	CUDA_HOSTDEV static void zeroForces(LBM_DATA &SD, idx x, idx y, idx z)
	{
		SD.macro(e_fx, x, y, z) = 0;
		SD.macro(e_fy, x, y, z) = 0;
		SD.macro(e_fz, x, y, z) = 0;
	}

	template < typename LBM_DATA, typename LBM_KS >
	CUDA_HOSTDEV static void copyQuantities(LBM_DATA &SD, LBM_KS &KS, idx x, idx y, idx z)
	{
		KS.lbmViscosity = SD.lbmViscosity;
		KS.fx = SD.macro(e_fx, x, y, z);
		KS.fy = SD.macro(e_fy, x, y, z);
		KS.fz = SD.macro(e_fz, x, y, z);
	}
};

template < typename NSE >
struct StateLocal : State<NSE>
{
	using TRAITS = typename NSE::TRAITS;
	using BC = typename NSE::BC;
	using MACRO = typename NSE::MACRO;
	using BLOCK = LBM_BLOCK< NSE >;

	using State<NSE>::nse;
	using State<NSE>::ibm;
	using State<NSE>::vtk_helper;
	using State<NSE>::id;

	using idx = typename TRAITS::idx;
	using real = typename TRAITS::real;
	using dreal = typename TRAITS::dreal;
	using point_t = typename TRAITS::point_t;
	using lat_t = Lattice<3, real, idx>;

	dreal lbm_inflow_vx = 0;
	bool firstrun=true;
	dreal ball_diameter=0.01;
	point_t ball_c;
	dreal ball_amplitude = 0;
	dreal ball_period = 1;

	virtual bool outputData(const BLOCK& block, int index, int dof, char *desc, idx x, idx y, idx z, real &value, int &dofs)
	{
		int k=0;
		if (index==k++)
		{
			switch (dof)
			{
				case 0: return vtk_helper("lbm_velocity", block.hmacro(MACRO::e_vx,x,y,z), 3, desc, value, dofs);
				case 1: return vtk_helper("lbm_velocity", block.hmacro(MACRO::e_vy,x,y,z), 3, desc, value, dofs);
				case 2: return vtk_helper("lbm_velocity", block.hmacro(MACRO::e_vz,x,y,z), 3, desc, value, dofs);
			}
		}
		if (index==k++)
		{
			switch (dof)
			{
				case 0: return vtk_helper("lbm_force", block.hmacro(MACRO::e_fx,x,y,z), 3, desc, value, dofs);
				case 1: return vtk_helper("lbm_force", block.hmacro(MACRO::e_fy,x,y,z), 3, desc, value, dofs);
				case 2: return vtk_helper("lbm_force", block.hmacro(MACRO::e_fz,x,y,z), 3, desc, value, dofs);
			}
		}
		if (index==k++) return vtk_helper("lbm_density", block.hmacro(MACRO::e_rho,x,y,z), 1, desc, value, dofs);
		if (index==k++) return vtk_helper("lbm_density_fluctuation", block.hmacro(MACRO::e_rho,x,y,z)-1.0, 1, desc, value, dofs);
		if (index==k++)
		{
			switch (dof)
			{
				case 0: return vtk_helper("velocity", nse.lat.lbm2physVelocity(block.hmacro(MACRO::e_vx,x,y,z)), 3, desc, value, dofs);
				case 1: return vtk_helper("velocity", nse.lat.lbm2physVelocity(block.hmacro(MACRO::e_vy,x,y,z)), 3, desc, value, dofs);
				case 2: return vtk_helper("velocity", nse.lat.lbm2physVelocity(block.hmacro(MACRO::e_vz,x,y,z)), 3, desc, value, dofs);
			}
		}
		if (index==k++)
		{
			switch (dof)
			{
				case 0: return vtk_helper("force", nse.lat.lbm2physForce(block.hmacro(MACRO::e_fx,x,y,z)), 3, desc, value, dofs);
				case 1: return vtk_helper("force", nse.lat.lbm2physForce(block.hmacro(MACRO::e_fy,x,y,z)), 3, desc, value, dofs);
				case 2: return vtk_helper("force", nse.lat.lbm2physForce(block.hmacro(MACRO::e_fz,x,y,z)), 3, desc, value, dofs);
			}
		}
		//if (index==k++) return vtk_helper("density", block.hmacro(MACRO::e_rho,x,y,z)*nse.physFluidDensity, 1, desc, value, dofs);
		return false;
	}

	virtual void probe1()
	{
		static idx cycle = 0;
		const std::string basename = fmt::format("ball_{:04d}", cycle);
		this->writeVTK_Points(basename.c_str(), nse.physTime(), cycle);

		// output the center alone in a vtk file for easier rendering
		const std::string center_basename = fmt::format("ball_center_{:04d}", cycle);
		typename Lagrange3D<NSE>::HLPVECTOR center_vector({nse.lat.phys2lbmPoint(ball_c)});
		this->writeVTK_Points(center_basename.c_str(), nse.physTime(), cycle, center_vector);

		cycle++;
	}

	virtual void computeBeforeLBMKernel()
	{
		// update ball position
		const dreal velocity_amplitude = 2 * ball_amplitude / ball_period;
		const dreal vz = TNL::sign( cos(2*TNL::pi*nse.iterations/ball_period) ) * velocity_amplitude;
		if (ibm.computeVariant == IbmCompute::CPU) {
			//ibm.hLL_velocity_lat = point_t{0,0,vz};
			ibm.hLL_velocity_lat = point_t{0,vz,0};
			ibm.hLL_lat += ibm.hLL_velocity_lat;	// Delta t = 1
		}
		else {
			//ibm.dLL_velocity_lat = point_t{0,0,vz};
			ibm.dLL_velocity_lat = point_t{0,vz,0};
			ibm.dLL_lat += ibm.dLL_velocity_lat;	// Delta t = 1
		}
		ibm.constructed = false;
		ibm.use_LL_velocity_in_solution = true;

		// update the ball center for drawing
		//ball_c += point_t{0,0,vz*nse.lat.physDl};
		ball_c += point_t{0,vz*nse.lat.physDl,0};
	}

	virtual void updateKernelVelocities()
	{
		for (auto& block : nse.blocks)
		{
			block.data.inflow_vx = lbm_inflow_vx;
			block.data.inflow_vy = 0;
			block.data.inflow_vz = 0;
		}
	}

	virtual void setupBoundaries()
	{
		nse.setBoundaryX(0, BC::GEO_INFLOW_LEFT); // left
		nse.setBoundaryX(nse.lat.global.x()-1, BC::GEO_OUTFLOW_RIGHT);// right
		nse.setBoundaryY(0, BC::GEO_INFLOW); // back
		nse.setBoundaryY(nse.lat.global.y()-1, BC::GEO_INFLOW);// front
		nse.setBoundaryZ(0, BC::GEO_INFLOW);// top
		nse.setBoundaryZ(nse.lat.global.z()-1, BC::GEO_INFLOW);// bottom
	}

	StateLocal(const std::string& id, const TNL::MPI::Comm& communicator, lat_t lat)
		: State<NSE>(id, communicator, lat)
	{}
};

template < typename NSE >
int sim(int RES=2, double i_Re=1000, double nasobek=2.0, int dirac_delta=2, int method=0, int compute=5)
{
	using idx = typename NSE::TRAITS::idx;
	using real = typename NSE::TRAITS::real;
	using point_t = typename NSE::TRAITS::point_t;
	using lat_t = Lattice<3, real, idx>;

	int block_size=32;
	real BALL_DIAMETER = 0.01;
	real real_domain_height= BALL_DIAMETER*11;// [m]
	//real real_domain_length= BALL_DIAMETER*11;// [m] // extra 1cm on both sides
	idx LBM_Y = RES*block_size; // for 4 cm
	idx LBM_Z = LBM_Y;
	real PHYS_DL = real_domain_height/((real)LBM_Y);
	idx LBM_X = 2*LBM_Y;//(int)(real_domain_length/PHYS_DL)+2;//block_size;//16*RESOLUTION;
	point_t PHYS_ORIGIN = {0., 0., 0.};

	// zvolit Re + LBM VELOCITY + PHYS_VISCOSITY
//	real i_Re = ;
	real i_LBM_VELOCITY = 0.07; // Geier
	real i_PHYS_VISCOSITY = 0.00001; // proc ne?
	// mam:
	real i_LBM_VISCOSITY = i_LBM_VELOCITY * BALL_DIAMETER / PHYS_DL / i_Re;
	real i_PHYS_VELOCITY = i_PHYS_VISCOSITY * i_Re / BALL_DIAMETER;
	fmt::print("input phys velocity {:f}\ninput lbm velocity {:f}\nRe {:f}\nlbm viscosity{:f}\nphys viscosity {:f}\n", i_PHYS_VELOCITY, i_LBM_VELOCITY, i_Re, i_LBM_VISCOSITY, i_PHYS_VISCOSITY);

	real LBM_VISCOSITY = i_LBM_VISCOSITY;// 0.0001*RES;//*SIT;//1.0/6.0; /// GIVEN: optimal is 1/6
	real PHYS_VISCOSITY = i_PHYS_VISCOSITY;//0.00001;// [m^2/s] fluid viscosity of water
	real Re=i_Re;//200;

//	real INIT_TIME = 1.0; // [s]
	real PHYS_DT = LBM_VISCOSITY / PHYS_VISCOSITY*PHYS_DL*PHYS_DL;

	// initialize the lattice
	lat_t lat;
	lat.global = typename lat_t::CoordinatesType( LBM_X, LBM_Y, LBM_Z );
	lat.physOrigin = PHYS_ORIGIN;
	lat.physDl = PHYS_DL;
	lat.physDt = PHYS_DT;
	lat.physViscosity = PHYS_VISCOSITY;

	const std::string state_id = fmt::format("sim_IBM4_{}_{}_dirac_{}_res_{}_Re_{}_nas_{:05.4f}_compute_{}", NSE::COLL::id, (method>0)?"original":"modified", dirac_delta, RES, Re, nasobek, compute);
	StateLocal<NSE> state(state_id, MPI_COMM_WORLD, lat);

	if (state.isMark())
		return 0;

	state.lbm_inflow_vx = i_LBM_VELOCITY;
	state.nse.physCharLength = BALL_DIAMETER; // [m]
	state.ball_diameter = BALL_DIAMETER; // [m]
	state.ball_amplitude = lat.global.z() / 5.;	// [lbm units]
	state.ball_period = 2.0 / PHYS_DT;	// [lbm units]

	state.cnt[PRINT].period = 0.1;
	state.nse.physFinalTime = 30.0;

//	state.cnt[VTK3D].period = 1.0;
	state.cnt[VTK2D].period = 0.01;
	state.cnt[PROBE1].period = 0.01;	// Lagrangian points VTK output

	// select compute method
	IbmCompute computeVariant;
	switch (compute)
	{
		case 0: computeVariant = IbmCompute::GPU; break;
		case 1: computeVariant = IbmCompute::CPU; break;
		case 2: computeVariant = IbmCompute::Hybrid; break;
		case 3: computeVariant = IbmCompute::Hybrid_zerocopy; break;
		default:
			spdlog::warn("Unknown parameter compute={}, selecting GPU as the default.", compute);
			computeVariant = IbmCompute::GPU;
			break;
	}

	// add cuts
	state.add2Dcut_X(LBM_X/2,"cut_X");
//	state.add2Dcut_X(2*BALL_DIAMETER/PHYS_DL,"cut_Xball");
	state.add2Dcut_Y(LBM_Y/2,"cut_Y");
	state.add2Dcut_Z(LBM_Z/2,"cut_Z");

	// create immersed objects
	state.ball_c[0] = 2*state.ball_diameter;
	state.ball_c[1] = 5.5*state.ball_diameter;
	state.ball_c[2] = 5.5*state.ball_diameter;
	real sigma = nasobek * PHYS_DL;
	ibmSetupRectangle(state.ibm, state.ball_c, state.ball_diameter/2.0, sigma);

	// configure IBM
	state.ibm.computeVariant = computeVariant;
	state.ibm.diracDeltaTypeEL = dirac_delta;
	if (method == 0)
		state.ibm.methodVariant = IbmMethod::modified;
	else
		state.ibm.methodVariant = IbmMethod::original;

	execute(state);

	return 0;
}

template < typename TRAITS=TraitsSP >
void run(int res, double Re, double h, int dirac, int method, int compute)
{
	using COLL = D3Q27_CUM<TRAITS>;
	using NSE_CONFIG = LBM_CONFIG<
				TRAITS,
				D3Q27_KernelStruct,
				NSE_Data_ConstInflow< TRAITS >,
				COLL,
				typename COLL::EQ,
				D3Q27_STREAMING< TRAITS >,
				D3Q27_BC_All,
				MacroLocal< TRAITS >,
				D3Q27_MACRO_Void< TRAITS >
			>;

	sim<NSE_CONFIG>(res, Re, h, dirac, method, compute);
}

int main(int argc, char **argv)
{
	TNLMPI_INIT mpi(argc, argv);

	// here "h" is the Lagrangian/Eulerian spacing ratio
//	const double hvals[] = { 2.0, 1.5, 1.0, 0.75, 0.50, 0.25 };
//	const double hvals[] = { 2.0 };
//	const double hvals[] = { 0.125 };
	const double hvals[] = { 0.25, 0.5, 0.75, 1.0, 1.5, 2.0 };
//	const double hvals[] = { 2.0, 1.8, 1.6, 1.4, 1.2, 1.0, 0.9, 0.8, 0.75, 0.50 };
//	const double hvals[] = { 2.0, 1.8, 1.6, 1.4, 1.2, 1.0, 0.75, 0.50, 0.25, 0.125 };
//	const double hvals[] = { 3.0, 2.5, 2.0, 1.8, 1.6, 1.4, 1.2, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1 }; // ostrava
	int hmax = sizeof(hvals)/sizeof(double);
	double h=1.0;

	bool use_command_line=true;

	if (!use_command_line)
	{
		int dirac=1;
		int res=2; // 3, 6, 12
		int hi=3;
		int method=0; //0 = modified
		int Re=100;
		int compute=5;
//		for (int Re=100;Re<=200; Re+=100)
//		for (hi=0;hi<hmax;hi++)
//		for (method=0;method<=1;method++)
//		for (res=3;res<=5;res++)
//		for (dirac=1;dirac<=4;dirac++)
//		for (compute=1;compute<=6;compute++)
		{
			if (hi<hmax) h=hvals[hi];
			run(res, (double)Re, h, dirac, method, compute);
		}
	} else
	{
		const int pars=6;
		if (argc <= pars)
		{
			fprintf(stderr, "error: %d parameters required:\n %s method{0,1} dirac{1,2,3,4} Re{100,200} hi[0,%d] res[1,22] compute[1,7]\n", pars, argv[0],hmax-1);
			return 1;
		}
		int method = atoi(argv[1]);	// 0=modified 1=original
		int dirac = atoi(argv[2]);
		int Re = atoi(argv[3]);		// type=0,1,2 (geometry selection)
		int hi = atoi(argv[4]);		// index in the hvals
		int res = atoi(argv[5]);	// res=1,2,3
		int compute = atoi(argv[6]); // compute=0,1,2,3

		if (method > 1 || method < 0) { fprintf(stderr, "error: method=%d out of bounds [0, 1]\n",method); return 1; }
		if (dirac < 1 || dirac > 4) { fprintf(stderr, "error: dirac=%d out of bounds [1,4]\n",dirac); return 1; }
		if (hi >= hmax || hi < 0) { fprintf(stderr, "error: hi=%d out of bounds [0, %d]\n",hi,hmax-1); return 1; }
		if (res < 1) { fprintf(stderr, "error: res=%d out of bounds [1, ...]\n",res); return 1; }
		if (compute < 0 || compute > 3) { fprintf(stderr, "error: compute=%d out of bounds [0,3]\n",compute); return 1; }
		if (hi<hmax) h=hvals[hi];
		run(res, (double)Re, h, dirac, method, compute);
	}
	return 0;
}
