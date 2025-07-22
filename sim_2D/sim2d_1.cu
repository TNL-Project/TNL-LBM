#include <argparse/argparse.hpp>
#include <utility>

#include "lbm3d/core.h"
#include "lbm3d/lbm_data.h"

#include "lbm3d/d2q9/bc.h"
#include "lbm3d/d2q9/col_srt.h"
#include "lbm3d/d2q9/col_clbm.h"
#include "lbm3d/d2q9/macro.h"

// exactly one streaming header must be included
#ifdef AA_PATTERN
	#include "lbm3d/d2q9/streaming_AA.h"
#endif
#ifdef AB_PATTERN
	#include "lbm3d/d2q9/streaming_AB.h"
#endif

template <typename TRAITS>
struct NSE2D_Data_ConstInflow : NSE_Data<TRAITS>
{
	using idx = typename TRAITS::idx;
	using dreal = typename TRAITS::dreal;

	dreal inflow_vx = 0;
	dreal inflow_vy = 0;

	template <typename LBM_KS>
	CUDA_HOSTDEV void inflow(LBM_KS& KS, idx x, idx y, idx z)
	{
		KS.vx = inflow_vx;
		KS.vy = inflow_vy;
	}
};

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
		nse.setBoundaryX(0, BC::GEO_INFLOW);							  // left
		nse.setBoundaryX(nse.lat.global.x() - 1, BC::GEO_OUTFLOW_RIGHT);  // right

		nse.setBoundaryY(1, BC::GEO_WALL);						 // back
		nse.setBoundaryY(nse.lat.global.y() - 2, BC::GEO_WALL);	 // front

		// extra layer needed due to A-A pattern
		nse.setBoundaryY(0, BC::GEO_NOTHING);						// back
		nse.setBoundaryY(nse.lat.global.y() - 1, BC::GEO_NOTHING);	// front

		// draw a wall with a hole
		int cx = floor(0.20 / nse.lat.physDl);
		int width = nse.lat.global.y() / 10;
		for (int px = cx; px <= cx + width; px++)
			for (int py = 1; py <= nse.lat.global.y() - 2; py++)
				if (! (py >= nse.lat.global.y() * 4 / 10 && py <= nse.lat.global.y() * 6 / 10)) {
					nse.setMap(px, py, 0, BC::GEO_WALL);
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
					return vtk_helper("velocity", nse.lat.lbm2physVelocity(block.hmacro(MACRO::e_vx, x, y, z)), 3, desc, value, dofs);
				case 1:
					return vtk_helper("velocity", nse.lat.lbm2physVelocity(block.hmacro(MACRO::e_vy, x, y, z)), 3, desc, value, dofs);
				case 2:
					// VTK does not know 2D vectors
					return vtk_helper("velocity", 0, 3, desc, value, dofs);
			}
		}
		return false;
	}

	void updateKernelVelocities() override
	{
		for (auto& block : nse.blocks) {
			block.data.inflow_vx = lbm_inflow_vx;
			block.data.inflow_vy = 0;
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
	real LBM_VISCOSITY = 1e-5;		  //1.0/6.0; /// GIVEN: optimal is 1/6
	real PHYS_HEIGHT = 0.41;		  // [m] domain height (physical)
	real PHYS_VISCOSITY = 1.5e-5;	  // [m^2/s] fluid viscosity .... blood?
	//real PHYS_VELOCITY = 2.25; // m/s ... will be multip
	real PHYS_VELOCITY = 1.0;
	real PHYS_DL = PHYS_HEIGHT / ((real) Y - 2);
	real PHYS_DT = LBM_VISCOSITY / PHYS_VISCOSITY * PHYS_DL * PHYS_DL;	//PHYS_HEIGHT/(real)LBM_HEIGHT;
	point_t PHYS_ORIGIN = {0., 0., 0.};

	// initialize the lattice
	lat_t lat;
	lat.global = typename lat_t::CoordinatesType(X, Y, 1);
	lat.physOrigin = PHYS_ORIGIN;
	lat.physDl = PHYS_DL;
	lat.physDt = PHYS_DT;
	lat.physViscosity = PHYS_VISCOSITY;

	const std::string state_id = fmt::format("sim2d_1_res{:02d}_np{:03d}", RESOLUTION, TNL::MPI::GetSize(MPI_COMM_WORLD));
	StateLocal<NSE> state(state_id, MPI_COMM_WORLD, lat);

	if (! state.canCompute())
		return 0;

	// problem parameters
	state.lbm_inflow_vx = lat.phys2lbmVelocity(PHYS_VELOCITY);

	state.nse.physFinalTime = 1.0;
	state.cnt[PRINT].period = 0.001;

	// 2D = cut in 3D at z=0
	state.cnt[VTK2D].period = 0.001;
	state.add2Dcut_Z(0, "");

	execute(state);

	return 0;
}

template <typename TRAITS = TraitsSP>
void run(int RES)
{
	//using COLL = D2Q9_SRT<TRAITS>;
	using COLL = D2Q9_CLBM<TRAITS>;

	using NSE_CONFIG = LBM_CONFIG<
		TRAITS,
		D2Q9_KernelStruct,
		NSE2D_Data_ConstInflow<TRAITS>,
		COLL,
		typename COLL::EQ,
		D2Q9_STREAMING<TRAITS>,
		D2Q9_BC_All,
		D2Q9_MACRO_Default<TRAITS>>;

	sim<NSE_CONFIG>(RES);
}

int main(int argc, char** argv)
{
	TNLMPI_INIT mpi(argc, argv);

	argparse::ArgumentParser program("sim2d_1");
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
