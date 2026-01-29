#define AB_PATTERN

#include <argparse/argparse.hpp>

#include "lbm3d/core.h"
#include "lbm3d/d3q7/eq.h"
#include "lbm3d/d3q7/col_srt.h"
#include "lbm3d/d3q7/col_mrt.h"
#include "lbm3d/d3q7/col_clbm.h"
// exactly one streaming header must be included
#ifdef AA_PATTERN
	#include "lbm3d/d3q7/streaming_AA.h"
#endif
#ifdef AB_PATTERN
	#include "lbm3d/d3q7/streaming_AB.h"
#endif
#include "lbm3d/d3q7/bc.h"
#include "lbm3d/d3q7/macro.h"
#include "lbm3d/state_NSE_ADE.h"

// 3D test domain
template <typename NSE, typename ADE>
struct StateLocal : State_NSE_ADE<NSE, ADE>
{
	using TRAITS = typename NSE::TRAITS;
	using BLOCK_NSE = LBM_BLOCK<NSE>;
	using BLOCK_ADE = LBM_BLOCK<ADE>;

	using State<NSE>::nse;
	using State_NSE_ADE<NSE, ADE>::ade;

	using idx = typename TRAITS::idx;
	using idx3d = typename TRAITS::idx3d;
	using real = typename TRAITS::real;
	using dreal = typename TRAITS::dreal;
	using point_t = typename TRAITS::point_t;
	using lat_t = Lattice<3, real, idx>;

	// problem parameters
	real phi_left = 0;
	real phi_right = 0;
	real diffusion_top = 0;
	real diffusion_bottom = 0;

	StateLocal(
		const std::string& id,
		const TNL::MPI::Comm& communicator,
		lat_t lat_nse,
		lat_t lat_ade,
		real iphysVelocity,
		const std::string& adiosConfigPath = "adios2.xml"
	)
	: State_NSE_ADE<NSE, ADE>(id, communicator, lat_nse, lat_ade, adiosConfigPath)
	{
		//for (auto& block : nse.blocks)
		//{
		//	block.data.inflow_rho = no1;
		//	block.data.inflow_vx = nse.lat.phys2lbmVelocity(iphysVelocity);
		//	block.data.inflow_vy = 0;
		//	block.data.inflow_vz = 0;
		//}

		for (auto& block : ade.blocks) {
			// TODO: phys -> lbm conversion for concentration?
			block.data.inflow_phi = phi_left;
		}
	}

	void setupBoundaries() override
	{
		nse.setBoundaryX(0, NSE::BC::GEO_INFLOW);  // left
		nse.setBoundaryX(nse.lat.global.x() - 1, NSE::BC::GEO_OUTFLOW_RIGHT);

		nse.setBoundaryZ(1, NSE::BC::GEO_WALL);						  // top
		nse.setBoundaryZ(nse.lat.global.z() - 2, NSE::BC::GEO_WALL);  // bottom
		nse.setBoundaryY(1, NSE::BC::GEO_WALL);						  // back
		nse.setBoundaryY(nse.lat.global.y() - 2, NSE::BC::GEO_WALL);  // front

		// extra layer needed due to A-A pattern
		nse.setBoundaryZ(0, NSE::BC::GEO_NOTHING);						 // top
		nse.setBoundaryZ(nse.lat.global.z() - 1, NSE::BC::GEO_NOTHING);	 // bottom
		nse.setBoundaryY(0, NSE::BC::GEO_NOTHING);						 // back
		nse.setBoundaryY(nse.lat.global.y() - 1, NSE::BC::GEO_NOTHING);	 // front

		// ADE boundaries
		//ade.setBoundaryX(0, ADE::BC::GEO_INFLOW);		// left
		ade.setBoundaryX(0, ADE::BC::GEO_WALL);	 // left
		ade.setBoundaryX(ade.lat.global.x() - 1, ADE::BC::GEO_OUTFLOW_RIGHT);

		ade.setBoundaryZ(1, ADE::BC::GEO_WALL);						  // top
		ade.setBoundaryZ(ade.lat.global.z() - 2, ADE::BC::GEO_WALL);  // bottom
		ade.setBoundaryY(1, ADE::BC::GEO_WALL);						  // back
		ade.setBoundaryY(ade.lat.global.y() - 2, ADE::BC::GEO_WALL);  // front

		// extra layer needed due to A-A pattern
		ade.setBoundaryZ(0, ADE::BC::GEO_NOTHING);						 // top
		ade.setBoundaryZ(ade.lat.global.z() - 1, ADE::BC::GEO_NOTHING);	 // bottom
		ade.setBoundaryY(0, ADE::BC::GEO_NOTHING);						 // back
		ade.setBoundaryY(ade.lat.global.y() - 1, ADE::BC::GEO_NOTHING);	 // front

		// setup variable diffusion coefficient and initial phi value in the domain
		ade.allocateDiffusionCoefficientArrays();

		const idx center_x = ade.lat.global.x() / 2;
		const idx center_z = ade.lat.global.z() / 2;

		ade.forAllLatticeSites(
			[&](BLOCK_ADE& block, idx x, idx y, idx z)
			{
				if (x > 0 && (block.hmap(x, y, z) == ADE::BC::GEO_WALL || block.hmap(x, y, z) == ADE::BC::GEO_NOTHING))
					return;
				if (z < center_z)
					block.hdiffusionCoeff(x, y, z) = ade.lat.phys2lbmViscosity(diffusion_bottom);
				else
					block.hdiffusionCoeff(x, y, z) = ade.lat.phys2lbmViscosity(diffusion_top);

#ifdef HAVE_MPI
				auto local_df = block.hfs[0].getLocalView();
				// shift global indices to local
				const auto local_begins = block.hfs[0].getLocalBegins();
				x -= local_begins.template getSize<1>();
				y -= local_begins.template getSize<2>();
				z -= local_begins.template getSize<3>();
#else
				auto local_df = block.hfs[0].getView();
#endif
				// TODO: phys -> lbm conversion for concentration?
				if (x < center_x)
					ADE::COLL::setEquilibriumLat(local_df, x, y, z, phi_left, 0, 0, 0);	 // phi, vx, vy, vz
				else
					ADE::COLL::setEquilibriumLat(local_df, x, y, z, phi_right, 0, 0, 0);  // phi, vx, vy, vz
			}
		);

		// copy the initialized DFs so that they are not overridden
		for (auto& block : ade.blocks)
			for (uint8_t dftype = 1; dftype < DFMAX; dftype++)
				block.hfs[dftype] = block.hfs[0];
		ade.copyDFsToDevice();
	}

	[[nodiscard]] std::vector<std::string> getOutputDataNames() const override
	{
		// return all quantity names used in outputData
		return {"lbm_density", "lbm_density_fluctuation", "velocity_x", "velocity_y", "velocity_z", "lbm_phi", "lbm_diffusion", "phys_diffusion"};
	}

	void outputData(UniformDataWriter<TRAITS>& writer, const BLOCK_NSE& block, const idx3d& begin, const idx3d& end) override
	{
		writer.write("lbm_density", getMacroView<TRAITS>(block.hmacro, NSE::MACRO::e_rho), begin, end);
		writer.write(
			"lbm_density_fluctuation",
			[&](idx x, idx y, idx z) -> dreal
			{
				return block.hmacro(NSE::MACRO::e_rho, x, y, z) - 1.0;
			},
			begin,
			end
		);
		writer.write(
			"velocity_x",
			[&](idx x, idx y, idx z) -> dreal
			{
				return nse.lat.lbm2physVelocity(block.hmacro(NSE::MACRO::e_vx, x, y, z));
			},
			begin,
			end
		);
		writer.write(
			"velocity_y",
			[&](idx x, idx y, idx z) -> dreal
			{
				return nse.lat.lbm2physVelocity(block.hmacro(NSE::MACRO::e_vy, x, y, z));
			},
			begin,
			end
		);
		writer.write(
			"velocity_z",
			[&](idx x, idx y, idx z) -> dreal
			{
				return nse.lat.lbm2physVelocity(block.hmacro(NSE::MACRO::e_vz, x, y, z));
			},
			begin,
			end
		);
	}

	void outputData(UniformDataWriter<TRAITS>& writer, const BLOCK_ADE& block, const idx3d& begin, const idx3d& end) override
	{
		writer.write("lbm_phi", getMacroView<TRAITS>(block.hmacro, ADE::MACRO::e_phi), begin, end);
		writer.write("lbm_diffusion", block.hdiffusionCoeff, begin, end);
		writer.write(
			"phys_diffusion",
			[&](idx x, idx y, idx z) -> dreal
			{
				return nse.lat.lbm2physViscosity(block.hdiffusionCoeff(x, y, z));
			},
			begin,
			end
		);
	}
};

template <typename NSE, typename ADE>
int sim(int RESOLUTION = 2, const std::string& adiosConfigPath = "adios2.xml")
{
	using idx = typename NSE::TRAITS::idx;
	using real = typename NSE::TRAITS::real;
	using point_t = typename NSE::TRAITS::point_t;
	using lat_t = Lattice<3, real, idx>;

	int X = 128 * RESOLUTION;		   // width in pixels
	int Y = 32 * RESOLUTION;		   // height in pixels --- top and bottom walls 1px
	int Z = X;						   // height in pixels --- top and bottom walls 1px
	real LBM_VISCOSITY = 0.001 / 3.0;  //1.0/6.0; /// GIVEN: optimal is 1/6
	real PHYS_HEIGHT = 1.0;			   // [m] domain height (physical)
	real PHYS_VISCOSITY = 1.552e-5;	   // [m^2/s] fluid viscosity of air
	real PHYS_VELOCITY = 1.0;
	real PHYS_DL = PHYS_HEIGHT / ((real) Z - 2);
	real PHYS_DT = LBM_VISCOSITY / PHYS_VISCOSITY * PHYS_DL * PHYS_DL;	//PHYS_HEIGHT/(real)LBM_HEIGHT;
	point_t PHYS_ORIGIN = {0., 0., 0.};

	// initialize the lattice
	lat_t lat_nse;
	lat_nse.global = typename lat_t::CoordinatesType(X, Y, Z);
	lat_nse.physOrigin = PHYS_ORIGIN;
	lat_nse.physDl = PHYS_DL;
	lat_nse.physDt = PHYS_DT;
	lat_nse.physViscosity = PHYS_VISCOSITY;

	// lattice for the ADE is the same as that for NSE, except for viscosity (diffusion)
	lat_t lat_ade = lat_nse;
	lat_ade.physViscosity = 0;	// unused, the diffusion coeff for ADE is set below

	const std::string state_id = fmt::format("sim_T2_res{:02d}_np{:03d}", RESOLUTION, TNL::MPI::GetSize(MPI_COMM_WORLD));
	StateLocal<NSE, ADE> state(state_id, MPI_COMM_WORLD, lat_nse, lat_ade, PHYS_VELOCITY, adiosConfigPath);

	if (! state.canCompute())
		return 0;

	// problem parameters
	state.phi_left = 10;
	state.phi_right = 1;
	state.diffusion_top = 1e-4;		// [m^2/s] diffusion coeff for the ADE
	state.diffusion_bottom = 1e-5;	// [m^2/s] diffusion coeff for the ADE

	state.nse.physFinalTime = 100.0;
	state.cnt[PRINT].period = 0.01;

	// add cuts
	state.cnt[VTK2D].period = 1;
	state.add2Dcut_X(X / 2, "cutsX/cut_X");
	state.add2Dcut_Y(Y / 2, "cutsY/cut_Y");
	state.add2Dcut_Z(Z / 2, "cutsZ/cut_Z");

	//state.cnt[VTK3D].period = 0.001;
	//state.cnt[VTK3DCUT].period = 0.001;
	//state.add3Dcut(X/4,Y/4,Z/4, X/2,Y/2,Z/2, "box");

	execute(state);

	return 0;
}

//template <typename TRAITS=TraitsSP>
template <typename TRAITS = TraitsDP>
void run(int RES, const std::string& adiosConfigPath)
{
	using NSE_COLL = D3Q27_CUM<TRAITS, D3Q27_EQ_INV_CUM<TRAITS>>;
	using NSE_CONFIG = LBM_CONFIG<
		TRAITS,
		D3Q27_KernelStruct,
		NSE_Data_ConstInflow<TRAITS>,
		NSE_COLL,
		typename NSE_COLL::EQ,
		D3Q27_STREAMING<TRAITS>,
		D3Q27_BC_All,
		D3Q27_MACRO_Default<TRAITS>>;

	//using ADE_COLL = D3Q7_SRT<TRAITS>;
	//using ADE_COLL = D3Q7_MRT<TRAITS>;
	using ADE_COLL = D3Q7_CLBM<TRAITS>;
	using ADE_CONFIG = LBM_CONFIG<
		TRAITS,
		D3Q7_KernelStruct,
		ADE_Data_ConstInflow<TRAITS>,
		ADE_COLL,
		typename ADE_COLL::EQ,
		D3Q7_STREAMING<TRAITS>,
		D3Q7_BC_All,
		D3Q7_MACRO_Default<TRAITS>>;

	sim<NSE_CONFIG, ADE_CONFIG>(RES, adiosConfigPath);
}

int main(int argc, char** argv)
{
	TNLMPI_INIT mpi(argc, argv);

	argparse::ArgumentParser program("sim_T2");
	program.add_description("Simple coupled D3Q27-D3Q7 simulation example.");
	program.add_argument("resolution").help("resolution of the lattice").scan<'i', int>().default_value(1);
	program.add_argument("--adios-config").help("path to ADIOS2 XML configuration file").default_value(std::string("adios2.xml"));

	try {
		program.parse_args(argc, argv);
	}
	catch (const std::exception& err) {
		std::cerr << err.what() << std::endl;
		std::cerr << program;
		return 1;
	}

	const auto resolution = program.get<int>("resolution");
	if (resolution < 1)
		throw std::invalid_argument("CLI error: resolution must be at least 1");

	const auto adiosConfigPath = program.get<std::string>("--adios-config");

	run(resolution, adiosConfigPath);

	return 0;
}
