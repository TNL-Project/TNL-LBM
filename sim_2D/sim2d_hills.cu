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

	using idx = typename TRAITS::idx;
	using idx3d = typename TRAITS::idx3d;
	using real = typename TRAITS::real;
	using dreal = typename TRAITS::dreal;
	using point_t = typename TRAITS::point_t;
	using lat_t = Lattice<3, real, idx>;

	real lbm_inflow_vx = 0;

	StateLocal(const std::string& id, const TNL::MPI::Comm& communicator, lat_t lat, const std::string& adios_config = "adios2.xml")
	: State<NSE>(id, communicator, std::move(lat), adios_config)
	{}

	void setupBoundaries() override
	{
		nse.setBoundaryX(1, BC::GEO_INFLOW_LEFT);								 // left: inflow
		nse.setBoundaryX(nse.lat.global.x() - 2, BC::GEO_OUTFLOW_RIGHT_INTERP);	 // right: outflow

		nse.setBoundaryY(1, BC::GEO_WALL);							// bottom: wall
		nse.setBoundaryY(nse.lat.global.y() - 2, BC::GEO_SYM_TOP);	// top: symmetry

		// extra layer needed due to A-A pattern
		nse.setBoundaryX(0, BC::GEO_NOTHING);						// left
		nse.setBoundaryX(nse.lat.global.x() - 1, BC::GEO_NOTHING);	// right
		nse.setBoundaryY(0, BC::GEO_NOTHING);						// bottom
		nse.setBoundaryY(nse.lat.global.y() - 1, BC::GEO_NOTHING);	// top

		// 3 identical hill-like bumps on the bottom wall, in the left half of the domain
		// Each bump is a half-sine shape: h(x) = bump_height * sin(pi * (x - x0) / bump_width)
		const real phys_dl = nse.lat.physDl;
		const idx Y = nse.lat.global.y();
		const int bump_height = std::max(2, (int) (Y / 8));
		const int bump_width = std::max(4, (int) (Y / 4));
		const int domain_half = nse.lat.global.x() / 2;
		const int gap = (domain_half - 3 * bump_width) / 4;

		int x0 = gap;
		for (int b = 0; b < 3; b++) {
			for (int px = x0; px < x0 + bump_width; px++) {
				int h = (int) std::round(bump_height * std::sin(M_PI * (px - x0) / bump_width));
				for (int py = 1; py <= 1 + h; py++)
					nse.setMap(px, py, 0, BC::GEO_WALL);
			}
			x0 += bump_width + gap;
		}
	}

	[[nodiscard]] std::vector<std::string> getOutputDataNames() const override
	{
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
				return 0;
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
		}
	}
};

template <typename NSE>
void sim(const std::string& adios_config, int RESOLUTION, typename NSE::TRAITS::real Re)
{
	using idx = typename NSE::TRAITS::idx;
	using real = typename NSE::TRAITS::real;
	using point_t = typename NSE::TRAITS::point_t;
	using lat_t = Lattice<3, real, idx>;

	int block_size = 32;
	int X = 4 * block_size * RESOLUTION;
	int Y = block_size * RESOLUTION;
	real LBM_VISCOSITY = 5e-4;
	real PHYS_HEIGHT = 0.41;
	real PHYS_VISCOSITY = 1.5e-5;
	real PHYS_DL = PHYS_HEIGHT / ((real) Y - 2);
	real PHYS_DT = LBM_VISCOSITY / PHYS_VISCOSITY * PHYS_DL * PHYS_DL;
	point_t PHYS_ORIGIN = {0., 0., 0.};

	lat_t lat;
	lat.global = typename lat_t::CoordinatesType(X, Y, 1);
	lat.physOrigin = PHYS_ORIGIN;
	lat.physDl = PHYS_DL;
	lat.physDt = PHYS_DT;
	lat.physViscosity = PHYS_VISCOSITY;

	const std::string state_id = fmt::format("sim2d_hills_res{:02d}_np{:03d}", RESOLUTION, TNL::MPI::GetSize(MPI_COMM_WORLD));
	StateLocal<NSE> state(state_id, MPI_COMM_WORLD, lat, adios_config);

	if (! state.canCompute())
		return;

	real PHYS_VELOCITY = Re * PHYS_VISCOSITY / PHYS_HEIGHT;
	state.lbm_inflow_vx = state.nse.lat.phys2lbmVelocity(PHYS_VELOCITY);
	state.nse.physFinalTime = 100.0;
	state.cnt[PRINT].period = 0.1;
	state.cnt[OUT2D].period = 0.5;
	state.add2Dcut_Z(0, "");

	if (state.nse.rank == 0)
		spdlog::info("Re={:.0f} Ma={:.4f} PHYS_VISCOSITY={:e} PHYS_VELOCITY={:e}",
			PHYS_VELOCITY * PHYS_HEIGHT / PHYS_VISCOSITY,
			state.lbm_inflow_vx * sqrt(3.0),
			PHYS_VISCOSITY,
			PHYS_VELOCITY);

	execute(state);
}

template <typename TRAITS = Traits<float, double, int>>
void run(const std::string& adios_config, int RES, typename TRAITS::real Re)
{
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

	sim<NSE_CONFIG>(adios_config, RES, Re);
}

int main(int argc, char** argv)
{
	TNLMPI_INIT mpi(argc, argv);

	argparse::ArgumentParser program("sim2d_hills");
	program.add_description("Channel flow over 3 hill-like bumps on the bottom wall, with inflow (left), symmetry (top), outflow (right).");
	program.add_argument("--adios-config").help("path to ADIOS2 configuration file").default_value(std::string("adios2.xml")).nargs(1);
	program.add_argument("--resolution").help("resolution of the lattice").scan<'i', int>().default_value(1).nargs(1);
	program.add_argument("--Re").help("target Reynolds number").scan<'g', double>().default_value(1000.0).nargs(1);

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

	const auto Re = program.get<double>("--Re");
	if (Re <= 0) {
		fmt::println(stderr, "CLI error: Re must be positive");
		return 1;
	}

	run(adios_config, resolution, Re);

	return 0;
}
