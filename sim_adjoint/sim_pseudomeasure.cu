#define AB_PATTERN

#include <argparse/argparse.hpp>
#include <magic_enum/magic_enum.hpp>

#include "lbm3d/core.h"
#include "saveload.h"
#include "velocity_profiles.h"

#define DIRECTORY "results_adjoint"

// FIXME: loadPrimaryAndMeasuredMacro is not implemented for non-steady adjoint simulations
static constexpr bool STEADY = true;
static constexpr int BLOCK_SIZE = 32;
static constexpr int MACRO_ALL_ITERS_MAX_GB = 5;

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

	int resolution = 0;
	bool steady = false;

	StateLocal(const std::string& id, const TNL::MPI::Comm& communicator, lat_t lat, const std::string& adiosConfigPath = "adios2.xml")
	: State<NSE>(id, communicator, std::move(lat), adiosConfigPath)
	{}

	void setupBoundaries() override
	{
		nse.setBoundaryX(0, BC::GEO_INFLOW_BB_LEFT);  // left

		nse.setBoundaryX(nse.lat.global.x() - 1, BC::GEO_WALL);	 // right
		nse.setBoundaryZ(0, BC::GEO_WALL);						 // top
		nse.setBoundaryZ(nse.lat.global.z() - 1, BC::GEO_WALL);	 // bottom
		nse.setBoundaryY(0, BC::GEO_WALL);						 // back
		nse.setBoundaryY(nse.lat.global.y() - 1, BC::GEO_WALL);	 // front
	}

	[[nodiscard]] std::vector<std::string> getOutputDataNames() const override
	{
		// return all quantity names used in outputData
		return {"lbm_density", "lbm_density_fluctuation", "lbm_velocity_x", "lbm_velocity_y", "lbm_velocity_z"};
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
		writer.write("lbm_velocity_x", getMacroView<TRAITS>(block.hmacro, MACRO::e_vx), begin, end);
		writer.write("lbm_velocity_y", getMacroView<TRAITS>(block.hmacro, MACRO::e_vy), begin, end);
		writer.write("lbm_velocity_z", getMacroView<TRAITS>(block.hmacro, MACRO::e_vz), begin, end);
	}

	void computeAfterLBMKernel() override
	{
		if (! steady) {
			nse.copyMacroToHost();	//! important - macro is stored on device
			const std::string fname = fmt::format("{}/adjoint_data_res{:02d}/macro_measured.bp", DIRECTORY, resolution);
			saveloadMacro(*this, adios2::Mode::Write, fname, steady);
		}
	}

	bool estimateMemoryDemands() override
	{
		if (! steady) {
			// calculate storage for macroscopic quantities (same as in the original estimateMemoryDemands function)
			long long memMacro = 0;
			for (const auto& block : nse.blocks) {
				const long long XYZ = block.local.x() * block.local.y() * block.local.z();
				memMacro += XYZ * sizeof(dreal) * NSE::MACRO::N;
			}

			// check storage for macroscopic quantities in all iterations
			long long memMacroAllIter = memMacro * (nse.physFinalTime / nse.lat.physDt);
			if (memMacroAllIter > MACRO_ALL_ITERS_MAX_GB * 1e9) {
				spdlog::error("Memory estimation for saving all macro is over {} GB", MACRO_ALL_ITERS_MAX_GB);
				return false;
			}
		}

		return State<NSE>::estimateMemoryDemands();
	}
};

template <typename NSE>
int sim(int resolution, double vy_amplitude, VelocityProfile vy_profile)
{
	using idx = typename NSE::TRAITS::idx;
	using real = typename NSE::TRAITS::real;
	using dreal = typename NSE::TRAITS::dreal;
	using point_t = typename NSE::TRAITS::point_t;
	using lat_t = Lattice<3, real, idx>;

	int X = BLOCK_SIZE * resolution;			   // width in pixels
	int Y = X;									   // height in pixels --- top and bottom walls 1px
	int Z = Y;									   // height in pixels --- top and bottom walls 1px
	real LBM_VISCOSITY = 0.0002 * (real) (Y - 1);  //1.0/6.0; /// GIVEN: optimal is 1/6
	real PHYS_HEIGHT = 0.10;					   // [m] domain height (physical)
	real PHYS_VISCOSITY = 1.5e-5;				   // [m^2/s] fluid viscosity .... blood?
	real PHYS_DL = PHYS_HEIGHT / ((real) Y - 1);
	real PHYS_DT = LBM_VISCOSITY / PHYS_VISCOSITY * PHYS_DL * PHYS_DL;	//PHYS_HEIGHT/(real)LBM_HEIGHT;
	point_t PHYS_ORIGIN = {0., 0., 0.};

	// initialize the lattice
	lat_t lat;
	lat.global = typename lat_t::CoordinatesType(X, Y, Z);
	lat.physOrigin = PHYS_ORIGIN;
	lat.physDl = PHYS_DL;
	lat.physDt = PHYS_DT;
	lat.physViscosity = PHYS_VISCOSITY;

	const std::string state_id = fmt::format("sim_pseudomeasure_res{:02d}_np{:03d}", resolution, TNL::MPI::GetSize(MPI_COMM_WORLD));
	StateLocal<NSE> state(state_id, MPI_COMM_WORLD, lat);

	// problem parameters
	state.resolution = resolution;
	state.steady = STEADY;

	state.nse.physFinalTime = 16.0;

	// add cuts
	state.cnt[OUT2D].period = state.nse.physFinalTime / 100.0;
	state.add2Dcut_X(0, "cutsX/cut_X");
	state.add2Dcut_X(X / 2, "cutsX2/cut_X");
	state.add2Dcut_Y(Y / 2, "cutsY/cut_Y");
	state.add2Dcut_Z(Z / 2, "cutsZ/cut_Z");

	state.cnt[OUT3D].period = state.nse.physFinalTime / 4.0;

	// set inflow velocity profile
	std::unique_ptr<double[]> velocityProfileX = initGuess(VelocityProfile::zero, Y, Z);
	std::unique_ptr<double[]> velocityProfileY = initGuess(vy_profile, Y, Z, vy_amplitude);
	std::unique_ptr<double[]> velocityProfileZ = initGuess(VelocityProfile::zero, Y, Z);

	for (auto& block : state.nse.blocks) {
		allocateCopyVelocityProfile<dreal, idx>(
			&(block.data.inflow_vx),
			block.local.y(),
			block.local.z(),
			block.offset.y(),
			block.offset.z(),
			velocityProfileX.get(),
			block.global.y(),
			block.global.z()
		);
		allocateCopyVelocityProfile<dreal, idx>(
			&(block.data.inflow_vy),
			block.local.y(),
			block.local.z(),
			block.offset.y(),
			block.offset.z(),
			velocityProfileY.get(),
			block.global.y(),
			block.global.z()
		);
		allocateCopyVelocityProfile<dreal, idx>(
			&(block.data.inflow_vz),
			block.local.y(),
			block.local.z(),
			block.offset.y(),
			block.offset.z(),
			velocityProfileZ.get(),
			block.global.y(),
			block.global.z()
		);
	}

	execute(state);

	if (state.steady) {
		state.nse.copyMacroToHost();  //! important - macro is stored on device
		const std::string fname = fmt::format("{}/adjoint_data_res{:02d}/macro_measured.bp", DIRECTORY, resolution);
		saveloadMacro(state, adios2::Mode::Write, fname, state.steady);
	}

	for (auto& block : state.nse.blocks) {
		deallocateVelocityProfile<dreal>(&(block.data.inflow_vx));
		deallocateVelocityProfile<dreal>(&(block.data.inflow_vy));
		deallocateVelocityProfile<dreal>(&(block.data.inflow_vz));
	}

	std::string dirname = fmt::format("{}/adjoint_data_res{:02d}/goal", DIRECTORY, resolution);
	saveVelocityProfile(dirname, velocityProfileX.get(), Y, Z, 'X');
	saveVelocityProfile_txt(dirname, velocityProfileX.get(), Y, Z, 'X');
	saveVelocityProfile(dirname, velocityProfileY.get(), Y, Z, 'Y');
	saveVelocityProfile_txt(dirname, velocityProfileY.get(), Y, Z, 'Y');
	saveVelocityProfile(dirname, velocityProfileZ.get(), Y, Z, 'Z');
	saveVelocityProfile_txt(dirname, velocityProfileZ.get(), Y, Z, 'Z');

	return 0;
}

template <typename TRAITS = TraitsDP>
void run(int resolution, double vy_amplitude, VelocityProfile vy_profile)
{
	//	using COLL = D3Q27_CUM< TRAITS >;
	using COLL = D3Q27_SRT<TRAITS, D3Q27_EQ<TRAITS>>;

	using NSE_CONFIG = LBM_CONFIG<
		TRAITS,
		D3Q27_KernelStruct,
		NSE_Data_InflowProfile<TRAITS>,
		COLL,
		typename COLL::EQ,
		D3Q27_STREAMING<TRAITS>,
		D3Q27_BC_All,
		D3Q27_MACRO_Default<TRAITS>>;

	sim<NSE_CONFIG>(resolution, vy_amplitude, vy_profile);
}

int main(int argc, char** argv)
{
	TNLMPI_INIT mpi(argc, argv);

	argparse::ArgumentParser program("sim_pseudomeasure");
	program.add_description("First part of the adjoint-LBM workflow - creates the 'measured' data.");
	program.add_argument("--resolution").help("resolution of the lattice").scan<'i', int>().default_value(1);
	program.add_argument("--vy-amplitude").help("parameter for the initial guess (velocity profile)").scan<'g', double>().default_value(0.1);
	program.add_argument("--vy-profile").help("type of the velocity profile").choices("sinus", "block", "flat").default_value("sinus");

	try {
		program.parse_args(argc, argv);
	}
	catch (const std::exception& err) {
		std::cerr << err.what() << '\n';
		std::cerr << program;
		return 1;
	}

	const auto resolution = program.get<int>("--resolution");
	if (resolution < 1) {
		fmt::println(stderr, "CLI error: resolution must be at least 1");
		return 1;
	}

	const auto vy_amplitude = program.get<double>("--vy-amplitude");
	if (vy_amplitude < 0) {
		fmt::println(stderr, "CLI error: vy-amplitude must be non-negative");
		return 1;
	}

	const auto vy_profile_string = program.get<std::string>("--vy-profile");
	const auto vy_profile = magic_enum::enum_cast<VelocityProfile>(vy_profile_string);
	if (! vy_profile.has_value()) {
		fmt::println(stderr, "CLI error: unknown vy-profile option '{}'", vy_profile_string);
		return 1;
	}

	run(resolution, vy_amplitude, vy_profile.value());

	return 0;
}
