#include <argparse/argparse.hpp>
#include <utility>

#include "lbm3d/core.h"
#include "lbm3d/inflow_openings_lbm.h"

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

	// authored target flux through the opening (unit cell area), resolved by
	// sim() before execute() starts reset(); setupBoundaries() only forwards
	dreal opening_flux = 0;
	// opening cross-section bounds on y/z (fluid interior of the channel)
	idx open_lo = 2;
	idx open_hi_y = 0;
	idx open_hi_z = 0;

	InflowOpeningsState<TRAITS> openings;

	StateLocal(const std::string& id, const TNL::MPI::Comm& communicator, lat_t lat, const std::string& adiosConfigPath = "adios2.xml")
	: State<NSE>(id, communicator, std::move(lat), adiosConfigPath)
	{}

	void setupBoundaries() override
	{
		const idx X = nse.lat.global.x();
		const idx Y = nse.lat.global.y();
		const idx Z = nse.lat.global.z();
		open_hi_y = Y - 3;
		open_hi_z = Z - 3;

		// base imposed velocity (unit base that PARABOLIC shapes through the
		// precomputed site velocities)
		for (auto& block : nse.blocks)
			block.data.inflow_vx = 1;

		// authored inflow on the interior plane x=1; plane 0 stays the
		// GEO_NOTHING ghost layer (A-A pattern idiom, same as sim_1)
		addInflowPlane(nse, openings, 0, -1, 1, idx3d{1, open_lo, open_lo}, idx3d{1, open_hi_y, open_hi_z}, ProfileType::PARABOLIC, opening_flux);

		nse.setBoundaryX(X - 2, BC::GEO_OUTFLOW_RIGHT_INTERP);	// right

		nse.setBoundaryZ(1, BC::GEO_WALL);		// bottom
		nse.setBoundaryZ(Z - 2, BC::GEO_WALL);	// top
		nse.setBoundaryY(1, BC::GEO_WALL);		// front
		nse.setBoundaryY(Y - 2, BC::GEO_WALL);	// back

		// extra layer needed due to A-A pattern
		nse.setBoundaryX(0, BC::GEO_NOTHING);	   // left
		nse.setBoundaryX(X - 1, BC::GEO_NOTHING);  // right
		nse.setBoundaryZ(0, BC::GEO_NOTHING);	   // bottom
		nse.setBoundaryZ(Z - 1, BC::GEO_NOTHING);  // top
		nse.setBoundaryY(0, BC::GEO_NOTHING);	   // front
		nse.setBoundaryY(Y - 1, BC::GEO_NOTHING);  // back

		finalizeInflowOpenings(nse, openings);
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
				return nse.lat.lbm2physVelocity(block.hmacro(MACRO::e_vz, x, y, z));
			},
			begin,
			end
		);
	}

	void AfterSimFinished() override
	{
		// final-time check: the registered PROBE1 counter keeps the device
		// macro current through the last probe cycle; refresh the host copy
		// and integrate the axial velocity over the opening cells (the moment
		// BC imposes exactly the authored paraboloid there, so the flux
		// approaches the amplitude)
		nse.copyMacroToHost();

		real local_flux = 0;
		for (auto& block : nse.blocks) {
			if (1 < block.offset.x() || 1 >= block.offset.x() + block.local.x())
				continue;
			const idx begin_y = TNL::max(open_lo, block.offset.y());
			const idx end_y = TNL::min(open_hi_y, block.offset.y() + block.local.y() - 1);
			const idx begin_z = TNL::max(open_lo, block.offset.z());
			const idx end_z = TNL::min(open_hi_z, block.offset.z() + block.local.z() - 1);
			for (idx k = begin_z; k <= end_z; k++)
				for (idx j = begin_y; j <= end_y; j++)
					local_flux += block.hmacro(MACRO::e_vx, 1, j, k);
		}
		const real flux = TNL::MPI::reduce(local_flux, MPI_SUM, MPI_COMM_WORLD);
		const real rel_error = opening_flux != 0 ? TNL::abs(flux - opening_flux) / TNL::abs(opening_flux) : 0;

		if (nse.rank == 0) {
			spdlog::info("opening_flux = {:.8e}", flux);
			spdlog::info("flux_rel_error = {:.8e}", rel_error);
		}

		State<NSE>::AfterSimFinished();
	}
};

template <typename NSE>
void sim(const std::string& adios_config, int RESOLUTION, double final_time)
{
	using idx = typename NSE::TRAITS::idx;
	using real = typename NSE::TRAITS::real;
	using point_t = typename NSE::TRAITS::point_t;
	using lat_t = Lattice<3, real, idx>;

	int block_size = 32;
	int X = 128 * RESOLUTION;		  // width in pixels
	int Y = block_size * RESOLUTION;  // height in pixels --- top and bottom walls 1px
	int Z = Y;						  // height in pixels --- top and bottom walls 1px
	real LBM_VISCOSITY = 1e-4;
	real PHYS_HEIGHT = 0.41;	   // [m] domain height (physical)
	real PHYS_VISCOSITY = 1.5e-5;  // [m^2/s] fluid viscosity .... blood?
	real PHYS_DL = PHYS_HEIGHT / ((real) Y - 2);
	real PHYS_DT = LBM_VISCOSITY / PHYS_VISCOSITY * PHYS_DL * PHYS_DL;
	point_t PHYS_ORIGIN = {0., 0., 0.};

	// PHYS_VELOCITY is the target centerline (peak) velocity of the paraboloid
	real PHYS_VELOCITY = 1.0;

	// Initialize the lattice
	lat_t lat;
	lat.global = typename lat_t::CoordinatesType(X, Y, Z);
	lat.physOrigin = PHYS_ORIGIN;
	lat.physDl = PHYS_DL;
	lat.physDt = PHYS_DT;
	lat.physViscosity = PHYS_VISCOSITY;

	const std::string state_id = fmt::format("sim_openings_res{:02d}_np{:03d}", RESOLUTION, TNL::MPI::GetSize(MPI_COMM_WORLD));
	StateLocal<NSE> state(state_id, MPI_COMM_WORLD, lat, adios_config);

	if (! state.canCompute())
		return;

	// Problem parameters
	const real u_peak_lbm = lat.phys2lbmVelocity(PHYS_VELOCITY);
	const real Re = PHYS_VELOCITY * PHYS_HEIGHT / PHYS_VISCOSITY;
	const real Ma = u_peak_lbm * std::sqrt(3.0);

	// authored amplitude = target flux through the opening: the peak velocity
	// times the sum of the cell-centered paraboloid weights over the opening
	// cross-section (same z-then-y accumulation order as the scale resolver)
	const idx open_lo = 2;
	const idx open_hi_y = Y - 3;
	const idx open_hi_z = Z - 3;
	double weight_sum = 0;
	for (idx k = open_lo; k <= open_hi_z; k++)
		for (idx j = open_lo; j <= open_hi_y; j++)
			weight_sum += openingProfileWeight<idx, double>(ProfileType::PARABOLIC, open_lo, open_hi_y, open_lo, open_hi_z, j, k);
	state.opening_flux = u_peak_lbm * weight_sum;

	spdlog::info("PHYS_VELOCITY (centerline) = {:e} m/s, authored opening flux = {:e}", PHYS_VELOCITY, state.opening_flux);
	spdlog::info("Re = {:e} (based on centerline velocity and channel height)", Re);
	spdlog::info("Ma = {:e} (based on lattice centerline velocity, c_s = 1/sqrt(3))", Ma);

	// Set up simulation parameters
	state.nse.physFinalTime = final_time;
	state.cnt[PRINT].period = 0.1;
	// no data outputs -- the PROBE1 cadence keeps the device macro fresh for
	// the final-time flux check via the regular counter-driven sync
	state.cnt[PROBE1].period = 0.1;

	execute(state);
}

template <typename TRAITS = TraitsSP>
void run(const std::string& adios_config, int resolution, double final_time)
{
	using COLL = D3Q27_CUM<TRAITS, D3Q27_EQ_INV_CUM<TRAITS>>;

	using NSE_CONFIG = LBM_CONFIG<
		TRAITS,
		D3Q27_KernelStruct,
		NSE_Data_OpeningInflow<TRAITS>,
		COLL,
		typename COLL::EQ,
		D3Q27_STREAMING<TRAITS>,
		D3Q27_BC_All,
		D3Q27_MACRO_Default<TRAITS>>;

	sim<NSE_CONFIG>(adios_config, resolution, final_time);
}

int main(int argc, char** argv)
{
	TNLMPI_INIT mpi(argc, argv);

	argparse::ArgumentParser program("sim_openings");
	program.add_description("Square channel with an authored parabolic inflow opening on the interior plane x=1.");
	program.add_argument("--adios-config").help("path to ADIOS2 configuration file").default_value(std::string("adios2.xml")).nargs(1);
	program.add_argument("--resolution").help("resolution of the lattice").scan<'i', int>().default_value(1).nargs(1);
	program.add_argument("--final-time").help("final time of the simulation").scan<'g', double>().default_value(1.0).nargs(1);

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

	const auto final_time = program.get<double>("--final-time");
	if (final_time <= 0) {
		fmt::println(stderr, "CLI error: final-time must be positive");
		return 1;
	}

	run(adios_config, resolution, final_time);

	return 0;
}
