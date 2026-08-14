#include <argparse/argparse.hpp>
#include <cmath>
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
struct NSE2D_Data_Periodic : NSE_Data<TRAITS>
{
	using idx = typename TRAITS::idx;
	using dreal = typename TRAITS::dreal;

	// minimal inflow method (required by BC interface, not used for periodic BCs)
	template <typename LBM_KS>
	CUDA_HOSTDEV void inflow(LBM_KS& KS, idx x, idx y, idx z)
	{
		KS.vx = 0;
		KS.vy = 0;
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
	using bool3d = typename TRAITS::bool3d;
	using lat_t = Lattice<3, real, idx>;

	// Taylor-Green vortex parameters
	dreal rho_0 = 1;
	dreal lbm_V_0 = 0;	// velocity amplitude in lattice units (set from sim())

	StateLocal(const std::string& id, const TNL::MPI::Comm& communicator, lat_t lat, const std::string& adios_config = "adios2.xml")
	: State<NSE>(
		  id,
		  communicator,
		  std::move(lat),
		  adios_config,
		  // fully periodic domain, so no setupBoundaries() override
		  bool3d{true, true, false}
	  )
	{}

	// Analytical Taylor-Green decay factor F(t) = exp(-2 * nu * k^2 * t)
	// In lattice units: k = 2*pi/X, nu = lbmViscosity, t = iterations
	// (See https://en.wikipedia.org/wiki/Taylor-Green_vortex)
	real decayFactor() const
	{
		const real X = nse.lat.global.x();
		const real nu = nse.lat.lbmViscosity();
		const real k = 2 * TNL::pi / X;
		const real t = nse.iterations;
		return std::exp(-2.0 * nu * k * k * t);
	}

	// Analytical velocity in lattice units at global lattice position (x, y)
	real analytical_vx(idx x, idx y) const
	{
		const real X = nse.lat.global.x();
		const real Y = nse.lat.global.y();
		const real F = decayFactor();
		const real px = nse.lat.lbm2physX(x) / (X * nse.lat.physDl);
		const real py = nse.lat.lbm2physY(y) / (Y * nse.lat.physDl);
		return lbm_V_0 * TNL::sin(2 * TNL::pi * px) * TNL::cos(2 * TNL::pi * py) * F;
	}

	real analytical_vy(idx x, idx y) const
	{
		const real X = nse.lat.global.x();
		const real Y = nse.lat.global.y();
		const real F = decayFactor();
		const real px = nse.lat.lbm2physX(x) / (X * nse.lat.physDl);
		const real py = nse.lat.lbm2physY(y) / (Y * nse.lat.physDl);
		return -lbm_V_0 * TNL::cos(2 * TNL::pi * px) * TNL::sin(2 * TNL::pi * py) * F;
	}

	void resetDFs() override
	{
		spdlog::info("Computing Taylor-Green initial condition");

		const auto lat = nse.lat;
		const dreal rho_0 = this->rho_0;
		const dreal V_0 = lbm_V_0;

		for (auto& block : nse.blocks) {
			const idx3d offset = block.offset;
#ifdef HAVE_MPI
			auto local_df = block.dfs[0].getLocalView();
#else
			auto local_df = block.dfs[0].getView();
#endif

			const idx3d begin = {0, 0, 0};
			const idx3d end = {block.local.y(), block.local.z(), block.local.x()};
			TNL::Algorithms::parallelFor<DeviceType>(
				begin,
				end,
				[lat, local_df, offset, V_0, rho_0] __cuda_callable__(const idx3d& yzx) mutable
				{
					const auto& [y_lat, z_lat, x_lat] = yzx;
					const idx x = offset.x() + x_lat;
					const idx y = offset.y() + y_lat;

					// Taylor-Green vortex at t=0 (F=1), using the same physical
					// coordinate mapping as analytical_vx/vy
					const real X = lat.global.x();
					const real Y = lat.global.y();
					const real px = lat.lbm2physX(x) / (X * lat.physDl);
					const real py = lat.lbm2physY(y) / (Y * lat.physDl);
					const dreal vx = V_0 * TNL::sin(2 * TNL::pi * px) * TNL::cos(2 * TNL::pi * py);
					const dreal vy = -V_0 * TNL::cos(2 * TNL::pi * px) * TNL::sin(2 * TNL::pi * py);
					const dreal vz = 0;

					NSE::COLL::setEquilibriumLat(local_df, x_lat, y_lat, z_lat, rho_0, vx, vy, vz);
				}
			);

			// copy the initialized DFs so that they are not overridden
			for (uint8_t dftype = 1; dftype < DFMAX; dftype++)
				block.dfs[dftype] = block.dfs[0];
		}

		nse.copyDFsToHost();
	}

	[[nodiscard]] std::vector<std::string> getOutputDataNames() const override
	{
		return {
			"lbm_density",
			"lbm_velocity_x",
			"lbm_velocity_y",
			"velocity_x",
			"velocity_y",
			"lbm_analytical_vx",
			"lbm_analytical_vy",
			"lbm_error_vx",
			"lbm_error_vy",
			"analytical_vx",
			"analytical_vy",
			"error_vx",
			"error_vy"
		};
	}

	void outputData(UniformDataWriter<TRAITS>& writer, const BLOCK& block, const idx3d& begin, const idx3d& end) override
	{
		writer.write("lbm_density", getMacroView<TRAITS>(block.hmacro, MACRO::e_rho), begin, end);
		writer.write("lbm_velocity_x", getMacroView<TRAITS>(block.hmacro, MACRO::e_vx), begin, end);
		writer.write("lbm_velocity_y", getMacroView<TRAITS>(block.hmacro, MACRO::e_vy), begin, end);
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
			"lbm_analytical_vx",
			[&](idx x, idx y, idx z) -> dreal
			{
				return analytical_vx(block.offset.x() + x, block.offset.y() + y);
			},
			begin,
			end
		);
		writer.write(
			"lbm_analytical_vy",
			[&](idx x, idx y, idx z) -> dreal
			{
				return analytical_vy(block.offset.x() + x, block.offset.y() + y);
			},
			begin,
			end
		);
		writer.write(
			"lbm_error_vx",
			[&](idx x, idx y, idx z) -> dreal
			{
				idx gx = block.offset.x() + x;
				idx gy = block.offset.y() + y;
				return TNL::abs(block.hmacro(MACRO::e_vx, x, y, z) - analytical_vx(gx, gy));
			},
			begin,
			end
		);
		writer.write(
			"lbm_error_vy",
			[&](idx x, idx y, idx z) -> dreal
			{
				idx gx = block.offset.x() + x;
				idx gy = block.offset.y() + y;
				return TNL::abs(block.hmacro(MACRO::e_vy, x, y, z) - analytical_vy(gx, gy));
			},
			begin,
			end
		);
		writer.write(
			"analytical_vx",
			[&](idx x, idx y, idx z) -> dreal
			{
				return nse.lat.lbm2physVelocity(analytical_vx(block.offset.x() + x, block.offset.y() + y));
			},
			begin,
			end
		);
		writer.write(
			"analytical_vy",
			[&](idx x, idx y, idx z) -> dreal
			{
				return nse.lat.lbm2physVelocity(analytical_vy(block.offset.x() + x, block.offset.y() + y));
			},
			begin,
			end
		);
		writer.write(
			"error_vx",
			[&](idx x, idx y, idx z) -> dreal
			{
				idx gx = block.offset.x() + x;
				idx gy = block.offset.y() + y;
				return nse.lat.lbm2physVelocity(TNL::abs(block.hmacro(MACRO::e_vx, x, y, z) - analytical_vx(gx, gy)));
			},
			begin,
			end
		);
		writer.write(
			"error_vy",
			[&](idx x, idx y, idx z) -> dreal
			{
				idx gx = block.offset.x() + x;
				idx gy = block.offset.y() + y;
				return nse.lat.lbm2physVelocity(TNL::abs(block.hmacro(MACRO::e_vy, x, y, z) - analytical_vy(gx, gy)));
			},
			begin,
			end
		);
	}

	void probe1() override
	{
		// compute L1 and L2 errors against the analytical solution
		// (skip non-fluid and non-periodic sites — only count interior fluid cells)
		auto& block = nse.blocks.front();
		real local_l1sum_vx = 0;
		real local_l1sum_vy = 0;
		real local_l2sum_vx = 0;
		real local_l2sum_vy = 0;
		for (int i = block.offset.x() + 1; i < block.offset.x() + block.local.x() - 1; i++)
			for (int j = block.offset.y() + 1; j < block.offset.y() + block.local.y() - 1; j++) {
				auto gi = block.hmap(i, j, 0);
				if (! NSE::BC::isFluid(gi))
					continue;
				real an_vx = analytical_vx(i, j);
				real an_vy = analytical_vy(i, j);
				real diff_vx = fabs(block.hmacro(MACRO::e_vx, i, j, 0) - an_vx);
				real diff_vy = fabs(block.hmacro(MACRO::e_vy, i, j, 0) - an_vy);
				local_l1sum_vx += diff_vx;
				local_l1sum_vy += diff_vy;
				local_l2sum_vx += TNL::sqr(diff_vx);
				local_l2sum_vy += TNL::sqr(diff_vy);
			}

		// MPI reduction of the local results
		real l1sum_vx = TNL::MPI::reduce(local_l1sum_vx, MPI_SUM, MPI_COMM_WORLD);
		real l1sum_vy = TNL::MPI::reduce(local_l1sum_vy, MPI_SUM, MPI_COMM_WORLD);
		real l2sum_vx = TNL::MPI::reduce(local_l2sum_vx, MPI_SUM, MPI_COMM_WORLD);
		real l2sum_vy = TNL::MPI::reduce(local_l2sum_vy, MPI_SUM, MPI_COMM_WORLD);

		// convert to physical units
		real area = nse.lat.physDl * nse.lat.physDl;
		auto to_phys = [&](real l1, real l2) -> std::pair<real, real>
		{
			real l1p = nse.lat.lbm2physVelocity(l1 * area);
			real l2p = nse.lat.lbm2physVelocity(sqrt(l2 * area));
			return {l1p, l2p};
		};
		auto [l1error_phys_vx, l2error_phys_vx] = to_phys(l1sum_vx, l2sum_vx);
		auto [l1error_phys_vy, l2error_phys_vy] = to_phys(l1sum_vy, l2sum_vy);

		if (nse.rank == 0)
			spdlog::info(
				"at t={:1.2f}s, iterations={:d} F={:e} l1error_phys_v=[{:e},{:e}] l2error_phys_v=[{:e},{:e}]",
				nse.physTime(),
				nse.iterations,
				decayFactor(),
				l1error_phys_vx,
				l1error_phys_vy,
				l2error_phys_vx,
				l2error_phys_vy
			);
	}
};

template <typename NSE>
void sim(const std::string& adios_config = "adios2.xml", int RESOLUTION = 1)
{
	using idx = typename NSE::TRAITS::idx;
	using real = typename NSE::TRAITS::real;
	using point_t = typename NSE::TRAITS::point_t;
	using lat_t = Lattice<3, real, idx>;

	int X = 64 * RESOLUTION;
	int Y = 64 * RESOLUTION;
	// Viscosity 0.01 gives tau = 0.53 (stable) and visible decay on a 64^2 grid.
	real LBM_VISCOSITY = 0.01;
	real PHYS_HEIGHT = 1.0;
	real PHYS_VISCOSITY = 1.5e-5;
	real PHYS_V_0 = 5e-3;  // velocity amplitude in physical units [m/s]
	real PHYS_DL = PHYS_HEIGHT / Y;
	real PHYS_DT = LBM_VISCOSITY / PHYS_VISCOSITY * PHYS_DL * PHYS_DL;
	point_t PHYS_ORIGIN = {-PHYS_HEIGHT / 2. + PHYS_DL, -PHYS_HEIGHT / 2. + PHYS_DL, 0.};

	lat_t lat;
	lat.global = typename lat_t::CoordinatesType(X, Y, 1);
	lat.physOrigin = PHYS_ORIGIN;
	lat.physDl = PHYS_DL;
	lat.physDt = PHYS_DT;
	lat.physViscosity = PHYS_VISCOSITY;

	const std::string state_id = fmt::format("sim2d_Taylor_Green_res{:02d}_np{:03d}", RESOLUTION, TNL::MPI::GetSize(MPI_COMM_WORLD));
	StateLocal<NSE> state(state_id, MPI_COMM_WORLD, lat, adios_config);

	if (! state.canCompute())
		return;

	state.lbm_V_0 = state.nse.lat.phys2lbmVelocity(PHYS_V_0);
	if (state.nse.rank == 0)
		spdlog::info("Taylor-Green: V_0_phys={:e} m/s, lbm_V_0={:e}, Ma={:e}", PHYS_V_0, state.lbm_V_0, state.lbm_V_0 * sqrt(3.0));

	// Run long enough for significant decay: F(physFinalTime) ~ 0.09
	// decay rate = 2 * nu * (2*pi/X)^2 per iteration, physFinalTime ~ 2000
	state.nse.physFinalTime = 2000.0;
	state.cnt[PRINT].period = 10.0;
	state.cnt[PROBE1].period = 10.0;
	state.cnt[OUT2D].period = 100.0;
	state.add2Dcut_Z(0, "");

	execute(state);
}

template <typename TRAITS = Traits<float, double, int>>
void run(const std::string& adios_config, int RES)
{
	using COLL = D2Q9_CLBM<TRAITS>;

	using NSE_CONFIG = LBM_CONFIG<
		TRAITS,
		D2Q9_KernelStruct,
		NSE2D_Data_Periodic<TRAITS>,
		COLL,
		typename COLL::EQ,
		D2Q9_STREAMING<TRAITS>,
		D2Q9_BC_All,
		D2Q9_MACRO_Default<TRAITS>>;

	sim<NSE_CONFIG>(adios_config, RES);
}

int main(int argc, char** argv)
{
	TNLMPI_INIT mpi(argc, argv);

	argparse::ArgumentParser program("sim2d_Taylor_Green");
	program.add_description(
		"2D Taylor-Green vortex with periodic BCs using D2Q9_CLBM, compared against the analytical solution F(t) = exp(-2*nu*k^2*t)."
	);
	program.add_argument("--adios-config").help("path to ADIOS2 configuration file").default_value(std::string("adios2.xml")).nargs(1);
	program.add_argument("--resolution").help("resolution of the lattice").scan<'i', int>().default_value(1).nargs(1);
	program.add_argument("--precision")
		.help("precision for numerical operations: single=32-bit (float), double=64-bit")
		.choices("single", "double")
		.default_value("single")
		.nargs(1);

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

	if (program.get<std::string>("--precision") == "double")
		run<Traits<double, double, int>>(adios_config, resolution);
	else
		run<Traits<float, double, int>>(adios_config, resolution);

	return 0;
}
