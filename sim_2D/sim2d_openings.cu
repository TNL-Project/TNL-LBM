#include <argparse/argparse.hpp>
#include <utility>

#include "lbm3d/core.h"
#include "lbm3d/lbm_data.h"
#include "lbm3d/inflow_openings_lbm.h"

#include "lbm3d/d2q9/bc.h"
#include "lbm3d/d2q9/col_srt.h"
#include "lbm3d/d2q9/col_clbm.h"
#include "lbm3d/d2q9/macro.h"

#ifdef AA_PATTERN
	#include "lbm3d/d2q9/streaming_AA.h"
#endif
#ifdef AB_PATTERN
	#include "lbm3d/d2q9/streaming_AB.h"
#endif

// Inflow mode: one parabolic opening spanning the channel (Poiseuille
// reference flux), or two uniform openings (amplitudes v and 2v by channel
// half) proving per-opening amplitudes.
enum class Mode
{
	parabolic,
	dual
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

#ifdef HAVE_MPI
	TNL::Containers::DistributedNDArray<typename TRAITS::template array3d<real, TNL::Devices::Host>> an_cache;
#else
	typename TRAITS::template array3d<real, TNL::Devices::Host> an_cache;
#endif

	Mode mode = Mode::parabolic;
	// authored amplitudes resolved by sim() before execute() starts reset();
	// setupBoundaries() only forwards them to addInflowPlane
	dreal opening_flux = 0;	 // parabolic: target volumetric flux (unit cell area)
	dreal uniform_v = 0;	 // dual: per-opening base velocity (opening 1 gets 2x)
	// y-ranges of the dual openings [2, split-1] / [split, Y-3], kept for the
	// final per-opening mean-vx report
	idx split_y = 0;

	InflowOpeningsState<TRAITS> openings;

	int errors_count;
	real* l1errors;
	int error_idx = 0;
	real l1error_initial = -1;

	StateLocal(const std::string& id, const TNL::MPI::Comm& communicator, lat_t lat, const std::string& adios_config = "adios2.xml")
	: State<NSE>(id, communicator, std::move(lat), adios_config)
	{
		errors_count = 10;
		l1errors = new real[errors_count];
		for (int i = 0; i < errors_count; i++)
			l1errors[i] = 1;
	}

	~StateLocal() override
	{
		delete[] l1errors;
	}

	// Hagen-Poiseuille analytical solution: u(y) = (G/(2*nu)) * (R^2 - y^2)
	// where G is the forcing term, R is half the channel height, nu is viscosity
	real raw_analytical_vx(idx lbm_y)
	{
		if (lbm_y == 0 || lbm_y == nse.lat.global.y() - 1)
			return 0;

		idx wall_low = 1;
		idx wall_high = nse.lat.global.y() - 2;
		real R = (real) (wall_high - wall_low) / 2.0;
		real y_rel = (real) lbm_y - (real) (wall_low + wall_high) / 2.0;

		real G = nse.blocks.front().data.fx;
		real nu = nse.lat.lbmViscosity();
		return G / (2.0 * nu) * (R * R - y_rel * y_rel);
	}

	real analytical_vx(idx lbm_y)
	{
		if (an_cache.getData() == nullptr)
			cache_analytical();
		return an_cache(0, lbm_y, 0);
	}

	void cache_analytical()
	{
		const auto& block = nse.blocks.front();
		an_cache.setSizes(1, block.global.y(), 1);
#ifdef HAVE_MPI
		an_cache.template setDistribution<1>(block.offset.y(), block.offset.y() + block.local.y(), block.communicator);
		an_cache.allocate();
#endif

#pragma omp parallel for schedule(static) default(none) shared(block)
		for (idx y = block.offset.y(); y < block.offset.y() + block.local.y(); y++)
			an_cache(0, y, 0) = raw_analytical_vx(y);
	}

	void setupBoundaries() override
	{
		const idx X = nse.lat.global.x();
		const idx Y = nse.lat.global.y();
		// fluid inflow rows: walls sit at y=1 and y=Y-2 (stamped below)
		const idx y_lo = 2;
		const idx y_hi = Y - 3;

		// base imposed velocity (unit base that PARABOLIC shapes through the
		// precomputed site velocities)
		for (auto& block : nse.blocks)
			block.data.inflow_vx = 1;

		// authored inflow on the interior plane x=1; plane 0 stays the
		// GEO_NOTHING ghost layer (A-A pattern idiom, same as sim2d_2)
		if (mode == Mode::parabolic) {
			addInflowPlane(nse, openings, 0, -1, 1, idx3d{1, y_lo, 0}, idx3d{1, y_hi, 0}, ProfileType::PARABOLIC, opening_flux);
		}
		else {
			addInflowPlane(nse, openings, 0, -1, 1, idx3d{1, y_lo, 0}, idx3d{1, split_y - 1, 0}, ProfileType::UNIFORM, uniform_v);
			addInflowPlane(nse, openings, 0, -1, 1, idx3d{1, split_y, 0}, idx3d{1, y_hi, 0}, ProfileType::UNIFORM, 2 * uniform_v);
		}

		nse.setBoundaryX(X - 2, BC::GEO_OUTFLOW_RIGHT_INTERP);	// right

		nse.setBoundaryY(1, BC::GEO_WALL);		// bottom
		nse.setBoundaryY(Y - 2, BC::GEO_WALL);	// top

		// extra layer needed due to A-A pattern
		nse.setBoundaryX(0, BC::GEO_NOTHING);	   // left
		nse.setBoundaryX(X - 1, BC::GEO_NOTHING);  // right
		nse.setBoundaryY(0, BC::GEO_NOTHING);	   // bottom
		nse.setBoundaryY(Y - 1, BC::GEO_NOTHING);  // top

		finalizeInflowOpenings(nse, openings);
	}

	[[nodiscard]] std::vector<std::string> getOutputDataNames() const override
	{
		return {
			"lbm_density",
			"lbm_density_fluctuation",
			"lbm_velocity_x",
			"lbm_velocity_y",
			"velocity_x",
			"velocity_y",
			"lbm_analytical_vx",
			"lbm_error_vx",
			"analytical_vx",
			"error_vx"
		};
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
				return analytical_vx(y);
			},
			begin,
			end
		);
		writer.write(
			"lbm_error_vx",
			[&](idx x, idx y, idx z) -> dreal
			{
				return TNL::abs(block.hmacro(MACRO::e_vx, x, y, z) - analytical_vx(y));
			},
			begin,
			end
		);
		writer.write(
			"analytical_vx",
			[&](idx x, idx y, idx z) -> dreal
			{
				return nse.lat.lbm2physVelocity(analytical_vx(y));
			},
			begin,
			end
		);
		writer.write(
			"error_vx",
			[&](idx x, idx y, idx z) -> dreal
			{
				return nse.lat.lbm2physVelocity(TNL::abs(block.hmacro(MACRO::e_vx, x, y, z) - analytical_vx(y)));
			},
			begin,
			end
		);
	}

	// compute L1 and L2 errors against the analytical solution and print them
	// in sim2d_2's format; the dynamic termination criterion of sim2d_2 is
	// evaluated for the stopping printout but deliberately not applied, so the
	// regression runs stay fixed-length in both streaming patterns
	void reportErrors()
	{
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
				real an_vx = analytical_vx(j);
				real diff_vx = fabs(block.hmacro(MACRO::e_vx, i, j, 0) - an_vx);
				real diff_vy = fabs(block.hmacro(MACRO::e_vy, i, j, 0));
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

		// dynamic stopping criterion bookkeeping (based on vx error, the primary component)
		real l1error_phys = l1error_phys_vx;

		// record the first probe's error as the initial reference value
		if (l1error_initial < 0)
			l1error_initial = l1error_phys;

		real l1prev = 0.0;
		for (int i = 0; i < errors_count; i++)
			l1prev += l1errors[i];
		l1prev /= errors_count;
		real stddev = 0.0;
		for (int i = 0; i < errors_count; i++)
			stddev += TNL::sqr(l1errors[i] - l1prev);
		stddev /= (errors_count - 1);
		stddev = sqrt(stddev);
		real stopping = l1error_phys > 0 ? abs(l1prev - l1error_phys) / l1error_phys : 0;

		error_idx = (error_idx + 1) % errors_count;
		l1errors[error_idx] = l1error_phys;

		if (nse.rank == 0)
			spdlog::info(
				"at t={:1.2f}s, iterations={:d} l1error_phys_v=[{:e},{:e}] l2error_phys_v=[{:e},{:e}] stopping={:e}",
				nse.physTime(),
				nse.iterations,
				l1error_phys_vx,
				l1error_phys_vy,
				l2error_phys_vx,
				l2error_phys_vy,
				stopping
			);
	}

	// dual mode: mean imposed velocity per authored opening at the x=1 plane
	// plus the total mass balance across the channel
	void reportOpenings()
	{
		const idx X = nse.lat.global.x();
		const idx y_lo = 2;
		const idx y_hi = nse.lat.global.y() - 3;
		const idx lo_y[2] = {y_lo, split_y};
		const idx hi_y[2] = {split_y - 1, y_hi};

		real local_mean[2] = {0, 0};
		real local_outlet_flux = 0;
		for (auto& block : nse.blocks) {
			// owned rows of the inflow plane x=1
			if (1 >= block.offset.x() && 1 < block.offset.x() + block.local.x())
				for (int k = 0; k < 2; k++) {
					const idx begin = TNL::max(lo_y[k], block.offset.y());
					const idx end = TNL::min(hi_y[k], block.offset.y() + block.local.y() - 1);
					for (idx j = begin; j <= end; j++)
						local_mean[k] += block.hmacro(MACRO::e_vx, 1, j, 0);
				}
			// owned rows of the outflow plane x=X-2 (walls contribute vx=0)
			if (X - 2 >= block.offset.x() && X - 2 < block.offset.x() + block.local.x())
				for (idx j = block.offset.y(); j < block.offset.y() + block.local.y(); j++)
					local_outlet_flux += block.hmacro(MACRO::e_vx, X - 2, j, 0);
		}

		real mean_vx[2] = {TNL::MPI::reduce(local_mean[0], MPI_SUM, MPI_COMM_WORLD), TNL::MPI::reduce(local_mean[1], MPI_SUM, MPI_COMM_WORLD)};
		const real outlet_flux = TNL::MPI::reduce(local_outlet_flux, MPI_SUM, MPI_COMM_WORLD);
		const real count[2] = {real(hi_y[0] - lo_y[0] + 1), real(hi_y[1] - lo_y[1] + 1)};
		mean_vx[0] /= count[0];
		mean_vx[1] /= count[1];

		const real target_vx[2] = {uniform_v, 2 * uniform_v};
		const real inlet_flux = count[0] * target_vx[0] + count[1] * target_vx[1];
		const real flux_balance = inlet_flux != 0 ? TNL::abs(outlet_flux - inlet_flux) / TNL::abs(inlet_flux) : 0;

		if (nse.rank == 0) {
			for (int k = 0; k < 2; k++) {
				spdlog::info("opening_mean_vx[{}] = {:.8e}", k, mean_vx[k]);
				spdlog::info("opening_target_vx[{}] = {:.8e}", k, target_vx[k]);
			}
			spdlog::info("flux_balance = {:.8e}", flux_balance);
		}
	}

	void probe1() override
	{
		if (mode == Mode::parabolic)
			reportErrors();
	}

	void AfterSimFinished() override
	{
		// final-time report: the registered OUT2D/PROBE1 counters keep the
		// device+host macro current through the last output cycle; refresh the
		// host copy and report (computeInitialMacro is NOT usable here -- its
		// gather is only valid at the initial parity of the A-A pattern)
		nse.copyMacroToHost();
		if (mode == Mode::parabolic)
			reportErrors();
		else
			reportOpenings();

		State<NSE>::AfterSimFinished();
	}
};

template <typename NSE>
void sim(const std::string& adios_config, int RESOLUTION, Mode mode, double final_time)
{
	using idx = typename NSE::TRAITS::idx;
	using real = typename NSE::TRAITS::real;
	using dreal = typename NSE::TRAITS::dreal;
	using point_t = typename NSE::TRAITS::point_t;
	using lat_t = Lattice<3, real, idx>;

	int block_size = 32;
	int X = block_size * RESOLUTION;
	int Y = block_size * RESOLUTION;
	real LBM_VISCOSITY = 0.001;
	real PHYS_HEIGHT = 0.25;
	real PHYS_VISCOSITY = 1.5e-5;
	real PHYS_DL = PHYS_HEIGHT / real(Y - 2);
	real PHYS_DT = LBM_VISCOSITY / PHYS_VISCOSITY * PHYS_DL * PHYS_DL;
	point_t PHYS_ORIGIN = {0., 0., 0.};

	lat_t lat;
	lat.global = typename lat_t::CoordinatesType(X, Y, 1);
	lat.physOrigin = PHYS_ORIGIN;
	lat.physDl = PHYS_DL;
	lat.physDt = PHYS_DT;
	lat.physViscosity = PHYS_VISCOSITY;

	const char* prec = (std::is_same_v<dreal, float>) ? "float" : "double";
	const char* mode_str = (mode == Mode::dual) ? "dual" : "parabolic";
	const std::string state_id =
		fmt::format("sim2d_openings_{}_{}_{}_res{:02d}_np{:03d}", NSE::COLL::id, prec, mode_str, RESOLUTION, TNL::MPI::GetSize(MPI_COMM_WORLD));
	StateLocal<NSE> state(state_id, MPI_COMM_WORLD, lat, adios_config);
	state.mode = mode;

	if (! state.canCompute())
		return;

	// Hagen-Poiseuille analytical solution: u(y) = (G/(2*nu)) * (R^2 - y^2)
	// where G is the forcing term, R is half the channel height, nu is
	// viscosity; the same discrete reference that sim2d_2 verifies against
	dreal force = 1e-4;
	state.nse.blocks.front().data.fx = state.nse.lat.phys2lbmForce(force);
	state.cache_analytical();
	// no forcing -- the authored openings drive the flow
	state.nse.blocks.front().data.fx = 0;

	// amplitudes derived from the reference profile over the fluid inflow rows
	// (cached values -- raw_analytical_vx needs the forcing term that was
	// already reset above)
	const idx y_lo = 2;
	const idx y_hi = Y - 3;
	real profile_flux = 0;
	for (idx y = y_lo; y <= y_hi; y++)
		profile_flux += state.analytical_vx(y);
	if (mode == Mode::parabolic) {
		// flux-matched amplitude: the authored parabolic opening injects the
		// same total volumetric flow as sim2d_2's discrete inflow profile, so
		// the developed centerline velocities coincide
		state.opening_flux = profile_flux;
	}
	else {
		// per-opening base velocity v: the channel-mean of the reference
		// profile; the two uniform openings impose v and 2v
		state.uniform_v = profile_flux / (y_hi - y_lo + 1);
		state.split_y = y_lo + (y_hi - y_lo + 1) / 2;
	}

	state.nse.physFinalTime = final_time;
	state.cnt[PRINT].period = 10.0;
	state.cnt[PROBE1].period = 1.0;

	// 2D = cut in 3D at z=0
	state.cnt[OUT2D].period = 10.0;
	state.add2Dcut_Z(0, "");

	spdlog::info("mode = {}, authored amplitudes: flux = {:e}, uniform_v = {:e}", mode_str, state.opening_flux, state.uniform_v);

	execute(state);
}

template <typename TRAITS = Traits<float, double, int>>
void run(const std::string& adios_config, int RES, Mode mode, double final_time)
{
	using COLL = D2Q9_CLBM<TRAITS>;

	using NSE_CONFIG = LBM_CONFIG<
		TRAITS,
		D2Q9_KernelStruct,
		NSE_Data_OpeningInflow<TRAITS>,
		COLL,
		typename COLL::EQ,
		D2Q9_STREAMING<TRAITS>,
		D2Q9_BC_All,
		D2Q9_MACRO_Default<TRAITS>>;

	sim<NSE_CONFIG>(adios_config, RES, mode, final_time);
}

int main(int argc, char** argv)
{
	TNLMPI_INIT mpi(argc, argv);

	argparse::ArgumentParser program("sim2d_openings");
	program.add_description("2D Hagen-Poiseuille flow with authored inflow openings on the interior plane x=1.");
	program.add_argument("--adios-config").help("path to ADIOS2 configuration file").default_value(std::string("adios2.xml")).nargs(1);
	program.add_argument("--resolution").help("resolution of the lattice").scan<'i', int>().default_value(1).nargs(1);
	program.add_argument("--final-time").help("final time of the simulation").scan<'g', double>().default_value(100.0).nargs(1);
	program.add_argument("--mode")
		.help("inflow opening layout: parabolic = single flux-matched opening, dual = two uniform openings with amplitudes v and 2v")
		.choices("parabolic", "dual")
		.default_value("parabolic")
		.nargs(1);
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

	const auto final_time = program.get<double>("--final-time");
	if (final_time <= 0) {
		fmt::println(stderr, "CLI error: final-time must be positive");
		return 1;
	}

	const Mode mode = (program.get<std::string>("--mode") == "dual") ? Mode::dual : Mode::parabolic;

	if (program.get<std::string>("--precision") == "double")
		run<Traits<double, double, int>>(adios_config, resolution, mode, final_time);
	else
		run<Traits<float, double, int>>(adios_config, resolution, mode, final_time);

	return 0;
}
