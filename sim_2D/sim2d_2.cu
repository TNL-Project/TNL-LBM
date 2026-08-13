#include <argparse/argparse.hpp>
#include <utility>

#include "lbm3d/core.h"
#include "lbm3d/lbm_data.h"

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

template <typename TRAITS>
struct NSE2D_Data_XProfileInflow : NSE_Data<TRAITS>
{
	using idx = typename TRAITS::idx;
	using dreal = typename TRAITS::dreal;

	dreal* vx_profile = nullptr;
	idx size_y = 0;

	template <typename LBM_KS>
	CUDA_HOSTDEV void inflow(LBM_KS& KS, idx x, idx y, idx z)
	{
		KS.vx = vx_profile[y];
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

	TNL::Containers::Array<dreal, DeviceType, idx> vx_profile;

#ifdef HAVE_MPI
	TNL::Containers::DistributedNDArray<typename TRAITS::template array3d<real, TNL::Devices::Host>> an_cache;
#else
	typename TRAITS::template array3d<real, TNL::Devices::Host> an_cache;
#endif

	int errors_count;
	real* l1errors;
	int error_idx = 0;

	StateLocal(const std::string& id, const TNL::MPI::Comm& communicator, lat_t lat, bool use_forcing, const std::string& adios_config = "adios2.xml")
	: State<NSE>(
		  id,
		  communicator,
		  std::move(lat),
		  adios_config,
		  // conditional periodic domain in x-direction
		  bool3d{use_forcing, false, false}
	  )
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
		if (nse.blocks.front().data.vx_profile) {
			nse.setBoundaryX(1, BC::GEO_INFLOW_LEFT);								 // left
			nse.setBoundaryX(nse.lat.global.x() - 2, BC::GEO_OUTFLOW_RIGHT_INTERP);	 // right
		}

		nse.setBoundaryY(1, BC::GEO_WALL);						 // bottom
		nse.setBoundaryY(nse.lat.global.y() - 2, BC::GEO_WALL);	 // top

		// extra layer needed due to A-A pattern
		if (nse.blocks.front().data.vx_profile) {
			nse.setBoundaryX(0, BC::GEO_NOTHING);						// left
			nse.setBoundaryX(nse.lat.global.x() - 1, BC::GEO_NOTHING);	// right
		}
		nse.setBoundaryY(0, BC::GEO_NOTHING);						// bottom
		nse.setBoundaryY(nse.lat.global.y() - 1, BC::GEO_NOTHING);	// top
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

		// dynamic stopping criterion (based on vx error, the primary component)
		real l1error_phys = l1error_phys_vx;
		real threshold = 1e-4;
		real threshold_stddev = 1e-3;
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
		real stopping_stddev = l1prev > 0 ? stddev / l1prev : 0;
		if (stopping < threshold && stopping_stddev < threshold_stddev)
			nse.terminate = true;

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
};

template <typename NSE>
void sim(const std::string& adios_config, int RESOLUTION, bool use_forcing, double final_time)
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
	const char* bc_variant = use_forcing ? "forcing" : "inflow";
	const std::string state_id =
		fmt::format("sim2d_2_{}_{}_{}_res{:02d}_np{:03d}", NSE::COLL::id, prec, bc_variant, RESOLUTION, TNL::MPI::GetSize(MPI_COMM_WORLD));
	StateLocal<NSE> state(state_id, MPI_COMM_WORLD, lat, use_forcing, adios_config);

	if (! state.canCompute())
		return;

	// Hagen-Poiseuille analytical solution: u(y) = (G/(2*nu)) * (R^2 - y^2)
	// where G is the forcing term, R is half the channel height, nu is viscosity
	dreal force = 1e-4;
	if (use_forcing) {
		state.nse.blocks.front().data.fx = state.nse.lat.phys2lbmForce(force);
		state.nse.blocks.front().data.fy = 0;
		state.nse.blocks.front().data.vx_profile = nullptr;
		state.cache_analytical();
	}
	else {
		// compute the analytical solution using the forcing term
		state.nse.blocks.front().data.fx = state.nse.lat.phys2lbmForce(force);
		state.nse.blocks.front().data.fy = 0;
		state.cache_analytical();
		// reset the forcing — the inflow profile drives the flow
		state.nse.blocks.front().data.fx = 0;

		// allocate array for the inflow profile (parabolic Poiseuille)
		state.vx_profile.setSize(state.nse.blocks.front().local.y());
		state.nse.blocks.front().data.vx_profile = state.vx_profile.getData();
		state.nse.blocks.front().data.size_y = state.nse.blocks.front().local.y();

#ifdef USE_CUDA
		// build the profile on the host, then copy to the device array
		std::unique_ptr<dreal[]> analytical{new dreal[state.nse.blocks.front().local.y()]};
		for (int j = 0; j < state.nse.blocks.front().local.y(); j++)
			analytical[j] = state.analytical_vx(state.nse.blocks.front().offset.y() + j);
		TNL::Backend::memcpy(
			state.nse.blocks.front().data.vx_profile,
			analytical.get(),
			state.nse.blocks.front().local.y() * sizeof(dreal),
			TNL::Backend::MemcpyHostToDevice
		);
#else
		for (int j = 0; j < state.nse.blocks.front().local.y(); j++)
			state.nse.blocks.front().data.vx_profile[j] = state.analytical_vx(state.nse.blocks.front().offset.y() + j);
#endif
	}

	state.nse.physFinalTime = final_time;
	state.cnt[PRINT].period = 0.01;
	state.cnt[PROBE1].period = 0.1;

	// 2D = cut in 3D at z=0
	state.cnt[OUT2D].period = 10.0;
	state.add2Dcut_Z(0, "");

	execute(state);
}

template <typename TRAITS = Traits<float, double, int>>
void run(const std::string& adios_config, int RES, bool use_forcing, double final_time)
{
	using COLL = D2Q9_CLBM<TRAITS>;

	using NSE_CONFIG = LBM_CONFIG<
		TRAITS,
		D2Q9_KernelStruct,
		NSE2D_Data_XProfileInflow<TRAITS>,
		COLL,
		typename COLL::EQ,
		D2Q9_STREAMING<TRAITS>,
		D2Q9_BC_All,
		D2Q9_MACRO_Default<TRAITS>>;

	sim<NSE_CONFIG>(adios_config, RES, use_forcing, final_time);
}

int main(int argc, char** argv)
{
	TNLMPI_INIT mpi(argc, argv);

	argparse::ArgumentParser program("sim2d_2");
	program.add_description("2D Hagen-Poiseuille flow with verification against analytical solution.");
	program.add_argument("--adios-config").help("path to ADIOS2 configuration file").default_value(std::string("adios2.xml")).nargs(1);
	program.add_argument("--resolution").help("resolution of the lattice").scan<'i', int>().default_value(1).nargs(1);
	program.add_argument("--final-time").help("final time of the simulation").scan<'g', double>().default_value(100.0).nargs(1);
	program.add_argument("--use-forcing").help("use forcing term with periodic boundary conditions instead of inflow boundary condition").flag();
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

	const bool use_forcing = program.get<bool>("--use-forcing");

	if (program.get<std::string>("--precision") == "double")
		run<Traits<double, double, int>>(adios_config, resolution, use_forcing, final_time);
	else
		run<Traits<float, double, int>>(adios_config, resolution, use_forcing, final_time);

	return 0;
}
