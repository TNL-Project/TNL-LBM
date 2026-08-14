#include <argparse/argparse.hpp>
#include <magic_enum/magic_enum.hpp>
#include <utility>

#include "lbm3d/core.h"

// 3D test problem: forcing/input velocity
// analytical solution for rectangular duct: forcing accelerated

enum class Scaling : std::uint8_t
{
	strong,
	weak_1d,
	weak_3d,
};

template <typename TRAITS>
struct NSE_Data_XProfileInflow : NSE_Data<TRAITS>
{
	using idx = typename TRAITS::idx;
	using dreal = typename TRAITS::dreal;

	dreal* vx_profile = nullptr;
	idx size_y = 0;

	template <typename LBM_KS>
	CUDA_HOSTDEV void inflow(LBM_KS& KS, idx x, idx y, idx z)
	{
		KS.vx = vx_profile[y + z * size_y];
		KS.vy = 0;
		KS.vz = 0;
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

	// array for the inflow velocity profile (pointer is passed to the LBM kernel)
	TNL::Containers::Array<dreal, DeviceType, idx> vx_profile;

#ifdef HAVE_MPI
	TNL::Containers::DistributedNDArray<typename TRAITS::template array3d<real, TNL::Devices::Host>> an_cache;
#else
	typename TRAITS::template array3d<real, TNL::Devices::Host> an_cache;
#endif
	int an_n = 50;

	int errors_count;
	real* l1errors;
	int error_idx = 0;
	real l1error_initial = -1;

	StateLocal(
		const std::string& id, const TNL::MPI::Comm& communicator, lat_t lat, bool use_forcing, const std::string& adiosConfigPath = "adios2.xml"
	)
	: State<NSE>(
		  id,
		  communicator,
		  std::move(lat),
		  adiosConfigPath,
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

	real raw_analytical_vx(int n, idx lbm_y, idx lbm_z)
	{
		if (lbm_y == 0 || lbm_y == nse.lat.global.y() - 1 || lbm_z == 0 || lbm_z == nse.lat.global.z() - 1)
			return 0;

		real a = nse.lat.global.y() / 2.0 - 1.0;
		real b = nse.lat.global.z() / 2.0 - 1.0;
		real y = ((real) lbm_y + 0.5 - nse.lat.global.y() / 2.) / a;
		real z = ((real) lbm_z + 0.5 - nse.lat.global.z() / 2.) / a;
		real b_ku_a = b / a;
		real sum = 0;
		real minusonek = 1.0;
		real kkk;
		real omega = PI / 2.0;
		for (int k = 0; k <= n; k++) {
			kkk = 2.0 * k + 1.;
			sum += minusonek
				 * (1.0 - exp(omega * kkk * (z - b_ku_a)) * (1.0 + exp(-omega * 2.0 * kkk * z)) / (1.0 + exp(-omega * 2.0 * kkk * b_ku_a)))
				 * cos(omega * kkk * y) / kkk / kkk / kkk;
			minusonek *= -1.0;
		}

		//real coef = (nse.blocks.front().data.fx != 0) ? nse.blocks.front().data.fx : nse.blocks.front().data.inflow_vx;
		real coef = nse.blocks.front().data.fx;
		return coef * 16.0 * a * a / PI / PI / PI * sum / nse.lat.lbmViscosity();
	}

	real analytical_vx(idx lbm_y, idx lbm_z)
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
		an_cache.template setDistribution<1>(block.offset.y(), block.offset.y() + block.local.y(), block.communicator);
		an_cache.template setDistribution<2>(block.offset.z(), block.offset.z() + block.local.z(), block.communicator);
		an_cache.allocate();
#endif

#pragma omp parallel for schedule(static) collapse(2) default(none) shared(block)
		for (idx z = block.offset.z(); z < block.offset.z() + block.local.z(); z++)
			for (idx y = block.offset.y(); y < block.offset.y() + block.local.y(); y++)
				an_cache(0, y, z) = raw_analytical_vx(an_n, y, z);
	}

	void setupBoundaries() override
	{
		if (nse.blocks.front().data.vx_profile) {
			nse.setBoundaryX(1, BC::GEO_INFLOW_LEFT);								 // left
			nse.setBoundaryX(nse.lat.global.x() - 2, BC::GEO_OUTFLOW_RIGHT_INTERP);	 // right
		}

		nse.setBoundaryZ(1, BC::GEO_WALL);						 // bottom
		nse.setBoundaryZ(nse.lat.global.z() - 2, BC::GEO_WALL);	 // top
		nse.setBoundaryY(1, BC::GEO_WALL);						 // front
		nse.setBoundaryY(nse.lat.global.y() - 2, BC::GEO_WALL);	 // back

		// extra layer needed due to A-A pattern
		if (nse.blocks.front().data.vx_profile) {
			nse.setBoundaryX(0, BC::GEO_NOTHING);						// left
			nse.setBoundaryX(nse.lat.global.x() - 1, BC::GEO_NOTHING);	// right
		}
		nse.setBoundaryZ(0, BC::GEO_NOTHING);						// bottom
		nse.setBoundaryZ(nse.lat.global.z() - 1, BC::GEO_NOTHING);	// top
		nse.setBoundaryY(0, BC::GEO_NOTHING);						// front
		nse.setBoundaryY(nse.lat.global.y() - 1, BC::GEO_NOTHING);	// back
	}

	[[nodiscard]] std::vector<std::string> getOutputDataNames() const override
	{
		// return all quantity names used in outputData
		return {
			"lbm_density",
			"lbm_density_fluctuation",
			"lbm_velocity_x",
			"lbm_velocity_y",
			"lbm_velocity_z",
			"velocity_x",
			"velocity_y",
			"velocity_z",
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
		writer.write("lbm_velocity_z", getMacroView<TRAITS>(block.hmacro, MACRO::e_vz), begin, end);
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
		writer.write(
			"lbm_analytical_vx",
			[&](idx x, idx y, idx z) -> dreal
			{
				return analytical_vx(y, z);
			},
			begin,
			end
		);
		writer.write(
			"lbm_error_vx",
			[&](idx x, idx y, idx z) -> dreal
			{
				return TNL::abs(block.hmacro(MACRO::e_vx, x, y, z) - analytical_vx(y, z));
			},
			begin,
			end
		);
		writer.write(
			"analytical_vx",
			[&](idx x, idx y, idx z) -> dreal
			{
				return nse.lat.lbm2physVelocity(analytical_vx(y, z));
			},
			begin,
			end
		);
		writer.write(
			"error_vx",
			[&](idx x, idx y, idx z) -> dreal
			{
				return nse.lat.lbm2physVelocity(TNL::abs(block.hmacro(MACRO::e_vx, x, y, z) - analytical_vx(y, z)));
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
		real local_l1sum_vz = 0;
		real local_l2sum_vx = 0;
		real local_l2sum_vy = 0;
		real local_l2sum_vz = 0;
		for (int i = block.offset.x() + 1; i < block.offset.x() + block.local.x() - 1; i++)
			for (int j = block.offset.y() + 1; j < block.offset.y() + block.local.y() - 1; j++)
				for (int k = block.offset.z() + 1; k < block.offset.z() + block.local.z() - 1; k++) {
					auto gi = block.hmap(i, j, k);
					if (! NSE::BC::isFluid(gi))
						continue;
					real an_vx = analytical_vx(j, k);
					real diff_vx = fabs(block.hmacro(MACRO::e_vx, i, j, k) - an_vx);
					real diff_vy = fabs(block.hmacro(MACRO::e_vy, i, j, k));
					real diff_vz = fabs(block.hmacro(MACRO::e_vz, i, j, k));
					local_l1sum_vx += diff_vx;
					local_l1sum_vy += diff_vy;
					local_l1sum_vz += diff_vz;
					local_l2sum_vx += TNL::sqr(diff_vx);
					local_l2sum_vy += TNL::sqr(diff_vy);
					local_l2sum_vz += TNL::sqr(diff_vz);
				}

		// MPI reduction of the local results
		real l1sum_vx = TNL::MPI::reduce(local_l1sum_vx, MPI_SUM, MPI_COMM_WORLD);
		real l1sum_vy = TNL::MPI::reduce(local_l1sum_vy, MPI_SUM, MPI_COMM_WORLD);
		real l1sum_vz = TNL::MPI::reduce(local_l1sum_vz, MPI_SUM, MPI_COMM_WORLD);
		real l2sum_vx = TNL::MPI::reduce(local_l2sum_vx, MPI_SUM, MPI_COMM_WORLD);
		real l2sum_vy = TNL::MPI::reduce(local_l2sum_vy, MPI_SUM, MPI_COMM_WORLD);
		real l2sum_vz = TNL::MPI::reduce(local_l2sum_vz, MPI_SUM, MPI_COMM_WORLD);

		// convert to physical units
		real vol = nse.lat.physDl * nse.lat.physDl * nse.lat.physDl;
		auto to_phys = [&](real l1, real l2) -> std::pair<real, real>
		{
			real l1p = nse.lat.lbm2physVelocity(l1 * vol);
			real l2p = nse.lat.lbm2physVelocity(sqrt(l2 * vol));
			return {l1p, l2p};
		};
		auto [l1error_phys_vx, l2error_phys_vx] = to_phys(l1sum_vx, l2sum_vx);
		auto [l1error_phys_vy, l2error_phys_vy] = to_phys(l1sum_vy, l2sum_vy);
		auto [l1error_phys_vz, l2error_phys_vz] = to_phys(l1sum_vz, l2sum_vz);

		// dynamic stopping criterion (based on vx error, the primary component)
		real l1error_phys = l1error_phys_vx;

		// record the first probe's error as the initial reference value
		if (l1error_initial < 0)
			l1error_initial = l1error_phys;

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
		// magnitude gate: do not allow termination until the error has actually
		// dropped below half of the initial error
		if (l1error_phys <= 0.5 * l1error_initial && stopping < threshold && stopping_stddev < threshold_stddev)
			nse.terminate = true;

		error_idx = (error_idx + 1) % errors_count;
		l1errors[error_idx] = l1error_phys;

		if (nse.rank == 0)
			spdlog::info(
				"at t={:1.2f}s, iterations={:d} l1error_phys_v=[{:e},{:e},{:e}] l2error_phys_v=[{:e},{:e},{:e}] stopping={:e}",
				nse.physTime(),
				nse.iterations,
				l1error_phys_vx,
				l1error_phys_vy,
				l1error_phys_vz,
				l2error_phys_vx,
				l2error_phys_vy,
				l2error_phys_vz,
				stopping
			);
	}
};

template <typename NSE>
void sim(const std::string& adios_config, int RES, bool use_forcing, Scaling scaling, double final_time)
{
	using idx = typename NSE::TRAITS::idx;
	using real = typename NSE::TRAITS::real;
	using dreal = typename NSE::TRAITS::dreal;
	using point_t = typename NSE::TRAITS::point_t;
	using lat_t = Lattice<3, real, idx>;

	int block_size = 32;
	int LBM_X = block_size;
	if (! use_forcing)
		LBM_X *= RES;
	int LBM_Y = RES * block_size;
	int LBM_Z = RES * block_size;
	if (scaling == Scaling::weak_1d)
		LBM_X *= TNL::MPI::GetSize(MPI_COMM_WORLD);
	else if (scaling == Scaling::weak_3d) {
		// NOTE: scale volume by nproc, preserve the proportions of the domain
		const real factor = std::cbrt(TNL::MPI::GetSize(MPI_COMM_WORLD));
		LBM_X = std::round(LBM_X * factor);
		LBM_Y = std::round(LBM_Y * factor);
		LBM_Z = std::round(LBM_Z * factor);
	}
	// NOTE: LBM_VISCOSITY must be less than 1/6
	real LBM_VISCOSITY = 0.001;
	real PHYS_VISCOSITY = 1.5e-5;  // [m^2/s] fluid viscosity air: 1.81e-5
	real PHYS_HEIGHT = 0.25;
	real PHYS_DL = PHYS_HEIGHT / real(LBM_Z - 2);
	real PHYS_DT = LBM_VISCOSITY / PHYS_VISCOSITY * PHYS_DL * PHYS_DL;
	point_t PHYS_ORIGIN = {0., 0., 0.};

	// initialize the lattice
	lat_t lat;
	lat.global = typename lat_t::CoordinatesType(LBM_X, LBM_Y, LBM_Z);
	lat.physOrigin = PHYS_ORIGIN;
	lat.physDl = PHYS_DL;
	lat.physDt = PHYS_DT;
	lat.physViscosity = PHYS_VISCOSITY;

	const char* prec = (std::is_same_v<dreal, float>) ? "float" : "double";
	const char* bc_variant = (use_forcing) ? "forcing" : "velocity";
	const auto scaling_variant = magic_enum::enum_name(scaling);
	const std::string state_id =
		fmt::format("sim_2_{}_{}_{}_{}_res_{}_np_{}", NSE::COLL::id, prec, bc_variant, scaling_variant, RES, TNL::MPI::GetSize(MPI_COMM_WORLD));
	StateLocal<NSE> state(state_id, MPI_COMM_WORLD, lat, use_forcing, adios_config);

	if (! state.canCompute())
		return;

	if (state.nse.blocks.front().local.x() <= 2) {
		std::cout << "Local block size " << state.nse.blocks.front().local.x() << " is too small, skipping this resolution." << std::endl;
		return;
	}

	// NOTE: this is for NSE_Data_ConstInflow
	//if (use_forcing)
	//{
	//	state.nse.blocks.front().data.fx = state.nse.lat.phys2lbmForce(1e-4);
	//	state.nse.blocks.front().data.fy = 0;
	//	state.nse.blocks.front().data.fz = 0;
	//	state.nse.blocks.front().data.inflow_vx = 0;
	//	state.nse.blocks.front().data.inflow_vy = 0;
	//	state.nse.blocks.front().data.inflow_vz = 0;
	//} else
	//{
	//	state.nse.blocks.front().data.fx = 0;
	//	state.nse.blocks.front().data.fy = 0;
	//	state.nse.blocks.front().data.fz = 0;
	//	state.nse.blocks.front().data.inflow_vx = state.nse.lat.phys2lbmVelocity(2e-6);
	//	state.nse.blocks.front().data.inflow_vy = 0;
	//	state.nse.blocks.front().data.inflow_vz = 0;
	//}

	// NOTE: this is for NSE_Data_XProfileInflow
	dreal force = 1e-4;
	if (use_forcing) {
		dreal fx_lbm = state.nse.lat.phys2lbmForce(force);
		state.nse.blocks.front().data.fx = fx_lbm;
		state.nse.blocks.front().data.fy = 0;
		state.nse.blocks.front().data.fz = 0;
		state.nse.blocks.front().data.vx_profile = nullptr;
		state.cache_analytical();
		if (std::abs(fx_lbm) < std::numeric_limits<dreal>::epsilon())
			spdlog::warn(
				"lattice force {:e} is below the precision floor {:e} — the body force will be lost in rounding",
				fx_lbm,
				std::numeric_limits<dreal>::epsilon()
			);
	}
	else {
		// calculate analytical solution using forcing just like above
		state.nse.blocks.front().data.fx = state.nse.lat.phys2lbmForce(force);
		state.nse.blocks.front().data.fy = 0;
		state.nse.blocks.front().data.fz = 0;
		state.cache_analytical();
		// reset the forcing for the LBM simulation
		state.nse.blocks.front().data.fx = 0;

		// allocate array for the inflow profile
		state.vx_profile.setSize(state.nse.blocks.front().local.y() * state.nse.blocks.front().local.z());
		state.nse.blocks.front().data.vx_profile = state.vx_profile.getData();
		state.nse.blocks.front().data.size_y = state.nse.blocks.front().local.y();

#ifdef USE_CUDA
		// convert analytical solution from double to float
		std::unique_ptr<dreal[]> analytical{new dreal[state.nse.blocks.front().local.y() * state.nse.blocks.front().local.z()]};
		for (int j = 0; j < state.nse.blocks.front().local.y(); j++)
			for (int k = 0; k < state.nse.blocks.front().local.z(); k++)
				analytical[k * state.nse.blocks.front().local.y() + j] =
					state.analytical_vx(state.nse.blocks.front().offset.y() + j, state.nse.blocks.front().offset.z() + k);
		// copy the analytical profile to the GPU
		TNL::Backend::memcpy(
			state.nse.blocks.front().data.vx_profile,
			analytical.get(),
			state.nse.blocks.front().local.y() * state.nse.blocks.front().local.z() * sizeof(dreal),
			TNL::Backend::MemcpyHostToDevice
		);
#else
		for (int j = 0; j < state.nse.blocks.front().local.y(); j++)
			for (int k = 0; k < state.nse.blocks.front().local.z(); k++)
				state.nse.blocks.front().data.vx_profile[k * state.nse.blocks.front().local.y() + j] =
					state.analytical_vx(state.nse.blocks.front().offset.y() + j, state.nse.blocks.front().offset.z() + k);
#endif
	}

	state.cnt[PRINT].period = 10.0;
	state.cnt[PROBE1].period = 1.0;
	//state.nse.physFinalTime = PHYS_DT * 1e7;
	state.nse.physFinalTime = final_time;
	//state.cnt[OUT2D].period = 1.0;

	if (scaling == Scaling::weak_3d) {
		// TRICK to keep the benchmark fast: decrease the periods and physFinalTime
		// to keep the compute time (more or less) constant
		const real factor = (LBM_Y - 2) / real(block_size * RES - 2) * RES / 2;
		state.cnt[PRINT].period /= factor;
		state.cnt[PROBE1].period /= factor;
		state.nse.physFinalTime /= factor;
	}

	spdlog::info("PHYS_DL = {:e}", PHYS_DL);
	//spdlog::info("in lbm units: forcing={:e} velocity={:e}", state.nse.blocks.front().data.fx,
	//state.nse.blocks.front().data.inflow_vx);
	spdlog::info("in lbm units: forcing={:e}", force);

	// add cuts
	//state.add2Dcut_X(LBM_X/2,"cut_X");
	//state.add2Dcut_Y(LBM_Y/2,"cut_Y");
	//state.add2Dcut_Z(LBM_Z/2,"cut_Z");

	execute(state);
}

template <typename TRAITS = TraitsSP>
void run(const std::string& adios_config, int RES, bool use_forcing, Scaling scaling, double final_time)
{
	using COLL = D3Q27_CUM<TRAITS, D3Q27_EQ_INV_CUM<TRAITS>>;
	//using COLL = D3Q27_FCLBM<TRAITS>;
	//using COLL = D3Q27_SRT<TRAITS>;
	//using COLL = D3Q27_SRT_WELL<TRAITS>;
	//using COLL = D3Q27_SRT_MODIF_FORCE<TRAITS>;
	//using COLL = D3Q27_BGK<TRAITS>;
	//using COLL = D3Q27_KBC_N1<TRAITS>;
	//using COLL = D3Q27_CUM<TRAITS>;
	//using COLL = D3Q27_CLBM<TRAITS>;
	//using COLL = D3Q27_CLBM_WELL<TRAITS>;
	//using COLL = D3Q27_CUM_SGS<TRAITS>;
	//using COLL = D3Q27_CUM_FIX<TRAITS>;
	//using COLL = D3Q27_CUM_WELL<TRAITS>;

	using NSE_CONFIG = LBM_CONFIG<
		TRAITS,
		D3Q27_KernelStruct,
		NSE_Data_XProfileInflow<TRAITS>,
		COLL,
		typename COLL::EQ,
		D3Q27_STREAMING<TRAITS>,
		D3Q27_BC_All,
		D3Q27_MACRO_Default<TRAITS>>;

	sim<NSE_CONFIG>(adios_config, RES, use_forcing, scaling, final_time);
}

int main(int argc, char** argv)
{
	TNLMPI_INIT mpi(argc, argv);

	argparse::ArgumentParser program("sim_2");
	program.add_description("Square duct flow with verification against analytical solution.");
	program.add_argument("--adios-config").help("path to ADIOS2 configuration file").default_value(std::string("adios2.xml")).nargs(1);
	program.add_argument("--min-resolution").help("minimum resolution of the lattice").scan<'i', int>().default_value(2).nargs(1);
	program.add_argument("--max-resolution").help("maximum resolution of the lattice").scan<'i', int>().default_value(4).nargs(1);
	program.add_argument("--final-time").help("final time of the simulation").scan<'g', double>().default_value(100.0).nargs(1);
	program.add_argument("--use-forcing").help("use forcing term with periodic boundary conditions instead of inflow boundary condition").flag();
	program.add_argument("--scaling")
		.help("parallel scaling mode (affects the global lattice size and its distribution into subdomains)")
		.choices("strong", "weak_1d", "weak_3d")
		.default_value("strong")
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

	const auto min_resolution = program.get<int>("--min-resolution");
	const auto max_resolution = program.get<int>("--max-resolution");
	if (min_resolution < 1) {
		fmt::println(stderr, "CLI error: min-resolution must be at least 1");
		return 1;
	}
	if (max_resolution < min_resolution) {
		fmt::println(stderr, "CLI error: max-resolution={} must be at least min-resolution={}", min_resolution);
		return 1;
	}

	const auto final_time = program.get<double>("--final-time");
	if (final_time <= 0) {
		fmt::println(stderr, "CLI error: final-time must be positive");
		return 1;
	}

	const auto adios_config = program.get<std::string>("--adios-config");
	const bool use_forcing = program.get<bool>("--use-forcing");
	const auto scaling_name = program.get<std::string>("--scaling");
	const Scaling scaling = magic_enum::enum_cast<Scaling>(scaling_name).value_or(Scaling::strong);

	for (int i = min_resolution; i <= max_resolution; i++) {
		int res = pow(2, i);
		if (program.get<std::string>("--precision") == "double")
			run<TraitsDP>(adios_config, res, use_forcing, scaling, final_time);
		else
			run<TraitsSP>(adios_config, res, use_forcing, scaling, final_time);
	}

	return 0;
}
