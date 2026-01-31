#define USE_GEIER_CUM_2017		 // use Geier 2017 Cummulant improvement A,B terms
#define USE_GEIER_CUM_ANTIALIAS	 // use antialiasing Dxu, Dyv, Dzw from Geier 2015/2017

#include <argparse/argparse.hpp>
#include <utility>

#include "lbm3d/core.h"
#include "lbm3d/lbm_data.h"

// Taylor-Green vortex

template <typename REAL>
struct Tensor
{
	REAL xx = 0, xy = 0, xz = 0;
	REAL yx = 0, yy = 0, yz = 0;
	REAL zx = 0, zy = 0, zz = 0;
};

// computes velocity gradient in lattice units and assuming periodic boundary conditions
template <typename MACRO, typename MACRO_ARRAY, typename lat_t, typename bool3d>
__cuda_callable__ Tensor<typename MACRO_ARRAY::ValueType> computeGradVel(
	const MACRO_ARRAY& macro,
	const lat_t& lat,
	const bool3d& distributed,	// multiindex specifying if the lattice is distributed along each axis
	typename lat_t::GlobalIndexType x,
	typename lat_t::GlobalIndexType y,
	typename lat_t::GlobalIndexType z
)
{
	using real = typename MACRO_ARRAY::ValueType;
	using idx = typename lat_t::GlobalIndexType;
	using tensor = Tensor<real>;

	const idx xp = (! distributed.x() && x == lat.global.x() - 1) ? 0 : (x + 1);
	const idx xm = (! distributed.x() && x == 0) ? (lat.global.x() - 1) : (x - 1);
	const idx yp = (! distributed.y() && y == lat.global.y() - 1) ? 0 : (y + 1);
	const idx ym = (! distributed.y() && y == 0) ? (lat.global.y() - 1) : (y - 1);
	const idx zp = (! distributed.z() && z == lat.global.z() - 1) ? 0 : (z + 1);
	const idx zm = (! distributed.z() && z == 0) ? (lat.global.z() - 1) : (z - 1);

	tensor G;
	// grad vel tensor
	G.xx = (real) 0.5 * (macro(MACRO::e_vx, xp, y, z) - macro(MACRO::e_vx, xm, y, z));
	G.xy = (real) 0.5 * (macro(MACRO::e_vx, x, yp, z) - macro(MACRO::e_vx, x, ym, z));
	G.xz = (real) 0.5 * (macro(MACRO::e_vx, x, y, zp) - macro(MACRO::e_vx, x, y, zm));
	G.yx = (real) 0.5 * (macro(MACRO::e_vy, xp, y, z) - macro(MACRO::e_vy, xm, y, z));
	G.yy = (real) 0.5 * (macro(MACRO::e_vy, x, yp, z) - macro(MACRO::e_vy, x, ym, z));
	G.yz = (real) 0.5 * (macro(MACRO::e_vy, x, y, zp) - macro(MACRO::e_vy, x, y, zm));
	G.zx = (real) 0.5 * (macro(MACRO::e_vz, xp, y, z) - macro(MACRO::e_vz, xm, y, z));
	G.zy = (real) 0.5 * (macro(MACRO::e_vz, x, yp, z) - macro(MACRO::e_vz, x, ym, z));
	G.zz = (real) 0.5 * (macro(MACRO::e_vz, x, y, zp) - macro(MACRO::e_vz, x, y, zm));
	return G;
}

template <typename MACRO, typename MACRO_ARRAY, typename lat_t, typename bool3d>
__cuda_callable__ typename lat_t::PointType computeVorticity(
	const MACRO_ARRAY& macro,
	const lat_t& lat,
	const bool3d& distributed,	// multiindex specifying if the lattice is distributed along each axis
	typename lat_t::GlobalIndexType x,
	typename lat_t::GlobalIndexType y,
	typename lat_t::GlobalIndexType z
)
{
	using tensor = Tensor<typename MACRO_ARRAY::ValueType>;
	using vector = typename lat_t::PointType;

	const tensor G = computeGradVel<MACRO>(macro, lat, distributed, x, y, z);
	vector vorticity;
	/*
		\vec \omega = \nabla \times \vec v =
		(
			\partial v_z / \partial y - \partial v_y - \partial z,
			\partial v_x / \partial z - \partial v_z - \partial x,
			\partial v_y / \partial x - \partial v_x - \partial y,
	   )
	*/
	vorticity.x() = G.zy - G.yz;
	vorticity.y() = G.xz - G.zx;
	vorticity.z() = G.yx - G.xy;
	return vorticity;
}

template <typename TRAITS>
struct D3Q27_MACRO_Sync : D3Q27_MACRO_Default<TRAITS>
{
	// needed for computing velocity gradient
	static const bool use_syncMacro = true;
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

	// problem parameters
	dreal rho_0 = 1;
	dreal V_0 = 0;
	dreal L = 0;

	std::shared_ptr<spdlog::logger> kinetic_energy_logger = nullptr;

	void setupBoundaries() override
	{
		nse.setBoundaryX(0, BC::GEO_PERIODIC);						 // left
		nse.setBoundaryX(nse.lat.global.x() - 1, BC::GEO_PERIODIC);	 // right
		nse.setBoundaryY(0, BC::GEO_PERIODIC);						 // back
		nse.setBoundaryY(nse.lat.global.y() - 1, BC::GEO_PERIODIC);	 // front
		nse.setBoundaryZ(0, BC::GEO_PERIODIC);						 // top
		nse.setBoundaryZ(nse.lat.global.z() - 1, BC::GEO_PERIODIC);	 // bottom
	}

	void resetDFs() override
	{
		spdlog::info("Computing initial condition");
		for (auto& block : nse.blocks) {
			// extract variables and views for capturing in the lambda function
			const idx3d offset = nse.blocks.front().offset;
			const lat_t lat = nse.lat;
			const dreal L = this->L;
			const dreal V_0 = lat.phys2lbmVelocity(this->V_0);
			const dreal rho_0 = this->rho_0;
#ifdef HAVE_MPI
			auto local_df = block.dfs[0].getLocalView();
#else
			auto local_df = block.dfs[0].getView();
#endif

			// compute the initial condition
			const idx3d begin = {0, 0, 0};
			const idx3d end = {block.local.y(), block.local.z(), block.local.x()};
			TNL::Algorithms::parallelFor<DeviceType>(
				begin,
				end,
				[local_df, offset, lat, L, V_0, rho_0] __cuda_callable__(const idx3d& yzx) mutable
				{
					const auto& [y_lat, z_lat, x_lat] = yzx;
					// convert local coordinates to physical coordinates
					const dreal x = lat.lbm2physX(offset.x() + x_lat);
					const dreal y = lat.lbm2physY(offset.y() + y_lat);
					const dreal z = lat.lbm2physZ(offset.z() + z_lat);
					// Taylor-Green vortex
					const dreal u = V_0 * TNL::sin(x / L) * TNL::cos(y / L) * TNL::cos(z / L);
					const dreal v = -V_0 * TNL::cos(x / L) * TNL::sin(y / L) * TNL::cos(z / L);
					const dreal w = 0;
					// 3 = 1/c_s^2
					const dreal rho = rho_0 + 3 * (V_0 * V_0 / 16) * (TNL::cos(2 * x / L) + TNL::cos(2 * y / L)) * (TNL::cos(2 * z / L) + 2);
					NSE::COLL::setEquilibriumLat(local_df, x_lat, y_lat, z_lat, rho, u, v, w);
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
		// return all quantity names used in outputData
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

	void probe1() override
	{
		spdlog::info("probe1 called at t={:f} iter={:d}", nse.physTime(), nse.iterations);

		// Initialize the logger
		if (kinetic_energy_logger == nullptr && nse.rank == 0) {
			const std::string dir = fmt::format("results_{}/probe1", this->id);
			mkdir_p(dir.c_str(), 0777);
			const std::string fname = fmt::format("{}/{}_rank{:03d}.txt", dir, "kinetic_energy", this->nse.rank);

			const bool truncate = true;
			auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(fname, truncate);
			file_sink->set_level(spdlog::level::trace);
			file_sink->set_pattern(fmt::format("%v"));	// log only the actual text

			kinetic_energy_logger = std::make_shared<spdlog::logger>("kinetic_energy", file_sink);
			kinetic_energy_logger->set_level(spdlog::level::trace);

			// Write the data format header line
			kinetic_energy_logger->info("# iter time kinetic_energy enstrophy enstrophy_dissipation");

			spdlog::info("probe1 logger initialized");
		}

		const dreal domain_volume = nse.lat.global.x() * nse.lat.global.y() * nse.lat.global.z();

		// extract variables and views for capturing in the lambda function
		const lat_t lat = nse.lat;
		const idx3d local_size = nse.blocks.front().local;
		const bool3d distributed = nse.blocks.front().is_distributed();
#ifdef HAVE_MPI
		const auto& dmacro_view = nse.blocks.front().dmacro.getLocalView();
#else
		const auto& dmacro_view = nse.blocks.front().dmacro.getView();
#endif

		// integrate the kinetic energy on the whole domain
		const real lbm_kinetic_energy_local =  //
			TNL::Algorithms::reduce<DeviceType>(
				idx(0),
				local_size.x() * local_size.y() * local_size.z(),
				[local_size, dmacro_view] __cuda_callable__(idx i) -> real
				{
					const idx x = i % local_size.x();
					const idx y = (i / local_size.x()) % local_size.y();
					const idx z = i / (local_size.x() * local_size.y());
					const dreal rho = dmacro_view(MACRO::e_rho, x, y, z);
					const dreal vx = dmacro_view(MACRO::e_vx, x, y, z);
					const dreal vy = dmacro_view(MACRO::e_vy, x, y, z);
					const dreal vz = dmacro_view(MACRO::e_vz, x, y, z);
					return rho * (vx * vx + vy * vy + vz * vz);
				},
				TNL::Plus{},
				real(0)
			);
		const real lbm_kinetic_energy = TNL::MPI::reduce(lbm_kinetic_energy_local, MPI_SUM, nse.communicator) / rho_0 / 2 / domain_volume;
		const real kinetic_energy = nse.lat.lbm2physVelocity(nse.lat.lbm2physVelocity(lbm_kinetic_energy));

		// integrate enstrophy on the whole domain
		const real lbm_enstrophy_local =  //
			TNL::Algorithms::reduce<DeviceType>(
				idx(0),
				local_size.x() * local_size.y() * local_size.z(),
				[lat, local_size, distributed, dmacro_view] __cuda_callable__(idx i) -> real
				{
					const idx x = i % local_size.x();
					const idx y = (i / local_size.x()) % local_size.y();
					const idx z = i / (local_size.x() * local_size.y());
					const dreal rho = dmacro_view(MACRO::e_rho, x, y, z);
					//const point_t vorticity = computeVorticity<MACRO>(dmacro_view, lat, distributed, x, y, z);
					//return rho * TNL::dot(vorticity, vorticity);
					const auto G = computeGradVel<MACRO>(dmacro_view, lat, distributed, x, y, z);
					const real G_norm_squared =
						G.xx * G.xx + G.yy * G.yy + G.zz * G.zz + G.xy * G.xy + G.xz * G.xz + G.yz * G.yz + G.yx * G.yx + G.zx * G.zx + G.zy * G.zy;
					return rho * G_norm_squared;
				},
				TNL::Plus{},
				real(0)
			);
		const real lbm_enstrophy = TNL::MPI::reduce(lbm_enstrophy_local, MPI_SUM, nse.communicator) / rho_0 / 2 / domain_volume;
		const real enstrophy = nse.lat.lbm2physVelocity(nse.lat.lbm2physVelocity(lbm_enstrophy)) / nse.lat.physDl / nse.lat.physDl;
		const real enstrophy_dissipation = enstrophy * 2 * nse.lat.physViscosity;

		if (nse.rank == 0) {
			kinetic_energy_logger->info(
				"{:06d} {:12.8e} {:12.8e} {:12.8e} {:12.8e}", nse.iterations, nse.physTime(), kinetic_energy, enstrophy, enstrophy_dissipation
			);
			kinetic_energy_logger->flush();
		}
	}

	StateLocal(const std::string& id, const TNL::MPI::Comm& communicator, lat_t lat, const std::string& adiosConfigPath, std::vector<BLOCK>&& blocks)
	: State<NSE>(id, communicator, std::move(lat), adiosConfigPath, std::move(blocks))
	{}
};

template <typename NSE>
int sim(
	const std::string& adiosConfigPath = "adios2.xml",
	int RESOLUTION = 2,
	double Re = 1600,			 // [-] Reynolds number
	double LBM_VISCOSITY = 1e-4	 // [Δx^2/Δt]
)
{
	using BLOCK = LBM_BLOCK<NSE>;
	using idx = typename NSE::TRAITS::idx;
	using real = typename NSE::TRAITS::real;
	using dreal = typename NSE::TRAITS::dreal;
	using point_t = typename NSE::TRAITS::point_t;
	using lat_t = Lattice<3, real, idx>;

	int X0 = 32;
	int X = X0 * RESOLUTION;
	int Y = X0 * RESOLUTION;
	int Z = X0 * RESOLUTION;
	// problem parameters - based on https://onlinelibrary.wiley.com/doi/pdf/10.1002/fld.3767
	real L = 1;															// [m] arbitrary
	real V_0 = 1;														// [m/s] amplitude of the wave
	real PHYS_VISCOSITY = V_0 * L / Re;									// [m^2/s]
	real PHYS_LENGTH = 2 * TNL::pi * L;									// [m] length of the domain
	real PHYS_DL = PHYS_LENGTH / X;										// [m] length of a lattice cell
	real PHYS_DT = LBM_VISCOSITY / PHYS_VISCOSITY * PHYS_DL * PHYS_DL;	// [s] physical time step
	real convective_time = L / V_0;										// [s] non-dimensional convective time
	// origin is in the center of the domain
	point_t PHYS_ORIGIN = -PHYS_DL / 2 * point_t{X, Y, Z} + PHYS_DL;

	// initialize the lattice
	lat_t lat;
	lat.global = typename lat_t::CoordinatesType(X, Y, Z);
	lat.physOrigin = PHYS_ORIGIN;
	lat.physDl = PHYS_DL;
	lat.physDt = PHYS_DT;
	lat.physViscosity = PHYS_VISCOSITY;

	// periodic lattice decomposition
	// (but do not add overlaps for single process simulations)
	const bool periodic_lattice = TNL::MPI::GetSize(MPI_COMM_WORLD) > 1;
	std::vector<BLOCK> blocks;
	blocks.emplace_back(decomposeLattice_D1Q3<NSE>(MPI_COMM_WORLD, lat.global, periodic_lattice));

	const std::string state_id = fmt::format(
		"sim_4_{}_np{:03d}/res={:02d}_Re={:g}_nu={:e}", TNL::getType<dreal>(), TNL::MPI::GetSize(MPI_COMM_WORLD), RESOLUTION, Re, LBM_VISCOSITY
	);
	StateLocal<NSE> state(state_id, MPI_COMM_WORLD, lat, adiosConfigPath, std::move(blocks));
	state.V_0 = V_0;
	state.L = L;
	spdlog::info("Physical parameters: L={}, V_0={}, Re={}, nu={}, dl={}, dt={}", L, V_0, Re, PHYS_VISCOSITY, PHYS_DL, PHYS_DT);

	if (! state.canCompute())
		return 0;

	state.nse.physFinalTime = 20 * convective_time;
	state.cnt[PRINT].period = state.nse.physFinalTime / 1000;
	// probe only in even iterations
	state.cnt[PROBE1].period = idx(state.nse.physFinalTime / 1000 / (2 * PHYS_DT)) * 2 * PHYS_DT;
	if (state.cnt[PROBE1].period == 0)
		state.cnt[PROBE1].period = 2 * PHYS_DT;

	// add outputs
	state.cnt[OUT2D].period = state.nse.physFinalTime / 1000;
	state.add2Dcut_X(0, "left_side");
	state.add2Dcut_X(X / 2, "cut_X");
	state.add2Dcut_Y(Y / 2, "cut_Y");
	state.add2Dcut_Z(Z / 2, "cut_Z");
	state.cnt[OUT3D].period = state.nse.physFinalTime / 10;

	execute(state);

	return 0;
}

template <typename TRAITS = TraitsSP>
void run(const std::string& adiosConfigPath, int RES, double Re, double lbm_viscosity)
{
	using COLL = D3Q27_CUM<TRAITS, D3Q27_EQ_INV_CUM<TRAITS>>;

	using NSE_CONFIG = LBM_CONFIG<
		TRAITS,
		D3Q27_KernelStruct,
		NSE_Data_ConstInflow<TRAITS>,
		COLL,
		typename COLL::EQ,
		D3Q27_STREAMING<TRAITS>,
		D3Q27_BC_All,
		D3Q27_MACRO_Sync<TRAITS>>;

	sim<NSE_CONFIG>(adiosConfigPath, RES, Re, lbm_viscosity);
}

int main(int argc, char** argv)
{
	TNLMPI_INIT mpi(argc, argv);

	argparse::ArgumentParser program("sim_1");
	program.add_description("3D Taylor-Green vortex simulation using incompressible Navier-Stokes equations.");
	program.add_argument("--adios-config").help("path to adios2.xml configuration file").default_value(std::string("adios2.xml"));
	program.add_argument("--resolution").help("resolution of the lattice").scan<'i', int>().default_value(1);
	program.add_argument("--Re").help("desired Reynolds number").scan<'g', double>().default_value(1600.0).nargs(1);
	program.add_argument("--lbm-viscosity").help("LBM viscosity [Δx^2/Δt]").scan<'g', double>().default_value(1e-4).nargs(1);

	try {
		program.parse_args(argc, argv);
	}
	catch (const std::exception& err) {
		std::cerr << err.what() << '\n';
		std::cerr << program;
		return 1;
	}

	const auto adiosConfigPath = program.get<std::string>("adios-config");
	const auto resolution = program.get<int>("--resolution");
	const auto Re = program.get<double>("--Re");
	const auto lbm_viscosity = program.get<double>("--lbm-viscosity");

	if (resolution < 1) {
		fmt::println(stderr, "CLI error: resolution must be at least 1");
		return 1;
	}
	if (Re < 1) {
		fmt::println(stderr, "CLI error: Re must be at least 1");
		return 1;
	}
	if (lbm_viscosity <= 0.0 || lbm_viscosity > 1. / 6.) {
		fmt::println(stderr, "CLI error: LBM viscosity must be in range (0, 1/6]");
		return 1;
	}

	run(adiosConfigPath, resolution, Re, lbm_viscosity);

	return 0;
}
