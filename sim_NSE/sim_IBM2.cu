#include <argparse/argparse.hpp>
#include <magic_enum/magic_enum.hpp>
#include <utility>

#include "lbm3d/core.h"
#include "lbm3d/lagrange_3D.h"
#include "lbm3d/obstacles_ibm.h"

// ball in 3D
// IBM-LBM

template <typename TRAITS>
struct MacroLocal : D3Q27_MACRO_Base<TRAITS>
{
	using dreal = typename TRAITS::dreal;
	using idx = typename TRAITS::idx;

	enum QuantityNames : std::uint8_t
	{
		e_fx,
		e_fy,
		e_fz,
		e_vx,
		e_vy,
		e_vz,
		e_rho,
		N
	};

	template <typename LBM_DATA, typename LBM_KS>
	CUDA_HOSTDEV static void outputMacro(LBM_DATA& SD, LBM_KS& KS, idx x, idx y, idx z)
	{
		SD.macro(e_rho, x, y, z) = KS.rho;
		SD.macro(e_vx, x, y, z) = KS.vx;
		SD.macro(e_vy, x, y, z) = KS.vy;
		SD.macro(e_vz, x, y, z) = KS.vz;
	}

	template <typename LBM_DATA>
	CUDA_HOSTDEV static void zeroForces(LBM_DATA& SD, idx x, idx y, idx z)
	{
		SD.macro(e_fx, x, y, z) = 0;
		SD.macro(e_fy, x, y, z) = 0;
		SD.macro(e_fz, x, y, z) = 0;
	}

	template <typename LBM_DATA, typename LBM_KS>
	CUDA_HOSTDEV static void copyQuantities(LBM_DATA& SD, LBM_KS& KS, idx x, idx y, idx z)
	{
		KS.lbmViscosity = SD.lbmViscosity;
		KS.fx = SD.macro(e_fx, x, y, z);
		KS.fy = SD.macro(e_fy, x, y, z);
		KS.fz = SD.macro(e_fz, x, y, z);
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
	using State<NSE>::ibm;
	using State<NSE>::id;

	using idx = typename TRAITS::idx;
	using idx3d = typename TRAITS::idx3d;
	using real = typename TRAITS::real;
	using dreal = typename TRAITS::dreal;
	using point_t = typename TRAITS::point_t;
	using lat_t = Lattice<3, real, idx>;

	dreal lbm_inflow_vx = 0;
	bool firstrun = true;
	real ball_diameter = 0.01;
	point_t ball_c;

	StateLocal(const std::string& id, const TNL::MPI::Comm& communicator, lat_t lat, const std::string& adiosConfigPath = "adios2.xml")
	: State<NSE>(id, communicator, std::move(lat), adiosConfigPath)
	{}

	void setupBoundaries() override
	{
		// symmetry planes first, so inflow/outflow overwrite them at edges/corners
		nse.setBoundaryY(1, BC::GEO_SYMMETRY);						 // front
		nse.setBoundaryY(nse.lat.global.y() - 2, BC::GEO_SYMMETRY);	 // back
		nse.setBoundaryZ(1, BC::GEO_SYMMETRY);						 // bottom
		nse.setBoundaryZ(nse.lat.global.z() - 2, BC::GEO_SYMMETRY);	 // top

		// inflow/outflow next, so they win over symmetry at edges/corners
		nse.setBoundaryX(1, BC::GEO_INFLOW_LEFT);								 // left
		nse.setBoundaryX(nse.lat.global.x() - 2, BC::GEO_OUTFLOW_RIGHT_INTERP);	 // right

		// extra layer needed due to A-A pattern
		nse.setBoundaryX(0, BC::GEO_NOTHING);						// left
		nse.setBoundaryX(nse.lat.global.x() - 1, BC::GEO_NOTHING);	// right
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
			"lbm_force_x",
			"lbm_force_y",
			"lbm_force_z",
			"force_x",
			"force_y",
			"force_z",
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
		writer.write("lbm_force_x", getMacroView<TRAITS>(block.hmacro, MACRO::e_fx), begin, end);
		writer.write("lbm_force_y", getMacroView<TRAITS>(block.hmacro, MACRO::e_fy), begin, end);
		writer.write("lbm_force_z", getMacroView<TRAITS>(block.hmacro, MACRO::e_fz), begin, end);
		writer.write(
			"force_x",
			[&](idx x, idx y, idx z) -> dreal
			{
				return nse.lat.lbm2physForce(block.hmacro(MACRO::e_fx, x, y, z));
			},
			begin,
			end
		);
		writer.write(
			"force_y",
			[&](idx x, idx y, idx z) -> dreal
			{
				return nse.lat.lbm2physForce(block.hmacro(MACRO::e_fy, x, y, z));
			},
			begin,
			end
		);
		writer.write(
			"force_z",
			[&](idx x, idx y, idx z) -> dreal
			{
				return nse.lat.lbm2physForce(block.hmacro(MACRO::e_fz, x, y, z));
			},
			begin,
			end
		);
	}

	void probe1() override
	{
		spdlog::info(
			"Reynolds = {:f} lbmvel {:f} physvel {:f}",
			lbm_inflow_vx * ball_diameter / nse.lat.physDl / nse.lat.lbmViscosity(),
			lbm_inflow_vx,
			nse.lat.lbm2physVelocity(lbm_inflow_vx)
		);

		// compute drag and lift coefficients (both are dimensionless numbers,
		// so we can do it just in LBM units and avoid converting force to physical units)
		const point_t F = ibm.integrateForce();
		const real reference_area = PI * ball_diameter * ball_diameter / nse.lat.physDl / nse.lat.physDl;
		const real C_D = -F.x() * 2.0 / lbm_inflow_vx / lbm_inflow_vx / reference_area;
		const real C_L = -F.z() * 2.0 / lbm_inflow_vx / lbm_inflow_vx / reference_area;
		spdlog::info("F=[{:e}, {:e}, {:e}] C_D={:e} C_L={:e}", F.x(), F.y(), F.z(), C_D, C_L);

		// empty files
		const char* iotype = (firstrun) ? "wt" : "at";
		firstrun = false;
		// output
		FILE* f;
		const std::string dir = fmt::format("results_{}/probes", id);
		mkdir_p(dir.c_str(), 0755);

		std::string str = fmt::format("{}/probe_cd", dir);
		f = fopen(str.c_str(), iotype);
		fprintf(f, "%e\t%e\n", nse.physTime(), C_D);
		fclose(f);

		str = fmt::format("{}/probe_cl", dir);
		f = fopen(str.c_str(), iotype);
		fprintf(f, "%e\t%e\n", nse.physTime(), C_L);
		fclose(f);
	}

	void updateKernelVelocities() override
	{
		for (auto& block : nse.blocks) {
			block.data.inflow_vx = lbm_inflow_vx;
			block.data.inflow_vy = 0;
			block.data.inflow_vz = 0;
		}
	}
};

template <typename NSE>
void
sim(const std::string& adios_config,
	int RES,
	double Re,
	double discretization_ratio,
	IbmCompute computeVariant,
	int dirac,
	IbmMethod methodVariant,
	int n_spheres,
	double final_time,
	bool mtx_output)
{
	using idx = typename NSE::TRAITS::idx;
	using real = typename NSE::TRAITS::real;
	using point_t = typename NSE::TRAITS::point_t;
	using lat_t = Lattice<3, real, idx>;

	int block_size = 32;
	real ball_diameter = 0.01;
	real real_domain_height = ball_diameter * 11;  // [m]
	real real_domain_length = real_domain_height;  // [m]
	idx LBM_Y = RES * block_size;
	idx LBM_Z = LBM_Y;
	real PHYS_DL = real_domain_height / ((real) LBM_Y);
	idx LBM_X = (int) (real_domain_length / PHYS_DL);
	point_t PHYS_ORIGIN = {0., 0., 0.};

	real PHYS_VISCOSITY = 1e-5;	 // [m^2/s]
	real PHYS_VELOCITY = Re * PHYS_VISCOSITY / ball_diameter;

	real LBM_VELOCITY = 0.07;  // Geier
	real LBM_VISCOSITY = LBM_VELOCITY * ball_diameter / PHYS_DL / Re;

	real PHYS_DT = LBM_VISCOSITY / PHYS_VISCOSITY * PHYS_DL * PHYS_DL;

	// initialize the lattice
	lat_t lat;
	lat.global = typename lat_t::CoordinatesType(LBM_X, LBM_Y, LBM_Z);
	lat.physOrigin = PHYS_ORIGIN;
	lat.physDl = PHYS_DL;
	lat.physDt = PHYS_DT;
	lat.physViscosity = PHYS_VISCOSITY;

	const auto compute_name = magic_enum::enum_name(computeVariant);
	const auto method_name = magic_enum::enum_name(methodVariant);
	const std::string state_id = fmt::format(
		"sim_IBM2_{}_{}_dirac_{}_res_{}_Re_{}_nas_{:05.4f}_compute_{}_spheres{}",
		NSE::COLL::id,
		method_name,
		dirac,
		RES,
		Re,
		discretization_ratio,
		compute_name,
		n_spheres
	);
	StateLocal<NSE> state(state_id, MPI_COMM_WORLD, lat, adios_config);

	if (! state.canCompute())
		return;

	state.lbm_inflow_vx = state.nse.lat.phys2lbmVelocity(PHYS_VELOCITY);
	state.nse.physCharLength = ball_diameter;  // [m]
	state.ball_diameter = ball_diameter;	   // [m]
	//state.nse.physFluidDensity = 1000.0; // [kg/m^3]

	state.cnt[PRINT].period = 0.1;
	state.cnt[PROBE1].period = 0.1;
	state.nse.physFinalTime = final_time;

	//state.cnt[OUT3D].period = 1.0;
	state.cnt[OUT2D].period = 1.0;

	// add cuts
	state.add2Dcut_X(LBM_X / 2, "cut_X");
	//state.add2Dcut_X(2*BALL_DIAMETER/PHYS_DL,"cut_Xball");
	state.add2Dcut_Y(LBM_Y / 2, "cut_Y");
	state.add2Dcut_Z(LBM_Z / 2, "cut_Z");

	// create immersed objects
	state.ball_c[0] = 2 * state.ball_diameter;
	state.ball_c[1] = 5.5 * state.ball_diameter;
	state.ball_c[2] = 5.5 * state.ball_diameter;
	real sigma = discretization_ratio * PHYS_DL;
	ibmDrawSphere(state.ibm, state.ball_c, state.ball_diameter / 2.0, sigma);

	// 2nd ball (only when --spheres 2)
	// FIXME: computation of drag and lift coefficients assumes only one sphere
	if (n_spheres == 2) {
		state.ball_c[0] = 5.5 * state.ball_diameter;
		ibmDrawSphere(state.ibm, state.ball_c, state.ball_diameter / 2.0, sigma);
	}

	state.writePoints("ball", 0, 0);

	// configure IBM
	state.ibm.computeVariant = computeVariant;
	state.ibm.diracDeltaTypeEL = dirac;
	state.ibm.methodVariant = methodVariant;
	// A/M matrix dump is opt-in via --mtx-output; the IBM matrix regression test relies on it
	state.ibm.mtx_output = mtx_output;

	execute(state);
}

template <typename TRAITS = TraitsSP>
void
run(const std::string& adios_config,
	int res,
	double Re,
	double discretization_ratio,
	IbmCompute compute,
	int dirac,
	IbmMethod method,
	int spheres,
	double final_time,
	bool mtx_output)
{
	using COLL = D3Q27_CUM<TRAITS>;
	using NSE_CONFIG = LBM_CONFIG<
		TRAITS,
		D3Q27_KernelStruct,
		NSE_Data_ConstInflow<TRAITS>,
		COLL,
		typename COLL::EQ,
		D3Q27_STREAMING<TRAITS>,
		D3Q27_BC_All,
		MacroLocal<TRAITS>>;

	sim<NSE_CONFIG>(adios_config, res, Re, discretization_ratio, compute, dirac, method, spheres, final_time, mtx_output);
}

int main(int argc, char** argv)
{
	TNLMPI_INIT mpi(argc, argv);

	argparse::ArgumentParser program("sim_IBM2");
	program.add_description("IBM-LBM simulation with spheres in 3D.");
	program.add_argument("--adios-config").help("path to ADIOS2 configuration file").default_value(std::string("adios2.xml")).nargs(1);
	program.add_argument("--resolution").help("resolution of the lattice").scan<'i', int>().default_value(1).nargs(1);
	program.add_argument("--Re").help("desired Reynolds number (affects the inflow velocity)").scan<'g', double>().default_value(100.0).nargs(1);
	program.add_argument("--spheres")
		.help("number of spheres: 1 = single sphere (default), 2 = upstream + downstream spheres")
		.scan<'i', int>()
		.default_value(1)
		.nargs(1);
	program.add_argument("--final-time").help("override the physical final time").scan<'g', double>().default_value(30.0).nargs(1);
	program.add_argument("--mtx-output").help("write IBM A/M matrices to .mtx files (used by the IBM matrix regression test)").flag();
	program.add_argument("--discretization-ratio")
		.help("ratio between the Lagrangian spacing step and the Eulerian spacing step")
		.scan<'g', double>()
		.default_value(0.25)
		.nargs(1);
	program.add_argument("--compute").help("IBM compute method").default_value("GPU").choices("GPU", "CPU", "hybrid", "hybrid_zerocopy").nargs(1);
	program.add_argument("--dirac").help("Dirac delta function to use in IBM").scan<'i', int>().default_value(1).choices(1, 2, 3, 4).nargs(1);
	program.add_argument("--method").help("IBM method").default_value("modified").choices("modified", "original").nargs(1);

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
	const auto Re = program.get<double>("--Re");
	const auto discretization_ratio = program.get<double>("--discretization-ratio");
	const auto compute = program.get<std::string>("--compute");
	const auto dirac = program.get<int>("--dirac");
	const auto method = program.get<std::string>("--method");
	const auto spheres = program.get<int>("--spheres");
	const auto final_time = program.get<double>("--final-time");

	if (resolution < 1) {
		fmt::println(stderr, "CLI error: resolution must be at least 1");
		return 1;
	}
	if (Re < 1) {
		fmt::println(stderr, "CLI error: Re must be at least 1");
		return 1;
	}
	if (discretization_ratio <= 0) {
		fmt::println(stderr, "CLI error: discretization-ratio must be positive");
		return 1;
	}
	if (spheres != 1 && spheres != 2) {
		fmt::println(stderr, "CLI error: spheres must be 1 or 2");
		return 1;
	}

	const IbmCompute computeEnum = magic_enum::enum_cast<IbmCompute>(compute).value_or(IbmCompute::GPU);
	const IbmMethod methodEnum = magic_enum::enum_cast<IbmMethod>(method).value_or(IbmMethod::modified);

	run(adios_config, resolution, Re, discretization_ratio, computeEnum, dirac, methodEnum, spheres, final_time, program.get<bool>("--mtx-output"));

	return 0;
}
