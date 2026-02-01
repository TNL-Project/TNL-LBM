#include <argparse/argparse.hpp>
#include <magic_enum/magic_enum.hpp>
#include <utility>

#include "lbm3d/core.h"
#include "lbm3d/lagrange_3D.h"
#include "lbm3d/obstacles_ibm.h"

// cylinder in 3D - Schafer-Turek problem
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

template <typename TRAITS>
struct NSE_Data_SpecialInflow : NSE_Data<TRAITS>
{
	using dreal = typename TRAITS::dreal;
	using idx = typename TRAITS::idx;

	dreal inflow_vx = 0;
	dreal inflow_vy = 0;
	dreal inflow_vz = 0;
	//dreal inflow_z0=0.00112;
	//dreal inflow_mez=0.0112;
	//dreal inflow_physDl=0.1;
	dreal physDl;
	dreal H;

	template <typename LBM_KS>
	CUDA_HOSTDEV void inflow(LBM_KS& KS, idx x, idx y, idx z)
	{
		KS.vx = 16.0 * inflow_vx
			  * TNL::max(0, (physDl * (y - 0.5) / H) * (physDl * (z - 0.5) / H) * (1.0 - physDl * (y - 0.5) / H) * (1.0 - physDl * (z - 0.5) / H));
		//(no1 - ((physDl*y-y0)*(physDl*y-y0) + (physDl*z-z0)*(physDl*z-z0))/delta/delta);
		//KS.vx = inflow_vx*(no1 - ((physDl*y-y0)*(physDl*y-y0) + (physDl*z-z0)*(physDl*z-z0))/delta/delta);
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
	using State<NSE>::ibm;
	using State<NSE>::id;

	using idx = typename TRAITS::idx;
	using idx3d = typename TRAITS::idx3d;
	using real = typename TRAITS::real;
	using dreal = typename TRAITS::dreal;
	using point_t = typename TRAITS::point_t;
	using lat_t = Lattice<3, real, idx>;

	// maximum velocity for the parabolic profile
	dreal lbm_inflow_vx_max = 0;
	// characteristic velocity (given by the Reynolds number)
	dreal lbm_char_velocity = 0;
	bool firstrun = true;
	real cylinder_diameter = 0.01;
	point_t cylinder_c;

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
			lbm_char_velocity * cylinder_diameter / nse.lat.physDl / nse.lat.lbmViscosity(),
			lbm_char_velocity,
			nse.lat.lbm2physVelocity(lbm_char_velocity)
		);

		// compute drag and lift coefficients (both are dimensionless numbers,
		// so we can do it just in LBM units and avoid converting force to physical units)
		const point_t F = ibm.integrateForce();
		const real reference_area = cylinder_diameter * nse.blocks.front().data.H / nse.lat.physDl / nse.lat.physDl;
		const real C_D = -F.x() * 2.0 / lbm_char_velocity / lbm_char_velocity / reference_area;
		const real C_L = -F.z() * 2.0 / lbm_char_velocity / lbm_char_velocity / reference_area;
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
			block.data.inflow_vx = lbm_inflow_vx_max;
			block.data.inflow_vy = 0;
			block.data.inflow_vz = 0;
			block.data.physDl = nse.lat.physDl;
			block.data.H = (nse.lat.global.y() - 2.0) * nse.lat.physDl;	 // domain width and height
		}
	}

	void setupBoundaries() override
	{
		nse.setBoundaryX(0, BC::GEO_INFLOW_LEFT);					   // left
		nse.setBoundaryX(nse.lat.global.x() - 1, BC::GEO_OUTFLOW_EQ);  // right
		nse.setBoundaryY(0, BC::GEO_WALL);							   // back
		nse.setBoundaryY(nse.lat.global.y() - 1, BC::GEO_WALL);		   // front
		nse.setBoundaryZ(0, BC::GEO_WALL);							   // top
		nse.setBoundaryZ(nse.lat.global.z() - 1, BC::GEO_WALL);		   // bottom
	}

	StateLocal(const std::string& id, const TNL::MPI::Comm& communicator, lat_t lat, const std::string& adiosConfigPath = "adios2.xml")
	: State<NSE>(id, communicator, std::move(lat), adiosConfigPath)
	{}
};

template <typename NSE>
int sim(
	int RES, double Re, double discretization_ratio, IbmCompute computeVariant, int dirac, IbmMethod methodVariant, const std::string& adiosConfigPath
)
{
	using idx = typename NSE::TRAITS::idx;
	using real = typename NSE::TRAITS::real;
	using point_t = typename NSE::TRAITS::point_t;
	using lat_t = Lattice<3, real, idx>;

	int block_size = 32;
	real cylinder_diameter = 0.10;	 // [m]
	real real_domain_height = 0.41;	 // [m]
	real real_domain_length = 2.00;	 // [m]
	idx LBM_Y = RES * block_size;	 // for 4 cm
	idx LBM_Z = LBM_Y;
	real PHYS_DL = real_domain_height / ((real) LBM_Y - 2.0);
	idx LBM_X = (int) (real_domain_length / PHYS_DL) + 2;
	point_t PHYS_ORIGIN = {0., 0., 0.};

	real PHYS_VISCOSITY = 0.001;  // [m^2/s]
	//real Umax = 0.45; // [m/s]
	real Ubar = Re * PHYS_VISCOSITY / cylinder_diameter;
	real Umax = 9.0 / 4.0 * Ubar;  // [m/s] // Re=20 --> 0.45m/s

	real LBM_VISCOSITY = 0.001;

	//fmt::print("input phys velocity {:f}\ninput lbm velocity {:f}\nRe {:f}\nlbm viscosity {:f}\nphys viscosity {:f}\n", i_PHYS_VELOCITY,
	//	i_LBM_VELOCITY, i_Re, i_LBM_VISCOSITY, i_PHYS_VISCOSITY);
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
		"sim_IBM1_{}_{}_dirac_{}_res_{}_Re_{}_nas_{:05.4f}_compute_{}", NSE::COLL::id, method_name, dirac, RES, Re, discretization_ratio, compute_name
	);
	StateLocal<NSE> state(state_id, MPI_COMM_WORLD, lat, adiosConfigPath);

	if (! state.canCompute())
		return 0;

	state.lbm_char_velocity = state.nse.lat.phys2lbmVelocity(Ubar);
	state.lbm_inflow_vx_max = state.nse.lat.phys2lbmVelocity(Umax);
	state.nse.physCharLength = cylinder_diameter;  // [m]
	state.cylinder_diameter = cylinder_diameter;   // [m]
	//state.nse.physFluidDensity = 1000.0; // [kg/m^3]

	spdlog::info(
		"Reynolds = {:f} Umax_lbm {:f} Umax_phys {:f} Ubar_lbm {:f} Ubar_phys {:f}", Re, state.lbm_inflow_vx_max, Umax, state.lbm_char_velocity, Ubar
	);

	state.cnt[PRINT].period = 0.1;
	state.cnt[PROBE1].period = 0.1;
	state.nse.physFinalTime = 10.0;

	//state.cnt[OUT3D].period = 1.0;
	state.cnt[OUT2D].period = 1.0;

	// add cuts
	state.add2Dcut_X(LBM_X / 2, "cut_X");
	//state.add2Dcut_X(2*cylinder_diameter/PHYS_DL,"cut_Xcylinder");
	state.add2Dcut_Y(LBM_Y / 2, "cut_Y");
	state.add2Dcut_Z(LBM_Z / 2, "cut_Z");

	// create immersed objects
	state.cylinder_c[0] = 0.50;	 // [m]
	state.cylinder_c[1] = 0;	 // [m]
	state.cylinder_c[2] = 0.20;	 // [m]
	real sigma = discretization_ratio * PHYS_DL;
	ibmSetupCylinder(state.ibm, state.cylinder_c, state.cylinder_diameter, sigma);
	state.writePoints("cylinder", 0, 0);

	// configure IBM
	state.ibm.computeVariant = computeVariant;
	state.ibm.diracDeltaTypeEL = dirac;
	state.ibm.methodVariant = methodVariant;

	execute(state);

	return 0;
}

template <typename TRAITS = TraitsSP>
void run(int res, double Re, double discretization_ratio, IbmCompute compute, int dirac, IbmMethod method, const std::string& adiosConfigPath)
{
	using COLL = D3Q27_CUM<TRAITS>;
	using NSE_CONFIG = LBM_CONFIG<
		TRAITS,
		D3Q27_KernelStruct,
		NSE_Data_SpecialInflow<TRAITS>,
		COLL,
		typename COLL::EQ,
		D3Q27_STREAMING<TRAITS>,
		D3Q27_BC_All,
		MacroLocal<TRAITS>>;

	sim<NSE_CONFIG>(res, Re, discretization_ratio, compute, dirac, method, adiosConfigPath);
}

int main(int argc, char** argv)
{
	TNLMPI_INIT mpi(argc, argv);

	argparse::ArgumentParser program("sim_IBM1");
	program.add_description("IBM-LBM simulation with cylinder in 3D - Schafer-Turek problem.");
	program.add_argument("--resolution").help("resolution of the lattice").scan<'i', int>().default_value(1).nargs(1);
	program.add_argument("--Re").help("desired Reynolds number (affects the inflow velocity)").scan<'g', double>().default_value(100.0).nargs(1);
	program.add_argument("--discretization-ratio")
		.help("ratio between the Lagrangian spacing step and the Eulerian spacing step")
		.scan<'g', double>()
		.default_value(0.25)
		.nargs(1);
	program.add_argument("--compute").help("IBM compute method").default_value("GPU").choices("GPU", "CPU", "hybrid", "hybrid_zerocopy").nargs(1);
	program.add_argument("--dirac").help("Dirac delta function to use in IBM").scan<'i', int>().default_value(1).choices(1, 2, 3, 4).nargs(1);
	program.add_argument("--method").help("IBM method").default_value("modified").choices("modified", "original").nargs(1);
	program.add_argument("--adios-config").help("path to adios2 configuration file").default_value(std::string("adios2.xml")).nargs(1);

	try {
		program.parse_args(argc, argv);
	}
	catch (const std::exception& err) {
		std::cerr << err.what() << '\n';
		std::cerr << program;
		return 1;
	}

	const auto resolution = program.get<int>("--resolution");
	const auto Re = program.get<double>("--Re");
	const auto discretization_ratio = program.get<double>("--discretization-ratio");
	const auto compute = program.get<std::string>("--compute");
	const auto dirac = program.get<int>("--dirac");
	const auto method = program.get<std::string>("--method");
	const auto adiosConfigPath = program.get<std::string>("--adios-config");

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

	const IbmCompute computeEnum = magic_enum::enum_cast<IbmCompute>(compute).value_or(IbmCompute::GPU);
	const IbmMethod methodEnum = magic_enum::enum_cast<IbmMethod>(method).value_or(IbmMethod::modified);

	run(resolution, Re, discretization_ratio, computeEnum, dirac, methodEnum, adiosConfigPath);

	return 0;
}
