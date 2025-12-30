#define AB_PATTERN

#include <argparse/argparse.hpp>
#include <filesystem>
#include <cmath>
#include <cfloat>

#include "lbm3d/core.h"
#include "saveload.h"
#include "velocity_profiles.h"

#define DIRECTORY "results_adjoint"

// FIXME: loadPrimaryAndMeasuredMacro is not implemented for non-steady adjoint simulations
static constexpr bool STEADY = true;
static constexpr double MIN_STEP_SIZE = 1e-9;
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
	using State<NSE>::vtk_helper;

	using idx = typename TRAITS::idx;
	using real = typename TRAITS::real;
	using dreal = typename TRAITS::dreal;
	using point_t = typename TRAITS::point_t;
	using lat_t = Lattice<3, real, idx>;

	int resolution = 0;
	bool steady = false;

	StateLocal(const std::string& id, const TNL::MPI::Comm& communicator, lat_t lat)
	: State<NSE>(id, communicator, lat)
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

	bool outputData(const BLOCK& block, int index, int dof, char* desc, idx x, idx y, idx z, real& value, int& dofs) override
	{
		int k = 0;
		if (index == k++)
			return vtk_helper("lbm_density", block.hmacro(MACRO::e_rho, x, y, z), 1, desc, value, dofs);
		if (index == k++) {
			switch (dof) {
				case 0:
					return vtk_helper("velocity", block.hmacro(MACRO::e_vx, x, y, z), 3, desc, value, dofs);
				case 1:
					return vtk_helper("velocity", block.hmacro(MACRO::e_vy, x, y, z), 3, desc, value, dofs);
				case 2:
					return vtk_helper("velocity", block.hmacro(MACRO::e_vz, x, y, z), 3, desc, value, dofs);
			}
		}
		return false;
	}

	void computeAfterLBMKernel() override
	{
		if (! steady) {
			nse.copyMacroToHost();	//! important - macro is stored on device
			const std::string fname = fmt::format("{}/adjoint_data_res{:02d}/macro_primary.bp", DIRECTORY, resolution);
			saveloadMacro(*this, adios2::Mode::Write, fname, steady);
		}
	}
};

template <typename NSE>
struct StateLocalAdjoint : State<NSE>
{
	using TRAITS = typename NSE::TRAITS;
	using BC = typename NSE::BC;
	using MACRO = typename NSE::MACRO;
	using BLOCK = LBM_BLOCK<NSE>;

	using State<NSE>::nse;
	using State<NSE>::vtk_helper;

	using idx = typename TRAITS::idx;
	using real = typename TRAITS::real;
	using dreal = typename TRAITS::dreal;
	using point_t = typename TRAITS::point_t;
	using lat_t = Lattice<3, real, idx>;

	double* lbm_inflow_vx_profile = nullptr;
	double* lbm_inflow_vy_profile = nullptr;
	double* lbm_inflow_vz_profile = nullptr;
	double* lossFunction = nullptr;
	double hide = 0;
	int resolution = 0;
	bool steady = false;

	StateLocalAdjoint(const std::string& id, const TNL::MPI::Comm& communicator, lat_t lat)
	: State<NSE>(id, communicator, lat)
	{}

	void setupBoundaries() override
	{
		// define where the measured data lies - first!
		int cx = floor(hide * nse.lat.global.x());	// domain where measured data are
		for (int px = cx; px <= nse.lat.global.x() - 1; px++)
			for (int pz = 1; pz <= nse.lat.global.z() - 1; pz++)
				for (int py = 1; py <= nse.lat.global.y() - 1; py++)
					nse.setMap(px, py, pz, BC::GEO_ADJOINT_FLUID_m);

		nse.setBoundaryX(0, BC::GEO_ADJOINT_INFLOW_BB_LEFT);  // left

		nse.setBoundaryX(nse.lat.global.x() - 1, BC::GEO_ADJOINT_WALL);	 // right
		nse.setBoundaryZ(0, BC::GEO_ADJOINT_WALL);						 // top
		nse.setBoundaryZ(nse.lat.global.z() - 1, BC::GEO_ADJOINT_WALL);	 // bottom
		nse.setBoundaryY(0, BC::GEO_ADJOINT_WALL);						 // back
		nse.setBoundaryY(nse.lat.global.y() - 1, BC::GEO_ADJOINT_WALL);	 // front
	}

	bool outputData(const BLOCK& block, int index, int dof, char* desc, idx x, idx y, idx z, real& value, int& dofs) override
	{
		int k = 0;
		if (index == k++)
			return vtk_helper("lbm_density", block.hmacro(MACRO::e_rho, x, y, z), 1, desc, value, dofs);
		if (index == k++) {
			switch (dof) {
				case 0:
					return vtk_helper("velocity", block.hmacro(MACRO::e_vx, x, y, z), 3, desc, value, dofs);
				case 1:
					return vtk_helper("velocity", block.hmacro(MACRO::e_vy, x, y, z), 3, desc, value, dofs);
				case 2:
					return vtk_helper("velocity", block.hmacro(MACRO::e_vz, x, y, z), 3, desc, value, dofs);
			}
		}
		if (index == k++)
			return vtk_helper("lbm_density_m", block.hmacro(MACRO::e_rho_m, x, y, z), 1, desc, value, dofs);
		if (index == k++) {
			switch (dof) {
				case 0:
					return vtk_helper("velocity_m", block.hmacro(MACRO::e_vx_m, x, y, z), 3, desc, value, dofs);
				case 1:
					return vtk_helper("velocity_m", block.hmacro(MACRO::e_vy_m, x, y, z), 3, desc, value, dofs);
				case 2:
					return vtk_helper("velocity_m", block.hmacro(MACRO::e_vz_m, x, y, z), 3, desc, value, dofs);
			}
		}
		return false;
	}

	void reset() override
	{
		// first load data from primary and measured
		const std::string fname_p = fmt::format("{}/adjoint_data_res{:02d}/macro_primary.bp", DIRECTORY, resolution);
		const std::string fname_m = fmt::format("{}/adjoint_data_res{:02d}/macro_measured.bp", DIRECTORY, resolution);
		loadPrimaryAndMeasuredMacro(*this, fname_p, fname_m, steady);
		nse.copyMacroToDevice();

		// compute initial DFs on GPU
		this->resetDFs();

		nse.resetMap(NSE::BC::GEO_ADJOINT_FLUID);

		// setup domain geometry after all resets, including setEquilibrium,
		// so it can override the defaults with different initial condition
		setupBoundaries();

		nse.copyMapToDevice();

		// compute initial macroscopic quantities on GPU and copy to CPU
		nse.computeInitialMacro();
		nse.copyMacroToHost();
	}

	void computeAfterLBMKernel() override
	{
		if (! steady) {
			const std::string fname_p = fmt::format("{}/adjoint_data_res{:02d}/macro_primary.bp", DIRECTORY, resolution);
			const std::string fname_m = fmt::format("{}/adjoint_data_res{:02d}/macro_measured.bp", DIRECTORY, resolution);
			loadPrimaryAndMeasuredMacro(*this, fname_p, fname_m, steady);
			nse.copyMacroToDevice();
			for (auto& block : this->nse.blocks) {
				copyVelocityProfile<dreal, idx>(
					block.data.vx_profile_result,
					block.local.y(),
					block.local.z(),
					block.offset.y(),
					block.offset.z(),
					lbm_inflow_vx_profile,
					block.global.y(),
					block.global.z()
				);
				copyVelocityProfile<dreal, idx>(
					block.data.vy_profile_result,
					block.local.y(),
					block.local.z(),
					block.offset.y(),
					block.offset.z(),
					lbm_inflow_vy_profile,
					block.global.y(),
					block.global.z()
				);
				copyVelocityProfile<dreal, idx>(
					block.data.vz_profile_result,
					block.local.y(),
					block.local.z(),
					block.offset.y(),
					block.offset.z(),
					lbm_inflow_vz_profile,
					block.global.y(),
					block.global.z()
				);
				*lossFunction += block.data.loss_function;
			}
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
int simAdjoint(
	double hide,
	double* velocityProfileX,
	double* velocityProfileY,
	double* velocityProfileZ,
	double* lossFunction,
	double eps,
	int RESOLUTION = 1,
	bool print = false
)
{
	using MACRO = typename NSE::MACRO;

	using idx = typename NSE::TRAITS::idx;
	using real = typename NSE::TRAITS::real;
	using dreal = typename NSE::TRAITS::dreal;
	using point_t = typename NSE::TRAITS::point_t;
	using lat_t = Lattice<3, real, idx>;

	int X = BLOCK_SIZE * RESOLUTION;			   // width in pixels
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

	const std::string state_id = fmt::format("sim_adjoint_res{:02d}_np{:03d}", RESOLUTION, TNL::MPI::GetSize(MPI_COMM_WORLD));
	StateLocalAdjoint<NSE> state(state_id, MPI_COMM_WORLD, lat);

	// problem parameters
	state.lbm_inflow_vx_profile = velocityProfileX;
	state.lbm_inflow_vy_profile = velocityProfileY;
	state.lbm_inflow_vz_profile = velocityProfileZ;

	state.lossFunction = lossFunction;
	state.hide = hide;

	state.resolution = RESOLUTION;
	state.steady = STEADY;

	state.nse.physFinalTime = 4.0;

	for (auto& block : state.nse.blocks) {
		//block.data.inflow_vy = 0;
		//block.data.inflow_vz = 0;
		block.data.eps = (dreal) eps;
		block.data.loss_function = 0.0;

		allocateCopyVelocityProfile<dreal, idx>(
			&(block.data.vx_profile),
			block.local.y(),
			block.local.z(),
			block.offset.y(),
			block.offset.z(),
			velocityProfileX,
			block.global.y(),
			block.global.z()
		);
		allocateCopyVelocityProfile<dreal, idx>(
			&(block.data.vy_profile),
			block.local.y(),
			block.local.z(),
			block.offset.y(),
			block.offset.z(),
			velocityProfileY,
			block.global.y(),
			block.global.z()
		);
		allocateCopyVelocityProfile<dreal, idx>(
			&(block.data.vz_profile),
			block.local.y(),
			block.local.z(),
			block.offset.y(),
			block.offset.z(),
			velocityProfileZ,
			block.global.y(),
			block.global.z()
		);
		allocateCopyVelocityProfile<dreal, idx>(
			&(block.data.vx_profile_result),
			block.local.y(),
			block.local.z(),
			block.offset.y(),
			block.offset.z(),
			velocityProfileX,
			block.global.y(),
			block.global.z()
		);
		allocateCopyVelocityProfile<dreal, idx>(
			&(block.data.vy_profile_result),
			block.local.y(),
			block.local.z(),
			block.offset.y(),
			block.offset.z(),
			velocityProfileY,
			block.global.y(),
			block.global.z()
		);
		allocateCopyVelocityProfile<dreal, idx>(
			&(block.data.vz_profile_result),
			block.local.y(),
			block.local.z(),
			block.offset.y(),
			block.offset.z(),
			velocityProfileZ,
			block.global.y(),
			block.global.z()
		);
		// init g*_profile with zeroes
		allocateCopyGradientProfile<dreal, idx>(&(block.data.gx_profile), block.local.y(), block.local.z());
		allocateCopyGradientProfile<dreal, idx>(&(block.data.gy_profile), block.local.y(), block.local.z());
		allocateCopyGradientProfile<dreal, idx>(&(block.data.gz_profile), block.local.y(), block.local.z());
		allocateCopyGradientProfile<bool, idx>(&(block.data.b_profile), block.local.y(), block.local.z());
	}

	if (print) {
		state.cnt[VTK2D].period = state.nse.physFinalTime / 100.0;
		state.add2Dcut_X(0, "cutsX/cut_X");
		state.add2Dcut_X(X / 2, "cutsX2/cut_X");
		state.add2Dcut_Y(Y / 2, "cutsY/cut_Y");
		state.add2Dcut_Z(Z / 2, "cutsZ/cut_Z");

		state.cnt[VTK3D].period = state.nse.physFinalTime / 4.0;
	}

	spdlog::info("eps = {:e}", eps);
	spdlog::info("hide = {:f}", hide);

	execute(state);

	if (state.steady) {
		for (auto& block : state.nse.blocks) {
			copyVelocityProfile<dreal, idx>(
				block.data.vx_profile_result,
				block.local.y(),
				block.local.z(),
				block.offset.y(),
				block.offset.z(),
				state.lbm_inflow_vx_profile,
				block.global.y(),
				block.global.z()
			);
			copyVelocityProfile<dreal, idx>(
				block.data.vy_profile_result,
				block.local.y(),
				block.local.z(),
				block.offset.y(),
				block.offset.z(),
				state.lbm_inflow_vy_profile,
				block.global.y(),
				block.global.z()
			);
			copyVelocityProfile<dreal, idx>(
				block.data.vz_profile_result,
				block.local.y(),
				block.local.z(),
				block.offset.y(),
				block.offset.z(),
				state.lbm_inflow_vz_profile,
				block.global.y(),
				block.global.z()
			);

			// loss function
			state.nse.copyMacroToHost();
			real sum = 0.0;
#pragma omp parallel for collapse(3) reduction(+ : sum)
			for (idx x = state.hide * block.local.x(); x < block.local.x(); x++)	// assume 1 block
				for (idx y = 0; y < block.local.y(); y++)
					for (idx z = 0; z < block.local.z(); z++) {
						const real err_rho = block.hmacro(MACRO::e_rho, x, y, z) - block.hmacro(MACRO::e_rho_m, x, y, z);
						const real err_vx = block.hmacro(MACRO::e_vx, x, y, z) - block.hmacro(MACRO::e_vx_m, x, y, z);
						const real err_vy = block.hmacro(MACRO::e_vy, x, y, z) - block.hmacro(MACRO::e_vy_m, x, y, z);
						const real err_vz = block.hmacro(MACRO::e_vz, x, y, z) - block.hmacro(MACRO::e_vz_m, x, y, z);
						sum += 0.5 * (err_rho * err_rho + err_vx * err_vx + err_vy * err_vy + err_vz * err_vz);
					}
			*lossFunction += sum;
		}
	}

	for (auto& block : state.nse.blocks) {
		deallocateVelocityProfile<dreal>(&(block.data.vx_profile));
		deallocateVelocityProfile<dreal>(&(block.data.vy_profile));
		deallocateVelocityProfile<dreal>(&(block.data.vz_profile));
		deallocateVelocityProfile<dreal>(&(block.data.vx_profile_result));
		deallocateVelocityProfile<dreal>(&(block.data.vy_profile_result));
		deallocateVelocityProfile<dreal>(&(block.data.vz_profile_result));
		deallocateGradientProfile<dreal>(&(block.data.gx_profile));
		deallocateGradientProfile<dreal>(&(block.data.gy_profile));
		deallocateGradientProfile<dreal>(&(block.data.gz_profile));
		deallocateGradientProfile<bool>(&(block.data.b_profile));
	}

	//! remove directories
	//std::string dirname = fmt::format("results_{}", state_id);
	//std::filesystem::remove_all(dirname.c_str());
	// remove primary data for new primary problem run
	const std::string dirname = fmt::format("{}/adjoint_data_res{:02d}/macro_primary.bp", DIRECTORY, RESOLUTION);
	std::filesystem::remove_all(dirname.c_str());

	return 0;
}

template <typename NSE>
int sim(double* velocityProfileX, double* velocityProfileY, double* velocityProfileZ, int RESOLUTION = 1, bool print = false)
{
	using idx = typename NSE::TRAITS::idx;
	using real = typename NSE::TRAITS::real;
	using dreal = typename NSE::TRAITS::dreal;
	using point_t = typename NSE::TRAITS::point_t;
	using lat_t = Lattice<3, real, idx>;

	int X = BLOCK_SIZE * RESOLUTION;			   // width in pixels
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

	const std::string state_id = fmt::format("sim_primary_res{:02d}_np{:03d}", RESOLUTION, TNL::MPI::GetSize(MPI_COMM_WORLD));
	StateLocal<NSE> state(state_id, MPI_COMM_WORLD, lat);

	// problem parameters
	state.resolution = RESOLUTION;
	state.steady = STEADY;

	state.nse.physFinalTime = 16.0;

	for (auto& block : state.nse.blocks) {
		allocateCopyVelocityProfile<dreal, idx>(
			&(block.data.inflow_vx),
			block.local.y(),
			block.local.z(),
			block.offset.y(),
			block.offset.z(),
			velocityProfileX,
			block.global.y(),
			block.global.z()
		);
		allocateCopyVelocityProfile<dreal, idx>(
			&(block.data.inflow_vy),
			block.local.y(),
			block.local.z(),
			block.offset.y(),
			block.offset.z(),
			velocityProfileY,
			block.global.y(),
			block.global.z()
		);
		allocateCopyVelocityProfile<dreal, idx>(
			&(block.data.inflow_vz),
			block.local.y(),
			block.local.z(),
			block.offset.y(),
			block.offset.z(),
			velocityProfileZ,
			block.global.y(),
			block.global.z()
		);
	}

	// add cuts
	if (print) {
		state.cnt[VTK2D].period = state.nse.physFinalTime / 100.0;
		state.add2Dcut_X(0, "cutsX/cut_X");
		state.add2Dcut_X(X / 2, "cutsX2/cut_X");
		state.add2Dcut_Y(Y / 2, "cutsY/cut_Y");
		state.add2Dcut_Z(Z / 2, "cutsZ/cut_Z");

		state.cnt[VTK3D].period = state.nse.physFinalTime / 4.0;
	}

	execute(state);

	if (state.steady) {
		state.nse.copyMacroToHost();  //! important - macro is stored on device
		const std::string fname = fmt::format("{}/adjoint_data_res{:02d}/macro_primary.bp", DIRECTORY, RESOLUTION);
		saveloadMacro(state, adios2::Mode::Write, fname, state.steady);
	}

	for (auto& block : state.nse.blocks) {
		deallocateVelocityProfile<dreal>(&(block.data.inflow_vx));
		deallocateVelocityProfile<dreal>(&(block.data.inflow_vy));
		deallocateVelocityProfile<dreal>(&(block.data.inflow_vz));
	}

	//! remove directories
	//std::string dirname = fmt::format("results_{}", state_id);
	//std::filesystem::remove_all(dirname.c_str());

	return 0;
}

template <typename TRAITS = TraitsDP>
void run(double* velocityProfileX, double* velocityProfileY, double* velocityProfileZ, int RES, bool print)
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

	sim<NSE_CONFIG>(velocityProfileX, velocityProfileY, velocityProfileZ, RES, print);
}

// ADJOINT
template <typename TRAITS = TraitsDP>
void runAdjoint(
	double hide, double* velocityProfileX, double* velocityProfileY, double* velocityProfileZ, double* lossFunction, double eps, int RES, bool print
)
{
	//	using COLL = D3Q27_CUM< TRAITS >;
	using COLL = D3Q27_SRT_ADJOINT<TRAITS, D3Q27_EQ_ADJOINT<TRAITS>>;

	using ADJ_CONFIG = LBM_CONFIG<
		TRAITS,
		D3Q27_KernelStruct_Adjoint,
		NSE_Data_Adjoint<TRAITS>,
		COLL,
		typename COLL::EQ,
		D3Q27_STREAMING<TRAITS>,
		D3Q27_BC_All,
		D3Q27_MACRO_Adjoint<TRAITS>>;

	simAdjoint<ADJ_CONFIG>(hide, velocityProfileX, velocityProfileY, velocityProfileZ, lossFunction, eps, RES, print);
}

void saveLossFunctionToFile(const std::string& dirname, int adjointIteration, double lossFunction)
{
	if (lossFunction != lossFunction) {
		spdlog::error("Loss function is NaN in iteration {}", adjointIteration);
		throw std::runtime_error("Loss function is NaN");
	}

	mkdir_p(dirname.c_str(), 0777);
	const std::string fname = fmt::format("{}/lossFunction.txt", dirname);
	if (adjointIteration == 1) {
		std::ofstream lossFunctionFile(fname);
		std::ostringstream s;
		s << std::setprecision(10) << lossFunction;
		lossFunctionFile << s.str() << "\n";
		lossFunctionFile.close();
		return;
	}
	std::ofstream lossFunctionFile;
	lossFunctionFile.open(fname, std::ofstream::app);
	std::ostringstream s;
	s << std::setprecision(10) << lossFunction;
	lossFunctionFile << s.str() << "\n";
	lossFunctionFile.close();
}

int adjointEpoch(int resolution, double hide, double* guessX, double* guessY, double* guessZ, int iteration, double eps, bool print)
{
	int Y = BLOCK_SIZE * resolution;
	int Z = Y;

	double lossFunction = 0;

	static double prev_lossFunction = DBL_MAX;

	const std::string dirname = fmt::format("{}/adjoint_data_res{:02d}", DIRECTORY, resolution);
	if (iteration > 1) {
		loadVelocityProfile(dirname, guessX, Y, Z, 'X');
		loadVelocityProfile(dirname, guessY, Y, Z, 'Y');
		loadVelocityProfile(dirname, guessZ, Y, Z, 'Z');
	}

	run<TraitsDP>(guessX, guessY, guessZ, resolution, print);
	runAdjoint<TraitsDP>(hide, guessX, guessY, guessZ, &lossFunction, eps, resolution, print);

	if (prev_lossFunction < lossFunction) {
		spdlog::warn("prev loss function = {}, loss function = {}", prev_lossFunction, lossFunction);
		return -1;
	}
	prev_lossFunction = lossFunction;

	saveVelocityProfile(dirname, guessX, Y, Z, 'X');
	saveVelocityProfile(dirname, guessY, Y, Z, 'Y');
	saveVelocityProfile(dirname, guessZ, Y, Z, 'Z');
	saveVelocityProfile_txt(dirname, guessX, Y, Z, 'X');
	saveVelocityProfile_txt(dirname, guessY, Y, Z, 'Y');
	saveVelocityProfile_txt(dirname, guessZ, Y, Z, 'Z');

	saveLossFunctionToFile(dirname, iteration, lossFunction);
	return 0;
}

void adjointFullSim(int resolution, std::size_t epochs, double eps, double hide)
{
	int Y = BLOCK_SIZE * resolution;
	int Z = Y;

	std::unique_ptr<double[]> guessX = initGuess(VelocityProfile::zero, Y, Z);
	std::unique_ptr<double[]> guessY = initGuess(VelocityProfile::zero, Y, Z);
	std::unique_ptr<double[]> guessZ = initGuess(VelocityProfile::zero, Y, Z);

	double step = eps;
	for (std::size_t i = (std::size_t) 1; i <= epochs; i++) {
		//! remove directories
		std::string dirname = fmt::format("results_sim_adjoint_res{:02d}_np{:03d}", resolution, TNL::MPI::GetSize(MPI_COMM_WORLD));
		std::filesystem::remove_all(dirname.c_str());
		dirname = fmt::format("results_sim_primary_res{:02d}_np{:03d}", resolution, TNL::MPI::GetSize(MPI_COMM_WORLD));
		std::filesystem::remove_all(dirname.c_str());

		const bool print = (i == epochs);
		if (adjointEpoch(resolution, hide, guessX.get(), guessY.get(), guessZ.get(), (int) i, step, print) != 0) {
			step /= 2.0;
			spdlog::warn("Loss function increased instead of decreased - halving step size = {}", step);
			if (step < MIN_STEP_SIZE) {
				loadVelocityProfile(dirname, guessX.get(), Y, Z, 'X');
				loadVelocityProfile(dirname, guessY.get(), Y, Z, 'Y');
				loadVelocityProfile(dirname, guessZ.get(), Y, Z, 'Z');
				run(guessX.get(), guessY.get(), guessZ.get(), resolution, true);
				runAdjoint(hide, guessX.get(), guessY.get(), guessZ.get(), &step, eps, resolution, true);
				spdlog::error("Step size too small - exiting");
				return;
			}
		}
	}
}

int main(int argc, char** argv)
{
	TNLMPI_INIT mpi(argc, argv);

	argparse::ArgumentParser program("sim_adjoint");
	program.add_description("Main part of the adjoint-LBM workflow.");
	program.add_argument("--resolution").help("resolution of the lattice").scan<'i', int>().default_value(1);
	program.add_argument("--epochs").help("number of epochs").scan<'i', int>().default_value(1000);
	program.add_argument("--eps").help("epsilon for the inflow velocity profile").scan<'g', double>().default_value(0.001);
	program.add_argument("--hide").help("fraction of hidden measured data (along the x-axis)").scan<'g', double>().default_value(0.2);

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

	const auto epochs = program.get<int>("--epochs");
	if (epochs < 1) {
		fmt::println(stderr, "CLI error: epochs must be at least 1");
		return 1;
	}

	const auto eps = program.get<double>("--eps");
	if (eps < 0.0 || eps > 1.0) {
		fmt::println(stderr, "CLI error: epsilon must be between 0 and 1");
		return 1;
	}

	const auto hide = program.get<double>("--hide");
	if (hide < 0.0 || hide > 1.0) {
		fmt::println(stderr, "CLI error: hidden fraction must be between 0 and 1");
		return 1;
	}

	adjointFullSim(resolution, epochs, eps, hide);

	return 0;
}
