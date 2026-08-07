#ifndef AA_PATTERN
	#define AB_PATTERN
#endif

#if ! defined(USE_POWERLAW) && ! defined(USE_CYMODEL) && ! defined(USE_CASSON)
	#define USE_POWERLAW
#endif

#include <argparse/argparse.hpp>
#include <algorithm>
#include <cmath>
#include <type_traits>
#include <utility>
#include <vector>

#include "lbm3d/core.h"
#include "lbm3d/nonNewtonian.h"

// Data struct for the power-law expansion channel.
// Inherits the inflow velocity profile arrays from NSE_Data_InflowProfile.
template <typename TRAITS>
struct NSE_Data_NonNewtonian_Expansion : NSE_Data_InflowProfile<TRAITS>
{
	using dreal = typename TRAITS::dreal;

	dreal lbm_K = 0;
	dreal lbm_n = 0;
};

// Simulation state for the symmetric 3:1 sudden expansion channel.
// Inlet-driven boundary conditions (no body force), interior wall carving
// for the expansion step, and analytical-solution error evaluation at x=28 m.
template <typename NSE>
struct StateLocal : State<NSE>
{
	using TRAITS = typename NSE::TRAITS;
	using BC = typename NSE::BC;
	using MACRO = typename NSE::MACRO;
	using BLOCK = LBM_BLOCK<NSE>;

	using State<NSE>::checkpoint;
	using State<NSE>::nse;

	using idx = typename TRAITS::idx;
	using idx3d = typename TRAITS::idx3d;
	using real = typename TRAITS::real;
	using dreal = typename TRAITS::dreal;
	using point_t = typename TRAITS::point_t;
	using lat_t = Lattice<3, real, idx>;

	// Analytical solution parameters (expanded section at x=28 m).
	// Set once in sim() before execute().
	dreal exp_n = 0;
	dreal exp_u_max_expanded_lbm = 0;
	dreal exp_R_expanded_lbm = 0;
	dreal exp_z_center_lbm = 0;
	int exp_x_probe = 0;

	// Device array for the inflow velocity profile (inlet section).
	// Indexed as inflow_vx[Y() * z + y] by the kernel's inflow() method.
	TNL::Containers::Array<dreal, DeviceType, idx> inflow_profile;

	int errors_count;
	real* l1errors;
	int error_idx = 0;

	// Cache for analytical solution (depends only on z, called per-cell).
	std::vector<real> analytical_cache;
	bool analytical_cache_ready = false;

	StateLocal(const std::string& id, const TNL::MPI::Comm& communicator, lat_t lat, const std::string& adiosConfigPath = "adios2.xml")
	: State<NSE>(id, communicator, std::move(lat), adiosConfigPath)
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

	// Analytical expanded-section velocity profile at x=28 m (lattice units).
	// Returns 0 outside the expanded channel (|z_rel| >= R_expanded_lbm).
	// Results are cached per-z since the solution depends only on z.
	real analytical_ux(idx lbm_z)
	{
		if (! analytical_cache_ready) {
			int Z = nse.lat.global.z();
			analytical_cache.resize(Z);
			for (int z = 0; z < Z; z++)
				analytical_cache[z] = compute_analytical_ux(z);
			analytical_cache_ready = true;
		}
		if (lbm_z < 0 || lbm_z >= (idx) analytical_cache.size())
			return 0;
		return analytical_cache[lbm_z];
	}

	// Model-specific analytical solution (uncached).
	// u_expanded(z) = u_max_expanded_lbm * (1 - (|z_rel|/R_expanded_lbm)^(1+1/n))
	real compute_analytical_ux(idx lbm_z)
	{
		real z_rel = std::abs((real) lbm_z - (real) exp_z_center_lbm);
		if (z_rel >= (real) exp_R_expanded_lbm)
			return 0;

		real n = (real) exp_n;
		real exponent = 1.0 + 1.0 / n;
		return (real) exp_u_max_expanded_lbm * (1.0 - std::pow(z_rel / (real) exp_R_expanded_lbm, exponent));
	}

	void setupBoundaries() override
	{
		// X: inlet ghost, inflow, outflow, outlet ghost
		nse.setBoundaryX(0, BC::GEO_NOTHING);
		nse.setBoundaryX(1, BC::GEO_INFLOW_LEFT);
		nse.setBoundaryX(nse.lat.global.x() - 2, BC::GEO_OUTFLOW_RIGHT_INTERP);
		nse.setBoundaryX(nse.lat.global.x() - 1, BC::GEO_NOTHING);

		// Y: periodic (pseudo-2D)
		nse.setBoundaryY(0, BC::GEO_PERIODIC);
		nse.setBoundaryY(nse.lat.global.y() - 1, BC::GEO_PERIODIC);

		// Z: ghost, wall, ..., wall, ghost
		nse.setBoundaryZ(0, BC::GEO_NOTHING);
		nse.setBoundaryZ(1, BC::GEO_WALL);
		nse.setBoundaryZ(nse.lat.global.z() - 2, BC::GEO_WALL);
		nse.setBoundaryZ(nse.lat.global.z() - 1, BC::GEO_NOTHING);

		// Expansion step: carve interior walls in the inlet section.
		// At x=1 (inflow plane), carve GEO_WALL for |z_phys| > 0.5 m so the
		// inflow BC is only 1 m tall (the inlet-section height), not the full
		// 3 m expanded height. Cells with |z_phys| <= 0.5 remain GEO_INFLOW_LEFT.
		// For x in [2, x_step], cells with |z_phys| > 0.5 m are walls.
		real PHYS_DL = nse.lat.physDl;
		int Z = nse.lat.global.z();
		real z_center_lbm = ((real) Z - 1.0) / 2.0;
		int x_step = 1 + (int) std::floor(5.0 / PHYS_DL);
		// x=1: carve inlet-section walls on the inflow plane
		for (int y = 1; y < nse.lat.global.y() - 1; y++)
			for (int z = 1; z < nse.lat.global.z() - 1; z++) {
				real z_phys = ((real) z - z_center_lbm) * PHYS_DL;
				if (std::abs(z_phys) > 0.5)
					nse.setMap(1, y, z, BC::GEO_WALL);
			}
		// x in [2, x_step]: carve inlet-section walls in the interior
		for (int x = 2; x <= x_step; x++)
			for (int y = 1; y < nse.lat.global.y() - 1; y++)
				for (int z = 1; z < nse.lat.global.z() - 1; z++) {
					real z_phys = ((real) z - z_center_lbm) * PHYS_DL;
					if (std::abs(z_phys) > 0.5)
						nse.setMap(x, y, z, BC::GEO_WALL);
				}
	}

	[[nodiscard]] std::vector<std::string> getOutputDataNames() const override
	{
		return {"lbm_density",	  "lbm_density_fluctuation",
				"lbm_velocity_x", "lbm_velocity_y",
				"lbm_velocity_z", "lbm_force_x",
				"lbm_force_y",	  "lbm_force_z",
				"lbm_S11",		  "lbm_S12",
				"lbm_S13",		  "lbm_S22",
				"lbm_S32",		  "lbm_S33",
				"velocity_x",	  "velocity_y",
				"velocity_z",	  "lbm_analytical_ux",
				"lbm_error_ux",	  "lbm_error_uy",
				"lbm_error_uz",	  "analytical_ux",
				"error_ux",		  "error_uy",
				"error_uz"};
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
		writer.write("lbm_force_x", getMacroView<TRAITS>(block.hmacro, MACRO::e_fx), begin, end);
		writer.write("lbm_force_y", getMacroView<TRAITS>(block.hmacro, MACRO::e_fy), begin, end);
		writer.write("lbm_force_z", getMacroView<TRAITS>(block.hmacro, MACRO::e_fz), begin, end);
		writer.write("lbm_S11", getMacroView<TRAITS>(block.hmacro, MACRO::e_S11), begin, end);
		writer.write("lbm_S12", getMacroView<TRAITS>(block.hmacro, MACRO::e_S12), begin, end);
		writer.write("lbm_S13", getMacroView<TRAITS>(block.hmacro, MACRO::e_S13), begin, end);
		writer.write("lbm_S22", getMacroView<TRAITS>(block.hmacro, MACRO::e_S22), begin, end);
		writer.write("lbm_S32", getMacroView<TRAITS>(block.hmacro, MACRO::e_S32), begin, end);
		writer.write("lbm_S33", getMacroView<TRAITS>(block.hmacro, MACRO::e_S33), begin, end);
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
			"lbm_analytical_ux",
			[&](idx x, idx y, idx z) -> dreal
			{
				return analytical_ux(z);
			},
			begin,
			end
		);
		writer.write(
			"lbm_error_ux",
			[&](idx x, idx y, idx z) -> dreal
			{
				return TNL::abs(block.hmacro(MACRO::e_vx, x, y, z) - analytical_ux(z));
			},
			begin,
			end
		);
		writer.write(
			"lbm_error_uy",
			[&](idx x, idx y, idx z) -> dreal
			{
				return TNL::abs(block.hmacro(MACRO::e_vy, x, y, z));
			},
			begin,
			end
		);
		writer.write(
			"lbm_error_uz",
			[&](idx x, idx y, idx z) -> dreal
			{
				return TNL::abs(block.hmacro(MACRO::e_vz, x, y, z));
			},
			begin,
			end
		);
		writer.write(
			"analytical_ux",
			[&](idx x, idx y, idx z) -> dreal
			{
				return nse.lat.lbm2physVelocity(analytical_ux(z));
			},
			begin,
			end
		);
		writer.write(
			"error_ux",
			[&](idx x, idx y, idx z) -> dreal
			{
				return nse.lat.lbm2physVelocity(TNL::abs(block.hmacro(MACRO::e_vx, x, y, z) - analytical_ux(z)));
			},
			begin,
			end
		);
		writer.write(
			"error_uy",
			[&](idx x, idx y, idx z) -> dreal
			{
				return nse.lat.lbm2physVelocity(TNL::abs(block.hmacro(MACRO::e_vy, x, y, z)));
			},
			begin,
			end
		);
		writer.write(
			"error_uz",
			[&](idx x, idx y, idx z) -> dreal
			{
				return nse.lat.lbm2physVelocity(TNL::abs(block.hmacro(MACRO::e_vz, x, y, z)));
			},
			begin,
			end
		);
	}

	void probe1() override
	{
		// Compute L1 and L2 errors against the analytical expanded-section
		// solution at x_probe only (the comparison plane at x=28 m).
		// Errors are reported as average (mean) error in physical velocity
		// units so they can be compared directly against V_mean_expanded.
		auto& block = nse.blocks.front();
		real local_l1sum_ux = 0;
		real local_l1sum_uy = 0;
		real local_l1sum_uz = 0;
		real local_l2sum_ux = 0;
		real local_l2sum_uy = 0;
		real local_l2sum_uz = 0;
		int local_count = 0;

		// Check if x_probe falls within this block's interior x-range.
		int x_start = block.offset.x() + 1;
		int x_end = block.offset.x() + block.local.x() - 1;

		if (exp_x_probe >= x_start && exp_x_probe < x_end) {
			int i = exp_x_probe;
			for (int j = block.offset.y() + 1; j < block.offset.y() + block.local.y() - 1; j++)
				for (int k = block.offset.z() + 1; k < block.offset.z() + block.local.z() - 1; k++) {
					auto gi = block.hmap(i, j, k);
					if (! (NSE::BC::isFluid(gi) || NSE::BC::isPeriodic(gi)))
						continue;
					real an_ux = analytical_ux(k);
					real diff_ux = fabs(block.hmacro(MACRO::e_vx, i, j, k) - an_ux);
					real diff_uy = fabs(block.hmacro(MACRO::e_vy, i, j, k));
					real diff_uz = fabs(block.hmacro(MACRO::e_vz, i, j, k));
					local_l1sum_ux += diff_ux;
					local_l1sum_uy += diff_uy;
					local_l1sum_uz += diff_uz;
					local_l2sum_ux += TNL::sqr(diff_ux);
					local_l2sum_uy += TNL::sqr(diff_uy);
					local_l2sum_uz += TNL::sqr(diff_uz);
					local_count++;
				}
		}

		// MPI reduction
		real l1sum_ux = TNL::MPI::reduce(local_l1sum_ux, MPI_SUM, MPI_COMM_WORLD);
		real l1sum_uy = TNL::MPI::reduce(local_l1sum_uy, MPI_SUM, MPI_COMM_WORLD);
		real l1sum_uz = TNL::MPI::reduce(local_l1sum_uz, MPI_SUM, MPI_COMM_WORLD);
		real l2sum_ux = TNL::MPI::reduce(local_l2sum_ux, MPI_SUM, MPI_COMM_WORLD);
		real l2sum_uy = TNL::MPI::reduce(local_l2sum_uy, MPI_SUM, MPI_COMM_WORLD);
		real l2sum_uz = TNL::MPI::reduce(local_l2sum_uz, MPI_SUM, MPI_COMM_WORLD);
		int total_count = TNL::MPI::reduce(local_count, MPI_SUM, MPI_COMM_WORLD);
		if (total_count == 0)
			total_count = 1;

		// Convert to physical units as mean error (velocity units)
		real inv_count = 1.0 / (real) total_count;
		auto to_phys = [&](real l1, real l2) -> std::pair<real, real>
		{
			real l1p = nse.lat.lbm2physVelocity(l1 * inv_count);
			real l2p = nse.lat.lbm2physVelocity(sqrt(l2 * inv_count));
			return {l1p, l2p};
		};
		auto [l1error_phys_ux, l2error_phys_ux] = to_phys(l1sum_ux, l2sum_ux);
		auto [l1error_phys_uy, l2error_phys_uy] = to_phys(l1sum_uy, l2sum_uy);
		auto [l1error_phys_uz, l2error_phys_uz] = to_phys(l1sum_uz, l2sum_uz);

		// Dynamic stopping criterion (based on ux error, the primary component)
		real l1error_phys = l1error_phys_ux;
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
				"at t={:1.6f}s, iterations={:d} l1error_phys_u=[{:e},{:e},{:e}] l2error_phys_u=[{:e},{:e},{:e}] "
				"stopping={:e} stopping_stddev={:e}",
				nse.physTime(),
				nse.iterations,
				l1error_phys_ux,
				l1error_phys_uy,
				l1error_phys_uz,
				l2error_phys_ux,
				l2error_phys_uy,
				l2error_phys_uz,
				stopping,
				stopping_stddev
			);
	}

	void updateKernelData() override
	{
		State<NSE>::updateKernelData();
		for (auto& block : nse.blocks) {
			block.data.fx = 0;
			block.data.fy = 0;
			block.data.fz = 0;
		}
	}

	void computeBeforeLBMKernel() override
	{
		computeNonNewtonianKernels(*this);
	}
};

// Simulation setup
template <typename NSE>
void sim(const std::string& adios_config, int RESOLUTION, double lbm_viscosity, double V, double n, double K_phys)
{
	using idx = typename NSE::TRAITS::idx;
	using real = typename NSE::TRAITS::real;
	using dreal = typename NSE::TRAITS::dreal;
	using point_t = typename NSE::TRAITS::point_t;
	using lat_t = Lattice<3, real, idx>;

	int Z = 30 * RESOLUTION + 4;
	real PHYS_HEIGHT = 3.0;
	real PHYS_DL = PHYS_HEIGHT / ((real) Z - 4);
	int X = std::round(30.0 / PHYS_DL) + 3;
	// Spanwise (Y) direction: 1 m physical width with periodic BCs.
	// Periodic faces at y=0 and y=Y-1; interior cells y=1..Y-2 span 1 m.
	int Y = std::max(4, (int) std::round(1.0 / PHYS_DL) + 2);

	const real rho = 1.0;
	real c_s = 1.0 / std::sqrt(3.0);

	// Inlet section parameters
	real R_inlet_phys = 0.5;
	real R_inlet_lbm = R_inlet_phys / PHYS_DL;
	real u_max_inlet = V * (2.0 * n + 1.0) / (n + 1.0);

	// Unit conversions (inlet scales, diffusive scaling)
	real gamma_ref = u_max_inlet / R_inlet_phys;
	real nu_ref = (K_phys / rho) * std::pow(gamma_ref, n - 1.0);
	real PHYS_DT = lbm_viscosity * PHYS_DL * PHYS_DL / nu_ref;
	real K_lbm = (K_phys / rho) * std::pow(PHYS_DT, 2.0 - n) / (PHYS_DL * PHYS_DL);
	real u_max_inlet_lbm = u_max_inlet * PHYS_DT / PHYS_DL;
	real Ma = u_max_inlet_lbm / c_s;
	real Re = rho * std::pow(V, 2.0 - n) * std::pow(1.0, n) / K_phys;
	real PHYS_VISCOSITY = nu_ref;

	if (Ma > 0.3)
		spdlog::warn("Mach number {:.4f} exceeds 0.3, simulation may be unstable", Ma);

	// Expanded-section analytical profile parameters (at x=28 m)
	real V_mean_expanded = V / 3.0;
	real u_max_expanded = V_mean_expanded * (2.0 * n + 1.0) / (n + 1.0);
	real R_expanded_phys = 1.5;
	real R_expanded_lbm = R_expanded_phys / PHYS_DL;
	real u_max_expanded_lbm = u_max_expanded * PHYS_DT / PHYS_DL;
	real z_center_lbm = ((real) Z - 1.0) / 2.0;
	int x_probe = 1 + (int) std::round(28.0 / PHYS_DL);

	spdlog::info(
		"Power-law expansion: n={}, K_phys={:.4e} Pa·s^n, K_lbm={:.4e}, nu_ref={:.4e} m²/s, gamma_ref={:.4e} 1/s, "
		"u_max_inlet={:.6e} m/s, u_max_inlet_lbm={:.6e}, Ma={:.6f}, omega={:.4f}, Re={:.2f}",
		n,
		K_phys,
		K_lbm,
		nu_ref,
		gamma_ref,
		u_max_inlet,
		u_max_inlet_lbm,
		Ma,
		1.0 / (3.0 * lbm_viscosity + 0.5),
		Re
	);
	spdlog::info(
		"Expanded section: V_mean_expanded={:.6e} m/s, u_max_expanded={:.6e} m/s, u_max_expanded_lbm={:.6e}, "
		"R_expanded_lbm={:.4f}, z_center_lbm={:.4f}, x_probe={}",
		V_mean_expanded,
		u_max_expanded,
		u_max_expanded_lbm,
		R_expanded_lbm,
		z_center_lbm,
		x_probe
	);

	std::string state_id = fmt::format(
		"sim_nonnewtonian_expansion_{}/n={}_K={:.2e}_nu={:.2e}_vmean={:.6e}_res={:02d}_np={:03d}",
		std::is_same_v<dreal, float> ? "SP" : "DP",
		n,
		K_phys,
		lbm_viscosity,
		V,
		RESOLUTION,
		TNL::MPI::GetSize(MPI_COMM_WORLD)
	);

	point_t PHYS_ORIGIN = {-0.5 * PHYS_DL, -0.5 * PHYS_DL, (0.5 - z_center_lbm) * PHYS_DL};

	lat_t lat;
	lat.global = typename lat_t::CoordinatesType(X, Y, Z);
	lat.physOrigin = PHYS_ORIGIN;
	lat.physDl = PHYS_DL;
	lat.physDt = PHYS_DT;
	lat.physViscosity = PHYS_VISCOSITY;

	StateLocal<NSE> state(state_id, MPI_COMM_WORLD, lat, adios_config);

	if (! state.canCompute())
		return;

	// Set analytical solution parameters (needed by probe1 and outputData).
	state.exp_n = (dreal) n;
	state.exp_u_max_expanded_lbm = (dreal) u_max_expanded_lbm;
	state.exp_R_expanded_lbm = (dreal) R_expanded_lbm;
	state.exp_z_center_lbm = (dreal) z_center_lbm;
	state.exp_x_probe = x_probe;

	// Set model-specific block data (needed by both the analytical solution
	// and the non-Newtonian kernel).
	// lbmViscosity is set here because the analytical solution is evaluated
	// during SimInit (iteration 0), before State::updateKernelData() runs.
	for (auto& block : state.nse.blocks) {
		block.data.lbmViscosity = lbm_viscosity;
		block.data.lbm_K = K_lbm;
		block.data.lbm_n = n;
	}

	// Set up the inflow velocity profile (inlet section power-law profile).
	// u_inlet_lbm(z) = u_max_inlet_lbm * (1 - (|z - z_center|/R_inlet_lbm)^(1+1/n))
	{
		auto& block = state.nse.blocks.front();
		int local_y = block.local.y();
		int local_z = block.local.z();
		state.inflow_profile.setSize(local_y * local_z);
		block.data.inflow_vx = state.inflow_profile.getData();
		block.data.inflow_vy = nullptr;
		block.data.inflow_vz = nullptr;

		std::vector<dreal> profile(local_y * local_z);
		real exponent = 1.0 + 1.0 / n;
		for (int j = 0; j < local_y; j++)
			for (int k = 0; k < local_z; k++) {
				real z_global = (real) (block.offset.z() + k);
				real z_rel = std::abs(z_global - z_center_lbm);
				real u_lbm;
				if (z_rel >= R_inlet_lbm)
					u_lbm = 0;
				else
					u_lbm = u_max_inlet_lbm * (1.0 - std::pow(z_rel / R_inlet_lbm, exponent));
				profile[k * local_y + j] = (dreal) u_lbm;
			}
#ifdef USE_CUDA
		TNL::Backend::memcpy(block.data.inflow_vx, profile.data(), local_y * local_z * sizeof(dreal), TNL::Backend::MemcpyHostToDevice);
#else
		std::copy(profile.begin(), profile.end(), block.data.inflow_vx);
#endif
	}

	// Steady state time (physical) = H^2/(4*nu) with expanded height.
	// physDt shrinks as 1/N^2, so physFinalTime is resolution-independent.
	// Use 10x steady state physical time as safety limit.
	real t_steady = (PHYS_HEIGHT * PHYS_HEIGHT) / (4.0 * PHYS_VISCOSITY);
	state.nse.physFinalTime = 10.0 * t_steady;
	state.cnt[PRINT].period = t_steady;
	state.cnt[PROBE1].period = 0.1 * t_steady;

	state.cnt[OUT3D].period = t_steady;

	execute(state);
}

template <typename TRAITS = TraitsSP>
void run(const std::string& adios_config, int resolution, double lbm_viscosity, double V, double n, double K_phys)
{
	using COLL = D3Q27_CUM<TRAITS, D3Q27_EQ<TRAITS>>;

	using NSE_CONFIG = LBM_CONFIG<
		TRAITS,
		D3Q27_KernelStruct,
		NSE_Data_NonNewtonian_Expansion<TRAITS>,
		COLL,
		typename COLL::EQ,
		D3Q27_STREAMING<TRAITS>,
		D3Q27_BC_All,
		MacroNonNewtonianDefault<TRAITS>>;

	sim<NSE_CONFIG>(adios_config, resolution, lbm_viscosity, V, n, K_phys);
}

int main(int argc, char** argv)
{
	TNLMPI_INIT mpi(argc, argv);

	argparse::ArgumentParser program("sim_nonnewtonian_expansion");
	program.add_description("Non-Newtonian power-law flow through a symmetric 3:1 sudden expansion channel.");

	program.add_argument("--adios-config").help("path to ADIOS2 configuration file").default_value(std::string("adios2.xml")).nargs(1);
	program.add_argument("--resolution").help("resolution of the lattice").scan<'i', int>().default_value(1).nargs(1);
	program.add_argument("--lbm-viscosity")
		.help("Reference lattice viscosity. Must be in range (0, 1/6].")
		.scan<'g', double>()
		.default_value(0.05)
		.nargs(1);
	program.add_argument("--v-mean").help("inlet mean velocity [m/s]").scan<'g', double>().default_value(0.5).nargs(1);
	program.add_argument("--n")
		.help("power-law index (1=Newtonian, <1=shear-thinning, >1=shear-thickening)")
		.scan<'g', double>()
		.default_value(1.0)
		.nargs(1);
	program.add_argument("--K").help("Power-law consistency index (Pa·s^n)").scan<'g', double>().default_value(0.0125).nargs(1);
	program.add_argument("--precision")
		.help("precision for numerical operations: single=32-bit (float), double=64-bit")
		.default_value(std::string("single"))
		.nargs(1);

	try {
		program.parse_args(argc, argv);
	}
	catch (const std::exception& err) {
		fmt::print(stderr, "Error: {}\n", err.what());
		std::exit(1);
	}

	const auto adios_config = program.get<std::string>("--adios-config");
	const int resolution = program.get<int>("--resolution");
	const double lbm_viscosity = program.get<double>("--lbm-viscosity");
	const double V = program.get<double>("--v-mean");
	const double n = program.get<double>("--n");
	const double K_phys = program.get<double>("--K");
	const auto precision = program.get<std::string>("--precision");

	if (resolution < 1) {
		fmt::println(stderr, "CLI error: resolution must be at least 1");
		return 1;
	}
	if (lbm_viscosity <= 0.0 || lbm_viscosity > 1. / 6.) {
		fmt::println(stderr, "CLI error: --lbm-viscosity must be in range (0, 1/6]");
		return 1;
	}
	if (V <= 0.0) {
		fmt::println(stderr, "CLI error: --v-mean must be positive");
		return 1;
	}
	if (n <= 0.0) {
		fmt::println(stderr, "CLI error: --n must be positive");
		return 1;
	}
	if (K_phys <= 0.0) {
		fmt::println(stderr, "CLI error: --K must be positive");
		return 1;
	}
	if (precision != "single" && precision != "double") {
		fmt::println(stderr, "CLI error: --precision must be 'single' or 'double'");
		return 1;
	}

	if (precision == "double")
		run<TraitsDP>(adios_config, resolution, lbm_viscosity, V, n, K_phys);
	else
		run<TraitsSP>(adios_config, resolution, lbm_viscosity, V, n, K_phys);

	return 0;
}
