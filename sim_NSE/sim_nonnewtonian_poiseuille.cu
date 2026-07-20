#ifndef AA_PATTERN
	#define AB_PATTERN
#endif

#if ! defined(USE_POWERLAW) && ! defined(USE_CYMODEL) && ! defined(USE_CASSON)
	#define USE_POWERLAW
#endif

#ifndef SIM_NONNEWTONIAN_POISEUILLE_STATE_SUFFIX
	#define SIM_NONNEWTONIAN_POISEUILLE_STATE_SUFFIX ""
#endif

#include <argparse/argparse.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

#include "lbm3d/core.h"
#include "lbm3d/nonNewtonian.h"

// Numerical helpers for non-Newtonian Poiseuille analytical solution.
//
// General solution: W(s) = R*u_R - s*u_0 + (F(u_0) - F(u_R))/A
// where f(u) = eta(u)*u = A*|x3|,  u_pos = solve_monotone(f, A*pos),
// F(u) = integral_0^u eta(t)*t dt  (constitutive potential).

// Bisection solver for f(u) = rhs, f monotone increasing on [0, inf).
// Handles both f(0) = 0 (standard case) and f(0) > 0 (yield-stress case):
// if f(0) > rhs, returns 0 (plug region for Casson).
template <typename Fn>
static double solve_monotone(Fn f, double rhs)
{
	if (rhs <= 0)
		return 0;
	if (f(0.0) >= rhs)
		return 0;
	double lo = 0;
	double hi = std::max(rhs, 1.0);
	while (f(hi) < rhs)
		hi *= 2;
	for (int i = 0; i < 100; i++) {
		double mid = lo + (hi - lo) * 0.5;
		if (f(mid) < rhs)
			lo = mid;
		else
			hi = mid;
	}
	return lo + (hi - lo) * 0.5;
}

// Composite Simpson's rule for integral_0^u g(t) dt.
template <typename Fn>
static double simpson_integral(Fn g, double u, int N = 500)
{
	if (u <= 0)
		return 0;
	double h = u / N;
	double sum = g(0.0) + g(u);
	for (int i = 1; i < N; i++) {
		double t = i * h;
		sum += (i % 2 == 0 ? 2.0 : 4.0) * g(t);
	}
	return sum * h / 3.0;
}

// General non-Newtonian plane Poiseuille velocity W at |z_rel| = s.
template <typename FluxFn, typename PotFn>
static double poiseuille_W(double s, double R, double A, FluxFn flux, PotFn pot)
{
	double u_0 = solve_monotone(flux, A * s);
	double u_R = solve_monotone(flux, A * R);
	return R * u_R - s * u_0 + (pot(u_0) - pot(u_R)) / A;
}

// Shared constitutive-law closures for CY and Casson models.
//
// The LBM kernel (nonNewtonian.h) computes the lattice shear rate
//   gamma_lbm = sqrt(2 * D:D)
// where D is the rate-of-strain tensor with components S11, S22, S33, S12, S13, S32,
// all computed from lattice-unit velocities and lattice-unit distances.
// For plane Poiseuille (only S13 = W'/2 != 0) this gives gamma_lbm = |W'|.
//
// Each struct below provides flux(u) = nu(u)*u and pot(u) = integral_0^u nu(t)*t dt
// using the lattice shear rate u = |W'| (in lattice units). Both compute_analytical_ux()
// and the sim() driving-force root-find use these shared definitions, so the analytical
// reference and the force solver cannot drift.
//
// All parameters passed to these structs are in LATTICE UNITS (kinematic viscosities,
// dimensionless shear rates). The physical-to-lattice conversion happens in sim().

#if defined(USE_CYMODEL)
struct CY_Constitutive
{
	double nu_inf, nu_0, lambda, a, n;

	// nu(u) = nu_inf + (nu_0 - nu_inf) * (1 + (lambda*u)^a)^((n-1)/a)
	// where u = |W'| is the lattice shear rate, matching the kernel's gamma_lbm.
	// nu_0 is the zero-shear lattice viscosity (also the collision reference),
	// nu_inf is the infinite-shear lattice viscosity.
	double nu(double u) const
	{
		return nu_inf + (nu_0 - nu_inf) * std::pow(1.0 + std::pow(lambda * u, a), (n - 1.0) / a);
	}
	double flux(double u) const
	{
		return nu(u) * u;
	}
	double pot(double u) const
	{
		return simpson_integral(
			[&](double t)
			{
				return nu(t) * t;
			},
			u
		);
	}
};
#elif defined(USE_CASSON)
struct Casson_Constitutive
{
	double k0, k1;

	// Standard Casson: nu(u) = (k0 + k1*sqrt(u))^2 / u
	// where u = |W'| is the lattice shear rate, matching the kernel's gamma_lbm.
	// Yield stress tau_y_lbm = k0^2, plastic viscosity nu_C_lbm = k1^2.
	//
	// flux(u) = nu(u) * u = (k0 + k1*sqrt(u))^2
	// pot(u) = integral_0^u (k0 + k1*sqrt(t))^2 dt
	//        = integral_0^u (k0^2 + 2*k0*k1*sqrt(t) + k1^2*t) dt
	//        = k0^2*u + (4/3)*k0*k1*u^(3/2) + (1/2)*k1^2*u^2

	double nu(double u) const
	{
		if (u <= 0.0)
			return std::numeric_limits<double>::infinity();
		return (k0 + k1 * std::sqrt(u)) * (k0 + k1 * std::sqrt(u)) / u;
	}
	double flux(double u) const
	{
		return (k0 + k1 * std::sqrt(u)) * (k0 + k1 * std::sqrt(u));
	}
	double pot(double u) const
	{
		double su = std::sqrt(u);
		return k0 * k0 * u + (4.0 / 3.0) * k0 * k1 * u * su + 0.5 * k1 * k1 * u * u;
	}
};
#endif

// Data struct
template <typename TRAITS>
struct NSE_Data_NonNewtonian_Poiseuille : NSE_Data_InflowProfile<TRAITS>
{
	using dreal = typename TRAITS::dreal;

#if defined(USE_POWERLAW)
	dreal lbm_K = 0;
	dreal lbm_n = 0;
#elif defined(USE_CYMODEL)
	dreal lbm_nu_inf = 0;
	dreal lbm_lambda = 0;
	dreal lbm_a = 0;
	dreal lbm_n = 0;
#elif defined(USE_CASSON)
	dreal lbm_k0 = 0;
	dreal lbm_k1 = 0;
#endif
};

// Simulation state
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

	dreal driving_force = 0;
	bool use_forcing = true;

	// Device array for the inflow velocity profile (pressure-driven mode).
	// Indexed as inflow_vx[Y()*z + y] by the kernel's inflow() method.
	TNL::Containers::Array<dreal, DeviceType, idx> inflow_profile;

	int errors_count;
	real* l1errors;
	int error_idx = 0;

	// Cache for analytical solution (depends only on z, but called per-cell).
	std::vector<real> analytical_cache;
	bool analytical_cache_ready = false;
	dreal cached_driving_force = -1;

	StateLocal(
		const std::string& id, const TNL::MPI::Comm& communicator, lat_t lat, bool use_forcing, const std::string& adiosConfigPath = "adios2.xml"
	)
	: State<NSE>(id, communicator, std::move(lat), adiosConfigPath)
	{
		this->use_forcing = use_forcing;
		errors_count = 10;
		l1errors = new real[errors_count];
		for (int i = 0; i < errors_count; i++)
			l1errors[i] = 1;
	}

	~StateLocal() override
	{
		delete[] l1errors;
	}

	// Exact analytical Poiseuille solution W(z).
	// Returns 0 outside the channel (wall and ghost cells).
	// Results are cached per-z since the solution depends only on z.
	real analytical_ux(idx lbm_z)
	{
		if (! analytical_cache_ready || cached_driving_force != driving_force) {
			int Z = nse.lat.global.z();
			analytical_cache.resize(Z);
			for (int z = 0; z < Z; z++)
				analytical_cache[z] = compute_analytical_ux(z);
			analytical_cache_ready = true;
			cached_driving_force = driving_force;
		}
		if (lbm_z < 0 || lbm_z >= (idx) analytical_cache.size())
			return 0;
		return analytical_cache[lbm_z];
	}

	// Model-specific analytical solution (uncached).
	real compute_analytical_ux(idx lbm_z)
	{
		real z_wall_low = 1.5;
		real z_wall_high = nse.lat.global.z() - 2.5;
		real R = (z_wall_high - z_wall_low) / 2.0;
		real z_c = (z_wall_high + z_wall_low) / 2.0;
		real z_rel = (real) lbm_z - z_c;

		if (std::abs(z_rel) >= R)
			return 0;

		real A = (real) driving_force;
		real s = std::abs(z_rel);

#if defined(USE_POWERLAW)
		// W(z) = (A/K)^(1/n) * n/(n+1) * (R^((n+1)/n) - |z|^((n+1)/n))
		real K = (real) nse.blocks.front().data.lbm_K;
		real n = (real) nse.blocks.front().data.lbm_n;

		real exponent = (n + 1.0) / n;
		real coeff = std::pow(A / K, 1.0 / n) * n / (n + 1.0);
		return coeff * (std::pow(R, exponent) - std::pow(s, exponent));

#elif defined(USE_CYMODEL)
		// W = R*u_R - s*u_0 + (F(u_0) - F(u_R))/A
		// CY constitutive with lattice shear rate u = |W'|, matching the kernel's gamma_lbm.
		double nu_inf = nse.blocks.front().data.lbm_nu_inf;
		double nu_0 = nse.blocks.front().data.lbmViscosity;
		double lambda = nse.blocks.front().data.lbm_lambda;
		double a = nse.blocks.front().data.lbm_a;
		double n = nse.blocks.front().data.lbm_n;

		CY_Constitutive cy{nu_inf, nu_0, lambda, a, n};
		return (real) poiseuille_W(
			(double) s,
			(double) R,
			(double) A,
			[&](double u)
			{
				return cy.flux(u);
			},
			[&](double u)
			{
				return cy.pot(u);
			}
		);

#elif defined(USE_CASSON)
		// W = R*u_R - s*u_0 + (F(u_0) - F(u_R))/A
		// Casson constitutive with lattice shear rate u = |W'|, matching the kernel's gamma_lbm.
		double k0 = (double) nse.blocks.front().data.lbm_k0;
		double k1 = (double) nse.blocks.front().data.lbm_k1;

		Casson_Constitutive cas{k0, k1};
		return (real) poiseuille_W(
			(double) s,
			(double) R,
			(double) A,
			[&](double u)
			{
				return cas.flux(u);
			},
			[&](double u)
			{
				return cas.pot(u);
			}
		);
#endif
	}

	void setupBoundaries() override
	{
		nse.setBoundaryY(0, BC::GEO_PERIODIC);
		nse.setBoundaryY(nse.lat.global.y() - 1, BC::GEO_PERIODIC);

		if (use_forcing) {
			nse.setBoundaryX(0, BC::GEO_PERIODIC);
			nse.setBoundaryX(nse.lat.global.x() - 1, BC::GEO_PERIODIC);
		}
		else {
			nse.setBoundaryX(0, BC::GEO_NOTHING);
			nse.setBoundaryX(1, BC::GEO_INFLOW_LEFT);
			nse.setBoundaryX(nse.lat.global.x() - 2, BC::GEO_OUTFLOW_RIGHT_INTERP);
			nse.setBoundaryX(nse.lat.global.x() - 1, BC::GEO_NOTHING);
		}

		nse.setBoundaryZ(0, BC::GEO_NOTHING);
		nse.setBoundaryZ(1, BC::GEO_WALL);
		nse.setBoundaryZ(nse.lat.global.z() - 2, BC::GEO_WALL);
		nse.setBoundaryZ(nse.lat.global.z() - 1, BC::GEO_NOTHING);
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
		// compute L1 and L2 errors against the analytical solution
		// (skip non-fluid and non-periodic sites — only count interior fluid cells)
		auto& block = nse.blocks.front();
		real local_l1sum_ux = 0;
		real local_l1sum_uy = 0;
		real local_l1sum_uz = 0;
		real local_l2sum_ux = 0;
		real local_l2sum_uy = 0;
		real local_l2sum_uz = 0;
		for (int i = block.offset.x() + 1; i < block.offset.x() + block.local.x() - 1; i++)
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
				}

		// MPI reduction
		real l1sum_ux = TNL::MPI::reduce(local_l1sum_ux, MPI_SUM, MPI_COMM_WORLD);
		real l1sum_uy = TNL::MPI::reduce(local_l1sum_uy, MPI_SUM, MPI_COMM_WORLD);
		real l1sum_uz = TNL::MPI::reduce(local_l1sum_uz, MPI_SUM, MPI_COMM_WORLD);
		real l2sum_ux = TNL::MPI::reduce(local_l2sum_ux, MPI_SUM, MPI_COMM_WORLD);
		real l2sum_uy = TNL::MPI::reduce(local_l2sum_uy, MPI_SUM, MPI_COMM_WORLD);
		real l2sum_uz = TNL::MPI::reduce(local_l2sum_uz, MPI_SUM, MPI_COMM_WORLD);

		// convert to physical units
		real vol = nse.lat.physDl * nse.lat.physDl * nse.lat.physDl;
		auto to_phys = [&](real l1, real l2) -> std::pair<real, real>
		{
			real l1p = nse.lat.lbm2physVelocity(l1 * vol);
			real l2p = nse.lat.lbm2physVelocity(sqrt(l2 * vol));
			return {l1p, l2p};
		};
		auto [l1error_phys_ux, l2error_phys_ux] = to_phys(l1sum_ux, l2sum_ux);
		auto [l1error_phys_uy, l2error_phys_uy] = to_phys(l1sum_uy, l2sum_uy);
		auto [l1error_phys_uz, l2error_phys_uz] = to_phys(l1sum_uz, l2sum_uz);

		// dynamic stopping criterion (based on ux error, the primary component)
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
			block.data.fx = use_forcing ? driving_force : 0;
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
void sim(
	const std::string& adios_config,
	int RESOLUTION,
	double Re,
	double lbm_viscosity,
	bool use_forcing,
	double u_max_phys
#if defined(USE_POWERLAW)
	,
	double n,
	double K_phys
#elif defined(USE_CYMODEL)
	,
	double n,
	double a,
	double lambda_phys,
	double eta_inf_phys,
	double eta_0_phys
#elif defined(USE_CASSON)
	,
	double tau_y_phys,
	double eta_C_phys
#endif
)
{
	using idx = typename NSE::TRAITS::idx;
	using real = typename NSE::TRAITS::real;
	using dreal = typename NSE::TRAITS::dreal;
	using point_t = typename NSE::TRAITS::point_t;
	using lat_t = Lattice<3, real, idx>;

	int block_size = 32;
	int Y = block_size * RESOLUTION;
	int Z = Y;
	real PHYS_HEIGHT = 1.0;
	real PHYS_DL = PHYS_HEIGHT / ((real) Z - 4);
	int X = std::round(3.0 * PHYS_HEIGHT / PHYS_DL);
	if (! use_forcing)
		X += 2;	 // for GEO_NOTHING layers

	real R_lbm = (Z - 4) / 2.0;
	real c_s = 1.0 / std::sqrt(3.0);
	real u_max_lbm = Re * lbm_viscosity / (2.0 * R_lbm);
	real Ma = u_max_lbm / c_s;
	const real rho = 1000.0;  // kg/m^3, constant density
	const char* bc_variant = use_forcing ? "force" : "inflow";

	real driving_force;
	std::string state_id;

#if defined(USE_POWERLAW)
	// K_phys: Pa·s^n (dynamic consistency index)
	// K_phys/ρ has units [m²·s^(n-2)], NOT a kinematic viscosity for n≠1.
	// Define reference shear rate γ_ref = u_max/R to get a proper kinematic viscosity:
	//   ν_ref = (K_phys/ρ) · γ_ref^(n-1)  [m²/s]
	// Then use diffusive scaling (same as CY/Casson):
	//   δ_t = ν_lbm · δ_x² / ν_ref
	// K_lbm = (K_phys/ρ) · δ_t^(2-n)/δ_x² (varies with resolution for n≠1)
	// ν_lbm = K_lbm · γ_ref_lbm^(n-1) = lbm_viscosity (invariant, controls ω at γ_ref)
	// For n=1: ν_ref = K/ρ, δ_t = ν_lbm·δ_x²·ρ/K, K_lbm = ν_lbm — identical to old formula.
	real R_phys = PHYS_HEIGHT / 2.0;
	real u_max_phys_local = u_max_phys;
	if (u_max_phys <= 0.0) {
		// Re-driven mode: Re = Re_ref = u_max·2R/ν_ref = 2·(ρ/K)·u_max^(2-n)·R^n
		// → u_max = (Re · K / (2·ρ·R^n))^(1/(2-n))
		u_max_phys_local = std::pow(Re * K_phys / (2.0 * std::pow(R_phys, n) * rho), 1.0 / (2.0 - n));
	}
	real gamma_ref = u_max_phys_local / R_phys;          // characteristic shear rate [1/s]
	real nu_ref = (K_phys / rho) * std::pow(gamma_ref, n - 1.0);  // [m²/s]
	real PHYS_DT = lbm_viscosity * PHYS_DL * PHYS_DL / nu_ref;     // diffusive scaling
	real K_lbm = (K_phys / rho) * std::pow(PHYS_DT, 2.0 - n) / (PHYS_DL * PHYS_DL);

	u_max_lbm = u_max_phys_local * PHYS_DT / PHYS_DL;
	Ma = u_max_lbm / c_s;
	Re = u_max_lbm * 2.0 * R_lbm / lbm_viscosity;  // Re_ref (resolution-invariant)
	real PHYS_VISCOSITY = nu_ref;  // reference kinematic viscosity for t_steady

	real K = K_lbm;
	real exponent = (n + 1.0) / n;
	driving_force = K * std::pow(u_max_lbm * (n + 1.0) / n / std::pow(R_lbm, exponent), n);
	spdlog::info(
		"Power-law Poiseuille: n={}, K_phys={:.4e} Pa·s^n, K_lbm={:.4e}, nu_ref={:.4e} m²/s, gamma_ref={:.4e} 1/s, A={:.6e}, u_max={:.6e}, Ma={:.6f}, omega={:.4f}, Re={}",
		n,
		K_phys,
		K_lbm,
		nu_ref,
		gamma_ref,
		driving_force,
		u_max_lbm,
		Ma,
		1.0 / (3.0 * lbm_viscosity + 0.5),
		Re
	);

	state_id = fmt::format(
		"sim_nonnewtonian_poiseuille/n={}_K={:.2e}_nu={:.2e}_{}_{}_res={:02d}_np={:03d}{}",
		n,
		K_phys,
		lbm_viscosity,
		u_max_phys > 0.0 ? fmt::format("umax={:.6e}", u_max_phys) : fmt::format("Re={}", Re),
		bc_variant,
		RESOLUTION,
		TNL::MPI::GetSize(MPI_COMM_WORLD),
		SIM_NONNEWTONIAN_POISEUILLE_STATE_SUFFIX
	);

#elif defined(USE_CYMODEL)
	// eta_inf_phys, eta_0_phys: Pa·s (dynamic viscosities)
	// nu_0_phys = eta_0 / rho, nu_inf_phys = eta_inf / rho (kinematic, m^2/s)
	// Collision uses eta_0 (maximum). nu_0_lbm = lbm_viscosity.
	// PHYS_DT = nu_0_lbm * dl^2 / nu_0_phys
	// nu_inf_lbm = nu_inf_phys * dt / dl^2 = lbm_viscosity * (eta_inf / eta_0)
	// lambda_lbm = lambda_phys / PHYS_DT (since lambda*gamma must be dimensionless)
	real nu_0_phys = eta_0_phys / rho;
	real nu_inf_phys = eta_inf_phys / rho;
	real PHYS_DT = lbm_viscosity * PHYS_DL * PHYS_DL / nu_0_phys;
	if (u_max_phys > 0.0) {
		u_max_lbm = u_max_phys * PHYS_DT / PHYS_DL;
		Ma = u_max_lbm / c_s;
		Re = u_max_lbm * 2.0 * R_lbm / lbm_viscosity;
	}
	real nu_0_lbm = lbm_viscosity;
	real nu_inf_lbm = nu_inf_phys * PHYS_DT / (PHYS_DL * PHYS_DL);
	real lambda_lbm = lambda_phys / PHYS_DT;
	real PHYS_VISCOSITY = nu_0_phys;

	CY_Constitutive cy{nu_inf_lbm, nu_0_lbm, lambda_lbm, a, n};

	// Root-find A from W(0; A) = u_max.  W(0) = R*u_R - F(u_R)/A.
	auto Wmax = [&](real A) -> real
	{
		return poiseuille_W(
			0.0,
			R_lbm,
			A,
			[&](real u)
			{
				return cy.flux(u);
			},
			[&](real u)
			{
				return cy.pot(u);
			}
		);
	};
	real A_lo = 0, A_hi = 1.0;
	while (Wmax(A_hi) < u_max_lbm)
		A_hi *= 2;
	for (int i = 0; i < 100; i++) {
		real A_mid = (A_lo + A_hi) * 0.5;
		if (Wmax(A_mid) < u_max_lbm)
			A_lo = A_mid;
		else
			A_hi = A_mid;
	}
	driving_force = (real) ((A_lo + A_hi) * 0.5);

	spdlog::info(
		"CY Poiseuille: n={}, a={}, lambda={:.4e} s, eta_inf={:.4e} Pa·s, eta_0={:.4e} Pa·s, nu_inf_lbm={:.4e}, nu_0_lbm={:.4e}, lambda_lbm={:.4e}, "
		"A={:.6e}, u_max={:.6e}, Ma={:.4f}, omega={:.4f}, Re={}",
		n,
		a,
		lambda_phys,
		eta_inf_phys,
		eta_0_phys,
		nu_inf_lbm,
		nu_0_lbm,
		lambda_lbm,
		driving_force,
		u_max_lbm,
		Ma,
		1.0 / (3.0 * nu_0_lbm + 0.5),
		Re
	);
	state_id = fmt::format(
		"sim_nonnewtonian_poiseuille_CY/n={}_a={}_lam={:.2e}_einf={:.2e}_e0={:.2e}_nu={:.2e}_{}_{}_res={:02d}_np={:03d}{}",
		n,
		a,
		lambda_phys,
		eta_inf_phys,
		eta_0_phys,
		lbm_viscosity,
		u_max_phys > 0.0 ? fmt::format("umax={:.6e}", u_max_phys) : fmt::format("Re={}", Re),
		bc_variant,
		RESOLUTION,
		TNL::MPI::GetSize(MPI_COMM_WORLD),
		SIM_NONNEWTONIAN_POISEUILLE_STATE_SUFFIX
	);

#elif defined(USE_CASSON)
	// tau_y_phys: Pa (yield stress), eta_C_phys: Pa·s (plastic dynamic viscosity)
	// nu_C_phys = eta_C / rho (plastic kinematic viscosity, m^2/s)
	// Collision uses nu_C (plastic viscosity, minimum). nu_C_lbm = lbm_viscosity.
	// PHYS_DT = nu_C_lbm * dl^2 / nu_C_phys
	// k0_lbm = sqrt(tau_y / rho) * dt / dl  (from dimensional analysis of Casson equation)
	// k1_lbm = sqrt(nu_C_lbm) = sqrt(lbm_viscosity)
	real nu_C_phys = eta_C_phys / rho;
	real PHYS_DT = lbm_viscosity * PHYS_DL * PHYS_DL / nu_C_phys;
	if (u_max_phys > 0.0) {
		u_max_lbm = u_max_phys * PHYS_DT / PHYS_DL;
		Ma = u_max_lbm / c_s;
		Re = u_max_lbm * 2.0 * R_lbm / lbm_viscosity;
	}
	real nu_C_lbm = lbm_viscosity;
	real k0_lbm = std::sqrt(tau_y_phys / rho) * PHYS_DT / PHYS_DL;
	real k1_lbm = std::sqrt(nu_C_lbm);
	real PHYS_VISCOSITY = nu_C_phys;

	Casson_Constitutive cas{k0_lbm, k1_lbm};

	// Root-find A from W(0; A) = u_max
	auto Wmax = [&](real A) -> real
	{
		return poiseuille_W(
			0.0,
			R_lbm,
			A,
			[&](real u)
			{
				return cas.flux(u);
			},
			[&](real u)
			{
				return cas.pot(u);
			}
		);
	};
	real A_lo = 0, A_hi = 1.0;
	while (Wmax(A_hi) < u_max_lbm)
		A_hi *= 2;
	for (int i = 0; i < 100; i++) {
		real A_mid = (A_lo + A_hi) * 0.5;
		if (Wmax(A_mid) < u_max_lbm)
			A_lo = A_mid;
		else
			A_hi = A_mid;
	}
	driving_force = (real) ((A_lo + A_hi) * 0.5);

	spdlog::info(
		"Casson Poiseuille: tau_y={:.4e} Pa, eta_C={:.4e} Pa·s, k0_lbm={:.4e}, k1_lbm={:.4e}, nu_C_lbm={:.4e}, A={:.6e}, u_max={:.6e}, Ma={:.4f}, "
		"omega={:.4f}, Re={}",
		tau_y_phys,
		eta_C_phys,
		k0_lbm,
		k1_lbm,
		nu_C_lbm,
		driving_force,
		u_max_lbm,
		Ma,
		1.0 / (3.0 * nu_C_lbm + 0.5),
		Re
	);
	state_id = fmt::format(
		"sim_nonnewtonian_poiseuille_Cas/ty={:.2e}_eC={:.2e}_nu={:.2e}_{}_{}_res={:02d}_np={:03d}{}",
		tau_y_phys,
		eta_C_phys,
		lbm_viscosity,
		u_max_phys > 0.0 ? fmt::format("umax={:.6e}", u_max_phys) : fmt::format("Re={}", Re),
		bc_variant,
		RESOLUTION,
		TNL::MPI::GetSize(MPI_COMM_WORLD),
		SIM_NONNEWTONIAN_POISEUILLE_STATE_SUFFIX
	);
#endif

	point_t PHYS_ORIGIN = {0., 0., 0.};

	lat_t lat;
	lat.global = typename lat_t::CoordinatesType(X, Y, Z);
	lat.physOrigin = PHYS_ORIGIN;
	lat.physDl = PHYS_DL;
	lat.physDt = PHYS_DT;
	lat.physViscosity = PHYS_VISCOSITY;

	StateLocal<NSE> state(state_id, MPI_COMM_WORLD, lat, use_forcing, adios_config);

	if (! state.canCompute())
		return;

	state.driving_force = driving_force;
	state.use_forcing = use_forcing;

	// Set model-specific block data (needed by both the analytical solution
	// and the non-Newtonian kernel).
	// lbmViscosity is set here because the analytical solution is evaluated
	// during SimInit (iteration 0), before State::updateKernelData() runs.
	for (auto& block : state.nse.blocks) {
#if defined(USE_POWERLAW)
		block.data.lbmViscosity = lbm_viscosity;
		block.data.lbm_K = K_lbm;
		block.data.lbm_n = n;
#elif defined(USE_CYMODEL)
		block.data.lbmViscosity = nu_0_lbm;
		block.data.lbm_nu_inf = nu_inf_lbm;
		block.data.lbm_lambda = lambda_lbm;
		block.data.lbm_a = a;
		block.data.lbm_n = n;
#elif defined(USE_CASSON)
		block.data.lbmViscosity = lbm_viscosity;
		block.data.lbm_k0 = k0_lbm;
		block.data.lbm_k1 = k1_lbm;
#endif
	}

	// In pressure-driven mode, set up the inflow velocity profile from the
	// analytical solution. The analytical reference still uses driving_force
	// (= pressure gradient A), but fx is set to 0 in updateKernelData().
	if (! use_forcing) {
		auto& block = state.nse.blocks.front();
		int local_y = block.local.y();
		int local_z = block.local.z();
		state.inflow_profile.setSize(local_y * local_z);
		block.data.inflow_vx = state.inflow_profile.getData();
		block.data.inflow_vy = nullptr;
		block.data.inflow_vz = nullptr;

		// Build the profile on the host, then copy to device.
		std::vector<dreal> profile(local_y * local_z);
		for (int j = 0; j < local_y; j++)
			for (int k = 0; k < local_z; k++)
				profile[k * local_y + j] = (dreal) state.analytical_ux(block.offset.z() + k);
#ifdef USE_CUDA
		TNL::Backend::memcpy(block.data.inflow_vx, profile.data(), local_y * local_z * sizeof(dreal), TNL::Backend::MemcpyHostToDevice);
#else
		std::copy(profile.begin(), profile.end(), block.data.inflow_vx);
#endif
	}

	// Steady state time (physical) = R^2/nu = (H/2)^2 / PHYS_VISCOSITY = H^2/(4*nu_phys).
	// physDt shrinks as 1/N^2, so physFinalTime must be resolution-independent.
	// Use 10x steady state physical time as safety limit.
	real t_steady = (PHYS_HEIGHT * PHYS_HEIGHT) / (4.0 * PHYS_VISCOSITY);
	state.nse.physFinalTime = 10.0 * t_steady;
	state.cnt[PRINT].period = t_steady;
	state.cnt[PROBE1].period = 0.1 * t_steady;

	state.cnt[OUT3D].period = t_steady;

	execute(state);
}

template <typename TRAITS = TraitsSP>
void run(
	const std::string& adios_config,
	int resolution,
	double Re,
	double lbm_viscosity,
	bool use_forcing,
	double u_max_phys
#if defined(USE_POWERLAW)
	,
	double n,
	double K_phys
#elif defined(USE_CYMODEL)
	,
	double n,
	double a,
	double lambda_phys,
	double eta_inf_phys,
	double eta_0_phys
#elif defined(USE_CASSON)
	,
	double tau_y_phys,
	double eta_C_phys
#endif
)
{
	using COLL = D3Q27_CUM<TRAITS, D3Q27_EQ<TRAITS>>;

	using NSE_CONFIG = LBM_CONFIG<
		TRAITS,
		D3Q27_KernelStruct,
		NSE_Data_NonNewtonian_Poiseuille<TRAITS>,
		COLL,
		typename COLL::EQ,
		D3Q27_STREAMING<TRAITS>,
		D3Q27_BC_All,
		MacroNonNewtonianDefault<TRAITS>>;

	sim<NSE_CONFIG>(
		adios_config,
		resolution,
		Re,
		lbm_viscosity,
		use_forcing,
		u_max_phys
#if defined(USE_POWERLAW)
		,
		n,
		K_phys
#elif defined(USE_CYMODEL)
		,
		n,
		a,
		lambda_phys,
		eta_inf_phys,
		eta_0_phys
#elif defined(USE_CASSON)
		,
		tau_y_phys,
		eta_C_phys
#endif
	);
}

int main(int argc, char** argv)
{
	TNLMPI_INIT mpi(argc, argv);

	argparse::ArgumentParser program("sim_nonnewtonian_poiseuille");
#if defined(USE_POWERLAW)
	program.add_description("Non-Newtonian plane Poiseuille flow benchmark (power-law model).");
#elif defined(USE_CYMODEL)
	program.add_description("Non-Newtonian plane Poiseuille flow benchmark (Carreau-Yasuda model).");
#elif defined(USE_CASSON)
	program.add_description("Non-Newtonian plane Poiseuille flow benchmark (Casson model).");
#endif

	program.add_argument("--adios-config").help("path to ADIOS2 configuration file").default_value(std::string("adios2.xml")).nargs(1);
	program.add_argument("--resolution").help("resolution of the lattice").scan<'i', int>().default_value(1).nargs(1);
	program.add_argument("--lbm-viscosity")
		.help("Reference lattice viscosity (nu_0 for CY, nu_C for Casson, K for power-law). Must be in range (0, 1/6].")
		.scan<'g', double>()
		.default_value(0.05)
		.nargs(1);
	program.add_argument("--use-forcing")
		.help("use forcing term with periodic boundary conditions instead of inflow/outflow boundary conditions")
		.flag();

	// Mutually exclusive driving modes: --Re or --u-max
	auto& drive_group = program.add_mutually_exclusive_group(true);
	drive_group.add_argument("--Re").help("Reynolds number based on reference viscosity").scan<'g', double>();
	drive_group.add_argument("--u-max").help("maximum physical velocity [m/s]").scan<'g', double>();

#if defined(USE_POWERLAW)
	program.add_argument("--n")
		.help("power-law index (1=Newtonian, <1=shear-thinning, >1=shear-thickening)")
		.scan<'g', double>()
		.default_value(1.0)
		.nargs(1);
	program.add_argument("--K").help("Power-law consistency index (Pa·s^n)").scan<'g', double>().default_value(0.01).nargs(1);
#elif defined(USE_CYMODEL)
	program.add_argument("--n").help("CY power-law index (<1=shear-thinning)").scan<'g', double>().default_value(0.5).nargs(1);
	program.add_argument("--a").help("Yasuda transition parameter (a=2 for Carreau model)").scan<'g', double>().default_value(2.0).nargs(1);
	program.add_argument("--lambda").help("CY relaxation time constant (seconds)").scan<'g', double>().default_value(1.0).nargs(1);
	program.add_argument("--eta-inf").help("CY infinite-shear dynamic viscosity (Pa·s)").scan<'g', double>().default_value(0.01).nargs(1);
	program.add_argument("--eta-0").help("CY zero-shear dynamic viscosity (Pa·s)").scan<'g', double>().default_value(0.1).nargs(1);
#elif defined(USE_CASSON)
	program.add_argument("--tau-y").help("Casson yield stress (Pa)").scan<'g', double>().default_value(0.01).nargs(1);
	program.add_argument("--eta-c").help("Casson plastic dynamic viscosity (Pa·s)").scan<'g', double>().default_value(0.01).nargs(1);
#endif

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
	const bool use_forcing = program.get<bool>("--use-forcing");

	const bool has_u_max = program.is_used("--u-max");
	const double u_max_phys = has_u_max ? program.get<double>("--u-max") : -1.0;
	double Re = has_u_max ? 0.0 : program.get<double>("--Re");

	if (resolution < 1) {
		fmt::println(stderr, "CLI error: resolution must be at least 1");
		return 1;
	}
	if (!has_u_max && Re < 1) {
		fmt::println(stderr, "CLI error: Re must be at least 1");
		return 1;
	}
	if (has_u_max && u_max_phys <= 0.0) {
		fmt::println(stderr, "CLI error: --u-max must be positive");
		return 1;
	}
	if (lbm_viscosity <= 0.0 || lbm_viscosity > 1. / 6.) {
		fmt::println(stderr, "CLI error: --lbm-viscosity must be in range (0, 1/6]");
		return 1;
	}

#if defined(USE_POWERLAW)
	const double n = program.get<double>("--n");
	const double K_phys = program.get<double>("--K");

	if (n <= 0.0) {
		fmt::println(stderr, "CLI error: n must be positive");
		return 1;
	}
	if (K_phys <= 0.0) {
		fmt::println(stderr, "CLI error: K must be positive");
		return 1;
	}

	run(adios_config, resolution, Re, lbm_viscosity, use_forcing, u_max_phys, n, K_phys);
#elif defined(USE_CYMODEL)
	const double n = program.get<double>("--n");
	const double a = program.get<double>("--a");
	const double lambda_phys = program.get<double>("--lambda");
	const double eta_inf_phys = program.get<double>("--eta-inf");
	const double eta_0_phys = program.get<double>("--eta-0");

	if (n <= 0.0) {
		fmt::println(stderr, "CLI error: n must be positive");
		return 1;
	}
	if (a <= 0.0) {
		fmt::println(stderr, "CLI error: a must be positive");
		return 1;
	}
	if (lambda_phys < 0.0) {
		fmt::println(stderr, "CLI error: lambda must be non-negative");
		return 1;
	}
	if (eta_inf_phys <= 0.0 || eta_0_phys <= 0.0) {
		fmt::println(stderr, "CLI error: eta-inf and eta-0 must be positive");
		return 1;
	}
	if (eta_0_phys <= eta_inf_phys) {
		fmt::println(stderr, "CLI error: eta-0 must be > eta-inf (zero-shear > infinite-shear)");
		return 1;
	}

	run(adios_config, resolution, Re, lbm_viscosity, use_forcing, u_max_phys, n, a, lambda_phys, eta_inf_phys, eta_0_phys);
#elif defined(USE_CASSON)
	const double tau_y_phys = program.get<double>("--tau-y");
	const double eta_C_phys = program.get<double>("--eta-c");

	if (tau_y_phys < 0.0) {
		fmt::println(stderr, "CLI error: tau-y must be non-negative");
		return 1;
	}
	if (eta_C_phys <= 0.0) {
		fmt::println(stderr, "CLI error: eta-c must be positive");
		return 1;
	}

	run(adios_config, resolution, Re, lbm_viscosity, use_forcing, u_max_phys, tau_y_phys, eta_C_phys);
#endif

	return 0;
}
