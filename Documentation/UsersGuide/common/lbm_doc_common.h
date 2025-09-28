#pragma once

#include <argparse/argparse.hpp>
#include <fmt/core.h>
#include <fmt/ostream.h>
#include <spdlog/spdlog.h>

#include <string>
#include <vector>
#include <stdexcept>
#include <iostream>

#include "lbm3d/core.h"

namespace tnl_lbm::doc
{
// [doc-lid-options-start]
struct LidDrivenCavityOptions
{
	int resolution = 1;
	double cavityLength = 0.01;    // [m]
	double lidVelocity = 0.1;      // [m/s]
	double fluidViscosity = 1.0e-6;  // [m^2/s]
	double latticeViscosity = 1.0 / 6.0;  // default SRT relaxation time
	double finalPhysicalTime = 0.05;      // [s]
	double printPeriod = 0.01;            // [s]
	double vtkPeriod = -1.0;              // disabled by default
	double cutPeriod = -1.0;
	bool docMode = false;
	std::string runIdSuffix;

	void applyDocMode()
	{
		if (! docMode)
			return;

		resolution = 1;
		finalPhysicalTime = 1.0e-3;
		printPeriod = finalPhysicalTime;
		vtkPeriod = -1.0;
		cutPeriod = -1.0;
		runIdSuffix = "_doc";
	}
};
// [doc-lid-options-end]

// [doc-lid-cli-start]
inline void setupLidDrivenCavityCLI(argparse::ArgumentParser& program)
{
	program.add_argument("--resolution")
		.help("multiplicative factor for the base 32^3 cavity grid")
		.scan<'i', int>()
		.default_value(1);
	program.add_argument("--lid-velocity")
		.help("lid velocity in m/s")
		.scan<'g', double>()
		.default_value(0.1);
	program.add_argument("--length")
		.help("edge length of the cubic cavity in metres")
		.scan<'g', double>()
		.default_value(0.01);
	program.add_argument("--viscosity")
		.help("kinematic viscosity in m^2/s")
		.scan<'g', double>()
		.default_value(1.0e-6);
	program.add_argument("--lbm-viscosity")
		.help("lattice viscosity (tau = 3*nu + 0.5)")
		.scan<'g', double>()
		.default_value(1.0 / 6.0);
	program.add_argument("--final-time")
		.help("simulation length in physical seconds")
		.scan<'g', double>()
		.default_value(0.05);
	program.add_argument("--print-period")
		.help("how often to print progress information in physical seconds")
		.scan<'g', double>()
		.default_value(0.01);
	program.add_argument("--vtk-period")
		.help("VTK export period in physical seconds (negative disables output)")
		.scan<'g', double>()
		.default_value(-1.0);
	program.add_argument("--cut-period")
		.help("2D slice export period in physical seconds (negative disables output)")
		.scan<'g', double>()
		.default_value(-1.0);
	program.add_argument("--id-suffix")
		.help("custom suffix added to the results directory identifier")
		.default_value(std::string());
	program.add_argument("--doc-mode")
		.help("run a minimal configuration used by the documentation build")
		.default_value(false)
		.implicit_value(true);
}

inline LidDrivenCavityOptions parseLidDrivenCavityOptions(const argparse::ArgumentParser& program)
{
	LidDrivenCavityOptions opts;
	opts.resolution = program.get<int>("--resolution");
	opts.lidVelocity = program.get<double>("--lid-velocity");
	opts.cavityLength = program.get<double>("--length");
	opts.fluidViscosity = program.get<double>("--viscosity");
	opts.latticeViscosity = program.get<double>("--lbm-viscosity");
	opts.finalPhysicalTime = program.get<double>("--final-time");
	opts.printPeriod = program.get<double>("--print-period");
	opts.vtkPeriod = program.get<double>("--vtk-period");
	opts.cutPeriod = program.get<double>("--cut-period");
	opts.runIdSuffix = program.get<std::string>("--id-suffix");
	opts.docMode = program.get<bool>("--doc-mode");
	return opts;
}
// [doc-lid-cli-end]

// Utility that converts options into a lattice description.
// [doc-lid-lattice-start]
template <typename TRAITS>
Lattice<3, typename TRAITS::real, typename TRAITS::idx> makeCavityLattice(const LidDrivenCavityOptions& opts)
{
	using real = typename TRAITS::real;
	using idx = typename TRAITS::idx;

	const idx interior = idx(32 * opts.resolution);
	const real physDl = opts.cavityLength / static_cast<real>(interior);
	const real physDt = opts.latticeViscosity / opts.fluidViscosity * physDl * physDl;

	Lattice<3, real, idx> lat;
	lat.global = typename Lattice<3, real, idx>::CoordinatesType(interior + 2, interior + 2, interior + 2);
	lat.physOrigin = typename Lattice<3, real, idx>::PointType(0, 0, 0);
	lat.physDl = physDl;
	lat.physDt = physDt;
	lat.physViscosity = opts.fluidViscosity;
	return lat;
}
// [doc-lid-lattice-end]

// [doc-lid-state-start]
template <typename NSE>
class LidDrivenCavityState : public State<NSE>
{
	using Base = State<NSE>;

public:
	using TRAITS = typename NSE::TRAITS;
	using BC = typename NSE::BC;
	using MACRO = typename NSE::MACRO;
	using BLOCK = LBM_BLOCK<NSE>;
	using idx = typename TRAITS::idx;
	using real = typename TRAITS::real;
	using point_t = typename TRAITS::point_t;

	LidDrivenCavityState(
		const std::string& id,
		const TNL::MPI::Comm& communicator,
		typename Base::lat_t lat,
		real lidVelocityLBM
	)
	: Base(id, communicator, std::move(lat)),
	  lidVelocityLBM(lidVelocityLBM)
	{}

	void setupBoundaries() override
	{
		auto& nse = Base::nse;
		for (auto& block : nse.blocks)
			block.resetMap(BC::GEO_FLUID);

		// Solid walls on all sides except for the moving lid at the top (z = max - 2).
		nse.setBoundaryX(1, BC::GEO_WALL);
		nse.setBoundaryX(nse.lat.global.x() - 2, BC::GEO_WALL);
		nse.setBoundaryY(1, BC::GEO_WALL);
		nse.setBoundaryY(nse.lat.global.y() - 2, BC::GEO_WALL);
		nse.setBoundaryZ(1, BC::GEO_WALL);
		nse.setBoundaryZ(nse.lat.global.z() - 2, BC::GEO_INFLOW_LEFT);

		// Guard layers required by the streaming schemes.
		nse.setBoundaryX(0, BC::GEO_NOTHING);
		nse.setBoundaryX(nse.lat.global.x() - 1, BC::GEO_NOTHING);
		nse.setBoundaryY(0, BC::GEO_NOTHING);
		nse.setBoundaryY(nse.lat.global.y() - 1, BC::GEO_NOTHING);
		nse.setBoundaryZ(0, BC::GEO_NOTHING);
		nse.setBoundaryZ(nse.lat.global.z() - 1, BC::GEO_NOTHING);
	}

	bool outputData(const BLOCK& block, int index, int dof, char* desc, idx x, idx y, idx z, real& value, int& dofs) override
	{
		int k = 0;
		if (index == k++)
			return Base::vtk_helper("density", block.hmacro(MACRO::e_rho, x, y, z), 1, desc, value, dofs);
		if (index == k++) {
			switch (dof) {
				case 0:
					return Base::vtk_helper("velocity", block.hmacro(MACRO::e_vx, x, y, z), 3, desc, value, dofs);
				case 1:
					return Base::vtk_helper("velocity", block.hmacro(MACRO::e_vy, x, y, z), 3, desc, value, dofs);
				case 2:
					return Base::vtk_helper("velocity", block.hmacro(MACRO::e_vz, x, y, z), 3, desc, value, dofs);
			}
		}
		return false;
	}

	void updateKernelVelocities() override
	{
		for (auto& block : Base::nse.blocks) {
			block.data.inflow_vx = lidVelocityLBM;
			block.data.inflow_vy = 0;
			block.data.inflow_vz = 0;
		}
	}

private:
	real lidVelocityLBM;
};
// [doc-lid-state-end]

// Convenience helpers for diagnostics.
// [doc-lid-diagnostics-start]
template <typename NSE>
auto sampleVelocityAt(const State<NSE>& state, typename NSE::TRAITS::idx gx, typename NSE::TRAITS::idx gy, typename NSE::TRAITS::idx gz)
{
	using point_t = typename NSE::TRAITS::point_t;
	for (const auto& block : state.nse.blocks) {
		if (! block.isLocalIndex(gx, gy, gz))
			continue;
		return point_t(
			block.hmacro(NSE::MACRO::e_vx, gx, gy, gz),
			block.hmacro(NSE::MACRO::e_vy, gx, gy, gz),
			block.hmacro(NSE::MACRO::e_vz, gx, gy, gz)
		);
	}
	return point_t(0, 0, 0);
}

inline void printSimulationHeader(const LidDrivenCavityOptions& opts, double reynolds, double physDt, double physDl)
{
	fmt::println("Cavity length: {:.3e} m, lattice spacing: {:.3e} m", opts.cavityLength, physDl);
	fmt::println("Time step: {:.3e} s, Reynolds number: {:.1f}", physDt, reynolds);
}
// [doc-lid-diagnostics-end]

// [doc-lid-runner-start]
template <typename TRAITS, typename COLLISION>
int runLidDrivenCavity(const std::string& executableId, LidDrivenCavityOptions opts)
{
	if (opts.resolution < 1)
		throw std::invalid_argument("resolution must be a positive integer");

	opts.applyDocMode();

	auto lat = makeCavityLattice<TRAITS>(opts);
	const double reynolds = opts.lidVelocity * opts.cavityLength / opts.fluidViscosity;

	const std::string runId = fmt::format("{}_res{:02d}{}", executableId, opts.resolution, opts.runIdSuffix);

	using point_t = typename TRAITS::point_t;
	using NSE_CONFIG = LBM_CONFIG<
		TRAITS,
		D3Q27_KernelStruct,
		NSE_Data_ConstInflow<TRAITS>,
		COLLISION,
		typename COLLISION::EQ,
		D3Q27_STREAMING<TRAITS>,
		D3Q27_BC_All,
		D3Q27_MACRO_Default<TRAITS>>;

	using real = typename TRAITS::real;
	const real lidVelocityLBM = lat.phys2lbmVelocity(static_cast<real>(opts.lidVelocity));

	LidDrivenCavityState<NSE_CONFIG> state(runId, MPI_COMM_WORLD, lat, lidVelocityLBM);
	if (! state.canCompute())
		return 0;

	state.nse.physFinalTime = static_cast<real>(opts.finalPhysicalTime);
	state.nse.physCharLength = static_cast<real>(opts.cavityLength);
	state.cnt[PRINT].period = static_cast<real>(opts.printPeriod);
	state.cnt[VTK3D].period = static_cast<real>(opts.vtkPeriod);
	state.cnt[VTK2D].period = static_cast<real>(opts.cutPeriod);

	if (opts.cutPeriod > 0) {
		const auto mid = state.nse.lat.global / 2;
		state.add2Dcut_Z(mid.z(), "cuts/plane_xy");
		state.add2Dcut_Y(mid.y(), "cuts/plane_xz");
	}

	printSimulationHeader(opts, reynolds, lat.physDt, lat.physDl);

	execute(state);

	state.nse.copyMacroToHost();
	const auto mid = state.nse.lat.global / 2;
	const auto velocityLbm = sampleVelocityAt(state, mid.x(), mid.y(), mid.z());
	const auto velocityPhys = point_t(
		state.nse.lat.lbm2physVelocity(velocityLbm.x()),
		state.nse.lat.lbm2physVelocity(velocityLbm.y()),
		state.nse.lat.lbm2physVelocity(velocityLbm.z())
	);

	fmt::println(
		"Velocity at cavity centre: ({:.3e}, {:.3e}, {:.3e}) m/s",
		velocityPhys.x(), velocityPhys.y(), velocityPhys.z()
	);

	return 0;
}
// [doc-lid-runner-end]

}  // namespace tnl_lbm::doc
