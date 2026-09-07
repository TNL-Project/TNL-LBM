#pragma once

#include <nanobind/nanobind.h>
#include <pytnl/pytnl.h>

#include "inlet_regions_lbm.h"
#include "lbm.h"

// Python surface of the inlet-regions management: the sim-side state is an
// InletRegionsState instance owned by the Python StateLocal, and the same
// universal free functions the C++ sims call (add* to stamp the claims,
// finalizeInletRegions to carve/validate/discover/bake/publish) operate on
// the State's LBM and that instance. The kernel-facing DATA is untouched -
// gating stays with the has_inlet_regions_v trait of the chosen CONFIG.
template <typename CONFIG>
void export_InletRegions(nb::module_& m)
{
	using TRAITS = typename CONFIG::TRAITS;
	using OS = InletRegionsState<TRAITS>;
	using idx = typename TRAITS::idx;
	using idx3d = typename TRAITS::idx3d;
	using dreal = typename TRAITS::dreal;

	nb::enum_<ProfileType>(m, "ProfileType").value("UNIFORM", ProfileType::UNIFORM).value("PARABOLIC", ProfileType::PARABOLIC);

	nb::class_<OS>(m, "InletRegionsState")
		.def(nb::init<>())
		.def_rw("discover", &OS::discover, "gate the discovery pass in finalizeInletRegions (default True)");

	m.def(
		"addInletRegion",
		[](LBM<CONFIG>& nse, OS& os, short axis, short sign, idx planeOffset, idx3d lo, idx3d hi, ProfileType profile, dreal amplitude)
		{
			return addInletRegion(nse, os, axis, sign, planeOffset, lo, hi, profile, amplitude).id;
		},
		nb::arg("nse"),
		nb::arg("regions"),
		nb::arg("axis"),
		nb::arg("sign"),
		nb::arg("plane_offset"),
		nb::arg("lo"),
		nb::arg("hi"),
		nb::arg("profile") = ProfileType::UNIFORM,
		nb::arg("amplitude") = 0.0,
		"register an authored inlet region on the plane (axis, sign, plane_offset) with authoritative "
		"direction provenance and claim its rect (returns the inlet region id)"
	);
	m.def(
		"finalizeInletRegions",
		[](LBM<CONFIG>& nse, OS& os)
		{
			finalizeInletRegions(nse, os);
		},
		nb::arg("nse"),
		nb::arg("regions"),
		"carve the authored outward ghost layers, validate the claim map, discover the complement, and "
		"bake+publish the precomputed site velocities"
	);
}
