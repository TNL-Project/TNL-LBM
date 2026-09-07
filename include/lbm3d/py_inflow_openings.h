#pragma once

#include <nanobind/nanobind.h>

#include "inflow_openings_lbm.h"
#include "lbm.h"

// Python surface of the inflow-openings management: the sim-side state is an
// InflowOpeningsState instance owned by the Python StateLocal, and the same
// universal free functions the C++ sims call (add* to stamp the claims,
// finalizeInflowOpenings to carve/validate/discover/bake/publish) operate on
// the State's LBM and that instance. The kernel-facing DATA is untouched -
// gating stays with the has_inflow_openings_v trait of the chosen CONFIG.
template <typename CONFIG>
void export_InflowOpenings(nb::module_& m)
{
	using TRAITS = typename CONFIG::TRAITS;
	using OS = InflowOpeningsState<TRAITS>;
	using idx = typename TRAITS::idx;
	using idx3d = typename TRAITS::idx3d;
	using dreal = typename TRAITS::dreal;

	nb::enum_<ProfileType>(m, "ProfileType").value("UNIFORM", ProfileType::UNIFORM).value("PARABOLIC", ProfileType::PARABOLIC);

	nb::class_<OS>(m, "InflowOpeningsState")
		.def(nb::init<>())
		.def_rw("discover", &OS::discover, "gate the discovery pass in finalizeInflowOpenings (default True)");

	m.def(
		"addInflowPlane",
		[](LBM<CONFIG>& nse, OS& os, short axis, short sign, idx planeOffset, idx3d lo, idx3d hi, ProfileType profile, dreal amplitude)
		{
			return addInflowPlane(nse, os, axis, sign, planeOffset, lo, hi, profile, amplitude).id;
		},
		nb::arg("nse"),
		nb::arg("openings"),
		nb::arg("axis"),
		nb::arg("sign"),
		nb::arg("plane_offset"),
		nb::arg("lo"),
		nb::arg("hi"),
		nb::arg("profile") = ProfileType::UNIFORM,
		nb::arg("amplitude") = 0.0,
		"register an authored inflow opening on the plane (axis, sign, plane_offset) with authoritative "
		"direction provenance and claim its rect (returns the opening id)"
	);
	m.def(
		"finalizeInflowOpenings",
		[](LBM<CONFIG>& nse, OS& os)
		{
			finalizeInflowOpenings(nse, os);
		},
		nb::arg("nse"),
		nb::arg("openings"),
		"carve the authored outward ghost layers, validate the claim map, discover the complement, and "
		"bake+publish the precomputed site velocities"
	);
}
