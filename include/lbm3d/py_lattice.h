#pragma once

#include <pytnl/pytnl.h>

#include "lattice.h"
#include "py_typedef.h"

template <int D, typename real, typename idx>
void export_Lattice(nb::module_& m, const char* name)
{
	using Lattice = ::Lattice<D, real, idx>;
	using PointType = typename Lattice::PointType;

	auto lattice =
		nb::class_<Lattice>(m, name)
			// Default constructor
			.def(nb::init<>())

			// Static attributes
			.def_ro_static("D", &Lattice::D, "Dimension of the lattice")

			// Attributes
			.def_rw("global_", &Lattice::global, "Global size of the lattice")
			.def_rw(
				"physOrigin", &Lattice::physOrigin, "Physical coordinates of the point at the center between $(0,0,0)$ and $(1,1,1)$ lattice sites"
			)
			.def_rw("physDl", &Lattice::physDl, "Spatial step, i.e. the distance between two neighboring lattice sites in physical coordinates")
			.def_rw("physDt", &Lattice::physDt, "Temporal step, i.e. fixed length of each time step in physical time domain")
			.def_rw("physViscosity", &Lattice::physViscosity, "Physical viscosity of the fluid")

			// Methods
			.def("lbmViscosity", &Lattice::lbmViscosity, "Returns the non-dimensional lattice viscosity")
			.def(
				"phys2lbmViscosity",
				&Lattice::phys2lbmViscosity,
				nb::arg("physViscosity"),
				"Converts the given viscosity from physical to non-dimensional units"
			)
			.def(
				"lbm2physViscosity",
				&Lattice::lbm2physViscosity,
				nb::arg("lbmViscosity"),
				"Converts the given viscosity from non-dimensional to physical units"
			)
			.def(
				"lbm2physPoint",
				[](Lattice& self, idx x, idx y, idx z)
				{
					return self.lbm2physPoint(x, y, z);
				},
				nb::arg("x"),
				nb::arg("y"),
				nb::arg("z"),
				"Converts the given point from non-dimensional to physical units"
			)
			.def(
				"lbm2physPoint",
				[](Lattice& self, const PointType& p)
				{
					return self.lbm2physPoint(p);
				},
				nb::arg("p"),
				"Converts the given point from non-dimensional to physical units"
			)
			.def("lbm2physX", &Lattice::lbm2physX, nb::arg("x"), "Converts the given x coordinate from non-dimensional to physical units")
			.def("lbm2physY", &Lattice::lbm2physY, nb::arg("y"), "Converts the given y coordinate from non-dimensional to physical units")
			.def("lbm2physZ", &Lattice::lbm2physZ, nb::arg("z"), "Converts the given z coordinate from non-dimensional to physical units")
			.def("phys2lbmPoint", &Lattice::phys2lbmPoint, nb::arg("p"), "Converts the given point from physical to non-dimensional units")
			.def("phys2lbmX", &Lattice::phys2lbmX, nb::arg("x"), "Converts the given x coordinate from physical to non-dimensional units")
			.def("phys2lbmY", &Lattice::phys2lbmY, nb::arg("y"), "Converts the given y coordinate from physical to non-dimensional units")
			.def("phys2lbmZ", &Lattice::phys2lbmZ, nb::arg("z"), "Converts the given z coordinate from physical to non-dimensional units")
			.def(
				"lbm2physVelocity",
				&Lattice::lbm2physVelocity,
				nb::arg("lbm_velocity"),
				"Converts the given velocity from non-dimensional to physical units"
			)
			.def(
				"phys2lbmVelocity",
				&Lattice::phys2lbmVelocity,
				nb::arg("phys_velocity"),
				"Converts the given velocity from physical to non-dimensional units"
			)
			.def("lbm2physForce", &Lattice::lbm2physForce, nb::arg("lbm_force"), "Converts the given force from non-dimensional to physical units")
			.def("phys2lbmForce", &Lattice::phys2lbmForce, nb::arg("phys_force"), "Converts the given force from physical to non-dimensional units")

			// Getters for TNL compatibility
			.def("getMeshDimension", &Lattice::getMeshDimension, "Returns the spatial dimension of the lattice")
			.def("getDimension", &Lattice::getDimension, "Returns the spatial dimension of the lattice")
			.def("size", &Lattice::size, "Returns the global lattice size", nb::rv_policy::reference_internal)
			.def(
				"getDimensions",
				&Lattice::getDimensions,
				"Returns the size of the **grid** represented by the lattice (i.e., the number of voxels between the lattice sites)",
				nb::rv_policy::reference_internal
			)
			.def("getOrigin", &Lattice::getOrigin, "Returns the origin of the lattice", nb::rv_policy::reference_internal)
			.def("getSpaceSteps", &Lattice::getSpaceSteps, "Returns the space steps of the grid/lattice")
		//
		;

	// Export typedefs
	export_typedef<typename Lattice::RealType>(lattice, "RealType");
	export_typedef<typename Lattice::GlobalIndexType>(lattice, "GlobalIndexType");
	export_typedef<typename Lattice::PointType>(lattice, "PointType");
	export_typedef<typename Lattice::CoordinatesType>(lattice, "CoordinatesType");
}
