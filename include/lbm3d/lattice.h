#pragma once

#include <TNL/Containers/StaticVector.h>

/**
 * \brief Minimal class representing an equidistant D-dimensional lattice.
 *
 * It does not have a \e Device template argument, because it does not describe
 * a data structure - it contains only the information describing the physical
 * domain of the lattice (i.e., the metadata of the \e ImageData tag in the VTI
 * file format). Note that VTK supports only 1, 2, or 3-dimensional lattices.
 * The class is usable as \e Mesh in the \ref TNL::Meshes::Writers::VTIWriter.
 */
template <int D_ = 3, typename real = float, typename idx = int>
struct Lattice
{
	// these type names are required for compatibility with TNL
	using RealType = real;
	using GlobalIndexType = idx;
	using PointType = TNL::Containers::StaticVector<D_, real>;
	using CoordinatesType = TNL::Containers::StaticVector<D_, idx>;

	//! \brief Dimension of the lattice.
	static constexpr int D = D_;

	//! \brief Global size of the lattice.
	CoordinatesType global = 0;

	//! \brief Physical coordinates of the point at the center between $(0,0,0)$
	//! and $(1,1,1)$ lattice sites.
	PointType physOrigin = 0;

	//! \brief Spatial step, i.e. the distance between two neighboring lattice
	//! sites in physical coordinates.
	RealType physDl = 0;

	//! \brief Temporal step, i.e. fixed length of each time step in physical
	//! time domain.
	real physDt = 0;

	//! \brief Physical viscosity of the fluid.
	real physViscosity = 0;

	//! \brief Returns the non-dimensional lattice viscosity.
	__cuda_callable__ real lbmViscosity() const
	{
		return phys2lbmViscosity(physViscosity);
	}

	//! \brief Converts the given viscosity from physical to non-dimensional units.
	__cuda_callable__ real phys2lbmViscosity(real physViscosity) const
	{
		return physDt / physDl / physDl * physViscosity;
	}

	//! \brief Converts the given viscosity from non-dimensional to physical units.
	__cuda_callable__ real lbm2physViscosity(real lbmViscosity) const
	{
		return physDl * physDl / physDt * lbmViscosity;
	}

	// getters for physical coordinates (note that here x,y,z are *global* lattice indices)
	__cuda_callable__ PointType lbm2physPoint(idx x, idx y, idx z) const
	{
		return physOrigin + (PointType(x, y, z) - 0.5) * physDl;
	}
	__cuda_callable__ PointType lbm2physPoint(PointType p) const
	{
		return physOrigin + (p - 0.5) * physDl;
	}
	__cuda_callable__ real lbm2physX(idx x) const
	{
		return physOrigin.x() + (x - 0.5) * physDl;
	}
	__cuda_callable__ real lbm2physY(idx y) const
	{
		return physOrigin.y() + (y - 0.5) * physDl;
	}
	__cuda_callable__ real lbm2physZ(idx z) const
	{
		return physOrigin.z() + (z - 0.5) * physDl;
	}

	// physical to lattice coordinates (but still real rather than idx, rounding can be done later)
	__cuda_callable__ PointType phys2lbmPoint(PointType p) const
	{
		return (p - physOrigin) / physDl + 0.5;
	}
	__cuda_callable__ real phys2lbmX(real x) const
	{
		return (x - physOrigin.x()) / physDl + 0.5;
	}
	__cuda_callable__ real phys2lbmY(real y) const
	{
		return (y - physOrigin.y()) / physDl + 0.5;
	}
	__cuda_callable__ real phys2lbmZ(real z) const
	{
		return (z - physOrigin.z()) / physDl + 0.5;
	}

	__cuda_callable__ real lbm2physVelocity(real lbm_velocity) const
	{
		return lbm_velocity / physDt * physDl;
	}
	__cuda_callable__ real phys2lbmVelocity(real phys_velocity) const
	{
		return phys_velocity * physDt / physDl;
	}

	__cuda_callable__ real lbm2physForce(real lbm_force) const
	{
		return lbm_force * physDl / physDt / physDt;
	}
	__cuda_callable__ real phys2lbmForce(real phys_force) const
	{
		return phys_force / physDl * physDt * physDt;
	}

	//! \brief Returns the spatial dimension of the lattice.
	static constexpr int getMeshDimension()
	{
		return D;
	}

	//! \brief Returns the spatial dimension of the lattice.
	static constexpr int getDimension()
	{
		return D;
	}

	//! \brief Returns the global lattice size.
	__cuda_callable__ const CoordinatesType& size() const
	{
		return global;
	}

	//! \brief Returns the size of the **grid** represented by the lattice
	//! (i.e., the number of voxels between the lattice sites).
	__cuda_callable__ CoordinatesType getDimensions() const
	{
		return global - 1;
	}

	//! \brief Returns the origin of the lattice.
	__cuda_callable__ const PointType& getOrigin() const
	{
		return physOrigin;
	}

	//! \brief Returns the space steps of the grid/lattice.
	__cuda_callable__ PointType getSpaceSteps() const
	{
		return physDl;
	}
};
