#pragma once

#include <string>

#include "DataManager.h"
#include "DataWriter.h"

template <typename TRAITS>
class UniformDataWriter : public DataWriter<TRAITS>
{
private:
	using idx = typename TRAITS::idx;
	using real = typename TRAITS::real;
	using point_t = typename TRAITS::point_t;
	using idx3d = typename TRAITS::idx3d;

	// data extent attributes
	idx3d global;
	idx3d local;
	idx3d offset;
	point_t physOrigin;
	real physDl;

protected:
	void addVTKAttributes() override;

	void addFidesAttributes() override;

public:
	UniformDataWriter() = delete;

	UniformDataWriter(idx3d global, idx3d local, idx3d offset, point_t physOrigin, real physDl, DataManager& dataManager, std::string ioName);

	// Returns the global extent of this writer's output domain.
	// Used by lbm3d::synchronizeOutputBuffer() (see output_buffer_sync.h) to determine,
	// per dimension, whether the ranks along that dimension participate in this output
	// (safe for MPI communication with them) or not (e.g. the cut dimension of 2D plane
	// cuts, or a dimension where a 3D bounding-box cut covers only part of the domain).
	const idx3d& getGlobal() const
	{
		return global;
	}

	// Returns the offset of this writer's local box within the output domain
	// (cut-relative, i.e. counted from the output box origin, not from the lattice
	// origin). Together with getGlobal() it reconstructs the output box: the global
	// position of the local box begin is getOffset() + the output box origin.
	const idx3d& getOffset() const
	{
		return offset;
	}

	template <typename T>
	void write(std::string varName, T val);

	// DataSource is a 3D array of type T or lambda function with signature src(idx x, idx y, idx z) -> T
	template <typename DataSource>
	void write(const std::string& varName, const DataSource& src, idx3d begin, idx3d end);

	~UniformDataWriter() override;
};

#include "UniformDataWriter.hpp"
