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

	template <typename T>
	void write(std::string varName, T val);

	// DataSource is a 3D array of type T or lambda function with signature src(idx x, idx y, idx z) -> T
	template <typename DataSource>
	void write(const std::string& varName, const DataSource& src, idx3d begin, idx3d end);

	virtual ~UniformDataWriter();
};

#include "UniformDataWriter.hpp"
