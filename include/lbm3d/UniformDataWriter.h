#pragma once

#include <map>
#include <string>

#include "DataManager.h"

template <typename TRAITS>
class UniformDataWriter
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

	// data variables recorded for output (mapping of name to dimension)
	std::map<std::string, int> variables;

	// DataManager reference
	DataManager* dataManager;
	const std::string& simType;

	void recordVariable(const std::string& name, int dim);

	void addVTKAttributes();

	void addFidesAttributes();

public:
	UniformDataWriter() = delete;

	UniformDataWriter(idx3d global, idx3d local, idx3d offset, point_t physOrigin, real physDl, DataManager& dataManager, const std::string& simType);

	template <typename T>
	void write(std::string varName, T val);

	template <typename T>
	void write(std::string varName, std::vector<T>& val, int dim);

	~UniformDataWriter();
};

#include "UniformDataWriter.hpp"
