#pragma once

#include <map>
#include <string>

#include "DataManager.h"
#include "DataWriter.h"

template <typename TRAITS>
class UnstructuredPointsWriter : public DataWriter<TRAITS>
{
private:
	using idx = typename TRAITS::idx;
	using real = typename TRAITS::real;
	using point_t = typename TRAITS::point_t;
	using idx3d = typename TRAITS::idx3d;

	// default data model attributes
	std::string coordinates_variable = "points";
	std::string connectivity_variable = "connectivity";
	std::string cell_types_variable = "cell_types";
	std::string num_points_variable = "number_of_points";

	// data variables recorded for output (mapping of name to dimension)
	std::map<std::string, int> variables;

	// DataManager reference
	DataManager* dataManager;
	const std::string& simType;

	void recordVariable(const std::string& name, int dim);

	void addVTKAttributes();

	void addFidesAttributes();

public:
	UnstructuredPointsWriter() = delete;

	UnstructuredPointsWriter(
		DataManager& dataManager,
		const std::string& simType,
		std::string coordinates_variable = "points",
		std::string connectivity_variable = "connectivity",
		std::string cell_types_variable = "cell_types"
	);

	template <typename T>
	void write(std::string varName, T val);

	template <typename T>
	void write(std::string varName, std::vector<T>& val, int dim, idx num_points);

	~UnstructuredPointsWriter();
};

#include "UnstructuredPointsWriter.hpp"
