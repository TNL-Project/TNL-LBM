#pragma once

#include <fmt/format.h>

#include "UniformDataWriter.h"

template <typename TRAITS>
UniformDataWriter<TRAITS>::UniformDataWriter(
	idx3d global, idx3d local, idx3d offset, point_t physOrigin, real physDl, DataManager& dataManager, const std::string& simType
)
: simType(simType)
{
	this->global = global;
	this->local = local;
	this->offset = offset;
	this->physOrigin = physOrigin;
	this->physDl = physDl;
	this->dataManager = &dataManager;
	dataManager.beginStep(simType);
}

template <typename TRAITS>
template <typename T>
void UniformDataWriter<TRAITS>::write(std::string varName, T val)
{
	if (! dataManager->isVariableDefined<T>(varName, simType)) {
		dataManager->defineData<T>(varName, simType);
	}

	dataManager->outputData<T>(varName, val, simType);
}

template <typename TRAITS>
template <typename T>
void UniformDataWriter<TRAITS>::write(std::string varName, std::vector<T>& val, int dim)
{
	recordVariable(varName, dim);

	if (! dataManager->isVariableDefined<T>(varName, simType)) {
		adios2::Dims shape{static_cast<std::size_t>(global.z()), static_cast<std::size_t>(global.y()), static_cast<std::size_t>(global.x())};
		adios2::Dims start{static_cast<std::size_t>(offset.z()), static_cast<std::size_t>(offset.y()), static_cast<std::size_t>(offset.x())};
		adios2::Dims count{static_cast<std::size_t>(local.z()), static_cast<std::size_t>(local.y()), static_cast<std::size_t>(local.x())};
		dataManager->defineData<T>(varName, shape, start, count, simType);
	}

	// keep internal copy of the data until EndStep()
	auto& buffer = this->template newBuffer<T>(val.size());
	// Avoid extra copy: move data into internal buffer (val stays usable as an empty preallocated buffer)
	buffer.swap(val);
	dataManager->outputData<T>(varName, buffer.data(), simType);
}

template <typename TRAITS>
void UniformDataWriter<TRAITS>::recordVariable(const std::string& name, int dim)
{
	if (variables.count(name) > 0)
		throw std::invalid_argument("Variable \"" + name + "\" is already defined.");
	if (dim != 0 && dim != 1 && dim != 3)
		throw std::invalid_argument("Invalid dimension of \"" + name + "\"(" + std::to_string(dim) + ").");

	variables[name] = dim;
}

template <typename TRAITS>
void UniformDataWriter<TRAITS>::addVTKAttributes()
{
	const std::string extentG = fmt::format("0 {} 0 {} 0 {}", global.z(), global.y(), global.x());
	const std::string extentL = fmt::format("0 {} 0 {} 0 {}", local.z(), local.y(), local.x());
	const std::string origin = fmt::format("{} {} {}", physOrigin.x(), physOrigin.y(), physOrigin.z());
	const std::string spacing = fmt::format("{} {} {}", physDl, physDl, physDl);

	std::string dataArrays;
	for (const auto& [name, dim] : variables) {
		switch (dim) {
			case 0:
				dataArrays += "<DataArray Name=\"" + name + "\"> " + name + " </DataArray>\n";
				break;
			case 1:
				dataArrays += "<DataArray Name=\"" + name + "\"/>\n";
				break;
			case 3:
				dataArrays += "<DataArray Name=\"" + name + "\"/>\n";
				//dataArrays += "<DataArray Name=\"" + name + "\" NumberOfComponents=\"3\"/>\n";
				break;
		}
	}

	// https://adios2.readthedocs.io/en/latest/ecosystem/visualization.html#saving-the-vtk-xml-data-model
	const std::string dataModel = R"(
        <?xml version="1.0"?>
        <VTKFile type="ImageData" version="0.1" byte_order="LittleEndian">
            <ImageData WholeExtent=")"
								+ extentG + R"(" Origin=")" + origin + R"(" Spacing=")" + spacing + R"(">
                <Piece Extent=")"
								+ extentL + R"(">
                    <CellData Scalars="data">)"
								+ dataArrays + R"(
                    </CellData>
                </Piece>
            </ImageData>
        </VTKFile>)";

	dataManager->defineAttribute<std::string>("vtk.xml", dataModel, simType);
}

template <typename TRAITS>
void UniformDataWriter<TRAITS>::addFidesAttributes()
{
	// add attributes for Fides
	// https://fides.readthedocs.io/en/latest/components/components.html#uniform-data-model
	dataManager->defineAttribute<std::string>("Fides_Data_Model", "uniform", simType);
	dataManager->defineAttribute<typename point_t::ValueType>("Fides_Origin", &physOrigin[0], point_t::getSize(), simType);
	real spacing[3] = {physDl, physDl, physDl};
	dataManager->defineAttribute<real>("Fides_Spacing", spacing, 3, simType);

	bool dimension_variable_set = false;
	std::vector<std::string> variable_list;
	std::vector<std::string> variable_associations;
	for (const auto& [name, dim] : variables) {
		if (dim > 0) {
			if (! dimension_variable_set) {
				// NOTE: Fides_Dimension_Variable must refer to a scalar variable
				// https://gitlab.kitware.com/vtk/fides/-/issues/22
				// FIXME: Fides requires this variable to be PointData for sizing,
				// but PointData leads to visual "gaps" between subdomains in Paraview
				// https://github.com/ornladios/ADIOS2-Examples/issues/90
				dataManager->defineAttribute<std::string>("Fides_Dimension_Variable", name, simType);
				dimension_variable_set = true;
			}
			variable_list.push_back(name);
			variable_associations.emplace_back("points");
		}
	}
	dataManager->defineAttribute<std::string>("Fides_Variable_List", variable_list.data(), variable_list.size(), simType);
	dataManager->defineAttribute<std::string>("Fides_Variable_Associations", variable_associations.data(), variable_associations.size(), simType);
	dataManager->defineAttribute<std::string>("Fides_Time_Variable", "TIME", simType);
}

template <typename TRAITS>
UniformDataWriter<TRAITS>::~UniformDataWriter()
{
	if (! variables.empty()) {
		addVTKAttributes();
		addFidesAttributes();
	}
	dataManager->performPutsAndStep(simType);
}
