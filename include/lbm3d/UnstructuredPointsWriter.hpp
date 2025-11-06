#pragma once

#include "UnstructuredPointsWriter.h"

template <typename TRAITS>
UnstructuredPointsWriter<TRAITS>::UnstructuredPointsWriter(
	DataManager& dataManager,
	const std::string& simType,
	std::string coordinates_variable,
	std::string connectivity_variable,
	std::string cell_types_variable
)
: dataManager(&dataManager),
  simType(simType),
  coordinates_variable(std::move(coordinates_variable)),
  connectivity_variable(std::move(connectivity_variable)),
  cell_types_variable(std::move(cell_types_variable))
{
	dataManager.beginStep(simType);
}

template <typename TRAITS>
template <typename T>
void UnstructuredPointsWriter<TRAITS>::write(std::string varName, T val)
{
	if (! dataManager->isVariableDefined<T>(varName, simType)) {
		dataManager->defineData<T>(varName, simType);
	}

	dataManager->outputData<T>(varName, val, simType);
}

template <typename TRAITS>
template <typename T>
void UnstructuredPointsWriter<TRAITS>::write(std::string varName, std::vector<T>& val, int dim, idx num_points)
{
	recordVariable(varName, dim);

	if (! dataManager->isVariableDefined<T>(varName, simType)) {
		// TODO: make it distributed
		adios2::Dims shape{static_cast<std::size_t>(num_points), std::size_t(dim)};
		adios2::Dims start{static_cast<std::size_t>(0), static_cast<std::size_t>(0)};
		adios2::Dims count{static_cast<std::size_t>(num_points), static_cast<std::size_t>(dim)};
		dataManager->defineData<T>(varName, shape, start, count, simType);
	}

	dataManager->outputData<T>(varName, val.data(), simType);
}

template <typename TRAITS>
void UnstructuredPointsWriter<TRAITS>::recordVariable(const std::string& name, int dim)
{
	if (variables.count(name) > 0)
		throw std::invalid_argument("Variable \"" + name + "\" is already defined.");
	if (dim != 0 && dim != 1 && dim != 3)
		throw std::invalid_argument("Invalid dimension of \"" + name + "\"(" + std::to_string(dim) + ").");

	variables[name] = dim;
}

template <typename TRAITS>
void UnstructuredPointsWriter<TRAITS>::addVTKAttributes()
{
	std::string dataArrays;
	for (const auto& [name, dim] : variables) {
		if (name == coordinates_variable || name == connectivity_variable || name == cell_types_variable)
			continue;
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
        <VTKFile type="UnstructuredGrid" version="0.1" byte_order="LittleEndian">
			<UnstructuredGrid>
				<Piece NumberOfPoints=")"
								+ num_points_variable + R"(" NumberOfCells=")" + num_points_variable + R"(" >
					<Points>
						<DataArray Name=")"
								+ coordinates_variable + R"(" />
					</Points>
					<Cells>
						<DataArray Name=")"
								+ connectivity_variable + R"(" />
						<DataArray Name=")"
								+ cell_types_variable + R"(" />
					</Cells>
					<CellData Scalars="data">)"
								+ dataArrays + R"(
					</CellData>
				</Piece>
			</UnstructuredGrid>
        </VTKFile>)";

	dataManager->defineAttribute<std::string>("vtk.xml", dataModel, simType);
}

template <typename TRAITS>
void UnstructuredPointsWriter<TRAITS>::addFidesAttributes()
{
	// add attributes for Fides
	// https://fides.readthedocs.io/en/latest/components/components.html#unstructured-with-single-cell-type-data-model
	dataManager->defineAttribute<std::string>("Fides_Data_Model", "unstructured_single", simType);
	dataManager->defineAttribute<std::string>("Fides_Cell_Type", "vertex", simType);
	dataManager->defineAttribute<std::string>("Fides_Coordinates_Variable", coordinates_variable, simType);
	dataManager->defineAttribute<std::string>("Fides_Connectivity_Variable", connectivity_variable, simType);

	std::vector<std::string> variable_list;
	std::vector<std::string> variable_associations;
	for (const auto& [name, dim] : variables) {
		if (name == coordinates_variable || name == connectivity_variable)
			continue;
		if (dim > 0) {
			variable_list.push_back(name);
			variable_associations.emplace_back("points");
		}
	}
	dataManager->defineAttribute<std::string>("Fides_Variable_List", variable_list.data(), variable_list.size(), simType);
	dataManager->defineAttribute<std::string>("Fides_Variable_Associations", variable_associations.data(), variable_associations.size(), simType);
	dataManager->defineAttribute<std::string>("Fides_Time_Variable", "TIME", simType);
}

template <typename TRAITS>
UnstructuredPointsWriter<TRAITS>::~UnstructuredPointsWriter()
{
	if (! variables.empty()) {
		addVTKAttributes();
		addFidesAttributes();
	}
	dataManager->performPutsAndStep(simType);
}
