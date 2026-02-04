#pragma once

#include "UnstructuredPointsWriter.h"

template <typename TRAITS>
UnstructuredPointsWriter<TRAITS>::UnstructuredPointsWriter(
	DataManager& dataManager, std::string ioName, std::string coordinates_variable, std::string connectivity_variable, std::string cell_types_variable
)
: DataWriter<TRAITS>::DataWriter(dataManager, std::move(ioName)),
  coordinates_variable(std::move(coordinates_variable)),
  connectivity_variable(std::move(connectivity_variable)),
  cell_types_variable(std::move(cell_types_variable))
{}

template <typename TRAITS>
template <typename T>
void UnstructuredPointsWriter<TRAITS>::write(std::string varName, T val)
{
	this->recordVariable(varName, 0);

	this->dataManager->template outputData<T>(varName, val, this->ioName);
}

template <typename TRAITS>
template <typename T>
void UnstructuredPointsWriter<TRAITS>::write(std::string varName, std::vector<T>& val, int dim, idx num_points)
{
	this->recordVariable(varName, dim);

	// keep internal copy of the data until EndStep()
	auto& buffer = this->template newBuffer<T>(val.size());
	// Avoid extra copy: move data into internal buffer (val stays usable as an empty preallocated buffer)
	buffer.swap(val);
	this->dataManager->template outputData<T>(varName, buffer.data(), this->ioName);
}

template <typename TRAITS>
void UnstructuredPointsWriter<TRAITS>::addVTKAttributes()
{
	std::string dataArrays;
	for (const auto& [name, dim] : this->variables) {
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

	this->dataManager->template defineAttribute<std::string>("vtk.xml", dataModel, this->ioName);
}

template <typename TRAITS>
void UnstructuredPointsWriter<TRAITS>::addFidesAttributes()
{
	// add attributes for Fides
	// https://fides.readthedocs.io/en/latest/components/components.html#unstructured-with-single-cell-type-data-model
	this->dataManager->template defineAttribute<std::string>("Fides_Data_Model", "unstructured_single", this->ioName);
	this->dataManager->template defineAttribute<std::string>("Fides_Cell_Type", "vertex", this->ioName);
	this->dataManager->template defineAttribute<std::string>("Fides_Coordinates_Variable", coordinates_variable, this->ioName);
	this->dataManager->template defineAttribute<std::string>("Fides_Connectivity_Variable", connectivity_variable, this->ioName);

	std::vector<std::string> variable_list;
	std::vector<std::string> variable_associations;
	for (const auto& [name, dim] : this->variables) {
		if (name == coordinates_variable || name == connectivity_variable)
			continue;
		if (dim > 0) {
			variable_list.push_back(name);
			variable_associations.emplace_back("points");
		}
	}
	this->dataManager->template defineAttribute<std::string>("Fides_Variable_List", variable_list.data(), variable_list.size(), this->ioName);
	this->dataManager->template defineAttribute<std::string>(
		"Fides_Variable_Associations", variable_associations.data(), variable_associations.size(), this->ioName
	);
	this->dataManager->template defineAttribute<std::string>("Fides_Time_Variable", "TIME", this->ioName);
}

template <typename TRAITS>
UnstructuredPointsWriter<TRAITS>::~UnstructuredPointsWriter()
{
	this->finalize();
}
