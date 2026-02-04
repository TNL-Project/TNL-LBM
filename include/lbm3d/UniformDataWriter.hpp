#pragma once

#include <fmt/format.h>

#include "UniformDataWriter.h"

template <typename TRAITS>
UniformDataWriter<TRAITS>::UniformDataWriter(
	idx3d global, idx3d local, idx3d offset, point_t physOrigin, real physDl, DataManager& dataManager, std::string ioName
)
: DataWriter<TRAITS>::DataWriter(dataManager, std::move(ioName)),
  global(global),
  local(local),
  offset(offset),
  physOrigin(physOrigin),
  physDl(physDl)
{}

template <typename TRAITS>
template <typename T>
void UniformDataWriter<TRAITS>::write(std::string varName, T val)
{
	this->recordVariable(varName, 0);

	this->dataManager->template outputData<T>(varName, val, this->ioName);
}

template <typename TRAITS>
template <typename DataSource>
void UniformDataWriter<TRAITS>::write(const std::string& varName, const DataSource& src, idx3d begin, idx3d end)
{
	// NOTE: vector fields are not handled because the VTX reader does not support vector fields on ImageData
	// https://github.com/ornladios/ADIOS2/discussions/4117
	// NOTE: we need to do buffering because ADIOS2 enforces row-major format
	// 		 and our DistributedNDArray has overlaps that do not map to ADIOS2 nicely

	using ValueType = std::decay_t<decltype(src(0, 0, 0))>;

	const std::size_t size = local.x() * local.y() * local.z();
	const std::size_t size2 = (end.x() - begin.x()) * (end.y() - begin.y()) * (end.z() - begin.z());
	if (size != size2)
		throw std::invalid_argument(
			fmt::format(
				"UniformDataWriter::write_cut: size mismatch: {} != {} (begin = [{}, {}, {}], end = [{}, {}, {}])",
				size,
				size2,
				begin.x(),
				begin.y(),
				begin.z(),
				end.x(),
				end.y(),
				end.z()
			)
		);

	// make internal buffer that is kept until EndStep()
	auto& buffer = this->template newBuffer<ValueType>(size);
	buffer.resize(size);

	for (idx z = begin.z(); z < end.z(); z++)
		for (idx y = begin.y(); y < end.y(); y++)
			for (idx x = begin.x(); x < end.x(); x++) {
				const idx index = (z - begin.z()) * (end.y() - begin.y()) * local.x() + (y - begin.y()) * local.x() + x - begin.x();
				buffer[index] = src(x, y, z);
			}

	this->recordVariable(varName, 1);
	this->dataManager->template outputData<ValueType>(varName, buffer.data(), this->ioName);
}

template <typename TRAITS>
void UniformDataWriter<TRAITS>::addVTKAttributes()
{
	const std::string extentG = fmt::format("0 {} 0 {} 0 {}", global.z(), global.y(), global.x());
	const std::string extentL = fmt::format("0 {} 0 {} 0 {}", local.z(), local.y(), local.x());
	const std::string origin = fmt::format("{} {} {}", physOrigin.x(), physOrigin.y(), physOrigin.z());
	const std::string spacing = fmt::format("{} {} {}", physDl, physDl, physDl);

	std::string dataArrays;
	for (const auto& [name, dim] : this->variables) {
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

	this->dataManager->template defineAttribute<std::string>("vtk.xml", dataModel, this->ioName);
}

template <typename TRAITS>
void UniformDataWriter<TRAITS>::addFidesAttributes()
{
	// add attributes for Fides
	// https://fides.readthedocs.io/en/latest/components/components.html#uniform-data-model
	this->dataManager->template defineAttribute<std::string>("Fides_Data_Model", "uniform", this->ioName);
	this->dataManager->template defineAttribute<typename point_t::ValueType>("Fides_Origin", &physOrigin[0], point_t::getSize(), this->ioName);
	real spacing[3] = {physDl, physDl, physDl};
	this->dataManager->template defineAttribute<real>("Fides_Spacing", spacing, 3, this->ioName);

	bool dimension_variable_set = false;
	std::vector<std::string> variable_list;
	std::vector<std::string> variable_associations;
	for (const auto& [name, dim] : this->variables) {
		if (dim > 0) {
			if (! dimension_variable_set) {
				// NOTE: Fides_Dimension_Variable must refer to a scalar variable
				// https://gitlab.kitware.com/vtk/fides/-/issues/22
				// FIXME: Fides requires this variable to be PointData for sizing,
				// but PointData leads to visual "gaps" between subdomains in Paraview
				// https://github.com/ornladios/ADIOS2-Examples/issues/90
				this->dataManager->template defineAttribute<std::string>("Fides_Dimension_Variable", name, this->ioName);
				dimension_variable_set = true;
			}
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
UniformDataWriter<TRAITS>::~UniformDataWriter()
{
	this->finalize();
}
