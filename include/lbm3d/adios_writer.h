#pragma once

#include <string>
#include <map>

#include <adios2.h>
#include "DataManager.h"
#include <TNL/MPI/Comm.h>

template <typename TRAITS>
class ADIOSWriter
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

	// ADIOS2 interface
	adios2::ADIOS adios;
	adios2::IO io;
	adios2::Engine engine;
	std::string filename;

	DataManager* dataManager;
	DataManager::SimulationType simType;
	// data variables recorded for output (mapping of name to dimension)
	std::map<std::string, int> variables;

	void recordVariable(const std::string& name, int dim);

	void addVTKAttributes();

	void addFidesAttributes();

public:
	ADIOSWriter() = delete;

	ADIOSWriter(
		TNL::MPI::Comm communicator,
		const std::string& basename,
		idx3d global,
		idx3d local,
		idx3d offset,
		point_t physOrigin,
		real physDl,
		int cycle,
		DataManager& dataManager,
		DataManager::SimulationType simType
	);

	template <typename T>
	void write(std::string varName, T val);

	template <typename T>
	void write(std::string varName, std::vector<T>& val, int dim);

	~ADIOSWriter();
};

#include "adios_writer.hpp"
