#pragma once

#include <map>
#include <string>
#include <vector>

#include <fmt/format.h>

#include "DataManager.h"

template <typename TRAITS>
class DataWriter
{
protected:
	using idx = typename TRAITS::idx;

	// DataManager reference
	DataManager* dataManager;

	// IO name in DataManager
	std::string ioName;

	// data variables recorded for output (mapping of name to dimension)
	std::map<std::string, int> variables;

	// plugin doesnt support now
	// template <typename T>
	// typename adios2::Variable<T>::Span newBuffer(const std::string& varName)
	// {
	// 	adios2::Engine& engine = dataManager->getEngine(ioName);
	// 	adios2::Variable<T> var = dataManager->getVariable<T>(varName, ioName);
	// 	typename adios2::Variable<T>::Span buffer = engine.Put(var);
	// 	return buffer;
	// }

	template <typename T>
	std::vector<T>& newBuffer(std::size_t reserve = 0)
	{
		return dataManager->newStepBuffer<T>(reserve);
	}
	void recordVariable(const std::string& name, int dim)
	{
		if (variables.count(name) > 0)
			throw std::invalid_argument(fmt::format("Variable \"{}\" is already defined.", name));
		if (dim != 0 && dim != 1 && dim != 3)
			throw std::invalid_argument(fmt::format("Invalid dimension of variable \"{}\": {}.", name, dim));

		variables[name] = dim;
	}

	virtual void addVTKAttributes() {}

	virtual void addFidesAttributes() {}

	// Calling virtual methods does not work from the destructor, so derived classes must
	// call this method from their own destructors.
	void finalize()
	{
		if (! variables.empty()) {
			addVTKAttributes();
			addFidesAttributes();
		}
		dataManager->performPuts(ioName);
	}

public:
	DataWriter() = delete;

	DataWriter(DataManager& dataManager, std::string ioName)
	: dataManager(&dataManager),
	  ioName(std::move(ioName))
	{}

	virtual ~DataWriter() = default;
};
