#pragma once

#include <any>
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

	// internal storage for array variables
	std::vector<std::any> buffers;

	// data variables recorded for output (mapping of name to dimension)
	std::map<std::string, int> variables;

	template <typename T>
	std::vector<T>& newBuffer(std::size_t reserve = 0)
	{
		std::any& any_buffer = buffers.emplace_back(std::make_any<std::vector<T>>());
		std::vector<T>& buffer = std::any_cast<std::vector<T>&>(any_buffer);
		if (reserve > 0)
			buffer.reserve(reserve);
		return buffer;
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
	void endStep()
	{
		if (! variables.empty()) {
			addVTKAttributes();
			addFidesAttributes();
		}
		dataManager->performPutsAndStep(ioName);
	}

public:
	DataWriter() = delete;

	DataWriter(DataManager& dataManager, std::string ioName)
	: dataManager(&dataManager),
	  ioName(std::move(ioName))
	{
		dataManager.beginStep(this->ioName);
	}

	virtual ~DataWriter() = default;
};
