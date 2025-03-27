#pragma once

#include <adios2.h>
#include <fmt/core.h>
#include "DataManager.h"

class CheckpointManager
{
private:
	adios2::ADIOS* adios;
	DataManager* dataManager;
	std::string currentCheckpointName;
	bool isActive = false;
	adios2::Mode currentMode = adios2::Mode::Undefined;

public:
	CheckpointManager(adios2::ADIOS& adios)
	: adios(&adios),
	  dataManager(nullptr)
	{}

	// Set DataManager to use for checkpoint operations
	void setDataManager(DataManager& dataManager)
	{
		this->dataManager = &dataManager;
	}

	// Open a checkpoint file for writing or reading
	void start(const std::string& filename, adios2::Mode mode)
	{
		if (dataManager == nullptr) {
			throw std::runtime_error("DataManager not set in CheckpointManager");
		}

		currentCheckpointName = filename;
		dataManager->initEngine(currentCheckpointName, mode);
		currentMode = mode;

		isActive = true;
		dataManager->beginStep(currentCheckpointName);
	}

	// Perform all deferred operations
	void performDeferred()
	{
		if (! isActive || dataManager == nullptr) {
			return;
		}
		// This is handled automatically by DataManager
	}

	// End the checkpoint session
	void finalize()
	{
		if (! isActive || dataManager == nullptr) {
			return;
		}

		dataManager->performPutsAndStep(currentCheckpointName);
		isActive = false;
		currentMode = adios2::Mode::Undefined;
		dataManager->closeEngine(currentCheckpointName);
	}

	// Save or load attributes
	template <typename T, typename CastToType = T>
	void saveLoadAttribute(const std::string& name, T& variable)
	{
		if (! isActive || dataManager == nullptr) {
			return;
		}

		if (currentMode == adios2::Mode::Write) {
			// When writing, define and output the attribute
			dataManager->defineAttribute<CastToType>(name, static_cast<CastToType>(variable), currentCheckpointName);
		}
		else if (currentMode == adios2::Mode::Read) {
			// Read the attribute
			try {
				CastToType value = dataManager->readAttribute<CastToType>(name, currentCheckpointName);
				variable = static_cast<T>(value);
			}
			catch (const std::runtime_error& e) {
				// Handle the case when the attribute doesn't exist
				spdlog::warn("Failed to read attribute {}: {}", name, e.what());
			}
		}
	}

	// Save or load variables
	template <typename LBM_BLOCK, typename Array>
	void saveLoadVariable(std::string name, LBM_BLOCK& block, Array& array)
	{
		if (! isActive || dataManager == nullptr) {
			return;
		}

		using T = typename Array::ValueType;

		// NOTE: For checkpointing functionality we need to save overlaps
		// which don't map nicely to the ADIOS2 model. Hence, we save/load
		// each block as a separate variable.
		adios2::Dims shape;
		adios2::Dims start;
		adios2::Dims count;
		//if constexpr (Array::getDimension() == 4) {
		//	const std::size_t N = array.template getSize<0>();
		//	shape = adios2::Dims({N, size_t(block.global.z()), size_t(block.global.y()), size_t(block.global.x())});
		//	start = adios2::Dims({0, size_t(block.offset.z()), size_t(block.offset.y()), size_t(block.offset.x())});
		//	count = adios2::Dims({N, size_t(block.local.z()), size_t(block.local.y()), size_t(block.local.x())});
		//}
		//else {
		//	shape = adios2::Dims({size_t(block.global.z()), size_t(block.global.y()), size_t(block.global.x())});
		//	start = adios2::Dims({size_t(block.offset.z()), size_t(block.offset.y()), size_t(block.offset.x())});
		//	count = adios2::Dims({size_t(block.local.z()), size_t(block.local.y()), size_t(block.local.x())});
		//}
		start = {0};
#ifdef HAVE_MPI
		shape = count = {std::size_t(array.getLocalStorageSize())};
#else
		shape = count = {std::size_t(array.getStorageSize())};
#endif
		name += fmt::format("_block_{}", block.id);

		if (currentMode == adios2::Mode::Write) {
			// Define and output the variable
			if (dataManager->getVariables(currentCheckpointName).count(name) == 0) {
				dataManager->defineData<T>(name, shape, start, count, currentCheckpointName);
			}

			dataManager->outputData<T>(name, array.getData(), currentCheckpointName);
		}
		else if (currentMode == adios2::Mode::Read) {
			// Read the variable
			try {
				dataManager->readVariable<T>(name, array.getData(), currentCheckpointName);
			}
			catch (const std::runtime_error& e) {
				// Handle the case when the variable doesn't exist
				spdlog::warn("Failed to read variable {}: {}", name, e.what());
			}
		}
	}

	// Save or load local arrays
	template <typename Array>
	void saveLoadLocalArray(std::string name, int rank, Array& array)
	{
		if (! isActive || ! dataManager) {
			return;
		}

		using T = typename Array::ValueType;

		adios2::Dims shape = {std::size_t(array.getStorageSize())};
		adios2::Dims start = {0};
		adios2::Dims count = shape;
		name += fmt::format("_rank_{}", rank);

		if (currentMode == adios2::Mode::Write) {
			// Define and output the variable
			if (dataManager->getVariables(currentCheckpointName).count(name) == 0) {
				dataManager->defineData<T>(name, shape, start, count, currentCheckpointName);
			}

			dataManager->outputData<T>(name, array.getData(), currentCheckpointName);
		}
		else if (currentMode == adios2::Mode::Read) {
			// Read the variable
			try {
				dataManager->readVariable<T>(name, array.getData(), currentCheckpointName);
			}
			catch (const std::runtime_error& e) {
				// Handle the case when the variable doesn't exist
				spdlog::warn("Failed to read local array {}: {}", name, e.what());
			}
		}
	}
};
