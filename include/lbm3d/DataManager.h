#pragma once

#include <string>
#include <map>
#include <any>
#include <adios2.h>
#include <stdexcept>
#include <memory>
#include <fmt/format.h>
#include <spdlog/spdlog.h>

class DataManager
{
public:
	DataManager(adios2::ADIOS* adios)
	: adios(adios)
	{
		defaultIO = adios->DeclareIO("Output");
	}

	void initEngine(const std::string& name, adios2::Mode mode = adios2::Mode::Write)
	{
		if (engines_.count(name) > 0 && current_mode_[name] != mode) {
			engines_[name]->Close();
			engines_.erase(name);
		}
		else if (engines_.count(name) > 0 && current_mode_[name] == mode) {
			spdlog::trace("Engine '{}' already initialized with requested mode.", name);
			engine_ = engines_[name].get();
			return;
		}
		//spdlog::warn("engine closed, count: {}", engines_.size());

		if (engines_.count(name) == 0) {
			std::string filename = name;
			adios2::IO io_;
			if (ios_.count(name) != 0) {
				io_ = ios_[name];
			}
			else {
				io_ = adios->DeclareIO(fmt::format("IO_{}", name));	 //vzit to co za poslednim lomitkem
																	 //
				if (! io_.InConfigFile()) {
					io_.SetEngine(defaultIO.EngineType());
					io_.SetParameters(defaultIO.Parameters());
					spdlog::warn("{}", defaultIO.EngineType());
				}
				ios_[name] = io_;
			}
			if (io_.EngineType().substr(0, 2) == "BP")
				filename += ".bp";
			engines_[name] = std::make_unique<adios2::Engine>(io_.Open(filename, mode));

			current_mode_[name] = mode;
			{
				variables_[name] = {};
				variable_dimensions_[name] = {};
			}
		}

		engine_ = engines_[name].get();
	}

	void ChangeEnginesToAppend()
	{
		for (auto& [name, engine] : engines_) {
			if (engine && current_mode_[name] == adios2::Mode::Write) {
				initEngine(name, adios2::Mode::Append);
				// engines_[name]->Close();
				// engines_[name] = std::make_unique<adios2::Engine>(ios_[name].Open(name, adios2::Mode::Append));
				// current_mode_[name] = adios2::Mode::Append;
			}
		}
	}

	adios2::Engine& getEngine(const std::string& name)
	{
		if (! engines_[name]) {
			throw std::runtime_error("Engine not initialized for this simulation type");
		}
		return *engines_[name];
	}

	adios2::Mode getEngineMode(const std::string& name)
	{
		if (! engines_[name]) {
			throw std::runtime_error("Engine not initialized for this simulation type");
		}
		return current_mode_[name];
	}

	// Explicitly close a specific engine
	void closeEngine(const std::string& name)
	{
		if (engines_.count(name) > 0) {
			engines_[name]->Close();
			engines_.erase(name);
			// ios_.erase(name);
			variables_[name].clear();
			variable_dimensions_[name].clear();
			spdlog::debug("Closed engine for {}", name);
		}
	}

	// Close all engines
	void closeAllEngines()
	{
		for (auto& [name, engine] : engines_) {
			if (engine) {
				closeEngine(name);
				//spdlog::warn("Closed engine for {}", name);
			}
		}
		engines_.clear();
		// ios_.clear();
		variables_.clear();
		variable_dimensions_.clear();
	}

	~DataManager()
	{
		closeAllEngines();
	}

	DataManager(const DataManager&) = delete;
	DataManager& operator=(const DataManager&) = delete;

	DataManager(DataManager&&) noexcept = default;
	DataManager& operator=(DataManager&&) noexcept = default;

	template <typename T>
	void defineData(const std::string& name, const adios2::Dims& shape, const adios2::Dims& start, const adios2::Dims& count, const std::string& type)
	{
		// Check if variable already exists in IO object
		auto var = ios_[type].InquireVariable<T>(name);

		if (! var) {
			// If not, define a new variable
			var = ios_[type].DefineVariable<T>(name, shape, start, count);
			spdlog::debug("Defined new variable {}", name);
		}
		else {
			// If it exists, update the shape, start, and count
			var.SetShape(shape);
			var.SetSelection({start, count});
			spdlog::debug("Reusing existing variable {}", name);
		}

		variables_[type][name] = var;
		variable_dimensions_[type][name] = static_cast<int>(shape.size());
	}

	template <typename T>
	void defineData(const std::string& name, const std::string& type)
	{
		// Check if variable already exists in IO object
		auto var = ios_[type].InquireVariable<T>(name);

		if (! var) {
			// If not, define a new variable
			var = ios_[type].DefineVariable<T>(name);
			spdlog::debug("Defined new variable {}", name);
		}

		variables_[type][name] = var;
		variable_dimensions_[type][name] = 0;
	}

	template <typename T>
	void defineAttribute(const std::string& name, const T& data, const std::string& type)
	{
		// Allow modification of attributes
		ios_[type].DefineAttribute<T>(name, data, "", "/", true);
	}

	template <typename T>
	void defineAttribute(const std::string& name, const T* data, size_t size, const std::string& type)
	{
		// Allow modification of attributes
		ios_[type].DefineAttribute<T>(name, data, size, "", "/", true);
	}

	template <typename T>
	void outputData(const std::string& name, const T& data, const std::string& type)
	{
		if (engines_[type] == nullptr) {
			throw std::runtime_error("Engine not initialized for this simulation type");
		}

		if (current_mode_[type] != adios2::Mode::Write && current_mode_[type] != adios2::Mode::Append) {
			throw std::runtime_error("Engine not in write mode");
		}

		auto it = variables_[type].find(name);
		if (it != variables_[type].end()) {
			adios2::Variable<T> var = std::any_cast<adios2::Variable<T>>(it->second);
			engines_[type]->Put(var, data);
			engines_[type]->PerformPuts();
		}
		else {
			throw std::runtime_error(fmt::format("Variable \"{}\" not found", name));
		}
	}

	template <typename T>
	void outputData(const std::string& name, const T* data, const std::string& type)
	{
		if (engines_[type] == nullptr) {
			throw std::runtime_error("Engine not initialized for this simulation type");
		}

		if (current_mode_[type] != adios2::Mode::Write && current_mode_[type] != adios2::Mode::Append) {
			throw std::runtime_error("Engine not in write mode");
		}

		auto it = variables_[type].find(name);
		if (it != variables_[type].end()) {
			adios2::Variable<T> var = std::any_cast<adios2::Variable<T>>(it->second);
			engines_[type]->Put(var, data);
			engines_[type]->PerformPuts();
		}
		else {
			throw std::runtime_error(fmt::format("Variable \"{}\" not found", name));
		}
	}

	// Methods for reading

	template <typename T>
	T readAttribute(const std::string& name, const std::string& type)
	{
		if (engines_[type] == nullptr) {
			throw std::runtime_error("Engine not initialized for this simulation type");
		}

		if (current_mode_[type] != adios2::Mode::Read) {
			throw std::runtime_error("Engine not in read mode");
		}

		auto attr = ios_[type].InquireAttribute<T>(name);
		if (! attr) {
			throw std::runtime_error(fmt::format("Attribute \"{}\" not found", name));
		}

		return attr.Data()[0];
	}

	template <typename T>
	std::vector<T> readAttributeArray(const std::string& name, const std::string& type)
	{
		if (engines_[type] == nullptr) {
			throw std::runtime_error("Engine not initialized for this simulation type");
		}

		if (current_mode_[type] != adios2::Mode::Read) {
			throw std::runtime_error("Engine not in read mode");
		}

		auto attr = ios_[type].InquireAttribute<T>(name);
		if (! attr) {
			throw std::runtime_error(fmt::format("Attribute \"{}\" not found", name));
		}

		return std::vector<T>(attr.Data(), attr.Data() + attr.Size());
	}

	template <typename T>
	void readVariable(const std::string& name, T* data, const std::string& type)
	{
		if (engines_[type] == nullptr) {
			throw std::runtime_error("Engine not initialized for this simulation type");
		}

		if (current_mode_[type] != adios2::Mode::Read) {
			throw std::runtime_error("Engine not in read mode");
		}

		// Try to get the variable
		adios2::Variable<T> var;
		auto it = variables_[type].find(name);
		if (it != variables_[type].end()) {
			var = std::any_cast<adios2::Variable<T>>(it->second);
		}
		else {
			// If not found in our map, try to inquire it from ADIOS2
			var = ios_[type].InquireVariable<T>(name);
			if (! var) {
				throw std::runtime_error(fmt::format("Variable \"{}\" not found", name));
			}
			variables_[type][name] = var;
		}

		engines_[type]->Get(var, data);
		engines_[type]->PerformGets();
	}

	void beginStep(const std::string& type)
	{
		if (engines_[type] == nullptr) {
			throw std::runtime_error("Engine not initialized for this simulation type");
		}
		engine_ = engines_[type].get();
		engine_->BeginStep();
	}

	void performPutsAndStep(const std::string& type)
	{
		if (engines_[type] == nullptr) {
			throw std::runtime_error("Engine not initialized for this simulation type");
		}
		engine_ = engines_[type].get();

		if (current_mode_[type] == adios2::Mode::Write || current_mode_[type] == adios2::Mode::Append) {
			engine_->PerformPuts();
		}
		else {
			engine_->PerformGets();
		}

		engine_->EndStep();
	}

	const std::map<std::string, std::any>& getVariables(const std::string& type) const
	{
		auto it = variables_.find(type);
		if (it != variables_.end()) {
			return it->second;
		}
		throw std::runtime_error("No variables defined for this simulation type");
	}

	const std::map<std::string, int>& getVariableDimensions(const std::string& type) const
	{
		auto it = variable_dimensions_.find(type);
		if (it != variable_dimensions_.end()) {
			return it->second;
		}
		throw std::runtime_error("No variable dimensions defined for this simulation type");
	}

private:
	adios2::ADIOS* adios;
	std::map<std::string, adios2::IO> ios_;
	adios2::IO defaultIO;
	std::map<std::string, std::unique_ptr<adios2::Engine>> engines_;
	std::map<std::string, std::map<std::string, std::any>> variables_;
	std::map<std::string, std::map<std::string, int>> variable_dimensions_;
	std::map<std::string, adios2::Mode> current_mode_;	// Track if in read or write mode
	adios2::Engine* engine_ = nullptr;
};
