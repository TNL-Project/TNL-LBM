#pragma once

#include <string>
#include <map>
#include <adios2.h>
#include <stdexcept>
#include <fmt/format.h>
#include <spdlog/spdlog.h>
#include <magic_enum/magic_enum.hpp>

class DataManager
{
public:
	DataManager(adios2::ADIOS* adios)
	: adios(adios)
	{
		defaultIO = adios->DeclareIO("Output");
	}

	// Prepare IO object without opening engine 
	void prepareIO(const std::string& ioName)
	{
		if (ios_.find(ioName) != ios_.end()) {
			return; // Already prepared
		}
		adios2::IO io_handle = adios->DeclareIO(fmt::format("IO_{}", ioName));
		if (! io_handle.InConfigFile()) {
			io_handle.SetEngine(defaultIO.EngineType());
			io_handle.SetParameters(defaultIO.Parameters());
		}
		ios_.emplace(ioName, io_handle);
		spdlog::info("Prepared IO for '{}' (variables can be defined now)", ioName);
	}

	// Open engine after variables are defined
	void openEngine(const std::string& ioName, adios2::Mode mode = adios2::Mode::Write)
	{
		auto io_it = ios_.find(ioName);
		if (io_it == ios_.end()) {
			prepareIO(ioName);
			io_it = ios_.find(ioName);
		}

		adios2::IO& io_ref = io_it->second;
		const bool is_bp_engine = io_ref.EngineType().rfind("BP", 0) == 0;

		if (! is_bp_engine && mode == adios2::Mode::Append) {
			mode = adios2::Mode::Write;
		}

		// Check if engine already exists and is open
		auto eng_it = engines_.find(ioName);
		if (eng_it != engines_.end() && eng_it->second) {
			if (current_mode_[ioName] == mode) {
				engine_ = &eng_it->second;
				return;
			}
			eng_it->second.Close();
			engines_.erase(ioName);
		}

		std::string filename = ioName;
		if (is_bp_engine) {
			filename += ".bp";
		}

		spdlog::info("Opening engine '{}' (after variable definitions)", ioName);
		engines_[ioName] = io_ref.Open(filename, mode);
		current_mode_[ioName] = mode;
		engine_ = &engines_[ioName];
	}

	void initEngine(const std::string& ioName, adios2::Mode mode = adios2::Mode::Write)
	{
		prepareIO(ioName);
		openEngine(ioName, mode);
	}

	//this function is not necessary right now
	void ChangeEnginesToAppend()
	{
		for (auto& [ioName, engine] : engines_) {
			if (engine && current_mode_[ioName] == adios2::Mode::Write) {
				initEngine(ioName, adios2::Mode::Append);
			}
		}
	}

	adios2::Engine& getEngine(const std::string& ioName)
	{
		if (! engines_[ioName]) {
			throw std::runtime_error("Engine not initialized for this simulation type");
		}
		return engines_[ioName];
	}

	adios2::Mode getEngineMode(const std::string& ioName)
	{
		if (! engines_[ioName]) {
			throw std::runtime_error("Engine not initialized for this simulation type");
		}
		return current_mode_[ioName];
	}

	// Explicitly close a specific engine
	void closeEngine(const std::string& ioName)
	{
		if (engines_.count(ioName) > 0) {
			engines_[ioName].Close();
			engines_.erase(ioName);
			spdlog::debug("Closed engine for {}", ioName);
		}
	}

	// Close all engines
	void closeAllEngines()
	{
		while (! engines_.empty()) {
			closeEngine(engines_.begin()->first);
		}
		engines_.clear();
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
	void
	defineData(const std::string& name, const adios2::Dims& shape, const adios2::Dims& start, const adios2::Dims& count, const std::string& ioName)
	{
		// Check if variable already exists in IO object
		auto var = ios_[ioName].InquireVariable<T>(name);

		if (! var) {
			// If not, define a new variable
			var = ios_[ioName].DefineVariable<T>(name, shape, start, count);
			spdlog::debug("Defined new variable {}", name);
		}
		else {
			// If it exists, update the shape, start, and count
			var.SetShape(shape);
			var.SetSelection({start, count});
			spdlog::debug("Reusing existing variable {}", name);
		}
	}

	template <typename T>
	void defineData(const std::string& name, const std::string& ioName)
	{
		// Check if variable already exists in IO object
		auto var = ios_[ioName].InquireVariable<T>(name);

		if (! var) {
			// If not, define a new variable
			var = ios_[ioName].DefineVariable<T>(name);
			spdlog::debug("Defined new variable {}", name);
		}
	}

	template <typename T>
	void defineAttribute(const std::string& name, const T& data, const std::string& ioName)
	{
		// Allow modification of attributes
		ios_[ioName].DefineAttribute<T>(name, data, "", "/", true);
	}

	template <typename T>
	void defineAttribute(const std::string& name, const T* data, size_t size, const std::string& ioName)
	{
		// Allow modification of attributes
		ios_[ioName].DefineAttribute<T>(name, data, size, "", "/", true);
	}

	template <typename T>
	void outputData(const std::string& varName, const T& data, const std::string& ioName)
	{
		if (! engines_[ioName]) {
			return;
		}

		if (current_mode_[ioName] != adios2::Mode::Write && current_mode_[ioName] != adios2::Mode::Append) {
			throw std::runtime_error("Engine not in write mode");
		}

		adios2::Variable<T> var = ios_[ioName].InquireVariable<T>(varName);
		if (var) {
			engines_[ioName].Put(var, data);
			// engines_[ioName].PerformPuts();
		}
		else {
			throw std::runtime_error(fmt::format("Variable \"{}\" not found", varName));
		}
	}

	template <typename T>
	void outputData(const std::string& varName, const T* data, const std::string& ioName)
	{
		if (! engines_[ioName]) {
			return;
		}

		if (current_mode_[ioName] != adios2::Mode::Write && current_mode_[ioName] != adios2::Mode::Append) {
			throw std::runtime_error("Engine not in write mode");
		}

		adios2::Variable<T> var = ios_[ioName].InquireVariable<T>(varName);
		if (var) {
			engines_[ioName].Put(var, data);
			// engines_[ioName].PerformPuts();
		}
		else {
			throw std::runtime_error(fmt::format("Variable \"{}\" not found", varName));
		}
	}

	// Methods for reading
	template <typename T>
	T readAttribute(const std::string& attrName, const std::string& ioName)
	{
		if (! engines_[ioName]) {
			throw std::runtime_error("Engine not initialized for this simulation type");
		}

		if (current_mode_[ioName] != adios2::Mode::Read) {
			throw std::runtime_error("Engine not in read mode");
		}

		auto attr = ios_[ioName].InquireAttribute<T>(attrName);
		if (! attr) {
			throw std::runtime_error(fmt::format("Attribute \"{}\" not found", attrName));
		}

		return attr.Data()[0];
	}

	template <typename T>
	std::vector<T> readAttributeArray(const std::string& attrName, const std::string& ioName)
	{
		if (! engines_[ioName]) {
			throw std::runtime_error("Engine not initialized for this simulation type");
		}

		if (current_mode_[ioName] != adios2::Mode::Read) {
			throw std::runtime_error("Engine not in read mode");
		}

		auto attr = ios_[ioName].InquireAttribute<T>(attrName);
		if (! attr) {
			throw std::runtime_error(fmt::format("Attribute \"{}\" not found", attrName));
		}

		return std::vector<T>(attr.Data(), attr.Data() + attr.Size());
	}

	template <typename T>
	void readVariable(const std::string& varName, T* data, const std::string& ioName)
	{
		if (! engines_[ioName]) {
			throw std::runtime_error("Engine not initialized for this simulation type");
		}

		if (current_mode_[ioName] != adios2::Mode::Read) {
			throw std::runtime_error("Engine not in read mode");
		}

		adios2::Variable<T> var = ios_.at(ioName).InquireVariable<T>(varName);
		if (! var) {
			throw std::runtime_error(fmt::format("Variable \"{}\" not found", varName));
		}

		engines_[ioName].Get(var, data);
		engines_[ioName].PerformGets();
	}

	void beginStep(const std::string& ioName)
	{
		if (! engines_[ioName]) {
			return;
		}
		engine_ = &engines_[ioName];
		engine_->BeginStep();
	}

	void performPutsAndStep(const std::string& ioName)
	{
		if (! engines_[ioName]) {
			return;
		}
		engine_ = &engines_[ioName];

		if (current_mode_[ioName] == adios2::Mode::Write || current_mode_[ioName] == adios2::Mode::Append) {
			engine_->PerformPuts();
		}
		else {
			engine_->PerformGets();
		}

		engine_->EndStep();
	}

	template <typename T>
	bool isVariableDefined(const std::string& varName, const std::string& ioName)
	{
		adios2::Variable<T> var = ios_.at(ioName).InquireVariable<T>(varName);
		return static_cast<bool>(var);
	}

private:
	adios2::ADIOS* adios;
	std::map<std::string, adios2::IO> ios_;
	adios2::IO defaultIO;
	std::map<std::string, adios2::Engine> engines_;
	std::map<std::string, adios2::Mode> current_mode_;	// Track if in read or write mode
	adios2::Engine* engine_ = nullptr;
};
