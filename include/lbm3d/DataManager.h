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

	void initEngine(const std::string& name, adios2::Mode mode = adios2::Mode::Write)
	{
		auto io_it = ios_.find(name);
		if (io_it == ios_.end()) {
			adios2::IO io_handle = adios->DeclareIO(fmt::format("IO_{}", name));

			if (! io_handle.InConfigFile()) {
				io_handle.SetEngine(defaultIO.EngineType());
				io_handle.SetParameters(defaultIO.Parameters());
				spdlog::warn("{}", defaultIO.EngineType());
			}

			io_it = ios_.emplace(name, io_handle).first;
		}

		adios2::IO& io_ref = io_it->second;
		const bool is_bp_engine = io_ref.EngineType().rfind("BP", 0) == 0;

		if (! is_bp_engine && mode == adios2::Mode::Append) {
			mode = adios2::Mode::Write;
		}

		if (engines_.count(name) > 0) {
			if (current_mode_[name] == mode) {
				spdlog::trace("Engine '{}' already initialized with requested mode.", name);
				engine_ = &engines_[name];
				return;
			}

			engines_[name].Close();
			engines_.erase(name);
		}

		if (engines_.count(name) == 0) {
			std::string filename = name;

			if (is_bp_engine) {
				filename += ".bp";
			}

			engines_[name] = io_ref.Open(filename, mode);
			current_mode_[name] = mode;
			spdlog::trace("Initialized engine '{}' with mode {}", filename, magic_enum::enum_name(mode));
		}

		engine_ = &engines_[name];
	}

	//this function is not necessary right now
	void ChangeEnginesToAppend()
	{
		for (auto& [name, engine] : engines_) {
			if (engine && current_mode_[name] == adios2::Mode::Write) {
				initEngine(name, adios2::Mode::Append);
			}
		}
	}

	adios2::Engine& getEngine(const std::string& name)
	{
		if (! engines_[name]) {
			throw std::runtime_error("Engine not initialized for this simulation type");
		}
		return engines_[name];
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
			engines_[name].Close();
			engines_.erase(name);
			spdlog::debug("Closed engine for {}", name);
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
		if (! engines_[type]) {
			throw std::runtime_error("Engine not initialized for this simulation type");
		}

		if (current_mode_[type] != adios2::Mode::Write && current_mode_[type] != adios2::Mode::Append) {
			throw std::runtime_error("Engine not in write mode");
		}

		adios2::Variable<T> var = ios_[type].InquireVariable<T>(name);
		if (var) {
			engines_[type].Put(var, data);
			engines_[type].PerformPuts();
		}
		else {
			throw std::runtime_error(fmt::format("Variable \"{}\" not found", name));
		}
	}

	template <typename T>
	void outputData(const std::string& name, const T* data, const std::string& type)
	{
		if (! engines_[type]) {
			throw std::runtime_error("Engine not initialized for this simulation type");
		}

		if (current_mode_[type] != adios2::Mode::Write && current_mode_[type] != adios2::Mode::Append) {
			throw std::runtime_error("Engine not in write mode");
		}

		adios2::Variable<T> var = ios_[type].InquireVariable<T>(name);
		if (var) {
			engines_[type].Put(var, data);
			engines_[type].PerformPuts();
		}
		else {
			throw std::runtime_error(fmt::format("Variable \"{}\" not found", name));
		}
	}

	// Methods for reading
	template <typename T>
	T readAttribute(const std::string& name, const std::string& type)
	{
		if (! engines_[type]) {
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
		if (! engines_[type]) {
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
		if (! engines_[type]) {
			throw std::runtime_error("Engine not initialized for this simulation type");
		}

		if (current_mode_[type] != adios2::Mode::Read) {
			throw std::runtime_error("Engine not in read mode");
		}

		adios2::Variable<T> var = ios_.at(type).InquireVariable<T>(name);
		if (! var) {
			throw std::runtime_error(fmt::format("Variable \"{}\" not found", name));
		}

		engines_[type].Get(var, data);
		engines_[type].PerformGets();
	}

	void beginStep(const std::string& type)
	{
		if (! engines_[type]) {
			throw std::runtime_error("Engine not initialized for this simulation type");
		}
		engine_ = &engines_[type];
		engine_->BeginStep();
	}

	void performPutsAndStep(const std::string& type)
	{
		if (! engines_[type]) {
			throw std::runtime_error("Engine not initialized for this simulation type");
		}
		engine_ = &engines_[type];

		if (current_mode_[type] == adios2::Mode::Write || current_mode_[type] == adios2::Mode::Append) {
			engine_->PerformPuts();
		}
		else {
			engine_->PerformGets();
		}

		engine_->EndStep();
	}

	template <typename T>
	bool isVariableDefined(const std::string& name, const std::string& type)
	{
		adios2::Variable<T> var = ios_.at(type).InquireVariable<T>(name);
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
