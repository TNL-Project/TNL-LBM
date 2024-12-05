#pragma once

#include <string>
#include <map>
#include <any>
#include <adios2.h>
#include <stdexcept>
#include <memory>
#include <fmt/format.h>

class DataManager
{
public:

    DataManager(adios2::ADIOS* adios) 
        : adios(adios)
    {
    }

    void initEngine(const std::string& name)
    {
        if (engines_.count(name) >0) {
            return;     
        }

        std::string filename = fmt::format("{}.bp", name);

        adios2::IO io_ = adios->DeclareIO(fmt::format("IO_{}", name));
        io_.SetEngine("BP4");
        engines_[name] = std::make_unique<adios2::Engine>(
            io_.Open(filename, adios2::Mode::Write)
        );
        ios_[name] = io_;
        engine_ = engines_[name].get();

        variables_[name] = {};
        variable_dimensions_[name] = {};
    }

    adios2::Engine& getEngine(const std::string& name) {
        if (!engines_[name]) {
            throw std::runtime_error("Engine not initialized for this simulation type");
        }
        return *engines_[name];
    }

    ~DataManager() {
        for (auto& [type, engine] : engines_) {
            if (engine) {
                engine->Close();
            }
        }
    }

    DataManager(const DataManager&) = delete;
    DataManager& operator=(const DataManager&) = delete;

    DataManager(DataManager&&) noexcept = default;
    DataManager& operator=(DataManager&&) noexcept = default;

    template<typename T>
    void defineData(const std::string& name, const adios2::Dims& shape,
                   const adios2::Dims& start, const adios2::Dims& count, const std::string& type)
    {
        adios2::Variable<T> var = ios_[type].DefineVariable<T>(name, shape, start, count);
        variables_[type][name] = var;
        variable_dimensions_[type][name] = static_cast<int>(shape.size());
    }

    template<typename T>
    void defineData(const std::string& name, const std::string& type)
    {
        adios2::Variable<T> var = ios_[type].DefineVariable<T>(name);
        variables_[type][name] = var;
        variable_dimensions_[type][name] = 0; 
    }

    template<typename T>
    void defineAttribute(const std::string& name, const T& data, const std::string& type)
    {
        ios_[type].DefineAttribute<T>(name, data);
    }

    template<typename T>
    void defineAttribute(const std::string& name, const T* data, size_t size, const std::string& type)
    {
         ios_[type].DefineAttribute<T>(name, data, size);
    }

    template<typename T>
    void outputData(const std::string& name, const T& data, const std::string& type)
    {
        if (engines_[type] == nullptr) {
            throw std::runtime_error("Engine not initialized for this simulation type");
        }
        auto it = variables_[type].find(name);
        if (it != variables_[type].end())
        {
            adios2::Variable<T> var = std::any_cast<adios2::Variable<T>>(it->second);
            engines_[type]->Put(var, data);
            engines_[type]->PerformPuts();
        }
        else
        {
            throw std::runtime_error(fmt::format("Variable \"{}\" not found", name));
        }
    }

    template<typename T>
    void outputData(const std::string& name, const T* data, const std::string& type)
    {
        if (engines_[type] == nullptr) {
            throw std::runtime_error("Engine not initialized for this simulation type");
        }

        auto it = variables_[type].find(name);
        if (it != variables_[type].end())
        {
            adios2::Variable<T> var = std::any_cast<adios2::Variable<T>>(it->second);
            engines_[type]->Put(var, data);
            engines_[type]->PerformPuts();
        }
        else
        {
            throw std::runtime_error(fmt::format("Variable \"{}\" not found", name));
        }
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
        engine_->PerformPuts();
        engine_->EndStep();
    }

    const std::map<std::string, std::any>& getVariables(const std::string& type) const
    {
        auto it = variables_.find(type);
        if(it != variables_.end()) {
            return it->second;
        }
        throw std::runtime_error("No variables defined for this simulation type");
    }

    const std::map<std::string, int>& getVariableDimensions(const std::string& type) const
    {
        auto it = variable_dimensions_.find(type);
        if(it != variable_dimensions_.end()) {
            return it->second;
        }
        throw std::runtime_error("No variable dimensions defined for this simulation type");
    }

private:

    adios2::ADIOS* adios;
    std::map<std::string, adios2::IO> ios_;
    std::map<std::string, std::unique_ptr<adios2::Engine>> engines_;
    std::map<std::string, std::map<std::string, std::any>> variables_;
    std::map<std::string, std::map<std::string, int>> variable_dimensions_; 
    adios2::Engine* engine_ = nullptr;
};
