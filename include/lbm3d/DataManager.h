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
    enum class SimulationType {
        SIM_3D,
        SIM_3D_CUT,
        SIM_2D_X,
        SIM_2D_Y,
        SIM_2D_Z
    };

    DataManager(const std::string& baseDir) 
        : baseDir_(baseDir)
    {
        engines_[SimulationType::SIM_3D] = nullptr;
        engines_[SimulationType::SIM_3D_CUT] = nullptr;
        engines_[SimulationType::SIM_2D_X] = nullptr;
        engines_[SimulationType::SIM_2D_Y] = nullptr;
        engines_[SimulationType::SIM_2D_Z] = nullptr;

        variables_[SimulationType::SIM_3D] = {};
       variables_[SimulationType::SIM_3D_CUT] = {};
          variables_[SimulationType::SIM_2D_X] = {};
       variables_[SimulationType::SIM_2D_Y] = {};
     variables_[SimulationType::SIM_2D_Z] = {};
    }

    void initEngine(SimulationType type, const std::string& name)
    {
        if (engines_[type]) {
            return;     
        }

        std::string filename = fmt::format("{}.bp", name);

        adios2::IO io_ = adios.DeclareIO(fmt::format("IO_{}", name));
        io_.SetEngine("BP4");
        engines_[type] = std::make_unique<adios2::Engine>(
            io_.Open(filename, adios2::Mode::Write)
        );
        ios_[type] = io_;
        engine_ = engines_[type].get();
        engine_->BeginStep(); 
        std::cout<<filename<<std::endl;
    }

    adios2::Engine& getEngine(SimulationType type) {
        if (!engines_[type]) {
            throw std::runtime_error("Engine not initialized for this simulation type");
        }
        return *engines_[type];
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
                   const adios2::Dims& start, const adios2::Dims& count, SimulationType type)
    {
        adios2::Variable<T> var = ios_[type].DefineVariable<T>(name, shape, start, count);
        variables_[type][name] = var;
        variable_dimensions_[type][name] = static_cast<int>(shape.size());
    }

    template<typename T>
    void defineData(const std::string& name, SimulationType type)
    {
        adios2::Variable<T> var = ios_[type].DefineVariable<T>(name);
        variables_[type][name] = var;
        variable_dimensions_[type][name] = 0; 
    }

    template<typename T>
    void defineAttribute(const std::string& name, const T& data, SimulationType type)
    {
        ios_[type].DefineAttribute<T>(name, data);
    }

    template<typename T>
    void defineAttribute(const std::string& name, const T* data, size_t size, SimulationType type)
    {
         ios_[type].DefineAttribute<T>(name, data, size);
    }

    template<typename T>
    void outputData(const std::string& name, const T& data, SimulationType type)
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
    void outputData(const std::string& name, const T* data, SimulationType type)
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


    void beginStep(SimulationType type)
    {
        if (engines_[type] == nullptr) {
            throw std::runtime_error("Engine not initialized for this simulation type");
        }
        engine_ = engines_[type].get(); 
        engine_->BeginStep();
    }

    void performPutsAndStep(SimulationType type)
    {
        if (engines_[type] == nullptr) {
            throw std::runtime_error("Engine not initialized for this simulation type");
        }
        engine_ = engines_[type].get();
        engine_->PerformPuts();
        engine_->EndStep();
    }

    const std::map<std::string, std::any>& getVariables(SimulationType type) const
    {
        auto it = variables_.find(type);
        if(it != variables_.end()) {
            return it->second;
        }
        throw std::runtime_error("No variables defined for this simulation type");
    }

    const std::map<std::string, int>& getVariableDimensions(SimulationType type) const
    {
        auto it = variable_dimensions_.find(type);
        if(it != variable_dimensions_.end()) {
            return it->second;
        }
        throw std::runtime_error("No variable dimensions defined for this simulation type");
    }

private:
    std::string getSimTypeString(SimulationType type) {
        switch(type) {
            case SimulationType::SIM_3D: return "3D";
            case SimulationType::SIM_3D_CUT: return "3D_CUT";
            case SimulationType::SIM_2D_X: return "2D_X";
            case SimulationType::SIM_2D_Y: return "2D_Y";
            case SimulationType::SIM_2D_Z: return "2D_Z";
            default: return "UNKNOWN";
        }
    }

    std::string baseDir_;
    adios2::ADIOS adios;
    std::map<SimulationType, adios2::IO> ios_;
    std::map<SimulationType, std::unique_ptr<adios2::Engine>> engines_;
    std::map<SimulationType, std::map<std::string, std::any>> variables_;
    std::map<SimulationType, std::map<std::string, int>> variable_dimensions_; 
    adios2::Engine* engine_ = nullptr;
};
