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
    }

    void initEngine(SimulationType type, const std::string& name)
    {
        if (engines_[type] != nullptr) {
            engines_[type]->Close();
        }

        std::string filename = fmt::format("{}.bp", name);

        io_ = adios.DeclareIO(fmt::format("IO_{}", name));
        io_.SetEngine("BP4");
        engines_[type] = std::make_unique<adios2::Engine>(
            io_.Open(name, adios2::Mode::Write)
        );
        engine_ = engines_[type].get();
        engine_->BeginStep(); 
       // std::cout<<filename<<std::endl;
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
                   const adios2::Dims& start, const adios2::Dims& count)
    {
        adios2::Variable<T> var = io_.DefineVariable<T>(name, shape, start, count);
        variables_[name] = var;
        variable_dimensions_[name] = static_cast<int>(shape.size());
    }

    template<typename T>
    void defineData(const std::string& name)
    {
        adios2::Variable<T> var = io_.DefineVariable<T>(name);
        variables_[name] = var;
        variable_dimensions_[name] = 0; 
    }

    template<typename T>
    void defineAttribute(const std::string& name, const T& data)
    {
        io_.DefineAttribute<T>(name, data);
    }

    template<typename T>
    void defineAttribute(const std::string& name, const T* data, size_t size)
    {
        io_.DefineAttribute<T>(name, data, size);
    }

    template<typename T>
    void outputData(const std::string& name, const T& data, SimulationType type)
    {
        if (engines_[type] == nullptr) {
            throw std::runtime_error("Engine not initialized for this simulation type");
        }
        engine_ = engines_[type].get();

        auto it = variables_.find(name);
        if (it != variables_.end())
        {
            adios2::Variable<T> var = std::any_cast<adios2::Variable<T>>(it->second);
            engine_->Put(var, data);
            engine_->PerformPuts();
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

    const std::map<std::string, std::any>& getVariables() const
    {
        return variables_;
    }

    const std::map<std::string, int>& getVariableDimensions() const
    {
        return variable_dimensions_;
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
    adios2::IO io_;
    std::map<SimulationType, std::unique_ptr<adios2::Engine>> engines_;
    std::map<std::string, std::any> variables_;
    std::map<std::string, int> variable_dimensions_; 
    adios2::Engine* engine_ = nullptr;
};
