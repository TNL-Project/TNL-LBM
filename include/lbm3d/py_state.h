#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/trampoline.h>
#include <pytnl/pytnl.h>

#include "lbm_block.h"
#include "state.h"
#include "py_mpi.h"
#include "py_typedef.h"

inline void export_actions_enum(nb::module_& m)
{
	nb::enum_<Actions>(m, "Actions", nb::is_arithmetic())
		.value("STAT_RESET", Actions::STAT_RESET)
		.value("STAT2_RESET", Actions::STAT2_RESET)
		.value("PRINT", Actions::PRINT)
		.value("OUT2D", Actions::OUT2D)
		.value("OUT3D", Actions::OUT3D)
		.value("OUT3DCUT", Actions::OUT3DCUT)
		.value("PROBE1", Actions::PROBE1)
		.value("PROBE2", Actions::PROBE2)
		.value("PROBE3", Actions::PROBE3)
		.value("SAVESTATE", Actions::SAVESTATE)
		.export_values();
}

template <typename REAL>
void export_counter(nb::module_& m, const char* name)
{
	nb::class_<counter<REAL>>(m, name)
		.def(nb::init<>())
		.def("action", &counter<REAL>::action, nb::arg("time"))
		.def_rw("count", &counter<REAL>::count)
		.def_rw("period", &counter<REAL>::period);
}

// Trampoline class for overriding virtual methods
// For details, see https://nanobind.readthedocs.io/en/latest/classes.html#overriding-virtual-functions-in-python
template <typename NSE>
struct PyState : public State<NSE>
{
	using TRAITS = typename NSE::TRAITS;
	using BLOCK_NSE = LBM_BLOCK<NSE>;
	using real = typename NSE::TRAITS::real;
	using dreal = typename NSE::TRAITS::dreal;
	using idx = typename NSE::TRAITS::idx;
	using idx3d = typename NSE::TRAITS::idx3d;

	NB_TRAMPOLINE(State<NSE>, 16);

	[[nodiscard]] std::vector<std::string> getOutputDataNames() const override
	{
		NB_OVERRIDE(getOutputDataNames);
	}

	void outputData(UniformDataWriter<TRAITS>& writer, const BLOCK_NSE& block, const idx3d& begin, const idx3d& end) override
	{
		//NB_OVERRIDE(outputData, writer, block, begin, end);
		// NOTE: code below is expanded from NB_OVERRIDE, but changed to use rv_policy::reference for arguments
		// (necessary because LBM_BLOCK is not copyable)
		using nb_ret_type = void;
		nanobind::detail::ticket nb_ticket(nb_trampoline, "outputData", false);
		if (nb_ticket.key.is_valid()) {
			return nanobind::cast<nb_ret_type>(
				nb_trampoline.base().attr(nb_ticket.key).template operator()<nb::rv_policy::reference>(writer, block, begin, end)
			);
		}
		else
			return NBBase::outputData(writer, block, begin, end);
	}

	void probe1() override
	{
		NB_OVERRIDE(probe1);
	}
	void probe2() override
	{
		NB_OVERRIDE(probe2);
	}
	void probe3() override
	{
		NB_OVERRIDE(probe3);
	}
	void statReset() override
	{
		NB_OVERRIDE(statReset);
	}
	void stat2Reset() override
	{
		NB_OVERRIDE(stat2Reset);
	}

	// Simulation control
	bool estimateMemoryDemands() override
	{
		NB_OVERRIDE(estimateMemoryDemands);
	}
	void reset() override
	{
		NB_OVERRIDE(reset);
	}
	void resetDFs() override
	{
		NB_OVERRIDE(resetDFs);
	}
	void setupBoundaries() override
	{
		NB_OVERRIDE(setupBoundaries);
	}
	void SimInit() override
	{
		NB_OVERRIDE(SimInit);
	}
	void updateKernelData() override
	{
		NB_OVERRIDE(updateKernelData);
	}
	void updateKernelVelocities() override
	{
		NB_OVERRIDE(updateKernelVelocities);
	}
	void SimUpdate() override
	{
		NB_OVERRIDE(SimUpdate);
	}
	void AfterSimUpdate() override
	{
		NB_OVERRIDE(AfterSimUpdate);
	}
	void AfterSimFinished() override
	{
		NB_OVERRIDE(AfterSimFinished);
	}
	void computeBeforeLBMKernel() override
	{
		NB_OVERRIDE(computeBeforeLBMKernel);
	}
	void computeAfterLBMKernel() override
	{
		NB_OVERRIDE(computeAfterLBMKernel);
	}
	void copyAllToDevice() override
	{
		NB_OVERRIDE(copyAllToDevice);
	}
	void copyAllToHost() override
	{
		NB_OVERRIDE(copyAllToHost);
	}

	// Checkpointing
	void checkpointStateLocal(adios2::Mode mode) override
	{
		NB_OVERRIDE(checkpointStateLocal, mode);
	}
};

template <typename NSE>
void export_State(nb::module_& m, const char* name)
{
	using State = ::State<NSE>;
	using PyState = ::PyState<NSE>;
	using idx = typename State::idx;
	using real = typename State::real;
	using point_t = typename State::point_t;
	using lat_t = typename State::lat_t;

	auto state =  //
		nb::class_<State, PyState>(m, name)
			// Constructors
			.def(
				"__init__",
				[](State* t, const std::string& id, nb::object communicator, lat_t lat, const std::string& adiosConfigPath = "adios2.xml")
				{
					// GOTCHA: the trampoline must be used here
					new (t) PyState(id, py2mpi(communicator), std::move(lat), adiosConfigPath);
				}
			)
			// Attributes
			.def_ro("id", &State::id, "Unique identifier of the state")
			// NOTE: .def_rw() requires copyable types, but .def_ro() still makes modifiable attributes
			.def_ro("nse", &State::nse, "Main class for LBM Navier-Stokes equations")
			.def_ro("ibm", &State::ibm, "Main class for immersed boundary method")
			.def_ro("cnt", &State::cnt, "Counter for actions during the simulation (print, probe, output, etc.)")
			// Virtual methods
			.def("outputData", &State::outputData)
			.def("probe1", &State::probe1)
			.def("probe2", &State::probe2)
			.def("probe3", &State::probe3)
			.def("statReset", &State::statReset)
			.def("stat2Reset", &State::stat2Reset)
			// Configuration methods
			.def(
				"add2Dcut_X",
				[](State& self, idx x, const std::string& name)
				{
					return self.add2Dcut_X(x, name.c_str());
				}
			)
			.def(
				"add2Dcut_Y",
				[](State& self, idx y, const std::string& name)
				{
					return self.add2Dcut_Y(y, name.c_str());
				}
			)
			.def(
				"add2Dcut_Z",
				[](State& self, idx z, const std::string& name)
				{
					return self.add2Dcut_Z(z, name.c_str());
				}
			)
			.def(
				"add3Dcut",
				[](State& self, idx ox, idx oy, idx oz, idx lx, idx ly, idx lz, const std::string& name)
				{
					return self.add3Dcut(ox, oy, oz, lx, ly, lz, name.c_str());
				}
			)
			// Projection of wall from PNG images
			.def(
				"projectPNG_X",
				&State::projectPNG_X,
				nb::arg("filename"),
				nb::arg("x0"),
				nb::arg("rotate") = false,
				nb::arg("mirror") = false,
				nb::arg("flip") = false,
				nb::arg("amin") = 0.0,
				nb::arg("amax") = 1.0,
				nb::arg("bmin") = 0.0,
				nb::arg("bmax") = 1.0
			)
			.def(
				"projectPNG_Y",
				&State::projectPNG_Y,
				nb::arg("filename"),
				nb::arg("y0"),
				nb::arg("rotate") = false,
				nb::arg("mirror") = false,
				nb::arg("flip") = false,
				nb::arg("amin") = 0.0,
				nb::arg("amax") = 1.0,
				nb::arg("bmin") = 0.0,
				nb::arg("bmax") = 1.0
			)
			.def(
				"projectPNG_Z",
				&State::projectPNG_Z,
				nb::arg("filename"),
				nb::arg("z0"),
				nb::arg("rotate") = false,
				nb::arg("mirror") = false,
				nb::arg("flip") = false,
				nb::arg("amin") = 0.0,
				nb::arg("amax") = 1.0,
				nb::arg("bmin") = 0.0,
				nb::arg("bmax") = 1.0
			)
			// Simulation control
			.def("estimateMemoryDemands", &State::estimateMemoryDemands)
			.def("reset", &State::reset)
			.def("resetDFs", &State::resetDFs)
			.def("setupBoundaries", &State::setupBoundaries)
			.def("SimInit", &State::SimInit)
			.def("updateKernelData", &State::updateKernelData)
			.def("updateKernelVelocities", &State::updateKernelVelocities)
			.def("SimUpdate", &State::SimUpdate)
			.def("AfterSimUpdate", &State::AfterSimUpdate)
			.def("AfterSimFinished", &State::AfterSimFinished)
			.def("computeBeforeLBMKernel", &State::computeBeforeLBMKernel)
			.def("computeAfterLBMKernel", &State::computeAfterLBMKernel)
			.def("copyAllToDevice", &State::copyAllToDevice)
			.def("copyAllToHost", &State::copyAllToHost)
			// Flag files
			.def("canCompute", &State::canCompute)
			.def("flagCreate", &State::flagCreate)
			.def("flagDelete", &State::flagDelete)
			.def("flagExists", &State::flagExists)
			// Checkpointing
			.def("checkpointState", &State::checkpointState)
			.def("checkpointStateLocal", &State::checkpointStateLocal)
			.def("saveState", &State::saveState)
			.def("loadState", &State::loadState)
			// Timing
			.def("wallTimeReached", &State::wallTimeReached)
			.def("getWallTime", &State::getWallTime, nb::arg("collective") = false)
		//
		;

	// Export typedefs
	export_typedef<typename State::BLOCK_NSE>(state, "BLOCK_NSE");
	export_typedef<typename State::map_t>(state, "map_t");
	export_typedef<typename State::idx>(state, "idx");
	export_typedef<typename State::dreal>(state, "dreal");
	export_typedef<typename State::real>(state, "real");
	export_typedef<typename State::point_t>(state, "point_t");
	export_typedef<typename State::idx3d>(state, "idx3d");
	export_typedef<typename State::lat_t>(state, "lat_t");
}
