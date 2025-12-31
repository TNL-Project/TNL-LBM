#pragma once

#include <pytnl/pytnl.h>

#include "lbm.h"
#include "py_bc.h"
#include "py_macro.h"
#include "py_typedef.h"

template <typename NSE>
void export_LBM(nb::module_& m, const char* name)
{
	using LBM = ::LBM<NSE>;

	auto lbm =	//
		nb::class_<LBM>(m, name)
			// MPI
			.def_ro("communicator", &LBM::communicator)
			.def_ro("rank", &LBM::rank)
			.def_ro("nproc", &LBM::nproc)
			// global lattice size and physical units conversion
			.def_ro("lat", &LBM::lat)
			// local lattice blocks (subdomains)
			.def_ro("blocks", &LBM::blocks)
			.def_ro("total_blocks", &LBM::total_blocks)
#ifdef HAVE_MPI
			// synchronization methods
			.def("synchronizeDFsAndMacroDevice", &LBM::synchronizeDFsAndMacroDevice)
			.def("synchronizeMapDevice", &LBM::synchronizeMapDevice)
#endif
			// input parameters: constant in time
			.def_rw(
				"physCharLength",
				&LBM::physCharLength,
				"characteristic length used for Re calculation, "
				"default is `physDl * Y` but you can specify that manually"
			)
			.def_rw("physFinalTime", &LBM::physFinalTime)
			.def_rw("physStartTime", &LBM::physStartTime, "used for ETA calculation only (default is 0)")
			.def_rw("iterations", &LBM::iterations, "number of LBM iterations at the start (physStartTime) -- used for GLUPS calculation only")
			// flag
			.def_rw("terminate", &LBM::terminate, "flag for terminal error detection")
			// getters
			.def("Re", &LBM::Re)
			.def("physTime", &LBM::physTime)
			// Helpers for indexing - methods check if the given GLOBAL (multi)index is in the local range
			.def(
				"isAnyLocalIndex",
				&LBM::isAnyLocalIndex,
				nb::arg("x"),
				nb::arg("y"),
				nb::arg("z"),
				"check if the given GLOBAL (multi)index is in the local range"
			)
			.def("isAnyLocalX", &LBM::isAnyLocalX, nb::arg("x"), "check if the given GLOBAL index is in the local range")
			.def("isAnyLocalY", &LBM::isAnyLocalY, nb::arg("y"), "check if the given GLOBAL index is in the local range")
			.def("isAnyLocalZ", &LBM::isAnyLocalZ, nb::arg("z"), "check if the given GLOBAL index is in the local range")
			// actions
			.def("copyMapToHost", &LBM::copyMapToHost)
			.def("copyMapToDevice", &LBM::copyMapToDevice)
			.def("copyMacroToHost", &LBM::copyMacroToHost)
			.def("copyMacroToDevice", &LBM::copyMacroToDevice)
			// TODO: overloads w/o parameter
			//.def("copyDFsToHost", &LBM::copyDFsToHost)
			//.def("copyDFsToDevice", &LBM::copyDFsToDevice)
			.def("allocateHostData", &LBM::allocateHostData)
			.def("allocateDeviceData", &LBM::allocateDeviceData)
			// TODO: need conditional build?
			//.def("allocateDiffusionCoefficientArrays", &LBM::allocateDiffusionCoefficientArrays)
			//.def("allocatePhiTransferDirectionArrays", &LBM::allocatePhiTransferDirectionArrays)
			.def("updateKernelData", &LBM::updateKernelData, "copy physical parameters to data structure accessible by the CUDA kernel")
			// Global methods - use GLOBAL indices !!!
			.def("setMap", &LBM::setMap, nb::arg("x"), nb::arg("y"), nb::arg("z"), nb::arg("value"), "set map value at the given GLOBAL index")
			.def("setBoundaryX", &LBM::setBoundaryX, nb::arg("x"), nb::arg("value"), "set boundary value at the given GLOBAL index")
			.def("setBoundaryY", &LBM::setBoundaryY, nb::arg("y"), nb::arg("value"), "set boundary value at the given GLOBAL index")
			.def("setBoundaryZ", &LBM::setBoundaryZ, nb::arg("z"), nb::arg("value"), "set boundary value at the given GLOBAL index")
			.def("resetMap", &LBM::resetMap)
			.def("setEquilibrium", &LBM::setEquilibrium)
			.def("computeInitialMacro", &LBM::computeInitialMacro)
		//
		;

	// Export typedefs
	export_typedef<typename LBM::map_t>(lbm, "map_t");
	export_typedef<typename LBM::idx>(lbm, "idx");
	export_typedef<typename LBM::dreal>(lbm, "dreal");
	export_typedef<typename LBM::real>(lbm, "real");
	export_typedef<typename LBM::point_t>(lbm, "point_t");
	export_typedef<typename LBM::idx3d>(lbm, "idx3d");
	export_typedef<typename LBM::lat_t>(lbm, "lat_t");

	export_bc<typename NSE::BC>(lbm, "BC");
	export_macro<typename NSE::MACRO>(lbm, "MACRO");
}
