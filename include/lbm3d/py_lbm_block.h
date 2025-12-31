#pragma once

#include <pytnl/pytnl.h>

#include "lbm_block.h"
#include "py_typedef.h"

template <typename NSE>
void export_LBM_BLOCK(nb::module_& m, const char* name)
{
	using LBM_BLOCK = ::LBM_BLOCK<NSE>;

	auto block =  //
		nb::class_<LBM_BLOCK>(m, name)
			.def_ro("data", &LBM_BLOCK::data)
			.def_ro("hmap", &LBM_BLOCK::hmap)
			.def_ro("dmap", &LBM_BLOCK::dmap)
			.def_ro("hmacro", &LBM_BLOCK::hmacro)
			.def_ro("dmacro", &LBM_BLOCK::dmacro)
			.def_ro("hdiffusionCoeff", &LBM_BLOCK::hdiffusionCoeff)
			.def_ro("ddiffusionCoeff", &LBM_BLOCK::ddiffusionCoeff)
			.def_ro("hphiTransferDirection", &LBM_BLOCK::hphiTransferDirection)
			.def_ro("dphiTransferDirection", &LBM_BLOCK::dphiTransferDirection)
			// TODO: hfs, dfs
			// MPI
			.def_ro("communicator", &LBM_BLOCK::communicator)
			.def_ro("rank", &LBM_BLOCK::rank)
			.def_ro("nproc", &LBM_BLOCK::nproc)
			// lattice sizes and offsets
			.def_ro("global_", &LBM_BLOCK::global)
			.def_ro("local", &LBM_BLOCK::local)
			.def_ro("offset", &LBM_BLOCK::offset)
			.def_ro("id", &LBM_BLOCK::id, "index of this block")
			.def_ro_static("macro_overlap_width", &LBM_BLOCK::macro_overlap_width, "maximum width of overlaps for the macro arrays")
			// Methods
			.def("df_overlap_X", &LBM_BLOCK::df_overlap_X)
			.def("df_overlap_Y", &LBM_BLOCK::df_overlap_Y)
			.def("df_overlap_Z", &LBM_BLOCK::df_overlap_Z)
			.def(
				"is_distributed", &LBM_BLOCK::is_distributed, "returns a tuple of bools indicating if the lattice is distributed along each dimension"
			)
			// Helpers for indexing - methods check if the given GLOBAL (multi)index is in the local range
			.def(
				"isLocalIndex",
				&LBM_BLOCK::isLocalIndex,
				nb::arg("x"),
				nb::arg("y"),
				nb::arg("z"),
				"check if the given GLOBAL (multi)index is in the local range"
			)
			.def("isLocalX", &LBM_BLOCK::isLocalX, nb::arg("x"), "check if the given GLOBAL index is in the local range")
			.def("isLocalY", &LBM_BLOCK::isLocalY, nb::arg("y"), "check if the given GLOBAL index is in the local range")
			.def("isLocalZ", &LBM_BLOCK::isLocalZ, nb::arg("z"), "check if the given GLOBAL index is in the local range")
			// actions
			.def("copyMapToHost", &LBM_BLOCK::copyMapToHost)
			.def("copyMapToDevice", &LBM_BLOCK::copyMapToDevice)
			.def("copyMacroToHost", &LBM_BLOCK::copyMacroToHost)
			.def("copyMacroToDevice", &LBM_BLOCK::copyMacroToDevice)
			// TODO: overloads w/o parameter
			//.def("copyDFsToHost", &LBM_BLOCK::copyDFsToHost)
			//.def("copyDFsToDevice", &LBM_BLOCK::copyDFsToDevice)
			.def("allocateHostData", &LBM_BLOCK::allocateHostData)
			.def("allocateDeviceData", &LBM_BLOCK::allocateDeviceData)
			// TODO: need conditional build?
			//.def("allocateDiffusionCoefficientArrays", &LBM_BLOCK::allocateDiffusionCoefficientArrays)
			//.def("allocatePhiTransferDirectionArrays", &LBM_BLOCK::allocatePhiTransferDirectionArrays)
			// Global methods - use GLOBAL indices !!!
			.def("setMap", &LBM_BLOCK::setMap, nb::arg("x"), nb::arg("y"), nb::arg("z"), nb::arg("value"), "set map value at the given GLOBAL index")
			.def("setBoundaryX", &LBM_BLOCK::setBoundaryX, nb::arg("x"), nb::arg("value"), "set boundary value at the given GLOBAL index")
			.def("setBoundaryY", &LBM_BLOCK::setBoundaryY, nb::arg("y"), nb::arg("value"), "set boundary value at the given GLOBAL index")
			.def("setBoundaryZ", &LBM_BLOCK::setBoundaryZ, nb::arg("z"), nb::arg("value"), "set boundary value at the given GLOBAL index")
			.def("resetMap", &LBM_BLOCK::resetMap)
			.def("setEquilibrium", &LBM_BLOCK::setEquilibrium)
			.def("computeInitialMacro", &LBM_BLOCK::computeInitialMacro)
		//
		;

	// Export typedefs
	export_typedef<typename LBM_BLOCK::map_t>(block, "map_t");
	export_typedef<typename LBM_BLOCK::idx>(block, "idx");
	export_typedef<typename LBM_BLOCK::dreal>(block, "dreal");
	export_typedef<typename LBM_BLOCK::real>(block, "real");
	export_typedef<typename LBM_BLOCK::point_t>(block, "point_t");
	export_typedef<typename LBM_BLOCK::idx3d>(block, "idx3d");
	export_typedef<typename LBM_BLOCK::bool3d>(block, "bool3d");
	export_typedef<typename LBM_BLOCK::lat_t>(block, "lat_t");
}
