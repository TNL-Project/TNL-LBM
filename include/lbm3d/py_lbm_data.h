#pragma once

#include <pytnl/pytnl.h>

#include "py_typedef.h"

template <typename NSE>
void export_LBM_Data(nb::module_& m, const char* name)
{
	using LBM_DATA = typename NSE::DATA;

	auto block =	//
		nb::class_<LBM_DATA>(m, name)
			.def_rw("even_iter", &LBM_DATA::even_iter)
			.def_rw("indexer", &LBM_DATA::indexer)
			.def_rw("XYZ", &LBM_DATA::XYZ)
			.def_rw("lbmViscosity", &LBM_DATA::lbmViscosity)
			.def_rw("stat_counter", &LBM_DATA::stat_counter)
			// FIXME: this approach does not work for user-defined classes and additional attributes
			// FIXME: the following is from NSE_Data_ConstInflow
			.def_rw("inflow_vx", &LBM_DATA::inflow_vx)
			.def_rw("inflow_vy", &LBM_DATA::inflow_vy)
			.def_rw("inflow_vz", &LBM_DATA::inflow_vz)
		//
		;

	// Export typedefs
	export_typedef<typename LBM_DATA::map_t>(block, "map_t");
	export_typedef<typename LBM_DATA::idx>(block, "idx");
	export_typedef<typename LBM_DATA::dreal>(block, "dreal");
	export_typedef<typename LBM_DATA::indexer_t>(block, "indexer_t");
}
