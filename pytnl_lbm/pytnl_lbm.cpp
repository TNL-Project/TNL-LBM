#include <pytnl/exceptions.h>
#include <pytnl/pytnl.h>
#include <pytnl/containers/NDArray.h>

#include "lbm3d/core.h"
#include "lbm3d/py_lattice.h"
#include "lbm3d/py_lbm_block.h"
#include "lbm3d/py_lbm_data.h"
#include "lbm3d/py_lbm.h"
#include "lbm3d/py_state.h"
#include "lbm3d/py_UniformDataWriter.h"
#include "typedefs.h"

NB_MODULE(pytnl_lbm, m)
{
	register_exceptions(m);

	// import depending modules
	nb::module_::import_("pytnl._containers");
#ifdef USE_CUDA
	nb::module_::import_("pytnl._containers_cuda");
#endif

	export_Lattice<3, float, int>(m, "Lattice_3_float_int");
	export_Lattice<3, float, long int>(m, "Lattice_3_float_long");
	export_Lattice<3, double, int>(m, "Lattice_3_double_int");
	export_Lattice<3, double, long int>(m, "Lattice_3_double_long");

	export_actions_enum(m);
	export_counter<float>(m, "counter_float");
	export_counter<double>(m, "counter_double");

	export_LBM_Data<SP_D3Q27_CUM_ConstInflow>(m, "LBM_Data_SP_D3Q27_CUM_ConstInflow");
	export_LBM_BLOCK<SP_D3Q27_CUM_ConstInflow>(m, "LBM_BLOCK_SP_D3Q27_CUM_ConstInflow");
	export_LBM<SP_D3Q27_CUM_ConstInflow>(m, "LBM_SP_D3Q27_CUM_ConstInflow");
	export_State<SP_D3Q27_CUM_ConstInflow>(m, "State_SP_D3Q27_CUM_ConstInflow");
	export_UniformDataWriter<TRAITS>(m, "UniformDataWriter");

	m.def("execute", execute<State<SP_D3Q27_CUM_ConstInflow>>);

#ifndef HAVE_MPI
	// FIXME: missing bindings for DistributedNDArray
	using macro_indexer_t = typename SP_D3Q27_CUM_ConstInflow::TRAITS::hmacro_array_t::IndexerType;
	export_NDArrayIndexer<macro_indexer_t>(m, "macro_indexer_SP_D3Q27_CUM_ConstInflow");
	using hmacro_array_t = typename SP_D3Q27_CUM_ConstInflow::TRAITS::hmacro_array_t;
	using dmacro_array_t = typename SP_D3Q27_CUM_ConstInflow::TRAITS::dmacro_array_t;
	export_NDArray<hmacro_array_t>(m, "hmacro_array_SP_D3Q27_CUM_ConstInflow");
	export_NDArray<dmacro_array_t>(m, "dmacro_array_SP_D3Q27_CUM_ConstInflow");
#endif
}
