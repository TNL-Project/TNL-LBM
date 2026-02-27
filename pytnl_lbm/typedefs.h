#pragma once

#include "lbm3d/d3q27/bc.h"
#include "lbm3d/d3q27/col_cum.h"
#include "lbm3d/d3q27/eq_inv_cum.h"
#include "lbm3d/d3q27/macro.h"
#include "lbm3d/d3q27/streaming_AB.h"
#include "lbm3d/defs.h"
#include "lbm3d/lbm_data.h"

using TRAITS = TraitsSP;
using COLL = D3Q27_CUM<TRAITS, D3Q27_EQ_INV_CUM<TRAITS>>;

using SP_D3Q27_CUM_ConstInflow = LBM_CONFIG<
	TRAITS,
	D3Q27_KernelStruct,
	NSE_Data_ConstInflow<TRAITS>,
	COLL,
	typename COLL::EQ,
	D3Q27_STREAMING<TRAITS>,
	D3Q27_BC_All,
	D3Q27_MACRO_Default<TRAITS>>;
