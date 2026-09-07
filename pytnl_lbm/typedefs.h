#pragma once

#include "lbm3d/defs.h"
#include "lbm3d/d3q27/bc.h"
#include "lbm3d/d3q27/col_cum.h"
#include "lbm3d/d3q27/eq_inv_cum.h"
#include "lbm3d/d3q27/macro.h"
// exactly one streaming header must be included
#ifdef AA_PATTERN
	#include "lbm3d/d3q27/streaming_AA.h"
#endif
#ifdef AB_PATTERN
	#include "lbm3d/d3q27/streaming_AB.h"
#endif
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

// openings-capable twin of the above: same collision/streaming/BC/macro, only
// the DATA carries the inflow-openings device state (per the AGENTS.md payload
// rule the legacy ConstInflow instantiation stays untouched)
using SP_D3Q27_CUM_OpeningInflow = LBM_CONFIG<
	TRAITS,
	D3Q27_KernelStruct,
	NSE_Data_OpeningInflow<TRAITS>,
	COLL,
	typename COLL::EQ,
	D3Q27_STREAMING<TRAITS>,
	D3Q27_BC_All,
	D3Q27_MACRO_Default<TRAITS>>;
