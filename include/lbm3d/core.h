#pragma once

// default
#include "lbm_data.h"  // LBM_Data is a general template (for any Q)
#include "d3q27/macro.h"
#include "d3q27/bc.h"

#include "d3q27/eq.h"
#include "d3q27/eq_inv_cum.h"
#include "d3q27/eq_well.h"
#include "d3q27/eq_entropic.h"

// exactly one streaming header must be included
#ifdef AA_PATTERN
	#include "d3q27/streaming_AA.h"
#endif
#ifdef AB_PATTERN
	#include "d3q27/streaming_AB.h"
#endif

#include "d3q27/col_cum.h"
#include "d3q27/col_bgk.h"
#include "d3q27/col_clbm.h"
#include "d3q27/col_mrt.h"
#include "d3q27/col_srt.h"
#include "d3q27/col_kbc_n.h"
#include "d3q27/col_kbc_c.h"
#include "d3q27/col_srt_modif_force.h"

#include "d3q27/col_srt_well.h"
#include "d3q27/col_clbm_well.h"
#include "d3q27/col_cum_well.h"
#include "d3q27/col_bgk_well.h"


#include "state.h"

template <typename STATE>
void execute(STATE& state)
{
	// initialize the simulation -- load state or allocate, copy to device and synchronize overlaps with MPI
	state.SimInit();

	// make snapshot for the initial condition
	state.AfterSimUpdate();

	bool quit = false;
	while (! quit) {
		// update kernel data (viscosity, swap df1 and df2)
		state.updateKernelData();
		state.updateKernelVelocities();

		state.SimUpdate();

		// post-processing: snapshots etc.
		state.AfterSimUpdate();

		// check wall time
		// (Note that state.wallTimeReached() must be called exactly once per iteration!)
		if (state.wallTimeReached()) {
			// copy all LBM quantities from device to host
			state.copyAllToHost();

			spdlog::info("maximum wall time reached");
			// copy data to CPU (if needed)
			state.saveState();
			quit = true;
		}
		// check savestate
		else if (state.cnt[SAVESTATE].action(state.getWallTime(true))) {
			// copy all LBM quantities from device to host
			state.copyAllToHost();

			state.saveState();
			state.cnt[SAVESTATE].count++;
		}

		// check final time
		if (state.nse.physTime() > state.nse.physFinalTime) {
			spdlog::info("physFinalTime reached");
			quit = true;
			state.flagCreate("finished");
			state.flagDelete("loadstate");
		}

		// handle termination (we must reduce the terminate flag first, because
		// only rank 0 can create and delete flags)
		state.nse.terminate = TNL::MPI::reduce(state.nse.terminate, MPI_LOR, MPI_COMM_WORLD);
		if (state.nse.terminate) {
			spdlog::info("terminate flag triggered");
			quit = true;
			state.flagCreate("terminated");
			state.flagDelete("loadstate");
		}

		// distribute quit among all MPI processes
		quit = TNL::MPI::reduce(quit, MPI_LOR, MPI_COMM_WORLD);
	}

	state.AfterSimFinished();
}
