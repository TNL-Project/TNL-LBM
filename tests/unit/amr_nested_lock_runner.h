#pragma once

// Templated scenario runner of the NESTED cross-pattern AMR bitwise lock
// (the locked surface and its provenance are documented in
// amr_nested_lock.h): one three-level periodic State_AMR instance (the mL1
// fixture geometry extended by a nested level-2 footprint) driven 10
// coarse cycles through the real SimInit/SimUpdate machinery, snapshotting
// the map and the macroscopic channels of all three levels after SimInit
// and after every cycle. Instantiated once per runner translation unit
// (test_amr_nested_lock_<pattern>.cu), so the five pattern instantiations
// compile in parallel; the traces are compared in the kernel-free
// comparator TU test_amr_nested_lock.cu.
//
// The config bundle, the macro policy shadow, the scenario state and the
// snapshot helpers are the mL1 lock's, shared by include (the parity
// surfaces under lock -- the Schönherr schedule, the subcycling parity
// discipline and the C2F/F2C transfers -- are identical; only the level
// count, the region set and the scenario entry point differ).

#include <string>

#include <fmt/core.h>

#include "amr_bitwise_lock_runner.h"
#include "amr_nested_lock.h"

// the mL1 fixture geometry with one nested level: the same centered
// level-1 footprint [4, 12)^3 on the 16^3 periodic box, plus the level-2
// region at parent-frame [12, 20)^3 (level-2 lines are expressed in the
// level's own fine-global frame, amr_decomposition.h: the raw line numbers
// resolve by >> (level - 1) to the immediate parent's global coordinates;
// footprint 4 parent-level cells per axis, telescoping gap 4 parent cells
// on every face -- above the recommended 3, so no V-suite warn tier fires)
constexpr const char* AMR_NESTED_LOCK_REGIONS = "1 4 4 4 8 8 8\n"
												"2 24 24 24 16 16 16";

// mirrored from the mL1 lock's runAMRLockScenario with max_level == 2 and
// the nested region set (see amr_bitwise_lock_runner.h for the lattice
// factory, the IC and the snapshot idiom)
template <typename NSE>
AMRLockTrace runAMRNestedLockScenario(const std::string& pattern_label)
{
	using lat_t = Lattice<3, typename NSE::TRAITS::real, typename NSE::TRAITS::idx>;
	using point_t = typename NSE::TRAITS::point_t;
	using bool3d = typename NSE::TRAITS::bool3d;

	AMRLockTrace trace;
	trace.pattern = pattern_label;

	// the subcycling-suite lattice factory (amr_test_fixture.h's
	// makeLattice): the fully periodic 16^3 box in the sim_AMR physical
	// scaling, so the per-level viscosity doubling and the time-step
	// halving are exact in binary arithmetic
	const typename NSE::TRAITS::real LBM_VISCOSITY = 0.005;
	const typename NSE::TRAITS::real PHYS_HEIGHT = 0.41;
	const typename NSE::TRAITS::real PHYS_VISCOSITY = 1.5e-5;
	const int N = 16;
	const typename NSE::TRAITS::real PHYS_DL = PHYS_HEIGHT / N;
	const typename NSE::TRAITS::real PHYS_DT = LBM_VISCOSITY / PHYS_VISCOSITY * PHYS_DL * PHYS_DL;

	lat_t lat;
	lat.global = typename lat_t::CoordinatesType(N, N, N);
	lat.physOrigin = point_t{0., 0., 0.};
	lat.physDl = PHYS_DL;
	lat.physDt = PHYS_DT;
	lat.physViscosity = PHYS_VISCOSITY;

	const std::string id = fmt::format("test_amr_nested_lock_{}", pattern_label);
	AMRLockState<NSE> state(id, MPI_COMM_WORLD, lat, "adios2.xml", /*periodic=*/bool3d{true, true, true}, /*max_level=*/2);
	if (! state.canCompute())
		return trace;

	createAMRBlocks(state.nse, parseAMRConfig<NSE>(AMR_NESTED_LOCK_REGIONS));

	state.SimInit();
	if (state.nse.terminate)
		return trace;
	trace.init_ok = true;

	// cycle 0: the post-SimInit state (map + macro; the map tagging --
	// GEO_AMR_INTERFACE ring + frozen footprint -- is init-time only)
	amrLockSnapshotCycle(trace, state, /*with_map=*/true);

	// the execute()-style iteration idiom of the mL1 lock: the global
	// updateKernelData clock arms level 0's rotation, one SimUpdate call
	// advances exactly one coarse cycle
	for (int c = 0; c < AMR_NESTED_LOCK_CYCLES; c++) {
		state.updateKernelData();
		state.SimUpdate();
		if (state.nse.terminate)
			break;
		amrLockSnapshotCycle(trace, state, /*with_map=*/false);
	}
	return trace;
}
