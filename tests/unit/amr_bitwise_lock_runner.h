#pragma once

// Templated scenario runner of the cross-pattern AMR bitwise lock (the lock
// surface and its provenance are documented in amr_bitwise_lock.h): one
// two-level periodic State_AMR instance (the mL1 fixture geometry shared
// with the subcycling/nesting suites) driven 10 coarse cycles through the
// real SimInit/SimUpdate machinery, snapshotting the map and the macroscopic
// channels of both levels after SimInit and after every cycle. Instantiated
// once per runner translation unit (test_amr_bitwise_lock_<pattern>.cu), so
// the five pattern instantiations compile in parallel; the traces are
// compared in the kernel-free comparator TU test_amr_bitwise_lock.cu.

#include <algorithm>
#include <utility>

#include <fmt/core.h>

#include "lbm3d/core.h"
#include "lbm3d/amr_state.h"

#include "amr_bitwise_lock.h"

// the mL1 fixture geometry of the subcycling/nesting suites: a centered
// level-1 region with coarse footprint [4, 12)^3 (a 14^3 fine interior) on
// the 16^3 periodic box
constexpr const char* AMR_LOCK_REGIONS = "1 4 4 4 8 8 8";

// lock-specific macro policy: D3Q27_MACRO_Default computes macroscopic
// quantities only on request (compute_in_each_iteration == false), which
// leaves the interior device macro stale across the Schönherr substeps (the
// transfer-written cells stay fresh, the kernel-integrated ones do not) —
// the per-cycle bitwise surface needs the default's output channels but a
// per-substep refresh, so the flag is shadowed to true. The refresh is
// read-only with respect to the populations (MACRO::outputMacro writes the
// macro array, nothing touches the DFs), so the DF trajectory — and hence
// the pattern-agreement claim — is identical to the default-policy run
template <typename TRAITS>
struct AMRLockMacro : D3Q27_MACRO_Default<TRAITS>
{
	static const bool compute_in_each_iteration = true;
};

// per-pattern config bundle: identical to the shared AMR fixture
// (amr_test_fixture.h's NSE_CONFIG: single-precision TraitsSP,
// NSE_Data_ConstInflow, BC_All dispatch and the default macro output
// channels), with the streaming pattern made a template parameter. The
// collision is D3Q27_SRT, NOT the production cumulant — a deliberate
// compile-time mitigation in the style of the streaming-MPI suite (which
// shares one collision instantiation across patterns because "a streaming
// pattern changes where populations live in memory, never the
// arithmetic"): the AMR machinery under lock — the Schönherr schedule, the
// subcycling parity discipline, the ghost-layer extents and the C2F/F2C
// transfers (which run their own pattern-dependent conversion kernels,
// instantiated here unchanged) — is collision-agnostic, while the CUM
// instantiation inflates the per-runner TU compile time (measured on this
// Debug tree: 32.2 min user for the lock's five runner TUs under CUM vs
// 23.1 min under SRT, 2.3x on the heaviest eso TU); the fp-contraction
// sensitivity of the production collision across patterns is owned by the
// regression suites and the streaming-MPI suites of test_cpp_units, not by
// this harness
template <typename STREAMING>
using AMRLockConfig = LBM_CONFIG<
	TraitsSP,
	D3Q27_KernelStruct,
	NSE_Data_ConstInflow,
	D3Q27_SRT<TraitsSP>,
	typename D3Q27_SRT<TraitsSP>::EQ,
	STREAMING,
	D3Q27_BC_All,
	AMRLockMacro<TraitsSP>>;

// scenario state: periodic boundaries and a traveling sinusoidal density
// gradient as the initial condition (the subcycling suite's smooth
// kernel-detectable IC: the perturbation field evolves acoustically and
// crosses the L0->L1 link from cycle 1, so the cascade C2F fills and the
// F2C feedback carry evolving content every cycle)
template <typename NSE>
struct AMRLockState : State_AMR<NSE>
{
	using dreal = typename NSE::TRAITS::dreal;

	template <typename... ARGS>
	AMRLockState(ARGS&&... args)
	: State_AMR<NSE>(std::forward<ARGS>(args)...)
	{}

	void resetDFs() override
	{
		using lat_t = typename State_AMR<NSE>::lat_t;
		using idx = typename NSE::TRAITS::idx;
		using point_t = typename NSE::TRAITS::point_t;

		for (auto& block : this->nse.blocks) {
			const lat_t lat_local = (block.level == 0) ? this->nse.lat : block.lat_local;
			const typename NSE::TRAITS::idx3d offset = block.offset;

			block.setInitialCondition(
				[lat_local, offset] __cuda_callable__(typename NSE::template KernelStruct<dreal> & KS, idx gx, idx gy, idx gz) mutable
				{
					const point_t phys = lat_local.lbm2physPoint(gx - offset.x(), gy - offset.y(), gz - offset.z());
					KS.rho = 1 + 0.01f * TNL::sin(8.0f * phys.x());
					KS.vx = 0;
					KS.vy = 0;
					KS.vz = 0;
					NSE::COLL::setEquilibrium(KS);
				}
			);
		}

		this->nse.copyDFsToHost();
	}

	[[nodiscard]] std::vector<std::string> getOutputDataNames() const override
	{
		return {};
	}

	void outputData(
		UniformDataWriter<typename NSE::TRAITS>&, const LBM_BLOCK<NSE>&, const typename NSE::TRAITS::idx3d&, const typename NSE::TRAITS::idx3d&
	) override
	{}
};

// host snapshot of one block: the GEO map (when requested — interface
// tagging is init-time, so only the post-SimInit snapshot carries it) and
// every macroscopic channel over the whole stored box [offset - ov,
// offset + local + ov) including the overlap rows (the host mirrors take
// offset + local coordinates with the +-overlap margin — the
// amr_test_fixture.h captureFineGhost idiom). Macro values are widened to
// double element-by-element; float -> double widening is exact, so the
// snapshot preserves the device bits losslessly.
template <typename NSE>
AMRLockLevelSnap amrLockSnapshotBlock(LBM_BLOCK<NSE>& block, bool with_map)
{
	using idx = typename NSE::TRAITS::idx;

	AMRLockLevelSnap snap;
	snap.level = block.level;
	const idx ov_x = block.df_overlap_X();
	const idx ov_y = block.df_overlap_Y();
	const idx ov_z = block.df_overlap_Z();
	const idx beg_x = block.offset.x() - ov_x;
	const idx beg_y = block.offset.y() - ov_y;
	const idx beg_z = block.offset.z() - ov_z;
	snap.origin[0] = beg_x;
	snap.origin[1] = beg_y;
	snap.origin[2] = beg_z;
	snap.size[0] = block.local.x() + 2 * ov_x;
	snap.size[1] = block.local.y() + 2 * ov_y;
	snap.size[2] = block.local.z() + 2 * ov_z;
	snap.macro_channels = NSE::MACRO::N;

	if (with_map)
		for (idx z = beg_z; z < beg_z + snap.size[2]; z++)
			for (idx y = beg_y; y < beg_y + snap.size[1]; y++)
				for (idx x = beg_x; x < beg_x + snap.size[0]; x++)
					snap.map.push_back(block.hmap(x, y, z));

	for (int m = 0; m < NSE::MACRO::N; m++)
		for (idx z = beg_z; z < beg_z + snap.size[2]; z++)
			for (idx y = beg_y; y < beg_y + snap.size[1]; y++)
				for (idx x = beg_x; x < beg_x + snap.size[0]; x++)
					snap.macro.push_back(block.hmacro(m, x, y, z));

	return snap;
}

template <typename NSE>
void amrLockSnapshotCycle(AMRLockTrace& trace, AMRLockState<NSE>& state, bool with_map)
{
	AMRLockCycle cycle;
	for (auto& block : state.nse.blocks) {
		if (with_map)
			block.copyMapToHost();
		block.copyMacroToHost();
		cycle.levels.push_back(amrLockSnapshotBlock<NSE>(block, with_map));
	}
	// LBM::blocks is populated in creation order (level 0 first), but the
	// ascending-level order is the documented comparison contract, so sort
	// defensively (two blocks only)
	std::sort(
		cycle.levels.begin(),
		cycle.levels.end(),
		[](const AMRLockLevelSnap& a, const AMRLockLevelSnap& b)
		{
			return a.level < b.level;
		}
	);
	trace.cycles.push_back(std::move(cycle));
	trace.cycles_completed++;
}

template <typename NSE>
AMRLockTrace runAMRLockScenario(const std::string& pattern_label)
{
	using lat_t = Lattice<3, typename NSE::TRAITS::real, typename NSE::TRAITS::idx>;
	using point_t = typename NSE::TRAITS::point_t;
	using bool3d = typename NSE::TRAITS::bool3d;

	AMRLockTrace trace;
	trace.pattern = pattern_label;

	// the subcycling-suite lattice factory (amr_test_fixture.h's
	// makeLattice): the fully periodic 16^3 box in the sim_AMR physical
	// scaling, so the fine-level viscosity doubling and the time-step
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

	const std::string id = fmt::format("test_amr_bitwise_lock_{}", pattern_label);
	AMRLockState<NSE> state(id, MPI_COMM_WORLD, lat, "adios2.xml", /*periodic=*/bool3d{true, true, true}, /*max_level=*/1);
	if (! state.canCompute())
		return trace;

	createAMRBlocks(state.nse, parseAMRConfig<NSE>(AMR_LOCK_REGIONS));

	state.SimInit();
	if (state.nse.terminate)
		return trace;
	trace.init_ok = true;

	// cycle 0: the post-SimInit state (map + macro; the map tagging —
	// GEO_AMR_INTERFACE ring + frozen footprint — is init-time only)
	amrLockSnapshotCycle(trace, state, /*with_map=*/true);

	// the execute()-style iteration idiom of the subcycling suite: the
	// global updateKernelData clock arms level 0's rotation, one SimUpdate
	// call advances exactly one coarse cycle
	for (int c = 0; c < AMR_LOCK_CYCLES; c++) {
		state.updateKernelData();
		state.SimUpdate();
		if (state.nse.terminate)
			break;
		amrLockSnapshotCycle(trace, state, /*with_map=*/false);
	}
	return trace;
}
