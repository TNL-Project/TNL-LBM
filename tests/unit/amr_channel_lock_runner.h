#pragma once

// Templated scenario runner of the cross-pattern channel-geometry lock
// (the surface is documented in amr_channel_lock.h): one mL1 developing-channel
// State_AMR instance — sim_AMR_channel's --resolution 1 boundary/IC plumbing
// verbatim minus the windbreak — with TWO level-1 footprints, one
// GEO_INFLOW_MOMENT-adjacent (the R = 1 Sec. 7.3 guarded inflow class of
// aa_seed_report.md §6) and one wall-attached (the channel's own byte-frozen
// slab), driven AMR_CHANNEL_LOCK_CYCLES coarse cycles through the real
// SimInit/SimUpdate machinery. Instantiated once per runner translation unit
// (test_amr_channel_lock_<pattern>.cu), so the five pattern instantiations
// compile in parallel; the traces are compared in the kernel-free comparator
// TU test_amr_channel_lock.cu.
//
// Per cycle the runner snapshots, for every block and over its whole stored
// box (interior + overlap rows):
//  - the GEO map (post-SimInit cycle only),
//  - the macroscopic channels (convention-free by construction),
//  - the DECODED CANONICAL DF FRAME: for direction q and site x the value a
//    kernel launch of the upcoming parity would gather as direction q's
//    pre-collision population there — the pattern's preCollisionSlot(q, x,
//    W) read with W the next consuming substep's parity (level 0: the
//    post-incremented global clock; level L: the cumulative substep counter,
//    mirroring updateKernelDataForLevel's even_iter = (substep % 2) == 1).
//    Pre-collision populations of a site are physical content, so this
//    frame is convention-free across streaming patterns; the host-side
//    decode below replicates each streaming header's preCollisionSlot
//    arithmetic exactly (and, for A-B pull, the physical frame the NEXT
//    rotation presents as logical df_cur — the complement of the present
//    one). Out-of-storage slots carry the amrChannelFrameMask() sentinel.
//    The canonical frame resolves differences the macroscopic surface is
//    blind to (non-hydrodynamic content) while staying pattern-agnostic.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <utility>

#include <fmt/core.h>

#include "lbm3d/core.h"
#include "lbm3d/amr_state.h"
#include "lbm3d/amr_decomposition.h"

#include "amr_channel_lock.h"

// (the mask helpers are defined once in the comparator TU — the runner TUs
// link them from amr_channel_lock.h's declarations)

// lock-specific macro policy (the AMRLockMacro precedent of
// amr_bitwise_lock_runner.h): D3Q27_MACRO_Default computes macroscopic
// quantities only on request, leaving the interior device macro stale across
// the Schönherr substeps; the per-cycle bitwise surface needs a per-substep
// refresh. The refresh is read-only with respect to the populations, so the
// DF trajectory — and hence the pattern-agreement claim — is identical to
// the default-policy run
template <typename TRAITS>
struct AMRChannelLockMacro : D3Q27_MACRO_Default<TRAITS>
{
	static const bool compute_in_each_iteration = true;
};

// per-pattern config bundle: identical policies to sim_AMR_channel's run<>
// (TraitsSP single precision, NSE_Data_ConstInflow, the production cumulant
// collision D3Q27_CUM with D3Q27_EQ_INV_CUM, the BC_All dispatch), with the
// streaming pattern made a template parameter and the lock macro policy
// above. The production collision is used (not the SRT mitigation of the
// mL1 bitwise lock): the SP-5 surfaces live in the BC+coupling interaction
// with GEO_INFLOW_MOMENT / the two-pass outflow, whose authored content is
// collision-arithmetic-specific, so the harness must run the production
// head (CUM) to reproduce the machine evidence it classifies.
template <typename STREAMING>
using AMRChannelLockConfig = LBM_CONFIG<
	TraitsSP,
	D3Q27_KernelStruct,
	NSE_Data_ConstInflow,
	D3Q27_CUM<TraitsSP, D3Q27_EQ_INV_CUM<TraitsSP>>,
	typename D3Q27_CUM<TraitsSP, D3Q27_EQ_INV_CUM<TraitsSP>>::EQ,
	STREAMING,
	D3Q27_BC_All,
	AMRChannelLockMacro<TraitsSP>>;

template <typename NSE>
AMRChannelLevelSnap amrChannelSnapshotBlock(LBM_BLOCK<NSE>& block, bool with_map, bool W);

// scenario state: sim_AMR_channel's R = 1 boundary map (inflow /
// outflow-interp / wall planes with the GEO_NOTHING edge layers, the
// face-generic fine-level wall machinery), uniform-at-rest IC, per-block
// inflow velocity — the windbreak rod array is omitted (the default
// --max-level 1 sim does not carry it either)
template <typename NSE>
struct AMRChannelLockState : State_AMR<NSE>
{
	using TRAITS = typename NSE::TRAITS;
	using BC = typename NSE::BC;
	using BLOCK = LBM_BLOCK<NSE>;
	using idx = typename TRAITS::idx;
	using idx3d = typename TRAITS::idx3d;
	using dreal = typename TRAITS::dreal;
	using real = typename TRAITS::real;
	using lat_t = typename State_AMR<NSE>::lat_t;

	dreal phys_inflow_velocity = 0;	 // [m/s] uniform inflow velocity

	template <typename... ARGS>
	AMRChannelLockState(ARGS&&... args)
	: State_AMR<NSE>(std::forward<ARGS>(args)...)
	{}

	// per-phase snapshot capture (launch-level attribution): armed by the
	// runner for cycles 1..phase_capture_upto; every capture point drains
	// the transfer streams first so the snapshot observes a quiesced state
	int phase_cycle = 0;
	int phase_capture_upto = 3;
	std::vector<std::pair<std::string, AMRChannelCycle>>* phase_out = nullptr;

	void capturePhase(const char* label)
	{
		if (phase_out == nullptr || phase_cycle < 1 || phase_cycle > phase_capture_upto)
			return;
		AMRChannelCycle cycle;
		for (auto& block : this->nse.blocks) {
			block.copyDFsToHost();
			block.copyMacroToHost();
			const bool W = block.data.even_iter;
			cycle.levels.push_back(amrChannelSnapshotBlock<NSE>(block, /*with_map=*/false, W));
		}
		std::sort(
			cycle.levels.begin(),
			cycle.levels.end(),
			[](const AMRChannelLevelSnap& a, const AMRChannelLevelSnap& b)
			{
				return a.level < b.level;
			}
		);
		phase_out->emplace_back(fmt::format("c{}:{}", phase_cycle, label), std::move(cycle));
	}

	void launchLBMKernelForLevel(int level, bool compute_macro, int ghost_layers) override
	{
		State_AMR<NSE>::launchLBMKernelForLevel(level, compute_macro, ghost_layers);
		// drain the default stream before the host snapshot so the device
		// state matches the launch order
		TNL::Backend::streamSynchronize(0);
		capturePhase(level == 0 ? "kernel(L0)" : (ghost_layers ? "substepA" : "substepB"));
	}

	void launchFineToCoarseTransfersInterior(int fine_level) override
	{
		State_AMR<NSE>::launchFineToCoarseTransfersInterior(fine_level);
		State_AMR<NSE>::synchronizeTransfers();
		capturePhase("f2c");
	}

	void launchCoarseToFineTransfers(int fine_level) override
	{
		State_AMR<NSE>::launchCoarseToFineTransfers(fine_level);
		State_AMR<NSE>::synchronizeTransfers();
		capturePhase("c2f");
	}

	// sim_AMR_channel::setupBoundaries at res == 1, verbatim except the
	// windbreak block (the header comment of sim_AMR_channel.cu documents
	// the same wall mechanics this suite exercises; stamp order is
	// load-bearing: symmetry/in-outflow first, walls next, GEO_NOTHING
	// ghost frame last)
	void setupBoundaries() override
	{
		this->nse.setBoundaryX(1, BC::GEO_INFLOW_MOMENT);									 // inflow (constant profile)
		this->nse.setBoundaryX(this->nse.lat.global.x() - 2, BC::GEO_OUTFLOW_RIGHT_INTERP);	 // outflow

		// bounce-back wall cells at z = 1 and z = 14 (res == 1: the sim's
		// res / 13*res+1 formulas)
		this->nse.setBoundaryZ(1, BC::GEO_WALL);   // bottom wall
		this->nse.setBoundaryZ(14, BC::GEO_WALL);  // top wall

		// GEO_NOTHING edge/dead layers (res == 1: z in {0, global-1=15})
		this->nse.setBoundaryZ(0, BC::GEO_NOTHING);								// below bottom wall
		this->nse.setBoundaryZ(this->nse.lat.global.z() - 1, BC::GEO_NOTHING);	// above top wall
		this->nse.setBoundaryX(0, BC::GEO_NOTHING);								// left edge
		this->nse.setBoundaryX(this->nse.lat.global.x() - 1, BC::GEO_NOTHING);	// right edge
		// y planes are intentionally untouched: periodic tangent direction

		// face-generic fine-level bounce-back walls (sim_AMR_channel's own
		// loop, windbreak omitted): on every footprint face whose
		// face-adjacent parent-level row is GEO_WALL the fine block imposes
		// its own wall — exactly the code the sim ships, minus its
		// windbreak section)
		// (storage precondition, the sim's own: the wall row sits at local
		// -2 and its GEO_NOTHING streaming buffer at local -3 — two rows
		// past the wall — so the hmap stamps stay inside the fine block's
		// stored host extent only because the scenario sets
		// storage_overlap_z = 3 on the fine blocks before SimInit
		// re-allocates; only the z faces fire in this registration, the
		// x/y parent rows are inflow/outflow/periodic fluid)
		for (auto& fine : this->nse.blocks) {
			if (fine.level == 0)
				continue;
			const idx3d& go = fine.global_offset;
			const idx3d gs{(fine.local.x() + 2) / 2, (fine.local.y() + 2) / 2, (fine.local.z() + 2) / 2};
			const struct FACE
			{
				const char* name;
				char axis_name;
				int axis;
				bool min_side;
			} faces[6] = {
				{"x-min", 'x', 0, true},
				{"x-max", 'x', 0, false},
				{"y-min", 'y', 1, true},
				{"y-max", 'y', 1, false},
				{"z-min", 'z', 2, true},
				{"z-max", 'z', 2, false},
			};
			for (const FACE& f : faces) {
				const int a = f.axis;
				const int b = (a + 1) % 3, c = (a + 2) % 3;
				const idx wall = f.min_side ? idx(-2) : fine.local[a] + 1;
				const idx buffer = f.min_side ? idx(-3) : fine.local[a] + 2;
				const idx plane = f.min_side ? go[a] - 1 : go[a] + gs[a];
				idx n_wall = 0;
				for (idx ib = 0; ib < fine.local[b]; ib++)
					for (idx ic = 0; ic < fine.local[c]; ic++) {
						idx3d fg{0, 0, 0};
						fg[b] = fine.offset[b] + ib;
						fg[c] = fine.offset[c] + ic;
						idx3d cg{0, 0, 0};
						cg[a] = plane;
						cg[b] = fg[b] / 2;
						cg[c] = fg[c] / 2;
						bool wall_column = false;
						for (auto& coarse : this->nse.blocks)
							if (coarse.level == fine.level - 1 && cg[0] >= coarse.offset.x() - coarse.df_overlap_X()
								&& cg[0] < coarse.offset.x() + coarse.local.x() + coarse.df_overlap_X()
								&& cg[1] >= coarse.offset.y() - coarse.df_overlap_Y()
								&& cg[1] < coarse.offset.y() + coarse.local.y() + coarse.df_overlap_Y()
								&& cg[2] >= coarse.offset.z() - coarse.df_overlap_Z()
								&& cg[2] < coarse.offset.z() + coarse.local.z() + coarse.df_overlap_Z()
								&& coarse.hmap(cg[0], cg[1], cg[2]) == BC::GEO_WALL)
								wall_column = true;
						if (! wall_column)
							continue;
						fg[a] = fine.offset[a] + wall;
						fine.hmap(fg[0], fg[1], fg[2]) = BC::GEO_WALL;
						fg[a] = fine.offset[a] + buffer;
						fine.hmap(fg[0], fg[1], fg[2]) = BC::GEO_NOTHING;
						n_wall++;
					}
				if (n_wall > 0)
					spdlog::info(
						"fine block {}: tagged {} {} columns with GEO_WALL (local {}={}) backed by GEO_NOTHING (local {}={})",
						fine.id,
						n_wall,
						f.name,
						f.axis_name,
						wall,
						f.axis_name,
						buffer
					);
			}
		}
	}

	// uniform-at-rest initial condition (developing channel), the sim's
	// setInitialCondition verbatim
	void setInitialCondition()
	{
		for (auto& block : this->nse.blocks)
			block.setEquilibrium(1, 0, 0, 0);

		this->nse.copyDFsToHost();
	}

	void resetDFs() override
	{
		setInitialCondition();
	}

	// per-block lattice conversion of the uniform inflow velocity (the sim's
	// updateKernelVelocities verbatim)
	void updateKernelVelocities() override
	{
		for (auto& block : this->nse.blocks) {
			const lat_t lat_local = (block.level == 0) ? this->nse.lat : block.lat_local;
			block.data.inflow_vx = lat_local.phys2lbmVelocity(phys_inflow_velocity);
			block.data.inflow_vy = 0;
			block.data.inflow_vz = 0;
		}
	}

	[[nodiscard]] std::vector<std::string> getOutputDataNames() const override
	{
		return {};
	}

	void outputData(UniformDataWriter<TRAITS>&, const BLOCK&, const idx3d&, const idx3d&) override {}
};

// ---- canonical-frame host replica of the streaming headers'
// preCollisionSlot arithmetic (see the file docstring). For A-B pull the
// physical frame the NEXT rotation presents as logical df_cur is the
// complement of the present one: the rotation state at snapshot time is
// S = dfsSwapped(block) (logical df_cur = phys[S ? 1 : 0]), so the next
// consuming launch rotates logical df_cur onto phys[1 - (S ? 1 : 0)] =
// phys[S ? 0 : 1]; single-array patterns read frame 0.
template <typename BLOCK>
int amrChannelCurFrame(BLOCK& block)
{
	return block.data.dfs[0] == block.dfs[1].getData() ? 1 : 0;
}

// physical host frame index the next consuming launch of an A-B two-frame
// block reads as df_cur (see above); DFMAX == 1 patterns ignore it
template <typename BLOCK>
int amrChannelNextCurPhysical(BLOCK& block)
{
	return amrChannelCurFrame(block) == 0 ? 1 : 0;
}

// range predicate of a candidate slot site in the block's GLOBAL host-array
// frame (mirrors preCollisionSlotInRange: the shifted slot site must lie in
// the stored extent [offset - ov, offset + local + ov) per axis)
template <typename BLOCK>
bool amrChannelSlotInRange(BLOCK& block, int x, int y, int z)
{
	return x >= block.offset.x() - block.df_overlap_X() && x < block.offset.x() + block.local.x() + block.df_overlap_X()
		&& y >= block.offset.y() - block.df_overlap_Y() && y < block.offset.y() + block.local.y() + block.df_overlap_Y()
		&& z >= block.offset.z() - block.df_overlap_Z() && z < block.offset.z() + block.local.z() + block.df_overlap_Z();
}

// canonical value of direction q at global site (x, y, z) for the upcoming
// parity W (see the file docstring); the amrChannelFrameMask() sentinel
// marks out-of-storage slots
template <typename NSE>
double amrChannelCanonical(LBM_BLOCK<NSE>& block, int q, int x, int y, int z, bool W)
{
	using STREAMING = typename NSE::STREAMING;
	using idx = typename NSE::TRAITS::idx;

	int rx = x, ry = y, rz = z;	 // candidate slot site (global frame)
	int slot = q;
	int frame = 0;
	if constexpr (NSE::DFMAX >= 2) {
		// A-B pull: own-direction slot one step upwind, in the physical
		// frame the next rotation presents as df_cur (parity arg ignored)
		frame = amrChannelNextCurPhysical(block);
		rx = x - dir27_cx(q);
		ry = y - dir27_cy(q);
		rz = z - dir27_cz(q);
	}
	else if constexpr (is_AA_v<STREAMING>) {
		if (W) {
			rx = x;
			ry = y;
			rz = z;
		}
		else {
			slot = opposite_direction(q);
			rx = x - dir27_cx(q);
			ry = y - dir27_cy(q);
			rz = z - dir27_cz(q);
		}
	}
	else if constexpr (is_ESO_TWIST_v<STREAMING>) {
		slot = W ? opposite_direction(q) : q;
		rx = x + (dir27_cx(q) < 0 ? 1 : 0);
		ry = y + (dir27_cy(q) < 0 ? 1 : 0);
		rz = z + (dir27_cz(q) < 0 ? 1 : 0);
	}
	else if constexpr (is_ESO_PULL_v<STREAMING>) {
		slot = W ? opposite_direction(q) : q;
		rx = x + (is_pair_head(q) ? 0 : -dir27_cx(q));
		ry = y + (is_pair_head(q) ? 0 : -dir27_cy(q));
		rz = z + (is_pair_head(q) ? 0 : -dir27_cz(q));
	}
	else if constexpr (is_ESO_PUSH_v<STREAMING>) {
		slot = W ? q : opposite_direction(q);
		rx = x + (is_pair_head(q) ? -dir27_cx(q) : 0);
		ry = y + (is_pair_head(q) ? -dir27_cy(q) : 0);
		rz = z + (is_pair_head(q) ? -dir27_cz(q) : 0);
	}
	else {
		static_assert(sizeof(STREAMING) == 0, "amrChannelCanonical is not specialized for this streaming pattern");
	}
	if (! amrChannelSlotInRange(block, rx, ry, rz))
		return amrChannelFrameMask();
	// +0.0f normalizes -0.0f (the sign patterns are the same physical
	// content; the production evidence surface compares values, not bits)
	return static_cast<double>(block.hfs[frame](slot, rx, ry, rz) + 0.0f);
}

// host snapshot of one block (the amrLockSnapshotBlock idiom extended with
// the canonical DF frame; W is the block's next consuming parity)
template <typename NSE>
AMRChannelLevelSnap amrChannelSnapshotBlock(LBM_BLOCK<NSE>& block, bool with_map, bool W)
{
	using idx = typename NSE::TRAITS::idx;

	AMRChannelLevelSnap snap;
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
				for (idx x = beg_x; x < beg_x + snap.size[0]; x++)	// +0.0f: the -0.0f normalization of amrChannelCanonical
					snap.macro.push_back(block.hmacro(m, x, y, z) + 0.0f);

	for (idx z = beg_z; z < beg_z + snap.size[2]; z++)
		for (idx y = beg_y; y < beg_y + snap.size[1]; y++)
			for (idx x = beg_x; x < beg_x + snap.size[0]; x++)
				for (int q = 0; q < NSE::Q; q++)
					snap.frame.push_back(amrChannelCanonical<NSE>(block, q, x, y, z, W));

	return snap;
}

template <typename NSE>
void amrChannelSnapshotCycle(AMRChannelTrace& trace, AMRChannelLockState<NSE>& state, bool with_map)
{
	AMRChannelCycle cycle;
	for (auto& block : state.nse.blocks) {
		if (with_map)
			block.copyMapToHost();
		block.copyMacroToHost();
		// the canonical frame decode reads hfs — it MUST be refreshed too
		// (the macro mirror alone leaves it stale across cycles)
		block.copyDFsToHost();
		const int level = block.level;
		// parity of the block's next consuming launch (level 0: the
		// post-incremented global clock — SimUpdate pre-increments
		// `iterations` for the cycle just executed, so the next launch's
		// substep index is the counter's current value verbatim; fine
		// levels: the cumulative substep counter — the same absolute
		// setters updateKernelDataForLevel applies)
		const bool W = (level == 0 ? state.nse.iterations : state.nse.totalSubstepCount[level]) % 2 == 1;
		cycle.levels.push_back(amrChannelSnapshotBlock<NSE>(block, with_map, W));
	}
	// LBM::blocks is populated in creation order (level 0 first), but the
	// ascending-level order is the documented comparison contract
	std::sort(
		cycle.levels.begin(),
		cycle.levels.end(),
		[](const AMRChannelLevelSnap& a, const AMRChannelLevelSnap& b)
		{
			return a.level < b.level;
		}
	);
	trace.cycles.push_back(std::move(cycle));
	trace.cycles_completed++;
}

template <typename NSE>
AMRChannelTrace runAMRChannelScenario(const std::string& pattern_label)
{
	using lat_t = Lattice<3, typename NSE::TRAITS::real, typename NSE::TRAITS::idx>;
	using point_t = typename NSE::TRAITS::point_t;
	using bool3d = typename NSE::TRAITS::bool3d;

	AMRChannelTrace trace;
	trace.pattern = pattern_label;

	// sim_AMR_channel's --resolution 1 lattice (the sim's formulas verbatim)
	const typename NSE::TRAITS::real LBM_VISCOSITY = 0.005f;	// [Δx^2/Δt]
	const typename NSE::TRAITS::real PHYS_HEIGHT = 0.041f;		// [m] channel height (z extent)
	const typename NSE::TRAITS::real PHYS_VISCOSITY = 1.5e-5f;	// [m^2/s]
	const typename NSE::TRAITS::real LBM_INFLOW = 0.1f;			// [Δx/Δt] uniform inflow lattice velocity
	const int X = 64, Y = 16, Z = 16;
	const typename NSE::TRAITS::real PHYS_DL = PHYS_HEIGHT / Z;
	const typename NSE::TRAITS::real PHYS_DT = LBM_VISCOSITY / PHYS_VISCOSITY * PHYS_DL * PHYS_DL;
	const typename NSE::TRAITS::real PHYS_INFLOW = LBM_INFLOW * PHYS_DL / PHYS_DT;

	lat_t lat;
	lat.global = typename lat_t::CoordinatesType(X, Y, Z);
	lat.physOrigin = point_t{0., 0., 0.};
	lat.physDl = PHYS_DL;
	lat.physDt = PHYS_DT;
	lat.physViscosity = PHYS_VISCOSITY;

	const std::string id = fmt::format("test_amr_channel_lock_{}", pattern_label);
	// planar channel: non-periodic x (inflow/outflow), periodic tangent y,
	// non-periodic z (walls) — the sim's periodicity triple
	AMRChannelLockState<NSE> state(id, MPI_COMM_WORLD, lat, "adios2.xml", /*periodic=*/bool3d{false, true, false}, /*max_level=*/1);
	if (! state.canCompute())
		return trace;

	state.phys_inflow_velocity = PHYS_INFLOW;

	state.nse.allocateHostData();
	state.nse.allocateDeviceData();
	state.nse.iterations = 0;
	// two level-1 slabs: x-min halo of the FIRST lies on the
	// GEO_INFLOW_MOMENT plane (origin 2 → halo on x = 1: the R = 1
	// inflow-adjacent Sec. 7.3 guard class of aa_seed_report.md §6 /
	// sim_AMR_ball's production geometry); the SECOND is sim_AMR_channel's
	// byte-frozen mL1 slab (z-min halo on the bottom wall plane z = 1)
	createAMRBlocks(state.nse, parseAMRConfig<NSE>("1 2 6 6 4 4 4\n1 24 4 2 16 8 8"));
	// the fine wall's GEO_NOTHING streaming buffer needs local z = -3:
	// the sim's per-block overlap override (SimInit's re-allocation
	// materializes it)
	for (auto& block : state.nse.blocks)
		if (block.level > 0)
			block.storage_overlap_z = 3;
	state.setInitialCondition();

	state.SimInit();
	if (state.nse.terminate)
		return trace;
	// markAMRInterface ran inside SimInit; the fine-level DFs must be pulled
	// back to the host mirrors before the first snapshot (reset()/SimInit
	// writes them on the device)
	state.nse.copyDFsToHost();
	state.nse.copyMapToHost();
	state.nse.copyMacroToHost();
	trace.init_ok = true;

	// cycle 0: the post-SimInit state (map + macro + canonical frame)
	amrChannelSnapshotCycle(trace, state, /*with_map=*/true);

	state.phase_out = &trace.phases;

	// the execute()-style iteration idiom (the wall-lane/subcycling
	// suites': updateKernelData arms level 0's rotation from the global
	// clock, one SimUpdate advances exactly one coarse cycle)
	for (int c = 0; c < AMR_CHANNEL_LOCK_CYCLES; c++) {
		state.phase_cycle = c + 1;
		state.updateKernelData();
		state.SimUpdate();
		if (state.nse.terminate)
			break;
		amrChannelSnapshotCycle(trace, state, /*with_map=*/false);
	}
	return trace;
}
