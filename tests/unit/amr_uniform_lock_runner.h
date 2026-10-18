#pragma once

// Templated scenario runner of the UNIFORM cross-pattern AMR bitwise lock
// (the locked surface and its provenance are documented in
// amr_uniform_lock.h — the SP-6 surface of the out-of-scope survey): one
// uniform (max_level == 0) single-block State_AMR instance of the
// sim_AMR_ball channel geometry driven 10 coarse cycles through the real
// SimInit/SimUpdate machinery (which at this level count IS the base
// State<NSE> driver byte-identically), snapshotting the GEO map and the
// macroscopic channels after SimInit and after every cycle. Instantiated
// once per runner translation unit (test_amr_uniform_lock_<pattern>.cu), so
// the five pattern instantiations compile in parallel; the traces are
// compared in the kernel-free comparator TU test_amr_uniform_lock.cu.
//
// The config bundle, the macro policy shadow and the snapshot idiom are
// the bitwise locks', shared by include; only the scenario state
// (sim_AMR_ball's map authoring at uniform refinement) and the entry
// point differ.

#include <algorithm>
#include <cmath>
#include <utility>

#include <fmt/core.h>

#include "lbm3d/core.h"
#include "lbm3d/amr_state.h"
#include "lbm3d/obstacles_lbm.h"

#include "amr_bitwise_lock_runner.h"
#include "amr_uniform_lock.h"

// scenario state: sim_AMR_ball's --resolution 1 uniform-reference channel
// (62 x 32 x 32, inflow x = 1, interp outflow x = X-2, wall planes y/z at
// 1 and N-2, GEO_NOTHING on every edge plane, the staircased ball in the
// interior) driven through State_AMR at max_level == 0. The map authoring
// is the sim's setupBoundaries sequence including the FIXED
// stampBallOnFineBlocks contract (the max_level == 0 skip — the sim's
// guard is the SP-6 repair, replicated here so the gate's map matches the
// production one exactly; deleting the guard here reproduces the
// historical single-array abort on the first odd-parity launch)
template <typename NSE>
struct AMRUniformLockState : State_AMR<NSE>
{
	using typename State_AMR<NSE>::idx;
	using typename State_AMR<NSE>::idx3d;
	using typename State_AMR<NSE>::real;
	using typename State_AMR<NSE>::dreal;
	using typename State_AMR<NSE>::point_t;
	using BC = typename NSE::BC;

	using State<NSE>::nse;

	// problem parameters (set by the runner before SimInit)
	dreal phys_inflow_velocity = 0;
	real ball_diameter = 0;
	point_t ball_c;

	template <typename... ARGS>
	AMRUniformLockState(ARGS&&... args)
	: State_AMR<NSE>(std::forward<ARGS>(args)...)
	{}

	void setupBoundaries() override
	{
		nse.setBoundaryX(1, BC::GEO_INFLOW_MOMENT);								 // left
		nse.setBoundaryX(nse.lat.global.x() - 2, BC::GEO_OUTFLOW_RIGHT_INTERP);	 // right

		nse.setBoundaryY(1, BC::GEO_WALL);						 // front
		nse.setBoundaryY(nse.lat.global.y() - 2, BC::GEO_WALL);	 // back
		nse.setBoundaryZ(1, BC::GEO_WALL);						 // bottom
		nse.setBoundaryZ(nse.lat.global.z() - 2, BC::GEO_WALL);	 // top

		// extra layer needed due to A-A pattern (sim_AMR_ball's idiom — the
		// ghost-layer frame must stay GEO_NOTHING-clean on every pattern)
		nse.setBoundaryX(0, BC::GEO_NOTHING);						// left
		nse.setBoundaryX(nse.lat.global.x() - 1, BC::GEO_NOTHING);	// right
		nse.setBoundaryZ(0, BC::GEO_NOTHING);						// bottom
		nse.setBoundaryZ(nse.lat.global.z() - 1, BC::GEO_NOTHING);	// top
		nse.setBoundaryY(0, BC::GEO_NOTHING);						// front
		nse.setBoundaryY(nse.lat.global.y() - 1, BC::GEO_NOTHING);	// back

		lbmDrawSphere(nse, BC::GEO_WALL, ball_c, ball_diameter * 0.5);

		stampBallOnFineBlocks();
	}

	// sim_AMR_ball::stampBallOnFineBlocks verbatim (guard included): the
	// uniform fallback (max_level == 0) must not stamp through the level-0
	// block's zero-initialized lat_local (the SP-6 seed); level 0's ball is
	// lbmDrawSphere's job above
	void stampBallOnFineBlocks()
	{
		if (nse.max_level == 0)
			return;
		for (auto& block : nse.blocks) {
			if (block.level != nse.max_level)
				continue;
			const idx3d c = block.lat_local.phys2lbmPoint(ball_c);
			const real r = ball_diameter * 0.5 / block.lat_local.physDl;
			const idx range = ceil(r) + 1;
			idx n_ball = 0;
			for (idx py = c.y() - range; py <= c.y() + range; py++)
				for (idx pz = c.z() - range; pz <= c.z() + range; pz++)
					for (idx px = c.x() - range; px <= c.x() + range; px++) {
						const idx3d p{px, py, pz};
						if (TNL::l2Norm(p - c) >= r)
							continue;
						if (px < 0 || py < 0 || pz < 0 || px >= block.local.x() || py >= block.local.y() || pz >= block.local.z())
							continue;
						block.hmap(block.offset.x() + px, block.offset.y() + py, block.offset.z() + pz) = BC::GEO_WALL;
						n_ball++;
					}
			spdlog::info("fine block {} (level {}): stamped {} ball cells", block.id, block.level, n_ball);
		}
	}

	// per-block lattice conversion of the physical inflow velocity
	// (sim_AMR_ball's updateKernelVelocities idiom: level 0 uses nse.lat)
	void updateKernelVelocities() override
	{
		for (auto& block : nse.blocks) {
			const auto lat_local = (block.level == 0) ? nse.lat : block.lat_local;
			block.data.inflow_vx = lat_local.phys2lbmVelocity(phys_inflow_velocity);
			block.data.inflow_vy = 0;
			block.data.inflow_vz = 0;
		}
	}

	void resetDFs() override
	{
		using lat_t = typename State_AMR<NSE>::lat_t;

		// the bitwise locks' traveling sinusoidal density gradient (the
		// smooth, kernel-detectable IC; at the uniform level count the
		// inflow-driven development from rest alone moves too few cells
		// for the comparator's evolution-sanity guard)
		for (auto& block : nse.blocks) {
			const lat_t lat_local = (block.level == 0) ? nse.lat : block.lat_local;
			const idx3d offset = block.offset;

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

		nse.copyDFsToHost();
	}

	[[nodiscard]] std::vector<std::string> getOutputDataNames() const override
	{
		return {};
	}

	void outputData(UniformDataWriter<typename NSE::TRAITS>&, const LBM_BLOCK<NSE>&, const idx3d&, const idx3d&) override {}
};

// host snapshot of one uniform-level cycle: the level-0 block's GEO map
// (post-SimInit only) and every macroscopic channel over the whole stored
// box. At max_level == 0 the block carries no DF overlap, so the stored
// box is the bare interior [offset, offset + local) — the frame planes
// the lock's GEO_NOTHING-cleanliness assertion checks
template <typename NSE>
void amrUniformLockSnapshotCycle(AMRLockTrace& trace, AMRUniformLockState<NSE>& state, bool with_map)
{
	using idx = typename NSE::TRAITS::idx;

	AMRLockCycle cycle;
	for (auto& block : state.nse.blocks) {
		const idx ov_x = block.df_overlap_X();
		const idx ov_y = block.df_overlap_Y();
		const idx ov_z = block.df_overlap_Z();
		const idx beg_x = block.offset.x() - ov_x;
		const idx beg_y = block.offset.y() - ov_y;
		const idx beg_z = block.offset.z() - ov_z;

		AMRLockLevelSnap snap;
		snap.level = block.level;
		snap.origin[0] = beg_x;
		snap.origin[1] = beg_y;
		snap.origin[2] = beg_z;
		snap.size[0] = block.local.x() + 2 * ov_x;
		snap.size[1] = block.local.y() + 2 * ov_y;
		snap.size[2] = block.local.z() + 2 * ov_z;
		snap.macro_channels = NSE::MACRO::N;

		if (with_map) {
			block.copyMapToHost();
			for (idx z = beg_z; z < beg_z + snap.size[2]; z++)
				for (idx y = beg_y; y < beg_y + snap.size[1]; y++)
					for (idx x = beg_x; x < beg_x + snap.size[0]; x++)
						snap.map.push_back(block.hmap(x, y, z));
		}

		block.copyMacroToHost();
		for (int m = 0; m < NSE::MACRO::N; m++)
			for (idx z = beg_z; z < beg_z + snap.size[2]; z++)
				for (idx y = beg_y; y < beg_y + snap.size[1]; y++)
					for (idx x = beg_x; x < beg_x + snap.size[0]; x++)
						snap.macro.push_back(block.hmacro(m, x, y, z));

		cycle.levels.push_back(std::move(snap));
	}
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

// mirrored from the bitwise locks' scenario idiom (amr_bitwise_lock_runner.h):
// SimInit snapshot, then updateKernelData-armed SimUpdate cycles
template <typename NSE>
AMRLockTrace runAMRUniformLockScenario(const std::string& pattern_label)
{
	using lat_t = Lattice<3, typename NSE::TRAITS::real, typename NSE::TRAITS::idx>;
	using point_t = typename NSE::TRAITS::point_t;
	using bool3d = typename NSE::TRAITS::bool3d;

	AMRLockTrace trace;
	trace.pattern = pattern_label;

	// sim_AMR_ball's --resolution 1 geometry and unit derivations (the
	// uniform reference run the SP-6 abort fired on): domain H = 11 D
	// tall/deep, L = 2 H long, ball of diameter D at (2 D, H/2, H/2)
	const typename NSE::TRAITS::real ball_diameter = 0.1;						   // [m]
	const typename NSE::TRAITS::real real_domain_height = 11 * ball_diameter;	   // [m]
	const typename NSE::TRAITS::real real_domain_length = 2 * real_domain_height;  // [m]
	const int LBM_Y = 32;
	const int LBM_Z = LBM_Y;
	const typename NSE::TRAITS::real PHYS_DL = real_domain_height / ((typename NSE::TRAITS::real) LBM_Y - 2.0);
	const int LBM_X = (int) (real_domain_length / PHYS_DL) + 2;

	const typename NSE::TRAITS::real PHYS_VISCOSITY = 0.001;										// [m^2/s]
	const typename NSE::TRAITS::real LBM_VISCOSITY = 0.001;											// [Δx^2/Δt] (sim_3's value)
	const typename NSE::TRAITS::real PHYS_DT = LBM_VISCOSITY / PHYS_VISCOSITY * PHYS_DL * PHYS_DL;	// [s]
	const typename NSE::TRAITS::real Re = 100;
	const typename NSE::TRAITS::real PHYS_VELOCITY = Re * PHYS_VISCOSITY / ball_diameter;  // [m/s]

	lat_t lat;
	lat.global = typename lat_t::CoordinatesType(LBM_X, LBM_Y, LBM_Z);
	lat.physOrigin = point_t{0., 0., 0.};
	lat.physDl = PHYS_DL;
	lat.physDt = PHYS_DT;
	lat.physViscosity = PHYS_VISCOSITY;

	const std::string id = fmt::format("test_amr_uniform_lock_{}", pattern_label);
	// the channel around the ball: non-periodic everywhere — the inflow/
	// outflow planes and the wall planes are explicit BC cells set in
	// setupBoundaries() (sim_AMR_ball's arrangement)
	AMRUniformLockState<NSE> state(id, MPI_COMM_WORLD, lat, "adios2.xml", /*periodic=*/bool3d{false, false, false}, /*max_level=*/0);
	if (! state.canCompute())
		return trace;

	state.ball_c[0] = 2 * ball_diameter;
	state.ball_c[1] = 0.5 * real_domain_height;
	state.ball_c[2] = 0.5 * real_domain_height;
	state.ball_diameter = ball_diameter;
	state.phys_inflow_velocity = PHYS_VELOCITY;

	state.SimInit();
	if (state.nse.terminate)
		return trace;
	trace.init_ok = true;

	// cycle 0: the post-SimInit state (map + macro; the map carries the
	// GEO_NOTHING frame the lock's cleanliness assertion checks)
	amrUniformLockSnapshotCycle(trace, state, /*with_map=*/true);

	// the bitwise locks' iteration idiom: the global updateKernelData clock
	// arms the rotation, one SimUpdate call advances exactly one cycle (the
	// base driver's launch — the uniform fallback — under every pattern)
	for (int c = 0; c < AMR_UNIFORM_LOCK_CYCLES; c++) {
		state.updateKernelData();
		state.SimUpdate();
		if (state.nse.terminate)
			break;
		amrUniformLockSnapshotCycle(trace, state, /*with_map=*/false);
	}
	return trace;
}
