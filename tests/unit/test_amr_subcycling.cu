// Unit tests for the Berger-Colella time-subcycling orchestration in
// State_AMR (include/lbm3d/amr_state.h), scheduled as the Schönherr
// cycle with simulated band (docs/AMR-schonherr-ch7-target-contract.md
// sec. 1.3).
//
// Different from test_amr_coupling.cu (which exercises the coupling kernels
// directly), these tests drive SMALL two-level State_AMR instances through
// the real SimInit/SimUpdate machinery and verify the subcycling schedule
// from host-observable per-level state:
//
// - Test 1 (launch census): State_AMR declares its per-stage launch
//   helpers virtual, so the test subclass StateSchedule_AMR overrides
//   them to record one event per stage launch (kind, level, the parity
//   state at the call site, and the kernel launch extent class) and
//   delegates to the production implementations. Per cycle and level
//   pair the census is exactly 2 fine substeps + 1 coarse step + 2 fill
//   launches (1 F2C + 1 C2F -- the simulated band needs one fill, in
//   the substep-0 frame), with the AB-frame pairing asserted at every
//   call site (the parity table in test_subcycling_schedule) and the
//   widened substep-1 extent asserted from the ghost_layers argument;
//   for the A-A pattern the frame identities map to data.even_iter
//   states. SimInit's initial single-frame fill is asserted the same
//   way (1 C2F event, frame 0).
// - Test 2 (time synchronization): with the 2:1 refinement the fine level
//   performs exactly 2 substeps per coarse step (proven by Test 1's
//   census) and lat_local.physDt is exactly physDt/2, hence the physical
//   time advanced on both levels is identical after every coarse step.
// - Test 3 (max_level == 0 fallthrough): State_AMR with max_level == 0 must
//   behave identically to the base State driver - verified against a plain
//   State sibling on the same lattice run through the same sequence
//   (bitwise-identical DFs and macroscopic quantities on the host).
//
// The streaming pattern is selected at compile time (AB_PATTERN/AA_PATTERN);
// this suite is compiled into the consolidated doctest binaries
// test_amr_units_{ab,aa} (tests/unit/CMakeLists.txt), which provide main().
// Everything is single-rank.

// The shared fixture machinery (lattice factory, spy states, census carriers,
// reference-stat helpers) lives in tests/unit/amr_test_fixture.h so that
// test_amr_nesting.cu reuses it; this file keeps the tests themselves
// (extraction precondition of the amr-nlevel-nesting commit B).
#include "amr_test_fixture.h"

TEST_SUITE_BEGIN("amr_subcycling");

// Tests 1 and 2: cycle launch census with parity-at-call-site locks, and
// time synchronization. Per SimUpdate call the schedule must record
// exactly this ordered census per level pair (plan contract sec. 1.3,
// cycle steps named in State_AMR's file docstring):
//
//   #   stage        AB frame pairing at the call site (P = dfs[0], Q = dfs[1])
//   1   kernel L1    fine df_cur == P, ghost_layers == 1 (simulated band:
//                    substep 1 integrates the inner ghost rows of P,
//                    sourcing the outer row's fill, produces Q)
//   2   kernel L1    fine df_cur == Q, ghost_layers == 0 (interior-only:
//                    substep 2 consumes substep 1's updated inner ghost
//                    rows of Q, produces P)
//   3   kernel L0    coarse step, ghost_layers == 0 (rotation from the
//                    global iterations clock)
//   4   f2c L1       fine df_out == P (reads the post-substep-2 array; the
//                    rotation is still substep-1's, so df_cur == Q)
//   5   c2f L1       fine df_cur == P (the single fill of the cycle:
//                    frame 0 for the next cycle's substep 1; prepared by
//                    updateKernelDataForLevel(L, 0))
//
// i.e. 2 fine substeps + 1 coarse step + 2 fill launches (1 F2C + 1 C2F)
// per cycle per level -- the simulated band consumes the other frame's
// inner ghost rows as substep 1's kernel output, so a frame-1 fill would
// be dead traffic. The coarse rotation is set once per cycle by the
// global updateKernelData() and must not change across events 3-5 (no
// fine-level preparation touches level 0). Under AA the frame identities
// map to the even_iter values false, true, -, true (f2c reads the twisted
// post-collision state), false.
void test_subcycling_schedule()
{
	lat_t lat = makeLattice();
	const std::string id = fmt::format("test_amr_subcycling_{}_sched", pattern_name);
	StateSchedule_AMR<NSE_CONFIG> state(id, MPI_COMM_WORLD, lat, "adios2.xml", /*periodic=*/TRAITS::bool3d{true, true, true}, /*max_level=*/1);
	if (! state.canCompute()) {
		CHECK_MESSAGE(false, "Test 1/2 setup: state.canCompute()");
		return;
	}

	// one centered level-1 region: coarse footprint [4, 12)^3, i.e. a 14^3
	// fine interior (local = 2*8 - 2) at fine-global offset (9, 9, 9)
	createAMRBlocks(state.nse, parseAMRConfig<NSE_CONFIG>("1 4 4 4 8 8 8"));

	const std::vector<BLOCK*> level0 = state.nse.getBlocksAtLevel(0);
	const std::vector<BLOCK*> level1 = state.nse.getBlocksAtLevel(1);
	CHECK_MESSAGE((level0.size() == 1 && level1.size() == 1), "Test 1 setup: one level-0 block and one level-1 block created");
	if (level0.empty() || level1.empty())
		return;
	BLOCK* coarse = level0.front();
	BLOCK* fine = level1.front();

	// full initialization (allocation, boundary setup, interface tagging,
	// coupling patches, initial condition, initial single-frame fill)
	state.SimInit();
	if (state.nse.terminate) {
		CHECK_MESSAGE(false, "Test 1/2 setup: SimInit triggered the terminate flag");
		return;
	}

	CHECK_MESSAGE((! state.couplings.empty() && ! state.couplings.front().patches.empty()), "Test 1 setup: interface coupling patches were built in SimInit");

	const double nu_lb_coarse = state.nse.lat.lbmViscosity();
	const double nu_lb_fine = fine->lat_local.lbmViscosity();
	CHECK_MESSAGE(
		std::abs(nu_lb_fine - 2 * nu_lb_coarse) <= 1e-12 * nu_lb_coarse,
		fmt::format("Test 1 setup: fine lattice viscosity is doubled (nu_fine = {:.6e}, nu_coarse = {:.6e})", nu_lb_fine, nu_lb_coarse)
	);

	// SimInit's initial single-frame fill (the cycle-0 anchor of the
	// Schönherr cycle): one C2F launch filling the substep-0 frame
	// rotation, before any SimUpdate
	if (state.events.size() != 1 || state.events[0].stage != StateSchedule_AMR<NSE_CONFIG>::Stage::c2f) {
		CHECK_MESSAGE(false, fmt::format("Test 1 setup: SimInit launched {} events, expected exactly 1 (C2F frame 0)", state.events.size()));
		return;
	}
#ifdef AB_PATTERN
	{
		const void* const P = fine->dfs[0].getData();
		CHECK_MESSAGE(
			state.events[0].fine_cur == P,
			"Test 1 setup: SimInit's initial fill targeted frame P (the substep-0 rotation)"
		);
	}
#elif defined(AA_PATTERN)
	CHECK_MESSAGE(
		state.events[0].fine_even == false,
		"Test 1 setup: SimInit's initial fill ran at even_iter false (the substep-0 parity)"
	);
#endif
	// consume the SimInit events so the cycle census starts empty
	state.events.clear();

	// execute()-style iteration: updateKernelData before each SimUpdate
	bool census_ok = true;
	bool visc_ok = true;
	bool sync_ok = true;
	for (int call = 1; call <= 3 && ! state.nse.terminate; call++) {
		const std::size_t base = state.events.size();
		state.updateKernelData();
		state.SimUpdate();

		// one call of SimUpdate = exactly one coarse iteration
		const bool iter_ok = (state.nse.iterations == call);

		// the launch census and the parity-at-call-site table (see the
		// header comment); every failing event is dumped for the log
		using Evt = typename StateSchedule_AMR<NSE_CONFIG>::Event;
		using Stage = typename StateSchedule_AMR<NSE_CONFIG>::Stage;
		const Evt* ev = state.events.size() >= base + 5 ? state.events.data() + base : nullptr;
		bool call_ok = iter_ok && ev != nullptr;
#ifdef AB_PATTERN
		const void* const P = fine->dfs[0].getData();
		const void* const Q = fine->dfs[1].getData();
		const void* const expected_coarse = ((call - 1) % 2 == 0) ? coarse->dfs[0].getData() : coarse->dfs[1].getData();
		if (call_ok) {
			call_ok = ev[0].stage == Stage::kernel && ev[0].level == 1 && ev[0].fine_cur == P && ev[0].fine_out == Q
				   && ev[0].ghost_layers == 1;
			call_ok = call_ok && ev[1].stage == Stage::kernel && ev[1].level == 1 && ev[1].fine_cur == Q && ev[1].fine_out == P
					   && ev[1].ghost_layers == 0;
			call_ok = call_ok && ev[2].stage == Stage::kernel && ev[2].level == 0 && ev[2].coarse_cur == expected_coarse
					   && ev[2].ghost_layers == 0;
			call_ok = call_ok && ev[3].stage == Stage::f2c && ev[3].level == 1 && ev[3].fine_cur == Q && ev[3].fine_out == P;
			call_ok = call_ok && ev[4].stage == Stage::c2f && ev[4].level == 1 && ev[4].fine_cur == P;
			// the coarse rotation must not change across events 3-5 (no
			// fine-level preparation may touch level 0)
			call_ok = call_ok && ev[3].coarse_cur == expected_coarse && ev[4].coarse_cur == expected_coarse;
		}
#elif defined(AA_PATTERN)
		const bool expected_coarse_even = ((call - 1) % 2 == 1);
		if (call_ok) {
			call_ok = ev[0].stage == Stage::kernel && ev[0].level == 1 && ev[0].fine_even == false && ev[0].ghost_layers == 1;
			call_ok = call_ok && ev[1].stage == Stage::kernel && ev[1].level == 1 && ev[1].fine_even == true
					   && ev[1].ghost_layers == 0;
			call_ok = call_ok && ev[2].stage == Stage::kernel && ev[2].level == 0 && ev[2].coarse_even == expected_coarse_even
					   && ev[2].ghost_layers == 0;
			call_ok = call_ok && ev[3].stage == Stage::f2c && ev[3].level == 1 && ev[3].fine_even == true;
			call_ok = call_ok && ev[4].stage == Stage::c2f && ev[4].level == 1 && ev[4].fine_even == false;
			call_ok = call_ok && ev[3].coarse_even == expected_coarse_even && ev[4].coarse_even == expected_coarse_even;
		}
#endif
		if (! call_ok) {
			census_ok = false;
			CHECK_MESSAGE(
				false,
				fmt::format(
					"Test 1 launch census after call {}: iterations = {}, {} events recorded -- {} | {}",
					call,
					state.nse.iterations,
					state.events.size() - base,
					levelStateString(state, *coarse, 0),
					levelStateString(state, *fine, 1)
				)
			);
		}

		// per-level viscosity restored by updateKernelDataForLevel during
		// the subcycling (level 0 must keep the global value); the values
		// are stored as floats, so the tolerance is the float rounding level
		const double nu_l1 = fine->data.lbmViscosity;
		const double nu_l0 = coarse->data.lbmViscosity;
		if (std::abs(nu_l1 - nu_lb_fine) > 1e-6 * nu_lb_fine || std::abs(nu_l0 - nu_lb_coarse) > 1e-6 * nu_lb_coarse) {
			visc_ok = false;
			CHECK_MESSAGE(false, fmt::format("Test 1 per-level viscosity after call {}: level0 = {:.6e}, level1 = {:.6e}", call, nu_l0, nu_l1));
		}

		// Test 2: time synchronization -- the fine level performs exactly 2
		// substeps per coarse step (steps 1-2 of the census above) and its
		// time step is exactly physDt/2, so both level clocks agree after
		// every cycle
		const double t_coarse = state.nse.iterations * static_cast<double>(state.nse.lat.physDt);
		const long fine_substeps = 2L * state.nse.iterations;
		const double t_fine = fine_substeps * static_cast<double>(fine->lat_local.physDt);
		const bool dt_ok = (2.0 * static_cast<double>(fine->lat_local.physDt) == static_cast<double>(state.nse.lat.physDt))
						&& (2.0 * static_cast<double>(fine->lat_local.physDl) == static_cast<double>(state.nse.lat.physDl));
		if (! (dt_ok && t_coarse == t_fine)) {
			sync_ok = false;
			CHECK_MESSAGE(false, fmt::format("Test 2 time sync after call {}: t_coarse = {:.17e}, t_fine = {:.17e}", call, t_coarse, t_fine));
		}
	}

	CHECK_MESSAGE((
		state.nse.iterations == 3 && state.events.size() == 3 * 5 && census_ok && visc_ok),
		"Test 1 launch census: 3 coarse iterations, each recording 2 fine substeps + 1 coarse step + 2 fill launches (1 F2C + 1 C2F) with the "
		"parity-at-call-site and launch-extent table asserted (per-level viscosities restored)"
	);
	CHECK_MESSAGE(
		sync_ok, "Test 2 time synchronization: fine clock (2 substeps of dt/2) equals coarse clock (1 step of dt) after every Berger-Colella step"
	);
	CHECK_MESSAGE(! state.nse.terminate, "Test 1/2: no termination or kernel failure during 3 subcycled iterations");
}

// Test 3: max_level == 0 fallthrough must be identical to the base driver.
// The AMR state and the plain base State run SEQUENTIALLY (one instance at
// a time) and the AMR snapshots are compared against the sibling's live
// host arrays afterwards.
void test_max_level_zero_fallthrough()
{
	lat_t lat = makeLattice();

	HostSnapshot snap_amr_init, snap_amr_step;
	{
		const std::string id_amr = fmt::format("test_amr_subcycling_{}_amr0", pattern_name);
		StateLocal_AMR<NSE_CONFIG> state_amr(id_amr, MPI_COMM_WORLD, lat, "adios2.xml", /*periodic=*/TRAITS::bool3d{true, true, true});
		if (! state_amr.canCompute()) {
			CHECK_MESSAGE(false, "Test 3 setup: state.canCompute()");
			return;
		}
		CHECK_MESSAGE(state_amr.nse.max_level == 0, "Test 3 setup: State_AMR constructed with max_level == 0 (no extra LBM args)");

		state_amr.SimInit();
		if (state_amr.nse.terminate) {
			CHECK_MESSAGE(false, "Test 3 setup: SimInit triggered the terminate flag");
			return;
		}
		snap_amr_init = snapshotBlock(state_amr.nse.blocks.front());

		// force macroscopic output inside the kernel so that the step's
		// effect is visible in dmacro (same decision in both drivers)
		state_amr.cnt[OUT3DCUT].period = 1e-30;

		// run exactly one iteration of the execute() loop body
		state_amr.updateKernelData();
		state_amr.SimUpdate();

		CHECK_MESSAGE((state_amr.nse.iterations == 1 && ! state_amr.nse.terminate), "Test 3: one SimUpdate completed (iterations == 1, no termination)");
		CHECK_MESSAGE((
			state_amr.nse.blocks.size() == 1 && state_amr.nse.blocks.front().level == 0),
			"Test 3: no fine-level blocks or per-level data exist at max_level == 0"
		);

		// snapshot the post-step state (copy to host first)
		for (auto& block : state_amr.nse.blocks) {
			block.copyDFsToHost();
			block.copyMacroToHost();
		}
		snap_amr_step = snapshotBlock(state_amr.nse.blocks.front());
	}

	{
		const std::string id_base = fmt::format("test_amr_subcycling_{}_base0", pattern_name);
		StateLocal_Base<NSE_CONFIG> state_base(id_base, MPI_COMM_WORLD, lat, "adios2.xml", /*periodic=*/TRAITS::bool3d{true, true, true});
		if (! state_base.canCompute()) {
			CHECK_MESSAGE(false, "Test 3 sibling setup: state.canCompute()");
			return;
		}

		state_base.SimInit();
		if (state_base.nse.terminate) {
			CHECK_MESSAGE(false, "Test 3 sibling setup: SimInit triggered the terminate flag");
			return;
		}

		// identical initialization (bitwise)
		const double init_diff = maxAbsDiffSnapshot(state_base.nse.blocks.front(), snap_amr_init);
		CHECK_MESSAGE(init_diff == 0, fmt::format("Test 3 initialization bitwise identical to base (max |diff| = {:.3e})", init_diff));

		state_base.cnt[OUT3DCUT].period = 1e-30;
		state_base.updateKernelData();
		state_base.SimUpdate();

		for (auto& block : state_base.nse.blocks) {
			block.copyDFsToHost();
			block.copyMacroToHost();
		}
		const double step_diff = maxAbsDiffSnapshot(state_base.nse.blocks.front(), snap_amr_step);
		CHECK_MESSAGE(step_diff == 0, fmt::format("Test 3 one-step result bitwise identical to the base driver (max |diff| = {:.3e})", step_diff));

		// the non-uniform IC must actually evolve in one step (guards
		// against the comparison passing vacuously when no kernel ran)
		double max_rho_change = 0;
		const BLOCK& block = state_base.nse.blocks.front();
		for (idx z = 0; z < block.local.z(); z++)
			for (idx y = 0; y < block.local.y(); y++)
				for (idx x = 0; x < block.local.x(); x++) {
					const double initial = 1.0 + 0.01 * TNL::sin(8.0 * (x * lat.physDl));
					max_rho_change = std::max(max_rho_change, std::abs(static_cast<double>(block.hmacro(NSE_CONFIG::MACRO::e_rho, x, y, z)) - initial));
				}
		CHECK_MESSAGE(max_rho_change > 0, fmt::format("Test 3 sanity: the step advanced the state (max |rho change| = {:.3e})", max_rho_change));
	}
}

// Test 4 (B.5 lock, sole variant since D.1): after one cycle every ring
// cell's macros must equal the KERNEL-PRODUCED values, and the fine-level
// C2F destination cells must carry the single-fill, simulated-band state of
// the Schönherr cycle contract: the cycle-end fill rewrites BOTH overlap
// layers of the substep-0 frame (frame 0), the kernel INTEGRATES the inner
// overlap layer during substep 1 (its updated rows land in the other
// frame), and the outer layer of the other frame keeps the SimInit-era
// fill (no frame-1 fill exists under the simulated band). Ring cells are
// collision-active, and no coupling channel writes them (the skin F2C
// rectangles lie inside the footprint = GEO_NOTHING region, disjoint from
// the ring; C2F writes fine ghost cells only), so the kernel is their only
// writer. Detection for the ring: compare against a DETERMINISTIC
// kernels-only reference -- the same initialization advanced by the SAME
// SimUpdate with ALL coupling launches disabled (couplings.clear() after
// SimInit; every transfer helper is then a silent no-op with empty
// couplings, while the parity, counter, and kernel flow of SimUpdate stay
// identical). The reference stays valid under the cycle: SimInit's initial
// fill ran BEFORE the clear() and touches fine ghost cells only, and in
// both runs the only writer of ring cells is the coarse kernel on
// identical input, so the expected maximum difference is exactly 0
// (assert == 0 and print the max). NOTE: the reference's FOOTPRINT cells
// intentionally diverge (frozen init state, never skin-written), so every
// cross-state ring comparison below is map-selected on GEO_AMR_INTERFACE;
// the FINE ghost cells are compared separately below against the fill and
// simulated-band locks (the subject's cycle-end frame-0 fill vs the
// reference's SimInit-era fill; the frame-1 inner overlap layer, which the
// kernel updates identically in both runs).
//
// Why not the placeholder probe alone: the (rho = 1, v = 0) freshness the
// historical test asserts does NOT detect a coupling channel's absence on
// this driver path -- on the non-uniform sine IC the collision-active
// kernel computes real macros at ring cells and evicts (1,0,0,0) either
// way (B.1 finding, mock-matrix.md subcycling item 4). The kernels-only
// reference comparison closes that blind spot and keeps this test the
// regression lock for the placeholder-defect class: a resurrected coupling
// write onto ring cells (e.g. a re-added ring launch) overwrites the ring
// macros with F2C-filtered values (trips the reference comparison), a
// collision-inactive kernel regression produces placeholders at ring cells
// (trips the kept placeholder probe), and a coupling channel misfiring
// onto ring cells trips the reference comparison.
//
// D.1 (2026-08-16, gate B ruling): the ring fine-to-coarse launch this
// test's former detection demonstration emulated was HARD-DELETED. The
// demonstration block (a third state + one manual emulation of the retired
// ring launch + its teeth/confinement assertions) was removed with it: at
// storage_overlap = 1 the emulated launch is a no-op BY DESIGN (the F2C
// kernel's per-cell storability guard skips every ring cell), which is the
// one red assertion this file carried (issues.md, Phase C.3). The gate-
// relevant LOCK-1 (bitwise kernels-only reference) and LOCK-2 (placeholder
// probe) assertions below are unchanged and remain green.
void test_interface_ring_freshness()
{
	lat_t lat = makeLattice();

	// subject: one full coupled Schönherr cycle
	CoarseMacroScan subject;
	FineGhostScan subject_ghost;
	FineGhostScan subject_ghost_init;
	HostSnapshot subject_init;
	{
		const std::string id = fmt::format("test_amr_subcycling_{}_ring", pattern_name);
		StateLocal_AMR<NSE_CONFIG> state(id, MPI_COMM_WORLD, lat, "adios2.xml", /*periodic=*/TRAITS::bool3d{true, true, true}, /*max_level=*/1);
		if (! state.canCompute()) {
			CHECK_MESSAGE(false, "Test 4 setup: state.canCompute()");
			return;
		}

		// one centered level-1 region with the coarse footprint [4, 12)^3: the
		// GEO_AMR_INTERFACE ring is the 10^3 - 6^3 = 784 cells around it
		// (halo 488 + reactivated c=0 shell 296, contract sec. 2.4)
		createAMRBlocks(state.nse, parseAMRConfig<NSE_CONFIG>("1 4 4 4 8 8 8"));

		state.SimInit();
		if (state.nse.terminate) {
			CHECK_MESSAGE(false, "Test 4 setup: SimInit triggered the terminate flag");
			return;
		}

		BLOCK* coarse = state.nse.getBlocksAtLevel(0).front();
		subject_init = snapshotBlock(*coarse);

		// pre-cycle fine ghost snapshot (the SimInit-era fill): baseline of
		// the simulated-band locks -- the frame-1 inner overlap rows must be
		// CHANGED by the widened substep-1 kernel integration, the frame-1
		// outer rows must be UNCHANGED (no frame-1 fill exists, and the
		// kernel never reaches the outer layer)
		BLOCK* fine = state.nse.getBlocksAtLevel(1).front();
		fine->copyDFsToHost();
		subject_ghost_init = captureFineGhost(*fine);

		// force macroscopic output inside the kernel (same flag as the
		// default variant) so the kernel-produced ring macros land in dmacro
		state.cnt[OUT3DCUT].period = 1e-30;

		// one coupled Schönherr cycle (2 fine substeps + coarse step + F2C +
		// the cycle-end single-frame C2F fill; ring F2C removed in D.1, the
		// skin F2C transfer runs unconditionally as the only fine-to-coarse
		// channel)
		state.updateKernelData();
		state.SimUpdate();
		if (state.nse.terminate) {
			CHECK_MESSAGE(false, "Test 4: SimUpdate triggered the terminate flag");
			return;
		}

		coarse->copyMapToHost();
		coarse->copyMacroToHost();
		subject = captureCoarseMacros(*coarse);

		// the fill locks' subject carrier: the fine block's C2F destination
		// cells of BOTH AB frames after the cycle (frame 0's complement was
		// rewritten by the cycle-end fill; frame 1's inner overlap layer was
		// INTEGRATED by the widened substep-1 kernel and frame 1's outer
		// layer still holds the SimInit-era fill, as no frame-1 fill exists)
		fine->copyMapToHost();
		fine->copyDFsToHost();
		subject_ghost = captureFineGhost(*fine);
	}

	// deterministic kernels-only reference (the plan's construction):
	// same initialization, advanced by the SAME SimUpdate with ALL coupling
	// launches disabled
	CoarseMacroScan reference;
	FineGhostScan reference_ghost;
	{
		const std::string id = fmt::format("test_amr_subcycling_{}_ringref", pattern_name);
		StateLocal_AMR<NSE_CONFIG> state(id, MPI_COMM_WORLD, lat, "adios2.xml", /*periodic=*/TRAITS::bool3d{true, true, true}, /*max_level=*/1);
		if (! state.canCompute()) {
			CHECK_MESSAGE(false, "Test 4 setup: reference state.canCompute()");
			return;
		}
		createAMRBlocks(state.nse, parseAMRConfig<NSE_CONFIG>("1 4 4 4 8 8 8"));
		state.SimInit();
		if (state.nse.terminate) {
			CHECK_MESSAGE(false, "Test 4 setup: reference SimInit triggered the terminate flag");
			return;
		}

		// deterministic-initialization anchor (Test 3 proves the init is
		// reproducible bitwise across instances)
		const HostSnapshot reference_init = snapshotBlock(*state.nse.getBlocksAtLevel(0).front());
		const double init_diff = maxAbsDiffSnapshots(subject_init, reference_init);
		CHECK_MESSAGE(
			init_diff == 0,
			fmt::format(
				"Test 4 reference init: kernels-only reference is bitwise identical to the subject after SimInit (max |diff| = {:.3e})", init_diff
			)
		);

		// disable ALL coupling launches (C2F, ring F2C, interior/skin F2C):
		// clear() after SimInit leaves the SimUpdate parity/counter/kernel
		// flow untouched and makes every transfer launch a no-op (the
		// reference's fine ghost cells therefore keep the SimInit-era
		// initial fill forever)
		state.couplings.clear();
		state.cnt[OUT3DCUT].period = 1e-30;
		state.updateKernelData();
		state.SimUpdate();
		if (state.nse.terminate) {
			CHECK_MESSAGE(false, "Test 4: reference SimUpdate triggered the terminate flag");
			return;
		}

		BLOCK* coarse = state.nse.getBlocksAtLevel(0).front();
		coarse->copyMapToHost();
		coarse->copyMacroToHost();
		reference = captureCoarseMacros(*coarse);

		BLOCK* fine = state.nse.getBlocksAtLevel(1).front();
		fine->copyDFsToHost();
		reference_ghost = captureFineGhost(*fine);
	}

	// assertions
	// the ring layout is fixed by markAMRInterface: the 784-cell ring of
	// the 8^3 K = 8 footprint (halo (K+2)^3 - K^3 = 488 plus the
	// reactivated c=0 surface shell K^3 - (K-2)^3 = 296, contract sec.
	// 2.4; same geometry in all three scans)
	idx ring_cells = 0;
	for (const int tag : subject.map)
		if (tag == NSE_CONFIG::BC::GEO_AMR_INTERFACE)
			ring_cells++;
	CHECK_MESSAGE(
		ring_cells == 10 * 10 * 10 - 6 * 6 * 6,
		fmt::format("Test 4 setup: GEO_AMR_INTERFACE ring has 784 cells around the 8^3 footprint (got {})", ring_cells)
	);

	// LOCK 1 (the plan's B.5 replacement assertion): every ring cell's
	// macros equal the kernels-only reference, bitwise; LOCK 2 (kept): no
	// ring cell holds the (rho = 1, v = 0) placeholder (collision-inactive
	// defect class: ring cells would hold fresh IC values when their only
	// writer regressed)
	double max_ring_diff = 0;
	idx placeholder_cells = 0;
	for (std::size_t i = 0, c = 0; i < subject.map.size(); i++, c += 4) {
		if (subject.map[i] != NSE_CONFIG::BC::GEO_AMR_INTERFACE)
			continue;
		for (int m = 0; m < 4; m++)
			max_ring_diff = std::max(max_ring_diff, std::abs(subject.vals[c + m] - reference.vals[c + m]));
		if (subject.vals[c] == static_cast<real>(1) && subject.vals[c + 1] == static_cast<real>(0) && subject.vals[c + 2] == static_cast<real>(0)
			&& subject.vals[c + 3] == static_cast<real>(0))
			placeholder_cells++;
	}
	CHECK_MESSAGE(
		max_ring_diff == 0,
		fmt::format(
			"Test 4 ring macros kernel-produced (ring F2C removed, D.1): all {} GEO_AMR_INTERFACE cells' macros are bitwise identical to the "
			"kernels-only reference (max |diff| = {:.3e})",
			ring_cells,
			max_ring_diff
		)
	);
	CHECK_MESSAGE(
		placeholder_cells == 0,
		fmt::format(
			"Test 4 freshness: all {} GEO_AMR_INTERFACE cells hold kernel-produced macros ({} still hold the rho=1, v=0 placeholder)",
			ring_cells,
			placeholder_cells
		)
	);

	// fill-destination census: the fine block's C2F destination complement
	// is (local + 2*ov)^3 - local^3 cells (K = 8 fixture: 18^3 - 14^3 =
	// 3,088 cells, 27 DF entries each)
	const long ghost_cells = static_cast<long>(subject_ghost.frame0.size()) / NSE_CONFIG::Q;
	CHECK_MESSAGE((
		subject_ghost.frame0.size() == 27 * 3088 && subject_ghost.frame0.size() == reference_ghost.frame0.size()),
		fmt::format("Test 4 fill census: the fine block has 3,088 fill-owned destination cells (got {})", ghost_cells)
	);

	// LOCK 3 (ghost-map / simulated-band precondition): every C2F
	// destination cell must be GEO_FLUID -- the widened substep-1 kernel
	// dispatches BCs on the inner overlap rows and must collide+stream
	// there like on interior fluid
	idx non_fluid_ghosts = 0;
	for (std::size_t i = 0; i < subject_ghost.map.size(); i++)
		if (subject_ghost.map[i] != NSE_CONFIG::BC::GEO_FLUID)
			non_fluid_ghosts++;
	CHECK_MESSAGE(
		non_fluid_ghosts == 0,
		fmt::format(
			"Test 4 ghost map: all {} destination cells are GEO_FLUID ({} non-fluid entries) so the widened substep-1 kernel "
			"integrates the inner overlap rows",
			ghost_cells,
			non_fluid_ghosts
		)
	);

	// LOCK 4 (fill coverage, Schönherr band): the cycle-end fill must have
	// rewritten BOTH overlap layers of frame 0 vs the reference's
	// SimInit-era fill -- the inner layer is what the interior pulls (and
	// the layer the widened substep-1 kernel integrates), the outer layer
	// is the integration's streaming source; a fill reaching only one
	// layer silently starves the band
	idx refilled_inner = 0;
	idx refilled_outer = 0;
	for (std::size_t i = 0; i < subject_ghost.frame0.size(); i++) {
		if (subject_ghost.frame0[i] == reference_ghost.frame0[i])
			continue;
		if (ghostLayerDepth(subject_ghost, subject_ghost.coords[i]) == 1)
			refilled_inner++;
		else
			refilled_outer++;
	}
	CHECK_MESSAGE((
		refilled_inner > 0 && refilled_outer > 0),
		fmt::format(
			"Test 4 fill coverage: the cycle-end fill re-wrote {} inner-layer and {} outer-layer destination entries of frame 0 "
			"(both ghost layers must be filled)",
			refilled_inner,
			refilled_outer
		)
	);

#ifdef AB_PATTERN
	// LOCK 5 (the pair discriminates the simulated band from both the old
	// both-frames fill and from a no-widening regression):
	// (SB1) the widened substep-1 kernel INTEGRATED the inner overlap rows:
	// frame-1 inner-layer entries must differ from the pre-cycle snapshot
	// (under the old both-frames fill they would carry the cycle-end fill;
	// under a no-widening regression they would keep the init fill);
	// (SB2) no frame-1 fill exists and the kernel never reaches the outer
	// layer: frame-1 outer-layer entries must equal the pre-cycle snapshot
	// bitwise (under the old both-frames fill they would have moved)
	idx sb1_inner_total = 0;
	idx sb1_changed = 0;
	idx sb2_outer_total = 0;
	double sb1_max_diff = 0;
	double sb2_max_diff = 0;
	for (std::size_t i = 0; i < subject_ghost.frame1.size(); i++) {
		const double diff = std::abs(subject_ghost.frame1[i] - subject_ghost_init.frame1[i]);
		if (ghostLayerDepth(subject_ghost, subject_ghost.coords[i]) == 1) {
			sb1_inner_total++;
			sb1_max_diff = std::max(sb1_max_diff, diff);
			if (diff > 0)
				sb1_changed++;
		}
		else {
			sb2_outer_total++;
			sb2_max_diff = std::max(sb2_max_diff, diff);
		}
	}
	CHECK_MESSAGE((
		sb1_inner_total > 0 && sb1_changed > 0),
		fmt::format(
			"Test 4 simulated band (substep-1 integration): {} of {} frame-1 inner-layer destination entries were updated by the "
			"widened kernel (max |change| = {:.3e})",
			sb1_changed,
			sb1_inner_total,
			sb1_max_diff
		)
	);
	CHECK_MESSAGE((
		sb2_outer_total > 0 && sb2_max_diff == 0),
		fmt::format(
			"Test 4 single fill (no frame-1 fill): all {} frame-1 outer-layer destination entries keep the SimInit-era fill "
			"bitwise (max |diff| = {:.3e})",
			sb2_outer_total,
			sb2_max_diff
		)
	);
#endif
}

// Test 8 (T8 generative fill-freshness model, N cycles): the fine block's
// C2F destination complement is filled ONCE per cycle (the single-fill
// simulated-band schedule: the frame-0 complement is fill-owned -- no
// kernel writes frame 0's ghost rows, a model premise locked by the census
// and Test 4's coverage lock), so the schedule forces an exact generative
// law for the fill content: after every cycle the frame-0 destination rows
// carry THE fill of the coarse ring state that the coarse kernel produced
// during that cycle, i.e. of the kernel's own evolution from the initial
// (SimInit) fill's source state. The frame-1 content follows the
// simulated-band law instead: its INNER layer is the substep-1 kernel's
// output (fresh every cycle), its OUTER layer is the SimInit anchor fill
// forever (no frame-1 fill exists and the kernel never reaches the outer
// layer). Over N = 3 coupled cycles on this fixture:
//
// (M1) fresh-fill cadence, never a stale re-write: cycle k's frame-0 fill
//      differs from EVERY earlier fill's content. An aliased/wedged
//      cycle-end fill would re-emit an older fill's content bitwise; the
//      evolving sine IC guarantees a genuinely different source state each
//      cycle (the same fixture property Test 3 uses to prove kernel
//      activity), so the lock is structural and carries no numeric
//      expectation;
// (M2) simulated-band generative law on frame 1 (AB only): the inner layer
//      is fresh kernel output every cycle (differs from the anchor and
//      from the previous cycle), the outer layer is the constant SimInit
//      anchor fill at every cycle end (bitwise; under AA the two frames
//      are parity states of one array and the parity evidence is Tests
//      1/9's even_iter census).
//
// Dedupe note vs Test 4: extending the kernels-only ring comparison past
// cycle 1 would be physically WRONG -- from cycle 2 on, ring cells
// legitimately stream from the F2C-refreshed GEO_NOTHING skin (see the
// d3q27/bc.h preCollision comment), so the coupled subject's ring is
// kernel-only in WRITE REACH but not kernels-only in content. The bitwise
// kernels-only reference comparison therefore stays the 1-cycle canary in
// Test 4; this test pins the fill and band sides' generative cadence
// instead.
void test_interface_ring_freshness_model()
{
	lat_t lat = makeLattice();
	const std::string id = fmt::format("test_amr_subcycling_{}_ringmodel", pattern_name);
	StateLocal_AMR<NSE_CONFIG> state(id, MPI_COMM_WORLD, lat, "adios2.xml", /*periodic=*/TRAITS::bool3d{true, true, true}, /*max_level=*/1);
	if (! state.canCompute()) {
		CHECK_MESSAGE(false, "Test 8 setup: state.canCompute()");
		return;
	}
	createAMRBlocks(state.nse, parseAMRConfig<NSE_CONFIG>("1 4 4 4 8 8 8"));
	state.SimInit();
	if (state.nse.terminate) {
		CHECK_MESSAGE(false, "Test 8 setup: SimInit triggered the terminate flag");
		return;
	}

	// scan the destination complement once per fill epoch: index 0 = the
	// SimInit single-frame fill (the cycle-0 anchor of the cycle), index
	// k = the cycle-k cycle-end fill
	constexpr int cycles = 3;
	BLOCK* fine = state.nse.getBlocksAtLevel(1).front();
	std::vector<FineGhostScan> fills;
	fills.reserve(cycles + 1);
	fine->copyDFsToHost();
	fills.push_back(captureFineGhost(*fine));
	for (int k = 1; k <= cycles && ! state.nse.terminate; k++) {
		state.updateKernelData();
		state.SimUpdate();
		fine->copyDFsToHost();
		fills.push_back(captureFineGhost(*fine));
	}
	CHECK_MESSAGE((
		static_cast<int>(fills.size()) == cycles + 1 && ! state.nse.terminate),
		"Test 8 setup: the SimInit anchor fill plus 3 coupled cycle-end fills were scanned without termination"
	);
	if (static_cast<int>(fills.size()) != cycles + 1)
		return;

	// every scan must cover the same 3,088-cell destination complement (the
	// fill ownership does not change across cycles)
	bool census_ok = true;
	for (const auto& scan : fills)
		census_ok = census_ok && scan.frame0.size() == 27 * 3088;
	CHECK_MESSAGE(census_ok, "Test 8 census: all 4 fill scans cover exactly the 3,088-cell C2F destination complement (27 x 3,088 entries each)");

	// M1: cycle k's fill is a fresh fill of the kernel-evolved coarse state,
	// never a stale re-write of an earlier cycle's fill
	for (int k = 1; k <= cycles; k++) {
		long min_diff = -1;
		for (int m = 0; m < k; m++) {
			long diff = 0;
			for (std::size_t i = 0; i < fills[k].frame0.size(); i++)
				if (fills[k].frame0[i] != fills[m].frame0[i])
					diff++;
			if (min_diff < 0 || diff < min_diff)
				min_diff = diff;
		}
		CHECK_MESSAGE(
			min_diff > 0,
			fmt::format(
				"Test 8 fill freshness, cycle {}: the cycle-end fill differs from every earlier fill ({} differing entries vs the "
				"nearest earlier fill, of {} entries)",
				k,
				min_diff,
				fills[k].frame0.size()
			)
		);
	}

#ifdef AB_PATTERN
	// M2a: the frame-1 INNER layer is fresh substep-1 kernel output every
	// cycle -- it must differ from the SimInit anchor and from the previous
	// cycle's inner content at every cycle end
	for (int k = 1; k <= cycles; k++) {
		long inner_vs_anchor = 0;
		long inner_vs_prev = 0;
		for (std::size_t i = 0; i < fills[k].frame1.size(); i++) {
			if (ghostLayerDepth(fills[k], fills[k].coords[i]) != 1)
				continue;
			if (fills[k].frame1[i] != fills[0].frame1[i])
				inner_vs_anchor++;
			if (fills[k].frame1[i] != fills[k - 1].frame1[i])
				inner_vs_prev++;
		}
		CHECK_MESSAGE((
			inner_vs_anchor > 0 && inner_vs_prev > 0),
			fmt::format(
				"Test 8 simulated-band inner layer, cycle {}: the substep-1 kernel's inner-overlap output is fresh ({} differing "
				"entries vs the SimInit anchor, {} vs the previous cycle)",
				k,
				inner_vs_anchor,
				inner_vs_prev
			)
		);
	}

	// M2b: the frame-1 OUTER layer is the fill-only row of frame 1 -- no
	// fill and no kernel write reaches it, so it stays the SimInit anchor
	// fill bitwise at every cycle end
	for (int k = 1; k <= cycles; k++) {
		long outer_total = 0;
		double outer_max_diff = 0;
		for (std::size_t i = 0; i < fills[k].frame1.size(); i++) {
			if (ghostLayerDepth(fills[k], fills[k].coords[i]) != 2)
				continue;
			outer_total++;
			outer_max_diff = std::max(outer_max_diff, static_cast<double>(std::abs(fills[k].frame1[i] - fills[0].frame1[i])));
		}
		CHECK_MESSAGE((
			outer_total > 0 && outer_max_diff == 0),
			fmt::format(
				"Test 8 fill-only outer layer, cycle {}: all {} frame-1 outer-layer destination entries still hold the SimInit "
				"anchor fill bitwise (max |diff| = {:.3e})",
				k,
				outer_total,
				outer_max_diff
			)
		);
	}
#endif
}

// Test 9 (T8 parity-structure lock): the re-paired 10-iter seam metric
// alternates even/odd cycles (approx. -3e-05 even vs -1.6e-05 odd after the
// six-step reorder; row-8 verification, artifact cited in the commit body).
// A sim belongs to the gate artifacts, not to a unit test, so this lock pins
// the schedule structure from which that alternation FOLLOWS, derived from
// the recorded launch stream of 4 cycles (even->odd and odd->even
// transitions both covered) on the same schedule spy as Test 1:
//
// (P1) cycle-invariant fine-frame parity per call-site slot: the ghost
//      frame consumed by substep 1, the frame substep 2 sources from it,
//      and the frame the single fill targets are the SAME across cycles
//      (the absolute identities P/Q are pinned by Test 1);
// (P2) strictly alternating fill-SOURCE parity: the coarse rotation
//      recorded at every coarse-touching slot (coarse kernel, F2C, the C2F
//      fill) flips with cycle parity and repeats at same-parity cycles --
//      the global updateKernelData() toggles the coarse rotation once per
//      cycle and nothing inside SimUpdate re-toggles it (the within-cycle
//      equality across the three slots is Test 1's assertion);
// (P3) cross-cycle consumption chain: cycle k's substep 1 consumes exactly
//      the frame that cycle k-1's single fill was recorded writing, and
//      substep 2 consumes the frame substep 1 wrote -- the fill written
//      once at the cycle end is what the next cycle runs on.
//
// Derivation (the point of the lock): the seam metric samples the fine
// interior face row, whose evolution each cycle runs on the fill ghost
// frames. P1+P3 say those are always the same-paired frames written once
// at the previous cycle's end, while P2 says the fill's source frame
// alternates with cycle parity -- hence the fill imprint carried into the
// seam MUST alternate with cycle parity. The observed even/odd seam
// oscillation is the schedule's frame-parity signature, not a drift
// signal; the locks above detect any structural break of the chain
// (mid-cycle rotation re-toggle, fill-target drift, cadence loss). AA
// mapping: frame identities are the even_iter states recorded at each
// call site (the F2C slot reads the twisted post-substep-2 state).
void test_schedule_parity_structure()
{
	lat_t lat = makeLattice();
	const std::string id = fmt::format("test_amr_subcycling_{}_parity", pattern_name);
	StateSchedule_AMR<NSE_CONFIG> state(id, MPI_COMM_WORLD, lat, "adios2.xml", /*periodic=*/TRAITS::bool3d{true, true, true}, /*max_level=*/1);
	if (! state.canCompute()) {
		CHECK_MESSAGE(false, "Test 9 setup: state.canCompute()");
		return;
	}
	createAMRBlocks(state.nse, parseAMRConfig<NSE_CONFIG>("1 4 4 4 8 8 8"));
	state.SimInit();
	if (state.nse.terminate) {
		CHECK_MESSAGE(false, "Test 9 setup: SimInit triggered the terminate flag");
		return;
	}
	// consume SimInit's C2F event (the cycle-0 anchor, asserted in Test 1)
	state.events.clear();

	constexpr int cycles = 4;
	for (int k = 0; k < cycles && ! state.nse.terminate; k++) {
		state.updateKernelData();
		state.SimUpdate();
	}

	using Evt = typename StateSchedule_AMR<NSE_CONFIG>::Event;
	using Stage = typename StateSchedule_AMR<NSE_CONFIG>::Stage;
	bool premise = ! state.nse.terminate && state.events.size() == static_cast<std::size_t>(5 * cycles);
	if (premise)
		for (int k = 0; k < cycles; k++) {
			const Evt* ev = state.events.data() + 5 * k;
			premise = premise && ev[0].stage == Stage::kernel && ev[0].level == 1 && ev[1].stage == Stage::kernel && ev[1].level == 1
				   && ev[2].stage == Stage::kernel && ev[2].level == 0 && ev[3].stage == Stage::f2c && ev[3].level == 1 && ev[4].stage == Stage::c2f
				   && ev[4].level == 1;
		}
	CHECK_MESSAGE(
		premise,
		"Test 9 premise: 4 cycles recorded exactly the simulated-band stage sequence each (20 events: kernel L1 x2, kernel L0, F2C, "
		"C2F per cycle; the indexing premise for the parity derivation)"
	);
	if (! premise)
		return;

	// P1: cycle-invariant fine-frame parity at every call-site slot
	bool invariant_ok = true;
	for (int slot = 0; slot < 5; slot++)
		for (int k = 1; k < cycles; k++) {
			const Evt& ref = state.events[slot];
			const Evt& cur = state.events[5 * k + slot];
#ifdef AB_PATTERN
			invariant_ok = invariant_ok && ref.fine_cur == cur.fine_cur && ref.fine_out == cur.fine_out;
#elif defined(AA_PATTERN)
			invariant_ok = invariant_ok && ref.fine_even == cur.fine_even;
#endif
		}
	CHECK_MESSAGE(
		invariant_ok,
		"Test 9 cycle-invariant fine parity: the fine ghost frame consumed at each substep slot and targeted at each fill slot is "
		"the same frame in every one of the 4 cycles (absolute frame identities are pinned by Test 1)"
	);

	// P2: strictly alternating coarse (fill-source) parity at every
	// coarse-touching slot; repeats at same-parity cycles
	bool alternation_ok = true;
	for (int slot = 2; slot < 5; slot++)
		for (int k = 0; k + 1 < cycles; k++) {
#ifdef AB_PATTERN
			alternation_ok = alternation_ok && state.events[5 * k + slot].coarse_cur != state.events[5 * (k + 1) + slot].coarse_cur;
#elif defined(AA_PATTERN)
			alternation_ok = alternation_ok && state.events[5 * k + slot].coarse_even != state.events[5 * (k + 1) + slot].coarse_even;
#endif
		}
	for (int slot = 2; slot < 5; slot++)
		for (int k = 0; k + 2 < cycles; k++) {
#ifdef AB_PATTERN
			alternation_ok = alternation_ok && state.events[5 * k + slot].coarse_cur == state.events[5 * (k + 2) + slot].coarse_cur;
#elif defined(AA_PATTERN)
			alternation_ok = alternation_ok && state.events[5 * k + slot].coarse_even == state.events[5 * (k + 2) + slot].coarse_even;
#endif
		}
	CHECK_MESSAGE(
		alternation_ok,
		"Test 9 alternating fill-source parity: the coarse rotation at every coarse-touching slot (kernel, F2C, the C2F fill) flips "
		"with cycle parity across all 3 transitions and repeats at same-parity cycles (one toggle per cycle, none inside)"
	);

	// P3: cross-cycle consumption chain -- cycle k's substep 1 consumes
	// exactly the frame that cycle k-1's single fill was recorded writing,
	// and substep 2 consumes the frame substep 1 wrote (the simulated-band
	// chain: the fill feeds the first substep, the updated band feeds the
	// second)
	bool consumption_ok = true;
	for (int k = 1; k < cycles; k++) {
		const Evt& substep1 = state.events[5 * k];
		const Evt& substep2 = state.events[5 * k + 1];
		const Evt& fill = state.events[5 * (k - 1) + 4];
#ifdef AB_PATTERN
		consumption_ok = consumption_ok && substep1.fine_cur == fill.fine_cur && substep2.fine_cur == substep1.fine_out;
#elif defined(AA_PATTERN)
		consumption_ok = consumption_ok && substep1.fine_even == fill.fine_even && substep2.fine_even != substep1.fine_even;
#endif
	}
	CHECK_MESSAGE(
		consumption_ok,
		"Test 9 consumption chain: every cycle's substep 1 consumes exactly the frame that the previous cycle end's fill was "
		"recorded writing, and substep 2 consumes the frame substep 1 wrote -- the fill imprint carried to the seam still "
		"alternates with the source parity locked above"
	);
}

// Test 5: State_AMR::computeConservationStats must EXCLUDE the coarse cells
// hidden under the fine footprint (tagged GEO_NOTHING) from the mass,
// momentum, and per-level kinetic-energy sums - the same physical region is
// already counted on the fine level, so adding the coarse-level (frozen
// placeholder) hidden cells double-counts it. The GEO_AMR_INTERFACE ring
// cells are real coarse fluid cells and must KEEP counting (they are part
// of the reference sums below).
// [B.5 catalog note (mock-matrix.md subcycling items 1-3), updated for the
// D.1 single-configuration reality: the printed conservation values are the
// skin-era numbers (mass 4.120695e+03, KE L0 4.285964e-04, KE L1
// 2.435120e-04 bitwise) -- the ring macros are kernel-produced since the
// ring-F2C launch was removed (gate B ruling, D.1 hard-delete); the
// pre-deletion default-arm values (mass 4.120714e+03, KE L0 4.170924e-04)
// are historical. The SimInit geometry line reads "6 interface patches, 6
// interior patches" (the 8^3 footprint's skin rectangles). This test's
// assertions are metric-vs-reference internal consistency and do not pin
// the feedback path, so NOTHING was adapted here in either transition.]
//
// Production-pipeline setup identical to Test 4: one coupled Berger-Colella
// cycle populates real macros everywhere; then the hidden cells' macros are
// replaced by unmistakable sentinel values (pushed to the device, because
// the metric refreshes the host mirrors from the device) and the metric is
// re-evaluated. The metric must be INVARIANT to the sentinel injection and
// equal to a direct host-side reference sum that excludes exactly the
// GEO_NOTHING cells. A single-level (max_level == 0) sibling verifies that
// the exclusion is keyed to the tag, not to "has a finer level": with no
// GEO_NOTHING cells present, the full mass is still counted.
void test_conservation_hidden_cell_exclusion()
{
	lat_t lat = makeLattice();

	// two-level state (scope-limited: the State constructor registers a
	// global spdlog logger per instance - see Test 3)
	{
		const std::string id = fmt::format("test_amr_subcycling_{}_cons", pattern_name);
		StateLocal_AMR<NSE_CONFIG> state(id, MPI_COMM_WORLD, lat, "adios2.xml", /*periodic=*/TRAITS::bool3d{true, true, true}, /*max_level=*/1);
		if (! state.canCompute()) {
			CHECK_MESSAGE(false, "Test 5 setup: state.canCompute()");
			return;
		}

		// same centered level-1 region as Test 4: coarse footprint [4, 12)^3
		createAMRBlocks(state.nse, parseAMRConfig<NSE_CONFIG>("1 4 4 4 8 8 8"));

		state.SimInit();
		if (state.nse.terminate) {
			CHECK_MESSAGE(false, "Test 5 setup: SimInit triggered the terminate flag");
			return;
		}

		// one coupled Berger-Colella cycle populates real macros everywhere
		// on both levels (same flag as Tests 3 and 4)
		state.cnt[OUT3DCUT].period = 1e-30;
		state.updateKernelData();
		state.SimUpdate();
		if (state.nse.terminate) {
			CHECK_MESSAGE(false, "Test 5: SimUpdate triggered the terminate flag");
			return;
		}

		BLOCK* coarse = state.nse.getBlocksAtLevel(0).front();
		coarse->copyMapToHost();

		// metric BEFORE the sentinel injection (the metric refreshes the
		// block host mirrors from the device itself)
		const AMRConservationStats s0 = state.computeConservationStats();

		idx hidden = 0;
		for (idx z = 0; z < coarse->local.z(); z++)
			for (idx y = 0; y < coarse->local.y(); y++)
				for (idx x = 0; x < coarse->local.x(); x++)
					if (coarse->hmap(x, y, z) == NSE_CONFIG::BC::GEO_NOTHING)
						hidden++;
		CHECK_MESSAGE(
			hidden == 6 * 6 * 6,
			fmt::format("Test 5 setup: GEO_NOTHING footprint has exactly 216 coarse cells (skin 152 + deep 64; got {})", hidden)
		);
		if (hidden == 0)
			return;

		// unmistakable sentinel macros on the hidden cells: written to the
		// host mirror and pushed to the device (the metric's own D2H copy
		// would clobber host-side writes otherwise)
		const real sentinel_rho = 177.0f;
		const real sentinel_vx = 0.5f, sentinel_vy = -0.25f, sentinel_vz = 0.125f;
		for (idx z = 0; z < coarse->local.z(); z++)
			for (idx y = 0; y < coarse->local.y(); y++)
				for (idx x = 0; x < coarse->local.x(); x++) {
					if (coarse->hmap(x, y, z) != NSE_CONFIG::BC::GEO_NOTHING)
						continue;
					coarse->hmacro(NSE_CONFIG::MACRO::e_rho, x, y, z) = sentinel_rho;
					coarse->hmacro(NSE_CONFIG::MACRO::e_vx, x, y, z) = sentinel_vx;
					coarse->hmacro(NSE_CONFIG::MACRO::e_vy, x, y, z) = sentinel_vy;
					coarse->hmacro(NSE_CONFIG::MACRO::e_vz, x, y, z) = sentinel_vz;
				}
		coarse->copyMacroToDevice();

		// metric AFTER the sentinel injection (refreshes the host mirrors
		// again - the reference below reads the same mirrors)
		const AMRConservationStats s1 = state.computeConservationStats();
		const RefStats ref = computeReferenceStats(state);
		CHECK_MESSAGE(
			ref.hidden == 6 * 6 * 6,
			fmt::format("Test 5 reference: exactly 216 hidden cells excluded from the reference sums (got {})", ref.hidden)
		);

		// the sentinel injection must be invisible to the metric on every
		// accumulated quantity (pre-fix the shift is ~9e4 on the mass)
		const double inv_diff = std::max(
			{std::abs(s1.total_mass - s0.total_mass),
			 std::abs(s1.total_momentum_x - s0.total_momentum_x),
			 std::abs(s1.total_momentum_y - s0.total_momentum_y),
			 std::abs(s1.total_momentum_z - s0.total_momentum_z),
			 std::abs(s1.per_level_kinetic_energy.at(0) - s0.per_level_kinetic_energy.at(0)),
			 std::abs(s1.per_level_kinetic_energy.at(1) - s0.per_level_kinetic_energy.at(1))});
		CHECK_MESSAGE((
			closeRel(s1.total_mass, s0.total_mass) && closeRel(s1.total_momentum_x, s0.total_momentum_x)
				&& closeRel(s1.total_momentum_y, s0.total_momentum_y) && closeRel(s1.total_momentum_z, s0.total_momentum_z)
				&& closeRel(s1.per_level_kinetic_energy.at(0), s0.per_level_kinetic_energy.at(0))
				&& closeRel(s1.per_level_kinetic_energy.at(1), s0.per_level_kinetic_energy.at(1))),
			fmt::format("Test 5 hidden-cell exclusion: conservation stats are invariant to sentinel macros injected into GEO_NOTHING cells (max |diff| = {:.3e})", inv_diff)
		);

		// the metric must equal the reference that excludes exactly the
		// GEO_NOTHING cells (this also proves the GEO_AMR_INTERFACE ring
		// cells keep counting: they are part of the reference sums)
		CHECK_MESSAGE(
			closeRel(s1.total_mass, ref.mass),
			fmt::format("Test 5 mass: metric equals the reference sum that excludes exactly the GEO_NOTHING cells (metric = {:.6e}, ref = {:.6e})", s1.total_mass, ref.mass)
		);
		CHECK_MESSAGE((
			closeRel(s1.total_momentum_x, ref.mx) && closeRel(s1.total_momentum_y, ref.my) && closeRel(s1.total_momentum_z, ref.mz)),
			fmt::format(
				"Test 5 momentum: metric equals the reference sum that excludes exactly the GEO_NOTHING cells (metric = {:.6e}, {:.6e}, {:.6e}; ref = {:.6e}, {:.6e}, {:.6e})",
				s1.total_momentum_x, s1.total_momentum_y, s1.total_momentum_z, ref.mx, ref.my, ref.mz
			)
		);
		CHECK_MESSAGE((
			s1.per_level_kinetic_energy.size() == 2 && ref.ke.size() == 2 && closeRel(s1.per_level_kinetic_energy.at(0), ref.ke.at(0))
				&& closeRel(s1.per_level_kinetic_energy.at(1), ref.ke.at(1))),
			fmt::format("Test 5 per-level kinetic energy: metric matches the reference (L0 metric = {:.6e} ref = {:.6e}; L1 metric = {:.6e} ref = {:.6e})",
				s1.per_level_kinetic_energy.at(0),
				ref.ke.at(0),
				s1.per_level_kinetic_energy.at(1),
				ref.ke.at(1))
		);
		CHECK_MESSAGE((s0.total_mass > 0 && s1.total_mass > 0), "Test 5 sanity: total mass is nonzero with two levels");
	}

	// single-level sibling: no fine block, hence no GEO_NOTHING cells - the
	// exclusion must be keyed to the tag, not to "has a finer level", so ALL
	// of the mass is still counted
	{
		const std::string id = fmt::format("test_amr_subcycling_{}_cons0", pattern_name);
		StateLocal_AMR<NSE_CONFIG> state(id, MPI_COMM_WORLD, lat, "adios2.xml", /*periodic=*/TRAITS::bool3d{true, true, true});
		if (! state.canCompute()) {
			CHECK_MESSAGE(false, "Test 5 single-level setup: state.canCompute()");
			return;
		}

		state.SimInit();
		if (state.nse.terminate) {
			CHECK_MESSAGE(false, "Test 5 single-level setup: SimInit triggered the terminate flag");
			return;
		}

		state.cnt[OUT3DCUT].period = 1e-30;
		state.updateKernelData();
		state.SimUpdate();
		if (state.nse.terminate) {
			CHECK_MESSAGE(false, "Test 5 single-level: SimUpdate triggered the terminate flag");
			return;
		}

		const AMRConservationStats s = state.computeConservationStats();
		const RefStats ref = computeReferenceStats(state);
		CHECK_MESSAGE(ref.hidden == 0, fmt::format("Test 5 single-level: no GEO_NOTHING cells present (got {})", ref.hidden));
		CHECK_MESSAGE((
			closeRel(s.total_mass, ref.mass) && s.total_mass > 1000),
			fmt::format("Test 5 single-level: the full mass is counted when no exclusion tag exists (metric = {:.6e}, ref = {:.6e})", s.total_mass, ref.mass)
		);
		CHECK_MESSAGE(s.per_level_kinetic_energy.size() == 1, "Test 5 single-level: one per-level kinetic-energy entry");
	}
}

// Test 6 (F3 F-2 lock): the interior_patches built by buildCouplings must
// be a DISJOINT partition of the depth-1 face shell INSIDE the fine
// footprint (the F2C skin one coarse row inside the reactivated c=0 ring;
// volume = prod(max(gs - 2, 0)) - prod(max(gs - 4, 0)) coarse cells),
// for on-cube and thin-axis footprints alike. Every pushed rectangle must
// bounds-check against the footprint and be non-empty (the clip must DROP
// zero-extent degenerate rectangles - a thin axis yields neither empty nor
// duplicate rectangles). Exact integer geometry - no tolerances.
void check_skin_partition(const char* config, const idx3d& go, const idx3d& gs, int expected_rects, const char* label)
{
	lat_t lat = makeLattice();
	const std::string id = fmt::format("test_amr_subcycling_{}_skin_{}", pattern_name, label);
	StateLocal_AMR<NSE_CONFIG> state(id, MPI_COMM_WORLD, lat, "adios2.xml", /*periodic=*/TRAITS::bool3d{true, true, true}, /*max_level=*/1);
	if (! state.canCompute()) {
		CHECK_MESSAGE(false, fmt::format("Test 6 {} setup: state.canCompute()", label));
		return;
	}
	createAMRBlocks(state.nse, parseAMRConfig<NSE_CONFIG>(config));
	state.SimInit();
	if (state.nse.terminate) {
		CHECK_MESSAGE(false, fmt::format("Test 6 {} setup: SimInit triggered the terminate flag", label));
		return;
	}

	if (state.couplings.size() != 1) {
		CHECK_MESSAGE(false, fmt::format("Test 6 {} setup: exactly one inter-level coupling was built (got {})", label, state.couplings.size()));
		return;
	}
	const auto& coupling = state.couplings.front();
	const auto& rects = coupling.interior_patches;

	// (iii) every rectangle bounds-checks against the footprint and is
	// non-empty; collect cell coverage as a set of global coarse coordinates
	std::set<std::tuple<idx, idx, idx>> covered;
	std::size_t pushed_cells = 0;
	bool refs_ok = rects.size() == coupling.interior_coarse_block_ids.size() && rects.size() == coupling.interior_fine_block_ids.size();
	bool bounds_ok = true;
	for (std::size_t i = 0; i < rects.size(); i++) {
		const BLOCK* coarse = state.findBlockById(coupling.coarse_level, coupling.interior_coarse_block_ids[i]);
		const BLOCK* fine = state.findBlockById(coupling.fine_level, coupling.interior_fine_block_ids[i]);
		if (coarse == nullptr || fine == nullptr) {
			refs_ok = false;
			continue;
		}
		const idx3d begin{
			rects[i].coarse_origin.x() + coarse->offset.x(), rects[i].coarse_origin.y() + coarse->offset.y(), rects[i].coarse_origin.z() + coarse->offset.z()};
		const idx3d end{begin.x() + rects[i].coarse_size.x(), begin.y() + rects[i].coarse_size.y(), begin.z() + rects[i].coarse_size.z()};
		// non-empty and fully inside the footprint [go, go + gs)
		if (rects[i].coarse_size.x() <= 0 || rects[i].coarse_size.y() <= 0 || rects[i].coarse_size.z() <= 0 || begin.x() < go.x()
			|| begin.y() < go.y() || begin.z() < go.z() || end.x() > go.x() + gs.x() || end.y() > go.y() + gs.y() || end.z() > go.z() + gs.z())
			bounds_ok = false;
		for (idx x = begin.x(); x < end.x(); x++)
			for (idx y = begin.y(); y < end.y(); y++)
				for (idx z = begin.z(); z < end.z(); z++) {
					covered.insert({x, y, z});
					pushed_cells++;
				}
	}
	CHECK_MESSAGE((
		refs_ok && static_cast<int>(rects.size()) == expected_rects),
		fmt::format("Test 6 {}: exactly {} non-degenerate skin rectangles pushed (got {})", label, expected_rects, rects.size())
	);
	CHECK_MESSAGE(bounds_ok, fmt::format("Test 6 {}: every skin rectangle is non-empty and lies inside the footprint", label));

	// (i) pairwise disjoint: the number of pushed cells equals the set size
	// (an overlap or a duplicate rectangle would shrink the set)
	CHECK_MESSAGE(
		pushed_cells == covered.size(),
		fmt::format("Test 6 {}: skin rectangles are pairwise disjoint ({} cells pushed, {} distinct)", label, pushed_cells, covered.size())
	);

	// (ii) the union equals exactly the depth-1 face shell INSIDE the
	// footprint: the inset box [go+1, go+gs-1) minus its own inset
	// [go+2, go+gs-2) (thin-axis insets clamp empty; gs >= 3 by the
	// createAMRBlocks validation, so the +1 bound never underflows)
	std::set<std::tuple<idx, idx, idx>> shell;
	for (idx x = go.x() + 1; x < go.x() + gs.x() - 1; x++)
		for (idx y = go.y() + 1; y < go.y() + gs.y() - 1; y++)
			for (idx z = go.z() + 1; z < go.z() + gs.z() - 1; z++)
				if (x == go.x() + 1 || x == go.x() + gs.x() - 2 || y == go.y() + 1 || y == go.y() + gs.y() - 2 || z == go.z() + 1
					|| z == go.z() + gs.z() - 2)
					shell.insert({x, y, z});
	const idx3d inset1{std::max(gs.x() - 2, idx(0)), std::max(gs.y() - 2, idx(0)), std::max(gs.z() - 2, idx(0))};
	const idx3d inset2{std::max(gs.x() - 4, idx(0)), std::max(gs.y() - 4, idx(0)), std::max(gs.z() - 4, idx(0))};
	const long analytic =
		static_cast<long>(inset1.x()) * inset1.y() * inset1.z() - static_cast<long>(inset2.x()) * inset2.y() * inset2.z();
	CHECK_MESSAGE(
		shell.size() == static_cast<std::size_t>(analytic),
		fmt::format("Test 6 {} sanity: analytic depth-1 shell size {} matches the enumerated shell", label, analytic)
	);
	CHECK_MESSAGE(
		covered == shell,
		fmt::format(
			"Test 6 {}: skin-rectangle union equals exactly the depth-1 face shell of the footprint ({} of {} cells)",
			label,
			covered.size(),
			shell.size()
		)
	);
}

void test_skin_partition_geometry()
{
	check_skin_partition("1 4 4 4 8 8 8", {4, 4, 4}, {8, 8, 8}, 6, "8x8x8");
	// gs = 3 cube: the depth-1 shell is a single cell, owned by the x-min
	// face alone -- the other 5 rectangles dedupe-clamp empty (commit-6
	// runtime evidence: "1 cells pushed, 1 distinct")
	check_skin_partition("1 4 4 4 3 3 3", {4, 4, 4}, {3, 3, 3}, 1, "3x3x3");
	// thin x-axis (gs = 3, the new minimum): the 1x6x6 depth-1 slab is
	// again owned by the x-min face alone; [2,8,8] is no longer a valid
	// fixture (dual-role row) and is pinned in Test 7's rejection set
	check_skin_partition("1 4 4 4 3 8 8", {4, 4, 4}, {3, 8, 8}, 1, "3x8x8");
}

// Test 7 (F3 F-1 lock): a refinement region below the 3-coarse-cell minimum
// on ANY axis is rejected by createAMRBlocks' validation (with the interior
// inset one fine cell per face a 2-thin axis would give one cell both the
// c=0 ring and c=1 skin destination roles -- a dual-role row the band
// structure does not admit; the F2C 4-node filter window would additionally
// read out of the storable fine-DF range);
// the rejection must happen in the read-only phase (no partial block
// creation) and the minimum valid [3,...] footprint must still pass.
void test_footprint_min_size_validation()
{
	lat_t lat = makeLattice();
	const std::string id = fmt::format("test_amr_subcycling_{}_minfp", pattern_name);
	StateLocal_AMR<NSE_CONFIG> state(id, MPI_COMM_WORLD, lat, "adios2.xml", /*periodic=*/TRAITS::bool3d{true, true, true}, /*max_level=*/1);
	if (! state.canCompute()) {
		CHECK_MESSAGE(false, "Test 7 setup: state.canCompute()");
		return;
	}

	const std::size_t level0_blocks = state.nse.blocks.size();
	for (const auto& [config, axis, size] : {
			 std::tuple{"1 4 4 4 1 8 8", "X", 1},
			 std::tuple{"1 4 4 4 8 1 8", "Y", 1},
			 std::tuple{"1 4 4 4 8 8 1", "Z", 1},
			 // the retired 2-coarse-cell fixture is rejected too: with the
			 // interior inset one fine cell per face, a 2-thin axis would
			 // give one cell both the c=0 ring and the c=1 skin roles
			 std::tuple{"1 4 4 4 2 8 8", "X", 2},
		 }) {
		std::string message;
		try {
			createAMRBlocks(state.nse, parseAMRConfig<NSE_CONFIG>(config));
		}
		catch (const std::runtime_error& e) {
			message = e.what();
		}
		const std::string expected = fmt::format(
			"AMR footprint size below the 3-coarse-cell minimum required by the interface band structure (distinct c=0 ring and c=1 "
			"destination rows) on axis {} (got {})",
			axis,
			size
		);
		CHECK_MESSAGE((
			message.find(expected) != std::string::npos && state.nse.blocks.size() == level0_blocks),
			fmt::format(
				"Test 7: a {}-coarse-cell-thin footprint on axis {} is rejected in the read-only validation phase ({})",
				size,
				axis,
				message.empty() ? "no exception thrown" : fmt::format("threw: {}", message)
			)
		);

		// T8 exact-wording lock: the FULL rejection message (createAMRBlocks'
		// reject()-envelope with the region index/level/origin/size fields plus
		// the reason text) must match verbatim -- the substring check above
		// cannot see envelope edits, and the wording is user-facing (the
		// dual-role row is documented through this message)
		const std::string axis_str{axis};
		const int sx = axis_str == "X" ? size : 8;
		const int sy = axis_str == "Y" ? size : 8;
		const int sz = axis_str == "Z" ? size : 8;
		const std::string expected_full = fmt::format(
			"createAMRBlocks: invalid region #0 (level 1, origin [4,4,4], size [{},{},{}]): AMR footprint size below the 3-coarse-cell "
			"minimum required by the interface band structure (distinct c=0 ring and c=1 destination rows) on axis {} (got {})",
			sx,
			sy,
			sz,
			axis,
			size
		);
		CHECK_MESSAGE(
			message == expected_full,
			fmt::format("Test 7 wording lock: the {}-thin rejection message matches the gs>=3 minimum-footprint wording verbatim", axis)
		);
	}

	// the minimum valid footprint ([3, 8, 8] coarse cells) must be accepted
	std::string message;
	try {
		createAMRBlocks(state.nse, parseAMRConfig<NSE_CONFIG>("1 4 4 4 3 8 8"));
	}
	catch (const std::runtime_error& e) {
		message = e.what();
	}
	CHECK_MESSAGE(message.empty(), fmt::format("Test 7: the minimum [3, 8, 8] footprint is accepted ({})", message.empty() ? "no exception" : message));
	const std::vector<BLOCK*> level1 = state.nse.getBlocksAtLevel(1);
	// re-anchored indexer (ruling): local' = 2*gs - 2 per axis
	CHECK_MESSAGE((
		(level1.size() == 1 && level1.front()->local == idx3d{4, 14, 14})),
		fmt::format("Test 7: the accepted [3, 8, 8] footprint created one level-1 block of 4x14x14 fine cells (got {} blocks)", level1.size())
	);
}

// State_AMR subclass for the fine-level wall tests (Tests 10-11): the
// setupBoundaries modes drive State_AMR::buildFineWallMasks into its three
// contract lanes (partial wall, missing override, registered wall). The
// coarse plane numbers below are the face-adjacent halo rows of the
// [4,12)^3 fixture footprint (z-min: 3, x-max: 12); the fine-row tagging
// of mode 3 keys on the coarse map exactly like the channel's
// setupBoundaries and places the GEO_WALL row at local+1 with the
// GEO_NOTHING streaming buffer at local+2 (the max-face band geometry).
template <typename NSE>
struct StateWall_AMR : StateLocal_AMR<NSE>
{
	using BC = typename NSE::BC;
	using BLOCK_NSE = LBM_BLOCK<NSE>;
	using idx = typename NSE::TRAITS::idx;

	// 1 = PARTIAL z-min wall, 2 = FULL z-min wall without the overlap
	// override, 3 = FULL x-max wall (the override is set by the test)
	int wall_mode = 0;

	template <typename... ARGS>
	StateWall_AMR(ARGS&&... args)
	: StateLocal_AMR<NSE>(std::forward<ARGS>(args)...)
	{}

	void setupBoundaries() override
	{
		// domain border layers GEO_NOTHING on every edge (the ghost-layer
		// idiom of the channel sims): this fixture is non-periodic, so a
		// plain-fluid edge cell would stream from outside the coarse
		// block's zero-overlap storage (dmap bounds assert + kernel trap)
		this->nse.setBoundaryX(0, BC::GEO_NOTHING);
		this->nse.setBoundaryX(this->nse.lat.global.x() - 1, BC::GEO_NOTHING);
		this->nse.setBoundaryY(0, BC::GEO_NOTHING);
		this->nse.setBoundaryY(this->nse.lat.global.y() - 1, BC::GEO_NOTHING);
		this->nse.setBoundaryZ(0, BC::GEO_NOTHING);
		this->nse.setBoundaryZ(this->nse.lat.global.z() - 1, BC::GEO_NOTHING);
		if (wall_mode == 1) {
			// half of the z-min cross-section: coarse x in [4, 8)
			for (idx x = 4; x < 8; x++)
				for (idx y = 4; y < 12; y++)
					this->nse.setMap(x, y, 3, BC::GEO_WALL);
		}
		else if (wall_mode == 2) {
			this->nse.setBoundaryZ(3, BC::GEO_WALL);
		}
		else if (wall_mode == 3) {
			// coarse wall plane on the x=12 face-adjacent halo row, spanning
			// y,z in [1, global-1): its outermost row keeps a GEO_NOTHING
			// backer at the domain border -- a wall cell sitting directly on
			// the zero-overlap edge would still gather from outside the
			// storage (the same idiom as the border layers above)
			for (idx y = 1; y < this->nse.lat.global.y() - 1; y++)
				for (idx z = 1; z < this->nse.lat.global.z() - 1; z++)
					this->nse.setMap(12, y, z, BC::GEO_WALL);
			for (auto& fine : this->nse.blocks) {
				if (fine.level == 0)
					continue;
				for (idx y = fine.offset.y(); y < fine.offset.y() + fine.local.y(); y++)
					for (idx z = fine.offset.z(); z < fine.offset.z() + fine.local.z(); z++) {
						// wall column iff the face-adjacent coarse column is
						// GEO_WALL on the parent level (floor(fine/2) is
						// exact for the positive fine-global coordinates)
						bool wall_column = false;
						for (const auto& coarse : this->nse.blocks)
							if (coarse.level == 0 && coarse.hmap(12, y / 2, z / 2) == BC::GEO_WALL)
								wall_column = true;
						if (! wall_column)
							continue;
						fine.hmap(fine.offset.x() + fine.local.x() + 1, y, z) = BC::GEO_WALL;
						fine.hmap(fine.offset.x() + fine.local.x() + 2, y, z) = BC::GEO_NOTHING;
					}
			}
		}
	}

	void resetDFs() override
	{
		if (wall_mode == 3) {
			// full-stored-extent equilibrium on every block: the walled
			// face's ghost band receives no coarse-to-fine fill, so its
			// rows must hold a valid state from the start (the channel's
			// setInitialCondition idiom; setEquilibrium covers the overlap
			// rows); the sine IC of the sibling tests is interior-only
			for (auto& block : this->nse.blocks)
				block.setEquilibrium(1, 0, 0, 0);
			this->nse.copyDFsToHost();
		}
		else
			StateLocal_AMR<NSE>::resetDFs();
	}
};

// Test 10 (fine-wall fail-fast rails): State_AMR::buildFineWallMasks must
// REJECT a partial fine wall with a runtime_error naming block, face,
// count, and the expected count (the launch window and the C2F fill are
// face-wide decisions, a column-wise mixture has no defined contract), and
// a full wall missing the per-axis storage override with a runtime_error
// naming block, face, and the override to set (with only the 2-deep C2F
// band, the GEO_NOTHING streaming-buffer row would lie outside the
// allocated storage and the coupling patch would overwrite the wall
// columns).
void test_fine_wall_failfast()
{
	// partial z-min wall: coarse x in [4, 8) tagged on the z = 3 plane
	// backs fine columns [9, 16) of the [9, 23) interior x range, i.e. 7 x
	// 14 = 98 of the 196 z-min cross-section columns -- a partial wall
	{
		lat_t lat = makeLattice();
		const std::string id = fmt::format("test_amr_subcycling_{}_wallpartial", pattern_name);
		StateWall_AMR<NSE_CONFIG> state(id, MPI_COMM_WORLD, lat, "adios2.xml", /*periodic=*/TRAITS::bool3d{false, false, false}, /*max_level=*/1);
		if (! state.canCompute()) {
			CHECK_MESSAGE(false, "Test 10 setup (partial): state.canCompute()");
			return;
		}
		state.wall_mode = 1;
		createAMRBlocks(state.nse, parseAMRConfig<NSE_CONFIG>("1 4 4 4 8 8 8"));
		std::string message;
		try {
			state.SimInit();
		}
		catch (const std::runtime_error& e) {
			message = e.what();
		}
		CHECK_MESSAGE((
			message.find("PARTIAL fine-level wall") != std::string::npos && message.find("z-min") != std::string::npos
				&& message.find("block 1") != std::string::npos && message.find("98 of 196") != std::string::npos),
			fmt::format(
				"Test 10 partial-wall rejection: SimInit throws the named error (block, face, count 98 of 196) -- {}",
				message.empty() ? "no exception thrown" : fmt::format("threw: {}", message)
			)
		);
	}

	// full z-min wall but no storage_overlap_z = 3 override: the wall row
	// is tagged but the GEO_NOTHING streaming-buffer row is unallocated
	{
		lat_t lat = makeLattice();
		const std::string id = fmt::format("test_amr_subcycling_{}_wallnooverlap", pattern_name);
		StateWall_AMR<NSE_CONFIG> state(id, MPI_COMM_WORLD, lat, "adios2.xml", /*periodic=*/TRAITS::bool3d{false, false, false}, /*max_level=*/1);
		if (! state.canCompute()) {
			CHECK_MESSAGE(false, "Test 10 setup (missing override): state.canCompute()");
			return;
		}
		state.wall_mode = 2;
		createAMRBlocks(state.nse, parseAMRConfig<NSE_CONFIG>("1 4 4 4 8 8 8"));
		std::string message;
		try {
			state.SimInit();
		}
		catch (const std::runtime_error& e) {
			message = e.what();
		}
		CHECK_MESSAGE((
			message.find("storage_overlap_z") != std::string::npos && message.find("z-min") != std::string::npos
				&& message.find("the z-axis overlap is 2 (< 3)") != std::string::npos),
			fmt::format(
				"Test 10 missing-override rejection: SimInit throws the named error (block, face, storage_overlap_z) -- {}",
				message.empty() ? "no exception thrown" : fmt::format("threw: {}", message)
			)
		);
	}
}

// Test 11 (fine wall on a non-z face): a full x-max wall with
// storage_overlap_x = 3 must be REGISTERED by buildFineWallMasks (exactly
// the Right-face bit of fine_wall_masks), the launch windows must be
// deepened to the GEO_WALL row at local+1 on x in both substeps while
// keeping the nominal band extents on y/z, the x-max coupling patch must
// carry an EMPTY fine destination (the BC-managed band is not a C2F
// destination) while every other face keeps its full band, and a coupled
// SimUpdate over this configuration must run cleanly (the deepened launch
// processes the bounce-back row against the allocated buffer row).
void test_fine_wall_maxface()
{
	using SyncDirection = TNL::Containers::SyncDirection;

	lat_t lat = makeLattice();
	const std::string id = fmt::format("test_amr_subcycling_{}_wallxmax", pattern_name);
	StateWall_AMR<NSE_CONFIG> state(id, MPI_COMM_WORLD, lat, "adios2.xml", /*periodic=*/TRAITS::bool3d{false, false, false}, /*max_level=*/1);
	if (! state.canCompute()) {
		CHECK_MESSAGE(false, "Test 11 setup: state.canCompute()");
		return;
	}
	state.wall_mode = 3;
	// [4,12)^3 footprint: the x = 12 plane is the x-max face-adjacent halo row
	createAMRBlocks(state.nse, parseAMRConfig<NSE_CONFIG>("1 4 4 4 8 8 8"));
	BLOCK* fine = state.nse.getBlocksAtLevel(1).front();
	fine->storage_overlap_x = 3;
	state.SimInit();
	if (state.nse.terminate) {
		CHECK_MESSAGE(false, "Test 11 setup: SimInit triggered the terminate flag");
		return;
	}

	// the mask registers exactly the x-max (Right) bit of the six
	const int right_bit = State_AMR<NSE_CONFIG>::fineWallFaceBit(SyncDirection::Right);
	CHECK_MESSAGE(
		state.fineWallMask(*fine) == (1 << right_bit),
		fmt::format(
			"Test 11 wall mask: exactly the x-max (Right) bit is registered (mask = {})", static_cast<int>(state.fineWallMask(*fine))
		)
	);

	// launch window geometry: deepened to the wall row at local+1 = 15 on
	// x in both substep classes; the y/z axes keep the nominal simulated
	// band ([-1, local+1) substep 1) and interior ([0, local) substep 2)
	// extents of a 14^3 fine interior with a 3-deep x overlap
	const auto w1 = state.kernelLaunchWindow(*fine, 1);
	CHECK_MESSAGE((
		(w1.first == idx3d{-1, -1, -1} && w1.second == idx3d{17, 16, 16})),
		fmt::format(
			"Test 11 substep-1 window: [{},{},{}] + [{},{},{}] (expected [-1,-1,-1] + [17,16,16])",
			w1.first.x(),
			w1.first.y(),
			w1.first.z(),
			w1.second.x(),
			w1.second.y(),
			w1.second.z()
		)
	);
	const auto w0 = state.kernelLaunchWindow(*fine, 0);
	CHECK_MESSAGE((
		w0.first == idx3d{0, 0, 0} && w0.second == idx3d{16, 14, 14}),
		fmt::format(
			"Test 11 substep-2 window: [{},{},{}] + [{},{},{}] (expected [0,0,0] + [16,14,14])",
			w0.first.x(),
			w0.first.y(),
			w0.first.z(),
			w0.second.x(),
			w0.second.y(),
			w0.second.z()
		)
	);

	// grid selection guard: the precomputed interior grid is sized by
	// roundup(local) with zero slack, so a max-face-wall substep-2 window
	// (begin {0,0,0} but size.x = local+2) MUST recompute the grid -- a
	// begin-only fast-path check would silently skip the wall row
	CHECK_MESSAGE(
		! State_AMR<NSE_CONFIG>::isInteriorLaunchWindow(w0.first, w0.second, fine->local),
		"Test 11 grid selection: the x-max-wall substep-2 window is NOT eligible for the precomputed interior grid"
	);
	CHECK_MESSAGE(
		! State_AMR<NSE_CONFIG>::isInteriorLaunchWindow(w1.first, w1.second, fine->local),
		"Test 11 grid selection: the widened substep-1 window is NOT eligible for the precomputed interior grid"
	);
	CHECK_MESSAGE((
		State_AMR<NSE_CONFIG>::isInteriorLaunchWindow(idx3d{0, 0, 0}, fine->local, fine->local)),
		"Test 11 grid selection: the plain interior [0, local) window IS eligible for the precomputed grid"
	);

	// coupling patches: the x-max face's destination is emptied while the
	// other five faces keep their full C2F bands
	bool saw_right = false, right_empty = false, others_nonempty = true;
	int n_patches = 0;
	for (const auto& patch : state.couplings.front().patches) {
		n_patches++;
		if (patch.face == SyncDirection::Right) {
			saw_right = true;
			right_empty = patch.fine_size == idx3d{0, 14, 14};
		}
		else
			others_nonempty = others_nonempty && patch.fine_size.x() > 0 && patch.fine_size.y() > 0 && patch.fine_size.z() > 0;
	}
	CHECK_MESSAGE((
		saw_right && right_empty && others_nonempty && n_patches == 6),
		fmt::format(
			"Test 11 coupling patches: x-max destination is empty ({}), the other 5 faces keep their full bands ({})",
			right_empty,
			others_nonempty
		)
	);

	// one coupled cycle over the deepened window (launch smoke: the
	// bounce-back row is processed against the allocated buffer row)
	state.updateKernelData();
	state.SimUpdate();
	CHECK_MESSAGE((
		! state.nse.terminate && state.nse.iterations == 1),
		"Test 11: one coupled SimUpdate over the x-max fine wall runs cleanly"
	);
}

TEST_CASE("T01-T02 schedule census and time sync") { test_subcycling_schedule(); }
TEST_CASE("T09 parity structure") { test_schedule_parity_structure(); }
TEST_CASE("T03 max-level-0 fallthrough") { test_max_level_zero_fallthrough(); }
TEST_CASE("T04 interface ring freshness") { test_interface_ring_freshness(); }
TEST_CASE("T08 freshness generative model") { test_interface_ring_freshness_model(); }
TEST_CASE("T05 conservation hidden-cell exclusion") { test_conservation_hidden_cell_exclusion(); }
TEST_CASE("T06 skin partition geometry") { test_skin_partition_geometry(); }
TEST_CASE("T07 footprint min-size validation") { test_footprint_min_size_validation(); }
TEST_CASE("T10 fine wall fail-fast") { test_fine_wall_failfast(); }
TEST_CASE("T11 fine wall max-face") { test_fine_wall_maxface(); }

TEST_SUITE_END();
