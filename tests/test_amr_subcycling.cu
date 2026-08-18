// Unit tests for the Berger-Colella time-subcycling orchestration in
// State_AMR (include/lbm3d/amr_state.h).
//
// Different from test_amr_coupling.cu (which exercises the coupling kernels
// directly), these tests drive SMALL two-level State_AMR instances through
// the real SimInit/SimUpdate machinery and verify the subcycling schedule
// from host-observable per-level state:
//
// - Test 1 (substep counting per level): State_AMR calls the non-virtual
//   launch helpers internally, so kernel launches cannot be intercepted from
//   a subclass. Instead, the test counts the per-level DATA updates that
//   each kernel launch requires: for the A-B pattern every kernel-launch
//   preparation applies one DF-pointer rotation (updateKernelData for level
//   0 driven by the global iteration clock, updateKernelDataForLevel for
//   level 1 driven by the substep clock), for the A-A pattern it toggles
//   data.even_iter. After one Berger-Colella step (1 coarse + 2 fine
//   substeps) the coarse and fine rotation/parity states must therefore
//   differ by exactly one preparation step - this is the host-visible
//   encoding of "1 coarse kernel call vs 2 fine kernel calls".
// - Test 2 (time synchronization): with the 2:1 refinement the fine level
//   performs exactly 2 substeps per coarse step (proven by Test 1) and
//   lat_local.physDt is exactly physDt/2, hence the physical time advanced
//   on both levels is identical after every coarse step.
// - Test 3 (max_level == 0 fallthrough): State_AMR with max_level == 0 must
//   behave identically to the base State driver - verified against a plain
//   State sibling on the same lattice run through the same sequence
//   (bitwise-identical DFs and macroscopic quantities on the host).
//
// The streaming pattern is selected at compile time (AB_PATTERN/AA_PATTERN
// from tests/CMakeLists.txt), everything is single-rank.

#include <algorithm>
#include <cmath>
#include <set>
#include <string>
#include <tuple>

#include <fmt/core.h>

#include "lbm3d/core.h"
#include "lbm3d/amr_state.h"

using TRAITS = TraitsSP;
using COLL = D3Q27_CUM<TRAITS, D3Q27_EQ_INV_CUM<TRAITS>>;
using NSE_CONFIG = LBM_CONFIG<
	TRAITS,
	D3Q27_KernelStruct,
	NSE_Data_ConstInflow<TRAITS>,
	COLL,
	typename COLL::EQ,
	D3Q27_STREAMING<TRAITS>,
	D3Q27_BC_All,
	D3Q27_MACRO_Default<TRAITS>>;

#ifdef AA_PATTERN
constexpr const char* pattern_name = "AA";
#else
constexpr const char* pattern_name = "AB";
#endif

using idx = typename TRAITS::idx;
using idx3d = typename TRAITS::idx3d;
using real = typename TRAITS::real;
using point_t = typename TRAITS::point_t;
using lat_t = Lattice<3, real, idx>;
using BLOCK = LBM_BLOCK<NSE_CONFIG>;

int g_failures = 0;

void report(bool ok, const std::string& what)
{
	if (ok)
		fmt::println("PASS: {}", what);
	else {
		fmt::println("FAIL: {}", what);
		g_failures++;
	}
}

// 16^3 periodic box in physical units (same scaling as sim_AMR at N = 16):
// LBM viscosity nu_lb_coarse = 0.005, hence nu_lb_fine = 0.01 and
// physDt_fine = physDt_coarse / 2 exactly (binary halving)
// periodicity is declared per-dimension via the bool3d passed to the State/LBM constructors
lat_t makeLattice()
{
	const int N = 16;
	const real LBM_VISCOSITY = 0.005;
	const real PHYS_HEIGHT = 0.41;
	const real PHYS_VISCOSITY = 1.5e-5;
	const real PHYS_DL = PHYS_HEIGHT / N;
	const real PHYS_DT = LBM_VISCOSITY / PHYS_VISCOSITY * PHYS_DL * PHYS_DL;

	lat_t lat;
	lat.global = typename lat_t::CoordinatesType(N, N, N);
	lat.physOrigin = point_t{0., 0., 0.};
	lat.physDl = PHYS_DL;
	lat.physDt = PHYS_DT;
	lat.physViscosity = PHYS_VISCOSITY;
	return lat;
}

// non-uniform (but smooth and stable) initial condition so that one LBM
// kernel launch has an observable effect (a uniform equilibrium state is a
// fixed point of both collision and streaming and could not distinguish
// "kernel ran" from "no-op" in Test 3)
template <typename STATE>
void setSineInitialCondition(STATE& state)
{
	using idx3d = typename STATE::idx3d;
	using dreal = typename STATE::dreal;

	for (auto& block : state.nse.blocks) {
#ifdef HAVE_MPI
		auto local_df = block.dfs[0].getLocalView();
#else
		auto local_df = block.dfs[0].getView();
#endif
		const lat_t lat_local = (block.level == 0) ? state.nse.lat : block.lat_local;

		const idx3d begin = {0, 0, 0};
		const idx3d end = {block.local.y(), block.local.z(), block.local.x()};
		TNL::Algorithms::parallelFor<DeviceType>(
			begin,
			end,
			[local_df, lat_local] __cuda_callable__(const idx3d& yzx) mutable
			{
				const auto& [y, z, x] = yzx;
				const point_t phys = lat_local.lbm2physPoint(x, y, z);
				const dreal rho = 1 + 0.01f * TNL::sin(8.0f * phys.x());
				NSE_CONFIG::COLL::setEquilibriumLat(local_df, x, y, z, rho, 0, 0, 0);
			}
		);

		// copy the initialized DFs so that they are not overridden
		for (uint8_t dftype = 1; dftype < DFMAX; dftype++)
			block.dfs[dftype] = block.dfs[0];
	}

	state.nse.copyDFsToHost();
}

// subclass of State_AMR with periodic boundaries and the non-uniform IC
template <typename NSE>
struct StateLocal_AMR : State_AMR<NSE>
{
	using idx3d = typename NSE::TRAITS::idx3d;
	using dreal = typename NSE::TRAITS::dreal;
	using lat_t = typename State_AMR<NSE>::lat_t;

	// pass-through constructor (forwards periodic/max_level to LBM)
	template <typename... ARGS>
	StateLocal_AMR(ARGS&&... args)
	: State_AMR<NSE>(std::forward<ARGS>(args)...)
	{}

	void resetDFs() override
	{
		setSineInitialCondition(*this);
	}

	[[nodiscard]] std::vector<std::string> getOutputDataNames() const override
	{
		return {};
	}

	void outputData(UniformDataWriter<typename NSE::TRAITS>&, const LBM_BLOCK<NSE>&, const idx3d&, const idx3d&) override
	{}
};

// same IC/boundaries on the plain base State (the Test-3 sibling)
template <typename NSE>
struct StateLocal_Base : State<NSE>
{
	using idx3d = typename NSE::TRAITS::idx3d;
	using dreal = typename NSE::TRAITS::dreal;
	using lat_t = typename State<NSE>::lat_t;

	template <typename... ARGS>
	StateLocal_Base(ARGS&&... args)
	: State<NSE>(std::forward<ARGS>(args)...)
	{}

	void resetDFs() override
	{
		setSineInitialCondition(*this);
	}

	[[nodiscard]] std::vector<std::string> getOutputDataNames() const override
	{
		return {};
	}

	void outputData(UniformDataWriter<typename NSE::TRAITS>&, const LBM_BLOCK<NSE>&, const idx3d&, const idx3d&) override
	{}
};

// For the A-B pattern each kernel-launch preparation rotates data.dfs by one
// position: `data.dfs[0] == dfs[0].getData()` is the identity (substep 0)
// state, `data.dfs[0] == dfs[1].getData()` is the swapped (substep 1) state.
bool dfsSwapped(const BLOCK& block)
{
	return block.data.dfs[0] == block.dfs[1].getData();
}

// count of CUDA kernels actually launched is inferred from the data
// preparation state; both levels' states are dumped for the test log
template <typename STATE>
std::string levelStateString(const STATE& state, const BLOCK& block, int level)
{
	std::string s = fmt::format("level {}:", level);
#ifdef AB_PATTERN
	s += fmt::format(" dfs rotation = {}", dfsSwapped(block) ? "substep-1 (swapped)" : "substep-0 (identity)");
#elif defined(AA_PATTERN)
	s += fmt::format(" even_iter = {}", block.data.even_iter);
#endif
	s += fmt::format(", lbmViscosity = {:.6e}", static_cast<double>(block.data.lbmViscosity));
	return s;
}

// Tests 1 and 2: Berger-Colella subcycling count and time synchronization
void test_subcycling_schedule()
{
	lat_t lat = makeLattice();
	const std::string id = fmt::format("test_amr_subcycling_{}_sched", pattern_name);
	StateLocal_AMR<NSE_CONFIG> state(id, MPI_COMM_WORLD, lat, "adios2.xml", /*periodic=*/TRAITS::bool3d{true, true, true}, /*max_level=*/1);
	if (! state.canCompute()) {
		report(false, "Test 1/2 setup: state.canCompute()");
		return;
	}

	// one centered level-1 region: coarse footprint [4, 12)^3, i.e. a 16^3
	// fine block at fine offset (8, 8, 8)
	createAMRBlocks(state.nse, parseAMRConfig<NSE_CONFIG>("1 4 4 4 8 8 8"));

	const std::vector<BLOCK*> level0 = state.nse.getBlocksAtLevel(0);
	const std::vector<BLOCK*> level1 = state.nse.getBlocksAtLevel(1);
	report(level0.size() == 1 && level1.size() == 1, "Test 1 setup: one level-0 block and one level-1 block created");
	if (level0.empty() || level1.empty())
		return;
	BLOCK* coarse = level0.front();
	BLOCK* fine = level1.front();

	// full initialization (allocation, boundary setup, interface tagging,
	// coupling patches, initial condition)
	state.SimInit();
	if (state.nse.terminate) {
		report(false, "Test 1/2 setup: SimInit triggered the terminate flag");
		return;
	}

	report(! state.couplings.empty() && ! state.couplings.front().patches.empty(), "Test 1 setup: interface coupling patches were built in SimInit");

	const double nu_lb_coarse = state.nse.lat.lbmViscosity();
	const double nu_lb_fine = fine->lat_local.lbmViscosity();
	report(
		std::abs(nu_lb_fine - 2 * nu_lb_coarse) <= 1e-12 * nu_lb_coarse,
		fmt::format("Test 1 setup: fine lattice viscosity is doubled (nu_fine = {:.6e}, nu_coarse = {:.6e})", nu_lb_fine, nu_lb_coarse)
	);

	// execute()-style iteration: updateKernelData before each SimUpdate
	bool counts_ok = true;
	bool visc_ok = true;
	bool sync_ok = true;
	for (int call = 1; call <= 3 && ! state.nse.terminate; call++) {
		state.updateKernelData();
		state.SimUpdate();

		// one call of SimUpdate = exactly one coarse iteration
		const bool iter_ok = (state.nse.iterations == call);

		// launch-preparation state per level (Tests 1's kernel-call counts):
		// level 0 was prepared exactly ONCE for this call (the pre-SimUpdate
		// updateKernelData driven by the global clock), level 1 was prepared
		// TWICE inside SimUpdate (substeps 0 and 1 driven by the substep
		// clock, the final state being the substep-1 preparation)
#ifdef AB_PATTERN
		const bool coarse_at_substep0 = ! dfsSwapped(*coarse);
		const bool expected_coarse_substep0 = ((call - 1) % 2 == 0);
		const bool level_ok = (coarse_at_substep0 == expected_coarse_substep0) && dfsSwapped(*fine);
#elif defined(AA_PATTERN)
		const bool expected_coarse_even = ((call - 1) % 2 == 1);
		const bool level_ok = (coarse->data.even_iter == expected_coarse_even) && fine->data.even_iter;
#endif
		if (! (iter_ok && level_ok)) {
			counts_ok = false;
			report(
				false,
				fmt::format(
					"Test 1 substep counting after call {}: iterations = {} -- {} | {}",
					call, state.nse.iterations, levelStateString(state, *coarse, 0), levelStateString(state, *fine, 1)
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
			report(false, fmt::format("Test 1 per-level viscosity after call {}: level0 = {:.6e}, level1 = {:.6e}", call, nu_l0, nu_l1));
		}

		// Test 2: time synchronization -- the fine level performs exactly 2
		// substeps per coarse step (the data-state evidence above) and its
		// time step is exactly physDt/2, so both level clocks agree after
		// every Berger-Colella step
		const double t_coarse = state.nse.iterations * static_cast<double>(state.nse.lat.physDt);
		const long fine_substeps = 2L * state.nse.iterations;
		const double t_fine = fine_substeps * static_cast<double>(fine->lat_local.physDt);
		const bool dt_ok = (2.0 * static_cast<double>(fine->lat_local.physDt) == static_cast<double>(state.nse.lat.physDt))
						&& (2.0 * static_cast<double>(fine->lat_local.physDl) == static_cast<double>(state.nse.lat.physDl));
		if (! (dt_ok && t_coarse == t_fine)) {
			sync_ok = false;
			report(false, fmt::format("Test 2 time sync after call {}: t_coarse = {:.17e}, t_fine = {:.17e}", call, t_coarse, t_fine));
		}
	}

	report(state.nse.iterations == 3 && counts_ok && visc_ok, "Test 1 substep counting: 3 coarse iterations, each driving 1 coarse + 2 fine kernel-launch preparations (with per-level viscosities restored)");
	report(sync_ok, "Test 2 time synchronization: fine clock (2 substeps of dt/2) equals coarse clock (1 step of dt) after every Berger-Colella step");
	report(! state.nse.terminate, "Test 1/2: no termination or kernel failure during 3 subcycled iterations");
}

// plain host-side snapshot of a block's DF and macro arrays in a fixed
// element order (for the sequential Test-3 sibling comparison; the State
// constructor registers a global spdlog logger per instance, so two States
// cannot coexist in one process)
struct HostSnapshot
{
	idx3d local{0, 0, 0};
	std::vector<double> dfs;
	std::vector<double> macro;
};

HostSnapshot snapshotBlock(const BLOCK& block)
{
	HostSnapshot snap;
	snap.local = block.local;
	for (uint8_t dfty = 0; dfty < DFMAX; dfty++)
		for (int q = 0; q < NSE_CONFIG::Q; q++)
			for (idx z = 0; z < block.local.z(); z++)
				for (idx y = 0; y < block.local.y(); y++)
					for (idx x = 0; x < block.local.x(); x++)
						snap.dfs.push_back(block.hfs[dfty](q, x, y, z));
	for (int m = 0; m < NSE_CONFIG::MACRO::N; m++)
		for (idx z = 0; z < block.local.z(); z++)
			for (idx y = 0; y < block.local.y(); y++)
				for (idx x = 0; x < block.local.x(); x++)
					snap.macro.push_back(block.hmacro(m, x, y, z));
	return snap;
}

// maximum absolute difference of the block's host arrays against a snapshot
// taken in the same element order (bitwise comparison, exact == 0 expected)
double maxAbsDiffSnapshot(const BLOCK& block, const HostSnapshot& snap)
{
	double max_diff = 0;
	std::size_t i = 0;
	for (uint8_t dfty = 0; dfty < DFMAX; dfty++)
		for (int q = 0; q < NSE_CONFIG::Q; q++)
			for (idx z = 0; z < block.local.z(); z++)
				for (idx y = 0; y < block.local.y(); y++)
					for (idx x = 0; x < block.local.x(); x++, i++)
						max_diff = std::max(max_diff, std::abs(snap.dfs[i] - static_cast<double>(block.hfs[dfty](q, x, y, z))));
	i = 0;
	for (int m = 0; m < NSE_CONFIG::MACRO::N; m++)
		for (idx z = 0; z < block.local.z(); z++)
			for (idx y = 0; y < block.local.y(); y++)
				for (idx x = 0; x < block.local.x(); x++, i++)
					max_diff = std::max(max_diff, std::abs(snap.macro[i] - static_cast<double>(block.hmacro(m, x, y, z))));
	return max_diff;
}

using dreal = typename TRAITS::dreal;

// snapshot-vs-snapshot variant of maxAbsDiffSnapshot (the subject's init is
// gone by the time the reference state exists -- the State constructor
// registers a global spdlog logger per instance, so the two states are
// compared through their snapshots)
double maxAbsDiffSnapshots(const HostSnapshot& a, const HostSnapshot& b)
{
	double max_diff = 0;
	for (std::size_t i = 0; i < a.dfs.size(); i++)
		max_diff = std::max(max_diff, std::abs(a.dfs[i] - b.dfs[i]));
	for (std::size_t i = 0; i < a.macro.size(); i++)
		max_diff = std::max(max_diff, std::abs(a.macro[i] - b.macro[i]));
	return max_diff;
}

// scan-order capture of a coarse block's (map tag, macro quad) per local
// cell -- the cross-state comparison carrier of Test 4's kernels-only
// reference lock (B.5)
struct CoarseMacroScan
{
	std::vector<int> map;
	std::vector<double> vals;  // 4 entries per cell (rho, vx, vy, vz), scan order
};

CoarseMacroScan captureCoarseMacros(const BLOCK& block)
{
	CoarseMacroScan scan;
	for (idx z = 0; z < block.local.z(); z++)
		for (idx y = 0; y < block.local.y(); y++)
			for (idx x = 0; x < block.local.x(); x++) {
				scan.map.push_back(block.hmap(x, y, z));
				scan.vals.push_back(block.hmacro(NSE_CONFIG::MACRO::e_rho, x, y, z));
				scan.vals.push_back(block.hmacro(NSE_CONFIG::MACRO::e_vx, x, y, z));
				scan.vals.push_back(block.hmacro(NSE_CONFIG::MACRO::e_vy, x, y, z));
				scan.vals.push_back(block.hmacro(NSE_CONFIG::MACRO::e_vz, x, y, z));
			}
	return scan;
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
			report(false, "Test 3 setup: state.canCompute()");
			return;
		}
		report(state_amr.nse.max_level == 0, "Test 3 setup: State_AMR constructed with max_level == 0 (no extra LBM args)");

		state_amr.SimInit();
		if (state_amr.nse.terminate) {
			report(false, "Test 3 setup: SimInit triggered the terminate flag");
			return;
		}
		snap_amr_init = snapshotBlock(state_amr.nse.blocks.front());

		// force macroscopic output inside the kernel so that the step's
		// effect is visible in dmacro (same decision in both drivers)
		state_amr.cnt[OUT3DCUT].period = 1e-30;

		// run exactly one iteration of the execute() loop body
		state_amr.updateKernelData();
		state_amr.SimUpdate();

		report(state_amr.nse.iterations == 1 && ! state_amr.nse.terminate, "Test 3: one SimUpdate completed (iterations == 1, no termination)");
		report(
			state_amr.nse.blocks.size() == 1 && state_amr.nse.blocks.front().level == 0,
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
			report(false, "Test 3 sibling setup: state.canCompute()");
			return;
		}

		state_base.SimInit();
		if (state_base.nse.terminate) {
			report(false, "Test 3 sibling setup: SimInit triggered the terminate flag");
			return;
		}

		// identical initialization (bitwise)
		const double init_diff = maxAbsDiffSnapshot(state_base.nse.blocks.front(), snap_amr_init);
		report(init_diff == 0, fmt::format("Test 3 initialization bitwise identical to base (max |diff| = {:.3e})", init_diff));

		state_base.cnt[OUT3DCUT].period = 1e-30;
		state_base.updateKernelData();
		state_base.SimUpdate();

		for (auto& block : state_base.nse.blocks) {
			block.copyDFsToHost();
			block.copyMacroToHost();
		}
		const double step_diff = maxAbsDiffSnapshot(state_base.nse.blocks.front(), snap_amr_step);
		report(step_diff == 0, fmt::format("Test 3 one-step result bitwise identical to the base driver (max |diff| = {:.3e})", step_diff));

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
		report(max_rho_change > 0, fmt::format("Test 3 sanity: the step advanced the state (max |rho change| = {:.3e})", max_rho_change));
	}
}

// Test 4 (B.5 lock, sole variant since D.1): after one Berger-Colella
// cycle every ring cell's macros must equal the KERNEL-PRODUCED values.
// Ring cells are collision-active, and no coupling channel writes them
// (the skin F2C rectangles lie inside the footprint = GEO_NOTHING region,
// disjoint from the ring; C2F writes fine ghost cells only), so the kernel
// is their only writer. Detection: compare against a DETERMINISTIC
// kernels-only reference -- the same initialization advanced by the SAME
// SimUpdate with ALL coupling launches disabled (couplings.clear() after
// SimInit; every transfer helper is then a silent no-op with empty
// couplings, while the parity, counter, and kernel flow of SimUpdate stay
// identical). Bitwise equality is the fp-tightest form of "fp-level": in
// both runs the only writer of ring cells is the coarse kernel on
// identical input, so the expected maximum difference is exactly 0
// (assert == 0 and print the max). NOTE: the reference's FOOTPRINT cells
// intentionally diverge (frozen init state, never skin-written), so every
// cross-state comparison below is map-selected on GEO_AMR_INTERFACE.
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

	// subject: one full coupled Berger-Colella cycle
	CoarseMacroScan subject;
	HostSnapshot subject_init;
	{
		const std::string id = fmt::format("test_amr_subcycling_{}_ring", pattern_name);
		StateLocal_AMR<NSE_CONFIG> state(id, MPI_COMM_WORLD, lat, "adios2.xml", /*periodic=*/TRAITS::bool3d{true, true, true}, /*max_level=*/1);
		if (! state.canCompute()) {
			report(false, "Test 4 setup: state.canCompute()");
			return;
		}

		// one centered level-1 region with the coarse footprint [4, 12)^3: the
		// GEO_AMR_INTERFACE ring is the 10^3 - 8^3 = 488 shell cells around it
		createAMRBlocks(state.nse, parseAMRConfig<NSE_CONFIG>("1 4 4 4 8 8 8"));

		state.SimInit();
		if (state.nse.terminate) {
			report(false, "Test 4 setup: SimInit triggered the terminate flag");
			return;
		}

		BLOCK* coarse = state.nse.getBlocksAtLevel(0).front();
		subject_init = snapshotBlock(*coarse);

		// force macroscopic output inside the kernel (same flag as the
		// default variant) so the kernel-produced ring macros land in dmacro
		state.cnt[OUT3DCUT].period = 1e-30;

		// one coupled Berger-Colella cycle (1 coarse step + 2 fine substeps;
		// ring F2C removed in D.1, the skin F2C transfer runs unconditionally
		// as the only fine-to-coarse channel)
		state.updateKernelData();
		state.SimUpdate();
		if (state.nse.terminate) {
			report(false, "Test 4: SimUpdate triggered the terminate flag");
			return;
		}

		coarse->copyMapToHost();
		coarse->copyMacroToHost();
		subject = captureCoarseMacros(*coarse);
	}

	// deterministic kernels-only reference (the plan's construction):
	// same initialization, advanced by the SAME SimUpdate with ALL coupling
	// launches disabled
	CoarseMacroScan reference;
	{
		const std::string id = fmt::format("test_amr_subcycling_{}_ringref", pattern_name);
		StateLocal_AMR<NSE_CONFIG> state(id, MPI_COMM_WORLD, lat, "adios2.xml", /*periodic=*/TRAITS::bool3d{true, true, true}, /*max_level=*/1);
		if (! state.canCompute()) {
			report(false, "Test 4 setup: reference state.canCompute()");
			return;
		}
		createAMRBlocks(state.nse, parseAMRConfig<NSE_CONFIG>("1 4 4 4 8 8 8"));
		state.SimInit();
		if (state.nse.terminate) {
			report(false, "Test 4 setup: reference SimInit triggered the terminate flag");
			return;
		}

		// deterministic-initialization anchor (Test 3 proves the init is
		// reproducible bitwise across instances)
		const HostSnapshot reference_init = snapshotBlock(*state.nse.getBlocksAtLevel(0).front());
		const double init_diff = maxAbsDiffSnapshots(subject_init, reference_init);
		report(
			init_diff == 0,
			fmt::format("Test 4 reference init: kernels-only reference is bitwise identical to the subject after SimInit (max |diff| = {:.3e})", init_diff)
		);

		// disable ALL coupling launches (C2F, ring F2C, interior/skin F2C):
		// clear() after SimInit leaves the SimUpdate parity/counter/kernel
		// flow untouched and makes every transfer launch a no-op
		state.couplings.clear();
		state.cnt[OUT3DCUT].period = 1e-30;
		state.updateKernelData();
		state.SimUpdate();
		if (state.nse.terminate) {
			report(false, "Test 4: reference SimUpdate triggered the terminate flag");
			return;
		}

		BLOCK* coarse = state.nse.getBlocksAtLevel(0).front();
		coarse->copyMapToHost();
		coarse->copyMacroToHost();
		reference = captureCoarseMacros(*coarse);
	}

	// assertions
	// the ring layout is fixed by markAMRInterface: 488 shell cells around
	// the 8^3 footprint (same geometry in all three scans)
	idx ring_cells = 0;
	for (const int tag : subject.map)
		if (tag == NSE_CONFIG::BC::GEO_AMR_INTERFACE)
			ring_cells++;
	report(
		ring_cells == 10 * 10 * 10 - 8 * 8 * 8,
		fmt::format("Test 4 setup: GEO_AMR_INTERFACE shell has 488 cells around the 8^3 footprint (got {})", ring_cells)
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
	report(
		max_ring_diff == 0,
		fmt::format(
			"Test 4 ring macros kernel-produced (ring F2C removed, D.1): all {} GEO_AMR_INTERFACE cells' macros are bitwise identical to the kernels-only reference (max |diff| = {:.3e})",
			ring_cells,
			max_ring_diff
		)
	);
	report(
		placeholder_cells == 0,
		fmt::format(
			"Test 4 freshness: all {} GEO_AMR_INTERFACE cells hold kernel-produced macros ({} still hold the rho=1, v=0 placeholder)",
			ring_cells,
			placeholder_cells
		)
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
struct RefStats
{
	double mass = 0, mx = 0, my = 0, mz = 0;
	std::vector<double> ke;
	long hidden = 0;
};

// direct host-side reference of the intended metric semantics: sum rho,
// rho*u and per-level 0.5*rho*|u|^2 over every local lattice site EXCEPT
// cells tagged GEO_NOTHING, with the same per-level volume weighting (1/8^L)
template <typename STATE>
RefStats computeReferenceStats(const STATE& state)
{
	RefStats ref;
	ref.ke.assign(state.nse.max_level + 1, 0.0);
	for (const auto& block : state.nse.blocks) {
		const double volume_factor = std::pow(0.5, 3.0 * block.level);
		for (idx x = block.offset.x(); x < block.offset.x() + block.local.x(); x++)
			for (idx z = block.offset.z(); z < block.offset.z() + block.local.z(); z++)
				for (idx y = block.offset.y(); y < block.offset.y() + block.local.y(); y++) {
					if (block.hmap(x, y, z) == NSE_CONFIG::BC::GEO_NOTHING) {
						ref.hidden++;
						continue;
					}
					const double rho = block.hmacro(NSE_CONFIG::MACRO::e_rho, x, y, z);
					const double vx = block.hmacro(NSE_CONFIG::MACRO::e_vx, x, y, z);
					const double vy = block.hmacro(NSE_CONFIG::MACRO::e_vy, x, y, z);
					const double vz = block.hmacro(NSE_CONFIG::MACRO::e_vz, x, y, z);
					ref.mass += rho * volume_factor;
					ref.mx += rho * vx * volume_factor;
					ref.my += rho * vy * volume_factor;
					ref.mz += rho * vz * volume_factor;
					ref.ke[block.level] += 0.5 * rho * (vx * vx + vy * vy + vz * vz);
				}
	}
	return ref;
}

// relative-or-absolute closeness: the metric accumulates via OpenMP atomics
// (summation order varies between calls), so exact equality is not expected;
// the double-count signal (512 hidden cells with sentinel rho) exceeds 9e4
// and dwarfs both this tolerance and any reassociation noise
bool closeRel(double a, double b)
{
	return std::abs(a - b) <= 1e-6 * std::max({1.0, std::abs(a), std::abs(b)});
}

void test_conservation_hidden_cell_exclusion()
{
	lat_t lat = makeLattice();

	// two-level state (scope-limited: the State constructor registers a
	// global spdlog logger per instance - see Test 3)
	{
		const std::string id = fmt::format("test_amr_subcycling_{}_cons", pattern_name);
		StateLocal_AMR<NSE_CONFIG> state(id, MPI_COMM_WORLD, lat, "adios2.xml", /*periodic=*/TRAITS::bool3d{true, true, true}, /*max_level=*/1);
		if (! state.canCompute()) {
			report(false, "Test 5 setup: state.canCompute()");
			return;
		}

		// same centered level-1 region as Test 4: coarse footprint [4, 12)^3
		createAMRBlocks(state.nse, parseAMRConfig<NSE_CONFIG>("1 4 4 4 8 8 8"));

		state.SimInit();
		if (state.nse.terminate) {
			report(false, "Test 5 setup: SimInit triggered the terminate flag");
			return;
		}

		// one coupled Berger-Colella cycle populates real macros everywhere
		// on both levels (same flag as Tests 3 and 4)
		state.cnt[OUT3DCUT].period = 1e-30;
		state.updateKernelData();
		state.SimUpdate();
		if (state.nse.terminate) {
			report(false, "Test 5: SimUpdate triggered the terminate flag");
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
		report(hidden == 8 * 8 * 8, fmt::format("Test 5 setup: GEO_NOTHING footprint has exactly 512 coarse cells (got {})", hidden));
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
		report(ref.hidden == 8 * 8 * 8, fmt::format("Test 5 reference: exactly 512 hidden cells excluded from the reference sums (got {})", ref.hidden));

		// the sentinel injection must be invisible to the metric on every
		// accumulated quantity (pre-fix the shift is ~9e4 on the mass)
		const double inv_diff = std::max(
			{std::abs(s1.total_mass - s0.total_mass),
			 std::abs(s1.total_momentum_x - s0.total_momentum_x),
			 std::abs(s1.total_momentum_y - s0.total_momentum_y),
			 std::abs(s1.total_momentum_z - s0.total_momentum_z),
			 std::abs(s1.per_level_kinetic_energy.at(0) - s0.per_level_kinetic_energy.at(0)),
			 std::abs(s1.per_level_kinetic_energy.at(1) - s0.per_level_kinetic_energy.at(1))});
		report(
			closeRel(s1.total_mass, s0.total_mass) && closeRel(s1.total_momentum_x, s0.total_momentum_x)
				&& closeRel(s1.total_momentum_y, s0.total_momentum_y) && closeRel(s1.total_momentum_z, s0.total_momentum_z)
				&& closeRel(s1.per_level_kinetic_energy.at(0), s0.per_level_kinetic_energy.at(0))
				&& closeRel(s1.per_level_kinetic_energy.at(1), s0.per_level_kinetic_energy.at(1)),
			fmt::format("Test 5 hidden-cell exclusion: conservation stats are invariant to sentinel macros injected into GEO_NOTHING cells (max |diff| = {:.3e})", inv_diff)
		);

		// the metric must equal the reference that excludes exactly the
		// GEO_NOTHING cells (this also proves the GEO_AMR_INTERFACE ring
		// cells keep counting: they are part of the reference sums)
		report(
			closeRel(s1.total_mass, ref.mass),
			fmt::format("Test 5 mass: metric equals the reference sum that excludes exactly the GEO_NOTHING cells (metric = {:.6e}, ref = {:.6e})", s1.total_mass, ref.mass)
		);
		report(
			closeRel(s1.total_momentum_x, ref.mx) && closeRel(s1.total_momentum_y, ref.my) && closeRel(s1.total_momentum_z, ref.mz),
			fmt::format(
				"Test 5 momentum: metric equals the reference sum that excludes exactly the GEO_NOTHING cells (metric = {:.6e}, {:.6e}, {:.6e}; ref = {:.6e}, {:.6e}, {:.6e})",
				s1.total_momentum_x, s1.total_momentum_y, s1.total_momentum_z, ref.mx, ref.my, ref.mz
			)
		);
		report(
			s1.per_level_kinetic_energy.size() == 2 && ref.ke.size() == 2 && closeRel(s1.per_level_kinetic_energy.at(0), ref.ke.at(0))
				&& closeRel(s1.per_level_kinetic_energy.at(1), ref.ke.at(1)),
			fmt::format("Test 5 per-level kinetic energy: metric matches the reference (L0 metric = {:.6e} ref = {:.6e}; L1 metric = {:.6e} ref = {:.6e})",
				s1.per_level_kinetic_energy.at(0),
				ref.ke.at(0),
				s1.per_level_kinetic_energy.at(1),
				ref.ke.at(1))
		);
		report(s0.total_mass > 0 && s1.total_mass > 0, "Test 5 sanity: total mass is nonzero with two levels");
	}

	// single-level sibling: no fine block, hence no GEO_NOTHING cells - the
	// exclusion must be keyed to the tag, not to "has a finer level", so ALL
	// of the mass is still counted
	{
		const std::string id = fmt::format("test_amr_subcycling_{}_cons0", pattern_name);
		StateLocal_AMR<NSE_CONFIG> state(id, MPI_COMM_WORLD, lat, "adios2.xml", /*periodic=*/TRAITS::bool3d{true, true, true});
		if (! state.canCompute()) {
			report(false, "Test 5 single-level setup: state.canCompute()");
			return;
		}

		state.SimInit();
		if (state.nse.terminate) {
			report(false, "Test 5 single-level setup: SimInit triggered the terminate flag");
			return;
		}

		state.cnt[OUT3DCUT].period = 1e-30;
		state.updateKernelData();
		state.SimUpdate();
		if (state.nse.terminate) {
			report(false, "Test 5 single-level: SimUpdate triggered the terminate flag");
			return;
		}

		const AMRConservationStats s = state.computeConservationStats();
		const RefStats ref = computeReferenceStats(state);
		report(ref.hidden == 0, fmt::format("Test 5 single-level: no GEO_NOTHING cells present (got {})", ref.hidden));
		report(
			closeRel(s.total_mass, ref.mass) && s.total_mass > 1000,
			fmt::format("Test 5 single-level: the full mass is counted when no exclusion tag exists (metric = {:.6e}, ref = {:.6e})", s.total_mass, ref.mass)
		);
		report(s.per_level_kinetic_energy.size() == 1, "Test 5 single-level: one per-level kinetic-energy entry");
	}
}

// Test 6 (F3 F-2 lock): the interior_patches built by buildCouplings must
// be a DISJOINT partition of the 1-coarse-cell-deep face shell INSIDE the
// fine footprint (volume = prod(gs) - prod(max(gs - 2, 0)) coarse cells),
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
		report(false, fmt::format("Test 6 {} setup: state.canCompute()", label));
		return;
	}
	createAMRBlocks(state.nse, parseAMRConfig<NSE_CONFIG>(config));
	state.SimInit();
	if (state.nse.terminate) {
		report(false, fmt::format("Test 6 {} setup: SimInit triggered the terminate flag", label));
		return;
	}

	if (state.couplings.size() != 1) {
		report(false, fmt::format("Test 6 {} setup: exactly one inter-level coupling was built (got {})", label, state.couplings.size()));
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
	report(
		refs_ok && static_cast<int>(rects.size()) == expected_rects,
		fmt::format("Test 6 {}: exactly {} non-degenerate skin rectangles pushed (got {})", label, expected_rects, rects.size())
	);
	report(bounds_ok, fmt::format("Test 6 {}: every skin rectangle is non-empty and lies inside the footprint", label));

	// (i) pairwise disjoint: the number of pushed cells equals the set size
	// (an overlap or a duplicate rectangle would shrink the set)
	report(
		pushed_cells == covered.size(),
		fmt::format("Test 6 {}: skin rectangles are pairwise disjoint ({} cells pushed, {} distinct)", label, pushed_cells, covered.size())
	);

	// (ii) the union equals exactly the 1-cell-deep face shell INSIDE the
	// footprint: footprint volume minus the (gs - 2) interior volume
	std::set<std::tuple<idx, idx, idx>> shell;
	for (idx x = go.x(); x < go.x() + gs.x(); x++)
		for (idx y = go.y(); y < go.y() + gs.y(); y++)
			for (idx z = go.z(); z < go.z() + gs.z(); z++)
				if (x == go.x() || x == go.x() + gs.x() - 1 || y == go.y() || y == go.y() + gs.y() - 1 || z == go.z() || z == go.z() + gs.z() - 1)
					shell.insert({x, y, z});
	const idx3d inner{std::max(gs.x() - 2, idx(0)), std::max(gs.y() - 2, idx(0)), std::max(gs.z() - 2, idx(0))};
	const long analytic = static_cast<long>(gs.x()) * gs.y() * gs.z() - static_cast<long>(inner.x()) * inner.y() * inner.z();
	report(shell.size() == static_cast<std::size_t>(analytic), fmt::format("Test 6 {} sanity: analytic shell size {} matches the enumerated shell", label, analytic));
	report(
		covered == shell,
		fmt::format(
			"Test 6 {}: skin-rectangle union equals exactly the 1-cell-deep face shell of the footprint ({} of {} cells)",
			label,
			covered.size(),
			shell.size()
		)
	);
}

void test_skin_partition_geometry()
{
	check_skin_partition("1 4 4 4 8 8 8", {4, 4, 4}, {8, 8, 8}, 6, "8x8x8");
	check_skin_partition("1 4 4 4 3 3 3", {4, 4, 4}, {3, 3, 3}, 6, "3x3x3");
	// thin x-axis (2 coarse cells = the F2C-window minimum): only the two
	// x-normal faces survive the clip (the y/z interior ranges are empty)
	check_skin_partition("1 4 4 4 2 8 8", {4, 4, 4}, {2, 8, 8}, 2, "2x8x8");
}

// Test 7 (F3 F-1 lock): a refinement region below the 2-coarse-cell minimum
// on ANY axis is rejected by createAMRBlocks' validation (the F2C 4-node
// filter window would otherwise read out of the storable fine-DF range);
// the rejection must happen in the read-only phase (no partial block
// creation) and the minimum valid [2,...] footprint must still pass.
void test_footprint_min_size_validation()
{
	lat_t lat = makeLattice();
	const std::string id = fmt::format("test_amr_subcycling_{}_minfp", pattern_name);
	StateLocal_AMR<NSE_CONFIG> state(id, MPI_COMM_WORLD, lat, "adios2.xml", /*periodic=*/TRAITS::bool3d{true, true, true}, /*max_level=*/1);
	if (! state.canCompute()) {
		report(false, "Test 7 setup: state.canCompute()");
		return;
	}

	const std::size_t level0_blocks = state.nse.blocks.size();
	for (const auto& [config, axis] : {
			 std::pair{"1 4 4 4 1 8 8", "X"},
			 std::pair{"1 4 4 4 8 1 8", "Y"},
			 std::pair{"1 4 4 4 8 8 1", "Z"},
		 }) {
		std::string message;
		try {
			createAMRBlocks(state.nse, parseAMRConfig<NSE_CONFIG>(config));
		}
		catch (const std::runtime_error& e) {
			message = e.what();
		}
		const std::string expected = fmt::format(
			"AMR footprint size below the 2-coarse-cell minimum required by the F2C 4-node filter window on axis {} (got 1)", axis
		);
		report(
			message.find(expected) != std::string::npos && state.nse.blocks.size() == level0_blocks,
			fmt::format(
				"Test 7: a 1-coarse-cell-thin footprint on axis {} is rejected in the read-only validation phase ({})",
				axis,
				message.empty() ? "no exception thrown" : fmt::format("threw: {}", message)
			)
		);
	}

	// the minimum valid footprint ([2, 8, 8] coarse cells) must be accepted
	std::string message;
	try {
		createAMRBlocks(state.nse, parseAMRConfig<NSE_CONFIG>("1 4 4 4 2 8 8"));
	}
	catch (const std::runtime_error& e) {
		message = e.what();
	}
	report(message.empty(), fmt::format("Test 7: the minimum [2, 8, 8] footprint is accepted ({})", message.empty() ? "no exception" : message));
	const std::vector<BLOCK*> level1 = state.nse.getBlocksAtLevel(1);
	report(
		level1.size() == 1 && level1.front()->local == idx3d{4, 16, 16},
		fmt::format("Test 7: the accepted [2, 8, 8] footprint created one level-1 block of 4x16x16 fine cells (got {} blocks)", level1.size())
	);
}

int main(int argc, char** argv)
{
	TNLMPI_INIT mpi(argc, argv);

	if (TNL::MPI::GetSize(MPI_COMM_WORLD) != 1) {
		fmt::println("RESULT: AMR subcycling tests are single-rank only (nproc = {})", TNL::MPI::GetSize(MPI_COMM_WORLD));
		return 1;
	}

	fmt::println("AMR subcycling unit tests (streaming pattern: {})", pattern_name);

	test_subcycling_schedule();
	test_max_level_zero_fallthrough();
	test_interface_ring_freshness();
	test_conservation_hidden_cell_exclusion();
	test_skin_partition_geometry();
	test_footprint_min_size_validation();

	if (g_failures == 0) {
		fmt::println("RESULT: all AMR subcycling tests passed");
		return 0;
	}
	fmt::println("RESULT: {} AMR subcycling check(s) FAILED", g_failures);
	return 1;
}
