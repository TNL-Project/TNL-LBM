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
#include <string>

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

	if (g_failures == 0) {
		fmt::println("RESULT: all AMR subcycling tests passed");
		return 0;
	}
	fmt::println("RESULT: {} AMR subcycling check(s) FAILED", g_failures);
	return 1;
}
