#pragma once

// Shared fixture machinery for the AMR gate binaries, extracted verbatim from
// test_amr_subcycling.cu so that test_amr_nesting.cu reuses the same census
// spies, map-scan carriers and reference-stat helpers (extraction precondition
// of the amr-nlevel-nesting plan's commit B; the subcycling tests compile from
// this header with byte-identical behavior):
//
// - makeLattice / setSineInitialCondition: the 16^3 periodic-box lattice and
//   the non-uniform (kernel-detectable) initial condition;
// - StateLocal_AMR / StateLocal_Base: pass-through subclasses wiring the sine
//   IC into State_AMR and its plain-State sibling;
// - StateSchedule_AMR: the schedule-observing spy overriding State_AMR's
//   virtual launch helpers to record one Event per stage launch (kind, level,
//   ghost_layers, parity-at-call-site);
// - HostSnapshot / snapshotBlock / maxAbsDiffSnapshot(s): host-side bitwise
//   comparison carriers for cross-state bitwise locks;
// - CoarseMacroScan / captureCoarseMacros and FineGhostScan /
//   captureFineGhost / ghostLayerDepth: scan-order map+macro censuses of the
//   coarse block and of the fine block's C2F destination complement;
// - RefStats / computeReferenceStats / closeRel: the host-side conservation
//   reference sums (GEO_NOTHING exclusion) and the OpenMP-tolerant closeness.

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

// N^3 periodic box in physical units (same scaling as sim_AMR at that N: the
// subcycling fixtures use N = 16): LBM viscosity nu_lb_coarse = 0.005, hence
// nu_lb_fine = 0.01 and physDt_fine = physDt_coarse / 2 exactly (binary
// halving); periodicity is declared per-dimension via the bool3d passed to
// the State/LBM constructors
lat_t makeLattice(int N = 16)
{
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

	void outputData(UniformDataWriter<typename NSE::TRAITS>&, const LBM_BLOCK<NSE>&, const idx3d&, const idx3d&) override {}
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

	void outputData(UniformDataWriter<typename NSE::TRAITS>&, const LBM_BLOCK<NSE>&, const idx3d&, const idx3d&) override {}
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

// Schedule-observing State_AMR subclass for Test 1: the virtual launch
// helpers of State_AMR are overridden to record one event per stage launch
// (kind, level, and the parity state at the call site) and then delegate to
// the production implementation, so the SIX-STEP cycle's launch census and
// AB-frame pairing are asserted from the real call graph (not from baked
// counts). Under AB the parity evidence is the DF-pointer rotation of each
// block (df_cur/df_out = data.dfs[0]/[1] compared against the physical
// arrays dfs[0]/dfs[1]); under AA (single array, DFMAX == 1) it is
// data.even_iter.
//
// For events at level >= 1 the parent_* fields carry the parity of the
// level-(level-1) block at the call site (at max_level == 1 that block is
// level 0, the same evidence as coarse_cur/coarse_even); next_parent_substep
// snapshots the write-side next-substep index the fine-to-coarse parity
// argument derives from (level-0 parents: the post-incremented global
// iterations clock; finer parents: the level's cumulative totalSubstepCount).
template <typename NSE>
struct StateSchedule_AMR : StateLocal_AMR<NSE>
{
	using Base = State_AMR<NSE>;
	using BLOCK_NSE = typename Base::BLOCK_NSE;

	// pass-through constructor (forwards periodic/max_level to LBM)
	template <typename... ARGS>
	StateSchedule_AMR(ARGS&&... args)
	: StateLocal_AMR<NSE>(std::forward<ARGS>(args)...)
	{}

	enum class Stage
	{
		kernel,	 // launchLBMKernelForLevel (fine substep or coarse step)
		c2f,	 // launchCoarseToFineTransfers (ghost fill of the cycle)
		f2c,	 // launchFineToCoarseTransfersInterior (skin feedback)
	};
	struct Event
	{
		Stage stage;
		int level = -1;
		int ghost_layers = 0;  // kernel launch extent class (0 = interior-only, 1 = widened simulated-band substep)
#ifdef AB_PATTERN
		const void* fine_cur = nullptr;	   // fine block's data.dfs[0] (df_cur) at the call site
		const void* fine_out = nullptr;	   // fine block's data.dfs[1] (df_out) at the call site
		const void* coarse_cur = nullptr;  // level-0 block's data.dfs[0] (df_cur) at the call site
		const void* parent_cur = nullptr;  // level-(level-1) block's data.dfs[0] at the call site (level >= 1 events)
		const void* parent_out = nullptr;  // level-(level-1) block's data.dfs[1] at the call site (level >= 1 events)
#elif defined(AA_PATTERN)
		bool fine_even = false;
		bool coarse_even = false;
		bool parent_even = false;
#endif
		// f2c events only: the write-side next-substep index sampled at the
		// call site (the expression launchFineToCoarseTransfersInterior's
		// parity argument derives from); -1 on every other event
		int next_parent_substep = -1;
	};
	std::vector<Event> events;

	void record(Stage stage, int level, int ghost_layers = 0)
	{
		Event e;
		e.stage = stage;
		e.level = level;
		e.ghost_layers = ghost_layers;
		// the fixture has exactly one block per level, so the coarse side of
		// every level-1 transfer is the level-0 block and the coarse side of
		// a level >= 2 event is the level-(level-1) block
		BLOCK_NSE* fine = level > 0 ? this->nse.getBlocksAtLevel(level).front() : nullptr;
		BLOCK_NSE* coarse = this->nse.getBlocksAtLevel(0).front();
		BLOCK_NSE* parent = level > 0 ? this->nse.getBlocksAtLevel(level - 1).front() : nullptr;
#ifdef AB_PATTERN
		if (fine != nullptr) {
			e.fine_cur = fine->data.dfs[0];
			e.fine_out = fine->data.dfs[1];
		}
		if (parent != nullptr) {
			e.parent_cur = parent->data.dfs[0];
			e.parent_out = parent->data.dfs[1];
		}
		e.coarse_cur = coarse->data.dfs[0];
#elif defined(AA_PATTERN)
		if (fine != nullptr)
			e.fine_even = fine->data.even_iter;
		if (parent != nullptr)
			e.parent_even = parent->data.even_iter;
		e.coarse_even = coarse->data.even_iter;
#endif
		if (stage == Stage::f2c && level > 0)
			e.next_parent_substep = level > 1 ? this->nse.totalSubstepCount[level - 1] : this->nse.iterations;
		events.push_back(e);
	}

	void launchLBMKernelForLevel(int level, bool compute_macro, int ghost_layers) override
	{
		record(Stage::kernel, level, ghost_layers);
		Base::launchLBMKernelForLevel(level, compute_macro, ghost_layers);
	}
	void launchCoarseToFineTransfers(int fine_level) override
	{
		record(Stage::c2f, fine_level);
		Base::launchCoarseToFineTransfers(fine_level);
	}
	void launchFineToCoarseTransfersInterior(int fine_level) override
	{
		record(Stage::f2c, fine_level);
		Base::launchFineToCoarseTransfersInterior(fine_level);
	}
};

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

// scan-order capture of the fine block's C2F destination cells (the stored
// overlap complement outside the simulated interior) from the AB frames --
// the cross-state carrier of Test 4's fill and simulated-band locks. Host
// mirrors take (offset + local) coordinates with the +-overlap margin (the
// computeReferenceStats idiom), so the destination window is
// [offset - ov, offset + local + ov)^3 minus the interior. The per-entry
// coordinates are kept so the assertions can split the INNER overlap layer
// (Chebyshev distance 1 from the interior box: the kernel-integrated
// simulated-band rows) from the OUTER layer (distance 2: the fill-only
// streaming source of the inner rows).
struct FineGhostScan
{
	idx3d offset{0, 0, 0};
	idx3d local{0, 0, 0};
	idx3d overlap{0, 0, 0};
	std::vector<idx3d> coords;	 // per-entry global coordinates (same order as frame0/frame1)
	std::vector<int> map;		 // per-entry GEO tag (the simulated band relies on GEO_FLUID ghost rows)
	std::vector<double> frame0;	 // per-destination-cell values of dfs[0]
	std::vector<double> frame1;	 // per-destination-cell values of dfs[1] (AB only)
};

FineGhostScan captureFineGhost(BLOCK& block)
{
	FineGhostScan scan;
	scan.offset = block.offset;
	scan.local = block.local;
	scan.overlap = {block.df_overlap_X(), block.df_overlap_Y(), block.df_overlap_Z()};
	for (int q = 0; q < NSE_CONFIG::Q; q++)
		for (idx z = scan.offset.z() - scan.overlap.z(); z < scan.offset.z() + scan.local.z() + scan.overlap.z(); z++)
			for (idx y = scan.offset.y() - scan.overlap.y(); y < scan.offset.y() + scan.local.y() + scan.overlap.y(); y++)
				for (idx x = scan.offset.x() - scan.overlap.x(); x < scan.offset.x() + scan.local.x() + scan.overlap.x(); x++) {
					// destination complement: skip the simulated interior
					// cells (they are kernel-owned, not fill-owned)
					const bool interior = x >= scan.offset.x() && x < scan.offset.x() + scan.local.x() && y >= scan.offset.y()
									   && y < scan.offset.y() + scan.local.y() && z >= scan.offset.z() && z < scan.offset.z() + scan.local.z();
					if (interior)
						continue;
					scan.coords.emplace_back(idx3d{x, y, z});
					scan.map.push_back(block.hmap(x, y, z));
					scan.frame0.push_back(block.hfs[0](q, x, y, z));
#ifdef AB_PATTERN
					scan.frame1.push_back(block.hfs[1](q, x, y, z));
#endif
				}
	return scan;
}

// Chebyshev distance of a destination-complement cell from the interior box
// [offset, offset + local): 1 = inner overlap layer (the simulated-band rows
// the substep-1 kernel integrates), 2 = outer overlap layer (the fill-only
// streaming source of the inner rows).
int ghostLayerDepth(const FineGhostScan& scan, const idx3d& c)
{
	const idx d_x = c.x() < scan.offset.x() ? scan.offset.x() - c.x() : c.x() - (scan.offset.x() + scan.local.x() - 1);
	const idx d_y = c.y() < scan.offset.y() ? scan.offset.y() - c.y() : c.y() - (scan.offset.y() + scan.local.y() - 1);
	const idx d_z = c.z() < scan.offset.z() ? scan.offset.z() - c.z() : c.z() - (scan.offset.z() + scan.local.z() - 1);
	return static_cast<int>(std::max(std::max(d_x, d_y), d_z));
}

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
// the double-count signal (216 hidden cells with sentinel rho ~= 3.8e4) exceeds 9e3
// and dwarfs both this tolerance and any reassociation noise
bool closeRel(double a, double b)
{
	return std::abs(a - b) <= 1e-6 * std::max({1.0, std::abs(a), std::abs(b)});
}
