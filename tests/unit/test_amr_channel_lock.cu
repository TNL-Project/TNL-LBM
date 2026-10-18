// Comparator translation unit of the cross-pattern channel-geometry lock
// (the SP-5 surface of results_drag_crisis/out_of_scope_survey.md): runs
// the five per-pattern scenario runners (test_amr_channel_lock_*.cu,
// instantiated through the explicit D3Q27_STREAMING_* types) SEQUENTIALLY
// in one process — the State constructor registers a global spdlog logger
// per instance, so the pattern instances cannot coexist, the sequential
// idiom of the mL1 bitwise lock — and compares the init-time GEO map, the
// macroscopic channels and the decoded canonical DF frames against the A-B
// pull reference trace, level by level, cycle by cycle, with the
// divergence anatomy classified by map-tag region (the census the SP-5
// deconstruction is built from).
//
// This TU instantiates no CUDA kernel: it links the five runner objects and
// works on their host-side traces only. The scenario and the comparison
// surface are documented in amr_channel_lock.h.

#include <cmath>
#include <cstdint>
#include <cstring>
#include <map>
#include <string>

#include <fmt/format.h>

#include <doctest/doctest.h>

#include "amr_channel_lock.h"

TEST_SUITE_BEGIN("amr_channel_lock");

// sentinel definitions (declared in amr_channel_lock.h; one definition for
// the whole binary so the runner TUs link cleanly)
double amrChannelFrameMask()
{
	return std::numeric_limits<double>::quiet_NaN();
}

bool amrChannelIsFrameMask(double v)
{
	return v != v;	// the sentinel is NaN (i != i); never matches a widened float
}

namespace
{

// GEO map tag names for the census (the D3Q27_BC_All enum values)
std::string regionName(int mapgi)
{
	switch (mapgi) {
		case 0: return "FLUID";
		case 1: return "WALL";
		case 2: return "INFLOW";
		case 3: return "INFLOW_MOMENT";
		case 4: return "INFLOW_BOUNCEBACK";
		case 5: return "INFLOW_EQ_LEFT";
		case 6: return "OUTFLOW_EQ";
		case 7: return "OUTFLOW_RIGHT";
		case 8: return "OUTFLOW_RIGHT_INTERP";
		case 9: return "NOTHING";
		case 10: return "SYMMETRY";
		case 16: return "AMR_INTERFACE";
		default: return fmt::format("GEO_{}", mapgi);
	}
}

// bitwise compare of two widened-float doubles, honoring the frame mask
// (masked-vs-masked is equal, masked-vs-content is a difference)
bool sameBits(double a, double b)
{
	const bool ma = amrChannelIsFrameMask(a), mb = amrChannelIsFrameMask(b);
	if (ma || mb)
		return ma == mb;
	std::uint64_t ua, ub;
	std::memcpy(&ua, &a, 8);
	std::memcpy(&ub, &b, 8);
	return ua == ub;
}

long floatUlpDistance(double a, double b)
{
	const float fa = static_cast<float>(a);
	const float fb = static_cast<float>(b);
	std::int32_t ia, ib;
	std::memcpy(&ia, &fa, 4);
	std::memcpy(&ib, &fb, 4);
	return std::abs(static_cast<long>(ia) - static_cast<long>(ib));
}

// per-cycle, per-level, per-surface census row
struct ChannelCycleCensus
{
	long ndiff = 0;
	double maxabs = 0;
	long maxulp = 0;
	std::map<std::string, long> by_region;
	int first_x = -1, first_y = -1, first_z = -1, first_channel = -1;
	double first_a = 0, first_b = 0;
	// coordinate histograms of the differing cells (the anatomy)
	std::map<int, long> by_x, by_y, by_z;
	// sample sites for drill-down (channel, x, y, z, ref, test)
	std::vector<std::array<std::string, 6>> samples;
};

struct ChannelPairDiff
{
	bool shape_ok = true;
	std::string shape_error;
	// [cycle][level] census per surface
	std::map<std::pair<int, int>, ChannelCycleCensus> macro;
	std::map<std::pair<int, int>, ChannelCycleCensus> frame;
	long macro_cells = 0, frame_cells = 0;
	long map_cells = 0;
	long frame_masked = 0;	// entries masked in either pattern (edge-slot class)
	int first_diff_cycle = -1;
};

// compare one cycle's per-level snapshots (channel uses plain indices into
// the flattened [z][y][x] boxes); map0 is the init-time map of the block
// (cycle 0 snapshot, topology tagging is init-time-static), used for the
// region classification on every cycle
void compareLevelSnap(
	const AMRChannelLevelSnap& rs,
	const AMRChannelLevelSnap& ts,
	const AMRChannelLevelSnap* map0,
	int cycle,
	int slot,
	ChannelPairDiff& out
)
{
	const long cells_xyz = static_cast<long>(rs.size[0]) * rs.size[1] * rs.size[2];
	auto classify = [&](long i) -> std::string
	{
		if (map0 == nullptr || map0->map.empty())
			return "?";
		return regionName(map0->map[i]);
	};
	auto record = [&](ChannelCycleCensus& c, double a, double b, long i, int channel)
	{
		c.ndiff++;
		const double d = std::abs(a - b);
		if (d > c.maxabs) {
			c.maxabs = d;
			c.maxulp = floatUlpDistance(a, b);
		}
		c.by_region[classify(i)]++;
		const int cx = static_cast<int>(i % rs.size[0]) + rs.origin[0];
		const int cy = static_cast<int>((i / rs.size[0]) % rs.size[1]) + rs.origin[1];
		const int cz = static_cast<int>(i / (rs.size[0] * rs.size[1])) + rs.origin[2];
		c.by_x[cx]++;
		c.by_y[cy]++;
		c.by_z[cz]++;
		if (c.first_channel < 0) {
			c.first_channel = channel;
			c.first_x = cx;
			c.first_y = cy;
			c.first_z = cz;
			c.first_a = a;
			c.first_b = b;
		}
		if (c.samples.size() < 6)
			c.samples.push_back({fmt::format("{}", channel),
				fmt::format("{} {} {}", cx, cy, cz),
				classify(i),
				fmt::format("{:.10e}", a),
				fmt::format("{:.10e}", b),
				fmt::format("{:+.3e}", a - b)});
	};

	// init-time GEO map (cycle 0 only)
	if (cycle == 0)
		for (long i = 0; i < cells_xyz; i++)
			if (rs.map[i] != ts.map[i])
				out.map_cells++;

	for (int m = 0; m < rs.macro_channels; m++)
		for (long i = 0; i < cells_xyz; i++) {
			const double a = rs.macro[static_cast<std::size_t>(m) * cells_xyz + i];
			const double b = ts.macro[static_cast<std::size_t>(m) * cells_xyz + i];
			if (! sameBits(a, b)) {
				record(out.macro[{cycle, slot}], a, b, i, m);
				out.macro_cells++;
			}
		}

	for (int qq = 0; qq < 27; qq++)
		for (long i = 0; i < cells_xyz; i++) {
			const double a = rs.frame[static_cast<std::size_t>(i) * 27 + qq];
			const double b = ts.frame[static_cast<std::size_t>(i) * 27 + qq];
			// edge-slot class: the candidate's storage validity is
			// pattern-specific at the block frame (the esoteric layouts give
			// physical roles to storage positions the A-B patterns leave
			// out of the gather), so masked-in-either entries are not
			// comparable content and count as their own class
			if (amrChannelIsFrameMask(a) || amrChannelIsFrameMask(b)) {
				out.frame_masked++;
				continue;
			}
			if (! sameBits(a, b)) {
				record(out.frame[{cycle, slot}], a, b, i, qq);
				out.frame_cells++;
			}
		}
}

ChannelPairDiff compareTraces(const AMRChannelTrace& ref, const AMRChannelTrace& test)
{
	ChannelPairDiff out;
	if (ref.cycles.size() != test.cycles.size()) {
		out.shape_ok = false;
		out.shape_error = fmt::format("cycle counts differ: {} (ref ab) vs {} ({})", ref.cycles.size(), test.cycles.size(), test.pattern);
		return out;
	}
	for (std::size_t c = 0; c < ref.cycles.size(); c++) {
		const AMRChannelCycle& rc = ref.cycles[c];
		const AMRChannelCycle& tc = test.cycles[c];
		if (rc.levels.size() != tc.levels.size()) {
			out.shape_ok = false;
			out.shape_error = fmt::format("cycle {}: level counts differ: {} vs {}", c, rc.levels.size(), tc.levels.size());
			return out;
		}
		for (std::size_t l = 0; l < rc.levels.size(); l++) {
			const AMRChannelLevelSnap& rs = rc.levels[l];
			const AMRChannelLevelSnap& ts = tc.levels[l];
			if (rs.level != ts.level || std::memcmp(rs.origin, ts.origin, sizeof rs.origin) != 0
				|| std::memcmp(rs.size, ts.size, sizeof rs.size) != 0 || rs.macro_channels != ts.macro_channels
				|| rs.macro.size() != ts.macro.size() || rs.frame.size() != ts.frame.size()) {
				out.shape_ok = false;
				out.shape_error = fmt::format("cycle {}, level slot {}: block geometry differs (level {} vs {})", c, l, rs.level, ts.level);
				return out;
			}
			const AMRChannelLevelSnap* map0 = nullptr;
			if (c != 0) {
				for (const auto& s0 : ref.cycles.front().levels)
					if (s0.level == rs.level && std::memcmp(s0.origin, rs.origin, sizeof rs.origin) == 0) {
						map0 = &s0;
						break;
					}
			}
			else
				map0 = &rs;
			const long before = out.macro_cells + out.frame_cells;
			compareLevelSnap(rs, ts, map0, static_cast<int>(c), static_cast<int>(l), out);
			if (out.first_diff_cycle < 0 && out.macro_cells + out.frame_cells > before)
				out.first_diff_cycle = static_cast<int>(c);
		}
	}
	return out;
}

// look up a macro/map entry of the reference trace for coordinate probing
// (used by the census print to name first-divergent regions)
const AMRChannelLevelSnap* findLevel(const AMRChannelCycle& c, int level)
{
	for (const auto& s : c.levels)
		if (s.level == level)
			return &s;
	return nullptr;
}

}  // namespace

TEST_CASE("R=1 channel scenario census (two-footprint registration over the BC map)")
{
	AMRChannelTrace ref = runAMRChannelScenario_ab();
	CHECK_MESSAGE(ref.init_ok, "AMR channel lock: the ab reference scenario failed to initialize");
	CHECK_MESSAGE(
		ref.cycles_completed == AMR_CHANNEL_LOCK_CYCLES + 1,
		fmt::format("AMR channel lock: the ab reference completed {} cycle snapshots, expected {}", ref.cycles_completed, AMR_CHANNEL_LOCK_CYCLES + 1)
	);
	if (! ref.init_ok || ref.cycles.empty())
		return;

	// two couplings (one per footprint), three blocks (one coarse, two fine)
	const AMRChannelCycle& c0 = ref.cycles.front();
	CHECK(c0.levels.size() == 3);
	const AMRChannelLevelSnap* L0 = findLevel(c0, 0);
	REQUIRE(L0 != nullptr);
	REQUIRE(! L0->map.empty());
	const long sx = L0->size[0], sy = L0->size[1];
	auto map_at = [&](int x, int y, int z) -> int
	{
		const long i = (static_cast<long>(z) - L0->origin[2]) * sx * sy + (static_cast<long>(y) - L0->origin[1]) * sx + (static_cast<long>(x) - L0->origin[0]);
		return L0->map[i];
	};
	// boundary map of the sim verbatim: inflow / outflow-interp / walls /
	// edge layers
	CHECK(map_at(1, 4, 4) == 3);		   // GEO_INFLOW_MOMENT
	CHECK(map_at(62, 4, 4) == 8);		   // GEO_OUTFLOW_RIGHT_INTERP
	CHECK(map_at(32, 4, 1) == 1);		   // GEO_WALL (bottom)
	CHECK(map_at(32, 4, 14) == 1);		   // GEO_WALL (top)
	CHECK(map_at(32, 4, 0) == 9);		   // GEO_NOTHING edge
	CHECK(map_at(0, 4, 4) == 9);		   // GEO_NOTHING edge
	CHECK(map_at(63, 4, 4) == 9);		   // GEO_NOTHING edge
	// the inflow-adjacent first footprint: halo row x = 1 on the inflow
	// plane, ring x = 2 tagged GEO_AMR_INTERFACE, skin x = 3 frozen
	CHECK(map_at(2, 7, 7) == 16);		   // GEO_AMR_INTERFACE ring
	CHECK(map_at(3, 7, 7) == 9);		   // frozen GEO_NOTHING skin/depth
	// the channel's wall-attached second footprint: z-min halo row IS the
	// bottom wall (stays GEO_WALL), ring shell above it reactivated
	CHECK(map_at(24, 4, 1) == 1);		   // halo row on the wall plane
	CHECK(map_at(24, 4, 2) == 16);		   // ring shell (z-min surface)
	CHECK(map_at(26, 6, 3) == 9);		   // frozen skin depth 1 (interior-tangent)
}

TEST_CASE("cross-pattern cycle compare vs ab (macro + decoded canonical DF frame, SP-5 evidence census)")
{
	// run the five pattern instances sequentially (they cannot coexist)
	AMRChannelTrace traces[5] = {
		runAMRChannelScenario_ab(),
		runAMRChannelScenario_aa(),
		runAMRChannelScenario_eso_twist(),
		runAMRChannelScenario_eso_pull(),
		runAMRChannelScenario_eso_push(),
	};
	for (const auto& trace : traces) {
		CHECK_MESSAGE(trace.init_ok, fmt::format("AMR channel lock: scenario '{}' failed to initialize", trace.pattern));
		CHECK_MESSAGE(
			trace.cycles_completed == AMR_CHANNEL_LOCK_CYCLES + 1,
			fmt::format("AMR channel lock: scenario '{}' completed {} cycle snapshots, expected {}", trace.pattern, trace.cycles_completed, AMR_CHANNEL_LOCK_CYCLES + 1)
		);
	}
	const AMRChannelTrace& ref = traces[0];
	if (! ref.init_ok || ref.cycles_completed != AMR_CHANNEL_LOCK_CYCLES + 1)
		return;

	for (int p = 1; p < 5; p++) {
		const AMRChannelTrace& test = traces[p];
		if (! test.init_ok || test.cycles_completed != AMR_CHANNEL_LOCK_CYCLES + 1)
			continue;
		const ChannelPairDiff diff = compareTraces(ref, test);
		if (! diff.shape_ok) {
			CHECK_MESSAGE(false, fmt::format("AMR channel lock: {} vs ab: {}", test.pattern, diff.shape_error));
			continue;
		}
		// the census print (evidence surface): per (cycle, slot) rows with
		// the region histogram of both comparison surfaces
		const int nslots = static_cast<int>(ref.cycles.front().levels.size());
		fmt::print(
			"AMR channel lock census: {} vs ab: map diffs {}, edge-slot masked {}, first-divergent cycle {} (macro cells {}, canonical-frame cells {})\n",
			test.pattern,
			diff.map_cells,
			diff.frame_masked,
			diff.first_diff_cycle,
			diff.macro_cells,
			diff.frame_cells
		);
		for (int c = 0; c <= AMR_CHANNEL_LOCK_CYCLES; c++)
			for (int l = 0; l < nslots; l++) {
				const auto mi = diff.macro.find({c, l});
				const auto fi = diff.frame.find({c, l});
				const long mn = mi != diff.macro.end() ? mi->second.ndiff : 0;
				const long fn = fi != diff.frame.end() ? fi->second.ndiff : 0;
				if (mn == 0 && fn == 0)
					continue;
				auto regions = [](const ChannelCycleCensus* c1, const ChannelCycleCensus* c2)
				{
					std::string s;
					auto merge = [&](const ChannelCycleCensus* c_)
					{
						if (! c_)
							return;
						for (const auto& kv : c_->by_region)
							s += fmt::format(" {}:{}", kv.first, kv.second);
					};
					merge(c1);
					merge(c2);
					return s;
				};
				const auto& s0 = ref.cycles.front().levels[l];
				fmt::print(
					"  cycle {:2d} block(level {}, origin {} {} {}): macro {:5d} (max {:9.3e}) frame {:7d} (max {:9.3e}):{}\n",
					c,
					s0.level,
					s0.origin[0],
					s0.origin[1],
					s0.origin[2],
					mn,
					mi != diff.macro.end() ? mi->second.maxabs : 0.0,
					fn,
					fi != diff.frame.end() ? fi->second.maxabs : 0.0,
					regions(mi != diff.macro.end() ? &mi->second : nullptr, fi != diff.frame.end() ? &fi->second : nullptr)
				);
				// first-divergent sites of the first divergence cycle
				if (c == diff.first_diff_cycle) {
					for (const auto* ci : {mi != diff.macro.end() ? &mi->second : nullptr, fi != diff.frame.end() ? &fi->second : nullptr})
						if (ci && ci->first_channel >= 0)
							fmt::print(
								"    first: channel/q {} cell ({} {} {}): ref {:.9e} vs {} {:.9e}\n",
								ci->first_channel,
								ci->first_x,
								ci->first_y,
								ci->first_z,
								ci->first_a,
								test.pattern,
								ci->first_b
							);
				}
				// coordinate-anatomy rows for the macro surface of the
				// first 4 divergence cycles (the birth anatomy of each
				// block)
				if (mi != diff.macro.end() && mi->second.ndiff > 0 && c <= 6) {
					auto coords = [](const std::map<int, long>& m)
					{
						std::string s;
						long shown = 0;
						for (const auto& kv : m) {
							if (m.size() > 12 && kv.second < m.begin()->second)
								continue;
							s += fmt::format(" {}({})", kv.first, kv.second);
							if (++shown >= 12) {
								s += " ...";
								break;
							}
						}
						return s;
					};
					fmt::print("    macro coords: x:{} | y:{} | z:{}\n", coords(mi->second.by_x), coords(mi->second.by_y), coords(mi->second.by_z));
					for (const auto& s : mi->second.samples)
						fmt::print("      ch {} cell ({}) [{}]: ref {} vs {} d={}\n", s[0], s[1], s[2], s[3], s[4], s[5]);
				}
				if (fi != diff.frame.end() && fi->second.ndiff > 0 && c <= 6) {
					auto coords = [](const std::map<int, long>& m)
					{
						std::string s;
						long shown = 0;
						for (const auto& kv : m) {
							if (m.size() > 12 && kv.second < m.begin()->second)
								continue;
							s += fmt::format(" {}({})", kv.first, kv.second);
							if (++shown >= 12) {
								s += " ...";
								break;
							}
						}
						return s;
					};
					fmt::print("    frame coords: x:{} | y:{} | z:{}\n", coords(fi->second.by_x), coords(fi->second.by_y), coords(fi->second.by_z));
					for (const auto& s : fi->second.samples)
						fmt::print("      q {} cell ({}) [{}]: ref {} vs {} d={}\n", s[0], s[1], s[2], s[3], s[4], s[5]);
				}
			}
		// topology must be bitwise identical in every trace (map tagging is
		// pattern-independent init-time work); any violation is a real bug
		CHECK_MESSAGE(diff.map_cells == 0, fmt::format("AMR channel lock: {} vs ab: {} differing map cells (init-time topology must be pattern-independent)", test.pattern, diff.map_cells));

		// classification contract of the SP-5 partition verdict
		// (results_drag_crisis/sp5/, survey annotation): macroscopic content
		// is bitwise identical through cycle 1 on every block; the only
		// admissible birth classes are (i) cycle 2 on the wall-attached
		// slab — the wall-adjacent band row, dyadic-scale dust — and
		// (ii) cycle 3 on level 0 — the x == 61 plane facing the frozen
		// outflow wall (the AMR driver never launches the base driver's
		// two-pass outflow) — and on the inflow-adjacent slab's band; every
		// birth cell is GEO_FLUID with dyadic-scale amplitudes; later cycles
		// only amplify these birth classes advectively — a violation of any
		// anchor means the documented partition boundary moved and the
		// verdict must be re-derived
		for (std::size_t l = 0; l < ref.cycles.front().levels.size(); l++) {
			const AMRChannelLevelSnap& s = ref.cycles.front().levels[l];
			int first = -1;
			const ChannelCycleCensus* birth = nullptr;
			for (int c = 0; c <= AMR_CHANNEL_LOCK_CYCLES; c++) {
				auto it = diff.macro.find({c, static_cast<int>(l)});
				if (it != diff.macro.end() && it->second.ndiff > 0) {
					first = c;
					birth = &it->second;
					break;
				}
			}
			const bool is_l0 = s.level == 0;
			const bool is_fp1 = s.origin[0] == 3 && s.origin[1] == 11 && s.origin[2] == 10;
			const bool is_fp2 = s.origin[0] == 47 && s.origin[1] == 7 && s.origin[2] == 2;
			const bool block_recognized = is_l0 || is_fp1 || is_fp2;
			REQUIRE(block_recognized);
			const int expected_first = is_fp2 ? 2 : 3;
			CHECK_MESSAGE(
				first == expected_first,
				fmt::format(
					"AMR channel lock partition: {} vs ab: block(level {}, origin {} {} {}) first macro-divergent cycle {} violates the SP-5 birth anchor {}",
					test.pattern, s.level, s.origin[0], s.origin[1], s.origin[2], first, expected_first
				)
			);
			if (birth == nullptr || first != expected_first)
				continue;
			const bool birth_region_only_fluid = birth->by_region.size() == 1 && birth->by_region.begin()->first == "FLUID";
			CHECK_MESSAGE(
				birth_region_only_fluid,
				fmt::format("AMR channel lock partition: {}: birth cells must all be GEO_FLUID, got regions", test.pattern)
			);
			const bool birth_amplitude_dust = birth->maxabs <= 5e-8;
			CHECK_MESSAGE(
				birth_amplitude_dust,
				fmt::format("AMR channel lock partition: {}: birth-cycle amplitude {:.3e} exceeds the dyadic-dust bound", test.pattern, birth->maxabs)
			);
			const bool birth_l0_outflow_plane = birth->by_x.size() == 1 && birth->by_x.begin()->first == 61;
			if (is_l0)
				CHECK_MESSAGE(
					birth_l0_outflow_plane,
					fmt::format("AMR channel lock partition: {}: level-0 birth must be confined to the outflow-adjacent plane x == 61", test.pattern)
				);
			const bool birth_fp2_band_row = birth->by_z.size() == 1 && birth->by_z.begin()->first == 3;
			if (is_fp2)
				CHECK_MESSAGE(
					birth_fp2_band_row,
					fmt::format("AMR channel lock partition: {}: wall-slab birth must be confined to the band row z == 3", test.pattern)
				);
		}
	}

	// probe: per-cycle macro time series at the outflow-adjacent plane
	// x = 61 (the thread that irreducible-at-first-sight L0 divergence at
	// cycle 3) — reconciles the macro refresh provenance per pattern
	{
		const int probe_yz[3][2] = {{6, 6}, {0, 2}, {15, 13}};
		for (const auto& trace : traces) {
			if (! trace.init_ok)
				continue;
			fmt::print("AMR channel lock probe ({}):", trace.pattern);
			for (int c = 0; c <= 3 && c < static_cast<int>(trace.cycles.size()); c++) {
				const auto* L0 = findLevel(trace.cycles[c], 0);
				if (L0 == nullptr || L0->macro.empty())
					continue;
				const long sx = L0->size[0], sy = L0->size[1];
				fmt::print("  c{}=[", c);
				for (const auto& p : probe_yz) {
					const long i = (static_cast<long>(p[1]) - L0->origin[2]) * sx * sy + (static_cast<long>(p[0]) - L0->origin[1]) * sx + (61L - L0->origin[0]);
					const long cells_xyz = sx * sy * L0->size[2];
					// channel 0 = rho, 1 = vx
					fmt::print("({},{})v {:.6e}|", p[0], p[1], L0->macro[1 * cells_xyz + i]);
				}
				fmt::print("]");
			}
			fmt::print("\n");
		}
	}

	// launch-level attribution: per-phase snapshots at the launch hooks
	// (substepA/substepB/kernel(L0)/f2c/c2f) inside cycles 1..3 — the
	// first divergent phase names the authoring launch of each class
	for (int p = 1; p < 5; p++) {
		const AMRChannelTrace& test = traces[p];
		if (! test.init_ok || ref.phases.empty() || test.phases.empty())
			continue;
		CHECK_MESSAGE(
			ref.phases.size() == test.phases.size(),
			fmt::format("AMR channel lock phases: {} has {} phase snapshots, ab has {}", test.pattern, test.phases.size(), ref.phases.size())
		);
		fmt::print("AMR channel lock phases: {} vs ab ({} phase points):\n", test.pattern, test.phases.size());
		const std::size_t nph = std::min(ref.phases.size(), test.phases.size());
		for (std::size_t ph = 0; ph < nph; ph++) {
			const auto& [rlabel, rc] = ref.phases[ph];
			const auto& [tlabel, tc] = test.phases[ph];
			CHECK(rlabel == tlabel);
			// per-block macro/frame diff counts at this phase point
			std::string row;
			for (std::size_t l = 0; l < rc.levels.size(); l++) {
				const AMRChannelLevelSnap& rs = rc.levels[l];
				const AMRChannelLevelSnap& ts = tc.levels[l];
				const long cells_xyz = static_cast<long>(rs.size[0]) * rs.size[1] * rs.size[2];
				long mn = 0, fn = 0;
				int mx = -1, my = -1, mz = -1, mc = -1;
				double ma = 0, mb = 0;
				for (int m = 0; m < rs.macro_channels; m++)
					for (long i = 0; i < cells_xyz; i++) {
						const double a = rs.macro[static_cast<std::size_t>(m) * cells_xyz + i];
						const double b = ts.macro[static_cast<std::size_t>(m) * cells_xyz + i];
						if (! sameBits(a, b)) {
							mn++;
							if (mc < 0) {
								mc = m;
								mx = static_cast<int>(i % rs.size[0]) + rs.origin[0];
								my = static_cast<int>((i / rs.size[0]) % rs.size[1]) + rs.origin[1];
								mz = static_cast<int>(i / (rs.size[0] * rs.size[1])) + rs.origin[2];
								ma = a;
								mb = b;
							}
						}
					}
				for (int qq = 0; qq < 27; qq++)
					for (long i = 0; i < cells_xyz; i++) {
						const double a = rs.frame[static_cast<std::size_t>(i) * 27 + qq];
						const double b = ts.frame[static_cast<std::size_t>(i) * 27 + qq];
						if (amrChannelIsFrameMask(a) || amrChannelIsFrameMask(b))
							continue;
						if (! sameBits(a, b))
							fn++;
					}
				if (mn == 0 && fn == 0)
					continue;
				row += fmt::format(" blk(level {} {}x{}x{} @ {} {} {}): macro {} frame {}", rs.level, rs.size[0], rs.size[1], rs.size[2], rs.origin[0], rs.origin[1], rs.origin[2], mn, fn);
				if (mc >= 0)
					row += fmt::format(" first m/{} ({} {} {}): {:.9e} vs {:.9e}", mc, mx, my, mz, ma, mb);
			}
			if (! row.empty())
				fmt::print("  {}:{}\n", rlabel, row);
		}
	}
}

TEST_SUITE_END();
