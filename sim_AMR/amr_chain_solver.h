#pragma once

// Footprint-chain solver of sim_AMR_channel's nested mode (commit F of the
// amr-nlevel-nesting plan): derives the level-2..4 AMR footprints instead of
// hand-writing the region string. Every derived level is wall-attached to the
// bottom plane (the user-locked target geometry: the V10 wall chain 0..4,
// each level's z-min footprint face aligned with its parent's edge, gap 0,
// reaching the level-1 face on the existing wall-candidate lane z = R+1
// level-0 cells). Every other face telescopes inward by `inset` parent-level
// cells per hop (inset = 3: the V-suite's no-warning tier; the z-min face is
// wall-shared/exempt, so the z budget carries only the z-max inset).
//
// The derivation works in integer parent-cell frames (createAMRBlocks' own
// frame, so all divisions are exact by construction): rect_L = the footprint
// rectangle in level-L cells; rect_1 is the 2-level channel's level-1 anchor
// footprint ("1 24R 4R (R+1) 16R 8R 8R") doubled into level-1 cells; each
// hop insets the parent rect and doubles into the child frame
// (rect_L = 2 * inset(rect_{L-1})). The region-file lines follow from
// createAMRBlocks' parent-frame conversion (amrParentFrameOrigin):
//
//     origin = (rect_L.lo / 2) * 2^(L-1),  size = (rect_L.span / 2) * 2^(L-1)
//
// per component (rect_L is even at every level, so the /2 is exact and the
// product is a multiple of 2^(L-1): the V4 alignment holds automatically).
//
// The solver's own checks mirror only the hard floors (per-axis footprint
// span >= 3 parent-level cells, z-min gap-0 chain consistency); the
// authoritative guard remains createAMRBlocks' full V-suite, which runs on
// the emitted region spec in the sim and throws on any violation. The
// derived spec is logged so the configuration is reproducible from the run
// log alone.

#include <array>
#include <stdexcept>
#include <string>
#include <vector>

#include <fmt/core.h>
#include <spdlog/spdlog.h>

struct AMRChainLevelGeometry
{
	int level = 0;
	std::array<int, 3> origin{};	   // region-file coordinates (the level-0 convention: physical = value / 4^(L-1) level-0 cells)
	std::array<int, 3> size{};		   // region-file footprint size
	std::array<int, 3> parent_span{};  // footprint span in parent-level cells (the V2/frame quantity)
	std::array<int, 3> fine_span{};	   // footprint span in fine cells of this level
};

struct AMRChannelChain
{
	std::vector<AMRChainLevelGeometry> levels;	// one entry per level 1..max_level
	std::string region_config;					// parseAMRConfig-ready region spec
};

inline AMRChannelChain deriveAMRChannelChain(int R, int max_level)
{
	// telescoping inset per non-wall face per hop, in parent-level cells
	// (3 = the no-warning tier of the V7/V9 gap rule); the z-min face is
	// wall-shared (gap 0 down the whole chain) and carries no inset
	constexpr int inset = 3;

	if (R < 1) {
		const std::string message = fmt::format("AMR chain solver: resolution R = {} is below 1", R);
		spdlog::error("{}", message);
		throw std::runtime_error(message);
	}
	if (max_level < 2 || max_level > 4) {
		const std::string message = fmt::format(
			"AMR chain solver: max_level = {} is outside the supported nested range 2..4 (the level-1 geometry is fixed "
			"by the channel anchor and the y/z budgets exhaust below level 4)",
			max_level
		);
		spdlog::error("{}", message);
		throw std::runtime_error(message);
	}

	struct Rect
	{
		std::array<int, 3> lo;
		std::array<int, 3> span;
	};

	// the anchor: footprint rect of the 2-level channel's level-1 footprint
	// in level-1 cells (2x the level-0 region values)
	std::vector<Rect> rects;
	rects.push_back(Rect{{0, 0, 0}, {0, 0, 0}});  // level 0 unused (indexing by level)
	rects.push_back(Rect{{48 * R, 8 * R, 2 * R + 2}, {32 * R, 16 * R, 16 * R}});

	for (int L = 2; L <= max_level; L++) {
		const Rect& parent = rects[L - 1];
		// the child in the parent frame: x/y inset on both faces, z-min
		// chained gap-0 (no inset), z-max inset only
		Rect child;
		child.lo = {{parent.lo[0] + inset, parent.lo[1] + inset, parent.lo[2]}};
		child.span = {{parent.span[0] - 2 * inset, parent.span[1] - 2 * inset, parent.span[2] - inset}};
		// hard floors (the solver's own guard; createAMRBlocks re-checks the
		// full V-suite authoritatively at SimInit)
		for (int a = 0; a < 3; a++) {
			if (child.span[a] < 3) {
				const std::string message = fmt::format(
					"AMR chain solver: level-{} footprint span below the 3-parent-cell minimum on axis {} ({} < 3 at R = {}): "
					"the telescoping budget is exhausted; do not nest this deep",
					L,
					char('x' + a),
					child.span[a],
					R
				);
				spdlog::error("{}", message);
				throw std::runtime_error(message);
			}
		}
		// z-min gap-0 chain consistency (construction invariant, checked so
		// that a broken derivation is attributable to the solver, not to a
		// downstream V10 reject)
		if (child.lo[2] != parent.lo[2]) {
			const std::string message = fmt::format(
				"AMR chain solver: level-{} z-min footprint face is inset {} parent-level cells from its parent's edge "
				"(the wall chain requires gap 0)",
				L,
				child.lo[2] - parent.lo[2]
			);
			spdlog::error("{}", message);
			throw std::runtime_error(message);
		}
		// double the child rect into level-L cells
		for (int a = 0; a < 3; a++) {
			child.lo[a] *= 2;
			child.span[a] *= 2;
		}
		rects.push_back(child);
	}

	AMRChannelChain chain;
	for (int L = 1; L <= max_level; L++) {
		const Rect& rect = rects[L];
		AMRChainLevelGeometry geometry;
		geometry.level = L;
		for (int a = 0; a < 3; a++) {
			// createAMRBlocks' parent-frame conversion inverted: footprint
			// rect in level-L cells is 2 * (value >> (L-1)) per component
			geometry.origin[a] = (rect.lo[a] / 2) << (L - 1);
			geometry.size[a] = (rect.span[a] / 2) << (L - 1);
			geometry.parent_span[a] = rect.span[a] / 2;
			geometry.fine_span[a] = rect.span[a];
		}
		chain.levels.push_back(geometry);
		chain.region_config += fmt::format(
			"{} {} {} {} {} {} {}",
			L,
			geometry.origin[0],
			geometry.origin[1],
			geometry.origin[2],
			geometry.size[0],
			geometry.size[1],
			geometry.size[2]
		);
		if (L < max_level)
			chain.region_config += "\n";
	}

	spdlog::info(
		"AMR chain solver: derived {} nested level(s) on top of the 2-level channel anchor (R = {}, insets >= {} "
		"parent-level cells per non-wall face, z-min wall-chained gap-0 to the level-1 face at level-0 z = {})",
		max_level - 1,
		R,
		inset,
		R + 1
	);
	for (const AMRChainLevelGeometry& geometry : chain.levels)
		spdlog::info(
			"AMR chain solver: level {} origin [{},{},{}] size [{},{},{}] (level-0 coordinates); footprint spans {}x{}x{} "
			"parent / {}x{}x{} fine cells",
			geometry.level,
			geometry.origin[0],
			geometry.origin[1],
			geometry.origin[2],
			geometry.size[0],
			geometry.size[1],
			geometry.size[2],
			geometry.parent_span[0],
			geometry.parent_span[1],
			geometry.parent_span[2],
			geometry.fine_span[0],
			geometry.fine_span[1],
			geometry.fine_span[2]
		);
	spdlog::info("AMR chain solver region spec (reproduces this configuration):\n{}", chain.region_config);

	return chain;
}
