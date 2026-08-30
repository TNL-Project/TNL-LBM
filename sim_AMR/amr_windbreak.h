#pragma once

// Windbreak rod layout and stamping of sim_AMR_channel's nested mode
// (commit G of the amr-nlevel-nesting plan; the user-locked windbreak
// geometry of plan sec. 9): two streamwise rows of thin vertical
// cylindrical posts, staggered by half a pitch, standing on the bottom
// wall plane and stopping below the footprint top (partial height),
// GEO_WALL-stamped on the FINEST level's map only -- every parent level
// treats the rod columns as plain fluid (locked item 1, the sub-grid
// geometry must not appear on the parents).
//
// The layout lives purely in integers of the finest block's local frame:
// axes sit on cell edges (lattice node coordinates, so an even diameter
// gives the symmetric stair-step disc), and a cell belongs to the disc
// iff its center lies within diameter/2 of the axis, i.e. on the
// axis-edge basis (2*dx+1)^2 + (2*dy+1)^2 <= diameter^2. The disc is
// vertical-extruded over `height` rows from local z = -1 -- the
// simulated-band row directly on the wall-chain GEO_WALL row at local
// z = -2, so the rod base sits flush on the wall link plane and every
// base cell bounces back like an interior obstacle in the widened
// substep. The rod top keeps `clearance` cells below the footprint's
// z-max face (partial height by construction).
//
// Guardrails (hard errors, the chain solver's spdlog::error +
// std::runtime_error style): diameter >= 3 (a rod cross-section smaller
// than ~2 cells is forbidden -- bounce-back needs a solid disc blob);
// pitch even and >= diameter (the half-pitch stagger is integer-exact);
// row spacing >= diameter (the two rows stay disjoint); rod cells keep
// >= 8 cells off the footprint's x faces and >= `clearance` cells off
// every other face except the wall plane; both staggered rows must be
// non-empty. The parents' fine-to-coarse windows can read coarse cells
// whose fine subcells are rod-tagged -- locked item 4 makes that a
// physics smear, not a validity issue (the destination-side filtering
// guards the map tags), and the conservation stats keep the constant
// GEO_WALL rod cells as a documented constant offset (locked item 5).
//
// Shared by sim_AMR/sim_AMR_channel.cu (the stamping consumer) and
// tests/unit/test_amr_nesting.cu (the rod-map census test), so the sim and
// the test always run the same arithmetic by construction. Pure
// integers/std types: no TNL or lbm3d dependency.

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <fmt/core.h>
#include <spdlog/spdlog.h>

struct WindbreakRodParams
{
	int diameter = 4;	   // rod disc diameter in finest-level cells (>= 3; even gives the symmetric disc)
	int pitch = 16;		   // rod spacing along y within a row in finest-level cells (even, >= diameter)
	int height = 40;	   // rod height in finest-level z rows above the wall plane (partial)
	int row_spacing = 34;  // streamwise spacing of row 2 behind row 1 in finest-level cells (~2d + wake room)
	int x_first = 32;	   // streamwise axis position of row 1 (the footprint x margins keep >= 8 cells)
	int clearance = 4;	   // hard floor: rod cells keep >= clearance cells off every footprint face except the wall plane
};

struct WindbreakRod
{
	int axis_x = 0;	 // axis lattice coordinate in the block's local frame (the axis sits on the cell edge
					 // between cells axis_x-1 and axis_x, so the disc is symmetric for even diameters)
	int axis_y = 0;
	int row = 0;  // 1 or 2 (row 2 staggers by +pitch/2 along y and +row_spacing along x)
};

struct WindbreakLayout
{
	WindbreakRodParams params;
	std::vector<WindbreakRod> rods;
	std::vector<std::pair<int, int>> disc;	// discretized cross-section: cell offsets (dx, dy) around the axis
	int z_first = -1;						// first tagged z row (the rod base row directly on the wall row)
	int cells_per_rod = 0;					// disc.size() * height
	int cells_total = 0;					// cells_per_rod * rods.size()
};

// the discretized disc cross-section: cell offsets around the axis-edge
// basis (cell centers sit at half-integer offsets from the axis, so a
// cell is in iff its center lies within diameter/2; for diameter = 4
// this is the 12-cell 4x4-minus-corners disc)
inline std::vector<std::pair<int, int>> windbreakDiscOffsets(int diameter)
{
	std::vector<std::pair<int, int>> disc;
	for (int dy = -diameter; dy <= diameter; dy++)
		for (int dx = -diameter; dx <= diameter; dx++) {
			const int sx = 2 * dx + 1;
			const int sy = 2 * dy + 1;
			if (sx * sx + sy * sy <= diameter * diameter)
				disc.emplace_back(dx, dy);
		}
	return disc;
}

// derive the staggered two-row layout on a finest-level block of
// footprint interior spans (local_x, local_y, local_z); throws
// std::runtime_error with a named reason on any guardrail violation
inline WindbreakLayout deriveWindbreakLayout(int local_x, int local_y, int local_z, const WindbreakRodParams& params)
{
	const auto reject = [](const std::string& message)
	{
		spdlog::error("{}", message);
		throw std::runtime_error(message);
	};

	if (params.diameter < 3)
		reject(
			fmt::format(
				"windbreak layout: rod diameter {} is below the 3-cell minimum (a rod cross-section smaller than ~2 cells "
				"cannot hold bounce-back disc structure)",
				params.diameter
			)
		);
	if (params.pitch % 2 != 0 || params.pitch < params.diameter)
		reject(
			fmt::format(
				"windbreak layout: pitch {} must be even (integer-exact half-pitch stagger) and at least the diameter {} "
				"(staggered rods must not merge)",
				params.pitch,
				params.diameter
			)
		);
	if (params.row_spacing < params.diameter)
		reject(
			fmt::format(
				"windbreak layout: row spacing {} must be at least the diameter {} (the two rows must not merge)", params.row_spacing, params.diameter
			)
		);

	WindbreakLayout layout;
	layout.params = params;
	layout.disc = windbreakDiscOffsets(params.diameter);

	// disc extents in cell offsets around the axis
	int dx_min = 0, dx_max = 0, dy_min = 0, dy_max = 0;
	for (const auto& [dx, dy] : layout.disc) {
		dx_min = std::min(dx_min, dx);
		dx_max = std::max(dx_max, dx);
		dy_min = std::min(dy_min, dy);
		dy_max = std::max(dy_max, dy);
	}

	// z extrusion: rows z_first .. z_first + height - 1 with >= clearance
	// cells below the footprint top (partial height)
	if (params.height < 1 || params.height - 2 > local_z - 1 - params.clearance)
		reject(
			fmt::format(
				"windbreak layout: rod height {} exceeds the partial-height budget of the footprint z span {} (need "
				"1 <= h <= {} for >= {} cells of clearance below the z-max face)",
				params.height,
				local_z,
				local_z - params.clearance + 1,
				params.clearance
			)
		);

	// x placement: rod cells keep >= 8 cells off both footprint x faces
	// (row 1 at x_first, row 2 at x_first + row_spacing)
	for (const int axis_x : {params.x_first, params.x_first + params.row_spacing}) {
		if (axis_x + dx_min < 8 || axis_x + dx_max > local_x - 9)
			reject(
				fmt::format(
					"windbreak layout: rod row at axis x = {} leaves a rod cell outside the 8-cell x margin of the "
					"footprint x span [0, {})",
					axis_x,
					local_x
				)
			);
	}

	// y band: admissible axis interval keeps >= clearance cells off both
	// y faces; row 1's rods are centered in the interval, row 2 staggers
	// by +pitch/2 (axes are integer lattice coordinates)
	const int ay_lo = params.clearance - dy_min;
	const int ay_hi = local_y - params.clearance - dy_max - 1;
	if (ay_hi < ay_lo)
		reject(
			fmt::format(
				"windbreak layout: footprint y span {} cannot host a disc of diameter {} with the {}-cell clearance",
				local_y,
				params.diameter,
				params.clearance
			)
		);
	const int n_row1 = 1 + (ay_hi - ay_lo) / params.pitch;
	const int ay_first = ay_lo + ((ay_hi - ay_lo) - (n_row1 - 1) * params.pitch) / 2;
	for (int k = 0; k < n_row1; k++) {
		WindbreakRod rod;
		rod.axis_x = params.x_first;
		rod.axis_y = ay_first + k * params.pitch;
		rod.row = 1;
		layout.rods.push_back(rod);
	}
	int n_row2 = 0;
	for (int ay = ay_first + params.pitch / 2; ay <= ay_hi; ay += params.pitch) {
		WindbreakRod rod;
		rod.axis_x = params.x_first + params.row_spacing;
		rod.axis_y = ay;
		rod.row = 2;
		layout.rods.push_back(rod);
		n_row2++;
	}
	if (n_row2 == 0)
		reject(
			fmt::format(
				"windbreak layout: the half-pitch stagger of row 2 lands outside the admissible y band [{}, {}] (the "
				"footprint y span {} is too narrow for a staggered two-row array at pitch {})",
				ay_lo,
				ay_hi,
				local_y,
				params.pitch
			)
		);

	layout.cells_per_rod = static_cast<int>(layout.disc.size()) * params.height;
	layout.cells_total = layout.cells_per_rod * static_cast<int>(layout.rods.size());
	return layout;
}

// stamp the layout: calls stamp(local_x, local_y, local_z) once per
// tagged cell (the sim and the census test pass their own hmap writer)
template <typename Stamp>
inline void stampWindbreak(const WindbreakLayout& layout, Stamp&& stamp)
{
	for (const WindbreakRod& rod : layout.rods)
		for (int r = 0; r < layout.params.height; r++)
			for (const auto& [dx, dy] : layout.disc)
				stamp(rod.axis_x + dx, rod.axis_y + dy, layout.z_first + r);
}

// the quiet census: one line per rod (axis, diameter, height, tagged
// cell count) plus one total line
inline void logWindbreakLayout(const WindbreakLayout& layout, int block_id)
{
	for (const WindbreakRod& rod : layout.rods)
		spdlog::info(
			"windbreak rod (block {}, row {}): axis at local lattice (x={}, y={}), d = {} cells, h = {} rows, {} "
			"tagged cells",
			block_id,
			rod.row,
			rod.axis_x,
			rod.axis_y,
			layout.params.diameter,
			layout.params.height,
			layout.cells_per_rod
		);
	spdlog::info(
		"windbreak census (block {}): {} rods, {} tagged cells total on the finest-level map", block_id, layout.rods.size(), layout.cells_total
	);
}
