#pragma once

// Inflow openings on an LBM lattice: the sim-side state is fully managed by
// the universal free functions in this header (addInflowPlane,
// finalizeInflowOpenings); the simulation's StateLocal just owns an
// InflowOpeningsState and calls them from its setupBoundaries() override, in
// the order: stamp claims (add*), finish all other setBoundary* stamps, then
// finalizeInflowOpenings() to carve the outward ghost layers, validate the
// map, discover unclaimed inflow planes, resolve the precomputed velocities
// and publish everything into the kernel DATA.
//
// Device-facing contract (see NSE_Data_OpeningInflow in lbm_data.h): the
// kernel reads a per-cell site id (>= 0) from the claim map and the imposed
// velocity from one compressed array indexed by it - computed entirely on the
// host, so the kernel never reads an opening record and never normalizes a
// profile.

#include <limits>
#include <vector>

#include <fmt/core.h>
#include <spdlog/spdlog.h>

#include "defs.h"
#include "lbm.h"
#include "lbm_data.h"
#include "inflow_openings.h"

// Sim-side owner for everything the inflow openings need to outlive the
// setupBoundaries() call: the opening records (host), the discovery gating
// flag, and the replicated compressed velocity array whose device pointer is
// published into every block's kernel DATA by finalizeInflowOpenings(). None
// of this is kernel-argument space.
template <typename TRAITS>
struct InflowOpeningsState
{
	using dreal = typename TRAITS::dreal;

	std::vector<LBM_INFLOW_OPENING<TRAITS>> openings;
	bool discover = true;

	TNL::Containers::Array<dreal, TNL::Devices::Host> hvelocities;
	TNL::Containers::Array<dreal, DeviceType> dvelocities;
};

// Registers one authored inflow opening on the plane (axis, sign,
// planeOffset) and immediately claims its rect's cells on the host: stamps
// hmap (GEO_INFLOW_MOMENT) and the claim map (plain opening id; the resolve
// pass later re-stamps live cells with their compressed-velocity site ids)
// in every local block, last-write-wins exactly like setBoundary*.
// The explicit (axis, sign) is the plane's authoritative provenance: it
// drives the outward ghost carve in finalizeInflowOpenings and is what the
// detectBCFace mirror validation checks - it is mandatory precisely for
// interior planes (a voxelized pipe inlet away from the bounding box), where
// the flow direction cannot be inferred from the neighbors (see
// docs/moment-bc-derivation.md, "Boundary contract").
// amplitude semantics depend on the profile: UNIFORM applies the amplitude
// itself as the imposed inward-normal velocity (record value verbatim,
// negative sign pointing out of the domain), PARABOLIC treats the amplitude
// as the target volumetric flux across the opening - the bake in
// finalizeInflowOpenings resolves it into the site velocities.
// Throws std::invalid_argument on an invalid plane (bad axis/sign,
// non-planar rect, planeOffset outside the domain) or an empty clamped rect.
template <typename CONFIG>
LBM_INFLOW_OPENING<typename CONFIG::TRAITS>& addInflowPlane(
	LBM<CONFIG>& nse,
	InflowOpeningsState<typename CONFIG::TRAITS>& os,
	short axis,
	short sign,
	typename CONFIG::TRAITS::idx planeOffset,
	typename CONFIG::TRAITS::idx3d lo,
	typename CONFIG::TRAITS::idx3d hi,
	ProfileType profile = ProfileType::UNIFORM,
	typename CONFIG::TRAITS::dreal amplitude = 0
)
{
	static_assert(has_inflow_openings_v<typename CONFIG::DATA>, "addInflowPlane requires an openings-capable DATA struct (NSE_Data_OpeningInflow)");

	using TRAITS = typename CONFIG::TRAITS;
	using idx = typename TRAITS::idx;
	using idx3d = typename TRAITS::idx3d;
	using dreal = typename TRAITS::dreal;

	if (axis < 0 || axis > 2)
		throw std::invalid_argument(fmt::format("addInflowPlane: axis {} is outside [0, 2]", axis));
	if (sign != -1 && sign != 1)
		throw std::invalid_argument(fmt::format("addInflowPlane: sign {} is neither -1 nor +1", sign));
	if (planeOffset < 0 || planeOffset >= nse.lat.global[axis])
		throw std::invalid_argument(
			fmt::format("addInflowPlane: planeOffset {} is outside the global extent [0, {}) on axis {}", planeOffset, nse.lat.global[axis], axis)
		);
	// v1: authored claims cover single-plane rects only
	if (lo[axis] != planeOffset || hi[axis] != planeOffset)
		throw std::invalid_argument(
			fmt::format(
				"addInflowPlane: lo/hi on the plane axis must equal planeOffset {} (single-plane rect), got {}/{}", planeOffset, lo[axis], hi[axis]
			)
		);

	// clamp the tangential rect to the global domain
	for (int d = 0; d < 3; d++) {
		if (d == axis)
			continue;
		if (lo[d] < 0)
			lo[d] = 0;
		if (hi[d] >= nse.lat.global[d])
			hi[d] = nse.lat.global[d] - 1;
		if (lo[d] > hi[d])
			throw std::invalid_argument(fmt::format("addInflowPlane: empty rect after clamping on axis {} ({} > {})", d, lo[d], hi[d]));
	}

	LBM_INFLOW_OPENING<TRAITS> opening;
	opening.id = int(os.openings.size());
	opening.axis = axis;
	opening.sign = sign;
	opening.planeOffset = planeOffset;	// the record keeps the GLOBAL plane coordinates
	opening.lo = lo;
	opening.hi = hi;
	opening.profile = profile;
	opening.amplitude = amplitude;
	opening.scale = 1;
	opening.origin = OpeningOrigin::AUTHORED;

	// claim stamp holds a plain id; anything >= 4096 is a programming error
	// (the sites are also bounded by the 32-bit map but the cap keeps the
	// replicated book-keeping small)
	if (os.openings.size() >= 4096)
		throw std::invalid_argument(fmt::format("addInflowPlane: {} inflow openings reach the claim id limit of 4096", os.openings.size()));
	os.openings.push_back(opening);

	// claim the cells on the host: per-block intersection of the global rect
	// with the block's owned range in GLOBAL coordinates (setBoundary* loop
	// arithmetic); the setter's isLocalIndex guard is the exact ownership test.
	// Later calls overwrite earlier claims on overlap (last-write-wins).
	for (auto& block : nse.blocks) {
		idx3d begin = lo, end = hi;
		for (int d = 0; d < 3; d++) {
			if (begin[d] < block.offset[d])
				begin[d] = block.offset[d];
			if (end[d] >= block.offset[d] + block.local[d])
				end[d] = block.offset[d] + block.local[d] - 1;
		}
		for (idx gx = begin.x(); gx <= end.x(); gx++)
			for (idx gy = begin.y(); gy <= end.y(); gy++)
				for (idx gz = begin.z(); gz <= end.z(); gz++) {
					block.setMap(gx, gy, gz, CONFIG::BC::GEO_INFLOW_MOMENT);
					block.setInflowOpeningMap(gx, gy, gz, opening.id);
				}
	}

	return os.openings.back();
}

// Everything the add* calls deferred: (1) carved the outward layer of every
// authored opening into GEO_NOTHING (walls yield; anything else throws),
// (2) validated the boundary contract on the final map by mirroring the
// runtime detectBCFace per claimed cell (interior inward, ghost/outward,
// detected face == plane face), (3) enumerated the remaining inflow planes,
// gathered the plane pictures, labeled 4-connected components of the unclaimed
// cells and appended them as DISCOVERED openings (gated by os.discover), then
// (4) resolved the precomputed site velocities - UNIFORM imposes the authored
// amplitude along the inward normal (base components elsewhere), PARABOLIC
// imposes base * scale * profileWeight with scale = amplitude / sum of the
// profile weights of its cells (computed on the replicated plane pictures, so
// every rank produces bitwise-identical arrays without a collective) - and
// re-stamped the claim map with the site ids and published map/velocities/
// count into every block's kernel DATA.
template <typename CONFIG>
void finalizeInflowOpenings(LBM<CONFIG>& nse, InflowOpeningsState<typename CONFIG::TRAITS>& os)
{
	using TRAITS = typename CONFIG::TRAITS;
	using idx = typename TRAITS::idx;
	using idx3d = typename TRAITS::idx3d;
	using dreal = typename TRAITS::dreal;
	using map_t = typename TRAITS::map_t;
	using BC = typename CONFIG::BC;

	const idx3d global = nse.lat.global;
	constexpr map_t INFLOW_TAG = BC::GEO_INFLOW_MOMENT;
	using InflowOpening = LBM_INFLOW_OPENING<TRAITS>;

	// drop DISCOVERED records on repeated calls - they are re-derived below
	std::size_t keep = 0;
	for (const auto& o : os.openings)
		if (o.origin == OpeningOrigin::AUTHORED)
			os.openings[keep++] = o;
	os.openings.resize(keep);

	// (1) outward-layer carve: the boundary contract requires an interior
	// (fluid/symmetry) inward side and a GEO_NOTHING outward layer owning the
	// runtime face detection; anything else on the outward layer throws
	if (! os.openings.empty()) {
		int violation = 0;
		idx3d violation_pos = 0;
		short violation_axis = 0, violation_sign = 0;
		idx violation_plane = 0;
		int violation_tag = -1;
		for (const auto& opening : os.openings) {
			const idx outward = opening.planeOffset + opening.sign;
			if (outward < 0 || outward >= global[opening.axis])
				continue;  // bounding plane: the domain edge is the outward side
			for (const auto& block : nse.blocks) {
				idx3d begin = opening.lo, end = opening.hi;
				for (int d = 0; d < 3; d++) {
					begin[d] = TNL::max(begin[d], block.offset[d]);
					end[d] = TNL::min(end[d], block.offset[d] + block.local[d] - 1);
				}
				begin[opening.axis] = end[opening.axis] = outward;
				for (idx gx = begin.x(); gx <= end.x(); gx++)
					for (idx gy = begin.y(); gy <= end.y(); gy++)
						for (idx gz = begin.z(); gz <= end.z(); gz++) {
							const map_t tag = block.hmap(gx, gy, gz);
							const int claim = block.hinflow_opening_map(gx, gy, gz);
							if ((tag == BC::GEO_WALL || tag == BC::GEO_NOTHING) && claim < 0)
								continue;  // carveable: wall stamps yield, ghost layers are idempotent
							if (violation)
								continue;  // one violation example per rank is enough
							violation = 1;
							violation_pos = idx3d{gx, gy, gz};
							violation_axis = opening.axis;
							violation_sign = opening.sign;
							violation_plane = opening.planeOffset;
							violation_tag = tag;
						}
			}
		}
		const bool local_carve_example = violation != 0;
#ifdef HAVE_MPI
		// settle the verdict on all ranks before any throw, so no rank blocks
		// in a later collective while a sibling aborts
		MPI_Allreduce(MPI_IN_PLACE, &violation, 1, MPI_INT, MPI_BOR, nse.communicator);
#endif
		if (violation) {
			const char* rule = "an inflow opening's outward layer may only contain walls or ghost layers; interior tags, boundary stamps, or "
							   "overlapping claims on the outward side mean the plane is not a boundary of the fluid domain "
							   "(see AGENTS.md interior-planes convention)";
			if (local_carve_example)
				throw std::runtime_error(
					fmt::format(
						"finalizeInflowOpenings: authored opening (axis={}, sign={:+d}, planeOffset={}) cannot carve its outward layer: "
						"cell ({},{},{}) carries boundary tag {}: {}",
						violation_axis,
						violation_sign,
						violation_plane,
						violation_pos.x(),
						violation_pos.y(),
						violation_pos.z(),
						violation_tag,
						rule
					)
				);
			throw std::runtime_error(
				fmt::format("finalizeInflowOpenings: an authored opening's outward layer on another rank cannot be carved: {}", rule)
			);
		}

		// carve pass (no throws below): converts the outward wall layer into
		// the one-cell GEO_NOTHING ghost the runtime detector reads
		for (const auto& opening : os.openings) {
			const idx outward = opening.planeOffset + opening.sign;
			if (outward < 0 || outward >= global[opening.axis])
				continue;
			for (auto& block : nse.blocks) {
				idx3d begin = opening.lo, end = opening.hi;
				for (int d = 0; d < 3; d++) {
					begin[d] = TNL::max(begin[d], block.offset[d]);
					end[d] = TNL::min(end[d], block.offset[d] + block.local[d] - 1);
				}
				begin[opening.axis] = end[opening.axis] = outward;
				for (idx gx = begin.x(); gx <= end.x(); gx++)
					for (idx gy = begin.y(); gy <= end.y(); gy++)
						for (idx gz = begin.z(); gz <= end.z(); gz++)
							block.setMap(gx, gy, gz, BC::GEO_NOTHING);
			}
		}

		// (2) detector-mirror boundary contract: reproduces the runtime
		// detectBCFace on the host (first interior side in the kernel's
		// fixed order, face = its opposite) for every live claimed cell
		// and requires (i) an interior inward neighbor, (ii) a GEO_NOTHING
		// (or out-of-domain, for bounding planes) outward neighbor, and
		// (iii) the detected face equal to the plane's face bit
		short mirror_axis[6] = {0, 0, 1, 1, 2, 2};
		short mirror_dir[6] = {-1, 1, -1, 1, -1, 1};
		violation = 0;
		int violation_inward_tag = -1, violation_outward_tag = -1;	// -1 = outside the domain
		int violation_detected = -1, violation_expected = -1;
		for (const auto& block : nse.blocks) {
			for (idx lz = 0; lz < block.local.z(); lz++)
				for (idx ly = 0; ly < block.local.y(); ly++)
					for (idx lx = 0; lx < block.local.x(); lx++) {
						idx3d g = 0;
						g[0] = block.offset.x() + lx;
						g[1] = block.offset.y() + ly;
						g[2] = block.offset.z() + lz;
						const int claim = block.hinflow_opening_map(g[0], g[1], g[2]);
						if (claim < 0)
							continue;  // no authored claim (discovery stamps nothing before this point)
						if (block.hmap(g[0], g[1], g[2]) != INFLOW_TAG)
							continue;  // dead claim: a later setBoundary* stamp won the cell
						const auto& opening = os.openings[static_cast<std::size_t>(claim)];

						int tags[6];
						bool interior[6];
						for (int k = 0; k < 6; k++) {
							const idx hn = g[mirror_axis[k]] + mirror_dir[k];
							tags[k] = -1;
							if (hn >= 0 && hn < global[mirror_axis[k]]) {
								idx3d gn = g;
								gn[mirror_axis[k]] = hn;
								tags[k] = block.hmap(gn[0], gn[1], gn[2]);
							}
							interior[k] = tags[k] >= 0 && BC::isOutflowInterior(static_cast<map_t>(tags[k]));
						}
						int detected = -1;
						for (int k = 0; k < 6 && detected < 0; k++)
							if (interior[k])
								detected = 1 << (2 * mirror_axis[k] + (mirror_dir[k] > 0 ? 1 : 0));
						const int expected = 1 << (2 * opening.axis + (opening.sign > 0 ? 0 : 1));

						// inward side: (axis, -sign) neighbor must be interior
						const int inward_k = 2 * opening.axis + (opening.sign < 0 ? 1 : 0);
						const bool inward_ok = interior[inward_k];
						// outward side: (axis, +sign) neighbor must be the ghost layer (or the domain edge)
						const int outward_k = 2 * opening.axis + (opening.sign > 0 ? 1 : 0);
						const bool outward_ok = tags[outward_k] < 0 || tags[outward_k] == BC::GEO_NOTHING;

						if (inward_ok && outward_ok && detected == expected)
							continue;
						if (violation)
							continue;  // one violation example per rank is enough
						violation = 1;
						violation_pos = g;
						violation_axis = opening.axis;
						violation_sign = opening.sign;
						violation_plane = opening.planeOffset;
						violation_inward_tag = tags[inward_k];
						violation_outward_tag = tags[outward_k];
						violation_detected = detected;
						violation_expected = expected;
					}
		}
		const bool local_example = violation != 0;
#ifdef HAVE_MPI
		MPI_Allreduce(MPI_IN_PLACE, &violation, 1, MPI_INT, MPI_BOR, nse.communicator);
#endif
		if (violation) {
			const char* rule = "inflow BC cells must have an interior (fluid/symmetry) inward neighbor and a GEO_NOTHING (or out-of-domain) "
							   "outward neighbor, and runtime face detection must resolve the cell to the opening's own (axis, sign) "
							   "(see AGENTS.md interior-planes convention)";
			if (local_example)
				throw std::runtime_error(
					fmt::format(
						"finalizeInflowOpenings: authored opening (axis={}, sign={:+d}, planeOffset={}) claims cell ({},{},{}) whose "
						"neighborhood violates the inflow boundary contract (inward tag {}, outward tag {}, detected face {:#06x}, "
						"expected {:#06x}; -1 means outside the domain): {}",
						violation_axis,
						violation_sign,
						violation_plane,
						violation_pos.x(),
						violation_pos.y(),
						violation_pos.z(),
						violation_inward_tag,
						violation_outward_tag,
						violation_detected,
						violation_expected,
						rule
					)
				);
			throw std::runtime_error(
				fmt::format("finalizeInflowOpenings: an authored opening claim on another rank violates the inflow boundary contract: {}", rule)
			);
		}
	}

	// (3) plane enumeration + per-plane gather/label; the per-plane
	// workspace (replicated plane picture + component labels) is what the
	// resolve pass consumes, so it outlives the loop
	struct PlaneWork
	{
		short axis, sign;
		idx plane;
		int authored = 0;
		int discovered = 0;
		int authored_cells = 0;
		int discovered_cells = 0;
		idx U = 0, V = 0;
		std::vector<int> plane_tc;
		std::vector<int> comp_ids;
		int base_id = 0;
	};
	std::vector<PlaneWork> planes;

	struct PlaneKey
	{
		short axis, sign;
		idx plane;
	};
	std::vector<PlaneKey> authored_planes;
	for (const auto& opening : os.openings) {
		for (const auto& s : authored_planes)
			if (s.axis == opening.axis && s.plane == opening.planeOffset && s.sign != opening.sign)
				throw std::runtime_error(
					fmt::format(
						"finalizeInflowOpenings: authored openings on the same plane (axis={}, planeOffset={}) disagree on sign "
						"({} vs {}); one plane must have a single (axis, sign) direction",
						opening.axis,
						opening.planeOffset,
						s.sign,
						opening.sign
					)
				);
		bool known = false;
		for (const auto& s : authored_planes)
			if (s.axis == opening.axis && s.sign == opening.sign && s.plane == opening.planeOffset)
				known = true;
		if (! known)
			authored_planes.push_back({opening.axis, opening.sign, opening.planeOffset});
	}

	// single sweep over owned inflow-tagged cells: boundary-face existence
	// (bits 0-5 in the fixed XP,XM,YP,YM,ZP,ZM order), any tagged cell
	// (bit 6) and the interior-provenance check (bit 7 = violation)
	int local_mask = 0;
	idx3d violation_pos = 0;
	for (const auto& block : nse.blocks) {
		for (idx lz = 0; lz < block.local.z(); lz++)
			for (idx ly = 0; ly < block.local.y(); ly++)
				for (idx lx = 0; lx < block.local.x(); lx++) {
					idx3d g = 0;
					g[0] = block.offset.x() + lx;
					g[1] = block.offset.y() + ly;
					g[2] = block.offset.z() + lz;
					if (block.hmap(g[0], g[1], g[2]) != INFLOW_TAG)
						continue;
					local_mask |= 1 << 6;
					// tagged cell ON a bounding plane; degenerate axes
					// (global <= 1, e.g. z in D2Q9) carry no real face
					if (global.x() > 1) {
						if (g[0] == 0)
							local_mask |= 1 << 1;  // XM
						if (g[0] == global.x() - 1)
							local_mask |= 1 << 0;  // XP
					}
					if (global.y() > 1) {
						if (g[1] == 0)
							local_mask |= 1 << 3;  // YM
						if (g[1] == global.y() - 1)
							local_mask |= 1 << 2;  // YP
					}
					if (global.z() > 1) {
						if (g[2] == 0)
							local_mask |= 1 << 5;  // ZM
						if (g[2] == global.z() - 1)
							local_mask |= 1 << 4;  // ZP
					}
					if (block.hinflow_opening_map(g[0], g[1], g[2]) != -1)
						continue;  // authored claims are exempt from the provenance check
					if (local_mask & (1 << 7))
						continue;  // one violation example is enough
					// unclaimed: covered when it intersects an authored plane
					bool covered = false;
					for (const auto& s : authored_planes)
						if (s.plane == g[s.axis])
							covered = true;
					if (covered)
						continue;
					local_mask |= 1 << 7;
					violation_pos = g;
				}
	}

#ifdef HAVE_MPI
	// merge the face-existence bits (OR) and keep also whether ANY rank holds
	// the provenance violation facially: both fit the same collective
	MPI_Allreduce(MPI_IN_PLACE, &local_mask, 1, MPI_INT, MPI_BOR, nse.communicator);
#endif
	if (local_mask & (1 << 7))
		throw std::runtime_error(
			fmt::format(
				"finalizeInflowOpenings: inflow BC cells at ({},{},{}) sit on no bounding face and no authored plane - "
				"interior inflow planes need explicit provenance via addInflowPlane "
				"(or set the sim state's InflowOpeningsState.discover flag to false to keep the legacy uniform inflow)",
				violation_pos.x(),
				violation_pos.y(),
				violation_pos.z()
			)
		);

	// nothing tagged anywhere: the legacy uniform inflow path stays untouched
	if (! (local_mask & (1 << 6)) && os.openings.empty())
		return;

	// bounding-face planes in the fixed XP,XM,YP,YM,ZP,ZM bit order (bit
	// value XP..ZM); degenerate axes never set their bits above
	for (int bit = 0; bit < 6; bit++)
		if (local_mask & (1 << bit)) {
			// face bit order XP,XM,YP,YM,ZP,ZM -> axis = bit/2, sign = +1 for even
			const short axis = static_cast<short>(bit / 2);
			const short sign = (bit % 2 == 0) ? 1 : -1;
			const idx plane = (sign > 0) ? global[axis] - 1 : 0;
			planes.push_back({axis, sign, plane, 0, 0, 0, 0, 0, 0, {}, {}, 0});
		}

	// authored planes appended sorted by (axis, sign, plane)
	std::sort(
		authored_planes.begin(),
		authored_planes.end(),
		[](const PlaneKey& A, const PlaneKey& B)
		{
			if (A.axis != B.axis)
				return A.axis < B.axis;
			if (A.sign != B.sign)
				return A.sign < B.sign;
			return A.plane < B.plane;
		}
	);
	for (const auto& s : authored_planes) {
		bool duplicate = false;
		for (const auto& w : planes)
			if (w.axis == s.axis && w.sign == s.sign && w.plane == s.plane)
				duplicate = true;
		if (! duplicate)
			planes.push_back({s.axis, s.sign, s.plane, 0, 0, 0, 0, 0, 0, {}, {}, 0});
	}
	for (auto& w : planes)
		for (const auto& opening : os.openings)
			if (opening.axis == w.axis && opening.sign == w.sign && opening.planeOffset == w.plane)
				w.authored++;

	// tangential axes per plane axis, ascending axis order (u is the faster
	// grid index on the plane)
	const short u_axis[3] = {1, 0, 0};
	const short v_axis[3] = {2, 2, 1};

	for (auto& w : planes) {
		const short ua = u_axis[w.axis], va = v_axis[w.axis];
		w.U = global[ua];
		w.V = global[va];
		const idx U = w.U, V = w.V;

		// pack owned tagged cells of the plane as (u, v, tc) triplets;
		// tc = 1 for tagged-unclaimed, 2 + id for authored claims
		std::vector<int> triplets;
		for (const auto& block : nse.blocks) {
			if (w.plane < block.offset[w.axis] || w.plane >= block.offset[w.axis] + block.local[w.axis])
				continue;
			for (idx lv = 0; lv < block.local[va]; lv++)
				for (idx lu = 0; lu < block.local[ua]; lu++) {
					idx3d g = 0;
					g[w.axis] = w.plane;
					g[ua] = block.offset[ua] + lu;
					g[va] = block.offset[va] + lv;
					if (block.hmap(g[0], g[1], g[2]) != INFLOW_TAG)
						continue;
					const int claim = block.hinflow_opening_map(g[0], g[1], g[2]);
					triplets.push_back(static_cast<int>(g[ua]));
					triplets.push_back(static_cast<int>(g[va]));
					// tagAndClaim widened to int: one payload type, one Allgatherv
					triplets.push_back(claim < 0 ? 1 : 2 + claim);
				}
		}

		std::vector<int> all_triplets;
#ifdef HAVE_MPI
		// rebuild the identical global plane picture on every rank: counts
		// first, then one Allgatherv of the local slices
		std::vector<int> counts(nse.nproc), displs(nse.nproc);
		int my_count = static_cast<int>(triplets.size());
		MPI_Allgather(&my_count, 1, MPI_INT, counts.data(), 1, MPI_INT, nse.communicator);
		int total = 0;
		for (int r = 0; r < nse.nproc; r++) {
			displs[r] = total;
			total += counts[r];
		}
		all_triplets.resize(total);
		MPI_Allgatherv(triplets.data(), my_count, MPI_INT, all_triplets.data(), counts.data(), displs.data(), MPI_INT, nse.communicator);
#else
		all_triplets = std::move(triplets);
#endif

		// untagged cells stay 0; duplicate (u,v) entries cannot occur because
		// cell ownership is disjoint across ranks
		w.plane_tc.assign(static_cast<std::size_t>(U) * static_cast<std::size_t>(V), 0);
		for (std::size_t i = 0; i < all_triplets.size(); i += 3) {
			const std::size_t uv =
				static_cast<std::size_t>(all_triplets[i + 1]) * static_cast<std::size_t>(U) + static_cast<std::size_t>(all_triplets[i]);
			w.plane_tc[uv] = all_triplets[i + 2];
		}
		for (const int tc : w.plane_tc) {
			if (tc == 1)
				w.discovered_cells++;
			else if (tc >= 2)
				w.authored_cells++;
		}

		if (! os.discover)
			continue;

		// candidates = tagged cells claimed by no authored opening
		auto isCandidate = [&w, U](int u, int v)
		{
			return w.plane_tc[static_cast<std::size_t>(v) * static_cast<std::size_t>(U) + static_cast<std::size_t>(u)] == 1;
		};
		auto components = labelComponents2D(static_cast<int>(U), static_cast<int>(V), isCandidate);
		w.comp_ids = std::move(components.first);
		const auto& comp_rects = components.second;

		// claim ids are plain 32-bit values; all inputs are replicated
		// identically, so the cap verdict is identical on every rank
		// (the throw follows Allgatherv, no rank hangs)
		const std::size_t would = os.openings.size() + comp_rects.size();
		if (would >= 4096)
			throw std::runtime_error(
				fmt::format(
					"finalizeInflowOpenings: plane (axis={} sign={:+d} offset={}) adds {} discovered openings to {} existing ones, "
					"reaching {} and exceeding the claim id limit of 4096",
					w.axis,
					w.sign,
					w.plane,
					comp_rects.size(),
					os.openings.size(),
					would
				)
			);

		// append DISCOVERED openings: the components already ascend in
		// row-major order, so ids continue deterministically
		for (const auto& rect : comp_rects) {
			InflowOpening opening;
			opening.id = static_cast<int>(os.openings.size());
			opening.axis = w.axis;
			opening.sign = w.sign;
			opening.planeOffset = w.plane;
			opening.lo[w.axis] = opening.hi[w.axis] = w.plane;
			opening.lo[ua] = rect.lo_u;
			opening.hi[ua] = rect.hi_u;
			opening.lo[va] = rect.lo_v;
			opening.hi[va] = rect.hi_v;
			opening.profile = ProfileType::UNIFORM;
			opening.amplitude = 0;
			opening.scale = 1;
			opening.origin = OpeningOrigin::DISCOVERED;
			os.openings.push_back(opening);
			w.discovered++;
		}
		w.base_id = static_cast<int>(os.openings.size()) - w.discovered;

		// stamp the discovered claims (complement only: claimed cells were
		// never candidates); the setter's isLocalIndex guard writes only the
		// share owned by this rank, hmap is NOT touched. Discovered planes lie
		// on bounding planes, so their outward side is outside the domain.
		for (idx v = 0; v < V; v++)
			for (idx u = 0; u < U; u++) {
				const int c = w.comp_ids[static_cast<std::size_t>(v) * static_cast<std::size_t>(U) + static_cast<std::size_t>(u)];
				if (c < 0)
					continue;
				idx3d g = 0;
				g[w.axis] = w.plane;
				g[ua] = u;
				g[va] = v;
				for (auto& block : nse.blocks)
					block.setInflowOpeningMap(g[0], g[1], g[2], w.base_id + c);
			}
	}

	// R5: init report, once per run (legacy runs have no plane at all and
	// print nothing)
	if (nse.rank == 0) {
		bool any = false;
		for (const auto& w : planes)
			if (w.authored > 0 || w.discovered > 0)
				any = true;
		if (any) {
			spdlog::info("inflow openings: per-plane discovery report");
			for (const auto& w : planes) {
				if (w.authored == 0 && w.discovered == 0)
					continue;
				spdlog::info(
					"  plane (axis={} sign={:+d} offset={}): authored={} discovered={} cells={} ({} authored, {} discovered)",
					w.axis,
					w.sign,
					w.plane,
					w.authored,
					w.discovered,
					w.authored_cells + w.discovered_cells,
					w.authored_cells,
					w.discovered_cells
				);
			}
		}
	}

	// (4) resolve the per-opening scales (PARABOLIC only) on the replicated
	// plane pictures; every rank computes the identical sums without a
	// collective
	for (auto& opening : os.openings) {
		opening.scale = 1;
		if (opening.profile != ProfileType::PARABOLIC)
			continue;
		for (const auto& w : planes) {
			if (w.axis != opening.axis || w.sign != opening.sign || w.plane != opening.planeOffset)
				continue;
			const short ua = u_axis[w.axis], va = v_axis[w.axis];
			double sum_w = 0;
			for (idx v = 0; v < w.V; v++)
				for (idx u = 0; u < w.U; u++) {
					const int tc = w.plane_tc[static_cast<std::size_t>(v) * static_cast<std::size_t>(w.U) + static_cast<std::size_t>(u)];
					if (tc < 2 || tc - 2 != opening.id)
						continue;
					sum_w += openingProfileWeight<idx, double>(
						opening.profile,
						static_cast<double>(opening.lo[ua]),
						static_cast<double>(opening.hi[ua]),
						static_cast<double>(opening.lo[va]),
						static_cast<double>(opening.hi[va]),
						static_cast<double>(u),
						static_cast<double>(v)
					);
				}
			// a degenerate (fully overwritten) claim rect has sum_w == 0 and
			// gets a zero scale
			opening.scale = sum_w != 0 ? static_cast<dreal>(static_cast<double>(opening.amplitude) / sum_w) : dreal(0);
		}
	}

	// base imposed velocity (DATA defaults, replicated on every rank)
	dreal base[3] = {0, 0, 0};
	if (! nse.blocks.empty()) {
		base[0] = nse.blocks.front().data.inflow_vx;
		base[1] = nse.blocks.front().data.inflow_vy;
		base[2] = nse.blocks.front().data.inflow_vz;
	}

	// site numbering: plane order (bounding bits then sorted authored
	// planes), cells row-major within each plane - replicated everywhere
	int sites = 0;
	std::vector<std::vector<int>> plane_site;
	plane_site.resize(planes.size());
	for (std::size_t wi = 0; wi < planes.size(); wi++) {
		const auto& w = planes[wi];
		plane_site[wi].assign(static_cast<std::size_t>(w.U) * static_cast<std::size_t>(w.V), -1);
		for (idx v = 0; v < w.V; v++)
			for (idx u = 0; u < w.U; u++) {
				const std::size_t uv = static_cast<std::size_t>(v) * static_cast<std::size_t>(w.U) + static_cast<std::size_t>(u);
				if (w.plane_tc[uv] == 0)
					continue;
				plane_site[wi][uv] = sites++;
			}
	}

	// overflow guard on the SETUP side: the kernel indexes the compressed
	// table as site * D + c in int arithmetic and the claim map stores the
	// site id as int, so the number of claimed sites must fit the int domain
	// with a factor of D headroom (unreachable on any allocatable lattice,
	// but this is the only place a violation can be diagnosed cleanly)
	if (static_cast<long long>(sites) * CONFIG::D > std::numeric_limits<int>::max())
		throw std::runtime_error(
			fmt::format(
				"finalizeInflowOpenings: {} claimed sites overflow the int site indexing (the limit is {} at D={})",
				sites,
				std::numeric_limits<int>::max() / CONFIG::D,
				CONFIG::D
			)
		);

	// bake the resolved velocity per site (component-major compressed layout
	// site * D + c: UNIFORM imposes the authored amplitude along the inward
	// normal and the DATA defaults elsewhere, PARABOLIC scales the defaults
	// by scale * profile weight)
	os.hvelocities.setSize(static_cast<std::size_t>(sites) * static_cast<std::size_t>(CONFIG::D));
	for (std::size_t wi = 0; wi < planes.size(); wi++) {
		const auto& w = planes[wi];
		const short ua = u_axis[w.axis], va = v_axis[w.axis];
		for (idx v = 0; v < w.V; v++)
			for (idx u = 0; u < w.U; u++) {
				const std::size_t uv = static_cast<std::size_t>(v) * static_cast<std::size_t>(w.U) + static_cast<std::size_t>(u);
				const int site = plane_site[wi][uv];
				if (site < 0)
					continue;
				const int tc = w.plane_tc[uv];
				const int oid = (tc >= 2) ? tc - 2 : w.base_id + w.comp_ids[uv];
				const InflowOpening& opening = os.openings[static_cast<std::size_t>(oid)];
				dreal vel[3] = {base[0], base[1], base[2]};
				if (opening.profile == ProfileType::UNIFORM) {
					if (opening.amplitude != 0)
						vel[opening.axis] = static_cast<dreal>(-opening.sign * static_cast<double>(opening.amplitude));
				}
				else {
					const double weight = openingProfileWeight<idx, double>(
						opening.profile,
						static_cast<double>(opening.lo[ua]),
						static_cast<double>(opening.hi[ua]),
						static_cast<double>(opening.lo[va]),
						static_cast<double>(opening.hi[va]),
						static_cast<double>(u),
						static_cast<double>(v)
					);
					for (int c = 0; c < 3; c++)
						vel[c] = static_cast<dreal>(static_cast<double>(base[c]) * static_cast<double>(opening.scale) * weight);
				}
				for (int c = 0; c < CONFIG::D; c++)
					os.hvelocities[static_cast<std::size_t>(site) * CONFIG::D + static_cast<std::size_t>(c)] = vel[c];
			}
	}

	// re-stamp the claim map with the site ids (per-block ownership already
	// covers the shared claim setter's guard)
	for (std::size_t wi = 0; wi < planes.size(); wi++) {
		const auto& w = planes[wi];
		const short ua = u_axis[w.axis], va = v_axis[w.axis];
		for (idx v = 0; v < w.V; v++)
			for (idx u = 0; u < w.U; u++) {
				const int site = plane_site[wi][static_cast<std::size_t>(v) * static_cast<std::size_t>(w.U) + static_cast<std::size_t>(u)];
				if (site < 0)
					continue;
				idx3d g = 0;
				g[w.axis] = w.plane;
				g[ua] = u;
				g[va] = v;
				for (auto& block : nse.blocks)
					block.setInflowOpeningMap(g[0], g[1], g[2], site);
			}
	}

	// upload and publish into every block's kernel DATA (with zero claimed
	// sites the empty array publishes a null pointer, which the inflow() hook
	// treats as the legacy uniform path)
	os.dvelocities = os.hvelocities;
	for (auto& block : nse.blocks)
		block.data.inflow_opening_velocities = os.dvelocities.getData();
}
