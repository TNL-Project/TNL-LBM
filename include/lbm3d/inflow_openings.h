#pragma once

#include <utility>
#include <vector>

#include <TNL/Backend/Macros.h>

// Inflow profile shape of one opening: selects the velocity-bake semantics
// applied to LBM_INFLOW_OPENING::amplitude in finalizeInflowOpenings
// (UNIFORM -> the amplitude verbatim along the inward normal, PARABOLIC ->
// the amplitude as target volumetric flux normalized by openingProfileWeight).
enum class ProfileType
{
	UNIFORM,
	PARABOLIC
};

// Provenance of one LBM_INFLOW_OPENING record: AUTHORED openings carry an
// authored rect and drive the outward ghost carve (their outward layer may
// only contain walls, which yield, or an existing ghost); DISCOVERED
// openings are synthesized by the discovery pass over the complement of the
// authored claims and get the default UNIFORM profile with zero amplitude.
enum class OpeningOrigin
{
	AUTHORED,
	DISCOVERED
};

// Host-side record of one inflow opening (derived configuration, not
// checkpointed). Written by addInflowPlane (AUTHORED) or by
// the discovery pass of finalizeInflowOpenings (DISCOVERED); consumed by the
// same finalize pass only - the outward carve, the detector-mirror
// validation, the replicated plane pictures, the scale resolution, the
// velocity bake, and the per-plane report. Never uploaded: the kernel sees
// only the per-cell site id and the compressed velocity table baked from
// these records.
template <typename TRAITS>
struct LBM_INFLOW_OPENING
{
	int id = -1;						   // index into InflowOpeningsState::openings; stamped into the per-cell claim map at add
										   // time (finalize re-stamps live cells with the compressed-velocity site id), and embedded
										   // as id+2 in the replicated plane pictures so the scale pass can attribute cells to this
										   // opening without any collective
	short axis = 0;						   // plane identity, together with sign + planeOffset: matches the record to its plane
	short sign = 0;						   // picture and selects the tangential axes; sign is the outward normal of the opening, so
										   // the carve removes the layer at planeOffset+sign, the detector-mirror validation requires
										   // the face bit of (axis, sign), and the UNIFORM bake writes the amplitude as
										   // -sign * amplitude along axis (positive amplitude flows INTO the domain)
	typename TRAITS::idx planeOffset = 0;  // cell coordinate of the boundary plane along axis
	typename TRAITS::idx3d lo = 0;
	typename TRAITS::idx3d hi = 0;					 // touching rect in GLOBAL coordinates (clamped to the domain at add time; for
													 // discovered openings the component's bounding rect). Stamped onto hmap + claim
													 // map at add time; the PARABOLIC weights and their sum are evaluated over its
													 // tangential bounds
	typename TRAITS::dreal amplitude = 0;			 // authored value, interpreted per profile (see above); zero means "no
													 // authored velocity" and bakes the DATA inflow defaults verbatim (the
													 // discovered case)
	ProfileType profile = ProfileType::UNIFORM;		 // selects the bake semantics of amplitude
	typename TRAITS::dreal scale = 1;				 // resolved normalization, PARABOLIC only: amplitude / sum of the profile
													 // weights over the claimed cells (0 for a degenerate, fully overwritten
													 // rect); computed in finalize on the replicated plane pictures, so every
													 // rank bakes identical site velocities with no collective; host-side only
	OpeningOrigin origin = OpeningOrigin::AUTHORED;	 // only AUTHORED records drive the outward carve and the
													 // detector-mirror validation; the report splits per-plane
													 // counts by origin
};

// Axis-aligned bounding rect of one 4-connected component on a row-major 2D
// grid (u = column index within a row, v = row index).
struct ComponentRect2D
{
	int lo_u = 0;
	int lo_v = 0;
	int hi_u = -1;
	int hi_v = -1;
};

// Two-pass union-find labeling of 4-connected (cardinal-only) components of
// candidate cells on a width x height row-major grid; isCandidate(u, v)
// decides whether the cell belongs to the labelable set. Cells touching only
// diagonally are NOT connected (4-connectivity is the documented choice for
// inflow-opening components - 8-connectivity would merge openings that meet
// at a corner).
//
// Returns {component id per cell (-1 for non-candidates), bounding rect per
// component}. Component ids are assigned in ascending row-major order of each
// component's topmost row / leftmost column cell, so the result is fully
// deterministic for a fixed input picture (all MPI ranks must label the
// reconstructed global plane picture identically).
template <typename F>
std::pair<std::vector<int>, std::vector<ComponentRect2D>> labelComponents2D(int width, int height, F&& isCandidate)
{
	std::vector<int> ids;
	std::vector<ComponentRect2D> rects;
	if (width <= 0 || height <= 0)
		return {ids, rects};

	ids.assign(static_cast<std::size_t>(width) * static_cast<std::size_t>(height), -1);
	// parent < 0 marks non-candidates; unions attach the larger root to the
	// smaller one so the surviving root is always the component's first cell
	// in row-major order
	std::vector<int> parent(ids.size(), -1);

	auto find_root = [&parent](int i)
	{
		int r = i;
		while (parent[r] != r)
			r = parent[r];
		while (parent[i] != r) {  // path compression
			const int p = parent[i];
			parent[i] = r;
			i = p;
		}
		return r;
	};
	auto unite = [&parent, &find_root](int a, int b)
	{
		a = find_root(a);
		b = find_root(b);
		if (a == b)
			return;
		if (b < a)
			std::swap(a, b);
		parent[b] = a;
	};

	// pass 1: link each candidate to its left and up cardinal neighbors
	for (int v = 0; v < height; v++) {
		for (int u = 0; u < width; u++) {
			if (! isCandidate(u, v))
				continue;
			const int i = v * width + u;
			parent[i] = i;
			if (u > 0 && parent[i - 1] >= 0)
				unite(i, i - 1);
			if (v > 0 && parent[i - width] >= 0)
				unite(i, i - width);
		}
	}

	// pass 2: assign final ids in row-major first-visit order; the root's own
	// slot in ids doubles as the root -> compact id map, which is safe because
	// the root is the component's smallest cell index (unions keep the
	// smaller root), i.e. the first cell of the component reached by the scan
	for (int v = 0; v < height; v++) {
		for (int u = 0; u < width; u++) {
			const int i = v * width + u;
			if (parent[i] < 0)
				continue;
			const int r = find_root(i);
			if (ids[r] < 0) {
				ids[r] = static_cast<int>(rects.size());
				ComponentRect2D empty;	// lo = full grid, hi = -1: first cell updates all bounds
				empty.lo_u = width;
				empty.lo_v = height;
				rects.push_back(empty);
			}
			const int c = ids[r];
			ids[i] = c;
			ComponentRect2D& rect = rects[c];
			if (u < rect.lo_u)
				rect.lo_u = u;
			if (u > rect.hi_u)
				rect.hi_u = u;
			if (v < rect.lo_v)
				rect.lo_v = v;
			if (v > rect.hi_v)
				rect.hi_v = v;
		}
	}
	return {ids, rects};
}

// Per-cell profile weight of the imposed velocity on the opening rect
// [u0,u1] x [v0,v1] (tangential coordinates, u/v = the cell, u0/v0 = rect lo,
// u1/v1 = rect hi). UNIFORM openings get weight 1 everywhere (the amplitude
// itself is the imposed velocity); the uniform branch deliberately does not
// touch the geometry arguments.
// PARABOLIC uses a product paraboloid in CELL-CENTERED normalized coords
// xi = (u - u0 + 0.5) / (u1 - u0 + 1) (likewise eta over v), so a rect that
// is a single cell wide along a tangential direction contributes factor 1
// along it; the per-cell imposed velocity is scale * weight with the scale
// resolved at init from the target flux (finalizeInflowOpenings).
template <typename IDX, typename REAL>
__cuda_callable__ REAL openingProfileWeight(ProfileType profile, REAL u0, REAL u1, REAL v0, REAL v1, REAL u, REAL v)
{
	if (profile == ProfileType::UNIFORM)
		return 1;

	const REAL nu = u1 - u0 + 1;
	const REAL nv = v1 - v0 + 1;
	const REAL xi = (u - u0 + REAL(0.5)) / nu;
	const REAL eta = (v - v0 + REAL(0.5)) / nv;
	return (1 - (2 * xi - 1) * (2 * xi - 1)) * (1 - (2 * eta - 1) * (2 * eta - 1));
}
