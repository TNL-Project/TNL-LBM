#pragma once

#include <algorithm>
#include <vector>

#include "lbm3d/lbm.h"
#include "lbm3d/output_buffer_sync.h"
#include "lbm3d/UniformDataWriter.h"

namespace lbm3d {

// The helpers below are 3D-only: they read the velocity components e_vx, e_vy and e_vz
// through the NSE policy, so they require a 3D lattice model (D3Q27, D3Q7). The
// Q-criterion is set to zero at sites adjacent to a domain boundary or a non-fluid cell,
// which is the conventional choice for a visualization quantity.

/**
 * \brief Velocity-gradient tensor in physical units (1/s).
 */
template <typename REAL>
struct VelocityGradient
{
	REAL xx = 0, xy = 0, xz = 0;
	REAL yx = 0, yy = 0, yz = 0;
	REAL zx = 0, zy = 0, zz = 0;
};

/**
 * \brief Compute the velocity gradient (in physical units) at a single lattice site.
 *
 * The gradient is evaluated with central differences in the interior, and with
 * one-sided (forward/backward) differences in the first/last column of the X dimension.
 * Sites outside the strictly local range of the block, sites adjacent to a domain
 * boundary and sites with a non-fluid face neighbor are assigned a zero gradient, which
 * keeps the derived quantities (e.g. the Q-criterion) free of boundary artifacts.
 *
 * \param block  LBM block providing hmicro/macro host accessors and isLocalIndex().
 * \param nse    Distributed lattice providing the physical conversion helpers.
 */
template <typename BLOCK, typename NSE>
VelocityGradient<typename BLOCK::real>
computeVelocityGradient(const BLOCK& block, const LBM<NSE>& nse, typename BLOCK::idx x, typename BLOCK::idx y, typename BLOCK::idx z)
{
	using real = typename BLOCK::real;
	using MACRO = typename NSE::MACRO;
	using BC = typename NSE::BC;

	VelocityGradient<real> G;

	// sites in the overlap region hold no valid macroscopic data
	if (! block.isLocalIndex(x, y, z))
		return G;

	// domain-boundary cells and non-fluid neighborhoods get a zero gradient
	if (x == 0 || y == 0 || z == 0 || x == nse.lat.global.x() - 1 || y == nse.lat.global.y() - 1 || z == nse.lat.global.z() - 1
		|| ! BC::isFluid(block.hmap(x + 1, y, z)) || ! BC::isFluid(block.hmap(x - 1, y, z)) || ! BC::isFluid(block.hmap(x, y + 1, z))
		|| ! BC::isFluid(block.hmap(x, y - 1, z)) || ! BC::isFluid(block.hmap(x, y, z + 1)) || ! BC::isFluid(block.hmap(x, y, z - 1)))
	{
		return G;
	}

	const real inv2dl = 1 / (2 * nse.lat.physDl);
	G.xx = nse.lat.lbm2physVelocity((real) block.hmacro(MACRO::e_vx, x + 1, y, z) - (real) block.hmacro(MACRO::e_vx, x - 1, y, z)) * inv2dl;
	G.xy = nse.lat.lbm2physVelocity((real) block.hmacro(MACRO::e_vx, x, y + 1, z) - (real) block.hmacro(MACRO::e_vx, x, y - 1, z)) * inv2dl;
	G.xz = nse.lat.lbm2physVelocity((real) block.hmacro(MACRO::e_vx, x, y, z + 1) - (real) block.hmacro(MACRO::e_vx, x, y, z - 1)) * inv2dl;
	G.yx = nse.lat.lbm2physVelocity((real) block.hmacro(MACRO::e_vy, x + 1, y, z) - (real) block.hmacro(MACRO::e_vy, x - 1, y, z)) * inv2dl;
	G.yy = nse.lat.lbm2physVelocity((real) block.hmacro(MACRO::e_vy, x, y + 1, z) - (real) block.hmacro(MACRO::e_vy, x, y - 1, z)) * inv2dl;
	G.yz = nse.lat.lbm2physVelocity((real) block.hmacro(MACRO::e_vy, x, y, z + 1) - (real) block.hmacro(MACRO::e_vy, x, y, z - 1)) * inv2dl;
	G.zx = nse.lat.lbm2physVelocity((real) block.hmacro(MACRO::e_vz, x + 1, y, z) - (real) block.hmacro(MACRO::e_vz, x - 1, y, z)) * inv2dl;
	G.zy = nse.lat.lbm2physVelocity((real) block.hmacro(MACRO::e_vz, x, y + 1, z) - (real) block.hmacro(MACRO::e_vz, x, y - 1, z)) * inv2dl;
	G.zz = nse.lat.lbm2physVelocity((real) block.hmacro(MACRO::e_vz, x, y, z + 1) - (real) block.hmacro(MACRO::e_vz, x, y, z - 1)) * inv2dl;

	return G;
}

/**
 * \brief Compute the Q-criterion at a single lattice site.
 *
 * Uses the second invariant of the velocity-gradient tensor, Q = G.xx*G.yy + G.yy*G.zz +
 * G.xx*G.zz - G.zx*G.xz - G.yz*G.zy - G.xy*G.yx, with the velocity gradient in physical
 * units. The value is zero wherever computeVelocityGradient() returns a zero gradient.
 */
template <typename BLOCK, typename NSE>
typename BLOCK::real computeQ(const BLOCK& block, const LBM<NSE>& nse, typename BLOCK::idx x, typename BLOCK::idx y, typename BLOCK::idx z)
{
	const VelocityGradient<typename BLOCK::real> G = computeVelocityGradient(block, nse, x, y, z);
	return G.xx * G.yy + G.yy * G.zz + G.xx * G.zz - G.zx * G.xz - G.yz * G.zy - G.xy * G.yx;
}

/**
 * \brief Write the Q-criterion to a UniformDataWriter, synchronizing the MPI overlaps.
 *
 * Unlike the macroscopic quantities, the Q-criterion cannot be evaluated in the overlap
 * cells of a subdomain, because it depends on neighbor sites that are not valid there.
 * This function therefore computes Q only on strictly local sites into a flat buffer,
 * exchanges the trailing overlap cells with the neighboring ranks via
 * synchronizeOutputBuffer(), and then writes the filled buffer through the writer. The
 * output extent passed by the caller (begin/end, in global coordinates) determines which
 * dimensions are distributed and therefore which of them take part in the exchange.
 */
template <typename NSE, typename BLOCK>
void writeQCriterion(
	UniformDataWriter<typename NSE::TRAITS>& writer,
	const BLOCK& block,
	const LBM<NSE>& nse,
	const typename NSE::TRAITS::idx3d& begin,
	const typename NSE::TRAITS::idx3d& end
)
{
	using TRAITS = typename NSE::TRAITS;
	using idx = typename TRAITS::idx;
	using dreal = typename TRAITS::dreal;
	using idx3d = typename TRAITS::idx3d;

	// The writer's local box is begin..end, which may include a trailing overlap cell in
	// each distributed dimension. The overlap is the part of the box that is not covered
	// by the strictly local subdomain (clamped to zero for 2D plane cuts and for 3D
	// bounding boxes that are smaller than the subdomain).
	const idx3d overlap = {
		std::max<idx>(0, end.x() - begin.x() - block.local.x()),
		std::max<idx>(0, end.y() - begin.y() - block.local.y()),
		std::max<idx>(0, end.z() - begin.z() - block.local.z()),
	};
	const idx3d local_count = {end.x() - begin.x() - overlap.x(), end.y() - begin.y() - overlap.y(), end.z() - begin.z() - overlap.z()};
	const idx nx = local_count.x() + overlap.x();
	const idx ny = local_count.y() + overlap.y();
	const idx nz = local_count.z() + overlap.z();

	std::vector<dreal> buffer(static_cast<std::size_t>(nx) * ny * nz, 0);

	for (idx z = begin.z(); z < begin.z() + local_count.z(); z++)
		for (idx y = begin.y(); y < begin.y() + local_count.y(); y++)
			for (idx x = begin.x(); x < begin.x() + local_count.x(); x++) {
				const idx bx = x - begin.x();
				const idx by = y - begin.y();
				const idx bz = z - begin.z();
				buffer[(bz * ny + by) * nx + bx] = computeQ(block, nse, x, y, z);
			}

	synchronizeOutputBuffer(buffer, block, local_count, overlap, writer.getGlobal());

	writer.write(
		"Q",
		[&](idx x, idx y, idx z) -> dreal
		{
			const idx bx = x - begin.x();
			const idx by = y - begin.y();
			const idx bz = z - begin.z();
			return buffer[(bz * ny + by) * nx + bx];
		},
		begin,
		end
	);
}

}  // namespace lbm3d
