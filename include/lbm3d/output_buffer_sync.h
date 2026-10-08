#pragma once

#include <vector>

#include <TNL/MPI.h>
#include <TNL/Containers/DistributedNDArraySyncDirections.h>
#include <TNL/Containers/StaticVector.h>

namespace lbm3d {

#ifdef HAVE_MPI

namespace detail {

/**
 * \brief Determine if this rank is the first in a given dimension.
 */
template <typename idx>
bool isFirstInDimension(const TNL::Containers::StaticVector<3, idx>& offset, int dim)
{
	return offset[dim] == 0;
}

/**
 * \brief Determine if this rank is the last in a given dimension.
 */
template <typename idx>
bool isLastInDimension(
	const TNL::Containers::StaticVector<3, idx>& offset,
	const TNL::Containers::StaticVector<3, idx>& local,
	const TNL::Containers::StaticVector<3, idx>& global,
	int dim
)
{
	return offset[dim] + local[dim] == global[dim];
}

}  // namespace detail

/**
 * \brief Synchronize a host-side flat output buffer across MPI subdomain overlaps.
 *
 * The buffer contains locally computed values (e.g., Q-criterion) that are only valid on
 * strictly local sites. After calling this function, trailing overlap regions
 * (right/top/front faces) are filled with values received from neighboring subdomains,
 * making the entire buffer (local + overlap) consistent for output.
 *
 * Buffer layout is row-major: local data occupies indices [0, local_size) in each
 * dimension, and the trailing overlap (if any) occupies indices
 * [local_size, local_size+overlap_size). There is no leading overlap margin. Only
 * nearest-neighbor exchange is performed, no global reduction.
 *
 * The three dimensions are synchronized sequentially: the exchange in X completes and is
 * unpacked before Y is packed, and both complete before Z is packed. This ordering makes
 * the sends of a later dimension carry the overlap columns already received in earlier
 * dimensions, so cells lying in the overlap of two dimensions (subdomain edges and
 * corners, e.g. gradient-derived quantities) are forwarded correctly. A single batched
 * exchange would pack the perpendicular overlap columns before they are received and
 * leave those cells stale.
 *
 * The `block.neighborRanks` map must be populated (by
 * LBM_BLOCK::setLatticeDecomposition()) before this function is called; otherwise the
 * neighbor lookup throws.
 *
 * \tparam BLOCK    LBM_BLOCK type providing communicator, global, local, offset,
 *                  is_distributed() and neighborRanks fields.
 *
 * \param buffer       Flat row-major buffer of size
 *                     (local.x()+overlap.x()) * (local.y()+overlap.y()) * (local.z()+overlap.z()).
 * \param block        LBM block with MPI metadata.
 * \param local_size   Local subdomain size WITHOUT overlap (3D).
 * \param overlap_size Trailing overlap width per dimension (3D). 0 for the last rank in
 *                     each distributed dimension, 1 for non-last ranks.
 * \param cut_global   Global extent of this output (the writer's `global` field). For
 *                     full-domain outputs this equals block.global. For 2D plane cuts the
 *                     cut dimension is 1. For 3D bounding-box cuts it may be arbitrary.
 *                     Communication in dimension D is safe only when
 *                     cut_global.D() == block.global.D(), because that guarantees ALL
 *                     ranks along D participate in this output; otherwise a participating
 *                     rank would post messages to a neighbor that never calls this
 *                     function and the wait would hang. Consequently, a partial-region
 *                     output (e.g. a 3D cut that spans only part of a distributed
 *                     dimension) leaves the overlap cells of the boundary ranks on that
 *                     dimension zero-initialized; those cells are outside the output's
 *                     interior and merely appear as zeros in the written field.
 */
template <typename BLOCK>
void synchronizeOutputBuffer(
	std::vector<typename BLOCK::dreal>& buffer,
	const BLOCK& block,
	const typename BLOCK::idx3d& local_size,
	const typename BLOCK::idx3d& overlap_size,
	const typename BLOCK::idx3d& cut_global
)
{
	using idx = typename BLOCK::idx;
	using dreal = typename BLOCK::dreal;
	using Dir = TNL::Containers::SyncDirection;

	// Message tags for the three directions. They are private to this helper: the caller
	// must not post other messages with these tags on the same communicator concurrently.
	constexpr int tag_x = 10;
	constexpr int tag_y = 11;
	constexpr int tag_z = 12;

	// Total buffer size per dimension (local + trailing overlap).
	const idx nx = local_size.x() + overlap_size.x();
	const idx ny = local_size.y() + overlap_size.y();
	const idx nz = local_size.z() + overlap_size.z();
	const idx plane_size = ny * nx;

	const TNL::MPI::Comm& comm = block.communicator;

	// Communication in dimension D is safe only when ALL ranks along D participate in
	// this output, i.e. when the cut spans the full global domain in D. For 2D plane
	// cuts the cut dimension has cut_global = 1 != block.global, so communication there
	// is skipped. For 3D bounding-box cuts that partially cover D, some ranks
	// participate and others skip, so communication is skipped as well.
	const bool dist_x = block.is_distributed().x() && (cut_global.x() == block.global.x());
	const bool dist_y = block.is_distributed().y() && (cut_global.y() == block.global.y());
	const bool dist_z = block.is_distributed().z() && (cut_global.z() == block.global.z());

	// Send width: the number of planes sent to the left/bottom/back neighbor. It must
	// match what the neighbor expects to receive, which is its overlap_size (1 for
	// non-last ranks). The last rank has zero overlap but still sends one plane; the
	// first rank sends nothing.
	constexpr idx halo = 1;

	// X-dimension: send the first halo X-planes to the left neighbor, receive the
	// trailing overlap X-planes from the right neighbor into the buffer.
	if (dist_x) {
		std::vector<dreal> send_x_left;
		std::vector<dreal> recv_x_right;
		MPI_Request requests[2];
		int request_count = 0;

		if (! detail::isFirstInDimension(block.offset, 0)) {
			const int dest = block.neighborRanks.at(Dir::Left);
			const idx count = halo * ny * nz;
			send_x_left.resize(count);
			idx si = 0;
			for (idx z = 0; z < nz; z++)
				for (idx y = 0; y < ny; y++)
					for (idx x = 0; x < halo; x++)
						send_x_left[si++] = buffer[z * plane_size + y * nx + x];
			requests[request_count++] = TNL::MPI::Isend(send_x_left.data(), static_cast<int>(count), dest, tag_x, comm);
		}

		if (! detail::isLastInDimension(block.offset, block.local, block.global, 0)) {
			const int src = block.neighborRanks.at(Dir::Right);
			const idx count = overlap_size.x() * ny * nz;
			recv_x_right.resize(count);
			requests[request_count++] = TNL::MPI::Irecv(recv_x_right.data(), static_cast<int>(count), src, tag_x, comm);
		}

		if (request_count > 0)
			TNL::MPI::Waitall(requests, request_count);

		if (! recv_x_right.empty()) {
			idx si = 0;
			for (idx z = 0; z < nz; z++)
				for (idx y = 0; y < ny; y++)
					for (idx x = local_size.x(); x < local_size.x() + overlap_size.x(); x++)
						buffer[z * plane_size + y * nx + x] = recv_x_right[si++];
		}
	}

	// Y-dimension: send the first halo Y-planes to the bottom neighbor, receive the
	// trailing overlap Y-planes from the top neighbor. The packed rows span the full nx
	// extent, including the X-overlap columns received above.
	if (dist_y) {
		std::vector<dreal> send_y_bottom;
		std::vector<dreal> recv_y_top;
		MPI_Request requests[2];
		int request_count = 0;

		if (! detail::isFirstInDimension(block.offset, 1)) {
			const int dest = block.neighborRanks.at(Dir::Bottom);
			const idx count = halo * nx * nz;
			send_y_bottom.resize(count);
			idx si = 0;
			for (idx z = 0; z < nz; z++)
				for (idx y = 0; y < halo; y++)
					for (idx x = 0; x < nx; x++)
						send_y_bottom[si++] = buffer[z * plane_size + y * nx + x];
			requests[request_count++] = TNL::MPI::Isend(send_y_bottom.data(), static_cast<int>(count), dest, tag_y, comm);
		}

		if (! detail::isLastInDimension(block.offset, block.local, block.global, 1)) {
			const int src = block.neighborRanks.at(Dir::Top);
			const idx count = overlap_size.y() * nx * nz;
			recv_y_top.resize(count);
			requests[request_count++] = TNL::MPI::Irecv(recv_y_top.data(), static_cast<int>(count), src, tag_y, comm);
		}

		if (request_count > 0)
			TNL::MPI::Waitall(requests, request_count);

		if (! recv_y_top.empty()) {
			idx si = 0;
			for (idx z = 0; z < nz; z++)
				for (idx y = local_size.y(); y < local_size.y() + overlap_size.y(); y++)
					for (idx x = 0; x < nx; x++)
						buffer[z * plane_size + y * nx + x] = recv_y_top[si++];
		}
	}

	// Z-dimension: send the first halo Z-planes to the back neighbor, receive the
	// trailing overlap Z-planes from the front neighbor. Z is the slowest-varying
	// dimension, so the front overlap is a contiguous block of planes and can be
	// received directly into the buffer. The packed planes span the full nx and ny
	// extents, including the X- and Y-overlap cells received above.
	if (dist_z) {
		std::vector<dreal> send_z_back;
		MPI_Request requests[2];
		int request_count = 0;

		if (! detail::isFirstInDimension(block.offset, 2)) {
			const int dest = block.neighborRanks.at(Dir::Back);
			const idx count = halo * ny * nx;
			send_z_back.resize(count);
			idx si = 0;
			for (idx z = 0; z < halo; z++)
				for (idx y = 0; y < ny; y++)
					for (idx x = 0; x < nx; x++)
						send_z_back[si++] = buffer[z * plane_size + y * nx + x];
			requests[request_count++] = TNL::MPI::Isend(send_z_back.data(), static_cast<int>(count), dest, tag_z, comm);
		}

		if (! detail::isLastInDimension(block.offset, block.local, block.global, 2)) {
			const int src = block.neighborRanks.at(Dir::Front);
			const idx count = overlap_size.z() * plane_size;
			requests[request_count++] = TNL::MPI::Irecv(
				&buffer[static_cast<std::size_t>(local_size.z()) * static_cast<std::size_t>(plane_size)], static_cast<int>(count), src, tag_z, comm
			);
		}

		if (request_count > 0)
			TNL::MPI::Waitall(requests, request_count);
	}
}

#else  // ! HAVE_MPI

template <typename BLOCK>
void synchronizeOutputBuffer(
	std::vector<typename BLOCK::dreal>& /*buffer*/,
	const BLOCK& /*block*/,
	const typename BLOCK::idx3d& /*local_size*/,
	const typename BLOCK::idx3d& /*overlap_size*/,
	const typename BLOCK::idx3d& /*cut_global*/
)
{
	// No-op when MPI is disabled: all data is local.
}

#endif	// HAVE_MPI

}  // namespace lbm3d
