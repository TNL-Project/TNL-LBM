#pragma once

#include <functional>
#include <map>

#include <fmt/core.h>

#include <TNL/Containers/Block.h>
#include <TNL/Containers/BlockPartitioning.h>
#include <TNL/Containers/DistributedNDArraySyncDirections.h>
#include <TNL/Containers/StaticVector.h>
#include <TNL/MPI/Comm.h>

#include "lbm_block.h"

template <typename CONFIG, typename idx>
LBM_BLOCK<CONFIG>
decomposeLattice_D1Q3(const TNL::MPI::Comm& communicator, const TNL::Containers::StaticVector<3, idx>& global_size, bool periodic_boundaries)
{
	using idx3d = TNL::Containers::StaticVector<3, idx>;

	const int rank = communicator.rank();
	const int nproc = communicator.size();

	// split the x-axis uniformly
	auto local_range = TNL::Containers::splitRange<idx>(global_size.x(), communicator);

	// local size of the block
	idx3d local;
	local.x() = local_range.getEnd() - local_range.getBegin();
	local.y() = global_size.y();
	local.z() = global_size.z();

	// offset of the block on the global lattice
	idx3d offset;
	offset.x() = local_range.getBegin();
	offset.y() = offset.z() = 0;

	// instantiate the block
	LBM_BLOCK<CONFIG> block(communicator, global_size, local, offset, rank);

	// set synchronization pattern
	std::map<TNL::Containers::SyncDirection, int> neighbors;
	if (periodic_boundaries || rank > 0)
		neighbors[TNL::Containers::SyncDirection::Left] = (rank - 1 + nproc) % nproc;
	else
		neighbors[TNL::Containers::SyncDirection::Left] = -1;
	if (periodic_boundaries || rank < nproc - 1)
		neighbors[TNL::Containers::SyncDirection::Right] = (rank + 1 + nproc) % nproc;
	else
		neighbors[TNL::Containers::SyncDirection::Right] = -1;
	block.setLatticeDecomposition(TNL::Containers::NDArraySyncPatterns::D1Q3, neighbors, neighbors);

	return block;
}

/**
 * \brief Wraps \ref TNL::Containers::decomposeBlockOptimal with a permutation.
 *
 * \tparam Permutation is an \ref std::index_sequence that determines the order
 *                     in which the \e x, \e y, \e z directions are preferably
 *                     decomposed.
 * \param global The large block to decompose.
 * \param num_blocks Number of blocks.
 * \return A vector of the blocks into which the input was decomposed.
 */
template <typename Permutation, typename Index>
std::vector<TNL::Containers::Block<3, Index>> decomposeBlockOptimalWithPermutation(const TNL::Containers::Block<3, Index>& global, Index num_blocks)
{
	static_assert(Permutation::size() == 3);

	// leading dimension
	int i = TNL::Containers::detail::get<2>(Permutation{});
	// second dimension
	int j = TNL::Containers::detail::get<1>(Permutation{});
	// last dimension
	int k = TNL::Containers::detail::get<0>(Permutation{});

	// permute the global block
	auto permuted_global = global;
	permuted_global.begin[0] = global.begin[i];
	permuted_global.begin[1] = global.begin[j];
	permuted_global.begin[2] = global.begin[k];
	permuted_global.end[0] = global.end[i];
	permuted_global.end[1] = global.end[j];
	permuted_global.end[2] = global.end[k];

	// set axes-weights for the objective function used in the optimization
	const std::array<Index, 3>& axes_weights = {64, 8, 1};
	using FunctionType = std::function<Index(const std::vector<TNL::Containers::Block<3, Index>>&)>;
	const FunctionType objective = std::bind(TNL::Containers::getInterfaceArea<Index>, std::placeholders::_1, axes_weights);

	// decompose the permuted global block
	const std::vector<TNL::Containers::Block<3, Index>> permuted_result =
		TNL::Containers::decomposeBlockOptimal(permuted_global, num_blocks, objective);

	// unpermute the blocks in the result
	std::vector<TNL::Containers::Block<3, Index>> result;
	for (const auto& permuted_block : permuted_result) {
		auto& block = result.emplace_back(permuted_block);
		block.begin[i] = permuted_block.begin[0];
		block.begin[j] = permuted_block.begin[1];
		block.begin[k] = permuted_block.begin[2];
		block.end[i] = permuted_block.end[0];
		block.end[j] = permuted_block.end[1];
		block.end[k] = permuted_block.end[2];
	}

	return result;
}

/**
 * \brief Set neighbors for a synchronizer according to given synchronization pattern
 * and decomposition of a global block.
 *
 * \ingroup ndarray
 *
 * \tparam Q is the number of elements in \e pattern.
 * \param synchronizer is an instance of \ref DistributedNDArraySynchronizer.
 * \param pattern is the synchronization pattern (array of directions
 *                in which the data will be sent). It must be consistent
 *                with the partitioning of the distributed array.
 * \param rank is the ID of the current MPI rank and also an index of the
 *             corresponding block in \e decomposition.
 * \param decomposition is a vector of blocks forming a decomposition of the
 *                      global block. Its size must be equal to the size of
 *                      the MPI communicator and indices of the blocks in the
 *                      vector determine the rank IDs of the neighbors.
 * \param global is the global block (used for setting neighbors over the
 *               periodic boundary).
 */
template <std::size_t Q, typename BlockType>
std::map<TNL::Containers::SyncDirection, int> findNeighbors(
	const std::array<TNL::Containers::SyncDirection, Q>& pattern,
	int rank,
	const std::vector<BlockType>& decomposition,
	const BlockType& global,
	bool periodic_lattice
)
{
	using namespace TNL::Containers;

	const BlockType& reference = decomposition.at(rank);
	std::map<TNL::Containers::SyncDirection, int> neighbors;

	auto find = [&](SyncDirection direction, typename BlockType::CoordinatesType point, SyncDirection vertexDirection)
	{
		if (periodic_lattice) {
			// handle periodic boundaries
			if ((direction & SyncDirection::Left) != SyncDirection::None && point.x() == global.begin.x())
				point.x() = global.end.x();
			if ((direction & SyncDirection::Right) != SyncDirection::None && point.x() == global.end.x())
				point.x() = global.begin.x();
			if ((direction & SyncDirection::Bottom) != SyncDirection::None && point.y() == global.begin.y())
				point.y() = global.end.y();
			if ((direction & SyncDirection::Top) != SyncDirection::None && point.y() == global.end.y())
				point.y() = global.begin.y();
			if ((direction & SyncDirection::Back) != SyncDirection::None && point.z() == global.begin.z())
				point.z() = global.end.z();
			if ((direction & SyncDirection::Front) != SyncDirection::None && point.z() == global.end.z())
				point.z() = global.begin.z();
		}

		for (std::size_t i = 0; i < decomposition.size(); i++) {
			const auto vertex = getBlockVertex(decomposition[i], vertexDirection);
			if (point == vertex) {
				neighbors[direction] = i;
				return;
			}
		}

		// no neighbor found in this direction
		if (periodic_lattice)
			throw std::runtime_error(fmt::format("coordinate [{},{},{}] was not found in the decomposition", point.x(), point.y(), point.z()));
		else
			neighbors[direction] = -1;
	};

	for (SyncDirection direction : pattern) {
		switch (direction) {
			case SyncDirection::Left:
				find(direction, getBlockVertex(reference, SyncDirection::FrontTopLeft), SyncDirection::FrontTopRight);
				break;
			case SyncDirection::Right:
				find(direction, getBlockVertex(reference, SyncDirection::BackBottomRight), SyncDirection::BackBottomLeft);
				break;
			case SyncDirection::Bottom:
				find(direction, getBlockVertex(reference, SyncDirection::FrontBottomRight), SyncDirection::FrontTopRight);
				break;
			case SyncDirection::Top:
				find(direction, getBlockVertex(reference, SyncDirection::BackTopLeft), SyncDirection::BackBottomLeft);
				break;
			case SyncDirection::Back:
				find(direction, getBlockVertex(reference, SyncDirection::BackTopRight), SyncDirection::FrontTopRight);
				break;
			case SyncDirection::Front:
				find(direction, getBlockVertex(reference, SyncDirection::FrontTopRight), SyncDirection::BackTopRight);
				break;
			case SyncDirection::BottomLeft:
				find(direction, getBlockVertex(reference, SyncDirection::FrontBottomLeft), SyncDirection::FrontTopRight);
				break;
			case SyncDirection::BottomRight:
				find(direction, getBlockVertex(reference, SyncDirection::BackBottomRight), SyncDirection::BackTopLeft);
				break;
			case SyncDirection::TopRight:
				find(direction, getBlockVertex(reference, SyncDirection::BackTopRight), SyncDirection::BackBottomLeft);
				break;
			case SyncDirection::TopLeft:
				find(direction, getBlockVertex(reference, SyncDirection::BackTopLeft), SyncDirection::BackBottomRight);
				break;
			case SyncDirection::BackLeft:
				find(direction, getBlockVertex(reference, SyncDirection::BackBottomLeft), SyncDirection::FrontBottomRight);
				break;
			case SyncDirection::BackRight:
				find(direction, getBlockVertex(reference, SyncDirection::BackBottomRight), SyncDirection::FrontBottomLeft);
				break;
			case SyncDirection::BackBottom:
				find(direction, getBlockVertex(reference, SyncDirection::BackBottomLeft), SyncDirection::FrontTopLeft);
				break;
			case SyncDirection::BackTop:
				find(direction, getBlockVertex(reference, SyncDirection::BackTopLeft), SyncDirection::FrontBottomLeft);
				break;
			case SyncDirection::FrontLeft:
				find(direction, getBlockVertex(reference, SyncDirection::FrontBottomLeft), SyncDirection::BackBottomRight);
				break;
			case SyncDirection::FrontRight:
				find(direction, getBlockVertex(reference, SyncDirection::FrontBottomRight), SyncDirection::BackBottomLeft);
				break;
			case SyncDirection::FrontBottom:
				find(direction, getBlockVertex(reference, SyncDirection::FrontBottomLeft), SyncDirection::BackTopLeft);
				break;
			case SyncDirection::FrontTop:
				find(direction, getBlockVertex(reference, SyncDirection::FrontTopLeft), SyncDirection::BackBottomLeft);
				break;
			case SyncDirection::BackBottomLeft:
			case SyncDirection::BackBottomRight:
			case SyncDirection::BackTopLeft:
			case SyncDirection::BackTopRight:
			case SyncDirection::FrontBottomLeft:
			case SyncDirection::FrontBottomRight:
			case SyncDirection::FrontTopLeft:
			case SyncDirection::FrontTopRight:
				find(direction, getBlockVertex(reference, direction), opposite(direction));
				break;
			default:
				throw std::logic_error("unhandled direction: " + std::to_string(static_cast<std::uint8_t>(direction)));
		}
	}

	return neighbors;
}

template <typename CONFIG, typename idx>
LBM_BLOCK<CONFIG>
decomposeLattice_D3Q27(const TNL::MPI::Comm& communicator, const TNL::Containers::StaticVector<3, idx>& global_size, bool periodic_lattice)
{
	using idx3d = TNL::Containers::StaticVector<3, idx>;

	const int rank = communicator.rank();
	const int nproc = communicator.size();

	// find optimal decomposition
	const TNL::Containers::Block<3, idx> globalBoundingBox = {idx3d{0, 0, 0}, global_size};
	using Permutation = typename CONFIG::TRAITS::xyz_permutation;
	const auto decomposition = decomposeBlockOptimalWithPermutation<Permutation>(globalBoundingBox, idx(nproc));
	const TNL::Containers::Block<3, idx>& localBoundingBox = decomposition.at(rank);

	// local size of the block
	const idx3d local = localBoundingBox.end - localBoundingBox.begin;

	// offset of the block on the global lattice
	const idx3d offset = localBoundingBox.begin;

	// instantiate the block
	LBM_BLOCK<CONFIG> block(communicator, global_size, local, offset, rank);

	// set synchronization pattern
	const std::map<TNL::Containers::SyncDirection, int> neighbors =
		findNeighbors(TNL::Containers::NDArraySyncPatterns::D3Q27, rank, decomposition, globalBoundingBox, periodic_lattice);
	block.setLatticeDecomposition(TNL::Containers::NDArraySyncPatterns::D3Q27, neighbors, neighbors);

	return block;
}
