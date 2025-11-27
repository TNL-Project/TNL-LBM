#pragma once

#include <type_traits>
#include <vector>

template <typename TRAITS>
class DataWriter
{
protected:
	using idx = typename TRAITS::idx;

	// internal storage for array variables
	std::vector<std::vector<int>> iBuffers;
	std::vector<std::vector<float>> fBuffers;
	std::vector<std::vector<idx>> idxBuffers;

	std::vector<int>& newIBuffer(std::size_t reserve = 0)
	{
		iBuffers.emplace_back();
		if (reserve > 0)
			iBuffers.back().reserve(reserve);
		return iBuffers.back();
	}

	std::vector<float>& newFBuffer(std::size_t reserve = 0)
	{
		fBuffers.emplace_back();
		if (reserve > 0)
			fBuffers.back().reserve(reserve);
		return fBuffers.back();
	}

	std::vector<idx>& newIdxBuffer(std::size_t reserve = 0)
	{
		idxBuffers.emplace_back();
		if (reserve > 0)
			idxBuffers.back().reserve(reserve);
		return idxBuffers.back();
	}

	template <typename T>
	std::vector<T>& newBuffer(std::size_t reserve = 0)
	{
		if constexpr (std::is_same_v<T, int>) {
			return newIBuffer(reserve);
		}
		else if constexpr (std::is_same_v<T, float>) {
			return newFBuffer(reserve);
		}
		else if constexpr (std::is_same_v<T, idx>) {
			return newIdxBuffer(reserve);
		}
		else {
			static_assert(! std::is_same_v<T, T>, "DataWriter::newBuffer: unsupported buffer type");
		}
	}
};
