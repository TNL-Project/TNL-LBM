#pragma once

#include <cstdint>

#include <TNL/Backend/Macros.h>

template <typename TRAITS>
struct D3Q7_MACRO_Default
{
	using dreal = typename TRAITS::dreal;
	using idx = typename TRAITS::idx;

	enum QuantityNames : std::uint8_t
	{
		e_phi,
		N
	};

	static const bool use_syncMacro = false;

	// maximum width of overlaps for the macro arrays
	static constexpr int overlap_width = 1;

	template <typename LBM_KS>
	__cuda_callable__ static void zeroForcesInKS(LBM_KS& KS)
	{}

	template <typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void outputMacro(LBM_DATA& SD, LBM_KS& KS, idx x, idx y, idx z)
	{
		SD.macro(e_phi, x, y, z) = KS.phi;
	}

	template <typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void copyQuantities(LBM_DATA& SD, LBM_KS& KS, idx x, idx y, idx z)
	{
		KS.lbmViscosity = SD.diffusionCoefficient(x, y, z);
	}
};

template <typename TRAITS>
struct D3Q7_MACRO_Void
{
	using dreal = typename TRAITS::dreal;
	using idx = typename TRAITS::idx;

	static const int N = 0;

	template <typename LBM_KS>
	__cuda_callable__ static void zeroForcesInKS(LBM_KS& KS)
	{}

	template <typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void outputMacro(LBM_DATA& SD, LBM_KS& KS, idx x, idx y, idx z)
	{}

	template <typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void copyQuantities(LBM_DATA& SD, LBM_KS& KS, idx x, idx y, idx z)
	{}
};
