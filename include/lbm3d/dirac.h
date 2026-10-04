#pragma once

#include <TNL/Math.h>

#include "defs.h"

template <typename real>
CUDA_HOSTDEV bool isDDNonZero(int i, real r)
{
	switch (i) {
		case 1:	 // VU: phi3
			return fabs(r) < (real) 1.0;
		case 2:	 // VU: phi2
			return fabs(r) < (real) 2.0;
		case 3:	 // VU: phi1
			return fabs(r) < (real) 2.0;
		case 4:	 // VU: phi4
			return fabs(r) < (real) 1.5;
	}
	return false;
}

// NOTE: it is assumed that isDDNonZero is called explicitly outside this function, it is not checked again in diracDelta
template <typename real>
CUDA_HOSTDEV real diracDelta(int i, real r)
{
	switch (i) {
		case 1:	 // VU: phi3
			return 1 - fabs(r);
		case 2:	 // VU: phi2
			return (real) 0.25 * (1 + cos((real) TNL::pi * r * (real) 0.5));
		case 3:	 // VU: phi1
			if (fabs(r) > (real) 1.0)
				return (5 - 2 * fabs(r) - sqrt(-7 + 12 * fabs(r) - 4 * r * r)) / (real) 8.0;
			else
				return (3 - 2 * fabs(r) + sqrt(1 + 4 * fabs(r) - 4 * r * r)) / (real) 8.0;
		case 4:	 // VU: phi4
			if (fabs(r) > (real) 0.5)
				return (5 - 3 * fabs(r) - sqrt(-2 + 6 * fabs(r) - 3 * r * r)) / (real) 6.0;
			else
				return (1 + sqrt(1 - 3 * r * r)) / (real) 3.0;
	}

	// Just a failsafe to avoid a compiler warning
	return 0;
}

template <typename LLVectorType>
CUDA_HOSTDEV bool is3DiracNonZero(int rDirac, int colIndex, int rowIndex, const LLVectorType& LL)
{
	if (isDDNonZero(rDirac, LL[rowIndex].x() - LL[colIndex].x())) {
		if (isDDNonZero(rDirac, LL[rowIndex].y() - LL[colIndex].y())) {
			if (isDDNonZero(rDirac, LL[rowIndex].z() - LL[colIndex].z())) {
				return true;
			}
		}
	}
	return false;
}

// NOTE: it is assumed that is3DiracNonZero is called explicitly outside this function, it is not checked again in calculate3Dirac
template <typename LLVectorType>
CUDA_HOSTDEV typename LLVectorType::ValueType::RealType calculate3Dirac(int rDirac, int colIndex, int rowIndex, const LLVectorType& LL)
{
	using real = typename LLVectorType::ValueType::RealType;
	real d1 = diracDelta(rDirac, LL[rowIndex].x() - LL[colIndex].x());
	real d2 = diracDelta(rDirac, LL[rowIndex].y() - LL[colIndex].y());
	real d3 = diracDelta(rDirac, LL[rowIndex].z() - LL[colIndex].z());
	return d1 * d2 * d3;
}
