#pragma once

#include <cmath>

#include <TNL/Backend/Macros.h>

// Rounding-pinned arithmetic for mirrored expression pairs: nvcc/hipcc never fuse the intrinsic,
// and lbm_fma_rn fuses explicitly in one canonical operand order,
// so all operands round identically on every architecture.
// The host fallback uses std::fma, which computes a*b+c with a single rounding (no per-statement contraction).
__cuda_callable__ inline float lbm_fma_rn(float a, float b, float c)
{
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
	return __fmaf_rn(a, b, c);
#else
	return std::fma(a, b, c);
#endif
}
__cuda_callable__ inline double lbm_fma_rn(double a, double b, double c)
{
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
	return __fma_rn(a, b, c);
#else
	return std::fma(a, b, c);
#endif
}
