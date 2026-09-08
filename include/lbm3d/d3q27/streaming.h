#pragma once

// all streaming patterns are always available; the alias selects the
// legacy name for leaf code (the TNL_LBM_STREAMING_PATTERN_* macros are
// consulted only in the three streaming.h umbrellas)
#include "streaming_AA.h"
#include "streaming_AB_PULL.h"
#include "streaming_AB_PUSH.h"

#if defined(TNL_LBM_STREAMING_PATTERN_AA)
template <typename TRAITS>
using D3Q27_STREAMING = D3Q27_STREAMING_AA<TRAITS>;
#elif defined(TNL_LBM_STREAMING_PATTERN_AB_PUSH)
template <typename TRAITS>
using D3Q27_STREAMING = D3Q27_STREAMING_AB_PUSH<TRAITS>;
#else
// A-B pull is the default pattern
template <typename TRAITS>
using D3Q27_STREAMING = D3Q27_STREAMING_AB_PULL<TRAITS>;
#endif
