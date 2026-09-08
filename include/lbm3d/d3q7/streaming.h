#pragma once

// both streaming patterns are always available; the alias selects the
// legacy name for leaf code (the AA_PATTERN/AB_PATTERN macros are consulted
// only here and in the d3q27/d2q9 streaming headers' umbrellas)
#include "streaming_AA.h"
#include "streaming_AB_PULL.h"
#include "streaming_AB_PUSH.h"

#ifdef AA_PATTERN
template <typename TRAITS>
using D3Q7_STREAMING = D3Q7_STREAMING_AA<TRAITS>;
#else
// A-B is the default pattern
template <typename TRAITS>
using D3Q7_STREAMING = D3Q7_STREAMING_AB_PULL<TRAITS>;
#endif
