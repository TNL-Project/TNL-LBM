// scenario-runner translation unit of the nested cross-pattern AMR bitwise
// lock (test_amr_nested_lock): instantiates the three-level scenario pinned
// to the esoteric-twist in-place pattern via the explicit
// D3Q27_STREAMING_ESO_TWIST type
#include "amr_nested_lock_runner.h"

AMRLockTrace runAMRNestedLockScenario_eso_twist()
{
	return runAMRNestedLockScenario<AMRLockConfig<D3Q27_STREAMING_ESO_TWIST<TraitsSP>>>("eso_twist");
}
