// scenario-runner translation unit of the nested cross-pattern AMR bitwise
// lock (test_amr_nested_lock): instantiates the three-level scenario pinned
// to the esoteric-push in-place pattern via the explicit
// D3Q27_STREAMING_ESO_PUSH type
#include "amr_nested_lock_runner.h"

AMRLockTrace runAMRNestedLockScenario_eso_push()
{
	return runAMRNestedLockScenario<AMRLockConfig<D3Q27_STREAMING_ESO_PUSH<TraitsSP>>>("eso_push");
}
