// scenario-runner translation unit of the nested cross-pattern AMR bitwise
// lock (test_amr_nested_lock): instantiates the three-level scenario pinned
// to the A-B pull pattern via the explicit D3Q27_STREAMING_AB_PULL type --
// the reference instance every other pattern's trace is compared against
#include "amr_nested_lock_runner.h"

AMRLockTrace runAMRNestedLockScenario_ab()
{
	return runAMRNestedLockScenario<AMRLockConfig<D3Q27_STREAMING_AB_PULL<TraitsSP>>>("ab");
}
