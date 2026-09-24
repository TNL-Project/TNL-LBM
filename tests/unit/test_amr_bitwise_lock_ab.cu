// scenario-runner translation unit of the cross-pattern AMR bitwise lock
// (test_amr_bitwise_lock): instantiates the two-level scenario pinned to the
// A-B pull pattern via the explicit D3Q27_STREAMING_AB_PULL type — the
// reference instance every other pattern's trace is compared against
#include "amr_bitwise_lock_runner.h"

AMRLockTrace runAMRLockScenario_ab()
{
	return runAMRLockScenario<AMRLockConfig<D3Q27_STREAMING_AB_PULL<TraitsSP>>>("ab");
}
