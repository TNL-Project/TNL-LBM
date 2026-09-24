// scenario-runner translation unit of the cross-pattern AMR bitwise lock
// (test_amr_bitwise_lock): instantiates the two-level scenario pinned to the
// esoteric-push in-place pattern via the explicit D3Q27_STREAMING_ESO_PUSH
// type
#include "amr_bitwise_lock_runner.h"

AMRLockTrace runAMRLockScenario_eso_push()
{
	return runAMRLockScenario<AMRLockConfig<D3Q27_STREAMING_ESO_PUSH<TraitsSP>>>("eso_push");
}
