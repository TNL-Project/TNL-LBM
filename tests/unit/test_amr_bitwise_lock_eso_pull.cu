// scenario-runner translation unit of the cross-pattern AMR bitwise lock
// (test_amr_bitwise_lock): instantiates the two-level scenario pinned to the
// esoteric-pull in-place pattern via the explicit D3Q27_STREAMING_ESO_PULL
// type
#include "amr_bitwise_lock_runner.h"

AMRLockTrace runAMRLockScenario_eso_pull()
{
	return runAMRLockScenario<AMRLockConfig<D3Q27_STREAMING_ESO_PULL<TraitsSP>>>("eso_pull");
}
