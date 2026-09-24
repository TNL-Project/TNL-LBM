// scenario-runner translation unit of the cross-pattern AMR bitwise lock
// (test_amr_bitwise_lock): instantiates the two-level scenario pinned to the
// esoteric-twist in-place pattern via the explicit D3Q27_STREAMING_ESO_TWIST
// type
#include "amr_bitwise_lock_runner.h"

AMRLockTrace runAMRLockScenario_eso_twist()
{
	return runAMRLockScenario<AMRLockConfig<D3Q27_STREAMING_ESO_TWIST<TraitsSP>>>("eso_twist");
}
