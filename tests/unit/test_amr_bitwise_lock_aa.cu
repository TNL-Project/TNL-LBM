// scenario-runner translation unit of the cross-pattern AMR bitwise lock
// (test_amr_bitwise_lock): instantiates the two-level scenario pinned to the
// A-A pattern via the explicit D3Q27_STREAMING_AA type
#include "amr_bitwise_lock_runner.h"

AMRLockTrace runAMRLockScenario_aa()
{
	return runAMRLockScenario<AMRLockConfig<D3Q27_STREAMING_AA<TraitsSP>>>("aa");
}
