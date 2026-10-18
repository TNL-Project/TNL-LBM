// scenario-runner translation unit of the nested cross-pattern AMR bitwise
// lock (test_amr_nested_lock): instantiates the three-level scenario pinned
// to the A-A pattern via the explicit D3Q27_STREAMING_AA type
#include "amr_nested_lock_runner.h"

AMRLockTrace runAMRNestedLockScenario_aa()
{
	return runAMRNestedLockScenario<AMRLockConfig<D3Q27_STREAMING_AA<TraitsSP>>>("aa");
}
