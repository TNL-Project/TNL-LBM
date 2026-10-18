// scenario-runner translation unit of the uniform cross-pattern AMR
// bitwise lock (test_amr_uniform_lock): instantiates the sim_AMR_ball
// uniform channel scenario pinned to the A-A pattern via the explicit
// D3Q27_STREAMING_AA type
#include "amr_uniform_lock_runner.h"

AMRLockTrace runAMRUniformLockScenario_aa()
{
	return runAMRUniformLockScenario<AMRLockConfig<D3Q27_STREAMING_AA<TraitsSP>>>("aa");
}
