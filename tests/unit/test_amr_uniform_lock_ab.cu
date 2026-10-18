// scenario-runner translation unit of the uniform cross-pattern AMR
// bitwise lock (test_amr_uniform_lock): instantiates the sim_AMR_ball
// uniform channel scenario pinned to the A-B pull pattern via the explicit
// D3Q27_STREAMING_AB_PULL type (the reference trace)
#include "amr_uniform_lock_runner.h"

AMRLockTrace runAMRUniformLockScenario_ab()
{
	return runAMRUniformLockScenario<AMRLockConfig<D3Q27_STREAMING_AB_PULL<TraitsSP>>>("ab");
}
