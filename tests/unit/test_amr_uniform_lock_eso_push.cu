// scenario-runner translation unit of the uniform cross-pattern AMR
// bitwise lock (test_amr_uniform_lock): instantiates the sim_AMR_ball
// uniform channel scenario pinned to the esoteric push pattern via the
// explicit D3Q27_STREAMING_ESO_PUSH type
#include "amr_uniform_lock_runner.h"

AMRLockTrace runAMRUniformLockScenario_eso_push()
{
	return runAMRUniformLockScenario<AMRLockConfig<D3Q27_STREAMING_ESO_PUSH<TraitsSP>>>("eso_push");
}
