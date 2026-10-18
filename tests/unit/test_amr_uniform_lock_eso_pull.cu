// scenario-runner translation unit of the uniform cross-pattern AMR
// bitwise lock (test_amr_uniform_lock): instantiates the sim_AMR_ball
// uniform channel scenario pinned to the esoteric pull pattern via the
// explicit D3Q27_STREAMING_ESO_PULL type
#include "amr_uniform_lock_runner.h"

AMRLockTrace runAMRUniformLockScenario_eso_pull()
{
	return runAMRUniformLockScenario<AMRLockConfig<D3Q27_STREAMING_ESO_PULL<TraitsSP>>>("eso_pull");
}
