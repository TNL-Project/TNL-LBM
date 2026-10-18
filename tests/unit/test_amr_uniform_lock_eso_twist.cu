// scenario-runner translation unit of the uniform cross-pattern AMR
// bitwise lock (test_amr_uniform_lock): instantiates the sim_AMR_ball
// uniform channel scenario pinned to the esoteric twist pattern via the
// explicit D3Q27_STREAMING_ESO_TWIST type
#include "amr_uniform_lock_runner.h"

AMRLockTrace runAMRUniformLockScenario_eso_twist()
{
	return runAMRUniformLockScenario<AMRLockConfig<D3Q27_STREAMING_ESO_TWIST<TraitsSP>>>("eso_twist");
}
