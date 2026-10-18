// scenario-runner translation unit of the cross-pattern channel-geometry
// lock (test_amr_channel_lock): instantiates the two-footprint
// developing-channel scenario pinned to the esoteric-push pattern via the
// explicit D3Q27_STREAMING_ESO_PUSH type
#include "amr_channel_lock_runner.h"

AMRChannelTrace runAMRChannelScenario_eso_push()
{
	return runAMRChannelScenario<AMRChannelLockConfig<D3Q27_STREAMING_ESO_PUSH<TraitsSP>>>("eso_push");
}
