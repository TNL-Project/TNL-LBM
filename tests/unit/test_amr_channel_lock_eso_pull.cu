// scenario-runner translation unit of the cross-pattern channel-geometry
// lock (test_amr_channel_lock): instantiates the two-footprint
// developing-channel scenario pinned to the esoteric-pull pattern via the
// explicit D3Q27_STREAMING_ESO_PULL type
#include "amr_channel_lock_runner.h"

AMRChannelTrace runAMRChannelScenario_eso_pull()
{
	return runAMRChannelScenario<AMRChannelLockConfig<D3Q27_STREAMING_ESO_PULL<TraitsSP>>>("eso_pull");
}
