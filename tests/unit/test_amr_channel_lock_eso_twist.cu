// scenario-runner translation unit of the cross-pattern channel-geometry
// lock (test_amr_channel_lock): instantiates the two-footprint
// developing-channel scenario pinned to the esoteric-twist pattern via the
// explicit D3Q27_STREAMING_ESO_TWIST type
#include "amr_channel_lock_runner.h"

AMRChannelTrace runAMRChannelScenario_eso_twist()
{
	return runAMRChannelScenario<AMRChannelLockConfig<D3Q27_STREAMING_ESO_TWIST<TraitsSP>>>("eso_twist");
}
