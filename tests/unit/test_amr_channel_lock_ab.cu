// scenario-runner translation unit of the cross-pattern channel-geometry
// lock (test_amr_channel_lock): instantiates the two-footprint
// developing-channel scenario pinned to the A-B pull pattern via the
// explicit D3Q27_STREAMING_AB_PULL type — the reference instance every
// other pattern's trace is compared against
#include "amr_channel_lock_runner.h"

AMRChannelTrace runAMRChannelScenario_ab()
{
	return runAMRChannelScenario<AMRChannelLockConfig<D3Q27_STREAMING_AB_PULL<TraitsSP>>>("ab");
}
