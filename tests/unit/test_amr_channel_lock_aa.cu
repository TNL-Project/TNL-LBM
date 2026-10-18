// scenario-runner translation unit of the cross-pattern channel-geometry
// lock (test_amr_channel_lock): instantiates the two-footprint
// developing-channel scenario pinned to the A-A pattern via the explicit
// D3Q27_STREAMING_AA type
#include "amr_channel_lock_runner.h"

AMRChannelTrace runAMRChannelScenario_aa()
{
	return runAMRChannelScenario<AMRChannelLockConfig<D3Q27_STREAMING_AA<TraitsSP>>>("aa");
}
