#include "../defs.h"

// NOTE: df_sync_directions must be kept consistent with this enum!
enum : std::uint8_t
{
	mmm,
	mmz,
	mmp,
	mzm,
	mzz,
	mzp,
	mpm,
	mpz,
	mpp,
	zmm,
	zmz,
	zmp,
	zzm,
	zzz,
	zzp,
	zpm,
	zpz,
	zpp,
	pmm,
	pmz,
	pmp,
	pzm,
	pzz,
	pzp,
	ppm,
	ppz,
	ppp
};

// array of sync directions for the MPI synchronizer
// (indexing must correspond to the enum above)
inline constexpr TNL::Containers::SyncDirection df_sync_directions[27] = {
	TNL::Containers::SyncDirection::BackBottomLeft,
	TNL::Containers::SyncDirection::BackBottom,
	TNL::Containers::SyncDirection::BackBottomRight,
	TNL::Containers::SyncDirection::BackLeft,
	TNL::Containers::SyncDirection::Back,
	TNL::Containers::SyncDirection::BackRight,
	TNL::Containers::SyncDirection::BackTopLeft,
	TNL::Containers::SyncDirection::BackTop,
	TNL::Containers::SyncDirection::BackTopRight,
	TNL::Containers::SyncDirection::BottomLeft,
	TNL::Containers::SyncDirection::Bottom,
	TNL::Containers::SyncDirection::BottomRight,
	TNL::Containers::SyncDirection::Left,
	TNL::Containers::SyncDirection::None,
	TNL::Containers::SyncDirection::Right,
	TNL::Containers::SyncDirection::TopLeft,
	TNL::Containers::SyncDirection::Top,
	TNL::Containers::SyncDirection::TopRight,
	TNL::Containers::SyncDirection::FrontBottomLeft,
	TNL::Containers::SyncDirection::FrontBottom,
	TNL::Containers::SyncDirection::FrontBottomRight,
	TNL::Containers::SyncDirection::FrontLeft,
	TNL::Containers::SyncDirection::Front,
	TNL::Containers::SyncDirection::FrontRight,
	TNL::Containers::SyncDirection::FrontTopLeft,
	TNL::Containers::SyncDirection::FrontTop,
	TNL::Containers::SyncDirection::FrontTopRight
};
