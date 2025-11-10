#pragma once
#include "../defs.h"


// array of sync directions for the MPI synchronizer
// (indexing must correspond to the enum above)
inline constexpr TNL::Containers::SyncDirection df_sync_directions[27] = {
	TNL::Containers::SyncDirection::BackBottomLeft,
TNL::Containers::SyncDirection::BottomLeft,
TNL::Containers::SyncDirection::FrontBottomLeft,
TNL::Containers::SyncDirection::BackLeft,
TNL::Containers::SyncDirection::Left,
TNL::Containers::SyncDirection::FrontLeft,
TNL::Containers::SyncDirection::BackTopLeft,
TNL::Containers::SyncDirection::TopLeft,
TNL::Containers::SyncDirection::FrontTopLeft,
TNL::Containers::SyncDirection::BackBottom,
TNL::Containers::SyncDirection::Bottom,
TNL::Containers::SyncDirection::FrontBottom,
TNL::Containers::SyncDirection::Back,
TNL::Containers::SyncDirection::None,
TNL::Containers::SyncDirection::Front,
TNL::Containers::SyncDirection::BackTop,
TNL::Containers::SyncDirection::Top,
TNL::Containers::SyncDirection::FrontTop,
TNL::Containers::SyncDirection::BackBottomRight,
TNL::Containers::SyncDirection::BottomRight,
TNL::Containers::SyncDirection::FrontBottomRight,
TNL::Containers::SyncDirection::BackRight,
TNL::Containers::SyncDirection::Right,
TNL::Containers::SyncDirection::FrontRight,
TNL::Containers::SyncDirection::BackTopRight,
TNL::Containers::SyncDirection::TopRight,
TNL::Containers::SyncDirection::FrontTopRight
};
