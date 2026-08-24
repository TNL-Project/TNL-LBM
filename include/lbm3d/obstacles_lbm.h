#pragma once

template <typename LBM>
void lbmDrawCube(LBM& lbm, typename LBM::map_t wall_tag, typename LBM::point_t phys_center, typename LBM::real phys_radius)
{
	typename LBM::idx3d lbm_center = lbm.lat.phys2lbmPoint(phys_center);
	typename LBM::real lbm_radius = phys_radius / lbm.lat.physDl;
	typename LBM::idx range = ceil(lbm_radius) + 1;
	for (typename LBM::idx py = lbm_center.y() - range; py <= lbm_center.y() + range; py++)
		for (typename LBM::idx pz = lbm_center.z() - range; pz <= lbm_center.z() + range; pz++)
			for (typename LBM::idx px = lbm_center.x() - range; px <= lbm_center.x() + range; px++) {
				const typename LBM::point_t p = lbm.lat.lbm2physPoint({px, py, pz});
				const typename LBM::point_t d = p - phys_center;
				if (std::abs(d.x()) < phys_radius && std::abs(d.y()) < phys_radius && std::abs(d.z()) < phys_radius) {
					lbm.setMap(px, py, pz, wall_tag);
				}
			}
}

template <typename LBM>
void lbmDrawSphere(LBM& lbm, typename LBM::map_t wall_tag, typename LBM::point_t phys_center, typename LBM::real phys_radius)
{
	typename LBM::idx3d lbm_center = lbm.lat.phys2lbmPoint(phys_center);
	typename LBM::real lbm_radius = phys_radius / lbm.lat.physDl;
	typename LBM::idx range = ceil(lbm_radius) + 1;
	for (typename LBM::idx py = lbm_center.y() - range; py <= lbm_center.y() + range; py++)
		for (typename LBM::idx pz = lbm_center.z() - range; pz <= lbm_center.z() + range; pz++)
			for (typename LBM::idx px = lbm_center.x() - range; px <= lbm_center.x() + range; px++) {
				const typename LBM::point_t p = lbm.lat.lbm2physPoint({px, py, pz});
				const typename LBM::real dist = TNL::l2Norm(p - phys_center);
				if (dist < phys_radius) {
					lbm.setMap(px, py, pz, wall_tag);
				}
			}
}

template <typename LBM>
void lbmDrawCylinder(LBM& lbm, typename LBM::map_t wall_tag, typename LBM::point_t phys_center, typename LBM::real phys_radius)
{
	typename LBM::idx3d lbm_center = lbm.lat.phys2lbmPoint(phys_center);
	typename LBM::real lbm_radius = phys_radius / lbm.lat.physDl;
	typename LBM::idx range = ceil(lbm_radius) + 1;
	for (typename LBM::idx py = 0; py <= lbm.lat.global.y() - 1; py++)
		for (typename LBM::idx pz = lbm_center.z() - range; pz <= lbm_center.z() + range; pz++)
			for (typename LBM::idx px = lbm_center.x() - range; px <= lbm_center.x() + range; px++) {
				const typename LBM::point_t p = lbm.lat.lbm2physPoint({px, py, pz});
				const typename LBM::real dx = p.x() - phys_center.x();
				const typename LBM::real dz = p.z() - phys_center.z();
				if (dx * dx + dz * dz < phys_radius * phys_radius) {
					lbm.setMap(px, py, pz, wall_tag);
				}
			}
}

template <typename LBM>
void lbmDrawBoundingBox(LBM& lbm, typename LBM::map_t wall_tag, typename LBM::point_t phys_point1, typename LBM::point_t phys_point2)
{
	// Both points are assumed to be wall coordinates and the wall is in the
	// middle between two lattice sites, so we shift the points by 1/2
	typename LBM::point_t lbm_point1 = lbm.lat.phys2lbmPoint(phys_point1);
	typename LBM::point_t lbm_point2 = lbm.lat.phys2lbmPoint(phys_point2);
	if (lbm_point1.x() < lbm_point2.x()) {
		lbm_point1.x() += 0.5f;
		lbm_point2.x() -= 0.5f;
	}
	else {
		lbm_point1.x() -= 0.5f;
		lbm_point2.x() += 0.5f;
	}
	if (lbm_point1.y() < lbm_point2.y()) {
		lbm_point1.y() += 0.5f;
		lbm_point2.y() -= 0.5f;
	}
	else {
		lbm_point1.y() -= 0.5f;
		lbm_point2.y() += 0.5f;
	}
	if (lbm_point1.z() < lbm_point2.z()) {
		lbm_point1.z() += 0.5f;
		lbm_point2.z() -= 0.5f;
	}
	else {
		lbm_point1.z() -= 0.5f;
		lbm_point2.z() += 0.5f;
	}

	for (typename LBM::idx py = 0; py <= std::round(std::abs(lbm_point1.y() - lbm_point2.y())); py++)
		for (typename LBM::idx pz = 0; pz <= std::round(std::abs(lbm_point1.z() - lbm_point2.z())); pz++)
			for (typename LBM::idx px = 0; px <= std::round(std::abs(lbm_point1.x() - lbm_point2.x())); px++)
				lbm.setMap(lbm_point1.x() + px, lbm_point1.y() + py, lbm_point1.z() + pz, wall_tag);
}

// TODO: add orientation parameter
template <typename LBM>
void lbmDrawCUBI(LBM& lbm, typename LBM::map_t wall_tag, typename LBM::point_t phys_center, typename LBM::real phys_edge_length)
{
	// bottom two cubes
	typename LBM::point_t phys_p1{phys_center.x() - phys_edge_length, phys_center.y() - phys_edge_length / 2, phys_center.z() - phys_edge_length};
	typename LBM::point_t phys_p2{phys_center.x() + phys_edge_length, phys_center.y() + phys_edge_length / 2, phys_center.z()};
	lbmDrawBoundingBox(lbm, wall_tag, phys_p1, phys_p2);

	// upper one cube
	typename LBM::point_t phys_p3{phys_center.x(), phys_center.y() - phys_edge_length / 2, phys_center.z()};
	typename LBM::point_t phys_p4{phys_center.x() + phys_edge_length, phys_center.y() + phys_edge_length / 2, phys_center.z() + phys_edge_length};
	lbmDrawBoundingBox(lbm, wall_tag, phys_p3, phys_p4);
}
