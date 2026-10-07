#pragma once

#include "defs.h"  // IWYU pragma: keep (provides KernelStruct and idx types used by the kernels)
#include "lagrange_3D.h"
#include "dirac.h"

template <typename LBM>
__global__ void dM_row_capacities_kernel(
	typename Lagrange3D<LBM>::DLPVECTOR::ConstViewType LL,
	typename Lagrange3D<LBM>::dEllpack::RowCapacitiesType::ViewType dM_row_capacities,
	typename LBM::TRAITS::idx3d lbmBlockLocal,
	int diracDeltaTypeEL
)
{
#ifdef __CUDACC__
	using idx = typename LBM::TRAITS::idx;
	using real = typename Lagrange3D<LBM>::DLPVECTOR::RealType::RealType;

	const idx support = 5;	// search in this support

	idx i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= LL.getSize())
		return;

	idx rowCapacity = 0;

	idx fi_x = floor(LL[i].x() - (real) 0.5);
	idx fi_y = floor(LL[i].y() - (real) 0.5);
	idx fi_z = floor(LL[i].z() - (real) 0.5);

	// FIXME: iterate over LBM blocks
	for (idx gz = TNL::max(0, fi_z - support); gz < TNL::min(lbmBlockLocal.z(), fi_z + support); gz++)
		for (idx gy = TNL::max(0, fi_y - support); gy < TNL::min(lbmBlockLocal.y(), fi_y + support); gy++)
			for (idx gx = TNL::max(0, fi_x - support); gx < TNL::min(lbmBlockLocal.x(), fi_x + support); gx++) {
				if (isDDNonZero(diracDeltaTypeEL, gx - LL[i].x()) && isDDNonZero(diracDeltaTypeEL, gy - LL[i].y())
					&& isDDNonZero(diracDeltaTypeEL, gz - LL[i].z()))
				{
					rowCapacity++;
				}
			}

	dM_row_capacities[i] = rowCapacity;
#endif
}

template <typename LBM>
__global__ void dM_construction_kernel(
	typename Lagrange3D<LBM>::DLPVECTOR::ConstViewType LL,
	typename Lagrange3D<LBM>::dEllpack::ViewType ws_tnl_dM,
	typename LBM::TRAITS::idx3d lbmBlockLocal,
#ifdef HAVE_MPI
	typename LBM::BLOCK::dmap_array_t::ConstLocalViewType dmap,
#else
	typename LBM::BLOCK::dmap_array_t::ConstViewType dmap,
#endif
	int diracDeltaTypeEL
)
{
#ifdef __CUDACC__
	using idx = typename LBM::TRAITS::idx;
	using real = typename Lagrange3D<LBM>::DLPVECTOR::RealType::RealType;

	const idx support = 5;	// search in this support

	idx i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= LL.getSize())
		return;

	auto row = ws_tnl_dM.getRow(i);
	idx element_idx = 0;

	idx fi_x = floor(LL[i].x() - (real) 0.5);
	idx fi_y = floor(LL[i].y() - (real) 0.5);
	idx fi_z = floor(LL[i].z() - (real) 0.5);

	// FIXME: iterate over LBM blocks
	for (idx gz = TNL::max(0, fi_z - support); gz < TNL::min(lbmBlockLocal.z(), fi_z + support); gz++)
		for (idx gy = TNL::max(0, fi_y - support); gy < TNL::min(lbmBlockLocal.y(), fi_y + support); gy++)
			for (idx gx = TNL::max(0, fi_x - support); gx < TNL::min(lbmBlockLocal.x(), fi_x + support); gx++) {
				if (isDDNonZero(diracDeltaTypeEL, gx - LL[i].x()) && isDDNonZero(diracDeltaTypeEL, gy - LL[i].y())
					&& isDDNonZero(diracDeltaTypeEL, gz - LL[i].z()))
				{
					real dd = diracDelta(diracDeltaTypeEL, gx - LL[i].x()) * diracDelta(diracDeltaTypeEL, gy - LL[i].y())
							* diracDelta(diracDeltaTypeEL, gz - LL[i].z());
					idx index = dmap.getStorageIndex(gx, gy, gz);
					row.setElement(element_idx++, index, dd);
				}
			}
#endif
}

template <typename LBM>
__global__ void dA_row_capacities_kernel(
	typename Lagrange3D<LBM>::DLPVECTOR::ConstViewType LL,
	typename Lagrange3D<LBM>::dEllpack::RowCapacitiesType::ViewType dA_row_capacities,
	typename Lagrange3D<LBM>::dEllpack::ConstViewType ws_tnl_dM,
	int diracDeltaTypeLL,
	IbmMethod methodVariant
)
{
#ifdef __CUDACC__
	using idx = typename LBM::TRAITS::idx;
	using real = typename Lagrange3D<LBM>::DLPVECTOR::RealType::RealType;

	idx index_row = blockIdx.x * blockDim.x + threadIdx.x;
	idx m = LL.getSize();
	if (index_row >= m)
		return;

	idx rowCapacity = 0;								 //Number of elements where DiracDelta > 0 //old
	for (idx index_col = 0; index_col < m; index_col++)	 //old
	{
		if (methodVariant == IbmMethod::modified) {
			if (is3DiracNonZero(diracDeltaTypeLL, index_col, index_row, LL)) {
				rowCapacity++;
			}
		}
		else {
			real val = 0;
			auto row1 = ws_tnl_dM.getRow(index_row);
			auto row2 = ws_tnl_dM.getRow(index_col);
			for (idx in1 = 0; in1 < row1.getSize(); in1++)
				for (idx in2 = 0; in2 < row2.getSize(); in2++) {
					if (row1.getColumnIndex(in1) == row2.getColumnIndex(in2)) {
						val += row1.getValue(in1) * row2.getValue(in2);
						break;
					}
				}
			if (val > 0)
				rowCapacity++;
		}
	}
	dA_row_capacities[index_row] = rowCapacity;
#endif
}

template <typename LBM>
__global__ void dA_construction_kernel(
	typename Lagrange3D<LBM>::DLPVECTOR::ConstViewType LL,
	typename Lagrange3D<LBM>::dEllpack::ViewType ws_tnl_dA,
	typename Lagrange3D<LBM>::dEllpack::ConstViewType ws_tnl_dM,
	int diracDeltaTypeLL,
	IbmMethod methodVariant
)
{
#ifdef __CUDACC__
	using idx = typename LBM::TRAITS::idx;
	using real = typename Lagrange3D<LBM>::DLPVECTOR::RealType::RealType;

	idx index_row = blockIdx.x * blockDim.x + threadIdx.x;
	idx m = LL.getSize();
	if (index_row >= m)
		return;

	auto row = ws_tnl_dA.getRow(index_row);
	idx element_idx = 0;

	for (idx index_col = 0; index_col < m; index_col++) {
		if (methodVariant == IbmMethod::modified) {
			if (is3DiracNonZero(diracDeltaTypeLL, index_col, index_row, LL)) {
				real ddd = calculate3Dirac(diracDeltaTypeLL, index_col, index_row, LL);
				row.setElement(element_idx++, index_col, ddd);
				if (element_idx == row.getSize())
					break;
			}
		}
		else {
			real val = 0;
			auto row1 = ws_tnl_dM.getRow(index_row);
			auto row2 = ws_tnl_dM.getRow(index_col);
			for (idx in1 = 0; in1 < row1.getSize(); in1++)
				for (idx in2 = 0; in2 < row2.getSize(); in2++) {
					if (row1.getColumnIndex(in1) == row2.getColumnIndex(in2)) {
						val += row1.getValue(in1) * row2.getValue(in2);
						break;
					}
				}
			if (val > 0) {
				row.setElement(element_idx++, index_col, val);
				if (element_idx == row.getSize())
					break;
			}
		}
	}
#endif
}
