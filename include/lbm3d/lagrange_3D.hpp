#pragma once

#include <TNL/Timer.h>
#include <TNL/Backend/KernelLaunch.h>
#include <TNL/Backend/Macros.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Devices/Host.h>
#include <TNL/DiscreteMath.h>
#include <TNL/Matrices/MatrixWriter.h>

#include <cstddef>
#include <string>
#include <omp.h>
#include <fmt/core.h>
#include <nlohmann/json.hpp>

#include "lbm_common/logging.h"
#include "lagrange_3D.h"
#include "ibm_kernels.h"
#include "dirac.h"

template< typename LBM >
auto Lagrange3D<LBM>::hmacroVector(int macro_idx) -> hVectorView
{
	if (macro_idx >= MACRO::N)
		throw std::logic_error("macro_idx must be less than MACRO::N");

	auto& block = lbm.blocks.front();

	// local size of the distributed array
	// FIXME: overlaps
	const idx n = block.local.x() * block.local.y() * block.local.z();

	// offset for the requested quantity
	// FIXME: overlaps
	const idx quantity_offset = block.hmacro.getStorageIndex(macro_idx, block.offset.x(), block.offset.y(), block.offset.z());

	// pointer to the data
	dreal* data = block.hmacro.getData() + quantity_offset;

	TNL_ASSERT_EQ(data, &block.hmacro(macro_idx, block.offset.x(), block.offset.y(), block.offset.z()), "indexing bug");

	return {data, n};
}

template< typename LBM >
auto Lagrange3D<LBM>::dmacroVector(int macro_idx) -> dVectorView
{
	if (macro_idx >= MACRO::N)
		throw std::logic_error("macro_idx must be less than MACRO::N");

	auto& block = lbm.blocks.front();

	// local size of the distributed array
	// FIXME: overlaps
	const idx n = block.local.x() * block.local.y() * block.local.z();

	// offset for the requested quantity
	// FIXME: overlaps
	const idx quantity_offset = block.dmacro.getStorageIndex(macro_idx, block.offset.x(), block.offset.y(), block.offset.z());

	// pointer to the data
	dreal* data = block.dmacro.getData() + quantity_offset;

	return {data, n};
}

template< typename LBM >
typename LBM::TRAITS::real Lagrange3D<LBM>::computeMinDist()
{
	minDist=1e10;
	maxDist=-1e10;
	for (std::size_t i=0;i<LL.size()-1;i++)
	{
		for (std::size_t j=i+1;j<LL.size();j++)
		{
			real d = TNL::l2Norm(LL[j] - LL[i]);
			if (d>maxDist) maxDist=d;
			if (d<minDist) minDist=d;
		}
	}
	return minDist;
}

template< typename LBM >
typename LBM::TRAITS::real Lagrange3D<LBM>::computeMaxDistFromMinDist(typename LBM::TRAITS::real mindist)
{
	maxDist=-1e10;
	for (std::size_t i=0;i<LL.size()-1;i++)
	{
		real search_dist = 2.0*mindist;
		for (std::size_t j=i+1;j<LL.size();j++)
		{
			real d = TNL::l2Norm(LL[j] - LL[i]);
			if (d < search_dist)
				if (d>maxDist) maxDist=d;
		}
	}
	return maxDist;
}


template< typename LBM >
void Lagrange3D<LBM>::convertLagrangianPoints()
{
	hLL_lat.setSize(LL.size());

	TNL::Algorithms::parallelFor< TNL::Devices::Host >(
		(idx) 0,
		(idx) LL.size(),
		[this] (idx i, typename LBM::lat_t lat) mutable
		{
			this->hLL_lat[i] = lat.phys2lbmPoint(this->LL[i]);
		},
		lbm.lat
	);

	hLL_velocity_lat.setSize(LL.size());
	hLL_velocity_lat = 0;
}

template< typename LBM >
void Lagrange3D<LBM>::allocateMatricesCPU()
{
	idx m = LL.size();	// number of lagrangian nodes
	idx n = lbm.lat.global.x()*lbm.lat.global.y()*lbm.lat.global.z();	// number of eulerian nodes

	spdlog::info("number of Lagrangian points: {}", m);

	// Convert Lagrangian points to lattice coordinates and to StaticVector with dreal
	convertLagrangianPoints();

	// Allocate matrices
	ws_tnl_hM.setDimensions(m, n);
	ws_tnl_hA = std::make_shared< hEllpack >();
	ws_tnl_hA->setDimensions(m, m);
	#ifdef USE_CUDA
	ws_tnl_dM.setDimensions(m, n);
	ws_tnl_dA = std::make_shared< dEllpack >();
	ws_tnl_dA->setDimensions(m, m);
	#endif

	// Allocate row capacity vectors
	hM_row_capacities.setSize(m);
	hA_row_capacities.setSize(m);

	// Create vectors for the solution of the linear system
	for (int k=0;k<3;k++)
	{
		ws_tnl_hx[k].setSize(m);
		ws_tnl_hb[k].setSize(m);
		#ifdef USE_CUDA
		ws_tnl_dx[k].setSize(m);
		ws_tnl_db[k].setSize(m);
		ws_tnl_hxz[k].setSize(m);
		ws_tnl_hbz[k].setSize(m);
		#endif
	}

	// Zero-initialize x1, x2, x3
	for (int k=0;k<3;k++) ws_tnl_hx[k].setValue(0);
	#ifdef USE_CUDA
	for (int k=0;k<3;k++) ws_tnl_dx[k].setValue(0);
	for (int k=0;k<3;k++) ws_tnl_hxz[k].setValue(0);
	#endif
}

template< typename LBM >
void Lagrange3D<LBM>::constructMatricesCPU()
{
	auto ibm_logger = spdlog::get("ibm");

	// Start timer to measure WuShu construction time
	TNL::Timer timer;
	TNL::Timer loopTimer;

	double time_M_capacities = 0;
	double time_M_construct = 0;
	double time_M_transpose = 0;
	double time_A_capacities = 0;
	double time_A_construct = 0;
	double time_matrixWrite = 0;
	double time_matrixCopy = 0;

	idx m = LL.size();	// number of lagrangian nodes

	timer.start();
	loopTimer.start();
	// Calculate row capacities of hM
	const idx support = 5; // search in this support
	#pragma omp parallel for schedule(static)
	for (idx i = 0; i < m; i++)
	{
		idx rowCapacity = 0;

		idx fi_x = floor(hLL_lat[i].x() - (dreal)0.5);
		idx fi_y = floor(hLL_lat[i].y() - (dreal)0.5);
		idx fi_z = floor(hLL_lat[i].z() - (dreal)0.5);

		// FIXME: iterate over LBM blocks
		for (idx gz = MAX(0, fi_z - support); gz < MIN(lbm.blocks.front().local.z(), fi_z + support); gz++)
		for (idx gy = MAX(0, fi_y - support); gy < MIN(lbm.blocks.front().local.y(), fi_y + support); gy++)
		for (idx gx = MAX(0, fi_x - support); gx < MIN(lbm.blocks.front().local.x(), fi_x + support); gx++)
		{
			if (
				isDDNonZero(diracDeltaTypeEL, gx - hLL_lat[i].x()) &&
				isDDNonZero(diracDeltaTypeEL, gy - hLL_lat[i].y()) &&
				isDDNonZero(diracDeltaTypeEL, gz - hLL_lat[i].z())
			)
			{
				rowCapacity++;
			}
		}

		hM_row_capacities[i] = rowCapacity;
	}

	loopTimer.stop();
	time_M_capacities = loopTimer.getRealTime();

	ws_tnl_hM.setRowCapacities(hM_row_capacities);

	loopTimer.reset();
	loopTimer.start();
	// Construct the matrix hM
	#pragma omp parallel for schedule(static)
	for (idx i = 0; i < m; i++)
	{
		auto row = ws_tnl_hM.getRow(i);
		idx element_idx = 0;

		idx fi_x = floor(hLL_lat[i].x() - (dreal)0.5);
		idx fi_y = floor(hLL_lat[i].y() - (dreal)0.5);
		idx fi_z = floor(hLL_lat[i].z() - (dreal)0.5);

		// FIXME: iterate over LBM blocks
		for (idx gz = MAX(0, fi_z - support); gz < MIN(lbm.blocks.front().local.z(), fi_z + support); gz++)
		for (idx gy = MAX(0, fi_y - support); gy < MIN(lbm.blocks.front().local.y(), fi_y + support); gy++)
		for (idx gx = MAX(0, fi_x - support); gx < MIN(lbm.blocks.front().local.x(), fi_x + support); gx++)
		{
			if (
				isDDNonZero(diracDeltaTypeEL, gx - hLL_lat[i].x()) &&
				isDDNonZero(diracDeltaTypeEL, gy - hLL_lat[i].y()) &&
				isDDNonZero(diracDeltaTypeEL, gz - hLL_lat[i].z())
			)
			{
				dreal dd =
					diracDelta(diracDeltaTypeEL, gx - hLL_lat[i].x()) *
					diracDelta(diracDeltaTypeEL, gy - hLL_lat[i].y()) *
					diracDelta(diracDeltaTypeEL, gz - hLL_lat[i].z());
				idx index = lbm.blocks.front().hmap.getStorageIndex(gx,gy,gz);
				row.setElement(element_idx++, index, dd);
			}
		}
	}
	loopTimer.stop();
	time_M_construct = loopTimer.getRealTime();

	// Transpose matrix M
	loopTimer.reset();
	loopTimer.start();
    ws_tnl_hMT.getTransposition(ws_tnl_hM);
	loopTimer.stop();
	time_M_transpose = loopTimer.getRealTime();

	int threads = omp_get_max_threads();
	// TODO: find the correct threshold for this condition
	//if( m < 1000 )
	//	threads = 1;
	(void) threads; // shut up clang warning

	loopTimer.reset();
	loopTimer.start();
	// Calculate row capacities of matrix A
	#pragma omp parallel for schedule(dynamic) num_threads(threads)
	for (idx index_row = 0; index_row < m; index_row++)
	{
		idx rowCapacity = 0;
		for (idx index_col = 0; index_col < m; index_col++)
		{
			if (methodVariant == IbmMethod::modified)
			{
				if(is3DiracNonZero(diracDeltaTypeLL, index_col, index_row, hLL_lat))
				{
					rowCapacity++;
				}
			}
			else
			{
				real val=0;
				auto row1 = ws_tnl_hM.getRow(index_row);
				auto row2 = ws_tnl_hM.getRow(index_col);
				for (idx in1=0; in1 < row1.getSize(); in1++)
				{
					for (idx in2=0; in2 < row2.getSize(); in2++)
					{
						if (row1.getColumnIndex(in1) == row2.getColumnIndex(in2))
						{
							val += row1.getValue(in1) * row2.getValue(in2);
							break;
						}
					}
				}
				if (val > 0)
					rowCapacity++;
			}
		}
		hA_row_capacities[index_row] = rowCapacity;
	}


	loopTimer.stop();
	time_A_capacities = loopTimer.getRealTime();
	ws_tnl_hA->setRowCapacities(hA_row_capacities);


	loopTimer.reset();
	loopTimer.start();
	// Construct the matrix A
	#pragma omp parallel for schedule(dynamic) num_threads(threads)
	for (idx index_row=0;index_row<m;index_row++)
	{
		auto row = ws_tnl_hA->getRow(index_row);
		idx element_idx = 0;

		for (idx index_col=0;index_col<m;index_col++)
		{
			if (methodVariant == IbmMethod::modified)
			{
				if(is3DiracNonZero(diracDeltaTypeLL, index_col, index_row, hLL_lat))
				{
					dreal ddd = calculate3Dirac(diracDeltaTypeLL, index_col, index_row, hLL_lat);
					row.setElement(element_idx++, index_col, ddd);
					if (element_idx == row.getSize())
						break;
				}
			} else
			{
				dreal val=0;
				auto row1 = ws_tnl_hM.getRow(index_row);
				auto row2 = ws_tnl_hM.getRow(index_col);
				for (idx in1=0; in1 < row1.getSize(); in1++)
				{
					for (idx in2=0; in2 < row2.getSize(); in2++)
					{
						if (row1.getColumnIndex(in1) == row2.getColumnIndex(in2))
						{
							val += row1.getValue(in1) * row2.getValue(in2);
							break;
						}
					}
				}
				if (val > 0) {
					row.setElement(element_idx++, index_col, val);
					if (element_idx == row.getSize())
						break;
				}
			}
		}
	}
	loopTimer.stop();
	time_A_construct = loopTimer.getRealTime();


	// Update the preconditioner
	ws_tnl_hprecond->update(ws_tnl_hA);
	ws_tnl_hsolver.setMatrix(ws_tnl_hA);

	#ifdef USE_CUDA
	// Copy matrices from host to the GPU
	loopTimer.reset();
	loopTimer.start();
	*ws_tnl_dA = *ws_tnl_hA;
	ws_tnl_dM = ws_tnl_hM;
	ws_tnl_dMT = ws_tnl_hMT;
	loopTimer.stop();
	time_matrixCopy = loopTimer.getRealTime();

	// Update the preconditioner
	ws_tnl_dprecond->update(ws_tnl_dA);
	ws_tnl_dsolver.setMatrix(ws_tnl_dA);
	#endif

	if (mtx_output)
	{
		loopTimer.reset();
		loopTimer.start();
		const std::string output_M = fmt::format("ibm_CPU_matrix-M_method-{}_dirac-{}.mtx", (int)methodVariant, (int)diracDeltaTypeEL);
		const std::string output_A = fmt::format("ibm_CPU_matrix-A_method-{}_dirac-{}.mtx", (int)methodVariant, (int)diracDeltaTypeEL);
		TNL::Matrices::MatrixWriter< hEllpack >::writeMtx( output_M, ws_tnl_hM );
		TNL::Matrices::MatrixWriter< hEllpack >::writeMtx( output_A, *ws_tnl_hA );
		loopTimer.stop();
		time_matrixWrite = loopTimer.getRealTime();
	}

	timer.stop();
	nlohmann::json j;
	j["threads"] = omp_get_max_threads();
	j["time_total"] = timer.getRealTime();
	j["time_A_capacities"] = time_A_capacities;
	j["time_A_construct"] = time_A_construct;
	j["time_M_capacities"] = time_M_capacities;
	j["time_M_construct"] = time_M_construct;
	j["time_M_transpose"] = time_M_transpose;
	j["time_matrixWrite"] = time_matrixWrite;
	j["time_matrixCopy"] = time_matrixCopy;
	ibm_logger->info("constructMatricesJSON: {}", j.dump());
}

template< typename LBM >
void Lagrange3D<LBM>::allocateMatricesGPU()
{
	idx m = LL.size();	// number of lagrangian nodes
	idx n = lbm.lat.global.x()*lbm.lat.global.y()*lbm.lat.global.z();	// number of eulerian nodes

	spdlog::info("number of Lagrangian points: {}", m);

	// Convert Lagrangian points to lattice coordinates and to StaticVector with dreal
	convertLagrangianPoints();
	dLL_lat = hLL_lat;
	dLL_velocity_lat.setSize(m);
	dLL_velocity_lat = 0;

	// Allocate matrices
	ws_tnl_dM.setDimensions(m, n);
	ws_tnl_dA = std::make_shared< dEllpack >();
	ws_tnl_dA->setDimensions(m, m);

	// Allocate row capacity vectors
	dM_row_capacities.setSize(m);
	dA_row_capacities.setSize(m);

	// Create vectors for the solution of the linear system
	for (int k=0;k<3;k++)
	{
		#ifdef USE_CUDA
		ws_tnl_dx[k].setSize(m);
		ws_tnl_db[k].setSize(m);
		#endif
	}

	// Zero-initialize x1, x2, x3
	#ifdef USE_CUDA
	for (int k=0;k<3;k++) ws_tnl_dx[k].setValue(0);
	#endif
}

template< typename LBM >
void Lagrange3D<LBM>::constructMatricesGPU()
{
	auto ibm_logger = spdlog::get("ibm");

	// Start timer to measure WuShu construction time
	TNL::Timer timer;
	TNL::Timer loopTimer;

	double time_M_capacities = 0;
	double time_M_construct = 0;
	double time_M_transpose = 0;
	double time_A_capacities = 0;
	double time_A_construct = 0;
	double time_matrixWrite = 0;
	double time_matrixCopy = 0;

	idx m = LL.size();	// number of lagrangian nodes

	timer.start();
	loopTimer.start();
	// Calculate row capacities of matrix M
	TNL::Backend::LaunchConfiguration dM_config;
	dM_config.blockSize.x = 256;
	dM_config.gridSize.x = TNL::roundUpDivision(m,dM_config.blockSize.x);
	TNL::Backend::launchKernelSync( dM_row_capacities_kernel<LBM>, dM_config,
		dLL_lat.getView(),
		dM_row_capacities.getView(),
		lbm.blocks.front().local,
		diracDeltaTypeEL);

	loopTimer.stop();
	time_M_capacities = loopTimer.getRealTime();

	ws_tnl_dM.setRowCapacities(dM_row_capacities);

	loopTimer.reset();
	loopTimer.start();
	// Construct the matrix M
	TNL::Backend::LaunchConfiguration dM_config_build;
	dM_config_build.blockSize.x = 256;
	dM_config_build.gridSize.x = TNL::roundUpDivision(m,dM_config_build.blockSize.x);

	TNL::Backend::launchKernelSync( dM_construction_kernel<LBM>, dM_config_build,
		dLL_lat.getConstView(),
		ws_tnl_dM.getView(),
		lbm.blocks.front().local,
		#ifdef HAVE_MPI
		lbm.blocks.front().dmap.getConstLocalView(),
		#else
		lbm.blocks.front().dmap.getConstView(),
		#endif
		diracDeltaTypeEL);

	loopTimer.stop();
	time_M_construct = loopTimer.getRealTime();

	// Transpose matrix M
	loopTimer.reset();
	loopTimer.start();
    ws_tnl_dMT.getTransposition(ws_tnl_dM);
	loopTimer.stop();
	time_M_transpose = loopTimer.getRealTime();

	loopTimer.reset();
	loopTimer.start();
	// Calculate row capacities of matrix A
	TNL::Backend::LaunchConfiguration dA_config;
	dA_config.blockSize.x = 256;
	dA_config.gridSize.x = TNL::roundUpDivision(m, dA_config.blockSize.x);
	TNL::Backend::launchKernelSync(dA_row_capacities_kernel<LBM>, dA_config,
		dLL_lat.getConstView(),
		dA_row_capacities.getView(),
		ws_tnl_dM.getConstView(),
		diracDeltaTypeLL,
		methodVariant
		);

	loopTimer.stop();
	time_A_capacities = loopTimer.getRealTime();
	ws_tnl_dA->setRowCapacities(dA_row_capacities);


	loopTimer.reset();
	loopTimer.start();
	// Construct the matrix A
	TNL::Backend::LaunchConfiguration dA_construct_config;
	dA_construct_config.blockSize.x =256;
	dA_construct_config.gridSize.x = TNL::roundUpDivision(m,dA_construct_config.blockSize.x);
	TNL::Backend::launchKernelSync(dA_construction_kernel<LBM>, dA_construct_config,
		dLL_lat.getConstView(),
		ws_tnl_dA->getView(),
		ws_tnl_dM.getConstView(),
		diracDeltaTypeLL,
		methodVariant
		);

	loopTimer.stop();
	time_A_construct = loopTimer.getRealTime();


	// Update the preconditioner
	ws_tnl_dprecond->update(ws_tnl_dA);
	ws_tnl_dsolver.setMatrix(ws_tnl_dA);

	if (mtx_output)
	{
		loopTimer.reset();
		loopTimer.start();
		const std::string output_M = fmt::format("ibm_GPU_matrix-M_method-{}_dirac-{}.mtx", (int)methodVariant, (int)diracDeltaTypeEL);
		const std::string output_A = fmt::format("ibm_GPU_matrix-A_method-{}_dirac-{}.mtx", (int)methodVariant, (int)diracDeltaTypeEL);
		TNL::Matrices::MatrixWriter< dEllpack >::writeMtx( output_M, ws_tnl_dM );
		TNL::Matrices::MatrixWriter< dEllpack >::writeMtx( output_A, *ws_tnl_dA );
		loopTimer.stop();
		time_matrixWrite = loopTimer.getRealTime();
	}

	timer.stop();
	nlohmann::json j;
	j["threads"] = omp_get_max_threads();
	j["time_total"] = timer.getRealTime();
	j["time_A_capacities"] = time_A_capacities;
	j["time_A_construct"] = time_A_construct;
	j["time_M_capacities"] = time_M_capacities;
	j["time_M_construct"] = time_M_construct;
	j["time_M_transpose"] = time_M_transpose;
	j["time_matrixWrite"] = time_matrixWrite;
	j["time_matrixCopy"] = time_matrixCopy;
	ibm_logger->info("constructMatricesJSON: {}", j.dump());
}

template< typename Matrix, typename Vector >
__cuda_callable__
typename Matrix::RealType
rowVectorProduct( const Matrix& matrix, typename Matrix::IndexType i, const Vector& vector )
{
    typename Matrix::RealType result = 0;
    const auto row = matrix.getRow( i );

    for( typename Matrix::IndexType c = 0; c < row.getSize(); c++ ) {
        const typename Matrix::IndexType column = row.getColumnIndex( c );
        if( column != TNL::Matrices::paddingIndex< typename Matrix::IndexType > )
            result += row.getValue( c ) * vector[ column ];
    }

    return result;
}

//require: rho, vx, vy, vz
template< typename LBM >
void Lagrange3D<LBM>::computeForces(real time)
{
	const char* compute_desc = "undefined";
	switch (computeVariant)
	{
		case IbmCompute::CPU:                compute_desc = "CPU"; break;
		case IbmCompute::GPU:                compute_desc = "GPU"; break;
		case IbmCompute::Hybrid:             compute_desc = "Hybrid"; break;
		case IbmCompute::Hybrid_zerocopy:    compute_desc = "Hybrid_zerocopy"; break;
	}

	switch (computeVariant)
	{
		case IbmCompute::CPU:
		case IbmCompute::Hybrid:
		case IbmCompute::Hybrid_zerocopy:
			if (!allocated)
				allocateMatricesCPU();
			allocated = true;
			if (!constructed)
				constructMatricesCPU();
			constructed = true;
			break;
		case IbmCompute::GPU:
			if (!allocated)
				allocateMatricesGPU();
			allocated = true;
			if (!constructed)
				constructMatricesGPU();
			constructed = true;
			break;
	}

	spdlog::warn("computing implicit IBM forces is disabled");
	return;

	auto ibm_logger = spdlog::get("ibm");
	ibm_logger->info("computing forces using computeVariant={}", compute_desc);

	TNL::Timer timer;
	timer.start();
	idx m = LL.size();
	idx n = lbm.lat.global.x()*lbm.lat.global.y()*lbm.lat.global.z();

	const auto drho = dmacroVector(MACRO::e_rho);
	const auto dvx = dmacroVector(MACRO::e_vx);
	const auto dvy = dmacroVector(MACRO::e_vy);
	const auto dvz = dmacroVector(MACRO::e_vz);
	auto dfx = dmacroVector(MACRO::e_fx);
	auto dfy = dmacroVector(MACRO::e_fy);
	auto dfz = dmacroVector(MACRO::e_fz);

	const auto hrho = hmacroVector(MACRO::e_rho);
	const auto hvx = hmacroVector(MACRO::e_vx);
	const auto hvy = hmacroVector(MACRO::e_vy);
	const auto hvz = hmacroVector(MACRO::e_vz);
	auto hfx = hmacroVector(MACRO::e_fx);
	auto hfy = hmacroVector(MACRO::e_fy);
	auto hfz = hmacroVector(MACRO::e_fz);

	switch (computeVariant)
	{
		#ifdef USE_CUDA
		case IbmCompute::GPU:
		{
			// no Device--Host copy is required
			ws_tnl_dM.vectorProduct(dvx, ws_tnl_db[0], -1.0);
			ws_tnl_dM.vectorProduct(dvy, ws_tnl_db[1], -1.0);
			ws_tnl_dM.vectorProduct(dvz, ws_tnl_db[2], -1.0);
			if (use_LL_velocity_in_solution) {
				auto dbx = ws_tnl_db[0].getView();
				auto dby = ws_tnl_db[1].getView();
				auto dbz = ws_tnl_db[2].getView();
				const auto LL_velocity_lat = dLL_velocity_lat.getConstView();
				auto kernel = [=] CUDA_HOSTDEV (idx i) mutable
				{
					const point_t v = LL_velocity_lat[i];
					dbx[i] += v.x();
					dby[i] += v.y();
					dbz[i] += v.z();
				};
				TNL::Algorithms::parallelFor< TNL::Devices::Cuda >((idx) 0, m, kernel);
			}
			// solver
			for (int k=0;k<3;k++) {
				auto start = std::chrono::steady_clock::now();
				ws_tnl_dsolver.solve(ws_tnl_db[k], ws_tnl_dx[k]);
				auto end = std::chrono::steady_clock::now();
				auto int_ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
				real WT = int_ms * 1e-6;
				ibm_logger->info("t={:e}s k={:d} TNL CG solver: WT={:e} iterations={:d} residual={:e}", time, k, WT, ws_tnl_dsolver.getIterations(), ws_tnl_dsolver.getResidue());
			}
			const auto x1 = ws_tnl_dx[0].getConstView();
			const auto x2 = ws_tnl_dx[1].getConstView();
			const auto x3 = ws_tnl_dx[2].getConstView();
			//TNL::Pointers::DevicePointer<dEllpack> MT(ws_tnl_dMT);
			TNL::Pointers::DevicePointer<dEllpack> MT_dptr(ws_tnl_dMT);
			const dEllpack* MT = &MT_dptr.template getData<TNL::Devices::Cuda>();
			auto kernel = [=] CUDA_HOSTDEV (idx i) mutable
			{
				// skipping empty rows explicitly is much faster
				if( MT->getRowCapacity(i) > 0 ) {
					dfx[i] += 2 * drho[i] * rowVectorProduct(*MT, i, x1);
					dfy[i] += 2 * drho[i] * rowVectorProduct(*MT, i, x2);
					dfz[i] += 2 * drho[i] * rowVectorProduct(*MT, i, x3);
				}
			};
			TNL::Algorithms::parallelFor< TNL::Devices::Cuda >((idx) 0, n, kernel);
			break;
		}

		case IbmCompute::Hybrid:
		{
			ws_tnl_dM.vectorProduct(dvx, ws_tnl_db[0], -1.0);
			ws_tnl_dM.vectorProduct(dvy, ws_tnl_db[1], -1.0);
			ws_tnl_dM.vectorProduct(dvz, ws_tnl_db[2], -1.0);
			if (use_LL_velocity_in_solution) {
				auto dbx = ws_tnl_db[0].getView();
				auto dby = ws_tnl_db[1].getView();
				auto dbz = ws_tnl_db[2].getView();
				const auto LL_velocity_lat = dLL_velocity_lat.getConstView();
				auto kernel = [=] CUDA_HOSTDEV (idx i) mutable
				{
					const point_t v = LL_velocity_lat[i];
					dbx[i] += v.x();
					dby[i] += v.y();
					dbz[i] += v.z();
				};
				TNL::Algorithms::parallelFor< TNL::Devices::Cuda >((idx) 0, m, kernel);
			}
			// copy to Host
			for (int k=0;k<3;k++) ws_tnl_hb[k] = ws_tnl_db[k];
			// solve on CPU
			for (int k=0;k<3;k++) {
				auto start = std::chrono::steady_clock::now();
				ws_tnl_hsolver.solve(ws_tnl_hb[k], ws_tnl_hx[k]);
				auto end = std::chrono::steady_clock::now();
				auto int_ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
				real WT = int_ms * 1e-6;
				ibm_logger->info("t={:e}s k={:d} TNL CG solver: WT={:e} iterations={:d} residual={:e}", time, k, WT, ws_tnl_hsolver.getIterations(), ws_tnl_hsolver.getResidue());
			}
			// copy to GPU
			for (int k=0;k<3;k++) ws_tnl_dx[k] = ws_tnl_hx[k];
			// continue on GPU
			const auto x1 = ws_tnl_dx[0].getConstView();
			const auto x2 = ws_tnl_dx[1].getConstView();
			const auto x3 = ws_tnl_dx[2].getConstView();
//			TNL::Pointers::DevicePointer<dEllpack> MT(ws_tnl_dMT);
			TNL::Pointers::DevicePointer<dEllpack> MT_dptr(ws_tnl_dMT);
			const dEllpack* MT = &MT_dptr.template getData<TNL::Devices::Cuda>();
			auto kernel = [=] CUDA_HOSTDEV (idx i) mutable
			{
				// skipping empty rows explicitly is much faster
				if( MT->getRowCapacity(i) > 0 ) {
					dfx[i] += 2 * drho[i] * rowVectorProduct(*MT, i, x1);
					dfy[i] += 2 * drho[i] * rowVectorProduct(*MT, i, x2);
					dfz[i] += 2 * drho[i] * rowVectorProduct(*MT, i, x3);
				}
			};
			TNL::Algorithms::parallelFor< TNL::Devices::Cuda >((idx) 0, n, kernel);
			break;
		}

		case IbmCompute::Hybrid_zerocopy:
		{
			ws_tnl_dM.vectorProduct(dvx, ws_tnl_hbz[0], -1.0);
			ws_tnl_dM.vectorProduct(dvy, ws_tnl_hbz[1], -1.0);
			ws_tnl_dM.vectorProduct(dvz, ws_tnl_hbz[2], -1.0);
			if (use_LL_velocity_in_solution) {
				auto dbx = ws_tnl_db[0].getView();
				auto dby = ws_tnl_db[1].getView();
				auto dbz = ws_tnl_db[2].getView();
				const auto LL_velocity_lat = dLL_velocity_lat.getConstView();
				auto kernel = [=] CUDA_HOSTDEV (idx i) mutable
				{
					const point_t v = LL_velocity_lat[i];
					dbx[i] += v.x();
					dby[i] += v.y();
					dbz[i] += v.z();
				};
				TNL::Algorithms::parallelFor< TNL::Devices::Cuda >((idx) 0, m, kernel);
			}
			// solve on CPU
			for (int k=0;k<3;k++) {
				auto start = std::chrono::steady_clock::now();
				ws_tnl_hsolver.solve(ws_tnl_hbz[k].getView(), ws_tnl_hxz[k]);
				auto end = std::chrono::steady_clock::now();
				auto int_ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
				real WT = int_ms * 1e-6;
				ibm_logger->info("t={:e}s k={:d} TNL CG solver: WT={:e} iterations={:d} residual={:e}", time, k, WT, ws_tnl_hsolver.getIterations(), ws_tnl_hsolver.getResidue());
			}
			// continue on GPU
			const auto x1 = ws_tnl_hxz[0].getConstView();
			const auto x2 = ws_tnl_hxz[1].getConstView();
			const auto x3 = ws_tnl_hxz[2].getConstView();
//			TNL::Pointers::DevicePointer<dEllpack> MT(ws_tnl_dMT);
			TNL::Pointers::DevicePointer<dEllpack> MT_dptr(ws_tnl_dMT);
			const dEllpack* MT = &MT_dptr.template getData<TNL::Devices::Cuda>();
			auto kernel = [=] CUDA_HOSTDEV (idx i) mutable
			{
				// skipping empty rows explicitly is much faster
				if( MT->getRowCapacity(i) > 0 ) {
					dfx[i] += 2 * drho[i] * rowVectorProduct(*MT, i, x1);
					dfy[i] += 2 * drho[i] * rowVectorProduct(*MT, i, x2);
					dfz[i] += 2 * drho[i] * rowVectorProduct(*MT, i, x3);
				}
			};
			TNL::Algorithms::parallelFor< TNL::Devices::Cuda >((idx) 0, n, kernel);
			break;
		}
		#endif // USE_CUDA
		case IbmCompute::CPU:
		{
			// vx, vy, vz, rho must be copied from the device
			ws_tnl_hM.vectorProduct(hvx, ws_tnl_hb[0], -1.0);
			ws_tnl_hM.vectorProduct(hvy, ws_tnl_hb[1], -1.0);
			ws_tnl_hM.vectorProduct(hvz, ws_tnl_hb[2], -1.0);
			if (use_LL_velocity_in_solution) {
				auto hbx = ws_tnl_hb[0].getView();
				auto hby = ws_tnl_hb[1].getView();
				auto hbz = ws_tnl_hb[2].getView();
				const auto LL_velocity_lat = hLL_velocity_lat.getConstView();
				auto kernel = [=] CUDA_HOSTDEV (idx i) mutable
				{
					const point_t v = LL_velocity_lat[i];
					hbx[i] += v.x();
					hby[i] += v.y();
					hbz[i] += v.z();
				};
				TNL::Algorithms::parallelFor< TNL::Devices::Host >((idx) 0, m, kernel);
			}
			// solver
			for (int k=0;k<3;k++) {
				auto start = std::chrono::steady_clock::now();
				ws_tnl_hsolver.solve(ws_tnl_hb[k], ws_tnl_hx[k]);
				auto end = std::chrono::steady_clock::now();
				auto int_ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
				real WT = int_ms * 1e-6;
				ibm_logger->info("t={:e}s k={:d} TNL CG solver: WT={:e} iterations={:d} residual={:e}", time, k, WT, ws_tnl_hsolver.getIterations(), ws_tnl_hsolver.getResidue());
			}
			auto kernel = [&] (idx i) mutable
			{
				// skipping empty rows explicitly is much faster
				if( ws_tnl_hMT.getRowCapacity(i) > 0 ) {
					hfx[i] += 2 * hrho[i] * rowVectorProduct(ws_tnl_hMT, i, ws_tnl_hx[0]);
					hfy[i] += 2 * hrho[i] * rowVectorProduct(ws_tnl_hMT, i, ws_tnl_hx[1]);
					hfz[i] += 2 * hrho[i] * rowVectorProduct(ws_tnl_hMT, i, ws_tnl_hx[2]);
				}
			};
			TNL::Algorithms::parallelFor< TNL::Devices::Host >((idx) 0, n, kernel);
			// copy forces to the device
			// FIXME: this is copied multiple times when there are multiple Lagrange3D objects
			// (ideally there should be only one Lagrange3D object that comprises all immersed bodies)
			dfx = hfx;
			dfy = hfy;
			dfz = hfz;
			break;
		}
	}
	timer.stop();

	nlohmann::json j;
	j["threads"] = omp_get_max_threads();
	j["time_total"] = timer.getRealTime();
	ibm_logger->info("computeForcesJSON: {}", j.dump());
}

template< typename LBM >
Lagrange3D<LBM>::Lagrange3D(LBM &inputLBM, const std::string& state_id) : lbm(inputLBM)
{
	if (!spdlog::get("ibm"))
		init_file_logger("ibm", state_id, inputLBM.communicator);

	ws_tnl_hsolver.setMaxIterations(10000);
	ws_tnl_hsolver.setConvergenceResidue(3e-4);
	ws_tnl_hprecond = std::make_shared< hPreconditioner >();
//	ws_tnl_hsolver.setPreconditioner(ws_tnl_hprecond);
	#ifdef USE_CUDA
	ws_tnl_dsolver.setMaxIterations(10000);
	ws_tnl_dsolver.setConvergenceResidue(3e-4);
	ws_tnl_dprecond = std::make_shared< dPreconditioner >();
//	ws_tnl_dsolver.setPreconditioner(ws_tnl_dprecond);
	#endif
}
