#pragma once

#include <png.h>
#include <TNL/Timer.h>

#include "state.h"
#include "kernels.h"
#include "vtk_writer.h"

#include "../lbm_common/png_tool.h"

template <typename NSE>
void State<NSE>::flagCreate(const char* flagname)
{
	if (nse.rank != 0)
		return;

	const std::string fname = fmt::format("results_{}/flag.{}", id, flagname);
	create_file(fname.c_str());
}

template <typename NSE>
void State<NSE>::flagDelete(const char* flagname)
{
	if (nse.rank != 0)
		return;

	const std::string fname = fmt::format("results_{}/flag.{}", id, flagname);
	if (fileExists(fname.c_str()))
		remove(fname.c_str());
}

template <typename NSE>
bool State<NSE>::flagExists(const char* flagname)
{
	const std::string fname = fmt::format("results_{}/flag.{}", id, flagname);
	return fileExists(fname.c_str());
}

template <typename NSE>
bool State<NSE>::canCompute()
{
	bool result;
	if (nse.rank == 0) {
		if (lock_fd < 0) {
			spdlog::warn("Failed to lock the results_{} directory. Is there another instance of the solver running?", id);
			result = false;
		}
		else if (flagExists("loadstate")) {
			result = true;
		}
		else if (flagExists("finished")) {
			spdlog::info("The simulation results directory is in finished state, there is nothing to compute.");
			result = false;
		}
		else if (flagExists("terminated")) {
			spdlog::warn("The simulation results directory is in terminated state, there is nothing to compute.");
			result = false;
		}
		else {
			result = true;
		}
	}
	TNL::MPI::Bcast(&result, 1, 0, nse.communicator);
	return result;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                                                                                                                                                                                                //
//                                                                                                                                                                                                                //
// VTK POINTS
//                                                                                                                                                                                                                //
//                                                                                                                                                                                                                //
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename NSE>
void State<NSE>::writeVTK_Points(const char* name, real time, int cycle)
{
	if (! ibm.allocated)
		ibm.convertLagrangianPoints();

	// synchronize hLL_lat if points movement is computed on the GPU
	if (ibm.use_LL_velocity_in_solution && ibm.computeVariant != IbmCompute::CPU) {
		ibm.hLL_lat = ibm.dLL_lat;
		ibm.hLL_velocity_lat = ibm.dLL_velocity_lat;
	}

	writeVTK_Points(name, time, cycle, ibm.hLL_lat);
}

template <typename NSE>
void State<NSE>::writeVTK_Points(const char* name, real time, int cycle, const typename Lagrange3D::HLPVECTOR& hLL_lat)
{
	VTKWriter vtk;

	const std::string fname = fmt::format("results_{}/vtk3D/rank{:03d}_{}.vtk", id, nse.rank, name);
	create_file(fname.c_str());

	FILE* fp = fopen(fname.c_str(), "w+");
	vtk.writeHeader(fp);

	fprintf(fp, "DATASET POLYDATA\n");

	fprintf(fp, "POINTS %d float\n", (int) hLL_lat.getSize());
	for (idx i = 0; i < hLL_lat.getSize(); i++) {
		const point_t phys = nse.lat.lbm2physPoint(hLL_lat[i]);
		vtk.writeFloat(fp, phys.x());
		vtk.writeFloat(fp, phys.y());
		vtk.writeFloat(fp, phys.z());
	}
	vtk.writeBuffer(fp);
	fclose(fp);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                                                                                                                                                                                                //
//                                                                                                                                                                                                                //
// VTK 1D CUT
//                                                                                                                                                                                                                //
//                                                                                                                                                                                                                //
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename NSE>
template <typename... ARGS>
void State<NSE>::add1Dcut(point_t from, point_t to, const char* fmts, ARGS... args)
{
	probe1Dlinevec.push_back(T_PROBE1DLINECUT());
	int last = probe1Dlinevec.size() - 1;
	probe1Dlinevec[last].name = fmt::format(fmts, args...);
	probe1Dlinevec[last].from = from;
	probe1Dlinevec[last].to = to;
	probe1Dlinevec[last].cycle = 0;
}

template <typename NSE>
template <typename... ARGS>
void State<NSE>::add1Dcut_X(real y, real z, const char* fmts, ARGS... args)
{
	probe1Dvec.push_back(T_PROBE1DCUT());
	int last = probe1Dvec.size() - 1;
	probe1Dvec[last].name = fmt::format(fmts, args...);
	probe1Dvec[last].type = 0;
	probe1Dvec[last].pos1 = nse.lat.phys2lbmY(y);
	probe1Dvec[last].pos2 = nse.lat.phys2lbmZ(z);
	probe1Dvec[last].cycle = 0;
}

template <typename NSE>
template <typename... ARGS>
void State<NSE>::add1Dcut_Y(real x, real z, const char* fmts, ARGS... args)
{
	probe1Dvec.push_back(T_PROBE1DCUT());
	int last = probe1Dvec.size() - 1;
	probe1Dvec[last].name = fmt::format(fmts, args...);
	probe1Dvec[last].type = 1;
	probe1Dvec[last].pos1 = nse.lat.phys2lbmX(x);
	probe1Dvec[last].pos2 = nse.lat.phys2lbmZ(z);
	probe1Dvec[last].cycle = 0;
}

template <typename NSE>
template <typename... ARGS>
void State<NSE>::add1Dcut_Z(real x, real y, const char* fmts, ARGS... args)
{
	probe1Dvec.push_back(T_PROBE1DCUT());
	int last = probe1Dvec.size() - 1;
	probe1Dvec[last].name = fmt::format(fmts, args...);
	probe1Dvec[last].type = 2;
	probe1Dvec[last].pos1 = nse.lat.phys2lbmX(x);
	probe1Dvec[last].pos2 = nse.lat.phys2lbmY(y);
	probe1Dvec[last].cycle = 0;
}

template <typename NSE>
void State<NSE>::writeVTKs_1D()
{
	if (probe1Dvec.size() > 0) {
		// browse all 1D probeline cuts
		for (std::size_t i = 0; i < probe1Dvec.size(); i++) {
			const std::string fname = fmt::format("results_{}/probes1D/{}_rank{:03d}_{:06d}", id, probe1Dvec[i].name, nse.rank, probe1Dvec[i].cycle);
			create_parent_directories(fname.c_str());
			spdlog::info("[1dcut {}]", fname);
			//			probeLine(probe1Dvec[i].from[0],probe1Dvec[i].from[1],probe1Dvec[i].from[2],probe1Dvec[i].to[0],probe1Dvec[i].to[1],probe1Dvec[i].to[2],fname);
			switch (probe1Dvec[i].type) {
				case 0:
					write1Dcut_X(probe1Dvec[i].pos1, probe1Dvec[i].pos2, fname);
					break;
				case 1:
					write1Dcut_Y(probe1Dvec[i].pos1, probe1Dvec[i].pos2, fname);
					break;
				case 2:
					write1Dcut_Z(probe1Dvec[i].pos1, probe1Dvec[i].pos2, fname);
					break;
			}
			probe1Dvec[i].cycle++;
		}
	}

	// browse all 1D probe cuts
	for (std::size_t i = 0; i < probe1Dlinevec.size(); i++) {
		const std::string fname =
			fmt::format("results_{}/probes1D/{}_rank{:03d}_{:06d}", id, probe1Dlinevec[i].name, nse.rank, probe1Dlinevec[i].cycle);
		create_parent_directories(fname.c_str());
		spdlog::info("[1dcut {}]", fname);
		write1Dcut(probe1Dlinevec[i].from, probe1Dlinevec[i].to, fname);
		probe1Dlinevec[i].cycle++;
	}
}

template <typename NSE>
void State<NSE>::write1Dcut(point_t from, point_t to, const std::string& fname)
{
	FILE* fout = fopen(fname.c_str(), "wt");  // append information
	point_t i = nse.lat.phys2lbmPoint(from);
	point_t f = nse.lat.phys2lbmPoint(to);
	real dist = TNL::l2Norm(i - f);
	real ds = 1.0 / (dist * 2.0);  // rozliseni najit
	// special case: sampling along an axis
	if ((i[0] == f[0] && i[1] == f[1]) || (i[1] == f[1] && i[2] == f[2]) || (i[0] == f[0] && i[2] == f[2]))
		ds = 1.0 / dist;

	char idd[500];
	real value;
	int dofs;
	fprintf(fout, "#time %f s\n", nse.physTime());
	fprintf(fout, "#1:rel_pos");

	int count = 2, index = 0;
	while (outputData(
		nse.blocks.front(), index++, 0, idd, nse.blocks.front().offset.x(), nse.blocks.front().offset.y(), nse.blocks.front().offset.z(), value, dofs
	))
	{
		if (dofs == 1)
			fprintf(fout, "\t%d:%s", count++, idd);
		else
			for (int i = 0; i < dofs; i++)
				fprintf(fout, "\t%d:%s[%d]", count++, idd, i);
	}
	fprintf(fout, "\n");

	for (real s = 0; s <= 1.0; s += ds) {
		point_t p = i + s * (f - i);
		for (const auto& block : nse.blocks) {
			if (! block.isLocalIndex((idx) p.x(), (idx) p.y(), (idx) p.z()))
				continue;
			fprintf(fout, "%e", (s * dist - 0.5) * nse.lat.physDl);
			index = 0;
			while (outputData(block, index++, 0, idd, block.offset.x(), block.offset.y(), block.offset.z(), value, dofs)) {
				for (int dof = 0; dof < dofs; dof++) {
					outputData(block, index - 1, dof, idd, (idx) p.x(), (idx) p.y(), (idx) p.z(), value, dofs);
					fprintf(fout, "\t%e", value);
				}
			}
			fprintf(fout, "\n");
		}
	}
	fclose(fout);
}

template <typename NSE>
void State<NSE>::write1Dcut_X(idx y, idx z, const std::string& fname)
{
	FILE* fout = fopen(fname.c_str(), "wt");  // append information
	// probe vertical profile at x_m
	char idd[500];
	real value;
	int dofs;
	fprintf(fout, "#time %f s\n", nse.physTime());
	fprintf(fout, "#1:x");
	int count = 2, index = 0;
	while (outputData(
		nse.blocks.front(), index++, 0, idd, nse.blocks.front().offset.x(), nse.blocks.front().offset.y(), nse.blocks.front().offset.z(), value, dofs
	))
	{
		if (dofs == 1)
			fprintf(fout, "\t%d:%s", count++, idd);
		else
			for (idx i = 0; i < dofs; i++)
				fprintf(fout, "\t%d:%s[%d]", count++, idd, (int) i);
	}
	fprintf(fout, "\n");

	for (const auto& block : nse.blocks)
		for (idx i = block.offset.x(); i < block.offset.x() + block.local.x(); i++) {
			fprintf(fout, "%e", nse.lat.lbm2physX(i));
			index = 0;
			if (outputData(block, index++, 0, idd, block.offset.x(), block.offset.y(), block.offset.z(), value, dofs)) {
				for (int dof = 0; dof < dofs; dof++) {
					outputData(block, index - 1, dof, idd, i, y, z, value, dofs);
					fprintf(fout, "\t%e", value);
				}
			}
			fprintf(fout, "\n");
		}
	fclose(fout);
}

template <typename NSE>
void State<NSE>::write1Dcut_Y(idx x, idx z, const std::string& fname)
{
	FILE* fout = fopen(fname.c_str(), "wt");  // append information
	// probe vertical profile at x_m
	char idd[500];
	real value;
	int dofs;
	fprintf(fout, "#time %f s\n", nse.physTime());
	fprintf(fout, "#1:y");
	int count = 2, index = 0;
	while (outputData(
		nse.blocks.front(), index++, 0, idd, nse.blocks.front().offset.x(), nse.blocks.front().offset.y(), nse.blocks.front().offset.z(), value, dofs
	))
	{
		if (dofs == 1)
			fprintf(fout, "\t%d:%s", count++, idd);
		else
			for (idx i = 0; i < dofs; i++)
				fprintf(fout, "\t%d:%s[%d]", count++, idd, (int) i);
	}
	fprintf(fout, "\n");

	for (const auto& block : nse.blocks)
		for (idx i = block.offset.y(); i < block.offset.y() + block.local.y(); i++) {
			fprintf(fout, "%e", nse.lat.lbm2physY(i));
			int index = 0;
			while (outputData(block, index++, 0, idd, block.offset.x(), block.offset.y(), block.offset.z(), value, dofs)) {
				for (int dof = 0; dof < dofs; dof++) {
					outputData(block, index - 1, dof, idd, x, i, z, value, dofs);
					fprintf(fout, "\t%e", value);
				}
			}
			fprintf(fout, "\n");
		}
	fclose(fout);
}

template <typename NSE>
void State<NSE>::write1Dcut_Z(idx x, idx y, const std::string& fname)
{
	FILE* fout = fopen(fname.c_str(), "wt");  // append information
	// probe vertical profile at x_m
	char idd[500];
	real value;
	int dofs;
	fprintf(fout, "#time %f s\n", nse.physTime());
	fprintf(fout, "#1:z");
	int count = 2, index = 0;
	while (outputData(
		nse.blocks.front(), index++, 0, idd, nse.blocks.front().offset.x(), nse.blocks.front().offset.y(), nse.blocks.front().offset.z(), value, dofs
	))
	{
		if (dofs == 1)
			fprintf(fout, "\t%d:%s", count++, idd);
		else
			for (idx i = 0; i < dofs; i++)
				fprintf(fout, "\t%d:%s[%d]", count++, idd, (int) i);
	}
	fprintf(fout, "\n");

	for (const auto& block : nse.blocks)
		for (idx i = block.offset.z(); i < block.offset.z() + block.local.z(); i++) {
			fprintf(fout, "%e", nse.lat.lbm2physZ(i));
			index = 0;
			while (outputData(block, index++, 0, idd, block.offset.x(), block.offset.y(), block.offset.z(), value, dofs)) {
				for (int dof = 0; dof < dofs; dof++) {
					outputData(block, index - 1, dof, idd, x, y, i, value, dofs);
					fprintf(fout, "\t%e", value);
				}
			}
			fprintf(fout, "\n");
		}
	fclose(fout);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                                                                                                                                                                                                //
//                                                                                                                                                                                                                //
// VTK 3D
//                                                                                                                                                                                                                //
//                                                                                                                                                                                                                //
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename NSE>
void State<NSE>::writeVTKs_3D()
{
	TNL::Timer timer;
	for (const auto& block : nse.blocks) {
		const std::string fname = fmt::format("results_{}/output_3D", id);
		create_parent_directories(fname.c_str());
		auto outputData = [this](const BLOCK_NSE& block, int index, int dof, char* desc, idx x, idx y, idx z, real& value, int& dofs) mutable
		{
			return this->outputData(block, index, dof, desc, x, y, z, value, dofs);
		};
		timer.start();
		block.writeVTK_3D(nse.lat, outputData, fname, nse.physTime(), cnt[VTK3D].count);
		timer.stop();
		std::cout << "write3D saved in: " << timer.getRealTime() << std::endl;
		timer.reset();
		spdlog::info("[vtk {} written, time {:f}, cycle {:d}] ", fname, nse.physTime(), cnt[VTK3D].count);
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                                                                                                                                                                                                //
//                                                                                                                                                                                                                //
// VTK 3D CUT
//                                                                                                                                                                                                                //
//                                                                                                                                                                                                                //
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename NSE>
template <typename... ARGS>
void State<NSE>::add3Dcut(idx ox, idx oy, idx oz, idx lx, idx ly, idx lz, idx step, const char* fmts, ARGS... args)
{
	probe3Dvec.push_back(T_PROBE3DCUT());
	int last = probe3Dvec.size() - 1;

	probe3Dvec[last].name = fmt::format(fmts, args...);

	probe3Dvec[last].ox = ox;
	probe3Dvec[last].oy = oy;
	probe3Dvec[last].oz = oz;
	probe3Dvec[last].lx = lx;
	probe3Dvec[last].ly = ly;
	probe3Dvec[last].lz = lz;
	probe3Dvec[last].step = step;
	probe3Dvec[last].cycle = 0;
}

template <typename NSE>
void State<NSE>::writeVTKs_3Dcut()
{
	if (probe3Dvec.size() <= 0)
		return;
	// browse all 3D vtk cuts
	for (auto& probevec : probe3Dvec) {
		for (const auto& block : nse.blocks) {
			const std::string fname = fmt::format("results_{}/output_3Dcut_{}", id, probevec.name);
			create_parent_directories(fname.c_str());
			auto outputData = [this](const BLOCK_NSE& block, int index, int dof, char* desc, idx x, idx y, idx z, real& value, int& dofs) mutable
			{
				return this->outputData(block, index, dof, desc, x, y, z, value, dofs);
			};
			block.writeVTK_3Dcut(
				nse.lat,
				outputData,
				fname,
				nse.physTime(),
				probevec.cycle,
				probevec.ox,
				probevec.oy,
				probevec.oz,
				probevec.lx,
				probevec.ly,
				probevec.lz,
				probevec.step
			);
			spdlog::info("[vtk {} written, time {:f}, cycle {:d}] ", fname, nse.physTime(), probevec.cycle);
		}
		probevec.cycle++;
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                                                                                                                                                                                                //
//                                                                                                                                                                                                                //
// VTK 2D CUT
//                                                                                                                                                                                                                //
//                                                                                                                                                                                                                //
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename NSE>
template <typename... ARGS>
void State<NSE>::add2Dcut_X(idx x, const char* fmts, ARGS... args)
{
	probe2Dvec.push_back(T_PROBE2DCUT());
	int last = probe2Dvec.size() - 1;

	probe2Dvec[last].name = fmt::format(fmts, args...);

	probe2Dvec[last].type = 0;
	probe2Dvec[last].cycle = 0;
	probe2Dvec[last].position = x;
}

template <typename NSE>
template <typename... ARGS>
void State<NSE>::add2Dcut_Y(idx y, const char* fmts, ARGS... args)
{
	probe2Dvec.push_back(T_PROBE2DCUT());
	int last = probe2Dvec.size() - 1;

	probe2Dvec[last].name = fmt::format(fmts, args...);

	probe2Dvec[last].type = 1;
	probe2Dvec[last].cycle = 0;
	probe2Dvec[last].position = y;
}

template <typename NSE>
template <typename... ARGS>
void State<NSE>::add2Dcut_Z(idx z, const char* fmts, ARGS... args)
{
	probe2Dvec.push_back(T_PROBE2DCUT());
	int last = probe2Dvec.size() - 1;

	probe2Dvec[last].name = fmt::format(fmts, args...);

	probe2Dvec[last].type = 2;
	probe2Dvec[last].cycle = 0;
	probe2Dvec[last].position = z;
}

template <typename NSE>
void State<NSE>::writeVTKs_2D()
{
	if (probe2Dvec.size() <= 0)
		return;
	// browse all 2D vtk cuts
	for (auto& probevec : probe2Dvec) {
		for (const auto& block : nse.blocks) {
			const std::string fname = fmt::format("results_{}/output_2D_{}", id, probevec.name);
			create_parent_directories(fname.c_str());
			auto outputData = [this](const BLOCK_NSE& block, int index, int dof, char* desc, idx x, idx y, idx z, real& value, int& dofs) mutable
			{
				return this->outputData(block, index, dof, desc, x, y, z, value, dofs);
			};
			switch (probevec.type) {
				case 0:
					block.writeVTK_2DcutX(nse.lat, outputData, fname, nse.physTime(), probevec.cycle, probevec.position);
					break;
				case 1:
					block.writeVTK_2DcutY(nse.lat, outputData, fname, nse.physTime(), probevec.cycle, probevec.position);
					break;
				case 2:
					block.writeVTK_2DcutZ(nse.lat, outputData, fname, nse.physTime(), probevec.cycle, probevec.position);
					break;
			}
			spdlog::info("[vtk {} written, time {:f}, cycle {:d}] ", fname, nse.physTime(), probevec.cycle);
		}
		probevec.cycle++;
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                                                                                                                                                                                                //
//                                                                                                                                                                                                                //
// PNG PROJECTION
//                                                                                                                                                                                                                //
//                                                                                                                                                                                                                //
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename NSE>
bool State<NSE>::projectPNG_X(const std::string& filename, idx x0, bool rotate, bool mirror, bool flip, real amin, real amax, real bmin, real bmax)
{
	if (! fileExists(filename.c_str())) {
		fmt::print(stderr, "file {} does not exist\n", filename);
		return false;
	}
	PNGTool P(filename.c_str());

	for (auto& block : nse.blocks) {
		if (! block.isLocalX(x0))
			continue;

		// plane y-z
		idx x = x0;
		for (idx z = block.offset.z(); z < block.offset.z() + block.local.z(); z++) {
			real a = (real) z / (real) (nse.lat.global.z() - 1);  // a in [0,1]
			a = amin + a * (amax - amin);						  // a in [amin, amax]
			if (mirror)
				a = 1.0 - a;
			for (idx y = block.offset.y(); y < block.offset.y() + block.local.y(); y++) {
				real b = (real) y / (real) (nse.lat.global.y() - 1);  // b in [0,1]
				b = bmin + b * (bmax - bmin);						  // b in [bmin, bmax]
				if (flip)
					b = 1.0 - b;
				if (rotate) {
					if (P.intensity(b, a) > 0)
						block.setMap(x, y, z, NSE::BC::GEO_WALL);
				}
				else {
					if (P.intensity(a, b) > 0)
						block.setMap(x, y, z, NSE::BC::GEO_WALL);
				}
			}
		}
	}
	return true;
}

template <typename NSE>
bool State<NSE>::projectPNG_Y(const std::string& filename, idx y0, bool rotate, bool mirror, bool flip, real amin, real amax, real bmin, real bmax)
{
	if (! fileExists(filename.c_str())) {
		fmt::print(stderr, "file {} does not exist\n", filename);
		return false;
	}
	PNGTool P(filename.c_str());

	for (auto& block : nse.blocks) {
		if (! block.isLocalY(y0))
			continue;

		// plane x-z
		idx y = y0;
		for (idx z = block.offset.z(); z < block.offset.z() + block.local.z(); z++) {
			real a = (real) z / (real) (nse.lat.global.z() - 1);  // a in [0,1]
			a = amin + a * (amax - amin);						  // a in [amin, amax]
			if (mirror)
				a = 1.0 - a;
			for (idx x = block.offset.x(); x < block.offset.x() + block.local.x(); x++) {
				real b = (real) x / (real) (nse.lat.global.x() - 1);  // b in [0,1]
				b = bmin + b * (bmax - bmin);						  // b in [bmin, bmax]
				if (flip)
					b = 1.0 - b;
				if (rotate) {
					if (P.intensity(b, a) > 0)
						block.setMap(x, y, z, NSE::BC::GEO_WALL);
				}
				else {
					if (P.intensity(a, b) > 0)
						block.setMap(x, y, z, NSE::BC::GEO_WALL);
				}
			}
		}
	}
	return true;
}

template <typename NSE>
bool State<NSE>::projectPNG_Z(const std::string& filename, idx z0, bool rotate, bool mirror, bool flip, real amin, real amax, real bmin, real bmax)
{
	if (! fileExists(filename.c_str())) {
		fmt::print(stderr, "file {} does not exist\n", filename);
		return false;
	}
	PNGTool P(filename.c_str());

	for (auto& block : nse.blocks) {
		if (! block.isLocalZ(z0))
			continue;

		// plane x-y
		idx z = z0;
		for (idx x = block.offset.x(); x < block.offset.x() + block.local.x(); x++) {
			real a = (real) x / (real) (nse.lat.global.x() - 1);  // a in [0,1]
			a = amin + a * (amax - amin);						  // a in [amin, amax]
			if (mirror)
				a = 1.0 - a;
			for (idx y = block.offset.y(); y < block.offset.y() + block.local.y(); y++) {
				real b = (real) y / (real) (nse.lat.global.y() - 1);  // b in [0,1]
				b = bmin + b * (bmax - bmin);						  // b in [bmin, bmax]
				if (flip)
					b = 1.0 - b;
				if (rotate) {
					if (P.intensity(b, a) > 0)
						block.setMap(x, y, z, NSE::BC::GEO_WALL);
				}
				else {
					if (P.intensity(a, b) > 0)
						block.setMap(x, y, z, NSE::BC::GEO_WALL);
				}
			}
		}
	}
	return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                                                                                                                                                                                                //
//                                                                                                                                                                                                                //
// SAVE & LOAD STATE
//                                                                                                                                                                                                                //
//                                                                                                                                                                                                                //
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename NSE>
void State<NSE>::checkpointState(adios2::Mode mode)
{
	checkpoint.saveLoadAttribute("LBM_total_blocks", nse.total_blocks);
	checkpoint.saveLoadAttribute("LBM_physCharLength", nse.physCharLength);
	checkpoint.saveLoadAttribute("LBM_physFinalTime", nse.physFinalTime);
	checkpoint.saveLoadAttribute("LBM_iterations", nse.iterations);

	// TODO: save/load nse.lat ?

	// save/load all counter states
	for (int c = 0; c < MAX_COUNTER; c++) {
		const std::string name = fmt::format("State_counter_{}", c);
		checkpoint.saveLoadAttribute(name + "_count", cnt[c].count);
		checkpoint.saveLoadAttribute(name + "_period", cnt[c].period);
	}

	// save/load probes
	for (std::size_t i = 0; i < probe3Dvec.size(); i++) {
		const std::string name = fmt::format("State_probe3D_{}_cycle", i);
		checkpoint.saveLoadAttribute(name, probe3Dvec[i].cycle);
	}
	for (std::size_t i = 0; i < probe2Dvec.size(); i++) {
		const std::string name = fmt::format("State_probe2D_{}_cycle", i);
		checkpoint.saveLoadAttribute(name, probe2Dvec[i].cycle);
	}
	for (std::size_t i = 0; i < probe1Dvec.size(); i++) {
		const std::string name = fmt::format("State_probe1D_{}_cycle", i);
		checkpoint.saveLoadAttribute(name, probe1Dvec[i].cycle);
	}
	for (std::size_t i = 0; i < probe1Dlinevec.size(); i++) {
		const std::string name = fmt::format("State_probe1Dline_{}_cycle", i);
		checkpoint.saveLoadAttribute(name, probe1Dlinevec[i].cycle);
	}

	for (auto& block : nse.blocks) {
		// save/load map
		checkpoint.saveLoadVariable("LBM_map", block, block.hmap);

		// save/load DFs
		for (int dfty = 0; dfty < DFMAX; dfty++) {
			const std::string name = fmt::format("LBM_df_{}", dfty);
			checkpoint.saveLoadVariable(name, block, block.hfs[dfty]);
		}

		// save/load macro
		if constexpr (NSE::MACRO::N > 0)
			checkpoint.saveLoadVariable("LBM_macro", block, block.hmacro);

		// TODO: save/load other arrays that were added later to LBM_BLOCK
	}

	if (mode == adios2::Mode::Read) {
		// set physStartTime based on the loaded values - used for ETA calculation only
		nse.physStartTime = nse.physTime();
		// set startIterations based on the loaded values - used for GLUPS calculation only
		nse.startIterations = nse.iterations;
		glups_prev_iterations = nse.startIterations;
		glups_prev_time = timer_total.getRealTime();
	}
}

template <typename NSE>
void State<NSE>::saveState()
{
	// checkpoint to a staging file first to not break the previous checkpoint if we fail to create another one
	const std::string filename_tmp = fmt::format("results_{}/checkpoint_tmp.bp", id);
	spdlog::info("Saving checkpoint in {}", filename_tmp);
	checkpoint.start(filename_tmp, adios2::Mode::Write);
	checkpointState(adios2::Mode::Write);
	checkpointStateLocal(adios2::Mode::Write);
	checkpoint.finalize();

	if (nse.rank == 0) {
		const std::string filename = fmt::format("results_{}/checkpoint.bp", id);
		spdlog::info("Moving checkpoint {} to {}", filename_tmp, filename);
		int status = rename_exchange(filename_tmp.c_str(), filename.c_str());
		if (status != 0) {
			spdlog::error("rename_exchange(\"{}\", \"{}\") failed: {}", filename_tmp, filename, strerror(errno));
			return;
		}
		// update the modification timestamp on the checkpoint directory
		// (it would be weird to keep the old timestamp of a moved directory)
		status = utimensat(AT_FDCWD, filename.c_str(), NULL, 0);
		if (status != 0) {
			spdlog::error("touch(\"{}\") failed: {}", filename, strerror(errno));
		}
	}

	// Indicate that state can be loaded after restart (e.g. after a
	// failed/cancelled run or running over the walltime limit). The flag will
	// be deleted from core.h when a finished/terminated flag is created.
	flagCreate("loadstate");
}

template <typename NSE>
void State<NSE>::loadState()
{
	const std::string filename = fmt::format("results_{}/checkpoint.bp", id);
	spdlog::info("Loading data from checkpoint in {}", filename);
	checkpoint.start(filename, adios2::Mode::Read);
	checkpointState(adios2::Mode::Read);
	checkpointStateLocal(adios2::Mode::Read);
	checkpoint.finalize();
}

template <typename NSE>
bool State<NSE>::wallTimeReached()
{
	bool local_result = false;
	if (wallTime > 0) {
		long actualtimediff = timer_total.getRealTime();
		if (actualtimediff >= wallTime) {
			spdlog::info("wallTime reached: {} / {} [sec]", actualtimediff, wallTime);
			local_result = true;
		}
	}
	return TNL::MPI::reduce(local_result, MPI_LOR, nse.communicator);
}

template <typename NSE>
double State<NSE>::getWallTime(bool collective)
{
	double result = 0;
	if (! collective || nse.rank == 0) {
		result = timer_total.getRealTime();
	}
	if (collective) {
		// collective operation - make sure that all MPI processes return the same walltime (taken from rank 0)
		TNL::MPI::Bcast(&result, 1, 0, nse.communicator);
	}
	return result;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                                                                                                                                                                                                //
//                                                                                                                                                                                                                //
// LBM RELATED
//                                                                                                                                                                                                                //
//                                                                                                                                                                                                                //
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename NSE>
bool State<NSE>::estimateMemoryDemands()
{
	long long memDFs = 0;
	long long memMacro = 0;
	long long memMap = 0;
	for (const auto& block : nse.blocks) {
		const long long XYZ = block.local.x() * block.local.y() * block.local.z();
		memDFs += XYZ * sizeof(dreal) * NSE::Q;
		memMacro += XYZ * sizeof(dreal) * NSE::MACRO::N;
		memMap += XYZ * sizeof(map_t);
	}

	long long CPUavail = sysconf(_SC_PHYS_PAGES) * sysconf(_SC_PAGE_SIZE);
	long long CPUtotal = memMacro + memMap + DFMAX * memDFs;
	long long CPUDFs = DFMAX * memDFs;
#ifdef USE_CUDA
	long long GPUavail = 0;
	long long GPUtotal_hw = 0;
	long long GPUtotal = DFMAX * memDFs + memMacro + memMap;

	const int gpu_id = TNL::Backend::getDevice();
	const std::string gpu_name = TNL::Backend::getDeviceName(gpu_id);
	spdlog::info("Rank {} uses GPU id {}: {}", nse.rank, gpu_id, gpu_name);
	const std::size_t free = TNL::Backend::getFreeGlobalMemory();
	const std::size_t total = TNL::Backend::getGlobalMemorySize(gpu_id);
	GPUavail += free;
	GPUtotal_hw += total;
#endif

	spdlog::info("Local memory budget analysis / estimation for MPI rank {}", nse.rank);
	spdlog::info("CPU RAM for DFs:   {:d} MiB", CPUDFs / 1024 / 1024);
	//	spdlog::info("CPU RAM for lat:   {:d} MiB", memDFs/1024/1024);
	spdlog::info("CPU RAM for map:   {:d} MiB", memMap / 1024 / 1024);
	spdlog::info("CPU RAM for macro: {:d} MiB", memMacro / 1024 / 1024);
	spdlog::info(
		"TOTAL CPU RAM {:d} MiB estimated needed, {:d} MiB available ({:6.4f}%)",
		CPUtotal / 1024 / 1024,
		CPUavail / 1024 / 1024,
		100.0 * CPUtotal / CPUavail
	);
#ifdef USE_CUDA
	spdlog::info("GPU RAM for DFs:   {:d} MiB", DFMAX * memDFs / 1024 / 1024);
	spdlog::info("GPU RAM for map:   {:d} MiB", memMap / 1024 / 1024);
	spdlog::info("GPU RAM for macro: {:d} MiB", memMacro / 1024 / 1024);
	spdlog::info(
		"TOTAL GPU RAM {:d} MiB estimated needed, {:d} MiB available ({:6.4f}%), total GPU RAM: {:d} MiB",
		GPUtotal / 1024 / 1024,
		GPUavail / 1024 / 1024,
		100.0 * GPUtotal / GPUavail,
		GPUtotal_hw / 1024 / 1024
	);
	if (GPUavail <= GPUtotal)
		return false;
#endif
	if (CPUavail <= CPUtotal)
		return false;
	return true;
}

template <typename NSE>
void State<NSE>::reset()
{
	// compute initial DFs on GPU
	resetDFs();

	nse.resetMap(NSE::BC::GEO_FLUID);

	// setup domain geometry after all resets, including setEquilibrium,
	// so it can override the defaults with different initial condition
	setupBoundaries();

	nse.copyMapToDevice();

	// compute initial macroscopic quantities on GPU and copy to CPU
	nse.computeInitialMacro();
	nse.copyMacroToHost();
}

template <typename NSE>
void State<NSE>::resetDFs()
{
	// compute initial DFs on GPU and copy to CPU
	nse.setEquilibrium(1, 0, 0, 0);	 // rho, vx, vy, vz
	nse.copyDFsToHost();
}

template <typename NSE>
void State<NSE>::SimInit()
{
	glups_prev_time = glups_prev_iterations = 0;

	timer_SimInit.reset();
	timer_SimUpdate.reset();
	timer_AfterSimUpdate.reset();
	timer_compute.reset();
	timer_compute_overlaps.reset();
	timer_wait_communication.reset();
	timer_wait_computation.reset();

	timer_SimInit.start();

	spdlog::info(
		"MPI info: rank={:d}, nproc={:d}, lat.global=[{:d},{:d},{:d}]",
		nse.rank,
		nse.nproc,
		nse.lat.global.x(),
		nse.lat.global.y(),
		nse.lat.global.z()
	);
	for (auto& block : nse.blocks)
		spdlog::info(
			"LBM block {:d}: local=[{:d},{:d},{:d}], offset=[{:d},{:d},{:d}]",
			block.id,
			block.local.x(),
			block.local.y(),
			block.local.z(),
			block.offset.x(),
			block.offset.y(),
			block.offset.z()
		);

	spdlog::info(
		"START: simulation NSE:{} lbmVisc {:e} physDl {:e} physDt {:e}", NSE::COLL::id, nse.lat.lbmViscosity(), nse.lat.physDl, nse.lat.physDt
	);

	// reset counters
	for (int c = 0; c < MAX_COUNTER; c++)
		cnt[c].count = 0;
	cnt[SAVESTATE].count = 1;  // skip initial save of state
	nse.iterations = 0;

	// check for loadState
	if (flagExists("loadstate")) {
		// load saved state into host memory
		loadState();
		// allocate device memory and copy the data
		nse.allocateDeviceData();
		copyAllToDevice();
	}
	else {
		// allocate before reset - it might initialize on the GPU...
		nse.allocateDeviceData();

		// initialize map, DFs, and macro both in CPU and GPU memory
		reset();

#ifdef HAVE_MPI
		if (nse.nproc > 1) {
			// synchronize overlaps with MPI (initial synchronization can be synchronous)
			nse.synchronizeMapDevice();
			nse.synchronizeDFsAndMacroDevice(df_cur);
		}
#endif
	}

	spdlog::info("Finished SimInit");
	timer_SimInit.stop();
}

template <typename NSE>
void State<NSE>::SimUpdate()
{
	timer_SimUpdate.start();

	// debug
	for (auto& block : nse.blocks)
		if (block.data.lbmViscosity == 0) {
			spdlog::error("error: LBM viscosity is 0");
			nse.terminate = true;
			return;
		}

	// NOTE: all Lagrangian points are assumed to be on the first GPU
	// TODO
	//	if (nse.data.rank == 0 && ibm.LL.size() > 0)
	if (ibm.LL.size() > 0) {
		for (auto& block : nse.blocks) {
#ifdef USE_CUDA
			const auto direction = TNL::Containers::SyncDirection::None;
			TNL::Backend::LaunchConfiguration launch_config;
			launch_config.blockSize = block.computeData.at(direction).blockSize;
			launch_config.gridSize = block.computeData.at(direction).gridSize;
			TNL::Backend::launchKernelAsync(cudaLBMComputeVelocitiesStarAndZeroForce<NSE>, launch_config, block.data, nse.total_blocks);
#else
	#pragma omp parallel for schedule(static) collapse(2)
			for (idx x = 0; x < block.local.x(); x++)
				for (idx z = 0; z < block.local.z(); z++)
					for (idx y = 0; y < block.local.y(); y++)
						LBMComputeVelocitiesStarAndZeroForce<NSE>(block.data, nse.total_blocks, x, y, z);
#endif
		}
		// synchronize the null-stream after all grids
		TNL::Backend::streamSynchronize(0);

		ibm.computeForces(nse.physTime());
	}

	// call hook method (used e.g. for extra kernels in the non-Newtonian model)
	computeBeforeLBMKernel();

#ifdef HAVE_MPI
	#ifdef AA_PATTERN
	uint8_t output_df = df_cur;
	#endif
	#ifdef AB_PATTERN
	uint8_t output_df = df_out;
	#endif
#endif

#ifdef USE_CUDA
	#ifdef HAVE_MPI
	if (nse.nproc == 1) {
	#endif
		timer_compute.start();
		for (auto& block : nse.blocks) {
			const auto direction = TNL::Containers::SyncDirection::None;
			TNL::Backend::LaunchConfiguration launch_config;
			launch_config.blockSize = block.computeData.at(direction).blockSize;
			launch_config.gridSize = block.computeData.at(direction).gridSize;
			TNL::Backend::launchKernelAsync(cudaLBMKernel<NSE>, launch_config, block.data, nse.total_blocks, idx3d{0, 0, 0}, block.local);
		}
		// synchronize the null-stream after all grids
		TNL::Backend::streamSynchronize(0);
		// copying of overlaps is not necessary for nproc == 1 (nproc is checked in streaming as well)
		timer_compute.stop();
	#ifdef HAVE_MPI
	}
	else {
		timer_compute.start();
		timer_compute_overlaps.start();

		const auto boundary_directions = {
			TNL::Containers::SyncDirection::Bottom,
			TNL::Containers::SyncDirection::Top,
			TNL::Containers::SyncDirection::Back,
			TNL::Containers::SyncDirection::Front,
			TNL::Containers::SyncDirection::Left,
			TNL::Containers::SyncDirection::Right,
		};

		// compute on boundaries
		for (auto& block : nse.blocks) {
			for (auto direction : boundary_directions)
				if (auto search = block.neighborIDs.find(direction); search != block.neighborIDs.end() && search->second >= 0) {
					TNL::Backend::LaunchConfiguration launch_config;
					launch_config.blockSize = block.computeData.at(direction).blockSize;
					launch_config.gridSize = block.computeData.at(direction).gridSize;
					launch_config.stream = block.computeData.at(direction).stream;
					const idx3d offset = block.computeData.at(direction).offset;
					const idx3d size = block.computeData.at(direction).size;
					TNL::Backend::launchKernelAsync(cudaLBMKernel<NSE>, launch_config, block.data, nse.total_blocks, offset, offset + size);
				}
		}

		// compute on interior lattice sites
		for (auto& block : nse.blocks) {
			const auto direction = TNL::Containers::SyncDirection::None;
			TNL::Backend::LaunchConfiguration launch_config;
			launch_config.blockSize = block.computeData.at(direction).blockSize;
			launch_config.gridSize = block.computeData.at(direction).gridSize;
			launch_config.stream = block.computeData.at(direction).stream;
			const idx3d offset = block.computeData.at(direction).offset;
			const idx3d size = block.computeData.at(direction).size;
			TNL::Backend::launchKernelAsync(cudaLBMKernel<NSE>, launch_config, block.data, nse.total_blocks, offset, offset + size);
		}

		// wait for the computations on boundaries to finish
		// TODO: pipeline the stream synchronization with the MPI synchronizer (wait using CUDA stream events in the DistributedNDArraySynchronizer)
		for (auto& block : nse.blocks)
			for (auto direction : boundary_directions)
				if (auto search = block.neighborIDs.find(direction); search != block.neighborIDs.end() && search->second >= 0) {
					const auto& stream = block.computeData.at(direction).stream;
					TNL::Backend::streamSynchronize(stream);
				}
		timer_compute_overlaps.stop();

		// exchange the latest DFs and dmacro on overlaps between blocks
		// (it is important to wait for the communication before waiting for the computation, otherwise MPI won't progress)
		timer_wait_communication.start();
		nse.synchronizeDFsAndMacroDevice(output_df);
		timer_wait_communication.stop();

		// wait for the computation on the interior to finish
		timer_wait_computation.start();
		for (auto& block : nse.blocks) {
			const auto& stream = block.computeData.at(TNL::Containers::SyncDirection::None).stream;
			TNL::Backend::streamSynchronize(stream);
		}
		timer_wait_computation.stop();

		timer_compute.stop();
	}
	#endif
#else
	timer_compute.start();
	for (auto& block : nse.blocks) {
	#pragma omp parallel for schedule(static) collapse(2)
		for (idx x = 0; x < block.local.x(); x++)
			for (idx z = 0; z < block.local.z(); z++)
				for (idx y = 0; y < block.local.y(); y++) {
					LBMKernel<NSE>(block.data, x, y, z, nse.total_blocks);
				}
	}
	timer_compute.stop();
	#ifdef HAVE_MPI
	if (nse.nproc > 1) {
		// TODO: overlap computation with synchronization, just like above
		timer_wait_communication.start();
		nse.synchronizeDFsAndMacroDevice(output_df);
		timer_wait_communication.stop();
	}
	#endif
#endif

	nse.iterations++;

	bool doit = false;
	for (int c = 0; c < MAX_COUNTER; c++)
		if (c != PRINT && c != SAVESTATE)
			if (cnt[c].action(nse.physTime()))
				doit = true;
	if (doit) {
		// common copy
		nse.copyMacroToHost();
	}

	timer_SimUpdate.stop();
}

template <typename NSE>
void State<NSE>::AfterSimUpdate()
{
	timer_AfterSimUpdate.start();

	// call hook method (used e.g. for the coupled LBM-MHFEM solver)
	computeAfterLBMKernel();

	bool write_info = false;

	if (cnt[PRINT].action(nse.physTime()) || cnt[VTK1D].action(nse.physTime()) || cnt[VTK2D].action(nse.physTime())
		|| cnt[VTK3D].action(nse.physTime()) || cnt[VTK3DCUT].action(nse.physTime()) || cnt[PROBE1].action(nse.physTime())
		|| cnt[PROBE2].action(nse.physTime()) || cnt[PROBE3].action(nse.physTime()))
	{
		write_info = true;
		cnt[PRINT].count++;
	}

	// check for NaN values, abusing the period of other actions
	bool nan_detected = false;
	if (nse.iterations > 1 && write_info && MACRO::e_rho < MACRO::N) {
		for (auto& block : nse.blocks) {
			auto data = block.data;
			auto check_nan = [=] __cuda_callable__(idx i) -> bool
			{
				auto value = data.dmacro[MACRO::e_rho * data.XYZ + i];
				return value != value;
			};
			bool result = TNL::Algorithms::reduce<DeviceType>(idx(0), data.XYZ, check_nan, TNL::LogicalOr{});
			if (result) {
				spdlog::error("NaN detected on rank {} block {}", block.rank, block.id);
				nan_detected = true;
			}
		}
		nan_detected = TNL::MPI::reduce(nan_detected, MPI_LOR);
		if (nan_detected) {
			spdlog::error("Detected NaN, terminating the simulation.");
			nse.terminate = true;
			// in order to save the proper data, we need to copy macros from device to host
			nse.copyMacroToHost();
		}
	}

	if (cnt[VTK1D].action(nse.physTime()) || cnt[VTK2D].action(nse.physTime()) || cnt[VTK3D].action(nse.physTime())
		|| cnt[VTK3DCUT].action(nse.physTime()) || cnt[PROBE1].action(nse.physTime()) || cnt[PROBE2].action(nse.physTime())
		|| cnt[PROBE3].action(nse.physTime()) || nan_detected)
	{
		// probe1
		if (cnt[PROBE1].action(nse.physTime())) {
			probe1();
			cnt[PROBE1].count++;
		}
		// probe2
		if (cnt[PROBE2].action(nse.physTime())) {
			probe2();
			cnt[PROBE2].count++;
		}
		// probe3
		if (cnt[PROBE3].action(nse.physTime())) {
			probe3();
			cnt[PROBE3].count++;
		}
		// 3D VTK
		if (cnt[VTK3D].action(nse.physTime()) || nan_detected) {
			writeVTKs_3D();
			cnt[VTK3D].count++;
		}
		// 3D VTK CUT
		if (cnt[VTK3DCUT].action(nse.physTime())) {
			writeVTKs_3Dcut();
			cnt[VTK3DCUT].count++;
		}
		// 2D VTK
		if (cnt[VTK2D].action(nse.physTime()) || nan_detected) {
			writeVTKs_2D();
			cnt[VTK2D].count++;
		}
		// 1D VTK
		if (cnt[VTK1D].action(nse.physTime())) {
			writeVTKs_1D();
			cnt[VTK1D].count++;
		}
	}

	// statReset is called after all probes and VTK output
	// copy macro from host to device after reset
	if (cnt[STAT_RESET].action(nse.physTime())) {
		statReset();
		nse.copyMacroToDevice();
		cnt[STAT_RESET].count++;
	}
	if (cnt[STAT2_RESET].action(nse.physTime())) {
		stat2Reset();
		nse.copyMacroToDevice();
		cnt[STAT2_RESET].count++;
	}

	// only the first process writes GLUPS
	// getting the rank from MPI_COMM_WORLD is intended here - other ranks may be redirected to a file when the ranks are reordered
	if (TNL::MPI::GetRank(MPI_COMM_WORLD) == 0)
		if (nse.iterations > 1)
			if (write_info) {
				// get time diff
				const double now = timer_total.getRealTime();
				const double timediff = TNL::max(1e-6, now - glups_prev_time);

				// to avoid numerical errors - split LUPS computation in two parts
				double LUPS = (nse.iterations - glups_prev_iterations) / timediff;
				LUPS *= nse.lat.global.x() * nse.lat.global.y() * nse.lat.global.z();

				// save prev time and iterations
				glups_prev_time = now;
				glups_prev_iterations = nse.iterations;

				// simple estimate of time of accomplishment
				double ETA =
					(getWallTime() - timer_SimInit.getRealTime()) * (nse.physFinalTime - nse.physTime()) / (nse.physTime() - nse.physStartTime);

				spdlog::info(
					"GLUPS={:.3f} iter={:d} t={:1.3f}s dt={:1.2e} lbmVisc={:1.2e} WT={:.0f}s ETA={:.0f}s",
					LUPS * 1e-9,
					nse.iterations,
					nse.physTime(),
					nse.lat.physDt,
					nse.lat.lbmViscosity(),
					getWallTime(),
					ETA
				);
			}

	timer_AfterSimUpdate.stop();
}

template <typename NSE>
void State<NSE>::AfterSimFinished()
{
	const int iterations = nse.iterations - nse.startIterations;

	// only the first process writes the info
	if (TNL::MPI::GetRank(MPI_COMM_WORLD) == 0)
		if (iterations > 1) {
			spdlog::info(
				"total walltime: {:.1f} s, SimInit time: {:.1f} s, SimUpdate time: {:.1f} s, AfterSimUpdate time: {:.1f} s",
				getWallTime(),
				timer_SimInit.getRealTime(),
				timer_SimUpdate.getRealTime(),
				timer_AfterSimUpdate.getRealTime()
			);
			spdlog::info(
				"compute time: {:.1f} s, compute overlaps time: {:.1f} s, wait for communication time: {:.1f} s, wait for computation time: {:.1f} s",
				timer_compute.getRealTime(),
				timer_compute_overlaps.getRealTime(),
				timer_wait_communication.getRealTime(),
				timer_wait_computation.getRealTime()
			);
			const double avgLUPS = nse.lat.global.x() * nse.lat.global.y() * nse.lat.global.z()
								 * (iterations / (timer_SimUpdate.getRealTime() + timer_AfterSimUpdate.getRealTime()));
			const double computeLUPS = nse.lat.global.x() * nse.lat.global.y() * nse.lat.global.z() * (iterations / timer_compute.getRealTime());
			spdlog::info(
				"final GLUPS: average (based on SimInit + SimUpdate + AfterSimUpdate time): {:.3f}, based on compute time: {:.3f}",
				avgLUPS * 1e-9,
				computeLUPS * 1e-9
			);
		}
}

template <typename NSE>
void State<NSE>::updateKernelData()
{
	nse.updateKernelData();

	// this is not in nse.updateKernelData so that it can be overridden for ADE
	for (auto& block : nse.blocks)
		block.data.lbmViscosity = nse.lat.lbmViscosity();
}

template <typename NSE>
void State<NSE>::copyAllToDevice()
{
	nse.copyMapToDevice();
	nse.copyDFsToDevice();
	nse.copyMacroToDevice();  // important when a state has been loaded
}

template <typename NSE>
void State<NSE>::copyAllToHost()
{
	nse.copyMapToHost();
	nse.copyDFsToHost();
	nse.copyMacroToHost();
}
