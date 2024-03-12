#pragma once

#include "state.h"

#include <curl/curl.h>
#include <chrono>
#include <ctime>

#include "lbm_common/png_tool.h"
#include "lbm_common/fileutils.h"
#include "lbm_common/timeutils.h"

template< typename NSE >
int State<NSE>::addLagrange3D()
{
	const std::string dir = fmt::format("results_{}", id);
	FF.emplace_back(nse, std::move(dir));
	return FF.size()-1;
}

template< typename NSE >
void State<NSE>::computeAllLagrangeForces()
{
	for (std::size_t i=0;i<FF.size();i++)
		if (FF[i].implicitWuShuForcing)
			FF[i].computeWuShuForcesSparse(nse.physTime());
}

template< typename NSE >
bool State<NSE>::getPNGdimensions(const char * filename, int &w, int &h)
{
	if (!fileExists(filename)) { printf("file %s does not exist\n",filename); return false; }
	FILE *fp = fopen(filename, "rb");

	png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
	if(!png) { printf("file %s png read error\n",filename); return false; }

	png_infop info = png_create_info_struct(png);
	if(!png) { printf("file %s png read error\n",filename); return false; }

	if(setjmp(png_jmpbuf(png))) { printf("file %s png read error\n",filename); return false; }

	png_init_io(png, fp);

	png_read_info(png, info);

	w = png_get_image_width(png, info);
	h = png_get_image_height(png, info);
	//  color_type = png_get_color_type(png, info);
	//  bit_depth  = png_get_bit_depth(png, info);
	fclose(fp);
	if (w>0 && h>0) return true;
	return false;
}


template< typename NSE >
template< typename... ARGS >
void State<NSE>::log(const char* fmts, ARGS... args)
{
	const std::string dir = fmt::format("results_{}", id);
	mkdir(dir.c_str(), 0777);
	const std::string fname = fmt::format("{}/log_rank{:03d}", dir, nse.rank);

	FILE* f = fopen(fname.c_str(), "at"); // append information
	if (f==0) {
		fmt::print(stderr, "unable to create/access file {}", fname);
		return;
	}
	// insert time stamp
	fmt::print(f, "{} ", timestamp());
	fmt::print(f, fmts, args...);
	fmt::print(f, "\n");
	fclose(f);

	fmt::print(fmts, args...);
	fmt::print("\n");

}

/// outputs information into log file "type"
template< typename NSE >
template< typename... ARGS >
void State<NSE>::setid(const char* fmts, ARGS... args)
{
	id = fmt::format(fmts, args...);
}

template< typename NSE >
void State<NSE>::flagCreate(const char* flagname)
{
	if (nse.rank != 0) return;

	const std::string fname = fmt::format("results_{}/{}", id, flagname);
	create_file(fname.c_str());

	FILE*f = fopen(fname.c_str(), "at"); // append information
	if (f==0) {
		fmt::print(stderr, "unable to create/access file {}", fname);
		return;
	}
	// insert time stamp
	fmt::print(f, "{}\n", timestamp());
	fclose(f);
}

template< typename NSE >
void State<NSE>::flagDelete(const char*flagname)
{
	if (nse.rank != 0) return;

	const std::string fname = fmt::format("results_{}/{}", id, flagname);
	if (fileExists(fname.c_str()))
		remove(fname.c_str());
}

template< typename NSE >
bool State<NSE>::flagExists(const char* flagname)
{
	const std::string fname = fmt::format("results_{}/{}", id, flagname);
	return fileExists(fname.c_str());
}

template< typename NSE >
template< typename... ARGS >
void State<NSE>::mark(const char* fmts, ARGS... args)
{
	if (nse.rank != 0) return;

	const std::string fname = fmt::format("results_{}/mark", id);
	create_file(fname.c_str());

	FILE* f = fopen(fname.c_str(), "at"); // append information
	if (f==0) {
		fmt::print(stderr, "unable to create/access file {}", fname);
		return;
	}
	// insert time stamp
	fmt::print(f, "{} ", timestamp());
	fmt::print(f, fmts, args...);
	fmt::print(f, "\n");
	fclose(f);
}

/// checks/creates mark and return status
template< typename NSE >
bool State<NSE>::isMark()
{
	bool result;
	if (nse.rank == 0)
	{
		const std::string fname = fmt::format("results_{}/mark", id);
		if (fileExists(fname.c_str()))
		{
			log("Mark {} already exists.", fname);
			result = true;
		}
		else
		{
			log("Mark {} does not exist. Creating new mark.", fname);
			mark("");
			result = false;
		}
	}
	TNL::MPI::Bcast(&result, 1, 0, nse.communicator);
	return result;
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                                                                                                                                                                                                //
//                                                                                                                                                                                                                //
// VTK SURFACE
//                                                                                                                                                                                                                //
//                                                                                                                                                                                                                //
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


template< typename NSE >
void State<NSE>::writeVTK_Surface(const char* name, real time, int cycle, Lagrange3D &fil)
{
	VTKWriter vtk;

	const std::string fname = fmt::format("results_{}/vtk3D/rank{:03d}_{}.vtk", id, nse.rank, name);
	create_file(fname.c_str());

	FILE* fp = fopen(fname.c_str(), "w+");
	vtk.writeHeader(fp);

	fprintf(fp, "DATASET POLYDATA\n");

	fprintf(fp, "POINTS %d float\n", fil.LL.size());
	for (std::size_t i=0;i<fil.LL.size();i++)
	{
		vtk.writeFloat(fp, fil.LL[i].x);
		vtk.writeFloat(fp, fil.LL[i].y);
		vtk.writeFloat(fp, fil.LL[i].z);
	}
	vtk.writeBuffer(fp);

	fprintf(fp, "POLYGONS %d %d\n", (fil.lag_X-1)*fil.lag_Y , 5*(fil.lag_X-1)*fil.lag_Y ); // first number: number of polygons, second number: total integers describing the list
	for (int i=0;i<fil.lag_X-1;i++)
	for (int j=0;j<fil.lag_Y;j++)
	{
		int ip = i+1;
		int jp = (j==fil.lag_Y-1) ? 0 : j+1;
		vtk.writeInt(fp,4);
		vtk.writeInt(fp,fil.findIndex(i,j));
		vtk.writeInt(fp,fil.findIndex(ip,j));
		vtk.writeInt(fp,fil.findIndex(ip,jp));
		vtk.writeInt(fp,fil.findIndex(i,jp));
	}
	fclose(fp);
}

template< typename NSE >
template< typename... ARGS >
void State<NSE>::WriteTempAVG(dreal AvgTemp, int res)
{
	const std::string dir = fmt::format("results_{}/LOG", id);
	mkdir(dir.c_str(), 0777);
	const std::string fname = fmt::format("results_{}/LOG/Temp_log{}", id, res);
	create_file(fname.c_str());


	FILE*f = fopen(fname.c_str(),"at");
	if (f==0) {
		log("Unable to create/access file {}", fname.c_str());
		return;
	}

	fprintf(f, "%e \t %d\n", AvgTemp, res);
	fclose(f);
}

template< typename NSE >
template< typename... ARGS >
void State<NSE>::WriteTempMinMax(dreal minTem, dreal maxTem, dreal maxPec, int res, double time, double vel)
{
	const std::string dir = fmt::format("results_{}/LOG", id);
	mkdir(dir.c_str(), 0777);
	const std::string fname = fmt::format("results_{}/LOG/MinMax_temp_{}", id, id);
	create_file(fname.c_str());


	FILE*f = fopen(fname.c_str(),"at");
	if (f==0) {
		log("Unable to create/access file {}", fname.c_str());
		return;
	}

	std::time_t current_time = std::time(nullptr); // Or simply time(0)

    // Convert time_t to tm struct
    struct tm * timeinfo = localtime(&current_time);

    char buffer[80];
    // Format the time into the buffer as "yyyy-MM-dd hh:mm:ss"
    strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", timeinfo);


	
	fprintf(f, "%d \t %f \t %e \t %e \t %e \t %e \t %s \n", res, vel, minTem, maxTem, maxPec, time, buffer);
	fclose(f);
}

template< typename NSE >
template< typename... ARGS >
void State<NSE>::WriteTempInFile(dreal AvgTemp, int res, double physDt, double physDl, double physDif, double transfer, double vel, double time, double bodyDif)
{
	const std::string dir = fmt::format("results_{}/LOG", id);
	mkdir(dir.c_str(), 0777);
	const std::string fname = fmt::format("results_{}/LOG/Log_res{}", id, res);
	create_file(fname.c_str());


	FILE*f = fopen(fname.c_str(),"at");
	if (f==0) {
		log("Unable to create/access file {}", fname.c_str());
		return;
	}

	fprintf(f, "%e \t %d \t %e \t %e \t %e \t %e \t %e \t %e \t %e\n", AvgTemp, res, physDt, physDl, physDif, transfer, vel, time, bodyDif);
	fclose(f);
}

template< typename NSE >
template< typename... ARGS >
void State<NSE>::ComputeKolmogorov(dreal &kolmo_sc, dreal meanFluDerxx, dreal meanFluDerxy, dreal meanFluDerxz, dreal meanFluDeryy, dreal meanFluDeryz, dreal meanFluDerzz)
{
	dreal nu = 1.552e-5;
	dreal sqnu = pow(nu,0.5); 
	kolmo_sc = sqnu/(pow((
					2*(meanFluDerxx)
					+2*(meanFluDeryy)
					+2*(meanFluDerzz)
					+(meanFluDerxy)
					+(meanFluDerxz)
					+(meanFluDeryz)
				),0.25));
}

template< typename NSE >
template< typename... ARGS >
void State<NSE>::WriteMean(std::string direction, dreal *mean, int numY, int numX, int numZ)
{
	const std::string dir = fmt::format("results_{}", id);
	mkdir(dir.c_str(), 0777);
	const std::string fname = fmt::format("results_{}/{}", id, direction);
	create_file(fname.c_str());

	FILE*f = fopen(fname.c_str(),"wt");
	if (f==0) {
		log("Unable to create/access file {}", fname.c_str());
		return;
	}

	for(int y = 0; y < numY; y++) {
		for(int z = 0; z<  numZ; z++) {
			for(int x = 0; x< numX; x++) {
				fprintf(f, "\t %.10e", mean[this->IndexKolmo(x,y,z, numX, numY)]);
			}
			fprintf(f, "\n");
		}
	}
	
	fclose(f);
}

template< typename NSE >
template< typename... ARGS >
void State<NSE>::WriteTempArr(std::string direction, dreal *mean, int numY, int numX, int numZ)
{
	const std::string dir = fmt::format("results_{}", id);
	mkdir(dir.c_str(), 0777);
	const std::string fname = fmt::format("results_{}/{}", id, direction);
	create_file(fname.c_str());

	FILE*f = fopen(fname.c_str(),"wt");
	if (f==0) {
		log("Unable to create/access file {}", fname.c_str());
		return;
	}

	for(int y = 1; y < numY-1; y++) {
		for(int z = 1; z<  numZ-1; z++) {
			for(int x = 1; x< numX-1; x++) {
				fprintf(f, "\t %.10e", mean[this->IndexKolmo(x,y,z, numX, numY)]);
			}
			fprintf(f, "\n");
		}
	}
	
	fclose(f);
}

template< typename NSE >
template< typename... ARGS >
void State<NSE>::ReadMean(char filename, dreal *mean)
{
    // Open the file
	const std::string fname = fmt::format("results_{}/MeanVelocity_{}", id, filename);
    std::ifstream inputFile(fname.c_str());

	std::cout << fname.c_str() << std::endl;
    // Check if the file is open
    if (!inputFile.is_open()) {
        std::cerr << "Error opening file\n";
        return; // return an error code
    }

	int i = 0;
    dreal value;
 	while (inputFile >> value) {		
		mean[i] = (dreal)value;
		i++;
    }

    // Close the file
    inputFile.close();
}


template< typename NSE >
template< typename... ARGS >
int State<NSE>::IndexKolmo(int x, int y, int z, int numX, int numY)
{
	return x+y*numX+z*numX*numY;
}

template< typename dreal >
void ComputeMean(int IterNum, dreal velocity, dreal &suma, dreal &Mean)
{
	suma += velocity;
	Mean = suma / (IterNum);
}

template< typename dreal >
void ComputeFluctuation(dreal velocity, dreal mean, dreal &fluctuation)
{	
	fluctuation = mean - velocity;
}

template< typename NSE >
template< typename... ARGS >
void State<NSE>::ComputeFlucDerivative(dreal Zfluc, dreal &flucDerivative, dreal Pfluc, dreal Mfluc)
{
	// std::cout << "Vlevo je flu: " << Mfluc << std::endl;
	// std::cout << "Stred je flu: " << Zfluc << std::endl;
	// std::cout << "Vpravo je flu: " << Pfluc << std::endl;
	// std::cout << "PhysDl: " << nse.lat.physDl << std::endl;
	flucDerivative = (Pfluc - Mfluc) / (2*nse.lat.physDl);
	// std::cout << "Derivative is: " << flucDerivative << std::endl;
}


size_t WriteCallback(void* contents, size_t size, size_t nmemb, std::string* output) {
    size_t total_size = size * nmemb;
    output->append(static_cast<char*>(contents), total_size);
    return total_size;
}



////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                                                                                                                                                                                                //
//                                                                                                                                                                                                                //
// VTK POINTS
//                                                                                                                                                                                                                //
//                                                                                                                                                                                                                //
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


template< typename NSE >
void State<NSE>::writeVTK_Points(const char* name, real time, int cycle, Lagrange3D &fil)
{
	VTKWriter vtk;

	const std::string fname = fmt::format("results_{}/vtk3D/rank{:03d}_{}.vtk", id, nse.rank, name);
	create_file(fname.c_str());

	FILE* fp = fopen(fname.c_str(), "w+");
	vtk.writeHeader(fp);

	fprintf(fp, "DATASET POLYDATA\n");

	fprintf(fp, "POINTS %lu float\n", fil.LL.size());
	for (std::size_t i=0;i<fil.LL.size();i++)
	{
		vtk.writeFloat(fp, fil.LL[i].x);
		vtk.writeFloat(fp, fil.LL[i].y);
		vtk.writeFloat(fp, fil.LL[i].z);
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


template< typename NSE >
template< typename... ARGS >
void State<NSE>::add1Dcut(point_t from, point_t to, const char* fmts, ARGS... args)
{
	probe1Dlinevec.push_back( T_PROBE1DLINECUT() );
	int last = probe1Dlinevec.size()-1;
	probe1Dlinevec[last].name = fmt::format(fmts, args...);
	probe1Dlinevec[last].from = from;
	probe1Dlinevec[last].to = to;
	probe1Dlinevec[last].cycle = 0;
}

template< typename NSE >
template< typename... ARGS >
void State<NSE>::add1Dcut_X(real y, real z, const char* fmts, ARGS... args)
{
	if (!nse.isAnyLocalY(nse.lat.phys2lbmY(y)) || !nse.isAnyLocalZ(nse.lat.phys2lbmZ(z))) return;

	probe1Dvec.push_back( T_PROBE1DCUT() );
	int last = probe1Dvec.size()-1;
	probe1Dvec[last].name = fmt::format(fmts, args...);
	probe1Dvec[last].type = 0;
	probe1Dvec[last].pos1 = nse.lat.phys2lbmY(y);
	probe1Dvec[last].pos2 = nse.lat.phys2lbmZ(z);
	probe1Dvec[last].cycle = 0;
}

template< typename NSE >
template< typename... ARGS >
void State<NSE>::add1Dcut_Y(real x, real z, const char* fmts, ARGS... args)
{
	if (!nse.isAnyLocalX(nse.lat.phys2lbmX(x)) || !nse.isAnyLocalZ(nse.lat.phys2lbmZ(z))) return;

	probe1Dvec.push_back( T_PROBE1DCUT() );
	int last = probe1Dvec.size()-1;
	probe1Dvec[last].name = fmt::format(fmts, args...);
	probe1Dvec[last].type = 1;
	probe1Dvec[last].pos1 = nse.lat.phys2lbmX(x);
	probe1Dvec[last].pos2 = nse.lat.phys2lbmZ(z);
	probe1Dvec[last].cycle = 0;
}

template< typename NSE >
template< typename... ARGS >
void State<NSE>::add1Dcut_Z(real x, real y, const char* fmts, ARGS... args)
{
	if (!nse.isAnyLocalX(nse.lat.phys2lbmX(x)) || !nse.isAnyLocalY(nse.lat.phys2lbmY(y))) return;

	probe1Dvec.push_back( T_PROBE1DCUT() );
	int last = probe1Dvec.size()-1;
	probe1Dvec[last].name = fmt::format(fmts, args...);
	probe1Dvec[last].type = 2;
	probe1Dvec[last].pos1 = nse.lat.phys2lbmX(x);
	probe1Dvec[last].pos2 = nse.lat.phys2lbmY(y);
	probe1Dvec[last].cycle = 0;
}

template< typename NSE >
void State<NSE>::writeVTKs_1D()
{
	if (probe1Dvec.size()>0)
	{
		// browse all 1D probeline cuts
		for (std::size_t i=0;i<probe1Dvec.size(); i++)
		{
			const std::string fname = fmt::format("results_{}/probes1D/{}_rank{:03d}_{:06d}", id, probe1Dvec[i].name, nse.rank, probe1Dvec[i].cycle);
			// create parent directories
			create_file(fname.c_str());
			log("[1dcut {}]", fname);
//			probeLine(probe1Dvec[i].from[0],probe1Dvec[i].from[1],probe1Dvec[i].from[2],probe1Dvec[i].to[0],probe1Dvec[i].to[1],probe1Dvec[i].to[2],fname);
			switch (probe1Dvec[i].type)
			{
				case 0: write1Dcut_X(probe1Dvec[i].pos1, probe1Dvec[i].pos2, fname);
					break;
				case 1: write1Dcut_Y(probe1Dvec[i].pos1, probe1Dvec[i].pos2, fname);
					break;
				case 2: write1Dcut_Z(probe1Dvec[i].pos1, probe1Dvec[i].pos2, fname);
					break;
			}
			probe1Dvec[i].cycle++;
		}
	}

	// browse all 1D probe cuts
	for (std::size_t i=0;i<probe1Dlinevec.size(); i++)
	{
		const std::string fname = fmt::format("results_{}/probes1D/{}_rank{:03d}_{:06d}", id, probe1Dlinevec[i].name, nse.rank, probe1Dlinevec[i].cycle);
		// create parent directories
		create_file(fname.c_str());
		log("[1dcut {}]", fname);
		write1Dcut(probe1Dlinevec[i].from, probe1Dlinevec[i].to, fname);
		probe1Dlinevec[i].cycle++;
	}
}

template< typename NSE >
void State<NSE>::write1Dcut(point_t from, point_t to, const std::string& fname)
{
	FILE* fout = fopen(fname.c_str(), "wt"); // append information
	point_t i = nse.lat.phys2lbmPoint(from);
	point_t f = nse.lat.phys2lbmPoint(to);
	real dist = NORM(i[0]-f[0],i[1]-f[1],i[2]-f[2]);
	real ds = 1.0/(dist*2.0); // rozliseni najit
	// special case: sampling along an axis
	if( (i[0] == f[0] && i[1] == f[1]) ||
		(i[1] == f[1] && i[2] == f[2]) ||
		(i[0] == f[0] && i[2] == f[2]) )
		ds = 1.0/dist;

	char idd[500];
	real value;
	int dofs;
	fprintf(fout,"#time %f s\n", nse.physTime());
	fprintf(fout,"#1:rel_pos");

	int count=2, index=0;
	while (outputData(nse.blocks.front(), index++, 0, idd, nse.blocks.front().offset.x(), nse.blocks.front().offset.y(), nse.blocks.front().offset.z(), value, dofs))
	{
		if (dofs==1) fprintf(fout,"\t%d:%s",count++,idd);
		else
		for (int i=0;i<dofs;i++) fprintf(fout,"\t%d:%s[%d]",count++,idd,i);
	}
	fprintf(fout,"\n");

	for (real s=0;s<=1.0;s+=ds)
	{
		point_t p = i + s * (f - i);
		for (const auto& block : nse.blocks)
		{
			if (!block.isLocalIndex((idx) p.x(), (idx) p.y(), (idx) p.z())) continue;
			if (block.isFluid((idx) p.x(), (idx) p.y(), (idx) p.z()))
			{
				fprintf(fout, "%e", (s*dist-0.5)*nse.lat.physDl);
				index=0;
				while (outputData(block, index++, 0, idd, block.offset.x(), block.offset.y(), block.offset.z(), value, dofs))
				{
					for (int dof=0;dof<dofs;dof++)
					{
						outputData(block, index-1, dof, idd, (idx) p.x(), (idx) p.y(), (idx) p.z(), value, dofs);
						fprintf(fout, "\t%e", value);
					}
				}
				fprintf(fout, "\n");
			}
		}
	}
	fclose(fout);
}

template< typename NSE >
void State<NSE>::write1Dcut_X(idx y, idx z, const std::string& fname)
{
	FILE* fout = fopen(fname.c_str(), "wt"); // append information
	// probe vertical profile at x_m
	char idd[500];
	real value;
	int dofs;
	fprintf(fout,"#time %f s\n", nse.physTime());
	fprintf(fout,"#1:x");
	int count=2, index=0;
	while (outputData(nse.blocks.front(), index++, 0, idd, nse.blocks.front().offset.x(), nse.blocks.front().offset.y(), nse.blocks.front().offset.z(), value, dofs))
	{
		if (dofs==1) fprintf(fout,"\t%d:%s",count++,idd);
		else
		for (idx i=0;i<dofs;i++) fprintf(fout,"\t%d:%s[%d]",count++,idd,(int)i);
	}
	fprintf(fout,"\n");

	for (const auto& block : nse.blocks)
	for (idx i = block.offset.x(); i < block.offset.x() + block.local.x(); i++)
	{
		if (block.isFluid(i,y,z))
		{
			fprintf(fout, "%e", nse.lat.lbm2physX(i));
			index=0;
			if (outputData(block, index++, 0, idd, block.offset.x(), block.offset.y(), block.offset.z(), value, dofs))
			{
				for (int dof=0;dof<dofs;dof++)
				{
					outputData(block, index-1,dof,idd,i,y,z,value,dofs);
					fprintf(fout, "\t%e", value);
				}
			}
			fprintf(fout, "\n");
		}
	}
	fclose(fout);
}

template< typename NSE >
void State<NSE>::write1Dcut_Y(idx x, idx z, const std::string& fname)
{
	FILE* fout = fopen(fname.c_str(), "wt"); // append information
	// probe vertical profile at x_m
	char idd[500];
	real value;
	int dofs;
	fprintf(fout,"#time %f s\n", nse.physTime());
	fprintf(fout,"#1:y");
	int count=2, index=0;
	while (outputData(nse.blocks.front(), index++, 0, idd, nse.blocks.front().offset.x(), nse.blocks.front().offset.y(), nse.blocks.front().offset.z(), value, dofs))
	{
		if (dofs==1) fprintf(fout,"\t%d:%s",count++,idd);
		else
		for (idx i=0;i<dofs;i++) fprintf(fout,"\t%d:%s[%d]",count++,idd,(int)i);
	}
	fprintf(fout,"\n");

	for (const auto& block : nse.blocks)
	for (idx i = block.offset.y(); i < block.offset.y() + block.local.y(); i++)
	{
		if (block.isFluid(x,i,z))
		{
			fprintf(fout, "%e", nse.lat.lbm2physY(i));
			int index=0;
			while (outputData(block, index++, 0, idd, block.offset.x(), block.offset.y(), block.offset.z(), value, dofs))
			{
				for (int dof=0;dof<dofs;dof++)
				{
					outputData(block, index-1,dof,idd,x,i,z,value,dofs);
					fprintf(fout, "\t%e", value);
				}
			}
			fprintf(fout, "\n");
		}
	}
	fclose(fout);
}

template< typename NSE >
void State<NSE>::write1Dcut_Z(idx x, idx y, const std::string& fname)
{
	FILE* fout = fopen(fname.c_str(), "wt"); // append information
	// probe vertical profile at x_m
	char idd[500];
	real value;
	int dofs;
	fprintf(fout,"#time %f s\n", nse.physTime());
	fprintf(fout,"#1:z");
	int count=2, index=0;
	while (outputData(nse.blocks.front(), index++, 0, idd, nse.blocks.front().offset.x(), nse.blocks.front().offset.y(), nse.blocks.front().offset.z(), value, dofs))
	{
		if (dofs==1) fprintf(fout,"\t%d:%s",count++,idd);
		else
		for (idx i=0;i<dofs;i++) fprintf(fout,"\t%d:%s[%d]",count++,idd,(int)i);
	}
	fprintf(fout,"\n");

	for (const auto& block : nse.blocks)
	for (idx i = block.offset.z(); i < block.offset.z() + block.local.z(); i++)
	{
		if (block.isFluid(x,y,i))
		{
			fprintf(fout, "%e", nse.lat.lbm2physZ(i));
			index=0;
			while (outputData(block, index++, 0, idd, block.offset.x(), block.offset.y(), block.offset.z(), value, dofs))
			{
				for (int dof=0;dof<dofs;dof++)
				{
					outputData(block, index-1,dof,idd,x,y,i,value,dofs);
					fprintf(fout, "\t%e", value);
				}
			}
			fprintf(fout, "\n");
		}
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


template< typename NSE >
void State<NSE>::writeVTKs_3D()
{
	const std::string dir = fmt::format("results_{}/vtk3D", id);
	mkdir_p(dir.c_str(), 0755);

	for (const auto& block : nse.blocks)
	{
		std::string tmp_dirname;
		const char* local_scratch = getenv("LOCAL_SCRATCH");
		if (!local_scratch || local_scratch[0] == '\0')
		{
			// $LOCAL_SCRATCH is not defined or empty - default to regular subdirectory in results_*
			tmp_dirname = dir;
			local_scratch = NULL;
		}
		else
		{
			// Write files temporarily into the local scratch and move them to final_dirname at
			// the end, after all MPI processes have completed. This avoids small buffered writes
			// into the shared filesystem on clusters as well as corruption of previous state due
			// to MPI errors...
			tmp_dirname = fmt::format("{}/{}", local_scratch, dir);
		}
		mkdir_p(tmp_dirname.c_str(), 0755);

		const std::string basename = fmt::format("block{:03d}_{:d}.vtk", block.id, cnt[VTK3D].count);
		const std::string filename = fmt::format("{}/{}", tmp_dirname, basename);
		auto outputData = [this] (const BLOCK_NSE& block, int index, int dof, char* desc, idx x, idx y, idx z, real& value, int& dofs) mutable
		{
			return this->outputData(block, index, dof, desc, x, y, z, value, dofs);
		};
		block.writeVTK_3D(nse.lat, outputData, filename, nse.physTime(), cnt[VTK3D].count);
		log("[vtk {} written, time {:f}, cycle {:d}] ", filename, nse.physTime(), cnt[VTK3D].count);

		if (local_scratch)
		{
			// move the files from local_scratch into final_dirname and create a backup of the existing files
			move(tmp_dirname, dir, basename, basename);
		}
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                                                                                                                                                                                                //
//                                                                                                                                                                                                                //
// VTK 3D CUT
//                                                                                                                                                                                                                //
//                                                                                                                                                                                                                //
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


template< typename NSE >
template< typename... ARGS >
void State<NSE>::add3Dcut(idx ox, idx oy, idx oz, idx lx, idx ly, idx lz, idx step, const char* fmts, ARGS... args)
{
	probe3Dvec.push_back( T_PROBE3DCUT() );
	int last = probe3Dvec.size()-1;

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

template< typename NSE >
void State<NSE>::writeVTKs_3Dcut()
{
	if (probe3Dvec.size()<=0) return;
	// browse all 3D vtk cuts
	for (auto& probevec : probe3Dvec)
	{
		for (const auto& block : nse.blocks)
		{
			const std::string fname = fmt::format("results_{}/vtk3Dcut/{}_block{:03d}_{:d}.vtk", id, probevec.name, block.id, probevec.cycle);
			// create parent directories
			create_file(fname.c_str());
			auto outputData = [this] (const BLOCK_NSE& block, int index, int dof, char* desc, idx x, idx y, idx z, real& value, int& dofs) mutable
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
			log("[vtk {} written, time {:f}, cycle {:d}] ", fname, nse.physTime(), probevec.cycle);
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

template< typename NSE >
template< typename... ARGS >
void State<NSE>::add2Dcut_X(idx x, const char* fmts, ARGS... args)
{
	if (!nse.isAnyLocalX(x)) return;

	probe2Dvec.push_back( T_PROBE2DCUT() );
	int last = probe2Dvec.size()-1;

	probe2Dvec[last].name = fmt::format(fmts, args...);

	probe2Dvec[last].type = 0;
	probe2Dvec[last].cycle = 0;
	probe2Dvec[last].position = x;
}

template< typename NSE >
template< typename... ARGS >
void State<NSE>::add2Dcut_Y(idx y, const char* fmts, ARGS... args)
{
	if (!nse.isAnyLocalY(y)) return;

	probe2Dvec.push_back( T_PROBE2DCUT() );
	int last = probe2Dvec.size()-1;

	probe2Dvec[last].name = fmt::format(fmts, args...);

	probe2Dvec[last].type = 1;
	probe2Dvec[last].cycle = 0;
	probe2Dvec[last].position = y;
}

template< typename NSE >
template< typename... ARGS >
void State<NSE>::add2Dcut_Z(idx z, const char* fmts, ARGS... args)
{
	if (!nse.isAnyLocalZ(z)) return;

	probe2Dvec.push_back( T_PROBE2DCUT() );
	int last = probe2Dvec.size()-1;

	probe2Dvec[last].name = fmt::format(fmts, args...);

	probe2Dvec[last].type = 2;
	probe2Dvec[last].cycle = 0;
	probe2Dvec[last].position = z;
}


template< typename NSE >
void State<NSE>::writeVTKs_2D()
{
	if (probe2Dvec.size()<=0) return;
	// browse all 2D vtk cuts
	for (auto& probevec : probe2Dvec)
	{
		for (const auto& block : nse.blocks)
		{
			const std::string fname = fmt::format("results_{}/vtk2D/{}_block{:03d}_{:d}.vtk", id, probevec.name, block.id, probevec.cycle);
			// create parent directories
			create_file(fname.c_str());
			auto outputData = [this] (const BLOCK_NSE& block, int index, int dof, char* desc, idx x, idx y, idx z, real& value, int& dofs) mutable
			{
				return this->outputData(block, index, dof, desc, x, y, z, value, dofs);
			};
			switch (probevec.type)
			{
				case 0: block.writeVTK_2DcutX(nse.lat, outputData, fname, nse.physTime(), probevec.cycle, probevec.position);
					break;
				case 1: block.writeVTK_2DcutY(nse.lat, outputData, fname, nse.physTime(), probevec.cycle, probevec.position);
					break;
				case 2: block.writeVTK_2DcutZ(nse.lat, outputData, fname, nse.physTime(), probevec.cycle, probevec.position);
					break;
			}
			log("[vtk {} written, time {:f}, cycle {:d}] ", fname, nse.physTime(), probevec.cycle);
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


template< typename NSE >
bool State<NSE>::projectPNG_X(const std::string& filename, idx x0, bool rotate, bool mirror, bool flip, real amin, real amax, real bmin, real bmax)
{
	if (!fileExists(filename.c_str())) {
		fmt::print(stderr, "file {} does not exist\n", filename);
		return false;
	}
	PNGTool P(filename.c_str());

	for (auto& block : nse.blocks)
	{
		if (!block.isLocalX(x0)) continue;

		// plane y-z
		idx x = x0;
		for (idx z = block.offset.z(); z < block.offset.z() + block.local.z(); z++)
		{
			real a = (real)z/(real)(nse.lat.global.z() - 1); // a in [0,1]
			a = amin + a * (amax - amin); // a in [amin, amax]
			if (mirror) a = 1.0 - a;
			for (idx y = block.offset.y(); y < block.offset.y() + block.local.y(); y++)
			{
				real b = (real)y/(real)(nse.lat.global.y() - 1); // b in [0,1]
				b = bmin + b * (bmax - bmin); // b in [bmin, bmax]
				if (flip) b = 1.0 - b;
				if (rotate)
				{
					if (P.intensity(b,a) > 0) block.setMap(x, y, z, NSE::BC::GEO_WALL);
				}
				else
				{
					if (P.intensity(a,b) > 0) block.setMap(x, y, z, NSE::BC::GEO_WALL);
				}
			}
		}
	}
	return true;
}

template< typename NSE >
bool State<NSE>::projectPNG_Y(const std::string& filename, idx y0, bool rotate, bool mirror, bool flip, real amin, real amax, real bmin, real bmax)
{
	if (!fileExists(filename.c_str())) {
		fmt::print(stderr, "file {} does not exist\n", filename);
		return false;
	}
	PNGTool P(filename.c_str());

	for (auto& block : nse.blocks)
	{
		if (!block.isLocalY(y0)) continue;

		// plane x-z
		idx y=y0;
		for (idx z = block.offset.z(); z < block.offset.z() + block.local.z(); z++)
		{
			real a = (real)z/(real)(nse.lat.global.z() - 1); // a in [0,1]
			a = amin + a * (amax - amin); // a in [amin, amax]
			if (mirror) a = 1.0 - a;
			for (idx x = block.offset.x(); x < block.offset.x() + block.local.x(); x++)
			{
				real b = (real)x/(real)(nse.lat.global.x() - 1); // b in [0,1]
				b = bmin + b * (bmax - bmin); // b in [bmin, bmax]
				if (flip) b = 1.0 - b;
				if (rotate)
				{
					if (P.intensity(b,a) > 0) block.setMap(x, y, z, NSE::BC::GEO_WALL);
				}
				else
				{
					if (P.intensity(a,b) > 0) block.setMap(x, y, z, NSE::BC::GEO_WALL);
				}
			}
		}
	}
	return true;
}


template< typename NSE >
bool State<NSE>::projectPNG_Z(const std::string& filename, idx z0, bool rotate, bool mirror, bool flip, real amin, real amax, real bmin, real bmax)
{
	if (!fileExists(filename.c_str())) {
		fmt::print(stderr, "file {} does not exist\n", filename);
		return false;
	}
	PNGTool P(filename.c_str());

	for (auto& block : nse.blocks)
	{
		if (!block.isLocalZ(z0)) continue;

		// plane x-y
		idx z=z0;
		for (idx x = block.offset.x(); x < block.offset.x() + block.local.x(); x++)
		{
			real a = (real)x/(real)(nse.lat.global.x() - 1); // a in [0,1]
			a = amin + a * (amax - amin); // a in [amin, amax]
			if (mirror) a = 1.0 - a;
			for (idx y = block.offset.y(); y < block.offset.y() + block.local.y(); y++)
			{
				real b = (real)y/(real)(nse.lat.global.y() - 1); // b in [0,1]
				b = bmin + b * (bmax - bmin); // b in [bmin, bmax]
				if (flip) b = 1.0 - b;
				if (rotate)
				{
					if (P.intensity(b,a) > 0) block.setMap(x, y, z, NSE::BC::GEO_WALL);
				}
				else
				{
					if (P.intensity(a,b) > 0) block.setMap(x, y, z, NSE::BC::GEO_WALL);
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


template< typename NSE >
void State<NSE>::move(const std::string& srcdir, const std::string& dstdir, const std::string& srcfilename, const std::string& dstfilename)
{
	const std::string src = fmt::format("{}/{}", srcdir, srcfilename);
	const std::string dst = fmt::format("{}/{}", dstdir, dstfilename);

	// rename works only on the same filesystem
	if (rename(src.c_str(), dst.c_str()) == 0)
	{
		log("renamed {} to {}", src, dst);
		return;
	}
	if (errno != EXDEV)
	{
		perror("move: something went wrong!!!");
		return;
	}

	// manual copy data and meta data
	if (move_file(src.c_str(), dst.c_str()) == 0)
		log("moved {} to {}", src, dst);
	else
		log("move: manual move failed");
}

template< typename NSE >
template< typename... ARGS >
int State<NSE>::saveLoadTextData(int direction, const std::string& dirname, const std::string& filename, ARGS&... args)
{
	// check if main dir exists
	mkdir_p(dirname.c_str(), 0777);
	const std::string fname = fmt::format("{}/{}_rank{:03d}", dirname, filename, nse.rank);

	const std::string fmt = getSaveLoadFmt(args...);

	if (direction==MemoryToFile)
	{
		FILE* f = fopen(fname.c_str(), "wt");
		if (f==0)
		{
			log("unable to create file {}", fname);
			return 0;
		}
		fprintf(f, fmt.c_str(), args...);
		fclose(f);
		log("[saveLoadTextData: saved data into {}]", fname);
	}
	if (direction==FileToMemory)
	{
		FILE* f = fopen(fname.c_str(), "rt");
		if (f==0)
		{
			log("unable to access file {}", fname);
			return 0;
		}
		fscanf(f, fmt.c_str(), &args...);
		fclose(f);
		log("[saveLoadTextData: read data from {}]", fname);
	}
	return 1;
}

template< typename NSE >
template< typename VARTYPE >
int State<NSE>::saveloadBinaryData(int direction, const std::string& dirname, const std::string& filename, VARTYPE* data, idx length)
{
	// check if main dir exists
	mkdir_p(dirname.c_str(), 0777);
	const std::string fname = fmt::format("{}/{}_rank{:03d}", dirname, filename, nse.rank);

	if (direction==MemoryToFile)
	{
		FILE* f = fopen(fname.c_str(), "wb");
		if (f==0)
		{
			log("unable to create file {}", fname);
			return 0;
		}
		fwrite(data, sizeof(VARTYPE), length, f);
		fclose(f);
		log("[saveLoadBinaryData: saved data into {}]", fname);
	}
	if (direction==FileToMemory)
	{
		FILE* f = fopen(fname.c_str(), "rb");
		if (f==0)
		{
			log("unable to access file {}", fname);
			return 0;
		}
		fread(data, sizeof(VARTYPE), length, f);
		fclose(f);
		log("[saveLoadBinaryData: read data from {}]", fname);
	}
	return 1;
}

template< typename NSE >
void State<NSE>::saveAndLoadState(int direction, const char*subdirname)
{
	const std::string final_dirname = fmt::format("results_{}/{}", id, subdirname);
	mkdir_p(final_dirname.c_str(), 0777);

	std::string tmp_dirname;
	const char* local_scratch = getenv("LOCAL_SCRATCH");
	if (direction == FileToMemory || !local_scratch || local_scratch[0] == '\0')
	{
		// $LOCAL_SCRATCH is not defined or empty - default to regular subdirectory in results_*
		tmp_dirname = final_dirname;
		local_scratch = NULL;
	}
	else
	{
		// Write files temporarily into the local scratch and move them to final_dirname at
		// the end, after all MPI processes have completed. This avoids small buffered writes
		// into the shared filesystem on clusters as well as corruption of previous state due
		// to MPI errors...
		tmp_dirname = fmt::format("{}/{}", local_scratch, final_dirname);
	}

	char nid[200];

//	saveLoadTextData(direction, tmp_dirname, "config", "%d\n%d\n%d\n%d\n%d\n%d\n%d\n%.20le\n",
//			nse.iterations, nse.lat.global.x(), nse.lat.global.y(), nse.lat.global.z(), nse.lat.local.x(), nse.lat.local.y(), nse.lat.local.z(), nse.physFinalTime);
// FIXME: save and restore number of blocks
//	saveLoadTextData(direction, tmp_dirname, "config", nse.iterations, nse.lat.global.x(), nse.lat.global.y(), nse.lat.global.z(), nse.lat.local.x(), nse.lat.local.y(), nse.lat.local.z(), nse.physFinalTime);
	saveLoadTextData(direction, tmp_dirname, "config", nse.iterations, nse.lat.global.x(), nse.lat.global.y(), nse.lat.global.z(), nse.physFinalTime);
//	for (auto& block : nse.blocks)
//		saveLoadTextData(direction, tmp_dirname, "config", block.local.x(), block.local.y(), block.local.z(), block.offset.x(), block.offset.y(), block.offset.z());

	// save all counter states
	for (int c=0;c<MAX_COUNTER;c++)
	{
		sprintf(nid,"cnt_%d",c);
//		saveLoadTextData(direction, tmp_dirname, nid, "%d\n%le\n", cnt[c].count, cnt[c].period);
		saveLoadTextData(direction, tmp_dirname, nid, cnt[c].count, cnt[c].period);
	}

	// save probes
	for (std::size_t i=0;i<probe1Dvec.size();i++)
	{
		sprintf(nid,"probe1D_%lu",i);
//		saveLoadTextData(direction, tmp_dirname, nid, "%d\n", probe1Dvec[i].cycle);
		saveLoadTextData(direction, tmp_dirname, nid, probe1Dvec[i].cycle);
	}
	for (std::size_t i=0;i<probe1Dlinevec.size();i++)
	{
		sprintf(nid,"probe1Dline_%lu",i);
//		saveLoadTextData(direction, tmp_dirname, nid, "%d\n", probe1Dlinevec[i].cycle);
		saveLoadTextData(direction, tmp_dirname, nid, probe1Dlinevec[i].cycle);
	}
	for (std::size_t i=0;i<probe2Dvec.size();i++)
	{
		sprintf(nid,"probe2D_%lu",i);
//		saveLoadTextData(direction, tmp_dirname, nid, "%d\n", probe2Dvec[i].cycle);
		saveLoadTextData(direction, tmp_dirname, nid, probe2Dvec[i].cycle);
	}

	for (auto& block : nse.blocks)
	{
		// save DFs
		for (int dfty=0;dfty<DFMAX;dfty++)
		{
			sprintf(nid,"df_%d",dfty);
			#ifdef HAVE_MPI
			saveloadBinaryData(direction, tmp_dirname, nid, block.hfs[dfty].getData(), block.hfs[dfty].getLocalStorageSize());
			#else
			saveloadBinaryData(direction, tmp_dirname, nid, block.hfs[dfty].getData(), block.hfs[dfty].getStorageSize());
			#endif
		}

		// save map
		sprintf(nid,"map");
		#ifdef HAVE_MPI
		saveloadBinaryData(direction, tmp_dirname, nid, block.hmap.getData(), block.hmap.getLocalStorageSize());
		#else
		saveloadBinaryData(direction, tmp_dirname, nid, block.hmap.getData(), block.hmap.getStorageSize());
		#endif

		// save macro
		if (NSE::MACRO::N>0)
		{
			sprintf(nid,"macro");
			#ifdef HAVE_MPI
			saveloadBinaryData(direction, tmp_dirname, nid, block.hmacro.getData(), block.hmacro.getLocalStorageSize());
			#else
			saveloadBinaryData(direction, tmp_dirname, nid, block.hmacro.getData(), block.hmacro.getStorageSize());
			#endif
		}
	}

	if (local_scratch)
	{
		// move the files from local_scratch into final_dirname and create a backup of the existing files
		for (int i = 0; i < 2; i++)
		{
			// wait for all processes to create temporary files
			TNL::MPI::Barrier(nse.communicator);

			// first iteration: create temporary files in the destination directory
			// second iteration: rename the temporary files to the target files
			std::string src_suffix;
			std::string dst_suffix;
			if (i == 0)
			{
				log("[moving files from local scratch to temporary files in the destination directory]");
				dst_suffix = ".tmp";
			}
			else
			{
				log("[renaming temporary files to the target files]");
				src_suffix = ".tmp";
				tmp_dirname = final_dirname;
			}

			std::string src = fmt::format("config_rank{:03d}{}", nse.rank, src_suffix);
			std::string dst = fmt::format("config_rank{:03d}{}", nse.rank, dst_suffix);
			move(tmp_dirname, final_dirname, src, dst);

			// save all counter states
			for (int c=0;c<MAX_COUNTER;c++)
			{
				src = fmt::format("cnt_{:d}_rank{:03d}{}", c, nse.rank, src_suffix);
				dst = fmt::format("cnt_{:d}_rank{:03d}{}", c, nse.rank, dst_suffix);
				move(tmp_dirname, final_dirname, src, dst);
			}

			// save probes
			for (std::size_t i=0;i<probe1Dvec.size();i++)
			{
				src = fmt::format("probe1D_{:d}_rank{:03d}{}", i, nse.rank, src_suffix);
				dst = fmt::format("probe1D_{:d}_rank{:03d}{}", i, nse.rank, dst_suffix);
				move(tmp_dirname, final_dirname, src, dst);
			}
			for (std::size_t i=0;i<probe1Dlinevec.size();i++)
			{
				src = fmt::format("probe1Dline_{:d}_rank{:03d}{}", i, nse.rank, src_suffix);
				dst = fmt::format("probe1Dline_{:d}_rank{:03d}{}", i, nse.rank, dst_suffix);
				move(tmp_dirname, final_dirname, src, dst);
			}
			for (std::size_t i=0;i<probe2Dvec.size();i++)
			{
				src = fmt::format("probe2D_{:d}_rank{:03d}{}", i, nse.rank, src_suffix);
				dst = fmt::format("probe2D_{:d}_rank{:03d}{}", i, nse.rank, dst_suffix);
				move(tmp_dirname, final_dirname, src, dst);
			}

			// save DFs
			for (int dfty=0;dfty<DFMAX;dfty++)
			{
				src = fmt::format("df_{:d}_rank{:03d}{}", dfty, nse.rank, src_suffix);
				dst = fmt::format("df_{:d}_rank{:03d}{}", dfty, nse.rank, dst_suffix);
				move(tmp_dirname, final_dirname, src, dst);
			}

			// save map
			src = fmt::format("map_rank{:03d}{}", nse.rank, src_suffix);
			dst = fmt::format("map_rank{:03d}{}", nse.rank, dst_suffix);
			move(tmp_dirname, final_dirname, src, dst);

			// save macro
			if (NSE::MACRO::N>0)
			{
				src = fmt::format("macro_rank{:03d}{}", nse.rank, src_suffix);
				dst = fmt::format("macro_rank{:03d}{}", nse.rank, dst_suffix);
				move(tmp_dirname, final_dirname, src, dst);
			}
		}
	}
}

template< typename NSE >
void State<NSE>::saveState(bool forced)
{
//	flagCreate("do_save_state");
	if (flagExists("savestate") || !check_savestate_flag || forced)
	{
		log("[saveState invoked]");
		saveAndLoadState(MemoryToFile, "current_state");
		if (delete_savestate_flag && !forced)
		{
			flagDelete("savestate");
//			flagRename("savestate","savestate_done");
			flagCreate("savestate_done");
		}
		if (forced) flagCreate("loadstate");
	}
	// debug
//	saveAndLoadState(FileToMemory, "current_state");
}

template< typename NSE >
void State<NSE>::loadState(bool forced)
{
//	flagCreate("do_save_state");
	if (flagExists("loadstate") || forced)
	{
		log("[loadState invoked]");
//		printf("Provadim cteni df\n");
		saveAndLoadState(FileToMemory, "current_state");
//		if (delete_savestate_flag)
//			flagDelete("savestate");
//			flagRename("savestate","savestate_saved");
	}
	// debug
//	saveAndLoadState(FileToMemory, "current_state");
}

template< typename NSE >
bool State<NSE>::wallTimeReached()
{
	bool local_result = false;
	if (wallTime > 0)
	{
		long actualtimediff = timer_total.getRealTime();
		if (actualtimediff >= wallTime)
		{
			log("wallTime reached: {} / {} [sec]", actualtimediff, wallTime);
			local_result = true;
		}
	}
	return TNL::MPI::reduce(local_result, MPI_LOR, nse.communicator);
}

template< typename NSE >
double State<NSE>::getWallTime(bool collective)
{
	double result = 0;
	if (!collective || nse.rank == 0)
	{
		result = timer_total.getRealTime();
	}
	if (collective)
	{
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


template< typename NSE >
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

	long long CPUavail = sysconf(_SC_PHYS_PAGES)*sysconf(_SC_PAGE_SIZE);
	long long GPUavail = 0;
	long long GPUtotal = 0;
	long long GPUtotal_hw = 0;
	long long CPUtotal = memMacro + memMap + DFMAX*memDFs;
	long long CPUDFs = DFMAX*memDFs;
	#ifdef USE_CUDA
	GPUavail = 0;
	GPUtotal_hw =0;
	GPUtotal += DFMAX*memDFs + memMacro + memMap;
//	CPUDFs = 0;

	// get number of CUDA GPUs
//	int num_gpus=0;
//	cudaGetDeviceCount(&num_gpus);

	// display CPU and GPU configuration
//	log("number of CUDA devices:\t{}", num_gpus);
//	for (int i = 0; i < num_gpus; i++)
	{
		int gpu_id;
		cudaGetDevice(&gpu_id);
		cudaDeviceProp dprop;
		cudaGetDeviceProperties(&dprop, gpu_id);
		log("Rank {} uses GPU id {}: {}", nse.rank, gpu_id, dprop.name);
		// NOTE: cudaSetDevice breaks MPI !!!
//		cudaSetDevice(i);
		size_t free=0, total=0;
		cudaMemGetInfo(&free, &total);
		GPUavail += free;
		GPUtotal_hw += total;
	}

	#else
//	CPUtotal += CPUDFs;
	#endif

	log("Local memory budget analysis / estimation for MPI rank {}", nse.rank);
	log("CPU RAM for DFs:   {:d} MiB", CPUDFs/1024/1024);
//	log("CPU RAM for lat:   {:d} MiB", memDFs/1024/1024);
	log("CPU RAM for map:   {:d} MiB", memMap/1024/1024);
	log("CPU RAM for macro: {:d} MiB", memMacro/1024/1024);
	log("TOTAL CPU RAM {:d} MiB estimated needed, {:d} MiB available ({:6.4f}%)", CPUtotal/1024/1024, CPUavail/1024/1024, 100.0*CPUtotal/CPUavail);
	#ifdef USE_CUDA
	log("GPU RAM for DFs:   {:d} MiB", DFMAX*memDFs/1024/1024);
	log("GPU RAM for map:   {:d} MiB", memMap/1024/1024);
	log("GPU RAM for macro: {:d} MiB", memMacro/1024/1024);
	log("TOTAL GPU RAM {:d} MiB estimated needed, {:d} MiB available ({:6.4f}%), total GPU RAM: {:d} MiB", GPUtotal/1024/1024, GPUavail/1024/1024, 100.0*GPUtotal/GPUavail, GPUtotal_hw/1024/1024);
	if (GPUavail <= GPUtotal) return false;
	#endif
	if (CPUavail <= CPUtotal) return false;
	return true;
}

// clear Lattice and boundary setup
template< typename NSE >
void State<NSE>::reset()
{
	nse.resetMap(NSE::BC::GEO_FLUID);
	setupBoundaries();		// this can be virtualized
	resetLattice(1.0, 0, 0, 0);
//	resetLattice(1.0, lbmInputVelocityX(), lbmInputVelocityY(),lbmInputVelocityZ());
}

template< typename NSE >
void State<NSE>::resetLattice(real rho, real vx, real vy, real vz)
{
	// NOTE: it is important to reset *all* lattice sites (i.e. including ghost layers) when using the A-A pattern
	// (because GEO_INFLOW and GEO_OUTFLOW_EQ access the ghost layer in streaming)
	nse.forAllLatticeSites( [&] (BLOCK_NSE& block, idx x, idx y, idx z) {
		block.setEqLat(x,y,z,rho,vx,vy,vz);
	} );
}

template< typename NSE >
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

	log("MPI info: rank={:d}, nproc={:d}, lat.global=[{:d},{:d},{:d}]", nse.rank, nse.nproc, nse.lat.global.x(), nse.lat.global.y(), nse.lat.global.z());
	for (auto& block : nse.blocks)
		log("LBM block {:d}: local=[{:d},{:d},{:d}], offset=[{:d},{:d},{:d}]", block.id, block.local.x(), block.local.y(), block.local.z(), block.offset.x(), block.offset.y(), block.offset.z());

	log("\nSTART: simulation NSE:{} lbmVisc {:e} physDl {:e} physDt {:e}", NSE::COLL::id, nse.lbmViscosity(), nse.lat.physDl, nse.physDt);

	// reset counters
	for (int c=0;c<MAX_COUNTER;c++) cnt[c].count = 0;
	cnt[SAVESTATE].count = 1;  // skip initial save of state
	nse.iterations = 0;

	// check for loadState
//	if(flagExists("current_state/df_0"))
	if(flagExists("loadstate"))
	{
		loadState(); // load saved state into CPU memory
		nse.physStartTime = nse.physTime();
		nse.allocateDeviceData();
	}
	else
	{
		// allocate before reset - it might initialize on the GPU...
		nse.allocateDeviceData();

		// setup map and DFs in CPU memory
		reset();

		for (auto& block : nse.blocks)
		{
			// create LBM_DATA with host pointers
			typename NSE::DATA SD;
			for (uint8_t dfty=0;dfty<DFMAX;dfty++)
				SD.dfs[dfty] = block.hfs[dfty].getData();
			#ifdef HAVE_MPI
			SD.indexer = block.hmap.getLocalView().getIndexer();
			#else
			SD.indexer = block.hmap.getIndexer();
			#endif
			SD.XYZ = SD.indexer.getStorageSize();
			SD.dmap = block.hmap.getData();
			SD.dmacro = block.hmacro.getData();

			// initialize macroscopic quantities on CPU
			#pragma omp parallel for schedule(static) collapse(2)
			for (idx x = 0; x < block.local.x(); x++)
			for (idx z = 0; z < block.local.z(); z++)
			for (idx y = 0; y < block.local.y(); y++)
				LBMKernelInit<NSE>(SD, x, y, z);
		}
	}

	copyAllToDevice();

#ifdef HAVE_MPI
	// synchronize overlaps with MPI (initial synchronization can be synchronous)
	nse.synchronizeMapDevice();
	nse.synchronizeDFsAndMacroDevice(df_cur);
#endif

	timer_SimInit.stop();
}

template< typename NSE >
void State<NSE>::SimUpdate()
{
	timer_SimUpdate.start();

	// debug
	for (auto& block : nse.blocks)
	if (block.data.lbmViscosity == 0) {
		log("error: LBM viscosity is 0");
		nse.terminate = true;
		return;
	}

	// flags
	bool doComputeVelocitiesStar=false;
	bool doCopyQuantitiesStarToHost=false;
	bool doZeroForceOnDevice=false;
	bool doZeroForceOnHost=false;
	bool doComputeLagrangePhysics=false;
	bool doCopyForceToDevice=false;

	// determine global flags
	// NOTE: all Lagrangian points are assumed to be on the first GPU
	// TODO
//	if (nse.data.rank == 0 && FF.size() > 0)
	if (FF.size() > 0)
	{
		doComputeLagrangePhysics=true;
		for (std::size_t i=0;i<FF.size();i++)
		if (FF[i].implicitWuShuForcing)
		{
			doComputeVelocitiesStar=true;
			switch (FF[i].ws_compute)
			{
				case ws_computeCPU:
				case ws_computeCPU_TNL:
					doCopyQuantitiesStarToHost=true;
					doZeroForceOnHost=true;
					doCopyForceToDevice=true;
					break;
				case ws_computeGPU_TNL:
				case ws_computeHybrid_TNL:
				case ws_computeHybrid_TNL_zerocopy:
				case ws_computeGPU_CUSPARSE:
				case ws_computeHybrid_CUSPARSE:
					doZeroForceOnDevice=true;
					break;
			}
		}
	}

	#ifdef USE_CUDA
	auto get_grid_size = [] (const auto& block, idx x = 0, idx y = 0, idx z = 0) -> dim3
	{
		dim3 gridSize;
		if (x > 0)
			gridSize.x = x;
		else
			gridSize.x = TNL::roundUpDivision(block.local.x(), block.block_size.x());
		if (y > 0)
			gridSize.y = y;
		else
			gridSize.y = TNL::roundUpDivision(block.local.y(), block.block_size.y());
		if (z > 0)
			gridSize.z = z;
		else
			gridSize.z = TNL::roundUpDivision(block.local.z(), block.block_size.z());

		return gridSize;
	};
	#endif

	if (doComputeVelocitiesStar)
	{
		for (auto& block : nse.blocks)
		{
		#ifdef USE_CUDA
			const dim3 blockSize = {unsigned(block.block_size.x()), unsigned(block.block_size.y()), unsigned(block.block_size.z())};
			const dim3 gridSize = get_grid_size(block);
			if (doZeroForceOnDevice)
				cudaLBMComputeVelocitiesStarAndZeroForce< NSE ><<<gridSize, blockSize>>>(block.data, nse.rank, nse.nproc);
			else
				cudaLBMComputeVelocitiesStar< NSE ><<<gridSize, blockSize>>>(block.data, nse.rank, nse.nproc);
			checkCudaDevice;
		#else
			#pragma omp parallel for schedule(static) collapse(2)
			for (idx x = 0; x < block.local.x(); x++)
			for (idx z = 0; z < block.local.z(); z++)
			for (idx y = 0; y < block.local.y(); y++)
			if (doZeroForceOnDevice)
				LBMComputeVelocitiesStarAndZeroForce< NSE >(block.data, nse.rank, nse.nproc, x, y, z);
			else
				LBMComputeVelocitiesStar< NSE >(block.data, nse.rank, nse.nproc, x, y, z);
		#endif
		}
		if (doCopyQuantitiesStarToHost)
		{
			nse.copyMacroToHost();
		}
	}


	// reset lattice force vectors dfx and dfy
	if (doZeroForceOnHost)
	{
		nse.resetForces();
	}

	if (doComputeLagrangePhysics)
	{
		computeAllLagrangeForces();
	}

	if (doCopyForceToDevice)
	{
		nse.copyForcesToDevice();
	}


	// call hook method (used e.g. for extra kernels in the non-Newtonian model)
	computeBeforeLBMKernel();


#ifdef AA_PATTERN
	uint8_t output_df = df_cur;
#endif
#ifdef AB_PATTERN
	uint8_t output_df = df_out;
#endif

#ifdef USE_CUDA
	#ifdef HAVE_MPI
	if (nse.nproc == 1)
	{
	#endif
		timer_compute.start();
		for (auto& block : nse.blocks)
		{
			const dim3 blockSize = {unsigned(block.block_size.x()), unsigned(block.block_size.y()), unsigned(block.block_size.z())};
			const dim3 gridSize = get_grid_size(block);
			cudaLBMKernel< NSE ><<<gridSize, blockSize>>>(block.data, nse.rank, nse.nproc, (idx) 0);
		}
		cudaDeviceSynchronize();
		checkCudaDevice;
		// copying of overlaps is not necessary for nproc == 1 (nproc is checked in streaming as well)
		timer_compute.stop();
	#ifdef HAVE_MPI
	}
	else
	{
		timer_compute.start();
		timer_compute_overlaps.start();

		for (auto& block : nse.blocks)
		{
			const dim3 blockSize = {unsigned(block.block_size.x()), unsigned(block.block_size.y()), unsigned(block.block_size.z())};
			const dim3 gridSizeForBoundary = get_grid_size(block, block.df_overlap_X());
			const dim3 gridSizeForInternal = get_grid_size(block, block.local.x() - 2*block.df_overlap_X());

			// get CUDA streams
			const cudaStream_t cuda_stream_left = block.streams.at(block.left_id);
			const cudaStream_t cuda_stream_right = block.streams.at(block.right_id);
			const cudaStream_t cuda_stream_main = block.streams.at(block.id);

			// compute on boundaries (NOTE: 1D distribution is assumed)
			cudaLBMKernel< NSE ><<<gridSizeForBoundary, blockSize, 0, cuda_stream_left>>>(block.data, block.id, nse.total_blocks, (idx) 0);
			cudaLBMKernel< NSE ><<<gridSizeForBoundary, blockSize, 0, cuda_stream_right>>>(block.data, block.id, nse.total_blocks, block.local.x() - block.df_overlap_X());

			// compute on internal lattice sites
			cudaLBMKernel< NSE ><<<gridSizeForInternal, blockSize, 0, cuda_stream_main>>>(block.data, block.id, nse.total_blocks, block.df_overlap_X());
		}

		// wait for the computations on boundaries to finish
		// TODO: pipeline the stream synchronization with the MPI synchronizer (wait using CUDA stream events in the DistributedNDArraySynchronizer)
		for (auto& block : nse.blocks)
		{
			const cudaStream_t cuda_stream_left = block.streams.at(block.left_id);
			const cudaStream_t cuda_stream_right = block.streams.at(block.right_id);

			cudaStreamSynchronize(cuda_stream_left);
			cudaStreamSynchronize(cuda_stream_right);
		}
		timer_compute_overlaps.stop();

		// exchange the latest DFs and dmacro on overlaps between blocks
		// (it is important to wait for the communication before waiting for the computation, otherwise MPI won't progress)
		timer_wait_communication.start();
		nse.synchronizeDFsAndMacroDevice(output_df);
		timer_wait_communication.stop();

		// wait for the computation on the interior to finish
		timer_wait_computation.start();
		for (auto& block : nse.blocks)
		{
			const cudaStream_t cuda_stream_main = block.streams.at(block.id);
			cudaStreamSynchronize(cuda_stream_main);
		}

		// synchronize the whole GPU and check errors
		cudaDeviceSynchronize();
		checkCudaDevice;
		timer_wait_computation.stop();

		timer_compute.stop();
	}
	#endif
#else
	timer_compute.start();
	for (auto& block : nse.blocks)
	{
		#pragma omp parallel for schedule(static) collapse(2)
		for (idx x=0; x<block.local.x(); x++)
		for (idx z=0; z<block.local.z(); z++)
		for (idx y=0; y<block.local.y(); y++)
		{
			LBMKernel< NSE >(block.data, x, y, z, nse.rank, nse.nproc);
		}
	}
	timer_compute.stop();
	#ifdef HAVE_MPI
	// TODO: overlap computation with synchronization, just like above
	timer_wait_communication.start();
	nse.synchronizeDFsAndMacroDevice(output_df);
	timer_wait_communication.stop();
	#endif
#endif

	nse.iterations++;

	bool doit=false;
	for (int c=0;c<MAX_COUNTER;c++) if (c!=PRINT && c!=SAVESTATE) if (cnt[c].action(nse.physTime())) doit = true;
	if (doit)
	{
		// common copy
		nse.copyMacroToHost();
		// to be able to compute rho, vx, vy, vz etc... based on DFs on CPU to save GPU memory FIXME may not work with ESOTWIST
		if (NSE::CPU_MACRO::N>0)
			nse.copyDFsToHost(output_df);
	}

	timer_SimUpdate.stop();
}

template< typename NSE >
void State<NSE>::AfterSimUpdate()
{
	timer_AfterSimUpdate.start();

	// call hook method (used e.g. for the coupled LBM-MHFEM solver)
	computeAfterLBMKernel();

	bool write_info = false;

	if (cnt[VTK1D].action(nse.physTime()) ||
	    cnt[VTK2D].action(nse.physTime()) ||
	    cnt[VTK3D].action(nse.physTime()) ||
	    cnt[VTK3DCUT].action(nse.physTime()) ||
	    cnt[PROBE1].action(nse.physTime()) ||
	    cnt[PROBE2].action(nse.physTime()) ||
	    cnt[PROBE5].action(nse.physTime()) ||
	    cnt[PROBE3].action(nse.physTime())
	    )
	{
		// cpu macro
		nse.computeCPUMacroFromLat();
		// probe1
		if (cnt[PROBE1].action(nse.physTime()))
		{
			probe1();
			cnt[PROBE1].count++;
		}
		// probe2
		if (cnt[PROBE2].action(nse.physTime()))
		{
			probe2();
			cnt[PROBE2].count++;
		}
		// probe3
		if (cnt[PROBE3].action(nse.physTime()))
		{
			probe3();
			cnt[PROBE3].count++;
		}
		// probe5
		if (cnt[PROBE5].action(nse.physTime()))
		{
			probe5();
			cnt[PROBE5].count++;
		}
		// 3D VTK
		if (cnt[VTK3D].action(nse.physTime()))
		{
			writeVTKs_3D();
			cnt[VTK3D].count++;
		}
		// 3D VTK CUT
		if (cnt[VTK3DCUT].action(nse.physTime()))
		{
			writeVTKs_3Dcut();
			cnt[VTK3DCUT].count++;
		}
		// 2D VTK
		if (cnt[VTK2D].action(nse.physTime()))
		{
			writeVTKs_2D();
			cnt[VTK2D].count++;
		}
		// 1D VTK
		if (cnt[VTK1D].action(nse.physTime()))
		{
			writeVTKs_1D();
			cnt[VTK1D].count++;
		}
		write_info = true;
	}

	if (cnt[PRINT].action(nse.physTime()))
	{
		write_info = true;
		cnt[PRINT].count++;
	}

	// statReset is called after all probes and VTK output
	// copy macro from host to device after reset
	if (cnt[STAT_RESET].action(nse.physTime()))
	{
		statReset();
		nse.copyMacroToDevice();
		cnt[STAT_RESET].count++;
	}
	if (cnt[STAT2_RESET].action(nse.physTime()))
	{
		stat2Reset();
		nse.copyMacroToDevice();
		cnt[STAT2_RESET].count++;
	}

	// only the first process writes GLUPS
	// getting the rank from MPI_COMM_WORLD is intended here - other ranks may be redirected to a file when the ranks are reordered
	if (TNL::MPI::GetRank(MPI_COMM_WORLD) == 0)
	if (nse.iterations > 1)
	if (write_info)
	{
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
		double ETA = getWallTime() * (nse.physFinalTime - nse.physTime()) / (nse.physTime() - nse.physStartTime);

		if (verbosity > 0)
		{
			log("GLUPS={:.3f} iter={:d} t={:1.3f}s dt={:1.2e} lbmVisc={:1.2e} WT={:.0f}s ETA={:.0f}s",
				LUPS * 1e-9,
				nse.iterations,
				nse.physTime(),
				nse.physDt,
				nse.lbmViscosity(),
				getWallTime(),
				ETA
			);
		}
	}

	timer_AfterSimUpdate.stop();
}

template< typename NSE >
void State<NSE>::AfterSimFinished()
{
	// only the first process writes the info
	if (TNL::MPI::GetRank(MPI_COMM_WORLD) == 0)
	if (nse.iterations > 1)
	if (verbosity > 0)
	{
		log("total walltime: {:.1f} s, SimInit time: {:.1f} s, SimUpdate time: {:.1f} s, AfterSimUpdate time: {:.1f} s",
			getWallTime(),
			timer_SimInit.getRealTime(),
			timer_SimUpdate.getRealTime(),
			timer_AfterSimUpdate.getRealTime()
		);
		log("compute time: {:.1f} s, compute overlaps time: {:.1f} s, wait for communication time: {:.1f} s, wait for computation time: {:.1f} s",
			timer_compute.getRealTime(),
			timer_compute_overlaps.getRealTime(),
			timer_wait_communication.getRealTime(),
			timer_wait_computation.getRealTime()
		);
		const double avgLUPS = nse.lat.global.x() * nse.lat.global.y() * nse.lat.global.z() * (nse.iterations / (timer_SimUpdate.getRealTime() + timer_AfterSimUpdate.getRealTime()));
		const double computeLUPS = nse.lat.global.x() * nse.lat.global.y() * nse.lat.global.z() * (nse.iterations / timer_compute.getRealTime());
		log("final GLUPS: average (based on SimInit + SimUpdate + AfterSimUpdate time): {:.3f}, based on compute time: {:.3f}",
			avgLUPS * 1e-9,
			computeLUPS * 1e-9
		);
	}
}

template< typename NSE >
void State<NSE>::updateKernelData()
{
	nse.updateKernelData();

	// this is not in nse.updateKernelData so that it can be overridden for ADE
	for( auto& block : nse.blocks )
		block.data.lbmViscosity = nse.lbmViscosity();
}

template< typename NSE >
void State<NSE>::copyAllToDevice()
{
	nse.copyMapToDevice();
	nse.copyDFsToDevice();
	nse.copyMacroToDevice();  // important when a state has been loaded
}

template< typename NSE >
void State<NSE>::copyAllToHost()
{
	nse.copyMapToHost();
	nse.copyDFsToHost();
	nse.copyMacroToHost();
}
