#pragma once

#include "lbm3d/lagrange_3D.h"
#include <vector>
//int n2=0;
//int n1=0;

template <typename LBM>
int ibmSetupFilament(Lagrange3D<LBM>& ibm, typename LBM::point_t center, double sigma, double length)
{
	using real = typename Lagrange3D<LBM>::real;
	using point_t = typename Lagrange3D<LBM>::point_t;
	// based on sigma, estimate N-number of points
	// sigma is the maximal diagonal of a quasi-square that 4 points on the rectangle area form
	// the points do not have to be between y=0 and y=Y-1 sharp, but equidistantly spaced as close to sigma as possible
	//number of points on z direction
	int N = ceil(length/sigma);
	//spacing dx between points on length direction
	real dx = length/N;
	
	real dy = dx;
	//dm centering the rectangle
	real dm = -length/2; 
	
	int points=0;
	for (int i=0;i<N;i++)
	{
		point_t fp3;
		real x = 0;
		real y = i*dx;
		real z = i*dx; 
		
		fp3.x() = center.x()+x;

		fp3.y() = center.y()+y;

		fp3.z() = center.z() + dm+z;

		ibm.LL.push_back(fp3);
		points++;
	}
	spdlog::info("added {} lagrangian points", points);

	// compute sigma: take lag grid into account
	//ibm.computeMaxMinDist();
	//real sigma_min = ibm.minDist;
	//real sigma_max = ibm.maxDist;

	real sigma_min = ibm.computeMinDist();
	real sigma_max = ibm.computeMaxDistFromMinDist(sigma_min);

	spdlog::info("String: wanted sigma {:e} dx={:e} dm={:e} ({:d} points total, N={:d}) sigma_min {:e}, sigma_max {:e}", sigma, dx, dm, points, N, sigma_min, sigma_max);
	spdlog::info("h=physdl {:e} sigma min {:e} sigma/h {:e}", ibm.lbm.lat.physDl, sigma_min, sigma_min/ibm.lbm.lat.physDl);
	spdlog::info("h=physdl {:e} sigma max {:e} sigma/h {:e}", ibm.lbm.lat.physDl, sigma_max, sigma_max/ibm.lbm.lat.physDl);

return N;

}

template < typename LBM>
std::pair<int, int> ibmSetupRectangle(Lagrange3D<LBM>& ibm, typename LBM::point_t center, double sigma,
	double width, double height, double angleX=0, double angleY=0, double angleZ =0)
{
	using real = typename Lagrange3D<LBM>::real;
	using point_t = typename Lagrange3D<LBM>::point_t;
	// based on sigma, estimate N-number of points
	// sigma is the maximal diagonal of a quasi-square that 4 points on the rectangle area form
	// the points do not have to be between y=0 and y=Y-1 sharp, but equidistantly spaced as close to sigma as possible
	//number of points on z direction
	int N1 = ceil(width/sigma);
	//spaccing dx between points on width direction
	real dx = width/N1;
	
	int N2 = ceil(height/sigma);
	real dy = height/N2;
	//dm centering the rectangle
	real dm = -(width+height)/4; 
	
	int points=0;
	for (int i=0;i<N1;i++)
	for (int j=0;j<N2;j++)// y-direction-height
	{
		point_t fp3;
		real x = 0;
		real y = i*dx;
		real z = j*dx; 
		//general rotation matrix
		//extrinsic rotation-euler angles
		fp3.x() = center.x() + cos(angleY)*cos(angleZ)*x + 
		(sin(angleX)*sin(angleY)*cos(angleZ) - cos(angleX)*sin(angleZ))*y + 
		(cos(angleX)*sin(angleY)*cos(angleZ) + sin(angleX)*sin(angleZ))*z;

		fp3.y() = center.y() + cos(angleY)*cos(angleZ)*x + 
		(sin(angleX)*sin(angleY)*sin(angleZ) + cos(angleX)*cos(angleZ))*y + 
		(cos(angleX)*sin(angleY)*sin(angleZ) - sin(angleZ)*sin(angleZ))*z;

		fp3.z() = center.z() + dm + (-sin(angleY))*x + 
		sin(angleX)*cos(angleY)*y + 
		cos(angleX)*cos(angleY)*z;
		

		ibm.LL.push_back(fp3);
		points++;
	}
	spdlog::info("added {} lagrangian points", points);

	// compute sigma: take lag grid into account
	//ibm.computeMaxMinDist();
	//real sigma_min = ibm.minDist;
	//real sigma_max = ibm.maxDist;

	real sigma_min = ibm.computeMinDist();
	real sigma_max = ibm.computeMaxDistFromMinDist(sigma_min);

	spdlog::info("Rectangle: wanted sigma {:e} dx={:e} dm={:e} ({:d} points total, N1={:d} N2={:d}) sigma_min {:e}, sigma_max {:e}", sigma, dx, dm, points, N1, N2, sigma_min, sigma_max);
	spdlog::info("h=physdl {:e} sigma min {:e} sigma/h {:e}", ibm.lbm.lat.physDl, sigma_min, sigma_min/ibm.lbm.lat.physDl);
	spdlog::info("h=physdl {:e} sigma max {:e} sigma/h {:e}", ibm.lbm.lat.physDl, sigma_max, sigma_max/ibm.lbm.lat.physDl);

return std::pair<int,int>{N1,N2};
}


template < typename LBM >
void ibmSetupCylinder(Lagrange3D<LBM>& ibm, typename LBM::point_t center, double diameter, double sigma)
{
	using real = typename Lagrange3D<LBM>::real;
	using point_t = typename Lagrange3D<LBM>::point_t;

	// based on sigma, estimate N
	// sigma is the maximal diagonal of a quasi-square that 4 points on the cylinder surface form
	// the points do not have to be between y=0 and y=Y-1 sharp, but equidistantly spaced as close to sigma as possible
	int N2 = ceil(sqrt(2.0) * PI * diameter / sigma  ); // minimal number of N2 points
	real dx = PI*diameter/((real)N2);
	//global-global size of the lattice

	real W = ibm.lbm.lat.physDl*(ibm.lbm.lat.global.y()-2);
	int N1 = floor( W / dx );
	real dm = (W - N1*dx)/2.0;
	real radius = diameter/2.0;

	// compute the amount of N for the lowest radius such that min_dist
	int points=0;
	for (int i=0;i<N1;i++) // y-direction
	for (int j=0;j<N2;j++)
	{
		point_t fp3;
		fp3.x() = center.x() + radius * cos( 2.0*PI*j/((real)N2) + PI);
		fp3.y() = dm + i * dx;
		fp3.z() = center.z() + radius * sin( 2.0*PI*j/((real)N2) + PI);
		ibm.LL.push_back(fp3);
		points++;
	}
	spdlog::info("added {} lagrangian points", points);

	// compute sigma: take lag grid into account
	//ibm.computeMaxMinDist();
	//real sigma_min = ibm.minDist;
	//real sigma_max = ibm.maxDist;

	real sigma_min = ibm.computeMinDist();
	real sigma_max = ibm.computeMaxDistFromMinDist(sigma_min);

	spdlog::info("Cylinder: wanted sigma {:e} dx={:e} dm={:e} ({:d} points total, N1={:d} N2={:d}) sigma_min {:e}, sigma_max {:e}", sigma, dx, dm, points, N1, N2, sigma_min, sigma_max);
//	spdlog::info("Added {} Lagrangian points (requested {}) partial area {:e}", Ncount, N, a);
	spdlog::info("h=physdl {:e} sigma min {:e} sigma/h {:e}", ibm.lbm.lat.physDl, sigma_min, sigma_min/ibm.lbm.lat.physDl);
	spdlog::info("h=physdl {:e} sigma max {:e} sigma/h {:e}", ibm.lbm.lat.physDl, sigma_max, sigma_max/ibm.lbm.lat.physDl);
}

// ball discretization algorithm: https://www.cmu.edu/biolphys/deserno/pdf/sphere_equi.pdf
template < typename LBM >
void ibmDrawSphere(Lagrange3D<LBM>& ibm, typename LBM::point_t center, double radius, double sigma)
{
	using real = typename Lagrange3D<LBM>::real;
	using point_t = typename Lagrange3D<LBM>::point_t;

	// based on sigma, estimate N
	real surface = 4.0*PI*radius*radius;
	// sigma is diagonal of a "quasi-square" that 4 points on the sphere surface form
	// so the quasi-square has area = b^2 where b^2 + b^2 = sigma^2, i.e., b^2 = 1/2*sigma^2
	real wanted_unit_area = sigma*sigma / 2.0;
	// count how many of these
	real count = surface/wanted_unit_area;
	int N = ceil(count);

	int points=0;
//	real a = 4.0*PI*radius*radius/(real)N;
	real a = 4.0*PI/(real)N;
	real d = sqrt(a);
//	int Mtheta = (int)(PI/d);
	int Mtheta = floor(PI/d);
	real dtheta = PI/Mtheta;
	real dphi = a/dtheta;
	for (int m = 0; m < Mtheta; m++)
	{
		// for a given phi and theta:
		real theta = PI*(m+0.5)/(real)Mtheta;
//		int Mphi = (int)(2.0*PI*sin(theta)/dphi);
		int Mphi = floor(2.0*PI*sin(theta)/dphi);
		for (int n = 0; n<Mphi; n++)
		{
			real phi = 2.0*PI*n/(real)Mphi;
			point_t fp;
			fp.x() = center.x() + radius * cos( phi ) * sin( theta );
			fp.y() = center.y() + radius * sin( phi ) * sin( theta );
			fp.z() = center.z() + radius * cos( theta );
			ibm.LL.push_back(fp);
			points++;
		}
	}
	spdlog::info("added {} lagrangian points", points);

	real sigma_min = ibm.computeMinDist();
	real sigma_max = ibm.computeMaxDistFromMinDist(sigma_min);

	spdlog::info("Ball surface: wanted sigma {:e} ({:f} i.e. {:d} points), wanted_unit_area {:e}, sigma_min {:e}, sigma_max {:e}", sigma, count, N, wanted_unit_area, sigma_min, sigma_max);
//	spdlog::info("Added {} Lagrangian points (requested {}) partial area {:e}",points, N, a);
	spdlog::info("h=physdl {:e} sigma min {:e} sigma/h {:e}", ibm.lbm.lat.physDl, sigma_min, sigma_min/ibm.lbm.lat.physDl);
	spdlog::info("h=physdl {:e} sigma max {:e} sigma/h {:e}", ibm.lbm.lat.physDl, sigma_max, sigma_max/ibm.lbm.lat.physDl);
}
