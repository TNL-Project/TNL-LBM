#include "lbm3d/core.h"
#include "lbm3d/lagrange_3D.h"
#include "lbm3d/obstacles_ibm.h"
// bouncing ball in 3D
// IBM-LBM
#include "lbm3d/obstacles_lbm.h"
#include "lbm3d/pibm.h"


template < typename TRAITS >
struct MacroLocal : D3Q27_MACRO_Base< TRAITS >
{
	using dreal = typename TRAITS::dreal;
	using idx = typename TRAITS::idx;

	enum { e_fx, e_fy, e_fz, e_vx, e_vy, e_vz, e_rho, N };

	template < typename LBM_DATA, typename LBM_KS >
	CUDA_HOSTDEV static void outputMacro(LBM_DATA &SD, LBM_KS &KS, idx x, idx y, idx z)
	{
		SD.macro(e_rho, x, y, z) = KS.rho;
		SD.macro(e_vx, x, y, z)  = KS.vx;
		SD.macro(e_vy, x, y, z)  = KS.vy;
		SD.macro(e_vz, x, y, z)  = KS.vz;
	}

	template < typename LBM_DATA >
	CUDA_HOSTDEV static void zeroForces(LBM_DATA &SD, idx x, idx y, idx z)
	{
		SD.macro(e_fx, x, y, z) = 0;
		SD.macro(e_fy, x, y, z) = 0;
		SD.macro(e_fz, x, y, z) = 0;
	}

	template < typename LBM_DATA, typename LBM_KS >
	CUDA_HOSTDEV static void copyQuantities(LBM_DATA &SD, LBM_KS &KS, idx x, idx y, idx z)
	{
		KS.lbmViscosity = SD.lbmViscosity;
		KS.fx = SD.macro(e_fx, x, y, z);
		KS.fy = SD.macro(e_fy, x, y, z);
		KS.fz = SD.macro(e_fz, x, y, z);
	}
};

template < typename NSE >
struct StateLocal : State<NSE>
{
	using TRAITS = typename NSE::TRAITS;
	using BC = typename NSE::BC;
	using MACRO = typename NSE::MACRO;
	using BLOCK = LBM_BLOCK< NSE >;

	using State<NSE>::nse;
	using State<NSE>::ibm;
	using State<NSE>::vtk_helper;
	using State<NSE>::id;

	using idx = typename TRAITS::idx;
	using real = typename TRAITS::real;
	using dreal = typename TRAITS::dreal;
	using point_t = typename TRAITS::point_t;
	using lat_t = Lattice<3, real, idx>;

	dreal lbm_inflow_vx = 0;
	bool firstrun=true;
	dreal ball_diameter=0.01;
	point_t ball_c;
	dreal ball_amplitude = 0;
	dreal ball_period = 1;

	//flag
	////////////////////////////////////////
	//flag coordinates
    int N1=0;
    int N2=0;

	dreal sigma=0.0001;
	dreal gama=0.0001;
	dreal density = 1;
	//dreal fi11 = 10*10*10;
	//dreal fi22 = fi11;
	//dreal fi12 =10;
	dreal kappa = pow(10,5);
	int L = N1;
	int H = N2;
	bool by_s1 =true;
	bool by_s2 = false;
	//gravity neglected g=0

	using HLPVECTOR = decltype(ibm.hLL_lat);
	using DLPVECTOR = decltype(ibm.dLL_lat);
	HLPVECTOR previous;
	HLPVECTOR next;

	DLPVECTOR previous_device;
	DLPVECTOR next_device;

	//fiber
	//////////////////////////////
	bool add_ball = false;
	bool free_end = false;

    int N=0;
    HLPVECTOR ref;
	HLPVECTOR  LL;
	HLPVECTOR  U;
	HLPVECTOR  Y;
	HLPVECTOR  previousY;
	HLPVECTOR  nextY;
	HLPVECTOR F_bE;
	HLPVECTOR F_bK;

	DLPVECTOR ref_device;
	DLPVECTOR  LL_device;
	DLPVECTOR  U_device;
	DLPVECTOR  Y_device;
	DLPVECTOR  previousY_device;
	DLPVECTOR  nextY_device;
	DLPVECTOR F_bE_device;
	DLPVECTOR F_bK_device;

	//real K= 1e+6 * nse.lat.physDl*nse.lat.physDl * nse.lat.physDt*nse.lat.physDt;
	//real K= 1e+4 * nse.lat.physDt*nse.lat.physDt;
	real K= 1e+4 * nse.lat.physDt*nse.lat.physDt;
	real local_density = 1e-1 / pow(nse.lat.physDl, 3);
	real fdist = 0;  //TNL::l2Norm(ref.getElement(0)-ref.getElement(1));


	virtual bool outputData(const BLOCK& block, int index, int dof, char *desc, idx x, idx y, idx z, real &value, int &dofs)
	{
		int k=0;
		if (index==k++)
		{
			switch (dof)
			{
				case 0: return vtk_helper("lbm_velocity", block.hmacro(MACRO::e_vx,x,y,z), 3, desc, value, dofs);
				case 1: return vtk_helper("lbm_velocity", block.hmacro(MACRO::e_vy,x,y,z), 3, desc, value, dofs);
				case 2: return vtk_helper("lbm_velocity", block.hmacro(MACRO::e_vz,x,y,z), 3, desc, value, dofs);
			}
		}
		if (index==k++)
		{
			switch (dof)
			{
				case 0: return vtk_helper("lbm_force", block.hmacro(MACRO::e_fx,x,y,z), 3, desc, value, dofs);
				case 1: return vtk_helper("lbm_force", block.hmacro(MACRO::e_fy,x,y,z), 3, desc, value, dofs);
				case 2: return vtk_helper("lbm_force", block.hmacro(MACRO::e_fz,x,y,z), 3, desc, value, dofs);
			}
		}
		if (index==k++) return vtk_helper("lbm_density", block.hmacro(MACRO::e_rho,x,y,z), 1, desc, value, dofs);
		if (index==k++) return vtk_helper("lbm_density_fluctuation", block.hmacro(MACRO::e_rho,x,y,z)-1.0, 1, desc, value, dofs);
		if (index==k++)
		{
			switch (dof)
			{
				case 0: return vtk_helper("velocity", nse.lat.lbm2physVelocity(block.hmacro(MACRO::e_vx,x,y,z)), 3, desc, value, dofs);
				case 1: return vtk_helper("velocity", nse.lat.lbm2physVelocity(block.hmacro(MACRO::e_vy,x,y,z)), 3, desc, value, dofs);
				case 2: return vtk_helper("velocity", nse.lat.lbm2physVelocity(block.hmacro(MACRO::e_vz,x,y,z)), 3, desc, value, dofs);
			}
		}
		if (index==k++)
		{
			switch (dof)
			{
				case 0: return vtk_helper("force", nse.lat.lbm2physForce(block.hmacro(MACRO::e_fx,x,y,z)), 3, desc, value, dofs);
				case 1: return vtk_helper("force", nse.lat.lbm2physForce(block.hmacro(MACRO::e_fy,x,y,z)), 3, desc, value, dofs);
				case 2: return vtk_helper("force", nse.lat.lbm2physForce(block.hmacro(MACRO::e_fz,x,y,z)), 3, desc, value, dofs);
			}
		}
		//if (index==k++) return vtk_helper("density", block.hmacro(MACRO::e_rho,x,y,z)*nse.physFluidDensity, 1, desc, value, dofs);
		return false;
	}

	virtual void probe1()
	{
		static idx cycle = 0;
		const std::string basename = fmt::format("ball_{:04d}", cycle);
		this->writeVTK_Points(basename.c_str(), nse.physTime(), cycle);

		// output the center alone in a vtk file for easier rendering
		const std::string center_basename = fmt::format("ball_center_{:04d}", cycle);
		typename Lagrange3D<NSE>::HLPVECTOR center_vector({nse.lat.phys2lbmPoint(ball_c)});
		this->writeVTK_Points(center_basename.c_str(), nse.physTime(), cycle, center_vector);

		cycle++;
	}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////	

    template<typename LL_array>
	void compute_F_bE(LL_array& F_bE,const LL_array& ref, const LL_array& LL, const  real& s, const real& b)
	{
		// TODO: fdist počítat jenom jednou v první iteraci 
		//,error?
		real eps = 1e-8*fdist;
/*
		if(ibm.computeVariant == IbmCompute::CPU)
		{
		    for(int i =0;i<N;i++)
		    {
			    compute_Stiffness_F_bE(F_bE, ref, LL, s, eps, N, i);
			    compute_Bending_F_bE(F_bE,ref, LL, b,fdist, N, i);
		    }
	    }
		else 
		{
			*/
			auto F_bE_view = F_bE.getView();
			const auto ref_view = ref.getConstView();
			const auto LL_view = LL.getConstView();
			
			int N = this->N;
			real fdist = this->fdist;
			bool free_end = this->free_end;
			//bool free_end = this->free_end;

			auto kernel = [=] CUDA_HOSTDEV (idx i) mutable
			{
				compute_Stiffness_F_bE(F_bE_view, ref_view, LL_view,  s, eps, N, i, free_end);
			    compute_Bending_F_bE(F_bE_view,ref_view, LL_view, b,fdist, N, i, free_end);
			};
			TNL::Algorithms::parallelFor< typename LL_array::DeviceType >((idx) 0, (idx) N, kernel);
		//}	
	}

    template<typename LL_array>
    void compute_F_bK(LL_array & F_bK,const LL_array&Y,LL_array&nextY,const LL_array& previousY, const LL_array& LL, const real& K)
    {
		/*
		if(ibm.computeVariant == IbmCompute::CPU)
		{
		    for(int i = 0;i<N;i++)
		    {
				F_bK[i] = K*(Y[i]-LL[i]);
		    }
		    for(int i = 1;i<N-1;i++)
		    {
				nextY[i] = +2*Y[i] -previousY[i] -(1/local_density)*F_bK[i];
		    }
	    }
		else 
		{*/
			// view je možné zachytit pomocí "=" v lambda funkci
		auto F_bK_view = F_bK.getView();
		const auto Y_view = Y.getConstView();
		auto nextY_view = nextY.getView();
		const auto previousY_view = previousY.getConstView();
		const auto LL_view = LL.getConstView();

		int N = this->N;
		//error with same name
		real Kk = this->K;
		real local_density = this->local_density;
		bool free_end = this->free_end;

		auto kernel = [=] CUDA_HOSTDEV (idx i) mutable
		{
			F_bK_view[i] = Kk*(Y_view[i]-LL_view[i]);
			// TODO: pro volny konec spocitat i N-1
			if(free_end == false)
			{
			if(i!=0 && i!= N-1)
			{
				nextY_view[i] = +2*Y_view[i] -previousY_view[i] -(1/local_density)*F_bK_view[i];
			}
		    }
			else
			{
				if(i!=0)
			{
				nextY_view[i] = +2*Y_view[i] -previousY_view[i] -(1/local_density)*F_bK_view[i];
			}

			}
		};
		TNL::Algorithms::parallelFor< typename LL_array::DeviceType >((idx) 0, (idx) N, kernel);

		//}


    }
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////	
	virtual void computeBeforeLBMKernel()
	{
		std::cout << "compute before kernel"<< std::endl;

		// update ball position
		const dreal velocity_amplitude = 2 * ball_amplitude / ball_period;
		const dreal vz = TNL::sign( cos(2*TNL::pi*nse.iterations/ball_period) ) * velocity_amplitude;
/////////////////////////////////////////////////////////////////////////

		//const auto drho = ibm.dmacroVector(MACRO::e_rho);
        const auto dvx = ibm.dmacroVector(MACRO::e_vx);
        const auto dvy = ibm.dmacroVector(MACRO::e_vy);
        const auto dvz = ibm.dmacroVector(MACRO::e_vz);
        auto dfx = ibm.dmacroVector(MACRO::e_fx);
        auto dfy = ibm.dmacroVector(MACRO::e_fy);
        auto dfz = ibm.dmacroVector(MACRO::e_fz);
        //const auto hrho = ibm.hmacroVector(MACRO::e_rho);
        const auto hvx = ibm.hmacroVector(MACRO::e_vx);
        const auto hvy = ibm.hmacroVector(MACRO::e_vy);
        const auto hvz = ibm.hmacroVector(MACRO::e_vz);
        auto hfx = ibm.hmacroVector(MACRO::e_fx);
        auto hfy = ibm.hmacroVector(MACRO::e_fy);
        auto hfz = ibm.hmacroVector(MACRO::e_fz);

//////////////////////////////////////////////////////////
		if (ibm.computeVariant == IbmCompute::CPU) {
			//ibm.hLL_velocity_lat = point_t{0,0,vz};
			next.setLike(ibm.hLL_lat);
			if(nse.iterations == 0)
			{
				previous = ibm.hLL_lat;
				next = ibm.hLL_lat;
				ref = ibm.hLL_lat;
				LL = ibm.hLL_lat;
				U=ibm.hLL_lat;
				Y=ibm.hLL_lat;
				previousY=ibm.hLL_lat;
				nextY = ibm.hLL_lat;
				fdist = TNL::l2Norm(ref.getElement(0)-ref.getElement(1));	
				F_bE.setSize(N);
				F_bK.setSize(N);
			}		
			else
			{
				std::cout << "else called"<<std::endl;
				//const auto hvx = ibm.hmacroVector(MACRO::e_vx);
				//const auto hvy = ibm.hmacroVector(MACRO::e_vy);
				//const auto hvz = ibm.hmacroVector(MACRO::e_vz);
				ibm.ws_tnl_hM.vectorProduct(hvx, ibm.ws_tnl_hb[0]);
				ibm.ws_tnl_hM.vectorProduct(hvy, ibm.ws_tnl_hb[1]);
				ibm.ws_tnl_hM.vectorProduct(hvz, ibm.ws_tnl_hb[2]);
				//deform(previous, ibm.hLL_lat, next);
				
				std::cout << TNL::max(TNL::abs(ibm.ws_tnl_hb[0])) << std::endl;
				std::cout << TNL::max(TNL::abs(ibm.ws_tnl_hb[1])) << std::endl;
				std::cout << TNL::max(TNL::abs(ibm.ws_tnl_hb[2])) << std::endl;
				
///////////////////////////////////////////////////////////////////////////////////////////////////////////
				real s = 1e+2 * nse.lat.physDt * nse.lat.physDt;
				real b = nse.lat.phys2lbmForce(1e-8);
				//HLPVECTOR F_bE(N);
	            //HLPVECTOR F_bK(N);
				
			for(int i =0;i<N;i++)
			{
			    U[i] =  point_t{
					ibm.ws_tnl_hb[0][i],
					ibm.ws_tnl_hb[1][i],
					ibm.ws_tnl_hb[2][i]
				};
			}
			compute_F_bE(F_bE, ref, LL, s, b);
			compute_F_bK(F_bK, Y, nextY, previousY, LL, K);
							
				// set hx = F_bK + F_bE
				// (nahrada implicitni metody computeForces v Lagrange3D)
				for(int i = 0; i< N;i++)
				{
						point_t F_b{
							F_bK[i][0] + F_bE[i][0],
							F_bK[i][1] + F_bE[i][1],
							F_bK[i][2] + F_bE[i][2]
						};
						ibm.ws_tnl_hx[0][i] = F_b[0];
						ibm.ws_tnl_hx[1][i] = F_b[1];
						ibm.ws_tnl_hx[2][i] = F_b[2];
				
				}
				std::cout << "force" << std::endl;
				std::cout << TNL::max(TNL::abs(ibm.ws_tnl_hx[0])) << std::endl;
				std::cout << TNL::max(TNL::abs(ibm.ws_tnl_hx[1])) << std::endl;
				std::cout << TNL::max(TNL::abs(ibm.ws_tnl_hx[2])) << std::endl;

				// transform the force from Lagrangian to Eulerian coordinates
				auto kernel = [&] (idx i) mutable
				{
					// skipping empty rows explicitly is much faster
					if( ibm.ws_tnl_hMT.getRowCapacity(i) > 0 ) {
						hfx[i] = rowVectorProduct(ibm.ws_tnl_hMT, i, ibm.ws_tnl_hx[0]);
						hfy[i] = rowVectorProduct(ibm.ws_tnl_hMT, i, ibm.ws_tnl_hx[1]);
						hfz[i] = rowVectorProduct(ibm.ws_tnl_hMT, i, ibm.ws_tnl_hx[2]);
					}
				};
				idx n = nse.lat.global.x()*nse.lat.global.y()*nse.lat.global.z();
				TNL::Algorithms::parallelFor< TNL::Devices::Host >((idx) 0, n, kernel);
				// copy forces to the device
				// FIXME: this is copied multiple times when there are multiple Lagrange3D objects
				// (ideally there should be only one Lagrange3D object that comprises all immersed bodies)
				
				dfx = hfx;
				dfy = hfy;
				dfz = hfz;

                previousY = Y;
				Y = nextY;

				next = LL + U;  
				next[0] = LL[0];
				if(free_end==false)
				next[N-1] = LL[N-1];
				else{

				}
				LL=next;

				previous = ibm.hLL_lat;
				ibm.hLL_lat = next;


				
			}
			//deform(ibm.hLL_lat);
			ibm.hLL_velocity_lat = ibm.hLL_lat - previous;
		}
		else 
		{//GPU
			//deform(ibm.dLL_lat);
			//ibm.dLL_velocity_lat = point_t{0,0,vz};
			//ibm.dLL_velocity_lat = point_t{0,vz,0};
			//ibm.dLL_lat += ibm.dLL_velocity_lat;	// Delta t = 1

			next_device.setLike(ibm.dLL_lat);
			if(nse.iterations == 0)
			{
				previous_device = ibm.dLL_lat;
				next_device = ibm.dLL_lat;
				ref_device = ibm.dLL_lat;
				LL_device = ibm.dLL_lat;
				U_device = ibm.dLL_lat;
				Y_device = ibm.dLL_lat;
				previousY_device = ibm.dLL_lat;
				nextY_device = ibm.dLL_lat;
				fdist = TNL::l2Norm(ref_device.getElement(0)-ref_device.getElement(1));	
				F_bE_device.setSize(N);
				F_bK_device.setSize(N);
				//std::cout<<"ref"<<(point_t)ref[0]<<std::endl;

			}
			else
			{
				
					
			    ibm.ws_tnl_dM.vectorProduct(dvx, ibm.ws_tnl_db[0]);
				ibm.ws_tnl_dM.vectorProduct(dvy, ibm.ws_tnl_db[1]);
				ibm.ws_tnl_dM.vectorProduct(dvz, ibm.ws_tnl_db[2]);
					//deform(previous, ibm.dLL_lat, next);
					
				std::cout << TNL::max(TNL::abs(ibm.ws_tnl_db[0])) << std::endl;
				std::cout << TNL::max(TNL::abs(ibm.ws_tnl_db[1])) << std::endl;
				std::cout << TNL::max(TNL::abs(ibm.ws_tnl_db[2])) << std::endl;
					
				//real s = 1e+2 * nse.lat.physDt * nse.lat.physDt;
				real s = 1e+2 * nse.lat.physDt * nse.lat.physDt;//2//1
				//real b = nse.lat.phys2lbmForce(1e-8);
				real b = nse.lat.phys2lbmForce(1e-8);//7//10 
				//DLPVECTOR F_bE(N);
				//DLPVECTOR F_bK(N);
					
				const auto Ux = ibm.ws_tnl_db[0].getConstView();
				const auto Uy = ibm.ws_tnl_db[1].getConstView();	
				const auto Uz = ibm.ws_tnl_db[2].getConstView();		
				auto Uview = U_device.getView();
				auto kernelU = [=] CUDA_HOSTDEV (idx i) mutable
				{
					Uview[i] =  point_t{
						Ux[i],
						Uy[i],
						Uz[i]
					};
				};
				TNL::Algorithms::parallelFor< TNL::Devices::Cuda >((idx) 0, (idx) N, kernelU);
			    
			    compute_F_bE(F_bE_device, ref_device, LL_device, s, b);
			    compute_F_bK(F_bK_device, Y_device, nextY_device, previousY_device, LL_device, K);
				// TODO: ref na GPU
				//compute_F_bE(F_bE,ref, s, b);
				// TODO: LL na GPU
				//compute_F_bK(F_bK,Y,nextY,previousY,LL,K);
								
					// set hx = F_bK + F_bE
					// (nahrada implicitni metody computeForces v Lagrange3D)
				auto dxx = ibm.ws_tnl_dx[0].getView();
				auto dxy = ibm.ws_tnl_dx[1].getView();
				auto dxz = ibm.ws_tnl_dx[2].getView();

				auto F_bK_view = F_bK_device.getView();
				auto F_bE_view = F_bE_device.getView();

				auto kernelFb = [=] CUDA_HOSTDEV (idx i) mutable
				{
					point_t F_b ={
						F_bK_view[i][0] + F_bE_view[i][0],
						F_bK_view[i][1] + F_bE_view[i][1],
						F_bK_view[i][2] + F_bE_view[i][2]
					};
					dxx[i] = F_b[0];
					dxy[i] = F_b[1];
					dxz[i] = F_b[2];
				};
				TNL::Algorithms::parallelFor< TNL::Devices::Cuda >((idx) 0, (idx) N, kernelFb);
				
				std::cout << "force" << std::endl;
				std::cout << TNL::max(TNL::abs(ibm.ws_tnl_dx[0])) << std::endl;
				std::cout << TNL::max(TNL::abs(ibm.ws_tnl_dx[1])) << std::endl;
				std::cout << TNL::max(TNL::abs(ibm.ws_tnl_dx[2])) << std::endl;
	
					// transform the force from Lagrangian to Eulerian coordinates
					/////////////////
				const auto x1 = ibm.ws_tnl_dx[0].getConstView();
				const auto x2 = ibm.ws_tnl_dx[1].getConstView();
				const auto x3 = ibm.ws_tnl_dx[2].getConstView();
					//vice moznosti pres view nebo neco jineho atd
				using dEllpack = typename Lagrange3D<NSE>::dEllpack;
				TNL::Pointers::DevicePointer<dEllpack> MT_dptr(ibm.ws_tnl_dMT);
				const dEllpack* MT = &MT_dptr.template getData<TNL::Devices::Cuda>();
				auto kernel = [=] CUDA_HOSTDEV (idx i) mutable
				{
						// skipping empty rows explicitly is much faster
					if( MT->getRowCapacity(i) > 0 ) 
					{
						dfx[i] = rowVectorProduct(*MT, i, x1);
						dfy[i] = rowVectorProduct(*MT, i, x2);
						dfz[i] = rowVectorProduct(*MT, i, x3);
					}
				};
				idx n = nse.lat.global.x()*nse.lat.global.y()*nse.lat.global.z();
				TNL::Algorithms::parallelFor< TNL::Devices::Cuda >((idx) 0, n, kernel);
	
				previousY_device = Y_device;
				Y_device = nextY_device;
	
				next_device = LL_device + U_device;  
				next_device.setElement(0, LL_device.getElement(0));
				// TODO: parametr pro nastaveni okrajove podminky: pevny/volny konec
				if(free_end==false)
				{
				next_device.setElement(N-1, LL_device.getElement(N-1));
				}
				else
				{

				}
				LL_device=next_device;

				previous_device = ibm.dLL_lat;
				ibm.dLL_lat = next_device;
	
	
					
			}
				//deform(ibm.hLL_lat);
			ibm.dLL_velocity_lat = ibm.dLL_lat - previous_device;
			
		}

		ibm.constructed = false;
		ibm.use_LL_velocity_in_solution = true;

////////////////////////////////////////////////////////////////////////////////		
		// update the ball center for drawing
		//ball_c += point_t{0,0,vz*nse.lat.physDl};
		//ball_c += point_t{0,vz*nse.lat.physDl,0};

		//rotation
		//The residue is NaN.
		// const point_t rotation_axis = point_t{0,0,1};
		// const point_t rotation_radius = point_t{1,1,2};
		// const point_t radius_axis_cross_product = point_t{
		// 	rotation_radius[1]*rotation_axis[2] - rotation_radius[2]*rotation_axis[1],
		// 	-(rotation_radius[0]*rotation_axis[2] - rotation_radius[2]*rotation_axis[0]),
		// 	rotation_radius[0]*rotation_axis[1] - rotation_radius[1]*rotation_axis[0]
		// };
		// const dreal radius_axis_cross_product_length = sqrt(pow(radius_axis_cross_product[0],2)+pow(radius_axis_cross_product[1],2) + pow(radius_axis_cross_product[2],2));
		// //T=1
		// const dreal T = nse.iterations;
		// const dreal angular_speed = (2*TNL::pi)/T;
		// const dreal length_of_radius = sqrt(pow(rotation_radius[0],2)+pow(rotation_radius[1],2)+pow(rotation_radius[2],2));
		// const dreal magnitude_orbital_velocity = angular_speed*length_of_radius;
		// const dreal scalar = magnitude_orbital_velocity/radius_axis_cross_product_length;
		// //           velocity = scalar*radius_axis_cross_product;

		// if (ibm.computeVariant == IbmCompute::CPU) {
		// 	//ibm.hLL_velocity_lat = point_t{0,0,vz};
		// 	ibm.hLL_velocity_lat = point_t{scalar*radius_axis_cross_product[0],scalar*radius_axis_cross_product[1],scalar*radius_axis_cross_product[2]};
		// 	ibm.hLL_lat += ibm.hLL_velocity_lat;	// Delta t = 1
		// }
		// else {
		// 	//ibm.dLL_velocity_lat = point_t{0,0,vz};
		// 	ibm.dLL_velocity_lat = point_t{scalar*radius_axis_cross_product[0],scalar*radius_axis_cross_product[1],scalar*radius_axis_cross_product[2]};
		// 	ibm.dLL_lat += ibm.dLL_velocity_lat;	// Delta t = 1
		// }

		// ibm.constructed = false;
		// ibm.use_LL_velocity_in_solution = true;


	}

	virtual void updateKernelVelocities()
	{
		for (auto& block : nse.blocks)
		{
			block.data.inflow_vx = lbm_inflow_vx;
			block.data.inflow_vy = 0;
			block.data.inflow_vz = 0;
		}
	}
	point_t fiber_start = {0,0,0};
real cube_radius =0;
	virtual void setupBoundaries()
	{
		nse.setBoundaryX(0, BC::GEO_INFLOW_LEFT); // left
		nse.setBoundaryX(nse.lat.global.x()-1, BC::GEO_OUTFLOW_RIGHT);// right
		nse.setBoundaryY(0, BC::GEO_INFLOW); // back
		nse.setBoundaryY(nse.lat.global.y()-1, BC::GEO_INFLOW);// front
		nse.setBoundaryZ(0, BC::GEO_INFLOW);// top
		nse.setBoundaryZ(nse.lat.global.z()-1, BC::GEO_INFLOW);// bottom
													
		// draw a sphere
		if(add_ball==true)
		{
		//coordinates of the first point of the fiber
		fiber_start = nse.lat.phys2lbmPoint(this->fiber_start);

		cube_radius = (this->cube_radius)/nse.lat.physDl;
		//for size

		//for non X coordinates i want the center to be fiber start
		point_t center = point_t{fiber_start.x() -cube_radius,///2
		fiber_start.y(), fiber_start.z()};

		for (int px=center.x() - cube_radius;px<=center.x()+cube_radius;px++)
		for (int pz=center.z() - cube_radius;pz<=center.z()+cube_radius;pz++)
		for (int py=center.y() - cube_radius;py<=center.y()+cube_radius;py++)
		{
			//cube
			/*
			if (px-center.x() < cube_radius &&
			py-center.y() < cube_radius &&
			pz-center.z() < cube_radius )
				nse.setMap(px,py,pz,BC::GEO_WALL);
				*/
			//ball 
			point_t p = {px,py,pz};
			real dist = TNL::l2Norm(p - center);
			if(dist < cube_radius)
			{
				nse.setMap(px,py,pz, BC::GEO_WALL);
			}
			
		}
		//*/
	//odkud vzit lbm?lbm=nse
		/*
		lbmDrawCube(nse, BC::GEO_WALL,
		center ,
		cube_radius
		);
		*/
		
		}
	
	}

	StateLocal(const std::string& id, const TNL::MPI::Comm& communicator, lat_t lat)
		: State<NSE>(id, communicator, lat)
	{}
};

template < typename NSE >
int sim(int RES=2, double i_Re=1000, double nasobek=2.0, int dirac_delta=2, int method=0, int compute=5)
{
	using idx = typename NSE::TRAITS::idx;
	using real = typename NSE::TRAITS::real;
	using point_t = typename NSE::TRAITS::point_t;
	using lat_t = Lattice<3, real, idx>;

	int block_size=32;
	real BALL_DIAMETER = 0.01;
	real real_domain_height= BALL_DIAMETER*11;// [m]
	//real real_domain_length= BALL_DIAMETER*11;// [m] // extra 1cm on both sides
	idx LBM_Y = RES*block_size; // for 4 cm
	idx LBM_Z = LBM_Y;
	real PHYS_DL = real_domain_height/((real)LBM_Y);
	idx LBM_X = 2*LBM_Y;//(int)(real_domain_length/PHYS_DL)+2;//block_size;//16*RESOLUTION;
	point_t PHYS_ORIGIN = {0., 0., 0.};

	// zvolit Re + LBM VELOCITY + PHYS_VISCOSITY
//	real i_Re = ;
	real i_LBM_VELOCITY = 0.07; // Geier
	real i_PHYS_VISCOSITY = 0.00001; // proc ne?
	// mam:
	real i_LBM_VISCOSITY = i_LBM_VELOCITY * BALL_DIAMETER / PHYS_DL / i_Re;
	real i_PHYS_VELOCITY = i_PHYS_VISCOSITY * i_Re / BALL_DIAMETER; //0
	fmt::print("input phys velocity {:f}\ninput lbm velocity {:f}\nRe {:f}\nlbm viscosity{:f}\nphys viscosity {:f}\n", i_PHYS_VELOCITY, i_LBM_VELOCITY, i_Re, i_LBM_VISCOSITY, i_PHYS_VISCOSITY);

	real LBM_VISCOSITY = i_LBM_VISCOSITY;// 0.0001*RES;//*SIT;//1.0/6.0; /// GIVEN: optimal is 1/6
	real PHYS_VISCOSITY = i_PHYS_VISCOSITY;//0.00001;// [m^2/s] fluid viscosity of water
	real Re=i_Re;//200;

//	real INIT_TIME = 1.0; // [s]
	real PHYS_DT = LBM_VISCOSITY / PHYS_VISCOSITY*PHYS_DL*PHYS_DL;

	// initialize the lattice
	lat_t lat;
	lat.global = typename lat_t::CoordinatesType( LBM_X, LBM_Y, LBM_Z );
	lat.physOrigin = PHYS_ORIGIN;
	lat.physDl = PHYS_DL;
	lat.physDt = PHYS_DT;
	lat.physViscosity = PHYS_VISCOSITY;

	const std::string state_id = fmt::format("sim_IBM4_{}_{}_dirac_{}_res_{}_Re_{}_nas_{:05.4f}_compute_{}", NSE::COLL::id, (method>0)?"original":"modified", dirac_delta, RES, Re, nasobek, compute);
	StateLocal<NSE> state(state_id, MPI_COMM_WORLD, lat);

	if (state.isMark())
		return 0;

	state.lbm_inflow_vx = i_LBM_VELOCITY;
	state.nse.physCharLength = BALL_DIAMETER; // [m]
	state.ball_diameter = BALL_DIAMETER; // [m]
	state.ball_amplitude = lat.global.z() / 5.;	// [lbm units]
	state.ball_period = 2.0 / PHYS_DT;	// [lbm units]

	state.cnt[PRINT].period = 0.1;

	//state.nse.physFinalTime = 10*PHYS_DT; //1.0; //30.0;
	//state.cnt[VTK3D].period = PHYS_DT; //1.0;
	//state.cnt[VTK2D].period = PHYS_DT; //0.01;
	//state.cnt[PROBE1].period = PHYS_DT; //0.01;	// Lagrangian points VTK output

	state.nse.physFinalTime = 1.0; //30.0;
	state.cnt[VTK3D].period = 0.01;
	state.cnt[VTK2D].period = 0.01;
	state.cnt[PROBE1].period = 0.01;	// Lagrangian points VTK output

	// select compute method
	IbmCompute computeVariant;
	switch (compute)
	{
		case 0: computeVariant = IbmCompute::GPU; break;
		case 1: computeVariant = IbmCompute::CPU; break;
		case 2: computeVariant = IbmCompute::Hybrid; break;
		case 3: computeVariant = IbmCompute::Hybrid_zerocopy; break;
		default:
			spdlog::warn("Unknown parameter compute={}, selecting GPU as the default.", compute);
			computeVariant = IbmCompute::GPU;
			break;
	}

	// add cuts
	state.add2Dcut_X(LBM_X/2,"cut_X");
//	state.add2Dcut_X(2*BALL_DIAMETER/PHYS_DL,"cut_Xball");
	state.add2Dcut_Y(LBM_Y/2,"cut_Y");
	state.add2Dcut_Z(LBM_Z/2,"cut_Z");

	// create immersed objects
	state.ball_c[0] = 5.5*state.ball_diameter;
	state.ball_c[1] = 5.5*state.ball_diameter;
	state.ball_c[2] = 5.5*state.ball_diameter;
	real sigma = nasobek * PHYS_DL;
	//std::pair<int,int> N1N2 = ibmSetupRectangle(state.ibm, state.ball_c, state.ball_diameter/2.0, state.ball_diameter/2.0, sigma);
	//state.N1 = N1N2.first;
	//state.N2 = N1N2.second;

	int N = ibmSetupFilament(state.ibm, state.ball_c,sigma, 2*state.ball_diameter);
	state.N=N;
	state.add_ball = true;
	state.free_end=true;
	state.fiber_start = state.ibm.LL[0];
	state.cube_radius = 1.5*state.ball_diameter;
	//state.fiber_start = lat.phys2lbmPoint(state.fiber_start);
	// configure IBM
	state.ibm.computeVariant = computeVariant;
	state.ibm.diracDeltaTypeEL = dirac_delta;
	if (method == 0)
		state.ibm.methodVariant = IbmMethod::modified;
	else
		state.ibm.methodVariant = IbmMethod::original;

	execute(state);

	return 0;
}

template < typename TRAITS=TraitsSP >
void run(int res, double Re, double h, int dirac, int method, int compute)
{
	using COLL = D3Q27_CUM<TRAITS>;
	using NSE_CONFIG = LBM_CONFIG<
				TRAITS,
				D3Q27_KernelStruct,
				NSE_Data_ConstInflow< TRAITS >,
				COLL,
				typename COLL::EQ,
				D3Q27_STREAMING< TRAITS >,
				D3Q27_BC_All,
				MacroLocal< TRAITS >,
				D3Q27_MACRO_Void< TRAITS >
			>;

	sim<NSE_CONFIG>(res, Re, h, dirac, method, compute);
}

int main(int argc, char **argv)
{
	TNLMPI_INIT mpi(argc, argv);

	// here "h" is the Lagrangian/Eulerian spacing ratio
//	const double hvals[] = { 2.0, 1.5, 1.0, 0.75, 0.50, 0.25 };
//	const double hvals[] = { 2.0 };
//	const double hvals[] = { 0.125 };
	const double hvals[] = { 0.25, 0.5, 0.75, 1.0, 1.5, 2.0 };
//	const double hvals[] = { 2.0, 1.8, 1.6, 1.4, 1.2, 1.0, 0.9, 0.8, 0.75, 0.50 };
//	const double hvals[] = { 2.0, 1.8, 1.6, 1.4, 1.2, 1.0, 0.75, 0.50, 0.25, 0.125 };
//	const double hvals[] = { 3.0, 2.5, 2.0, 1.8, 1.6, 1.4, 1.2, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1 }; // ostrava
	int hmax = sizeof(hvals)/sizeof(double);
	double h=1.0;

	bool use_command_line=true;

	if (!use_command_line)
	{
		int dirac=1;
		int res=2; // 3, 6, 12
		int hi=3;
		int method=0; //0 = modified
		int Re=100;
		int compute=5;
//		for (int Re=100;Re<=200; Re+=100)
//		for (hi=0;hi<hmax;hi++)
//		for (method=0;method<=1;method++)
//		for (res=3;res<=5;res++)
//		for (dirac=1;dirac<=4;dirac++)
//		for (compute=1;compute<=6;compute++)
		{
			if (hi<hmax) h=hvals[hi];
			run(res, (double)Re, h, dirac, method, compute);
		}
	} else
	{
		const int pars=6;
		if (argc <= pars)
		{
			fprintf(stderr, "error: %d parameters required:\n %s method{0,1} dirac{1,2,3,4} Re{100,200} hi[0,%d] res[1,22] compute[0,3]\n", pars, argv[0],hmax-1);
			return 1;
		}
		int method = atoi(argv[1]);	// 0=modified 1=original
		int dirac = atoi(argv[2]);
		int Re = atoi(argv[3]);		// type=0,1,2 (geometry selection)
		int hi = atoi(argv[4]);		// index in the hvals
		int res = atoi(argv[5]);	// res=1,2,3
		int compute = atoi(argv[6]); // compute=0,1,2,3

		if (method > 1 || method < 0) { fprintf(stderr, "error: method=%d out of bounds [0, 1]\n",method); return 1; }
		if (dirac < 1 || dirac > 4) { fprintf(stderr, "error: dirac=%d out of bounds [1,4]\n",dirac); return 1; }
		if (hi >= hmax || hi < 0) { fprintf(stderr, "error: hi=%d out of bounds [0, %d]\n",hi,hmax-1); return 1; }
		if (res < 1) { fprintf(stderr, "error: res=%d out of bounds [1, ...]\n",res); return 1; }
		if (compute < 0 || compute > 3) { fprintf(stderr, "error: compute=%d out of bounds [0,3]\n",compute); return 1; }
		if (hi<hmax) h=hvals[hi];
		run(res, (double)Re, h, dirac, method, compute);
	}
	return 0;
}
