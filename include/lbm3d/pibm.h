#pragma once

#include "defs.h"

//function for stifness
template<typename LL_array, typename ConstLL_array, typename real>
CUDA_HOSTDEV
void compute_Stiffness_F_bE(LL_array& F_bE, const ConstLL_array& ref, const ConstLL_array& LL, const real& s, const real& eps, int N, int i, bool free_end)
{
	if(i==0)
	{
		real norm1 = l2Norm(LL[1]-LL[0]);
		real fdist1 = l2Norm(ref[1]-ref[0]);
		F_bE[i][0]= s/fdist1/fdist1*(1-fdist1/norm1)*(LL[1][0] -LL[0][0] );
		F_bE[i][1]= s/fdist1/fdist1*(1-fdist1/norm1)*(LL[1][1] -LL[0][1] );
		F_bE[i][2]= s/fdist1/fdist1*(1-fdist1/norm1)*(LL[1][2] -LL[0][2] );
	}
	else if(i==N-1)
	{
		
		// TODO: tohle je pro pevny konec, co se ma delat pro volny konec?
		
		if(free_end == false)
		{
		real norm2 = l2Norm(LL[N-1]-LL[N-2]);
		real fdist2 = l2Norm(ref[N-1]-ref[N-2]);
		
		F_bE[i][0]= s/fdist2/fdist2*(1-fdist2/norm2)*(LL[N-2][0] -LL[N-1][0] );
		F_bE[i][1]= s/fdist2/fdist2*(1-fdist2/norm2)*(LL[N-2][1] -LL[N-1][1] );
		F_bE[i][2]= s/fdist2/fdist2*(1-fdist2/norm2)*(LL[N-2][2] -LL[N-1][2] );
		}
		else
		{
			

		}
	}
	else
	{
		real norm1 = l2Norm(LL[i+1]-LL[i]);
		real norm2 = l2Norm(LL[i]-LL[i-1]);
		real fdist1 = l2Norm(ref[i+1]-ref[i]);
		real fdist2 = l2Norm(ref[i]-ref[i-1]);
		if(norm1<eps) norm1=eps;
		if(norm2<eps)norm2=eps;
		F_bE[i][0]= s/fdist1/fdist1*(1-fdist1/norm1)*(LL[i+1][0] -LL[i][0] )+
		s/fdist2/fdist2*(1-fdist2/norm2)*(LL[i+1][0] -LL[i][0] );
		F_bE[i][1]= s/fdist1/fdist1*(1-fdist1/norm1)*(LL[i+1][1] -LL[i][1] )+
		s/fdist2/fdist2*(1-fdist2/norm2)*(LL[i+1][1] -LL[i][1] );
		F_bE[i][2]= s/fdist1/fdist1*(1-fdist1/norm1)*(LL[i+1][2] -LL[i][2] )+
		s/fdist2/fdist2*(1-fdist2/norm2)*(LL[i+1][2] -LL[i][2] );	
	}

}
//function for bending
template<typename LL_array, typename ConstLL_array, typename real>
CUDA_HOSTDEV
void compute_Bending_F_bE(LL_array& F_bE, const ConstLL_array& ref, const ConstLL_array& LL, const real& b, const real& fdist, int N, int i, bool free_end)
{
	// bending
	if(i==0)
	{
		//continue;
	}
	else if(i==N-1)
	{
		//continue;
		// TODO: tohle je pro pevny konec, co se ma delat pro volny konec?
		if(free_end == false)
		{

		}
		else
		{


		}
	}
	else if(i==1)
	{
		F_bE[i][0]+=b/fdist/fdist/fdist/fdist*(2*(LL[0][0]-ref[0][0])-5*(LL[1][0]-ref[1][0])
		+4*(LL[2][0]-ref[2][0])-(LL[3][0]-ref[3][0]));

		F_bE[i][1]+=b/fdist/fdist/fdist/fdist*(2*(LL[0][1]-ref[0][1])-5*(LL[1][1]-ref[1][1])
		+4*(LL[2][1]-ref[2][1])-(LL[3][1]-ref[3][1]));

		F_bE[i][2]+=b/fdist/fdist/fdist/fdist*(2*(LL[0][2]-ref[0][2])-5*(LL[1][2]-ref[1][2])
		+4*(LL[2][2]-ref[2][2])-(LL[3][2]-ref[3][2]));
	}
	else if(i==N-2)
	{
		// TODO: tohle je pro pevny konec, co se ma delat pro volny konec?
		if(free_end == false)
		{
		F_bE[i][0]+=b/fdist/fdist/fdist/fdist*(2*(LL[N-1][0]-ref[N-1][0])-5*(LL[N-2][0]-ref[N-2][0])
		+4*(LL[N-3][0]-ref[N-3][0])-(LL[N-4][0]-ref[N-4][0]));

		F_bE[i][1]+=b/fdist/fdist/fdist/fdist*(2*(LL[N-1][1]-ref[N-1][1])-5*(LL[N-2][1]-ref[N-2][1])
		+4*(LL[N-3][1]-ref[N-3][1])-(LL[N-4][1]-ref[N-4][1]));
		
		F_bE[i][2]+=b/fdist/fdist/fdist/fdist*(2*(LL[N-1][2]-ref[N-1][2])-5*(LL[N-2][2]-ref[N-2][2])
		+4*(LL[N-3][2]-ref[N-3][2])-(LL[N-4][2]-ref[N-4][2]));
		}
		else
		{

		}
	}
	else
	{
		F_bE[i][0]+=b/fdist/fdist/fdist/fdist*((LL[i-2][0]-ref[i-2][0])-
		4*(LL[i-1][0]-ref[i-1][0])+6*(LL[i][0]-ref[i][0])-4*(LL[i+1][0]-ref[i+1][0])+(LL[i+2][0]-ref[i+2][0])
		);
		F_bE[i][1]+=b/fdist/fdist/fdist/fdist*((LL[i-2][1]-ref[i-2][1])-
		4*(LL[i-1][1]-ref[i-1][1])+6*(LL[i][1]-ref[i][1])-4*(LL[i+1][1]-ref[i+1][1])+(LL[i+2][1]-ref[i+2][1])
		);
		F_bE[i][2]+=b/fdist/fdist/fdist/fdist*((LL[i-2][2]-ref[i-2][2])-
		4*(LL[i-1][2]-ref[i-1][2])+6*(LL[i][2]-ref[i][2])-4*(LL[i+1][2]-ref[i+1][2])+(LL[i+2][2]-ref[i+2][2])
		);
	}

}


//flag
//////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//Delta t =1
//h = 1
//number of j and i  coordinates
//int N1=0;
//int N2=0;

//differences
template<typename LL_array, typename point_t>
point_t second_central_difference(int i, int j,bool by_s1, LL_array& LL, int N2)
{
	point_t X = LL[i*N2+j];
	point_t Xnext = by_s1? LL[(i+1)*N2+j] : LL[(i)*N2+j +1];
	point_t Xprev = by_s1 ? LL[(i-1)*N2+j] : LL[(i)*N2+j -1];
	return Xnext -2*X + Xprev;
}
template<typename LL_array, typename point_t>
point_t second_central_difference(int i, int j, LL_array& LL, int N2)
{
	return (LL[(i+1)*N2 +j + 1] - LL[(i+1)*N2 +j - 1] - LL[(i-1)*N2 +j + 1] + LL[(i-1)*N2 +j - 1])/4;
}
template<typename LL_array, typename point_t>
point_t fourth_central_difference(int i, int j, bool by_s1, LL_array& LL, int N1, int N2)
{
	point_t X = LL[i*N2+j];
	// if (by_s1) {

	// }
	// else {

	// }
	point_t Xnext = by_s1? LL[(i+1)*N2+j] : LL[(i)*N2+j +1];
	point_t Xnext2 = by_s1? LL[(i+1)*N2+j] : LL[(i)*N2+j +1];	// zkontrolovat jestli tu nema byt +-1 nebo doplnit dalsi podminky?
	point_t Xprev = by_s1 ? LL[(i-1)*N2+j] : LL[(i)*N2+j -1];
	point_t Xprev2 = by_s1 ? LL[(i-1)*N2+j] : LL[(i)*N2+j -1];  // zkontrolovat jestli tu nema byt +-2
	if(i<2 && j>=2)
	{
		if(j<N2-2) 
		{
			Xnext2 = by_s1? LL[(i+1)*N2+j] : LL[(i)*N2+j +2];
		Xprev2 = by_s1 ? LL[(i-1)*N2+j] : LL[(i)*N2+j -2];
		}
		else
		{
			Xnext2 = by_s1? LL[(i+1)*N2+j] : LL[(i)*N2+j +1];
		Xprev2 = by_s1 ? LL[(i-1)*N2+j] : LL[(i)*N2+j -1];

		}
	}
	if(i>=2 && j<2)
	{  
		if(i<N1-2)
		{
		Xnext2 = by_s1? LL[(i+2)*N2+j] : LL[(i)*N2+j +1];
		Xprev2 = by_s1 ? LL[(i-2)*N2+j] : LL[(i)*N2+j -1];
		}
		else
		{
			Xnext2 = by_s1? LL[(i+1)*N2+j] : LL[(i)*N2+j +1];
		Xprev2 = by_s1 ? LL[(i-1)*N2+j] : LL[(i)*N2+j -1];
		}
	}
	if(i<2 && j<2)
	{
		Xnext2 = by_s1? LL[(i+1)*N2+j] : LL[(i)*N2+j +1];
		Xprev2 = by_s1 ? LL[(i-1)*N2+j] : LL[(i)*N2+j -1];
	}
	if(i>=2 && j>=2)
	{
		if(i<N1-2 && j<N2-2)
		{
		Xprev2 = by_s1 ? LL[(i-2)*N2+j] : LL[(i)*N2+j -2];
		Xnext2 = by_s1? LL[(i+2)*N2+j] : LL[(i)*N2+j +2];
		}
		else
		{
			Xprev2 = by_s1 ? LL[(i-1)*N2+j] : LL[(i)*N2+j -1];
		Xnext2 = by_s1? LL[(i+1)*N2+j] : LL[(i)*N2+j +1];
		}
	}
	return Xnext2 - 4*Xnext + 6*X - 4*Xprev +Xprev2;
}
template<typename LL_array, typename point_t>
point_t fourth_central_difference(int i, int j, LL_array& LL, int N1, int N2)
{
	if(i<2 && j>=2)
	{
		if(j<N2-2)
		{
		return (
		LL[(i+1)*N2+j+2] +LL[(i+1)*N2+j-2] + LL[(i-1)*N2+j+2] + LL[(i-1)*N2+j-2]
		- 2*LL[(i+1)*N2+j]- 2*LL[(i)*N2+j+2]- 2*LL[(i)*N2+j-2]- 2*LL[(i-1)*N2+j]
		+ 4*LL[(i)*N2+j]
		)/16;
		}
		else
		{
		return (
		LL[(i+1)*N2+j+1] +LL[(i+1)*N2+j-1] + LL[(i-1)*N2+j+1] + LL[(i-1)*N2+j-1]
		- 2*LL[(i+1)*N2+j]- 2*LL[(i)*N2+j+1]- 2*LL[(i)*N2+j-1]- 2*LL[(i-1)*N2+j]
		+ 4*LL[(i)*N2+j]
		)/16;
		}
	}
	else if(i>=2 && j<2)
	{
		if(i<N1-2)
		{
		return (
		LL[(i+2)*N2+j+1] +LL[(i+2)*N2+j-1] + LL[(i-2)*N2+j+1] + LL[(i-2)*N2+j-1]
		- 2*LL[(i+2)*N2+j]- 2*LL[(i)*N2+j+1]- 2*LL[(i)*N2+j-1]- 2*LL[(i-2)*N2+j]
		+ 4*LL[(i)*N2+j]
		)/16;
		}
		else
		{
		return (
		LL[(i+1)*N2+j+1] +LL[(i+1)*N2+j-1] + LL[(i-1)*N2+j+1] + LL[(i-1)*N2+j-1]
		- 2*LL[(i+1)*N2+j]- 2*LL[(i)*N2+j+1]- 2*LL[(i)*N2+j-1]- 2*LL[(i-1)*N2+j]
		+ 4*LL[(i)*N2+j]
		)/16;
		}
	}
	else if(i<2 && j<2)
	{
		return (
		LL[(i+1)*N2+j+1] +LL[(i+1)*N2+j-1] + LL[(i-1)*N2+j+1] + LL[(i-1)*N2+j-1]
		- 2*LL[(i+1)*N2+j]- 2*LL[(i)*N2+j+1]- 2*LL[(i)*N2+j-1]- 2*LL[(i-1)*N2+j]
		+ 4*LL[(i)*N2+j]
		)/16;
	}
	else
	{
		if(i<N1-2 && j < N2-2)  // rozdelit podminku:  i < N1 - 2, j < N2 - 2
		{
		return (
		LL[(i+2)*N2+j+2] +LL[(i+2)*N2+j-2] + LL[(i-2)*N2+j+2] + LL[(i-2)*N2+j-2]
		- 2*LL[(i+2)*N2+j]- 2*LL[(i)*N2+j+2]- 2*LL[(i)*N2+j-2]- 2*LL[(i-2)*N2+j]
		+ 4*LL[(i)*N2+j]
		)/16;
		}
		else
		{
		return (
		LL[(i+1)*N2+j+1] +LL[(i+1)*N2+j-1] + LL[(i-1)*N2+j+1] + LL[(i-1)*N2+j-1]
		- 2*LL[(i+1)*N2+j]- 2*LL[(i)*N2+j+1]- 2*LL[(i)*N2+j-1]- 2*LL[(i-1)*N2+j]
		+ 4*LL[(i)*N2+j]
		)/16;
		}
	}

}
//////////////////////////////////////////////////////////////////////////////////////
template<typename LL_array, typename point_t>
point_t second_forward_diff_RHS(int i, int j,bool by_s1, LL_array& LL, int N2)
{
	//point_t X = LL[i*N2+j];
	point_t Xnext = by_s1? LL[(i+1)*N2+j] : LL[(i)*N2+j +1];
	point_t Xnext2 = by_s1 ? LL[(i+2)*N2+j] : LL[(i)*N2+j +2];
	return -(Xnext2 -2*Xnext);
}
template<typename LL_array, typename point_t>
point_t second_backward_diff_RHS(int i, int j,bool by_s1, LL_array& LL, int N2)
{
	//point_t X = LL[i*N2+j];
	point_t Xprev2 = by_s1? LL[(i-2)*N2+j] : LL[(i)*N2+j -2];
	point_t Xprev = by_s1 ? LL[(i-1)*N2+j] : LL[(i)*N2+j -1];
	return -(-2*Xprev + Xprev2);
}
template<typename LL_array, typename point_t>
point_t third_forward_diff_RHS(int i, int j,bool by_s1, LL_array& LL, int N2)
{
	//point_t X = LL[i*N2+j];
	point_t Xnext = by_s1? LL[(i+1)*N2+j] : LL[(i)*N2+j +1];
	point_t Xnext2 = by_s1 ? LL[(i+2)*N2+j] : LL[(i)*N2+j +2];
	point_t Xnext3 = by_s1 ? LL[(i+3)*N2+j] : LL[(i)*N2+j +3];
	return +3*Xnext -3*Xnext2 + Xnext3;//-X
}
template<typename LL_array, typename point_t>
point_t third_backward_diff_RHS(int i, int j,bool by_s1, LL_array& LL, int N2)
{
	//point_t X = LL[i*N2+j];
	point_t Xprev = by_s1? LL[(i-1)*N2+j] : LL[(i)*N2+j -1];
	point_t Xprev2 = by_s1 ? LL[(i-2)*N2+j] : LL[(i)*N2+j -2];
	point_t Xprev3 = by_s1 ? LL[(i-3)*N2+j] : LL[(i)*N2+j -3];
	return -(-3*Xprev +3*Xprev2 - Xprev3);
}

template<typename LL_array, typename point_t, typename dreal>
point_t elastic_force_sum(int i,int j,LL_array& LL, bool by_s1, bool by_s2, dreal sigma, dreal gama)
{
std::cout<< "elastic force sum is called"<<" i = "<<i<<" j = "<<j <<std::endl;
	//sum from i,j = 1 to i,j =2
	//i=j=1 s1,s1
	std::cout<<"second"<<std::endl;
	point_t sigma_s1s1 = second_central_difference(i,j,by_s1,LL);
	point_t sigma_s1s2 = second_central_difference(i,j,LL);
	point_t sigma_s2s2 = second_central_difference(i,j,by_s2,LL);
	point_t sigmaSum = sigma*(sigma_s1s1 + 2*sigma_s1s2 + sigma_s2s2);
std::cout<<"fourtth"<<std::endl;
	point_t gama_s1s1 = fourth_central_difference(i,j,by_s1,LL);
	std::cout<<"fourth difference by both i j"<<std::endl;
	point_t gama_s1s2 = fourth_central_difference(i,j,LL);
	point_t gama_s2s2 = fourth_central_difference(i,j,by_s2,LL);
	point_t gamaSum = gama*(gama_s1s1 + 2*gama_s1s2 + gama_s2s2);
	std::cout<<"gama sum = "<< gamaSum<<std::endl;
	std::cout<<"sigma sum = "<< sigmaSum<<std::endl;
	   return sigmaSum - gamaSum;
}
template<typename LL_array, typename point_t, typename dreal>
point_t lagrangian_force(LL_array& previous, LL_array& LL, int i, int j, int N2, dreal kappa, point_t U_ib)
{
	/*
	point_t U_ib;
	//if (...)   rozhodnout jestli hb nebo db
	U_ib = point_t{
		ibm.ws_tnl_hb[0][i*N2+j],
		ibm.ws_tnl_hb[1][i*N2+j],
		ibm.ws_tnl_hb[2][i*N2+j]
	};
	*/
	point_t Xn = LL[i*N2+j];
	//point_t Xn_ib = point_t{dvx,dvy,dvz};
	point_t Xn_ib = Xn;  // mozna to je jinak...
	point_t waveX_ibPlus1 = Xn_ib + U_ib*1;//delta t =1
	point_t XnMinus1 = previous[i*N2+j];
	std::cout << " langangian_force \n i= " << i << "\n" <<
		"j= " << j <<"\n" << 
	" result " << -kappa*(waveX_ibPlus1 -2*Xn + XnMinus1)<<std::endl; 
	return -kappa*(waveX_ibPlus1 -2*Xn + XnMinus1);
}
//operations for point_t
//how to get time--the point in previous time
//is point_t the proper type
//are all of point_t coordinates defined?
//HLPVECTOR previous2;
//LL array indexed from 0-> N1-1 N2-1
template<typename LL_array, typename dreal>
void deformX(int i, int j,LL_array& previous ,LL_array& LL,LL_array& next, int N1, int N2, dreal density)
{
	//boundaries
	//s1=0
	//i*N2 +j
	//does it work correctly for (0,0)and (N1,N2)
	//scaling factors for forward and backword differenes to mitigate oscilations etc
	double a =0.5;//0.9999999;
	double b =0.5;//0.0001;
	 if(i ==0)
	 {
		std::cout<< "i ==0" <<" i "<<i<<" j " << j<<std::endl;
		next[i*N2+j] = LL[i*N2+j];
		 
		//next[i*N2+j] = point_t{0,0,j};
		//next[i*N2+j]= a*second_forward_diff_RHS(i,j,by_s1,LL);
		//previous[i*N2+j]= ;

	 }
	 //s1 = L = N1
	 else if(i==N1-1)
	 {std::cout<< "i ==N1"  <<" i "<<i<<" j " << j<<std::endl;
		//next[i*N2+j] =a*second_backward_diff_RHS(i,j,by_s1,LL);
		//next[i*N2+j]+=b*third_backward_diff_RHS(i,j,by_s1,LL);
		next[i*N2+j] = LL[i*N2+j];
		//previous[i*N2+j]= ;
		//sigma=0;
		//gama=0;

	 }
	 //s2=0 or s2 = H = N2
	 else if(j==0)
	 {
		std::cout<< "j ==0 or N2"  <<" i "<<i<<" j " << j<<std::endl;
		//next[i*N2+j]=a*second_forward_diff_RHS(i,j,by_s2,LL);
		//next[i+N2+j]+=b*third_forward_diff_RHS(i,j,by_s2,LL);
		 next[i*N2+j] = LL[i*N2+j];
		 //previous[i*N2+j]= ;
		//sigma=0;
		//gama=0;

	 }
	 else if(j == N2-1)
	 {
		std::cout<< "j ==0 or N2"  <<" i "<<i<<" j " << j<<std::endl;
		//next[i*N2+j]=a*second_backward_diff_RHS(i,j,by_s2,LL);
		//next[i+N2+j]+=b*third_backward_diff_RHS(i,j,by_s2,LL);
		 next[i*N2+j] = LL[i*N2+j];
		 //previous[i*N2+j]= ;
		//sigma=0;
		//gama=0;

	 }
	 else
	 {
		std::cout<< "else " <<" i "<<i<<" j " << j <<std::endl;
		//next[i*N2+j] += (elastic_force_sum(i,j,LL) -lagrangian_force(previous,LL,i,j) -2*LL[i*N2+j] +previous[i*N2+j])/density;
		
		next[i*N2+j] = (elastic_force_sum(i,j,LL) -lagrangian_force(previous,LL,i,j))/density +2*LL[i*N2+j] -previous[i*N2+j];
		//previous[i*N2+j] = (elastic_force_sum(i,j,LL) -lagrangian_force(previous2,LL,i,j))/density +2*LL[i*N2+j] -previous2[i*N2+j];

	 }


}
template<typename LL_array>
void deform(LL_array&previous, LL_array& LL, LL_array&next, int N1, int N2)
{
	std::cout << "deform is called" << std::endl;

std::cout<<"N1 == "<<N1<<" N2 == " <<N2<<std::endl;

for(int i = 0; i< N1;i++)
{
	for(int j = 0; j< N2;j++)
	{
		std::cout << "firs for i = "<<i<< " j = "<<j<<std::endl;
		deformX(i,j,previous,LL,next);

	}
}

}