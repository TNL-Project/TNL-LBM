#pragma once

template< typename CONFIG >
struct D3Q7_BC_All
{
	using COLL = typename CONFIG::COLL;
	using STREAMING = typename CONFIG::STREAMING;
	using DATA = typename CONFIG::DATA;

	using map_t = typename CONFIG::TRAITS::map_t;
	using idx = typename CONFIG::TRAITS::idx;
	using dreal = typename CONFIG::TRAITS::dreal;

	enum GEO : map_t {
		GEO_FLUID, 		// compulsory
		GEO_WALL, 		// compulsory
		GEO_WALL_BODY,
		GEO_SOLID,
		GEO_TRANSFER_FS,
		GEO_TRANSFER_SF,
		GEO_TRANSFER_SW,
		GEO_INFLOW,
		GEO_OUTFLOW_RIGHT,
		GEO_PERIODIC,
		GEO_NOTHING,
		GEO_SYM_TOP,
		GEO_OUTFLOW_ABB,
		GEO_OUTFLOW_PE,
		GEO_SYM_TOP_right
	};

	CUDA_HOSTDEV static bool isPeriodic(map_t mapgi)
	{
		return (mapgi==GEO_PERIODIC);
	}

	CUDA_HOSTDEV static bool isFluid(map_t mapgi)
	{
		return (mapgi==GEO_FLUID);
	}

	CUDA_HOSTDEV static bool isWall(map_t mapgi)
	{
		return (mapgi==GEO_WALL);
	}
	
	CUDA_HOSTDEV static bool isFluidPhase(map_t mapgi)		//Solid phase is solid + boundary StoF
	{
		return (mapgi==GEO_FLUID || mapgi==GEO_TRANSFER_SF);
	}
	
	CUDA_HOSTDEV static bool isSolid(map_t mapgi)
	{
		return (mapgi==GEO_SOLID);
	}
	
	CUDA_HOSTDEV static bool isSolidPhase(map_t mapgi)		//Solid phase is solid + boundary SF
	{
		return (mapgi==GEO_SOLID || mapgi==GEO_TRANSFER_SF || mapgi==GEO_TRANSFER_SW);
	}
	
	CUDA_HOSTDEV static bool isTransferSF(map_t mapgi)
	{
		return (mapgi==GEO_TRANSFER_SF);
	}
	
	CUDA_HOSTDEV static bool isTransferFS(map_t mapgi)
	{
		return (mapgi==GEO_TRANSFER_FS);
	}

	template< typename LBM_KS >
	CUDA_HOSTDEV static void preCollision(DATA &SD, LBM_KS &KS, map_t mapgi, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm, idx z, idx zp)
	{
		if (mapgi == GEO_NOTHING) {
			// nema zadny vliv na vypocet, jen pro output
			KS.phi = 0;
			return;
		}

		// modify pull location for streaming
		if (mapgi == GEO_OUTFLOW_RIGHT)
			xp = x = xm;

		if(mapgi == GEO_INFLOW || mapgi == GEO_WALL || mapgi == GEO_WALL_BODY || mapgi == GEO_TRANSFER_FS || mapgi == GEO_TRANSFER_SF || mapgi == GEO_TRANSFER_SW)
			STREAMING::streaming(SD,KS,xm,x,xp,ym,y,yp,zm,z,zp);

		// boundary conditions
		switch (mapgi)
		{
		case GEO_INFLOW:
			SD.inflow(KS,x,y,z);
			COLL::setEquilibrium(KS);
			break;
		case GEO_OUTFLOW_PE:
			STREAMING::streaming(SD,KS,xm-1,xm,x,ym,y,yp,zm,z,zp);
			COLL::computeDensityAndVelocity(KS);
			COLL::setEquilibrium(KS);
			break;

		case GEO_OUTFLOW_RIGHT:
			STREAMING::streaming(SD,KS,xm,x,xm,ym,y,yp,zm,z,zp);
			COLL::computeDensityAndVelocity(KS);
			break;
		case GEO_WALL:
			//KS.phi = 0;
			COLL::computeDensityAndVelocity(KS);
			// collision step: bounce-back
			TNL::swap( KS.f[mzz], KS.f[pzz] );
			TNL::swap( KS.f[zmz], KS.f[zpz] );
			TNL::swap( KS.f[zzm], KS.f[zzp] );
			break;
		
		case GEO_WALL_BODY:
			COLL::computeDensityAndVelocity(KS);
			// collision step: bounce-back
			TNL::swap( KS.f[mzz], KS.f[pzz] );
			TNL::swap( KS.f[zmz], KS.f[zpz] );
			TNL::swap( KS.f[zzm], KS.f[zzp] );
			// anti-bounce-back (recovers zero gradient across the wall boundary, see Kruger section 8.5.2.1)
			for (int q = 0; q < 7; q++)
			{
				if(q == zzz)
					KS.f[q] = -KS.f[q] + 2 * n1o4 * KS.phi;
				else
					KS.f[q] = -KS.f[q] + 2 * n1o8 * KS.phi;
			}
			// TODO: Kruger's eq (8.54) includes concentration imposed on the wall - does it diffusively propagate into the domain? -- Yes DH2022
			break;

		case GEO_OUTFLOW_ABB:
			COLL::computeDensityAndVelocity(KS);
			// collision step: bounce-back
			TNL::swap( KS.f[mzz], KS.f[pzz] );
			TNL::swap( KS.f[zmz], KS.f[zpz] );
			TNL::swap( KS.f[zzm], KS.f[zzp] );
			// anti-bounce-back (recovers zero gradient across the wall boundary, see Kruger section 8.5.2.1)
			for (int q = 0; q < 7; q++)
			{
				if(q == zzz)
					KS.f[q] = -KS.f[q] + 2 * n1o4 * KS.phi;
				else
					KS.f[q] = -KS.f[q] + 2 * n1o8 * KS.phi;
			}
			// TODO: Kruger's eq (8.54) includes concentration imposed on the wall - does it diffusively propagate into the domain? -- Yes DH2022
			break;
		
		case GEO_SYM_TOP:
			// double change = KS.f[zzp];
			// STREAMING::streaming(SD,KS,xm,x,xp,ym,y,yp,zm,z,zp);
			
			// KS.f[mmm] = KS.f[mmp];
			// KS.f[mzm] = KS.f[mzp];
			// KS.f[mpm] = KS.f[mpp];
			// KS.f[zmm] = KS.f[zmp];
			KS.f[zzm] = KS.f[zzp];
			// KS.f[zpm] = KS.f[zpp];
			// KS.f[pmm] = KS.f[pmp];
			// KS.f[pzm] = KS.f[pzp];
			// KS.f[ppm] = KS.f[ppp];
			// KS.f[zzm] = change;
			COLL::computeDensityAndVelocity(KS);
			break;
		
		case GEO_SYM_TOP_right:
			// double change = KS.f[zzp];
			xp = x = xm;
			STREAMING::streaming(SD,KS,xm,x,xp,ym,y,yp,zm,z,zp);

			
			// KS.f[mmm] = KS.f[mmp];
			// KS.f[mzm] = KS.f[mzp];
			// KS.f[mpm] = KS.f[mpp];
			// KS.f[zmm] = KS.f[zmp];
			KS.f[mzz] = KS.f[pzz];
			KS.f[zzm] = KS.f[zzp];
			// KS.f[zpm] = KS.f[zpp];
			// KS.f[pmm] = KS.f[pmp];
			// KS.f[pzm] = KS.f[pzp];
			// KS.f[ppm] = KS.f[ppp];
			// KS.f[zzm] = change;
			COLL::computeDensityAndVelocity(KS);
			break;

		case GEO_TRANSFER_FS: {
			//Streaming
			// STREAMING::streaming(SD,KS,xm,x,xp,ym,y,yp,zm,z,zp);
			dreal Temp[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
				Temp[0] = SD.df(df_cur,zzz,xp,y,z) + SD.df(df_cur,pzz,xp,y,z) + SD.df(df_cur,zpz,xp,y,z) + SD.df(df_cur,zzp,xp,y,z) + SD.df(df_cur,mzz,xp,y,z) + SD.df(df_cur,zmz,xp,y,z) +  SD.df(df_cur,zzm,xp,y,z);
				Temp[1] = SD.df(df_cur,zzz,x,yp,z) + SD.df(df_cur,pzz,x,yp,z) + SD.df(df_cur,zpz,x,yp,z) + SD.df(df_cur,zzp,x,yp,z) + SD.df(df_cur,mzz,x,yp,z) + SD.df(df_cur,zmz,x,yp,z) +  SD.df(df_cur,zzm,x,yp,z);
				Temp[2] = SD.df(df_cur,zzz,x,y,zp) + SD.df(df_cur,pzz,x,y,zp) + SD.df(df_cur,zpz,x,y,zp) + SD.df(df_cur,zzp,x,y,zp) + SD.df(df_cur,mzz,x,y,zp) + SD.df(df_cur,zmz,x,y,zp) +  SD.df(df_cur,zzm,x,y,zp);
				Temp[3] = SD.df(df_cur,zzz,xm,y,z) + SD.df(df_cur,pzz,xm,y,z) + SD.df(df_cur,zpz,xm,y,z) + SD.df(df_cur,zzp,xm,y,z) + SD.df(df_cur,mzz,xm,y,z) + SD.df(df_cur,zmz,xm,y,z) +  SD.df(df_cur,zzm,xm,y,z);
				Temp[4] = SD.df(df_cur,zzz,x,ym,z) + SD.df(df_cur,pzz,x,ym,z) + SD.df(df_cur,zpz,x,ym,z) + SD.df(df_cur,zzp,x,ym,z) + SD.df(df_cur,mzz,x,ym,z) + SD.df(df_cur,zmz,x,ym,z) +  SD.df(df_cur,zzm,x,ym,z);
				Temp[5] = SD.df(df_cur,zzz,x,y,zm) + SD.df(df_cur,pzz,x,y,zm) + SD.df(df_cur,zpz,x,y,zm) + SD.df(df_cur,zzp,x,y,zm) + SD.df(df_cur,mzz,x,y,zm) + SD.df(df_cur,zmz,x,y,zm) +  SD.df(df_cur,zzm,x,y,zm);
				
			if(SD.transferDir(pzz, x, y, z)){	KS.f[mzz] = SD.df(df_cur,pzz,x, y, z) + SD.C*(Temp[0] - SD.macro(0, x, y, z));}
			if(SD.transferDir(zpz, x, y, z)){	KS.f[zmz] = SD.df(df_cur,zpz,x, y, z) + SD.C*(Temp[1] - SD.macro(0, x, y, z));}
			if(SD.transferDir(zzp, x, y, z)){	KS.f[zzm] = SD.df(df_cur,zzp,x, y, z) + SD.C*(Temp[2] - SD.macro(0, x, y, z));}
			if(SD.transferDir(mzz, x, y, z)){	KS.f[pzz] = SD.df(df_cur,mzz,x, y, z) + SD.C*(Temp[3] - SD.macro(0, x, y, z));}
			if(SD.transferDir(zmz, x, y, z)){	KS.f[zpz] = SD.df(df_cur,zmz,x, y, z) + SD.C*(Temp[4] - SD.macro(0, x, y, z));}
			if(SD.transferDir(zzm, x, y, z)){	KS.f[zzp] = SD.df(df_cur,zzm,x, y, z) + SD.C*(Temp[5] - SD.macro(0, x, y, z));}
			COLL::computeDensityAndVelocity(KS);
			break;
		}	
			
		case GEO_TRANSFER_SF: {
			//Streaming
			// STREAMING::streaming(SD,KS,xm,x,xp,ym,y,yp,zm,z,zp);

			dreal Temp[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
				Temp[0] = SD.df(df_cur,zzz,xp,y,z) + SD.df(df_cur,pzz,xp,y,z) + SD.df(df_cur,zpz,xp,y,z) + SD.df(df_cur,zzp,xp,y,z) + SD.df(df_cur,mzz,xp,y,z) + SD.df(df_cur,zmz,xp,y,z) +  SD.df(df_cur,zzm,xp,y,z);
				Temp[1] = SD.df(df_cur,zzz,x,yp,z) + SD.df(df_cur,pzz,x,yp,z) + SD.df(df_cur,zpz,x,yp,z) + SD.df(df_cur,zzp,x,yp,z) + SD.df(df_cur,mzz,x,yp,z) + SD.df(df_cur,zmz,x,yp,z) +  SD.df(df_cur,zzm,x,yp,z);
				Temp[2] = SD.df(df_cur,zzz,x,y,zp) + SD.df(df_cur,pzz,x,y,zp) + SD.df(df_cur,zpz,x,y,zp) + SD.df(df_cur,zzp,x,y,zp) + SD.df(df_cur,mzz,x,y,zp) + SD.df(df_cur,zmz,x,y,zp) +  SD.df(df_cur,zzm,x,y,zp);
				Temp[3] = SD.df(df_cur,zzz,xm,y,z) + SD.df(df_cur,pzz,xm,y,z) + SD.df(df_cur,zpz,xm,y,z) + SD.df(df_cur,zzp,xm,y,z) + SD.df(df_cur,mzz,xm,y,z) + SD.df(df_cur,zmz,xm,y,z) +  SD.df(df_cur,zzm,xm,y,z);
				Temp[4] = SD.df(df_cur,zzz,x,ym,z) + SD.df(df_cur,pzz,x,ym,z) + SD.df(df_cur,zpz,x,ym,z) + SD.df(df_cur,zzp,x,ym,z) + SD.df(df_cur,mzz,x,ym,z) + SD.df(df_cur,zmz,x,ym,z) +  SD.df(df_cur,zzm,x,ym,z);
				Temp[5] = SD.df(df_cur,zzz,x,y,zm) + SD.df(df_cur,pzz,x,y,zm) + SD.df(df_cur,zpz,x,y,zm) + SD.df(df_cur,zzp,x,y,zm) + SD.df(df_cur,mzz,x,y,zm) + SD.df(df_cur,zmz,x,y,zm) +  SD.df(df_cur,zzm,x,y,zm);
			
			
			if(SD.transferDir(pzz, x, y, z))	KS.f[mzz] = SD.df(df_cur,pzz,x, y, z) + SD.C*(Temp[0] - SD.macro(0, x, y, z));
			if(SD.transferDir(zpz, x, y, z))	KS.f[zmz] = SD.df(df_cur,zpz,x, y, z) + SD.C*(Temp[1] - SD.macro(0, x, y, z));
			if(SD.transferDir(zzp, x, y, z))	KS.f[zzm] = SD.df(df_cur,zzp,x, y, z) + SD.C*(Temp[2] - SD.macro(0, x, y, z));
			if(SD.transferDir(mzz, x, y, z))	KS.f[pzz] = SD.df(df_cur,mzz,x, y, z) + SD.C*(Temp[3] - SD.macro(0, x, y, z));
			if(SD.transferDir(zmz, x, y, z))	KS.f[zpz] = SD.df(df_cur,zmz,x, y, z) + SD.C*(Temp[4] - SD.macro(0, x, y, z));
			if(SD.transferDir(zzm, x, y, z))	KS.f[zzp] = SD.df(df_cur,zzm,x, y, z) + SD.C*(Temp[5] - SD.macro(0, x, y, z));
			COLL::computeDensityAndVelocity(KS);
			break;
			}

			case GEO_TRANSFER_SW: {

			// STREAMING::streaming(SD,KS,xm,x,xp,ym,y,yp,zm,z,zp);
			if(SD.transferDir(pzz, x, y, z))	KS.f[mzz] = SD.df(df_cur,pzz,x, y, z);//TNL::swap( KS.f[mzz], KS.f[pzz] );//KS.f[mzz] = SD.df(df_cur,pzz,x, y, z) + SD.C*(Temp[0] - SD.macro(0, x, y, z));
			if(SD.transferDir(zpz, x, y, z))	KS.f[zmz] = SD.df(df_cur,zpz,x, y, z);//TNL::swap( KS.f[zmz], KS.f[zpz] );//KS.f[zmz] = SD.df(df_cur,zpz,x, y, z) + SD.C*(Temp[1] - SD.macro(0, x, y, z));
			if(SD.transferDir(zzp, x, y, z))	KS.f[zzm] = SD.df(df_cur,zzp,x, y, z);//TNL::swap( KS.f[zzm], KS.f[zzp] );//KS.f[zzm] = SD.df(df_cur,zzp,x, y, z) + SD.C*(Temp[2] - SD.macro(0, x, y, z));
			if(SD.transferDir(mzz, x, y, z))	KS.f[pzz] = SD.df(df_cur,mzz,x, y, z);//TNL::swap( KS.f[mzz], KS.f[pzz] );//KS.f[pzz] = SD.df(df_cur,mzz,x, y, z) + SD.C*(Temp[3] - SD.macro(0, x, y, z));
			if(SD.transferDir(zmz, x, y, z))	KS.f[zpz] = SD.df(df_cur,zmz,x, y, z);//TNL::swap( KS.f[zmz], KS.f[zpz] );//KS.f[zpz] = SD.df(df_cur,zmz,x, y, z) + SD.C*(Temp[4] - SD.macro(0, x, y, z));
			if(SD.transferDir(zzm, x, y, z))	KS.f[zzp] = SD.df(df_cur,zzm,x, y, z);//TNL::swap( KS.f[zzm], KS.f[zzp] );//KS.f[zzp] = SD.df(df_cur,zzm,x, y, z) + SD.C*(Temp[5] - SD.macro(0, x, y, z));
			COLL::computeDensityAndVelocity(KS);
			break;
			}
		default:
			COLL::computeDensityAndVelocity(KS);
			break;
		}
	}

	CUDA_HOSTDEV static bool doCollision(map_t mapgi)
	{
		// by default, collision is done on non-BC sites only
		// additionally, BCs which include the collision step should be specified here
		return isFluid(mapgi) || isPeriodic(mapgi) || isSolid(mapgi) || mapgi == GEO_TRANSFER_SF || mapgi == GEO_TRANSFER_FS || mapgi == GEO_OUTFLOW_RIGHT || mapgi == GEO_TRANSFER_SW;
	}

	template< typename LBM_KS >
	CUDA_HOSTDEV static void postCollision(DATA &SD, LBM_KS &KS, map_t mapgi, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm, idx z, idx zp)
	{
		if (mapgi == GEO_NOTHING)
			return;

		STREAMING::postCollisionStreaming(SD,KS,xm,x,xp,ym,y,yp,zm,z,zp);
	}
};
