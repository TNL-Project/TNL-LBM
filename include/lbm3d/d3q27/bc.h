#pragma once

template< typename CONFIG >
struct D3Q27_BC_All
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
		GEO_INFLOW,
		GEO_OUTFLOW_EQ,
		GEO_OUTFLOW_RIGHT,
		GEO_PERIODIC,
		GEO_NOTHING,
		GEO_SYM_TOP,
		GEO_SYM_TOP_right,
		GEO_SYM_LEFT,
		GEO_SYM_RIGHT,
		GEO_SYM_BACK,
		GEO_SYM_FRONT
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

	template< typename LBM_KS >
	CUDA_HOSTDEV static void preCollision(DATA &SD, LBM_KS &KS, map_t mapgi, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm, idx z, idx zp)
	{
		if (mapgi == GEO_NOTHING) {
			// nema zadny vliv na vypocet, jen pro output
			KS.rho = 1;
			KS.vx = 0;
			KS.vy = 0;
			KS.vz = 0;
			return;
		}

		// modify pull location for streaming
		if (mapgi == GEO_OUTFLOW_RIGHT)
			xp = x = xm;

		// if(mapgi == GEO_INFLOW || mapgi == GEO_OUTFLOW_RIGHT || mapgi == GEO_WALL  || mapgi == GEO_FLUID)
			STREAMING::streaming(SD,KS,xm,x,xp,ym,y,yp,zm,z,zp);

		// boundary conditions
		switch (mapgi)
		{
		case GEO_INFLOW:
			SD.inflow(KS,x,y,z);
			COLL::setEquilibrium(KS);
			break;
		case GEO_OUTFLOW_EQ:
			COLL::computeDensityAndVelocity(KS);
			KS.rho = 1;
			COLL::setEquilibrium(KS);
			break;
		case GEO_OUTFLOW_RIGHT:
			COLL::computeDensityAndVelocity(KS);
			KS.rho = 1;
			break;
		case GEO_WALL:
			// nema zadny vliv na vypocet, jen pro output
			KS.rho = 1;
			KS.vx = 0;
			KS.vy = 0;
			KS.vz = 0;
			// collision step: bounce-back
			TNL::swap( KS.f[mmm], KS.f[ppp] );
			TNL::swap( KS.f[mmz], KS.f[ppz] );
			TNL::swap( KS.f[mmp], KS.f[ppm] );
			TNL::swap( KS.f[mzm], KS.f[pzp] );
			TNL::swap( KS.f[mzz], KS.f[pzz] );
			TNL::swap( KS.f[mzp], KS.f[pzm] );
			TNL::swap( KS.f[mpm], KS.f[pmp] );
			TNL::swap( KS.f[mpz], KS.f[pmz] );
			TNL::swap( KS.f[mpp], KS.f[pmm] );
			TNL::swap( KS.f[zmm], KS.f[zpp] );
			TNL::swap( KS.f[zzm], KS.f[zzp] );
			TNL::swap( KS.f[zmz], KS.f[zpz] );
			TNL::swap( KS.f[zmp], KS.f[zpm] );
			break;
		case GEO_SYM_TOP:
			KS.f[mmm] = KS.f[mmp];
			KS.f[mzm] = KS.f[mzp];
			KS.f[mpm] = KS.f[mpp];
			KS.f[zmm] = KS.f[zmp];
			KS.f[zzm] = KS.f[zzp];
			KS.f[zpm] = KS.f[zpp];
			KS.f[pmm] = KS.f[pmp];
			KS.f[pzm] = KS.f[pzp];
			KS.f[ppm] = KS.f[ppp];
			COLL::computeDensityAndVelocity(KS);
			break;
		case GEO_SYM_LEFT:
			KS.f[pmm] = KS.f[mmm];
			KS.f[pmz] = KS.f[mmz];
			KS.f[pmp] = KS.f[mmp];
			KS.f[pzm] = KS.f[mzm];
			KS.f[pzz] = KS.f[mzz];
			KS.f[pzp] = KS.f[mzp];
			KS.f[ppm] = KS.f[mpm];
			KS.f[ppz] = KS.f[mpz];
			KS.f[ppp] = KS.f[mpp];
			COLL::computeDensityAndVelocity(KS);
			break;
		case GEO_SYM_RIGHT:
			KS.f[mmm] = KS.f[pmm];
			KS.f[mmz] = KS.f[pmz];
			KS.f[mmp] = KS.f[pmp];
			KS.f[mzm] = KS.f[pzm];
			KS.f[mzz] = KS.f[pzz];
			KS.f[mzp] = KS.f[pzp];
			KS.f[mpm] = KS.f[ppm];
			KS.f[mpz] = KS.f[ppz];
			KS.f[mpp] = KS.f[ppp];
			COLL::computeDensityAndVelocity(KS);
			break;
		case GEO_SYM_BACK:
			KS.f[mpm] = KS.f[mmm];
			KS.f[mpz] = KS.f[mmz];
			KS.f[mpp] = KS.f[mmp];
			KS.f[zpm] = KS.f[zmm];
			KS.f[zpz] = KS.f[zmz];
			KS.f[zpp] = KS.f[zmp];
			KS.f[ppm] = KS.f[pmm];
			KS.f[ppz] = KS.f[pmz];
			KS.f[ppp] = KS.f[pmp];
			COLL::computeDensityAndVelocity(KS);
			break;
		case GEO_SYM_FRONT:
			KS.f[mmm] = KS.f[mpm];
			KS.f[mmz] = KS.f[mpz];
			KS.f[mmp] = KS.f[mpp];
			KS.f[zmm] = KS.f[zpm];
			KS.f[zmz] = KS.f[zpz];
			KS.f[zmp] = KS.f[zpp];
			KS.f[pmm] = KS.f[ppm];
			KS.f[pmz] = KS.f[ppz];
			KS.f[pmp] = KS.f[ppp];
			COLL::computeDensityAndVelocity(KS);
			break;

		case GEO_SYM_TOP_right:

			xp = x = xm;
			STREAMING::streaming(SD,KS,xm,x,xp,ym,y,yp,zm,z,zp);

			KS.f[mmm] = KS.f[mmp];
			KS.f[mzm] = KS.f[mzp];
			KS.f[mpm] = KS.f[mpp];
			KS.f[zmm] = KS.f[zmp];
			KS.f[zzm] = KS.f[zzp];
			KS.f[zpm] = KS.f[zpp];
			KS.f[pmm] = KS.f[pmp];
			KS.f[pzm] = KS.f[pzp];
			KS.f[ppm] = KS.f[ppp];
			KS.f[pmz] = KS.f[mmz];
			KS.f[pmp] = KS.f[mmp];
			KS.f[pzz] = KS.f[mzz];
			KS.f[pzp] = KS.f[mzp];
			KS.f[ppz] = KS.f[mpz];
			KS.f[ppp] = KS.f[mpp];

			COLL::computeDensityAndVelocity(KS);
			break;
		default:
			COLL::computeDensityAndVelocity(KS);
			break;
		}
	}

	CUDA_HOSTDEV static bool doCollision(map_t mapgi)
	{
		// by default, collision is done on non-BC sites only
		// additionally, BCs which include the collision step should be specified here
		return isFluid(mapgi) || isPeriodic(mapgi)
			|| mapgi == GEO_OUTFLOW_RIGHT;
	}

	template< typename LBM_KS >
	CUDA_HOSTDEV static void postCollision(DATA &SD, LBM_KS &KS, map_t mapgi, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm, idx z, idx zp)
	{
		if (mapgi == GEO_NOTHING)
			return;

		STREAMING::postCollisionStreaming(SD,KS,xm,x,xp,ym,y,yp,zm,z,zp);
	}
};
