#pragma once

#include "lbm3d/defs.h"
#include "lbm_common/rounding.h"

// Esoteric Twist (Geier & Schönherr 2017): single in-place DF array.
// Directions with negative components are pulled from the upper-octant
// neighbors while positive directions are read at the own site; after
// collision the values are written back in opposite (twisted) direction, so
// every memory location is read and written by the same thread in each
// launch. The pointer swap of the paper is realized by alternating two
// parities:
// - even_iter == false (phase A): natural slots, read at site n + sigma(c)
//   with sigma(c) = |min(c, 0)| componentwise; write back in opposite slots
//   at the same sites (each site writes post[opp(c)] into slot c).
// - even_iter == true (phase B): the twist counterpart: read from opposite
//   slots at the same sites; write back to own slots at site n + p(c) with
//   p(c) = max(c, 0) componentwise.
// The pre-collision populations read here are identical to the A-B pull
// scheme's at every launch, provided the initial DF field is placed as:
// slot (c, s) = eq_c(s - p(c)).
//
// LIMITATION (same as the A-A pattern): the non-Newtonian methods
// streamingRho / streamingVx / streamingVy / streamingVz dereference slots
// at TWO-step neighbor offsets. They are safe only from a separate,
// read-only-on-df_cur kernel launch (computeNonNewtonianKernels), never from
// the main LBM kernel where df_cur is read-write.
template <typename TRAITS>
struct D3Q27_STREAMING_ESO_TWIST
{
	static constexpr int DFMAX = 1;
	// DF slot that holds the freshly written field after a kernel launch
	static constexpr std::uint8_t output_df = df_cur;

	using idx = typename TRAITS::idx;
	using dreal = typename TRAITS::dreal;

	using SyncDirection = TNL::Containers::SyncDirection;

	template <typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void
	postCollisionStreaming(LBM_DATA& SD, LBM_KS& KS, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm, idx z, idx zp)
	{
		if (! SD.even_iter) {
			// phase A: slot (c, n + sigma(c)) receives post[opp(c)]
			SD.df(df_cur, dir27::zzz, x, y, z) = KS.f[dir27::zzz];
			SD.df(df_cur, dir27::pzz, x, y, z) = KS.f[dir27::mzz];
			SD.df(df_cur, dir27::mzz, xp, y, z) = KS.f[dir27::pzz];
			SD.df(df_cur, dir27::zpz, x, y, z) = KS.f[dir27::zmz];
			SD.df(df_cur, dir27::zmz, x, yp, z) = KS.f[dir27::zpz];
			SD.df(df_cur, dir27::zzp, x, y, z) = KS.f[dir27::zzm];
			SD.df(df_cur, dir27::zzm, x, y, zp) = KS.f[dir27::zzp];
			SD.df(df_cur, dir27::ppz, x, y, z) = KS.f[dir27::mmz];
			SD.df(df_cur, dir27::mmz, xp, yp, z) = KS.f[dir27::ppz];
			SD.df(df_cur, dir27::pmz, x, yp, z) = KS.f[dir27::mpz];
			SD.df(df_cur, dir27::mpz, xp, y, z) = KS.f[dir27::pmz];
			SD.df(df_cur, dir27::pzp, x, y, z) = KS.f[dir27::mzm];
			SD.df(df_cur, dir27::mzm, xp, y, zp) = KS.f[dir27::pzp];
			SD.df(df_cur, dir27::pzm, x, y, zp) = KS.f[dir27::mzp];
			SD.df(df_cur, dir27::mzp, xp, y, z) = KS.f[dir27::pzm];
			SD.df(df_cur, dir27::zpp, x, y, z) = KS.f[dir27::zmm];
			SD.df(df_cur, dir27::zmm, x, yp, zp) = KS.f[dir27::zpp];
			SD.df(df_cur, dir27::zpm, x, y, zp) = KS.f[dir27::zmp];
			SD.df(df_cur, dir27::zmp, x, yp, z) = KS.f[dir27::zpm];
			SD.df(df_cur, dir27::ppp, x, y, z) = KS.f[dir27::mmm];
			SD.df(df_cur, dir27::mmm, xp, yp, zp) = KS.f[dir27::ppp];
			SD.df(df_cur, dir27::ppm, x, y, zp) = KS.f[dir27::mmp];
			SD.df(df_cur, dir27::mmp, xp, yp, z) = KS.f[dir27::ppm];
			SD.df(df_cur, dir27::pmp, x, yp, z) = KS.f[dir27::mpm];
			SD.df(df_cur, dir27::mpm, xp, y, zp) = KS.f[dir27::pmp];
			SD.df(df_cur, dir27::pmm, x, yp, zp) = KS.f[dir27::mpp];
			SD.df(df_cur, dir27::mpp, xp, y, z) = KS.f[dir27::pmm];
		}
		else {
			// phase B: slot (d, n + p(d)) keeps post[d]
			SD.df(df_cur, dir27::zzz, x, y, z) = KS.f[dir27::zzz];
			SD.df(df_cur, dir27::pzz, xp, y, z) = KS.f[dir27::pzz];
			SD.df(df_cur, dir27::mzz, x, y, z) = KS.f[dir27::mzz];
			SD.df(df_cur, dir27::zpz, x, yp, z) = KS.f[dir27::zpz];
			SD.df(df_cur, dir27::zmz, x, y, z) = KS.f[dir27::zmz];
			SD.df(df_cur, dir27::zzp, x, y, zp) = KS.f[dir27::zzp];
			SD.df(df_cur, dir27::zzm, x, y, z) = KS.f[dir27::zzm];
			SD.df(df_cur, dir27::ppz, xp, yp, z) = KS.f[dir27::ppz];
			SD.df(df_cur, dir27::mmz, x, y, z) = KS.f[dir27::mmz];
			SD.df(df_cur, dir27::pmz, xp, y, z) = KS.f[dir27::pmz];
			SD.df(df_cur, dir27::mpz, x, yp, z) = KS.f[dir27::mpz];
			SD.df(df_cur, dir27::pzp, xp, y, zp) = KS.f[dir27::pzp];
			SD.df(df_cur, dir27::mzm, x, y, z) = KS.f[dir27::mzm];
			SD.df(df_cur, dir27::pzm, xp, y, z) = KS.f[dir27::pzm];
			SD.df(df_cur, dir27::mzp, x, y, zp) = KS.f[dir27::mzp];
			SD.df(df_cur, dir27::zpp, x, yp, zp) = KS.f[dir27::zpp];
			SD.df(df_cur, dir27::zmm, x, y, z) = KS.f[dir27::zmm];
			SD.df(df_cur, dir27::zpm, x, yp, z) = KS.f[dir27::zpm];
			SD.df(df_cur, dir27::zmp, x, y, zp) = KS.f[dir27::zmp];
			SD.df(df_cur, dir27::ppp, xp, yp, zp) = KS.f[dir27::ppp];
			SD.df(df_cur, dir27::mmm, x, y, z) = KS.f[dir27::mmm];
			SD.df(df_cur, dir27::ppm, xp, yp, z) = KS.f[dir27::ppm];
			SD.df(df_cur, dir27::mmp, x, y, zp) = KS.f[dir27::mmp];
			SD.df(df_cur, dir27::pmp, xp, y, zp) = KS.f[dir27::pmp];
			SD.df(df_cur, dir27::mpm, x, yp, z) = KS.f[dir27::mpm];
			SD.df(df_cur, dir27::pmm, xp, y, z) = KS.f[dir27::pmm];
			SD.df(df_cur, dir27::mpp, x, yp, zp) = KS.f[dir27::mpp];
		}
	}

	template <typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void streaming(LBM_DATA& SD, LBM_KS& KS, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm, idx z, idx zp)
	{
		if (! SD.even_iter) {
			// phase A: KS.f[c] = slot (c, n + sigma(c))
			KS.f[dir27::zzz] = TNL::Backend::ldg(SD.df(df_cur, dir27::zzz, x, y, z));
			KS.f[dir27::pzz] = TNL::Backend::ldg(SD.df(df_cur, dir27::pzz, x, y, z));
			KS.f[dir27::mzz] = TNL::Backend::ldg(SD.df(df_cur, dir27::mzz, xp, y, z));
			KS.f[dir27::zpz] = TNL::Backend::ldg(SD.df(df_cur, dir27::zpz, x, y, z));
			KS.f[dir27::zmz] = TNL::Backend::ldg(SD.df(df_cur, dir27::zmz, x, yp, z));
			KS.f[dir27::zzp] = TNL::Backend::ldg(SD.df(df_cur, dir27::zzp, x, y, z));
			KS.f[dir27::zzm] = TNL::Backend::ldg(SD.df(df_cur, dir27::zzm, x, y, zp));
			KS.f[dir27::ppz] = TNL::Backend::ldg(SD.df(df_cur, dir27::ppz, x, y, z));
			KS.f[dir27::mmz] = TNL::Backend::ldg(SD.df(df_cur, dir27::mmz, xp, yp, z));
			KS.f[dir27::pmz] = TNL::Backend::ldg(SD.df(df_cur, dir27::pmz, x, yp, z));
			KS.f[dir27::mpz] = TNL::Backend::ldg(SD.df(df_cur, dir27::mpz, xp, y, z));
			KS.f[dir27::pzp] = TNL::Backend::ldg(SD.df(df_cur, dir27::pzp, x, y, z));
			KS.f[dir27::mzm] = TNL::Backend::ldg(SD.df(df_cur, dir27::mzm, xp, y, zp));
			KS.f[dir27::pzm] = TNL::Backend::ldg(SD.df(df_cur, dir27::pzm, x, y, zp));
			KS.f[dir27::mzp] = TNL::Backend::ldg(SD.df(df_cur, dir27::mzp, xp, y, z));
			KS.f[dir27::zpp] = TNL::Backend::ldg(SD.df(df_cur, dir27::zpp, x, y, z));
			KS.f[dir27::zmm] = TNL::Backend::ldg(SD.df(df_cur, dir27::zmm, x, yp, zp));
			KS.f[dir27::zpm] = TNL::Backend::ldg(SD.df(df_cur, dir27::zpm, x, y, zp));
			KS.f[dir27::zmp] = TNL::Backend::ldg(SD.df(df_cur, dir27::zmp, x, yp, z));
			KS.f[dir27::ppp] = TNL::Backend::ldg(SD.df(df_cur, dir27::ppp, x, y, z));
			KS.f[dir27::mmm] = TNL::Backend::ldg(SD.df(df_cur, dir27::mmm, xp, yp, zp));
			KS.f[dir27::ppm] = TNL::Backend::ldg(SD.df(df_cur, dir27::ppm, x, y, zp));
			KS.f[dir27::mmp] = TNL::Backend::ldg(SD.df(df_cur, dir27::mmp, xp, yp, z));
			KS.f[dir27::pmp] = TNL::Backend::ldg(SD.df(df_cur, dir27::pmp, x, yp, z));
			KS.f[dir27::mpm] = TNL::Backend::ldg(SD.df(df_cur, dir27::mpm, xp, y, zp));
			KS.f[dir27::pmm] = TNL::Backend::ldg(SD.df(df_cur, dir27::pmm, x, yp, zp));
			KS.f[dir27::mpp] = TNL::Backend::ldg(SD.df(df_cur, dir27::mpp, xp, y, z));
		}
		else {
			// phase B: KS.f[c] = slot (opp(c), n + sigma(c))
			KS.f[dir27::zzz] = TNL::Backend::ldg(SD.df(df_cur, dir27::zzz, x, y, z));
			KS.f[dir27::pzz] = TNL::Backend::ldg(SD.df(df_cur, dir27::mzz, x, y, z));
			KS.f[dir27::mzz] = TNL::Backend::ldg(SD.df(df_cur, dir27::pzz, xp, y, z));
			KS.f[dir27::zpz] = TNL::Backend::ldg(SD.df(df_cur, dir27::zmz, x, y, z));
			KS.f[dir27::zmz] = TNL::Backend::ldg(SD.df(df_cur, dir27::zpz, x, yp, z));
			KS.f[dir27::zzp] = TNL::Backend::ldg(SD.df(df_cur, dir27::zzm, x, y, z));
			KS.f[dir27::zzm] = TNL::Backend::ldg(SD.df(df_cur, dir27::zzp, x, y, zp));
			KS.f[dir27::ppz] = TNL::Backend::ldg(SD.df(df_cur, dir27::mmz, x, y, z));
			KS.f[dir27::mmz] = TNL::Backend::ldg(SD.df(df_cur, dir27::ppz, xp, yp, z));
			KS.f[dir27::pmz] = TNL::Backend::ldg(SD.df(df_cur, dir27::mpz, x, yp, z));
			KS.f[dir27::mpz] = TNL::Backend::ldg(SD.df(df_cur, dir27::pmz, xp, y, z));
			KS.f[dir27::pzp] = TNL::Backend::ldg(SD.df(df_cur, dir27::mzm, x, y, z));
			KS.f[dir27::mzm] = TNL::Backend::ldg(SD.df(df_cur, dir27::pzp, xp, y, zp));
			KS.f[dir27::pzm] = TNL::Backend::ldg(SD.df(df_cur, dir27::mzp, x, y, zp));
			KS.f[dir27::mzp] = TNL::Backend::ldg(SD.df(df_cur, dir27::pzm, xp, y, z));
			KS.f[dir27::zpp] = TNL::Backend::ldg(SD.df(df_cur, dir27::zmm, x, y, z));
			KS.f[dir27::zmm] = TNL::Backend::ldg(SD.df(df_cur, dir27::zpp, x, yp, zp));
			KS.f[dir27::zpm] = TNL::Backend::ldg(SD.df(df_cur, dir27::zmp, x, y, zp));
			KS.f[dir27::zmp] = TNL::Backend::ldg(SD.df(df_cur, dir27::zpm, x, yp, z));
			KS.f[dir27::ppp] = TNL::Backend::ldg(SD.df(df_cur, dir27::mmm, x, y, z));
			KS.f[dir27::mmm] = TNL::Backend::ldg(SD.df(df_cur, dir27::ppp, xp, yp, zp));
			KS.f[dir27::ppm] = TNL::Backend::ldg(SD.df(df_cur, dir27::mmp, x, y, zp));
			KS.f[dir27::mmp] = TNL::Backend::ldg(SD.df(df_cur, dir27::ppm, xp, yp, z));
			KS.f[dir27::pmp] = TNL::Backend::ldg(SD.df(df_cur, dir27::mpm, x, yp, z));
			KS.f[dir27::mpm] = TNL::Backend::ldg(SD.df(df_cur, dir27::pmp, xp, y, zp));
			KS.f[dir27::pmm] = TNL::Backend::ldg(SD.df(df_cur, dir27::mpp, x, yp, zp));
			KS.f[dir27::mpp] = TNL::Backend::ldg(SD.df(df_cur, dir27::pmm, xp, y, z));
		}
	}

	// streaming with the bounce-back rule applied: the identity write-back of
	// this pattern preserves the implicit bounce-back of the esoteric schemes
	// (a swapped wall cell writes back exactly what it read)
	template <typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void streamingBounceBack(LBM_DATA& SD, LBM_KS& KS, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm, idx z, idx zp)
	{
		streaming(SD, KS, xm, x, xp, ym, y, yp, zm, z, zp);
		TNL::swap(KS.f[dir27::mmm], KS.f[dir27::ppp]);
		TNL::swap(KS.f[dir27::mmz], KS.f[dir27::ppz]);
		TNL::swap(KS.f[dir27::mmp], KS.f[dir27::ppm]);
		TNL::swap(KS.f[dir27::mzm], KS.f[dir27::pzp]);
		TNL::swap(KS.f[dir27::mzz], KS.f[dir27::pzz]);
		TNL::swap(KS.f[dir27::mzp], KS.f[dir27::pzm]);
		TNL::swap(KS.f[dir27::mpm], KS.f[dir27::pmp]);
		TNL::swap(KS.f[dir27::mpz], KS.f[dir27::pmz]);
		TNL::swap(KS.f[dir27::mpp], KS.f[dir27::pmm]);
		TNL::swap(KS.f[dir27::zmm], KS.f[dir27::zpp]);
		TNL::swap(KS.f[dir27::zzm], KS.f[dir27::zzp]);
		TNL::swap(KS.f[dir27::zmz], KS.f[dir27::zpz]);
		TNL::swap(KS.f[dir27::zmp], KS.f[dir27::zpm]);
	}

	// the post-collision population of direction i authored at site w, given
	// the layout finalized by the previous launch and the current parity: the
	// site offset p(c) = max(c, 0) is parity-independent (the sigma components
	// of each phase's layout are absorbed by the slot, which is the only thing
	// flipping):
	// - even_iter == true  (post-phase-A layout): slot (opp i, w + p(i))
	// - even_iter == false (post-phase-B layout): slot (i, w + p(i))
	template <typename LBM_DATA>
	__cuda_callable__ static dreal postCollValue(LBM_DATA& SD, int i, idx wx, idx wy, idx wz)
	{
		const idx ox = dir27_cx(i) > 0 ? 1 : 0;
		const idx oy = dir27_cy(i) > 0 ? 1 : 0;
		const idx oz = dir27_cz(i) > 0 ? 1 : 0;
		const int slot = SD.even_iter ? opposite_direction(i) : i;
		return TNL::Backend::ldg(SD.df(df_cur, slot, wx + ox, wy + oy, wz + oz));
	}

	// adjoint ("reversed") gather: reads the post-collision state one full hop
	// downstream in direction i (the A-B-even equivalent), expressed through
	// the same postCollValue layout resolution. EXPERIMENTAL, same status as
	// the A-A pattern's adjoint (sim_adjoint is A-B pull only).
	template <typename LBM_DATA, typename LBM_KS>
	CUDA_HOSTDEV static void
	streamingAdjoint(uint8_t type, LBM_DATA& SD, LBM_KS& KS, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm, idx z, idx zp)
	{
		static_cast<void>(type);
		for (int i = 0; i < 27; i++)
			KS.f[i] = postCollValue(SD, i, x + dir27_cx(i), y + dir27_cy(i), z + dir27_cz(i));
	}

	template <typename LBM_DATA, typename LBM_KS>
	CUDA_HOSTDEV static void streamingAdjoint(LBM_DATA& SD, LBM_KS& KS, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm, idx z, idx zp)
	{
		streamingAdjoint(df_cur, SD, KS, xm, x, xp, ym, y, yp, zm, z, zp);
	}

	// Computes the post-stream density at position P = (xp, y, z). Used by the
	// non-Newtonian kernel to set KS.rho for inflow cells before calling
	// setEquilibrium. Sums the arrival populations like the A-A pattern:
	// rho(P) = sum_i postcoll_{n-1}(i, P - c_i), resolved per layout via
	// postCollValue.
	template <typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void streamingRho(LBM_DATA& SD, LBM_KS& KS, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm, idx z, idx zp)
	{
		dreal rho = postCollValue(SD, dir27::mmm, xp + 1, y + 1, z + 1) + postCollValue(SD, dir27::mmz, xp + 1, y + 1, z)
				  + postCollValue(SD, dir27::mmp, xp + 1, y + 1, z - 1) + postCollValue(SD, dir27::mzm, xp + 1, y, z + 1)
				  + postCollValue(SD, dir27::mzz, xp + 1, y, z) + postCollValue(SD, dir27::mzp, xp + 1, y, z - 1)
				  + postCollValue(SD, dir27::mpm, xp + 1, y - 1, z + 1) + postCollValue(SD, dir27::mpz, xp + 1, y - 1, z)
				  + postCollValue(SD, dir27::mpp, xp + 1, y - 1, z - 1) + postCollValue(SD, dir27::zmm, xp, y + 1, z + 1)
				  + postCollValue(SD, dir27::zmz, xp, y + 1, z) + postCollValue(SD, dir27::zmp, xp, y + 1, z - 1)
				  + postCollValue(SD, dir27::zzm, xp, y, z + 1) + postCollValue(SD, dir27::zzz, xp, y, z)
				  + postCollValue(SD, dir27::zzp, xp, y, z - 1) + postCollValue(SD, dir27::zpm, xp, y - 1, z + 1)
				  + postCollValue(SD, dir27::zpz, xp, y - 1, z) + postCollValue(SD, dir27::zpp, xp, y - 1, z - 1)
				  + postCollValue(SD, dir27::pmm, xm, y + 1, z + 1) + postCollValue(SD, dir27::pmz, xm, y + 1, z)
				  + postCollValue(SD, dir27::pmp, xm, y + 1, z - 1) + postCollValue(SD, dir27::pzm, xm, y, z + 1)
				  + postCollValue(SD, dir27::pzz, xm, y, z) + postCollValue(SD, dir27::pzp, xm, y, z - 1)
				  + postCollValue(SD, dir27::ppm, xm, y - 1, z + 1) + postCollValue(SD, dir27::ppz, xm, y - 1, z)
				  + postCollValue(SD, dir27::ppp, xm, y - 1, z - 1);
		KS.rho = rho;
	}

	// post-stream x-velocity at P = (xm, y, z): momentum-first-moment of the
	// arrival populations (see the A-A pattern's streamingVx).
	template <typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void streamingVx(LBM_DATA& SD, LBM_KS& KS, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm, idx z, idx zp)
	{
		dreal vx = postCollValue(SD, dir27::pmm, xm - 1, y + 1, z + 1) + postCollValue(SD, dir27::pmz, xm - 1, y + 1, z)
				 + postCollValue(SD, dir27::pmp, xm - 1, y + 1, z - 1) + postCollValue(SD, dir27::ppm, xm - 1, y - 1, z + 1)
				 + postCollValue(SD, dir27::ppz, xm - 1, y - 1, z) + postCollValue(SD, dir27::ppp, xm - 1, y - 1, z - 1)
				 + postCollValue(SD, dir27::pzm, xm - 1, y, z + 1) + postCollValue(SD, dir27::pzz, xm - 1, y, z)
				 + postCollValue(SD, dir27::pzp, xm - 1, y, z - 1) - postCollValue(SD, dir27::mmm, xm + 1, y + 1, z + 1)
				 - postCollValue(SD, dir27::mmz, xm + 1, y + 1, z) - postCollValue(SD, dir27::mmp, xm + 1, y + 1, z - 1)
				 - postCollValue(SD, dir27::mzm, xm + 1, y, z + 1) - postCollValue(SD, dir27::mzz, xm + 1, y, z)
				 - postCollValue(SD, dir27::mzp, xm + 1, y, z - 1) - postCollValue(SD, dir27::mpm, xm + 1, y - 1, z + 1)
				 - postCollValue(SD, dir27::mpz, xm + 1, y - 1, z) - postCollValue(SD, dir27::mpp, xm + 1, y - 1, z - 1);
		KS.vx = vx;
	}

	// post-stream y-velocity at P = (xm, y, z)
	template <typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void streamingVy(LBM_DATA& SD, LBM_KS& KS, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm, idx z, idx zp)
	{
		dreal vy = postCollValue(SD, dir27::mpm, xm + 1, y - 1, z + 1) + postCollValue(SD, dir27::mpz, xm + 1, y - 1, z)
				 + postCollValue(SD, dir27::mpp, xm + 1, y - 1, z - 1) + postCollValue(SD, dir27::zpm, xm, y - 1, z + 1)
				 + postCollValue(SD, dir27::zpz, xm, y - 1, z) + postCollValue(SD, dir27::zpp, xm, y - 1, z - 1)
				 + postCollValue(SD, dir27::ppm, xm - 1, y - 1, z + 1) + postCollValue(SD, dir27::ppz, xm - 1, y - 1, z)
				 + postCollValue(SD, dir27::ppp, xm - 1, y - 1, z - 1) - postCollValue(SD, dir27::zmm, xm, y + 1, z + 1)
				 - postCollValue(SD, dir27::zmz, xm, y + 1, z) - postCollValue(SD, dir27::zmp, xm, y + 1, z - 1)
				 - postCollValue(SD, dir27::pmm, xm - 1, y + 1, z + 1) - postCollValue(SD, dir27::pmz, xm - 1, y + 1, z)
				 - postCollValue(SD, dir27::pmp, xm - 1, y + 1, z - 1) - postCollValue(SD, dir27::mmm, xm + 1, y + 1, z + 1)
				 - postCollValue(SD, dir27::mmz, xm + 1, y + 1, z) - postCollValue(SD, dir27::mmp, xm + 1, y + 1, z - 1);
		KS.vy = vy;
	}

	// post-stream z-velocity at P = (xm, y, z)
	template <typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void streamingVz(LBM_DATA& SD, LBM_KS& KS, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm, idx z, idx zp)
	{
		dreal vz = postCollValue(SD, dir27::mmp, xm + 1, y + 1, z - 1) + postCollValue(SD, dir27::pmp, xm - 1, y + 1, z - 1)
				 + postCollValue(SD, dir27::zmp, xm, y + 1, z - 1) + postCollValue(SD, dir27::pzp, xm - 1, y, z - 1)
				 + postCollValue(SD, dir27::zzp, xm, y, z - 1) + postCollValue(SD, dir27::mzp, xm + 1, y, z - 1)
				 + postCollValue(SD, dir27::ppp, xm - 1, y - 1, z - 1) + postCollValue(SD, dir27::zpp, xm, y - 1, z - 1)
				 + postCollValue(SD, dir27::mpp, xm + 1, y - 1, z - 1) - postCollValue(SD, dir27::mmm, xm + 1, y + 1, z + 1)
				 - postCollValue(SD, dir27::pmm, xm - 1, y + 1, z + 1) - postCollValue(SD, dir27::zmm, xm, y + 1, z + 1)
				 - postCollValue(SD, dir27::pzm, xm - 1, y, z + 1) - postCollValue(SD, dir27::zzm, xm, y, z + 1)
				 - postCollValue(SD, dir27::mzm, xm + 1, y, z + 1) - postCollValue(SD, dir27::ppm, xm - 1, y - 1, z + 1)
				 - postCollValue(SD, dir27::zpm, xm, y - 1, z + 1) - postCollValue(SD, dir27::mpm, xm + 1, y - 1, z + 1);
		KS.vz = vz;
	}

	// Outflow-pass gather (a separate kernel launched before the main one for
	// deterministic outflow handling - the outflow pass is a type of
	// processing, not a BC type): reconstructs the pre-collision populations
	// at the translated A-B pull sites of the outflow cell:
	// postcoll_{n-1}(i) at site t_i = (anchor, tangential -c_i), where the
	// anchor is the fluid-side neighbor column one cell inward.
	// FACE is a compile-time template parameter, so the per-direction
	// components and site offsets fold to constants. Tangential coordinates
	// select among the kernel's already-wrapped neighbors (kernelInitIndices):
	// interior rows read the raw neighbor, periodic-seam rows the wrapped
	// neighbor, distributed axes the halo; boundary rows on non-periodic
	// tangential axes are unreachable (ghost-layer idiom stamps walls/
	// GEO_NOTHING there).
	template <int FACE, typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void streamingOutflowImpl(LBM_DATA& SD, LBM_KS& KS, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm, idx z, idx zp)
	{
		constexpr int axis = (FACE & (bc_face::XP | bc_face::XM)) ? 0
						   : (FACE & (bc_face::YP | bc_face::YM)) ? 1
																  : 2;	// normal axis: 0 = x, 1 = y, 2 = z
		// anchor column: the fluid-side neighbor one cell inward (opposite the outward normal)
		const idx anchor = (FACE == bc_face::XP) ? xm
						 : (FACE == bc_face::XM) ? xp
						 : (FACE == bc_face::YP) ? ym
						 : (FACE == bc_face::YM) ? yp
						 : (FACE == bc_face::ZP) ? zm
												 : zp;
		for (int i = 0; i < 27; i++) {
			// placement shift + slot identical to postCollValue, but folded into the
			// net tangential offset used for the neighbor select: that engages on
			// the FINAL storage coordinate (logical site + shift), not on the
			// logical gather site
			const idx px = dir27_cx(i) > 0 ? 1 : 0;
			const idx py = dir27_cy(i) > 0 ? 1 : 0;
			const idx pz = dir27_cz(i) > 0 ? 1 : 0;
			const int slot = SD.even_iter ? opposite_direction(i) : i;
			idx wx;
			idx wy;
			idx wz;
			if constexpr (axis == 0) {
				wx = anchor + px;
				wy = dir27_cy(i) - py > 0 ? ym : (dir27_cy(i) - py < 0 ? yp : y);
				wz = dir27_cz(i) - pz > 0 ? zm : (dir27_cz(i) - pz < 0 ? zp : z);
			}
			else if constexpr (axis == 1) {
				wx = dir27_cx(i) - px > 0 ? xm : (dir27_cx(i) - px < 0 ? xp : x);
				wy = anchor + py;
				wz = dir27_cz(i) - pz > 0 ? zm : (dir27_cz(i) - pz < 0 ? zp : z);
			}
			else {
				wx = dir27_cx(i) - px > 0 ? xm : (dir27_cx(i) - px < 0 ? xp : x);
				wy = dir27_cy(i) - py > 0 ? ym : (dir27_cy(i) - py < 0 ? yp : y);
				wz = anchor + pz;
			}
			KS.f[i] = TNL::Backend::ldg(SD.df(df_cur, slot, wx, wy, wz));
		}
	}

	template <typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void
	streamingOutflow(LBM_DATA& SD, LBM_KS& KS, int face, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm, idx z, idx zp)
	{
		switch (face) {
			case bc_face::XP:
				streamingOutflowImpl<bc_face::XP>(SD, KS, xm, x, xp, ym, y, yp, zm, z, zp);
				break;
			case bc_face::XM:
				streamingOutflowImpl<bc_face::XM>(SD, KS, xm, x, xp, ym, y, yp, zm, z, zp);
				break;
			case bc_face::YP:
				streamingOutflowImpl<bc_face::YP>(SD, KS, xm, x, xp, ym, y, yp, zm, z, zp);
				break;
			case bc_face::YM:
				streamingOutflowImpl<bc_face::YM>(SD, KS, xm, x, xp, ym, y, yp, zm, z, zp);
				break;
			case bc_face::ZP:
				streamingOutflowImpl<bc_face::ZP>(SD, KS, xm, x, xp, ym, y, yp, zm, z, zp);
				break;
			default:
				streamingOutflowImpl<bc_face::ZM>(SD, KS, xm, x, xp, ym, y, yp, zm, z, zp);
				break;
		}
	}

	// interpolated outflow (Geier 2015) for an arbitrary face: the population
	// moving against the outward normal blends postcoll_{n-1} from the anchor
	// column with the outflow cell's own postcoll, the perpendicular population
	// takes the cell's own postcoll, the outward-moving population comes from
	// the anchor column; all of it is previous-launch state finalized before
	// the pass runs; the tangential coordinates select among the wrapped
	// neighbors like in the plain gather above.
	template <int FACE, typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void
	streamingOutflowInterpImpl(LBM_DATA& SD, LBM_KS& KS, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm, idx z, idx zp)
	{
		constexpr int axis = (FACE & (bc_face::XP | bc_face::XM)) ? 0 : (FACE & (bc_face::YP | bc_face::YM)) ? 1 : 2;
		constexpr int out_sign = (FACE & (bc_face::XM | bc_face::YM | bc_face::ZM)) ? -1 : 1;
		// anchor column: the fluid-side neighbor one cell inward (opposite the outward normal)
		const idx anchor = (FACE == bc_face::XP) ? xm
						 : (FACE == bc_face::XM) ? xp
						 : (FACE == bc_face::YP) ? ym
						 : (FACE == bc_face::YM) ? yp
						 : (FACE == bc_face::ZP) ? zm
												 : zp;
		for (int i = 0; i < 27; i++) {
			const int cn = (axis == 0) ? dir27_cx(i) : (axis == 1) ? dir27_cy(i) : dir27_cz(i);	 // normal component of c_i
			// placement shift + slot identical to postCollValue, but folded into the
			// net tangential offset used for the neighbor select: that engages on
			// the FINAL storage coordinate (logical site + shift), not on the
			// logical gather site
			const idx px = dir27_cx(i) > 0 ? 1 : 0;
			const idx py = dir27_cy(i) > 0 ? 1 : 0;
			const idx pz = dir27_cz(i) > 0 ? 1 : 0;
			const int slot = SD.even_iter ? opposite_direction(i) : i;
			// value at the anchor column and at the own column, tangential -c offsets
			idx nx;
			idx ny;
			idx nz;
			idx ox;
			idx oy;
			idx oz;
			if constexpr (axis == 0) {
				nx = anchor + px;
				ny = dir27_cy(i) - py > 0 ? ym : (dir27_cy(i) - py < 0 ? yp : y);
				nz = dir27_cz(i) - pz > 0 ? zm : (dir27_cz(i) - pz < 0 ? zp : z);
				ox = x + px;
				oy = ny;
				oz = nz;
			}
			else if constexpr (axis == 1) {
				nx = dir27_cx(i) - px > 0 ? xm : (dir27_cx(i) - px < 0 ? xp : x);
				ny = anchor + py;
				nz = dir27_cz(i) - pz > 0 ? zm : (dir27_cz(i) - pz < 0 ? zp : z);
				ox = nx;
				oy = y + py;
				oz = nz;
			}
			else {
				nx = dir27_cx(i) - px > 0 ? xm : (dir27_cx(i) - px < 0 ? xp : x);
				ny = dir27_cy(i) - py > 0 ? ym : (dir27_cy(i) - py < 0 ? yp : y);
				nz = anchor + pz;
				ox = nx;
				oy = ny;
				oz = z + pz;
			}
			if (cn == out_sign) {
				KS.f[i] = TNL::Backend::ldg(SD.df(df_cur, slot, nx, ny, nz));
			}
			else if (cn == 0) {
				KS.f[i] = TNL::Backend::ldg(SD.df(df_cur, slot, ox, oy, oz));
			}
			else {
				// the outflowInterpBlend form (velocity neglected)
				constexpr dreal SpeedOfSound = 0.5773502691896257;
				KS.f[i] = lbm_fma_rn(
					SpeedOfSound,
					TNL::Backend::ldg(SD.df(df_cur, slot, nx, ny, nz)),
					(1 - SpeedOfSound) * TNL::Backend::ldg(SD.df(df_cur, slot, ox, oy, oz))
				);
			}
		}
	}

	template <typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void
	streamingOutflowInterp(LBM_DATA& SD, LBM_KS& KS, int face, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm, idx z, idx zp)
	{
		switch (face) {
			case bc_face::XP:
				streamingOutflowInterpImpl<bc_face::XP>(SD, KS, xm, x, xp, ym, y, yp, zm, z, zp);
				break;
			case bc_face::XM:
				streamingOutflowInterpImpl<bc_face::XM>(SD, KS, xm, x, xp, ym, y, yp, zm, z, zp);
				break;
			case bc_face::YP:
				streamingOutflowInterpImpl<bc_face::YP>(SD, KS, xm, x, xp, ym, y, yp, zm, z, zp);
				break;
			case bc_face::YM:
				streamingOutflowInterpImpl<bc_face::YM>(SD, KS, xm, x, xp, ym, y, yp, zm, z, zp);
				break;
			case bc_face::ZP:
				streamingOutflowInterpImpl<bc_face::ZP>(SD, KS, xm, x, xp, ym, y, yp, zm, z, zp);
				break;
			default:
				streamingOutflowInterpImpl<bc_face::ZM>(SD, KS, xm, x, xp, ym, y, yp, zm, z, zp);
				break;
		}
	}
};

template <typename TRAITS>
inline constexpr bool is_ESO_TWIST_v<D3Q27_STREAMING_ESO_TWIST<TRAITS>> = true;
template <typename TRAITS>
inline constexpr bool is_esoteric_in_place_v<D3Q27_STREAMING_ESO_TWIST<TRAITS>> = true;
template <typename TRAITS>
inline constexpr bool requires_ghost_layer_v<D3Q27_STREAMING_ESO_TWIST<TRAITS>> = true;
