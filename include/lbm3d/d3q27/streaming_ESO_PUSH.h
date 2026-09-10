#pragma once

#include "lbm3d/defs.h"
#include "lbm_common/rounding.h"

// Esoteric Push (Lehmann 2022): single in-place DF array. Directions are
// processed as opposite pairs (head h = the odd-numbered slot, tail
// t = opposite_direction(h)) with the head's direction vector c_h; the rest
// population is always local.
// - even_iter == false: each head is pushed into its own slot at the own
//   site, each tail into the tail slot shifted upstream by -c_h; populations
//   are read from the tail slot upstream/the head slot at the own site
//   (every memory location is read and written by the same thread, so no
//   race).
// - even_iter == true: the parity counterpart with swapped push/pull roles.
// The pre-collision populations read here are identical to the A-B pull
// scheme's at every launch, provided the initial DF field is placed as the
// streamed push-scheme state: slot (opp h, s) = eq_h(s) for heads, natural
// for tails and rest.
//
// LIMITATION (same as the A-A pattern): the non-Newtonian methods
// streamingRho / streamingVx / streamingVy / streamingVz dereference slots
// at TWO-step neighbor offsets. They are safe only from a separate,
// read-only-on-df_cur kernel launch (computeNonNewtonianKernels), never from
// the main LBM kernel where df_cur is read-write.
template <typename TRAITS>
struct D3Q27_STREAMING_ESO_PUSH
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
			// heads pushed to their own slot at the own site, tails to the
			// tail slot shifted by -c_h
			SD.df(df_cur, pzz, x, y, z) = KS.f[pzz];
			SD.df(df_cur, mzz, xm, y, z) = KS.f[mzz];

			SD.df(df_cur, zpz, x, y, z) = KS.f[zpz];
			SD.df(df_cur, zmz, x, ym, z) = KS.f[zmz];

			SD.df(df_cur, zzp, x, y, z) = KS.f[zzp];
			SD.df(df_cur, zzm, x, y, zm) = KS.f[zzm];

			SD.df(df_cur, ppz, x, y, z) = KS.f[ppz];
			SD.df(df_cur, mmz, xm, ym, z) = KS.f[mmz];

			SD.df(df_cur, pmz, x, y, z) = KS.f[pmz];
			SD.df(df_cur, mpz, xm, yp, z) = KS.f[mpz];

			SD.df(df_cur, pzp, x, y, z) = KS.f[pzp];
			SD.df(df_cur, mzm, xm, y, zm) = KS.f[mzm];

			SD.df(df_cur, pzm, x, y, z) = KS.f[pzm];
			SD.df(df_cur, mzp, xm, y, zp) = KS.f[mzp];

			SD.df(df_cur, zpp, x, y, z) = KS.f[zpp];
			SD.df(df_cur, zmm, x, ym, zm) = KS.f[zmm];

			SD.df(df_cur, zpm, x, y, z) = KS.f[zpm];
			SD.df(df_cur, zmp, x, ym, zp) = KS.f[zmp];

			SD.df(df_cur, ppp, x, y, z) = KS.f[ppp];
			SD.df(df_cur, mmm, xm, ym, zm) = KS.f[mmm];

			SD.df(df_cur, ppm, x, y, z) = KS.f[ppm];
			SD.df(df_cur, mmp, xm, ym, zp) = KS.f[mmp];

			SD.df(df_cur, pmp, x, y, z) = KS.f[pmp];
			SD.df(df_cur, mpm, xm, yp, zm) = KS.f[mpm];

			SD.df(df_cur, pmm, x, y, z) = KS.f[pmm];
			SD.df(df_cur, mpp, xm, yp, zp) = KS.f[mpp];

			SD.df(df_cur, zzz, x, y, z) = KS.f[zzz];
		}
		else {
			SD.df(df_cur, pzz, xm, y, z) = KS.f[mzz];
			SD.df(df_cur, mzz, x, y, z) = KS.f[pzz];

			SD.df(df_cur, zpz, x, ym, z) = KS.f[zmz];
			SD.df(df_cur, zmz, x, y, z) = KS.f[zpz];

			SD.df(df_cur, zzp, x, y, zm) = KS.f[zzm];
			SD.df(df_cur, zzm, x, y, z) = KS.f[zzp];

			SD.df(df_cur, ppz, xm, ym, z) = KS.f[mmz];
			SD.df(df_cur, mmz, x, y, z) = KS.f[ppz];

			SD.df(df_cur, pmz, xm, yp, z) = KS.f[mpz];
			SD.df(df_cur, mpz, x, y, z) = KS.f[pmz];

			SD.df(df_cur, pzp, xm, y, zm) = KS.f[mzm];
			SD.df(df_cur, mzm, x, y, z) = KS.f[pzp];

			SD.df(df_cur, pzm, xm, y, zp) = KS.f[mzp];
			SD.df(df_cur, mzp, x, y, z) = KS.f[pzm];

			SD.df(df_cur, zpp, x, ym, zm) = KS.f[zmm];
			SD.df(df_cur, zmm, x, y, z) = KS.f[zpp];

			SD.df(df_cur, zpm, x, ym, zp) = KS.f[zmp];
			SD.df(df_cur, zmp, x, y, z) = KS.f[zpm];

			SD.df(df_cur, ppp, xm, ym, zm) = KS.f[mmm];
			SD.df(df_cur, mmm, x, y, z) = KS.f[ppp];

			SD.df(df_cur, ppm, xm, ym, zp) = KS.f[mmp];
			SD.df(df_cur, mmp, x, y, z) = KS.f[ppm];

			SD.df(df_cur, pmp, xm, yp, zm) = KS.f[mpm];
			SD.df(df_cur, mpm, x, y, z) = KS.f[pmp];

			SD.df(df_cur, pmm, xm, yp, zp) = KS.f[mpp];
			SD.df(df_cur, mpp, x, y, z) = KS.f[pmm];

			SD.df(df_cur, zzz, x, y, z) = KS.f[zzz];
		}
	}

	template <typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void streaming(LBM_DATA& SD, LBM_KS& KS, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm, idx z, idx zp)
	{
		if (! SD.even_iter) {
			// heads read from the tail slot one step upstream (-c_h), tails
			// from the head slot at the own site
			KS.f[pzz] = TNL::Backend::ldg(SD.df(df_cur, mzz, xm, y, z));
			KS.f[mzz] = TNL::Backend::ldg(SD.df(df_cur, pzz, x, y, z));

			KS.f[zpz] = TNL::Backend::ldg(SD.df(df_cur, zmz, x, ym, z));
			KS.f[zmz] = TNL::Backend::ldg(SD.df(df_cur, zpz, x, y, z));

			KS.f[zzp] = TNL::Backend::ldg(SD.df(df_cur, zzm, x, y, zm));
			KS.f[zzm] = TNL::Backend::ldg(SD.df(df_cur, zzp, x, y, z));

			KS.f[ppz] = TNL::Backend::ldg(SD.df(df_cur, mmz, xm, ym, z));
			KS.f[mmz] = TNL::Backend::ldg(SD.df(df_cur, ppz, x, y, z));

			KS.f[pmz] = TNL::Backend::ldg(SD.df(df_cur, mpz, xm, yp, z));
			KS.f[mpz] = TNL::Backend::ldg(SD.df(df_cur, pmz, x, y, z));

			KS.f[pzp] = TNL::Backend::ldg(SD.df(df_cur, mzm, xm, y, zm));
			KS.f[mzm] = TNL::Backend::ldg(SD.df(df_cur, pzp, x, y, z));

			KS.f[pzm] = TNL::Backend::ldg(SD.df(df_cur, mzp, xm, y, zp));
			KS.f[mzp] = TNL::Backend::ldg(SD.df(df_cur, pzm, x, y, z));

			KS.f[zpp] = TNL::Backend::ldg(SD.df(df_cur, zmm, x, ym, zm));
			KS.f[zmm] = TNL::Backend::ldg(SD.df(df_cur, zpp, x, y, z));

			KS.f[zpm] = TNL::Backend::ldg(SD.df(df_cur, zmp, x, ym, zp));
			KS.f[zmp] = TNL::Backend::ldg(SD.df(df_cur, zpm, x, y, z));

			KS.f[ppp] = TNL::Backend::ldg(SD.df(df_cur, mmm, xm, ym, zm));
			KS.f[mmm] = TNL::Backend::ldg(SD.df(df_cur, ppp, x, y, z));

			KS.f[ppm] = TNL::Backend::ldg(SD.df(df_cur, mmp, xm, ym, zp));
			KS.f[mmp] = TNL::Backend::ldg(SD.df(df_cur, ppm, x, y, z));

			KS.f[pmp] = TNL::Backend::ldg(SD.df(df_cur, mpm, xm, yp, zm));
			KS.f[mpm] = TNL::Backend::ldg(SD.df(df_cur, pmp, x, y, z));

			KS.f[pmm] = TNL::Backend::ldg(SD.df(df_cur, mpp, xm, yp, zp));
			KS.f[mpp] = TNL::Backend::ldg(SD.df(df_cur, pmm, x, y, z));

			KS.f[zzz] = TNL::Backend::ldg(SD.df(df_cur, zzz, x, y, z));
		}
		else {
			KS.f[pzz] = TNL::Backend::ldg(SD.df(df_cur, pzz, xm, y, z));
			KS.f[mzz] = TNL::Backend::ldg(SD.df(df_cur, mzz, x, y, z));

			KS.f[zpz] = TNL::Backend::ldg(SD.df(df_cur, zpz, x, ym, z));
			KS.f[zmz] = TNL::Backend::ldg(SD.df(df_cur, zmz, x, y, z));

			KS.f[zzp] = TNL::Backend::ldg(SD.df(df_cur, zzp, x, y, zm));
			KS.f[zzm] = TNL::Backend::ldg(SD.df(df_cur, zzm, x, y, z));

			KS.f[ppz] = TNL::Backend::ldg(SD.df(df_cur, ppz, xm, ym, z));
			KS.f[mmz] = TNL::Backend::ldg(SD.df(df_cur, mmz, x, y, z));

			KS.f[pmz] = TNL::Backend::ldg(SD.df(df_cur, pmz, xm, yp, z));
			KS.f[mpz] = TNL::Backend::ldg(SD.df(df_cur, mpz, x, y, z));

			KS.f[pzp] = TNL::Backend::ldg(SD.df(df_cur, pzp, xm, y, zm));
			KS.f[mzm] = TNL::Backend::ldg(SD.df(df_cur, mzm, x, y, z));

			KS.f[pzm] = TNL::Backend::ldg(SD.df(df_cur, pzm, xm, y, zp));
			KS.f[mzp] = TNL::Backend::ldg(SD.df(df_cur, mzp, x, y, z));

			KS.f[zpp] = TNL::Backend::ldg(SD.df(df_cur, zpp, x, ym, zm));
			KS.f[zmm] = TNL::Backend::ldg(SD.df(df_cur, zmm, x, y, z));

			KS.f[zpm] = TNL::Backend::ldg(SD.df(df_cur, zpm, x, ym, zp));
			KS.f[zmp] = TNL::Backend::ldg(SD.df(df_cur, zmp, x, y, z));

			KS.f[ppp] = TNL::Backend::ldg(SD.df(df_cur, ppp, xm, ym, zm));
			KS.f[mmm] = TNL::Backend::ldg(SD.df(df_cur, mmm, x, y, z));

			KS.f[ppm] = TNL::Backend::ldg(SD.df(df_cur, ppm, xm, ym, zp));
			KS.f[mmp] = TNL::Backend::ldg(SD.df(df_cur, mmp, x, y, z));

			KS.f[pmp] = TNL::Backend::ldg(SD.df(df_cur, pmp, xm, yp, zm));
			KS.f[mpm] = TNL::Backend::ldg(SD.df(df_cur, mpm, x, y, z));

			KS.f[pmm] = TNL::Backend::ldg(SD.df(df_cur, pmm, xm, yp, zp));
			KS.f[mpp] = TNL::Backend::ldg(SD.df(df_cur, mpp, x, y, z));

			KS.f[zzz] = TNL::Backend::ldg(SD.df(df_cur, zzz, x, y, z));
		}
	}

	// streaming with the bounce-back rule applied: the identity write-back of
	// this pattern preserves the implicit bounce-back of the esoteric schemes
	// (a swapped wall cell writes back exactly what it read)
	template <typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void streamingBounceBack(LBM_DATA& SD, LBM_KS& KS, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm, idx z, idx zp)
	{
		streaming(SD, KS, xm, x, xp, ym, y, yp, zm, z, zp);
		TNL::swap(KS.f[mmm], KS.f[ppp]);
		TNL::swap(KS.f[mmz], KS.f[ppz]);
		TNL::swap(KS.f[mmp], KS.f[ppm]);
		TNL::swap(KS.f[mzm], KS.f[pzp]);
		TNL::swap(KS.f[mzz], KS.f[pzz]);
		TNL::swap(KS.f[mzp], KS.f[pzm]);
		TNL::swap(KS.f[mpm], KS.f[pmp]);
		TNL::swap(KS.f[mpz], KS.f[pmz]);
		TNL::swap(KS.f[mpp], KS.f[pmm]);
		TNL::swap(KS.f[zmm], KS.f[zpp]);
		TNL::swap(KS.f[zzm], KS.f[zzp]);
		TNL::swap(KS.f[zmz], KS.f[zpz]);
		TNL::swap(KS.f[zmp], KS.f[zpm]);
	}

	// the post-collision population of direction i authored at site w, given
	// the layout finalized by the previous launch and the current parity:
	// - even_iter == true  (post-phase-1 layout): slot (i, w + [tail] c_i)
	// - even_iter == false (post-phase-2 layout): slot (opp i, w + [tail] c_i)
	template <typename LBM_DATA>
	__cuda_callable__ static dreal postCollValue(LBM_DATA& SD, int i, idx wx, idx wy, idx wz)
	{
		const idx ox = is_pair_head(i) ? 0 : dir27_cx(i);
		const idx oy = is_pair_head(i) ? 0 : dir27_cy(i);
		const idx oz = is_pair_head(i) ? 0 : dir27_cz(i);
		const int slot = SD.even_iter ? i : opposite_direction(i);
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
		dreal rho = postCollValue(SD, mmm, xp + 1, y + 1, z + 1) + postCollValue(SD, mmz, xp + 1, y + 1, z)
				  + postCollValue(SD, mmp, xp + 1, y + 1, z - 1) + postCollValue(SD, mzm, xp + 1, y, z + 1) + postCollValue(SD, mzz, xp + 1, y, z)
				  + postCollValue(SD, mzp, xp + 1, y, z - 1) + postCollValue(SD, mpm, xp + 1, y - 1, z + 1) + postCollValue(SD, mpz, xp + 1, y - 1, z)
				  + postCollValue(SD, mpp, xp + 1, y - 1, z - 1) + postCollValue(SD, zmm, xp, y + 1, z + 1) + postCollValue(SD, zmz, xp, y + 1, z)
				  + postCollValue(SD, zmp, xp, y + 1, z - 1) + postCollValue(SD, zzm, xp, y, z + 1) + postCollValue(SD, zzz, xp, y, z)
				  + postCollValue(SD, zzp, xp, y, z - 1) + postCollValue(SD, zpm, xp, y - 1, z + 1) + postCollValue(SD, zpz, xp, y - 1, z)
				  + postCollValue(SD, zpp, xp, y - 1, z - 1) + postCollValue(SD, pmm, xm, y + 1, z + 1) + postCollValue(SD, pmz, xm, y + 1, z)
				  + postCollValue(SD, pmp, xm, y + 1, z - 1) + postCollValue(SD, pzm, xm, y, z + 1) + postCollValue(SD, pzz, xm, y, z)
				  + postCollValue(SD, pzp, xm, y, z - 1) + postCollValue(SD, ppm, xm, y - 1, z + 1) + postCollValue(SD, ppz, xm, y - 1, z)
				  + postCollValue(SD, ppp, xm, y - 1, z - 1);
		KS.rho = rho;
	}

	// post-stream x-velocity at P = (xm, y, z): momentum-first-moment of the
	// arrival populations (see the A-A pattern's streamingVx).
	template <typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void streamingVx(LBM_DATA& SD, LBM_KS& KS, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm, idx z, idx zp)
	{
		dreal vx = postCollValue(SD, pmm, xm - 1, y + 1, z + 1) + postCollValue(SD, pmz, xm - 1, y + 1, z)
				 + postCollValue(SD, pmp, xm - 1, y + 1, z - 1) + postCollValue(SD, ppm, xm - 1, y - 1, z + 1)
				 + postCollValue(SD, ppz, xm - 1, y - 1, z) + postCollValue(SD, ppp, xm - 1, y - 1, z - 1) + postCollValue(SD, pzm, xm - 1, y, z + 1)
				 + postCollValue(SD, pzz, xm - 1, y, z) + postCollValue(SD, pzp, xm - 1, y, z - 1) - postCollValue(SD, mmm, xm + 1, y + 1, z + 1)
				 - postCollValue(SD, mmz, xm + 1, y + 1, z) - postCollValue(SD, mmp, xm + 1, y + 1, z - 1) - postCollValue(SD, mzm, xm + 1, y, z + 1)
				 - postCollValue(SD, mzz, xm + 1, y, z) - postCollValue(SD, mzp, xm + 1, y, z - 1) - postCollValue(SD, mpm, xm + 1, y - 1, z + 1)
				 - postCollValue(SD, mpz, xm + 1, y - 1, z) - postCollValue(SD, mpp, xm + 1, y - 1, z - 1);
		KS.vx = vx;
	}

	// post-stream y-velocity at P = (xm, y, z)
	template <typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void streamingVy(LBM_DATA& SD, LBM_KS& KS, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm, idx z, idx zp)
	{
		dreal vy = postCollValue(SD, mpm, xm + 1, y - 1, z + 1) + postCollValue(SD, mpz, xm + 1, y - 1, z)
				 + postCollValue(SD, mpp, xm + 1, y - 1, z - 1) + postCollValue(SD, zpm, xm, y - 1, z + 1) + postCollValue(SD, zpz, xm, y - 1, z)
				 + postCollValue(SD, zpp, xm, y - 1, z - 1) + postCollValue(SD, ppm, xm - 1, y - 1, z + 1) + postCollValue(SD, ppz, xm - 1, y - 1, z)
				 + postCollValue(SD, ppp, xm - 1, y - 1, z - 1) - postCollValue(SD, zmm, xm, y + 1, z + 1) - postCollValue(SD, zmz, xm, y + 1, z)
				 - postCollValue(SD, zmp, xm, y + 1, z - 1) - postCollValue(SD, pmm, xm - 1, y + 1, z + 1) - postCollValue(SD, pmz, xm - 1, y + 1, z)
				 - postCollValue(SD, pmp, xm - 1, y + 1, z - 1) - postCollValue(SD, mmm, xm + 1, y + 1, z + 1)
				 - postCollValue(SD, mmz, xm + 1, y + 1, z) - postCollValue(SD, mmp, xm + 1, y + 1, z - 1);
		KS.vy = vy;
	}

	// post-stream z-velocity at P = (xm, y, z)
	template <typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void streamingVz(LBM_DATA& SD, LBM_KS& KS, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm, idx z, idx zp)
	{
		dreal vz = postCollValue(SD, mmp, xm + 1, y + 1, z - 1) + postCollValue(SD, pmp, xm - 1, y + 1, z - 1)
				 + postCollValue(SD, zmp, xm, y + 1, z - 1) + postCollValue(SD, pzp, xm - 1, y, z - 1) + postCollValue(SD, zzp, xm, y, z - 1)
				 + postCollValue(SD, mzp, xm + 1, y, z - 1) + postCollValue(SD, ppp, xm - 1, y - 1, z - 1) + postCollValue(SD, zpp, xm, y - 1, z - 1)
				 + postCollValue(SD, mpp, xm + 1, y - 1, z - 1) - postCollValue(SD, mmm, xm + 1, y + 1, z + 1)
				 - postCollValue(SD, pmm, xm - 1, y + 1, z + 1) - postCollValue(SD, zmm, xm, y + 1, z + 1) - postCollValue(SD, pzm, xm - 1, y, z + 1)
				 - postCollValue(SD, zzm, xm, y, z + 1) - postCollValue(SD, mzm, xm + 1, y, z + 1) - postCollValue(SD, ppm, xm - 1, y - 1, z + 1)
				 - postCollValue(SD, zpm, xm, y - 1, z + 1) - postCollValue(SD, mpm, xm + 1, y - 1, z + 1);
		KS.vz = vz;
	}

	// Outflow-pass gather (a separate kernel launched before the main one for
	// deterministic outflow handling - the outflow pass is a type of
	// processing, not a BC type): reconstructs the pre-collision populations
	// at the translated A-B pull sites of the outflow cell:
	// postcoll_{n-1}(i) at site t_i = (anchor, tangential -c_i), where the
	// anchor is the fluid-side neighbor column one cell inward.
	// FACE is a compile-time template parameter, so the per-direction
	// components and site offsets fold to constants.
	template <int FACE, typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void streamingOutflowImpl(LBM_DATA& SD, LBM_KS& KS, idx anchor, idx x, idx y, idx z)
	{
		constexpr int axis = (FACE & (bc_face::XP | bc_face::XM)) ? 0
						   : (FACE & (bc_face::YP | bc_face::YM)) ? 1
																  : 2;	// normal axis: 0 = x, 1 = y, 2 = z
		for (int i = 0; i < 27; i++) {
			idx wx, wy, wz;
			if constexpr (axis == 0) {
				wx = anchor;
				wy = y - dir27_cy(i);
				wz = z - dir27_cz(i);
			}
			else if constexpr (axis == 1) {
				wx = x - dir27_cx(i);
				wy = anchor;
				wz = z - dir27_cz(i);
			}
			else {
				wx = x - dir27_cx(i);
				wy = y - dir27_cy(i);
				wz = anchor;
			}
			KS.f[i] = postCollValue(SD, i, wx, wy, wz);
		}
	}

	template <typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void
	streamingOutflow(LBM_DATA& SD, LBM_KS& KS, int face, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm, idx z, idx zp)
	{
		switch (face) {
			case bc_face::XP:
				streamingOutflowImpl<bc_face::XP>(SD, KS, xm, x, y, z);
				break;
			case bc_face::XM:
				streamingOutflowImpl<bc_face::XM>(SD, KS, xp, x, y, z);
				break;
			case bc_face::YP:
				streamingOutflowImpl<bc_face::YP>(SD, KS, ym, x, y, z);
				break;
			case bc_face::YM:
				streamingOutflowImpl<bc_face::YM>(SD, KS, yp, x, y, z);
				break;
			case bc_face::ZP:
				streamingOutflowImpl<bc_face::ZP>(SD, KS, zm, x, y, z);
				break;
			default:
				streamingOutflowImpl<bc_face::ZM>(SD, KS, zp, x, y, z);
				break;
		}
	}

	// interpolated-outflow blend in the pinned lbm_fma_rn form:
	// the first site delivers the anchor-column postcoll (weight cs),
	// the second site the own-column postcoll (weight 1-cs)
	template <typename LBM_DATA>
	__cuda_callable__ static dreal outflowInterpBlend(LBM_DATA& SD, int i, idx ax, idx ay, idx az, idx bx, idx by, idx bz)
	{
		// NOTE: velocity is neglected (for the case velocity << speed of sound)
		constexpr dreal SpeedOfSound = 0.5773502691896257;
		return lbm_fma_rn(SpeedOfSound, postCollValue(SD, i, ax, ay, az), (1 - SpeedOfSound) * postCollValue(SD, i, bx, by, bz));
	}

	// interpolated outflow (Geier 2015) for an arbitrary face: the population
	// moving against the outward normal blends postcoll_{n-1} from the anchor
	// column with the outflow cell's own postcoll, the perpendicular population
	// takes the cell's own postcoll, the outward-moving population comes from
	// the anchor column; all of it is previous-launch state finalized before
	// the pass runs.
	template <int FACE, typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void streamingOutflowInterpImpl(LBM_DATA& SD, LBM_KS& KS, idx anchor, idx x, idx y, idx z)
	{
		constexpr int axis = (FACE & (bc_face::XP | bc_face::XM)) ? 0 : (FACE & (bc_face::YP | bc_face::YM)) ? 1 : 2;
		constexpr int out_sign = (FACE & (bc_face::XM | bc_face::YM | bc_face::ZM)) ? -1 : 1;
		for (int i = 0; i < 27; i++) {
			const int cn = (axis == 0) ? dir27_cx(i) : (axis == 1) ? dir27_cy(i) : dir27_cz(i);	 // normal component of c_i
			// value at the anchor column and at the own column, tangential -c offsets
			idx nx, ny, nz, ox, oy, oz;
			if constexpr (axis == 0) {
				nx = anchor;
				ny = y - dir27_cy(i);
				nz = z - dir27_cz(i);
				ox = x;
				oy = y - dir27_cy(i);
				oz = z - dir27_cz(i);
			}
			else if constexpr (axis == 1) {
				nx = x - dir27_cx(i);
				ny = anchor;
				nz = z - dir27_cz(i);
				ox = x - dir27_cx(i);
				oy = y;
				oz = z - dir27_cz(i);
			}
			else {
				nx = x - dir27_cx(i);
				ny = y - dir27_cy(i);
				nz = anchor;
				ox = x - dir27_cx(i);
				oy = y - dir27_cy(i);
				oz = z;
			}
			if (cn == out_sign)
				KS.f[i] = postCollValue(SD, i, nx, ny, nz);
			else if (cn == 0)
				KS.f[i] = postCollValue(SD, i, ox, oy, oz);
			else
				KS.f[i] = outflowInterpBlend(SD, i, nx, ny, nz, ox, oy, oz);
		}
	}

	template <typename LBM_DATA, typename LBM_KS>
	__cuda_callable__ static void
	streamingOutflowInterp(LBM_DATA& SD, LBM_KS& KS, int face, idx xm, idx x, idx xp, idx ym, idx y, idx yp, idx zm, idx z, idx zp)
	{
		switch (face) {
			case bc_face::XP:
				streamingOutflowInterpImpl<bc_face::XP>(SD, KS, xm, x, y, z);
				break;
			case bc_face::XM:
				streamingOutflowInterpImpl<bc_face::XM>(SD, KS, xp, x, y, z);
				break;
			case bc_face::YP:
				streamingOutflowInterpImpl<bc_face::YP>(SD, KS, ym, x, y, z);
				break;
			case bc_face::YM:
				streamingOutflowInterpImpl<bc_face::YM>(SD, KS, yp, x, y, z);
				break;
			case bc_face::ZP:
				streamingOutflowInterpImpl<bc_face::ZP>(SD, KS, zm, x, y, z);
				break;
			default:
				streamingOutflowInterpImpl<bc_face::ZM>(SD, KS, zp, x, y, z);
				break;
		}
	}

	// DF halo exchange descriptors for slot dir on lattice axis a (0=x, 1=y,
	// 2=z). The fresh cross-boundary populations sit one plane beyond the block
	// boundary on the side opposite to their PARITY-dependent storage layout:
	// - after a phase-1 launch (even_iter == false): canonical mask,
	//   shift [tail]
	// - after a phase-2 launch (even_iter == true): opposite mask,
	//   shift [head]
	// SyncDirection::None means no exchange on this axis.
	__cuda_callable__ static constexpr SyncDirection dfSyncDirection(int dir, int axis, bool even_iter)
	{
		const int c = axis == 0 ? dir27_cx(dir) : axis == 1 ? dir27_cy(dir) : dir27_cz(dir);
		if (c == 0)
			return SyncDirection::None;
		const bool positive = even_iter ? c < 0 : c > 0;
		if (positive)
			return axis == 0 ? SyncDirection::Right : axis == 1 ? SyncDirection::Top : SyncDirection::Front;
		return axis == 0 ? SyncDirection::Left : axis == 1 ? SyncDirection::Bottom : SyncDirection::Back;
	}

	__cuda_callable__ static constexpr int dfSyncOffset(int dir, int axis, bool even_iter)
	{
		(void) axis;
		return even_iter ? int(is_pair_head(dir)) : int(! is_pair_head(dir));
	}
};

template <typename TRAITS>
inline constexpr bool is_ESO_PUSH_v<D3Q27_STREAMING_ESO_PUSH<TRAITS>> = true;
template <typename TRAITS>
inline constexpr bool is_esoteric_in_place_v<D3Q27_STREAMING_ESO_PUSH<TRAITS>> = true;
template <typename TRAITS>
inline constexpr bool requires_ghost_layer_v<D3Q27_STREAMING_ESO_PUSH<TRAITS>> = true;
