/*
 * Type-parametrized unit test for all streaming patterns (A-A, A-B pull,
 * A-B push, esoteric pull/push/twist) over all lattice models (D2Q9, D3Q7, D3Q27).
 *
 * Each pattern is initialized in its own DF layout (twisted for A-A, pre-stream
 * for A-B pull, post-stream for A-B push) so that the first collide reads
 * identical populations. One full stream+collide iteration with an identity
 * collision is executed per pattern (two parity sub-steps for A-A), and the
 * gathered populations are compared after each phase: geometrically, every
 * pattern must implement the same shift of the injected injective pattern.
 */

#include <vector>

#include <doctest/doctest.h>

#include <TNL/Containers/Array.h>
#include <TNL/Devices/Cuda.h>

#include "lbm3d/defs.h"
#include "lbm3d/d2q9/streaming.h"
#include "lbm3d/d3q27/streaming.h"
#include "lbm3d/d3q7/streaming.h"

// mock coordinate space per dimension (as in the outflow-gather tests)
static constexpr int MS = 32;

// injective pattern value over (slot, x, y, z); exact in double
static constexpr double pat(int slot, int x, int y, int z)
{
	return 1.0 + 1e6 * slot + 1e4 * x + 1e2 * y + z;
}

static __cuda_callable__ int clampMS(int v)
{
	return v < 0 ? 0 : (v >= MS ? MS - 1 : v);
}

// lattice-model shims unifying the direction access for the parametrized test
template <typename KS>
struct Dirs9
{
	using ks_t = KS;
	static constexpr int Q = 9;
	static constexpr int cx(int i)
	{
		return dir9_cx(i);
	}
	static constexpr int cy(int i)
	{
		return dir9_cy(i);
	}
	static constexpr int cz(int)
	{
		return 0;
	}
};

template <typename KS>
struct Dirs27
{
	using ks_t = KS;
	static constexpr int Q = 27;
	static constexpr int cx(int i)
	{
		return dir27_cx(i);
	}
	static constexpr int cy(int i)
	{
		return dir27_cy(i);
	}
	static constexpr int cz(int i)
	{
		return dir27_cz(i);
	}
};

template <typename KS>
struct Dirs7
{
	using ks_t = KS;
	static constexpr int Q = 7;
	static constexpr int cx(int i)
	{
		return dir27_cx(i);
	}
	static constexpr int cy(int i)
	{
		return dir27_cy(i);
	}
	static constexpr int cz(int i)
	{
		return dir27_cz(i);
	}
};

// device-side DF storage mock: two type slots so the A-B patterns can rotate
// between them; the A-A pattern uses slot df_cur only (mirroring production)
template <typename DIRS>
struct StreamMock
{
	double* mem;
	bool even_iter;

	__cuda_callable__ double& df(int type, int slot, int x, int y, int z) const
	{
		const size_t idx = ((size_t) type * DIRS::Q + slot) * MS * MS * MS + ((size_t) clampMS(x) * MS + clampMS(y)) * MS + clampMS(z);
		return mem[idx];
	}
};
// one gather + identity collision + scatter launch over the whole mock
// lattice (like production): every thread handles its own site and the
// kernel struct is captured at the probe site
template <typename STREAMING, typename DIRS>
__global__ void streamCollideKernel(StreamMock<DIRS> sd, typename DIRS::ks_t* out, int px, int py, int pz)
{
	const int x = threadIdx.x + blockIdx.x * blockDim.x;
	const int y = threadIdx.y + blockIdx.y * blockDim.y;
	const int z = threadIdx.z + blockIdx.z * blockDim.z;
	if (x >= MS || y >= MS || z >= MS)
		return;
	typename DIRS::ks_t ks;
	STREAMING::streaming(sd, ks, x - 1, x, x + 1, y - 1, y, y + 1, z - 1, z, z + 1);
	if (x == px && y == py && z == pz)
		*out = ks;
	STREAMING::postCollisionStreaming(sd, ks, x - 1, x, x + 1, y - 1, y, y + 1, z - 1, z, z + 1);
}

// launch sequence with the pattern's own iteration schedule and rotation
// semantics; returns the kernel structs gathered at the first and last launch
template <typename DIRS, typename STREAMING>
static void runRoundTrip(int x, int y, int z, typename DIRS::ks_t& ks1, typename DIRS::ks_t& ks2)
{
	using KS = typename DIRS::ks_t;
	static constexpr int Q = DIRS::Q;

	const size_t slotSize = (size_t) Q * MS * MS * MS;
	std::vector<double> host(2 * slotSize);
	for (int slot = 0; slot < Q; slot++)
		for (int xx = 0; xx < MS; xx++)
			for (int yy = 0; yy < MS; yy++)
				for (int zz = 0; zz < MS; zz++) {
					double v;
					if constexpr (is_AA_v<STREAMING>) {
						// twisted initial layout: slot opposite(i) holds pat(i, s)
						v = pat(opposite_direction(slot), xx, yy, zz);
					}
					else if constexpr (is_esoteric_in_place_v<STREAMING>) {
						// parity-0 initial placements (see LBM_BLOCK::setInitialCondition)
						if constexpr (is_ESO_TWIST_v<STREAMING>) {
							// slot (i, s) holds pat(i, s - p(c_i)), p(c) = max(c, 0)
							v =
								pat(slot,
									clampMS(xx - (DIRS::cx(slot) > 0 ? 1 : 0)),
									clampMS(yy - (DIRS::cy(slot) > 0 ? 1 : 0)),
									clampMS(zz - (DIRS::cz(slot) > 0 ? 1 : 0)));
						}
						else if constexpr (is_ESO_PULL_v<STREAMING>) {
							// heads shifted by -c_i, tails natural
							if (is_pair_head(slot))
								v = pat(slot, clampMS(xx - DIRS::cx(slot)), clampMS(yy - DIRS::cy(slot)), clampMS(zz - DIRS::cz(slot)));
							else
								v = pat(slot, xx, yy, zz);
						}
						else {
							// ESO_PUSH: head slot holds the tail's population shifted by +c_h,
							// tail slot the head's population at the own site
							const int pair = opposite_direction(slot);
							if (is_pair_head(slot))
								v = pat(pair, clampMS(xx + DIRS::cx(slot)), clampMS(yy + DIRS::cy(slot)), clampMS(zz + DIRS::cz(slot)));
							else
								v = pat(pair, xx, yy, zz);
						}
					}
					else if constexpr (is_AB_PUSH_v<STREAMING>) {
						// post-stream layout: slot (i, s) holds the population that arrived from s - c_i
						v = pat(slot, clampMS(xx - DIRS::cx(slot)), clampMS(yy - DIRS::cy(slot)), clampMS(zz - DIRS::cz(slot)));
					}
					else {
						// pre-stream layout: slot (i, s) holds pat(i, s)
						v = pat(slot, xx, yy, zz);
					}
					host[((size_t) slot) * MS * MS * MS + ((size_t) xx * MS + yy) * MS + zz] = v;
				}

	TNL::Containers::Array<double, TNL::Devices::Host> hostArr(host.size());
	for (size_t i = 0; i < host.size(); i++)
		hostArr[i] = host[i];
	TNL::Containers::Array<double, TNL::Devices::Cuda> dev;
	dev = hostArr;
	TNL::Containers::Array<KS, TNL::Devices::Cuda> devOut(2);

	StreamMock<DIRS> sd{dev.getData(), /*even_iter=*/false};
	const dim3 block(8, 8, 4);
	const dim3 grid(MS / 8, MS / 8, MS / 4);

	// first launch: gather ks1, collide(identity), scatter
	streamCollideKernel<STREAMING, DIRS><<<grid, block>>>(sd, devOut.getData(), x, y, z);
	TNL::Backend::deviceSynchronize();

	if constexpr (is_AA_v<STREAMING> || is_esoteric_in_place_v<STREAMING>) {
		// the second sub-step runs with the opposite parity on the same array
		sd.even_iter = true;
	}
	else {
		// A-B rotation: out becomes cur (emulates the pointer swap)
		TNL::Backend::memcpy(dev.getData(), dev.getData() + slotSize, slotSize * sizeof(double), TNL::Backend::MemcpyDeviceToDevice);
	}

	// second launch: the parity counterpart for A-A; a gather from the rotated
	// array for the A-B patterns
	streamCollideKernel<STREAMING, DIRS><<<grid, block>>>(sd, devOut.getData() + 1, x, y, z);
	TNL::Backend::deviceSynchronize();

	KS hostOut[2];
	TNL::Backend::memcpy(hostOut, devOut.getData(), 2 * sizeof(KS), TNL::Backend::MemcpyDeviceToHost);
	ks1 = hostOut[0];
	ks2 = hostOut[1];
}

template <typename DIRS, typename STREAMING>
static void checkRoundTripAt(int x, int y, int z)
{
	typename DIRS::ks_t ks1;
	typename DIRS::ks_t ks2;
	runRoundTrip<DIRS, STREAMING>(x, y, z, ks1, ks2);
	for (int i = 0; i < DIRS::Q; i++) {
		INFO("slot=", i);
		// one effective stream: the population traveling c_i arrived from s - c_i
		CHECK_EQ(ks1.f[i], pat(i, clampMS(x - DIRS::cx(i)), clampMS(y - DIRS::cy(i)), clampMS(z - DIRS::cz(i))));
		// two effective streams: pat(i, s - 2 c_i)
		CHECK_EQ(ks2.f[i], pat(i, clampMS(x - 2 * DIRS::cx(i)), clampMS(y - 2 * DIRS::cy(i)), clampMS(z - 2 * DIRS::cz(i))));
	}
}

template <typename S>
struct D2Q9Case
{
	using dirs = Dirs9<D2Q9_KernelStruct<double>>;
	using stream = S;
};

template <typename S>
struct D3Q7Case
{
	using dirs = Dirs7<D3Q7_KernelStruct<double>>;
	using stream = S;
};

template <typename S>
struct D3Q27Case
{
	using dirs = Dirs27<D3Q27_KernelStruct<double>>;
	using stream = S;
};

TEST_SUITE_BEGIN("streamingpatterns");

TEST_CASE_TEMPLATE(
	"stream-collide round trip",
	T,
	D2Q9Case<D2Q9_STREAMING_AA<TraitsDP>>,
	D2Q9Case<D2Q9_STREAMING_AB_PULL<TraitsDP>>,
	D2Q9Case<D2Q9_STREAMING_AB_PUSH<TraitsDP>>,
	D2Q9Case<D2Q9_STREAMING_ESO_PULL<TraitsDP>>,
	D2Q9Case<D2Q9_STREAMING_ESO_PUSH<TraitsDP>>,
	D2Q9Case<D2Q9_STREAMING_ESO_TWIST<TraitsDP>>,
	D3Q7Case<D3Q7_STREAMING_AA<TraitsDP>>,
	D3Q7Case<D3Q7_STREAMING_AB_PULL<TraitsDP>>,
	D3Q7Case<D3Q7_STREAMING_AB_PUSH<TraitsDP>>,
	D3Q7Case<D3Q7_STREAMING_ESO_PULL<TraitsDP>>,
	D3Q7Case<D3Q7_STREAMING_ESO_PUSH<TraitsDP>>,
	D3Q7Case<D3Q7_STREAMING_ESO_TWIST<TraitsDP>>,
	D3Q27Case<D3Q27_STREAMING_AA<TraitsDP>>,
	D3Q27Case<D3Q27_STREAMING_AB_PULL<TraitsDP>>,
	D3Q27Case<D3Q27_STREAMING_AB_PUSH<TraitsDP>>,
	D3Q27Case<D3Q27_STREAMING_ESO_PULL<TraitsDP>>,
	D3Q27Case<D3Q27_STREAMING_ESO_PUSH<TraitsDP>>,
	D3Q27Case<D3Q27_STREAMING_ESO_TWIST<TraitsDP>>
)
{
	INFO("site=(8,9,10)");
	checkRoundTripAt<typename T::dirs, typename T::stream>(8, 9, 10);
	INFO("site=(20,18,16)");
	checkRoundTripAt<typename T::dirs, typename T::stream>(20, 18, 16);
}

TEST_SUITE_END();
