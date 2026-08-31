/*
 * Unit tests for the any-face moment inflow boundary condition
 * (GEO_INFLOW_MOMENT), both D3Q27 and D2Q9.
 *
 * For every domain face construction (normal axis + outward sign) the
 * generalized moment BC (D3Q27: BC::inflowMoment(face, KS) runtime body;
 * D2Q9: BC::inflowMoment<AXIS, SIGN>(KS) per-face instantiations - see
 * docs/moment-bc-derivation.md) must satisfy the Eichler moment system
 * after the write: mass, three momentum components, the two tangential
 * second-order stresses, the tangential shear, and the 3rd/4th-order
 * cross-moments at their equilibrium values, with the not-to-be-written
 * layers left bit-untouched. The expectations are computed from the
 * population moments directly (ground truth = the moment system itself), so
 * a wrong formula cannot hide behind matching arithmetic.
 * On the legacy face (XM) the body evaluates the pre-generalization
 * expression tree bit-exactly by construction; bitwise compatibility with
 * legacy simulations is enforced by the production regression suites
 * (tests/regression/test_d2q9.py, tests/regression/test_d3q27_nse.py sim_1)
 * rather than by a local duplicate XM body here.
 */

#include <doctest/doctest.h>

#include "lbm3d/defs.h"
#include "lbm3d/lbm_data.h"

#ifdef AB_PATTERN
	#include "lbm3d/d2q9/streaming_AB.h"
#endif
#ifdef AA_PATTERN
	#include "lbm3d/d2q9/streaming_AA.h"
#endif
#include "lbm3d/d2q9/bc.h"
#include "lbm3d/d2q9/col_srt.h"
#include "lbm3d/d2q9/macro.h"

#include "lbm3d/core.h"  // d3q27 umbrella
#include "lbm3d/d3q27/bc.h"
#include "lbm3d/d3q27/col_srt.h"
#include "lbm3d/d3q27/macro.h"

using TRAITS = Traits<double>;  // dreal = double: control precision for constraint comparisons

// ---------------------------------------------------------------------------
// face geometry, written independently of the production AXIS/SIGN mapping:
// face -> (normal axis 0/1/2 = x/y/z, outward sign, tangential axes t1, t2)
// ---------------------------------------------------------------------------
struct FaceSpec
{
	int face;
	int axis;
	int sign;
	int t1;
	int t2;
};

static constexpr FaceSpec d3q27_faces[6] = {
	{bc_face::XP, 0, +1, 1, 2},
	{bc_face::XM, 0, -1, 1, 2},
	{bc_face::YP, 1, +1, 2, 0},
	{bc_face::YM, 1, -1, 2, 0},
	{bc_face::ZP, 2, +1, 0, 1},
	{bc_face::ZM, 2, -1, 0, 1},
};

static constexpr FaceSpec d2q9_faces[4] = {
	{bc_face::XP, 0, +1, 1, -1},
	{bc_face::XM, 0, -1, 1, -1},
	{bc_face::YP, 1, +1, 0, -1},
	{bc_face::YM, 1, -1, 0, -1},
};

// velocity component along axis a (face-local access in tests)
template <typename VEC>
static double axisComp(const VEC& v, int axis)
{
	return axis == 0 ? v[0] : axis == 1 ? v[1] : v[2];
}

static double dirComp3(int i, int axis)
{
	return axis == 0 ? dir27_cx(i) : axis == 1 ? dir27_cy(i) : dir27_cz(i);
}

static double dirComp2(int i, int axis)
{
	return axis == 0 ? dir9_cx(i) : dir9_cy(i);
}

// ---------------------------------------------------------------------------
// D3Q27
// ---------------------------------------------------------------------------
using TRAITS3 = TRAITS;
using KS3 = D3Q27_KernelStruct<typename TRAITS3::dreal>;
using COLL3 = D3Q27_SRT<TRAITS3>;
using CONFIG3 =
	LBM_CONFIG<TRAITS3, D3Q27_KernelStruct, NSE_Data<TRAITS3>, COLL3, typename COLL3::EQ, D3Q27_STREAMING<TRAITS3>, D3Q27_BC_All, D3Q27_MACRO_Default<TRAITS3>>;
using BC3 = typename CONFIG3::BC;

// driver kernel: run the inflow moment body for one face on one KS
__global__ void runInflowMoment3D(int face, KS3 in, KS3* out)
{
	*out = in;
	BC3::inflowMoment(face, *out);
}

static void checkFace3D(const FaceSpec& fs, const double vel[3])
{
	// deterministic distinct input pattern (small integers, exactly representable)
	KS3 in;
	for (int i = 0; i < 27; i++)
		in.f[i] = 1.0 + 0.25 * i;
	in.vx = vel[0];
	in.vy = vel[1];
	in.vz = vel[2];
	in.rho = -123.0;  // must be overwritten

	TNL::Containers::Array<KS3, TNL::Devices::Cuda> devOut(1);
	runInflowMoment3D<<<1, 1>>>(fs.face, in, devOut.getData());
	TNL::Backend::deviceSynchronize();
	KS3 out;
	TNL::Backend::memcpy(&out, devOut.getData(), sizeof(KS3), TNL::Backend::MemcpyDeviceToHost);

	const double vn = axisComp(vel, fs.axis);
	const double vt1 = axisComp(vel, fs.t1);
	const double vt2 = axisComp(vel, fs.t2);

	const auto verify = [&](const KS3& out)
	{
		const double rho = out.rho;
		const double tol = 1e-10 * rho;	 // generous absolute scale, tight relative

		// (i) untouched layers are left bit-exact: every slot with cn != -sign
		for (int i = 0; i < 27; i++)
			if (int(dirComp3(i, fs.axis)) != -fs.sign)
				CHECK_MESSAGE(out.f[i] == in.f[i], "slot ", i, " of face ", fs.face, " must stay untouched");

		// (ii) mass
		double mass = 0, mx = 0, my = 0, mz = 0;
		for (int i = 0; i < 27; i++) {
			mass += out.f[i];
			mx += dir27_cx(i) * out.f[i];
			my += dir27_cy(i) * out.f[i];
			mz += dir27_cz(i) * out.f[i];
		}
		CHECK(mass == doctest::Approx(rho).epsilon(1e-12));
		// (iii) momentum
		CHECK(mx == doctest::Approx(rho * vel[0]).scale(tol));
		CHECK(my == doctest::Approx(rho * vel[1]).scale(tol));
		CHECK(mz == doctest::Approx(rho * vel[2]).scale(tol));

		// (iv) tangential second-order stresses and shear
		double s11 = 0, s22 = 0, s12 = 0, q112 = 0, q122 = 0, m22 = 0;
		for (int i = 0; i < 27; i++) {
			const double c1 = dirComp3(i, fs.t1), c2 = dirComp3(i, fs.t2);
			s11 += c1 * c1 * out.f[i];
			s22 += c2 * c2 * out.f[i];
			s12 += c1 * c2 * out.f[i];
			q112 += c1 * c1 * c2 * out.f[i];
			q122 += c1 * c2 * c2 * out.f[i];
			m22 += c1 * c1 * c2 * c2 * out.f[i];
		}
		CHECK(s11 == doctest::Approx(rho / 3 + rho * vt1 * vt1).scale(tol));
		CHECK(s22 == doctest::Approx(rho / 3 + rho * vt2 * vt2).scale(tol));
		CHECK(s12 == doctest::Approx(rho * vt1 * vt2).scale(tol));
		// (v) 3rd and 4th order cross-moments (equilibrium)
		CHECK(q112 == doctest::Approx(rho * vt2 / 3 + rho * vt1 * vt1 * vt2).scale(tol));
		CHECK(q122 == doctest::Approx(rho * vt1 / 3 + rho * vt1 * vt2 * vt2).scale(tol));
		CHECK(m22 == doctest::Approx(rho / 9 + rho / 3 * (vt1 * vt1 + vt2 * vt2) + rho * vt1 * vt1 * vt2 * vt2).scale(tol));

		// (vi) the written layer is exactly the in-domain-moving family cn == -sign
		int written = 0;
		for (int i = 0; i < 27; i++)
			if (int(dirComp3(i, fs.axis)) == -fs.sign) {
				written++;
				CHECK_MESSAGE(out.f[i] != in.f[i], "face ", fs.face, " slot ", i, " should be rewritten");
			}
		CHECK(written == 9);
	};

	verify(out);
}

TEST_SUITE_BEGIN("inflowmoment3d");

TEST_CASE("faces-constraints")
{
	SUBCASE("v1")
	{
		const double vel[3] = {0.05, -0.02, 0.03};
		for (const auto& fs : d3q27_faces) {
			INFO("face=", fs.face);
			checkFace3D(fs, vel);
		}
	}
	SUBCASE("v2")
	{
		const double vel[3] = {-0.03, 0.04, -0.01};
		for (const auto& fs : d3q27_faces) {
			INFO("face=", fs.face);
			checkFace3D(fs, vel);
		}
	}
}

TEST_SUITE_END();

// ---------------------------------------------------------------------------
// D2Q9
// ---------------------------------------------------------------------------
using KS2 = D2Q9_KernelStruct<typename TRAITS::dreal>;
using COLL2 = D2Q9_SRT<TRAITS, D2Q9_EQ<TRAITS>>;
using CONFIG2 =
	LBM_CONFIG<TRAITS, D2Q9_KernelStruct, NSE_Data<TRAITS>, COLL2, typename COLL2::EQ, D2Q9_STREAMING<TRAITS>, D2Q9_BC_All, D2Q9_MACRO_Default<TRAITS>>;
using BC2 = typename CONFIG2::BC;

__global__ void runInflowMoment2D(int face, KS2 in, KS2* out)
{
	*out = in;
	switch (face) {
		case bc_face::XM:
			BC2::inflowMoment<0, -1>(*out);
			break;
		case bc_face::XP:
			BC2::inflowMoment<0, 1>(*out);
			break;
		case bc_face::YP:
			BC2::inflowMoment<1, 1>(*out);
			break;
		case bc_face::YM:
			BC2::inflowMoment<1, -1>(*out);
			break;
	}
}

static void checkFace2D(const FaceSpec& fs, const double vel[2])
{
	KS2 in;
	for (int i = 0; i < 9; i++)
		in.f[i] = 1.0 + 0.25 * i;
	in.vx = vel[0];
	in.vy = vel[1];
	in.rho = -123.0;

	TNL::Containers::Array<KS2, TNL::Devices::Cuda> devOut(1);
	runInflowMoment2D<<<1, 1>>>(fs.face, in, devOut.getData());
	TNL::Backend::deviceSynchronize();
	KS2 out;
	TNL::Backend::memcpy(&out, devOut.getData(), sizeof(KS2), TNL::Backend::MemcpyDeviceToHost);

	const double vt = fs.t1 == 0 ? vel[0] : vel[1];

	const auto verify = [&](const KS2& out)
	{
		const double rho = out.rho;
		const double tol = 1e-10 * rho;

		for (int i = 0; i < 9; i++)
			if (int(dirComp2(i, fs.axis)) != -fs.sign)
				CHECK_MESSAGE(out.f[i] == in.f[i], "slot ", i, " of face ", fs.face, " must stay untouched");

		double mass = 0, mx = 0, my = 0, stt = 0;
		for (int i = 0; i < 9; i++) {
			mass += out.f[i];
			mx += dir9_cx(i) * out.f[i];
			my += dir9_cy(i) * out.f[i];
			const double ct = dirComp2(i, fs.t1);
			stt += ct * ct * out.f[i];
		}
		CHECK(mass == doctest::Approx(rho).epsilon(1e-12));
		CHECK(mx == doctest::Approx(rho * vel[0]).scale(tol));
		CHECK(my == doctest::Approx(rho * vel[1]).scale(tol));
		CHECK(stt == doctest::Approx(rho / 3 + rho * vt * vt).scale(tol));

		int written = 0;
		for (int i = 0; i < 9; i++)
			if (int(dirComp2(i, fs.axis)) == -fs.sign) {
				written++;
				CHECK_MESSAGE(out.f[i] != in.f[i], "face ", fs.face, " slot ", i, " should be rewritten");
			}
		CHECK(written == 3);
	};

	verify(out);
}

TEST_SUITE_BEGIN("inflowmoment2d");

TEST_CASE("faces-constraints")
{
	const double vel[2] = {0.05, -0.02};
	for (const auto& fs : d2q9_faces) {
		INFO("face=", fs.face);
		checkFace2D(fs, vel);
	}
}

TEST_SUITE_END();
