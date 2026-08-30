// Dedicated 3-level nesting mock simulation for the ParaView end-to-end
// nesting arm (tests/regression/test_amr_paraview.py driving
// tests/amr_paraview_e2e_nesting.py under pvpython, the
// amr-nlevel-nesting plan's commit D). sim_AMR/sim_AMR_channel hardcode
// their region specs, so a compact dedicated driver is the clean path to a
// nested-region VTKHDF frame: a 32^3 periodic Taylor-Green box with three
// fine blocks telescoping [6,26) -> [15, 15+10) -> [33, 33+8) in each
// level's parent frame (gaps >= 3 on every face, no warnings), advanced for
// a few coarse steps with a per-cycle VTKHDF frame.
//
// Per-level expectations pinned by the e2e script (same conversions as
// tests/test_amr_nesting.cu and tests/test_amr_vtkhdf_writer.cu):
// - 4 VTKHDF levels (0..3), one block per level, spacing halving per level;
// - per-level emitted-cell censuses: L0 32^3, L1 40^3, L2 20^3, L3 16^3
//   (the writer emits the interior plus the footprint-covering ghost rows);
// - per-level REFINEDCELL censuses: L0 20^3 (the level-1 footprint
//   [6,26)^3), L1 10^3 (the level-2 footprint [15,25)^3), L2 8^3 (the
//   level-3 footprint [33,41)^3), L3 0 (finest).
//
// Single-rank only, no CLI: the run is deterministic (4 coarse steps, one
// frame per cycle) so the e2e wrapper can point straight at
// results_test_amr_nesting_sim_np001/output_amr_0000.vtkhdf.

#include "lbm3d/core.h"
#include "lbm3d/amr_state.h"

// the telescoping nesting chain of the e2e arm ("level ox oy oz lx ly lz"
// in level-0 cells; parent-frame footprints [6,26) -> [15,25) -> [33,41),
// telescoping gaps of 9/6/6, 6/6/6... every face >= 3 parent cells:
// V-suite valid, no advisory warnings)
constexpr const char* amr_config = "1 6 6 6 20 20 20\n"
								   "2 30 30 30 20 20 20\n"
								   "3 132 132 132 32 32 32";

template <typename NSE>
struct StateLocal_AMR : State_AMR<NSE>
{
	using TRAITS = typename NSE::TRAITS;
	using BLOCK = LBM_BLOCK<NSE>;

	using State<NSE>::nse;

	using idx = typename TRAITS::idx;
	using idx3d = typename TRAITS::idx3d;
	using bool3d = typename TRAITS::bool3d;
	using lat_t = typename State_AMR<NSE>::lat_t;
	using dreal = typename State<NSE>::TRAITS::dreal;
	using point_t = typename State<NSE>::TRAITS::point_t;

	// problem parameters
	dreal V_0 = 0;	// [m/s] velocity amplitude
	dreal k = 0;	// [1/m] wave number

	StateLocal_AMR(const std::string& id, const TNL::MPI::Comm& communicator, lat_t lat, const std::string& adiosConfigPath, int max_level = 3)
	: State_AMR<NSE>(
		  id,
		  communicator,
		  std::move(lat),
		  adiosConfigPath,
		  // fully periodic domain, so no setupBoundaries() override
		  bool3d{true, true, true},
		  max_level
	  )
	{}

	// Taylor-Green initial condition on all blocks (the sim_AMR idiom);
	// per-level lattice parameters come from block.lat_local (level 0 uses
	// nse.lat), so local coordinates are passed directly
	void setInitialCondition()
	{
		for (auto& block : nse.blocks) {
#ifdef HAVE_MPI
			auto local_df = block.dfs[0].getLocalView();
#else
			auto local_df = block.dfs[0].getView();
#endif
			const lat_t lat_local = (block.level == 0) ? nse.lat : block.lat_local;
			const dreal V_0 = lat_local.phys2lbmVelocity(this->V_0);
			const dreal k = this->k;

			const idx3d begin = {0, 0, 0};
			const idx3d end = {block.local.y(), block.local.z(), block.local.x()};
			TNL::Algorithms::parallelFor<DeviceType>(
				begin,
				end,
				[local_df, lat_local, V_0, k] __cuda_callable__(const idx3d& yzx) mutable
				{
					const auto& [y, z, x] = yzx;
					const point_t phys = lat_local.lbm2physPoint(x, y, z);
					const dreal u = V_0 * TNL::sin(k * phys.x()) * TNL::cos(k * phys.y()) * TNL::cos(k * phys.z());
					const dreal v = -V_0 * TNL::cos(k * phys.x()) * TNL::sin(k * phys.y()) * TNL::cos(k * phys.z());
					const dreal w = 0;
					const dreal rho =
						1 + 3 * (V_0 * V_0 / 16) * (TNL::cos(2 * k * phys.x()) + TNL::cos(2 * k * phys.y())) * (TNL::cos(2 * k * phys.z()) + 2);
					NSE::COLL::setEquilibriumLat(local_df, x, y, z, rho, u, v, w);
				}
			);

			// copy the initialized DFs so that they are not overridden
			for (uint8_t dftype = 1; dftype < DFMAX; dftype++)
				block.dfs[dftype] = block.dfs[0];
		}

		nse.copyDFsToHost();
	}

	void resetDFs() override
	{
		setInitialCondition();
	}

	[[nodiscard]] std::vector<std::string> getOutputDataNames() const override
	{
		return {};
	}

	void outputData(UniformDataWriter<TRAITS>&, const BLOCK&, const idx3d&, const idx3d&) override {}
};

int main(int argc, char** argv)
{
	TNLMPI_INIT mpi(argc, argv);

	if (TNL::MPI::GetSize(MPI_COMM_WORLD) != 1) {
		fmt::println("test_amr_nesting_sim is single-rank only (nproc = {})", TNL::MPI::GetSize(MPI_COMM_WORLD));
		return 1;
	}

	using TRAITS = TraitsSP;
	using COLL = D3Q27_CUM<TRAITS, D3Q27_EQ_INV_CUM<TRAITS>>;
	using NSE_CONFIG = LBM_CONFIG<
		TRAITS,
		D3Q27_KernelStruct,
		NSE_Data_ConstInflow<TRAITS>,
		COLL,
		typename COLL::EQ,
		D3Q27_STREAMING<TRAITS>,
		D3Q27_BC_All,
		D3Q27_MACRO_Default<TRAITS>>;

	using idx = typename TRAITS::idx;
	using real = typename TRAITS::real;
	using point_t = typename TRAITS::point_t;
	using lat_t = Lattice<3, real, idx>;

	// 32^3 periodic box (same physical scaling as the fixtures/sim_AMR)
	const int N = 32;
	const real LBM_VISCOSITY = 0.005f;
	const real PHYS_HEIGHT = 0.41f;
	const real PHYS_VISCOSITY = 1.5e-5f;
	const real REYNOLDS = 100;
	const real PHYS_VELOCITY = REYNOLDS * PHYS_VISCOSITY / PHYS_HEIGHT;
	const real PHYS_DL = PHYS_HEIGHT / N;
	const real PHYS_DT = LBM_VISCOSITY / PHYS_VISCOSITY * PHYS_DL * PHYS_DL;

	lat_t lat;
	lat.global = typename lat_t::CoordinatesType(N, N, N);
	lat.physOrigin = point_t{0., 0., 0.};
	lat.physDl = PHYS_DL;
	lat.physDt = PHYS_DT;
	lat.physViscosity = PHYS_VISCOSITY;

	const std::string state_id = "test_amr_nesting_sim_np001";
	StateLocal_AMR<NSE_CONFIG> state(state_id, MPI_COMM_WORLD, lat, "adios2.xml", /*max_level=*/3);

	if (! state.canCompute())
		return 1;

	state.V_0 = PHYS_VELOCITY;
	state.k = 2 * TNL::pi / (N * PHYS_DL);

	// 4 coarse steps, one VTKHDF frame per coarse step (the OUT3D hook
	// fires once per cycle at a full sync point); PRINT goes quiet
	state.nse.physFinalTime = 4.5 * PHYS_DT;
	state.cnt[PRINT].period = 1.0f;
	state.cnt[OUT3D].period = PHYS_DT;

	state.nse.allocateHostData();
	state.nse.allocateDeviceData();
	state.nse.iterations = 0;
	createAMRBlocks(state.nse, parseAMRConfig<NSE_CONFIG>(amr_config));
	markAMRInterface(state.nse);
	state.setInitialCondition();

	execute(state);
	return 0;
}
