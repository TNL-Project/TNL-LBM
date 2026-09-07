"""Python twin of sim_openings.cu: square channel with an authored parabolic
inflow opening on the interior plane x=1, driven through the pytnl_lbm
bindings (uses the openings-capable State_SP_D3Q27_CUM_OpeningInflow).

At the final time the script integrates the axial velocity over the opening
cells and prints opening_flux and its relative error against the authored
target flux, exactly like the C++ sim.
"""

import argparse
import sys
from pathlib import Path

from mpi4py import MPI

PROJECT_DIR = Path(__file__).parent.parent

sys.path.append(str(PROJECT_DIR / "build/_deps/pytnl-build/src"))
sys.path.append(str(PROJECT_DIR / "build/pytnl_lbm/"))

from pytnl_lbm import (  # noqa: E402
    PRINT,
    PROBE1,
    InflowOpeningsState,
    ProfileType,
    addInflowPlane,
    execute,
    finalizeInflowOpenings,
    getMacroView,
)
from pytnl_lbm import State_SP_D3Q27_CUM_OpeningInflow as State  # noqa: E402


# Define the StateLocal class in Python
class StateLocal(State):
    def __init__(
        self,
        id: str,
        communicator: MPI.Intracomm,
        lat: State.lat_t,
        adiosConfigPath: str = "adios2.xml",
    ) -> None:
        super().__init__(id, communicator, lat, adiosConfigPath)
        self.openings = InflowOpeningsState()
        # authored target flux through the opening (unit cell area), resolved
        # by sim() before execute() starts reset()
        self.opening_flux = 0.0
        # opening cross-section bounds on y/z (fluid interior of the channel),
        # resolved from the lattice in setupBoundaries
        self.open_lo = 2
        self.open_hi_y = 0
        self.open_hi_z = 0

    def setupBoundaries(self) -> None:
        nse = self.nse
        lat = nse.lat
        self.open_hi_y = lat.global_.y - 3
        self.open_hi_z = lat.global_.z - 3

        # base imposed velocity (unit base that PARABOLIC shapes through the
        # precomputed site velocities)
        for block in nse.blocks:
            block.data.inflow_vx = 1

        lat_t = State.lat_t
        addInflowPlane(
            nse,
            self.openings,
            0,
            -1,
            1,
            lat_t.CoordinatesType((1, self.open_lo, self.open_lo)),
            lat_t.CoordinatesType((1, self.open_hi_y, self.open_hi_z)),
            ProfileType.PARABOLIC,
            self.opening_flux,
        )

        nse.setBoundaryX(lat.global_.x - 2, nse.BC.GEO_OUTFLOW_RIGHT_INTERP)  # right

        nse.setBoundaryZ(1, nse.BC.GEO_WALL)  # top
        nse.setBoundaryZ(lat.global_.z - 2, nse.BC.GEO_WALL)  # bottom
        nse.setBoundaryY(1, nse.BC.GEO_WALL)  # back
        nse.setBoundaryY(lat.global_.y - 2, nse.BC.GEO_WALL)  # front

        # extra layer needed due to A-A pattern
        nse.setBoundaryX(0, nse.BC.GEO_NOTHING)  # left
        nse.setBoundaryX(lat.global_.x - 1, nse.BC.GEO_NOTHING)  # right
        nse.setBoundaryZ(0, nse.BC.GEO_NOTHING)  # top
        nse.setBoundaryZ(lat.global_.z - 1, nse.BC.GEO_NOTHING)  # bottom
        nse.setBoundaryY(0, nse.BC.GEO_NOTHING)  # back
        nse.setBoundaryY(lat.global_.y - 1, nse.BC.GEO_NOTHING)  # front

        finalizeInflowOpenings(nse, self.openings)

    def AfterSimFinished(self) -> None:
        # final-time check: the PROBE1 counter keeps the device macro current;
        # refresh the host copy and integrate the axial velocity over the
        # opening cells (the moment BC imposes exactly the authored paraboloid
        # there, so the flux approaches the amplitude)
        nse = self.nse
        nse.copyMacroToHost()

        local_flux = 0.0
        e_vx = int(nse.MACRO.e_vx)
        for block in nse.blocks:
            if not block.isLocalX(1):
                continue
            vx = getMacroView(block.hmacro, e_vx)
            for j in range(self.open_lo, self.open_hi_y + 1):
                for k in range(self.open_lo, self.open_hi_z + 1):
                    if block.isLocalIndex(1, j, k):
                        local_flux += vx[1, j, k]

        flux = MPI.COMM_WORLD.allreduce(local_flux, op=MPI.SUM)
        rel_error = (
            abs(flux - self.opening_flux) / abs(self.opening_flux)
            if self.opening_flux != 0
            else 0.0
        )
        if nse.rank == 0:
            print(f"opening_flux = {flux:.8e}")
            print(f"flux_rel_error = {rel_error:.8e}")

        super().AfterSimFinished()


# Define the simulation function in Python
def sim(
    adiosConfigPath: str = "adios2.xml", RESOLUTION: int = 1, final_time: float = 1.0
) -> None:
    point_t = State.point_t
    lat_t = State.lat_t

    block_size = 32
    X = 128 * RESOLUTION  # width in pixels
    Y = block_size * RESOLUTION  # height in pixels --- top and bottom walls 1px
    Z = Y  # height in pixels --- top and bottom walls 1px
    LBM_VISCOSITY = 1e-4
    PHYS_HEIGHT = 0.41  # [m] domain height (physical)
    PHYS_VISCOSITY = 1.5e-5  # [m^2/s] fluid viscosity
    PHYS_DL = PHYS_HEIGHT / (Y - 2)
    PHYS_DT = LBM_VISCOSITY / PHYS_VISCOSITY * PHYS_DL * PHYS_DL
    PHYS_ORIGIN = point_t((0.0, 0.0, 0.0))

    # PHYS_VELOCITY is the target centerline (peak) velocity of the paraboloid
    PHYS_VELOCITY = 1.0

    # Initialize the lattice
    lat = lat_t()
    lat.global_ = lat_t.CoordinatesType((X, Y, Z))
    lat.physOrigin = PHYS_ORIGIN
    lat.physDl = PHYS_DL
    lat.physDt = PHYS_DT
    lat.physViscosity = PHYS_VISCOSITY

    state_id = f"sim_openings_py_res{RESOLUTION:02d}_np{MPI.COMM_WORLD.size:03d}"
    state = StateLocal(state_id, MPI.COMM_WORLD, lat, adiosConfigPath)

    if not state.canCompute():
        return

    # Problem parameters
    u_peak_lbm = lat.phys2lbmVelocity(PHYS_VELOCITY)
    Re = PHYS_VELOCITY * PHYS_HEIGHT / PHYS_VISCOSITY
    Ma = u_peak_lbm * 3**0.5

    # authored amplitude = target flux through the opening: the peak velocity
    # times the sum of the cell-centered paraboloid weights over the opening
    # cross-section (mirrors the C++ sim's accumulation)
    open_lo = 2
    open_hi_y = Y - 3
    open_hi_z = Z - 3
    nu = open_hi_y - open_lo + 1
    nv = open_hi_z - open_lo + 1
    weight_sum = 0.0
    for k in range(open_lo, open_hi_z + 1):
        eta = (k - open_lo + 0.5) / nv
        w_z = 1 - (2 * eta - 1) ** 2
        for j in range(open_lo, open_hi_y + 1):
            xi = (j - open_lo + 0.5) / nu
            weight_sum += (1 - (2 * xi - 1) ** 2) * w_z
    state.opening_flux = u_peak_lbm * weight_sum

    print(f"PHYS_VELOCITY (centerline) = {PHYS_VELOCITY:e} m/s")
    print(f"authored opening flux = {state.opening_flux:e}")
    print(f"Re = {Re:e} (based on centerline velocity and channel height)")
    print(f"Ma = {Ma:e} (based on lattice centerline velocity, c_s = 1/sqrt(3))")

    # Set up simulation parameters
    state.nse.physFinalTime = final_time
    state.cnt[PRINT].period = 0.1
    # no data outputs -- the PROBE1 cadence keeps the device macro fresh for
    # the final-time flux check via the regular counter-driven sync
    state.cnt[PROBE1].period = 0.1

    execute(state)


# Define the main function
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Square channel with an authored parabolic inflow on plane x=1."
    )
    parser.add_argument(
        "--adios-config",
        type=str,
        default="adios2.xml",
        help="path to ADIOS2 configuration file (default: %(default)s)",
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1,
        help="resolution of the lattice (default: %(default)s)",
    )
    parser.add_argument(
        "--final-time",
        type=float,
        default=1.0,
        help="final time of the simulation in physical units (default: %(default)s)",
    )
    args = parser.parse_args()

    if args.resolution < 1:
        raise ValueError("CLI error: resolution must be at least 1")
    if args.final_time <= 0:
        raise ValueError("CLI error: final-time must be positive")

    sim(args.adios_config, args.resolution, args.final_time)


if __name__ == "__main__":
    main()
