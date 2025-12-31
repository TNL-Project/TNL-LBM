import argparse
import sys
from pathlib import Path

import numpy as np
from mpi4py import MPI

PROJECT_DIR = Path(__file__).parent.parent

sys.path.append(str(PROJECT_DIR / "build/pytnl_lbm/"))

from pytnl_lbm import (  # noqa: E402
    OUT2D,
    OUT3D,
    OUT3DCUT,
    PRINT,
    UniformDataWriter,
    execute,
)
from pytnl_lbm import State_SP_D3Q27_CUM_ConstInflow as State  # noqa: E402


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
        self.lbm_inflow_vx = 0

    def setupBoundaries(self) -> None:
        nse = self.nse
        lat = nse.lat

        nse.setBoundaryX(0, nse.BC.GEO_INFLOW_LEFT)  # left
        nse.setBoundaryX(lat.global_.x - 1, nse.BC.GEO_OUTFLOW_RIGHT)  # right

        nse.setBoundaryZ(1, nse.BC.GEO_WALL)  # top
        nse.setBoundaryZ(lat.global_.z - 2, nse.BC.GEO_WALL)  # bottom
        nse.setBoundaryY(1, nse.BC.GEO_WALL)  # back
        nse.setBoundaryY(lat.global_.y - 2, nse.BC.GEO_WALL)  # front

        # extra layer needed due to A-A pattern
        nse.setBoundaryZ(0, nse.BC.GEO_NOTHING)  # top
        nse.setBoundaryZ(lat.global_.z - 1, nse.BC.GEO_NOTHING)  # bottom
        nse.setBoundaryY(0, nse.BC.GEO_NOTHING)  # back
        nse.setBoundaryY(lat.global_.y - 1, nse.BC.GEO_NOTHING)  # front

        # draw a wall with a hole
        cx = int(0.20 / lat.physDl)
        width = lat.global_.z // 10
        for px in range(cx, cx + width + 1):
            for pz in range(1, lat.global_.z - 1):
                for py in range(1, lat.global_.y - 1):
                    if not (
                        pz >= lat.global_.z * 4 // 10
                        and pz <= lat.global_.z * 6 // 10
                        and py >= lat.global_.y * 4 // 10
                        and py <= lat.global_.y * 6 // 10
                    ):
                        nse.setMap(px, py, pz, nse.BC.GEO_WALL)

    def getOutputDataNames(self) -> list[str]:
        # return all quantity names used in outputData
        return [
            "lbm_density",
            "velocity_x",
            "velocity_y",
            "velocity_z",
        ]

    def outputData(
        self,
        writer: UniformDataWriter,
        block: State.BLOCK_NSE,
        begin: State.idx3d,
        end: State.idx3d,
    ) -> None:
        if hasattr(block.hmacro, "getLocalView"):
            hmacro_np = np.from_dlpack(block.hmacro.getLocalView())
        else:
            hmacro_np = np.from_dlpack(block.hmacro)
        # convert global begin and end to local coordinates so we can pass them along
        # with a local ndarray to the writer.write function
        begin -= block.offset
        end -= block.offset
        # NOTE: output is in lattice units
        writer.write(
            "lbm_density", hmacro_np[self.nse.MACRO.e_rho, :, :, :], begin, end
        )
        writer.write("velocity_x", hmacro_np[self.nse.MACRO.e_vx, :, :, :], begin, end)
        writer.write("velocity_y", hmacro_np[self.nse.MACRO.e_vy, :, :, :], begin, end)
        writer.write("velocity_z", hmacro_np[self.nse.MACRO.e_vz, :, :, :], begin, end)

    def updateKernelVelocities(self) -> None:
        for block in self.nse.blocks:
            block.data.inflow_vx = self.lbm_inflow_vx
            block.data.inflow_vy = 0
            block.data.inflow_vz = 0


# Define the simulation function in Python
def sim(RESOLUTION: int = 2, adiosConfigPath: str = "adios2.xml") -> None:
    point_t = State.point_t
    lat_t = State.lat_t

    block_size = 32
    X = 128 * RESOLUTION  # width in pixels
    Y = block_size * RESOLUTION  # height in pixels --- top and bottom walls 1px
    Z = Y  # height in pixels --- top and bottom walls 1px
    LBM_VISCOSITY = 0.00001  # 1.0/6.0; /// GIVEN: optimal is 1/6
    PHYS_HEIGHT = 0.41  # [m] domain height (physical)
    PHYS_VISCOSITY = 1.5e-5  # [m^2/s] fluid viscosity .... blood?
    PHYS_VELOCITY = 1.0
    PHYS_DL = PHYS_HEIGHT / (Y - 2)
    PHYS_DT = LBM_VISCOSITY / PHYS_VISCOSITY * PHYS_DL * PHYS_DL
    PHYS_ORIGIN = point_t((0.0, 0.0, 0.0))

    # initialize the lattice
    lat = lat_t()
    lat.global_ = lat_t.CoordinatesType((X, Y, Z))
    lat.physOrigin = PHYS_ORIGIN
    lat.physDl = PHYS_DL
    lat.physDt = PHYS_DT
    lat.physViscosity = PHYS_VISCOSITY

    state_id = f"sim_1_res{RESOLUTION:02d}_np{MPI.COMM_WORLD.size:03d}"
    state = StateLocal(state_id, MPI.COMM_WORLD, lat, adiosConfigPath)

    if not state.canCompute():
        return

    # problem parameters
    state.lbm_inflow_vx = lat.phys2lbmVelocity(PHYS_VELOCITY)

    state.nse.physFinalTime = 1.0
    state.cnt[PRINT].period = 0.001

    # add cuts
    state.cnt[OUT2D].period = 0.001
    state.add2Dcut_X(X // 2, "cutsX/cut_X")
    state.add2Dcut_Y(Y // 2, "cutsY/cut_Y")
    state.add2Dcut_Z(Z // 2, "cutsZ/cut_Z")

    state.cnt[OUT3D].period = 0.1
    state.cnt[OUT3DCUT].period = 0.1
    state.add3Dcut(X // 4, Y // 4, Z // 4, X // 2, Y // 2, Z // 2, "box")

    execute(state)

    return


# Define the main function
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simple incompressible Navier-Stokes simulation example."
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1,
        help="resolution of the lattice (default: %(default)s)",
    )
    parser.add_argument(
        "--adios-config",
        type=str,
        default="adios2.xml",
        help="path to adios2.xml configuration file",
    )
    args = parser.parse_args()

    if args.resolution < 1:
        raise ValueError("CLI error: resolution must be at least 1")

    sim(args.resolution, args.adios_config)


if __name__ == "__main__":
    main()
