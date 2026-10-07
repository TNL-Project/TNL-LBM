#! /usr/bin/env python3

import argparse
import json
import os
import subprocess
from pathlib import Path

from tabulate import tabulate

SIM_NAME = "sim_IBM3"


def run_sim(
    *,
    compute: str = "gpu",
    dirac: int = 1,
    method: str = "modified",
    Re: float = 100,
    hi: int = 0,
    resolution: int = 5,
) -> Path:
    hvals = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
    assert hi in range(len(hvals))
    assert method in {"modified", "original"}
    assert compute in {"cpu", "gpu"}
    compute_id = 0 if compute == "gpu" else 1

    results_dir = (
        f"results_{SIM_NAME}_CUM_{method}_dirac_{dirac}_res_{resolution}_Re_{Re}_nas_{hvals[hi]:.4f}_compute_{compute_id}"
    )
    log_file = Path(results_dir) / "log_ibm_rank000"
    if log_file.exists():
        log_file.unlink()

    print("Running simulation for Dirac", dirac)
    subprocess.run(
        [
            f"./build/sim_NSE/{SIM_NAME}",
            "0" if method == "modified" else "1",
            str(dirac),
            str(Re),
            str(hi),
            str(resolution),
            str(compute_id),
        ],
        check=True,
    )
    if not log_file.exists():
        raise Exception(f"log file {log_file} does not exist after running the simulation")

    return log_file


def run_simulations(compute: str, diracmin: int = 1, diracmax: int = 4) -> tuple[list[list], list[dict], str]:
    constructMatricesTableElements = []
    computeForcesTableElements = []

    variantString = ""
    for dirac in range(diracmin, diracmax + 1):
        log_file = run_sim(compute=compute, dirac=dirac)
        lines = log_file.read_text().splitlines()
        for line in lines:
            if line.find("constructMatricesJSON:") >= 0:
                splitString = line.split("constructMatricesJSON:")
                parsedJson = json.loads(splitString[1])
                constructMatricesTableElements.append(
                    [
                        parsedJson["threads"],
                        str(dirac),
                        parsedJson["time_total"],
                        parsedJson["time_M_capacities"],
                        parsedJson["time_M_construct"],
                        parsedJson["time_M_transpose"],
                        parsedJson["time_A_capacities"],
                        parsedJson["time_A_construct"],
                        parsedJson["time_matrixWrite"],
                        parsedJson["time_matrixCopy"],
                    ]
                )
                variantString = (
                    "hACapacities-"
                    + str(parsedJson.get("variant_Ha_capacities", 1))
                    + "_hA-"
                    + str(parsedJson.get("variant_Ha", 1))
                )
            if line.find("computeForcesJSON:") >= 0:
                splitString = line.split("computeForcesJSON:")
                parsedJson = json.loads(splitString[1])
                computeForcesTableElements.append(
                    {
                        "threads": parsedJson["threads"],
                        "dirac": str(dirac),
                        "time_total": parsedJson["time_total"],
                    }
                )

    return (constructMatricesTableElements, computeForcesTableElements, variantString)


def build(variantHaCapacities: int, variantHa: int) -> None:
    subprocess.run(
        ["cmake -B build -DHA_CAPACITY_VARIANT=" + str(variantHaCapacities) + " -DHA_VARIANT=" + str(variantHa)],
        shell=True,
        check=True,
    )
    subprocess.run("cmake --build build", shell=True, check=True)


def cleanFiles() -> None:
    subprocess.run(f"rm -rf ./results_{SIM_NAME}_*", shell=True, check=True)
    subprocess.run("rm -f ./ibm_*.mtx", shell=True, check=True)


def main() -> None:
    # parse arguments
    parser = argparse.ArgumentParser(
        prog="Parallel LBM-IBM Performance Table Maker",
        description="This program runs the simulations and displays relevant info about the simulation run",
    )
    parser.add_argument("-o", "--output", default="prefix")
    parser.add_argument("-b", "--build", action="store_true")
    parser.add_argument(
        "-c",
        "--clean",
        action="store_true",
        help="clean results directories and .mtx files after running the simulation",
    )
    parser.add_argument(
        "-t",
        "--threads",
        type=int,
        default=int(os.environ.get("OMP_NUM_THREADS", "0")),
        help="Number of CPU threads (does not apply to GPU compute)",
    )
    parser.add_argument("--variantHaCapacities", type=int, default=1)
    parser.add_argument("--variantHa", type=int, default=1)
    parser.add_argument("--compute", choices=["gpu", "cpu"], default="gpu")
    args = parser.parse_args()

    if args.compute == "cpu":
        print("CPU selected")
        print(f"Running script for {args.threads} thread(s).")
    elif args.compute == "gpu":
        print("GPU selected")
        print("Running script for GPU")

    if args.threads > 0:
        os.environ["OMP_NUM_THREADS"] = str(args.threads)

    # run cmake
    if args.build:
        build(args.variantHaCapacities, args.variantHa)
    # run simulations
    constructTableElements, computeTableElements, variantString = run_simulations(args.compute)
    # clean files
    if args.clean:
        cleanFiles()

    filename = args.output + "_UNDEFINED_threads-" + str(args.threads) + "_" + variantString + ".txt"
    if args.compute == "cpu":
        filename = args.output + "_CPU_threads-" + str(args.threads) + "_" + variantString + ".txt"
    elif args.compute == "gpu":
        filename = args.output + "_GPU" + "_" + variantString + ".txt"

    file = open(filename, "w")
    file.write("Variants: " + variantString + "\n")
    constructTableHeaders = [
        "Threads",
        "Dirac",
        "Total Time",
        "Time M capacities",
        "Time M construct",
        "Time M transpose",
        "Time A capacities",
        "Time A construct",
        "Time matrix write",
        "Time matrix copy",
    ]
    constructTableElements = list(sorted(constructTableElements, key=lambda x: x[2]))
    print(
        tabulate(constructTableElements, tablefmt="github", headers=constructTableHeaders),
        file=file,
    )

    file.write("\n" * 2)

    computeTableHeaders = ["Threads", "Dirac", "Total Time"]
    computeTable = []
    for r in computeTableElements:
        computeTable.append(
            [
                r["threads"],
                r["dirac"],
                r["time_total"],
            ]
        )
    print(
        tabulate(computeTable, tablefmt="github", headers=computeTableHeaders),
        file=file,
    )


if __name__ == "__main__":
    main()
