import os

import httpx
import matplotlib.pyplot as plt
import numpy as np

nu = 1e-4
precision = "float"

# source: https://github.com/chaos-polymtl/lethe/tree/master/examples/incompressible-flow/3d-taylor-green-vortex/reference
reference_url = "https://raw.githubusercontent.com/chaos-polymtl/lethe/e51db26ad0bbb2e872bedb6dce3c6a8ca3330fb8/examples/incompressible-flow/3d-taylor-green-vortex/reference/wang_2013.dat"
reference_path = f"results_sim_4_{precision}_np001/wang_2013.txt"

input_paths = {
    "reference (Wang et al.)": reference_path,
    f"res =  1, nu = {nu:.0e}": f"results_sim_4_{precision}_np001/res=01_Re=1600_nu={nu:.6e}/probe1/kinetic_energy_rank000.txt",
    f"res =  2, nu = {nu:.0e}": f"results_sim_4_{precision}_np001/res=02_Re=1600_nu={nu:.6e}/probe1/kinetic_energy_rank000.txt",
    f"res =  4, nu = {nu:.0e}": f"results_sim_4_{precision}_np001/res=04_Re=1600_nu={nu:.6e}/probe1/kinetic_energy_rank000.txt",
    f"res =  8, nu = {nu:.0e}": f"results_sim_4_{precision}_np001/res=08_Re=1600_nu={nu:.6e}/probe1/kinetic_energy_rank000.txt",
    f"res = 16, nu = {nu:.0e}": f"results_sim_4_{precision}_np001/res=16_Re=1600_nu={nu:.6e}/probe1/kinetic_energy_rank000.txt",
}

output_path = f"results_sim_4_{precision}_np001/Taylor-Green_kinetic_energy_dissipation_{nu=:.0e}_{precision}.png"

if not os.path.exists(reference_path):
    print(f"Downloading reference data from {reference_url}")
    r = httpx.get(reference_url)
    r.raise_for_status()
    with open(reference_path, "wb") as f:
        f.write(r.content)


def get_data(label, path):
    # Read data from file
    data = np.loadtxt(path, comments="#", skiprows=1)

    # Convert to numpy array
    data = np.array(data).T
    if "reference" in label.lower():
        time_col = data[0]
        epsilon = data[1]
        return time_col, None, epsilon, None
    else:
        time_col = data[1]
        kinetic_energy = data[2]
        # enstrophy = data[3]
        enstrophy_dissipation = data[4]

    # Calculate dissipation rate epsilon = -dEk/dt using finite differences
    epsilon = np.zeros_like(kinetic_energy)
    for i in range(len(kinetic_energy)):
        if i == 0:
            epsilon[i] = -(kinetic_energy[i + 1] - kinetic_energy[i]) / (
                time_col[i + 1] - time_col[i]
            )
        elif i == len(kinetic_energy) - 1:
            epsilon[i] = -(kinetic_energy[i] - kinetic_energy[i - 1]) / (
                time_col[i] - time_col[i - 1]
            )
        elif i == 1 or i == len(kinetic_energy) - 2:
            epsilon[i] = -(kinetic_energy[i + 1] - kinetic_energy[i - 1]) / (
                time_col[i + 1] - time_col[i - 1]
            )
        else:
            h = time_col[i + 1] - time_col[i]
            epsilon[i] = (
                -(
                    1.0 / 12 * kinetic_energy[i - 2]
                    - 2.0 / 3.0 * kinetic_energy[i - 1]
                    + 2.0 / 3.0 * kinetic_energy[i + 1]
                    - 1.0 / 12.0 * kinetic_energy[i + 2]
                )
                / h
            )

    return time_col, kinetic_energy, epsilon, enstrophy_dissipation


def plot(input_paths, output_path):
    # Create subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))

    for label, path in input_paths.items():
        if not os.path.exists(path):
            print(f"Skipping non-existent file: {path}")
            continue

        time_col, kinetic_energy, epsilon, enstrophy_dissipation = get_data(label, path)

        if "reference" in label.lower():
            ax2.plot(time_col, epsilon, "k-", linewidth=1, label=label)
            ax3.plot(time_col, epsilon, "k-", linewidth=1, label=label)
        else:
            assert enstrophy_dissipation is not None
            ax1.plot(time_col, kinetic_energy, "-", linewidth=1, label=label)
            ax2.plot(time_col, epsilon, "-", linewidth=1, label=label)
            ax3.plot(time_col, enstrophy_dissipation, "-", linewidth=1, label=f"{label}")

    ax1.set_xlabel("Time")
    ax1.set_ylabel("Kinetic energy")
    ax1.set_title("Temporal evolution of the kinetic energy")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Time")
    ax2.set_ylabel("Dissipation rate")
    ax2.set_title(
        "Temporal evolution of the dissipation rate (measured by kinetic energy $E_K$)"
    )
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3.set_xlabel("Time")
    ax3.set_ylabel("Dissipation rate")
    ax3.set_title(
        "Temporal evolution of the dissipation rate (measured by enstrophy $\\mathcal{E}$)"
    )
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    ax1.set_xlim([0, 20])
    ax2.set_xlim([0, 20])
    ax3.set_xlim([0, 20])

    ax1.set_ylim([0, 0.14])
    ax2.set_ylim([0, 0.014])
    ax3.set_ylim([0, 0.014])

    plt.tight_layout()

    plt.savefig(output_path, dpi=300)

    plt.show()


plot(input_paths, output_path)
