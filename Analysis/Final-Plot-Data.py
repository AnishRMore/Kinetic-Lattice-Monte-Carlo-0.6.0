# Import modules
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Make the project package importable when this script is run from Analysis.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from KMC.Analysis import weighted_average


# Declare constants
PERIODIC = "Periodic interface"
PHYSICAL = "Physical interface"

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_FILE = SCRIPT_DIR / "conductivities-advanced.csv"
OUTPUT_FILE = SCRIPT_DIR / "Conductivity vs InvT.png"


# Prepare plot
fig, ax = plt.subplots(1, 1)

# Load conductivities data
data = pd.read_csv(DATA_FILE, sep=",", index_col=0)
temperatures = np.unique(data["Temperature"].values)

# Compile data into plottable form
shown_invT = []
shown_phys = []
shown_phys_err = []
shown_per = []
shown_per_err = []

for T in temperatures:
    # Get data corresponding to this temperature
    is_temp = data["Temperature"] == T

    # Take weighted average
    sigma_physical, sigma_physical_err = weighted_average(
        data[is_temp]["Sigma (Physical)"].to_numpy(dtype=float),
        data[is_temp]["Sigma Error (Physical)"].to_numpy(dtype=float),
    )
    sigma_periodic, sigma_periodic_err = weighted_average(
        data[is_temp]["Sigma (Periodic)"].to_numpy(dtype=float),
        data[is_temp]["Sigma Error (Periodic)"].to_numpy(dtype=float),
    )

    # Convert temperature to 1/T
    invT = 1.0 / T

    shown_invT.append(invT)
    shown_phys.append(sigma_physical)
    shown_phys_err.append(sigma_physical_err)
    shown_per.append(sigma_periodic)
    shown_per_err.append(sigma_periodic_err)

# Convert to numpy arrays for sorting
shown_invT = np.array(shown_invT)
shown_phys = np.array(shown_phys)
shown_phys_err = np.array(shown_phys_err)
shown_per = np.array(shown_per)
shown_per_err = np.array(shown_per_err)

# Sort by increasing 1/T
sort_idx = np.argsort(shown_invT)

shown_invT = shown_invT[sort_idx]
shown_phys = shown_phys[sort_idx]
shown_phys_err = shown_phys_err[sort_idx]
shown_per = shown_per[sort_idx]
shown_per_err = shown_per_err[sort_idx]

# Plot data
ax.errorbar(
    shown_invT,
    shown_per,
    yerr=shown_per_err,
    color="red",
    ecolor="red",
    fmt="o-",
    capsize=10.0,
    capthick=1.0,
    label=PERIODIC,
)

ax.errorbar(
    shown_invT,
    shown_phys,
    yerr=shown_phys_err,
    color="black",
    ecolor="black",
    fmt="o-",
    capsize=10.0,
    capthick=1.0,
    label=PHYSICAL,
)

# Configure plot
ax.legend(fontsize=12)
ax.grid()

ax.set_xlabel(r"$1/T$ (K$^{-1}$)", fontsize=14)
ax.set_ylabel("Ionic Conductivity (S/cm)", fontsize=14)

ax.set_title(f"{PHYSICAL} and {PERIODIC}", fontsize=14)

ax.tick_params(axis="both", labelsize=12)
ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))

fig.suptitle("Average Ionic Conductivity vs Inverse Temperature", size=15)

fig.tight_layout()

# Save and show plot
fig.savefig(OUTPUT_FILE, dpi=300)

plt.show()

print(f"Saved {OUTPUT_FILE}")