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
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_FILE = SCRIPT_DIR / "conductivities-advanced.csv"
OUTPUT_FILE = SCRIPT_DIR / "Conductivity_vs_1000_over_T.png"
LOG_OUTPUT_FILE = SCRIPT_DIR / "Ln_(Conductivity)_vs_1000_over_T.png"


# Load conductivities data
data = pd.read_csv(DATA_FILE, sep=",")

# Average the two interface values for each simulation.
data["Sigma (Average Interface)"] = (
    data["Sigma (Physical)"].to_numpy(dtype=float)
    + data["Sigma (Periodic)"].to_numpy(dtype=float)
) / 2.0
data["Sigma Error (Average Interface)"] = np.sqrt(
    data["Sigma Error (Physical)"].to_numpy(dtype=float) ** 2
    + data["Sigma Error (Periodic)"].to_numpy(dtype=float) ** 2
) / 2.0

temperatures = np.unique(data["Temperature"].values)


# Compile data into plottable form
shown_1000_over_T = []
shown_sigma = []
shown_sigma_err = []

for T in temperatures:
    is_temp = data["Temperature"] == T

    sigma, sigma_err = weighted_average(
        data[is_temp]["Sigma (Average Interface)"].to_numpy(dtype=float),
        data[is_temp]["Sigma Error (Average Interface)"].to_numpy(dtype=float),
    )

    shown_1000_over_T.append(1000.0 / T)
    shown_sigma.append(sigma)
    shown_sigma_err.append(sigma_err)


# Convert to numpy arrays for sorting
shown_1000_over_T = np.array(shown_1000_over_T)
shown_sigma = np.array(shown_sigma)
shown_sigma_err = np.array(shown_sigma_err)

sort_idx = np.argsort(shown_1000_over_T)

shown_1000_over_T = shown_1000_over_T[sort_idx]
shown_sigma = shown_sigma[sort_idx]
shown_sigma_err = shown_sigma_err[sort_idx]


# Prepare conductivity plot
fig, ax = plt.subplots(1, 1)

ax.errorbar(
    shown_1000_over_T,
    shown_sigma,
    yerr=shown_sigma_err,
    color="black",
    ecolor="black",
    fmt="o-",
    capsize=10.0,
    capthick=1.0,
    label="Average interface",
)


# Configure plot
#ax.legend(fontsize=12)
ax.grid()

ax.set_xlabel(r"$1000/T$ (K$^{-1}$)", fontsize=14)
ax.set_ylabel("Ionic Conductivity (S/cm)", fontsize=14)
#ax.set_title("Average Interface Conductivity", fontsize=14)

ax.tick_params(axis="both", labelsize=12)
ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))

ax.set_title("Ionic Conductivity vs 1000/T", size=15)
fig.tight_layout()


# Save and show plot
fig.savefig(OUTPUT_FILE, dpi=300)

print(f"Saved {OUTPUT_FILE}")


# Prepare ln(conductivity) plot
valid_log = shown_sigma > 0.0
log_1000_over_T = shown_1000_over_T[valid_log]
shown_ln_sigma = np.log(shown_sigma[valid_log])
shown_ln_sigma_err = shown_sigma_err[valid_log] / shown_sigma[valid_log]

log_fig, log_ax = plt.subplots(1, 1)

log_ax.errorbar(
    log_1000_over_T,
    shown_ln_sigma,
    yerr=shown_ln_sigma_err,
    color="black",
    ecolor="black",
    fmt="o-",
    capsize=10.0,
    capthick=1.0,
    label="Average interface",
)


# Configure ln(conductivity) plot
#log_ax.legend(fontsize=12)
log_ax.grid()

log_ax.set_xlabel(r"$1000/T$ (K$^{-1}$)", fontsize=14)
log_ax.set_ylabel(r"$\ln(\sigma)$", fontsize=14)
log_ax.set_title(r"ln(Ionic Conductivity) vs 1000/T", size=15)

log_ax.tick_params(axis="both", labelsize=12)

log_fig.tight_layout()


# Save and show plot
log_fig.savefig(LOG_OUTPUT_FILE, dpi=300)

print(f"Saved {LOG_OUTPUT_FILE}")

plt.show()
