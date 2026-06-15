# Import modules
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Declare constants
ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent

BARRIER_FILE = ROOT / "Barrier" / "KMC_Barriers.csv"
CONDUCTIVITY_FILE = SCRIPT_DIR / "conductivities-advanced.csv"

BARRIER_HISTOGRAM_FILE = SCRIPT_DIR / "Energy Barrier Histogram.png"
CUMULATIVE_BARRIER_FILE = SCRIPT_DIR / "Cumulative Barrier Distribution.png"
ARRHENIUS_WEIGHTED_FILE = SCRIPT_DIR / "Arrhenius Weighted Barrier Distribution.png"

KB = 8.61733e-5
N_BINS = 80
DEFAULT_TEMPERATURES = np.array([1500, 2000, 2500, 3000, 3500], dtype=float)


def load_temperatures():
    """Loads temperatures from the conductivity table, with a fallback for standalone use."""
    if not CONDUCTIVITY_FILE.exists():
        return DEFAULT_TEMPERATURES

    conductivities = pd.read_csv(CONDUCTIVITY_FILE, sep=",", index_col=0)
    temperatures = conductivities["Temperature"].to_numpy(dtype=float)
    temperatures = temperatures[np.isfinite(temperatures)]

    if len(temperatures) == 0:
        return DEFAULT_TEMPERATURES

    return np.unique(temperatures)


# Load barrier data
barriers = pd.read_csv(BARRIER_FILE)
energy_barriers = barriers["Energy Barrier"].to_numpy(dtype=float)
energy_barriers = energy_barriers[np.isfinite(energy_barriers)]
temperatures = load_temperatures()


# Plot energy barrier histogram
fig, ax = plt.subplots(1, 1)

ax.hist(
    energy_barriers,
    bins=N_BINS,
    color="black",
    alpha=0.8,
)

ax.grid()
ax.set_xlabel("Energy Barrier (eV)", fontsize=14)
ax.set_ylabel("Transition Count", fontsize=14)
ax.set_title("Energy Barrier Distribution", fontsize=15)
ax.tick_params(axis="both", labelsize=12)

fig.tight_layout()
fig.savefig(BARRIER_HISTOGRAM_FILE, dpi=300)
print(f"Saved {BARRIER_HISTOGRAM_FILE}")


# Plot cumulative barrier distribution
sorted_barriers = np.sort(energy_barriers)
cumulative_fraction = np.arange(1, len(sorted_barriers) + 1) / len(sorted_barriers)

fig, ax = plt.subplots(1, 1)

ax.plot(
    sorted_barriers,
    cumulative_fraction,
    color="black",
    linewidth=2.0,
)

ax.grid()
ax.set_xlabel("Energy Barrier (eV)", fontsize=14)
ax.set_ylabel("Cumulative Fraction of Transitions", fontsize=14)
ax.set_title("Cumulative Energy Barrier Distribution", fontsize=15)
ax.tick_params(axis="both", labelsize=12)

fig.tight_layout()
fig.savefig(CUMULATIVE_BARRIER_FILE, dpi=300)
print(f"Saved {CUMULATIVE_BARRIER_FILE}")


# Plot Arrhenius-weighted barrier distribution
bin_edges = np.linspace(energy_barriers.min(), energy_barriers.max(), N_BINS + 1)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

fig, ax = plt.subplots(1, 1)

for T in temperatures:
    weights = np.exp(-energy_barriers / (KB * T))
    weighted_counts, _ = np.histogram(
        energy_barriers,
        bins=bin_edges,
        weights=weights,
    )

    total_weight = np.sum(weighted_counts)
    if total_weight == 0.0:
        continue

    weighted_fraction = weighted_counts / total_weight

    ax.plot(
        bin_centers,
        weighted_fraction,
        linewidth=2.0,
        label=f"{int(T)} K",
    )

ax.grid()
ax.legend(fontsize=11)
ax.set_xlabel("Energy Barrier (eV)", fontsize=14)
ax.set_ylabel("Fraction of Arrhenius Weight", fontsize=14)
ax.set_title("Arrhenius-Weighted Barrier Distribution", fontsize=15)
ax.tick_params(axis="both", labelsize=12)

fig.tight_layout()
fig.savefig(ARRHENIUS_WEIGHTED_FILE, dpi=300)
print(f"Saved {ARRHENIUS_WEIGHTED_FILE}")


plt.show()
