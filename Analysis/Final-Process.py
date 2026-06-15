# Import modules
from pathlib import Path
from re import search
import sys

import numpy as np
import pandas as pd


# Make the project package importable when this script is run from Analysis.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from KMC.Analysis import KMCResults, Structure, weighted_average


# Declare constants
STRUCTURE_FILE = ROOT / "Structures" / "min.xyz"
LOG_DIR = ROOT / "logs"
OUTPUT_FILE = Path(__file__).resolve().parent / "conductivities-advanced.csv"

LOG_GLOB = "*.log"
WINDOW_SIZE = 10000
center_z = 27.0
dt = 1e-12
q = 2.0


# Load structure data
structure = Structure.fromXYZ(str(STRUCTURE_FILE))
zlo, zhi = structure.bounds.zlo, structure.bounds.zhi
lz = structure.lz

# Check if bounds were automatically detected
if zlo is None or zhi is None:
    raise Exception(
        "Unable to proceed: bounding box data not detected in .xyz structure file. "
        "Try adding the following data to the comment line (second line of .xyz file): "
        "xlo: [lower x], xhi: [upper x], ylo: [lower y], yhi: [upper y], zlo: [lower z], zhi: [upper z]"
    )


# Locate log files
log_files = sorted(LOG_DIR.glob(LOG_GLOB))

if len(log_files) == 0:
    raise FileNotFoundError(f"No log files matching {LOG_GLOB!r} were found in {LOG_DIR}")


# Construct dataframe to dump results to
results = pd.DataFrame(
    columns=[
        "Temperature",
        "Log File",
        "Sigma (Physical)",
        "Sigma Error (Physical)",
        "Sigma (Periodic)",
        "Sigma Error (Periodic)",
    ]
)


# Declare helper functions
def classify_side(z_pos):
    """Determines whether a vacancy is closer to the interface across the periodic boundary."""
    z = ((z_pos - zlo) % lz) + zlo
    dist_to_center = np.abs(z - center_z)
    dist_to_periodic1 = np.abs(z - zlo)
    dist_to_periodic2 = np.abs(z - zhi)
    dist_to_periodic = np.min([dist_to_periodic1, dist_to_periodic2], axis=0)
    return dist_to_periodic < dist_to_center


def temperature_from_file(log_file, sim_results):
    """Prefer the temperature stored in the log header, then fall back to the file name."""
    if sim_results.T not in (None, 0.0):
        return int(sim_results.T)

    match = search(r"(\d+)K", log_file.name)
    if match is None:
        raise ValueError(f"Could not determine temperature for {log_file.name}")

    return int(match.group(1))


# Run analysis for each log file
for log_file in log_files:
    print(f"Processing {log_file.name}")

    # Open this log file
    sim_results = KMCResults.fromFile(str(log_file))
    T = temperature_from_file(log_file, sim_results)

    if len(sim_results) < WINDOW_SIZE:
        print(f"Skipping {log_file.name}: fewer than {WINDOW_SIZE} KMC steps")
        continue

    # Calculate ionic conductivity. Ensure units are correct.
    sigmas = [
        sigma * 1.602e-11
        for sigma in sim_results.calcWindowedIonicConductivity(structure, WINDOW_SIZE, dt, q)
    ]
    windowed_positions = sim_results.getWindowedPositionHistories(structure, WINDOW_SIZE)

    # Disentangle which interface the conductivities belong to
    periodic_sigma_all = []
    physical_sigma_all = []
    periodic_sigma_all_err = []
    physical_sigma_all_err = []

    for positions, sigma in zip(windowed_positions, sigmas):
        # Classify the side of the structure these values correspond to
        z = positions.T[2]
        side = classify_side(z[:-1])

        # Compute averages and errors for each side
        if np.sum(side) > 0:
            periodic_sigma, periodic_sigma_err = np.mean(sigma[side]), np.std(sigma[side])
        else:
            periodic_sigma, periodic_sigma_err = np.nan, np.nan

        if np.sum(~side) > 0:
            physical_sigma, physical_sigma_err = np.mean(sigma[~side]), np.std(sigma[~side])
        else:
            physical_sigma, physical_sigma_err = np.nan, np.nan

        periodic_sigma_all.append(periodic_sigma)
        physical_sigma_all.append(physical_sigma)
        periodic_sigma_all_err.append(periodic_sigma_err)
        physical_sigma_all_err.append(physical_sigma_err)

    # Compute weighted average across the vacancies
    avg_periodic_sigma, avg_periodic_sigma_err = weighted_average(
        np.array(periodic_sigma_all),
        np.array(periodic_sigma_all_err),
    )
    avg_physical_sigma, avg_physical_sigma_err = weighted_average(
        np.array(physical_sigma_all),
        np.array(physical_sigma_all_err),
    )

    # Add to results table
    row_name = log_file.stem
    results.loc[row_name, "Sigma (Periodic)"] = avg_periodic_sigma
    results.loc[row_name, "Sigma Error (Periodic)"] = avg_periodic_sigma_err
    results.loc[row_name, "Sigma (Physical)"] = avg_physical_sigma
    results.loc[row_name, "Sigma Error (Physical)"] = avg_physical_sigma_err
    results.loc[row_name, "Temperature"] = T
    results.loc[row_name, "Log File"] = log_file.name


# ==========================================
# AGGREGATE RESULTS ACROSS SEEDS
# ==========================================

# 1. Ensure columns are numeric (Pandas sometimes stores them as objects when appending rows)
numeric_cols = [
    "Temperature", 
    "Sigma (Physical)", 
    "Sigma Error (Physical)", 
    "Sigma (Periodic)", 
    "Sigma Error (Periodic)"
]
results[numeric_cols] = results[numeric_cols].apply(pd.to_numeric)

# 2. Group by Temperature and calculate the mean and standard deviation
aggregated_results = results.groupby("Temperature").agg({
    "Sigma (Physical)": ["mean", "std"],
    "Sigma (Periodic)": ["mean", "std"]
})

# 3. Flatten the column names
aggregated_results.columns = [
    "Mean Sigma (Physical)", 
    "Std Sigma (Physical)",
    "Mean Sigma (Periodic)", 
    "Std Sigma (Periodic)"
]

# 4. Save both the aggregated summary and the raw per-seed data
AGG_OUTPUT_FILE = Path(__file__).resolve().parent / "conductivities-advanced-summary.csv"

# Save the new aggregated table
aggregated_results.to_csv(AGG_OUTPUT_FILE)
print(f"Saved aggregated (mean/std) results to {AGG_OUTPUT_FILE}")

# Save the original raw table
results.to_csv(OUTPUT_FILE, index=False)
print(f"Saved raw per-seed results to {OUTPUT_FILE}")