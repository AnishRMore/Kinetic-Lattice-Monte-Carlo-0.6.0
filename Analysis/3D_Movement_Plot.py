from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from KMC.Analysis import KMCResults, Structure
import numpy as np
import matplotlib.pyplot as plt

STRUCTURE_FILE = ROOT / "Structures" / "min.xyz"
LOG_FILE = ROOT / "logs" / "benchmark_2000K_seed1.log"

WINDOW_SIZE = 10000
dt = 1e-12
q = 2.0

# Limit the maximum number of points sent to Matplotlib to prevent crashes
MAX_RENDER_POINTS = 250000 

# 1. Load the structure and your specific log file
structure = Structure.fromXYZ(str(STRUCTURE_FILE))
sim_results = KMCResults.fromFile(str(LOG_FILE))

# 2. Calculate the 3D spatial data
sigmas = [i * 1.602e-11 for i in sim_results.calcWindowedIonicConductivity(structure, WINDOW_SIZE, dt, q)]
positions = sim_results.getWindowedPositionHistories(structure, WINDOW_SIZE, include_crossings=False)

# 3. OPTIMIZATION: Fast Numpy Concatenation instead of slow list appending
x_arrays, y_arrays, z_arrays, cond_arrays = [], [], [], []

for pos, sig in zip(positions, sigmas):
    x_arrays.append(pos[:, 0][:-1])
    y_arrays.append(pos[:, 1][:-1])
    z_arrays.append(pos[:, 2][:-1])
    cond_arrays.append(sig)

# Concatenate all arrays at once (orders of magnitude faster for millions of points)
all_x = np.concatenate(x_arrays)
all_y = np.concatenate(y_arrays)
all_z = np.concatenate(z_arrays)
all_cond = np.concatenate(cond_arrays)

print(f"Total calculated points: {len(all_cond):,}")

# 4. FILTERING: Remove the top and bottom 2% of outliers
p2 = np.percentile(all_cond, 0)
p98 = np.percentile(all_cond, 100)

# Create a mask keeping only data within the 2nd and 98th percentiles
valid_mask = (all_cond >= p2) & (all_cond <= p98)

# Apply the mask to physically discard the outliers
filt_x = all_x[valid_mask]
filt_y = all_y[valid_mask]
filt_z = all_z[valid_mask]
filt_cond = all_cond[valid_mask]

print(f"Points after discarding outliers: {len(filt_cond):,}")

# The scale is now strictly defined by the absolute min/max of the remaining data
scale_min = filt_cond.min()
scale_max = filt_cond.max()
print(f"Absolute Color Scale Bounds: {scale_min:.2e} to {scale_max:.2e} S/cm")

# 5. DOWNSAMPLING: Protect Matplotlib from crashing
if len(filt_cond) > MAX_RENDER_POINTS:
    step = len(filt_cond) // MAX_RENDER_POINTS
    plot_x = filt_x[::step]
    plot_y = filt_y[::step]
    plot_z = filt_z[::step]
    plot_cond = filt_cond[::step]
    print(f"Downsampled to {len(plot_cond):,} points for 3D rendering.")
else:
    plot_x, plot_y, plot_z, plot_cond = filt_x, filt_y, filt_z, filt_cond

# 6. Generate the 3D Plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Plot the downsampled, filtered scatter points
scatter = ax.scatter(
    plot_x, plot_y, plot_z, 
    c=plot_cond, 
    cmap='viridis', 
    vmin=scale_min, 
    vmax=scale_max, 
    alpha=0.6,
    s=2  # Reduced marker size to prevent overlapping blobbing with high point counts
)

# Add the colorbar
cbar = fig.colorbar(scatter, ax=ax, shrink=0.5, pad=0.1)
cbar.set_label('Conductivity (S cm$^{-1}$)')

ax.set_xlabel('X Position (Å)')
ax.set_ylabel('Y Position (Å)')
ax.set_zlabel("Z Position (Å)")
ax.set_title('3D Windowed Ionic Conductivity (Outliers Removed)')

plt.show()