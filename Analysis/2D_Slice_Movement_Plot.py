from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from KMC.Analysis import KMCResults, Structure
import numpy as np
import matplotlib.pyplot as plt

STRUCTURE_FILE = ROOT / "Structures" / "min.xyz"
LOG_FILE = ROOT / "logs" / "benchmark_2000K.log"

WINDOW_SIZE = 10000
dt = 1e-12
q = 2.0

# Define the Z-slice parameters
Z_CENTER = 27
Z_WINDOW = 27
Z_MIN = Z_CENTER - (Z_WINDOW / 2.0)  # 27.42 Å
Z_MAX = Z_CENTER + (Z_WINDOW / 2.0)  # 37.42 Å

# Limit the maximum number of points sent to Matplotlib to prevent crashes
MAX_RENDER_POINTS = 150000  

# 2. Load the structure and your specific log file
structure = Structure.fromXYZ(str(STRUCTURE_FILE))
sim_results = KMCResults.fromFile(str(LOG_FILE))

# 3. Calculate the 3D spatial data
sigmas = [i * 1.602e-11 for i in sim_results.calcWindowedIonicConductivity(structure, WINDOW_SIZE, dt, q)]
positions = sim_results.getWindowedPositionHistories(structure, WINDOW_SIZE, include_crossings=False)

# OPTIMIZATION 1: Fast Numpy Concatenation instead of slow list appending
x_arrays, y_arrays, z_arrays, cond_arrays = [], [], [], []

for pos, sig in zip(positions, sigmas):
    x_arrays.append(pos[:, 0][:-1])
    y_arrays.append(pos[:, 1][:-1])
    z_arrays.append(pos[:, 2][:-1])
    cond_arrays.append(sig)

all_x = np.concatenate(x_arrays)
all_y = np.concatenate(y_arrays)
all_z = np.concatenate(z_arrays)
all_cond = np.concatenate(cond_arrays)

# First, isolate the data for the specific Z-slice
slice_mask = (all_z >= Z_MIN) & (all_z <= Z_MAX)
x_slice = all_x[slice_mask]
y_slice = all_y[slice_mask]
cond_slice = all_cond[slice_mask]

print(f"Total points within the Z-window: {len(cond_slice):,}")

if len(cond_slice) == 0:
    print(f"No vacancies passed through the Z-window ({Z_MIN} Å to {Z_MAX} Å).")
    sys.exit()

# OPTIMIZATION 2: FILTERING Outliers (Remove top and bottom 2%)
p2 = np.percentile(cond_slice, 0)
p98 = np.percentile(cond_slice, 100)

# Create a mask keeping only data within the 2nd and 98th percentiles
valid_mask = (cond_slice >= p2) & (cond_slice <= p98)

# Apply the mask to physically discard the outliers
filt_x = x_slice[valid_mask]
filt_y = y_slice[valid_mask]
filt_cond = cond_slice[valid_mask]

print(f"Points after discarding outliers: {len(filt_cond):,}")

# The scale is now strictly defined by the absolute min/max of the remaining data
scale_min = filt_cond.min()
scale_max = filt_cond.max()
print(f"Absolute Color Scale Bounds: {scale_min:.2e} to {scale_max:.2e} S/cm")

# OPTIMIZATION 3: DOWNSAMPLING (Protect Matplotlib from crashing)
if len(filt_cond) > MAX_RENDER_POINTS:
    step = len(filt_cond) // MAX_RENDER_POINTS
    plot_x = filt_x[::step]
    plot_y = filt_y[::step]
    plot_cond = filt_cond[::step]
    print(f"Downsampled to {len(plot_cond):,} points for 2D rendering.")
else:
    plot_x, plot_y, plot_cond = filt_x, filt_y, filt_cond

# 4. Generate the 2D Plot
fig, ax = plt.subplots(figsize=(10, 8))

# Plot the downsampled, filtered scatter points on the XY plane
scatter = ax.scatter(
    plot_x, plot_y, 
    c=plot_cond, 
    cmap='viridis', 
    vmin=scale_min, 
    vmax=scale_max, 
    alpha=0.7, 
    s=15  # Adjust marker size if necessary after downsampling
)

# Add the colorbar
cbar = fig.colorbar(scatter, ax=ax, shrink=0.5, pad=0.05)
cbar.set_label('Conductivity (S cm$^{-1}$)')

ax.set_xlabel('X Position (Å)')
ax.set_ylabel('Y Position (Å)')
ax.set_title(f'2D Cross-Section of Ionic Conductivity\nXY Plane (Z = {Z_CENTER} ± {Z_WINDOW/2} Å)')
ax.grid(True, linestyle='--', alpha=0.5)

# Force the aspect ratio to be equal so the physical space isn't stretched/distorted
ax.set_aspect('equal', adjustable='box')

plt.savefig('2d_slice_conductivity.png', dpi=600)
plt.show()