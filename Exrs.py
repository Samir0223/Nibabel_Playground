import numpy as np
import matplotlib.pyplot as plt
from ndslib.data import load_data
from skimage.filters import threshold_multiotsu, try_all_threshold
from skimage.filters import try_all_threshold 
# Load the brain slice
brain = load_data("bold_volume")
slice10 = brain[:, :, 10]

# --- Manual 2-threshold segmentation (3 segments) ---
min_var = np.inf
best_thresholds = (0, 0)

unique_vals = np.unique(slice10)

# Brute-force search for two thresholds
for t1 in unique_vals:
    for t2 in unique_vals:
        if t1 >= t2:
            continue
        bg = slice10[slice10 < t1]
        mid = slice10[(slice10 >= t1) & (slice10 < t2)]
        fg = slice10[slice10 >= t2]
        if len(bg) == 0 or len(mid) == 0 or len(fg) == 0:
            continue
        var = (
            np.var(bg) * len(bg)
            + np.var(mid) * len(mid)
            + np.var(fg) * len(fg)
        )
        if var < min_var:
            min_var = var
            best_thresholds = (t1, t2)

# Apply manual thresholds
t1, t2 = best_thresholds
manual_seg = np.zeros_like(slice10)
manual_seg[slice10 >= t1] = 1
manual_seg[slice10 >= t2] = 2

# --- Use Skimage's built-in Multi-Otsu ---
otsu_thresholds = threshold_multiotsu(slice10, classes=3)
otsu_seg = np.digitize(slice10, bins=otsu_thresholds)

# --- Visualization ---
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].imshow(slice10, cmap="bone")
axs[0].set_title("Original Slice")

axs[1].imshow(manual_seg, cmap="viridis")
axs[1].set_title(f"Manual 3-class Segmentation\nThresholds: {t1:.2f}, {t2:.2f}")

axs[2].imshow(otsu_seg, cmap="viridis")
axs[2].set_title(f"Multi-Otsu Segmentation\nThresholds: {otsu_thresholds}")



fig, ax = try_all_threshold(slice10, figsize=(12, 8), verbose=False)
plt.suptitle("Comparison of Thresholding Methods")
plt.show()