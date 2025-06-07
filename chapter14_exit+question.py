from skimage.morphology import opening, closing, ball
from ndslib.data import load_data
import matplotlib.pyplot as plt
import numpy as np

brain = load_data("bold_volume")  # 3D (x, y, z)

# Work on a 2D slice
slice_2d = brain[:, :, 10]

# Try removing skull using closing followed by opening
from skimage.morphology import disk

cleaned_slice = opening(closing(slice_2d, disk(5)), disk(5))

fig, ax = plt.subplots(1, 2)
ax[0].imshow(slice_2d, cmap="gray")
ax[0].set_title("Original Slice")
ax[1].imshow(cleaned_slice, cmap="gray")
ax[1].set_title("After Morphological Cleaning")


# Apply opening to entire 3D volume to remove skull
brain_cleaned = opening(brain, ball(2))  # Try ball(2), ball(3), etc.

# View slice
fig, ax = plt.subplots()
ax.imshow(brain_cleaned[:, :, 10], cmap="gray")
ax.set_title("3D Morphological Opening on Slice 10")
plt.show()