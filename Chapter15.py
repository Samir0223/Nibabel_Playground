from ndslib import load_data
import matplotlib.pyplot as plt
import numpy as np
from skimage.data import camera
from skimage import data
from skimage.filters import threshold_otsu

brain = load_data("bold_volume")
slice10 = brain[:, :, 10]
# The input image.
image = data.camera(slice10)
thresh = threshold_otsu(image)

fig, ax = plt.subplots()
im = ax.imshow(slice10, cmap="bone")

mean = np.mean(slice10)
fig, ax = plt.subplots()
p= ax.hist(slice10.flat)


fig, ax = plt.subplots()
ax.hist(slice10.flat)
p = ax.axvline(mean, linestyle='dashed')



segmentation = np.zeros_like(slice10)
segmentation[slice10 > mean] = 1
fig, ax = plt.subplots()
ax.imshow(slice10, cmap="bone")
im = ax.imshow(segmentation, alpha=0.5)


min_intraclass_variance = np.inf

for candidate in np.unique(slice10):
    background = slice10[slice10 < candidate]
    foreground = slice10[slice10 >= candidate]
    if len(foreground) and len(background):
        foreground_variance = np.var(foreground) * len(foreground)
        background_variance = np.var(background) * len(background)
        intraclass_variance = foreground_variance + background_variance
        if intraclass_variance < min_intraclass_variance:
            min_intraclass_variance = intraclass_variance
            threshold = candidate

mean = np.mean(slice10)
fig, ax = plt.subplots()
ax.hist(slice10.flat)
ax.axvline(mean, linestyle='dashed')
p= ax.axvline(threshold, linestyle='dotted')


segmentation = np.zeros_like(slice10)
segmentation[slice10 > threshold] = 1
fig, ax = plt.subplots()
ax.imshow(slice10, cmap="bone")
p= ax.imshow(segmentation, alpha=0.5)
plt.show()