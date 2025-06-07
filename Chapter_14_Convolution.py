from ndslib.viz import imshow_with_annot
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import rgb2gray
from ndslib.image import gaussian_kernel
import skimage
from skimage.filters import gaussian
from ndslib.data import load_data
from skimage import data
from skimage import filters
from skimage.filters import sobel
from skimage.morphology import erosion
from skimage.morphology import dilation
from skimage.morphology import disk
# Gray img
small_image = np.concatenate([np.arange(10), np.arange(10, 0,-1)]).reshape((4, 5))
# imshow_with_annot(small_image)
# plt.show()

# Another example:
small_result = np.zeros(small_image.shape)

# 3 x 3 Kernel = filter 3 x 3
small_kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
# imshow_with_annot(small_kernel)
# plt.show()

padded_small_image = np.pad(small_image, 1)
# imshow_with_annot(padded_small_image)
# plt.show()

# We can start doing our convolution
neighborhood = padded_small_image[:3, :3]
weighted_neighborhood = neighborhood * small_kernel
# imshow_with_annot(neighborhood)
# imshow_with_annot(weighted_neighborhood)

conv_pixel = np.sum(weighted_neighborhood)
# Pixel is still in coordinate [0, 0] in the result array:
small_result[0, 0] = conv_pixel

# We can repeat this same sequence of operations for the next pixel: the second pixel in the top row of the image.
neighborhood = small_image[:3, 1:4]
# weighted_neighborhood = neighborhood * small_image
conv_pixel = np.sum(weighted_neighborhood)
small_result[0, 1] = conv_pixel

# We need a for loop to complete the convolution
for ii in range(small_result.shape[0]):
    for jj in range(small_result.shape[1]):
        neighborhood = padded_small_image[ii:ii+3, jj:jj+3]
        weighted_neighborhood = neighborhood * small_kernel
        conv_pixel = np.sum(weighted_neighborhood)
        small_result[ii, jj] = conv_pixel

# imshow_with_annot(small_image)
# imshow_with_annot(small_result)
# plt.show()        

img = skimage.data.astronaut()
gray_img = rgb2gray(img)
kernel = gaussian_kernel()
# fig, ax = plt.subplots()
# im = ax.imshow(kernel, cmap="gray")

result = np.zeros(gray_img.shape)
padded_gray_img = np.pad(gray_img, int(kernel.shape[0] / 2))

for ii in range(result.shape[0]):
    for jj in range(result.shape[1]):
        neighborhood = padded_gray_img[ii:ii+kernel.shape[0], jj:jj+kernel.shape[1]]
        weighted_neighborhood = neighborhood * kernel
        conv_pixel = np.sum(weighted_neighborhood)
        result[ii, jj] = conv_pixel

# fig, ax = plt.subplots(1, 2)
# ax[0].imshow(gray_img, cmap="gray")
# im = ax[1].imshow(result, cmap="gray")
# plt.show()

# fig, ax = plt.subplots()
# im = ax.imshow(gaussian(gray_img, sigma=5))
# plt.show()

brain = load_data("bold_volume")
brain1 = skimage.filters.sobel(brain)
# test = filters.sobel()
smoothed_brain = gaussian(brain1, sigma=1)
# fig, ax = plt.subplots(1, 2)
# ax[0].matshow(smoothed_brain[:, :, 10])
# im = ax[1].matshow(brain[:, :, 10])



sobel_volume = np.zeros_like(brain)

for i in range(brain.shape[2]):  # iterate over z-slices
    sobel_volume[:, :, i] = sobel(brain[:, :, i])

# fig, ax = plt.subplots(1, 2)
# ax[0].matshow(smoothed_brain[:, :, 10])
# im = ax[1].matshow(brain[:, :, 10])
shepp_logan = skimage.data.shepp_logan_phantom()
fig, ax = plt.subplots()
im = ax.matshow(shepp_logan)

noise = np.random.rand(400, 400) < 0.1
fig, ax = plt.subplots()
im = ax.matshow(shepp_logan + noise)

fig, ax = plt.subplots()
im = ax.matshow(erosion(shepp_logan + noise))


fig, ax = plt.subplots()
im = ax.matshow(dilation(erosion(shepp_logan + noise)))


fig, ax = plt.subplots()
im = ax.matshow(shepp_logan- erosion(shepp_logan))

fig, ax = plt.subplots()
im = ax.matshow(erosion(dilation(shepp_logan, selem=disk(7)), selem=disk(7)))
plt.show()