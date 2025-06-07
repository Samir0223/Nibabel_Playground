import skimage
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from ndslib.image import gaussian_kernel



# img = skimage.data.astronaut()
# print(img.shape)
# print(img.dtype)

# fig, ax = plt.subplots()
# ax.imshow(img)
# ax.plot(70, 200, marker='o', markersize=5, color="white")
# p = ax.plot(200, 400, marker='o', markersize=5, color="white")

# plt.show()
# print(img[200, 70])
# print(img[400, 200])

# gray_img = rgb2gray(img)
# print(gray_img.shape)
# fig, ax = plt.subplots()
# im = ax.imshow(gray_img)
# # plt.show()

# # print(img.shape)
# # print(img.dtype)

# # print(img[200, 70])
# print(img[400, 200])