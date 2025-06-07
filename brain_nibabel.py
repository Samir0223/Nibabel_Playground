import nibabel as nib
import matplotlib.pyplot as plt
epi_img = nib.load('downloads/someones_epi.nii.gz')
epi_img_data = epi_img.get_fdata()
# print(epi_img_data.shape)

def show_slices(slices):
   """ Function to display row of image slices """
   fig, axes = plt.subplots(1, len(slices))
   for i, slice in enumerate(slices):
       axes[i].imshow(slice.T, cmap="gray", origin="lower")

slice_0 = epi_img_data[26, :, :]
slice_1 = epi_img_data[:, 30, :]
slice_2 = epi_img_data[:, :, 16]
show_slices([slice_0, slice_1, slice_2])
plt.suptitle("Center slices for EPI image")  


# We collected an anatomical image in the same session. We can load that image and look at slices in the three axes:
anat_img = nib.load('downloads/someones_anatomy.nii.gz')
anat_img_data = anat_img.get_fdata()
anat_img_data.shape
(57, 67, 56)
show_slices([anat_img_data[28, :, :],
             anat_img_data[:, 33, :],
             anat_img_data[:, :, 28]])
plt.suptitle("Center slices for anatomical image")
# plt.show()

n_i, n_j, n_k = epi_img_data.shape
center_i = (n_i - 1) // 2  # // for integer division
center_j = (n_j - 1) // 2
center_k = (n_k - 1) // 2
center_i, center_j, center_k
(26, 30, 16)
center_vox_value = epi_img_data[center_i, center_j, center_k]
print(center_vox_value)