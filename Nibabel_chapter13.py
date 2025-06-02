# Importing libraries for neuro imaging
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from nibabel.processing import resample_from_to

img_bold = nib.load("ds001233/sub-17/ses-pre/func/sub-17_ses-pre_task-cuedSFM_run-01_bold.nii.gz")

data_bold = img_bold.get_fdata()
data_bold_t0 = data_bold[:, :, :, 0]
# print(data_bold)

img_t1 = nib.load("ds001233/sub-17/ses-pre/anat/sub-17_ses-pre_T1w.nii.gz")
data_t1 = img_t1.get_fdata()

# print(img_t1.shape)



# Example: coordinate in RAS (in mm)
coord_ras = np.array([1, 0, 0, 1])  # Add 1 for homogeneous coordinates

# If using LAS orientation, flip X
coord_las = np.array([-1, 0, 0, 1])

# Get the inverse of the affine
affine_inv = np.linalg.inv(img_t1.affine)

# Convert real-world coord to voxel index
voxel_index = affine_inv @ coord_las

# Drop the homogeneous part
i, j, k = voxel_index[:3]
i, j, k = np.round(voxel_index[:3]).astype(int)
t = i, j, k

# # Plot the T1 slice at that voxel's Z-location (or Y or X as desired)
# fig, ax = plt.subplots(1, 2, figsize=(10, 5))

# # Show a slice of the BOLD data
# ax[0].matshow(data_bold_t0[:, :, data_bold_t0.shape[-1]//2], cmap='gray')
# ax[0].set_title("BOLD Slice (t=0)")

# # Show the anatomical slice at k (Z slice)
# ax[1].matshow(data_t1[:, :, k], cmap='gray')
# ax[1].plot(j, i, 'ro')  # Mark the point (i, j) in the slice at z = k
# ax[1].set_title("T1 Slice with Transformed Coordinate")

# plt.tight_layout()
# plt.show()

fig, ax = plt.subplots(1, 3)
ax[0].matshow(data_bold_t0[:, :, 20])
ax[1].matshow(data_bold_t0[:, :, data_bold_t0.shape[-1]//2])
ax[2].matshow(data_bold_t0[:, :, 46])
fig.set_tight_layout("tight")
# plt.tight_layout()
# plt.show()

# Matrix Multiplication Function in Python
def matrix_multiply(A, B):
    # Get dimensions
    rows_A = len(A)
    cols_A = len(A[0])
    rows_B = len(B)
    cols_B = len(B[0])

    # Check matrix multiplication validity
    if cols_A != rows_B:
        raise ValueError("Number of columns in A must match number of rows in B")

    # Initialize result matrix with zeros
    result = [[0 for _ in range(cols_B)] for _ in range(rows_A)]

    # Compute each element
    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):  # or rows_B
                result[i][j] += A[i][k] * B[k][j]

    return result
#  Example Usage:
A = [[1, 2, 3],
     [4, 5, 6]]

B = [[7, 8],
     [9, 10],
     [11, 12]]

C = matrix_multiply(A, B)
# print(C)

shape = data_bold.shape[:3]  # Ignore time dimension
last_voxel = [s - 1 for s in shape]


affine_t1 = img_t1.affine
affine_bold = img_bold.affine

# print(affine_t1 @ np.array([0, 0, 0, 1]))
# 1. Where is data_bold[0, 0, 0] in scanner space?
corner_voxel = np.array([0, 0, 0, 1])
corner_phys = affine_bold @ corner_voxel
# print("data_bold[0,0,0] is at:", corner_phys)

# 2. Where is data_bold[-1, -1, -1]?
shape = data_bold.shape[:3]
last_voxel = np.array([shape[0]-1, shape[1]-1, shape[2]-1, 1])
last_phys = affine_bold @ last_voxel
# print("data_bold[max,max,max] is at:", last_phys)

# First, we calculate the location of this voxel in the [i, j, k]coordinatesofthefMRIimagespace:
central_bold_voxel = np.array([img_bold.shape[0]//2,
img_bold.shape[1]//2,
img_bold.shape[2]//2, 1])

# Next, we move this into the scanner space, using the affine transform of the fMRI image:
bold_center_scanner = affine_bold @ central_bold_voxel
# print(bold_center_scanner)
#  Next,weusetheinverseoftheT1-weightedaffinetomovethiscoordinateintothespace of theT1-weightedimagespace.
bold_center_t1 = np.linalg.inv(affine_t1) @ bold_center_scanner
# print(bold_center_t1)

#resample the T1-weighted image to the space and resolution of the fMRI data:
img_t1_resampled = resample_from_to(img_t1, (img_bold.shape[:3], img_bold.affine))
#  The image that results from this computation has the shape of the fMRI data, as well as the affine of the fMRI data:
# print(img_t1_resampled.shape)
# print(img_t1_resampled.affine == img_bold.affine)
#  And,importantly, if we extract the data from this image, we can showthatthetwodata modalities are now well aligned:
data_t1_resampled = img_t1_resampled.get_fdata()
fig, ax = plt.subplots(1, 2)
ax[0].matshow(data_bold_t0[:, :, data_bold_t0.shape[-1]//2])
im = ax[1].matshow(data_t1_resampled[:, :, data_t1_resampled.shape[-1]//2])
# plt.show()
#Finally, if you would like to write out this result into another NIfTI file to be used later in someother computation, you can do so byusing NiBabel’s save function:
nib.save(nib.Nifti1Image(data_t1_resampled, img_t1_resampled.affine), 't1_resampled.nii.gz')

# Resample BOLD data (t=0) into T1-weighted image space
img_bold_t0 = nib.Nifti1Image(data_bold_t0, affine_bold)
img_bold_resampled = resample_from_to(img_bold_t0, img_t1)

# Extract the resampled data
data_bold_resampled = img_bold_resampled.get_fdata()

# Visualize to check alignment in T1 space
fig, ax = plt.subplots(1, 2)
ax[0].matshow(data_t1[:, :, data_t1.shape[-1]//2], cmap='gray')
ax[0].set_title("Original T1")
ax[1].matshow(data_bold_resampled[:, :, data_bold_resampled.shape[-1]//2], cmap='gray')
ax[1].set_title("BOLD (t=0) resampled to T1 space")
plt.show()

# Save the resampled BOLD image
nib.save(img_bold_resampled, "bold_t0_resampled_to_t1.nii.gz")













