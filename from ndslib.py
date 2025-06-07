# Import neuroimaging pkgs
from ndslib.data import download_bids_dataset
from nipype.algorithms.confounds import TSNR
import nibabel as nib
import numpy as np
download_bids_dataset()

# ds001233 assigned to img_bold
img_bold = nib.load("ds001233/sub-17/ses-pre/func/sub-17_ses-pre_task-cuedSFM_run-01_bold.nii.gz")

# What kind of object is img_bold ?
# print(type(img_bold))

# tells us that the data in this file are a four-dimensional array with 
# dimensions of 96×96×66 voxels, collected in two hundred forty-one time points.
# print(img_bold.shape)

# The‘f’inthismethodnameindicates to us that NiBabel is going to take 
# the data that is stored in the file and do its best to represent it as floating point numbers
data_bold = img_bold.get_fdata()
# print(type(data_bold))

# What data type does it use?
# print(data_bold.dtype)

#  Whatshapedoesthis array have?
# print(data_bold.shape)

# computing the temporal signal-to-noise ratio (tSNR) of the measurement in each voxel.
# This is quantified as the mean signal in the voxel across time points, divided by the standard deviation of the signal across time points.
tanr_numpy = np.mean(data_bold, -1 ) / np.std(data_bold, - 1)
print(tanr_numpy)


