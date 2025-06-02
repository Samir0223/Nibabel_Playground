from nipype.algorithms.confounds import TSNR
import os
import nibabel as nib

# Correct full path to your file
in_file = r'C:\\Users\\samir\\ds001233\\sub-17\\ses-pre\\func\\sub-17_ses-pre_task-cuedSFM_run-01_bold.nii.gz'

# Ensure it exists
if not os.path.exists(in_file):
    raise FileNotFoundError(f"File does not exist: {in_file}")

tsnr = TSNR()
tsnr.inputs.in_file = in_file
res = tsnr.run()

#  Theoutputfile (tsnr.nii.gz)canbeimmediatelyloadedwithNiBabel:
tsnr_img = nib.load("tsnr.nii.gz")
print(tsnr_img)
tsnr_data = tsnr_img.get_fdata()
print(tsnr_data)
