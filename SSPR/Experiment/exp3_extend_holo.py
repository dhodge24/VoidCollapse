"""

The purpose of this code is the makes areas outside the circular area of the object extend radially outward. If the
image already takes up the entire computational domain, skip this step and continue. You may need to inpaint the
image if the white fields were taken too early or too late.


"""

import numpy as np
from tifffile import imread, imwrite
from SSPR.utilities import fadeoutImage, showImg, padToSize


save = True
use_PCA_corrected = True
use_imreg_corrected = False
use_inpainted_image = True
extend_image = False
plot_result = True

# For the image extension - if True
ellipse_size_y = 0.8
ellipse_size_x = 0.8
transition_length_y = 50
transition_length_x = 50
fade_to_val = None
num_segments = 250

run_wfs = "562"
run_holo = "571"

# Main directories
dir_main = ("/Users/danielhodge/Library/CloudStorage/Box-Box/BYU_CXI_Research_Team/ProjectFolders/SingleShotImaging/"
            "meclx4819/Tifs/run_data/")
dir_holo_preprocessed = "run" + run_holo + "_exp_preprocessed/"
dir_wfs_to_holo_img_reg = "run_" + run_wfs + "_to_" + run_holo + "_img_reg_exp/"
dir_ffc = "run" + run_holo + "_ffc/"

# Files to import
if use_inpainted_image:
    # TODO: Use MATLAB to apply exemplar based inpainting if needed
    # inpainted image comes from MATLAB
    tiff_holo_with_speckle_ffc = "run" + run_holo + "_exp_holos_with_speckle_FFC_inpainted.tiff"  # PCA + image reg
else:
    tiff_holo_with_speckle_ffc = "run" + run_holo + "_exp_holos_with_speckle_FFC.tiff"  # PCA + image registration


tiff_holo_only_speckle_ffc = "ffc_evt_1.tiff"  # Only image registration

# Files to save
tiff_holo_with_speckle_ffc_extended = "run" + run_holo + "_exp_holos_with_speckle_FFC_extended.tiff"

# Read in ffc image
if use_PCA_corrected:
    ffc_img = np.array(imread(dir_main + dir_holo_preprocessed + tiff_holo_with_speckle_ffc))
elif use_imreg_corrected:
    ffc_img = np.array(imread(dir_main + dir_wfs_to_holo_img_reg + dir_ffc + tiff_holo_only_speckle_ffc))
else:
    print("No valid image")

if extend_image:
    ffc_img, _ = fadeoutImage(img=ffc_img,
                              fadeMethod='ellipse',
                              ellipseSize=[ellipse_size_y, ellipse_size_x],
                              transitionLength=[transition_length_y, transition_length_x],
                              fadeToVal=fade_to_val,
                              numSegments=num_segments,
                              bottomApply=False)
    ffc_img = padToSize(img=ffc_img, outputSize=[2500, 2500], padMethod='replicate', padType='both', padValue=None)
else:
    ffc_img[np.isnan(ffc_img) | np.isinf(ffc_img)] = 0

if plot_result:
    showImg(ffc_img, clim=(0, 2))

if save:
    imwrite(dir_main + dir_holo_preprocessed + tiff_holo_with_speckle_ffc_extended, ffc_img)
