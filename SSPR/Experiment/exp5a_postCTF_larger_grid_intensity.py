"""

References:
    1) "Digital simulation of scalar optical diffraction: revisiting chirp function sampling criteria and consequences"
    by D. Voelz et al. (for sampling criteria)

The purpose of this code is to extend/pad the final image such that the sampling conditions are met for forward and
backward propagation, otherwise periodic patterns and false interference occurs. This is not needed for all data, but
should be checked, otherwise that phase retrieval result will be completely wrong


"""

import numpy as np
from tifffile import imread, imwrite

from utilities import showImg, padToSize, fadeoutImage


save = True
extend_image = False
plot_result = True

N_pad = 6000  # Pad size to satisfy sampling criteria

# For the image extension - if True
ellipse_size_y = 0.33
ellipse_size_x = 0.33
transition_length_y = 50
transition_length_x = 50
fade_to_val = None
num_segments = 250

run_holo = "571"

# Directories with data
dir_main = ("/Users/danielhodge/Library/CloudStorage/Box-Box/BYU_CXI_Research_Team/ProjectFolders/SingleShotImaging/"
            "meclx4819/Tifs/run_data/")
dir_exp = "run" + run_holo + "_exp_preprocessed/"

# File to import
tiff_holo_with_speckle_ffc_extended_decon = "run" + run_holo + "_exp_holos_with_speckle_FFC_extended_decon.tiff"

# File to save
tiff_holo_with_speckle_ffc_extended_decon_larger_grid = ("run" + run_holo +
                                                         "_exp_holos_with_speckle_FFC_extended_decon_larger_grid.tiff")

# Import hologram intensity
I = np.array(imread(dir_main + dir_exp + tiff_holo_with_speckle_ffc_extended_decon), dtype=np.float32)

I = padToSize(img=I, outputSize=[N_pad, N_pad], padMethod='replicate', padType='both', padValue=None)

if extend_image:
    I, _ = fadeoutImage(img=I,
                             fadeMethod='ellipse',
                             ellipseSize=[ellipse_size_y, ellipse_size_x],
                             transitionLength=[transition_length_y, transition_length_x],
                             fadeToVal=fade_to_val,
                             numSegments=num_segments,
                             bottomApply=False)

if plot_result:
    showImg(I)

imwrite(dir_main + dir_exp + tiff_holo_with_speckle_ffc_extended_decon_larger_grid, I)
