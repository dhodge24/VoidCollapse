"""

References:
    1) "Digital simulation of scalar optical diffraction: revisiting chirp function sampling criteria and consequences"
    by D. Voelz et al. (for sampling criteria)

The purpose of this code is to match the size of the intensity image for forward and backward propagation. The
sampling conditions needs to be met to obtain the correct phase reconstruction result.

"""


import numpy as np
from tifffile import imread, imwrite

from SSPR.utilities import showImg, padToSize, fadeoutImage


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

run_holo = "579"

# Directories with the data
dir_main = ("/Users/danielhodge/Library/CloudStorage/Box-Box/BYU_CXI_Research_Team/ProjectFolders/SingleShotImaging/"
            "meclx4819/Tifs/run_data/")
dir_exp = "run" + run_holo + "_exp_preprocessed/"

# Import CTF phase reconstruction
tiff_ph = "run" + run_holo + "_exp_phase_CTF.tiff"

# Save file
tiff_phase_larger_grid = "run" + run_holo + "_exp_phase_CTF_larger_grid.tiff"

# Import hologram intensity
ph = np.array(imread(dir_main + dir_exp + tiff_ph), dtype=np.float32)

ph = padToSize(img=ph, outputSize=[N_pad, N_pad], padMethod='replicate', padType='both', padValue=None)

if extend_image:
    ph, _ = fadeoutImage(img=ph,
                             fadeMethod='ellipse',
                             ellipseSize=[ellipse_size_y, ellipse_size_x],
                             transitionLength=[transition_length_y, transition_length_x],
                             fadeToVal=fade_to_val,
                             numSegments=num_segments,
                             bottomApply=False)

if plot_result:
    showImg(ph)

imwrite(dir_main + dir_exp + tiff_phase_larger_grid, ph)
