import numpy as np
from tifffile import imread, imwrite

from SSPR.utilities import showImg, padToSize, fadeoutImage


save = True

run_holo = "579"

# Directories with the data
dir_main = ("/Users/danielhodge/Library/CloudStorage/Box-Box/BYU_CXI_Research_Team/ProjectFolders/SingleShotImaging/"
            "meclx4819/Tifs/run_data/")
dir_sim = "run" + run_holo + "_sim/"

# File to import
tiff_holo_with_speckle_ffc_extended_decon = "run" + run_holo + "_sim_holos_with_speckle_FFC_extended_decon.tiff"

# File to save
tiff_holo_with_speckle_ffc_extended_decon_larger_grid = ("run" + run_holo +
                                                         "_sim_holos_with_speckle_FFC_extended_decon_larger_grid.tiff")

# Import hologram intensity
I = np.array(imread(dir_main + dir_sim + tiff_holo_with_speckle_ffc_extended_decon), dtype=np.float32)

I = padToSize(img=I, outputSize=[6000, 6000], padMethod='replicate', padType='both', padValue=None)

ellipse_size_y = 0.33
ellipse_size_x = 0.33
transition_length_y = 50
transition_length_x = 50
fade_to_val = None
num_segments = 250
I, _ = fadeoutImage(img=I,
                         fadeMethod='ellipse',
                         ellipseSize=[ellipse_size_y, ellipse_size_x],
                         transitionLength=[transition_length_y, transition_length_x],
                         fadeToVal=fade_to_val,
                         numSegments=num_segments,
                         bottomApply=False)


# result = g
showImg(I)

imwrite(dir_main + dir_sim + tiff_holo_with_speckle_ffc_extended_decon_larger_grid, I)
