import numpy as np
from tifffile import imread, imwrite

from SSPR.utilities import showImg, padToSize, fadeoutImage


save = True

# Run number
run_holo = "577"

# Directories with the data
dir_main = ("/Users/danielhodge/Library/CloudStorage/Box-Box/BYU_CXI_Research_Team/ProjectFolders/SingleShotImaging/"
            "meclx4819/Tifs/run_data/")
dir_sim = "run" + run_holo + "_sim/"

# Import CTF phase reconstruction
tiff_ph = "run" + run_holo + "_sim_phase_CTF.tiff"

# Save file
tiff_phase_larger_grid = "run" + run_holo + "_sim_phase_CTF_larger_grid.tiff"

# Import hologram intensity
ph = np.array(imread(dir_main + dir_sim + tiff_ph), dtype=np.float32)

ph = padToSize(img=ph, outputSize=[6000, 6000], padMethod='replicate', padType='both', padValue=None)

ellipse_size_y = 0.33
ellipse_size_x = 0.33
transition_length_y = 50
transition_length_x = 50
fade_to_val = None
num_segments = None
ph, _ = fadeoutImage(img=ph,
                         fadeMethod='ellipse',
                         ellipseSize=[ellipse_size_y, ellipse_size_x],
                         transitionLength=[transition_length_y, transition_length_x],
                         fadeToVal=fade_to_val,
                         numSegments=num_segments,
                         bottomApply=False)

showImg(ph)

imwrite(dir_main + dir_sim + tiff_phase_larger_grid, ph)
