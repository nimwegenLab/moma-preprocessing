"""
Abstract:
The goal of this script is to test out the memory-mapped loading of OME-TIFF files.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from mmpreprocesspy.MicroManagerOmeTiffReader import MicroManagerOmeTiffReader

test_data_base_path: str = '/media/micha/T7/data_michael_mell/preprocessing_test_data/MM_Testing'
image_path = os.path.join(test_data_base_path, '16_thomas_20201229_glc_lac_1/MMStack/20201229_glc_lac_1_MMStack.ome.tif')

# image_path = '/scicore/home/nimwegen/GROUP/MM_Data/Thomas/20201229/20201229_glc_lac_1/20201229_glc_lac_1_MMStack.ome.tif'

# image_path = '/home/micha/Documents/01_work/git/MM_Testing/11_20190910_glc_spcm_1/MMStack/20190910_glc_spcm_1_MMStack.ome.tif'

# image_path = '/home/micha/Documents/01_work/git/MM_Testing/15_lis_20201119_VNG1040_AB2h_2h_1/MMStack/20201119_VNG1040_AB2h_2h_1_MMStack.ome.tif'

data = MicroManagerOmeTiffReader(image_path)


channel_index = 0
position_name = 'Pos0'
start_frame = 0

max_index = 4
grid_size = int(np.sqrt(max_index))
fig, ax = plt.subplots(grid_size, grid_size)
ax = ax.flatten()

# plot `max_index` frames of the selected position and channel
imgs = []
for ax_index, frame_index in enumerate(range(start_frame, start_frame + max_index)):
    img = data.get_image_stack(position_name, frame_index, z_slice=0)[..., channel_index]
    min = np.min(img)
    max = np.max(img)
    img = (img - min)/(max - min)
    ax[ax_index].imshow(img)
    ax[ax_index].set_title(f"S/Pos: {position_name}, 'C': {channel_index}, T: {frame_index}")
    imgs.append(img)
plt.show()

# get all channels at specified position and time-frame and show them
imgs = data.get_channel_stack(position_name, frame_index)
fig, ax = plt.subplots(1, imgs.shape[0])
for ind, ax in enumerate(ax.flatten()):
    ax.imshow(imgs[ind, ...])
plt.show()

print('DONE')
