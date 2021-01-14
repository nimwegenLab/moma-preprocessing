"""
Abstract:
The goal of this script is to test out the memory-mapped loading of OME-TIFF files.
"""

import matplotlib.pyplot as plt
import numpy as np
from MicroManagerTiffReader import MicroManagerTiffReader
import tifffile as tff
import zarr

# image_path = '/home/micha/Documents/01_work/git/MM_Testing/16_thomas_20201229_glc_lac_1/MMStack/20201229_glc_lac_1_MMStack.ome.tif'
# image_path = '/scicore/home/nimwegen/GROUP/MM_Data/Thomas/20201229/20201229_glc_lac_1/20201229_glc_lac_1_MMStack.ome.tif'

image_path = '/home/micha/Documents/01_work/git/MM_Testing/11_20190910_glc_spcm_1/MMStack/20190910_glc_spcm_1_MMStack.ome.tif'

# image_path = '/home/micha/Documents/01_work/git/MM_Testing/15_lis_20201119_VNG1040_AB2h_2h_1/MMStack/20201119_VNG1040_AB2h_2h_1_MMStack.ome.tif'

# image_sequence = tff.TiffSequence(image_path, axesorder='STCZYX')
# image_sequence.shape

store = tff.imread(image_path, aszarr=True)
# store = tff.imread(image_path, aszarr=True, axesorder='STCZYX')
z = zarr.open(store, mode='r')

data = MicroManagerTiffReader(image_path)

# frame_index = 0

data.get_copy_of_page(page=510)

###
# exit(0)

#######################
page_nr = 1
# plt.imshow(tff.memmap(image_path, page=page_nr, mode='r'))
# plt.show()

import warnings

tif = tff.TiffFile(image_path)

len(tif.series)
pos1 = tif.series[0]
img1 = pos1[0, 0, 0, 0][0].asarray()
plt.imshow(img1)
plt.show()


tif = tff.TiffFile(image_path)
pos1 = tif.series[0]


t = 4
c = 0
full_image = pos1.asarray()
img = full_image[t, c, ...]
plt.imshow(img)
plt.show()




tif.series[1].shape

# with tff.TiffFile(image_path) as tif:
#     data = tif.pages[5].asarray()

#######################



# for ind in range(100):
#     try:
#         tff.memmap(image_path, page=ind, mode='r')
#     except:
#         print(f"Failed frame: {ind}")

#######################

channel_index = 0
position_index = 1
start_frame = 0

max_index = 4
grid_size = int(np.sqrt(max_index))
fig, ax = plt.subplots(grid_size, grid_size)
ax = ax.flatten()



imgs = []
for ax_index, frame_index in enumerate(range(start_frame, start_frame + max_index)):
    img = data.get_image(frame_index, channel_index, position_index)
    page_nr = data.calculate_page_nr(frame_index, channel_index, position_index)
    min = np.min(img)
    max = np.max(img)
    img = (img - min)/(max - min)
    ax[ax_index].imshow(img)
    ax[ax_index].set_title(f"page_nr: {page_nr}")
    imgs.append(img)
plt.show()
pass

print('test')
