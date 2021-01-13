"""
Abstract:
The goal of this script is to test out the memory-mapped loading of OME-TIFF files.
"""

import matplotlib.pyplot as plt
import numpy as np
from MicroManagerTiffReader import MicroManagerTiffReader

image_path = '/home/micha/Documents/01_work/git/MM_Testing/16_thomas_20201229_glc_lac_1/MMStack/20201229_glc_lac_1_MMStack.ome.tif'

data = MicroManagerTiffReader(image_path)

# frame_index = 0


#######################
import tifffile as tff
page_nr = 1
# plt.imshow(tff.memmap(image_path, page=page_nr, mode='r'))
# plt.show()

tif = tff.TiffFile(image_path)

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
start_index = 0

# img = data.get_image(start_index, channel_index, position_index)
img = tif.pages[8].asarray()
plt.imshow(img)
plt.show()

# exit(0)

max_index = 9
grid_size = int(np.sqrt(max_index))
fig, ax = plt.subplots(grid_size, grid_size)
ax = ax.flatten()

imgs = []
for ax_index, frame_index in enumerate(range(start_index, start_index + max_index)):
    # img = data.get_image(frame_index, channel_index, position_index)
    img = tif.pages[frame_index].asarray()
    img = img.astype(dtype=np.float)
    min = np.min(img)
    max = np.max(img)
    img = (img - min)/(max - min)
    ax[ax_index].imshow(img)
    ax[ax_index].set_title(f"page: {frame_index}")
    imgs.append(img)
plt.show()
pass


# img = tff.memmap(image_path, mode='r', page=0)

# img.shape

# plt.imshow(img)
# plt.show()

print('test')
