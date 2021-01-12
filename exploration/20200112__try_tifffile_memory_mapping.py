"""
Abstract:
The goal of this script is to test out the memory-mapped loading of OME-TIFF files.
"""

import matplotlib.pyplot as plt
from MMDataNew import MMDataNew

image_path = '/home/micha/Documents/01_work/git/MM_Testing/16_thomas_20201229_glc_lac_1/MMStack/20201229_glc_lac_1_MMStack.ome.tif'

data = MMDataNew(image_path)

frame_index = 0
channel_index = 0
position_index = 0
img = data.get_image_fast(frame_index, channel_index, position_index)

# img = tff.memmap(image_path, mode='r', page=0)

# img.shape

plt.imshow(img)
plt.show()

print('test')
