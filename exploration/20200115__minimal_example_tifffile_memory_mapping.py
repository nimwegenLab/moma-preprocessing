"""
Abstract:
The goal of this script is to test out the memory-mapped loading of OME-TIFF files.
"""

import tifffile as tff
import zarr
import matplotlib.pyplot as plt


image_path = '/home/micha/Documents/01_work/git/MM_Testing/16_thomas_20201229_glc_lac_1/MMStack/20201229_glc_lac_1_MMStack.ome.tif'


tif = tff.TiffFile(image_path)

s = 0  # position/scene/series
t = 0  # time-frame
c = 0  # channel

len(tif.series)
#Out: 32  # my tiff has 32 positions
position1_series = tif.series[s]
position1_series.axes
#Out: 'TCYX'  # my tiffs do not have a z-axis

# this is not memory-mapped (judging from memory usage)
position1 = position1_series.asarray()  # read as numpy-array
position1.shape
#Out: (480, 2, 2048, 2048)
img = position1[t, c, ...]
plt.imshow(img)
plt.show()


# this should be memory-mapped (judging from memory usage)
position1_zarr = zarr.open(position1_series.aszarr(), mode='r')  # read as zarr
position1_zarr.shape
#Out: (480, 2, 2048, 2048)
img_mm = position1_zarr[t, c, ...]
plt.imshow(img_mm)
plt.show()