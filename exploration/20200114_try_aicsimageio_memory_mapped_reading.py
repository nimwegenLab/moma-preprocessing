from aicsimageio import AICSImage, imread_dask

# Notes:
# * To run this script you need to install: pip install aicsimageio
# * I did not get this to run and in the end used tifffile package after I figured out how to use it.

image_path = '/home/micha/Documents/01_work/git/MM_Testing/16_thomas_20201229_glc_lac_1/MMStack/20201229_glc_lac_1_MMStack.ome.tif'
# image_path = '/scicore/home/nimwegen/GROUP/MM_Data/Thomas/20201229/20201229_glc_lac_1/20201229_glc_lac_1_MMStack.ome.tif'


img = AICSImage(image_path)

img.dask_data  # returns 6D STCZYX dask array
img_dims = img.dims  # returns string "STCZYX"
img.shape  # returns tuple of dimension sizes in STCZYX order
# img.size("STC")

# Read specified portion of dask array
lazy_s0t0 = img.get_image_dask_data("CZYX", S=0, T=0)  # returns 4D CZYX dask array
s0t0 = lazy_s0t0.compute()  # returns 4D CZYX numpy array
