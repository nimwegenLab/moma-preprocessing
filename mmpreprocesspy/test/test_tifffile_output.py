import os
import numpy as np
import tifffile as tff

dir = os.path.dirname(__file__)
filename = os.path.splitext(os.path.basename(__file__))[0]

output_path = os.path.join(dir, f'{filename}_result')
os.makedirs(output_path, exist_ok=True)


image_height = 301
image_width = 219
nr_of_timesteps = 10
nr_of_z_planes = 1
nr_of_channels = 3

data2 = np.float32(np.random.rand(nr_of_timesteps, nr_of_z_planes, nr_of_channels, image_height, image_width))
tff.imwrite(os.path.join(output_path, 'test_1.tif'), data2)  # NOT WHAT WE WANT: this displays in ImageJ all frames as a single channel (i.e. there is only a single slider below image in ImageJ)
tff.imwrite(os.path.join(output_path, 'test_2.tif'), data2, compression='zlib', metadata={'axes': 'TZCYX'}, imagej=True)  # WHAT WE WANT: this outputs all frames to single channel (single slider below image in ImageJ)
tff.imwrite(os.path.join(output_path, 'test_3.tif'), data2, metadata={'axes': 'TZCYX'}, imagej=True)  # WHAT WE WANT: this outputs all frames to single channel (single slider below image in ImageJ)
tff.imwrite(os.path.join(output_path, 'test_4.tif'), data2, imagej=True)

metadata = {'axes': 'TZCYX', 'channels': nr_of_channels, 'slices': 1, 'frames': 2*nr_of_timesteps}
tff.imwrite(os.path.join(output_path, 'test_5.tif'), data2, imagej=True, metadata=metadata)
tff.imwrite(os.path.join(output_path, 'test_5.tif'), data2, imagej=True, append='force', metadata=metadata)


data2_1_read_back = tff.imread(os.path.join(output_path, 'test_1.tif'))  # shape: (10, 1, 3, 301, 219)
data2_2_read_back = tff.imread(os.path.join(output_path, 'test_2.tif'))  # shape: (10, 3, 301, 219)

print("stop")