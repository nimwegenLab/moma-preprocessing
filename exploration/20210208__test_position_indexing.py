from unittest import TestCase
import os
import numpy as np
import re
import matplotlib.pyplot as plt
from mmpreprocesspy.MicroManagerOmeTiffReader import MicroManagerOmeTiffReader

test_dataset_path = '/home/micha/Documents/01_work/git/MM_Testing/21__dany__20190515_hi1_med1_med2_rpmB_glu_gly_7/20190515_hi1_med1_med2_rpmB_glu_gly_7_MMStack.ome.tif'


# output_path = os.path.join(os.path.basename(__file__), '/data/20210208__test_position_indexing/output/')
# path = os.path.join(test_dataset_path, '16_thomas_20201229_glc_lac_1/MMStack/')
dataset = MicroManagerOmeTiffReader(test_dataset_path)
# dataset = MMData(path)
position_names = dataset._position_names

position_inds = []
for name in dataset._position_names:
    position_inds.append(int(re.match('Pos[0]*(\d+)', name)[1]))

# position_inds = list(range(len(position_names)))
frame_nr = 0
for name, ind in zip(position_names, position_inds):
    image_stack = dataset.get_image_stack(position_index=ind, frame_index=frame_nr, z_slice=0)
    plt.imshow(image_stack[:, :, 0], cmap='gray')
    plt.title(f'Position: {name}')
    plt.show()

# np.save('resources/data__test__MicroManagerOmeTiffReader/expected_001.npy', image_stack)
# expected = np.load('resources/data__test__MicroManagerOmeTiffReader/expected_001.npy')

