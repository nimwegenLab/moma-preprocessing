from unittest import TestCase
import os
import numpy as np


class test_MicroManagerOmeTiffReader(TestCase):

    def test_getImageStack(self):
        from mmpreprocesspy.MMdata import MMData
        from mmpreprocesspy.MicroManagerOmeTiffReader import MicroManagerOmeTiffReader

        test_data_base_path = '/home/micha/Documents/01_work/git/MM_Testing'
        path = os.path.join(test_data_base_path, '16_thomas_20201229_glc_lac_1/MMStack/')
        dataset = MicroManagerOmeTiffReader(path)
        # dataset = MMData(path)

        image_stack = dataset.get_image_stack(0, 0)

        # np.save('resources/data__test__MicroManagerOmeTiffReader/expected_001.npy', image_stack)
        expected = np.load('resources/data__test__MicroManagerOmeTiffReader/expected_001.npy')

        self.assertTrue(np.all(expected == image_stack), msg="result image does not match expected image")

        image_stack

        pass
