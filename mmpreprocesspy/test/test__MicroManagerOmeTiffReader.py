from unittest import TestCase
import os
import numpy as np


class test_MicroManagerOmeTiffReader(TestCase):

    def test__get_image_stack(self):
        from mmpreprocesspy.MicroManagerOmeTiffReader import MicroManagerOmeTiffReader

        test_data_base_path = '/home/micha/Documents/01_work/git/MM_Testing'
        path = os.path.join(test_data_base_path, '16_thomas_20201229_glc_lac_1/MMStack/')
        dataset = MicroManagerOmeTiffReader(path)
        # dataset = MMData(path)

        image_stack = dataset.get_image_stack(0, 0)

        # np.save('resources/data__test__MicroManagerOmeTiffReader/expected_001.npy', image_stack)
        expected = np.load('resources/data__test__MicroManagerOmeTiffReader/expected_001.npy')

        self.assertTrue(np.all(expected == image_stack), msg="result image does not match expected image")


    def test__get_image_stack__returns_different_images_for_different_frame_indexes(self):
        from mmpreprocesspy.MicroManagerOmeTiffReader import MicroManagerOmeTiffReader

        test_data_base_path = '/home/micha/Documents/01_work/git/MM_Testing'
        path = os.path.join(test_data_base_path, '16_thomas_20201229_glc_lac_1/MMStack/')
        dataset = MicroManagerOmeTiffReader(path)

        position_index = 0
        min_frame = 00
        max_frame = 19

        for frame_index in range(min_frame, max_frame):
            current_frame = dataset.get_image_stack(frame_index=frame_index,
                                                        position_index=position_index)
            next_frame = dataset.get_image_stack(frame_index=frame_index + 1,
                                                        position_index=position_index)

            print('')
            print(f'frame {frame_index}:')
            print(f'PhC equal: {np.all(current_frame[:, :, 0] == next_frame[:, :, 0])}')
            print(f'FL equal: {np.all(current_frame[:, :, 1] == next_frame[:, :, 1])}')

        # self.assertTrue(np.all(expected == image_stack), msg="result image does not match expected image")
