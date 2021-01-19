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


    def get_test_data___test__get_image_stack__returns_different_images_for_different_frame_indexes(self):
        test_data_base_path = '/home/micha/Documents/01_work/git/MM_Testing'
        test_configs = []

        test_configs.append({'name': 'dataset_19',
                             'path': os.path.join(test_data_base_path, '19__dany__20201123_comlac_3conds_5/MMStack/'),
                             'position_index': 0,
                             'min_frame_index': 0,
                             'max_frame_index': 8})
        # test_configs.append({'name': 'dataset_18',
        #                      'path': os.path.join(test_data_base_path, '18__theo__20210112_ara-rha_glu-lac_1/MMStack/'),
        #                      'position_index': 0,
        #                      'min_frame_index': 0,
        #                      'max_frame_index': 5})
        test_configs.append({'name': 'dataset_17',
                             'path': os.path.join(test_data_base_path, '17_lis_20201218_VNG40_AB6min_2h_1_1/MMStack/'),
                             'position_index': 0,
                             'min_frame_index': 0,
                             'max_frame_index': 8})
        # test_configs.append({'name': 'dataset_16',
        #                      'path': os.path.join(test_data_base_path, '16_thomas_20201229_glc_lac_1/MMStack/'),
        #                      'position_index': 0,
        #                      'min_frame_index': 0,
        #                      'max_frame_index': 8})
        test_configs.append({'name': 'dataset_15',
                             'path': os.path.join(test_data_base_path, '15_lis_20201119_VNG1040_AB2h_2h_1/MMStack/'),
                             'position_index': 0,
                             'min_frame_index': 0,
                             'max_frame_index': 8})
        # test_configs.append({'name': 'dataset_14',
        #                      'path': os.path.join(test_data_base_path, '14_thomas_20201228_glc_ara_1/MMStack/'),
        #                      'position_index': 0,
        #                      'min_frame_index': 0,
        #                      'max_frame_index': 8})
        test_configs.append({'name': 'dataset_13',
                             'path': os.path.join(test_data_base_path, '13_20200128_glcIPTG_glc_1/MMStack/RawData/measurement/'),
                             'position_index': 0,
                             'min_frame_index': 0,
                             'max_frame_index': 8})
        test_configs.append({'name': 'dataset_12',
                             'path': os.path.join(test_data_base_path, '12_20190816_Theo/MMStack/'),
                             'position_index': 0,
                             'min_frame_index': 0,
                             'max_frame_index': 8})
        test_configs.append({'name': 'dataset_11',
                             'path': os.path.join(test_data_base_path, '11_20190910_glc_spcm_1/MMStack/'),
                             'position_index': 0,
                             'min_frame_index': 0,
                             'max_frame_index': 8})
        test_configs.append({'name': 'dataset_10',
                             'path': os.path.join(test_data_base_path, '10_20190424_hi2_hi3_med2_rplN_glu_gly/MMStack/RawData/measurement/'),
                             'position_index': 0,
                             'min_frame_index': 0,
                             'max_frame_index': 8})
        test_configs.append({'name': 'dataset_8',
                             'path': os.path.join(test_data_base_path, '08_20190222_LB_SpentLB_TrisEDTA_LB_1/MMStack/'),
                             'position_index': 0,
                             'min_frame_index': 0,
                             'max_frame_index': 8})
        test_configs.append({'name': 'dataset_4',
                             'path': os.path.join(test_data_base_path, '04_20180531_gluIPTG5uM_lac_1/MMStack/'),
                             'position_index': 0,
                             'min_frame_index': 0,
                             'max_frame_index': 8})

        return test_configs

    def test__get_image_stack__returns_different_images_for_different_frame_indexes(self):
        from mmpreprocesspy.MicroManagerOmeTiffReader import MicroManagerOmeTiffReader

        test_configs = self.get_test_data___test__get_image_stack__returns_different_images_for_different_frame_indexes()

        for test_config in test_configs:
            with self.subTest(test=test_config['name']):
                position_index = test_config['position_index']
                min_frame = test_config['min_frame_index']
                max_frame = test_config['max_frame_index']
                path = test_config['path']

                dataset = MicroManagerOmeTiffReader(path)

                for frame_index in range(min_frame, max_frame):
                    current_frame = dataset.get_image_stack(frame_index=frame_index,
                                                                position_index=position_index)
                    next_frame = dataset.get_image_stack(frame_index=frame_index + 1,
                                                                position_index=position_index)

                    # print('')
                    # print(f'frame {frame_index}:')
                    # print(f'PhC equal: {np.all(current_frame[:, :, 0] == next_frame[:, :, 0])}')
                    # print(f'FL equal: {np.all(current_frame[:, :, 1] == next_frame[:, :, 1])}')

                    for channel_index in range(current_frame.shape[2]):
                        self.assertFalse(np.all(current_frame[:, :, channel_index] == next_frame[:, :, channel_index]),
                                                msg=f'consecutive frames are identical: frame_index {frame_index}, channel {channel_index}')

                    # if np.all(current_frame[:, :, 0] == next_frame[:, :, 0]):
                    #     import matplotlib.pyplot as plt
                    #     plt.imshow(current_frame[:, :, 0])
                    #     plt.show()
                    #     pass

                # self.assertTrue(np.all(expected == image_stack), msg="result image does not match expected image")


