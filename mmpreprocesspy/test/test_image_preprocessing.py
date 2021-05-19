from unittest import TestCase

import cv2
import numpy as np
from mmpreprocesspy.MicroManagerOmeTiffReader import MicroManagerOmeTiffReader
from mmpreprocesspy.image_preprocessing import ImagePreprocessor

import mmpreprocesspy.dev_auxiliary_functions as aux


class TestImagePreprocessor(TestCase):
    def test__Process(self):
        data_directory = '/home/micha/Documents/01_work/git/MM_Testing/10_20190424_hi2_hi3_med2_rplN_glu_gly/MMStack/RawData/measurement'
        flatfield_directory = '/home/micha/Documents/01_work/git/MM_Testing/10_20190424_hi2_hi3_med2_rplN_glu_gly/MMStack/RawData/flatfield'

        dark_noise = 100
        gaussian_sigma = 10

        dataset = MicroManagerOmeTiffReader(data_directory)
        flatfield = MicroManagerOmeTiffReader(flatfield_directory)

        preprocessor = ImagePreprocessor(dataset, flatfield, dark_noise, gaussian_sigma)

        roi_shape = (dataset.get_image_height(), dataset.get_image_width())
        preprocessor.calculate_flatfields(roi_shape)

        nr_of_colors = len(dataset.get_channels())
        image_stack = np.zeros((dataset.height, dataset.width, nr_of_colors))
        for color in range(1, nr_of_colors):
            image_stack[:, :, color] = dataset.get_image_stack(frame_index=0, position_index=6, z_slice=0)[..., color]

        images_to_correct = image_stack[:, :, 1:]
        processed_stack = preprocessor.process_image_stack(images_to_correct)

        # np.save('./resources/data__test_image_preprocessing_py/test__Process__expected_image_00.npy', processed_stack)  # this is for updating the expected data
        expected = np.load('resources/data__test_image_preprocessing_py/test__Process__expected_image_00.npy')

        self.assertTrue(np.all(expected == processed_stack))
        #
        # aux.show_image(processed_stack[:,:,0])
        # cv2.waitKey()

    # def test__flat_field_correction_works_correctly(self):
    # NOTE-2019-05-17: this is test for checking, if the flatfield works correctly. This was done using the same flatfield dataset
    # as flatfield and dataset to correct. It showed that the code works correctly. However we need to comment out this code,
    # because the class ImagePreprocessor expects that the input dataset that is to be corrected has one more color-channel
    # than the flatflield (which hold the fluorescence data). This causes this test to fail, when using the same dataset as
    # as flatfield and data to correct.
    #
    #     data_directory = '/home/micha/Documents/git/MM_Testing/10_20190424_hi2_hi3_med2_rplN_glu_gly/MMStack/RawData/flatfield'
    #     flatfield_directory = '/home/micha/Documents/git/MM_Testing/10_20190424_hi2_hi3_med2_rplN_glu_gly/MMStack/RawData/flatfield'
    #
    #     dark_noise = 100
    #     gaussian_sigma = 10
    #
    #     dataset = MMData(data_directory)
    #     flatfield = MMData(flatfield_directory)
    #
    #     preprocessor = ImagePreprocessor(dataset, flatfield, dark_noise, gaussian_sigma)
    #     preprocessor.initialize()
    #
    #     nr_of_colors = len(dataset.get_channels())
    #     image_stack = np.zeros((dataset.height, dataset.width, nr_of_colors))
    #     for color in range(nr_of_colors):
    #         image_stack[:, :, color] = dataset.get_image_fast(channel=color, frame=0, position=0)
    #     # image_stack[:, :, 1] = dataset.get_image_fast(channel=color, frame=0, position=0)
    #
    #     image_stack_bkp = image_stack.copy()
    #     processed_stack = preprocessor.process_image_stack(image_stack)
    #
    #     aux.show_image(processed_stack[:,:,1])
    #     cv2.waitKey()
