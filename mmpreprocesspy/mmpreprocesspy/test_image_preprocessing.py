from unittest import TestCase

import cv2
import numpy as np
from mmpreprocesspy.MMdata import MMData
from mmpreprocesspy.image_preprocessing import PreprocessedImageProvider

import mmpreprocesspy.dev_auxiliary_functions as aux


class TestImagePreprocessor(TestCase):
    def test__Process(self):
        data_directory = '/home/micha/Documents/git/MM_Testing/10_20190424_hi2_hi3_med2_rplN_glu_gly/MMStack/RawData/measurement'
        flatfield_directory = '/home/micha/Documents/git/MM_Testing/10_20190424_hi2_hi3_med2_rplN_glu_gly/MMStack/RawData/flatfield'

        dark_noise = 100
        gaussian_sigma = 10

        dataset = MMData(data_directory)
        flatfield = MMData(flatfield_directory)

        preprocessor = PreprocessedImageProvider(dataset, flatfield, dark_noise, gaussian_sigma)
        preprocessor.initialize()

        nr_of_colors = len(dataset.get_channels())
        image_stack = np.zeros((dataset.height, dataset.width, nr_of_colors))
        for color in range(nr_of_colors):
            image_stack[:, :, color] = dataset.get_image_fast(channel=color, frame=0, position=0)

        processed_stack = preprocessor.process_image_stack(image_stack)

        aux.show_image(processed_stack[:,:,1])
        cv2.waitKey()
