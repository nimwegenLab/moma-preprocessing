from unittest import TestCase
import os
from skimage.transform import AffineTransform, warp

class test_MomaImageProcessor(TestCase):
    data_dir = os.path.join(os.path.dirname(__file__), 'resources/data__test__moma_image_processing')

    def test__image_shift_is_correctly_detected(self):
        import numpy as np
        import tifffile as tff
        import matplotlib.pyplot as plt
        from mmpreprocesspy.moma_image_processing import MomaImageProcessor

        original_image = tff.imread(os.path.join(self.data_dir, '12_20190816_Theo_MMStack.ome.tif'))
        sut = MomaImageProcessor()
        sut._image_for_registration = original_image

        shifts_x_y = [5.7, 67.3]

        for xshift in shifts_x_y:
            for yshift in shifts_x_y:
                expected_shift = [xshift, yshift]

                shifted_image = self.support__shift_image(original_image, expected_shift)

                sut.determine_image_shift(shifted_image)
                actual_shift = (sut.horizontal_shift, sut.vertical_shift)

                np.testing.assert_array_almost_equal(expected_shift,
                                                     actual_shift,
                                                     decimal=2,
                                                     err_msg=f'expected_shift was not recovered: {expected_shift}\n'
                                                     f'actual shift: {actual_shift}')

    def support__shift_image(self, image, vector):
        transform = AffineTransform(translation=vector)
        shifted = warp(image, transform, mode='wrap', preserve_range=True)
        shifted = shifted.astype(image.dtype)
        return shifted


