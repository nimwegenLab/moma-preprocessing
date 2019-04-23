from unittest import TestCase
import skimage.transform
import matplotlib.pyplot as plt


class TestPreprocessing(TestCase):
    test_data_base_path = '/home/micha/Documents/git/MM_Testing'

    def test_split_channels_init_dataset_03(self):
        from mmpreprocesspy import preprocessing

        image_array = read_tiff_to_nparray(
            self.test_data_base_path + '/03_20180604_gluIPTG10uM_lac_lacIoe_1/first_images/Pos0/03_img_000000000_ DIA Ph3 (GFP)_000.tif')
        image_rot, angle, mincol, maxcol, channel_centers = preprocessing.split_channels_init(image_array)
        self.assertEqual(359, mincol)
        self.assertEqual(675, maxcol)
        self.assertEqual(0, angle)

    def test_split_channels_init_dataset_03_rotated(self):
        from mmpreprocesspy import preprocessing

        image_array = read_tiff_to_nparray(
            self.test_data_base_path + '/03_20180604_gluIPTG10uM_lac_lacIoe_1/first_images/Pos0/03_img_000000000_ DIA Ph3 (GFP)_000.tif')
        image_array = skimage.transform.rotate(image_array, 90, resize=True)
        image_rot, angle, mincol, maxcol, channel_centers = preprocessing.split_channels_init(image_array)
        self.assertEqual(360, mincol)  # NOTE: for some reason mincol and maxcol are shifted by 1 in comparison to test_split_channels_init_dataset_03
        self.assertEqual(676, maxcol)
        self.assertEqual(90, angle)

    def test_split_channels_init_dataset_04(self):
        from mmpreprocesspy import preprocessing

        image_array = read_tiff_to_nparray(
            self.test_data_base_path + '/04_20180531_gluIPTG5uM_lac_1/first_images/Pos0/04_img_000000000_ DIA Ph3 (GFP)_000.tif')
        image_rot, angle, mincol, maxcol, channel_centers = preprocessing.split_channels_init(image_array)
        self.assertEqual(380, mincol)
        self.assertEqual(693, maxcol)
        self.assertEqual(0, angle)

    def test_find_main_channel_orientation__returns_angle_0__for_main_channel_in_vertical_direction(self):
        from mmpreprocesspy import preprocessing

        image_array = read_tiff_to_nparray(
            self.test_data_base_path + '/04_20180531_gluIPTG5uM_lac_1/first_images/Pos0/04_img_000000000_ DIA Ph3 (GFP)_000.tif')

        angle = preprocessing.find_main_channel_orientation(image_array)

        self.assertEqual(0, angle)

    def test_find_main_channel_orientation__returns_angle_90__for_main_channel_in_horizontal_direction(self):
        from mmpreprocesspy import preprocessing

        image_array = read_tiff_to_nparray(
            self.test_data_base_path + '/04_20180531_gluIPTG5uM_lac_1/first_images/Pos0/04_img_000000000_ DIA Ph3 (GFP)_000.tif')
        image_array = skimage.transform.rotate(image_array, 90)

        angle = preprocessing.find_main_channel_orientation(image_array)

        self.assertEqual(90, angle)


def read_tiff_to_nparray(image_path):
    """Reads tiff-image and returns it as a numpy-array."""

    from PIL import Image
    import numpy as np

    image_base = Image.open(image_path)
    return np.array(image_base)


    # @staticmethod
    # def show_image(image): # this method is currently broken
    #     image.mode = 'I'
    #     im2 = image.point(lambda i: i * (1. / 256)).convert('L')
    #     im2.show()

