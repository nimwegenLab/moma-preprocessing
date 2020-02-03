from unittest import TestCase

# import matplotlib.pyplot as plt
import mmpreprocesspy.dev_auxiliary_functions as dev_aux
import numpy as np
import skimage.transform
from skimage.io import imread
from skimage.transform import AffineTransform, warp


class TestPreprocessing(TestCase):
    test_data_base_path = '/home/micha/Documents/01_work/git/MM_Testing'

    def test__find_channels_in_region_dataset_10(self):
        from mmpreprocesspy import preprocessing

        image = imread("./resources/10_20190424_hi2_hi3_med2_rplN_4_MMStack.ome-2.tif")
        centers = preprocessing.get_gl_center_positions_in_growthlane_region(image)
        self.assertEqual(18, centers[0])
        self.assertEqual(125, centers[1])

        # image_with_channel_indicators = get_image_with_hlines(image, centers)
        # plt.imshow(image_with_channel_indicators, cmap="gray")
        # plt.show()

    def test__find_channels_in_region_dataset_4(self):
        from mmpreprocesspy import preprocessing

        image = imread("./resources/04_20180531_gluIPTG5uM_lac_1_MMStack.ome-2_channel_region.tif")
        centers = preprocessing.get_gl_center_positions_in_growthlane_region(image)
        self.assertEqual(74, centers[1])
        self.assertEqual(147, centers[2])

        # image_with_channel_indicators = get_image_with_lines(image, centers)
        # plt.imshow(image_with_channel_indicators, cmap="gray")
        # plt.show()

    def test__find_channels_in_region_dataset_11(self):
        from mmpreprocesspy import preprocessing

        image = imread("./resources/rotated_channel_region.tiff")
        centers = preprocessing.get_gl_center_positions_in_growthlane_region(image)
        self.assertEqual(78, centers[0])
        self.assertEqual(185, centers[1])

        # image_with_channel_indicators = get_image_with_lines(image, centers)
        # plt.imshow(image_with_channel_indicators, cmap="gray")
        # plt.show()

    def test__find_channels_in_region_dataset_11_with_cropped_region(self):
        from mmpreprocesspy import preprocessing

        image = imread("./resources/rotated_channel_region.tiff")
        image = shift(image, [0, 1250])
        centers = preprocessing.get_gl_center_positions_in_growthlane_region(image)
        self.assertEqual(105, centers[0])
        self.assertEqual(212, centers[1])

        # image_with_channel_indicators = get_image_with_lines(image, centers)
        # plt.imshow(image_with_channel_indicators, cmap="gray")
        # plt.show()

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

    def test_create_growthlane_objects(self):
        from mmpreprocesspy import preprocessing

        regions = preprocessing.get_growthlane_rois([1, 20], 20, 50)
        pass

    def test_get_rotation_matrix(self):
        from mmpreprocesspy import preprocessing
        import cv2 as cv

        image_array = read_tiff_to_nparray(
            self.test_data_base_path + '/04_20180531_gluIPTG5uM_lac_1/first_images/Pos0/04_img_000000000_ DIA Ph3 (GFP)_000.tif')

        rotation_angle = -45
        rotation_center = (image_array.shape[1] / 2 - 0.5, image_array.shape[0] / 2 - 0.5)
        matrix = preprocessing.get_rotation_matrix(rotation_angle, rotation_center)

        image_array = cv.warpAffine(image_array, matrix, (image_array.shape[1], image_array.shape[0]))
        dev_aux.show_image(image_array)

    def test_get_translation_matrix(self):
        from mmpreprocesspy import preprocessing
        import cv2 as cv

        image_array = read_tiff_to_nparray(
            self.test_data_base_path + '/04_20180531_gluIPTG5uM_lac_1/first_images/Pos0/04_img_000000000_ DIA Ph3 (GFP)_000.tif')

        horizontal_shift = 50
        vertical_shift = 100
        matrix = preprocessing.get_translation_matrix(horizontal_shift, vertical_shift)

        image_array = cv.warpAffine(image_array, matrix, (image_array.shape[1], image_array.shape[0]))
        # dev_aux.show_image(image_array)
        # cv.waitKey()


def read_tiff_to_nparray(image_path):
    """Reads tiff-image and returns it as a numpy-array."""

    from PIL import Image
    import numpy as np

    image_base = Image.open(image_path)
    return np.array(image_base, dtype=np.uint16)


def get_image_with_hlines(channel_region_image, channel_positions):
    new_image = np.float32(channel_region_image).copy()
    new_image /= np.max(new_image)
    for pos in channel_positions:
        new_image[int(pos) - 2:int(pos) + 2, :] = 1
    return new_image


def shift(image, vector):
    transform = AffineTransform(translation=vector)
    output_shape = (image.shape[0] - vector[1], image.shape[1] - vector[0])
    shifted = warp(image, transform, mode='constant', preserve_range=True, output_shape=output_shape)
    shifted_image = shifted.astype(image.dtype)
    return shifted_image[:]
