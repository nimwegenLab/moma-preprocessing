from unittest import TestCase

# import matplotlib.pyplot as plt
import mmpreprocesspy.dev_auxiliary_functions as dev_aux
import numpy as np
import skimage.transform
from skimage.io import imread
from skimage.transform import AffineTransform, warp
from parameterized import parameterized


class TestPreprocessing(TestCase):
    test_data_base_path = '/home/micha/Documents/01_work/git/MM_Testing'

    def get_tests__test__find_channel_regions(self):
        tests = []
        tests.append({'name': 'dataset_16',
                      'path': "./resources/data__test_preprocessing_py/16_thomas__20201229_glc_lac_1_MMStack__Pos0__rotated.tif",
                      'angle': 0,
                      'glt': 200,
                      'centers': [538, 1343]})
        tests.append({'name': 'dataset_15',
                      'path': "./resources/data__test_preprocessing_py/15_lis__20201119_VNG1040_AB2h_2h_1_MMStack.ome.tif",
                      'angle': 90,
                      'glt': 200,
                      'centers': [573, 1409]})
        tests.append({'name': 'dataset_14',
                      'path': "./resources/data__test_preprocessing_py/14_thomas_20201228_glc_ara_1__Pos0__rotated.tif",
                      'angle': -.5,
                      'glt': 300,
                      'centers': [489, 1315]})
        tests.append({'name': 'dataset_13',
                      'path': "./resources/data__test_preprocessing_py/13_20200128_glcIPTG_glc_1__MMStack.ome.tif",
                      'angle': 0,
                      'glt': 200,
                      'centers': [465, 752]})
        tests.append({'name': 'dataset_12',
                      'path': "./resources/data__test_preprocessing_py/12_20190816_Theo_MMStack.ome.tif",
                      'angle': 0,
                      'glt': 200,
                      'centers': [526]})
        tests.append({'name': 'dataset_11',
                      'path': "./resources/data__test_preprocessing_py/11_20190910_glc_spcm_1_MMStack.ome.tif",
                      'angle': 0,
                      'glt': 300,
                      'centers': [518, 1355]})
        tests.append({'name': 'dataset_10',
                      'path': "./resources/data__test_preprocessing_py/10_20190424_hi2_hi3_med2_rplN_glu_gly.ome.tif",
                      'angle': 0,
                      'glt': 300,
                      'centers': [392, 1249]})
        tests.append({'name': 'dataset_9',
                      'path': "./resources/data__test_preprocessing_py/09_20190325_hi1_hi2_med1_rpmB_glu_gly_pl_chr_1.tif",
                      'angle': 0,
                      'glt': 300,
                      'centers': [519, 1372]})
        tests.append({'name': 'dataset_8',
                      'path': "./resources/data__test_preprocessing_py/08_20190222_LB_SpentLB_TrisEDTA_LB_1.tif",
                      'angle': -4,
                      'glt': 300,
                      'centers': [591, 1441]})
        tests.append({'name': 'dataset_7',
                      'path': "./resources/data__test_preprocessing_py/07_20181203_glu_lac_switch16h_1__Pos0.tif",
                      'angle': 0,
                      'glt': 200,
                      'centers': [461]})
        tests.append({'name': 'dataset_6',
                      'path': "./resources/data__test_preprocessing_py/06_20180313_glu_lac_switch24h_1__Pos0.tif",
                      'angle': 0,
                      'glt': 300,
                      'centers': [512]})
        tests.append({'name': 'dataset_5',
                      'path': "./resources/data__test_preprocessing_py/05_20180404_glu_lacCM-ara_1__Pos0.tif",
                      'angle': 0,
                      'glt': 300,
                      'centers': [513]})
        tests.append({'name': 'dataset_4',
                      'path': "./resources/data__test_preprocessing_py/04_20180531_gluIPTG5uM_lac_1__Pos0.tif",
                      'angle': 0,
                      'glt': 300,
                      'centers': [513]})
        tests.append({'name': 'dataset_3',
                      'path': "./resources/data__test_preprocessing_py/03_20180604_gluIPTG10uM_lac_lacIoe_1__Pos0.tif",
                      'angle': 0,
                      'glt': 300,
                      'centers': [484]})
        tests.append({'name': 'dataset_0',
                      'path': "./resources/data__test_preprocessing_py/00_20150710_mmtest_2ch__Pos0__rotated.tif",
                      'angle': -1.5,
                      'glt': 300,
                      'centers': [660]})
        return tests

    def test__find_channel_regions(self):
        import tifffile as tff
        from mmpreprocesspy import preprocessing
        import matplotlib.pyplot as plt
        import skimage

        region_center_tolerance = 10

        tests = self.get_tests__test__find_channel_regions()

        for test in tests:
            path = test['path']
            rotation_angle = test['angle']
            growthlane_length_threshold = test['glt']
            region_centers = test['centers']

            with self.subTest(test=test['name']):
                imdata = tff.imread(path)
                imdata = skimage.transform.rotate(imdata, rotation_angle)
                region_list = preprocessing.find_channel_regions(imdata, use_smoothing=True, minimum_required_growthlane_length=growthlane_length_threshold)

                # if test['name'] == 'dataset_14' or test['name'] == 'dataset_16':
                #     plt.imshow(imdata, cmap='gray')
                #     for ind, region in enumerate(region_list):
                #         plt.axvline(region.start, color='r')
                #         plt.axvline(region.end, color='g')
                #     plt.title(test['name'])
                #     plt.show()

                for ind, region in enumerate(region_list):
                    expected = region_centers[ind]
                    actual = region_horizontal_center = region.start + region.width/2
                    self.assertAlmostEqual(expected, actual, delta=region_center_tolerance)


    def test__find_channels_in_region_dataset_10(self):
        from mmpreprocesspy import preprocessing

        image = imread("./resources/10_20190424_hi2_hi3_med2_rplN_4_MMStack.ome-2.tif")
        centers = preprocessing.get_gl_center_positions_in_growthlane_region(image)
        self.assertAlmostEqual(18, centers[0], delta=5)
        self.assertAlmostEqual(125, centers[1], delta=5)

        # image_with_channel_indicators = get_image_with_hlines(image, centers)
        # plt.imshow(image_with_channel_indicators, cmap="gray")
        # plt.show()

    def test__find_channels_in_region_dataset_4(self):
        from mmpreprocesspy import preprocessing

        image = imread("./resources/04_20180531_gluIPTG5uM_lac_1_MMStack.ome-2_channel_region.tif")
        centers = preprocessing.get_gl_center_positions_in_growthlane_region(image)
        self.assertAlmostEqual(74, centers[1], delta=5)
        self.assertAlmostEqual(147, centers[2], delta=5)

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
