import os
from unittest import TestCase

# import matplotlib.pyplot as plt
import mmpreprocesspy.dev_auxiliary_functions as dev_aux
from mmpreprocesspy.data_region import DataRegion
import numpy as np
import skimage.transform
from skimage.io import imread
from skimage.transform import AffineTransform, warp
from parameterized import parameterized

def set_axis_size(w, h, ax=None):
    import matplotlib.pyplot as plt
    """ w, h: width, height in inches """
    if not ax: ax=plt.gca()
    l = ax.figure.subplotpars.left
    r = ax.figure.subplotpars.right
    t = ax.figure.subplotpars.top
    b = ax.figure.subplotpars.bottom
    figw = float(w)/(r-l)
    figh = float(h)/(t-b)
    ax.figure.set_size_inches(figw, figh)

def calculate_vertical_ft(image):
    vertical_ft = np.zeros_like(image, dtype=complex)
    for col in range(image.shape[1]):
        vertical_ft[:, col] = np.fft.fft(image[:, col])
    return vertical_ft

class TestPreprocessing(TestCase):

    def test__explore__plot_spectra_at_different_horizontal_positons(self):
        import tifffile as tff
        from mmpreprocesspy import preprocessing
        import matplotlib.pyplot as plt
        import skimage

        test_data_base_path = '/home/micha/Documents/01_work/git/mmpreprocesspy/mmpreprocesspy/test'

        tests = self.get_tests__test__find_channel_regions()

        positions = [150, 300, 340, 600, 1000]
        colors = ['r', 'g', 'b', 'c', 'k']

        for test in [tests[1]]:
            path = os.path.join(test_data_base_path, test['path'])
            rotation_angle = test['angle']
            growthlane_length_threshold = test['glt']
            region_centers = test['centers']

            imdata = tff.imread(path)
            image_orig = skimage.transform.rotate(imdata, rotation_angle)
            image = image_orig - np.mean(image_orig, axis=0)

            vertical_ft = calculate_vertical_ft(image)
            spectral_power = np.abs(np.power(vertical_ft, 2))

            plt.imshow(image_orig)
            for ind, col_ind in enumerate(positions):
                plt.gca().axvline(col_ind, color=colors[ind])
                # ax[1].plot(np.log(spectral_power[:, col_ind]), color=colors[ind])
            plt.show()

            fig, ax = plt.subplots(1, 1)
            for ind, col_ind in enumerate(positions):
                # ax[0].axvline(col_ind, color=colors[ind])
                # ax[1].plot(np.log(spectral_power[:, col_ind]), color=colors[ind])
                ax.plot(spectral_power[0:50, col_ind], color=colors[ind])
            # ax[1].set_xlim([15,25])
            # ax[1].set_ylim([0, 200])
            ax.axvline(19)

            # set_axis_size(5, 5, ax=ax[0])
            # set_axis_size(5, 5, ax=ax)

            plt.show()

    def test__explore__relative_power_of_spectral_peak(self):
        import tifffile as tff
        from mmpreprocesspy import preprocessing
        import matplotlib.pyplot as plt
        import skimage

        test_data_base_path = '/home/micha/Documents/01_work/git/mmpreprocesspy/mmpreprocesspy/test'

        region_center_tolerance = 10

        tests = self.get_tests__test__find_channel_regions()

        # for test in [tests[1]]:
        for test in tests:
            path = os.path.join(test_data_base_path, test['path'])
            rotation_angle = test['angle']
            growthlane_length_threshold = test['glt']
            region_centers = test['centers']

            imdata = tff.imread(path)
            image_orig = skimage.transform.rotate(imdata, rotation_angle)

            # vertical_ft = calculate_vertical_ft(image_orig)
            # spectral_power = np.abs(np.power(vertical_ft, 2))
            # result = spectral_power[19, :] / np.sum(spectral_power, axis=0)
            # result /= np.max(result)
            # plt.plot(result, 'r')

            image = image_orig - np.mean(image_orig, axis=0)
            vertical_ft = calculate_vertical_ft(image)
            spectral_power = np.abs(np.power(vertical_ft, 2))
            peak_pos = 19

            peak_value_1 = spectral_power[peak_pos, :]
            result_1 = peak_value_1 / np.sum(spectral_power, axis=0)
            result_1 /= np.max(result_1)
            plt.plot(result_1, 'r', label='norm. spec. pow.')

            peak_value_2 = np.sum(spectral_power[peak_pos-2:peak_pos+3, :], axis=0)
            result_2 = peak_value_2 / np.sum(spectral_power, axis=0)
            result_2 /= np.max(result_2)
            plt.plot(result_2, 'g', label='norm. spec. pow.')

            # image = image_orig - np.min(image_orig, axis=0)
            # image = image / np.max(image_orig, axis=0)
            # vertical_ft = calculate_vertical_ft(image)
            # spectral_power = np.abs(np.power(vertical_ft, 2))
            # result = spectral_power[19, :] / np.sum(spectral_power, axis=0)
            # result /= np.max(result)
            # plt.plot(result, 'c')

            # max_projection = np.max(image_orig, axis=0)
            # result = max_projection / np.max(max_projection)
            # # result = np.mean(image_orig, axis=0)
            # # result = result / np.max(result)
            # plt.plot(result, label='norm. proj. int.')

            plt.legend()
            plt.title(test['name'])
            plt.show()

            # if test['name'] == 'dataset_16':
            #     with self.subTest(test=test['name']):
            #         imdata = tff.imread(path)
            #         imdata = skimage.transform.rotate(imdata, rotation_angle)
            #         region_list = preprocessing.find_channel_regions_using_fourierspace(imdata, use_smoothing=True, minimum_required_growthlane_length=growthlane_length_threshold)
            #
            #         # if not region_centers:  # output needed data for asserts, if it is not defined
            #         res = [region.start + region.width/2 for region in region_list]
            #         print(f"'centers': {res}")
            #         plt.imshow(imdata, cmap='gray')
            #         for ind, region in enumerate(region_list):
            #             plt.axvline(region.start, color='r')
            #             plt.axvline(region.end, color='g')
            #         plt.title(test['name'])
            #         plt.show()
            #
            #         for ind, region in enumerate(region_list):
            #             expected = region_centers[ind]
            #             actual = region_horizontal_center = region.start + region.width/2
            #             self.assertAlmostEqual(expected, actual, delta=region_center_tolerance)


    def get_tests__test__find_channel_regions(self):
        tests = []
        tests.append({'name': 'dataset_17',
                      'path': "./resources/data__test_preprocessing_py/17_lis__20201218_VNG40_AB6min_2h_1_1_MMStack.ome.tif",
                      'angle': 90,
                      'glt': 200,
                      'centers': [435.0, 1415.0]})
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

