import os
from unittest import TestCase
from scipy.signal import find_peaks

# import matplotlib.pyplot as plt
import mmpreprocesspy.dev_auxiliary_functions as dev_aux
from mmpreprocesspy.data_region import DataRegion
from mmpreprocesspy.moma_image_processing import MomaImageProcessor
import numpy as np
import skimage.transform
from skimage.io import imread
from skimage.transform import AffineTransform, warp
from parameterized import parameterized


class TestPreprocessing(TestCase):
    def get_tests__test__get_all_growthlane_rois(self):
        tests = []
        # tests.append({'name': 'dataset_17',
        #               'path': "./resources/data__test_preprocessing_py/17_lis__20201218_VNG40_AB6min_2h_1_1_MMStack.ome.tif",
        #               'angle': 90,
        #               'glt': 200,
        #               'centers': [538, 1343],
        #               'dataregions': [[317, 966], [1211, 1781]]
        #               })
        tests.append({'name': 'dataset_16',
                      'path': "./resources/data__test_preprocessing_py/16_thomas__20201229_glc_lac_1_MMStack__Pos0__rotated.tif",
                      'angle': 0,
                      'glt': 200,
                      'centers': [538, 1343],
                      'dataregions': [[354, 910], [1160, 1751]]
                      })
        tests.append({'name': 'dataset_15',
                      'path': "./resources/data__test_preprocessing_py/15_lis__20201119_VNG1040_AB2h_2h_1_MMStack.ome.tif",
                      'angle': 90,
                      'glt': 200,
                      'centers': [573, 1409],
                      'dataregions': [[325, 908], [1184, 1733]]
                      })
        # tests.append({'name': 'dataset_14',
        #               'path': "./resources/data__test_preprocessing_py/14_thomas_20201228_glc_ara_1__Pos0__rotated.tif",
        #               'angle': -.5,
        #               'glt': 300,
        #               'centers': [489, 1315],
        #               'dataregions': [[323, 873], [1129, 1721]]
        #               })
        tests.append({'name': 'dataset_11',
                      'path': "./resources/data__test_preprocessing_py/11_20190910_glc_spcm_1_MMStack.ome.tif",
                      'angle': 0,
                      'glt': 300,
                      'centers': [518, 1355],
                      'dataregions': [[332, 901], [1173, 1739]]
                      })

        tests.append({'name': 'dataset_10',
                      'path': "./resources/data__test_preprocessing_py/10_20190424_hi2_hi3_med2_rplN_glu_gly.ome.tif",
                      'angle': 0,
                      'glt': 300,
                      'centers': [392, 1249],
                      'dataregions': [[163, 714], [1008, 1575]]
                      })

        tests.append({'name': 'dataset_9',
                      'path': "./resources/data__test_preprocessing_py/09_20190325_hi1_hi2_med1_rpmB_glu_gly_pl_chr_1.tif",
                      'angle': 0,
                      'glt': 300,
                      'centers': [519, 1372],
                      'dataregions': [[294, 845], [1135, 1701]]
                      })

        tests.append({'name': 'dataset_8',
                      'path': "./resources/data__test_preprocessing_py/08_20190222_LB_SpentLB_TrisEDTA_LB_1.tif",
                      'angle': -4,
                      'glt': 300,
                      'centers': [591, 1441],
                      'dataregions': [[360, 922], [1209, 1756]]
                      })

        return tests

    def test__get_all_growthlane_rois(self):
        import tifffile as tff
        from mmpreprocesspy import preprocessing
        import matplotlib.pyplot as plt
        import skimage

        test_data_base_path = '/home/micha/Documents/01_work/git/mmpreprocesspy/mmpreprocesspy/test/'
        tests = self.get_tests__test__get_all_growthlane_rois()
        tests.reverse()
        region_reduction_margin = 100

        normalization_range_results = {}

        for test in tests:
            path = os.path.join(test_data_base_path, test['path'])
            rotation_angle = test['angle']
            growthlane_length_threshold = test['glt']
            region_centers = test['centers']

            imdata = tff.imread(path)
            imdata = skimage.transform.rotate(imdata, rotation_angle, preserve_range=True)

            region_list = [DataRegion(start=entry[0] + region_reduction_margin,
                                      end=entry[1] - region_reduction_margin,
                                      width=(entry[1] - entry[0]) - 2 * region_reduction_margin) for entry in
                           test['dataregions']]
            if is_debugging():
                plt.imshow(imdata, cmap='gray')
                for ind, region in enumerate(region_list):
                    plt.axvline(region.start, color='r')
                    plt.axvline(region.end, color='g')
                plt.title(test['name'])
                plt.show()

            if test['name'] == 'dataset_16':
                plt.imshow(imdata, cmap='gray')
                for ind, region in enumerate(region_list):
                    plt.axvline(region.start, color='r')
                    plt.axvline(region.end, color='g')
                plt.title(test['name'])
                plt.show()

                labels = ['region_1', 'region_2']
                for ind, region in enumerate(region_list):
                    profile = np.mean(imdata[:, region.start:region.end], axis=1)
                    plt.plot(profile, '')
                plt.title(test['name'])
                plt.show()

                pass


            normalization_ranges = get_normalization_ranges(imdata, region_list, box_pts=11)

            normalization_range_results[test['name']] = normalization_ranges

        # get data for unnormalized plot
        dataset_inds = []
        dataset_range_mins = []
        dataset_range_maxs = []
        tick_labels = []
        for dataset_ind, key in enumerate(normalization_range_results.keys()):
            range1 = normalization_range_results[key][0]
            dataset_range_mins.append(range1[0])
            dataset_range_maxs.append(range1[1])
            range2 = normalization_range_results[key][1]
            dataset_range_mins.append(range2[0])
            dataset_range_maxs.append(range2[1])
            dataset_inds.append(dataset_ind), dataset_inds.append(dataset_ind)  # add ind two times for each region
            tick_labels.append(key), tick_labels.append(key)  # add ind two times for each region

        # plot unnormalized comparison of min/max between regions
        plt.scatter(dataset_inds, dataset_range_mins, color='r')
        plt.scatter(dataset_inds, dataset_range_maxs, color='g')
        plt.xticks(dataset_inds, tick_labels, rotation=20)
        plt.xlabel('dataset')
        plt.ylabel('intensity')
        plt.show()

        # get data for normalized plot
        values_relative = []
        dataset_inds_relative = []
        tick_labels_relative = []
        for dataset_ind, key in enumerate(normalization_range_results.keys()):
            range1 = normalization_range_results[key][0]
            range2 = normalization_range_results[key][1]
            range1_diff = range1[1] - range1[0]
            range2_diff = range2[1] - range2[0]
            relative_range_deviation = np.abs(range1_diff - range2_diff) / np.min([range1_diff, range2_diff])
            values_relative.append(relative_range_deviation)
            dataset_inds_relative.append(dataset_ind)
            tick_labels_relative.append(key)


        # plot normalized comparison of min/max between regions
        plt.scatter(dataset_inds_relative, values_relative, color='r')
        plt.xticks(dataset_inds_relative, tick_labels_relative, rotation=20)
        plt.xlabel('dataset')
        plt.ylabel('relative deviation of normalization ranges')
        plt.show()


        pass



        pass
            # with self.subTest(test=test['name']):
            #     if not test['dataregions'][0]:   # output information for setting up test, if test assert-data is not yet fully defined
            #         region_list = preprocessing.find_channel_regions(imdata, use_smoothing=True,
            #                                                          minimum_required_growthlane_length=growthlane_length_threshold)
            #         plt.imshow(imdata, cmap='gray')
            #         for ind, region in enumerate(region_list):
            #             plt.axvline(region.start, color='r')
            #             plt.axvline(region.end, color='g')
            #         plt.title(test['name'])
            #         plt.show()
            #
            #         res = [[region.start, region.end] for region in region_list]
            #         print(f"'dataregions': {res}")
            #     else:
            #         region_list = [DataRegion(start=entry[0], end=entry[1], width=entry[1] - entry[0])
            #                        for entry in test['dataregions']]
            #
            #     growthlane_rois, channel_centers = preprocessing.get_all_growthlane_rois(imdata, region_list)
            #     growthlane_rois = preprocessing.rotate_rois(imdata, growthlane_rois, 0)
            #     growthlane_rois = preprocessing.remove_rois_not_fully_in_image(imdata, growthlane_rois)
            #
            #     if not test['roi_vertical_positions']:   # output information for setting up test, if test assert-data is not yet fully defined
            #         print(f"'roi_vertical_positions': {[roi.roi.center[1] for roi in growthlane_rois]}")
            #         self.show_gl_index_image(growthlane_rois, imdata, figure_title=test['name'])
            #
            #     actual_vertical_positions = [roi.roi.center[1] for roi in growthlane_rois]
            #
            #     # ASSERTS
            #     actual_periodicity = self.calculate_periodicty(actual_vertical_positions)
            #     expected_periodicity = self.calculate_periodicty(expected_vertical_positions)
            #
            #     self.assertAlmostEqual(expected_periodicity, actual_periodicity, delta=3)
            #
            #     for ind, roi in enumerate(growthlane_rois):
            #         expected = expected_vertical_positions[ind]
            #         actual = roi.roi.center[1]
            #         self.assertAlmostEqual(expected, actual, delta=region_center_tolerance)


def get_normalization_ranges(image, region_list, box_pts=11):
    norm_ranges = []
    for region in region_list:
        norm_range_min_value, norm_range_max_value = get_normalization_range_in_region(image, region.start,
                                                                                       region.end,
                                                                                       box_pts=box_pts)
        norm_ranges.append([norm_range_min_value, norm_range_max_value])
    return np.array(norm_ranges)


def get_normalization_range_in_region(aligned_image, region_start, region_end, box_pts=11):
    box_pts = 1
    gl_region = aligned_image[:, region_start:region_end]
    projected_intensity = np.mean(gl_region, axis=1)
    projected_intensity_smoothed = smooth(projected_intensity, box_pts)
    valid_region_offset = int(
        np.ceil(box_pts / 2))  # we keep only the region, where the smoothing operation is well-defined
    projected_intensity_smoothed = projected_intensity_smoothed[
                                   valid_region_offset:-valid_region_offset]  # keep only valid region
    # mean_peak_vals = projected_intensity_smoothed

    peak_inds = find_peaks(projected_intensity_smoothed, distance=25)[0]
    peak_vals = projected_intensity_smoothed[peak_inds]

    min = peak_vals.min()
    max = peak_vals.max()
    range = (max - min)
    threshold_lower = min + range * 0.1
    threshold_upper = min + range * 0.8

    pdms_peak_vals = peak_vals[peak_vals < threshold_lower]
    pdms_peak_inds = peak_inds[peak_vals < threshold_lower]
    empty_peak_vals = peak_vals[peak_vals > threshold_upper]
    empty_peak_inds = peak_inds[peak_vals > threshold_upper]

    norm_range_min_value = np.max(pdms_peak_vals)
    norm_range_max_value = np.max(empty_peak_vals)

    # if is_debugging():
    #     import matplotlib.pyplot as plt
    #     plt.plot(projected_intensity_smoothed)
    #     plt.plot(projected_intensity)
    #     plt.scatter(peak_inds, peak_vals, color='k')
    #     plt.scatter(pdms_peak_inds, pdms_peak_vals, color='b')
    #     plt.scatter(empty_peak_inds, empty_peak_vals, color='g')
    #     plt.axhline(threshold_lower, linestyle='--', color='k')
    #     plt.axhline(threshold_upper, linestyle='--', color='k')
    #     plt.show()
    #
    # if is_debugging():
    #     import matplotlib.pyplot as plt
    #     plt.plot(projected_intensity, color='r')
    #     plt.plot(projected_intensity_smoothed, color='g')
    #     plt.show()
    #
    # if is_debugging():
    #     import matplotlib.pyplot as plt
    #     plt.plot(projected_intensity, color='r')
    #     plt.scatter(peak_inds, projected_intensity_smoothed[peak_inds])
    #     plt.show()
    #
    # if is_debugging():
    #     import matplotlib.pyplot as plt
    #     plt.plot(projected_intensity, color='r')
    #     plt.plot(projected_intensity_smoothed, color='g')
    #     plt.show()

    if is_debugging():
        import matplotlib.pyplot as plt
        plt.plot(projected_intensity_smoothed)
        plt.scatter(np.argwhere(projected_intensity_smoothed == norm_range_min_value), norm_range_min_value, color='r')
        plt.scatter(np.argwhere(projected_intensity_smoothed == norm_range_max_value), norm_range_max_value, color='g')
        plt.show()


    return norm_range_min_value, norm_range_max_value


def smooth(y, box_pts):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def is_debugging():
    try:
        import pydevd
        return True
    except ImportError:
        return False

