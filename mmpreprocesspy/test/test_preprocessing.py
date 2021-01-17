from unittest import TestCase

# import matplotlib.pyplot as plt
import mmpreprocesspy.dev_auxiliary_functions as dev_aux
from mmpreprocesspy.data_region import DataRegion
import numpy as np
import skimage.transform
from skimage.io import imread
from skimage.transform import AffineTransform, warp
from parameterized import parameterized


class TestPreprocessing(TestCase):
    test_data_base_path = '/home/micha/Documents/01_work/git/MM_Testing'

    def get_tests__test__get_all_growthlane_rois(self):
        tests = []
        tests.append({'name': 'dataset_17',
                      'path': "./resources/data__test_preprocessing_py/17_lis__20201218_VNG40_AB6min_2h_1_1_MMStack.ome.tif",
                      'angle': 90,
                      'glt': 200,
                      'centers': [538, 1343],
                      'dataregions': [[317, 966], [1211, 1781]],
                      'roi_vertical_positions': [118.0, 225.0, 331.0, 438.0, 544.0, 651.0, 757.0, 864.0, 970.0, 1077.0,
                                                 1183.0, 1290.0, 1396.0, 1503.0, 1609.0, 1716.0, 1822.0, 1929.0, 126.0,
                                                 232.0, 339.0, 445.0, 552.0, 658.0, 765.0, 871.0, 978.0, 1084.0, 1191.0,
                                                 1297.0, 1404.0, 1510.0, 1617.0, 1723.0, 1830.0, 1936.0]})
        tests.append({'name': 'dataset_16',
                      'path': "./resources/data__test_preprocessing_py/16_thomas__20201229_glc_lac_1_MMStack__Pos0__rotated.tif",
                      'angle': 0,
                      'glt': 200,
                      'centers': [538, 1343],
                      'dataregions': [[354, 910], [1160, 1751]],
                      'roi_vertical_positions': [137.0, 243.0, 349.0, 454.0, 560.0, 666.0, 771.0, 877.0, 983.0, 1088.0,
                                                 1194.0, 1300.0, 1405.0, 1511.0, 1617.0, 1722.0, 1828.0, 1934.0, 142.0,
                                                 247.0, 352.0, 458.0, 563.0, 668.0, 774.0, 879.0, 984.0, 1090.0, 1195.0,
                                                 1300.0, 1406.0, 1511.0, 1616.0, 1722.0, 1827.0, 1932.0]
                      })
        tests.append({'name': 'dataset_15',
                      'path': "./resources/data__test_preprocessing_py/15_lis__20201119_VNG1040_AB2h_2h_1_MMStack.ome.tif",
                      'angle': 90,
                      'glt': 200,
                      'centers': [573, 1409],
                      'dataregions': [[325, 908], [1184, 1733]],
                      'roi_vertical_positions': [157.0, 264.0, 371.0, 478.0, 585.0, 692.0, 799.0, 906.0, 1013.0, 1120.0,
                                                 1227.0, 1334.0, 1441.0, 1548.0, 1655.0, 1762.0, 1869.0, 1976.0, 158.0,
                                                 265.0, 372.0, 479.0, 586.0, 693.0, 800.0, 907.0, 1014.0, 1121.0,
                                                 1228.0, 1335.0, 1442.0, 1549.0, 1656.0, 1763.0, 1870.0, 1977.0]})
        tests.append({'name': 'dataset_14',
                      'path': "./resources/data__test_preprocessing_py/14_thomas_20201228_glc_ara_1__Pos0__rotated.tif",
                      'angle': -.5,
                      'glt': 300,
                      'centers': [489, 1315],
                      'dataregions': [[323, 873], [1129, 1721]],
                      'roi_vertical_positions': [89.0, 195.0, 300.0, 406.0, 511.0, 617.0, 722.0, 828.0, 933.0, 1039.0,
                                                 1144.0, 1250.0, 1355.0, 1461.0, 1566.0, 1672.0, 1777.0, 1883.0, 1988.0,
                                                 89.0, 195.0, 300.0, 406.0, 511.0, 617.0, 722.0, 828.0, 933.0, 1039.0,
                                                 1144.0, 1250.0, 1355.0, 1461.0, 1566.0, 1672.0, 1777.0, 1883.0,
                                                 1988.0]})
        tests.append({'name': 'dataset_13',
                      'path': "./resources/data__test_preprocessing_py/13_20200128_glcIPTG_glc_1__MMStack.ome.tif",
                      'angle': 0,
                      'glt': 200,
                      'centers': [465, 752],
                      'dataregions': [[353, 685]],
                      'roi_vertical_positions': [78.0, 184.0, 289.0, 395.0, 500.0, 606.0, 711.0, 817.0, 922.0, 1028.0,
                                                 1133.0, 1239.0, 1344.0, 1450.0, 1555.0, 1661.0, 1766.0, 1872.0,
                                                 1977.0]})
        tests.append({'name': 'dataset_12',
                      'path': "./resources/data__test_preprocessing_py/12_20190816_Theo_MMStack.ome.tif",
                      'angle': 0,
                      'glt': 200,
                      'centers': [526],
                      'dataregions': [[361, 759]],
                      'roi_vertical_positions': [87.0, 160.0, 233.0, 306.0, 379.0, 452.0, 525.0, 598.0, 671.0, 744.0,
                                                 817.0, 890.0, 963.0, 1036.0, 1109.0, 1182.0, 1255.0, 1328.0, 1401.0,
                                                 1474.0, 1547.0, 1620.0, 1693.0, 1766.0, 1839.0, 1912.0, 1985.0]})
        tests.append({'name': 'dataset_11',
                      'path': "./resources/data__test_preprocessing_py/11_20190910_glc_spcm_1_MMStack.ome.tif",
                      'angle': 0,
                      'glt': 300,
                      'centers': [518, 1355],
                      'dataregions': [[332, 901], [1173, 1739]],
                      'roi_vertical_positions': [74.0, 181.0, 287.0, 394.0, 500.0, 607.0, 713.0, 820.0, 926.0, 1033.0,
                                                 1139.0, 1246.0, 1352.0, 1459.0, 1565.0, 1672.0, 1778.0, 1885.0, 1991.0,
                                                 80.0, 187.0, 293.0, 400.0, 506.0, 613.0, 719.0, 826.0, 932.0, 1039.0,
                                                 1145.0, 1252.0, 1358.0, 1465.0, 1571.0, 1678.0, 1784.0, 1891.0]})
        tests.append({'name': 'dataset_10',
                      'path': "./resources/data__test_preprocessing_py/10_20190424_hi2_hi3_med2_rplN_glu_gly.ome.tif",
                      'angle': 0,
                      'glt': 300,
                      'centers': [392, 1249],
                      'dataregions': [[163, 714], [1008, 1575]],
                      'roi_vertical_positions': [126.0, 232.0, 339.0, 445.0, 552.0, 658.0, 765.0, 871.0, 978.0, 1084.0,
                                                 1191.0, 1297.0, 1404.0, 1510.0, 1617.0, 1723.0, 1830.0, 1936.0, 127.0,
                                                 233.0, 340.0, 446.0, 553.0, 659.0, 766.0, 872.0, 979.0, 1085.0, 1192.0,
                                                 1298.0, 1405.0, 1511.0, 1618.0, 1724.0, 1831.0, 1937.0]})
        tests.append({'name': 'dataset_9',
                      'path': "./resources/data__test_preprocessing_py/09_20190325_hi1_hi2_med1_rpmB_glu_gly_pl_chr_1.tif",
                      'angle': 0,
                      'glt': 300,
                      'centers': [519, 1372],
                      'dataregions': [[294, 845], [1135, 1701]],
                      'roi_vertical_positions': [94.0, 201.0, 307.0, 414.0, 520.0, 627.0, 733.0, 840.0, 946.0, 1053.0,
                                                 1159.0, 1266.0, 1372.0, 1479.0, 1585.0, 1692.0, 1798.0, 1905.0, 93.0,
                                                 200.0, 306.0, 413.0, 519.0, 626.0, 732.0, 839.0, 945.0, 1052.0, 1158.0,
                                                 1265.0, 1371.0, 1478.0, 1584.0, 1691.0, 1797.0, 1904.0]})
        tests.append({'name': 'dataset_8',
                      'path': "./resources/data__test_preprocessing_py/08_20190222_LB_SpentLB_TrisEDTA_LB_1.tif",
                      'angle': -4,
                      'glt': 300,
                      'centers': [591, 1441],
                      'dataregions': [[360, 922], [1209, 1756]],
                      'roi_vertical_positions': [106.0, 213.0, 319.0, 426.0, 532.0, 639.0, 745.0, 852.0, 958.0, 1065.0,
                                                 1171.0, 1278.0, 1384.0, 1491.0, 1597.0, 1704.0, 1810.0, 1917.0, 109.0,
                                                 216.0, 322.0, 429.0, 535.0, 642.0, 748.0, 855.0, 961.0, 1068.0, 1174.0,
                                                 1281.0, 1387.0, 1494.0, 1600.0, 1707.0, 1813.0, 1920.0]})
        tests.append({'name': 'dataset_7',
                      'path': "./resources/data__test_preprocessing_py/07_20181203_glu_lac_switch16h_1__Pos0.tif",
                      'angle': 0,
                      'glt': 200,
                      'centers': [461],
                      'dataregions': [[336, 686]],
                      'roi_vertical_positions': [111.0, 218.0, 325.0, 432.0, 539.0, 646.0, 753.0, 860.0, 967.0, 1074.0,
                                                 1181.0, 1288.0, 1395.0, 1502.0, 1609.0, 1716.0, 1823.0, 1930.0]})
        tests.append({'name': 'dataset_6',
                      'path': "./resources/data__test_preprocessing_py/06_20180313_glu_lac_switch24h_1__Pos0.tif",
                      'angle': 0,
                      'glt': 300,
                      'centers': [512],
                      'dataregions': [[362, 762]],
                      'roi_vertical_positions': [75.0, 148.0, 221.0, 294.0, 367.0, 440.0, 513.0, 586.0, 659.0, 732.0,
                                                 805.0, 878.0, 951.0, 1024.0, 1097.0, 1170.0, 1243.0, 1316.0, 1389.0,
                                                 1462.0, 1535.0, 1608.0, 1681.0, 1754.0, 1827.0, 1900.0, 1973.0]})
        tests.append({'name': 'dataset_5',
                      'path': "./resources/data__test_preprocessing_py/05_20180404_glu_lacCM-ara_1__Pos0.tif",
                      'angle': 0,
                      'glt': 300,
                      'centers': [513],
                      'dataregions': [[362, 759]],
                      'roi_vertical_positions': [68.0, 142.0, 215.0, 289.0, 362.0, 436.0, 509.0, 583.0, 656.0, 730.0,
                                                 803.0, 877.0, 950.0, 1024.0, 1097.0, 1171.0, 1244.0, 1318.0, 1391.0,
                                                 1465.0, 1538.0, 1612.0, 1685.0, 1759.0, 1832.0, 1906.0, 1979.0]})
        tests.append({'name': 'dataset_4',
                      'path': "./resources/data__test_preprocessing_py/04_20180531_gluIPTG5uM_lac_1__Pos0.tif",
                      'angle': 0,
                      'glt': 300,
                      'centers': [513],
                      'dataregions': [[356, 753]],
                      'roi_vertical_positions': [75.0, 148.0, 221.0, 294.0, 367.0, 440.0, 513.0, 586.0, 659.0, 732.0,
                                                 805.0, 878.0, 951.0, 1024.0, 1097.0, 1170.0, 1243.0, 1316.0, 1389.0,
                                                 1462.0, 1535.0, 1608.0, 1681.0, 1754.0, 1827.0, 1900.0, 1973.0]})
        tests.append({'name': 'dataset_3',
                      'path': "./resources/data__test_preprocessing_py/03_20180604_gluIPTG10uM_lac_lacIoe_1__Pos0.tif",
                      'angle': 0,
                      'glt': 300,
                      'centers': [484],
                      'dataregions': [[332, 736]],
                      'roi_vertical_positions': [72.0, 145.0, 218.0, 291.0, 364.0, 437.0, 510.0, 583.0, 656.0, 729.0,
                                                 802.0, 875.0, 948.0, 1021.0, 1094.0, 1167.0, 1240.0, 1313.0, 1386.0,
                                                 1459.0, 1532.0, 1605.0, 1678.0, 1751.0, 1824.0, 1897.0, 1970.0]})
        tests.append({'name': 'dataset_0',
                      'path': "./resources/data__test_preprocessing_py/00_20150710_mmtest_2ch__Pos0__rotated.tif",
                      'angle': -1.5,
                      'glt': 300,
                      'centers': [660],
                      'dataregions': [[509, 907]],
                      'roi_vertical_positions': [41.0, 115.0, 188.0, 262.0, 335.0, 409.0, 482.0, 556.0, 629.0, 703.0,
                                                 776.0, 850.0, 923.0, 997.0, 1070.0, 1144.0, 1217.0, 1291.0, 1364.0,
                                                 1438.0, 1511.0, 1585.0, 1658.0, 1732.0, 1805.0, 1879.0, 1952.0]})
        return tests

    def test__get_all_growthlane_rois(self):
        import tifffile as tff
        from mmpreprocesspy import preprocessing
        import matplotlib.pyplot as plt
        import skimage

        region_center_tolerance = 5

        tests = self.get_tests__test__get_all_growthlane_rois()

        for test in tests:
            path = test['path']
            rotation_angle = test['angle']
            growthlane_length_threshold = test['glt']
            region_centers = test['centers']
            roi_vertical_positions = test['roi_vertical_positions']

            imdata = tff.imread(path)
            imdata = skimage.transform.rotate(imdata, rotation_angle)

            with self.subTest(test=test['name']):
                if not test['dataregions'][0]:   # output information for setting up test, if test assert-data is not yet fully defined
                    region_list = preprocessing.find_channel_regions(imdata, use_smoothing=True,
                                                                     minimum_required_growthlane_length=growthlane_length_threshold)
                    plt.imshow(imdata, cmap='gray')
                    for ind, region in enumerate(region_list):
                        plt.axvline(region.start, color='r')
                        plt.axvline(region.end, color='g')
                    plt.title(test['name'])
                    plt.show()

                    res = [[region.start, region.end] for region in region_list]
                    print(f"'dataregions': {res}")
                else:
                    region_list = [DataRegion(start=entry[0], end=entry[1], width=entry[1] - entry[0])
                                   for entry in test['dataregions']]

                growthlane_rois, channel_centers = preprocessing.get_all_growthlane_rois(imdata, region_list)
                growthlane_rois = preprocessing.rotate_rois(imdata, growthlane_rois, 0)
                growthlane_rois = preprocessing.remove_rois_not_fully_in_image(imdata, growthlane_rois)

                if not test['roi_vertical_positions']:   # output information for setting up test, if test assert-data is not yet fully defined
                    print(f"'roi_vertical_positions': {[roi.roi.center[1] for roi in growthlane_rois]}")
                    self.show_gl_index_image(growthlane_rois, imdata, figure_title=test['name'])

                # ASSERT
                for ind, roi in enumerate(growthlane_rois):
                    expected = roi_vertical_positions[ind]
                    actual = roi.roi.center[1]
                    self.assertAlmostEqual(expected, actual, delta=region_center_tolerance)

    def show_gl_index_image(self, growthlane_rois, full_frame_image, figure_title=None):
        import cv2 as cv
        import matplotlib.pyplot as plt

        """ Draw the growthlane ROIs and indices onto the image and save it. """
        font = cv.FONT_HERSHEY_SIMPLEX
        normalized_image = cv.normalize(full_frame_image, None, 0, 255, cv.NORM_MINMAX)
        final_image = np.array(normalized_image, dtype=np.uint8)

        for roi in growthlane_rois:
            roi.roi.draw_to_image(final_image, False)
            gl_index = self.calculate_gl_output_index(roi.id)
            cv.putText(final_image, str(gl_index), (np.int0(roi.roi.center[0]), np.int0(roi.roi.center[1])), font, 1,
                       (255, 255, 255), 2, cv.LINE_AA)

        plt.imshow(final_image, cmap='gray')
        if figure_title:
            plt.title(figure_title)
        plt.show()

    def calculate_gl_output_index(self, gl_id):
        return gl_id + 1  # start GL indexing with 1 to be compatible with legacy preprocessing

        # cv.imwrite(path, final_image)

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

                if not region_centers:  # output needed data for asserts, if it is not defined
                    res = [region.start + region.width/2 for region in region_list]
                    print(f"'centers': {res}")
                    plt.imshow(imdata, cmap='gray')
                    for ind, region in enumerate(region_list):
                        plt.axvline(region.start, color='r')
                        plt.axvline(region.end, color='g')
                    plt.title(test['name'])
                    plt.show()

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
