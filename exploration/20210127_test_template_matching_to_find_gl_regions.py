import inspect
import os
import shutil
from unittest import TestCase


class TestPreproc_fun(TestCase):
    def test__find_channel_regions(self):
        import tifffile as tff
        from mmpreprocesspy import preprocessing
        from mmpreprocesspy.MicroManagerOmeTiffReader import MicroManagerOmeTiffReader
        import matplotlib.pyplot as plt
        import skimage
        from skimage.feature import match_template
        from skimage.transform import rotate
        import numpy as np

        test_data_base_path = '/home/micha/Documents/01_work/git/MM_Testing'

        region_center_tolerance = 10

        tests = self.get_tests__test__find_channel_regions()

        for test in tests:
            path = os.path.join(test_data_base_path, test['path'])
            rotation_angle = test['angle']
            growthlane_length_threshold = test['glt']
            region_centers = test['centers']

            with self.subTest(test=test['name']):
                template_config = self.load_gl_detection_template(test['gl_detection_template'])

                print(f'path: {path}')
                dataset = MicroManagerOmeTiffReader(path)
                frame_index = 0  # work with first frame of each position
                position_index = 0
                pos_ind_range = range(0, 4)

                for position_index in pos_ind_range:
                    current_stack = dataset.get_image_stack(frame_index=frame_index,
                                                            position_index=position_index)
                    imdata = current_stack[:, :, 0]
                    imdata = rotate(imdata, rotation_angle)

                    gl_regions = self.get_gl_regions(imdata, template_config)

                    # self.inspect_template_config(template_config)

                    self.show_gl_regions_on_image(gl_regions, imdata, position_index, test)

                    # plt.imshow(imdata)
                    # plt.title(f"{test['name']}, position: {position_index}")
                    # plt.show()
                    #
                    # template_image = tff.imread(template_config['template_path'])
                    #
                    # # plt.imshow(template)
                    # # plt.show()
                    #
                    # # self.inspect_template_config(template_config)
                    #
                    # normalized_cross_correlation = match_template(imdata, template_image, pad_input=True)
                    #
                    # plt.imshow(normalized_cross_correlation)
                    # plt.title(f"{test['name']}, position: {position_index}")
                    # plt.show()
                    #
                    # plt.plot(np.mean(normalized_cross_correlation, axis=0), color='r', label='mean proj.')
                    # plt.plot(np.max(normalized_cross_correlation, axis=0), color='g', label='max proj.')
                    # plt.legend()
                    # plt.title(f"{test['name']}, position: {position_index}")
                    # plt.show()
                    #
                    # plt.plot(np.mean(normalized_cross_correlation, axis=1), color='r', label='mean proj.')
                    # plt.plot(np.max(normalized_cross_correlation, axis=1), color='g', label='max proj.')
                    # plt.legend()
                    # plt.title(f"{test['name']}, position: {position_index}")
                    # plt.show()

                    pass

    def show_gl_regions_on_image(self, gl_regions, imdata, position_index, test):
        import matplotlib.pyplot as plt

        plt.imshow(min_max_normalize(imdata), cmap='gray', vmin=0, vmax=0.2)
        for region in gl_regions:
            plt.axvline(region.start, color='r', linewidth=0.5)
            plt.axvline(region.end, color='g', linewidth=0.5)
        plt.title(f"{test['name']}, position: {position_index}")
        plt.show()

    def get_gl_regions(self, image, template_config):
        from skimage.feature import match_template
        import tifffile as tff
        import numpy as np
        from mmpreprocesspy.data_region import DataRegion

        template_image = tff.imread(template_config['template_path'])

        normalized_cross_correlation = match_template(image, template_image, pad_input=True)

        max_correlation_ind = np.argmax(np.max(normalized_cross_correlation, axis=0))

        template_gl_regions = np.array(template_config['gl_regions'])
        template_gl_regions = template_gl_regions - (template_image.shape[1] // 2)
        gl_regions = []
        for template_gl_region in template_gl_regions:
            gl_region_in_image = max_correlation_ind + template_gl_region
            gl_regions.append(DataRegion(start=gl_region_in_image[0],
                                         end=gl_region_in_image[1],
                                         width=gl_region_in_image[1] - gl_region_in_image[0]))
        return gl_regions

    def inspect_template_config(self, template_config):
        import tifffile as tff
        import matplotlib.pyplot as plt
        import numpy as np

        template_image = tff.imread(template_config['template_path'])

        first_gl_position = template_config['first_gl_position']
        gl_spacing = template_config['gl_spacing']
        range_max = int(np.floor(template_image.shape[0] / gl_spacing))
        gl_positions = np.floor(first_gl_position + np.arange(range_max) * gl_spacing).astype(dtype=np.int)
        gl_positions = gl_positions[gl_positions < template_image.shape[0]]  # refine valid positions

        plt.imshow(min_max_normalize(template_image), cmap='gray', vmin=0, vmax=0.5)
        for regions in template_config['gl_regions']:
            plt.axvline(regions[0], color='r', linestyle='--', linewidth=1)
            plt.axvline(regions[1], color='g', linestyle='--', linewidth=1)
        for gl_position in gl_positions:
            plt.axhline(gl_position, color='gray', linestyle='--', linewidth=0.5)
        plt.show()

        import json
        print(json.dumps(template_config, indent=2))

        pass

    def load_gl_detection_template(self, gl_detection_template):
        configs = self.get_template_configs()
        for config in configs:
            if config['name'] == gl_detection_template:
                return config

    def get_template_configs(self):
        configs = []

        configs.append({'name': 'template__thomas_20201229_glc_lac_1',
                        'template_path': './data/20210127_test_template_matching_to_find_gl_regions/16_thomas_20201229_glc_lac_1_MMStack.ome-1___template_v01_2.tif',
                        'gl_regions': [[30, 565], [840, 1410]],
                        'first_gl_position': 52,
                        'gl_spacing': 105.75,
                        })
        return configs

    def get_tests__test__find_channel_regions(self):
        tests = []
        tests.append({'name': 'dataset_17',
                      'path': "./17_lis_20201218_VNG40_AB6min_2h_1_1/MMStack/20201218_VNG40_AB6min_2h_1_1_MMStack.ome.tif",
                      'angle': 91,
                      'glt': 200,
                      'centers': [538, 1343],
                      'gl_detection_template': 'template__thomas_20201229_glc_lac_1'})
        tests.append({'name': 'dataset_16',
                      'path': "./16_thomas_20201229_glc_lac_1/MMStack/20201229_glc_lac_1_MMStack.ome.tif",
                      'angle': 90,
                      'glt': 200,
                      'centers': [435.0, 1415.0],
                      'gl_detection_template': 'template__thomas_20201229_glc_lac_1'})
        # tests.append({'name': 'dataset_15',
        #               'path': "./resources/data__test_preprocessing_py/15_lis__20201119_VNG1040_AB2h_2h_1_MMStack.ome.tif",
        #               'angle': 90,
        #               'glt': 200,
        #               'centers': [573, 1409]})
        # tests.append({'name': 'dataset_14',
        #               'path': "./resources/data__test_preprocessing_py/14_thomas_20201228_glc_ara_1__Pos0__rotated.tif",
        #               'angle': -.5,
        #               'glt': 300,
        #               'centers': [489, 1315]})
        # tests.append({'name': 'dataset_13',
        #               'path': "./resources/data__test_preprocessing_py/13_20200128_glcIPTG_glc_1__MMStack.ome.tif",
        #               'angle': 0,
        #               'glt': 200,
        #               'centers': [465, 752]})
        # tests.append({'name': 'dataset_12',
        #               'path': "./resources/data__test_preprocessing_py/12_20190816_Theo_MMStack.ome.tif",
        #               'angle': 0,
        #               'glt': 200,
        #               'centers': [526]})
        # tests.append({'name': 'dataset_11',
        #               'path': "./resources/data__test_preprocessing_py/11_20190910_glc_spcm_1_MMStack.ome.tif",
        #               'angle': 0,
        #               'glt': 300,
        #               'centers': [518, 1355]})
        # tests.append({'name': 'dataset_10',
        #               'path': "./resources/data__test_preprocessing_py/10_20190424_hi2_hi3_med2_rplN_glu_gly.ome.tif",
        #               'angle': 0,
        #               'glt': 300,
        #               'centers': [392, 1249]})
        # tests.append({'name': 'dataset_9',
        #               'path': "./resources/data__test_preprocessing_py/09_20190325_hi1_hi2_med1_rpmB_glu_gly_pl_chr_1.tif",
        #               'angle': 0,
        #               'glt': 300,
        #               'centers': [519, 1372]})
        # tests.append({'name': 'dataset_8',
        #               'path': "./resources/data__test_preprocessing_py/08_20190222_LB_SpentLB_TrisEDTA_LB_1.tif",
        #               'angle': -4,
        #               'glt': 300,
        #               'centers': [591, 1441]})
        # tests.append({'name': 'dataset_7',
        #               'path': "./resources/data__test_preprocessing_py/07_20181203_glu_lac_switch16h_1__Pos0.tif",
        #               'angle': 0,
        #               'glt': 200,
        #               'centers': [461]})
        # tests.append({'name': 'dataset_6',
        #               'path': "./resources/data__test_preprocessing_py/06_20180313_glu_lac_switch24h_1__Pos0.tif",
        #               'angle': 0,
        #               'glt': 300,
        #               'centers': [512]})
        # tests.append({'name': 'dataset_5',
        #               'path': "./resources/data__test_preprocessing_py/05_20180404_glu_lacCM-ara_1__Pos0.tif",
        #               'angle': 0,
        #               'glt': 300,
        #               'centers': [513]})
        # tests.append({'name': 'dataset_4',
        #               'path': "./resources/data__test_preprocessing_py/04_20180531_gluIPTG5uM_lac_1__Pos0.tif",
        #               'angle': 0,
        #               'glt': 300,
        #               'centers': [513]})
        # tests.append({'name': 'dataset_3',
        #               'path': "./resources/data__test_preprocessing_py/03_20180604_gluIPTG10uM_lac_lacIoe_1__Pos0.tif",
        #               'angle': 0,
        #               'glt': 300,
        #               'centers': [484]})
        # tests.append({'name': 'dataset_0',
        #               'path': "./resources/data__test_preprocessing_py/00_20150710_mmtest_2ch__Pos0__rotated.tif",
        #               'angle': -1.5,
        #               'glt': 300,
        #               'centers': [660]})
        return tests

def min_max_normalize(imdata):
    import numpy as np
    return (imdata - np.min(imdata)) / (np.max(imdata) - np.min(imdata))

