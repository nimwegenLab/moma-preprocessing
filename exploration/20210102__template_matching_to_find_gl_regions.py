import inspect
import os
import shutil
from unittest import TestCase
import json

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

        for test in [tests[0]]:
            path = os.path.join(test_data_base_path, test['path'])
            rotation_angle = test['angle']
            growthlane_length_threshold = test['glt']
            region_centers = test['centers']

            configs = self.get_template_configs()
            for template_config in configs:
                converted_template_config = self.convert_old_config(template_config)
                converted_template_micron = self.convert_config_to_microns(converted_template_config, pixel_size_in_microns=0.065)
                base_path = '/home/micha/Documents/01_work/git/mmpreprocesspy/exploration/data/20210127_test_template_matching_to_find_gl_regions/'
                output_path = os.path.join(base_path, template_config['name'] + '.json')
                self.write_template_config_to_json(converted_template_micron, output_path)

    def write_template_config_to_json(self, template_config, path):
        import json

        json_dump = json.dumps(template_config, indent=2)
        with open(path, 'w') as output_file:
            json.dump(template_config, output_file, indent=2, sort_keys=True)
        print(json_dump)
        pass

    def convert_old_config(self, old_template_config):
        new_template_config = {}
        # new_template_config['template_image_path'] = old_template_config['template_path']
        new_template_config['name'] = old_template_config['name']
        new_template_config['description'] = ''
        new_template_config['template_image_path'] = os.path.join('./', os.path.basename(old_template_config['template_path']))
        new_template_config['gl_regions'] = []
        for range in old_template_config['gl_regions']:
            gl_entry = {}
            gl_entry['horizontal_range'] = range
            gl_entry['first_gl_position_from_top'] = old_template_config['first_gl_position']
            gl_entry['gl_spacing_vertical'] = old_template_config['gl_spacing']
            new_template_config['gl_regions'].append(gl_entry)
        return new_template_config

    def convert_config_to_microns(self, template_config, pixel_size_in_microns):
        pixel_size = pixel_size_in_microns
        template_config['pixel_size_micron'] = pixel_size_in_microns
        for dict in template_config['gl_regions']:
            dict['first_gl_position_from_top'] *= pixel_size
            dict['gl_spacing_vertical'] *= pixel_size
            for ind, entry in enumerate(dict['horizontal_range']):
                dict['horizontal_range'][ind] *= pixel_size
        return template_config

    def get_template_configs(self):
        configs = []

        configs.append({'name': 'template__thomas_20201229_glc_lac_1',
                        'template_path': './data/20210127_test_template_matching_to_find_gl_regions/16_thomas_20201229_glc_lac_1_MMStack.ome-1___template_v01_2.tif',
                        'gl_regions': [[30, 565], [840, 1410]],
                        'first_gl_position': 52,
                        'gl_spacing': 105.75,
                        })
        configs.append({'name': '17_lis_20201218_VNG40_AB6min_2h_1_1__template_v000',
                        'template_path': './data/20210127_test_template_matching_to_find_gl_regions/17_lis_20201218_VNG40_AB6min_2h_1_1__template_v000.tif',
                        'gl_regions': [[40, 610], [888, 1417]],
                        'first_gl_position': 51,
                        'gl_spacing': 105.75,
                        })
        configs.append({'name': '13_20200128_glcIPTG_glc_1_MMStack__template__left_side__v000',
                        'template_path': './data/20210127_test_template_matching_to_find_gl_regions/13_20200128_glcIPTG_glc_1_MMStack__template__left_side__v000.tif',
                        'gl_regions': [[27, 333]],
                        'first_gl_position': 58,
                        'gl_spacing': 105.75,
                        })
        configs.append({'name': '12_theo_20190816_glc_spcm_1_MMStack_2__template_v00',
                        'template_path': './data/20210127_test_template_matching_to_find_gl_regions/12_theo_20190816_glc_spcm_1_MMStack_2__template_v00.tif',
                        'gl_regions': [[40, 422]],
                        'first_gl_position': 38,
                        'gl_spacing': 72.8,
                        })
        configs.append({'name': '11_20190910_glc_spcm_1_MMStack__template_v00',
                        'template_path': './data/20210127_test_template_matching_to_find_gl_regions/11_20190910_glc_spcm_1_MMStack__template_v00.tif',
                        'gl_regions': [[342, 894], [1180, 1732]],
                        'first_gl_position': 45,
                        'gl_spacing': 106.71,
                        })
        return configs