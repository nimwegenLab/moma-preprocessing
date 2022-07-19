import os
import shutil
import dotenv
from unittest import TestCase
from mmpreprocesspy.preproc_fun import PreprocessingRunner


class TestPreproc_fun(TestCase):
    def setUp(self):
        dotenv.load_dotenv('.env_for_testing')
        self.test_data_base_path = os.getenv('PREPROCDATADIR')

    def test__35__lis__20220320__repeat(self):
        data_directory = self.test_data_base_path + '/35__lis__20220320__repeat/MMSTACK/20220320_VNG1040_AB2h_1_Frame0-478_resaved/'
        directory_to_save = self.test_data_base_path + '/35__lis__20220320__repeat/result_without_flatfield/'
        gl_detection_template_path = self.test_data_base_path + '/35__lis__20220320__repeat/TEMPLATE_MICHAEL_v002/template_config.json'
        image_registration_method = 1
        positions = [0]
        # minframe = 0
        maxframe = 2
        dark_noise = 90
        gaussian_sigma = 5
        # main_channel_angle = -90.7
        # main_channel_angle = 180
        main_channel_angle = 90.1
        normalization_config_path = 'True'
        normalization_region_offset = 120
        frames_to_ignore = []

        if os.path.isdir(directory_to_save):
            shutil.rmtree(directory_to_save)

        runner = PreprocessingRunner()
        runner.preproc_fun(data_directory, directory_to_save, positions,
                           # minframe=minframe,
                           maxframe=maxframe,
                           dark_noise=dark_noise,
                           gaussian_sigma=gaussian_sigma,
                           main_channel_angle=main_channel_angle,
                           gl_detection_template_path=gl_detection_template_path,
                           normalization_config_path=normalization_config_path,
                           image_registration_method=image_registration_method,
                           normalization_region_offset=normalization_region_offset,
                           frames_to_ignore=frames_to_ignore)

        self.read_and_show_gl_index_image(os.path.join(directory_to_save, f'Pos{positions[0]}_GL_index_initial.tif'),
                                          title='test__dataset_21_no_flatfield')