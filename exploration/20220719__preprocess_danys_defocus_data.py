import os
import shutil
import dotenv
from unittest import TestCase
from mmpreprocesspy.preproc_fun import PreprocessingRunner

def __main__():
    tests = TestDanysData()
    tests.test__20220716_zstacks_hi1_500_1()

class TestDanysData(TestCase):

    def setUp(self):
        dotenv.load_dotenv('.env_for_testing')
        self.test_data_base_path = "/media/micha/T7/data_michael_mell/20220718__defocus_analysis_dany/20220716_zstacks/20220716_zstacks"
        self.output_path = '/media/micha/T7/data_michael_mell/20220718__defocus_analysis_dany/20220716_zstacks_preprocessing'

    def test__20220716_zstacks_hi1_500_1(self):
        data_directory = os.path.join(self.test_data_base_path, '20220716_zstacks_hi1_500_1')
        directory_to_save = os.path.join(self.output_path, '20220716_zstacks_output/20220716_zstacks_hi1_500_1')
        gl_detection_template_path = os.path.join(self.output_path, 'template/template_config.json')
        image_registration_method = 1
        positions = [0]
        # minframe = 0
        maxframe = 2
        dark_noise = 90
        gaussian_sigma = 5
        # main_channel_angle = -90.7
        # main_channel_angle = 180
        main_channel_angle = 0.01
        normalization_config_path = 'True'
        normalization_region_offset = 120
        frames_to_ignore = []

        if os.path.isdir(directory_to_save):
            shutil.rmtree(directory_to_save)
        os.makedirs(directory_to_save, exist_ok=True)

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