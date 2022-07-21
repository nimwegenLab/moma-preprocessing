import os
import shutil
import dotenv
from unittest import TestCase
from mmpreprocesspy.preproc_fun import PreprocessingRunner

class TestDanysData(object):
    def __init__(self):
        dotenv.load_dotenv('.env_for_testing')
        self.image_base_path = "/media/micha/T7/data_michael_mell/20220718__defocus_analysis_dany/20220716_zstacks/20220716_zstacks"
        self.flatfield_path = "/media/micha/T7/data_michael_mell/20220718__defocus_analysis_dany/20220719_flatfield/20220719_flatfield 20 exposure cyan30 110ms blanking_4"
        self.output_path = '/media/micha/T7/data_michael_mell/20220718__defocus_analysis_dany/20220716_zstacks_preprocessing'

    def test__20220716_zstacks_hi1_500_1(self):
        data_directory = os.path.join(self.image_base_path, '20220716_zstacks_hi1_500_1')
        flatfield_directory = self.flatfield_path
        directory_to_save = os.path.join(self.output_path, '20220716_zstacks_output/20220716_zstacks_hi1_500_1')
        gl_detection_template_path = os.path.join(self.output_path, 'template/template_config.json')
        image_registration_method = 1
        # positions = list(range(0, 21))
        positions = list(range(0, 21))
        maxframe = None
        dark_noise = 90
        gaussian_sigma = 5
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
                           flatfield_directory=flatfield_directory,
                           image_registration_method=image_registration_method,
                           normalization_region_offset=normalization_region_offset,
                           frames_to_ignore=frames_to_ignore)

    def test__20220716_zstacks_hi1_500_downward_1(self):
        data_directory = os.path.join(self.image_base_path, '20220716_zstacks_hi1_500_downward_1')
        flatfield_directory = self.flatfield_path
        directory_to_save = os.path.join(self.output_path, '20220716_zstacks_output/20220716_zstacks_hi1_500_downward_1')
        gl_detection_template_path = os.path.join(self.output_path, 'template/template_config.json')
        image_registration_method = 1
        positions = list(range(0, 21))
        maxframe = None
        dark_noise = 90
        gaussian_sigma = 5
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
                           flatfield_directory=flatfield_directory,
                           image_registration_method=image_registration_method,
                           normalization_region_offset=normalization_region_offset,
                           frames_to_ignore=frames_to_ignore)

class DataPreprocessor(object):
    def __init__(self, image_base_path, flatfield_path, output_path, gl_detection_template_path, position):
        self.image_path = image_base_path
        self.flatfield_path = flatfield_path
        self.output_path = output_path
        self.gl_detection_template_path = gl_detection_template_path
        self.position = position

    def process(self):
        data_directory = self.image_path
        flatfield_directory = self.flatfield_path
        output_path = self.output_path
        gl_detection_template_path = self.gl_detection_template_path
        image_registration_method = 1
        positions = [self.position]
        maxframe = None
        dark_noise = 90
        gaussian_sigma = 5
        main_channel_angle = 0.01
        normalization_config_path = 'True'
        normalization_region_offset = 120
        frames_to_ignore = []

        runner = PreprocessingRunner()
        runner.preproc_fun(data_directory, output_path, positions,
                           # minframe=minframe,
                           maxframe=maxframe,
                           dark_noise=dark_noise,
                           gaussian_sigma=gaussian_sigma,
                           main_channel_angle=main_channel_angle,
                           gl_detection_template_path=gl_detection_template_path,
                           normalization_config_path=normalization_config_path,
                           flatfield_directory=flatfield_directory,
                           image_registration_method=image_registration_method,
                           normalization_region_offset=normalization_region_offset,
                           frames_to_ignore=frames_to_ignore)

def process__20220716_zstacks_hi1_500_1():
    image_base_path = "/media/micha/T7/data_michael_mell/20220718__defocus_analysis_dany/20220716_zstacks/20220716_zstacks/20220716_zstacks_hi1_500_1"
    flatfield_path = "/media/micha/T7/data_michael_mell/20220718__defocus_analysis_dany/20220719_flatfield/20220719_flatfield 20 exposure cyan30 110ms blanking_4"
    output_path = '/media/micha/T7/data_michael_mell/20220718__defocus_analysis_dany/20220716_zstacks_preprocessing/20220716_zstacks_output/20220716_zstacks_hi1_500_1'
    gl_detection_template_path = '/media/micha/T7/data_michael_mell/20220718__defocus_analysis_dany/20220716_zstacks_preprocessing/template/template_config.json'

    if os.path.isdir(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path, exist_ok=True)

    positions = list(range(0, 21))
    for position in positions:
        dataprocessor = DataPreprocessor(image_base_path, flatfield_path, output_path, gl_detection_template_path, position)
        dataprocessor.process()

def process__20220716_zstacks_hi1_500_downward_1():
    image_base_path = "/media/micha/T7/data_michael_mell/20220718__defocus_analysis_dany/20220716_zstacks/20220716_zstacks/20220716_zstacks_hi1_500_downward_1"
    flatfield_path = "/media/micha/T7/data_michael_mell/20220718__defocus_analysis_dany/20220719_flatfield/20220719_flatfield 20 exposure cyan30 110ms blanking_4"
    output_path = '/media/micha/T7/data_michael_mell/20220718__defocus_analysis_dany/20220716_zstacks_preprocessing/20220716_zstacks_output/20220716_zstacks_hi1_500_downward_1'
    gl_detection_template_path = '/media/micha/T7/data_michael_mell/20220718__defocus_analysis_dany/20220716_zstacks_preprocessing/template/template_config.json'

    if os.path.isdir(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path, exist_ok=True)

    positions = list(range(0, 21))
    for position in positions:
        dataprocessor = DataPreprocessor(image_base_path, flatfield_path, output_path, gl_detection_template_path, position)
        dataprocessor.process()

if __name__ == "__main__":
    process__20220716_zstacks_hi1_500_1()
    process__20220716_zstacks_hi1_500_downward_1()


