import os
import shutil
from unittest import TestCase


class TestPreproc_fun(TestCase):
    test_data_base_path = '/home/micha/Documents/01_work/git/MM_Testing'

    def test__dataset_04(self):
        from mmpreprocesspy import preproc_fun

        data_directory = self.test_data_base_path + '/04_20180531_gluIPTG5uM_lac_1/MMStack'
        directory_to_save = data_directory
        positions = [0]
        maxframe = 10

        results_directory = directory_to_save + '/result/'
        if os.path.isdir(results_directory):
            shutil.rmtree(results_directory)

        preproc_fun.preproc_fun(data_directory, results_directory, positions, maxframe)

    def test__dataset_08(self):
        from mmpreprocesspy import preproc_fun

        data_directory = self.test_data_base_path + '/08_20190222_LB_SpentLB_TrisEDTA_LB_1/MMStack'
        directory_to_save = data_directory
        positions = [0]
        maxframe = 10

        results_directory = directory_to_save + '/result/'
        if os.path.isdir(results_directory):
            shutil.rmtree(results_directory)

        preproc_fun.preproc_fun(data_directory, results_directory, positions, maxframe)

    def test__dataset_10(self):
        from mmpreprocesspy import preproc_fun

        data_directory = self.test_data_base_path + '/10_20190424_hi2_hi3_med2_rplN_glu_gly/MMStack/RawData/measurement'
        flatfield_directory = self.test_data_base_path + '/10_20190424_hi2_hi3_med2_rplN_glu_gly/MMStack/RawData/flatfield'
        directory_to_save = self.test_data_base_path + '/10_20190424_hi2_hi3_med2_rplN_glu_gly/MMStack'
        positions = [0]
        minframe = None
        maxframe = 8
        dark_noise = 90
        gaussian_sigma = 5

        results_directory = directory_to_save + '/result/'
        if os.path.isdir(results_directory):
            shutil.rmtree(results_directory)

        preproc_fun.preproc_fun(data_directory, results_directory, positions, minframe=minframe, maxframe=maxframe, flatfield_directory=flatfield_directory, dark_noise=dark_noise,
                                gaussian_sigma=gaussian_sigma)

    def test__dataset_10__shift_is_determined_correctly(self):
        # There is jump of the image position in dataset 10 going from frame 603 to 604. This trips up the current
        # current implementation. This test was created for debugging this issue.

        from mmpreprocesspy import preproc_fun

        data_directory = self.test_data_base_path + '/10_20190424_hi2_hi3_med2_rplN_glu_gly/MMStack/RawData/measurement'
        flatfield_directory = self.test_data_base_path + '/10_20190424_hi2_hi3_med2_rplN_glu_gly/MMStack/RawData/flatfield'
        directory_to_save = self.test_data_base_path + '/10_20190424_hi2_hi3_med2_rplN_glu_gly/MMStack/result'
        positions = [0]
        minframe = 590
        maxframe = 610
        dark_noise = 90
        gaussian_sigma = 5

        results_directory = directory_to_save + '/result/'
        if os.path.isdir(results_directory):
            shutil.rmtree(results_directory)

        preproc_fun.preproc_fun(data_directory, results_directory, positions, minframe=minframe, maxframe=maxframe, flatfield_directory=flatfield_directory, dark_noise=dark_noise,
                                gaussian_sigma=gaussian_sigma)

    def test__get_gl_tiff_path(self):
        from mmpreprocesspy import preproc_fun

        gl_file_path = preproc_fun.get_gl_tiff_path('/path/to/file', 'experiment_name', '1', '5')
        print(gl_file_path)
        print(os.path.dirname(gl_file_path))

    def test__get_kymo_tiff_path(self):
        from mmpreprocesspy import preproc_fun

        kymo_file_path = preproc_fun.get_kymo_tiff_path('/path/to/file', 'experiment_name', '1', '5', '0')
        print(kymo_file_path)
        print(os.path.dirname(kymo_file_path))
