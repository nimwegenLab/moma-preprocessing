import os
import shutil
from unittest import TestCase


class TestPreproc_fun(TestCase):
    def test__dataset_04(self):
        from mmpreprocesspy import preproc_fun

        data_directory = '/home/micha/Documents/git/MM_Testing/04_20180531_gluIPTG5uM_lac_1/MMStack'
        directory_to_save = '/home/micha/Documents/git/MM_Testing/04_20180531_gluIPTG5uM_lac_1/MMStack'
        positions = [0]
        maxframe = 10

        results_directory = directory_to_save + '/result/'
        if os.path.isdir(results_directory):
            shutil.rmtree(results_directory)

        preproc_fun.preproc_fun(data_directory, results_directory, positions, maxframe)

    def test__dataset_08(self):
        from mmpreprocesspy import preproc_fun

        data_directory = '/home/micha/Documents/git/MM_Testing/08_20190222_LB_SpentLB_TrisEDTA_LB_1/MMStack'
        directory_to_save = '/home/micha/Documents/git/MM_Testing/08_20190222_LB_SpentLB_TrisEDTA_LB_1/MMStack'
        positions = [0]
        maxframe = 10

        results_directory = directory_to_save + '/result/'
        if os.path.isdir(results_directory):
            shutil.rmtree(results_directory)

        preproc_fun.preproc_fun(data_directory, results_directory, positions, maxframe)
