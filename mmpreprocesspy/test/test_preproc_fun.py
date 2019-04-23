from unittest import TestCase


class TestPreproc_fun(TestCase):
    def test_preproc_fun(self):
        from mmpreprocesspy import preproc_fun

        # '/home/micha/Documents/git/MM_Testing/Moma_stacks/20180531_gluIPTG5uM_lac_1_MMStack.ome.tif'

        data_folder = '/home/micha/Documents/git/MM_Testing/MMStacks/20180531_gluIPTG5uM_lac_1_MMStack'
        folder_to_save = '/home/micha/Documents/git/MM_Testing/MMStacks/20180531_gluIPTG5uM_lac_1_MMStack'
        positions = [0]
        maxframe = 10

        preproc_fun.preproc_fun(data_folder, folder_to_save, positions, maxframe)

        pass
