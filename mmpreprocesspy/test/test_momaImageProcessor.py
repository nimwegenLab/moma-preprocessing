from unittest import TestCase


class TestMomaImageProcessor(TestCase):
    test_data_base_path = '/home/micha/Documents/git/MM_Testing'

    def test_process_image_dataset_03(self):
        from mmpreprocesspy import preprocessing
        from mmpreprocesspy.moma_image_processing import MomaImageProcessor

        sut = MomaImageProcessor()
        sut.read_image(self.test_data_base_path + '/03_20180604_gluIPTG10uM_lac_lacIoe_1/first_images/Pos0/03_img_000000000_ DIA Ph3 (GFP)_000.tif')
        sut.process_image()

        self.assertEqual(359, sut.mincol)
        self.assertEqual(675, sut.maxcol)
        self.assertEqual(0, sut.main_channel_angle)
