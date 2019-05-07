from unittest import TestCase

import cv2
from mmpreprocesspy.dev_auxiliary_functions import show_image_with_growthlane_rois


class TestMomaImageProcessor(TestCase):
    test_data_base_path = '/home/micha/Documents/git/MM_Testing'

    def test_process_image_dataset_03(self):
        from mmpreprocesspy.moma_image_processing import MomaImageProcessor

        sut = MomaImageProcessor()
        sut.read_image(
            self.test_data_base_path + '/03_20180604_gluIPTG10uM_lac_lacIoe_1/first_images/Pos0/03_img_000000000_ DIA Ph3 (GFP)_000.tif')
        sut.process_image()

        self.assertEqual(359, sut.mincol)
        self.assertEqual(675, sut.maxcol)
        self.assertEqual(0, sut.main_channel_angle)

    def test_split_channels_init_dataset_03_rotated(self):
        from mmpreprocesspy.moma_image_processing import MomaImageProcessor
        import skimage

        image_array = read_tiff_to_nparray(
            self.test_data_base_path + '/03_20180604_gluIPTG10uM_lac_lacIoe_1/first_images/Pos0/03_img_000000000_ DIA Ph3 (GFP)_000.tif')
        image_array = skimage.transform.rotate(image_array, 90, resize=True)

        sut = MomaImageProcessor()
        sut.load_numpy_image_array(image_array)
        sut.process_image()

        self.assertEqual(360,
                         sut.mincol)  # NOTE: for some reason mincol and maxcol are shifted by 1 in comparison to test_split_channels_init_dataset_03
        self.assertEqual(676, sut.maxcol)
        self.assertEqual(90, sut.main_channel_angle)

    def test_split_channels_init_dataset_04(self):
        from mmpreprocesspy.moma_image_processing import MomaImageProcessor

        sut = MomaImageProcessor()
        sut.read_image(
            self.test_data_base_path + '/04_20180531_gluIPTG5uM_lac_1/first_images/Pos0/04_img_000000000_ DIA Ph3 (GFP)_000.tif')
        sut.process_image()

        self.assertEqual(380, sut.mincol)
        self.assertEqual(693, sut.maxcol)
        self.assertEqual(0, sut.main_channel_angle)

    def test_split_channels_init_dataset_08(self):
        from mmpreprocesspy.moma_image_processing import MomaImageProcessor

        sut = MomaImageProcessor()
        sut.read_image(
            self.test_data_base_path + '/08_20190222_LB_SpentLB_TrisEDTA_LB_1/first_images/Pos0/img_000000000_ DIA Ph3 (Dual)_000.tif')
        sut.process_image()

        self.assertEqual(380, sut.mincol)
        self.assertEqual(693, sut.maxcol)
        self.assertEqual(0, sut.main_channel_angle)

    def test_get_growthlane_rois_dataset_3(self):
        from mmpreprocesspy.moma_image_processing import MomaImageProcessor

        sut = MomaImageProcessor()
        sut.read_image(
            self.test_data_base_path + '/03_20180604_gluIPTG10uM_lac_lacIoe_1/first_images/Pos0/03_img_000000000_ DIA Ph3 (GFP)_000.tif')
        sut.process_image()

        gl_rois = sut.growthlane_rois

        show_image_with_growthlane_rois(sut.rotated_image, gl_rois)
        cv2.waitKey()

    def test_get_growthlane_rois_dataset_4(self):
        from mmpreprocesspy.moma_image_processing import MomaImageProcessor

        sut = MomaImageProcessor()
        sut.read_image(
            self.test_data_base_path + '/04_20180531_gluIPTG5uM_lac_1/first_images/Pos0/04_img_000000000_ DIA Ph3 (GFP)_000.tif')
        sut.process_image()
        gl_rois = sut.growthlane_rois

        show_image_with_growthlane_rois(sut.rotated_image, gl_rois)
        cv2.waitKey()

    def test_get_growthlane_rois_dataset_8(self):
        from mmpreprocesspy.moma_image_processing import MomaImageProcessor

        sut = MomaImageProcessor()
        sut.read_image(
            self.test_data_base_path + '/08_20190222_LB_SpentLB_TrisEDTA_LB_1/first_images/Pos0/img_000000000_ DIA Ph3 (Dual)_000.tif')
        sut.process_image()
        gl_rois = sut.growthlane_rois

        show_image_with_growthlane_rois(sut.image, gl_rois)
        cv2.waitKey()

    def test_get_growthlane_rois_dataset_9(self):
        from mmpreprocesspy.moma_image_processing import MomaImageProcessor

        sut = MomaImageProcessor()
        sut.read_image(
            self.test_data_base_path + '/09_20190325_hi1_hi2_med1_rpmB_glu_gly_pl_chr_1/first_images/20190325_hi1_hi2_med1_rpmB_glu_gly_pl_chr_1_MMStack_1.ome-1.tif')
        sut.process_image()
        gl_rois = sut.growthlane_rois

        show_image_with_growthlane_rois(sut.rotated_image, gl_rois)
        cv2.waitKey()


def read_tiff_to_nparray(image_path):
    """Reads tiff-image and returns it as a numpy-array."""

    from PIL import Image
    import numpy as np

    image_base = Image.open(image_path)
    return np.array(image_base)
