from unittest import TestCase

import cv2
from mmpreprocesspy.dev_auxiliary_functions import show_image_with_growthlane_rois
import mmpreprocesspy.dev_auxiliary_functions as aux


class TestMomaImageProcessor(TestCase):
    test_data_base_path = '/home/micha/Documents/01_work/git/MM_Testing'

    def test_process_image_dataset_03(self):
        from mmpreprocesspy.moma_image_processing import MomaImageProcessor

        sut = MomaImageProcessor()
        sut.read_image(
            self.test_data_base_path + '/03_20180604_gluIPTG10uM_lac_lacIoe_1/first_images/Pos0/03_img_000000000_ DIA Ph3 (GFP)_000.tif')
        sut.process_image()

        self.assertEqual(358, sut.mincol)
        self.assertEqual(678, sut.maxcol)
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

        self.assertEqual(359,
                         sut.mincol)  # NOTE: for some reason mincol and maxcol are shifted by 1 in comparison to test_split_channels_init_dataset_03
        self.assertEqual(679, sut.maxcol)
        self.assertEqual(90, sut.main_channel_angle)

    def test_split_channels_init_dataset_04(self):
        from mmpreprocesspy.moma_image_processing import MomaImageProcessor

        sut = MomaImageProcessor()
        sut.read_image(
            self.test_data_base_path + '/04_20180531_gluIPTG5uM_lac_1/first_images/Pos0/04_img_000000000_ DIA Ph3 (GFP)_000.tif')
        sut.process_image()

        self.assertEqual(378, sut.mincol)
        self.assertEqual(687, sut.maxcol)
        self.assertEqual(0, sut.main_channel_angle)

    def test_split_channels_init_dataset_08(self):
        from mmpreprocesspy.moma_image_processing import MomaImageProcessor

        sut = MomaImageProcessor()
        sut.read_image(
            self.test_data_base_path + '/08_20190222_LB_SpentLB_TrisEDTA_LB_1/first_images/Pos0/img_000000000_ DIA Ph3 (Dual)_000.tif')
        sut.process_image()

        self.assertEqual(375, sut.mincol)
        # self.assertEqual(687, sut.maxcol)
        self.assertEqual(-4, sut.main_channel_angle)

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

    def test__get_growthlane_rois__dataset_11(self):
        from mmpreprocesspy.moma_image_processing import MomaImageProcessor

        sut = MomaImageProcessor()
        sut.growthlane_length_threshold = 300
        sut.read_image("./resources/11_20190910_glc_spcm_1_MMStack.ome-1.tif")
        sut.process_image()
        gl_rois = sut.growthlane_rois

        show_image_with_growthlane_rois(sut.rotated_image, gl_rois)
        cv2.waitKey()

    def test__get_growthlane_rois__20200128_glcIPTG_glc_1_MMStack(self):
        from mmpreprocesspy.moma_image_processing import MomaImageProcessor

        sut = MomaImageProcessor()
        sut.read_image("./resources/20200128_glcIPTG_glc_1_MMStack.ome-1.tif")
        sut.process_image()
        gl_rois = sut.growthlane_rois

        show_image_with_growthlane_rois(sut.rotated_image, gl_rois)
        cv2.waitKey()

    def test_get_oriented_growthlane_rois_dataset_8(self):
        from mmpreprocesspy.moma_image_processing import MomaImageProcessor

        sut = MomaImageProcessor()
        sut.read_image(
            self.test_data_base_path + '/08_20190222_LB_SpentLB_TrisEDTA_LB_1/first_images/Pos0/img_000000000_ DIA Ph3 (Dual)_000.tif')
        sut.process_image()
        gl_rois = sut.growthlane_rois

        first_roi = gl_rois[0].get_oriented_roi_image(sut.image)
        last_roi = gl_rois[-1].get_oriented_roi_image(sut.image)

        aux.show_image(first_roi, 'first_roi')
        aux.show_image(last_roi, 'last_roi')
        cv2.waitKey()

        # show_image_with_growthlane_rois(sut.image, gl_rois)
        # cv2.waitKey()


def read_tiff_to_nparray(image_path):
    """Reads tiff-image and returns it as a numpy-array."""

    from PIL import Image
    import numpy as np

    image_base = Image.open(image_path)
    return np.array(image_base)
