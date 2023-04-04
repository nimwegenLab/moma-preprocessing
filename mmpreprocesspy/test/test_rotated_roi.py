from unittest import TestCase

import numpy as np
from PIL import Image
import mmpreprocesspy.dev_auxiliary_functions as aux
from mmpreprocesspy.roi import Roi
from mmpreprocesspy.rotated_roi import RotatedRoi
import cv2


class TestRoi(TestCase):
    test_data_base_path: str = '/media/micha/T7/data_michael_mell/preprocessing_test_data/MM_Testing'

    def test__get_roi_from_image(self):
        imdata = read_image('resources/data__test_rotated_roi_py/03_img_000000000_ DIA Ph3 (GFP)_000.tif')
        angle = -15
        center = (imdata.shape[1]/2, imdata.shape[0]/2)
        M = cv2.getRotationMatrix2D(center,angle, 1.0)
        imdata = cv2.warpAffine(imdata, M, (imdata.shape[1], imdata.shape[0]), cv2.INTER_CUBIC)
        # aux.show_image(imdata)

        im_center = imdata.shape[1]/2, imdata.shape[0]/2
        roi_size = (400, 100)

        sut = RotatedRoi(im_center, roi_size, angle)
        # aux.show_image_with_rotated_rois(imdata, [sut])

        imdata_extended = imdata[np.newaxis, ...]  # add axis at front, because RotatedRoi.get_from_image now expects multi-channel images/np-arrays
        roi_image_extended = sut.get_from_image(imdata_extended)
        roi_image = roi_image_extended[0, ...]

        # np.save('resources/data__test_rotated_roi_py/test__get_roi_from_image__expected_00.npy',roi_image)  # for updating the expected image
        expected = np.load('resources/data__test_rotated_roi_py/test__get_roi_from_image__expected_00.npy')
        self.assertTrue(np.all(expected == roi_image))
        # aux.show_image(roi_image)
        #
        # cv2.waitKey()

    def test__rotate(self):
        imdata = read_image('resources/data__test_rotated_roi_py/03_img_000000000_ DIA Ph3 (GFP)_000.tif')
        angle = -15
        # angle = 0
        image_center = (imdata.shape[1]/2, imdata.shape[0]/2)
        roi_center = 542, 140
        roi_size = (400, 100)
        sut = RotatedRoi(roi_center, roi_size, 0)

        # Plot non-rotated ROI against non-rotated image
        # aux.show_image_with_rotated_rois(imdata, [sut])

        M = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        imdata_rotated = cv2.warpAffine(imdata, M, (imdata.shape[1], imdata.shape[0]), cv2.INTER_CUBIC)

        sut.rotate(image_center, angle)

        # Plot rotated ROI against rotated image
        # aux.show_image_with_rotated_rois(imdata_rotated, [sut])

        imdata_rotated_extended = imdata_rotated[np.newaxis, ...]  # add axis at front, because RotatedRoi.get_from_image now expects multi-channel images/np-arrays
        roi_image_extended = sut.get_from_image(imdata_rotated_extended)
        roi_image = roi_image_extended[0, ...]

        # np.save('resources/data__test_rotated_roi_py/test__rotate__expected_00.npy',roi_image)  # for updating the expected image
        expected = np.load('resources/data__test_rotated_roi_py/test__rotate__expected_00.npy')
        self.assertTrue(np.all(expected == roi_image))

        # aux.show_image(roi_image)
        #
        # cv2.waitKey()

    def test__translate__updates_roi_center_correctly(self):
        roi_center = 542, 140
        roi_size = (400, 100)
        sut = RotatedRoi(roi_center, roi_size, 0)
        shift_value = (50, 100)
        sut.translate(shift_value)
        self.assertEqual(542+50, sut.center[0])
        self.assertEqual(140 + 100, sut.center[1])

    def test__is_inside_image__for_roi_in_image__returns_true(self):
        imdata = read_image(
            self.test_data_base_path + '/03_20180604_gluIPTG10uM_lac_lacIoe_1/first_images/Pos0/03_img_000000000_ DIA Ph3 (GFP)_000.tif')
        roi_center = 200, 200
        roi_size = (100, 100)
        sut = RotatedRoi(roi_center, roi_size, 0)

        self.assertTrue(sut.is_inside_image(imdata))

    def test__is_inside_image__for_roi_same_as_image_on_left__returns_true(self):
        imdata = read_image(
            self.test_data_base_path + '/03_20180604_gluIPTG10uM_lac_lacIoe_1/first_images/Pos0/03_img_000000000_ DIA Ph3 (GFP)_000.tif')
        roi_center = (imdata.shape[1]/2, imdata.shape[0]/2)
        roi_size = (imdata.shape[1], imdata.shape[0])
        sut = RotatedRoi(roi_center, roi_size, 0)

        self.assertTrue(sut.is_inside_image(imdata))

    def test__is_inside_image__for_roi_outside_image_on_left__returns_false(self):
        imdata = read_image(
            self.test_data_base_path + '/03_20180604_gluIPTG10uM_lac_lacIoe_1/first_images/Pos0/03_img_000000000_ DIA Ph3 (GFP)_000.tif')
        roi_center = 0, 200
        roi_size = (100, 100)
        sut = RotatedRoi(roi_center, roi_size, 0)

        self.assertFalse(sut.is_inside_image(imdata))

    def test__is_inside_image__for_roi_outside_image_on_right__returns_false(self):
        imdata = read_image(
            self.test_data_base_path + '/03_20180604_gluIPTG10uM_lac_lacIoe_1/first_images/Pos0/03_img_000000000_ DIA Ph3 (GFP)_000.tif')
        im_width = imdata.shape[1]
        roi_center = im_width, 200
        roi_size = (100, 100)
        sut = RotatedRoi(roi_center, roi_size, 0)

        self.assertFalse(sut.is_inside_image(imdata))

    def test__is_inside_image__for_roi_outside_image_at_top__returns_false(self):
        imdata = read_image(
            self.test_data_base_path + '/03_20180604_gluIPTG10uM_lac_lacIoe_1/first_images/Pos0/03_img_000000000_ DIA Ph3 (GFP)_000.tif')
        roi_center = 200, 0
        roi_size = (100, 100)
        sut = RotatedRoi(roi_center, roi_size, 0)

        self.assertFalse(sut.is_inside_image(imdata))

    def test__is_inside_image__for_roi_outside_image_at_bottom__returns_false(self):
        imdata = read_image(
            self.test_data_base_path + '/03_20180604_gluIPTG10uM_lac_lacIoe_1/first_images/Pos0/03_img_000000000_ DIA Ph3 (GFP)_000.tif')
        im_height = imdata.shape[0]
        roi_center = 200, im_height
        roi_size = (100, 100)
        sut = RotatedRoi(roi_center, roi_size, 0)

        self.assertFalse(sut.is_inside_image(imdata))

    def test__create_from_roi__returns_rotated_roi_with_correct_center_coordinate(self):
        roi = Roi(0, 0, 100, 100)

        sut = RotatedRoi.create_from_roi(roi)

        self.assertEqual(50, sut.center[0])
        self.assertEqual(50, sut.center[1])

    def test__create_from_roi__returns_rotated_roi_with_correct_width(self):
        roi = Roi(0, 0, 100, 100)

        sut = RotatedRoi.create_from_roi(roi)

        self.assertEqual(100, sut.width)

    def test__create_from_roi__returns_rotated_roi_with_correct_height(self):
        roi = Roi(0, 0, 100, 100)

        sut = RotatedRoi.create_from_roi(roi)

        self.assertEqual(100, sut.height)

    def test__create_from_roi__returns_rotated_roi_returns_same_image_as_input_roi(self):
        from mmpreprocesspy.roi import Roi
        imdata = read_image(
            self.test_data_base_path + '/03_20180604_gluIPTG10uM_lac_lacIoe_1/first_images/Pos0/03_img_000000000_ DIA Ph3 (GFP)_000.tif')

        roi = Roi(90, 342, 190, 742)  # size: (width=400, height=100)
        sut = RotatedRoi.create_from_roi(roi)

        imdata_extended = imdata[np.newaxis, ...]  # add axis at front, because RotatedRoi.get_from_image now expects multi-channel images/np-arrays
        roi_image_extended = sut.get_from_image(imdata_extended)
        roi_image = roi_image_extended[0, ...]


        self.assertEqual(roi_image.shape[0], 100)
        self.assertEqual(roi_image.shape[1], 400)
        # aux.show_image(roi_image)
        # cv2.waitKey()

    def test__width_setter__modifies_width_value_correctly(self):
        roi_center = 100, 100
        roi_size = (100, 100)
        sut = RotatedRoi(roi_center, roi_size, 0)

        sut.width += 100

        self.assertEqual(200, sut.width)

    def test__width_setter__modifies_value_of_size_correctly(self):
        roi_center = 100, 100
        roi_size = (100, 100)
        sut = RotatedRoi(roi_center, roi_size, 0)
        sut.width += 100

        self.assertEqual(200, sut.size[0])

    def test__height_setter__modifies_height_value_correctly(self):
        roi_center = 100, 100
        roi_size = (100, 100)
        sut = RotatedRoi(roi_center, roi_size, 0)

        sut.height += 100

        self.assertEqual(200, sut.height)

    def test__height_setter__modifies_modifies_value_of_size_correctly(self):
        roi_center = 100, 100
        roi_size = (100, 100)
        sut = RotatedRoi(roi_center, roi_size, 0)

        sut.height += 100

        self.assertEqual(200, sut.size[1])

    def test__size_setter__modifies_size_correctly(self):
        roi_center = 100, 100
        roi_size = (100, 100)
        sut = RotatedRoi(roi_center, roi_size, 0)

        sut.size = (200, 200)

        self.assertEqual((200, 200), sut.size)

    def test__size_setter__modifies_width_correctly(self):
        roi_center = 100, 100
        roi_size = (100, 100)
        sut = RotatedRoi(roi_center, roi_size, 0)

        sut.size = (200, 300)

        self.assertEqual(200, sut.width)

    def test__size_setter__modifies_height_correctly(self):
        roi_center = 100, 100
        roi_size = (100, 100)
        sut = RotatedRoi(roi_center, roi_size, 0)

        sut.size = (200, 300)

        self.assertEqual(300, sut.height)


def read_image(image_path):
    """Reads tiff-image and returns it as a numpy-array."""
    image_base = Image.open(image_path)
    return np.array(image_base, dtype=np.uint16)
