from unittest import TestCase

import numpy as np
from PIL import Image
import mmpreprocesspy.dev_auxiliary_functions as aux
from mmpreprocesspy.rotated_roi import RotatedRoi
import cv2

class TestRoi(TestCase):
    test_data_base_path = '/home/micha/Documents/git/MM_Testing'

    def test__get_roi_from_image(self):
        imdata = read_image(
            self.test_data_base_path + '/03_20180604_gluIPTG10uM_lac_lacIoe_1/first_images/Pos0/03_img_000000000_ DIA Ph3 (GFP)_000.tif')
        angle = -15
        center = (imdata.shape[1]/2, imdata.shape[0]/2)
        M = cv2.getRotationMatrix2D(center,angle, 1.0)
        imdata = cv2.warpAffine(imdata, M, (imdata.shape[1], imdata.shape[0]), cv2.INTER_CUBIC)
        # aux.show_image(imdata)

        im_center = imdata.shape[1]/2, imdata.shape[0]/2
        roi_size = (400, 100)

        sut = RotatedRoi(im_center, roi_size, angle)
        # aux.show_image_with_rotated_rois(imdata, [sut])

        roi_image = sut.get_roi_from_image(imdata)
        aux.show_image(roi_image)

        cv2.waitKey()

    def test__rotate(self):
        imdata = read_image(
            self.test_data_base_path + '/03_20180604_gluIPTG10uM_lac_lacIoe_1/first_images/Pos0/03_img_000000000_ DIA Ph3 (GFP)_000.tif')
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
        aux.show_image_with_rotated_rois(imdata_rotated, [sut])

        roi_image = sut.get_roi_from_image(imdata_rotated)
        aux.show_image(roi_image)

        cv2.waitKey()

    def test__translate__updates_roi_center_correctly(self):
        roi_center = 542, 140
        roi_size = (400, 100)
        sut = RotatedRoi(roi_center, roi_size, 0)
        shift_value = (50, 100)
        sut.translate(shift_value)
        self.assertEqual(542+50, sut.center[0])
        self.assertEqual(140 + 100, sut.center[1])

def read_image(image_path):
    """Reads tiff-image and returns it as a numpy-array."""
    image_base = Image.open(image_path)
    return np.array(image_base, dtype=np.uint16)
