import skimage
from mmpreprocesspy import preprocessing
from PIL import Image
import numpy as np
from mmpreprocesspy.preprocessing import get_growthlane_regions
from skimage.feature import match_template
import cv2 as cv

class MomaImageProcessor(object):
    """ MomaImageProcessor encapsulates the processing of a Mothermachine image. """

    def __init__(self):
        self.image = None
        self.rotated_image = None
        self.main_channel_angle = None
        self.mincol = None
        self.maxcol = None
        self.channel_centers = None
        self.growthlane_rois = None
        self.template = None
        self.hor_mid = None
        self.hor_width = None
        self.mid_row = None
        self.vertical_shift = None
        self.horizontal_shift = None

    def load_numpy_image_array(self, image):
        self.image = image

    def read_image(self, image_path):
        """Reads tiff-image and returns it as a numpy-array."""
        image_base = Image.open(image_path)
        self.image = np.array(image_base, dtype=np.uint16)

    def process_image(self):
        self.rotated_image, self.main_channel_angle, self.mincol, self.maxcol, self.channel_centers = preprocessing.split_channels_init(
            self.image)
        self.growthlane_rois = get_growthlane_regions(self.channel_centers, self.mincol, self.maxcol)
        self.get_image_registration_template()

    def get_image_registration_template(self):
        self.template, self.mid_row, self.hor_mid, self.hor_width = preprocessing.get_image_registration_template(self.image, self.mincol)

    def determine_image_shift(self, image):
        image_number_region = image[self.mid_row - 50:self.mid_row + 50, self.hor_mid - self.hor_width:self.hor_mid + self.hor_width]

        result = match_template(self.template, image_number_region, pad_input=True)
        ij = np.unravel_index(np.argmax(result), result.shape)
        t1, t0 = ij[::-1]
        self.vertical_shift = int(t0 - self.template.shape[0] / 2)
        self.horizontal_shift = int(t1 - self.template.shape[1] / 2)

    def get_registered_image(self, image_to_register):
        registered_image = self._translate_image(image_to_register)
        registered_image = self._rotate_image(registered_image)
        return registered_image

    def _translate_image(self, image):
        return np.roll(image, (self.vertical_shift, self.horizontal_shift), axis=(0, 1))  # shift image

    def _rotate_image(self, image):
        return skimage.transform.rotate(image, self.main_channel_angle, cval=0)  # rotate image

    def get_transformation_matrix(self):
        if self.main_channel_angle is None:
            raise ValueError("self.main_channel_angle must be set before calling self.get_transformation_matrix")
        if self.vertical_shift is None:
            raise ValueError("self.vertical_shift must be set before calling self.get_transformation_matrix")
        if self.horizontal_shift is None:
            raise ValueError("self.horizontal_shift must be set before calling self.get_transformation_matrix")

        rotation_center = self.image.shape/2

        return preprocessing.get_transformation_matrix(self.horizontal_shift, self.vertical_shift, self.main_channel_angle, rotation_center)

        raise NotImplementedError