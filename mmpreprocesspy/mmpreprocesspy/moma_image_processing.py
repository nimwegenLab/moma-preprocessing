import skimage
from mmpreprocesspy import preprocessing
from PIL import Image
import numpy as np
from mmpreprocesspy.preprocessing import get_growthlane_regions
from skimage.feature import match_template


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

    def load_numpy_image_array(self, image):
        self.image = image

    def read_image(self, image_path):
        """Reads tiff-image and returns it as a numpy-array."""
        image_base = Image.open(image_path)
        self.image = np.array(image_base)

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
        self.t0 = int(t0 - self.template.shape[0] / 2)
        self.t1 = int(t1 - self.template.shape[1] / 2)

    def get_registered_image(self, image_to_register):
        registered_image = np.roll(image_to_register, (self.t0, self.t1), axis=(0, 1))  # shift image
        registered_image = skimage.transform.rotate(registered_image, self.main_channel_angle, cval=0)  # rotate image
        return registered_image
