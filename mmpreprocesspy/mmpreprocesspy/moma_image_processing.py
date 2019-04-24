from mmpreprocesspy import preprocessing
from PIL import Image
import numpy as np
from mmpreprocesspy.preprocessing import get_growthlane_regions


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
