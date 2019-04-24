from mmpreprocesspy import preprocessing
from mmpreprocesspy.GrowthlaneRoi import GrowthlaneRoi
from PIL import Image
import numpy as np


class MomaImageProcessor(object):
    """ MomaImageProcessor encapsulates the processing of a Mothermachine image. """

    def __init__(self):
        self.image = None
        self.rotated_image = None
        self.main_channel_angle = None
        self.mincol = None
        self.maxcol = None
        self.channel_centers = None

    def load_numpy_image_array(self, image):
        self.image = image

    def read_image(self, image_path):
        """Reads tiff-image and returns it as a numpy-array."""
        image_base = Image.open(image_path)
        self.image = np.array(image_base)

    def process_image(self):
        self.rotated_image, self.main_channel_angle, self.mincol, self.maxcol, channel_centers = preprocessing.split_channels_init(
            self.image)

    def get_growthlane_regions(self, channel_centers, mincol, maxcol):
        rois = []
        for center in channel_centers:
            tmp = GrowthlaneRoi()
            tmp.roi = self.get_roi(center, mincol, maxcol)
            rois.append(tmp)
        return rois

    def get_roi(self, center, mincol, maxcol):
        channel_width = 100  # TODO-MM-2019-04-23: This will need to be determined dynamically or made configurable.
        half_width = channel_width / 2

        x = center - half_width
        y = mincol
        width = maxcol - mincol
        height = channel_width
        return (x, y), (width, height)
