import operator
import skimage
from mmpreprocesspy import preprocessing
from PIL import Image
import numpy as np
from mmpreprocesspy.GrowthlaneRoi import GrowthlaneExitLocation
from mmpreprocesspy.preprocessing import get_growthlane_rois
from skimage.feature import match_template
import cv2 as cv
import mmpreprocesspy.dev_auxiliary_functions as aux
# import matplotlib.pyplot as plt
from pystackreg import StackReg
import matplotlib.pyplot as plt

class MomaImageProcessor(object):
    """ MomaImageProcessor encapsulates the processing of a Mothermachine image. """

    def __init__(self):
        self.image = None
        self.rotated_image = None
        self.main_channel_angle = None
        self.growthlane_rois = []
        self.vertical_shift = None
        self.horizontal_shift = None
        self.gl_orientation_search_area = 80  # TODO: this is a MM specific parameter; should be made configurable
        self._image_for_registration = None
        self._sr = StackReg(StackReg.TRANSLATION)
        self.growthlane_length_threshold = 0
        self.roi_boundary_offset_at_mother_cell = 0
        self.gl_detection_template = None
        self.gl_regions = None

    def load_numpy_image_array(self, image):
        self.image = image

    def read_image(self, image_path):
        """Reads tiff-image and returns it as a numpy-array."""
        image_base = Image.open(image_path)
        self.image = np.array(image_base, dtype=np.uint16)

    def process_image(self):
        self.rotated_image, self.main_channel_angle = preprocessing.get_rotated_image(
            self.image, main_channel_angle=self.main_channel_angle)

        if self.gl_detection_template:
            self.growthlane_rois, self.gl_regions = preprocessing.get_gl_rois_using_template(self.rotated_image,
                                                                                             self.gl_detection_template,
                                                                                             roi_boundary_offset_at_mother_cell=self.roi_boundary_offset_at_mother_cell)
        else:
            self.growthlane_rois, self.gl_regions = preprocessing.get_gl_regions(self.rotated_image,
                                                                                 growthlane_length_threshold=self.growthlane_length_threshold,
                                                                                 roi_boundary_offset_at_mother_cell=self.roi_boundary_offset_at_mother_cell)

        self.growthlane_rois = preprocessing.rotate_rois(self.image, self.growthlane_rois, self.main_channel_angle)
        self.growthlane_rois = preprocessing.remove_rois_not_fully_in_image(self.image, self.growthlane_rois)

        self.reset_growthlane_roi_ids()
        self.get_image_registration_template()

    def reset_growthlane_roi_ids(self):
        self.growthlane_rois = sorted(self.growthlane_rois, key=operator.attrgetter("id"))  # make sure that the GLs are sorted by ID
        for new_id, roi in enumerate(self.growthlane_rois):
            roi.id = new_id

    def get_image_registration_template(self):
        self._image_for_registration = self.image.copy()

    def determine_image_shift(self, image):
        self._sr.register(self._image_for_registration, image)
        translation_matrix = self._sr.get_matrix()
        self.horizontal_shift = -translation_matrix[0][2]
        self.vertical_shift = -translation_matrix[1][2]

    def get_registered_image(self, image_to_register):
        # registered_image = self._transform_image(image_to_register)
        registered_image = self._rotate_image(image_to_register)
        registered_image = self._translate_image(registered_image)
        return registered_image

    def _translate_image(self, image):
        return cv.warpAffine(image, self.get_translation_matrix(), (image.shape[1], image.shape[0]))

    def _rotate_image(self, image):
        return cv.warpAffine(image, self.get_rotation_matrix(), (image.shape[1], image.shape[0]))

    def get_rotation_matrix(self):
        if self.main_channel_angle is None:
            raise ValueError("self.main_channel_angle must be set before calling self.get_transformation_matrix")

        rotation_center = (self.image.shape[1] / 2 - 0.5, self.image.shape[0] / 2 - 0.5)  # see center-definition here: https://scikit-image.org/docs/dev/api/skimage.transform.html#skimage.transform.rotate

        return preprocessing.get_rotation_matrix(self.main_channel_angle, rotation_center)

    def get_translation_matrix(self):
        if self.vertical_shift is None:
            raise ValueError("self.vertical_shift must be set before calling self.get_transformation_matrix")
        if self.horizontal_shift is None:
            raise ValueError("self.horizontal_shift must be set before calling self.get_transformation_matrix")

        return preprocessing.get_translation_matrix(self.horizontal_shift, self.vertical_shift)

    def normalize_image(self, image):
        """
        This method registers the input `image` and rotates it, so that the
        GL regions are correctly positioned on the resulting image.

        :param unmodified_image:
        :return:
        """

        imageProcessor.determine_image_shift(original_image)
        shifted_image = imageProcessor._translate_image(original_image)
        shifted_image = imageProcessor._rotate_image(shifted_image)

        pass