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

class MomaImageProcessor(object):
    """ MomaImageProcessor encapsulates the processing of a Mothermachine image. """

    def __init__(self):
        self.image = None
        self.rotated_image = None
        self.main_channel_angle = None
        self.mincol = None
        self.maxcol = None
        self.channel_centers = None
        self.growthlane_rois = []
        self.template = None
        self.hor_mid = None
        self.hor_width = None
        self.mid_row = None
        self.vertical_shift = None
        self.horizontal_shift = None
        self.gl_orientation_search_area = 80  # TODO: this is a MM specific parameter; should be made configurable
        self._image_for_registration = None
        self._sr = StackReg(StackReg.TRANSLATION)
        self.growthlane_length_threshold = 0

    def load_numpy_image_array(self, image):
        self.image = image

    def read_image(self, image_path):
        """Reads tiff-image and returns it as a numpy-array."""
        image_base = Image.open(image_path)
        self.image = np.array(image_base, dtype=np.uint16)

    def process_image(self):
        self.rotated_image, self.main_channel_angle, self.mincol, self.maxcol, self.channel_centers, self.growthlane_rois = preprocessing.process_image(
            self.image, self.growthlane_length_threshold)
        self.rotate_rois()
        self.set_growthlane_orientation(self.gl_orientation_search_area)
        self.get_image_registration_template()

    def set_growthlane_orientation(self, search_area):
        """
        Finds the orientation of the growthlane within the ROI.
        :param growthlane_rois:
        :param search_area: the area before and after the ROI the will be looked to determine the direction; unit: [px]
        :return:
        """
        gl_indexes_outside_of_image  = []
        for index, gl_roi in enumerate(self.growthlane_rois):
            # gl_roi.roi.width += 2 * search_area  # extend ROI to include search area before and after
            if not gl_roi.roi.is_inside_image(self.image):  # if extended ROI is outside image keep index for removal below
                gl_indexes_outside_of_image.append(index)
                continue
            roi_image = gl_roi.roi.get_from_image(self.image)
            # gl_roi.roi.width -= 2 * search_area  # revert ROI extension
            gl_roi.exit_location = self.determine_location_of_growthlane_exit(roi_image, search_area)
        [self.growthlane_rois.pop(i) for i in reversed(gl_indexes_outside_of_image)]  # remove GL ROIs outside of the image

    def determine_location_of_growthlane_exit(self, growthlane_roi_image, search_area):
        """
        This function determines the location of the growthlane by comparing the value sum of values
        at the start of the *extend* GL to those at the end.
        :param growthlane_roi_image: the image of from the extended GL ROI.
        :param search_area: the value by which the GL was extended in *both* directions.
        :return:
        """
        sum_at_start = np.sum(growthlane_roi_image[:, 0:search_area].flatten(), 0)
        sum_at_end = np.sum(growthlane_roi_image[:, -search_area:].flatten(), 0)
        if sum_at_start > sum_at_end:
            return GrowthlaneExitLocation.AT_LEFT
        elif sum_at_end > sum_at_start:
            return GrowthlaneExitLocation.AT_RIGHT
        else:
            raise ValueError("Could not determine location of growthlane exit.")

    def rotate_rois(self):
        rotation_center = (np.int0(self.image.shape[1]/2), np.int0(self.image.shape[0]/2))
        for growthlane_roi in self.growthlane_rois:
            growthlane_roi.roi.rotate(rotation_center, -self.main_channel_angle)

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

    def store_gl_index_image(self, path):
        """ Draw the growthlane ROIs and indices onto the image and save it. """
        font = cv.FONT_HERSHEY_SIMPLEX
        rotated_rois = [x.roi for x in self.growthlane_rois]
        # show_image_with_rotated_rois(image, rotated_rois)
        # normalizedImg = None
        normalized_image = cv.normalize(self.image, None, 0, 255, cv.NORM_MINMAX)
        final_image = np.array(normalized_image, dtype=np.uint8)

        for gl_index, roi in enumerate(rotated_rois):
            roi.draw_to_image(final_image, False)
            cv.putText(final_image, str(gl_index + 1), (np.int0(roi.center[0]), np.int0(roi.center[1])), font, 1, (255, 255, 255), 2, cv.LINE_AA)

        cv.imwrite(path, final_image)
