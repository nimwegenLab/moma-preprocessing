import operator
import skimage
from scipy.signal import find_peaks
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
        self.channel_centers = None
        self.growthlane_rois = []
        self.region_list = None
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
        self.rotated_image, self.main_channel_angle, self.channel_centers, self.growthlane_rois, self.region_list = preprocessing.process_image(
            self.image, self.growthlane_length_threshold, main_channel_angle=self.main_channel_angle)
        self.reset_growthlane_roi_ids()
        self.get_image_registration_template()


    def get_normalization_ranges(self, image, box_pts=11):
        norm_ranges = []
        for region in self.region_list:
            norm_range_min_value, norm_range_max_value = self.get_normalization_range_in_region(image, region.start,
                                                                                                region.end,
                                                                                                box_pts=box_pts)
            norm_ranges.append([norm_range_min_value, norm_range_max_value])
            # norm_range_min_values.append(norm_range_min_value)
            # norm_range_max_values.append(norm_range_max_value)

        # norm_range_min_values = np.array(norm_range_min_values)
        # norm_range_max_values = np.array(norm_range_max_values)
        return np.array(norm_ranges)


    def get_normalization_range_in_region(self, aligned_image, region_start, region_end, box_pts=11):
        box_pts = 1
        gl_region = aligned_image[:, region_start:region_end]
        projected_intensity = np.mean(gl_region, axis=1)
        projected_intensity_smoothed = self.smooth(projected_intensity, box_pts)
        valid_region_offset = int(np.ceil(box_pts/2))  # we keep only the region, where the smoothing operation is well-defined
        projected_intensity_smoothed = projected_intensity_smoothed[valid_region_offset:-valid_region_offset]  # keep only valid region
        # mean_peak_vals = projected_intensity_smoothed

        peak_inds = find_peaks(projected_intensity_smoothed, distance=25)[0]
        peak_vals = projected_intensity_smoothed[peak_inds]

        min = peak_vals.min()
        max = peak_vals.max()
        range = (max - min)
        threshold_lower = min + range * 0.1
        threshold_upper = min + range * 0.8

        pdms_peak_vals = peak_vals[peak_vals < threshold_lower]
        pdms_peak_inds = peak_inds[peak_vals < threshold_lower]
        empty_peak_vals = peak_vals[peak_vals > threshold_upper]
        empty_peak_inds = peak_inds[peak_vals > threshold_upper]

        norm_range_min_value = np.max(pdms_peak_vals)
        norm_range_max_value = np.max(empty_peak_vals)

        # if is_debugging():
        #     import matplotlib.pyplot as plt
        #     plt.plot(projected_intensity_smoothed)
        #     plt.plot(projected_intensity)
        #     plt.scatter(peak_inds, peak_vals, color='k')
        #     plt.scatter(pdms_peak_inds, pdms_peak_vals, color='b')
        #     plt.scatter(empty_peak_inds, empty_peak_vals, color='g')
        #     plt.axhline(threshold_lower, linestyle='--', color='k')
        #     plt.axhline(threshold_upper, linestyle='--', color='k')
        #     plt.show()
        #
        # if is_debugging():
        #     import matplotlib.pyplot as plt
        #     plt.plot(projected_intensity, color='r')
        #     plt.plot(projected_intensity_smoothed, color='g')
        #     plt.show()
        #
        # if is_debugging():
        #     import matplotlib.pyplot as plt
        #     plt.plot(projected_intensity, color='r')
        #     plt.scatter(peak_inds, projected_intensity_smoothed[peak_inds])
        #     plt.show()
        #
        # if is_debugging():
        #     import matplotlib.pyplot as plt
        #     plt.plot(projected_intensity, color='r')
        #     plt.plot(projected_intensity_smoothed, color='g')
        #     plt.show()
        #
        # if is_debugging():
        #     import matplotlib.pyplot as plt
        #     plt.plot(projected_intensity_smoothed)
        #     plt.scatter(np.argwhere(projected_intensity_smoothed == norm_range_min_value), norm_range_min_value, color='r')
        #     plt.scatter(np.argwhere(projected_intensity_smoothed == norm_range_max_value), norm_range_max_value, color='g')
        #     plt.show()

        return norm_range_min_value, norm_range_max_value

    def smooth(self, y, box_pts):
        box = np.ones(box_pts) / box_pts
        y_smooth = np.convolve(y, box, mode='same')
        return y_smooth

    def translate_and_rotate_image(self, image):
        image_orig = image
        image = self._translate_image(image)
        image = self._rotate_image(image)
        return image

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

def is_debugging():
    try:
        import pydevd
        return True
    except ImportError:
        return False

