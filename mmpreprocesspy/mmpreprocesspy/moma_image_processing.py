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
from scipy.signal import find_peaks

def is_debugging():
    try:
        import pydevd
        return True
    except ImportError:
        return False


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

        offset = 100

        original_image = image

        self.determine_image_shift(image)
        image_registered = self._translate_image(image)
        image_registered = self._rotate_image(image_registered)

        if is_debugging():
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            ax[0].imshow(original_image, cmap='gray')
            ax[1].imshow(image_registered, cmap='gray')
            for region in self.gl_regions:
                ax[1].axvline(region.start+offset, color='r')
                ax[1].axvline(region.end-offset, color='g')
                pass
            plt.show()

        intensity_profiles = []
        for region in self.gl_regions:
            intensity_profile_region = image_registered[:, region.start+offset:region.end-offset]
            intensity_profiles.append(np.mean(intensity_profile_region, axis=1))

        if is_debugging():
            for ind, profile in enumerate(intensity_profiles):
                plt.plot(profile, label=f'region {ind}')
            plt.legend()
            plt.show()

        min_vals, max_vals = [], []
        for ind, profile in enumerate(intensity_profiles):
            min_val, max_val = self.get_pdms_and_empty_channel_intensities(profile)
            min_vals.append(min_val)
            max_vals.append(max_val)

        if is_debugging():
            for ind, profile in enumerate(intensity_profiles):
                plt.plot(profile, label=f'region {ind}')
                plt.scatter(np.argwhere(profile == min_vals[ind]), min_vals[ind], color='r')
                plt.scatter(np.argwhere(profile == max_vals[ind]), max_vals[ind], color='g')
            plt.legend()
            plt.show()

        min_reference_value = np.min(min_vals)
        max_reference_value = np.max(max_vals)
        image_normalized = self._normalize_image_with_min_and_max_values(image, min_reference_value, max_reference_value)
        normalization_range = (min_reference_value, max_reference_value)
        return image_normalized, normalization_range

    def get_pdms_and_empty_channel_intensities(self, intensity_profile):
        # df_means_smoothed = pd.DataFrame(df_means.apply(lambda x: smooth(x, box_pts)))
        # intensity_profile = self.smooth(x, box_pts)
        mean_peak_inds = find_peaks(intensity_profile, distance=25)[0]
        mean_peak_vals = intensity_profile[mean_peak_inds]

        if is_debugging():
            plt.plot(intensity_profile)
            plt.scatter(mean_peak_inds, mean_peak_vals)
            plt.show()

        min = mean_peak_vals.min()
        max = mean_peak_vals.max()
        range = (max - min)
        lim1 = min + range * 1 / 4
        lim2 = min + range * 3 / 4

        pdms_peak_vals = mean_peak_vals[mean_peak_vals < lim1]
        empty_peak_vals = mean_peak_vals[mean_peak_vals > lim2]

        pdms_peak_min_value = pdms_peak_vals.max()
        empty_peak_max_value = empty_peak_vals.max()
        return pdms_peak_min_value, empty_peak_max_value

    def _normalize_image_with_min_and_max_values(self, image, pdms_intensity, empty_intensity):
        return (image - pdms_intensity) / (empty_intensity - pdms_intensity)

    def smooth(self, y, box_pts):
        box = np.ones(box_pts)/box_pts
        y_smooth = np.convolve(y, box, mode='same')
        return y_smooth
