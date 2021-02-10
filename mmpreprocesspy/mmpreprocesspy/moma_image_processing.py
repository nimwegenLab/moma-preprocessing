import os
import csv
import operator
import skimage
from mmpreprocesspy import preprocessing
from PIL import Image
import numpy as np
from mmpreprocesspy.GrowthlaneRoi import GrowthlaneExitLocation
from mmpreprocesspy.preprocessing import get_growthlane_rois
from mmpreprocesspy.support import saturate_image
from skimage.feature import match_template
import cv2 as cv
import mmpreprocesspy.dev_auxiliary_functions as aux
from pystackreg import StackReg
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from tifffile import TiffWriter
import tifffile as tff


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
        self._image_for_registration = None
        self._sr = StackReg(StackReg.TRANSLATION)
        self.growthlane_length_threshold = 0
        self.roi_boundary_offset_at_mother_cell = 0
        self.gl_detection_template = None
        self.gl_regions = None
        self._gl_region_indicator_images = []
        self._intensity_profiles = [[], []]  # we assume that at max. we will have two regions: one to each side of the main channel
        self.image_save_fequency = 2

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

    def set_normalization_ranges_and_save_log_data(self, growthlane_rois, phc_image, frame_nr, position_nr, output_path):
        """
        This method registers the input `image` and rotates it, so that the
        GL regions are correctly positioned on the resulting image.

        :param unmodified_image:
        :return:
        """

        offset = 100  # offset to both sides of the actual region range; this reduces the range where we will calculate the averaged profile by 2*offset
        box_pts = 11  # number of point to average over

        original_image = phc_image

        self.determine_image_shift(phc_image)
        image_registered = self._translate_image(phc_image)
        image_registered = self._rotate_image(image_registered)

        intensity_profiles_unprocessed = []
        intensity_profiles = []
        for region in self.gl_regions:
            intensity_profile_region = image_registered[:, region.start+offset:region.end-offset]
            intensity_profile_unprocessed = np.mean(intensity_profile_region, axis=1)
            intensity_profiles_unprocessed.append(intensity_profile_unprocessed)
            intensity_profile_processed = self.smooth(intensity_profile_unprocessed, box_pts=box_pts)
            intensity_profiles.append(intensity_profile_processed)

        normalization_ranges = []
        for ind, profile in enumerate(intensity_profiles):
            min_val, max_val = self.get_pdms_and_empty_channel_intensities(profile)
            normalization_ranges.append((min_val, max_val))

        self.set_gl_roi_normalization_ranges(growthlane_rois, normalization_ranges)

        self.save_normalization_range_to_csv_log(normalization_ranges, position_nr, frame_nr, output_path)

        self.save_image_with_region_indicators(image_registered,
                                               offset,
                                               position_nr,
                                               frame_nr,
                                               output_path)

        self.plot_and_save_intensity_profiles_with_peaks(intensity_profiles,
                                                         intensity_profiles_unprocessed,
                                                         normalization_ranges,
                                                         position_nr,
                                                         frame_nr,
                                                         output_path)

    def set_gl_roi_normalization_ranges(self, gl_rois, normalization_ranges):
        for roi in gl_rois:
            roi.normalization_range = normalization_ranges[roi.parent_gl_region_id]

    def save_normalization_range_to_csv_log(self,
                                            normalization_ranges,
                                            position_nr,
                                            frame_nr,
                                            output_path):
        path = os.path.join(output_path, f'intensity_normalization_ranges_pos_{position_nr}.csv')

        with open(path, mode='a') as normalization_ranges_file:
            csv_writer = csv.writer(normalization_ranges_file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            if frame_nr == 0:  # write header, if we are at the first frame
                header = []
                header.append('frame')
                for ind, range in enumerate(normalization_ranges):
                    header.append(f'region_{ind}_min')
                    header.append(f'region_{ind}_max')
                csv_writer.writerow(header)
            row = []
            row.append(frame_nr)
            for ind, range in enumerate(normalization_ranges):
                row.append(np.round(range[0], decimals=2))
                row.append(np.round(range[1], decimals=2))
            csv_writer.writerow(row)

    def convert_figure_to_numpy_array(self, canvas):
        canvas.draw()
        data = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
        image = data.reshape(canvas.get_width_height()[::-1] + (3,))
        return image

    def save_image_with_region_indicators(self,
                                          image_registered,
                                          offset,
                                          position_nr,
                                          frame_nr,
                                          output_path):
            image_registered_orig = image_registered
            image_registered = saturate_image(image_registered, 0.1, 0.3)
            plt.imshow(image_registered, cmap='gray')
            for region in self.gl_regions:
                plt.axvline(region.start+offset, color='r')
                plt.axvline(region.end-offset, color='g')

            plt.title(f'region indicators: pos {position_nr}, frame {frame_nr}')
            # plt.show()

            figure_canvas_handle = plt.gcf().canvas
            result = self.convert_figure_to_numpy_array(figure_canvas_handle)
            self._gl_region_indicator_images.append(result)
            plt.close(plt.gcf())

            if frame_nr % self.image_save_fequency == 0:
                image_to_save = np.array(self._gl_region_indicator_images)
                tff.imwrite(os.path.join(output_path, f'region_indiator_images__pos_{position_nr}.tif'), image_to_save)

    def plot_and_save_intensity_profiles_with_peaks(self,
                                                    intensity_profiles,
                                                    intensity_profiles_unprocessed,
                                                    normalization_ranges,
                                                    position_nr,
                                                    frame_nr,
                                                    output_path):

        for region_ind, intensity_profile in enumerate(intensity_profiles):

            intensity_profile_unprocessed = intensity_profiles_unprocessed[region_ind]
            normalization_range = normalization_ranges[region_ind]
            mean_peak_inds = find_peaks(intensity_profile, distance=25)[0]
            mean_peak_vals = intensity_profile[mean_peak_inds]

            plt.plot(intensity_profile_unprocessed, 'gray', label='profile')
            plt.plot(intensity_profile, label='smoothed')
            plt.scatter(mean_peak_inds, mean_peak_vals)

            plt.axhline(normalization_range[0], linestyle='--', color='k')
            plt.axhline(normalization_range[1], linestyle='--', color='k', label='norm range')

            plt.scatter(np.argwhere(intensity_profile == normalization_range[0]), normalization_range[0], color='k')
            plt.scatter(np.argwhere(intensity_profile == normalization_range[1]), normalization_range[1], color='k',
                        label='norm values')

            plt.ylim([0, 1.1 * np.max(intensity_profile_unprocessed)])
            plt.ylabel('intensity [a.u.]')
            plt.xlabel('vertical position [px]')
            # plt.legend(loc='center right')
            plt.title(f'intensity profile: pos {position_nr}, frame {frame_nr}, region {region_ind}')
            # plt.show()

            figure_canvas_handle = plt.gcf().canvas
            result = self.convert_figure_to_numpy_array(figure_canvas_handle)
            self._intensity_profiles[region_ind].append(result)
            plt.close(plt.gcf())

            if frame_nr % self.image_save_fequency == 0:
                image_to_save = np.array(self._intensity_profiles[region_ind])
                path = os.path.join(output_path, f'intensity_profile__pos_{position_nr}__region_{region_ind}.tif')
                tff.imwrite(path, image_to_save)

    def get_pdms_and_empty_channel_intensities(self, intensity_profile):
        mean_peak_inds = find_peaks(intensity_profile, distance=25)[0]
        mean_peak_vals = intensity_profile[mean_peak_inds]
        return mean_peak_vals.min(), mean_peak_vals.max()

    def smooth(self, y, box_pts):
        box = np.ones(box_pts)/box_pts
        y_smooth = np.convolve(y, box, mode='same')
        return y_smooth
