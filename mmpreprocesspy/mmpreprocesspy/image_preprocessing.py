import os
from pathlib import Path

import numpy as np
import tifffile as tff
from scipy.ndimage.filters import gaussian_filter


class ImagePreprocessor(object):
    def __init__(self, mm_dataset, flatfield_dataset, dark_noise=None, gaussian_sigma=None):
        self.mm_dataset = mm_dataset
        self.flatfield_dataset = flatfield_dataset
        self.dark_noise = dark_noise
        self.gaussian_sigma = gaussian_sigma

        if not self.dark_noise:
            self.dark_noise = 100  # set default dark-noise, if not specified
        if not self.gaussian_sigma:
            self.gaussian_sigma = 10  # set default gaussian value, if not specified

        self.assert_flatfield_color_channel_numbers()

    def save_flatfields(self, folder_to_save):
        if not os.path.isdir(folder_to_save):
            path = Path(folder_to_save)
            path.mkdir(parents=True, exist_ok=True)
        out_image = np.zeros((self.flatfields.shape[2], self.flatfields.shape[0], self.flatfields.shape[1]))
        for color_ind in range(self.flatfields.shape[2]):
            out_image[color_ind, ...] = self.flatfields[:, :, color_ind]
        tff.imsave(folder_to_save + '/flatfields.tiff', np.float32(out_image))

    def calculate_flatfields(self, roi_shape):
        self.calculate_averaged_flatfields(roi_shape)
        self.substract_dark_noise()
        self.smoothen_flatfields()
        self.normalize_flatfields()

    def calculate_averaged_flatfields(self, roi_shape):
        nr_of_flatfield_positions = len(self.flatfield_dataset.get_position_names())
        nr_of_colors = len(flatfield_dataset.get_channels())
        height = roi_shape[0]
        width = roi_shape[1]

        self.flatfields = np.zeros((height, width, nr_of_colors))
        for color_ind in range(0, nr_of_colors):
            for pos_ind in range(0, nr_of_flatfield_positions):
                next_image = self.flatfield_dataset.get_image_fast(channel=color_ind, frame=0, position=pos_ind)
                self.flatfields[:, :, color_ind] += next_image[:height, :width]
            self.flatfields[:, :, color_ind] /= nr_of_flatfield_positions
        pass

    def substract_dark_noise(self):
        for color_ind in range(0, self.flatfields.shape[2]):
            self.flatfields[:, :, color_ind] -= self.dark_noise
        # if np.any(self.flatfields < 0):
        #     raise ValueError("self.flatfields < 0: flatfield contains negative values")

    def smoothen_flatfields(self):
        for color_ind in range(0, self.flatfields.shape[2]):
            self.flatfields[:, :, color_ind] = gaussian_filter(self.flatfields[:, :, color_ind], self.gaussian_sigma)

    def normalize_flatfields(self):
        for color_ind in range(0, self.flatfields.shape[2]):
            self.flatfields[:, :, color_ind] /= self.flatfields[:, :, color_ind].max()

    def process_image_stack(self, channels_to_correct):
        colors_to_correct = channels_to_correct.copy()

        for color_ind in range(0, colors_to_correct.shape[2]):
            colors_to_correct[..., color_ind] -= self.dark_noise
            # colors_to_correct[color_ind] = np.divide(colors_to_correct[color_ind], self.flatfields[color_ind])
            colors_to_correct[..., color_ind] /= self.flatfields[..., color_ind]
        # colors_to_correct[colors_to_correct < 0] = 0
        return colors_to_correct

    def assert_flatfield_color_channel_numbers(self):
        nr_of_flatfield_channels = self.flatfield_dataset.get_channels().__len__()
        nr_of_data_channels = self.mm_dataset.get_channels().__len__()
        if nr_of_flatfield_channels != nr_of_data_channels - 1:
            raise AssertionError(
                "Number of flat-field color channels must be N-1, where N is the number of channels in the dataset.")
