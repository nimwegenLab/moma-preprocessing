import numpy as np
from scipy.ndimage.filters import gaussian_filter


class PreprocessedImageProvider(object):
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

    def initialize(self):
        self.calculate_averaged_flatfields()
        self.smoothen_flatfields()
        self.normalize_flatfields()

    def calculate_averaged_flatfields(self):
        nr_of_flatfield_positions = self.flatfield_dataset.get_position_names()[0].__len__()
        nr_of_colors = self.flatfield_dataset.get_channels().__len__()
        height = self.flatfield_dataset.height
        width = self.flatfield_dataset.width

        self.flatfields = np.zeros((height, width, nr_of_colors))
        for color_ind in range(0, nr_of_colors):
            for pos_ind in range(0, nr_of_flatfield_positions):
                self.flatfields[:, :, color_ind] += self.flatfield_dataset.get_image_fast(channel=color_ind, frame=0,
                                                                                          position=pos_ind)
        self.flatfields[:, :, color_ind] /= nr_of_flatfield_positions
        self.flatfields -= self.dark_noise

    def smoothen_flatfields(self):
        # self.flatfields = gaussian_filter(self.flatfields, (self.gaussian_sigma,self.gaussian_sigma,0))
        for color_ind in range(0, self.flatfields.shape[2]):
            self.flatfields[:, :, color_ind] = gaussian_filter(self.flatfields[:, :, color_ind], self.gaussian_sigma)

    def normalize_flatfields(self):
        for color_ind in range(0, self.flatfields.shape[2]):
            self.flatfields[:, :, color_ind] /= self.flatfields[:, :, color_ind].max()

    def process_image_stack(self, frame_image_stack):
        height = frame_image_stack.shape[0]
        width = frame_image_stack.shape[1]

        tmp = frame_image_stack[:, :, 1:]
        tmp = np.divide(tmp, self.flatfields[0:height, 0:width, :])
        tmp -= self.dark_noise
        frame_image_stack[:, :, 1:] = tmp
        return frame_image_stack

    def assert_flatfield_color_channel_numbers(self):
        nr_of_flatfield_channels = self.flatfield_dataset.get_channels().__len__()
        nr_of_data_channels = self.mm_dataset.get_channels().__len__()
        if nr_of_flatfield_channels != nr_of_data_channels - 1:
            raise AssertionError(
                "Number of flat-field color channels must be N-1, where N is the number of channels in the dataset.")
