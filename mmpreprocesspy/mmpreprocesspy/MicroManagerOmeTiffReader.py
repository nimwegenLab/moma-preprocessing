import os
import re
import numpy as np
import tifffile as tff
import zarr


class MicroManagerOmeTiffReader(object):

    def __init__(self, path):
        if not os.path.exists(path):
            raise FileExistsError('file does not exist')
        if os.path.isdir(path):
            path_orig = path
            path = self.get_first_tiff_in_path(path)

        self.tiff = tff.TiffFile(path)

        self._position_zarr = []
        self._position_series = []
        for position in self.tiff.series:
            self._position_series.append(position)
            self._position_zarr.append(zarr.open(position.aszarr(), mode='r'))

        metadata = self.tiff.micromanager_metadata['Summary']
        self.height = metadata['Height']
        self.width = metadata['Width']
        self.channels = [c.strip(' ') for c in metadata['ChNames']]
        self.number_of_channels = len(self.channels)
        self.number_of_frames = metadata['Frames']
        if 'InitialPositionList' in metadata:
            self._position_names = [c['Label'] for c in
                                    metadata['InitialPositionList']]  # this is for TIFF format from MicroManager 1
        elif 'StagePositions' in metadata:
            self._position_names = [c['Label'] for c in
                                    metadata['StagePositions']]  # this is for TIFF format from MicroManager 2
        else:
            raise LookupError(
                "TIFF metadata contains no entry for either 'InitialPositionList' or 'StagePositions'")
        self.number_of_positions = len(self._position_names)

    def __del__(self):
        del self.tiff

    def get_first_tiff_in_path(self, dir_path):
        tiffs = self.get_all_tiffs(dir_path)
        tiffs.sort()
        return os.path.join(dir_path, tiffs[0])

    def get_all_tiffs(self, dir_path):
        """Return list of all files composing an acquisition"""
        image_files = [re.search('.*(MMStack).*', f).group(0) for f in os.listdir(dir_path) if
                       re.search('.*(MMStack).*ome.tif', f)]
        return image_files

    def get_image(self, position_index, frame_index, channel_index):
        """
        Get single image for the specified position, frame and channel.

        :param position_index:
        :param frame_index:
        :param channel_index:
        :return:
        """
        axes = self._position_series[position_index].axes
        if axes == 'YX':
            return self._position_zarr[position_index][:, :].astype(dtype=np.float).copy()
        if axes == 'IYX':
            return self._position_zarr[position_index][frame_index, :, :].astype(dtype=np.float).copy()
        return self._position_zarr[position_index][frame_index, channel_index].astype(dtype=np.float).copy()

    def get_image_fast(self, frame=0, channel=0, plane=None, position=0):
        """
        This is to stay compatible with MMData.py

        :param frame:
        :param channel:
        :param plane:
        :param position:
        :return:
        """
        return self.get_image(position, frame, channel)

    def get_image_stack(self, position_index, frame_index):
        """
        Get image stack containing the different channels for the specified position and frame.

        :param position_index:
        :param frame_index:
        :return:
        """
        if self._position_series[position_index].axes == 'IYX':
            image_stack = self._position_zarr[position_index][frame_index, :].astype(dtype=np.float).copy()
            image_stack = np.expand_dims(image_stack, 2)  # return image only has 2 dimensions (no color); append color axis as it is expected by the preprocessing algorithm
            return image_stack
        else:
            image_stack = self._position_zarr[position_index][frame_index, :].astype(dtype=np.float).copy()
            return np.moveaxis(image_stack, 0, -1)

    def get_channels(self):
        """
        Returns a list with the names of the channels

        :return: list with channel names
        """

        return self.channels

    def get_image_height(self):
        """
        Return height of an individual image.

        :return: image height
        """
        return self.height

    def get_image_width(self):
        """
        Return width of an individual image.

        :return: image width
        """
        return self.width

    def get_position_names(self):
        """
        Return list with names of the positions.

        :return: List[str] with names of the positions.
        """

        return self._position_names


    def get_first_tiff(self):
        """
        Return name of the first tiff-file.
        
        :return: name of first tiff-file in series.
        """

        return self.tiff.filename

