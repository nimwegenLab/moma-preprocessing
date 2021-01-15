import os
import numpy as np
import tifffile as tff
import zarr


class MicroManagerOmeTiffReader(object):

    def __init__(self, path):
        if not os.path.exists(path):
            raise FileExistsError('file does not exist')

        self.tiff = tff.TiffFile(path)

        self._position_zarr = []
        for position in self.tiff.series:
             self._position_zarr.append(zarr.open(position.aszarr(), mode='r'))

        metadata = self.tiff.micromanager_metadata['Summary']
        self.height = metadata['Height']
        self.width = metadata['Width']
        self.channels = [c.strip(' ') for c in metadata['ChNames']]
        self.number_of_channels = len(self.channels)
        self.number_of_frames = metadata['Frames']
        if 'InitialPositionList' in metadata:
            self.positions = [c['Label'] for c in
                         metadata['InitialPositionList']]  # this is for TIFF format from MicroManager 1
        elif 'StagePositions' in metadata:
            self.positions = [c['Label'] for c in
                         metadata['StagePositions']]  # this is for TIFF format from MicroManager 2
        else:
            raise LookupError(
                "TIFF metadata contains no entry for either 'InitialPositionList' or 'StagePositions'")
        self.number_of_positions = len(self.positions)

    def __del__(self):
        del self.tiff

    def get_image(self, position_index, frame_index, channel_index):
        """
        Get single image for the specified position, frame and channel.

        :param position_index:
        :param frame_index:
        :param channel_index:
        :return:
        """
        return self._position_zarr[position_index][frame_index, channel_index].astype(dtype=np.float).copy()

    def get_channel_stack(self, position_index, frame_index):
        """
        Get image stack containing the different channels for the specified position and frame.

        :param position_index:
        :param frame_index:
        :return:
        """
        return self._position_zarr[position_index][frame_index, :].astype(dtype=np.float).copy()
