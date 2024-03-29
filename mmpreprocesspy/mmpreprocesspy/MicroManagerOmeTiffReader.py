import os
import re
import numpy as np
import tifffile as tff
import zarr


class MicroManagerOmeTiffReader(object):

    def __init__(self, path):
        if not os.path.exists(path):
            self.tiff = object()  # set this to an object, so that the call to `del self.tiff` in `self.__del__()` does not fail
            raise FileExistsError(f'path does not exist: {path}')
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
        if 'InitialPositionList' in metadata:
            self._position_names = [c['Label'] for c in
                                    metadata['InitialPositionList']]  # this is for TIFF format from MicroManager 1
        elif 'StagePositions' in metadata:
            if 'Label' in metadata['StagePositions'][0].keys():
                self._position_names = [c['Label'] for c in
                                        metadata['StagePositions']]  # this is for TIFF format from MicroManager 2
            elif 'label' in metadata['StagePositions'][0].keys():
                self._position_names = [c['label'] for c in
                                        metadata['StagePositions']]  # this is for TIFF format from MicroManager 2
            else:
                raise LookupError(
                    "TIFF metadata['StagePositions'] contains no key 'Label' or 'label'")
        else:
            raise LookupError(
                "TIFF metadata contains no entry for either 'InitialPositionList' or 'StagePositions'")
        self.number_of_positions = len(self._position_names)

    def __del__(self):
        del self.tiff

    def get_position_series_lut(self) -> dict():
        return dict((name, pos_series) for (name, pos_series) in zip(self._position_names, self._position_series))

    def generate_position_index_lut(self) -> dict():
        """
        Returns a look-up table (LUT) that maps the position name as stored inside the
        MicroManager OME-TIFF to the corresponding index of the image stack in self._position_series
        and self._position_zarr.

        :return: LUT
        """

        return dict((name, pos_zarr) for (name, pos_zarr) in zip(self._position_names, self._position_zarr))

    def get_first_tiff_in_path(self, dir_path):
        tiffs = self.get_all_tiffs(dir_path)
        tiffs.sort()
        return os.path.join(dir_path, tiffs[0])

    def get_all_tiffs(self, dir_path):
        """Return list of all files composing an acquisition"""
        image_files = [re.search('.*(MMStack).*', f).group(0) for f in os.listdir(dir_path) if
                       re.search('.*(MMStack).*ome.tif', f)]
        return image_files

    def get_image_stack(self, position_name: str, frame_index, z_slice):
        """
        Get image stack containing the different channels for the specified position and frame.
        For frame indexes >0, we check if the image returned by `_get_image_stack_with_adapted_dimensions`
        is the same as the previous image. If so, we return an all-NaN image for the corresponding channel.
        We do this, because we assume that in this case no image was recorded for the corresponding time-step
        in the corresponding channel. This is because the tifffile package (that we use as backend)
        returns the closest existing previous frame for frames that were not recorded.

        :param position_name: name of the position as set in MicroManager
        :param frame_index:
        :return:
        """
        if frame_index == 0:
            return self._get_image_stack_with_adapted_dimensions(position_name, frame_index, z_slice=z_slice)
        elif frame_index > 0:
            image_stack_current = self._get_image_stack_with_adapted_dimensions(position_name, frame_index, z_slice=z_slice)
            image_stack_previous = self._get_image_stack_with_adapted_dimensions(position_name, frame_index - 1, z_slice=z_slice)

            for channel_index in range(image_stack_current.shape[2]):
                if np.all(image_stack_current[:,:,channel_index] == image_stack_previous[:,:,channel_index]):
                    image_stack_current[:, :, channel_index] = np.nan
            return image_stack_current
        else:
            raise ValueError('frame_index cannot be negative')

    def get_position_series(self, position_name: str):
        position_series_lut = self.get_position_series_lut()
        return position_series_lut[position_name]

    def get_position_zarr(self, position_name: str):
        zarr_lut = self.generate_position_index_lut()
        return zarr_lut[position_name]

    def _get_image_stack_with_adapted_dimensions(self, position_name: str, frame_index, z_slice=0):
        """
        Get image stack containing the different channels for the specified position and frame.

        :param position_name: position name as set in MicroManger
        :param frame_index:
        :return:
        """

        position_series = self.get_position_series(position_name)
        position_zarr = self.get_position_zarr(position_name)

        if position_series.axes == 'IYX':
            image_stack = position_zarr[frame_index, :].astype(dtype=float).copy()
            image_stack = np.expand_dims(image_stack, 2)  # return image only has 2 dimensions (no color); append color axis as it is expected by the preprocessing algorithm
            return image_stack
        elif position_series.axes == 'YX':
            image_stack = position_zarr[:, :].astype(dtype=float).copy()  # position contains only one frame, so ignore `frame_index`
            image_stack = np.expand_dims(image_stack, 2)  # return image only has 2 dimensions (no color); append color axis as it is expected by the preprocessing algorithm
            return image_stack
            pass
        elif position_series.axes == 'CYX':
            image_stack = position_zarr[:, :, :].astype(dtype=float).copy()
            return np.moveaxis(image_stack, 0, -1)
        elif position_series.axes == 'TZCYX':
            image_stack = position_zarr[frame_index, z_slice, ...].astype(dtype=float).copy()
            return np.moveaxis(image_stack, 0, -1)
        else:
            image_stack = position_zarr[frame_index, :].astype(dtype=float).copy()
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

    def get_number_of_frames(self):
        """
        Return the number of frames in the tiff,

        :return: number of frames
        """
        lut = self.generate_position_index_lut()
        name_of_pos_1: str = list(lut.keys())[0]
        position_zarr = self.get_position_zarr(position_name=name_of_pos_1)
        return position_zarr.shape[0]  # since all our our positions have the same number of frames, just return the number of frames for the first position.
