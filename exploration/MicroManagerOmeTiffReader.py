import os
import tifffile as tff


class MicroManagerOmeTiffReader(object):

    def __init__(self, path):
        if not os.path.exists(path):
            raise FileExistsError('file does not exist')

        self.tiff = tff.TiffFile(path)

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

    def get_image(self, frame_index, channel_index, position_index):
        stack = self.get_channel_stack(frame_index, position_index)
        return stack[channel_index, ...]

    def get_channel_stack(self, frame_index, position_index):
        series = self.tiff.series[position_index].asarray()
        return series[frame_index, ...]
