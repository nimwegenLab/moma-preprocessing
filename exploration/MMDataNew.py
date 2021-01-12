import tifffile as tff
import numpy as np
import matplotlib.pyplot as plt

class MMDataNew():

    def __init__(self, image_path):
        self.image_path = image_path
        with tff.TiffFile(self.image_path) as tiff:
            metadata = tiff.micromanager_metadata['Summary']
            self.height = metadata['Height']
            self.width = metadata['Width']
            self.channels = [c.strip(' ') for c in metadata['ChNames']]
            self.number_of_channels = len(self.channels)
            self.number_of_frames = metadata['Frames']
            if 'InitialPositionList' in metadata:
                self.positions = [c['Label'] for c in
                             metadata['InitialPositionList']]  # this is for OME-TIFF format from MicroManager 1
            elif 'StagePositions' in metadata:
                self.positions = [c['Label'] for c in
                             metadata['StagePositions']]  # this is for OME-TIFF format from MicroManager 2
            else:
                raise LookupError(
                    "TIFF metadata contains no entry for either 'InitialPositionList' or 'StagePositions'")
            self.number_of_positions = len(self.positions)

    def get_image_fast(self, frame_index, channel_index, position_index):
        page_nr = self.calculate_page_nr(frame_index, channel_index, position_index)
        return self.get_copy_of_page(page=page_nr)

    def get_image_stack(self, frame_index, position_index):
        pass

    def calculate_page_nr(self, frame_index, channel_index, position_index):
        page_nr = frame_index * self.number_of_channels * self.number_of_positions \
                  + position_index * self.number_of_channels \
                  + channel_index
        return page_nr

    def get_copy_of_page(self, page):
        img_memmap = tff.memmap(self.image_path, page=page, mode='r')
        img = np.copy(img_memmap)
        del img_memmap
        return img
