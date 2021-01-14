import os
import re

import tifffile as tff
import numpy as np

class MicroManagerTiffReader(object):

    def __init__(self, path):
        self.image_path = path
        if os.path.isdir(path):
            self.dir_path = path
        elif os.path.isfile(path):
            self.dir_path = os.path.dirname(path)
        else:
            raise FileNotFoundError("cannot read specified path")
        self.image_list = self.get_all_tiffs()

        self.memory_map_files()
        self.tiff = tff.TiffFile(self.image_path)

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
        page_nr = self.calculate_page_nr(frame_index, channel_index, position_index)
        print(f"page_nr: {page_nr}")
        return self.get_copy_of_page(page=page_nr)

    def get_channel_stack(self, frame_index, position_index):
        pass

    def calculate_page_nr(self, frame_index, channel_index, position_index):
        page_nr = frame_index * self.number_of_channels * self.number_of_positions \
                  + position_index * self.number_of_channels \
                  + channel_index
        return page_nr

    def get_copy_of_page(self, page):
        file_mask = (self.page_range_index[:, 0] <= page) & (self.page_range_index[:, 1] >= page)
        tiffile_index = np.argwhere(file_mask)[0][0]
        # print(f"Loaded from file: {self.image_list[tiffile_index]}")
        current_tiff = self.memmap_tiffs[tiffile_index]
        page_nr_within_current_tiff = page - self.page_range_index[tiffile_index, 0]
        img = np.copy(current_tiff.pages[page_nr_within_current_tiff].asarray().astype(dtype=np.float))
        return img

    def get_all_tiffs(self):
        """Return list of all files composing an acquisition"""
        image_files = [re.search('.*(MMStack).*', f).group(0) for f in os.listdir(self.dir_path) if re.search('.*(MMStack).*ome.tif', f)]
        image_files.sort()
        return image_files

    def memory_map_files(self):
        self.memmap_tiffs = []
        self.page_range_index = np.zeros((len(self.image_list), 2), dtype=np.int)

        page_counter = 0
        for index, file_name in enumerate(self.image_list):
            self.memmap_tiffs.append(tff.TiffFile(os.path.join(self.dir_path, file_name)))
            pages_in_file = len(self.memmap_tiffs[-1].pages)
            self.page_range_index[index, 0] = page_counter  # calculate index of first page in file
            self.page_range_index[index, 1] = page_counter + pages_in_file - 1  # calculate index of last page in file
            page_counter = page_counter + pages_in_file

    def __del__(self):
        del self.memmap_tiffs
