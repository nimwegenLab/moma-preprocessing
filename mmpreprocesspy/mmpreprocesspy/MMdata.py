import os
import re
import struct

import numpy as np
import pandas as pd
import tifffile as tff
import xml.etree.ElementTree
import io

import warnings


class MMData:
    """Parsing of MicroManager metadata"""
    def __init__(self, folder = None, tiffs = None, mm_meta = None, height = None, 
                 width = None, mm_map = None, interval = None, channels = None,  num_planes = None):
        """Standard __init__ method.

        Parameters
        ----------
        folder : str
            Folder containing the acquisition.
        tiffs : : list of str
            List of tiffs composing the acquisition.
        mm_meta : str
            String of MicroManager metadata
        height : int
            Image height
        width : int
            Image widht
        mm_map : pandas dataframe
            Dataframe indicating the multi-dimensional position of each frame in the acquisition
        interval : float
            Time between frames of time-lapse
        channels : list of str
            List of channel names
        num_planes : list of int
            Number of planes per channel
        """
        
        self.folder = folder
        self.tiffs = self.get_all_tiffs()
        self.read_mm_metadata()
        # self.channels = self.get_channels()
        self.mm_meta = mm_meta
        self.mm_map = mm_map
        self.interval = interval

    def get_all_tiffs(self):
        """Return list of all files composing an acquisition"""
        image_files = [re.search('.*(MMStack).*', f).group(0) for f in os.listdir(self.folder) if re.search('.*(MMStack).*ome.tif', f)]
        return image_files
        
    def get_first_tiff(self):
        """Return name of first .tif block of the acquisition"""
        first_chunk = [f for f in self.tiffs if not re.search('.*(MMStack.ome).*',f)==None][0]
        return first_chunk
        
    def read_mm_metadata(self):
        """Return MicroManager metadata string contained in Tag 50839"""
        path_to_first_tiff = self.folder+'/'+self.get_first_tiff()
        with warnings.catch_warnings():
            with tff.TiffFile(path_to_first_tiff) as tiff:
                metadata = tiff.micromanager_metadata['Summary']
                self.height = metadata['Height']
                self.width = metadata['Width']
                self.channels = [c.strip(' ') for c in metadata['ChNames']]
                self.number_of_frames = metadata['Frames']
                if 'InitialPositionList' in metadata:
                    self.positions = [c['Label'] for c in metadata['InitialPositionList']]  # this is for OME-TIFF format from MicroManager 1
                elif 'StagePositions' in metadata:
                    self.positions = [c['Label'] for c in metadata['StagePositions']]  # this is for OME-TIFF format from MicroManager 2
                else:
                    raise LookupError("TIFF metadata contains no entry for either 'InitialPositionList' or 'StagePositions'")

    def get_image_height(self):
        return self.height

    def get_image_width(self):
        return self.width

    def get_map_indices(self):
        """Return the index map of the acquisition as a list of dict
        
        The returned dataframe contains 7 keys giving for each row its exact multi-dimensional position.
        Rows have no defined order. The keys are:
        channel: which channel in a multic-wavelength acquisition
        slice: which slice in a z-stack acqusition
        frame: which time-point in a time-lapse acquisition
        position: which position in a multi-position acquisition
        image_index: which image in a given tiff chunk
        offset: binaray offset in a given tiff chunk
        chunk: which tiff chunk in case the acquisition is >4GB and split in severla chunks
        """
        item_order = ('channel','slice','frame','position','offset')
        if self.mm_map is None:
            mm_map = {'channel':[], 'slice':[], 'frame':[], 'position':[], 'image_index':[], 'offset':[], 'chunk':[]}
            for f in self.tiffs:
                with open(self.folder+'/'+f, "rb") as binary_file:

                    # Seek position and read N bytes
                    binary_file.seek(0)  # Go to beginning
                    byte_order = struct.unpack('<ss',binary_file.read(2))
                    tiff_id = struct.unpack('<H',binary_file.read(2))
                    ifd_pos = struct.unpack('<L',binary_file.read(4))
                    map_tag = struct.unpack('<L',binary_file.read(4))
                    map_pos = struct.unpack('<L',binary_file.read(4))

                    binary_file.seek(map_pos[0])
                    binary_file.read(4)

                    nb_images = struct.unpack('<L',binary_file.read(4))

                    for i in range(nb_images[0]):
                        for x in item_order:
                            mm_map[x].append(struct.unpack('<L',binary_file.read(4))[0])

                        mm_map['chunk'].append(f)
                        mm_map['image_index'].append(i)
            self.mm_map = pd.DataFrame(mm_map)
        return self.mm_map
    
    #@profile
    def get_image(self, frame=0,channel=0,plane=0,position=0, compress = 1):
        """Return image at a given frame, channel, plane, position. One can skip every n'th pixel by setting 
        compress to n"""
        if self.mm_map is None:
            self.get_map_indices()
        selected = self.mm_map.loc[(self.mm_map['frame'] == frame) & (self.mm_map['position']==position) &
                                 (self.mm_map['channel']==channel) & (self.mm_map['slice']==plane),
                                 ['chunk','image_index','offset']].values
        
        
        with open(self.folder+'/'+selected[0,0], "rb") as binary_file:

            ifd_pos = selected[0,2]
            binary_file.seek(ifd_pos)  # Go to beginning
            nb_tags = struct.unpack('<H',binary_file.read(2))[0]
            #binary_file.seek(mm_map[0:1].offset.values[0])
    
            binary_file.seek(ifd_pos+2+(nb_tags)*12)
            next_ifd = struct.unpack('<L',binary_file.read(4))[0]
            im_bytes = struct.unpack('<'+str(self.height*self.width)+'H',binary_file.read(2*self.height*self.width))
                        
            width = self.width
            height = self.height
            
            im_bytes = np.fromiter(im_bytes, np.float)
            image = np.reshape(im_bytes,newshape=[self.height,self.width])

        return image

    def get_image_stack(self, frame=0, plane=0, position=0):
        """load all colors and return them as image stack"""
        nr_of_colors = len(self.channels)
        image_stack = np.zeros((self.height, self.width, nr_of_colors))
        for color in range(nr_of_colors):
            image_stack[:, :, color] = self.get_image_fast(channel=color, frame=frame, plane=plane, position=position)
        return image_stack

    def get_image_fast(self, frame=0,channel=0,plane=0,position=0):
        """Return image at a given frame, channel, plane, position. One can skip every n'th pixel by setting 
        compress to n"""
        if self.mm_map is None:
            self.get_map_indices()
        selected = self.mm_map.loc[(self.mm_map['frame'] == frame) &
                                   (self.mm_map['position'] == position) &
                                   (self.mm_map['channel'] == channel) &
                                   (self.mm_map['slice'] == plane),
                                   ['chunk', 'image_index', 'offset']].values

        with open(self.folder+'/'+selected[0,0], "rb") as binary_file:

            ifd_pos = selected[0,2]
            binary_file.seek(ifd_pos) 
            nb_tags = np.fromfile(binary_file, np.dtype('<H'),count=1)
            binary_file.seek(ifd_pos+2+(nb_tags[0])*12+0)
            next_ifd = np.fromfile(binary_file,np.dtype(dtype=np.int32),count=1)[0]
            im_bytes = np.fromfile(binary_file, np.dtype('<H'),count = self.height * self.width)
            image= np.reshape(im_bytes,newshape=[self.height,self.width])
            image = image.astype(float)

        return image
    
    def get_stack(self, frame=0,channel=0,position=0, compress = 1):
        """Return complete z-stack for given frame, channel, position"""
        mm_map = self.get_map_indices()
        planes = mm_map[(mm_map['frame']==0)&(mm_map['position']==0)&(mm_map['channel']==channel)]['slice'].values
        stack = np.empty((self.height,self.width,planes.shape[0]))
        for i in range(planes.shape[0]):
            stack[:,:,i] = self.get_image(frame=frame,channel=channel,plane=planes[i],position=position, compress = compress)
        return stack
    
    def get_stack_fast(self, frame=0,channel=0,position=0, compress = 1):
        """Return complete z-stack for given frame, channel, position"""
        mm_map = self.get_map_indices()
        planes = mm_map[(mm_map['frame']==0)&(mm_map['position']==0)&(mm_map['channel']==channel)]['slice'].values
        stack = np.empty((self.height,self.width,planes.shape[0]))
        for i in range(planes.shape[0]):
            stack[:,:,i] = self.get_image_fast(frame=frame,channel=channel,plane=planes[i],position=position, compress = compress)
        return stack
        
    def get_number_of_frames(self):
        """Return total number of time-points in the acquisition"""
        return self.number_of_frames
    
    def get_channels(self):
        """Return channels of the acquisition"""
        return self.channels

    def get_position_names(self):
        """Return list with position names"""
        return self.positions

def onselect(eclick, erelease):
    'eclick and erelease are matplotlib events at press and release'
    print(' startposition : (%f, %f)' % (eclick.xdata, eclick.ydata))
    print(' endposition   : (%f, %f)' % (erelease.xdata, erelease.ydata))
    print(' used button   : ', eclick.button)

def toggle_selector(event):
    print(' Key pressed.')
    if event.key in ['Q', 'q'] and toggle_selector.RS.active:
        print(' RectangleSelector deactivated.')
        toggle_selector.RS.set_active(False)
    if event.key in ['A', 'a'] and not toggle_selector.RS.active:
        print(' RectangleSelector activated.')
        toggle_selector.RS.set_active(True)
    
