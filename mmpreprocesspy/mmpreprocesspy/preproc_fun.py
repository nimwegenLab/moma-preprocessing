import os, re, time, sys, shutil
import copy
import cv2
import numpy as np
import matplotlib.pyplot as plt

import scipy.optimize
import scipy.ndimage.interpolation as ndii
import skimage.external.tifffile
import skimage.transform
import skimage.measure
import skimage.filters
from mmpreprocesspy.moma_image_processing import MomaImageProcessor
from skimage.feature import match_template
import mmpreprocesspy.dev_auxiliary_functions as aux

from mmpreprocesspy.MMdata import MMData

import mmpreprocesspy.preprocessing as pre

def preproc_fun(data_folder, folder_to_save,positions,maxframe):
    #create a micro-manager image object
    dataset = MMData(data_folder)

    #recover the basic experiment name
    base_name = dataset.get_first_tiff().split('.')[0]

    #define places to save the output
    folder_analyzed = os.path.normpath(folder_to_save)+'/'+base_name

    kymo_folder = folder_analyzed+'/Kymographs/'
    if not os.path.exists(kymo_folder):
        os.makedirs(kymo_folder)

    GL_folder = folder_analyzed+'/GrowthLanes/'
    if not os.path.exists(GL_folder):
        os.makedirs(GL_folder)

    #define basic parameters
    colors = dataset.get_channels()
    phase_channel_index = 0
    half_width = 50

    #define metadata for imagej
    metadata = {'channels':len(colors),'slices':1,'frames':maxframe,'hyperstack':True,'loop':False}

    # start measurement of processing time
    start1 = time.time()
    for indp in positions:  # MM: Currently proproc_fun.py in only run for a single position; so this loop is not needed

        current_saveto_folder = folder_to_save+'/'+base_name+'/GrowthLanes/'+base_name+'_Pos'+str(indp)

        if not os.path.exists(current_saveto_folder):
            os.makedirs(current_saveto_folder)

        #load first phase image
        image_base = dataset.get_image_fast(channel=phase_channel_index,frame=0,position=indp)

        # Process first image to find ROIs, etc.
        imageProcessor = MomaImageProcessor()
        imageProcessor.load_numpy_image_array(image_base)
        imageProcessor.process_image()
        mincol = imageProcessor.mincol
        maxcol = imageProcessor.maxcol
        channel_centers = imageProcessor.channel_centers

        #create empty kymographs to fill
        gl_length = imageProcessor.growthlane_rois[0].length
        nr_of_rois = len(imageProcessor.growthlane_rois)
        kymo = np.zeros((gl_length, maxframe, len(colors), nr_of_rois))
        # kymo = np.zeros((maxcol-mincol+60,maxframe, len(colors), len(channel_centers)))
        metadataK = {'channels':len(colors),'slices':1,'frames':len(channel_centers),'hyperstack':True,'loop':False}

        #calculate channel spacing 
        # fourier_ch = np.abs(np.fft.fft(np.sum(image_base[:,mincol:maxcol],axis=1)))
        # fourier_sort = np.sort(fourier_ch)
        # channel_spacing = image_base.shape[0]/np.where(fourier_ch==fourier_sort[-2])[0][0]

        frame_counter = np.zeros(len(channel_centers))  # stores per growthlane, the number of processed images
        #go through time-lapse and cut out channels
        for t in range(maxframe):

            if np.mod(t,10)==0:
                print('time: '+str(t))
            #load image and align
            image = dataset.get_image_fast(channel=phase_channel_index,frame=t,position=indp)

            '''#register images using FFT. Sometimes fails, probably because of border effect
            mindim = np.int(np.min([int(image.shape[0]/3),mincol-50])/2)
            t0, t1 = pre.fft_align(image_base[0:int(image.shape[0]/3),0:mincol-50],
                                       image[0:int(image.shape[0]/3),0:mincol-50],pixlim =mindim)
            image = np.roll(image,(t0,t1),axis=(0,1))
            t0b, t1b = pre.fft_align(image_base[0:int(image.shape[0]/3),mincol-50:mincol+200],
                                       image[0:int(image.shape[0]/3),mincol-50:mincol+200],pixlim=20)
            image = np.roll(image,(t0b,t1b),axis=(0,1))
            t0 = t0+t0b
            t1 = t1+t1b
            '''

            imageProcessor.determine_image_shift(image)
            growthlane_rois = copy.deepcopy(imageProcessor.growthlane_rois)

            for gl_roi in growthlane_rois:
                gl_roi.roi.translate((-imageProcessor.horizontal_shift, -imageProcessor.vertical_shift))

            # image_rot = imageProcessor.get_registered_image(image)

            #find channels in new image
            # channels = pre.find_channels(image_rot, mincol, maxcol)


            ###############################################
            # aux.show_image_with_rotated_rois(image, [g.roi for g in growthlane_rois])
            # cv2.waitKey()
            # cv2.destroyAllWindows()
            ###############################################

            #load all colors
            image_stack = np.zeros((image_base.shape[0],image_base.shape[1],len(colors)))
            for i in range(len(colors)):
                # image = dataset.get_image_fast(channel=i,frame=t,position=indp)
                # image_stack[:,:,i] = imageProcessor.get_registered_image(image)
                image_stack[:, :, i] = dataset.get_image_fast(channel=i,frame=t,position=indp)

            #go through all channels, check if there's a corresponding one in the new image. If yes go through all colors,
            #cut out channel, and append to tif stack. Completel also the kymograph for each color.
            for gl_index, gl_roi in enumerate(growthlane_rois):
                if gl_roi.roi.is_inside_image(image):
            # for c in gl_roi:
            #     if (np.min(c-channel_centers)<5)&(int(c)-half_width>0)&(int(c)+half_width+1<image.shape[0]):
            #         gl = np.argmin(np.abs(c-channel_centers)) # index of the growthlane
            #         gl = gl_roi.index
                    frame_counter[gl_index]+=1
                    gl_str='0'+str(gl_index) if gl_index<10 else str(gl_index)
                    pos_gl_name = dataset.get_first_tiff().split('.')[0]+'_Pos'+str(indp)+'_GL'+gl_str

                    if not os.path.exists(current_saveto_folder+'/'+pos_gl_name):
                        os.makedirs(current_saveto_folder+'/'+pos_gl_name)

                    filename = current_saveto_folder+'/'+pos_gl_name+'/'+pos_gl_name+'.tif'

                    # MM-2019-04-23: Here we get the GL ROI and store it to the GL stack.
                    for i in range(len(colors)):
                        # imtosave = image_stack[:,:,i][int(c)-half_width:int(c)+half_width+1,mincol-30:maxcol+30]
                        imtosave = gl_roi.get_oriented_roi_image(image_stack[:,:,i])

                        ################################################
                        # if i == 0 and gl_roi.index == 11:
                        #     aux.show_image(imtosave, "Image ROI "+str(gl_roi.index))
                        #     cv2.waitKey()
                        ################################################

                        # imtosave_flip = np.flipud(imtosave.T)
                        skimage.external.tifffile.imsave(filename,imtosave.astype(np.uint16),append = 'force',imagej = True, metadata = metadata)
                        kymo[:,t,i,gl_index] = np.mean(imtosave, axis = 1)

        #remove growth lanes that don't have all time points (e.g. because of drift)
        incomplete_GL = np.where(frame_counter<maxframe)[0]
        for inc in incomplete_GL:
            gl_str='0'+str(inc) if inc<10 else str(inc)
            pos_gl_name = dataset.get_first_tiff().split('.')[0]+'_Pos'+str(indp)+'_GL'+gl_str
            filename = current_saveto_folder+'/'+pos_gl_name
            if os.path.exists(filename):
                shutil.rmtree(filename)

        print(incomplete_GL)
        #save kymograph
        for gl_index in range(len(channel_centers)):
            if gl_index not in incomplete_GL:

                for c in range(len(colors)):
                    filename = kymo_folder+'/'+dataset.get_first_tiff().split('.')[0]+'_Pos'+str(indp)+'_GL'+str(gl_index)+'_col'+str(c)+'_kymo.tif'
                    skimage.external.tifffile.imsave(filename,kymo[:,:,c,gl_index].astype(np.uint16),append = 'force',imagej = True, metadata = metadataK)


    # finalize measurement of processing time
    end1 = time.time()
    print("Processing time [s]:" + str(end1 - start1))
