import copy
import os
import shutil
import time

import numpy as np
import matplotlib.pyplot as plt
import tifffile
import skimage.filters
import skimage.measure
import skimage.transform
from mmpreprocesspy.MMdata import MMData
from mmpreprocesspy.image_preprocessing import ImagePreprocessor
from mmpreprocesspy.moma_image_processing import MomaImageProcessor


def get_gl_index_tiff_path(result_base_path, indp):
    return result_base_path + '/' + 'Pos' + str(indp) + '_GL_index.tiff'


def get_position_folder_path(result_base_path, indp):
    return result_base_path + '/' + 'Pos' + str(indp) + '/'

def get_gl_tiff_path(result_base_path, base_name, indp, gl_index):
    gl_index += 1  # we do this to comply with legacy indexing of growthlanes, which starts at 1
    return result_base_path + '/' + 'Pos' + str(indp) + '/GL' + str(
        gl_index) + '/' + base_name + '_Pos' + str(indp) + '_GL' + str(gl_index) + '.tiff'


def get_kymo_tiff_path(result_base_path, base_name, indp, gl_index, color_index):
    gl_index += 1  # we do this to comply with legacy indexing of growthlanes, which starts at 1
    return result_base_path + '/' + 'Pos' + str(indp) + '/GL' + str(
        gl_index) + '/' + base_name + '_Pos' + str(indp) + '_GL' + str(gl_index) + '_Col' + str(
        color_index) + '_kymo.tiff'


def preproc_fun(data_folder, folder_to_save, positions=None, minframe=None, maxframe=None, flatfield_directory=None, dark_noise=None, gaussian_sigma=None, growthlane_length_threshold=0):
    # create a micro-manager image object
    dataset = MMData(data_folder)

    # define basic parameters
    colors = dataset.get_channels()
    phase_channel_index = 0

    # get default values for non-specified optional parameters
    if minframe is None:
        minframe = 0
    if maxframe is None:
        maxframe = dataset.get_max_frame() + 1  # +1 needed, because range(0,N) goes from 0 to N-1 (see below)
    if positions is None:
        nr_of_positions_in_data = dataset.get_position_names()[0].__len__()
        positions = range(0, nr_of_positions_in_data)

    nrOfFrames = maxframe - minframe

    # recover the basic experiment name
    base_name = dataset.get_first_tiff().split('.')[0]

    # define metadata for imagej
    metadata = {'channels': len(colors), 'slices': 1, 'frames': nrOfFrames, 'hyperstack': True, 'loop': False}

    # start measurement of processing time
    start1 = time.time()
    for indp in positions:  # MM: Currently proproc_fun.py in only run for a single position; so this loop is not needed
        # load and use flatfield data, if provided
        preprocessor = None
        if flatfield_directory is not None:
            flatfield = MMData(flatfield_directory)
            preprocessor = ImagePreprocessor(dataset, flatfield, dark_noise, gaussian_sigma)
            roi_shape = (dataset.get_image_height(), dataset.get_image_width())
            preprocessor.calculate_flatfields(roi_shape)
            # since we are correcting the images: correct the number and naming of the available colors
            colors_orig = colors.copy()
            colors[1:] = [name + '_corrected' for name in colors[1:]]
            colors = colors + colors_orig[1:]
            position_folder = get_position_folder_path(folder_to_save, indp)
            preprocessor.save_flatfields(position_folder)

        # load first phase image
        image_base = dataset.get_image_fast(channel=phase_channel_index, frame=minframe, position=indp)

        # Process first image to find ROIs, etc.
        imageProcessor = MomaImageProcessor()
        imageProcessor.load_numpy_image_array(image_base)
        imageProcessor.growthlane_length_threshold = growthlane_length_threshold
        imageProcessor.process_image()
        channel_centers = imageProcessor.channel_centers

        # store GL index image
        if not os.path.exists(os.path.dirname(folder_to_save)):
            os.makedirs(os.path.dirname(folder_to_save))
        path = get_gl_index_tiff_path(folder_to_save, indp)
        imageProcessor.store_gl_index_image(path)

        # create empty kymographs to fill
        kymographs = [np.zeros((roi.length, nrOfFrames, len(colors))) for roi in imageProcessor.growthlane_rois]
        metadataK = {'channels': len(colors), 'slices': 1, 'frames': len(channel_centers), 'hyperstack': True,
                     'loop': False}

        frame_counter = np.zeros(len(channel_centers))  # stores per growthlane, the number of processed images
        # go through time-lapse and cut out channels
        for t in range(minframe, maxframe):
            if np.mod(t, 10) == 0:
                print('working on frame: ' + str(t))  # output frame number

            image = dataset.get_image_fast(channel=phase_channel_index, frame=t, position=indp)
            imageProcessor.determine_image_shift(image)
            growthlane_rois = copy.deepcopy(imageProcessor.growthlane_rois)

            print("Shift frame "+str(t)+": "+str(imageProcessor.horizontal_shift)+", "+str(imageProcessor.vertical_shift))

            for gl_roi in growthlane_rois:
                gl_roi.roi.translate((-imageProcessor.horizontal_shift, -imageProcessor.vertical_shift))

            color_image_stack = dataset.get_image_stack(frame=t, position=indp)

            # correct images and append corrected and non-corrected images
            if preprocessor is not None:
                corrected_colors = preprocessor.process_image_stack(color_image_stack[:, :, 1:])  # correct all colors, but the PhC channel
                # corrected_color_image_stack = np.append(corrected_color_image_stack,
                #                                         color_image_stack[:, :, 1:], 2)
                color_image_stack_corr = np.append(color_image_stack[:, :, 0, np.newaxis], corrected_colors, 2)  # append corrected channel values
                color_image_stack_corr = np.append(color_image_stack_corr, color_image_stack[:, :, 1:], 2)  # append original channel values
                color_image_stack = color_image_stack_corr

            # go through all channels, check if there's a corresponding one in the new image. If yes go through all
            #  colors,cut out channel, and append to tif stack. Append also to the Kymograph for each color.
            for gl_index, gl_roi in enumerate(growthlane_rois):
                if gl_roi.roi.is_inside_image(image):
                    frame_counter[gl_index] += 1

                    gl_file_path = get_gl_tiff_path(folder_to_save, base_name, indp, gl_index)
                    if not os.path.exists(os.path.dirname(gl_file_path)):
                        os.makedirs(os.path.dirname(gl_file_path))

                    save_gl_roi(metadata, color_image_stack, gl_roi, gl_file_path)
                    kymographs = append_to_kymographs(color_image_stack, gl_roi, kymographs, gl_index, t, minframe)

        # remove growth lanes that don't have all time points (e.g. because of drift)
        incomplete_GL = np.where(frame_counter < nrOfFrames)[0]
        for inc in incomplete_GL:
            gl_result_folder = os.path.dirname(get_gl_tiff_path(folder_to_save, base_name, indp, inc))
            if os.path.exists(gl_result_folder):
                shutil.rmtree(gl_result_folder)

        # save kymograph
        for gl_index in range(len(channel_centers)):
            if gl_index not in incomplete_GL:
                for color in range(len(colors)):
                    kymo_file_path = get_kymo_tiff_path(folder_to_save, base_name, indp, gl_index, color)
                    if not os.path.exists(os.path.dirname(kymo_file_path)):
                        os.makedirs(os.path.dirname(kymo_file_path))
                    tifffile.imwrite(kymo_file_path,
                                                     kymographs[gl_index][:, :, color].astype(np.uint16),
                                                     append='force', imagej=True, metadata=metadataK)

    # finalize measurement of processing time
    print("Out of bounds ROIs: " + str(incomplete_GL))
    end1 = time.time()
    print("Processing time [s]:" + str(end1 - start1))


def save_gl_roi(metadata, color_image_stack, gl_roi, gl_file_path):
    nr_of_colors = color_image_stack.shape[2]
    for color in range(nr_of_colors):
        imtosave = gl_roi.get_oriented_roi_image(color_image_stack[:, :, color])
        tifffile.imwrite(gl_file_path, np.float32(imtosave), append='force',
                                         imagej=True, metadata=metadata)


def append_to_kymographs(color_image_stack, gl_roi, kymographs, gl_index, t, minframe):
    kymo_index = t - minframe
    nr_of_colors = color_image_stack.shape[2]
    for color in range(nr_of_colors):
        imtosave = gl_roi.get_oriented_roi_image(color_image_stack[:, :, color])
        kymographs[gl_index][:, kymo_index, color] = np.mean(imtosave, axis=1)
    return kymographs

