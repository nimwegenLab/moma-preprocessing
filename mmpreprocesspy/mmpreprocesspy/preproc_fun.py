import copy
import os
import shutil
import time

import numpy as np
import skimage.external.tifffile
import skimage.filters
import skimage.measure
import skimage.transform
from mmpreprocesspy.MMdata import MMData
from mmpreprocesspy.image_preprocessing import ImagePreprocessor
from mmpreprocesspy.moma_image_processing import MomaImageProcessor


def get_gl_tiff_path(result_base_path, base_name, indp, gl_index):
    gl_index += 1  # we do this to comply with legacy indexing of growthlanes, which starts at 1
    return result_base_path + '/' + 'Pos' + str(indp) + '/GL' + str(
        gl_index) + '/' + base_name + '_Pos' + str(indp) + '_GL' + str(gl_index) + '.tiff'


def get_kymo_tiff_path(result_base_path, base_name, indp, gl_index, color_index):
    gl_index += 1  # we do this to comply with legacy indexing of growthlanes, which starts at 1
    return result_base_path + '/' + 'Pos' + str(indp) + '/GL' + str(
        gl_index) + '/' + base_name + '_Pos' + str(indp) + '_GL' + str(gl_index) + '_Col' + str(
        color_index) + '_kymo.tiff'


def preproc_fun(data_folder, folder_to_save, positions=None, minframe=None, maxframe=None, flatfield_directory=None, dark_noise=None, gaussian_sigma=None):
    print("This is test-output to see if logging works ...")
    # create a micro-manager image object
    dataset = MMData(data_folder)

    # define basic parameters
    colors = dataset.get_channels()
    phase_channel_index = 0

    # load and use flatfield data, if provided
    preprocessor = None
    if flatfield_directory is not None:
        flatfield = MMData(flatfield_directory)
        preprocessor = ImagePreprocessor(dataset, flatfield, dark_noise, gaussian_sigma)
        preprocessor.initialize()
        # since we are correcting the images: correct the number and naming of the available colors
        colors_orig = colors.copy()
        colors[1:] = [name+'_corrected' for name in colors[1:]]
        colors = colors + colors_orig[1:]


    # get default values for non-specified optional parameters
    if minframe is None:
        minframe = 0
    if maxframe is None:
        maxframe = dataset.get_max_frame() + 1  # +1 needed, because range(0,N) goes from 0 to N-1 (see below)
    if positions is None:
        nr_of_positions_in_data = dataset.get_position_names()[0].__len__()
        positions = range(0, nr_of_positions_in_data)

    # recover the basic experiment name
    base_name = dataset.get_first_tiff().split('.')[0]

    # define metadata for imagej
    metadata = {'channels': len(colors), 'slices': 1, 'frames': maxframe, 'hyperstack': True, 'loop': False}

    # start measurement of processing time
    start1 = time.time()
    for indp in positions:  # MM: Currently proproc_fun.py in only run for a single position; so this loop is not needed
        # load first phase image
        image_base = dataset.get_image_fast(channel=phase_channel_index, frame=0, position=indp)

        # Process first image to find ROIs, etc.
        imageProcessor = MomaImageProcessor()
        imageProcessor.load_numpy_image_array(image_base)
        imageProcessor.process_image()
        channel_centers = imageProcessor.channel_centers

        # create empty kymographs to fill
        kymographs = [np.zeros((roi.length, maxframe, len(colors))) for roi in imageProcessor.growthlane_rois]
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

            for gl_roi in growthlane_rois:
                gl_roi.roi.translate((-imageProcessor.horizontal_shift, -imageProcessor.vertical_shift))

            color_image_stack = dataset.get_image_stack(frame=t, position=indp)

            # correct images and append corrected and non-corrected images
            if preprocessor is not None:
                corrected_color_image_stack = preprocessor.process_image_stack(color_image_stack)
                color_image_stack = np.append(corrected_color_image_stack,
                                              color_image_stack[:, :, 1:], 2)

            # go through all channels, check if there's a corresponding one in the new image. If yes go through all
            #  colors,cut out channel, and append to tif stack. Append also to the Kymograph for each color.
            for gl_index, gl_roi in enumerate(growthlane_rois):
                if gl_roi.roi.is_inside_image(image):
                    frame_counter[gl_index] += 1

                    gl_file_path = get_gl_tiff_path(folder_to_save, base_name, indp, gl_index)
                    if not os.path.exists(os.path.dirname(gl_file_path)):
                        os.makedirs(os.path.dirname(gl_file_path))

                    save_gl_roi(metadata, color_image_stack, gl_roi, gl_file_path, preprocessor)
                    kymographs = append_to_kymographs(color_image_stack, gl_roi, kymographs, gl_index, t)

        # remove growth lanes that don't have all time points (e.g. because of drift)
        incomplete_GL = np.where(frame_counter < maxframe)[0]
        for inc in incomplete_GL:
            gl_result_folder = os.path.dirname(get_gl_tiff_path(folder_to_save, base_name, indp, inc))
            if os.path.exists(gl_result_folder):
                shutil.rmtree(gl_result_folder)

        print(incomplete_GL)
        # save kymograph
        for gl_index in range(len(channel_centers)):
            if gl_index not in incomplete_GL:
                for color in range(len(colors)):
                    kymo_file_path = get_kymo_tiff_path(folder_to_save, base_name, indp, gl_index, color)
                    if not os.path.exists(os.path.dirname(kymo_file_path)):
                        os.makedirs(os.path.dirname(kymo_file_path))
                    skimage.external.tifffile.imsave(kymo_file_path,
                                                     kymographs[gl_index][:, :, color].astype(np.uint16),
                                                     append='force', imagej=True, metadata=metadataK)

    # finalize measurement of processing time
    end1 = time.time()
    print("Processing time [s]:" + str(end1 - start1))


def save_gl_roi(metadata, color_image_stack, gl_roi, gl_file_path, preprocessor):
    nr_of_colors = color_image_stack.shape[2]
    for color in range(nr_of_colors):
        imtosave = gl_roi.get_oriented_roi_image(color_image_stack[:, :, color])
        skimage.external.tifffile.imsave(gl_file_path, imtosave.astype(np.uint16), append='force',
                                         imagej=True, metadata=metadata)


def append_to_kymographs(color_image_stack, gl_roi, kymographs, gl_index, t):
    nr_of_colors = color_image_stack.shape[2]
    for color in range(nr_of_colors):
        imtosave = gl_roi.get_oriented_roi_image(color_image_stack[:, :, color])
        kymographs[gl_index][:, t, color] = np.mean(imtosave, axis=1)
    return kymographs