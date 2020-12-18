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
import cv2 as cv


def get_position_folder_path(result_base_path, indp):
    return result_base_path + '/' + 'Pos' + str(indp) + '/'

def get_gl_tiff_path(result_base_path, base_name, indp, gl_index):
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
    for position_index in positions:  # MM: Currently proproc_fun.py in only run for a single position; so this loop is not needed
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
            position_folder = get_position_folder_path(folder_to_save, position_index)
            preprocessor.save_flatfields(position_folder)

        # load first phase contrast image
        image_base = dataset.get_image_fast(channel=phase_channel_index, frame=minframe, position=position_index)

        # Process first image to find ROIs, etc.
        imageProcessor = MomaImageProcessor()
        imageProcessor.load_numpy_image_array(image_base)
        imageProcessor.growthlane_length_threshold = growthlane_length_threshold
        imageProcessor.process_image()
        channel_centers = imageProcessor.channel_centers

        # store GL index image
        if not os.path.exists(os.path.dirname(folder_to_save)):
            os.makedirs(os.path.dirname(folder_to_save))

        path = folder_to_save + '/' + 'Pos' + str(position_index) + '_GL_index_initial.tiff'
        store_gl_index_image(imageProcessor.growthlane_rois, imageProcessor.image, path)

        # create empty kymographs to fill
        kymographs = [np.zeros((roi.length, nrOfFrames, len(colors))) for roi in imageProcessor.growthlane_rois]
        metadataK = {'channels': len(colors), 'slices': 1, 'frames': len(channel_centers), 'hyperstack': True,
                     'loop': False}

        frame_counter = np.zeros(len(channel_centers))  # stores per growthlane, the number of processed images

        # initialize list of images to hold the final GL crop images
        nr_of_timesteps = maxframe - minframe
        nr_of_color_channels = len(colors)
        gl_image_path_dict = get_gl_image_image_paths(imageProcessor.growthlane_rois, folder_to_save, base_name, position_index)
        gl_image_dict = get_gl_image_stacks(imageProcessor.growthlane_rois, nr_of_timesteps, nr_of_color_channels, gl_image_path_dict)
        kymo_image_path_dict = get_kymo_image_image_paths(imageProcessor.growthlane_rois, folder_to_save, base_name, position_index)
        kymo_image_dict = get_kymo_image_stacks(imageProcessor.growthlane_rois, nr_of_timesteps, nr_of_color_channels, kymo_image_path_dict)

        # go through time-lapse and cut out channels
        for frame_index, t in enumerate(range(minframe, maxframe)):
            if np.mod(t, 10) == 0:
                print('working on frame: ' + str(t))  # output frame number

            image = dataset.get_image_fast(channel=phase_channel_index, frame=t, position=position_index)
            imageProcessor.determine_image_shift(image)
            growthlane_rois = copy.deepcopy(imageProcessor.growthlane_rois)

            print("Shift frame "+str(t)+": "+str(imageProcessor.horizontal_shift)+", "+str(imageProcessor.vertical_shift))

            growthlane_rois = translate_gl_rois(growthlane_rois, (-imageProcessor.horizontal_shift, -imageProcessor.vertical_shift))

            growthlane_rois, gl_image_dict = remove_gls_outside_of_image(growthlane_rois, image.shape, gl_image_dict)

            color_image_stack = dataset.get_image_stack(frame=t, position=position_index)  # TODO: rename this to e.g. 'current_image_frame'

            # correct images and append corrected and non-corrected images
            if preprocessor is not None:
                corrected_colors = preprocessor.process_image_stack(color_image_stack[:, :, 1:])  # correct all colors, but the PhC channel
                # corrected_color_image_stack = np.append(corrected_color_image_stack,
                #                                         color_image_stack[:, :, 1:], 2)
                color_image_stack_corr = np.append(color_image_stack[:, :, 0, np.newaxis], corrected_colors, 2)  # append corrected channel values
                color_image_stack_corr = np.append(color_image_stack_corr, color_image_stack[:, :, 1:], 2)  # append original channel values
                color_image_stack = color_image_stack_corr

            append_gl_roi_images(frame_index, growthlane_rois, gl_image_dict, color_image_stack)
            append_to_kymo_graph(frame_index, growthlane_rois, kymo_image_dict, color_image_stack)

            # go through all channels, check if there's a corresponding one in the new image. If yes go through all
            #  colors,cut out channel, and append to tif stack. Append also to the Kymograph for each color.
            for gl_index, gl_roi in enumerate(growthlane_rois):
                if gl_roi.roi.is_inside_image(image):
                    frame_counter[gl_index] += 1

                    gl_file_path = get_gl_tiff_path(folder_to_save, base_name, position_index, gl_index + 1)  # gl_index+1 to comply with legacy indexing of growthlanes, which starts at 1

                    if not os.path.exists(os.path.dirname(gl_file_path)):
                        os.makedirs(os.path.dirname(gl_file_path))

                    save_gl_roi(metadata, color_image_stack, gl_roi, gl_file_path)
                    kymographs = append_to_kymographs(color_image_stack, gl_roi, kymographs, gl_index, t, minframe)

        # save_gl_roi_image(growthlane_rois, gl_image_dict, gl_image_path_dict)
        finalize_memmap_images(growthlane_rois, gl_image_dict)
        finalize_memmap_images(growthlane_rois, kymo_image_dict)

        path = folder_to_save + '/' + 'Pos' + str(position_index) + '_GL_index_final.tiff'
        store_gl_index_image(growthlane_rois, imageProcessor.image, path)

        # remove growth lanes that don't have all time points (e.g. because of drift)
        incomplete_GL = np.where(frame_counter < nrOfFrames)[0]
        for inc in incomplete_GL:
            gl_result_folder = os.path.dirname(get_gl_tiff_path(folder_to_save, base_name, position_index, inc + 1))  # inc+1 to comply with legacy indexing of growthlanes, which starts at 1
            if os.path.exists(gl_result_folder):
                shutil.rmtree(gl_result_folder)

        # # save kymograph
        # for gl_index in range(len(channel_centers)):
        #     if gl_index not in incomplete_GL:
        #         for color in range(len(colors)):
        #             kymo_file_path = get_kymo_tiff_path(folder_to_save, base_name, position_index, gl_index, color)
        #             if not os.path.exists(os.path.dirname(kymo_file_path)):
        #                 os.makedirs(os.path.dirname(kymo_file_path))
        #             tifffile.imwrite(kymo_file_path,
        #                                              kymographs[gl_index][:, :, color].astype(np.uint16),
        #                                              append='force', imagej=True, metadata=metadataK)

    # finalize measurement of processing time
    print("Out of bounds ROIs: " + str(incomplete_GL))
    end1 = time.time()
    print("Processing time [s]:" + str(end1 - start1))


def get_gl_image_stacks(growthlane_rois, nr_of_timesteps, nr_of_color_channels, gl_image_path_dict):
    gl_image_stacks = {}
    for gl_roi in growthlane_rois:
        image_path = gl_image_path_dict[gl_roi.id]
        gl_image_stacks[gl_roi.id] = initialize_gl_roi_image_stack(gl_roi, nr_of_timesteps, nr_of_color_channels, image_path)
    return gl_image_stacks


def get_kymo_image_stacks(growthlane_rois, nr_of_timesteps, nr_of_color_channels, gl_image_path_dict):
    kymo_image_stacks = {}
    for gl_roi in growthlane_rois:
        image_path = gl_image_path_dict[gl_roi.id]
        kymo_image_stacks[gl_roi.id] = initialize_kymo_image_stack(gl_roi, nr_of_timesteps, nr_of_color_channels, image_path)
    return kymo_image_stacks


def get_gl_image_image_paths(growthlane_rois, folder_to_save, base_name, position_ind):
    gl_image_paths = {}
    for gl_roi in growthlane_rois:
        gl_image_paths[gl_roi.id] = get_gl_tiff_path(folder_to_save, base_name, position_ind, gl_roi.id)
    return gl_image_paths


def get_kymo_image_image_paths(growthlane_rois, folder_to_save, base_name, position_index):
    kymo_image_paths = {}
    for gl_roi in growthlane_rois:
        kymo_image_paths[gl_roi.id] =  get_kymo_tiff_path_2(folder_to_save, base_name, position_index, gl_roi.id)
    return kymo_image_paths


def get_kymo_tiff_path_2(result_base_path, base_name, indp, gl_index):
    return result_base_path + '/' + 'Pos' + str(indp) + '/GL' + str(
        gl_index) + '/' + base_name + '_Pos' + str(indp) + '_GL' + str(gl_index) + '_kymo.tiff'


def translate_gl_rois(growthlane_rois, shift_x_y):
    for gl_roi in growthlane_rois:
        gl_roi.roi.translate(shift_x_y)
    return growthlane_rois


def remove_gls_outside_of_image(growthlane_rois, image_shape, gl_image_dict):
    # if gl_roi.roi.is_inside_image(image):
    #     frame_counter[gl_index] += 1

    return growthlane_rois, gl_image_dict


def append_gl_roi_images(time_index, growthlane_rois, gl_image_dict, color_image_stack):
    z_index = 0
    nr_of_colors = color_image_stack.shape[2]
    for gl_roi in growthlane_rois:
        color_index = 0
        gl_image_dict[gl_roi.id][time_index, z_index, color_index, ...] = gl_roi.get_oriented_roi_image(color_image_stack[:, :, color_index])
        for color_index in range(1, nr_of_colors):  # add remaining channels
            gl_image_dict[gl_roi.id][time_index, z_index, color_index, ...] = gl_roi.get_oriented_roi_image(color_image_stack[:, :, color_index])


def append_to_kymo_graph(time_index, growthlane_rois, kymo_image_dict, color_image_stack):
    stack_time_index = 0  # kymo-graph does not have a time-index
    z_index = 0
    nr_of_colors = color_image_stack.shape[2]
    for gl_roi in growthlane_rois:
        color_index = 0
        gl_roi_crop = gl_roi.get_oriented_roi_image(color_image_stack[:, :, color_index])
        kymo_image_dict[gl_roi.id][stack_time_index, z_index, color_index, :, time_index] = get_kymo_graph_slice(gl_roi_crop)
        for color_index in range(1, nr_of_colors):  # add remaining channels
            gl_roi_crop = gl_roi.get_oriented_roi_image(color_image_stack[:, :, color_index])
            kymo_image_dict[gl_roi.id][stack_time_index, z_index, color_index, :, time_index] = get_kymo_graph_slice(gl_roi_crop)


def get_kymo_graph_slice(gl_roi_crop):
    """
    Calculate the row-wise average intensity in the region of the GL channel.
    This constitutes one slice vertical column in the final kymo-graph.

    :param gl_roi_crop:
    :return:
    """

    gl_channel_width_halved = 10/2  # unit: [px]; channel width is roughly 10px
    image_width = gl_roi_crop.shape[1]
    image_center = int(image_width/2)
    start_ind = image_center - int(gl_channel_width_halved)
    end_ind = image_center + int(gl_channel_width_halved)
    kymo_slice = np.mean(gl_roi_crop[:, start_ind:end_ind], axis=1)
    return kymo_slice

def save_gl_roi_image(growthlane_rois, gl_image_dict, gl_image_path_dict):
    for gl_roi in growthlane_rois:
        gl_file_path = gl_image_path_dict[gl_roi.id]
        if not os.path.exists(os.path.dirname(gl_file_path)):
            os.makedirs(os.path.dirname(gl_file_path))
        tifffile.imwrite(gl_file_path, gl_image_dict[gl_roi.id], metadata={'axes': 'TZCYX'}, imagej=True)


def finalize_memmap_images(growthlane_rois, gl_image_dict):
    for gl_roi in growthlane_rois:
        gl_image_dict[gl_roi.id].flush()
        del gl_image_dict[gl_roi.id]


def save_gl_roi(metadata, color_image_stack, gl_roi, gl_file_path):
    pass
    # nr_of_colors = color_image_stack.shape[2]
    # for color in range(nr_of_colors):
    #     imtosave = gl_roi.get_oriented_roi_image(color_image_stack[:, :, color])
    #     tifffile.imwrite(gl_file_path, np.float32(imtosave), append='force',
    #                                      imagej=True, metadata=metadata)

def initialize_gl_roi_image_stack(gl_roi, nr_of_timesteps, nr_of_color_channels, image_path):
    nr_of_z_planes = 1
    image_height = gl_roi.length
    image_width = gl_roi.width
    image_shape = (nr_of_timesteps, nr_of_z_planes, nr_of_color_channels, image_height, image_width)
    # image_stack = np.float32(np.zeros(image_shape))
    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    image_stack = tifffile.memmap(image_path, shape=image_shape, dtype='float32', metadata={'axes': 'TZCYX'}, imagej=True)
    image_stack[:] = np.nan  # initialize to nan, so that we can test later that all pixels were correctly written to
    return image_stack


def initialize_kymo_image_stack(gl_roi, nr_of_timesteps, nr_of_color_channels, image_path):
    nr_stack_time_steps = 1  # the kymo-graph does not have any timesteps
    nr_of_z_planes = 1
    image_height = gl_roi.length
    image_width = nr_of_timesteps  # the kymo-graph has a many columns as frames in the movie
    image_shape = (nr_stack_time_steps, nr_of_z_planes, nr_of_color_channels, image_height, image_width)
    os.makedirs(os.path.dirname(image_path), exist_ok=True)
    image_stack = tifffile.memmap(image_path, shape=image_shape, dtype='float32', metadata={'axes': 'TZCYX'}, imagej=True)
    image_stack[:] = np.nan  # initialize to nan, so that we can test later that all pixels were correctly written to
    return image_stack


def append_to_kymographs(color_image_stack, gl_roi, kymographs, gl_index, t, minframe):
    kymo_index = t - minframe
    nr_of_colors = color_image_stack.shape[2]
    for color in range(nr_of_colors):
        imtosave = gl_roi.get_oriented_roi_image(color_image_stack[:, :, color])
        kymographs[gl_index][:, kymo_index, color] = np.mean(imtosave, axis=1)
    return kymographs

def store_gl_index_image(growthlane_rois, full_frame_image, path):
    """ Draw the growthlane ROIs and indices onto the image and save it. """
    font = cv.FONT_HERSHEY_SIMPLEX
    rotated_rois = [x.roi for x in growthlane_rois]
    # show_image_with_rotated_rois(image, rotated_rois)
    # normalizedImg = None
    normalized_image = cv.normalize(full_frame_image, None, 0, 255, cv.NORM_MINMAX)
    final_image = np.array(normalized_image, dtype=np.uint8)

    for gl_index, roi in enumerate(rotated_rois):
        roi.draw_to_image(final_image, False)
        cv.putText(final_image, str(gl_index + 1), (np.int0(roi.center[0]), np.int0(roi.center[1])), font, 1, (255, 255, 255), 2, cv.LINE_AA)

    cv.imwrite(path, final_image)
