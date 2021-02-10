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
from mmpreprocesspy.MicroManagerOmeTiffReader import MicroManagerOmeTiffReader
from mmpreprocesspy.image_preprocessing import ImagePreprocessor
from mmpreprocesspy.moma_image_processing import MomaImageProcessor
from mmpreprocesspy.GlDetectionTemplate import GlDetectionTemplate
from mmpreprocesspy.support import saturate_image
import cv2 as cv
import csv

def get_position_folder_path(result_base_path, indp):
    """
    Return the path to the position folder containing the growthlane folders.

    :param result_base_path:
    :param indp:
    :return:
    """
    return result_base_path + '/' + 'Pos' + str(indp) + '/'

def get_normalization_log_folder_path(result_base_path, indp):
    """
    Return the path to the folder, where we will save log-data about how the
    was calcuatedd and performed normalization.

    :param result_base_path:
    :param indp:
    :return:
    """
    folder_path = os.path.normpath(os.path.join(result_base_path, 'Pos' + str(indp) + '_normalization_log'))
    os.makedirs(folder_path, exist_ok=True)
    return folder_path

def get_gl_folder_path(result_base_path, indp, gl_index):
    """
    Return path to the folder containing the growthlane and kymo image stacks.

    :param result_base_path:
    :param indp:
    :param gl_index:
    :return:
    """
    return get_position_folder_path(result_base_path, indp) + '/Pos' + str(indp) + '_GL' + str(
        gl_index)

def get_gl_tiff_path(result_base_path, base_name, indp, gl_index):
    """
    Return path to the growthlane image stack.

    :param result_base_path:
    :param base_name:
    :param indp:
    :param gl_index:
    :return:
    """
    return get_gl_folder_path(result_base_path, indp, gl_index) + '/' + base_name + '_Pos' + str(indp) + '_GL' + str(gl_index) + '.tiff'


def get_gl_csv_path(result_base_path, base_name, indp, gl_index):
    """
    Return path to the growthlane image stack.

    :param result_base_path:
    :param base_name:
    :param indp:
    :param gl_index:
    :return:
    """
    return get_gl_folder_path(result_base_path, indp, gl_index) + '/' + base_name + '_Pos' + str(indp) + '_GL' + str(gl_index) + '.csv'


def get_kymo_tiff_path(result_base_path, base_name, indp, gl_index):
    """
    Return path to the kymo-graph image-stack.

    :param result_base_path:
    :param base_name:
    :param indp:
    :param gl_index:
    :param color_index:
    :return:
    """

    return get_gl_folder_path(result_base_path, indp, gl_index) + '/' + base_name + '_Pos' + str(indp) + '_GL' + str(gl_index) + '_kymo.tiff'


def preproc_fun(data_folder,
                folder_to_save,
                positions=None,
                minframe=None,
                maxframe=None,
                flatfield_directory=None,
                dark_noise=None,
                gaussian_sigma=None,
                growthlane_length_threshold=0,
                main_channel_angle=None,
                roi_boundary_offset_at_mother_cell=None,
                gl_detection_template_path=None,
                normalization_config_path=None):

    # create a micro-manager image object
    dataset = MicroManagerOmeTiffReader(data_folder)

    # define basic parameters
    colors = dataset.get_channels()
    phase_channel_index = 0

    # get default values for non-specified optional parameters
    if minframe is None:
        minframe = 0
    if maxframe is None:
        maxframe = dataset.get_number_of_frames()
    if positions is None:
        nr_of_positions_in_data = len(dataset.get_position_names())
        positions = range(0, nr_of_positions_in_data)
    nrOfFrames = maxframe - minframe

    if roi_boundary_offset_at_mother_cell is None:
        roi_boundary_offset_at_mother_cell = 0

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
            flatfield = MicroManagerOmeTiffReader(flatfield_directory)
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
        first_phc_image = dataset.get_image_fast(channel=phase_channel_index, frame=minframe, position=position_index)

        # Process first image to find ROIs, etc.
        imageProcessor = MomaImageProcessor()
        if gl_detection_template_path:
            gl_detection_template = GlDetectionTemplate()
            gl_detection_template.load_config(gl_detection_template_path)
            imageProcessor.gl_detection_template = gl_detection_template
        imageProcessor.load_numpy_image_array(first_phc_image)
        imageProcessor.growthlane_length_threshold = growthlane_length_threshold
        imageProcessor.main_channel_angle = main_channel_angle
        imageProcessor.roi_boundary_offset_at_mother_cell = roi_boundary_offset_at_mother_cell

        imageProcessor.process_image()

        # store GL index image
        if not os.path.exists(os.path.dirname(folder_to_save)):
            os.makedirs(os.path.dirname(folder_to_save))

        path = folder_to_save + '/' + 'Pos' + str(position_index) + '_GL_index_initial.tiff'
        store_gl_index_image(imageProcessor.growthlane_rois, imageProcessor.image, path)

        # create empty kymographs to fill
        kymographs = [np.zeros((roi.length, nrOfFrames, len(colors))) for roi in imageProcessor.growthlane_rois]

        # initialize list of images to hold the final GL crop images
        nr_of_timesteps = maxframe - minframe
        nr_of_color_channels = len(colors)
        gl_image_path_dict = get_gl_image_image_paths(imageProcessor.growthlane_rois, folder_to_save, base_name, position_index)
        gl_csv_path_dict = get_gl_image_csv_paths(imageProcessor.growthlane_rois, folder_to_save, base_name, position_index)
        gl_image_dict = get_gl_image_stacks(imageProcessor.growthlane_rois, nr_of_timesteps, nr_of_color_channels, gl_image_path_dict)
        kymo_image_path_dict = get_kymo_image_image_paths(imageProcessor.growthlane_rois, folder_to_save, base_name, position_index)
        kymo_image_dict = get_kymo_image_stacks(imageProcessor.growthlane_rois, nr_of_timesteps, nr_of_color_channels, kymo_image_path_dict)

        # go through time-lapse and cut out channels
        for frame_index, t in enumerate(range(minframe, maxframe)):
            image = dataset.get_image_fast(channel=phase_channel_index, frame=t, position=position_index)
            imageProcessor.determine_image_shift(image)
            growthlane_rois = copy.deepcopy(imageProcessor.growthlane_rois)

            print(f"Shift of frame {t}: {imageProcessor.horizontal_shift:.2}, {imageProcessor.vertical_shift:.2}")

            growthlane_rois = translate_gl_rois(growthlane_rois, (-imageProcessor.horizontal_shift, -imageProcessor.vertical_shift))

            growthlane_rois, gl_image_dict, kymo_image_dict, gl_image_path_dict = remove_gls_outside_of_image(image, growthlane_rois, imageProcessor, gl_image_dict, kymo_image_dict, gl_image_path_dict)

            color_image_stack = dataset.get_image_stack(frame_index=t, position_index=position_index)  # TODO: rename this to e.g. 'current_image_frame'

            # correct images and append corrected and non-corrected images
            if preprocessor is not None:
                corrected_colors = preprocessor.process_image_stack(color_image_stack[:, :, 1:])  # correct all colors, but the PhC channel
                # corrected_color_image_stack = np.append(corrected_color_image_stack,
                #                                         color_image_stack[:, :, 1:], 2)
                color_image_stack_corr = np.append(color_image_stack[:, :, 0, np.newaxis], corrected_colors, 2)  # append corrected channel values
                color_image_stack_corr = np.append(color_image_stack_corr, color_image_stack[:, :, 1:], 2)  # append original channel values
                color_image_stack = color_image_stack_corr

            if normalization_config_path:
                phc_image = color_image_stack[:, :, 0]
                output_path = get_normalization_log_folder_path(folder_to_save, position_index)
                imageProcessor.set_normalization_ranges_and_save_log_data(growthlane_rois, phc_image, frame_index, position_index, output_path)

            # if normalization_mode is 1:
            #     phc_image = color_image_stack[:, :, 0]
            #     imageProcessor.get_registered_image()
            #     pass

            append_gl_roi_images(frame_index, growthlane_rois, gl_image_dict, color_image_stack)
            append_to_kymo_graph(frame_index, growthlane_rois, kymo_image_dict, color_image_stack)
            append_gl_csv(frame_index, growthlane_rois, gl_csv_path_dict)

        finalize_memmap_images(growthlane_rois, gl_image_dict)
        finalize_memmap_images(growthlane_rois, kymo_image_dict)

        path = folder_to_save + '/' + 'Pos' + str(position_index) + '_GL_index_final.tiff'
        store_gl_index_image(imageProcessor.growthlane_rois, imageProcessor.image, path)

    # # finalize measurement of processing time
    # print("Out of bounds ROIs: " + str(incomplete_GL))
    end1 = time.time()
    print("Processing time [s]:" + str(end1 - start1))


def get_gl_image_stacks(growthlane_rois, nr_of_timesteps, nr_of_color_channels, gl_image_path_dict):
    gl_image_stacks = {}
    for gl_roi in growthlane_rois:
        image_path = gl_image_path_dict[gl_roi.id]
        gl_image_stacks[gl_roi.id] = initialize_gl_roi_image_stack(gl_roi, nr_of_timesteps, nr_of_color_channels, image_path)
    return gl_image_stacks


def get_kymo_image_stacks(growthlane_rois, nr_of_timesteps, nr_of_color_channels, kymo_image_path_dict):
    kymo_image_stacks = {}
    for gl_roi in growthlane_rois:
        image_path = kymo_image_path_dict[gl_roi.id]
        kymo_image_stacks[gl_roi.id] = initialize_kymo_image_stack(gl_roi, nr_of_timesteps, nr_of_color_channels, image_path)
    return kymo_image_stacks


def get_gl_image_image_paths(growthlane_rois, folder_to_save, base_name, position_ind):
    gl_image_paths = {}
    for gl_roi in growthlane_rois:
        gl_image_paths[gl_roi.id] = get_gl_tiff_path(folder_to_save, base_name, position_ind, calculate_gl_output_index(gl_roi.id))
    return gl_image_paths


def get_gl_image_csv_paths(growthlane_rois, folder_to_save, base_name, position_ind):
    gl_image_paths = {}
    for gl_roi in growthlane_rois:
        gl_image_paths[gl_roi.id] = get_gl_csv_path(folder_to_save, base_name, position_ind, calculate_gl_output_index(gl_roi.id))
    return gl_image_paths


def get_kymo_image_image_paths(growthlane_rois, folder_to_save, base_name, position_index):
    kymo_image_paths = {}
    for gl_roi in growthlane_rois:
        kymo_image_paths[gl_roi.id] = get_kymo_tiff_path(folder_to_save, base_name, position_index, calculate_gl_output_index(gl_roi.id))
    return kymo_image_paths


def calculate_gl_output_index(gl_id):
    return gl_id + 1  # start GL indexing with 1 to be compatible with legacy preprocessing


def translate_gl_rois(growthlane_rois, shift_x_y):
    for gl_roi in growthlane_rois:
        gl_roi.roi.translate(shift_x_y)
    return growthlane_rois


def remove_gls_outside_of_image(image, growthlane_rois, imageProcessor, gl_image_dict, kymo_image_dict, gl_image_path_dict):
    """
    This method checks, if a GL ROI lies outside of the image.
    If so, it is removed from all lists/dicts.

    :param image:
    :param growthlane_rois:
    :param gl_image_dict:
    :param kymo_image_dict:
    :param gl_image_path_dict:
    :return:
    """

    inds = list(range(len(growthlane_rois)))
    inds.reverse()
    for ind in inds:
        gl_roi = growthlane_rois[ind]
        if not gl_roi.roi.is_inside_image(image):
            del gl_image_dict[gl_roi.id]
            del kymo_image_dict[gl_roi.id]
            gl_folder_path = os.path.dirname(gl_image_path_dict[gl_roi.id])
            del gl_image_path_dict[gl_roi.id]
            del growthlane_rois[ind]
            del imageProcessor.growthlane_rois[ind]
            if os.path.exists(gl_folder_path):
                shutil.rmtree(gl_folder_path)

    return growthlane_rois, gl_image_dict, kymo_image_dict, gl_image_path_dict


def append_gl_csv(frame_index, growthlane_rois, gl_csv_path_dict):
    for ind, gl_roi in enumerate(growthlane_rois):
        path = gl_csv_path_dict[ind]
        with open(path, mode='a') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            if frame_index == 0:  # write header, if we are at the first frame
                csv_writer.writerow(['frame', 'norm_range_min', 'norm_range_max'])
            range = gl_roi.normalization_range
            csv_writer.writerow([frame_index, np.round(range[0], decimals=2), np.round(range[1], decimals=2)])


def append_gl_roi_images(frame_index, growthlane_rois, gl_image_dict, color_image_stack):
    z_index = 0
    nr_of_colors = color_image_stack.shape[2]
    for gl_roi in growthlane_rois:
        gl_image_dict[gl_roi.id][frame_index, z_index, ...] = gl_roi.get_oriented_roi_image(np.moveaxis(color_image_stack, -1, 0))


def append_to_kymo_graph(frame_index, growthlane_rois, kymo_image_dict, color_image_stack):
    stack_time_index = 0  # kymo-graph does not have a time-index
    z_index = 0
    nr_of_colors = color_image_stack.shape[2]
    for gl_roi in growthlane_rois:
        gl_roi_crop = gl_roi.get_oriented_roi_image(np.moveaxis(color_image_stack, -1, 0))
        for color_index in range(0, nr_of_colors):  # add remaining channels
            kymo_image_dict[gl_roi.id][stack_time_index, z_index, color_index, :, frame_index] = get_kymo_graph_slice(gl_roi_crop[color_index, ...])


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


def finalize_memmap_images(growthlane_rois, gl_image_dict):
    for gl_roi in growthlane_rois:
        gl_image_dict[gl_roi.id].flush()
        del gl_image_dict[gl_roi.id]


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


def store_gl_index_image(growthlane_rois, full_frame_image, path):
    """ Draw the growthlane ROIs and indices onto the image and save it. """
    font = cv.FONT_HERSHEY_SIMPLEX
    full_frame_image = saturate_image(full_frame_image, 0.1, 0.3)
    normalized_image = cv.normalize(full_frame_image, None, 0, 255, cv.NORM_MINMAX)
    final_image = np.array(normalized_image, dtype=np.uint8)

    for roi in growthlane_rois:
        roi.roi.draw_to_image(final_image, False)
        gl_index = calculate_gl_output_index(roi.id)
        cv.putText(final_image, str(gl_index), (np.int0(roi.roi.center[0]), np.int0(roi.roi.center[1])), font, 1, (255, 255, 255), 2, cv.LINE_AA)

    cv.imwrite(path, final_image)
