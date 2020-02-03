import cv2
# import mmpreprocesspy.dev_auxiliary_functions as aux
import numpy as np
import skimage.transform
from mmpreprocesspy.GrowthlaneRoi import GrowthlaneRoi
from mmpreprocesspy.data_region import DataRegion
from mmpreprocesspy.roi import Roi
from mmpreprocesspy.rotated_roi import RotatedRoi
from scipy.signal import savgol_filter, find_peaks
from skimage.filters import threshold_otsu


# find rotation, channel boundaries and positions for first image that is then used as reference
def process_image(image, growthlane_length_threshold=0):
    main_channel_angle = find_main_channel_orientation(image)
    if main_channel_angle != 0:
        image = skimage.transform.rotate(image, -main_channel_angle,
                                         resize=True)  # rotate image angle back to 0, if needed

    # find rotation angle
    angle = find_rotation(image)
    main_channel_angle += angle

    # recalculate channel region boundary on rotated image
    image_rot = skimage.transform.rotate(image, angle, cval=0)
    mincol, maxcol, region_list = pattern_limits(image_rot, use_smoothing=True)
    refine_regions(image_rot, region_list)
    region_list = filter_regions(region_list, minimum_required_growthlane_length=growthlane_length_threshold)
    growthlane_rois, channel_centers = get_all_growthlane_rois(image_rot, region_list)

    return image_rot, main_channel_angle, mincol, maxcol, channel_centers, growthlane_rois


def filter_regions(region_list, minimum_required_growthlane_length):
    """
    Filter the regions using

    :param region_list:
    :return:
    """
    new_region_list = []
    for region in region_list:
        if region.end - region.start > minimum_required_growthlane_length:
            new_region_list.append(region)
    return new_region_list


def refine_regions(rotated_image, region_list):
    for region in region_list:
        refine_region(rotated_image, region)


def refine_region(rotated_image, region):
    """
    This method extends start and end-indexes of the channel areas, if necessary. It determines the maximum intensity in
    direction of the image columns in the channel region, which was determined using the FFT spectrum.
    The median of these maximum intensities is used as threshold value.
    It then shifts the start and end indices of the region in both directions until their respective intensity values
    fall below this threshold (it does this using a lookahead-interval to avoid aborting prematurely).

    :param rotated_image:
    :param region:
    :return:
    """
    look_ahead_length = 20  # the distance that the algorithm will look ahead to see if the threshold is passed
    projected_max_intensities = np.max(rotated_image, axis=0)
    sorted_max_intensities = np.sort(projected_max_intensities)
    threshold = np.median(sorted_max_intensities)

    #  extend region at the channel start using lookahead interval
    while np.any(projected_max_intensities[region.start - look_ahead_length:region.start] > threshold):
        region.start -= 1
    #  extend region at the channel end using lookahead interval
    while np.any(projected_max_intensities[region.end:region.end + look_ahead_length] > threshold):
        region.end += 1


def get_growthlane_periodicity(growthlane_region_image):
    '''
    Determine the periodicity of the growthlanes and return it. It does so by calculating the average distance between
    the `number_of_peaks` of highest peaks.

    :param growthlane_region_image: region of the image containing growthlanes. The growthlanes are assumed to be
    oriented horizontally and the image should be cropped, so that it does not contain too much space before and behind
    the GLs. This method also assumed that the growthlanes appear bright foreground on dark background.
    '''
    min_distance_between_peaks = 10  # TODO-MM-20191205: Minimum distance between peaks should probably be a parameter.
    number_of_peaks = 5  # TODO-MM-20191205: Number of peaks for periodicity calculation probably better be a parameter.

    projected_image_intensity = np.sum(growthlane_region_image, axis=1)
    projected_image_intensity_zero_mean = projected_image_intensity - np.mean(projected_image_intensity)
    cross_corr = np.correlate(projected_image_intensity_zero_mean, projected_image_intensity_zero_mean, 'same')

    peak_inds = find_peaks(cross_corr, distance=min_distance_between_peaks)[0]
    peak_vals = cross_corr[peak_inds]

    peak_inds_sorted = [x for _, x in sorted(zip(peak_vals, peak_inds), key=lambda pair: pair[0])]
    periodicity = np.mean(np.diff(sorted(peak_inds_sorted[-number_of_peaks:])))
    return periodicity


def get_index_of_intensity_maximum_closest_to_position(intensity_profile, position):
    """
    Returns the index of the intensity maximum closest to :position:.

    :intensity_profile: intensity profile from which to get the closes maximum.
    :position: position to which the maximum should be closest.
    """
    peak_inds = find_peaks(intensity_profile, distance=10)[0]
    peak_vals = intensity_profile[peak_inds]
    peak_inds = peak_inds[peak_vals > 0]  # keep only positive maxima, because they correspond to the growthlane centers
    return peak_inds[np.argmin(np.abs(peak_inds - position))]


def get_position_of_first_growthlane(growthlane_region_image, periodicity):
    """
    Returns the position of the first growthlane in :growthlane_region_image: closest to the image origin.

    :param growthlane_region_image: region of the image containing growthlanes. The growthlanes are assumed to be
    oriented horizontally and the image should be cropped, so that it does not contain too much space before and behind
    the GLs. This method also assumed that the growthlanes appear bright foreground on dark background.
   """
    projected_image_intensity = np.sum(growthlane_region_image, axis=1)
    projected_image_intensity_zero_mean = projected_image_intensity - np.mean(projected_image_intensity)
    acf_of_intensity_profile = np.correlate(projected_image_intensity_zero_mean, projected_image_intensity_zero_mean,
                                            'same')
    ccf_of_acf_with_intensity_profile = np.correlate(projected_image_intensity_zero_mean, acf_of_intensity_profile,
                                                     'same')
    center_index = np.round(projected_image_intensity.shape[0] / 2)
    index = get_index_of_intensity_maximum_closest_to_position(ccf_of_acf_with_intensity_profile, center_index)
    shift = index - periodicity * np.floor(index / periodicity)  # get growthlane position closest to the image origin
    return shift


def get_gl_center_positions_in_growthlane_region(growthlane_region_image):
    """
    Return the center positions of the growthlanes. The centers refer to the vertical position of the centers of the
    growthlanes.

    :param growthlane_region_image: region of the image containing growthlanes. The growthlanes are assumed to be
    oriented horizontally and the image should be cropped, so that it does not contain too much space before and behind
    the GLs. This method also assumed that the growthlanes appear bright foreground on dark background.
    """
    periodicity = get_growthlane_periodicity(growthlane_region_image)
    first_gl_position = get_position_of_first_growthlane(growthlane_region_image, periodicity)
    fft_length = growthlane_region_image.shape[0]
    channel_positions = list(np.arange(first_gl_position, fft_length, periodicity).astype(np.int))
    return channel_positions


def get_all_growthlane_rois(rotated_image, region_list):
    """Gets the growthlane ROIs from all growthlane regions that were found in the image."""
    growthlane_rois = []
    channel_centers = []
    for gl_region in region_list:
        channel_region_image = rotated_image[:, gl_region.start:gl_region.end]
        centers = get_gl_center_positions_in_growthlane_region(channel_region_image)
        rois = get_growthlane_rois(centers, gl_region.start, gl_region.end)
        growthlane_rois += rois
        channel_centers += centers
    return growthlane_rois, channel_centers


def get_mean_distance_between_growthlanes(channel_centers):
    """
    Get the mean distance between two adjacent growth-lanes.

    :param channel_centers:
    :return:
    """
    return np.int0(np.mean(np.diff(channel_centers)))


def find_main_channel_orientation(image):
    """ Find the orientation of the main channel.
    It distinguishes between 0 and 90 degrees, where '0' is in vertical direction
    and '90' is horizontal.

    :param image:
    :return:
    """
    fourier_ratio = calculate_fourier_ratio(image)

    rotated_image = skimage.transform.rotate(image, 90, cval=0)
    fourier_ratio_rotated = calculate_fourier_ratio(rotated_image)

    diff = np.max(fourier_ratio) - np.min(fourier_ratio)
    diff_rotated = np.max(fourier_ratio_rotated) - np.min(fourier_ratio_rotated)

    if diff > diff_rotated:
        return 0
    else:
        return 90


def find_rotation(image):
    """ Find the rotation of the image region containing the GLs.
    The rotation is determined using the 2D spectrum and find the direction along this spectrum, where the sum
    of the spectrum is maximal.

    :param image:
    :return: Returns the angle of the GL ROI-region
    """
    tofft = image
    tofft = np.pad(tofft, ((0, 0), (tofft.shape[0] - tofft.shape[1], 0)), mode='constant', constant_values=0)

    f0 = np.fft.fftshift(np.abs(np.fft.fft2(tofft)))
    allproj = []

    angles = list(np.arange(-10, 10, 0.1))
    for i in angles:
        basicim = skimage.transform.rotate(f0, i, cval=0)

        allproj.append(np.max(np.sum(basicim, axis=0)))

    angle = angles[np.argmax(allproj)]
    return angle


def pattern_limits(image, threshold_factor=None, use_smoothing=False):
    fourier_ratio_orig = calculate_fourier_ratio(image)

    if use_smoothing:
        fourier_ratio = savgol_filter(fourier_ratio_orig, 51, 1)  # window size 51, polynomial order 1
    else:
        fourier_ratio = fourier_ratio_orig

    if threshold_factor is None:
        threshold = threshold_otsu(fourier_ratio)  # use Otsu method to determine threshold value
    else:
        threshold = threshold_factor * fourier_ratio.max()

    region_mask = fourier_ratio > threshold
    region_list = get_regions_from_mask(region_mask)
    region_list = filter_date_regions_by_width(region_list, minimum_region_width=100)

    column_start_position = np.argwhere(fourier_ratio > threshold)[0][0]
    column_end_position = np.argwhere(fourier_ratio > threshold)[-1][0]

    return column_start_position, column_end_position, region_list


def filter_date_regions_by_width(region_list, minimum_region_width):
    """
    Filters removes data regions from region_list with width < minimum_region_width.

    :param region_list: list of DataRegion objects, that we want to filter.
    :param minimum_region_width: minimum width of a region, so that it is not filtered out.
    :return:
    """
    filtered_list = []
    for region in region_list:
        if region.width >= minimum_region_width:
            filtered_list.append(region)
    return filtered_list


def get_regions_from_mask(region_mask):
    """
    Get list of the DataRegion objects that were found inside region_mask.

    :param region_mask: the np.array of 0 and 1 in which we want find regions of value == 1.
    :return: region_list: a list of DataRegion objects.
    """
    region_list = []  # list that will hold all regions that were found
    region = DataRegion()  # object to hold the start and end values of the region
    inside_region = False  # indicates, if we are currently inside a mask_region with value=1
    for index, value in enumerate(region_mask):
        if value == 1 and not inside_region:
            # entered a region
            inside_region = True
            region.start = index
        if value == 0 and inside_region:
            # left a region
            inside_region = False
            region.end = index
            region.width = region.end - region.start
            region_list.append(region)
            region = DataRegion()
    return region_list


def calculate_fourier_ratio(image):
    """Calculates the ratio between highest and second-highest value of the absolute FFT of 'image'
    along the vertical dimension.
    """
    fourier_ratio = []
    for i in range(image.shape[1]):
        fourier_col = np.fft.fftshift(np.abs(np.fft.fft(image[:, i])))
        fourier_col[np.argmax(fourier_col) - 20:np.argmax(fourier_col)] = 0
        fourier_col[np.argmax(fourier_col) + 1:np.argmax(fourier_col) + 20] = 0

        # fourier_col = np.fft.fftshift(np.abs(np.fft.fft(skimage.transform.rotate(image,-5,cval=0)[:,1000])))
        fourier_sort = np.sort(fourier_col)
        fourier_ratio.append(fourier_sort[-2] / fourier_sort[-1])
    fourier_ratio = np.array(fourier_ratio)
    return fourier_ratio


def fft_align(im0, im1, pixlim=None):
    shape = im0.shape
    f0 = np.fft.fft2(im0)
    f1 = np.fft.fft2(im1)
    ir = abs(np.fft.ifft2((f0 * f1.conjugate()) / (np.abs(f0) * np.abs(f1))))

    if pixlim is None:
        t0, t1 = np.unravel_index(np.argmax(ir), shape)
    else:
        shape = ir[0:pixlim, 0:pixlim].shape
        t0, t1 = np.unravel_index(np.argmax(ir[0:pixlim, 0:pixlim]), shape)
    return t0, t1


def get_growthlane_rois(channel_centers, mincol, maxcol):
    """
    Returns ROIs for individual GLs at the positions of `channel_centers`.
    :param channel_centers:
    :param mincol:
    :param maxcol:
    :return:
    """
    growthlaneRois = []
    roi_height = get_mean_distance_between_growthlanes(channel_centers)
    for center in channel_centers:
        roi = get_roi(center, roi_height, mincol, maxcol)
        growthlaneRois.append(GrowthlaneRoi(roi))
    return growthlaneRois


def get_roi(vertical_center_index, height, start_index, stop_index):
    """
    Create a rotated ROI from provided center, width, start- and stop-index
    :param vertical_center_index:
    :param height:
    :param start_index:
    :param stop_index:
    :return:
    """
    roi_half_width = int(height / 2)
    m1 = vertical_center_index - roi_half_width
    m2 = vertical_center_index + roi_half_width
    n1 = start_index
    n2 = stop_index
    return RotatedRoi.create_from_roi(Roi(m1, n1, m2, n2))


def get_translation_matrix(horizontal_shift, vertical_shift):
    return np.float32([[1, 0, horizontal_shift], [0, 1, vertical_shift]])


def get_rotation_matrix(rotation_angle, rotation_center):
    return cv2.getRotationMatrix2D(rotation_center, rotation_angle, 1)
