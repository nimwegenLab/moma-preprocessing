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
def get_rotated_image(image, main_channel_angle=None):
    if main_channel_angle == None:
        main_channel_angle = find_main_channel_orientation(image)

    if main_channel_angle != 0:
        image = skimage.transform.rotate(image, main_channel_angle,
                                         resize=True)  # rotate image angle back to 0, if needed

    # find rotation angle
    angle = find_rotation(image)
    main_channel_angle += angle

    # recalculate channel region boundary on rotated image
    image_rotated = skimage.transform.rotate(image, angle, cval=0)

    return image_rotated, main_channel_angle

def get_gl_regions(image_rotated, growthlane_length_threshold=0, roi_boundary_offset_at_mother_cell=0):
    region_list = find_channel_regions(image_rotated, use_smoothing=True,
                                       minimum_required_growthlane_length=growthlane_length_threshold,
                                       roi_boundary_offset_at_mother_cell=roi_boundary_offset_at_mother_cell)
    growthlane_rois, channel_centers = get_all_growthlane_rois(image_rotated, region_list)
    growthlane_rois = rotate_rois(image, growthlane_rois, main_channel_angle)
    growthlane_rois = remove_rois_not_fully_in_image(image, growthlane_rois)
    return channel_centers, growthlane_rois

def rotate_rois(image, growthlane_rois, main_channel_angle):
    rotation_center = (np.int0(image.shape[1]/2), np.int0(image.shape[0]/2))
    for growthlane_roi in growthlane_rois:
        growthlane_roi.roi.rotate(rotation_center, -main_channel_angle)
    return growthlane_rois


def remove_rois_not_fully_in_image(image, growthlane_rois):
    """
    This method removes ROI that do not lie fully inside the image.
    :return:
    """
    inds = list(range(len(growthlane_rois)))
    inds.reverse()
    for ind in inds:
        gl_roi = growthlane_rois[ind]
        if not gl_roi.roi.is_inside_image(image):
            del growthlane_rois[ind]
    return growthlane_rois


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


def refine_regions(rotated_image, region_list, roi_boundary_offset_at_mother_cell=0):
    for region in region_list:
        refine_region(rotated_image, region, roi_boundary_offset_at_mother_cell=roi_boundary_offset_at_mother_cell)


def refine_region(rotated_image, region, roi_boundary_offset_at_mother_cell=0):
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
    look_ahead_length = 30  # the distance that the algorithm will look ahead to see if the threshold is passed
    projected_max_intensities = np.max(rotated_image, axis=0)
    threshold = 0.8 * np.median(projected_max_intensities[region.start:region.end])

    #  extend region at the channel start using lookahead interval
    while np.any(projected_max_intensities[region.start - look_ahead_length:region.start] > threshold):
        region.start -= 1
    #  extend region at the channel end using lookahead interval
    while np.any(projected_max_intensities[region.end:region.end + look_ahead_length] > threshold):
        region.end += 1

    region = extend_region_at_mothercell(rotated_image, region, roi_boundary_offset_at_mother_cell=roi_boundary_offset_at_mother_cell)


def extend_region_at_mothercell(rotated_image, region, roi_boundary_offset_at_mother_cell = 0, search_interval_length = 80):
    """
    This method figures out at what side of the region-interval the mother-cell is located.
    It then extends the region boundary in that direction by.

    :param rotated_image:
    :param region: region that will be extended
    :param roi_boundary_offset_at_mother_cell: the amount of pixels by which the region will be extended
    :param search_interval_length: length in pixel of the inteval for which we compare the summed intensities; in our images, the ROI image is brighter at the exit of the GL
    :return: the extended region
    """

    projected_max_intensities = np.max(rotated_image, axis=0)
    sum_at_start = np.sum(projected_max_intensities[region.start: region.start + search_interval_length])
    sum_at_end = np.sum(projected_max_intensities[region.end - search_interval_length: region.end])

    if sum_at_start > sum_at_end:  # mother cell is located at region.end, so we extend there
        region.end = region.end + roi_boundary_offset_at_mother_cell
    elif sum_at_start < sum_at_end:  # mother cell is located at region.start, so we extend there
        region.start = region.start - roi_boundary_offset_at_mother_cell

    return region


def get_growthlane_periodicity(growthlane_region_image):
    from scipy.ndimage.filters import uniform_filter1d
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
    auto_correlation = np.correlate(projected_image_intensity_zero_mean, projected_image_intensity_zero_mean, 'same')
    auto_correlation = auto_correlation / np.max(
        auto_correlation)  # normalize the auto-correlation function, so that we can use an absolut prominence of 0.5

    peak_inds = find_peaks(auto_correlation, prominence=0.5, distance=min_distance_between_peaks)[0]
    peak_vals = auto_correlation[peak_inds]

    peak_ind = np.argmax(peak_vals)
    start_ind = peak_ind - int(number_of_peaks / 2)
    end_ind = peak_ind + int(number_of_peaks / 2) + 1
    selected_peak_inds = peak_inds[start_ind:end_ind]  # get `number_of_peaks` around the center peak with shift=0 and value=1
    periodicity = np.mean(np.diff(selected_peak_inds))

    return periodicity


def get_index_of_intensity_maximum_closest_to_position(ccf_profile_orig, position):
    """
    Returns the index of the intensity maximum closest to :position:.

    :ccf_profile: intensity profile from which to get the closes maximum.
    :position: position to which the maximum should be closest.
    """
    ccf_profile = (ccf_profile_orig - np.amin(ccf_profile_orig)) / (np.amax(ccf_profile_orig) - np.amin(ccf_profile_orig))
    peak_inds = find_peaks(ccf_profile, distance=10)[0]
    peak_vals = ccf_profile[peak_inds]
    peak_inds = peak_inds[peak_vals > 0.5]  # keep only highest maxima, because they correspond to the growthlane centers
    return peak_inds[np.argmin(np.abs(peak_inds - position))]


def get_position_of_first_growthlane(growthlane_region_image, periodicity):
    """
    Returns the position of the first growthlane in :growthlane_region_image: closest to the image origin.

    :param growthlane_region_image: region of the image containing growthlanes. The growthlanes are assumed to be
    oriented horizontally and the image should be cropped, so that it does not contain too much space before and behind
    the GLs. This method also assumed that the growthlanes appear bright foreground on dark background.
   """
    growthlane_region_image_orig = growthlane_region_image
    growthlane_region_image = skimage.filters.gaussian(growthlane_region_image_orig, sigma=20)
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
        growthlane_rois += get_growthlane_rois(centers, gl_region.start, gl_region.end)
        channel_centers += centers
    growthlane_rois = fix_roi_ids(growthlane_rois)
    return growthlane_rois, channel_centers


def fix_roi_ids(rois):
    """
    This function fixes the indexing of the ROIs to have a single continuously increasing index.
    This is necessary because in the function get_growthlane_rois assigns the IDs per vertical growthlane-column.
    This means that for two-sided MMs we get the same IDs twice (i.e. 0, 1, 2, ...) for each side. But what we
    want is to have a monotonously increasing ID-value.

    :param rois:
    :return:
    """
    for id, roi in enumerate(rois):  # we can simple enumarte contiguously here, because the ROIs are sorted per-column/side of the MM main-channel
        roi.id = id
    return rois


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


def find_channel_regions(image, threshold_factor=None, use_smoothing=False, minimum_required_growthlane_length=0,
                         roi_boundary_offset_at_mother_cell=0):
    region_mask = find_plateaus(image)
    region_list = get_regions_from_mask(region_mask)
    region_list = filter_regions(region_list, minimum_required_growthlane_length=minimum_required_growthlane_length)
    refine_regions(image, region_list, roi_boundary_offset_at_mother_cell=roi_boundary_offset_at_mother_cell)
    return region_list


def find_plateaus(image, buffer_size = 50, treshold = 0.2):
    """
    This function scans for plateaus in the projected intensity profile along the main-channel.
    It does this by calculating the coefficient of variation for a moving window of size :buffer_size:.
    If the coefficient of variation is smaller than :treshold: for a given location, then that location
    is considered to be part of a plateau.

    :param image:
    :return:
    """

    image = image - np.mean(image, axis=0)
    projected_max_intensities = np.max(image, axis=0)
    projected_max_intensities_orig = projected_max_intensities
    projected_max_intensities = projected_max_intensities - np.min(projected_max_intensities)

    output_size = len(projected_max_intensities) - buffer_size
    stds = np.zeros(output_size)
    output = np.zeros(output_size)
    for ind in range(output_size):
        buffer = projected_max_intensities[ind:ind+buffer_size]
        output[ind] = np.std(buffer)/np.mean(buffer)

    return output < treshold


def is_debugging():
    try:
        import pydevd
        return True
    except ImportError:
        return False

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
    for id, center in enumerate(channel_centers):
        roi = get_roi(center, roi_height, mincol, maxcol)
        growthlaneRois.append(GrowthlaneRoi(roi, id))
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
