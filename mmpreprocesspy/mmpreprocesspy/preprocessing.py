import cv2
# import mmpreprocesspy.dev_auxiliary_functions as aux
import numpy as np
import skimage.transform
from mmpreprocesspy.GrowthlaneRoi import GrowthlaneRoi
from mmpreprocesspy.data_region import DataRegion
from mmpreprocesspy.roi import Roi
from mmpreprocesspy.rotated_roi import RotatedRoi
from scipy.signal import savgol_filter, find_peaks
from skimage.filters import threshold_otsu, gaussian
import matplotlib.pyplot as plt


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
    look_ahead_length = 50  # the distance that the algorithm will look ahead to see if the threshold is passed
    projected_max_intensities = np.max(rotated_image, axis=0)
    sorted_max_intensities = np.sort(projected_max_intensities)
    threshold = np.median(sorted_max_intensities)

    #  extend region at the channel start using lookahead interval
    while np.any(projected_max_intensities[region.start-look_ahead_length:region.start] > threshold):
        region.start -= 1
    #  extend region at the channel end using lookahead interval
    while np.any(projected_max_intensities[region.end:region.end+look_ahead_length] > threshold):
        region.end += 1

    # normalizedImg = None
    # normalizedImg = cv.normalize(rotated_image,  normalizedImg, 0, 255, cv.NORM_MINMAX)
    # im = np.array(normalizedImg, dtype=np.uint8)
    # cv.line(im, (region.start, 0), (region.start, im.shape[0]), (255, 0, 0), 5)
    # cv.line(im, (region.end, 0), (region.end, im.shape[0]), (255, 0, 0), 5)
    # aux.show_image(im, 'rotated image')
    # cv.waitKey()
    #
    # #######################
    #
    # plt.hist(projected_max_intensities, 50), plt.show()
    # plt.plot(projected_max_intensities), plt.hlines(threshold, region.start, region.end), plt.show()
    # plt.plot(projected_max_intensities[region.start:region.end]), plt.hlines(threshold, region.start, region.end), plt.show()
    # plt.plot(projected_max_intensities[region.start-look_ahead_length:region.start]), plt.hlines(threshold, region.start, region.end), plt.show()
    # plt.plot(projected_max_intensities[region.end:region.end+look_ahead_length]), plt.hlines(threshold, region.start, region.end), plt.show()


def get_channel_periodicity(channel_region_image):
    projected_image_intensity = np.sum(channel_region_image, axis=1)
    projected_image_intensity_zero_mean = projected_image_intensity - np.mean(projected_image_intensity)
    cross_corr = np.correlate(projected_image_intensity_zero_mean, projected_image_intensity_zero_mean, 'same')

    peak_inds = find_peaks(cross_corr, distance=10)[0]
    peak_vals = cross_corr[peak_inds]

    peak_inds_sorted = [x for _, x in sorted(zip(peak_vals, peak_inds), key=lambda pair: pair[0])]
    periodicity = np.mean(np.diff(sorted(peak_inds_sorted[-6:])))
    return periodicity

def get_index_of_maximum_closest_to_position(fnc, position):
    peak_inds = find_peaks(fnc, distance=10)[0]
    peak_vals = fnc[peak_inds]
    # keep only positive maxima
    peak_inds = peak_inds[peak_vals > 0]
    return peak_inds[np.argmin(np.abs(peak_inds-position))]

def get_position_of_first_growthlane(channel_region_image, periodicity):
    projected_image_intensity = np.sum(channel_region_image, axis=1)
    projected_image_intensity_zero_mean = projected_image_intensity - np.mean(projected_image_intensity)
    acf_of_intensity_profile = np.correlate(projected_image_intensity_zero_mean, projected_image_intensity_zero_mean, 'same')
    ccf_of_acf_with_intensity_profile = np.correlate(projected_image_intensity_zero_mean, acf_of_intensity_profile, 'same')
    center_index = np.round(projected_image_intensity.shape[0]/2)
    index = get_index_of_maximum_closest_to_position(ccf_of_acf_with_intensity_profile, center_index)
    shift = index - periodicity * np.floor(index/periodicity)  # get growthlane position closest to the image origin
    return shift

def find_channels_in_region_new(channel_region_image):
    periodicity = get_channel_periodicity(channel_region_image)
    shift = get_position_of_first_growthlane(channel_region_image, periodicity)
    fft_length = channel_region_image.shape[0]
    channel_positions = get_channel_positions(periodicity, shift, fft_length)
    return channel_positions

def get_channel_positions(periodicity, shift, fft_size):
    starting_value = shift
    return list(np.arange(starting_value, fft_size, periodicity).astype(np.int))

def get_all_growthlane_rois(rotated_image, region_list):
    """Gets the growthlane ROIs from all growthlane regions that were found in the image."""
    growthlane_rois = []
    channel_centers = []
    for gl_region in region_list:
        channel_region_image = rotated_image[:, gl_region.start:gl_region.end]
        centers = find_channels_in_region_new(channel_region_image)
        # centers = find_channels_in_region(channel_region_image)
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

    # plt.plot(fourier_ratio_orig)
    # plt.plot(fourier_ratio)
    # plt.plot(region_mask * np.max(fourier_ratio))
    # plt.show()

    # # compare smoothed vs. non-smoothed data
    # yhat = savgol_filter(fourier_ratio, 51, 1)  # window size 31, polynomial order 3
    # plt.plot(fourier_ratio)
    # plt.plot(yhat)
    # plt.hlines(threshold,0,fourier_ratio.__len__())
    # plt.show()
    #
    # # look at histogram
    # plt.hist(yhat,100)
    # plt.show()
    #
    # # look at differential values
    # plt.plot(np.diff(yhat))
    # plt.show()
    #
    # # try box-smoothing assuming, we know the length of the growthlanes
    # filt_sig = filters.uniform_filter(fourier_ratio_orig, size=400, output=None, mode='reflect', cval=0.0, origin=0)
    # plt.plot(fourier_ratio_orig)
    # plt.plot(filt_sig)
    # plt.show()


    # threshold_factor = threshold_otsu(yhat)
    # print(threshold_factor)

    mincol = np.argwhere(fourier_ratio > threshold)[0][0]
    maxcol = np.argwhere(fourier_ratio > threshold)[-1][0]

    return mincol, maxcol, region_list


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


def find_channels_in_region(channel_region_image):
    # find channels as peak of intensity in a projection
    # define a threshold between inter-channel and peak intensity.
    # For each chunk of rows corresponding to a channel, calculate a mean position as mid-channel

    projected_image_intensity = np.sum(channel_region_image, axis=1)
    inter_channel_val = np.mean(np.sort(projected_image_intensity)[0:100])

    window = 30
    peaks = np.array([x for x in np.arange(window, len(projected_image_intensity) - window)
                      if np.all(projected_image_intensity[x] > projected_image_intensity[x - window:x]) & np.all(
            projected_image_intensity[x] > projected_image_intensity[x + 1:x + window])])

    peaks = peaks[projected_image_intensity[peaks] > 1.5 * inter_channel_val]

    channel_val = np.mean(projected_image_intensity[peaks])
    # mid_range = 0.5*(inter_channel_val+channel_val)
    mid_range = inter_channel_val + 0.3 * (channel_val - inter_channel_val)

    chunks = np.concatenate(np.argwhere(projected_image_intensity > mid_range))

    channel_center = []
    initchunk = [chunks[0]]
    for x in range(1, len(chunks)):
        if chunks[x] - chunks[x - 1] == 1:
            initchunk.append(chunks[x])
        else:
            channel_center.append(np.mean(initchunk))
            initchunk = [chunks[x]]
    channel_center = channel_center
    return channel_center


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

