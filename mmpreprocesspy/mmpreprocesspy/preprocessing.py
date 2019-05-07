import numpy as np
import skimage.transform
from mmpreprocesspy.GrowthlaneRoi import GrowthlaneRoi
from mmpreprocesspy.roi import Roi
from mmpreprocesspy.rotated_roi import RotatedRoi
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import cv2 as cv


# find rotation, channel boundaries and positions for first image that is then used as reference
def process_image(image):
    main_channel_angle = find_main_channel_orientation(image)
    if main_channel_angle != 0:
        image = skimage.transform.rotate(image, -main_channel_angle,
                                         resize=True)  # rotate image angle back to 0, if needed

    # find the boundary region containing channels by finding columns with repetitive pattern
    mincol, maxcol = pattern_limits(image, use_smoothing=True)
    # find rotation angle
    angle = find_rotation(image[:, mincol:maxcol])
    main_channel_angle += angle

    # recalculate channel region boundary on rotated image
    image_rot = skimage.transform.rotate(image, angle, cval=0)
    mincol, maxcol = pattern_limits(image_rot, use_smoothing=True)

    channel_centers = find_channels(image_rot, mincol, maxcol)
    return image_rot, main_channel_angle, mincol, maxcol, channel_centers


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

    for i in np.arange(-10, 10, 1):
        basicim = skimage.transform.rotate(f0, i, cval=0)

        allproj.append(np.max(np.sum(basicim, axis=0)))

    angle = np.arange(-10, 10, 1)[np.argmax(allproj)]
    return angle


def pattern_limits(image, threshold_factor=None, use_smoothing=False):
    fourier_ratio = calculate_fourier_ratio(image)

    if use_smoothing:
        fourier_ratio = savgol_filter(fourier_ratio, 51, 1)  # window size 51, polynomial order 1

    if threshold_factor is None:
        threshold = threshold_otsu(fourier_ratio)  # use Otsu method to determine threshold value
    else:
        threshold = threshold_factor * fourier_ratio.max()

    # yhat = savgol_filter(fourier_ratio, 51, 1)  # window size 31, polynomial order 3
    # plt.plot(fourier_ratio)
    # plt.plot(yhat)
    # plt.show()
    #
    #
    # plt.hist(yhat)
    # plt.show()
    # threshold_factor = threshold_otsu(yhat)
    # print(threshold_factor)

    mincol = np.argwhere(fourier_ratio > threshold)[0][0]
    maxcol = np.argwhere(fourier_ratio > threshold)[-1][0]

    return mincol, maxcol


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


def find_channels(image, mincol, maxcol, window=30):
    # find channels as peak of intensity in a projection
    # define a threshold between inter-channel and peak intensity.
    # For each chunk of rows corresponding to a channel, calculate a mean position as mid-channel

    channel_proj = np.sum(image[:, mincol:maxcol], axis=1)
    inter_channel_val = np.mean(np.sort(channel_proj)[0:100])

    window = 30
    peaks = np.array([x for x in np.arange(window, len(channel_proj) - window)
                      if np.all(channel_proj[x] > channel_proj[x - window:x]) & np.all(
            channel_proj[x] > channel_proj[x + 1:x + window])])

    peaks = peaks[channel_proj[peaks] > 1.5 * inter_channel_val]

    channel_val = np.mean(channel_proj[peaks])
    # mid_range = 0.5*(inter_channel_val+channel_val)
    mid_range = inter_channel_val + 0.3 * (channel_val - inter_channel_val)

    chunks = np.concatenate(np.argwhere(channel_proj > mid_range))

    channel_center = []
    initchunk = [chunks[0]]
    for x in range(1, len(chunks)):
        if chunks[x] - chunks[x - 1] == 1:
            initchunk.append(chunks[x])
        else:
            channel_center.append(np.mean(initchunk))
            initchunk = [chunks[x]]
    channel_center = np.array(channel_center)
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


def get_growthlane_regions(channel_centers, mincol, maxcol):
    growthlaneRois = []
    for index, center in enumerate(channel_centers):
        roi = get_roi(center, mincol, maxcol)
        growthlaneRois.append(GrowthlaneRoi(roi, index))
    return growthlaneRois


def get_roi(center, mincol, maxcol):
    channel_width = 100  # TODO-MM-2019-04-23: This will need to be determined dynamically or made configurable.
    half_width = int(channel_width / 2)
    m1 = center - half_width
    m2 = center + half_width
    n1 = mincol
    n2 = maxcol
    return RotatedRoi.create_from_roi(Roi(m1, n1, m2, n2))


def get_image_registration_template(image, mincol):
    # find regions with large local derivatives in BOTH directions, which should be "number-regions".
    # keep the one the most in the middle
    # take a sub-region around that point as a template on which to do template matching
    large_feature = skimage.filters.gaussian(
        np.abs(skimage.filters.scharr_h(image) * skimage.filters.scharr_v(image)), sigma=5)
    mask = np.zeros(image.shape)
    mask[large_feature > 0.5 * np.max(large_feature)] = 1
    mask_lab = skimage.measure.label(mask)
    mask_reg = skimage.measure.regionprops(mask_lab)
    middle_num_pos = mask_reg[
        np.argmin([np.linalg.norm(np.array(x.centroid) - np.array(image.shape) / 2) for x in mask_reg])].centroid
    mid_row = np.int(middle_num_pos[0])

    hor_space = int(mincol) + 100
    hor_mid = int(hor_space / 2)
    hor_width = int(0.3 * hor_space)

    template = image[mid_row - 100:mid_row + 100, 0:hor_space]
    return template, mid_row, hor_mid, hor_width


def get_translation_matrix(horizontal_shift, vertical_shift):
    return np.float32([[1, 0, horizontal_shift], [0, 1, vertical_shift]])


def get_rotation_matrix(rotation_angle, rotation_center):
    return cv.getRotationMatrix2D(rotation_center, rotation_angle, 1)

