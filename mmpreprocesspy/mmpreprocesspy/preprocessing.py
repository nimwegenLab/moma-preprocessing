import numpy as np
import skimage.transform
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# find rotation, channel boundaries and positions for first image that is then used as reference
def split_channels_init(image):
    # find the boundary region containing channels by finding columns with repetitive pattern
    mincol, maxcol = pattern_limits(image, use_smoothing=True)
    # find rotation angle
    angle = find_rotation(image[:, mincol:maxcol])

    # recalculate channel region boundary on rotated image
    image_rot = skimage.transform.rotate(image, angle, cval=0)
    mincol, maxcol = pattern_limits(image_rot, use_smoothing=True)

    channel_centers = find_channels(image_rot, mincol, maxcol)
    return image_rot, angle, mincol, maxcol, channel_centers


def find_rotation(image):
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
    fourier_ratio = []
    for i in range(image.shape[1]):
        fourier_col = np.fft.fftshift(np.abs(np.fft.fft(image[:, i])))
        fourier_col[np.argmax(fourier_col) - 20:np.argmax(fourier_col)] = 0
        fourier_col[np.argmax(fourier_col) + 1:np.argmax(fourier_col) + 20] = 0

        # fourier_col = np.fft.fftshift(np.abs(np.fft.fft(skimage.transform.rotate(image,-5,cval=0)[:,1000])))
        fourier_sort = np.sort(fourier_col)
        fourier_ratio.append(fourier_sort[-2] / fourier_sort[-1])
    fourier_ratio = np.array(fourier_ratio)

    if use_smoothing:
        fourier_ratio = savgol_filter(fourier_ratio, 31, 3)  # window size 51, polynomial order 3

    if threshold_factor is None:
        threshold = threshold_otsu(fourier_ratio) # use Otsu method to determine threshold value
    else:
        threshold = threshold_factor * fourier_ratio.max()


    # yhat = savgol_filter(fourier_ratio, 31, 3)  # window size 31, polynomial order 3
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
