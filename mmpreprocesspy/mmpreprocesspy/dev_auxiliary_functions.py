import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def show_image(image, window_name=None):
    """ Draw the growthlane ROIs onto the image for control purposes. """

    if not window_name:
        window_name = ""

    normalizedImg = None
    normalizedImg = cv.normalize(image,  normalizedImg, 0, 255, cv.NORM_MINMAX)
    im = np.array(normalizedImg, dtype=np.uint8)

    cv.namedWindow(window_name, cv.WINDOW_NORMAL)
    cv.imshow(window_name, im)
    cv.resizeWindow(window_name, 600, 600)

def show_image_with_rois(image, growthlaneRoiList):
    """ Draw the growthlane ROIs onto the image for control purposes. """
    normalizedImg = None
    normalizedImg = cv.normalize(image,  normalizedImg, 0, 255, cv.NORM_MINMAX)
    im = np.array(normalizedImg, dtype=np.uint8)

    fig, ax = plt.subplots(1)
    ax.imshow(im)

    for growthlane in growthlaneRoiList:
        roi = growthlane.roi
        rect = patches.Rectangle((roi.n1, roi.m1), roi.width, roi.height, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    plt.show()

def show_image_with_rotated_rois(image, rotated_rois):
    """ Draw the growthlane ROIs onto the image for control purposes. """
    """ Draw the growthlane ROIs onto the image for control purposes. """
    normalizedImg = None
    normalizedImg = cv.normalize(image,  normalizedImg, 0, 255, cv.NORM_MINMAX)
    im = np.array(normalizedImg, dtype=np.uint8)

    for roi in rotated_rois:
        roi.draw_to_image(im, True)

    cv.namedWindow("image with roi", cv.WINDOW_NORMAL)
    cv.imshow("image with roi", im)
    cv.resizeWindow("image with roi", 600, 600)
