import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def show_image(image):
    """ Draw the growthlane ROIs onto the image for control purposes. """
    fig, ax = plt.subplots(1)
    ax.imshow(image)

    plt.show()

def show_image_with_rois(image, growthlaneRoiList):
    """ Draw the growthlane ROIs onto the image for control purposes. """
    normalizedImg = None
    normalizedImg = cv.normalize(image,  normalizedImg, 0, 255, cv.NORM_MINMAX)
    im = np.array(normalizedImg, dtype=np.uint8)

    fig, ax = plt.subplots(1)
    ax.imshow(im)

    for growthlane in growthlaneRoiList:
        roi = growthlane.roi
        rect = patches.Rectangle((roi[0][1], roi[0][0]), roi[1][0], roi[1][1], linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    plt.show()
