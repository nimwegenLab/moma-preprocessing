import numpy as np

def saturate_image(image, range_min, range_max):
    image_copy = image.copy()
    min_thresh = np.min(image_copy) * range_min
    max_thresh = np.max(image_copy) * range_max
    image_copy[image_copy < min_thresh] = min_thresh
    image_copy[image_copy > max_thresh] = max_thresh
    return image_copy


def show_gl_index_image(growthlane_rois, full_frame_image):
    import cv2 as cv
    import matplotlib.pyplot as plt

    """ Draw the growthlane ROIs and indices onto the image and save it."""
    font = cv.FONT_HERSHEY_SIMPLEX
    normalized_image = saturate_image(full_frame_image.copy(), 0.1, 0.3)
    normalized_image = cv.normalize(normalized_image, None, 0, 255, cv.NORM_MINMAX)
    final_image = np.array(normalized_image, dtype=np.uint8)

    for roi in growthlane_rois:
        roi.roi.draw_to_image(final_image, False)
        gl_index = roi.id + 1
        cv.putText(final_image, str(gl_index), (np.int0(roi.roi.center[0]), np.int0(roi.roi.center[1])), font, 1,
                   (255, 255, 255), 2, cv.LINE_AA)
    # plt.rcParams['figure.figsize'] = (15, 5)
    plt.imshow(final_image, cmap='gray')
    for roi in growthlane_rois:
        # plt.hlines(roi.roi.center[1], roi.roi.center[0]-roi.roi.width/2, roi.roi.center[0]+roi.roi.width/2, color='r', linestyle='--', linewidth=.5)
        plt.vlines(roi.roi.center[0], roi.roi.center[1]-roi.roi.width/2, roi.roi.center[1]+roi.roi.width/2, color='r', linestyle='--', linewidth=.5)
    plt.show()
