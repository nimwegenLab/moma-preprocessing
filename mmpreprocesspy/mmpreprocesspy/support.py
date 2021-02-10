import numpy as np

def saturate_image(image, range_min, range_max):
    image_copy = image.copy()
    min_thresh = np.min(image_copy) * range_min
    max_thresh = np.max(image_copy) * range_max
    image_copy[image_copy < min_thresh] = min_thresh
    image_copy[image_copy > max_thresh] = max_thresh
    return image_copy

