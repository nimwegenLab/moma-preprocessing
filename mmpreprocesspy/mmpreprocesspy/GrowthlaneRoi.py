import cv2 as cv


class GrowthlaneRoi(object):
    """ Represents the growth-lane present inside a Mother-machine image. """
    def __init__(self, roi = None):
        self.roi = roi
