from enum import Enum
import numpy as np

class GrowthlaneRoi(object):
    """ Represents the growth-lane present inside a Mother-machine image. """

    def __init__(self, roi=None, id=None):
        self.roi = roi
        self.id = id
        self.exit_location: GrowthlaneExitLocation = None

    def get_oriented_roi_image(self, image):
        roi_image = self.roi.get_from_image(image)
        if self.exit_location is GrowthlaneExitLocation.AT_RIGHT:
            return np.flipud(roi_image.T)
        if self.exit_location is GrowthlaneExitLocation.AT_LEFT:
            return roi_image.T
        else:
            raise ValueError("Growthlane orientation is not set.")

    @property
    def length(self):
        """
        Returns the length of the growthlane.
        """
        if self.roi.width >= self.roi.height:
            return self.roi.width
        else:
            return self.roi.height

    @property
    def width(self):
        """
        Returns the width of the channel ROI.
        Note that this is not identical with RotatedRoi.width (but it can be).
        """
        if self.roi.height < self.roi.width:
            return self.roi.height
        else:
            return self.roi.width


class GrowthlaneExitLocation(Enum):
    """Enum for passing back the result from determining the location of the growthlane exit."""
    AT_LEFT = 0
    AT_RIGHT = 1

