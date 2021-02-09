from enum import Enum
import numpy as np

class GrowthlaneRoi(object):
    """ Represents the growth-lane present inside a Mother-machine image. """

    def __init__(self, roi=None, id=None, parent_gl_region_id=None):
        self.roi = roi
        self.id = id
        self.parent_gl_region_id = parent_gl_region_id
        self.exit_location: GrowthlaneExitLocation = None

    def get_oriented_roi_image(self, image):
        roi_image = self.roi.get_from_image(image)
        if self.exit_location is None:
            self.exit_location = self.determine_location_of_growthlane_exit(roi_image, search_area=50)

        if self.exit_location is GrowthlaneExitLocation.AT_TOP:
            return roi_image
        elif self.exit_location is GrowthlaneExitLocation.AT_LEFT:
            return np.rot90(roi_image,-1)
        elif self.exit_location is GrowthlaneExitLocation.AT_BOTTOM:
            return np.rot90(roi_image, 2)
        elif self.exit_location is GrowthlaneExitLocation.AT_RIGHT:
            return np.rot90(roi_image, 1)
        else:
            raise ValueError("Growthlane orientation is not set.")

    def determine_location_of_growthlane_exit(self, growthlane_roi_image, search_area):
        """
        This function determines the location of the growthlane by comparing the value sum of values
        at the start of the *extended* GL to those at the end.
        :param growthlane_roi_image: the image of from the extended GL ROI.
        :param search_area: the value by which the GL was extended in *both* directions.
        :return:
        """
        height, width = growthlane_roi_image.shape

        if width > height:
            sum_at_left = np.sum(growthlane_roi_image[:, 0:search_area].flatten(), 0)
            sum_at_right = np.sum(growthlane_roi_image[:, -search_area:].flatten(), 0)
            if sum_at_left > sum_at_right:
                return GrowthlaneExitLocation.AT_LEFT
            elif sum_at_right > sum_at_left:
                return GrowthlaneExitLocation.AT_RIGHT
            else:
                raise ValueError("Could not determine location of growthlane exit.")
        elif height > width:
            sum_at_top = np.sum(growthlane_roi_image[0:search_area, :].flatten(), 0)
            sum_at_bottom = np.sum(growthlane_roi_image[-search_area:, :].flatten(), 0)
            if sum_at_top > sum_at_bottom:
                return GrowthlaneExitLocation.AT_TOP
            elif sum_at_bottom > sum_at_top:
                return GrowthlaneExitLocation.AT_BOTTOM
            else:
                raise ValueError("Could not determine location of growthlane exit.")
        else:
            raise ValueError("Could not determine location of growthlane exit.")

        print("stop")

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
    AT_TOP = 2
    AT_BOTTOM = 3
