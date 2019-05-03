class GrowthlaneRoi(object):
    """ Represents the growth-lane present inside a Mother-machine image. """

    def __init__(self, roi=None, index=None):
        self.roi = roi
        self.index = index

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
