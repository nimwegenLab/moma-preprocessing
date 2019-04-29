
class Roi(object):
    """Defines a ROI in an image."""
    def __init__(self, m1, n1, m2, n2):
        if m1 > m2:
            raise ValueError("Invalid ROI bounds: m1 > m2")
        if m1 == m2:
            raise ValueError("Invalid ROI bounds: m1 == m2")
        if n1 > n2:
            raise ValueError("Invalid ROI bounds: n1 > n2")
        if n1 == n2:
            raise ValueError("Invalid ROI bounds: n1 == n2")
        self.m1 = m1
        self.n1 = n1
        self.m2 = m2
        self.n2 = n2

    def is_subset_of(self, other_roi):
        """ Returns true if this ROI is a subset of 'other_roi'. False otherwise.

        :param other_roi:
        :return:
        """
        if self.m1 >= other_roi.m1 and self.n1 >= other_roi.n1 and self.m2 <= other_roi.m2 and self.n2 <= other_roi.n2:
            return True
        else:
            return False

    @property
    def width(self):
        return self.n2 - self.n1

    @property
    def height(self):
        return self.m2 - self.m1
