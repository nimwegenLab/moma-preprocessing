
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
        self.m1 = m1  # vertical starting position in matrix-indices convention
        self.n1 = n1  # horizontal starting position in matrix-indices convention
        self.m2 = m2  # vertical end position in matrix-indices convention
        self.n2 = n2  # horizontal end position in matrix-indices convention

    def get_from_image(self, image):
        return image[:, self.m1:self.m2, self.n1:self.n2]

    def is_subset_of(self, other_roi):
        """ Returns true if this ROI is a subset of 'other_roi'. False otherwise.

        :param other_roi:
        :return:
        """
        if self.m1 >= other_roi.m1 and self.n1 >= other_roi.n1 and self.m2 <= other_roi.m2 and self.n2 <= other_roi.n2:
            return True
        else:
            return False

    def translate(self, shift_x_y):
        if not isinstance(shift_x_y[0], int) or not isinstance(shift_x_y[1], int):
            raise ValueError("shift_x_y must be integer.")
        self.n1 += shift_x_y[0]
        self.n2 += shift_x_y[0]
        self.m1 += shift_x_y[1]
        self.m2 += shift_x_y[1]

    @property
    def width(self):
        return self.n2 - self.n1

    @width.setter
    def width(self, new_width):
        self.n2 = self.n1 + new_width

    @property
    def height(self):
        return self.m2 - self.m1

    @height.setter
    def height(self, new_height):
        self.m2 = self.m1 + new_height
