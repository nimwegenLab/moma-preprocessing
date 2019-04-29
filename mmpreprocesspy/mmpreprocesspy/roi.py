
class Roi(object):
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

