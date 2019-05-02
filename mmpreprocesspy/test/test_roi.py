from unittest import TestCase
import cv2 as cv
import numpy as np

class TestRoi(TestCase):
    def test__pointPolygonTest(self):
        x = 0
        y = 0
        width = 100
        height = 100
        angle = 0
        # rect = ((x, y), (x + width, y + height), angle)
        x = 50
        y = 50
        rect = ((x, y), (width, height), angle)
        contour = cv.boxPoints(rect)
        result = cv.pointPolygonTest(contour, (0, 0), measureDist=False)
        print(result)

    def test__rectangle_rotation(self):
        x = 0
        y = 0
        width = 100
        height = 100
        angle = -45
        rot_rect = ((x, y), (width, height), angle)
        box = cv.boxPoints(rot_rect)
        pass

    def test__shift__for_vertical_shift__updates_bounds_correctly(self):
        from mmpreprocesspy.roi import Roi
        sut = Roi(0, 0, 100, 100)

        sut.shift(0, 100)

        self.assertEqual(100, sut.m1)
        self.assertEqual(200, sut.m2)

    def test__shift__for_horizontal_shift__updates_bounds_correctly(self):
        from mmpreprocesspy.roi import Roi
        sut = Roi(0, 0, 100, 100)

        sut.shift(100, 0)

        self.assertEqual(100, sut.n1)
        self.assertEqual(200, sut.n2)

    def test__set_height__for_updated_height__correctly_updates_second_bound(self):
        from mmpreprocesspy.roi import Roi
        sut = Roi(0, 0, 100, 100)

        sut.height = 200

        self.assertEqual(200, sut.m2)

    def test__get_height__returns_correct_value(self):
        from mmpreprocesspy.roi import Roi
        sut = Roi(1, 1, 100, 100)
        self.assertEqual(99, sut.height)

    def test__set_width__for_updated_width__correctly_updates_second_bound(self):
        from mmpreprocesspy.roi import Roi
        sut = Roi(0, 0, 100, 100)

        sut.width = 200

        self.assertEqual(200, sut.n2)

    def test__get_width__returns_correct_value(self):
        from mmpreprocesspy.roi import Roi
        sut = Roi(1, 1, 100, 100)
        self.assertEqual(99, sut.width)

    def test__is_subset_of__for_simple_superset__returns_True(self):
        from mmpreprocesspy.roi import Roi
        superset = Roi(1, 1, 100, 100)
        sut = Roi(1, 1, 100, 100)
        self.assertTrue(sut.is_subset_of(superset))

    def test__is_subset_of__for__true_superset__returns_True(self):
        from mmpreprocesspy.roi import Roi
        superset = Roi(1, 1, 100, 100)
        sut = Roi(2, 2, 99, 99)
        self.assertTrue(sut.is_subset_of(superset))

    def test__init__for__equal_n_bounds__raises__ValueError(self):
        from mmpreprocesspy.roi import Roi
        with self.assertRaises(ValueError):
            Roi(1, 2, 3, 2)

    def test__init__for__invalid_n_bounds__raises__ValueError(self):
        from mmpreprocesspy.roi import Roi
        with self.assertRaises(ValueError):
            Roi(1, 4, 3, 2)

    def test__init__for__equal_m_bounds__raises__ValueError(self):
        from mmpreprocesspy.roi import Roi
        with self.assertRaises(ValueError):
            Roi(1, 2, 1, 4)

    def test__init__for__invalid_m_bounds__raises__ValueError(self):
        from mmpreprocesspy.roi import Roi
        with self.assertRaises(ValueError):
            Roi(3, 2, 1, 4)

    def test_roi_initializes_n2_correctly(self):
        from mmpreprocesspy.roi import Roi
        sut = Roi(1, 2, 3, 4)
        self.assertEqual(sut.n2, 4)

    def test_roi_initializes_m2_correctly(self):
        from mmpreprocesspy.roi import Roi
        sut = Roi(1, 2, 3, 4)
        self.assertEqual(sut.m2, 3)

    def test_roi_initializes_n1_correctly(self):
        from mmpreprocesspy.roi import Roi
        sut = Roi(1, 2, 3, 4)
        self.assertEqual(sut.n1, 2)

    def test_roi_initializes_m1_correctly(self):
        from mmpreprocesspy.roi import Roi
        sut = Roi(1, 2, 3, 4)
        self.assertEqual(sut.m1, 1)

