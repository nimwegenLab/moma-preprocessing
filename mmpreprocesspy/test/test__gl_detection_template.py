from unittest import TestCase
from mmpreprocesspy.GlDetectionTemplate import GlDetectionTemplate

class TestGlDetectionTemplate(TestCase):

    def test__constructor__constructs_class(self):
        sut = GlDetectionTemplate()

    def test__load_config__loads_config(self):
        config_path = self.support__get_full_path_to_config()
        sut = GlDetectionTemplate()
        sut.load_config(config_path)

    def test__load_config__returns_correct_pixel_size(self):
        sut = self.support__get_sut_with_config()
        self.assertEquals(sut.pixel_size, 0.065)

    def test__nr_of_gl_regions__is_correct(self):
        sut = self.support__get_sut_with_config()
        actual = sut.nr_of_gl_region
        self.assertEqual(2, actual)

    def test__get_gl_region_in_pixel__returns_correct_values__for_region_0(self):
        sut = self.support__get_sut_with_config()
        gl_region_index = 0
        pixel_size = 0.065
        region = sut.get_gl_region_in_pixel(gl_region_index)
        self.assertEqual(to_pixel(1.95, pixel_size), region.start)
        self.assertEqual(to_pixel(36.725, pixel_size), region.end)
        self.assertEqual(to_pixel(36.725, pixel_size) - to_pixel(1.95, pixel_size), region.width)
        self.assertEqual(to_pixel(6.87375, pixel_size), region.gl_spacing_vertical)
        self.assertEqual(to_pixel(3.38, pixel_size), region.first_gl_position_from_top)

    def test__get_gl_region_in_micron__returns_correct_values__for_region_0(self):
        sut = self.support__get_sut_with_config()
        gl_region_index = 0
        region = sut.get_gl_region_in_micron(gl_region_index)
        self.assertEqual(1.95, region.start)
        self.assertEqual(36.725, region.end)
        self.assertEqual(34.775, region.width)
        self.assertEqual(6.87375, region.gl_spacing_vertical)
        self.assertEqual(3.38, region.first_gl_position_from_top)

    def test__get_gl_region_in_micron__returns_correct_values__for_region_1(self):
        sut = self.support__get_sut_with_config()
        gl_region_index = 1
        region = sut.get_gl_region_in_micron(gl_region_index)
        self.assertEqual(54.6, region.start)
        self.assertEqual(91.65, region.end)
        self.assertAlmostEqual(37.05, region.width)
        self.assertEqual(6.87375, region.gl_spacing_vertical)
        self.assertEqual(3.38, region.first_gl_position_from_top)

    def support__get_sut_with_config(self) -> GlDetectionTemplate:
        config_path = self.support__get_full_path_to_config()
        sut = GlDetectionTemplate()
        sut.load_config(config_path)
        return sut

    def support__get_full_path_to_config(self):
        import os
        relative_config_path = './resources/data__test__gl_detection_template/16_thomas_thomas_20201229_glc_lac_1.json'
        path = os.path.join(os.path.dirname(__file__), relative_config_path)
        return path

def to_pixel(micron_value, pixel_size):
    return int(micron_value / pixel_size)
