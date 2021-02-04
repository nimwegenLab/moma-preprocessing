from unittest import TestCase
import os
from skimage.transform import AffineTransform, warp

class test_MomaImageProcessor(TestCase):
    data_dir = os.path.join(os.path.dirname(__file__), 'resources/data__test__moma_image_processing')

    def test__2(self):
        import numpy as np
        import tifffile as tff
        import matplotlib.pyplot as plt
        import copy
        from mmpreprocesspy.moma_image_processing import MomaImageProcessor
        from mmpreprocesspy.GlDetectionTemplate import GlDetectionTemplate
        from mmpreprocesspy.preproc_fun import translate_gl_rois

        original_image = tff.imread(os.path.join(self.data_dir, '12_20190816_Theo_MMStack.ome.tif'))
        gl_detection_template_path = os.path.join(self.data_dir, 'gl_detection_templates/12_theo_20190816_glc_spcm_1_MMStack_2__template_v00.json')
        main_channel_angle = 0
        imageProcessor = MomaImageProcessor()
        if gl_detection_template_path:
            gl_detection_template = GlDetectionTemplate()
            gl_detection_template.load_config(gl_detection_template_path)
            imageProcessor.gl_detection_template = gl_detection_template
        imageProcessor.load_numpy_image_array(original_image)
        # imageProcessor.growthlane_length_threshold = growthlane_length_threshold
        imageProcessor.main_channel_angle = main_channel_angle
        # imageProcessor.roi_boundary_offset_at_mother_cell = roi_boundary_offset_at_mother_cell

        imageProcessor.process_image()

        growthlane_rois_orig = copy.deepcopy(imageProcessor.growthlane_rois)
        expected_shift = [0, 10.1]
        shifted_image_0 = self.support__shift_image(original_image, expected_shift)
        imageProcessor.determine_image_shift(shifted_image_0)
        growthlane_rois = copy.deepcopy(imageProcessor.growthlane_rois)
        growthlane_rois_shifted_0 = translate_gl_rois(growthlane_rois,
                                            (-imageProcessor.horizontal_shift, -imageProcessor.vertical_shift))
        print(f'determined_shift: {(imageProcessor.horizontal_shift, imageProcessor.vertical_shift)}')
        expected_shift = [0, 10.9]
        shifted_image_1 = self.support__shift_image(original_image, expected_shift)
        imageProcessor.determine_image_shift(shifted_image_1)
        growthlane_rois = copy.deepcopy(imageProcessor.growthlane_rois)
        growthlane_rois_shifted_1 = translate_gl_rois(growthlane_rois,
                                            (-imageProcessor.horizontal_shift, -imageProcessor.vertical_shift))
        print(f'determined_shift: {(imageProcessor.horizontal_shift, imageProcessor.vertical_shift)}')

        for ind, roi in enumerate(growthlane_rois_orig):
            print(f'orig: {growthlane_rois_shifted_0[ind].roi.points[0]}; shifted:  {growthlane_rois_shifted_1[ind].roi.points[0]}')
            print(f'orig: {growthlane_rois_shifted_0[ind].roi.points[1]}; shifted:  {growthlane_rois_shifted_1[ind].roi.points[1]}')
            print(f'orig: {growthlane_rois_shifted_0[ind].roi.points[2]}; shifted:  {growthlane_rois_shifted_1[ind].roi.points[2]}')
            print(f'orig: {growthlane_rois_shifted_0[ind].roi.points[3]}; shifted:  {growthlane_rois_shifted_1[ind].roi.points[3]}')

    def test__image_shift_is_correctly_detected(self):
        import numpy as np
        import tifffile as tff
        import matplotlib.pyplot as plt
        from mmpreprocesspy.moma_image_processing import MomaImageProcessor

        original_image = tff.imread(os.path.join(self.data_dir, '12_20190816_Theo_MMStack.ome.tif'))
        sut = MomaImageProcessor()
        sut._image_for_registration = original_image

        shifts_x_y = [5.7, 67.3]

        for xshift in shifts_x_y:
            for yshift in shifts_x_y:
                expected_shift = [xshift, yshift]

                shifted_image = self.support__shift_image(original_image, expected_shift)

                sut.determine_image_shift(shifted_image)
                actual_shift = (sut.horizontal_shift, sut.vertical_shift)

                np.testing.assert_array_almost_equal(expected_shift,
                                                     actual_shift,
                                                     decimal=2,
                                                     err_msg=f'expected_shift was not recovered: {expected_shift}\n'
                                                     f'actual shift: {actual_shift}')

    def support__shift_image(self, image, vector):
        transform = AffineTransform(translation=vector)
        shifted = warp(image, transform, mode='wrap', preserve_range=True)
        shifted = shifted.astype(image.dtype)
        return shifted


