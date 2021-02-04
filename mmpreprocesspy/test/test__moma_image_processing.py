from unittest import TestCase
import os
from skimage.transform import AffineTransform, warp

class test_MomaImageProcessor(TestCase):
    data_dir = os.path.join(os.path.dirname(__file__), 'resources/data__test__moma_image_processing')

    def test__bla(self):
        import tifffile as tff
        import matplotlib.pyplot as plt
        from mmpreprocesspy.moma_image_processing import MomaImageProcessor

        sut = MomaImageProcessor()
        original_image = tff.imread(os.path.join(data_dir, '12_20190816_Theo_MMStack.ome.tif'))
        sut._image_for_registration = original_image
        shifted_image = self.support__shift_image(original_image, [50, 50])

        fig, ax = plt.subplots(1, 2, figsize=(5, 10))
        ax[0].imshow(original_image)
        ax[1].imshow(shifted_image)
        plt.show()

        sut.determine_image_shift(self, image)
        pass

    def support__shift_image(image, vector):
        transform = AffineTransform(translation=vector)
        shifted = warp(image, transform, mode='wrap', preserve_range=True)
        shifted = shifted.astype(image.dtype)


