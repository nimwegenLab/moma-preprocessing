from unittest import TestCase


class TestPreprocessing(TestCase):
    def test_split_channels_init_dataset_03(self):
        from PIL import Image

        from mmpreprocesspy import preprocessing
        import numpy as np

        image_path = '/home/micha/Documents/git/MM_Testing/03_20180604_gluIPTG10uM_lac_lacIoe_1/first_images/Pos0/03_img_000000000_ DIA Ph3 (GFP)_000.tif'

        image_base = Image.open(image_path)
        image_array = np.array(image_base)
        image_rot, angle, mincol, maxcol, channel_centers = preprocessing.split_channels_init(image_array)
        self.assertEqual(359, mincol)
        self.assertEqual(675, maxcol)

    def test_split_channels_init_dataset_04(self):
        from PIL import Image

        from mmpreprocesspy import preprocessing
        import numpy as np

        image_path = '/home/micha/Documents/git/MM_Testing/04_20180531_gluIPTG5uM_lac_1/first_images/Pos0/04_img_000000000_ DIA Ph3 (GFP)_000.tif'

        image_base = Image.open(image_path)
        image_array = np.array(image_base)
        image_rot, angle, mincol, maxcol, channel_centers = preprocessing.split_channels_init(image_array)
        self.assertEqual(380, mincol)
        self.assertEqual(693, maxcol)

    # @staticmethod
    # def show_image(image): # this method is currently broken
    #     image.mode = 'I'
    #     im2 = image.point(lambda i: i * (1. / 256)).convert('L')
    #     im2.show()

