from unittest import TestCase


class TestPreprocessing(TestCase):
    def test_split_channels_init(self):
        from PIL import Image

        #import mmpreprocesspy.preprocessing as pre
        #import mmpreprocesspy.preprocessing
        import mmpreprocesspy
        # from matplotlib.pyplot import imshow
        # import numpy as np

        image_path = '/home/micha/Documents/git/MM_Testing/04_20180531_gluIPTG5uM_lac_1/first_images/Pos0/img_000000000_ DIA Ph3 (GFP)_000.tif'

        image_base = Image.open(image_path)

        # image_base.convert(mode="RGB")
        image_base.mode = 'I'
        im2 = image_base.point(lambda i: i * (1. / 256)).convert('L')
        im2.show()
        im2.save('/home/micha/Documents/git/MM_Testing/04_20180531_gluIPTG5uM_lac_1/first_images/Pos0/img_000000000_ DIA Ph3 (GFP)_000__.jpeg')



        # image_base.save('/home/micha/Documents/git/MM_Testing/04_20180531_gluIPTG5uM_lac_1/first_images/Pos0/img_000000000_ DIA Ph3 (GFP)_000__.jpeg')

        # image_base.convert()
        # table = [i / 256 for i in range(65536)]
        #
        # im2 = image_base.point(table, 'L')
        #
        # print
        # im2.mode
        #
        # im2.show()

        # image_base.convert('L').save('/home/micha/Documents/git/MM_Testing/04_20180531_gluIPTG5uM_lac_1/first_images/Pos0/img_000000000_ DIA Ph3 (GFP)_000__.jpeg')
        # image_base.show()
        # imshow(np.asarray(image_base))
        # image_rot, angle, mincol, maxcol, channel_centers = mmpreprocesspy.preprocessing.split_channels_init(image_base)
        # self.fail()
