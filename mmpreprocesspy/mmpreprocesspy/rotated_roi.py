import cv2 as cv
import numpy as np
import mmpreprocesspy.dev_auxiliary_functions as aux


class RotatedRoi(object):
    def __init__(self, center, size, angle):
        """center = (x, y)
           size = (width, height)"""
        self.center = center
        self.size = size
        self.width = size[0]
        self.height = size[1]
        self.angle = angle
        pass

    @staticmethod
    def create_from_roi(roi):
        """
         Create an instance of rotated ROI from non-rotated ROI object.
        :param roi:
        :return:
        """
        center_y = roi.m1 + roi.height/2
        center_x = roi.n1 + roi.width/2
        rotated_roi = RotatedRoi((center_x, center_y), (roi.width, roi.height), 0)
        return rotated_roi

    def get_from_image(self, image):
        """ Return the cropped and rotated image data of the ROI. """
        if self.angle < -45.0: # REF: https://answers.opencv.org/question/497/extract-a-rotatedrect-area/?answer=518#post-id-518
            self.angle += 90.0
            width = self.width
            self.width = self.height
            self.height = width

        # boundRect = cv.boundingRect(self.points)
        x, y, w, h = cv.boundingRect(self.points)
        # boundingRectCenter = (x+w/2, y+h/2)
        # aux.show_image_with_rotated_rois(image, [self])
        # cv.waitKey()
        bounding_box_roi = RotatedRoi((x+w/2, y+h/2), (w, h), 0)
        bounding_box_image = image[y:y+h,x:x+w]
        bounding_image_center = (bounding_box_image.shape[1]/2, bounding_box_image.shape[0]/2)
        # cv.imshow("boundingBoxImage", bounding_box_image)
        # aux.show_image(bounding_box_image)
        # cv.waitKey()

        M = cv.getRotationMatrix2D(bounding_image_center, -self.angle, 1.0)
        rotated_image = cv.warpAffine(bounding_box_image, M, (bounding_box_image.shape[1], bounding_box_image.shape[0]), cv.INTER_CUBIC)

        # cv.imshow("rotated_image", rotated_image)
        # cv.waitKey()

        # aux.show_image_with_rotated_rois(image, [self, bounding_box_roi])
        # cv.waitKey()

        rotated_image = np.float32(rotated_image)
        cropped = cv.getRectSubPix(rotated_image, self.size, bounding_image_center)

        # cv.imshow("tmp", np.int16(cropped))
        # aux.show_image(cropped)
        # cv.waitKey()
        return cropped

    def draw_to_image(self, image, with_bounding_box=False):
        """
         Draw the ROI to image.
        :param image: image to which the ROI will be drawn.
        :param with_bounding_box: whether to also draw the bounding box of the rotated ROI.
        :return:
        """

        if not with_bounding_box:
            cv.drawContours(image, [self.points], 0, (255, 0, 0), 2)
        else:
            x, y, w, h = cv.boundingRect(self.points)
            bounding_box_roi = RotatedRoi((x+w/2, y+h/2), (w, h), 0)
            for roi in [self, bounding_box_roi]:
                cv.drawContours(image, [roi.points], 0, (255, 0, 0), 2)
        # return image

    def rotate(self, rotation_center, rotation_angle):
        M = cv.getRotationMatrix2D(rotation_center, rotation_angle, 1.0)
        center_coordinate = np.array([self.center[0], self.center[1], 1])
        self.center = M.dot(center_coordinate)
        self.angle = rotation_angle

    def translate(self, shift):
        self.center = (self.center[0] + shift[0], self.center[1] + shift[1])

    def is_inside_image(self, image):
        """ Returns whether this ROI lies with in the given image. """
        im_height = image.shape[0]
        im_width = image.shape[1]

        for point in self.points:
            if not 0 <= point[0] <= im_width:
                return False
            if not 0 <= point[1] <= im_height:
                return False
        return True

    @property
    def points(self):
        rot_rect = (self.center, (self.width, self.height), -self.angle) #  NOTE-MM-2019-05-02: I do not understand why it has to be '-self.angle' instead of self.angle
        return np.int0(cv.boxPoints(rot_rect))
