import os
import json
from dataclasses import dataclass
import numpy as np
import tifffile as tff
from mmpreprocesspy.GrowthlaneRoi import GrowthlaneExitLocation

@dataclass
class GlRegion:
    """
    Stores information of a GlRegion. Field-values can be either
    type int or float as returned by
    GlDetectionTemplate.get_gl_region_in_micron and
    GlDetectionTemplate.get_gl_region_in_pixel.
    """
    start = None
    end = None
    width = None
    first_gl_position_from_top = None
    gl_spacing_vertical = None


class GlDetectionTemplate(object):
    """
    This class loads the template config from disk using `load_config`.
    It returns instances of `GlRegion` in either micron or pixel for
    each region in the config-file. For this use methods `get_gl_region_in_pixel`
    and `get_gl_region_in_micron`.
    """

    config_dict = None

    def load_config(self, config_path):
        self.config_path = config_path
        with open(config_path) as fp:
            self.config_dict = json.load(fp)

    @property
    def pixel_size(self):
        return self.config_dict['pixel_size_micron']

    @property
    def nr_of_gl_region(self) -> int:
        return len(self.config_dict['gl_regions'])

    @property
    def absolute_image_path(self):
        return self._get_absolute_path(self.config_path, self.config_dict['template_image_path'])

    @property
    def template_image(self) -> np.ndarray:
        return tff.imread(self.absolute_image_path)

    def get_gl_region_in_micron(self, index: int) -> GlRegion:
        region = GlRegion()
        region.start = self.config_dict['gl_regions'][index]['horizontal_range'][0]
        region.end = self.config_dict['gl_regions'][index]['horizontal_range'][1]
        region.width = region.end - region.start
        region.gl_spacing_vertical = self.config_dict['gl_regions'][index]['gl_spacing_vertical']
        region.first_gl_position_from_top = self.config_dict['gl_regions'][index]['first_gl_position_from_top']
        region.gl_exit_orientation = self.get_gl_exit_orientation(index)
        return region

    def get_gl_region_in_pixel(self, index: int) -> GlRegion:
        pixel_size = self.config_dict['pixel_size_micron']
        region = GlRegion()
        region.start = self.config_dict['gl_regions'][index]['horizontal_range'][0] / pixel_size
        region.end = self.config_dict['gl_regions'][index]['horizontal_range'][1] / pixel_size
        region.width = region.end - region.start
        region.gl_spacing_vertical = self.config_dict['gl_regions'][index]['gl_spacing_vertical'] / pixel_size
        region.first_gl_position_from_top = self.config_dict['gl_regions'][index]['first_gl_position_from_top'] / pixel_size
        region.gl_exit_orientation = self.get_gl_exit_orientation(index)
        return region

    def get_gl_exit_orientation(self, index: int) -> GrowthlaneExitLocation:
        try:
            gl_exit_orientation = self.config_dict['gl_regions'][index]['gl_exit_orientation']
        except KeyError:
            gl_exit_orientation = ""  # if the orientation is not part of the config, set it to ""; this will make it be detected (see below)
        if gl_exit_orientation == "":
            return None  # if exit orientation not specified, set to None; this will lead to it being automatically determined in GrowthlaneRoi.get_oriented_roi_image
        elif gl_exit_orientation == "left":
            return GrowthlaneExitLocation.AT_TOP
        elif gl_exit_orientation == "right":
            return GrowthlaneExitLocation.AT_BOTTOM

    def get_gl_regions_in_pixel(self):
        regions = []
        for index in range(len(self.config_dict['gl_regions'])):
            regions.append(self.get_gl_region_in_pixel(index))
        return regions

    def _get_absolute_path(self, config_path, template_image_path):
        if os.path.isabs(template_image_path):
            return template_image_path
        return os.path.normpath(os.path.join(os.path.dirname(config_path), template_image_path))  # construct absolute path from path to the config files; we assume the image-location is given relative to the location of the template config-file
