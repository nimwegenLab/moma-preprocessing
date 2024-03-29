#!/bin/python

'''#
This script corresponds to the script mm_pre_slurm.sh, which is used for the original Java MMPreproc implementation.
'''

import sys
#print("Python version:")
#print(sys.version)

import argparse
import re

from mmpreprocesspy.preproc_fun import PreprocessingRunner

parser = argparse.ArgumentParser()

parser.add_argument("-i", "--input", type=str,
                    help="input directory")
parser.add_argument("-o", "--output", type=str,
                    help="output directory")
parser.add_argument("-p", "--positions", type=str,
                    help="positions of the images")
parser.add_argument("-r", "--rotation", type=float,
                    help="rotation of the images")
parser.add_argument("-tmin", "--tmin", type=int,
                    help="minimum time frame after which the data is processed")
parser.add_argument("-tmax", "--tmax", type=int,
                    help="maximum time frame upto which the data is processed")
parser.add_argument("-ff", "--flatfieldpath", type=str,
                    help="path to the folder containing the flatfield OME-Tiff stack")
parser.add_argument("-log", "--logfile", type=str,
                    help="path to log-file")
parser.add_argument("-glt", "--growthlanelengththreshold", type=int,
                    help="minimum length to be considered as a growth-lane")
parser.add_argument("-roioffsetmc", "--roi_boundary_offset_at_mother_cell", type=int,
                    help="shift the detected position of the ROI at the location of the mother-cell")
parser.add_argument("-gldtp", "--gl_detection_template_path", type=str,
                    help="")
parser.add_argument("-normconfpath", "--normalization_config_path", type=str,
                    help="")
parser.add_argument("-zslice", "--z_slice_index", type=int,
                    help="")
parser.add_argument("-nro", "--normalization_region_offset", type=int,
                    help="offset by which the normalization region is reduced relative to the GL ROI length")
parser.add_argument("-fti", "--frames_to_ignore", type=str,
                    help="pass the frames which should be ignored as a comma-separated list of integers")
parser.add_argument("-irm", "--image_registration_method", type=int,
                    help="pass the frames which should be ignored as a comma-separated list of integers")
parser.add_argument("-finr", "--forced_intensity_normalization_range", type=str,
                    help="pass the minimum and maximum normalization intensities as a comma-separated list of floats")
parser.add_argument("-inrc", "--intensity_normalization_range_cutoffs", type=str,
                    help="cutoff values for detecting the minimum and maximum normalization intensities as a comma-separated list of floats, i.e. [min_value, max_value]; intensity values outside this range will be ignored")


# parse values
args = parser.parse_args()
frames_to_ignore = []
if args.frames_to_ignore:
    frames_to_ignore.extend([int(strInt) for strInt in args.frames_to_ignore.split(",")])
    frames_to_ignore = [val - 1 for val in frames_to_ignore]  # convert from 1-based index (used by ImageJ) to 0-based (used by Python)

forced_intensity_normalization_range = None
if args.forced_intensity_normalization_range:
    [range_min, range_max] = args.forced_intensity_normalization_range.split(",")
    forced_intensity_normalization_range = (float(range_min.strip()), float(range_max.strip()))

intensity_normalization_range_cutoffs = None
if args.intensity_normalization_range_cutoffs:
    [range_min, range_max] = args.intensity_normalization_range_cutoffs.split(",")
    intensity_normalization_range_cutoffs = (float(range_min.strip()), float(range_max.strip()))

# overwrite sys.stdout and sys.stderr for logging
if args.logfile is not None:
    logfile = open(args.logfile, 'w')
    sys.stdout = logfile
    sys.stderr = logfile

print("Input path:")
print(args.input)
print("Output path:")
print(args.output)
print("Flatfield path:")
print(args.flatfieldpath)
print("Log-file path:")
print(args.logfile)
print("Position:")
print(args.positions)
print("Rotation:")
print(args.rotation)
print("Start frame (tmin):")
print(args.tmin)
print("End frame (tmax):")
print(args.tmax)
print("Growthlane length threshold (glt):")
print(args.growthlanelengththreshold)
print("roi_boundary_offset_at_mother_cell:")
print(args.roi_boundary_offset_at_mother_cell)
print("gl_detection_template_path:")
print(args.gl_detection_template_path)
print("normalization_config_path:")
print(args.normalization_config_path)
print("z_slice_index:")
print(args.z_slice_index)
print("normalization_region_offset:")
print(args.normalization_region_offset)
print("frames_to_ignore:")
print(frames_to_ignore)
print("image_registration_method:")
print(args.image_registration_method)
print("forced_intensity_normalization_range:")
print(forced_intensity_normalization_range)
print("intensity_normalization_range_cutoffs:")
print(intensity_normalization_range_cutoffs)


# parse position argument; IMPORTANT: this only works for a single position argument
runner = PreprocessingRunner()
runner.preproc_fun(args.input,
                   args.output,
                   positions=[args.positions],
                   minframe=args.tmin,
                   maxframe=args.tmax,
                   flatfield_directory=args.flatfieldpath,
                   growthlane_length_threshold=args.growthlanelengththreshold,
                   main_channel_angle=args.rotation,
                   roi_boundary_offset_at_mother_cell=args.roi_boundary_offset_at_mother_cell,
                   gl_detection_template_path=args.gl_detection_template_path,
                   normalization_config_path=args.normalization_config_path,
                   z_slice_index=args.z_slice_index,
                   normalization_region_offset=args.normalization_region_offset,
                   frames_to_ignore=frames_to_ignore,
                   image_registration_method=args.image_registration_method,
                   forced_intensity_normalization_range=forced_intensity_normalization_range,
                   intensity_normalization_range_cutoffs=intensity_normalization_range_cutoffs)
