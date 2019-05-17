#!/bin/python

'''#
This script corresponds to the script mm_pre_slurm.sh, which is used for the original Java MMPreproc implementation.
'''

import sys
#print("Python version:")
#print(sys.version)

import argparse
import re

from mmpreprocesspy.preproc_fun import preproc_fun

parser = argparse.ArgumentParser()

parser.add_argument("-i", "--input", type=str,
                    help="input directory")
parser.add_argument("-o", "--output", type=str,
                    help="output directory")
parser.add_argument("-p", "--positions", type=str,
                    help="positions of the images")
parser.add_argument("-r", "--rotations", type=str,
                    help="rotations of the images")
parser.add_argument("-tmax", "--timeframemaximum", type=int,
                    help="maximum time frame upto which the data is processed")
parser.add_argument("-ff", "--flatfieldpath", type=str,
                    help="path to the folder containing the flatfield OME-Tiff stack")
parser.add_argument("-log", "--logfile", type=str,
                    help="path to log-file")
args = parser.parse_args()

# overwrite sys.stdout and sys.stderr for logging
if args.logfile is not None:
    logfile = open(args.logfile, 'w')
    sys.stdout = logfile
    sys.stderr = logfile

print("Input path:")
print(args.input)
print("Output path:")
print(args.output)
print("Position:")
print(args.positions)
print("Rotations:")
print(args.rotations)
print("Max frame:")
print(args.timeframemaximum)
print("Flatfield path:")
print(args.flatfieldpath)
print("Log-file path:")
print(args.logfile)

# parse position argument; IMPORTANT: this only works for a single position argument
res = re.match('Pos[0]*(\d+)', args.positions)
posval = int(res.group(1))
posval = [posval];

preproc_fun(args.input, args.output, posval, args.timeframemaximum, args.flatfieldpath)
