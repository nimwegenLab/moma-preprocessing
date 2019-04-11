#!/bin/python

'''#
This script corresponds to the script mm_pre_slurm.sh, which is used for the original Java MMPreproc implementation.
'''

import sys
print("Python version:")
print(sys.version)

import argparse
from preproc_fun import preproc_fun

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
args = parser.parse_args()

print("My input:")
print(args.input)
print(args.output)
print(args.positions)
print(args.rotations)
print(args.timeframemaximum)

# here we need to convert the string 'Pos0' to just '[0]', which we then pass to preproc_fun

#preproc_fun(args.input,args.output,args.positions,args.timeframemaximum)
preproc_fun(args.input,args.output,[0],args.timeframemaximum)
