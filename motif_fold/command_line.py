from .motif_fold import motif_fold
import os
import argparse
import numpy
from matplotlib import pyplot
import pandas
import seaborn
import glob


def main():
    # Use current working directory as default input/output folder
    cwd = os.getcwd()
    input_folder = cwd
    output_folder = cwd

    # Parse arguments for input/output folder
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', type=str, help='Full path to input folder where sequence excel data is. ')
    args = parser.parse_args()

    if args.i:
        input_folder = args.i

    # Call sequence matching function
    motif_fold(input_folder)