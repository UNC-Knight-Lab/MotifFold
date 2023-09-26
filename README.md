# MotifFold

## Description
This Python package was developed by the Knight Lab at UNC Chapel Hill to regress sequence features against a colorimetric assay. It consists of two parts, an image quantification section that returns RGB values of selected beads, and a second will classify lists of RGB values into groups based on pre-classified test data.

## Installation Instructions
A version of Python >= 3.7 is required to use this package. We recommend using [Anaconda](https://www.anaconda.com) to install Python for those new to Python.
1. Open the terminal (MacOS) or Command Prompt (Windows).
2. Download the package by either:
   1. Download the zip from GitHub (Code -> Download ZIP). Unzip the package somewhere (note the extraction path). The extracted package can be deleted after installation.
   2. Clone this repository (requires git to be installed) with:
      
   `git clone https://github.com/UNC-Knight-Lab/MotifFold.git`

3. Install the package using pip. This command will install this package to your Python environment.
    The package path should be the current working directory `.` if cloned using git. Otherwise, replace it with the path to the `peptoid-sequence-tools` folder.
      
   `pip install .`
   or `pip install /path/to/package/MotifFold`

That's it!

## How to use
This tool can be run as a Python function or from the command line terminal. The tools will prompt for user input. Sample data is included for both parts of this script. 

### Sequence quantification


### To run from terminal:

    motif-fold -i "/path/to/input_folder"
    
Instead of specifying an input or output folder, you can also navigate to your data input folder in the terminal and run the script.
The current working directory will be used as default.
Use the help `-h` tag to see more options.

### To run in Python:

    from motif-fold import motif-fold
    motif-fold(input_folder)
