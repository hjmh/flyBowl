__author__ = 'hannah'

from scipy.io import loadmat
import csv
import numpy as np
from scipy import sparse as sps

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import gridspec
import matplotlib.colors as colors

import os
from glob import glob
from os.path import isfile, sep

import seaborn as sns

from flyBowlChrAssay_plottingFunctions import plotSparseMatrix, plotPosInRange, plotVeloHistogram,\
    veloPerTrial, veloPerTrialAverage, plotPosAndAngleInRange_singleFly_colorStim,\
    plotPosAndAngleInRange_singleFly_colorTrial, plotPosAndAngleInRange_singleFly_separateTrials

from flyBowlChrAssay_screenSingleExpAnalysis import screenSingleExpAnalysis

rootDir = '/Volumes/jayaraman/Chuntao/Ming_FlyBowl/'
analysisDir = '/Volumes/jayaraman/Hannah/Analysis_Ming_FlyBowl/'

# ----------------------------------------------------------------------------------------------------------------------
dateDir = 'Lori_CsChrimson_Screen'
expList = os.walk(rootDir+dateDir).next()[1]

for currDir in range(len(expList)):
    folder = expList[currDir]

    # Per experiment analysis
    # ------------------------------------------------------------------------------------------------------------------
    screenSingleExpAnalysis(rootDir, analysisDir, dateDir, folder)
    # ------------------------------------------------------------------------------------------------------------------

# rootDir = '/Volumes/jayaraman/Chuntao/Ming_FlyBowl/'
# analysisDir = '/Volumes/jayaraman/Hannah/Analysis_Ming_FlyBowl/'
#
# ----------------------------------------------------------------------------------------------------------------------
# dateDirs = os.walk(rootDir).next()[1]
#
# for currDateDir in range(len(dateDirs)):
#     dateDir = dateDirs[currDateDir]
#     expList = os.walk(rootDir+dateDir).next()[1]
#
# ----------------------------------------------------------------------------------------------------------------------
# Per experiment analysis
# ----------------------------------------------------------------------------------------------------------------------
#
#     for currDir in range(len(expList)):
#         folder = expList[currDir]
#
#         screenSingleExpAnalysis(rootDir, analysisDir, dateDir, folder)

