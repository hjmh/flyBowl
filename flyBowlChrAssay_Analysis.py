"""
 Analysis of trajectories from flies in fly bowl during CsChrimson stimulation

 Project: Flp-out based CsChrimson activation of putative CX-output neurons in walking flies
     About 20 flies are placed in fly bowl and full-field red light stimulation is delivered repeatedly
     for short periods. Only a few of the 20 flies will express CsChr in putative CX-output neurons,
     either uni- or bilateral.
"""

from scipy.io import loadmat
import csv
import matplotlib.colors as colors
import numpy as np
from scipy import sparse as sps

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import gridspec

import os
import sys

# Set path to analysis code directory
codeDir = sep.join(os.getcwd().split(sep)[:-2])
path.insert(1, codeDir)

# import custom written plotting functions
from plottingUtilities.flyBowlChrAssay_plottingFunctions import *
from plottingUtilities.ctraxFlyTracePlots import *

rootDir = '/Volumes/public/Ming2Chuntao/2015-06-09'
analysisDir = '/Volumes/jayaramanlab/Hannah/FlyBowlAnalysis'

with open(rootDir + '/experimentList.txt') as expList_handle:
    expList = expList_handle.read().splitlines()

numExperiments = len(expList)

for folder in expList:
    # Example:
    #   folder = 'SS02191_UAS-FRT-CsChrimson-mVeuns_attP18_hsFLP-PEST_attP3_HS0-3hALH120min_120min_flyBowlMing_...
    #               protocolVer3_Chuntao_CsChrimson_1intensities_20150609T111151'
    #   genotype = 'SS02191_UAS-FRT-CsChrimson-mVeuns_attP18_hsFLP-PEST_attP3_HS0-3hALH120min_120min'
    #   experiment = 'flyBowlMing_protocolVer3_Chuntao_CsChrimson_1intensities'

    fileNameParts = folder.split('_')

    genotype = '_'.join(fileNameParts[0:fileNameParts.index('flyBowlMing')])
    experiment = '_'.join(fileNameParts[fileNameParts.index('flyBowlMing'):-1])

    print(genotype)
    print(experiment)

    fileName = 'ctrax_results'

    fps = 30  # Video sampling rate

    plotSaveDir = analysisDir + '/' + genotype + '/' + experiment

    try:
        os.mkdir(analysisDir + '/' + genotype)
    except OSError:
        print('Genotype folder already exists')

    try:
        os.mkdir(plotSaveDir)
    except OSError:
        print('Experiment folder already exists')

    # ------------------------------------------------------------------------------------------------------------------
    # Import and rearrange data
    # ------------------------------------------------------------------------------------------------------------------

    # Extract protocol parameter
    protocol = loadmat(rootDir + '/' + folder + '/protocol.mat')  # load protocol parameter
    protocol = protocol['protocol']  # extract values from dict

    intensity = protocol['intensity'].squeeze().astype('int')
    pulseWidth = protocol['pulseWidthSP'].squeeze().astype('int')
    pulsePeriod = protocol['pulsePeriodSP'].squeeze().astype('int')
    stimTms = pulseWidth
    pauseTms = protocol['offTime'].squeeze().astype('int')
    numRepeat = protocol['iteration'].squeeze().astype('int')
    delayStart = protocol['delayTime'].squeeze().astype('int')

    stimCycle = (stimTms + pauseTms) / 1000

    # total length of protocol
    experimentT = (delayStart + numRepeat * stimCycle * fps)

    skipFrame = 3  # downsample from 30Hz --> 10Hz

    # Build indicator vector for red light stimulation
    stimBlock = np.hstack((np.ones((stimTms / 1000) * fps), np.zeros((pauseTms / 1000) * fps)))
    stimCode = np.tile(stimBlock, numRepeat)

    trialBlock = np.ones(((stimTms + pauseTms) / 1000) * fps)
    trialCode = np.repeat(range(1, numRepeat + 1), ((stimTms + pauseTms) / 1000) * fps)

    trialBegin = [(k * (stimTms + pauseTms) / 1000.0) * fps / skipFrame for k in range(numRepeat)]
    trialBlockPts = ((stimTms + pauseTms) / 1000.0) * fps / skipFrame
    trialBlockT = np.linspace(0, (stimTms + pauseTms) / 1000.0, trialBlockPts)

    # ------------------------------------------------------------------------------------------------------------------
    # Extract relevant tracking parameter
    # ------------------------------------------------------------------------------------------------------------------

    keyList = ['timestamps', 'x_pos', 'y_pos', 'ntargets', 'identity',
               'angle']  # data columns to be extracted from ctrax file

    # load matlab data and convert
    indat = loadmat(rootDir + '/' + folder + '/' + fileName + '.mat')

    dat = [indat[k] for k in keyList]

    # Reorganise fly position arrays into lists (sorted by frame)
    numFrames = len(dat[0])
    xPos = []
    yPos = []
    angle = []
    flyID = []

    pointer = 0
    for t in range(numFrames):
        numFlies = dat[3][t].astype('int')[0]

        xPos.append(dat[1][pointer:pointer + numFlies])
        yPos.append(dat[2][pointer:pointer + numFlies])
        angle.append(dat[5][pointer:pointer + numFlies])
        flyID.append(dat[4][pointer:pointer + numFlies])

        pointer += numFlies

    xPos = np.array(xPos)
    yPos = np.array(yPos)
    angle = np.array(angle)
    flyID = np.array(flyID)
    maxFlyID = max(dat[4])

    # ------------------------------------------------------------------------------------------------------------------
    # Visualise tracking performance
    # ------------------------------------------------------------------------------------------------------------------

    # Reorganise fly track fragments into matrix (frame x fly id )
    flyIDperFrame = np.zeros((numFrames, maxFlyID + 1))
    for frame in range(numFrames):
        for idx in np.array(flyID[frame]).squeeze().astype('int'):
            flyIDperFrame[frame][idx] = 1

    # visualise resluting matrix
    fragmentFig = plotSparseMatrix((7, 5), 0.003, flyIDperFrame, genotype + '\n' + experiment + '\n')

    fragmentFig.savefig(plotSaveDir + '/' + genotype + '_' + experiment + '_traceFragments.pdf', format='pdf')

    # ------------------------------------------------------------------------------------------------------------------
    # Plot tracking data
    # ------------------------------------------------------------------------------------------------------------------

    # Visualise response of all flies to first light pulse

    fig = plt.figure(figsize=(10, 10))
    sbplt = fig.add_subplot(111)

    firstStim = range(delayStart * fps, (delayStart + stimTms / 1000) * fps, 6)
    plotPosAndAngleInRange(sbplt, firstStim, xPos, yPos, angle, flyID, -1, 'OrRd')

    firstPause = range((delayStart + stimTms / 1000) * fps, (delayStart + ((stimTms + pauseTms) / 1000)) * fps, 5)
    plotPosAndAngleInRange(sbplt, firstPause, xPos, yPos, angle, flyID, -1, 'PuBu')

    fig.savefig(plotSaveDir + '/' + genotype + '_' + experiment + '_firstTrialTraces.pdf', format='pdf')

    # Plot per-fly walking statistics

    # compute translational and rotational velocity over entire experiment
    frameRange = range((delayStart) * fps, (delayStart + numRepeat * ((stimTms + pauseTms) / 1000)) * fps, skipFrame)
    activeFragments = np.array(np.nonzero(sum(flyIDperFrame[frameRange]))).squeeze()

    transV = np.zeros((len(frameRange), len(activeFragments)))
    rotV = np.zeros((len(frameRange), len(activeFragments)))

    for k, fly in enumerate(activeFragments):

        currTransV = np.zeros((len(frameRange), 1))
        currRotV = np.zeros((len(frameRange), 1))

        for ind, frame in enumerate(frameRange):
            if ((not np.any(flyID[frame] == fly)) or (not np.any(flyID[frame - skipFrame] == fly))):
                continue

            currTransV[ind] = np.hypot(
                xPos[frame][flyID[frame] == fly] - xPos[frame - skipFrame][flyID[frame - skipFrame] == fly],
                yPos[frame][flyID[frame] == fly] - yPos[frame - skipFrame][flyID[frame - skipFrame] == fly])

            currRotV[ind] = angle[frame][flyID[frame] == fly] - angle[frame - skipFrame][
                flyID[frame - skipFrame] == fly]
            if currRotV[ind] > np.pi:
                currRotV[ind] -= 2 * np.pi
            if currRotV[ind] < -np.pi:
                currRotV[ind] += 2 * np.pi

        transV[:, k] = currTransV.squeeze()
        rotV[:, k] = currRotV.squeeze()

    # Plot results
    Vfig = plt.figure(figsize=(15, 9))
    tVplt = Vfig.add_subplot(211)
    rVplt = Vfig.add_subplot(212)

    cNorm = colors.Normalize(vmin=min(activeFragments), vmax=max(activeFragments))
    FlyMap = plt.cm.ScalarMappable(norm=cNorm, cmap='Paired')

    for k, fly in enumerate(activeFragments):
        flyColor = FlyMap.to_rgba(fly)

        histVal, binEdges = np.histogram(transV[:, k], bins=np.linspace(0, 30, 50))
        binLoc = binEdges[0:-1]  # -0.5*np.mean(np.diff(binEdges))
        tVplt.plot(binLoc, histVal, color=flyColor, marker='.', linewidth=1.5)

        histVal, binEdges = np.histogram(abs(rotV[:, k]), bins=np.linspace(0, np.pi, 50))
        binLoc = binEdges[0:-1]  # -0.5*np.mean(np.diff(binEdges))
        rVplt.plot(binLoc, histVal, color=flyColor, marker='.', linewidth=1.5)

    tVplt.set_ylim((0, 200))
    tVplt.set_xlim((0, 30))
    tVplt.set_xlabel('translational velocity [px/0.1s]', fontsize=14)
    tVplt.set_ylabel('count', fontsize=14)
    tVplt.legend(activeFragments, ncol=4, fontsize=12)
    tVplt.yaxis.set_ticks_position('left')
    tVplt.xaxis.set_ticks_position('bottom')

    rVplt.set_ylim((0, 100))
    rVplt.set_xlim((0, np.pi))
    rVplt.set_xlabel('rotational velocity [rad/0.1s]', fontsize=14)
    rVplt.set_ylabel('count', fontsize=14)
    rVplt.legend(activeFragments, ncol=4, fontsize=12)
    rVplt.yaxis.set_ticks_position('left')
    rVplt.xaxis.set_ticks_position('bottom')

    Vfig.savefig(plotSaveDir + '/' + genotype + '_' + experiment + '_veloDistribution.pdf', format='pdf')

    # Trigger on light ON
    windowPreMS = 1000
    windowPostMS = 3000

    # Note: to be able to look at pre-stimulus window, the first trial is not regarded
    trialBegin = [((k + 1) * (stimTms + pauseTms) / 1000.0) * fps / skipFrame for k in range(numRepeat - 1)]
    trialWindowBlockPts = ((stimTms + windowPreMS + windowPostMS) / 1000.0) * fps / skipFrame
    trialWindowBlockT = np.linspace(-windowPreMS / 1000.0, (stimTms + windowPostMS) / 1000.0, trialWindowBlockPts)
    plotBegin = [trialBegin[i] - windowPreMS / 1000.0 * fps / skipFrame for i in range(len(trialBegin))]

    fig = plt.figure(figsize=(15, 1.1 * len(activeFragments)))

    sbpltCounter = 1
    for fly in activeFragments:
        try:
            vTplt = fig.add_subplot(len(activeFragments) / 2, 4, sbpltCounter)
            veloPerTrial(transV, trialWindowBlockT, plotBegin, fly, 'transl. velocity', vTplt)
            vTplt.add_patch(
                patches.Rectangle((0, -1), stimTms / 1000, 50, alpha=0.2, facecolor='red', edgecolor='none')
            )
            vTplt.set_ylim((-0.5, 30))

            vRplt = fig.add_subplot(len(activeFragments) / 2, 4, sbpltCounter + 1)
            veloPerTrial(rotV, trialWindowBlockT, plotBegin, fly, 'rot. velocity', vRplt)
            vRplt.add_patch(
                patches.Rectangle((0, -5), stimTms / 1000, 10, alpha=0.2, facecolor='red', edgecolor='none')
            )
            vRplt.set_ylim((-0.75 * np.pi, 0.75 * np.pi))

            sbpltCounter += 2
        except:
            print('problem with trace fragment of fly' + str(fly))

    plt.tight_layout()
    fig.savefig(plotSaveDir + '/' + genotype + '_' + experiment + '_summary.pdf', format='pdf')
