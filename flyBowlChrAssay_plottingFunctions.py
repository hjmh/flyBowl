__author__ = 'hannah'

import matplotlib.colors as colors
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns


def plotSparseMatrix(figsize, aspectRatio, matrixToPlot, titleString):
    fig = plt.figure(figsize=figsize)
    fig.set_canvas(plt.gcf().canvas)
    sns.set_style('ticks')
    ax = fig.add_subplot(111)
    ax.spy(matrixToPlot)
    ax.set_aspect(aspectRatio)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('top')
    ax.set_title(titleString)
    sns.axes_style({'axes.linewidth': 1, 'axes.edgecolor': '.8'})
    return fig


def plotVeloHistogram(velo, flyID, binVals, sbplot, xlabelString, colorCode):
    histVal, binEdges = np.histogram(velo[:, flyID], bins=binVals)
    binLoc = binEdges[0:-1]
    sbplot.plot(binLoc, histVal, color=colorCode, marker='.', linewidth=1.5)

    sbplot.set_xlabel(xlabelString, fontsize=14)
    sbplot.set_ylabel('count', fontsize=14)

    sbplot.yaxis.set_ticks_position('left')
    sbplot.xaxis.set_ticks_position('bottom')


def veloPerTrial(velo, windowT, tBegin, fly, ylableString, subplotHandle):
    [subplotHandle.plot(windowT, velo[tBegin[i]:tBegin[i]+len(windowT), fly], alpha=0.5)
     for i in range(len(tBegin))]
    subplotHandle.set_ylabel(ylableString, fontsize=12)
    subplotHandle.set_xlabel('time [s]', fontsize=12)
    subplotHandle.set_title('fly' + str(fly))
    subplotHandle.yaxis.set_ticks_position('left')
    subplotHandle.xaxis.set_ticks_position('bottom')


def veloPerTrialAverage(velo, windowT, tBegin, ylableString, subplotHandle):
    [subplotHandle.plot(windowT, np.nanmean(velo[tBegin[i]:tBegin[i]+len(windowT), :], 1), alpha=0.5, linewidth=2)
     for i in range(len(tBegin))]
    subplotHandle.set_ylabel(ylableString, fontsize=12)
    subplotHandle.set_xlabel('time [s]', fontsize=12)
    subplotHandle.set_title('average response')
    subplotHandle.yaxis.set_ticks_position('left')
    subplotHandle.xaxis.set_ticks_position('bottom')


def plotPosInRange_singleFly(ax, frameRange, xPos, yPos, angle, fly, currCmap):
    cNorm  = colors.Normalize(vmin=-0.5*len(frameRange), vmax=1*len(frameRange))
    scalarMap = plt.cm.ScalarMappable(norm=cNorm, cmap=currCmap)

    for frame in frameRange:
        currCol=scalarMap.to_rgba(len(frameRange)-ind)
        ax.plot(xPos[frame][flyID[frame] == fly], yPos[frame][flyID[frame] == fly],
                marker='.', markersize=6, linestyle='none', alpha=0.5, color=currCol)

        plotBodyAngle(ax, xPos[frame][flyID[frame] == fly], yPos[frame][flyID[frame] == fly],
                      angle[frame][flyID[frame] == fly], currCol, 0.5, 20)

    ax.set_aspect('equal')


def plotPosInRange(ax, frameRange, xPos, yPos, angle, flyID, currCmap):
    cNorm  = colors.Normalize(vmin=-0.5*len(frameRange), vmax=1*len(frameRange))
    scalarMap = plt.cm.ScalarMappable(norm=cNorm, cmap=currCmap)
    for frameind, frame in enumerate(frameRange):
        currCol=scalarMap.to_rgba(len(frameRange)-frameind)
        ax.plot(xPos[frame], yPos[frame], marker='.', markersize=10, linestyle='none', alpha=0.5,
                color=currCol)

        for fly in flyID[frame]:
            plotBodyAngle(ax, xPos[frame][flyID[frame] == fly], yPos[frame][flyID[frame] == fly],
                          angle[frame][flyID[frame] == fly], currCol, 0.5, 20)

    ax.set_aspect('equal')

    plt.xlim([0, 1000])
    plt.ylim([0, 1000])


def plotBodyAngle(ax, x, y, angle, markerColor, alphaVal, arrowScale):
    try:
        newArrow = patches.Arrow(x, y, np.cos(angle).squeeze()*arrowScale, np.sin(angle).squeeze()*arrowScale, width=2,
                                 edgecolor=markerColor, alpha=alphaVal)
        ax.add_patch(newArrow)
    except:
        couldNotPrint = True


def plotPosAndAngleInRange_singleFly_colorStim(ax, frameRange, stimCode, xPos, yPos, angle, flyID, fly, currCmap):
    cNorm  = colors.Normalize(vmin=-0.5*len(frameRange), vmax=1*len(frameRange))
    scalarMap = plt.cm.ScalarMappable(norm=cNorm, cmap=currCmap)
    for ind, frame in enumerate(frameRange):
        currCol=scalarMap.to_rgba(len(frameRange)-ind)
        if(stimCode[ind]):
            # we are within stimulation period
            currCol = 'red'

        ax.plot(xPos[frame][flyID[frame] == fly], yPos[frame][flyID[frame] == fly],
                marker='.', markersize=6, linestyle='none', alpha=0.5, color=currCol)

        plotBodyAngle(ax, xPos[frame][flyID[frame] == fly], yPos[frame][flyID[frame] == fly],
                      angle[frame][flyID[frame] == fly], currCol, 0.5, 20)

    ax.set_aspect('equal')


def plotPosAndAngleInRange_singleFly_colorTrial(ax, frameRange, stimCode, trialCode, trialLength, xPos, yPos, angle,
                                                flyID, fly, currCmap):
    cNorm  = colors.Normalize(vmin=min(trialCode), vmax=max(trialCode))
    scalarMap = plt.cm.ScalarMappable(norm=cNorm, cmap=currCmap)

    for ind, frame in enumerate(frameRange):
        currCol=scalarMap.to_rgba(trialCode[ind])
        alphaVal = max(0, 1-1.9*(ind % trialLength.astype('float'))/trialLength)
        if(stimCode[ind]):
            # we are within stimulation period
            currCol = 'black'

        ax.plot(xPos[frame][flyID[frame] == fly], yPos[frame][flyID[frame] == fly],
                marker='.', markersize=6, linestyle='none', alpha=alphaVal, color=currCol)
        # alpha=(1-float(ind)/(len(frameRange)+100))
        plotBodyAngle(ax, xPos[frame][flyID[frame] == fly], yPos[frame][flyID[frame] == fly],
                      angle[frame][flyID[frame] == fly], currCol, alphaVal, 20)

    ax.set_aspect('equal')
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')


def plotPosAndAngleInRange_singleFly_separateTrials(fig, trialBegin_traces, numTrials, stimFrames, postStimFrames,
                                                    skipFrame, xPos, yPos, angle, flyID, fly, currCmap, trialCode):
    cNorm  = colors.Normalize(vmin=min(trialCode), vmax=max(trialCode))
    scalarMap = plt.cm.ScalarMappable(norm=cNorm, cmap=currCmap)

    for trial in range(numTrials):
        ax = fig.add_subplot(np.ceil(numTrials/5.0),5,trial)

        #define window around stimulation pulse which should be plotted
        frameRangeStim = range(trialBegin_traces[trial], trialBegin_traces[trial]+stimFrames, skipFrame)
        frameRangePost = range(trialBegin_traces[trial]+stimFrames, trialBegin_traces[trial]+postStimFrames, skipFrame)

        for ind, frame in enumerate(frameRangeStim):
            currCol = 'grey'
            ax.plot(xPos[frame][flyID[frame] == fly], yPos[frame][flyID[frame] == fly],
                    marker='.', markersize=6, linestyle='none', alpha=0.5, color=currCol)
            plotBodyAngle(ax, xPos[frame][flyID[frame] == fly], yPos[frame][flyID[frame] == fly],
                          angle[frame][flyID[frame] == fly], currCol, 0.5, 22)

        for ind, frame in enumerate(frameRangePost):
            currCol=scalarMap.to_rgba(trial)
            ax.plot(xPos[frame][flyID[frame] == fly],yPos[frame][flyID[frame] == fly],
                    marker='.', markersize=6, linestyle='none', alpha=0.5, color=currCol)
            plotBodyAngle(ax, xPos[frame][flyID[frame] == fly], yPos[frame][flyID[frame] == fly],
                          angle[frame][flyID[frame] == fly], currCol, 0.5, 22)

        ax.set_aspect('equal')
        plt.axis('off')
        sns.despine(right=True, left=True, bottom=True, top=True)
