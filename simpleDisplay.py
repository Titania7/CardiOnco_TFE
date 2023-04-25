# -*- coding: utf-8 -*-
"""
Created on Tue Feb 21 17:13:42 2023
@author: Tatiana Dehon

 ========= Quick data display =========
 
"""
from SingleGraph import SingleGraph

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from matplotlib.lines import Line2D
from scipy import signal
from scipy.fft import rfft, rfftfreq, irfft
import numpy as np
import scipy.signal as sc

import neurokit2 as nk
import pandas as pd
import math

#%% Very basic displays

def display_plt(myGraphs : list):
    
    for currentGraph in myGraphs :
        #figure(figsize=(20, 15), dpi=80)
        figure(figsize=(10, 6))
        plt.plot(currentGraph._x, currentGraph._y)
        plt.title(currentGraph._title)
        plt.xlabel("Time [s]")
        plt.ylabel("Magnitude")
        #plt.xlim(0,1)
        plt.grid()
        plt.show()

def display_bp(bpSignal, lims = []):
    print("===== Signal : ===== \n", bpSignal, '\nSelected time :', lims, "\n===============")
    
    signal = bpSignal._y # list
    sfreq = bpSignal._samplerate
    step = 1/sfreq
    
    limSamples = []
    if len(lims) > 0 :
        for item in lims: # Go through the 2 items
            limSamples.append(item*sfreq)
        x = np.arange(min(lims), max(lims), step)
        signal = signal[int(min(limSamples)):int(max(limSamples))]
        
    else :
        x = np.arange(bpSignal._x[0], bpSignal._x[-1], step)
    
    #print( "before (x, y) : ", len(x), len(signal))
    while len(x) < len(signal):
        x = np.append(x, x[len(x)-1]+1)
    while len(x) > len(signal):
        x = np.delete(x, -1)
    #print( "after (x, y) : ", len(x), len(signal))
    
    # Display the selected data
    plt.plot(x, signal)
    plt.legend()
    plt.xlabel("Time [s]")
    plt.ylabel("Magnitude")
    plt.grid()
    plt.title("BPAo")
    plt.show()
    plt.close()


#%% Detection of peaks and bumps onsets

# Detection of the QRS complexes
def detect_qrs(ecgSignal, clean_technique, ecg_delineate, lims = [], show = False):

    """
    currentGraph is an ECG signal
    """
    
    print("===== Signal : ===== \n", ecgSignal, '\nSelected time :', lims, "\n===============")
    
    #%% Creation of the signals to be displayed
    
    signal = ecgSignal._y # list
    sfreq = ecgSignal._samplerate
    step = 1/sfreq
    
    limSamples = []
    if len(lims) > 0 :
        for item in lims: # Go through the 2 items
            limSamples.append(item*sfreq)
        x = np.arange(min(lims), max(lims), step)
        signal = signal[int(min(limSamples)):int(max(limSamples))]
        
    else :
        x = np.arange(ecgSignal._x[0], ecgSignal._x[-1], step)
    
    #print( "before (x, y) : ", len(x), len(signal))
    while len(x) < len(signal):
        x = np.append(x, x[len(x)-1]+1)
    while len(x) > len(signal):
        x = np.delete(x, -1)
    #print( "after (x, y) : ", len(x), len(signal))
    
    #%% Cleaning process
    
    # Test multiple cleaning processes
    cleaned_sig = pd.DataFrame({"ECG_Raw" : signal,
                                "ECG_NeuroKit" : nk.ecg_clean(signal, sampling_rate=sfreq, method="neurokit"),
                                "ECG_BioSPPy" : nk.ecg_clean(signal, sampling_rate=sfreq, method="biosppy"),
                                "ECG_PanTompkins" : nk.ecg_clean(signal, sampling_rate=sfreq, method="pantompkins1985"),
                                "ECG_Hamilton" : nk.ecg_clean(signal, sampling_rate=sfreq, method="hamilton2002"),
                                "ECG_Elgendi" : nk.ecg_clean(signal, sampling_rate=sfreq, method="elgendi2010"),
                                "ECG_EngZeeMod" : nk.ecg_clean(signal, sampling_rate=sfreq, method="engzeemod2012")})
    
    """
    # Visualize all types of ECG cleaning
    for signal in list(cleaned_sig.keys()):
        cleaned_sig[signal].plot()
        plt.title(signal)
        plt.show()
    """
    
    
    # Manually choosing the Neurokit process
    signal = cleaned_sig[clean_technique]
    
    #%% Process the analysis
    _, rpeaks = nk.ecg_peaks(signal, sfreq, correct_artifacts=True)
    # Delineate the ECG signal and NOT visualizing all peaks of ECG complexes
    _, waves_peak = nk.ecg_delineate(signal, rpeaks, sampling_rate=sfreq, method=ecg_delineate, show_type='peaks')
    
    p_peaks_indexes = [item for item in waves_peak["ECG_P_Peaks"] if not np.isnan(item)]
    q_peaks_indexes = [item for item in waves_peak["ECG_Q_Peaks"] if not np.isnan(item)]
    r_peaks_indexes = [item for item in rpeaks["ECG_R_Peaks"] if not np.isnan(item)]
    s_peaks_indexes = [item for item in waves_peak["ECG_S_Peaks"] if not np.isnan(item)]
    t_peaks_indexes = [item for item in waves_peak["ECG_T_Peaks"] if not np.isnan(item)]
    p_onsets_indexes = [item for item in waves_peak["ECG_P_Onsets"] if not np.isnan(item)]
    t_offsets_indexes = [item for item in waves_peak["ECG_T_Offsets"] if not np.isnan(item)]

    
    # If the absolute time is needed
    p_peaks_time = [item/sfreq + min(x) for item in p_peaks_indexes]
    q_peaks_time = [item/sfreq + min(x) for item in q_peaks_indexes]
    r_peaks_time = [item/sfreq + min(x) for item in r_peaks_indexes]
    s_peaks_time = [item/sfreq + min(x) for item in s_peaks_indexes]
    t_peaks_time = [item/sfreq + min(x) for item in t_peaks_indexes]
    p_onsets_time = [item/sfreq + min(x) for item in p_onsets_indexes]
    t_offsets_time = [item/sfreq + min(x) for item in t_offsets_indexes]
    
    #print("R peaks : ", r_peaks_time, "\nP peaks : ", p_peaks_time, "\nQ peaks : ", q_peaks_time, "\nS peaks : ", s_peaks_time,
    #      "\nT peaks : ", t_peaks_time, "\nP onsets : ", p_onsets_time, "\nT offsets : ", t_offsets_time)
    
    if show == True :
        plt.plot(x, signal, label='Signal')
        plt.plot(x[p_peaks_indexes], signal[p_peaks_indexes], 'o', color='blue', label='P peaks')
        plt.plot(x[q_peaks_indexes], signal[q_peaks_indexes], 'o', color='orange', label='Q peaks')
        plt.plot(x[r_peaks_indexes], signal[r_peaks_indexes], 'o', color='cyan', label='R peaks')
        plt.plot(x[s_peaks_indexes], signal[s_peaks_indexes], 'o', color='green', label='S peaks')
        plt.plot(x[t_peaks_indexes], signal[t_peaks_indexes], 'o', color='red', label='T peaks')
        plt.plot(x[p_onsets_indexes], signal[p_onsets_indexes], 'o', color='magenta', label='P onsets')
        plt.plot(x[t_offsets_indexes], signal[t_offsets_indexes], 'o', color='yellow', label='T offsets')
        plt.xlabel("Time [s]")
        plt.ylabel("Magnitude")
        plt.grid()
        plt.title("Visualisation of the peaks (Neurokit cleaning)")
        plt.show()
        plt.close()
    
    return [p_peaks_indexes, p_peaks_time], [q_peaks_indexes, q_peaks_time], [r_peaks_indexes, r_peaks_time], [s_peaks_indexes, s_peaks_time], [t_peaks_indexes, t_peaks_time], [p_onsets_indexes, p_onsets_time], [t_offsets_indexes, t_offsets_time]

# Computation of the onset points with the intersecting tangents method
def getBpOnsets_tang(bpGraph, rPeaks, lims = [], show = False):
    
    limSamples = []
    if len(lims) > 0 :
        for item in lims: # Go through the 2 items
            limSamples.append(item*bpGraph._samplerate)
        x = np.arange(min(lims), max(lims), bpGraph._step)
        
    else :
        x = np.arange(bpGraph._x[0], bpGraph._x[-1], bpGraph._step)

    # We stock the ranges of indexes inside x_min and x_max
    x_min = []
    x_max = []
    for i in range(len(rPeaks[0])-1):
        if rPeaks[0][i] >= x[0]*bpGraph._samplerate and rPeaks[0][i+1]-10 < x[-1]*bpGraph._samplerate:
            x_min.append(rPeaks[0][i])
            x_max.append(rPeaks[0][i+1]-10)
            
    #print(x_min, x_max)
    
    #print(x_min, x_max)
    bpOnsetsindex = []
    bpOnsetstime = []
    bpOnsetsY = []
    # For each R peak :
    for k in range(len(x_min)):
        limMin = x_min[k]
        limMax = x_max[k]
        
        #print("lims = ", limMin, limMax)
        
        x2 = np.arange(limMin, limMax)
        firstGraph = bpGraph._y[limMin:limMax]
        
        #print(len(x), len(firstGraph))
        
        # Compute first derivative
        slope_Y = np.diff(firstGraph)/np.diff(x2)
        # Add a the last value twice to deriv => same shape as firstGraph
        derivNotCleaned = np.append(slope_Y, slope_Y[-1])

        """ Cleaning of the gradient with polynomial approximation - Not good enough
        
        plt.figure()
        poly = np.polyfit(x,derivNotCleaned,40)
        deriv = np.poly1d(poly)(x)
        plt.plot(x,derivNotCleaned)
        plt.plot(x,deriv)
        plt.show()
        """
        
        """ Cleaning of the derivative with Fourier transform - OK
        
        # Visualization of the fourier signal
        yf = rfft(derivNotCleaned)
        xf = rfftfreq(len(x), bpGraph._step)
        plt.plot(xf, np.abs(yf))
        plt.show()
        """
    
        # Use of a butterworth filter to cut out the noise
        b, a = signal.butter(3, 0.05)
        zi = signal.lfilter_zi(b, a)
        z, _ = signal.lfilter(b, a, derivNotCleaned, zi=zi*derivNotCleaned[0])
        z2, _ = signal.lfilter(b, a, z, zi=zi*z[0])
        y = signal.filtfilt(b, a, derivNotCleaned)
    
        """ Visualization of the Fourier-cleaned signal
        
        # Visualization of the cleaned signal
        plt.figure
        plt.plot(x, derivNotCleaned, 'b', alpha=0.75)
        plt.plot(x, z, 'r--', x, z2, 'r', x, y, 'k')
        plt.legend(('noisy signal', 'lfilter, once', 'lfilter, twice',
                    'filtfilt'), loc='best')
        plt.grid(True)
        plt.show()
        """

        # Cleaned signal is stored in "deriv" variable
        deriv = y
    
    
    
        # Detection of the min value in firstGraph and max value in deriv
        maxVal = np.max(firstGraph)
        for i in range(len(firstGraph)):
            if firstGraph[i] == maxVal:
                relmaxValIndex = i
        searchZone = firstGraph[:relmaxValIndex]
        
        minIndex = np.argmin(searchZone)
        maxIndex = np.argmax(deriv)

    
        # First tangent calculation :
        valy = searchZone[minIndex]
        valx = x2[minIndex]
        #amin = deriv[minIndex]
        # We force the derivative of the minumum to zero
        amin = 0
        bmin = valy-amin*valx
        tgMin = amin*x2+bmin
        # Second tangent calculation :
        valy = firstGraph[maxIndex]
        valx = x2[maxIndex]
        amax = deriv[maxIndex]
        bmax = valy-amax*valx
        tgMaxDeriv = amax*x2+bmax
        # Determincation of the intersection point :
        x_inters = (bmax-bmin)/(amin-amax)
        y_inters = amin*x_inters+bmin
        
        """ Verification of values found :
        print("Graph : min_y = ", minVal)
        print("Deriv : y_max' = ", maxDeriv)
        print("Graph : x_min of y_min = ", (minIndex-halfDiff)/bpGraph._samplerate)
        print("Deriv : y_min' = f'(x_min) = ", deriv[minIndex])
        print("Deriv : x_max' = f'(y_max') = ", (maxIndex-halfDiff)/bpGraph._samplerate)
        print("Graph : y_gradMax = f(x_max') = ", firstGraph[maxIndex])
        print("Point of intersection = [", x_inters, ";", y_inters,"]")
        """    
        
        # Visualisation of the intersection with signal cleaned :
        if show == True :
            plt.figure()
            fig, axis = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
            axis[0].plot(x2, firstGraph, label = bpGraph._title)
            axis[0].plot(x2[minIndex], firstGraph[minIndex], 'o', color = 'magenta', label = "Min pressure point")
            axis[0].plot(x2, tgMin, '--',color = 'magenta', label = 'Min pressure tangent')
            axis[0].plot(x2[maxIndex], firstGraph[maxIndex], 'o', color = 'purple')
            axis[0].plot(x2, tgMaxDeriv, '--',color = 'purple', label = 'Max gradient tangent')
            axis[0].set_title("Graph number "+ str(k))
            axis[0].set_ylim([np.min(firstGraph)-0.5*(np.mean(firstGraph)-np.min(firstGraph)),np.max(firstGraph)+0.5*(np.max(firstGraph)-np.mean(firstGraph))])
            axis[0].plot(x_inters, y_inters, 'o', color = 'red', label = 'Onset of the pressure bump')
            axis[0].legend(loc="best")
        
            axis[0].grid()
            axis[1].plot(x2, derivNotCleaned, label = 'Raw gradient')
            axis[1].plot(x2, deriv, label = 'Cleaned gradient')
            axis[1].plot(x2[maxIndex], deriv[maxIndex], 'o', color = 'purple', label = "Max gradient point")
            axis[1].plot(x2[minIndex], deriv[minIndex], 'o', color = 'magenta')
            axis[1].set_title("Value of its gradient")
            axis[1].legend(loc="best")
            axis[1].grid()
    
            #axis.set_xlim([0.25, 0.3])
            fig.tight_layout(pad=2.0)
            plt.setp(axis[1], xlabel='Time [s]')
            plt.setp(axis, ylabel='Magnitude')
            plt.show()
            plt.close()
        
        x_int_index = int(x_inters)
        bpOnsetsindex.append(x_int_index)
        bpOnsetstime.append(x_int_index/bpGraph._samplerate)
        bpOnsetsY.append(y_inters)
   
    return [bpOnsetsindex, bpOnsetstime], bpOnsetsY # Index then time then Y

# Computation of the onset points with the second derivative method
def getBpOnsets_2dDeriv(bp, rPeaks, lims =[], show = False):
    
    limSamples = []
    if len(lims) > 0 :
        for item in lims: # Go through the 2 items
            limSamples.append(item*bp._samplerate)
        x = np.arange(min(lims), max(lims), bp._step)
        
    else :
        x = np.arange(bp._x[0], bp._x[-1], bp._step)
    
    
    # We stock the ranges of indexes inside x_min and x_max
    x_min = []
    x_max = []    
    for i in range(len(rPeaks[0])-1):
        if rPeaks[0][i] >= int(min(x)*bp._samplerate) and rPeaks[0][i+1]-10 <= int(max(x)*bp._samplerate):
            x_min.append(rPeaks[0][i])
            x_max.append(rPeaks[0][i+1]-10)
    
    bpOnsetsindex = []
    bpOnsetstime = []
    bpOnsetsY = []
    
    
    # For each R peak :
    for k in range(len(x_min)):
        limMin = x_min[k]
        limMax = x_max[k]

        x2 = np.arange(limMin, limMax)
        firstGraph = bp._y[limMin:limMax]
    
        # Compute first derivative
        slope_Y = np.diff(firstGraph)/np.diff(x2)
        slope_Y = np.append(slope_Y, slope_Y[-1])
        
        # Clean first derivative with butterworth filter
        b, a = signal.butter(3, 0.05)
        zi = signal.lfilter_zi(b, a)
        z, _ = signal.lfilter(b, a, slope_Y, zi=zi*slope_Y[0])
        z2, _ = signal.lfilter(b, a, z, zi=zi*z[0])
        clean_slope_Y = signal.filtfilt(b, a, slope_Y)
        
    
        # Compute second derivative
        slope_YY = np.diff(clean_slope_Y)/np.diff(x2)
        slope_YY = np.append(slope_YY, slope_YY[-1])
        
        # Clean second derivative with butterworth filter
        b, a = signal.butter(3, 0.05)
        zi = signal.lfilter_zi(b, a)
        z, _ = signal.lfilter(b, a, slope_YY, zi=zi*slope_YY[0])
        z2, _ = signal.lfilter(b, a, z, zi=zi*z[0])
        clean_slope_YY = signal.filtfilt(b, a, slope_YY)
        
        
        # Locate the max first derivative
        maxIndex = np.argmax(clean_slope_Y)    
        minIndex = maxIndex-30
        searchZone = clean_slope_YY[minIndex:maxIndex]
        
        
        


        # Locate the max locale second derivative
        max2d_Index = np.argmax(searchZone)+minIndex

        if show == True :
            plt.figure()
            fig, axis = plt.subplots(3, 1, figsize=(15, 8), sharex=True)
            axis[0].plot(x2, firstGraph, label = bp._title)
            axis[0].plot(x2[max2d_Index], firstGraph[max2d_Index], 'o', color = 'red', label = "Bump onset")
            axis[0].set_title("Graph number " + str(k))
            axis[0].set_ylim([np.min(firstGraph)-0.5*(np.mean(firstGraph)-np.min(firstGraph)),np.max(firstGraph)+0.5*(np.max(firstGraph)-np.mean(firstGraph))])
            axis[0].legend(loc="best")
            axis[0].grid()
    
            axis[1].plot(x2, slope_Y, label = 'Raw gradient')
            axis[1].plot(x2, clean_slope_Y, label = 'Cleaned gradient')
            axis[1].set_title("Value of its first gradient")
            axis[1].legend(loc="best")
            axis[1].grid()
    
            axis[2].plot(x2, slope_YY, label = 'Raw gradient (from cleaned)', color = 'orange')
            axis[2].plot(x2[maxIndex], slope_YY[maxIndex], 'o', color = 'blue', label = "End of search zone")
            axis[2].plot(x2[minIndex], slope_YY[minIndex], 'o', color = 'green', label = "End of search zone")
            axis[2].plot(x2[max2d_Index], slope_YY[max2d_Index], 'o', color = 'red', label = "Max locale")
            axis[2].set_title("Value of its second gradient")
            axis[2].legend(loc="best")
            axis[2].grid()
        
            fig.tight_layout(pad=2.0)
            plt.setp(axis[1], xlabel='Time [s]')
            plt.setp(axis, ylabel='Magnitude')
            plt.show()
            plt.close()
        
        bpOnsetsindex.append(max2d_Index+limMin)
        bpOnsetstime.append((max2d_Index+limMin)/bp._samplerate)
    
    return [bpOnsetsindex, bpOnsetstime]
    
#%% Mean graphs
          
# Show the mean bp aligned with their onsets
def showBPmean(bpGraph, indexTime, timelim = None):

    if timelim:
        temp = []
        temptime = []
        for peak in indexTime[1]:
            if peak > timelim[0] and peak < timelim[1]:
                temp.append(int(peak*bpGraph._samplerate))
                temptime.append(peak)
        indexTime = [temp, temptime]
        
    diff = []
    for i in range(len(indexTime[0])-1):
        diff.append(indexTime[0][i+1] - indexTime[0][i])
    
    mean_diff = int(np.mean(diff))
    
    # We stock the ranges of indexes inside x_min and x_max
    x_min = []
    x_max = []
    for i in range(len(indexTime[0])-1):
        x_min.append(indexTime[0][i]-int(0.1*mean_diff))
        x_max.append(indexTime[0][i]+int(0.9*mean_diff))
        #print(x_max[i]-x_min[i])

    

    x = np.arange(x_min[0], x_max[0])
    x = x-x[int(0.1*mean_diff)]
    x = x/bpGraph._samplerate


    for i in range(len(x_min)):
        plt.plot(x, bpGraph._y[x_min[i]:x_max[i]], color = 'grey', alpha = 0.3)
        
        
        
    plt.title("Mean "+bpGraph._title)
    plt.grid()
    plt.xlabel("Time [s]")
    plt.ylabel("Magnitude")
    plt.show()
    plt.close()

# Show the mean ecg aligned on the R peaks
def showECGmean(ecg, pPeaks, qPeaks, rPeaks, sPeaks, tPeaks, pOnsets, tOffsets, timelim = None):
    
    diff = []
    if timelim:
        temp = []
        temptime = []
        for peak in rPeaks[1]:
            if peak > timelim[0] and peak < timelim[1]:
                temp.append(int(peak*ecg._samplerate))
                temptime.append(peak)
        rPeaks = [temp, temptime]
    

    for i in range(len(rPeaks[0])-1):
        diff.append(rPeaks[0][i+1] - rPeaks[0][i])

    mean_diff = int(np.mean(diff))
    
    # We stock the ranges of indexes inside x_min and x_max
    x_min = []
    x_max = []
    
    for i in range(len(rPeaks[0])-1):
        x_min.append(rPeaks[0][i]-int(0.4*mean_diff))
        x_max.append(rPeaks[0][i]+int(0.4*mean_diff))
        #print(x_max[i]-x_min[i])
    
    
    x = np.arange(x_min[0], x_max[0])   
    x = x-x[int(0.4*mean_diff)]
    x = x/ecg._samplerate
    

    for i in range(len(x_min)):
        thisECG = ecg._y[x_min[i]:x_max[i]]
        plt.plot(x, thisECG, color = 'grey', alpha = 0.3)
        for a in pPeaks[0]:
            if a < x_max[i] and a > x_min[i]:
                relativeAbs = a - x_min[i]
                plt.plot(x[relativeAbs], thisECG[relativeAbs], 'o', color = 'red', alpha = 0.3)
        for a in qPeaks[0]:
            if a < x_max[i] and a > x_min[i]:
                relativeAbs = a - x_min[i]
                plt.plot(x[relativeAbs], thisECG[relativeAbs], 'o', color = 'blue', alpha = 0.3)
        for a in sPeaks[0]:
            if a < x_max[i] and a > x_min[i]:
                relativeAbs = a - x_min[i]
                plt.plot(x[relativeAbs], thisECG[relativeAbs], 'o', color = 'green', alpha = 0.3)
        for a in tPeaks[0]:
            if a < x_max[i] and a > x_min[i]:
                relativeAbs = a - x_min[i]
                plt.plot(x[relativeAbs], thisECG[relativeAbs], 'o', color = 'purple', alpha = 0.3)
        for a in rPeaks[0]:
            if a < x_max[i] and a > x_min[i]:
                relativeAbs = a - x_min[i]
                plt.plot(x[relativeAbs], thisECG[relativeAbs], 'o', color = 'orange', alpha = 0.3)
    
    legend_elements = [Line2D([0], [0], marker='o', color='red', label='P peaks', alpha = 0.3),
                       Line2D([0], [0], marker='o', color='blue', label='Q peaks', alpha = 0.3),
                       Line2D([0], [0], marker='o', color='orange', label='R peaks', alpha = 0.3),
                       Line2D([0], [0], marker='o', color='green', label='S peaks', alpha = 0.3),
                       Line2D([0], [0], marker='o', color='purple', label='T peaks', alpha = 0.3)]
    
    
    plt.title("Mean "+ecg._title)
    plt.legend(handles=legend_elements, loc = 'best')
    plt.grid()
    plt.xlabel("Time [s]")
    plt.ylabel("Magnitude")
    plt.show()
    plt.close()

#%%

def getStartStopIndexes(ecg_JSON, ecg_MLd):
    # Compare the 2 files
    
    plt.plot(ecg_JSON._y)
    plt.show()
    plt.plot(ecg_MLd._y)
    plt.show()    
    
    plt.figure(figsize=(10, 20))
    fig, axis = plt.subplots(2, 1, figsize=(15, 12))

    # ECG
    axis[0].plot(ecg_JSON._x, ecg_JSON._y)
    axis[0].set_title(ecg_JSON._title+" JSON")
    axis[0].legend(loc="lower right")
    axis[0].grid()

    # xLin
    axis[1].plot(ecg_MLd._x, ecg_MLd._y)
    axis[1].set_title(ecg_MLd._title + " MatLab")
    axis[1].legend(loc="lower right")
    axis[1].grid()

    fig.tight_layout(pad=2.0)
    plt.setp(axis, xlabel='Time [s]')
    plt.setp(axis, ylabel='Magnitude')
    plt.show()
    plt.close()

    # Select the shortest file :
    if len(ecg_JSON._x) > len(ecg_MLd._x):
        smallestLen = ecg_MLd
        biggestLen = ecg_JSON
    else :
        smallestLen = ecg_JSON
        biggestLen = ecg_MLd


    #print(biggestLen._samplerate, smallestLen._samplerate)
    difference = len(biggestLen._y) - len(smallestLen._y)

    # Create a temporary 2d signal of same dimensions :
    temp = np.zeros(difference)
    temp = np.append(temp, smallestLen._y)



    fig, axis = plt.subplots(2, 1, figsize=(15, 12))

    # ECG
    axis[0].set_title("Signals to be correlated")
    axis[0].plot(temp, label = "Signal 2")
    axis[0].plot(biggestLen._y, label = "Signal 1")
    axis[0].legend(loc="lower right")
    axis[0].grid()

    a = sc.correlate(biggestLen._y, temp, mode='full')

    # xLin
    axis[1].plot(a)
    axis[1].set_title("Value of cross-correlation")
    axis[1].grid()

    fig.tight_layout(pad=2.0)
    plt.setp(axis, ylabel='Magnitude')
    plt.setp(axis, xlabel='Samples')
    plt.show()
    plt.close()


    indexMaxCorr = np.argmax(a)
    #print(indexMaxCorr)
    # Il suffit de décaler le 2e signal de indexMaxCorr = 14599 depuis la droite pour avoir l'alignement correct

    #fin du plus signal le plus court : index 14599
    longest_reshaped = biggestLen._y[indexMaxCorr-len(smallestLen._y):indexMaxCorr]
    #print(type(longest_reshaped))


    plt.figure(figsize=(20, 10))
    plt.title("Alignment visualisation")
    plt.plot(smallestLen._x, smallestLen._y, label = "ECG_shortest")
    plt.plot(smallestLen._x, longest_reshaped, label = "ECG_longest")
    plt.grid()
    plt.setp(axis, xlabel='Time [s]')
    plt.setp(axis, ylabel='Magnitude')
    plt.show()
    plt.close()
    
    return [indexMaxCorr-len(smallestLen._y), indexMaxCorr]