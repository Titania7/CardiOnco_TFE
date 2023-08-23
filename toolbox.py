# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 11:17:56 2023

@author: Tatiana Dehon

 ========= Toolbox for common routines =========
"""

from SingleGraph import SingleGraph
from sklearn.neighbors import LocalOutlierFactor

import neurokit2 as nk
import pandas as pd
import math
from scipy.signal import hilbert

import matplotlib.pyplot as plt

import numpy as np
from numpy.fft import *


from scipy import signal
import scipy.signal as sc
from scipy.signal import freqz
from scipy.fft import rfft, rfftfreq



#%% Small useful functions

def getBegEnd_bpArm(bpSC : SingleGraph, show = False):
    
    slope_Y = np.diff(bpSC._y)/np.diff(bpSC._x)
    slope_Y = np.append(slope_Y, slope_Y[-1])
    
    maxDeriv = np.max(slope_Y)
    minDeriv = np.min(slope_Y)
    meanDeriv = np.mean(slope_Y)
    
    minValy = minDeriv/6
    maxValy = maxDeriv/6
    
    plt.plot(bpSC._x, slope_Y)
    plt.axvspan(0.4*max(bpSC._x), max(bpSC._x), color = "orange", alpha = 0.3)
    plt.axhspan(minValy, maxValy, color = "orange", alpha = 0.3)
    plt.show()
    
    offset_index = int(0.4*max(bpSC._x)*bpSC._samplerate)
    search = slope_Y[offset_index:]
    
    cleanSearch_min = []
    cleanSearch_max = []
    for index, item in enumerate(search):
        if item < minValy:
            cleanSearch_min.append(item)
        elif item > maxValy:
            cleanSearch_max.append(item)
        else :
            cleanSearch_min.append(np.mean(slope_Y))
            cleanSearch_max.append(np.mean(slope_Y))
    
    #  + offset_index
    
    minimum = sc.argrelmin(np.array(cleanSearch_min), order = int(2*bpSC._samplerate))
    maximum = sc.argrelmax(np.array(cleanSearch_max), order = int(2*bpSC._samplerate))
    
    minimum = np.squeeze(minimum)
    maximum = np.squeeze(maximum)
    
    
    try :
        minimum = minimum[-1]+offset_index
    except:
        minimum += offset_index
    try :
        maximum = maximum[-1]+offset_index
    except:
        maximum += offset_index
    
    if show == True:

        
        plt.plot(bpSC._x, slope_Y)
        plt.plot(bpSC._x[minimum], slope_Y[minimum], "o")
        plt.plot(bpSC._x[maximum], slope_Y[maximum], "o")
        plt.axvspan(0.4*max(bpSC._x), max(bpSC._x), color = "orange", alpha = 0.3)
        plt.axhspan(minValy, maxValy, color = "orange", alpha = 0.3)
        plt.plot(bpSC._x[minimum], slope_Y[minimum])
        plt.plot(bpSC._x[maximum], slope_Y[maximum])
        plt.show()
        
        plt.plot(bpSC._x, bpSC._y)
        plt.plot(bpSC._x[minimum], bpSC._y[minimum], "o", label = "Begin = "+str(minimum*bpSC._step))
        plt.plot(bpSC._x[maximum], bpSC._y[maximum], "o", label = "End = "+ str(maximum*bpSC._step))
        plt.title("Check good detection")
        plt.legend(loc = "best")
        plt.show()

    return[minimum, minimum*bpSC._step], [maximum, maximum*bpSC._step]

def index_conv(index : int, scg_fs, ecg_fs, indexType : str = None):
    """
    Function that converts an scg/ecg index into the other with different sampling rate for the same time.
    
    ----
    Inputs
    ----
    - index : integer representing the index to be translated
    - scg_fs : int or float being the sampling rate of the SCG
    - ecg_fs : int or float being the sampling rate of the ECG
    - indexType : "Scg2ecg" or "Ecg2scg" strings that choose the conversion
    
    ----
    Outputs
    ----
    - converted_index : the converted index (integer)
    """
    if indexType:
        #print("Before conversion : index =", index, "scg_fs =", scg_fs, "ecg_fs =", ecg_fs)
        if indexType == "Scg2ecg":
            ecg_index = index * ecg_fs / scg_fs
            converted_index = int(np.round(ecg_index))
        elif indexType == "Ecg2scg":
            
            scg_index = index * scg_fs / ecg_fs
            converted_index = int(np.round(scg_index))
        #print("After conversion : index =", converted_index)
    return converted_index

def getStartStopIndexes(ecg_JSON, ecg_MLd, minHP = False, show = False):
    
    """
    This function allows to synchronize the 2 ECGs json and matlab in order to give the Matlab's begin and end indexes for
    a synchronous display.

    ----
    Inputs
    ----
    - ecg_JSON : SingleGraph class of the ECG from the JSON file
    - ecg_MLd : SingleGraph class of the ECG from the Matlab file
    - show : boolean to say if we want to see the alignment process or not (defaults to False)
    
    ----
    Outputs
    ----
    - [start_index, stop_index] : list of integers containing the start and stop indexes of the ECG for the MatLab file
        (MatLab file is usually the longest)
    """
    print("Hello World")

    # Select the shortest file in NUMBER OF SAMPLES :
    if len(ecg_JSON._x) > len(ecg_MLd._x):
        smallestLen = ecg_MLd
        biggestLen = ecg_JSON
    else :
        smallestLen = ecg_JSON
        biggestLen = ecg_MLd

    
    print("Biggest len and smallest len :",biggestLen._title, len(biggestLen._y), biggestLen._samplerate, smallestLen._title, len(smallestLen._y), smallestLen._samplerate)
    difference = len(biggestLen._y) - len(smallestLen._y)

    # Create a temporary 2d signal of same dimensions :
    temp = np.zeros(difference)
    temp = np.append(temp, smallestLen._y)

    a = sc.correlate(biggestLen._y, temp, mode='full')
    
    x = np.arange(0, len(a), 1)
    aSG = SingleGraph(x, a, samplerate=1, step = 1)
    # Clean the a from its low frequency variations to avoid the V shapes
    if minHP :
        a = butterHighPass(aSG, filtorder=3, limfreq=0.05)
    
    # Create a search Window for the max of correlation in a in case the mean graph looks like a V
    # in which case => max correlation too soon of too late can happen
    # From 20% len(a) to 80% len(a)
    minSearch_a = int(0.15*len(a))
    maxSearch_a = int(0.75*len(a))
    
    
    indexMaxCorr = np.argmax(a[minSearch_a:maxSearch_a])+minSearch_a
    #print("IndexMaxCorr = ", indexMaxCorr) # 140

    # Il suffit de décaler le 2e signal de indexMaxCorr depuis la droite pour avoir l'alignement correct

    #fin du plus signal le plus court : index 14599
    longest_reshaped = biggestLen._y[indexMaxCorr-len(smallestLen._y):indexMaxCorr]
    print("This is the length of the longest reshaped", len(longest_reshaped)/biggestLen._samplerate)
    
    # Compare the 2 files
    if show == True :
        plt.plot(ecg_JSON._y)
        plt.show()
        plt.plot(ecg_MLd._y)
        plt.show()    
        

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
        
        fig, axis = plt.subplots(2, 1, figsize=(15, 12))

        # ECG
        axis[0].set_title("Signals to be correlated")
        axis[0].plot(temp, label = "Signal 2")
        axis[0].plot(biggestLen._y, label = "Signal 1")
        axis[0].legend(loc="lower right")
        axis[0].grid()
        
        # xLin
        axis[1].plot(aSG._y, label = "Noisy cross-correlation")
        axis[1].set_title("Value of cross-correlation")
        axis[1].plot(a, label = "Cleaned cross-correlation")
        axis[1].axvspan(minSearch_a, maxSearch_a, alpha=0.3, color='yellow', label = "Search zone for the correlation peak")
        axis[1].legend(loc = "best")
        axis[1].grid()

        fig.tight_layout(pad=2.0)
        plt.setp(axis, ylabel='Magnitude')
        plt.setp(axis, xlabel='Samples')
        plt.show()
        plt.close()
        
        plt.figure(figsize=(20, 10))
        plt.title("Alignment visualisation")
        plt.plot(smallestLen._x, smallestLen._y, label = "ECG_shortest")
        plt.plot(smallestLen._x, longest_reshaped, label = "ECG_longest")
        plt.grid()
        plt.setp(axis, xlabel='Time [s]')
        plt.setp(axis, ylabel='Magnitude')
        plt.show()
        plt.close()

        
    print(biggestLen._title)
    return biggestLen, [indexMaxCorr-len(smallestLen._y), indexMaxCorr]

def butterCutPass(graph : SingleGraph, filtorder : int, limfreqs : list, show=False):
    
    # Retrieving of Fourier components of the raw signal
    yf = rfft(graph._y)
    xf = rfftfreq(len(graph._x), graph._step)
    
    # Creation of the Butterworth bandpass filter
    lowcut = limfreqs[0] # 30bpm => 0.5 Hz
    highcut = limfreqs[1] # 250 bpm => ~ 4.17 Hz
    
    #lowcut = bpm_th-0.5
    #highcut = bpm_th+0.5
    order = filtorder
    fs = graph._samplerate
    b, a = sc.butter(N=order, Wn=[lowcut, highcut], btype='bandpass', fs=fs)
    w, h = freqz(b, a, fs=fs, worN=len(xf))
    
    # Cleaning of the signal -> y
    zi = signal.lfilter_zi(b, a)
    z, _ = signal.lfilter(b, a, graph._y, zi=zi*graph._y[0])
    z2, _ = signal.lfilter(b, a, z, zi=zi*z[0])
    y = signal.filtfilt(b, a, graph._y)
    
    """
    # Visualization of the Butterworth BP filter and the raw signal spectrum
    fig, axis = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain')
    
    axis[0].set_title("Frequency content of "+graph._title)
    axis[0].plot(xf, np.abs(yf), label = "Raw signal spectrum")
    axis[0].plot(w, abs(h), label="BP filter - order = %d" % order)
    axis[0].legend(loc="best")
    axis[0].grid()
    
    y_cleaned_fourier = np.abs(rfft(y))
    axis[1].plot(xf, y_cleaned_fourier, label = "Cleaned signal spectrum")
    axis[1].plot(w, abs(h), label="BP filter - order = %d" % order)
    axis[1].legend(loc="best")
    axis[1].grid()
    
    axis[0].set_xlim([bpm_th*0.5, bpm_th*1.5])
    axis[0].set_ylim([0, np.max(y_cleaned_fourier)*1.25])
    plt.show()
    
    print("Max freq of cleaned Fourier  = ", np.argmax(y_cleaned_fourier)*graph._step)
    """
    
    # Visualization of the cleaned signal    
    if show == True :
        plt.figure(figsize=(10, 6))
        plt.title("Cleaning of "+graph._title)
        plt.plot(graph._x, graph._y, 'grey', alpha = 0.5, label = "Raw signal")
        plt.plot(graph._x, y, label = "Cleaned signal")
        plt.legend(loc='best')
        plt.grid()
        plt.show()

    return y

def butterLowPass(graph : SingleGraph, filtorder : int, limfreq : float, show=False):
    
    # Retrieving of Fourier components of the raw signal
    yf = rfft(graph._y)
    xf = rfftfreq(len(graph._x), graph._step)
    fs = graph._samplerate
    
    b, a = sc.butter(N=filtorder, Wn=limfreq, btype='lowpass', fs=fs)
    w, h = freqz(b, a, fs=fs, worN=len(xf))
    
    # Cleaning of the signal -> y
    zi = signal.lfilter_zi(b, a)
    z, _ = signal.lfilter(b, a, graph._y, zi=zi*graph._y[0])
    z2, _ = signal.lfilter(b, a, z, zi=zi*z[0])
    y = signal.filtfilt(b, a, graph._y)    

    
    # Visualization of the cleaned signal    
    if show == True :
        plt.figure(figsize=(10, 6))
        plt.title("Cleaning of "+graph._title)
        plt.plot(graph._x, graph._y, 'grey', alpha = 0.5, label = "Raw signal")
        plt.plot(graph._x, y, label = "Cleaned signal")
        plt.legend(loc='best')
        plt.grid()
        plt.show()

    return y

def butterHighPass(graph : SingleGraph, filtorder : int, limfreq : float, show=False):
    
    # Retrieving of Fourier components of the raw signal
    yf = rfft(graph._y)
    xf = rfftfreq(len(graph._x), graph._step)
    fs = graph._samplerate
    
    b, a = sc.butter(N=filtorder, Wn=limfreq, btype='highpass', fs=fs)
    w, h = freqz(b, a, fs=fs, worN=len(xf))
    
    # Cleaning of the signal -> y
    zi = signal.lfilter_zi(b, a)
    z, _ = signal.lfilter(b, a, graph._y, zi=zi*graph._y[0])
    z2, _ = signal.lfilter(b, a, z, zi=zi*z[0])
    y = signal.filtfilt(b, a, graph._y)    

    
    # Visualization of the cleaned signal    
    if show == True :
        plt.figure(figsize=(10, 6))
        plt.title("Cleaning of "+graph._title)
        plt.plot(graph._x, graph._y, 'grey', alpha = 0.5, label = "Raw signal")
        plt.plot(graph._x, y, label = "Cleaned signal")
        plt.legend(loc='best')
        plt.grid()
        plt.show()

    return y

def cleanLOF(allVect, n_neighbors = 10, contamination = 0.5, show=False):
    
    minLen = np.min([len(vect) for vect in allVect])

    
    for i, item in enumerate(allVect):
        while len(allVect[i]) > minLen :
            allVect[i] = np.delete(allVect[i], -1)
    
    # make sure the data is an array
    data = np.array(allVect) #74x77 : 74 vectors of length 77
    
    # Processing
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    outlier_scores = lof.fit_predict(data)
    outliers_indices = np.where(outlier_scores == -1)[0]
    
    
    # Delete the arrays that are detected as too different from the others
    filtered_allVect = [allVect[i] for i in range(len(allVect)) if i not in outliers_indices]
    
    if show == True:
        oldMean = np.mean(data, axis=0)
        newMean = np.mean(filtered_allVect, axis=0)
        for i, graph in enumerate(data):
            if i == 0:
                plt.plot(graph, color = "grey", alpha = 0.3, label = "Original list : "+str(len(data))+ " vectors")
            else:
                plt.plot(graph, color = "grey", alpha = 0.3)
        for i, graph in enumerate(filtered_allVect):
            if i == 0:
                plt.plot(graph, color = "cyan", alpha = 0.3, label = "Cleaned list : "+str(len(filtered_allVect))+ " vectors")
            else:
                plt.plot(graph, color = "cyan", alpha = 0.3)
        
        plt.plot(oldMean, color = "k", label = "Original mean graph")
        plt.plot(newMean, color = "blue", label="Cleaned mean graph")
        
        plt.legend(loc = "best")
        plt.show()
    return filtered_allVect, outliers_indices

#%% Analysis of signals : QRS and BP onsets

def detect_qrs(ecgSignal, clean_technique, ecg_delineate, lims = [], show = False):

    """
    # Detection of the QRS complexes
    currentGraph is an ECG signal
    """
    
    #print("===== Signal : ===== \n", ecgSignal, '\nSelected time :', lims, "\n===============")
    
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
    
    
    """
    signal = np.array([monECG])
    sfreq = freq_echantillonnage_monECG
    """
    
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
    #%%
    

    
    return [p_peaks_indexes, p_peaks_time], [q_peaks_indexes, q_peaks_time], [r_peaks_indexes, r_peaks_time], [s_peaks_indexes, s_peaks_time], [t_peaks_indexes, t_peaks_time], [p_onsets_indexes, p_onsets_time], [t_offsets_indexes, t_offsets_time]

def getBpOnsets_tang(bpGraphList, fs, title, filt = False, show = False):
    
    """
    # Computation of the onset points with the intersecting tangents method

    """
    bpOnsetsindex = []
    bpOnsetstime = []
    
    x = np.arange(0, len(bpGraphList[0])/fs, 1/fs)
    while len(x)>len(bpGraphList[0]):
        x = np.delete(x, -1)
    while len(x)<len(bpGraphList[0]):
        x = np.append(x, x[-1]+1/fs)
    
    for k, bpGraph in enumerate(bpGraphList):

        # Compute first derivative
        slope_Y = np.diff(bpGraph)/np.diff(x)
        # Add a the last value twice to deriv => same shape as firstGraph
        derivNotCleaned = np.append(slope_Y, slope_Y[-1])

        
        if filt == True:
            # Use of a butterworth filter to cut out the noise
            b, a = signal.butter(5, 0.1)
            zi = signal.lfilter_zi(b, a)
            z, _ = signal.lfilter(b, a, derivNotCleaned, zi=zi*derivNotCleaned[0])
            z2, _ = signal.lfilter(b, a, z, zi=zi*z[0])
            y = signal.filtfilt(b, a, derivNotCleaned)
            
            # Cleaned signal is stored in "deriv" variable
            deriv = y
        else:
            deriv = derivNotCleaned     
    
    
    
        # Detection of the min value in firstGraph and max value in deriv
        relmaxValIndex = np.argmax(bpGraph)
        
        print("The rel max value index is = to ... ", relmaxValIndex)
        if relmaxValIndex >1:
            searchZone = bpGraph[:relmaxValIndex]
            
            minIndex = np.argmin(searchZone)
            #maxIndex = np.argmax(deriv)
            
            local_maxima = np.array(sc.argrelextrema(deriv, np.greater_equal, order = int(0.05*fs)))
            local_maxima = local_maxima.reshape([local_maxima.shape[1],])

            if len(local_maxima) == 1 :
                maxIndex = local_maxima[0]
            else :
                maxIndex = local_maxima[1]
        
            # First tangent calculation :
            valy = searchZone[minIndex]
            valx = x[minIndex]
            #amin = deriv[minIndex]
            # We force the derivative of the minumum to zero
            amin = 0
            bmin = valy-amin*valx
            tgMin = amin*x+bmin
            # Second tangent calculation :
            valy = bpGraph[maxIndex]
            valx = x[maxIndex]
            amax = deriv[maxIndex]
            bmax = valy-amax*valx
            tgMaxDeriv = amax*x+bmax
            # Determincation of the intersection point :
            x_inters = (bmax-bmin)/(amin-amax)
            y_inters = amin*x_inters+bmin
    
            # Visualisation of the intersection with signal cleaned :
            if show == True :
                plt.figure()
                fig, axis = plt.subplots(2, 1, figsize=(15, 8), sharex=True)
                axis[0].plot(x, bpGraph, label = title)
                axis[0].plot(x[minIndex], bpGraph[minIndex], 'o', color = 'magenta', label = "Min pressure point")
                axis[0].plot(x, tgMin, '--',color = 'magenta', label = 'Min pressure tangent')
                axis[0].plot(x[maxIndex], bpGraph[maxIndex], 'o', color = 'purple')
                axis[0].plot(x, tgMaxDeriv, '--',color = 'purple', label = 'Max gradient tangent')
                axis[0].set_title("Graph number "+ str(k))
                axis[0].set_ylim([np.min(bpGraph)-0.5*(np.mean(bpGraph)-np.min(bpGraph)),np.max(bpGraph)+0.5*(np.max(bpGraph)-np.mean(bpGraph))])
                axis[0].plot(x_inters, y_inters, 'o', color = 'red', label = 'Onset of the pressure bump')
                axis[0].legend(loc="best")
            
                axis[0].grid()
                axis[1].plot(x, derivNotCleaned, label = 'Raw gradient')
                axis[1].plot(x, deriv, label = 'Cleaned gradient')
                axis[1].plot(x[maxIndex], deriv[maxIndex], 'o', color = 'purple', label = "Max gradient point")
                axis[1].plot(x[minIndex], deriv[minIndex], 'o', color = 'magenta')
                axis[1].set_title("Value of its gradient")
                axis[1].legend(loc="best")
                axis[1].grid()
        
                #axis.set_xlim([0.25, 0.3])
                fig.tight_layout(pad=2.0)
                plt.setp(axis[1], xlabel='Time [s]')
                plt.setp(axis, ylabel='Magnitude')
                plt.show()
                plt.close()
            
            if x_inters > 0:
                bpOnsetstime.append(x_inters)
                bpOnsetsindex.append(int(np.round(x_inters*fs)))
        
        
    return [bpOnsetsindex, bpOnsetstime]# Index then time then Y

def getBpOnsets_2dDeriv(bp, rPeaks, lims =[], show = False):
    
    """
    # Computation of the onset points with the second derivative method
    """
    
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

#%% AO points detection from articles

def jafaritadi2015(cleanAllLinGraphs, fs, show = False):
    """
    cleanAllLinGraphs must be a list of arrays fron allLinSCG. 
    Each array is a SCG graph between a R-R interval.
    All the arrays must be the same length.
    Works better if the SCG signal has been cleaned before.
    """
    
    step = 1/fs
    
    time = np.arange(0, len(cleanAllLinGraphs[0])*step, step)
    while len(time)>len(cleanAllLinGraphs[0]):
        time = np.delete(time, -1)

    ao_90rel = np.array([])
    for vect in cleanAllLinGraphs:
        
        # Cut the frequencies between 4-40 Hz
        graph = SingleGraph(time, vect, title = "", samplerate=fs, step=step)
        vect = butterCutPass(graph, filtorder=3, limfreqs=[4,40], show = False)
        
        # Search each max peak in the R-90 ms interval (assumed to be AO)
        maxSearch = 0.09 #90 ms
        local_maxima = np.array(sc.argrelextrema(vect[:int(np.round(maxSearch*fs))], np.greater_equal, order = 2))
        local_maxima = local_maxima.reshape([local_maxima.shape[1],])
        # Select the last peak detected as AO
        ao = local_maxima[-1]
        
        if show == True :
            plt.plot(time, vect)
            plt.plot(time[ao], vect[ao], "o", color = "orange", label="AO (method R-90 ms)")
            plt.axvspan(0.0,0.090, alpha=0.3, color='orange', label="Search zone R-90 ms")
            plt.legend(loc = "best")
            plt.show()
        
        
        ao_90rel = np.append(ao_90rel, ao)

    ao_90rel = ao_90rel.astype(int)
    return ao_90rel
    

def khosrow_khavar2015(cleanAllLinGraphs, fs, show = False):
    step = 1/fs
    
    time = np.arange(0, len(cleanAllLinGraphs[0])*step, step)
    while len(time)>len(cleanAllLinGraphs[0]):
        time = np.delete(time, -1)

    ao_HFenvelope = np.array([])
    for vect in cleanAllLinGraphs:
        # Cut the frequencies between 4-40 Hz
        graph = SingleGraph(time, vect, title = "", samplerate=fs, step=step)
        vectLF = butterLowPass(graph, filtorder=5, limfreq=30, show = False)
        vectHF = butterHighPass(graph, filtorder=5, limfreq=20, show = False)
    
        envHF = np.abs(hilbert(vectHF))
        
        maxSearch = 0.2 #s
        peak_env = np.argmax(envHF[:int(np.round(maxSearch*fs))])
        
        windowSearch = int(np.round(0.025*fs)) #ms
        ao = np.argmax(vect[peak_env-windowSearch:peak_env+windowSearch])
        ao = ao+peak_env-windowSearch
        
        
        if show == True :
            plt.plot(time, vect, label = "Raw ACC")
            plt.plot(time[ao], vect[ao],'o', color = "blue", label = "AO")
            #plt.plot(time, vectLF, label = "LFACC")
            plt.plot(time, vectHF, label = "HFACC")
            plt.plot(time, envHF, color = "red", label = "HFENV")
            plt.plot(time[peak_env], envHF[peak_env], 'o', color = "red", label = "HFACC peak")
            plt.axvspan(0.0,0.2, alpha=0.3, color='yellow', label="Search zone HFACC peak")
            plt.axvspan(time[peak_env] - 0.025, time[peak_env]+0.025, alpha=0.5, color='lightblue', label="Where is AO")
            plt.legend(loc = "best")
            plt.show()
        
        ao_HFenvelope = np.append(ao_HFenvelope, ao)
    ao_HFenvelope = ao_HFenvelope.astype(int)
    return ao_HFenvelope

def yang2017(cleanXrot, cleanYrot, cleanZrot, zLin, fs, show = False):
    
    step = 1/fs
    
    # make them all the same length
    minsize = np.min([len(vectCollection) for vectCollection in [cleanXrot, cleanYrot, cleanZrot]])
    for vectCollection in [cleanXrot, cleanYrot, cleanZrot]:
        while len(vectCollection)>minsize:
            vectCollection.pop()
    
    
    time = np.arange(0, len(cleanXrot[0])*step, step)
    while len(time)>len(cleanXrot[0]):
        time = np.delete(time, -1)
    
    ao_DSProtation = np.array([])
    i_x = 11.5
    i_y = 21.4
    i_z = 1
    
    for i in range(len(cleanXrot)):
        
        graphX = SingleGraph(time, cleanXrot[i], title = "", samplerate=fs, step=step)
        vectX = butterCutPass(graphX, filtorder=3, limfreqs=[0.8,25], show = False)
        graphY = SingleGraph(time, cleanYrot[i], title = "", samplerate=fs, step=step)
        vectY = butterCutPass(graphY, filtorder=3, limfreqs=[0.8,25], show = False)
        graphZ = SingleGraph(time, cleanZrot[i], title = "", samplerate=fs, step=step)
        vectZ = butterCutPass(graphZ, filtorder=3, limfreqs=[0.8,25], show = False)

        kinEnergy = 0.5*(i_x*vectX**2 + i_y*vectY**2 + i_z*vectZ**2)   
        kE_peak = np.argmax(kinEnergy[:int(np.round(0.11*fs))])
        
        searchWin = 0.025 #s
        ao_DSProt = np.argmax(zLin[i][kE_peak-int(np.round(searchWin*fs)):kE_peak+int(np.round(searchWin*fs))])
        ao = ao_DSProt + kE_peak-int(np.round(searchWin*fs))
        
        if show == True:
            plt.plot(time, 0.5*i_x*vectX**2, label = "x component")
            plt.plot(time, 0.5*i_y*vectY**2, label="y component")
            plt.plot(time, 0.5*i_z*vectZ**2, label="z component");
            plt.plot(time, kinEnergy, label="kinetic energy")
            #plt.axvspan(kinEnergy*step-searchWin, kinEnergy*step+searchWin, alpha=0.3, color='orange', label="Search zone AO peak")
            plt.legend(loc = "best")
            plt.show()
            
            plt.plot(time, zLin[i])
            plt.plot(time[ao], zLin[i][ao], "o", color = "red")
            #plt.axvspan(kinEnergy*step-searchWin, kinEnergy*step+searchWin, alpha=0.3, color='orange', label="Search zone AO peak")
            plt.show()
        
        ao_DSProtation = np.append(ao_DSProtation, ao)
        
    ao_DSProtation = ao_DSProtation.astype(int)
    return ao_DSProtation

def siecinski2020(cleanYrot, fs, show = False):
    
    step = 1/fs
    time = np.arange(0, len(cleanYrot[0])*step, step)
    while len(time)>len(cleanYrot[0]):
        time = np.delete(time, -1)
    
    ao_peak_GCGy = np.array([])
    for vect in cleanYrot:
        
        graph = SingleGraph(time, vect, title = "", samplerate=fs, step=step)
        vect = butterCutPass(graph, filtorder=3, limfreqs=[4,40], show = False)
        
        searchZone = 0.1 #s
        ao = np.argmax(vect[:int(np.round(searchZone*fs))])
        
        if show == True:
            plt.plot(time, vect)
            plt.plot(time[ao], vect[ao], "o")
            plt.show()
        
        
        ao_peak_GCGy = np.append(ao_peak_GCGy, ao)
        
    ao_peak_GCGy = ao_peak_GCGy.astype(int)
    return ao_peak_GCGy

        

    
#%% Retrieval of the AO point techniques
def getAO_4090ms(cleanTrimGraphs, fs, show = False):
    """
    cleanTrimGraphs must be a list of arrays. 
    Each array is a SCG graph between a R-R interval.
    All the arrays must be the same length.
    Works better if the SCG signal has been cleaned before.
    """
    step = 1/fs
    time = np.arange(0, len(cleanTrimGraphs[0])*step, step)
    while len(time)>len(cleanTrimGraphs[0]):
        time = np.delete(time, -1)

    ao40_90rel = np.array([])
    for vect in cleanTrimGraphs:
        # Search each max peak in the 40-90 ms interval (assumed to be AO)
        minSearch = 0.04 #40 ms
        maxSearch = 0.09 #90 ms
        indMin = int(np.round(minSearch*fs))
        indMax = int(np.round(maxSearch*fs))
        ao = np.argmax(vect[indMin:indMax])
        
        if show == True :
            plt.plot(time, vect)
            plt.plot(time[ao+indMin], vect[ao+indMin], "o", color = "orange", label="AO (method 40-90 ms)")
            plt.axvspan(0.040,0.090, alpha=0.3, color='orange', label="Search zone 40-90 ms")
            plt.legend(loc = "best")
            plt.show()
        
        
        ao40_90rel = np.append(ao40_90rel, ao+indMin)

    ao40_90rel = ao40_90rel.astype(int)
    return ao40_90rel


def getAO_2pAfter40ms(cleanTrimGraphs, fs, show = False):
    """
    cleanTrimGraphs must be a list of arrays. 
    Each array is a SCG graph between a R-R interval.
    All the arrays must be the same length.
    Works better if the SCG signal has been cleaned before.
    """
    step = 1/fs
    time = np.arange(0, len(cleanTrimGraphs[0])*step, step)
    while len(time)>len(cleanTrimGraphs[0]):
        time = np.delete(time, -1)

    aoLocMax_rel = np.array([])
    for vect in cleanTrimGraphs:
        # Search each max peak in the 40-90 ms interval (assumed to be AO)
        minSearch = 0.04 #40 ms
        indMin = int(np.round(minSearch*fs))

        # Mise en place de la recherche du 2e pic temporel
        local_maxima = np.array(sc.argrelextrema(vect[int(np.round(minSearch*fs)):], np.greater_equal, order = 5))
        #local_maxima = np.array(sc.argrelextrema(vect, np.greater_equal, order = 10))
        local_maxima = local_maxima.reshape([local_maxima.shape[1],])
        local_maxima = local_maxima + indMin
        aoIndex = local_maxima[0]
        
        if show == True :
            plt.plot(time, vect)
            plt.plot(time[aoIndex], vect[aoIndex], "o", color = "green", label="AO (2d peak)")
            for i, element in enumerate(local_maxima):
                if i == 0:
                    plt.axvline(x=int(element)*step, color='green', alpha = 0.3, linestyle='-', label="Peaks found")
                else :
                    plt.axvline(x=int(element)*step, color='green', alpha = 0.3, linestyle='-')
            plt.legend(loc = "best")
            plt.show()
        
        
        aoLocMax_rel = np.append(aoLocMax_rel, aoIndex)

    aoLocMax_rel = aoLocMax_rel.astype(int)
    return aoLocMax_rel


#%% Generation of signals

def generateSCG(ecgTrack, ecgfs=200, fs=101.73333333333333, r_AOdelay=0.15, SNR=None, var_AOdelay=None, show=False):
    
    # Hardcoding of the values contained in the lin and rot waveforms (previously extracted from a real recording)
    scgLin1 = np.array([-0.03626822, -0.03870697, -0.00964752,  0.02038663,  0.02848642,
                0.01989584,  0.00837839,  0.0006357 , -0.00365017, -0.00318611,
                0.00481745,  0.01534936,  0.02104031,  0.02084285,  0.01382098,
               -0.00016782, -0.01479889, -0.02508601, -0.02971852, -0.02584436,
               -0.01483193, -0.00226942,  0.01156295,  0.02531506,  0.0310444 ,
                0.02796013,  0.02362599,  0.02140001,  0.0197706 ,  0.01487342,
                0.00741844,  0.00555026,  0.00391745, -0.01084618, -0.02521297,
               -0.0206576 , -0.01182744, -0.01191237, -0.00700524,  0.00537457,
                0.01004216,  0.00673381,  0.00471471,  0.00277742, -0.00287553,
               -0.00794067, -0.00724092, -0.00310047, -0.00060224, -0.00023085,
                0.00064106,  0.00227746,  0.00178822, -0.00204702, -0.00671426,
               -0.01041014, -0.01248146, -0.01093802, -0.00809598, -0.00929433,
               -0.01025527, -0.00362902,  0.00606335,  0.01240714,  0.0162938 ,
                0.01690147,  0.01171817,  0.00192271, -0.01078177, -0.02298959,
               -0.02808359, -0.02406762, -0.01393661,  0.00064302,  0.01483415,
                0.02082076,  0.01404954, -0.03626822])

    scgRot1 = np.array([-0.360629  , -0.22114051,  0.15731504,  0.45565534,  0.53953206,
                0.53031621,  0.57298787,  0.65679691,  0.67830599,  0.61923792,
                0.53194885,  0.40580583,  0.20132473, -0.03334415, -0.19911633,
               -0.25969107, -0.24106041, -0.17072136, -0.0832329 , -0.0230603 ,
                0.00298374,  0.02682715,  0.07609008,  0.15467076,  0.24270346,
                0.31709747,  0.36934003,  0.39485716,  0.40497235,  0.42423199,
                0.42380373,  0.35235835,  0.2502772 ,  0.18523734,  0.1419561 ,
                0.10473192,  0.09344186,  0.06200861, -0.06081055, -0.25030502,
               -0.42985757, -0.55514373, -0.60012997, -0.55457304, -0.44258582,
               -0.29960534, -0.15586829, -0.03336053,  0.06554428,  0.14675658,
                0.20844332,  0.24919573,  0.26665038,  0.24824081,  0.19257122,
                0.12314145,  0.06009084,  0.00322463, -0.0576325 , -0.12584692,
               -0.18129197, -0.19653134, -0.16942098, -0.12786883, -0.1199006 ,
               -0.18060727, -0.28765928, -0.39124245, -0.4892011 , -0.60808031,
               -0.72288554, -0.75841369, -0.66055705, -0.45137282, -0.21199069,
               -0.02827505, -0.00645258, -0.19120843])
        
    aoIndex1 = 13

    #%% Waveform 2 (BTJ)
    scgLin2 = np.array([0.09057197, 0.07034667, 0.03331333, 0.03457835, 0.03612876,
           0.03158454, 0.03611013, 0.04193628, 0.05349914, 0.05185851,
           0.02856389, 0.04950974, 0.07136503, 0.07388632, 0.05938739,
           0.03655772, 0.02718515, 0.03386344, 0.04683886, 0.04988764,
           0.0473964 , 0.04050721, 0.03365616, 0.02819578, 0.02516365,
           0.02585079, 0.03027803, 0.02855269, 0.03052878, 0.0280388 ,
           0.0304432 , 0.03127658, 0.03468609, 0.04065665, 0.03789825,
           0.03818528, 0.04587591, 0.04682701, 0.04738941, 0.03753161,
           0.02887513, 0.02874035, 0.03831973, 0.03901155, 0.03580583,
           0.03467188, 0.0277311 , 0.02307484, 0.02757051, 0.0296405 ,
           0.02647585, 0.02051212, 0.02591434, 0.02433831, 0.01910302,
           0.0220671 , 0.02055586, 0.01839016, 0.01956393, 0.01976707,
           0.01907223, 0.02197947, 0.0215241 , 0.0192687 , 0.02072702,
           0.02126777, 0.0218165 , 0.02117086, 0.01745327, 0.02244866,
           0.02424272, 0.02068188, 0.02217133, 0.02235653, 0.0225453 ,
           0.0240195 , 0.02056655, 0.02270786, 0.02620799, 0.03180583,
           0.02740589, 0.02536325, 0.02402016, 0.02335981, 0.02337992,
           0.02672862, 0.02091067, 0.02332674, 0.02676471, 0.02535256])

    scgRot2 = np.array([0.25063071, 0.26301377, 0.29180481, 0.33917601, 0.48929762,
           0.64133047, 0.5750245 , 0.35785225, 0.46722534, 0.67957104,
           0.75002679, 0.61998183, 0.38942575, 0.2893345 , 0.35677143,
           0.461261  , 0.46498591, 0.37650343, 0.24792469, 0.13546855,
           0.14907076, 0.20264935, 0.20633737, 0.2151712 , 0.19194011,
           0.14833024, 0.12024713, 0.12110984, 0.12064513, 0.16317772,
           0.1672904 , 0.16332906, 0.18828691, 0.29994771, 0.32572586,
           0.43681993, 0.29930101, 0.16315811, 0.16359082, 0.25498732,
           0.37783183, 0.41052823, 0.32488122, 0.1895865 , 0.17398912,
           0.23737043, 0.26574291, 0.22224856, 0.16824704, 0.1610582 ,
           0.15637627, 0.15543787, 0.14333073, 0.11171166, 0.1164013 ,
           0.10342948, 0.10480634, 0.11286438, 0.12904782, 0.12627919,
           0.11780902, 0.09789913, 0.09609532, 0.09221365, 0.10031571,
           0.11109609, 0.09325254, 0.09411569, 0.09111579, 0.10368325,
           0.11017931, 0.1186765 , 0.1101768 , 0.09729995, 0.1000237 ,
           0.10203668, 0.10628903, 0.10576931, 0.14397248, 0.14772739,
           0.16385453, 0.18071877, 0.20520856, 0.19850533, 0.17334995,
           0.1673472 , 0.19001721, 0.1764934 , 0.17220505, 0.20618449])

    aoIndex2 = 8

    #%% Waveform 3 (CUG)

    scgLin3 = np.array([0.04302079, 0.04788943, 0.0520282 , 0.05897618, 0.07373362,
           0.07740778, 0.07794657, 0.06738467, 0.05524546, 0.04356535,
           0.04591563, 0.05888388, 0.06712213, 0.06143086, 0.05107416,
           0.04221458, 0.04201164, 0.05114309, 0.06168191, 0.06826295,
           0.07109613, 0.06197265, 0.05619423, 0.04940868, 0.05016285,
           0.0487977 , 0.04682517, 0.04797173, 0.04590545, 0.04786364,
           0.04537532, 0.04451445, 0.04036719, 0.03859724, 0.0447429 ,
           0.04418811, 0.04671037, 0.04788447, 0.05374119, 0.05640385,
           0.05824323, 0.05402645, 0.04754198, 0.05118169, 0.04582275,
           0.04281745, 0.04728643, 0.04374172, 0.04074369, 0.03781244,
           0.03917603, 0.04153536, 0.03990405, 0.04315465, 0.03874084,
           0.03740562, 0.03521566, 0.03349938, 0.03550497, 0.03786293,
           0.03518827, 0.03386386, 0.0323262 , 0.0303586 , 0.03235823,
           0.0375707 , 0.03868296, 0.03817335, 0.0349388 , 0.03716044,
           0.03692181, 0.03241081, 0.03954713, 0.03913125, 0.04093005,
           0.04044715, 0.03748501, 0.03729435, 0.03846505])

    scgRot3 = np.array([0.56795909, 0.54417064, 0.65957072, 0.83898752, 0.84409389,
           0.78821618, 0.90443027, 0.90438195, 0.86908415, 0.968513  ,
           1.0090653 , 1.00374038, 0.89399713, 0.70431308, 0.7308614 ,
           0.78910559, 0.80525127, 0.88685808, 0.92422333, 0.8477323 ,
           0.71746933, 0.71396203, 0.86406753, 0.92014233, 0.91272166,
           0.8472741 , 0.7690298 , 0.70023683, 0.64599963, 0.62510939,
           0.54530211, 0.50412003, 0.50072497, 0.55376983, 0.55791002,
           0.64589338, 0.67768045, 0.63471727, 0.64577514, 0.62249187,
           0.67910022, 0.77344321, 0.68928548, 0.63995049, 0.5981155 ,
           0.60797145, 0.59737969, 0.53172724, 0.56173665, 0.62219376,
           0.59128556, 0.52063056, 0.46199325, 0.45267748, 0.54653328,
           0.65569266, 0.62849387, 0.58026636, 0.50560949, 0.46438824,
           0.51849145, 0.53595425, 0.5779763 , 0.60239124, 0.53274881,
           0.4618965 , 0.50497044, 0.5263222 , 0.54005242, 0.53686089,
           0.52529047, 0.54171549, 0.55066219, 0.52981028, 0.53100999,
           0.5622302 , 0.58027161, 0.60210404, 0.6314961 ])

    aoIndex3 = 6

    #%% Waveform 4 (PST)

    scgLin4 = np.array([0.09418131, 0.07032691, 0.06248753, 0.06274699, 0.06383538,
           0.05628469, 0.06292619, 0.06414603, 0.05905926, 0.05884486,
           0.07090294, 0.07151866, 0.0627352 , 0.05674482, 0.04936031,
           0.05312225, 0.05964942, 0.05359043, 0.04820964, 0.04300576,
           0.03492561, 0.03976779, 0.03692159, 0.03669779, 0.03506303,
           0.03543542, 0.03452836, 0.04118985, 0.042159  , 0.04282419,
           0.04834479, 0.0573647 , 0.05136109, 0.04170122, 0.04869809,
           0.05063891, 0.04909805, 0.04379039, 0.04274507, 0.04274976,
           0.04235517, 0.04147523, 0.04196409, 0.04071684, 0.03622721,
           0.03461141, 0.03281351, 0.03455676, 0.03697215, 0.03888737,
           0.03758797, 0.03693456, 0.03415731, 0.03640711, 0.03669494,
           0.03443714, 0.03552617, 0.03256644, 0.03785074, 0.03894564,
           0.03595944, 0.04300926, 0.04141085, 0.04400543, 0.04472394,
           0.04144134, 0.04806331, 0.05178214])

    scgRot4 = np.array([1.15143851, 1.7889186 , 2.07390062, 1.82165431, 1.30716887,
           1.10520424, 1.13479212, 1.14398884, 1.03861377, 0.86439563,
           0.7287011 , 0.65348306, 0.67631221, 0.75014269, 0.72989681,
           0.6779204 , 0.55827217, 0.41891597, 0.37383458, 0.39036354,
           0.41115138, 0.38982777, 0.37386482, 0.358823  , 0.33368561,
           0.38397327, 0.43007634, 0.42270709, 0.33440654, 0.46203492,
           0.65506555, 0.72473825, 0.64835624, 0.66182444, 0.62510453,
           0.50229555, 0.43326678, 0.50261597, 0.65963601, 0.71849365,
           0.65170415, 0.51777751, 0.4048873 , 0.34665506, 0.34637843,
           0.40287243, 0.47135597, 0.49466214, 0.44619877, 0.38883144,
           0.3355751 , 0.31585466, 0.28632955, 0.30684843, 0.31124426,
           0.32131985, 0.35693505, 0.35370235, 0.32401016, 0.34689816,
           0.36764061, 0.38186956, 0.37621427, 0.36706274, 0.40941379,
           0.4957053 , 0.49384053, 0.60686863])

    aoIndex4 = 7

    #%% Waveform 5 (VVF)

    scgLin5 = np.array([0.19164527, 0.12123737, 0.21688428, 0.13059957, 0.08598622,
           0.08027864, 0.10928295, 0.12034917, 0.09418478, 0.0506561 ,
           0.05788812, 0.09407817, 0.09980691, 0.0778042 , 0.05109262,
           0.04665478, 0.05213242, 0.05264707, 0.04836713, 0.04653668,
           0.0524177 , 0.05571436, 0.05350019, 0.03690765, 0.0295194 ,
           0.02685548, 0.02485494, 0.03189309, 0.03498468, 0.03469559,
           0.03481571, 0.03848306, 0.04296764, 0.04616625, 0.04892047,
           0.05352049, 0.06343513, 0.07790427, 0.0837841 , 0.10978994,
           0.0885495 , 0.06797298, 0.05062763, 0.05980329, 0.04680635,
           0.04405089, 0.04959019, 0.04391171, 0.04080335, 0.03866177,
           0.03427197, 0.02956502, 0.0371551 , 0.03553711, 0.03510048,
           0.02739482, 0.02629465, 0.02302452, 0.02688997, 0.02520357,
           0.02693551, 0.02232835, 0.02768744, 0.03008712, 0.02665146,
           0.02785172, 0.0244021 , 0.0247502 , 0.02766257, 0.02515235,
           0.02167074, 0.02194478, 0.02580477, 0.02379866, 0.02955708,
           0.02596781, 0.02346529, 0.02873186, 0.02384763, 0.02544828,
           0.02594515, 0.023935  , 0.0226134 , 0.02819358, 0.02532185,
           0.02591544, 0.0209773 , 0.02544229, 0.02563854, 0.02510961,
           0.02650447])

    scgRot5 = np.array([1.55159738, 2.19693189, 1.42088071, 0.68195313, 1.4110812 ,
           1.72663417, 1.34209758, 0.73796468, 0.64080789, 1.13822813,
           1.35164638, 1.13590572, 0.66620519, 0.48492707, 0.56119684,
           0.60583193, 0.57573945, 0.50801664, 0.49504268, 0.59068016,
           0.5185345 , 0.41858974, 0.35283565, 0.29846055, 0.29815364,
           0.35170323, 0.4035618 , 0.38976913, 0.31919646, 0.31588467,
           0.32195368, 0.29936934, 0.28641678, 0.30531745, 0.32317335,
           0.4236963 , 0.64218806, 1.18661442, 1.25379806, 0.84579987,
           0.88400462, 1.18899213, 0.94273587, 0.50580876, 0.39319978,
           0.50385254, 0.39120333, 0.33939471, 0.40641057, 0.37991618,
           0.42071883, 0.55397636, 0.51155865, 0.33195578, 0.31054249,
           0.36689634, 0.349888  , 0.30833921, 0.29677458, 0.30928415,
           0.26537499, 0.23369187, 0.24074786, 0.22495622, 0.25129522,
           0.23845101, 0.23188198, 0.22409388, 0.23991749, 0.23988497,
           0.22935353, 0.25587372, 0.23708397, 0.24715696, 0.28493137,
           0.2490975 , 0.21825863, 0.16346598, 0.19251062, 0.22096706,
           0.24495032, 0.20661498, 0.20726851, 0.20808759, 0.24903522,
           0.25626228, 0.24180976, 0.23958989, 0.22002605, 0.21022888,
           0.21926769])

    aoIndex5 = 7

    #%% Waveform 6 (YJR)

    scgLin6 = np.array([0.04786148, 0.0646523 , 0.06349963, 0.07900918, 0.06506242,
           0.06213829, 0.04879369, 0.0432466 , 0.04149156, 0.04660583,
           0.05062898, 0.04593793, 0.04904035, 0.04458973, 0.04332179,
           0.04434513, 0.03993531, 0.04018733, 0.04087447, 0.03795699,
           0.03970858, 0.03412702, 0.02964346, 0.0268528 , 0.02586859,
           0.02823148, 0.02427932, 0.02227978, 0.02695155, 0.02779553,
           0.02710511, 0.02659448, 0.02771513, 0.03195549, 0.03231667,
           0.03405972, 0.03649379, 0.04514287, 0.04506561, 0.03833233,
           0.03675907, 0.03075627, 0.02797533])

    scgRot6 = np.array([0.4094141 , 0.46976824, 0.6876721 , 0.77344574, 0.73568865,
           0.60723285, 0.49955058, 0.58872453, 0.76948668, 0.88012425,
           0.84468462, 0.70985745, 0.5677569 , 0.49181803, 0.53595179,
           0.6226311 , 0.78320591, 0.84807977, 0.77835995, 0.6343921 ,
           0.48871984, 0.43365563, 0.42332805, 0.4220213 , 0.38254991,
           0.30081103, 0.26810747, 0.2654105 , 0.27503225, 0.27677664,
           0.26058031, 0.28625677, 0.31167514, 0.3334321 , 0.42256661,
           0.43082162, 0.39040753, 0.33190751, 0.32885651, 0.33707168,
           0.41088124, 0.50116954, 0.56212778])

    aoIndex6 = 10

    #%% Waveform 7 (OSW)

    scgLin7 = np.array([0.05783115, 0.06951603, 0.07333327, 0.06741648, 0.06719036,
           0.08101553, 0.08123712, 0.08700639, 0.07970207, 0.08608203,
           0.08739946, 0.06448191, 0.06688581, 0.06948492, 0.06331278,
           0.04824523, 0.04625908, 0.04217001, 0.04658113, 0.04834234,
           0.04884107, 0.04859473, 0.04106845, 0.0374245 , 0.03934849,
           0.04385594, 0.04139688, 0.03979863, 0.04085991, 0.04199854,
           0.04563914, 0.04189642, 0.04957359, 0.05014549, 0.05965641,
           0.06349567, 0.06965945, 0.06214103, 0.05588765, 0.05793054,
           0.05401601, 0.05652222, 0.04902854, 0.04601896, 0.04595097,
           0.05027944, 0.04989729, 0.04925588, 0.04680906, 0.04923571,
           0.03953964, 0.03892563, 0.03971869, 0.03803983, 0.03776407,
           0.03876187, 0.03714696, 0.03587674, 0.03392555, 0.03600114,
           0.03753783, 0.03332807, 0.03735351, 0.03970651, 0.04062862,
           0.03424725, 0.03393879, 0.03702415, 0.03717099, 0.03872224])

    scgRot7 = np.array([0.59316391, 0.66745123, 0.69849518, 0.69037808, 0.63393742,
           0.56947466, 0.64193702, 0.66348855, 0.76741608, 0.80888648,
           0.88159537, 0.8508373 , 0.7440408 , 0.64890401, 0.54100953,
           0.44666813, 0.52110004, 0.54115534, 0.51047885, 0.45325351,
           0.45419226, 0.39515404, 0.33425067, 0.34125659, 0.3530697 ,
           0.37826842, 0.36182176, 0.35785394, 0.40839998, 0.38455208,
           0.33253567, 0.28672823, 0.36483095, 0.41655654, 0.49770296,
           0.60694605, 0.59127462, 0.60746693, 0.56317762, 0.57057156,
           0.58722413, 0.57479434, 0.53943684, 0.61762791, 0.60473267,
           0.5512909 , 0.55965727, 0.47870842, 0.4423    , 0.38822269,
           0.37514859, 0.38231822, 0.37134514, 0.36503495, 0.32929376,
           0.30004084, 0.30090231, 0.28074365, 0.27884085, 0.2589178 ,
           0.29441127, 0.32838126, 0.29500862, 0.29125388, 0.32645941,
           0.35405945, 0.35181234, 0.30421066, 0.28405315, 0.29029771])

    aoIndex7 = 7

    #%% Waveform 8 (INA)

    scgLin8 = np.array([0.04230204, 0.05312064, 0.05513886, 0.05504556, 0.053758  ,
           0.05715793, 0.05852413, 0.06739937, 0.07213218, 0.0660915 ,
           0.06175174, 0.06439476, 0.05585021, 0.04678413, 0.04601396,
           0.04655491, 0.04093961, 0.0384468 , 0.03416362, 0.03521298,
           0.03586916, 0.04120489, 0.04146412, 0.0385775 , 0.03183404,
           0.03091365, 0.02898329, 0.03236562, 0.0304674 , 0.03491377,
           0.04057003, 0.0434066 , 0.0500297 , 0.05623387, 0.05416896,
           0.04304403, 0.04702974, 0.0415029 , 0.03496911, 0.03239089,
           0.03401133, 0.03119574, 0.02902408, 0.02959099, 0.02434033,
           0.03028756, 0.02712871, 0.03121226, 0.02984455, 0.02833751,
           0.02899331, 0.02979947, 0.02512434, 0.02547189, 0.02931425,
           0.03038659, 0.02953814, 0.02571336, 0.0265942 , 0.02460228,
           0.03129255, 0.03033976, 0.02602545, 0.02912132, 0.02800107,
           0.02694547, 0.02609882, 0.02845212, 0.02811819])

    scgRot8 = np.array([0.55773545, 0.58676312, 0.60670511, 0.55837241, 0.55272166,
           0.60509574, 0.70464559, 0.71724633, 0.70418294, 0.87028281,
           0.8569082 , 0.69540501, 0.62482389, 0.58746073, 0.42321274,
           0.28799499, 0.34649515, 0.4339505 , 0.47175222, 0.5189612 ,
           0.47654272, 0.38191618, 0.33005349, 0.35640734, 0.41790254,
           0.46540953, 0.45090215, 0.38864033, 0.32525449, 0.29469944,
           0.30697374, 0.48138292, 0.6807067 , 0.67433411, 0.66002334,
           0.69253883, 0.55843995, 0.47422243, 0.53182284, 0.46472567,
           0.37062281, 0.28763168, 0.27479497, 0.30567966, 0.29784704,
           0.26565319, 0.24884284, 0.23461563, 0.21277252, 0.25091695,
           0.2350886 , 0.23289842, 0.24974603, 0.24247383, 0.20659033,
           0.21327256, 0.2056393 , 0.24346809, 0.22415754, 0.19553148,
           0.20592157, 0.19740608, 0.18150058, 0.19739817, 0.21786057,
           0.23137691, 0.21689318, 0.24079027, 0.23898048])

    aoIndex8 = 8

    #%% Waveform 9 (FRJ)

    scgLin9 = np.array([0.05238122, 0.04594892, 0.04586712, 0.05889741, 0.06502755,
           0.06297981, 0.05618844, 0.05895719, 0.06276738, 0.05919426,
           0.04859424, 0.05181263, 0.06308342, 0.06806825, 0.05717752,
           0.04827386, 0.04017366, 0.03755648, 0.04171141, 0.0418856 ,
           0.03855126, 0.0385069 , 0.0348949 , 0.03394783, 0.04031645,
           0.04076524, 0.04026016, 0.04316862, 0.04287306, 0.03809764,
           0.03965982, 0.03543559, 0.03766434, 0.04122515, 0.04766772,
           0.04338931, 0.05586402, 0.07270714, 0.06918083, 0.04868997,
           0.04026904, 0.04402617, 0.05207731, 0.05107924, 0.04970462,
           0.0395624 , 0.02996512, 0.03100323, 0.03402097, 0.03245764,
           0.02938335, 0.03249629, 0.03372646, 0.0337982 , 0.0343814 ,
           0.03069951, 0.03152422, 0.02666484, 0.02824117, 0.02657548,
           0.02602925, 0.02660625, 0.02869196, 0.02981361, 0.0298759 ,
           0.03002494, 0.03006905, 0.03011321, 0.03091601, 0.02835529,
           0.02924302, 0.03100206, 0.02963708, 0.03312727, 0.03011053,
           0.02878179, 0.03310406, 0.03063681, 0.0306911 ])

    scgRot9 = np.array([0.54089542, 0.64896967, 0.71770441, 0.69397579, 0.7172124 ,
           0.84755033, 0.98562076, 0.98428173, 0.76209803, 0.74246767,
           0.96882571, 1.1947493 , 1.20962633, 1.04365455, 0.93005819,
           0.9997593 , 0.98640258, 0.86364424, 0.73265964, 0.688764  ,
           0.72577114, 0.79697083, 0.80126566, 0.73109119, 0.65130492,
           0.54290124, 0.52312376, 0.53810962, 0.53873632, 0.58500116,
           0.57643264, 0.53480491, 0.49718323, 0.49739336, 0.48543815,
           0.5226603 , 0.61258991, 0.5952258 , 0.55204886, 0.66702169,
           0.81033165, 0.943449  , 0.95791473, 0.80533008, 0.54787052,
           0.56005557, 0.60303911, 0.55669741, 0.51070119, 0.46497169,
           0.4905264 , 0.53369174, 0.48811867, 0.40548191, 0.39395655,
           0.40086973, 0.37343235, 0.33117993, 0.32786149, 0.37694818,
           0.40367465, 0.38701523, 0.35206159, 0.36327042, 0.38170665,
           0.37243197, 0.37920648, 0.35689026, 0.40151626, 0.42157811,
           0.4020627 , 0.38456228, 0.37495053, 0.37778932, 0.39793727,
           0.40941572, 0.42187182, 0.41173793, 0.44156621])

    aoIndex9 = 8

    #%% Waveform 10 (DLE)

    scgLin10 = np.array([0.03618474, 0.0443291 , 0.03640514, 0.03542899, 0.03765679,
           0.03563278, 0.03016794, 0.0318693 , 0.03558141, 0.03563129,
           0.03366583, 0.03482938, 0.04070593, 0.03922242, 0.04616583,
           0.04684197, 0.04254669, 0.03710945, 0.03045355, 0.0318385 ,
           0.03207045, 0.03331473, 0.03589732, 0.03439699, 0.0357856 ,
           0.03546849, 0.03141528, 0.02857688, 0.02945697, 0.0282797 ,
           0.03130085, 0.03930692, 0.03700345, 0.04173902, 0.04047248,
           0.03722528, 0.03177666, 0.03045724, 0.03104173, 0.03510628,
           0.03495818, 0.0295392 , 0.03003196, 0.03354145, 0.0362317 ,
           0.04315962, 0.04068429, 0.03303348, 0.02936944, 0.02498236,
           0.02659638, 0.02720565, 0.03053767, 0.03214437, 0.03616542,
           0.03062722, 0.02676388, 0.02456057, 0.02450578, 0.02805932,
           0.02684794, 0.02483424, 0.02290041, 0.02592421, 0.02488406,
           0.02409587, 0.02283622, 0.02092327, 0.02690984, 0.02445478,
           0.02840865, 0.02488647, 0.02474791, 0.02827235, 0.0246932 ,
           0.02627268])

    scgRot10 = np.array([0.52660685, 0.52333781, 0.3787383 , 0.47217802, 0.5898056 ,
           0.62234119, 0.51041816, 0.39551609, 0.3595725 , 0.37755353,
           0.49024221, 0.55868018, 0.51835379, 0.44740532, 0.3548395 ,
           0.31389072, 0.30579156, 0.2972628 , 0.28972265, 0.25694387,
           0.25324038, 0.26486227, 0.23060365, 0.20917523, 0.19442439,
           0.18816214, 0.19505864, 0.21501467, 0.26882428, 0.27971717,
           0.33934757, 0.42278684, 0.4245737 , 0.38863698, 0.30342111,
           0.23692635, 0.25572968, 0.29688138, 0.30812049, 0.27047748,
           0.28471298, 0.2765058 , 0.2551688 , 0.23853064, 0.19492078,
           0.19875247, 0.2051315 , 0.2070036 , 0.22324935, 0.22103387,
           0.23212554, 0.24155058, 0.20598128, 0.17702296, 0.17641579,
           0.18997316, 0.20835716, 0.20907457, 0.20526163, 0.19433426,
           0.18559041, 0.17845242, 0.17924335, 0.16795531, 0.16157566,
           0.16203553, 0.17357307, 0.18387114, 0.16343583, 0.14457524,
           0.15963721, 0.16905542, 0.16540818, 0.16188698, 0.17927163,
           0.17123878])

    aoIndex10 = 4

    #%% Choose the data randomly

    data1 = [scgLin1, scgRot1, aoIndex1]
    data2 = [scgLin2, scgRot2, aoIndex2]
    data3 = [scgLin3, scgRot3, aoIndex3]
    data4 = [scgLin4, scgRot4, aoIndex4]
    data5 = [scgLin5, scgRot5, aoIndex5]
    data6 = [scgLin6, scgRot6, aoIndex6]
    data7 = [scgLin7, scgRot7, aoIndex7]
    data8 = [scgLin8, scgRot8, aoIndex8]
    data9 = [scgLin9, scgRot9, aoIndex9]
    data10 = [scgLin10, scgRot10, aoIndex10]

    allData = [data1, data2, data3, data4, data5, data6, data7, data8, data9, data10]

    chosen = np.random.randint(0,10)
    scgLin = allData[chosen][0]
    scgRot = allData[chosen][1]
    aoIndex = allData[chosen][2]
    
    
    
    
    
    fs = fs
    step = 1/fs
    
    # First part is the SCG points before AO and second part is the ones after AO
    firstPart = scgLin[0:aoIndex-1]
    secondPart = scgLin[aoIndex:]
    
    firstPartROT = scgRot[0:aoIndex-1]
    secondPartROT = scgRot[aoIndex:]
    
    
    # Process the ECG to get its R peaks detections
    ecg_results = nk.ecg_process(ecgTrack, sampling_rate=ecgfs)
    r_peaks = ecg_results[1]["ECG_R_Peaks"]
    r_times = r_peaks/ecgfs

    # resample the ECG so that it is the same fs as the scg
    ecgTrack = sc.resample(ecgTrack, int(np.round(len(ecgTrack)*fs/ecgfs))+1)
    r_peaks = np.round(r_times*fs).astype(int)
    r_ecarts = np.diff(r_peaks)
    
    print(len(ecgTrack))
    
    time = np.arange(0, len(ecgTrack)*step, step)
    while len(ecgTrack)>len(time):
        time = np.append(time, time[-1]+step)
    while len(ecgTrack)<len(time):
        time = np.delete(time, -1)
    
    # Create an array to store the values of the SCG track generated
    scg_constr = np.array([])
    scgROT_constr = np.array([])

    # We specify the delay between the R peaks and the AO peaks in seconds
    delayAO_samples = int(np.round(r_AOdelay*fs))
    delays = np.array([])
    for i in range(len(r_peaks)):
        if var_AOdelay :
            delay = delayAO_samples + int(np.round(np.random.uniform(-var_AOdelay, var_AOdelay)*fs))
        else :
            delay = delayAO_samples
        delays = np.append(delays, delay).astype(int)
    
    print("Delays SCG = ", delays)
    
    # At each R peak detected
    for i in range(len(r_peaks)):
        # No waveform until the 1st R peak : 0 for now
        begin_offset = 0
        # If it is the first R peak
        if i == 0:
            # And its index is not 0
            if not r_peaks[i] == 0:
                # Add "null" signal until we meet the 1st R peak
                begin_offset = r_peaks[i]
                print("Beginning offset of SCG in samples = ", begin_offset)
                for k in range(begin_offset):
                    scg_constr = np.append(scg_constr, np.mean(scgLin))
                    scgROT_constr = np.append(scgROT_constr, np.mean(scgRot))

        if i == len(r_ecarts):
            rSample_ecart = r_ecarts[i-1]
        else:
            rSample_ecart = r_ecarts[i]
        
        # Add the delay R-AO for the current peak
        scg_first = sc.resample(firstPart, delays[i])
        scg_constr = np.append(scg_constr, scg_first)
        scg_firstROT = sc.resample(firstPartROT, delays[i])
        scgROT_constr = np.append(scgROT_constr, scg_firstROT)
        
        rest = rSample_ecart - delays[i]
        # Add the rest of the SCG waveform
        scg_sec = sc.resample(secondPart, rest)
        scg_constr = np.append(scg_constr, scg_sec)
        scg_secROT = sc.resample(secondPartROT, rest)
        scgROT_constr = np.append(scgROT_constr, scg_secROT)
        
    # Reshape the arrays in case one is longer than the other
    while len(scg_constr) < len(ecgTrack):
        scg_constr = np.append(scg_constr, scg_constr[-1])
    while len(scg_constr) > len(ecgTrack):
        scg_constr = np.delete(scg_constr, -1)
    while len(scgROT_constr) < len(ecgTrack):
        scgROT_constr = np.append(scgROT_constr, scgROT_constr[-1])
    while len(scgROT_constr) > len(ecgTrack):
        scgROT_constr = np.delete(scgROT_constr, -1)

    
    # Clean the high frequency components with a LP filter :
    def getSmoothSCG(waveform, fs):
        yf = rfft(waveform)
        xf = rfftfreq(len(waveform), 1/fs)

        b, a = signal.butter(N=3, Wn=30, btype='lowpass', fs=fs)
        w, h = freqz(b, a, fs=fs, worN=len(xf))
     
        # Cleaning of the signal -> y
        zi = signal.lfilter_zi(b, a)
        z, _ = signal.lfilter(b, a, waveform, zi=zi*waveform[0])
        z2, _ = signal.lfilter(b, a, z, zi=zi*z[0])
        y = signal.filtfilt(b, a, waveform)
        return y
    
    # Smoothing of the result
    scg_constr_sm = getSmoothSCG(scg_constr, fs)
    scgROT_constr_sm = getSmoothSCG(scgROT_constr, fs)
    
    
    # Add noise to the BP signal
    if SNR:
        noiseMag = SNR*(np.max(scg_constr_sm)-np.min(scg_constr_sm))
        noise = np.random.normal(loc=0, scale=noiseMag, size=len(scg_constr_sm))
        scg_constr_sm = scg_constr_sm + noise
        
        noiseMag = SNR*(np.max(scgROT_constr_sm)-np.min(scgROT_constr_sm))
        noise = np.random.normal(loc=0, scale=noiseMag, size=len(scgROT_constr_sm))
        scgROT_constr_sm = scgROT_constr_sm + noise

    # Visualize the result
    delaystimes = delays/fs
    print("Delays  times= ", delaystimes)

    if show == True:
        fig, axis = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
        axis[0].plot(time, ecgTrack)
        axis[0].set_title("Signal ECG simulé")
        #axis[1].plot(time, scg_constr, label = "before smoothing")
        axis[1].plot(time, scg_constr_sm, label="after smoothing")
        axis[1].set_title("Signal SCG linéaire simulé avec délai de "+str(r_AOdelay)+" s")
        axis[2].plot(time, scgROT_constr_sm, label="after smoothing")
        axis[2].set_title("Signal SCG rotationnel simulé avec délai de "+str(r_AOdelay)+" s")

        for ax in axis:
            for i, rPeak in enumerate(r_times):
                ax.axvline(x=rPeak, color='red', linestyle='-')
                ax.axvline(x=rPeak+delaystimes[i], color='green', linestyle='-')
            ax.grid()

        plt.xlim(5,7)

        fig.tight_layout(pad=2.0)
        plt.setp(axis, xlabel='Time [s]')
        plt.setp(axis, ylabel='Magnitude')
        plt.show()
        
    index_AOpeaks = r_peaks+delays
    return scg_constr_sm, scgROT_constr_sm, [index_AOpeaks.astype(int), r_times+delaystimes]


def generateBP(ecgTrack, fs, r_BPdelay=0.15, maxsigMag=820, SNR=None, var_BPdelay=None, show=False):
    
    # Hardcoding of the values contained in waveform (previously extracted from a real recording)
    waveform = np.array([825.78605472, 825.81271714, 825.83999747, 825.86890543,
           825.90034695, 825.93504471, 825.97348287, 826.01587744,
           826.0621713 , 826.11205043, 826.16497702, 826.22023524,
           826.2769855 , 826.33432285, 826.39133396, 826.44714732,
           826.50097296, 826.55213019, 826.600064  , 826.64435214,
           826.68470457, 826.72095644, 826.75305553, 826.7810451 ,
           826.80504382, 826.82522451, 826.84179392, 826.85497552,
           826.86499623, 826.87207709, 826.87642713, 826.87823971,
           826.87769045, 826.8749355 , 826.87010986, 826.86332615,
           826.85467523, 826.84422885, 826.83204415, 826.81816879,
           826.8026461 , 826.7855198 , 826.76683888, 826.7466628 ,
           826.72506693, 826.70214807, 826.67802982, 826.65286688,
           826.62684702, 826.60018971, 826.57314121, 826.54596673,
           826.51894044, 826.4923338 , 826.46640378, 826.4413819 ,
           826.41746511, 826.39480899, 826.37352371, 826.35367295,
           826.33527609, 826.31831328, 826.30273277, 826.28845975,
           826.27540514, 826.26347359, 826.25257005, 826.24260436,
           826.23349339, 826.22516019, 826.21753052, 826.21052817,
           826.20407126, 826.19807093, 826.19243257, 826.1870589 ,
           826.18185511, 826.17673551, 826.17162956, 826.16648472,
           826.16126458, 826.15594392, 826.15050302, 826.14492379,
           826.13918861, 826.13328129, 826.12718821, 826.12089944,
           826.11440989, 826.10771956, 826.10083267, 826.09375563,
           826.08649465, 826.07905349, 826.07143169, 826.06362382,
           826.05562023, 826.04741018, 826.03898759, 826.03035787,
           826.02154314, 826.01258375, 826.00353553, 825.99446394,
           825.98543635, 825.97651411, 825.96774601, 825.95916407,
           825.95078255, 825.94259981, 825.93460251, 825.92677063,
           825.91908209, 825.91151637, 825.90405758, 825.89669671,
           825.88943281, 825.88227278, 825.87523064, 825.86832733,
           825.86159127, 825.85505845, 825.84877069, 825.84277241,
           825.8371071 , 825.8318149 , 825.82693127, 825.82248561,
           825.81849897, 825.81498118, 825.81192849, 825.80932198,
           825.8071272 , 825.80529581, 825.80376964, 825.80248579,
           825.8013807 , 825.80039289, 825.79946597, 825.79855293,
           825.79762079, 825.79665318, 825.79564954, 825.79462115,
           825.79358632, 825.79256567, 825.79157761, 825.79063428,
           825.78973926, 825.78888798, 825.78807049, 825.7872745 ,
           825.78648702, 825.7856946 , 825.7848838 , 825.78404333,
           825.78316719, 825.78225692, 825.78132131, 825.78037381,
           825.7794297 , 825.77850462, 825.77761353, 825.77676875,
           825.77597752, 825.77524104])
    
    # Adjust the amplitude of the signal
    waveform = waveform - ( np.min(waveform) - maxsigMag)

    # Process the ECG to get its R peaks detections
    ecg_results = nk.ecg_process(ecgTrack, sampling_rate=fs)
    r_peaks = ecg_results[1]["ECG_R_Peaks"]
    r_times = r_peaks/fs
    
    r_ecarts = np.diff(r_peaks)
    
    # Create an array to store the values of the BP track generated
    bp_constr = np.array([])
    delays = np.array([])
    
    # We specify the delay between the R peaks and the BP onsets ins seconds
    delayBP_seconds = r_BPdelay
    # And in number of samples
    delayBP_samples = int(np.round(delayBP_seconds*fs))
    
    # No waveform until the 1st R peak : 0 for now
    begin_offset = 0
    
    # At each R peak detected
    for i in range(len(r_peaks)):
        
        # If it is the first R peak
        if i == 0:
            # And its index is not 0
            if not r_peaks[i] == 0:
                # Add "null" signal until we meet the 1st R peak
                begin_offset = r_peaks[i]
                print("Beginning offset of BP in samples = ", begin_offset)
                for k in range(begin_offset):
                    bp_constr = np.append(bp_constr, waveform[0])

        if i == len(r_ecarts):
            rSample_ecart = r_ecarts[i-1]
        else:
            rSample_ecart = r_ecarts[i]
        
        if var_BPdelay :
            print("There is VarBPdelay")
            delay = delayBP_samples + int(np.round(np.random.uniform(-var_BPdelay, var_BPdelay)*fs))
        else :
            print("There is no VarBPdelay")
            delay = delayBP_samples
        delays = np.append(delays, delay)
        
        for k in range(delay):
            bp_constr = np.append(bp_constr, bp_constr[-1])
        rest = rSample_ecart - delay
        
        bp_respld = sc.resample(waveform, rest)
        bp_constr = np.append(bp_constr, bp_respld)
    
    print("Delays = ", delays)
    
    # Reshape the arrays in case one is longer than the other
    while len(bp_constr) < len(ecgTrack):
        bp_constr = np.append(bp_constr, bp_constr[-1])
    while len(bp_constr) > len(ecgTrack):
        bp_constr = np.delete(bp_constr, -1)
    
    # Add noise to the BP signal
    if SNR:
        noiseMag = SNR*(np.max(bp_constr)-np.min(bp_constr))
        noise = np.random.normal(loc=0, scale=noiseMag, size=len(bp_constr))
        bp_constr = bp_constr + noise
    
    delaystimes = delays/fs
    print("Delays times = ", delaystimes)
    
    # Show the results
    if show == True :
        # Visualize the result   
        time = np.arange(0, len(ecgTrack)*1/200, 1/200)
        fig, axis = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        axis[0].plot(time, ecgTrack)
        axis[0].set_title("Signal ECG simulé")
        axis[1].plot(time, bp_constr)
        axis[1].set_title("Signal BP simulé avec délai de "+str(delayBP_seconds)+" s")

        for ax in axis:
            for i, rPeak in enumerate(r_times):
                ax.axvline(x=rPeak, color='red', linestyle='-')
                ax.axvline(x=rPeak+delaystimes[i], color='green', linestyle='-')
            ax.grid()

        plt.xlim(0,10)

        fig.tight_layout(pad=2.0)
        plt.setp(axis, xlabel='Time [s]')
        plt.setp(axis, ylabel='Magnitude')
        plt.show()
        
        
        index_bpOnsets = (r_times+delaystimes)*fs
        return bp_constr, [index_bpOnsets.astype(int), r_times+delaystimes]


def simulate_file(duration=60, ecg_fs=200, ecg_noise = 0.01, hr=70, # About the ECG
                  r_BPlegdelay=0.25, maxBPleg=750, SNRbpLeg=0.005, varLeg_delay=None, # About the BPleg
                  r_BParmdelay=0.15, maxBParm=800, SNRbpArm=0.005, varArm_delay=None, # About the BParm
                  scg_fs=101.73333333333333, r_AOdelay=0.1, SNR_AO=0.05, varAO_delay=None, # About the SCG
                  show = False): # If results need to be displayed or not
    
    # Generate a ECG track
    ecgTrack = nk.ecg_simulate(duration=duration+20, sampling_rate=ecg_fs, noise=ecg_noise, heart_rate=hr)
    # Construction of the Arm BP graph
    bpArm_constr, onsetsBParm = generateBP(ecgTrack, fs=ecg_fs, r_BPdelay=r_BParmdelay, maxsigMag=maxBParm, SNR=SNRbpArm, var_BPdelay=varArm_delay, show=True)
    # Construction of the Leg BP graph
    bpLeg_constr, onsetsBPleg = generateBP(ecgTrack, fs=ecg_fs, r_BPdelay=r_BPlegdelay, maxsigMag=maxBPleg, SNR=SNRbpLeg, var_BPdelay=varLeg_delay, show=True)
    # Construction of the linear and rotationnal SCG graphs
    scgLin, scgRot, locAO = generateSCG(ecgTrack, ecgfs=ecg_fs, fs=scg_fs, r_AOdelay=r_AOdelay, SNR=SNR_AO, var_AOdelay=varAO_delay, show=True)
    
    # Resizing the graphs ML to the right duration
    actuLenML = len(ecgTrack)
    diffML = actuLenML - duration*ecg_fs
    halfdiffML = int(diffML/2)
    ecgTrack = ecgTrack[halfdiffML:actuLenML-halfdiffML]
    bpArm_constr = bpArm_constr[halfdiffML:actuLenML-halfdiffML]
    bpLeg_constr = bpLeg_constr[halfdiffML:actuLenML-halfdiffML]
    
    # Correct the onsets indexes and times
    onsetsBParm_0 = np.array([x for x in onsetsBParm[0] if x>halfdiffML and x<actuLenML-halfdiffML])
    onsetsBParm_0 = onsetsBParm_0 - halfdiffML
    onsetsBParm_1 = np.array([x for x in onsetsBParm[1] if x>halfdiffML/ecg_fs and x<(actuLenML-halfdiffML)/ecg_fs])
    onsetsBParm_1 = onsetsBParm_1 - halfdiffML/ecg_fs
    onsetsBParm = [onsetsBParm_0, onsetsBParm_1]
    
    onsetsBPleg_0 = np.array([x for x in onsetsBPleg[0] if x>halfdiffML and x<actuLenML-halfdiffML])
    onsetsBPleg_0 = onsetsBPleg_0 - halfdiffML
    onsetsBPleg_1 = np.array([x for x in onsetsBPleg[1] if x>halfdiffML/ecg_fs and x<(actuLenML-halfdiffML)/ecg_fs])
    onsetsBPleg_1 = onsetsBPleg_1 - halfdiffML/ecg_fs
    onsetsBPleg = [onsetsBPleg_0, onsetsBPleg_1]
    
    # Resizing if the graphs JSON to the right duration
    actuLenJSON = len(scgLin)
    diffJSON = actuLenJSON - duration*scg_fs
    halfdiffJSON = int(diffJSON/2)
    scgLin = scgLin[halfdiffJSON:actuLenJSON-halfdiffJSON]
    scgRot = scgRot[halfdiffJSON:actuLenJSON-halfdiffJSON]
    
    # Correct the AO peaks indexes and times
    newlocAO_1 = np.array([x for x in locAO[1] if x>halfdiffJSON/scg_fs and x<(actuLenJSON-halfdiffJSON)/scg_fs])
    newlocAO_1 = newlocAO_1 - halfdiffJSON/scg_fs
    newlocAO_0 = np.array([x for x in locAO[0] if x>halfdiffJSON and x<actuLenJSON-halfdiffJSON])
    newlocAO_0 = newlocAO_0 - halfdiffJSON
    locAO = [newlocAO_0, newlocAO_1]
    
    # Show the result
    if show == True :
        time_ML = np.arange(0, len(ecgTrack)*1/ecg_fs, 1/ecg_fs)
        scg_time = np.arange(0, len(scgLin)*1/scg_fs, 1/scg_fs)
        
        fig, axis = plt.subplots(5, 1, figsize=(10, 15), sharex=True)
        
        axis[0].set_title("Generated ECG (same for JSON and MatLab)")
        axis[0].plot(time_ML, ecgTrack)
        axis[1].set_title("Generated BP arm")
        axis[1].plot(time_ML, bpArm_constr)
        axis[1].plot(time_ML[onsetsBParm[0]], bpArm_constr[onsetsBParm[0]], "o")
        axis[2].set_title("Generated BP leg")
        axis[2].plot(time_ML, bpLeg_constr)
        axis[2].plot(time_ML[onsetsBPleg[0]], bpLeg_constr[onsetsBPleg[0]], "o")
        axis[3].set_title("Generated linear SCG")
        axis[3].plot(scg_time, scgLin)
        axis[3].plot(scg_time[locAO[0]], scgLin[locAO[0]], "o")
        axis[4].set_title("Generated angular SCG")
        axis[4].plot(scg_time, scgRot)
        axis[4].plot(scg_time[locAO[0]], scgRot[locAO[0]], "o")
        
        for ax in axis:
            ax.grid()
        
        fig.tight_layout(pad=2.0)
        plt.xlim([5,10])
        plt.setp(axis, xlabel='Time [s]')
        plt.setp(axis, ylabel='Magnitude')
        plt.show()


    # Store the values in a dictionary to be returned    
    dataML = {}
    dataML["fs"] = ecg_fs
    dataML["ECG"] = ecgTrack
    dataML["BPArm"] = bpArm_constr
    dataML["BPleg"] = bpLeg_constr
    dataML["BPArm_onsets"] = onsetsBParm
    dataML["BPLeg_onsets"] = onsetsBPleg
    dataJSON = {}
    dataJSON["fs"] = scg_fs
    dataJSON["LinSCG"] = scgLin
    dataJSON["RotSCG"] = scgRot
    dataJSON["AOlocations"] = locAO
    
    data={}
    data["MatLab"] = dataML
    data["JSON"] = dataJSON
    # We cut to have the resulting tracks to get the right duration for the output files
    
    return data # To test this function : new_data = simulate_file(show = True)