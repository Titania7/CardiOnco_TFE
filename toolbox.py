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

import matplotlib.pyplot as plt

import numpy as np
from numpy.fft import *

from scipy import signal
import scipy.signal as sc
from scipy.signal import freqz
from scipy.fft import rfft, rfftfreq



#%% Small useful functions

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

def getStartStopIndexes(ecg_JSON, ecg_MLd, show = False):
    
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
        fig, axis = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        axis[0].set_title("Corresponding ECG")
        axis[0].plot(ecg._x, ecg._y)
        axis[0].grid()
        
        axis[1].set_title("Cleaning of "+graph._title)
        axis[1].plot(graph._x, graph._y, 'grey', alpha = 0.5, label = "Raw signal")
        axis[1].plot(graph._x, y, label = "Cleaned signal")
        axis[1].legend(loc='best')
        axis[1].grid()
        
        for r_time in r_p[1]:
            axis[0].axvline(x=r_time, color='red', linestyle='--')
            axis[1].axvline(x=r_time, color='red', linestyle='--')
        
        axis[0].set_xlim([r_p[1][0]-0.05, r_p[1][5]+0.05])
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
        fig, axis = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        axis[0].set_title("Corresponding ECG")
        axis[0].plot(ecg._x, ecg._y)
        axis[0].grid()
        
        axis[1].set_title("Cleaning of "+graph._title)
        axis[1].plot(graph._x, graph._y, 'grey', alpha = 0.5, label = "Raw signal")
        axis[1].plot(graph._x, y, label = "Cleaned signal")
        axis[1].legend(loc='best')
        axis[1].grid()
        
        for r_time in r_p[1]:
            axis[0].axvline(x=r_time, color='red', linestyle='--')
            axis[1].axvline(x=r_time, color='red', linestyle='--')
        
        axis[0].set_xlim([r_p[1][0]-0.05, r_p[1][5]+0.05])
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
        fig, axis = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        axis[0].set_title("Corresponding ECG")
        axis[0].plot(ecg._x, ecg._y)
        axis[0].grid()
        
        axis[1].set_title("Cleaning of "+graph._title)
        axis[1].plot(graph._x, graph._y, 'grey', alpha = 0.5, label = "Raw signal")
        axis[1].plot(graph._x, y, label = "Cleaned signal")
        axis[1].legend(loc='best')
        axis[1].grid()
        
        for r_time in r_p[1]:
            axis[0].axvline(x=r_time, color='red', linestyle='--')
            axis[1].axvline(x=r_time, color='red', linestyle='--')
        
        axis[0].set_xlim([r_p[1][0]-0.05, r_p[1][5]+0.05])
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

def getBpOnsets_tang(bpGraph, rPeaks, lims = [], filt = False, show = False):
    
    """
    # Computation of the onset points with the intersecting tangents method

    """
    
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
        if filt == True:
            # Use of a butterworth filter to cut out the noise
            #b, a = signal.butter(3, 0.05)
            b, a = signal.butter(5, 0.1)
            zi = signal.lfilter_zi(b, a)
            z, _ = signal.lfilter(b, a, derivNotCleaned, zi=zi*derivNotCleaned[0])
            z2, _ = signal.lfilter(b, a, z, zi=zi*z[0])
            y = signal.filtfilt(b, a, derivNotCleaned)
            
            # Cleaned signal is stored in "deriv" variable
            deriv = y
        else:
            deriv = derivNotCleaned
    
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

    aoLocMax_rel = np.array([])
    for vect in cleanTrimGraphs:
        # Search each max peak in the 40-90 ms interval (assumed to be AO)
        minSearch = 0.04 #40 ms
        indMin = int(np.round(minSearch*fs))

        # Mise en place de la recherche du 2e pic temporel
        local_maxima = np.array(sc.argrelextrema(vect[int(np.round(minSearch*fs)):], np.greater_equal, order = 2))
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
    scgLin = np.array([-0.03626822, -0.03870697, -0.00964752,  0.02038663,  0.02848642,
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

    scgRot = np.array([-0.360629  , -0.22114051,  0.15731504,  0.45565534,  0.53953206,
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
    
    aoIndex = 13
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
