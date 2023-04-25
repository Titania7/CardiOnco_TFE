# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 16:49:11 2023

@author: Tatiana Dehon
"""

from openFiles import *
from simpleDisplay import *
import numpy as np

# Test HK JSON
#pathJSON = "iCure_JSON\\3aa9adf0-8fec-4aa9-a39d-cf7cee921c48.json"
pathJSON = "iCure_JSON\\PackHIZtest\\test-Onco_HIZ_84.json"

"""
# 2 mesures de 80 secondes environ :
    BTJ 16032023 2.json
"""

#with open(pathJSON) as f:
#    contents = json.load(f)

dataJSON, typeJSON = openFile(pathJSON)
allDataJSON = getDataToDisplay(dataJSON, typeJSON)
#display_plt(allDataJSON)

p_p = []
q_p = []
r_p = []
s_p = []
t_p = []
p_on = []
t_off = []
ecg = None
bpleg = None

lims = [52*200, 67*200]

for graph in allDataJSON:
    if graph._title == "ECG" :
        ecg = graph
        p_p, q_p, r_p, s_p, t_p, p_on, t_off = detect_qrs(graph, clean_technique="ECG_NeuroKit", ecg_delineate="peaks", show=False)
        
def butterCutPass(graph):
    # Use of a butterworth filter to cut out the noise
    b, a = signal.butter(N=3, Wn=[0.3, 8], btype='bandpass', fs=graph._samplerate)
    zi = signal.lfilter_zi(b, a)
    z, _ = signal.lfilter(b, a, graph._y, zi=zi*graph._y[0])
    z2, _ = signal.lfilter(b, a, z, zi=zi*z[0])
    y = signal.filtfilt(b, a, graph._y)

    return y

for graph in allDataJSON:
    if graph._title == "x_scgLin[m/s^2]" :
        xLin = graph
        
        """
        # Visualization of the fourier signal
        plt.figure(figsize=(10, 6))
        yf = rfft(graph._y)
        xf = rfftfreq(len(graph._x), graph._step)
        plt.title(graph._title+" before cleaning")
        plt.xlabel("Frequency[Hz]")
        plt.ylabel("Magnitude")
        plt.plot(xf, np.abs(yf))
        plt.show()
        """
        
        xLin_clean = butterCutPass(graph)
        
        # Visualization of the fourier signal
        plt.figure(figsize=(10, 6))
        yf = rfft(xLin_clean)
        xf = rfftfreq(len(graph._x), graph._step)
        plt.title(graph._title+" after cleaning")
        plt.xlabel("Frequency[Hz]")
        plt.ylabel("Magnitude")
        plt.plot(xf, np.abs(yf))
        plt.show()
        
        """
        # Visualization of the cleaned signal
        plt.figure(figsize=(10, 6))
        plt.title("After cleaning")
        plt.plot(graph._x, graph._y, 'b', alpha=0.6)
        plt.plot(graph._x, y, 'red')
        plt.legend(('noisy signal','filtfilt'), loc='best')
        plt.grid(True)
        plt.show()
        """
        
    elif graph._title == "y_scgLin[m/s^2]" :
        yLin = graph
        yLin_clean = butterCutPass(graph)

    elif graph._title == "z_scgLin[m/s^2]" :
        zLin = graph
        zLin_clean = butterCutPass(graph)
    
    elif graph._title == "x_scgRot[deg/s]" :
        xRot = graph        
        xRot_clean = butterCutPass(graph)
        
    elif graph._title == "y_scgRot[deg/s]" :
        yRot = graph        
        yRot_clean = butterCutPass(graph)
    
    elif graph._title == "z_scgRot[deg/s]" :
        zRot = graph        
        zRot_clean = butterCutPass(graph)



plt.figure(figsize=(10, 20))
fig, axis = plt.subplots(7, 1, figsize=(15, 12), sharex=True)

# ECG
axis[0].plot(ecg._x, ecg._y)
axis[0].plot(ecg._x[r_p[0]], ecg._y[r_p[0]], 'o', color = 'red', label = 'R peaks')
axis[0].set_title(ecg._title)
axis[0].legend(loc="lower right")
axis[0].grid()

# xLin
axis[1].plot(xLin._x, xLin_clean, label = 'Without respiration and noise')
axis[1].set_title(xLin._title)
axis[1].legend(loc="lower right")
axis[1].grid()

# xLin
axis[2].plot(yLin._x, yLin_clean, label = 'Without respiration and noise')
axis[2].set_title(yLin._title)
axis[2].legend(loc="lower right")
axis[2].grid()

# xLin
axis[3].plot(zLin._x, zLin_clean, label = 'Without respiration and noise')
axis[3].set_title(zLin._title)
axis[3].legend(loc="lower right")
axis[3].grid()

# xLin
axis[4].plot(xRot._x, xRot_clean, label = 'Without respiration and noise')
axis[4].set_title(xRot._title)
axis[4].legend(loc="lower right")
axis[4].grid()

# xLin
axis[5].plot(yRot._x, yRot_clean, label = 'Without respiration and noise')
axis[5].set_title(yRot._title)
axis[5].legend(loc="lower right")
axis[5].grid()

# xLin
axis[6].plot(zRot._x, zRot_clean, label = 'Without respiration and noise')
axis[6].set_title(zRot._title)
axis[6].legend(loc="lower right")
axis[6].grid()

"""
axis[1].plot(xLin._x, xLin._y, label = 'With respiration and noise')
axis[2].plot(yLin._x, yLin._y, label = 'With respiration and noise')
axis[3].plot(zLin._x, zLin._y, label = 'With respiration and noise')
axis[4].plot(xRot._x, xRot._y, label = 'With respiration and noise')
axis[5].plot(yRot._x, yRot._y, label = 'With respiration and noise')
axis[6].plot(zRot._x, zRot._y, label = 'With respiration and noise')
"""

fig.tight_layout(pad=2.0)
plt.setp(axis[1], xlabel='Time [s]')
plt.setp(axis, ylabel='Magnitude')
axis[6].set_xlim(0,5)
plt.show()
plt.close()