# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 19:29:06 2023

@author: Tatiana Dehon
"""

import scipy.io as sc
import matplotlib.pyplot as plt
import numpy as np


pathfile = "C:/Users/32493/Documents/Mes_Trucs/Cours/CardiOnco_projectMa2/DonnéesTFEWivine/GROUPE 1 validation/pat0101022022_pwaohu_ffr_kino_doppler/Cathlab/BLOC 3/fBPAo.mat"
mat = sc.loadmat(pathfile)

fs = np.squeeze(mat["Fsam"])
ecg = np.squeeze(mat["ECG"].T)
bpAO = np.squeeze(mat["BPAo"].T)
bpAOons = np.squeeze(mat["onsetBPAo"].T)

x = np.arange(0, len(bpAO)/fs, 1/fs)

plt.plot(x, ecg)
plt.title("ECG")
plt.show()

plt.plot(x, bpAO)
plt.plot(x[bpAOons], bpAO[bpAOons], 'o')
plt.title("BPAo")
plt.show()


pathfile = "C:/Users/32493/Documents/Mes_Trucs/Cours/CardiOnco_projectMa2/DonnéesTFEWivine/GROUPE 1 validation/pat0101022022_pwaohu_ffr_kino_doppler/Cathlab/BLOC 3/fBPplus.mat"
mat = sc.loadmat(pathfile)

fs = np.squeeze(mat["Fsam"])
bpArm = np.squeeze(mat["BPplus"].T)
bpArmons = np.squeeze(mat["onsetBPplus"].T)

x = np.arange(0, len(bpArm)/fs, 1/fs)

plt.plot(x, bpArm)
plt.plot(x[bpArmons], bpArm[bpArmons], 'o')
plt.title("BPArm")
plt.show()


pathfile = "C:/Users/32493/Documents/Mes_Trucs/Cours/CardiOnco_projectMa2/DonnéesTFEWivine/GROUPE 1 validation/pat0101022022_pwaohu_ffr_kino_doppler/Cathlab/BLOC 3/fBPleg.mat"
mat = sc.loadmat(pathfile)

fs = np.squeeze(mat["Fsam"])
bpLeg = np.squeeze(mat["BPleg"].T)
bpLegons = np.squeeze(mat["onsetBPleg"].T)

x = np.arange(0, len(bpLeg)/fs, 1/fs)

plt.plot(x, bpLeg)
plt.plot(x[bpLegons], bpLeg[bpLegons], 'o')
plt.title("BPLeg")
plt.show()

pathfile = "C:/Users/32493/Documents/Mes_Trucs/Cours/CardiOnco_projectMa2/DonnéesTFEWivine/GROUPE 1 validation/pat0101022022_pwaohu_ffr_kino_doppler\Cathlab\BLOC 6\_L2.mat"
mat = sc.loadmat(pathfile)

