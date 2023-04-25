# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 23:36:45 2023

@author: Tatiana Dehon
"""

from openFiles import *
from simpleDisplay import *
import matplotlib.pyplot as plt
import numpy as np

path1 = "C:\\Users\\32493\\Documents\\Mes_Trucs\\Cours\\CardiOnco_projectMa2\\Scripts_perso\\iCure_matlab\\AQB 20032023.mat"
data1, typefile1 = openFile(path1)
allData1 = getDataToDisplay(data1, typefile1)
path2 = "C:\\Users\\32493\\Documents\\Mes_Trucs\\Cours\\CardiOnco_projectMa2\\Scripts_perso\\iCure_JSON\\0001ff15-5a69-445b-bf91-d9a2bf5501db.json"
data2, typefile2 = openFile(path2)
allData2 = getDataToDisplay(data2, typefile2)



iD = "Name"
measures_iCure = ["ECG_iCure", "sfreq_ECG_iCure", "SCGlin_x", "SCGlin_y", "SCGlin_z", "SCGrot_x", "SCGrot_y", "SCGrot_z", "sfreq_SCG"]
measures_MLb = ["ECG_MatLab", "sfreq_ECG_MatLab", "BPAo", "sfreq_BPAo", "BPleg", "sfreq_BPleg"]
keys = ["Date", "Age", "Height", "Weight", "Sex", "iCure", "MatLab"]

iCure_dict = dict.fromkeys(measures_iCure)
mLb_dict = dict.fromkeys(measures_MLb)

for graph in allData2 :
    if graph._title == "ECG":
        iCure_dict["ECG_iCure"] = np.array(graph._y)
        iCure_dict["sfreq_ECG_iCure"] = graph._samplerate
    elif graph._title == "x_scgLin[m/s^2]":
        iCure_dict["SCGlin_x"] = np.array(graph._y)
    elif graph._title == "y_scgLin[m/s^2]":
        iCure_dict["SCGlin_y"] = np.array(graph._y)
    elif graph._title == "z_scgLin[m/s^2]":
        iCure_dict["SCGlin_z"] = np.array(graph._y)
    elif graph._title == "x_scgRot[deg/s]":
        iCure_dict["SCGrot_x"] = np.array(graph._y)
    elif graph._title == "y_scgRot[deg/s]":
        iCure_dict["SCGrot_y"] = np.array(graph._y)
    elif graph._title == "z_scgRot[deg/s]":
        iCure_dict["SCGrot_z"] = np.array(graph._y)
        iCure_dict["sfreq_SCG"] = graph._samplerate

for graph in allData1 :
    if graph._title == "ECG":
        mLb_dict["ECG_MatLab"] = np.array(graph._y)
        mLb_dict["sfreq_ECG_MatLab"] = graph._samplerate
    elif graph._title == "BPAo":
        mLb_dict["BPAo"] = np.array(graph._y)
        mLb_dict["sfreq_BPAo"] = graph._samplerate
    elif graph._title == "BPleg":
        mLb_dict["BPleg"] = np.array(graph._y)
        mLb_dict["sfreq_BPleg"] = graph._samplerate

subDict = dict.fromkeys(keys)

subDict["Age"] = data2["Age[y]"]
subDict["Height"] = data2["Height[m]"]
subDict["Sex"] = data2["Sex[m/f]"]
subDict["Weight"] = data2["Weight[kg]"]

"""
Facteurs de risque : hypertension, diabète, pwv
2 courbes de pression : identifier fichier avec début de la courbe de pression

"""



subDict["iCure"] = iCure_dict
subDict["MatLab"] = mLb_dict
myFinalDict = {iD : None, "Data" : subDict}