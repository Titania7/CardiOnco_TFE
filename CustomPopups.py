# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 23:10:36 2023

@author: Tatiana Dehon
"""
from PyQt6.QtCore import pyqtSignal, Qt, QSize
from PyQt6.QtWidgets import *
from PyQt6.QtGui import *


from toolbox import *
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image
from PIL.ImageQt import ImageQt
import io
import os
import json
import neurokit2 as nk

from sklearn.neighbors import LocalOutlierFactor

#%% Other types of PopUps

class DialogPopup(QDialog):
    def __init__(self, title, message):
        super().__init__()

        self.setWindowTitle(title)

        QBtn = QDialogButtonBox.StandardButton.Ok #| QDialogButtonBox.StandardButton.Cancel

        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.accept)
        #self.buttonBox.rejected.connect(self.reject)

        self.layout = QVBoxLayout()
        message = QLabel(message)
        self.layout.addWidget(message)
        self.layout.addWidget(self.buttonBox)
        self.setLayout(self.layout)
        

class OptionPopup(QDialog):
    
    """
    Valuable variables :
    - freq (sampling frequency) => f
    - offset (temporal offset) => b
    - scale (multiplier value) => a
    
    final_y = [ay+b]
    """
    # Connect the metadata to the first window
    newName = pyqtSignal(str)
    newSex = pyqtSignal(str)
    newAge = pyqtSignal(int)
    newWeight = pyqtSignal(float)
    newHeight = pyqtSignal(float)
    
    def __init__(self, name, sex, age, height, weight):
        super().__init__()

        self.setWindowTitle("Metadata modification")
        
        # Begin of stacked widgets
        self.layout = QVBoxLayout()
        
        
        # Beginning of side-by-side widgets
        
        #%% Number 1 : Name
        
        self.nameLabel = QHBoxLayout()
        nametxt = QLabel("Name")
        self.nameLabel.addWidget(nametxt)
        self.nameEdit = QLineEdit()
        self.nameEdit.setText(name)
        self.nameEdit.textChanged.connect(self.nameEdit.setText)
        self.nameLabel.addWidget(self.nameEdit)
        self.layout.addLayout(self.nameLabel)
        
        #%% Number 2 : Sex
        
        self.sexLabel = QHBoxLayout()
        sextxt = QLabel("Sex")
        self.sexLabel.addWidget(sextxt)
        self.sexEdit = QLineEdit()
        self.sexEdit.setText(sex)
        self.sexEdit.textChanged.connect(self.sexEdit.setText)
        self.sexLabel.addWidget(self.sexEdit)
        self.layout.addLayout(self.sexLabel)
        
        
        #%% Number 3 : Age
        
        self.ageLabel = QHBoxLayout()
        agetxt = QLabel("Age [y]")
        self.ageLabel.addWidget(agetxt)
        self.ageEdit = QLineEdit()
        self.ageEdit.setText(str(int(age)))
        self.ageEdit.textChanged.connect(self.ageEdit.setText)
        self.ageLabel.addWidget(self.ageEdit)
        self.layout.addLayout(self.ageLabel)
        
        #%% Number 4 : Height
        
        self.heightLabel = QHBoxLayout()
        heighttxt = QLabel("Height [m]")
        self.heightLabel.addWidget(heighttxt)
        self.heightEdit = QLineEdit()
        self.heightEdit.setText(str(height))
        self.heightEdit.textChanged.connect(self.heightEdit.setText)
        self.heightLabel.addWidget(self.heightEdit)
        self.layout.addLayout(self.heightLabel)
        
        #%% Number 5 : Weight
        
        self.weightLabel = QHBoxLayout()
        weighttxt = QLabel("Weight [kg]")
        self.weightLabel.addWidget(weighttxt)
        self.weightEdit = QLineEdit()
        self.weightEdit.setText(str(weight))
        self.weightEdit.textChanged.connect(self.weightEdit.setText)
        self.weightLabel.addWidget(self.weightEdit)
        self.layout.addLayout(self.weightLabel)
        
        QBtn = QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel

        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.clicked_OK)
        self.buttonBox.rejected.connect(self.reject)
        
        self.layout.addWidget(self.buttonBox)
        
        self.setLayout(self.layout)
        
        
    def clicked_OK(self):
        # Send the new value of visible information to the Main Window
        self.newName.emit(self.nameEdit.text())
        self.newSex.emit(self.sexEdit.text())
        self.newAge.emit(int(self.ageEdit.text()))
        self.newHeight.emit(float(self.heightEdit.text()))
        self.newWeight.emit(float(self.weightEdit.text()))
        self.accept()


class DisplDataPopup(QDialog):
    # Value connected to the Main Window to send the resulting new state
    newVisVect = pyqtSignal(list)
    
    # Add a table that keeps all checkboxes to be used later
    boxes = [] # [checkbox object, name of the object]
    initialState = []
    
    def __init__(self, infoVect, infoVis):
        super().__init__()
        
        self.setWindowTitle("Visible data")
        self.layout = QVBoxLayout()
        message = QLabel("Check the data to be displayed :")
        self.layout.addWidget(message)
        
        # If the user chooses to cancel the operation
        self.intialState = infoVis
        
        
        for i in range(len(infoVect)):
            checkbox = QCheckBox(infoVect[i][0]._title)
            # Fulfills the list of boxes
            self.boxes.append([checkbox, infoVect[i][0]._title])
            if infoVis[i] == True:
                checkbox.setChecked(True)
            else : checkbox.setChecked(False)
            self.layout.addWidget(checkbox)
                

        QBtn = QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel

        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.clicked_OK)
        self.buttonBox.rejected.connect(self.reject)
        
        self.layout.addWidget(self.buttonBox)
        self.setLayout(self.layout)
        
        
    def clicked_OK(self):
        # For each box in the list of boxes available
        for count in range(len(self.boxes)):
            box = self.boxes[count]
            if box[0].isChecked():
                # Verification of the checkbox state
                #print(box[1], " is set to visible !")
                # Update of the Initial State
                self.intialState[count] = True  
            else :
                #print(box[1], " is set to invisible ...")
                self.intialState[count] = False
                
        # Send the new value of visible information to the Main Window
        self.newVisVect.emit(self.intialState)
        self.boxes.clear()
        self.accept()




    
#%% Secondary window (mean graphs)

class DisplMeanWindow(QMainWindow):
    
    ecgML = None
    bpArm = None
    bpLeg = None
    ecgJSON = None
    linSCG = None
    rotSCG = None
    
    allecgML = None
    allbpLeg = None
    allbpArm = None
    allecgJSON = None
    allVectlin = None
    allVectrot = None
 
    
    hrJSON = None
    hrML = None
    qtJSON = None
    qtML = None
    prJSON = None
    prML = None
    
    meanR_Leg = None
    meanR_Arm = None
    
    meanR_AO_4090 = None
    meanR_AO_2dP = None
    pwv_4090 = None
    pwv_2dP = None
    
    meta = []
    
    #%% initialization
    def __init__(self, meta, pqrstJSON, pqrstMAT, infoVect, infoVis, timelim = None):
        super().__init__()
        
        # Create a central widget and set the layout
        self.central_widget = QWidget()
        
        self.meta = meta
        # Add a toolbar
        toolbar = QToolBar()
        self.addToolBar(toolbar)

        save_button = QAction("Save View", self)
        toolbar.addAction(save_button)
        save_button.triggered.connect(self.saveMeanView)
        
        # Create a scroll bar
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)

        # Create a layout to stack the graphs vertically
        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)
        
        self.scroll_area.setWidget(self.central_widget)
        # Set the central widget of the main window
        self.setCentralWidget(self.scroll_area)
        
        
        # Permet de resize les éléments du QVBoxLayout
        self.splitter = QSplitter()
        self.splitter.setOrientation(Qt.Orientation.Vertical)
        #self.splitter.setChildrenCollapsible(True)
        self.splitter.setOpaqueResize(False)
        

        
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setWidget(self.splitter)
        
        
        
        #self.setMinimumSize(700, 900)
        # Full screen mode
        self.showMaximized()
        self.setWindowTitle("Mean graphs display")
        self.setWindowIcon(QIcon("icon.png"))
        
        pwvtxt_meta = QLabel("Identifier : "+meta[0]+" ; Sex : "+ meta[1]+" ; Age : "+str(meta[2])+" years ; Weight : "+ str(meta[3])+ " kg ; Height : "+ str(meta[4])+" m")
        self.layout.addWidget(pwvtxt_meta)
        
        
        
        bpLeg_ons = []
        bpArm_ons = []
        startindex = None
        stopindex = None
        
        
        #%% get the peaks for both ECGs
        
        [p_pML, q_pML, r_pML, s_pML, t_pML, p_onML, t_offML] = pqrstMAT
        [p_pJSON, q_pJSON, r_pJSON, s_pJSON, t_pJSON, p_onJSON, t_offJSON] = pqrstJSON
        
       
        
        # Make them (p_pML, q_pML etc) all the same length
        for pqrstList in [pqrstMAT, pqrstJSON]:
            min_len = np.min([len(item[0]) for item in pqrstList])
            for item in pqrstList:
                while len(item[0]) > min_len:
                    item[0].pop()
                    item[1].pop()

        #%% Retrieval + identification of all the tracks available
        

        legCut = False
        for i in range(len(infoVect)):
            if infoVis[i] == True:
                if infoVect[i][0]._title == "ECG mat":
                    ecgML = infoVect[i][0]
                elif infoVect[i][0]._title == "ECG json":
                    ecgJSON = infoVect[i][0]
                elif infoVect[i][0]._title == "BPleg":
                    bpLeg = infoVect[i][0]
                    # We get the BPleg onsets on the whole graph then on the timelim if non-successful
                    try:
                        onsBPLeg, bpLegY = getBpOnsets_tang(bpLeg, r_pML, filt = True, show = False)
                        print("BPleg whole graph onset detection with tangent")
                    except:
                        try:
                            onsBPLeg, bpLegY = getBpOnsets_2dDeriv(bpLeg, r_pML, filt = True, show = False)
                            print("BPleg whole graph onset detection with 2d deriv")
                        except:
                            try:
                                onsBPLeg, bpLegY = getBpOnsets_tang(bpLeg, r_pML, lims=timelim, filt = True, show = False)
                                print("BPleg partial graph onset detection with tangent")
                                legCut = True
                            except:
                                try:
                                    onsBPLeg, bpLegY = getBpOnsets_2dDeriv(bpLeg, r_pML, lims=timelim, filt = True, show = False)
                                    print("BPleg partial graph onset detection with 2d deriv")
                                    legCut = True
                                except:
                                    DialogPopup("Error", "The BP onsets detection failed for BPleg").exec()
                            
                            startindex = int(timelim[0]*bpLeg._samplerate)
                            stopindex = int(timelim[1]*bpLeg._samplerate)
                            onsBPLeg[0] = list(np.array(onsBPLeg[0])-startindex)
                            onsBPLeg[1] = list(np.array(onsBPLeg[1])-timelim[0])
                            
                            """ #Verification => OK
                            plt.plot(bpLeg._x[startindex:stopindex], bpLeg._y[startindex:stopindex])
                            plt.plot(bpLeg._x[onsBPLeg[0]]+timelim[0], bpLegY, 'o')
                            plt.title("Verification of the onsets detections for BPleg")
                            plt.show()
                            """

                elif infoVect[i][0]._title == "BParm":
                    orig_bpArm = infoVect[i][0]
                    # We take into account the time limitations asked by the user for this one
                    startindex = int(timelim[0]*orig_bpArm._samplerate)
                    stopindex = int(timelim[1]*orig_bpArm._samplerate)
                    # We create another instance of bpArm in order to update the timelims at each instance of this window
                    self.bpArm = SingleGraph(orig_bpArm._x[startindex:stopindex], orig_bpArm._y[startindex:stopindex], orig_bpArm._title, orig_bpArm._samplerate, orig_bpArm._step)

                    # We get the BParm onsets on the time limitations given by the user
                    try:
                        onsBPArm, bpArmY = getBpOnsets_tang(orig_bpArm, r_pML, lims=timelim, filt = True, show = False)    
                        print("BParm partial graph onset detection with tangent")
                    except:
                        try:
                            onsBPArm, bpArmY = getBpOnsets_2dDeriv(orig_bpArm, r_pML, lims=timelim, filt = True, show = False)
                            print("BParm partial graph onset detection with 2d deriv")
                        except: 
                            DialogPopup("Error", "The BP onsets detection failed for the given time limits of BParm").exec()

                        
                    onsBPArm[0] = list(np.array(onsBPArm[0])-startindex)
                    onsBPArm[1] = list(np.array(onsBPArm[1])-timelim[0])
                    
                    
                    """ #Verification => OK
                    plt.plot(self.bpArm._x, self.bpArm._y)
                    plt.plot(self.bpArm._x[onsBPArm[0]], bpArmY, 'o')
                    plt.title("Verification of the restricted zone for BParm")
                    plt.show()
                    """
                
                elif infoVect[i][0]._title == "x_scgLin[m/s^2]":
                    xlinSCG = infoVect[i][0]
                elif infoVect[i][0]._title == "y_scgLin[m/s^2]":
                    ylinSCG = infoVect[i][0]
                elif infoVect[i][0]._title == "z_scgLin[m/s^2]":
                    zlinSCG = infoVect[i][0]
                elif infoVect[i][0]._title == "x_scgRot[deg/s]":
                    xrotSCG = infoVect[i][0]
                elif infoVect[i][0]._title == "y_scgRot[deg/s]":
                    yrotSCG = infoVect[i][0]
                elif infoVect[i][0]._title == "z_scgRot[deg/s]":
                    zrotSCG = infoVect[i][0]
                
                elif infoVect[i][0]._title == "lin_SCGvect[m/s^2]":
                    linSCG = infoVect[i][0]
                elif infoVect[i][0]._title == "rot_SCGvect[deg/s]":
                    rotSCG = infoVect[i][0]

        #%% prepare the lists of arrays for the mean display
        
        mean_ecgML = []
        mean_bpLeg = []
        mean_bpArm = []
        
        authorizedList = []
        
        # Go through the ecg_indexes of the r peaks (matlab graphs)
        for i in range(len(r_pML[0])-1):
            
            #print(startindex, r_pML[0][i]-startindex, r_pML[0][i+1]-startindex, len(self.bpArm._x))
            
            if r_pML[0][i+1] < len(ecgML._x):
                y_ecgML = ecgML._y[r_pML[0][i]:r_pML[0][i+1]]
                mean_ecgML.append(y_ecgML)
                authorizedList.append(i)
            if r_pML[0][i+1] < len(bpLeg._x):
                y_bpLeg = bpLeg._y[r_pML[0][i]:r_pML[0][i+1]]
                mean_bpLeg.append(y_bpLeg)
            if r_pML[0][i]-startindex> 0 and r_pML[0][i+1]-startindex < len(self.bpArm._x):
                y_bpArm = self.bpArm._y[r_pML[0][i]-startindex:r_pML[0][i+1]-startindex]
                mean_bpArm.append(y_bpArm)
        
        
        p_onML0 = [p_onML[0][i] for i in range(len(p_onML[0])) if i in authorizedList]
        p_onML1 = [p_onML[1][i] for i in range(len(p_onML[1])) if i in authorizedList]
        p_onML = [p_onML0, p_onML1]
        p_pML0 = [p_pML[0][i] for i in range(len(p_pML[0])) if i in authorizedList]
        p_pML1 = [p_pML[1][i] for i in range(len(p_pML[1])) if i in authorizedList]
        p_pML = [p_pML0, p_pML1]
        q_pML0 = [q_pML[0][i] for i in range(len(q_pML[0])) if i in authorizedList]
        q_pML1 = [q_pML[1][i] for i in range(len(q_pML[1])) if i in authorizedList]
        q_pML = [q_pML0, q_pML1]
        r_pML0 = [r_pML[0][i] for i in range(len(r_pML[0])) if i in authorizedList]
        r_pML1 = [r_pML[1][i] for i in range(len(r_pML[1])) if i in authorizedList]
        r_pML = [r_pML0, r_pML1]
        s_pML0 = [s_pML[0][i] for i in range(len(s_pML[0])) if i in authorizedList]
        s_pML1 = [s_pML[1][i] for i in range(len(s_pML[1])) if i in authorizedList]
        s_pML = [s_pML0, s_pML1]
        t_pML0 = [t_pML[0][i] for i in range(len(t_pML[0])) if i in authorizedList]
        t_pML1 = [t_pML[1][i] for i in range(len(t_pML[1])) if i in authorizedList]
        t_pML = [t_pML0, t_pML1]
        t_offML0 = [t_offML[0][i] for i in range(len(t_offML[0])) if i in authorizedList]
        t_offML1 = [t_offML[1][i] for i in range(len(t_offML[1])) if i in authorizedList]
        t_offML = [t_offML0, t_offML1]
        
        
        


        # Artificially put the beginning level of all the bpArm and bpLeg chunks at the same amplitude at the beginning
        minAmpl_bpArm = []
        minAmpl_bpLeg = []
        
        for chunk in mean_bpArm:
            minAmpl_bpArm.append(chunk[0])
        for chunk in mean_bpLeg:
            minAmpl_bpLeg.append(chunk[0])
        minAmpl_bpArm = np.min(minAmpl_bpArm)
        minAmpl_bpLeg = np.min(minAmpl_bpLeg)
        #print("Minimum amplitudes for bpleg and bparm :", minAmpl_bpLeg, minAmpl_bpArm)
        for i in range(len(mean_bpArm)):
            mean_bpArm[i] = list(np.array(mean_bpArm[i] - (mean_bpArm[i][0]- minAmpl_bpArm)))
        for i in range(len(mean_bpLeg)):
            mean_bpLeg[i] = list(np.array(mean_bpLeg[i] - (mean_bpLeg[i][0]- minAmpl_bpLeg)))

        

        # Definition of the minimum length to take into account the mean graphs
        alldiff = []
        for i in range(len(r_pJSON[0])-1):
            # conversion of the indexes
            minIndex_scg = index_conv(index=r_pJSON[0][i], scg_fs=linSCG._samplerate , ecg_fs=ecgJSON._samplerate, indexType="Ecg2scg")
            maxIndex_scg = index_conv(index=r_pJSON[0][i+1], scg_fs=linSCG._samplerate , ecg_fs=ecgJSON._samplerate, indexType="Ecg2scg")
            diff = maxIndex_scg-minIndex_scg
            alldiff.append(diff)
        length = np.min(alldiff)
        
        mean_ecgJSON = []
        all_xlin = []
        all_ylin = []
        all_zlin = []
        all_xrot = []
        all_yrot = []
        all_zrot = []
        # Stock all the in-between graphs (and prepare SCG for average graph)
        allVectlin = []
        allVectrot = []
        
        #print("all lengths = ", len(ecgJSON._y), len(linSCG._y), len(rotSCG._y))
        
        # Go through the ecg_indexes of the r peaks
        for i in range(len(r_pJSON[0])-1):
            # We align every graph on the R peak
            y_ecgJSON = ecgJSON._y[r_pJSON[0][i]:r_pJSON[0][i+1]]
            
            # conversion of the indexes
            minIndex_scg = index_conv(index=r_pJSON[0][i], scg_fs=linSCG._samplerate , ecg_fs=ecgJSON._samplerate, indexType="Ecg2scg")
            maxIndex_scg = index_conv(index=r_pJSON[0][i+1], scg_fs=rotSCG._samplerate , ecg_fs=ecgJSON._samplerate, indexType="Ecg2scg")
            
            #print(r_pJSON[0][i], minIndex_scg, ":", r_pJSON[0][i+1], maxIndex_scg)
            
            y_xLin = xlinSCG._y[minIndex_scg:maxIndex_scg]
            y_xRot = xrotSCG._y[minIndex_scg:maxIndex_scg]
            y_yLin = ylinSCG._y[minIndex_scg:maxIndex_scg]
            y_yRot = yrotSCG._y[minIndex_scg:maxIndex_scg]
            y_zLin = zlinSCG._y[minIndex_scg:maxIndex_scg]
            y_zRot = zrotSCG._y[minIndex_scg:maxIndex_scg]
            y_vectLin = linSCG._y[minIndex_scg:maxIndex_scg]
            y_vectRot = rotSCG._y[minIndex_scg:maxIndex_scg]

            mean_ecgJSON.append(y_ecgJSON)
            
            # Sum of the graphs in each cycle
            all_xlin.append(y_xLin[0:length-1])
            all_ylin.append(y_yLin[0:length-1])
            all_zlin.append(y_zLin[0:length-1])
            all_xrot.append(y_xRot[0:length-1])
            all_yrot.append(y_yRot[0:length-1])
            all_zrot.append(y_zRot[0:length-1])
            allVectlin.append(y_vectLin[0:length-1])
            allVectrot.append(y_vectRot[0:length-1])
        
        for a in [p_pJSON, q_pJSON, r_pJSON, s_pJSON, t_pJSON, p_onJSON, t_offJSON]:
            if len(mean_ecgJSON) < len(a[0]):
                a[0].pop()
                a[1].pop()
        

        #print(len(mean_ecgML), len(r_pML[0]), len(s_pML[0]), len(t_pML[0]), len(t_offML[0]), len(p_onML[0]), len(p_pML[0]), len(q_pML[0]))
        #print(len(mean_ecgJSON), len(r_pJSON[0]), len(s_pJSON[0]), len(t_pJSON[0]), len(t_offJSON[0]), len(p_onJSON[0]), len(p_pJSON[0]), len(q_pJSON[0]))

        
        #%% Rejection of all the aberrant graphs for the superposed graphs
        
        try:
            mean_ecgML, idx_ecgML = cleanLOF(mean_ecgML, 10, 0.25)
        # Make sure all the graphs are the same length (max one)
        except: mean_ecgML = self.all_SameLen(mean_ecgML) ; print("Rejection of outliers graphs for ecgML unsuccessful")
        
        try: mean_bpLeg, idx_bpLeg = cleanLOF(mean_bpLeg, 5, 0.3)
        except: mean_bpLeg = self.all_SameLen(mean_bpLeg) ; print("Rejection of outliers graphs for bpLeg unsuccessful")
        
        # Does not work with the bpArm if it is too short
        try: mean_bpArm, idx_bpArm = cleanLOF(mean_bpArm, 1, 0.05)
        except: mean_bpArm = self.all_SameLen(mean_bpArm) ; print("Rejection of outliers graphs for bpArm unsuccessful")
        
        try: mean_ecgJSON, idx_ecgJSON = cleanLOF(mean_ecgJSON, 10, 0.25)
        except: mean_ecgJSON = self.all_SameLen(mean_ecgJSON) ; print("Rejection of outliers graphs for ecgJSON unsuccessful")
      
        try: allVectlin, idx_Vectlin = cleanLOF(allVectlin)
        except: allVectlin = self.all_SameLen(allVectlin) ; print("Rejection of outliers graphs for scgLin unsuccessful")
        
        try: allVectrot, idx_Vectrot = cleanLOF(allVectrot)
        except: allVectrot = self.all_SameLen(allVectrot) ; print("Rejection of outliers graphs for scgRot unsuccessful")
        
        
        try: all_xlin, idx_xLin = cleanLOF(all_xlin)
        except: all_xlin = self.all_SameLen(all_xlin) ; print("Rejection of outliers graphs for xLin unsuccessful")
        
        try: all_ylin, idx_yLin = cleanLOF(all_ylin)
        except: all_ylin = self.all_SameLen(all_ylin) ; print("Rejection of outliers graphs for yLin unsuccessful")
        
        try: all_zlin, idx_zLin = cleanLOF(all_zlin)
        except: all_zlin = self.all_SameLen(all_zlin) ; print("Rejection of outliers graphs for zLin unsuccessful")
        
        try: all_xrot, idx_xRot = cleanLOF(all_xrot)
        except: all_xrot = self.all_SameLen(all_xrot) ; print("Rejection of outliers graphs for xRot unsuccessful")
        
        try: all_yrot, idx_yRot = cleanLOF(all_yrot)
        except: all_yrot = self.all_SameLen(all_yrot) ; print("Rejection of outliers graphs for yRot unsuccessful")
        
        try: all_zrot, idx_zRot = cleanLOF(all_zrot)
        except: all_zrot = self.all_SameLen(all_zrot) ; print("Rejection of outliers graphs for zRot unsuccessful")

        
        
        
        #%% Computation of the mean SCG graphs values
        
        meanAllvect = np.mean(allVectlin, axis=0)
        meanAllvectrot = np.mean(allVectrot, axis=0)
        
        relAO_4090 = getAO_4090ms(allVectlin, linSCG._samplerate, show = False)
        relAO_2dPeak = getAO_2pAfter40ms(allVectlin, linSCG._samplerate, show = False)
        
        #%% Definition of colors for the graphs
        
        rouge = QtGui.QBrush(QtGui.QColor(255, 0, 0, 85)) #R
        orange = QtGui.QBrush(QtGui.QColor(255, 128, 0, 85)) #S
        bleu = QtGui.QBrush(QtGui.QColor(0, 128, 255, 85)) #T
        vert = QtGui.QBrush(QtGui.QColor(0, 153, 0, 85)) #Toff
        violet = QtGui.QBrush(QtGui.QColor(51, 0, 102, 85)) #Pon
        turquoise = QtGui.QBrush(QtGui.QColor(0, 204, 204, 85)) #P
        rose = QtGui.QBrush(QtGui.QColor(255, 0, 255, 85)) #Q
        
        #%% Display of all the mean graphs (6)
        
        self.allecgML, self.allbpLeg, self.allbpArm, self.allecgJSON, self.allVectlin, self.allVectrot = mean_ecgML, mean_bpLeg, mean_bpArm, mean_ecgJSON, allVectlin, allVectrot
        self.ecgML, self.bpLeg, self.bpArm, self.ecgJSON, self.linSCG, self.rotSCG = ecgML, bpLeg, self.bpArm, ecgJSON, linSCG, rotSCG
        
        def returnCleaned(liste2D, indexes):
            new0 = []
            new1 = []
            new0 = [liste2D[0][i] for i in range(len(liste2D[0])) if i not in indexes]
            new1 = [liste2D[1][i] for i in range(len(liste2D[1])) if i not in indexes]
            new = [new0, new1]
            
            return new
        
        
        
        ecgp_pML = returnCleaned(p_pML, idx_ecgML)
        ecgq_pML = returnCleaned(q_pML, idx_ecgML)    
        ecgr_pML = returnCleaned(r_pML, idx_ecgML)
        ecgs_pML = returnCleaned(s_pML, idx_ecgML)
        ecgt_pML = returnCleaned(t_pML, idx_ecgML)
        ecgp_onML = returnCleaned(p_onML, idx_ecgML)
        ecgt_offML = returnCleaned(t_offML, idx_ecgML)
        pqrstMATcl = [ecgp_pML, ecgq_pML, ecgr_pML, ecgs_pML, ecgt_pML, ecgp_onML, ecgt_offML]
        
        ecgp_pJSON = returnCleaned(p_pJSON, idx_ecgJSON)
        ecgq_pJSON = returnCleaned(q_pJSON, idx_ecgJSON)
        ecgr_pJSON = returnCleaned(r_pJSON, idx_ecgJSON)
        ecgs_pJSON = returnCleaned(s_pJSON, idx_ecgJSON)
        ecgt_pJSON = returnCleaned(t_pJSON, idx_ecgJSON)
        ecgp_onJSON = returnCleaned(p_onJSON, idx_ecgJSON)
        ecgt_offJSON = returnCleaned(t_offJSON, idx_ecgJSON)
        pqrstJSONcl = [ecgp_pJSON, ecgq_pJSON, ecgr_pJSON, ecgs_pJSON, ecgt_pJSON, ecgp_onJSON, ecgt_offJSON]
        
        
        # Add the ECGs centered on the R peaks instead
        self.addECG_centerR(ecgML, pqrstMATcl, realR=r_pML)
        self.printHRV(ecgML, pqrstMAT)
        self.addECG_centerR(ecgJSON, pqrstJSONcl, realR=r_pJSON)
        self.printHRV(ecgJSON, pqrstJSON)
        
        onsets = onsBPLeg
        if legCut == False :
            temp_Rp = r_pML[0]
            while len(temp_Rp) > len(onsets[0]):
                temp_Rp.pop()
            onsets_rel = np.array(onsets[0]) - np.array(temp_Rp)
        else :
            temp_Rp = [x for x in r_pML[0] if x>timelim[0]*self.bpLeg._samplerate and x<timelim[1]*self.bpLeg._samplerate]
            while len(temp_Rp) > len(onsets[0]):
                temp_Rp.pop()
            temp_Rp = np.array(temp_Rp) - int(timelim[0]*self.bpLeg._samplerate)
            onsets_rel = np.array(onsets[0]) - np.array(temp_Rp)
        onsBPLeg_rel = onsets_rel
        
        onsets = onsBPArm
        temp_Rp = [x for x in r_pML[0] if x>timelim[0]*self.bpArm._samplerate and x<timelim[1]*self.bpArm._samplerate]
        while len(temp_Rp) > len(onsets[0]):
            temp_Rp.pop()
        temp_Rp = np.array(temp_Rp) - int(timelim[0]*self.bpArm._samplerate)
        onsets_rel = np.array(onsets[0]) - np.array(temp_Rp)
        onsBPArm_rel = onsets_rel

        onsBPArmCl, _ = returnCleaned([onsBPArm_rel,onsBPArm_rel], idx_bpArm)
        onsBPLegCl, _ = returnCleaned([onsBPLeg_rel,onsBPLeg_rel], idx_bpLeg)
        
        
        self.addBP(self.bpLeg, mean_bpLeg, onsBPLegCl)
        self.addBP(self.bpArm, mean_bpArm, onsBPArmCl)
        

        


        
        
        
        #%%
        listToDisplay = [allVectlin, allVectrot]
        corresponding = [self.linSCG, self.rotSCG]

        refPlotWidget = None
        # i is the index of the current R peak
        
        for i in range(len(listToDisplay)) : # passera 6 fois (1 pour chaque graphe moyenné)
            graphWidget = pg.PlotWidget()              
            countTrace = 0
            for trace in listToDisplay[i]: # Passera 71 fois (1 pour chaque morceau du graphe courant)
                x = np.arange(0, len(trace)*corresponding[i]._step, corresponding[i]._step)
                
                while len(x)>len(trace):
                    x=np.delete(x,-1)
                
                # Plot a single trace
                graphWidget.addLegend()
                courbe = graphWidget.plot(x, trace, pen="grey")
                courbe.setOpacity(0.3)
                
                
                if corresponding[i]._title == "lin_SCGvect[m/s^2]":
                    point4090 = relAO_4090[countTrace]
                    point_2dpeak = relAO_2dPeak[countTrace]
                    points4090 = pg.ScatterPlotItem(x=[x[point4090]], y=[trace[point4090]], brush=rose, pen=None)
                    graphWidget.addItem(points4090)
                    points2dP = pg.ScatterPlotItem(x=[x[point_2dpeak]], y=[trace[point_2dpeak]], brush=turquoise, pen=None)
                    graphWidget.addItem(points2dP)
                  
                            
                countTrace += 1
                    
            #%% Add the mean traces of the SCGs in red
            if corresponding[i]._title == "lin_SCGvect[m/s^2]":
                posStart = x[0]
                posStop = x[-1]
                self.aoCursor = pg.InfiniteLine(pos=posStart, bounds=[posStart, posStop], label='Manual AO', angle=90, movable=True, pen=pg.mkPen(width=3))
                graphWidget.addItem(self.aoCursor)
                self.aoCursor.sigPositionChanged.connect(self.manualAO)
                
                x = np.arange(0, len(meanAllvect)*linSCG._step, linSCG._step)
                if len(x)>len(meanAllvect):
                    x = np.delete(x,-1)
                elif len(meanAllvect)>len(x):
                    meanAllvect = np.delete(meanAllvect,-1)
                
                courbe = graphWidget.plot(x, meanAllvect, pen='red')
                points1 = pg.ScatterPlotItem(x=[x[int(np.round(np.mean(relAO_4090)))]], y=[meanAllvect[int(np.round(np.mean(relAO_4090)))]], brush = "magenta", pen=None)
                graphWidget.addItem(points1)
                points2 = pg.ScatterPlotItem(x=[x[int(np.round(np.mean(relAO_2dPeak)))]], y=[meanAllvect[int(np.round(np.mean(relAO_2dPeak)))]], brush = "turquoise", pen=None)
                graphWidget.addItem(points2) 
            
            
                
                
            elif corresponding[i]._title == "rot_SCGvect[deg/s]":
                x = np.arange(0, len(meanAllvectrot)*rotSCG._step, rotSCG._step)
                if len(x)>len(meanAllvect):
                    x = np.delete(x,-1)
                elif len(meanAllvect)>len(x):
                    meanAllvect = np.delete(meanAllvect,-1)
                    
                courbe = graphWidget.plot(x, meanAllvectrot, pen='red')
                
            graphWidget.setTitle("Mean "+corresponding[i]._title)
            graphWidget.setLabel('left', 'Magnitude')
            graphWidget.setLabel('bottom', 'Time [s]')
            graphWidget.setMinimumHeight(300)
            #graphWidget.addLegend()
            graphWidget.showGrid(x=True, y=True)
            
            # Make all the graphs share the same x axis
            if i == 0:
                refPlotWidget = graphWidget
            else :
                graphWidget.setXLink(refPlotWidget)
            
            self.layout.addWidget(graphWidget)
            
            if corresponding[i]._title == "BPleg" or corresponding[i]._title == "BParm":
                legend = QLabel("<font color='red'>●</font> BP onsets")
                legend.setAlignment(Qt.AlignmentFlag.AlignCenter)
                self.layout.addWidget(legend)
                
            elif corresponding[i]._title == "lin_SCGvect[m/s^2]":
                legend = QLabel("<font color='magenta'>●</font> AO detected 40-90 ms ; <font color='turquoise'>●</font> AO detected as 2d peak")
                legend.setAlignment(Qt.AlignmentFlag.AlignCenter)
                self.layout.addWidget(legend)
                
            if corresponding[i]._title == "lin_SCGvect[m/s^2]":
                 thenToDisplay = [all_xlin,all_ylin,all_zlin]
                 thencorresp = [xlinSCG,ylinSCG,zlinSCG]
                 for i, graph in enumerate(thenToDisplay):
                     graphWidget = pg.PlotWidget()
                     for trace in graph:
                         x = np.arange(0, len(trace)*thencorresp[i]._step, thencorresp[i]._step)
                         while len(x)>len(trace):
                             x=np.delete(x,-1)
                         courbe = graphWidget.plot(x, trace, pen="grey")
                         courbe.setOpacity(0.3)
                     courbe = graphWidget.plot(x, np.mean(np.array(graph), axis = 0), pen="magenta")
                     graphWidget.setTitle("Mean "+thencorresp[i]._title)
                     graphWidget.setLabel('left', 'Magnitude')
                     graphWidget.setLabel('bottom', 'Time [s]')
                     graphWidget.setMinimumHeight(250)
                     graphWidget.showGrid(x=True, y=True)
                     self.layout.addWidget(graphWidget)
             
            elif corresponding[i]._title == "rot_SCGvect[deg/s]":
                 thenToDisplay = [all_xrot,all_yrot,all_zrot]
                 thencorresp = [xrotSCG,yrotSCG,zrotSCG]
                 for i, graph in enumerate(thenToDisplay):
                     graphWidget = pg.PlotWidget()
                     for trace in graph:
                         x = np.arange(0, len(trace)*thencorresp[i]._step, thencorresp[i]._step)
                         while len(x)>len(trace):
                             x=np.delete(x,-1)
                         courbe = graphWidget.plot(x, trace, pen="grey")
                         courbe.setOpacity(0.3)
                     courbe = graphWidget.plot(x, np.mean(np.array(graph), axis = 0), pen="magenta")
                     graphWidget.setTitle("Mean "+thencorresp[i]._title)
                     graphWidget.setLabel('left', 'Magnitude')
                     graphWidget.setLabel('bottom', 'Time [s]')
                     graphWidget.setMinimumHeight(250)
                     graphWidget.showGrid(x=True, y=True)
                     self.layout.addWidget(graphWidget)

            
        
    #%% Display of the MetaData and the approximative PWV
        
        
        
        
        # Computation of the mean R-Leg and R-Arm
        self.meanR_Leg = np.mean(onsBPLeg_rel)*self.bpLeg._step
        self.meanR_Arm = np.mean(onsBPArm_rel)*self.bpArm._step
        
        self.stdR_Leg = np.std(onsBPLeg_rel*self.bpLeg._step)
        self.lenR_Leg = len(onsBPLeg_rel)
        self.stdR_Arm = np.std(onsBPArm_rel*self.bpArm._step)
        self.lenR_Arm = len(onsBPArm_rel)
        
        # Computation of the mean R-AO for each technique
        self.meanR_AO_4090 = np.mean(relAO_4090)*linSCG._step
        self.meanR_AO_2dP = np.round(np.mean(relAO_2dPeak))*linSCG._step
        
        self.stdR_AO_4090 = np.std(relAO_4090*linSCG._step)
        self.stdR_AO_2dP = np.std(relAO_2dPeak*linSCG._step)
        
        # Computation of the mean PWV for each AO measure
        self.pwv_4090 = (0.8*meta[-1])/(self.meanR_Leg - self.meanR_AO_4090)
        self.pwv_2dP = (0.8*meta[-1])/(self.meanR_Leg - self.meanR_AO_2dP)
        
        
        
        #%% Prepare the displays
        #txt_HR = QLabel("Mean HR (JSON/ML) = "+ str(self.hrJSON) +" +- "+str(np.round(self.stdhrJSON,3)) +' (N = '+str(self.lenHRjson)+') / '+ str(self.hrML)+" +- "+str(np.round(self.stdhrML,3)) +" (N = "+str(self.lenHRml)+") bpm")
        #txt_QT = QLabel("Mean Q-T intervals (JSON/ML) = "+str(self.qtJSON)+" +- "+str(np.round(self.stdQTjson,3)) +' (N = '+str(self.lenQTjson)+') / '+str(self.qtML)+" +- "+str(np.round(self.stdQTml,3)) +" (N = "+str(self.lenQTml)+") s")
        #txt_PR = QLabel("Mean P-R intervals (JSON/ML) = "+str(self.prJSON)+" +- "+str(np.round(self.stdPRjson,3)) +' (N = '+str(self.lenPRjson)+') / '+str(self.prML)+" +- "+str(np.round(self.stdPRml,3)) +" (N = "+str(self.lenPRml)+") s")
        txt_leg_arm = QLabel("Mean R-bpLeg onset delay : "+str(np.round(self.meanR_Leg,3))+" +- "+str(np.round(self.stdR_Leg,3))+" s (N = "+str(self.lenR_Leg)+") ; Mean R-bpArm onset delay : "+str(np.round(self.meanR_Arm,3))+" +- "+str(np.round(self.stdR_Arm,3))+" s (N = "+str(self.lenR_Arm)+")")
        pwvtxt_AO_4090 = QLabel("Delay R-AO (40-90 ms technique): "+str(np.round(self.meanR_AO_4090,3))+" +- "+str(np.round(self.stdR_AO_4090,3))+" s => afPWV = "+ str(np.round(self.pwv_4090 ,3))+" m/s")
        pwvtxt_AO_2dP = QLabel("Delay R-AO (2d peak techniaue): "+str(np.round(self.meanR_AO_2dP,3))+" +- "+str(np.round(self.stdR_AO_2dP,3))+" s => afPWV = "+str(np.round(self.pwv_2dP ,3))+" m/s")
        self.manualAO_label = QLabel(f"Manual AO : {self.aoCursor.pos().x()} s => afPWV = {self.meanR_Leg - self.aoCursor.pos().x()} m/s")
        # Display the results
        for txt in [txt_leg_arm, pwvtxt_AO_4090, pwvtxt_AO_2dP, self.manualAO_label]:#pwvtxt_AO, pwvtxt_leg_arm, pwvtxt]:
            self.layout.addWidget(txt)

        
        self.showMaximized()
    
    #%% Useful functions
    def saveMeanView(self):
        
        pix = self.central_widget.grab()
                
        name, sex, age, weight, height = self.meta
        
        # Get the Desktop path of the user
        path_desktop = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop')

        # Nom du dossier d'enregistrements
        name_folder = "CardiOnco recordings"

        # Complete path to the folder
        path_folder = os.path.join(path_desktop, name_folder)
        path_patientfolder = os.path.join(path_folder, name[0:3])

        # Check if the folder already exists
        if not os.path.exists(path_patientfolder): # If not, create it
            DialogPopup("Warning", "Associated forlder does not exist yet.\nPlease register the raw file before the mean graphs.").exec()
        else :
            # Save the QPixmap to a PNG file
            file_name = os.path.join(path_patientfolder, name+".png")
            pix.save(file_name)
        return
    
    def manualAO(self):
        newPosition = self.aoCursor.pos().x()
        newPWV = (0.8*self.meta[-1])/(self.meanR_Leg - newPosition)
        self.manualAO_label.setText(f"Manual AO : "+str(np.round(newPosition,3))+" s => afPWV = "+str(np.round(newPWV, 3))+" m/s")
    
    def addBP(self, bpSC, mean_bp, onsBPrel):
        
        rouge = QtGui.QBrush(QtGui.QColor(255, 0, 0, 85)) #R
        
        graphWidget = pg.PlotWidget()
        
        for i, trace in enumerate(mean_bp):
            x = np.arange(0, len(trace)*bpSC._step, bpSC._step)
            
            while len(x)>len(trace):
                x=np.delete(x,-1)
            
            # Plot a single trace
            courbe = graphWidget.plot(x, trace, pen="grey")
            courbe.setOpacity(0.3)     
            
            points = pg.ScatterPlotItem(x=[x[onsBPrel[i]]], y=[trace[onsBPrel[i]]], brush=rouge, pen=None)
            graphWidget.addItem(points) 
         
        
        graphWidget.setTitle("Mean "+bpSC._title)
        graphWidget.setLabel('left', 'Magnitude')
        graphWidget.setLabel('bottom', 'Time [s]')
        graphWidget.setMinimumHeight(300)
        #graphWidget.addLegend()
        graphWidget.showGrid(x=True, y=True)
        self.layout.addWidget(graphWidget)
        
        text = "<font color='red'>●</font> BP onsets"
        legend = QLabel(text)
        legend.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(legend)
        
        
        
    
    def addECG_centerR(self, ecg, pqrst, realR):
        
        [p_p, q_p, r_p, s_p, t_p, p_on, t_off] = pqrst

        # Compute the mean temporal difference between 2 R peaks
        rrTime = np.mean(np.diff(r_p[1]))
        halfRRtime = rrTime/2
        
        trois4RRindexes = int(np.round(np.mean(halfRRtime*ecg._samplerate)))
        halfRRindexes = int(np.round(np.mean(halfRRtime*ecg._samplerate)))

        meanECG = []
        nextP = []
        nextQ = []
        nextR = []
        nextS = []
        nextT = []
        nextToff = []
        nextPon = []
        
        for i in range(len(r_p[0])-1):
            if r_p[0][i] > halfRRindexes and len(ecg._y)> r_p[0][i]+trois4RRindexes:
                y_ecg = ecg._y[r_p[0][i]-halfRRindexes:r_p[0][i]+trois4RRindexes]
                meanECG.append(y_ecg)
                nextR.append(r_p[0][i] + halfRRindexes - r_p[0][i])
                
                
                
                
                if p_on[0][i] - r_p[0][i] < 0 and p_on[0][i] - r_p[0][i] > -100:
                    nextPon.append(p_on[0][i] + halfRRindexes - r_p[0][i])
                elif p_on[0][i] - r_p[0][i-1] < 0 and p_on[0][i] - r_p[0][i-1] > -100:
                    nextPon.append(p_on[0][i] + halfRRindexes - r_p[0][i-1])
                elif p_on[0][i] - r_p[0][i+1] < 0 and p_on[0][i] - r_p[0][i+1] > -100:
                    nextPon.append(p_on[0][i] + halfRRindexes - r_p[0][i+1])
                
                if p_p[0][i] - r_p[0][i] < 0 and p_p[0][i] - r_p[0][i] > -100:
                    nextP.append(p_p[0][i] + halfRRindexes - r_p[0][i])
                elif p_p[0][i] - r_p[0][i-1] < 0 and p_p[0][i] - r_p[0][i-1] > -100:
                    nextP.append(p_p[0][i] + halfRRindexes - r_p[0][i-1])
                elif p_p[0][i] - r_p[0][i+1] < 0 and p_p[0][i] - r_p[0][i+1] > -100:
                    nextP.append(p_p[0][i] + halfRRindexes - r_p[0][i+1])
                
                if q_p[0][i] - r_p[0][i] < 0 and q_p[0][i] - r_p[0][i] > -100:
                    nextQ.append(q_p[0][i] + halfRRindexes - r_p[0][i])
                elif q_p[0][i] - r_p[0][i-1] < 0 and q_p[0][i] - r_p[0][i-1] > -100:
                    nextQ.append(q_p[0][i] + halfRRindexes - r_p[0][i-1])
                elif q_p[0][i] - r_p[0][i+1] < 0 and q_p[0][i] - r_p[0][i+1] > -100:
                    nextQ.append(q_p[0][i] + halfRRindexes - r_p[0][i+1])
                
                
                if s_p[0][i] - r_p[0][i] > 0 and s_p[0][i] - r_p[0][i] < 100:
                    nextS.append(s_p[0][i] + halfRRindexes - r_p[0][i])
                elif s_p[0][i] - r_p[0][i-1] > 0 and s_p[0][i] - r_p[0][i-1] < 100:
                    nextS.append(s_p[0][i] + halfRRindexes - r_p[0][i-1])
                elif s_p[0][i] - r_p[0][i+1] > 0 and s_p[0][i] - r_p[0][i+1] < 100:
                    nextS.append(s_p[0][i] + halfRRindexes - r_p[0][i+1])
                    
                if t_p[0][i] - r_p[0][i] > 0 and t_p[0][i] - r_p[0][i] < 100:
                    nextT.append(t_p[0][i] + halfRRindexes - r_p[0][i])
                elif t_p[0][i] - r_p[0][i-1] > 0 and t_p[0][i] - r_p[0][i-1] < 100:
                    nextT.append(t_p[0][i] + halfRRindexes - r_p[0][i-1])
                elif t_p[0][i] - r_p[0][i+1] > 0 and t_p[0][i] - r_p[0][i+1] < 100:
                    nextT.append(t_p[0][i] + halfRRindexes - r_p[0][i+1])
                
                
                if t_off[0][i] - r_p[0][i] > 0 and t_off[0][i] - r_p[0][i] < 100:
                    nextToff.append(t_off[0][i] + halfRRindexes - r_p[0][i])
                elif t_off[0][i] - r_p[0][i-1] > 0 and t_off[0][i] - r_p[0][i-1] < 100:
                    nextToff.append(t_off[0][i] + halfRRindexes - r_p[0][i-1])
                elif t_off[0][i] - r_p[0][i+1] > 0 and t_off[0][i] - r_p[0][i+1] < 100:
                    nextToff.append(t_off[0][i] + halfRRindexes - r_p[0][i+1])

        nextPon = np.array(nextPon)
        nextP = np.array(nextP)
        nextQ = np.array(nextQ)
        nextR = np.array(nextR)
        nextS = np.array(nextS)
        nextT = np.array(nextT)
        nextToff = np.array(nextToff)
        
        # Computation of the HR
        
        print(60/np.diff(realR[1]))
        hr = np.round(np.mean(60/np.diff(realR[1])), 3)
        stdHR = np.std(60/np.diff(realR[1]))
        lenHR = len(np.diff(realR[1]))
        
        minlen = min(len(nextT), len(nextQ))
        nextTtr = nextT[:minlen]
        nextQtr = nextQ[:minlen]
        qt = np.round(np.mean((nextTtr-nextQtr)*ecg._step), 3)
        stdQT = np.std((nextTtr-nextQtr)*ecg._step)
        lenQT = len((nextTtr-nextQtr)*ecg._step)
        
        minlen = min(len(nextP), len(nextR))
        nextPtr = nextP[:minlen]
        nextRtr = nextR[:minlen]
        pr = np.round(np.mean((nextRtr-nextPtr)*ecg._step),3)
        stdPR = np.std((nextRtr-nextPtr)*ecg._step)
        lenPR = len((nextRtr-nextPtr)*ecg._step)
        
        minlen = min(len(nextQ), len(nextS))
        nextQtr = nextQ[:minlen]
        nextStr = nextS[:minlen]
        qrs = np.round(np.mean((nextStr-nextQtr)*ecg._step),3)
        stdQRS = np.std((nextStr-nextQtr)*ecg._step)
        lenQRS = len((nextStr-nextQtr)*ecg._step)
        


        
        try: 
            meanECG, outIdx = cleanLOF(meanECG, 10, 0.25)
            for nextPeak in [nextR, nextP, nextQ, nextS, nextT, nextToff, nextPon]:
                nextPeak = [nextPeak[i] for i in range(len(nextPeak)) if i not in outIdx]
        # Make sure all the graphs are the same length (max one)
        except: meanECG = self.all_SameLen(meanECG) ; print("Failed rejection of ECG outliers graphs")

        
        

        x = np.arange(0, len(meanECG[0])*ecg._step, ecg._step)
        while len(x)>len(meanECG[0]):
            x = np.delete(x, -1)
        while len(x)<len(meanECG[0]):
            x = np.append(x, x[-1]+ecg._step)
            
        x = x - rrTime/2   
        
        rouge = QtGui.QBrush(QtGui.QColor(255, 0, 0, 85)) #R
        orange = QtGui.QBrush(QtGui.QColor(255, 128, 0, 85)) #S
        bleu = QtGui.QBrush(QtGui.QColor(0, 128, 255, 85)) #T
        vert = QtGui.QBrush(QtGui.QColor(0, 153, 0, 85)) #Toff
        violet = QtGui.QBrush(QtGui.QColor(51, 0, 102, 85)) #Pon
        turquoise = QtGui.QBrush(QtGui.QColor(0, 204, 204, 85)) #P
        rose = QtGui.QBrush(QtGui.QColor(255, 0, 255, 85)) #Q
        
        
        graphWidget = pg.PlotWidget()
        for elem in meanECG:
            courbe = graphWidget.plot(x, elem, pen='grey')
            courbe.setOpacity(0.3)
            
        for i, elem in enumerate(meanECG):
            countPon, countP, countQ, countR, countS, countT, countToff = [0,0,0,0,0,0,0]
            if nextPon[i] < len(x) :
                if countPon == 0 :
                    pointsPon = pg.ScatterPlotItem(x=[x[nextPon[i]]], y=[elem[nextPon[i]]], brush=bleu, pen=None, label = "P onsets")
                    countPon += 1
                else :
                    pointsPon = pg.ScatterPlotItem(x=[x[nextPon[i]]], y=[elem[nextPon[i]]], brush=bleu, pen=None)
            if nextP[i] < len(x) :
                if countP == 0 :
                    pointsP = pg.ScatterPlotItem(x=[x[nextP[i]]], y=[elem[nextP[i]]], brush=violet, pen=None, label = "P peaks")
                    countP += 1
                else :
                    pointsP = pg.ScatterPlotItem(x=[x[nextP[i]]], y=[elem[nextP[i]]], brush=violet, pen=None)
            if nextQ[i] < len(x) :
                if countQ == 0 :
                    pointsQ = pg.ScatterPlotItem(x=[x[nextQ[i]]], y=[elem[nextQ[i]]], brush=rose, pen=None, label = "Q peaks")
                    countQ += 1
                else :
                    pointsQ = pg.ScatterPlotItem(x=[x[nextQ[i]]], y=[elem[nextQ[i]]], brush=rose, pen=None)
            if nextR[i] < len(x) :
                if countR == 0 :
                    pointsR = pg.ScatterPlotItem(x=[x[nextR[i]]], y=[elem[nextR[i]]], brush=rouge, pen=None, label = "R peaks")
                    countR += 1
                else :
                    pointsR = pg.ScatterPlotItem(x=[x[nextR[i]]], y=[elem[nextR[i]]], brush=rouge, pen=None)           
            if nextS[i] < len(x) :
                if countS == 0 :
                    pointsS = pg.ScatterPlotItem(x=[x[nextS[i]]], y=[elem[nextS[i]]], brush=orange, pen=None, label = "S peaks")
                    countS += 1
                else :
                    pointsS = pg.ScatterPlotItem(x=[x[nextS[i]]], y=[elem[nextS[i]]], brush=orange, pen=None)            
            if nextT[i] < len(x) :
                if countT == 0 :
                    pointsT = pg.ScatterPlotItem(x=[x[nextT[i]]], y=[elem[nextT[i]]], brush=vert, pen=None, label = "T peaks")
                    countT += 1
                else :
                    pointsT = pg.ScatterPlotItem(x=[x[nextT[i]]], y=[elem[nextT[i]]], brush=vert, pen=None)            
            if nextToff[i] < len(x) :
                if countToff == 0 :
                    pointsToff = pg.ScatterPlotItem(x=[x[nextToff[i]]], y=[elem[nextToff[i]]], brush=turquoise, pen=None, label = "T offsets")
                    countToff += 1
                else :
                    pointsToff = pg.ScatterPlotItem(x=[x[nextToff[i]]], y=[elem[nextToff[i]]], brush=turquoise, pen=None)
            graphWidget.addLegend()
            for points in [pointsPon, pointsP, pointsQ, pointsR, pointsS, pointsT, pointsToff]:
                graphWidget.addItem(points) 
        
        graphWidget.setTitle("Mean "+ecg._title)
        graphWidget.setLabel('left', 'Magnitude')
        graphWidget.setLabel('bottom', 'Time [s]')
        graphWidget.setMinimumHeight(250)
        graphWidget.addLegend()
        graphWidget.showGrid(x=True, y=True)
        self.layout.addWidget(graphWidget)
        
        text = "<font color='blue'>●</font> P onsets ; "
        text += "<font color='purple'>●</font> P peaks ; "
        text += "<font color='magenta'>●</font> Q peaks ; "
        text += "<font color='red'>●</font> R peaks ; "
        text += "<font color='orange'>●</font> S peaks ; "
        text += "<font color='green'>●</font> T peaks ; "
        text += "<font color='turquoise'>●</font> P peaks"
        
        legend = QLabel(text)
        legend.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(legend)
        
        

        
        
        txt_hr = "HR = "+str(hr)+ " ± "+ str(stdHR) + " bpm (N = "+ str(lenHR) +")"
        txt_hr += "\nQRS = "+str(qrs)+ " ± "+ str(stdQRS) + " bpm (N = "+ str(lenQRS) +")"
        txt_hr += "\nQT = "+str(qt)+ " ± "+ str(stdQT) + " bpm (N = "+ str(lenQT) +")"
        txt_hr += "\nPR = "+str(pr)+ " ± "+ str(stdPR) + " bpm (N = "+ str(lenPR) +")"
        
        #text += "<font color='purple'>●</font> P peaks ; "
        #text += "<font color='magenta'>●</font> Q peaks ; "
        #text += "<font color='red'>●</font> R peaks ; "
        #text += "<font color='orange'>●</font> S peaks ; "
        #text += "<font color='green'>●</font> T peaks ; "
        #text += "<font color='turquoise'>●</font> P peaks"
        
        legend = QLabel(txt_hr)
        legend.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(legend)
        
        
        
        return
    
    def printHRV(self, ecg, pqrst):
        
        [p_p, q_p, r_p, s_p, t_p, p_on, t_off] = pqrst
        
        # Get the HRV
        nk.hrv(r_p[0], ecg._samplerate, show = True)
        hrvPlot = plt.gcf()
        plt.show()
        hrvPlot.set_figwidth(20)
        hrvPlot.set_figheight(5)
        hrvPlot.tight_layout(pad=1.0)
        
        buffer = io.BytesIO()
        hrvPlot.savefig(buffer, format='png')
        buffer.seek(0)

        # Charger l'image depuis la variable en utilisant PIL
        image_pil = Image.open(buffer)
        
        pixmap = QPixmap.fromImage(ImageQt(image_pil))

        # Créer un widget QLabel pour afficher l'image
        label = QLabel()
        label.setPixmap(pixmap)
        label.setSizePolicy(QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding))
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)


        # Ajouter le QLabel au layout
        self.layout.addWidget(label)
    
    def all_SameLen(self, vect):
        # Make sure all the graphs are the same length (max one)
        
        #print("BEFORE :", len(vect), len(vect[0]))
        
        all_len = []
        vect_copy = vect.copy()
        for item in vect:
            all_len.append(len(item))
        max_len = int(np.max(all_len))

        for i in range(len(vect_copy)):
            while len(vect_copy[i])<max_len:
                vect_copy[i] = np.append(vect_copy[i], vect_copy[i][-1])

        return vect_copy
        #print("AFTER :", len(vect_copy), len(vect_copy[0]))
        

