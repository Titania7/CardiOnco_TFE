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



class PWVhypothesis(QDialog):
    
    distLeg = "?"
    distArm = "?"
    timeR_Leg = None
    timeR_Arm = None
    timeR_Ao4090 = None
    timeR_Ao2dP = None
    
    
    def __init__(self, timeR_Leg, timeR_Arm, timeR_Ao4090, timeR_Ao2dP):
        super().__init__()



        self.timeR_Leg = timeR_Leg
        self.timeR_Arm = timeR_Arm
        self.timeR_Ao4090 = timeR_Ao4090
        self.timeR_Ao2dP = timeR_Ao2dP


        self.setWindowTitle("PWV hypothesis computation")
        
        # Begin of stacked widgets
        self.layout = QVBoxLayout()
        
        
        # Beginning of side-by-side widgets
        
        #%% Number 1 : Leg distance
        
        self.legdistLabel = QHBoxLayout()
        legdisttxt = QLabel("Leg distance :")
        self.legdistLabel.addWidget(legdisttxt)
        self.legdistEdit = QLineEdit()
        self.legdistEdit.setText("?")
        self.legdistEdit.textChanged.connect(self.legdistEdit.setText)
        self.legdistEdit.textChanged.connect(self.updateDistleg)
        self.legdistLabel.addWidget(self.legdistEdit)
        self.layout.addLayout(self.legdistLabel)
        
        #%% Number 2 : Arm distance
        
        self.armdistLabel = QHBoxLayout()
        armdisttxt = QLabel("Arm distance :")
        self.armdistLabel.addWidget(armdisttxt)
        self.armdistEdit = QLineEdit()
        self.armdistEdit.setText("?")
        self.armdistEdit.textChanged.connect(self.armdistEdit.setText)
        self.armdistEdit.textChanged.connect(self.updateDistarm)
        self.armdistLabel.addWidget(self.armdistEdit)
        self.layout.addLayout(self.armdistLabel)
        
        
        #%% Number 3 : Leg and arm delays
        
        self.delaysLabel = QHBoxLayout()
        delaystxt = QLabel("Leg delay = " + str(np.round(timeR_Leg,3)) + " s")
        self.delaysLabel.addWidget(delaystxt)
        delaystxt = QLabel("Arm delay = "+ str(np.round(timeR_Arm,3))+" s")
        self.delaysLabel.addWidget(delaystxt)
        self.layout.addLayout(self.delaysLabel)

        
        #%% Number 4 : Print approximation for both calculated AO
        
        computePWV = QPushButton('Compute PWV', self)
        computePWV.clicked.connect(self.updatepwvHyp)
        computePWV.setGeometry(50, 50, 200, 30)
        self.layout.addWidget(computePWV)
        
        
        self.pwvHyp = QLabel("Hypothesis PWV = ? m/s")
        self.layout.addWidget(self.pwvHyp)

            
        self.setLayout(self.layout)

    def updateDistarm(self):
        if not self.armdistEdit == "?" or not self.armdistEdit == "" :
            self.distArm = float(self.armdistEdit.text())
        return
    
    def updateDistleg(self):
        if not self.legdistEdit == "?" or not self.legdistEdit == "" :
            self.distLeg = float(self.legdistEdit.text())
    
    def updatepwvHyp(self):
        print(self.distLeg, self.distArm, self.timeR_Leg, self.timeR_Arm)
        if not self.distLeg == "?" and not self.distArm == "?":
            pwvHypothese = self.distArm / (self.timeR_Arm - (self.distArm*self.timeR_Leg - self.distLeg*self.timeR_Arm)/(self.distArm - self.distLeg))
            self.pwvHyp.setText("Hypothesis PWV = " + str(np.round(pwvHypothese, 3)) + " m/s")
        
        return
    
#%% Secondary window (mean graphs)

class DisplMeanWindow(QMainWindow):
    
    ecgML = None
    bpArm = None
    bpLeg = None
    ecgJSON = None
    linSCG = None
    rotSCG = None
    
    bpAo = None
    
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
        
        pwvHyp_button = QAction("Hypothesis PWV", self)
        toolbar.addAction(pwvHyp_button)
        pwvHyp_button.triggered.connect(self.popupPWVhypothesis)
                                
        
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
        
        
        
        self.setMinimumSize(700, 900)
        # Full screen mode
        #self.showMaximized()
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
        #print(len(r_pML[0]), len(r_pJSON[0]))
       
        #%% Retrieval + identification of all the tracks available
        startindex = timelim[0]
        stopindex = timelim[1]

        legCut = False
        for i in range(len(infoVect)):
            if infoVis[i] == True:
                if infoVect[i][0]._title == "ECG mat":
                    ecgML = infoVect[i][0]
                elif infoVect[i][0]._title == "ECG json":
                    ecgJSON = infoVect[i][0]
                elif infoVect[i][0]._title == "BPAo":
                    self.bpAo = infoVect[i][0]
                     
                elif infoVect[i][0]._title == "BPleg":
                    bpLeg = infoVect[i][0]
                    bpLeg._y = butterCutPass(bpLeg, 4, [1,bpLeg._samplerate/2-10])
                    """
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
                    """
                elif infoVect[i][0]._title == "BParm":
                    orig_bpArm = infoVect[i][0]
                    # We take into account the time limitations asked by the user for this one
                    #print(timelim)
                    startindex = int(timelim[0]*orig_bpArm._samplerate)
                    stopindex = int(timelim[1]*orig_bpArm._samplerate)
                    print(timelim, startindex, stopindex, len(orig_bpArm._x))
                    
                    # We create another instance of bpArm in order to update the timelims at each instance of this window
                    self.bpArm = SingleGraph(orig_bpArm._x[startindex:stopindex], orig_bpArm._y[startindex:stopindex], orig_bpArm._title, orig_bpArm._samplerate, orig_bpArm._step)
                    self.bpArm._y = butterCutPass(self.bpArm, 4, [1,self.bpArm._samplerate/2-10])

                    
                    """ 
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
        if not self.bpAo == None :
            mean_bpAo = []
        
        authorizedList = []
        
        # Go through the ecg_indexes of the r peaks (matlab graphs)
        for i in range(len(r_pML[0])-1):
            #print(startindex, r_pML[1][i]*self.bpArm._samplerate, r_pML[1][i+1]*self.bpArm._samplerate, len(self.bpArm._x))
            
            if r_pML[0][i+1] < len(ecgML._x):
                y_ecgML = ecgML._y[r_pML[0][i]:r_pML[0][i+1]]
                mean_ecgML.append(y_ecgML)

                authorizedList.append(i)
                
            if r_pML[0][i+1] < len(bpLeg._x):
                y_bpLeg = bpLeg._y[r_pML[0][i]:r_pML[0][i+1]]
                mean_bpLeg.append(y_bpLeg)
            
            if not self.bpAo == None :
                y_bpAo = self.bpAo._y[r_pML[0][i]:r_pML[0][i+1]]
                mean_bpAo.append(y_bpAo)
            
            if r_pML[0][i]-startindex> 0 and r_pML[0][i+1]-startindex < len(self.bpArm._x):
                #print("I'm in !")
                y_bpArm = self.bpArm._y[r_pML[0][i]-startindex:r_pML[0][i+1]-startindex]
                mean_bpArm.append(y_bpArm)
            
        
        minsize = np.min([len(vect) for vect in mean_bpArm])
        for i, vect in enumerate(mean_bpArm):
            while len(mean_bpArm[i])>minsize:
                mean_bpArm[i] = np.delete(mean_bpArm[i],-1)
                
        if not self.bpAo == None :
            for i, vect in enumerate(mean_bpAo):
                while len(mean_bpAo[i])>minsize:
                    mean_bpAo[i] = np.delete(mean_bpAo[i],-1)
                
        minsize  = np.min([len(vect) for vect in mean_bpLeg])
        for i, vect in enumerate(mean_bpLeg):
            while len(mean_bpLeg[i])>minsize:
                mean_bpLeg[i] = np.delete(mean_bpLeg[i],-1)
        
        

                
        
        """
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
        """
        
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
        

        
        #%% Rejection of all the aberrant graphs for the superposed graphs
        
        
        def returnCleaned(liste2D, indexes):
            new0 = [liste2D[0][i] for i in range(len(liste2D[0])) if i not in indexes]
            new1 = [liste2D[1][i] for i in range(len(liste2D[1])) if i not in indexes]
            new = [new0, new1]
            
            return new


        self.addECG_centerR(ecgML)
        self.printHRV(ecgML, pqrstMAT)
        
        try: mean_bpLeg, idx_bpLeg = cleanLOF(mean_bpLeg, 5, 0.4)
        except: print("Rejection of outliers graphs for bpLeg unsuccessful")
        
        # Does not work with the bpArm if it is too short
        try: mean_bpArm, idx_bpArm = cleanLOF(mean_bpArm, 1, 0.1)
        except: print("Rejection of outliers graphs for bpArm unsuccessful")
        
        if not self.bpAo == None :
            try: mean_bpAo, idx_bpAo = cleanLOF(mean_bpAo, 5, 0.4)
            except: print("Rejection of outliers graphs for bpAo unsuccessful")
        
        
        self.addECG_centerR(ecgJSON)
        self.printHRV(ecgJSON, pqrstJSON)
      
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
        
        onsBPLeg = getBpOnsets_tang(mean_bpLeg, bpLeg._samplerate, bpLeg._title, filt = True, show = False)
        onsBPArm = getBpOnsets_tang(mean_bpArm, self.bpArm._samplerate, self.bpArm._title, filt = True, show = False)
        if not self.bpAo == None :
            onsBPAo = getBpOnsets_tang(mean_bpAo, self.bpAo._samplerate, self.bpAo._title, filt = True, show = False)
        
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
     
               
        self.addBP(self.bpLeg, mean_bpLeg, onsBPLeg[0])
        self.addBP(self.bpArm, mean_bpArm, onsBPArm[0])
        if not self.bpAo == None :
            self.addBP(self.bpAo, mean_bpAo, onsBPAo[0])
        
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
                self.aolinCursor = pg.InfiniteLine(pos=posStart, bounds=[posStart, posStop], label='Manual AO', angle=90, movable=True, pen=pg.mkPen(width=3))
                graphWidget.addItem(self.aolinCursor)
                self.aolinCursor.sigPositionChanged.connect(self.manualAOlin)
                
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
                posStart = x[0]
                posStop = x[-1]
                self.aorotCursor = pg.InfiniteLine(pos=posStart, bounds=[posStart, posStop], label='Manual AO', angle=90, movable=True, pen=pg.mkPen(width=3))
                graphWidget.addItem(self.aorotCursor)
                self.aorotCursor.sigPositionChanged.connect(self.manualAOrot)
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
                
                
            if corresponding[i]._title == "lin_SCGvect[m/s^2]":
                legend = QLabel("<font color='magenta'>●</font> AO detected 40-90 ms ; <font color='turquoise'>●</font> AO detected as 2d peak")
                legend.setAlignment(Qt.AlignmentFlag.AlignCenter)
                self.layout.addWidget(legend)
                
                self.ao_4090_label = QLabel("")
                self.layout.addWidget(self.ao_4090_label)
                
                self.aO_2dP_label = QLabel("")
                self.layout.addWidget(self.aO_2dP_label)
                
                self.manualAOlin_label = QLabel(f" Manual AO (linSCG) : {self.aolinCursor.pos().x()} s")
                self.layout.addWidget(self.manualAOlin_label)
                                
                self.xLinCursor = self.displaySCGGraph(all_xlin, xlinSCG)
                self.xLin_label = QLabel("")
                self.layout.addWidget(self.xLin_label)
                self.xLinCursor.sigPositionChanged.connect(self.manualAOxlin)
                
                self.yLinCursor = self.displaySCGGraph(all_ylin, ylinSCG)
                self.yLin_label = QLabel("")
                self.layout.addWidget(self.yLin_label)
                self.yLinCursor.sigPositionChanged.connect(self.manualAOylin)
                
                self.zLinCursor = self.displaySCGGraph(all_zlin, zlinSCG)
                self.zLin_label = QLabel("")
                self.layout.addWidget(self.zLin_label)
                self.zLinCursor.sigPositionChanged.connect(self.manualAOzlin)
                    
             
            elif corresponding[i]._title == "rot_SCGvect[deg/s]":
                 self.manualAOrot_label = QLabel(f"\tManual AO (rotSCG) : {self.aorotCursor.pos().x()} s")
                 self.layout.addWidget(self.manualAOrot_label)
                 
                 self.xRotCursor = self.displaySCGGraph(all_xrot, xrotSCG)
                 self.xRot_label = QLabel("")
                 self.layout.addWidget(self.xRot_label)
                 self.xRotCursor.sigPositionChanged.connect(self.manualAOxrot)
                 
                 self.yRotCursor = self.displaySCGGraph(all_yrot, yrotSCG)
                 self.yRot_label = QLabel("")
                 self.layout.addWidget(self.yRot_label)
                 self.yRotCursor.sigPositionChanged.connect(self.manualAOyrot)
                 
                 self.zRotCursor = self.displaySCGGraph(all_zrot, zrotSCG)
                 self.zRot_label = QLabel("")
                 self.layout.addWidget(self.zRot_label)
                 self.zRotCursor.sigPositionChanged.connect(self.manualAOzrot)

            
        
    #%% Display of the MetaData and the approximative PWV
        
        self.meanR_Leg = np.mean(onsBPLeg[1])
        self.stdR_Leg = np.std(onsBPLeg[1])
        self.lenR_Leg = len(onsBPLeg[1])
        self.bpLeg_label.setText(" Mean R-bpLeg onset delay : "+str(np.round(self.meanR_Leg,3))+" ± "+str(np.round(self.stdR_Leg,3))+" s (N = "+str(self.lenR_Leg)+")")
        # Computation of the mean R-Leg and R-Arm
        
        self.meanR_Arm = np.mean(onsBPArm[1])
        self.stdR_Arm = np.std(onsBPArm[1])
        self.lenR_Arm = len(onsBPArm[1])
        self.bpArm_label.setText(" Mean R-bpArm onset delay : "+str(np.round(self.meanR_Arm,3))+" ± "+str(np.round(self.stdR_Arm,3))+" s (N = "+str(self.lenR_Arm)+")")
        
        if not self.bpAo == None :
            self.meanR_Ao = np.mean(onsBPAo[1])
            self.stdR_Ao = np.std(onsBPAo[1])
            self.lenR_Ao = len(onsBPAo[1])
            self.bpAo_label.setText(" Mean R-bpAo onset delay : "+str(np.round(self.meanR_Ao,3))+" ± "+str(np.round(self.stdR_Ao,3))+" s (N = "+str(self.lenR_Ao)+")")
            
        
        # Computation of the mean R-AO for each technique
        self.meanR_AO_4090 = np.mean(relAO_4090)*linSCG._step
        self.stdR_AO_4090 = np.std(relAO_4090*linSCG._step)
        self.lenAO_4090 = len(relAO_4090)
        text = " Delay R-AO (40-90 ms technique): "+str(np.round(self.meanR_AO_4090,3))+" ± "+str(np.round(self.stdR_AO_4090,3))+" s (N = "+str(self.lenAO_4090)+")"
        self.pwv_4090 = (0.8*meta[-1])/(self.meanR_Leg - self.meanR_AO_4090)
        text +=  "     => afPWV = "+ str(np.round(self.pwv_4090 ,3))+" m/s"
        self.ao_4090_label.setText(text)
        
        self.meanR_AO_2dP = np.round(np.mean(relAO_2dPeak))*linSCG._step
        self.stdR_AO_2dP = np.std(relAO_2dPeak*linSCG._step) 
        self.lenAO_2dP = len(relAO_2dPeak)
        text = " Delay R-AO (2d peak technique): "+str(np.round(self.meanR_AO_2dP,3))+" ± "+str(np.round(self.stdR_AO_2dP,3))+" s (N = "+str(self.lenAO_2dP)+")"
        self.pwv_2dP = (0.8*meta[-1])/(self.meanR_Leg - self.meanR_AO_2dP)
        text +=  "     => afPWV = "+ str(np.round(self.pwv_2dP ,3))+" m/s"
        self.aO_2dP_label.setText(text)
        
        #%%
        


        self.setMinimumSize(900, 800)
        #self.showMaximized()
    
    #%% Useful functions
    
    def displaySCGGraph(self, graph, corresp):
        graphWidget = pg.PlotWidget()
        for trace in graph:
            x = np.arange(0, len(trace)*corresp._step, corresp._step)
            while len(x)>len(trace):
                x=np.delete(x,-1)
            courbe = graphWidget.plot(x, trace, pen="grey")
            courbe.setOpacity(0.3)
        courbe = graphWidget.plot(x, np.mean(np.array(graph), axis = 0), pen="magenta")

        graphWidget.setTitle("Mean "+corresp._title)
        graphWidget.setLabel('left', 'Magnitude')
        graphWidget.setLabel('bottom', 'Time [s]')
        graphWidget.setMinimumHeight(250)
        graphWidget.showGrid(x=True, y=True)
        
        posStart = x[0]
        posStop = x[-1]
        cursor = pg.InfiniteLine(pos=posStart, bounds=[posStart, posStop], label='Manual AO', angle=90, movable=True, pen=pg.mkPen(width=3))
        graphWidget.addItem(cursor)
        
        self.layout.addWidget(graphWidget)
        
        return cursor
        
        
    def popupPWVhypothesis(self):

        timeR_Leg = self.meanR_Leg
        timeR_Arm = self.meanR_Arm
        timeR_Ao4090 = self.meanR_AO_4090
        timeR_Ao2dP = self.meanR_AO_2dP
        
        PWVhypothesis(self.meanR_Leg, self.meanR_Arm, self.meanR_AO_4090, self.meanR_AO_2dP).exec()
        
        return
    
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
    
    def manualAOxlin(self):
        newPosition = self.xLinCursor.pos().x()
        newPWV = (0.8*self.meta[-1])/(self.meanR_Leg - newPosition)
        self.xLin_label.setText(f" Manual AO (xlinSCG) : "+str(np.round(newPosition,3))+" s     => afPWV = "+str(np.round(newPWV, 3))+" m/s")
        
    def manualAOylin(self):
        newPosition = self.yLinCursor.pos().x()
        newPWV = (0.8*self.meta[-1])/(self.meanR_Leg - newPosition)
        self.yLin_label.setText(f" Manual AO (ylinSCG) : "+str(np.round(newPosition,3))+" s     => afPWV = "+str(np.round(newPWV, 3))+" m/s")

    def manualAOzlin(self):
        newPosition = self.zLinCursor.pos().x()
        newPWV = (0.8*self.meta[-1])/(self.meanR_Leg - newPosition)
        self.zLin_label.setText(f" Manual AO (zlinSCG) : "+str(np.round(newPosition,3))+" s     => afPWV = "+str(np.round(newPWV, 3))+" m/s")
    
    
    def manualAOlin(self):
        newPosition = self.aolinCursor.pos().x()
        newPWV = (0.8*self.meta[-1])/(self.meanR_Leg - newPosition)
        self.manualAOlin_label.setText(f" Manual AO (linSCG) : "+str(np.round(newPosition,3))+" s     => afPWV = "+str(np.round(newPWV, 3))+" m/s")
    
    
    def manualAOxrot(self):
        newPosition = self.xRotCursor.pos().x()
        newPWV = (0.8*self.meta[-1])/(self.meanR_Leg - newPosition)
        self.xRot_label.setText(f" Manual AO (xrotSCG) : "+str(np.round(newPosition,3))+" s     => afPWV = "+str(np.round(newPWV, 3))+" m/s")
        
    def manualAOyrot(self):
        newPosition = self.yRotCursor.pos().x()
        newPWV = (0.8*self.meta[-1])/(self.meanR_Leg - newPosition)
        self.yRot_label.setText(f" Manual AO (yrotSCG) : "+str(np.round(newPosition,3))+" s     => afPWV = "+str(np.round(newPWV, 3))+" m/s")

    def manualAOzrot(self):
        newPosition = self.zRotCursor.pos().x()
        newPWV = (0.8*self.meta[-1])/(self.meanR_Leg - newPosition)
        self.zRot_label.setText(f" Manual AO (zrotSCG) : "+str(np.round(newPosition,3))+" s     => afPWV = "+str(np.round(newPWV, 3))+" m/s")
    
    
    
    def manualAOrot(self):
        newPosition = self.aorotCursor.pos().x()
        newPWV = (0.8*self.meta[-1])/(self.meanR_Leg - newPosition)
        self.manualAOrot_label.setText(f" Manual AO (rotSCG) : "+str(np.round(newPosition,3))+" s     => afPWV = "+str(np.round(newPWV, 3))+" m/s")
    
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
        
        if bpSC._title == "BPleg":
            self.bpLeg_label = QLabel("")
            self.layout.addWidget(self.bpLeg_label)
        elif bpSC._title == "BParm":
            self.bpArm_label = QLabel("")
            self.layout.addWidget(self.bpArm_label)
        elif bpSC._title == "BPAo":
            self.bpAo_label = QLabel("")
            self.layout.addWidget(self.bpAo_label)
        
        
        
    
    def addECG_centerR(self, ecg):
        
        [p_p, q_p, r_p, s_p, t_p, p_on, t_off] = detect_qrs(ecg, clean_technique = "ECG_NeuroKit", ecg_delineate = "peaks", show=False)
        realR = r_p
            
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


        
        def getNextVect(points, rpoints, inflim, suplim):
            nextVect = []
            indrec = []
            for i, pt in enumerate(points):
                for rp in rpoints:
                    difference = pt-rp
                    if difference < suplim and difference > inflim:
                        nextVect.append(difference + halfRRindexes)
                        indrec.append(i)
            return nextVect, indrec
        
        nextPon, indPon = getNextVect(p_on[0], r_p[0], inflim = -100, suplim = 0)
        nextP, indP = getNextVect(p_p[0], r_p[0], inflim = -100, suplim = 0)
        nextQ, indQ = getNextVect(q_p[0], r_p[0], inflim = -100, suplim = 0)
        nextS, indS  = getNextVect(s_p[0], r_p[0], inflim = 0, suplim = 100)
        nextT, indT  = getNextVect(t_p[0], r_p[0], inflim = 0, suplim = 100)
        nextToff, indToff = getNextVect(t_off[0], r_p[0], inflim = 0, suplim = 100)
            
        nextPon = np.array(nextPon)
        nextPon = np.delete(nextPon, -1)
        nextP = np.array(nextP)
        nextP = np.delete(nextP, -1)
        nextQ = np.array(nextQ)
        nextQ = np.delete(nextQ, -1)
        nextR = np.array(nextR)
        nextR = np.delete(nextR, -1)
        nextS= np.array(nextS)
        nextS= np.delete(nextS, -1)
        nextT= np.array(nextT)
        nextT= np.delete(nextT, -1)
        nextToff = np.array(nextToff)
        nextToff = np.delete(nextToff, -1)
        
        print("Before LOF ", len(meanECG), len(nextPon), len(nextP), len(nextQ), len(nextR), len(nextS), len(nextT), len(nextToff))
        # Computation of the HR
        
        #print(60/np.diff(realR[1]))
        hr = np.round(np.mean(60/np.diff(realR[1])), 3)
        stdHR = np.round(np.std(60/np.diff(realR[1])),3)
        lenHR = len(np.diff(realR[1]))
        
        minlen = min(len(nextToff), len(nextQ))
        nextTofftr = nextToff[:minlen]
        nextQtr = nextQ[:minlen]
        qt = np.round(np.mean((nextTofftr-nextQtr)*ecg._step), 3)
        stdQT = np.round(np.std((nextTofftr-nextQtr)*ecg._step),3)
        lenQT = len((nextTofftr-nextQtr)*ecg._step)
        
        minlen = min(len(nextP), len(nextR))
        nextPtr = nextP[:minlen]
        nextRtr = nextR[:minlen]
        pr = np.round(np.mean((nextRtr-nextPtr)*ecg._step),3)
        stdPR = np.round(np.std((nextRtr-nextPtr)*ecg._step),3)
        lenPR = len((nextRtr-nextPtr)*ecg._step)
        
        minlen = min(len(nextQ), len(nextS))
        nextQtr = nextQ[:minlen]
        nextStr = nextS[:minlen]
        qrs = np.round(np.mean((nextStr-nextQtr)*ecg._step),3)
        stdQRS = np.round(np.std((nextStr-nextQtr)*ecg._step),3)
        lenQRS = len((nextStr-nextQtr)*ecg._step)
        

        try: 
            meanECG, outIdx = cleanLOF(meanECG, 10, 0.25)
            nextPon = [nextPon[i] for i in range(len(nextPon)) if i not in outIdx]
            nextP = [nextP[i] for i in range(len(nextP)) if i not in outIdx]
            nextQ = [nextQ[i] for i in range(len(nextQ)) if i not in outIdx]
            nextR = [nextR[i] for i in range(len(nextR)) if i not in outIdx]
            nextS = [nextS[i] for i in range(len(nextS)) if i not in outIdx]
            nextT = [nextT[i] for i in range(len(nextT)) if i not in outIdx]
            nextToff = [nextToff[i] for i in range(len(nextToff)) if i not in outIdx]
        # Make sure all the graphs are the same length (max one)
        except: meanECG = self.all_SameLen(meanECG) ; print("Failed rejection of ECG outliers graphs")

        print("After LOF ", len(meanECG), len(nextPon), len(nextP), len(nextQ), len(nextR), len(nextS), len(nextT), len(nextToff))
        
        allpoints = [nextPon, nextP, nextQ, nextR, nextS, nextT, nextToff]
        for i in range(len(allpoints)):
            while len(allpoints[i])>len(meanECG):
                allpoints[i] = np.delete(allpoints[i], -1)
        [nextPon, nextP, nextQ, nextR, nextS, nextT, nextToff] = allpoints
        
        print("After LOF+resizing ", len(meanECG), len(nextPon), len(nextP), len(nextQ), len(nextR), len(nextS), len(nextT), len(nextToff))
        
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
        
        for i, p in enumerate(nextPon):
            currentgraph = meanECG[i]
            if i == 0:
                pointsPon = pg.ScatterPlotItem(x=[x[p]], y=[currentgraph[p]], brush=bleu, pen=None, label = "P onsets")
            else :
                pointsPon = pg.ScatterPlotItem(x=[x[p]], y=[currentgraph[p]], brush=bleu, pen=None)
            graphWidget.addItem(pointsPon) 
        
        for i, p in enumerate(nextP):
            currentgraph = meanECG[i]
            if i == 0:
                pointsP = pg.ScatterPlotItem(x=[x[p]], y=[currentgraph[p]], brush=violet, pen=None, label = "P peaks")
            else :
                pointsP = pg.ScatterPlotItem(x=[x[p]], y=[currentgraph[p]], brush=violet, pen=None)
            graphWidget.addItem(pointsP) 
         
        for i, p in enumerate(nextQ):
            currentgraph = meanECG[i]
            if i == 0:
                pointsQ = pg.ScatterPlotItem(x=[x[p]], y=[currentgraph[p]], brush=rose, pen=None, label = "Q peaks")
            else :
                pointsQ = pg.ScatterPlotItem(x=[x[p]], y=[currentgraph[p]], brush=rose, pen=None)
            graphWidget.addItem(pointsQ) 
        
        for i, p in enumerate(nextR):
            currentgraph = meanECG[i]
            if i == 0:
                pointsR = pg.ScatterPlotItem(x=[x[p]], y=[currentgraph[p]], brush=rouge, pen=None, label = "R peaks")
            else :
                pointsR = pg.ScatterPlotItem(x=[x[p]], y=[currentgraph[p]], brush=rouge, pen=None)
            graphWidget.addItem(pointsR)
            
        for i, p in enumerate(nextS):
            currentgraph = meanECG[i]
            if i == 0:
                pointsS = pg.ScatterPlotItem(x=[x[p]], y=[currentgraph[p]], brush=orange, pen=None, label = "S peaks")
            else :
                pointsS = pg.ScatterPlotItem(x=[x[p]], y=[currentgraph[p]], brush=orange, pen=None)
            graphWidget.addItem(pointsS)
            
        for i, p in enumerate(nextT):
            currentgraph = meanECG[i]
            if i == 0:
                pointsT = pg.ScatterPlotItem(x=[x[p]], y=[currentgraph[p]], brush=vert, pen=None, label = "T peaks")
            else :
                pointsT = pg.ScatterPlotItem(x=[x[p]], y=[currentgraph[p]], brush=vert, pen=None)
            graphWidget.addItem(pointsT)
        
        
        for i, p in enumerate(nextToff):
            currentgraph = meanECG[i]
            if i == 0:
                pointsToff = pg.ScatterPlotItem(x=[x[p]], y=[currentgraph[p]], brush=turquoise, pen=None, label = "T offsets")
            else :
                pointsToff = pg.ScatterPlotItem(x=[x[p]], y=[currentgraph[p]], brush=turquoise, pen=None)
            graphWidget.addItem(pointsToff)
        
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
        txt_hr += " ; QRS = "+str(qrs)+ " ± "+ str(stdQRS) + " bpm (N = "+ str(lenQRS) +")"
        txt_hr += "\nQT = "+str(qt)+ " ± "+ str(stdQT) + " bpm (N = "+ str(lenQT) +")"
        txt_hr += " ; PR = "+str(pr)+ " ± "+ str(stdPR) + " bpm (N = "+ str(lenPR) +")"
        
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
        
        minLen = np.min([len(v) for v in vect])

        for i, item in enumerate(vect):
            while len(vect[i]) > minLen :
                vect[i] = np.delete(vect[i], -1)

        return vect
        #print("AFTER :", len(vect_copy), len(vect_copy[0]))
        
