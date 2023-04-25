# -*- coding: utf-8 -*-
"""
Created on Sun Mar  5 23:10:36 2023

@author: Tatiana Dehon
"""
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import *
# QDialog, QDialogButtonBox, QVBoxLayout, QHBoxLayout, QLabel, QCheckBox, QLineEdit, QComboBox
from simpleDisplay import *
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui
import numpy as np


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
    
    
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Default file reading parameters")
        
        # Begin of stacked widgets
        self.layout = QVBoxLayout()
        
        
        # Beginning of side-by-side widgets
        
        #%% Number 1 : sampling frequency
        """
        self.subFreq = QHBoxLayout()
        freqtxt = QLabel("Sampling frequency [Hz]")
        self.subFreq.addWidget(freqtxt)
        self.freq = QLineEdit()
        self.freq.setText("1000")
        self.freq.textChanged.connect(self.freq.setText)
        self.subFreq.addWidget(self.freq)
        self.layout.addLayout(self.subFreq)
        """
        
        #%% Number 2 : ECG cleaning technique
        self.ecgClean = QHBoxLayout()
        ecgcleantxt = QLabel("ECG cleaning technique")
        self.ecgClean.addWidget(ecgcleantxt)
        self.combo_box = QComboBox()
        self.combo_box.addItems(["None", "NeuroKit", "BioSPPy", "PanTompkins", "Hamilton", "Elgendi", "EngZeeMod"])
        self.combo_box.setCurrentIndex(1)
        self.combo_box.setGeometry(10, 10, 150, 30)
        self.ecgClean.addWidget(self.combo_box)
        self.layout.addLayout(self.ecgClean)
        
        #%% Number 3 : ECG delineate technique
        self.ecgDelin = QHBoxLayout()
        ecgdelintxt = QLabel("ECG delineating technique")
        self.ecgDelin.addWidget(ecgdelintxt)
        self.combo_box = QComboBox()
        self.combo_box.addItems(["Peak", "Continuous Wavelet Transform", "Discrete Wavelet Transform"])
        self.combo_box.setCurrentIndex(0)
        self.combo_box.setGeometry(10, 10, 150, 30)
        self.ecgDelin.addWidget(self.combo_box)
        self.layout.addLayout(self.ecgDelin)
        
        #%% Number 4 : pressure Onsets detection technique
        """
        self.bpOnsets = QHBoxLayout()
        bponsetstxt = QLabel("BP onsets detection technique")
        self.bpOnsets.addWidget(bponsetstxt)
        self.combo_box = QComboBox()
        self.combo_box.addItems(["Intersecting tangents", "Maximum second derivative"])
        self.combo_box.setCurrentIndex(0)
        self.combo_box.setGeometry(10, 10, 150, 30)
        self.bpOnsets.addWidget(self.combo_box)
        self.layout.addLayout(self.bpOnsets)
        """
        
        QBtn = QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel

        self.buttonBox = QDialogButtonBox(QBtn)
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)
        
        self.layout.addWidget(self.buttonBox)
        
        
        self.setLayout(self.layout)


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
    
    


    
class DisplMeanPopup(QWidget):
    def __init__(self, height, infoVect, infoVis, clean_technique, ecg_delineate, timelim = None):
        super().__init__()
        
        self.setWindowTitle("Mean graphs display")
        
        # Create a central widget and set the layout
        central_widget = QWidget()
        
        # Create a scroll bar
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        

        # Create a layout to stack the graphs vertically
        self.layout = QVBoxLayout()
        central_widget.setLayout(self.layout)
        scroll_area.setWidget(central_widget)
        
        # Set the central widget of the main window
        self.setLayout(self.layout)


        bpAo_ons = []
        bpLeg_ons = []
        bpArm_ons = []
        
        pPeaks = None
        qPeaks = None
        rPeaks = None
        sPeaks = None
        tPeaks = None
        pOnsets = None
        tOffsets = None
        
        for i in range(len(infoVect)):
            if infoVis[i] == True:
                if infoVect[i][0]._title == "ECG":
                    oneData = infoVect[i][0]
                    pPeaks, qPeaks, rPeaks, sPeaks, tPeaks, pOnsets, tOffsets = detect_qrs(oneData, clean_technique, ecg_delineate, show = False)
                    
                    
        for i in range(len(infoVect)):

            if infoVis[i] == True:
                
                if infoVect[i][0]._title == "ECG":
                    oneData = infoVect[i][0]                
                    
                    # Create the graph and plot the data
                    oneGraph = pg.PlotWidget()
                    oneGraph.setMouseEnabled(x=False, y=False)
                    
                    diff = []
                    if timelim:
                        temp = []
                        temptime = []
                        for peak in rPeaks[1]:
                            if peak > timelim[0] and peak < timelim[1]:
                                temp.append(int(peak*oneData._samplerate))
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
                    x = x/oneData._samplerate

                    # Add the graph to the layout
                    self.layout.addWidget(oneGraph)
                    oneGraph.addLegend()

                

                    for i in range(len(x_min)):
                        thisECG = oneData._y[x_min[i]:x_max[i]]
                        courbe = oneGraph.plot(x, thisECG, pen='grey')
                        courbe.setOpacity(0.3)
                        
                        rouge = QtGui.QBrush(QtGui.QColor(255, 0, 0, 85))
                        orange = QtGui.QBrush(QtGui.QColor(255, 128, 0, 85))
                        bleu = QtGui.QBrush(QtGui.QColor(0, 128, 255, 85))
                        vert = QtGui.QBrush(QtGui.QColor(0, 153, 0, 85))
                        violet = QtGui.QBrush(QtGui.QColor(51, 0, 102, 85))
                        
                        turquoise = QtGui.QBrush(QtGui.QColor(0, 204, 204, 85))
                        rose = QtGui.QBrush(QtGui.QColor(255, 0, 255, 85))
                        
                        for a in pOnsets[0]:
                            if a < x_max[i] and a > x_min[i]:
                                relativeAbs = a - x_min[i]
                                if i == 0 :
                                    points = pg.ScatterPlotItem(x=[x[relativeAbs]], y=[thisECG[relativeAbs]], brush=turquoise, pen = None, name = "P onsets")
                                else :
                                    points = pg.ScatterPlotItem(x=[x[relativeAbs]], y=[thisECG[relativeAbs]], brush=turquoise, pen = None)
                                oneGraph.addItem(points)

                        for a in pPeaks[0]:
                            if a < x_max[i] and a > x_min[i]:
                                relativeAbs = a - x_min[i]
                                if i == 0 :
                                    points = pg.ScatterPlotItem(x=[x[relativeAbs]], y=[thisECG[relativeAbs]], brush=vert, pen = None, name = "P peaks")
                                else :
                                    points = pg.ScatterPlotItem(x=[x[relativeAbs]], y=[thisECG[relativeAbs]], brush=vert, pen = None)
                                oneGraph.addItem(points)
                        
                        for a in qPeaks[0]:
                            if a < x_max[i] and a > x_min[i]:
                                relativeAbs = a - x_min[i]
                                if i == 0 :
                                    points = pg.ScatterPlotItem(x=[x[relativeAbs]], y=[thisECG[relativeAbs]], brush=orange, pen = None, name = "Q peaks")
                                else :
                                    points = pg.ScatterPlotItem(x=[x[relativeAbs]], y=[thisECG[relativeAbs]], brush=orange, pen = None)
                                oneGraph.addItem(points)
                        
                        for a in rPeaks[0]:
                            if a < x_max[i] and a > x_min[i]:
                                relativeAbs = a - x_min[i]
                                if i == 0 :
                                    points = pg.ScatterPlotItem(x=[x[relativeAbs]], y=[thisECG[relativeAbs]], brush=rouge, pen = None, name = "R peaks")
                                else :
                                    points = pg.ScatterPlotItem(x=[x[relativeAbs]], y=[thisECG[relativeAbs]], brush=rouge, pen = None)
                                oneGraph.addItem(points)

                        for a in sPeaks[0]:
                            if a < x_max[i] and a > x_min[i]:
                                relativeAbs = a - x_min[i]
                                if i == 0 :
                                    points = pg.ScatterPlotItem(x=[x[relativeAbs]], y=[thisECG[relativeAbs]], brush=rose, pen = None, name = "S peaks")
                                else :
                                    points = pg.ScatterPlotItem(x=[x[relativeAbs]], y=[thisECG[relativeAbs]], brush=rose, pen = None)
                                oneGraph.addItem(points)
                        
                        for a in tPeaks[0]:
                            if a < x_max[i] and a > x_min[i]:
                                relativeAbs = a - x_min[i]
                                if i == 0 :
                                    points = pg.ScatterPlotItem(x=[x[relativeAbs]], y=[thisECG[relativeAbs]], brush=violet, pen = None, name = "T peaks")
                                else :
                                    points = pg.ScatterPlotItem(x=[x[relativeAbs]], y=[thisECG[relativeAbs]], brush=violet, pen = None)
                                oneGraph.addItem(points) 

                        for a in tOffsets[0]:
                            if a < x_max[i] and a > x_min[i]:
                                relativeAbs = a - x_min[i]
                                if i == 0 :
                                    points = pg.ScatterPlotItem(x=[x[relativeAbs]], y=[thisECG[relativeAbs]], brush=bleu, pen = None, name = "T offsets")
                                else :
                                    points = pg.ScatterPlotItem(x=[x[relativeAbs]], y=[thisECG[relativeAbs]], brush=bleu, pen = None)
                                oneGraph.addItem(points)

                    
                    oneGraph.setTitle("Mean "+oneData._title)
                    oneGraph.setLabel('left', 'Magnitude')
                    oneGraph.setLabel('bottom', 'Time [s]')
                    oneGraph.setMinimumHeight(150)
                    
                    oneGraph.showGrid(x=True, y=True)
        
                
                elif infoVect[i][0]._title == "BPAo" or infoVect[i][0]._title == "BPleg" or infoVect[i][0]._title == "BParm":
                    
                    oneData = infoVect[i][0]
                    
                    # Create the graph and plot the data
                    oneGraph = pg.PlotWidget()
                    oneGraph.setMouseEnabled(x=False, y=False)
                    
                    try:
                        onsBPAo, bpY = getBpOnsets_tang(infoVect[i][0], rPeaks, lims = timelim, show = False)
                    except:
                        DialogPopup("Warning", "Detection of the onsets of the BP onsets with tangent method was unsuccessful. New try with the 2d derivative method.").exec()
                        try:
                            onsBPAo, bpY = getBpOnsets_tang(infoVect[i][0], rPeaks, lims = timelim, show = False)
                        except:
                            DialogPopup("Warning", "No sucessful detection of the BP onsets").exec()
                        
                    if infoVect[i][0]._title == "BPAo":
                       bpAo_ons = onsBPAo 
                    elif infoVect[i][0]._title == "BPleg":
                       bpLeg_ons = onsBPAo
                    elif infoVect[i][0]._title == "BParm":
                       bpArm_ons = onsBPAo 
                    
                    if timelim:
                        temp = []
                        temptime = []
                        for peak in onsBPAo[1]:
                            if peak > timelim[0] and peak < timelim[1]:
                                temp.append(int(peak*oneData._samplerate))
                                temptime.append(peak)
                        indexTime = [temp, temptime]
                        
                    diff = []
                    for i in range(len(onsBPAo[0])-1):
                        diff.append(onsBPAo[0][i+1] - onsBPAo[0][i])
                    
                    mean_diff = int(np.mean(diff))
                    
                    # We stock the ranges of indexes inside x_min and x_max
                    x_min = []
                    x_max = []
                    for i in range(len(onsBPAo[0])-1):
                        x_min.append(onsBPAo[0][i]-int(0.1*mean_diff))
                        x_max.append(onsBPAo[0][i]+int(0.9*mean_diff))                    

                    x = np.arange(x_min[0], x_max[0])
                    x = x-x[int(0.1*mean_diff)]
                    x = x/oneData._samplerate
                    
                    # Add the graph to the layout
                    self.layout.addWidget(oneGraph)
                    oneGraph.addLegend()
                    
                    for i in range(len(x_min)):
                        
                        thisBPAo = oneData._y[x_min[i]:x_max[i]]
                        courbe = oneGraph.plot(x, thisBPAo, pen='grey')
                        courbe.setOpacity(0.3)
                        
                        if i == 0 :
                            points = pg.ScatterPlotItem(x=[x[onsBPAo[0][i]-x_min[i]]], y=[bpY[i]], brush=rouge, pen = None, name = "BP onsets")
                        else :
                            points = pg.ScatterPlotItem(x=[x[onsBPAo[0][i]-x_min[i]]], y=[bpY[i]], brush=rouge, pen = None)
                        oneGraph.addItem(points)
                        

                    oneGraph.setTitle("Mean "+oneData._title)
                    oneGraph.setLabel('left', 'Magnitude')
                    oneGraph.setLabel('bottom', 'Time [s]')
                    oneGraph.setMinimumHeight(150)
                    oneGraph.showGrid(x=True, y=True)
                
        
        # Computation of the approximative PWV based on the BParm and BPleg
        meanDt_AoLeg = np.mean(np.array(bpLeg_ons[1]) - np.array(bpArm_ons[1]))
        #print("TEST : ", meanDt_AoLeg)
        
        if not height == 0:
            PWV = np.round(0.01* height * 0.8/meanDt_AoLeg, 2)
            pwvtxt = QLabel("The PWV for this person is ->"+ str(PWV) + " (" + str(0.01*height) + "m tall)")
        else :
            height = np.arange(1, 2.1, 0.1)  # [m]
            PWV = np.round(height * 0.8/meanDt_AoLeg, 2)  # [m/s]
            pwvtxt = QLabel("The PWV for this person are between ->"+ str(PWV[0]) + " (1m tall) -> " + str(PWV[5]) + " (1.5m tall) -> "+ str(PWV[-1]) + " (2m tall)")
        
        self.layout.addWidget(pwvtxt)
        #popup = DialogPopup("PWV measurement", "The PWV for this person are between\n->"+ str(PWV[0]) + " (1m tall)\n-> " + str(PWV[5]) + " (1.5m tall)\n-> "+ str(PWV[-1]) + " (2m tall)")
        #popup.exec()
        #del popup
        
        self.setMinimumSize(1000, 700)
    
    
    