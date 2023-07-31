# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 15:18:41 2023

@author: Tatiana Dehon

Research : decide if the ballistocardiography signals derived pulse wave velocity is
correlated significantly with the classical one, defined through the ECG and pression wave.
"""

#%%
"""
===== TO DO LIST : =====

TO DO list :
-------------
1) Decal meang graphs avant Q peaks
2) Add x y z lin et rot scg aux mean graphs pour voir influences
3) Register the mean graphs for a couple of files + metadata
4) Read the special types of files and special display + evolution of the PWV
5) Add special Mean Windows for only JSON or only MatLab files
6) Add the other AO points detections
7) Statistics on the efficiency of each technique
8) WRITE THE REPORT

    
    
===== DONE : =====
        
OK - Link display data choices to the main window
OK - Automatic frequency recognition in JSON files
OK - Make all ECG data readable for qrs detection
OK - Create a routine for finding the local minimum for the end of signal (shallow after the T peak : Q-T delay)
OK - Display mean curves for ECG
OK - Repair synchronysation of cursors after modification of the displayed data
OK - Repair acc.JSON files display
OK - Repair TXT files reading
OK - Create the different cases of file reading based on the user (external text file)
OK - Detect beginning of pressure curve : FIX SLOPE OF THE DERIVATIVE !!
OK - Create new window to display data to analyze (PQRST signals)
OK - Réparer la 2d derivative
OK - Cross-correlation between the ECGs pour les réaligner et avoir les 2 infos de Florine et HK
    Ce qui est envoyé à HK est APRES ce qu'ils ont enregistré avec les fichiers de florine!
OK - Create the "complete" option to append new signals to the current file
OK - Allow the user to zoom on the x-axis (mouse wheel + horizontal scrollbar controls)
OK - Pouvoir ouvrir fichiers matlab depuis les iCare selon leur code TG, SM, UI etc - SUR BASE DE LA DATE ET L'HEURE
OK - Cross corrélation entre les onsets de pression pour trouver le delta T
OK - Save all datas selected in a single type of file.
OK - Create "save" option to keep analyzed data
"""
#%% Librairies

from openFiles import*
from CustomPopups import *
from toolbox import *
from saveFiles import saveRaw

import numpy as np
import neurokit2 as nk

import pyqtgraph as pg
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'black')


from PyQt6.QtWidgets import *
# QMainWindow, QVBoxLayout, QWidget, QToolBar, QFileDialog, QCheckBox, QScrollArea, QApplication, QHBoxLayout
from PyQt6.QtGui import *
from PyQt6.QtCore import *


import os
cwd = os.path.dirname(__file__)

#%% 

class MyWindow(QMainWindow):
    
    infosVector = [] # All the Data loaded in the program (each item = [SingleGraph, [cursor1, cursor2]])
    infosVisible = [] # Boolean list that tells if the corresponding info of the infosVector is set to visible or not

    
    # Metadata useful for storage of analyzed graphs
    name = ""
    sex = ""
    age = 0
    weight = 0
    height = 0
    pwv = None
    
    
#%% Initialization of the main Window
    def __init__(self):
        super().__init__()
        
        # Create a central widget and set the layout
        self.central_widget = QWidget()
        self.central_widget.setSizePolicy(QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding))
        
        
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("File")
        
        open_action = file_menu.addAction("Open file")
        open_action.triggered.connect(self.open_file)
        save_action = file_menu.addAction("Save file")
        save_action.triggered.connect(self.save_file)

        
        reading_menu = menu_bar.addMenu("Preferences")
        
        display_action = reading_menu.addAction("Visible data")
        display_action.triggered.connect(self.chooseDataDispl)
        defRead_action = reading_menu.addAction("Edit metadata")
        defRead_action.triggered.connect(self.optionMenuPopUp)

        
        # Add a toolbar
        toolbar = QToolBar()
        self.addToolBar(toolbar)

        button_analyze = QAction("Analyze selection", self)
        button_analyze.triggered.connect(self.analyzeSelection)
        toolbar.addAction(button_analyze)
        
        button_reset = QAction("Reset", self)
        button_reset.triggered.connect(self.reset)
        toolbar.addAction(button_reset)



        # Create a layout to stack the graphs vertically
        self.layout = QVBoxLayout(self.central_widget)
        #central_widget.setLayout(self.layout)
        
        
        # Permet de resize les éléments du QVBoxLayout
        self.splitter = QSplitter()
        self.splitter.setOrientation(Qt.Orientation.Vertical)
        #self.splitter.setChildrenCollapsible(True)
        self.splitter.setOpaqueResize(False)
        

        
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(self.splitter)
        
        
        # Set the central widget of the main window
        self.setCentralWidget(scroll_area)
        
        #self.setMinimumSize(1000, 900)
        # Full screen mode
        self.showMaximized()
        self.setWindowTitle("CardiOnco")
        self.setWindowIcon(QIcon("icon.png"))

#%% Add an existing graph
    
    def addExistingGraph(self, infoVect_item):
        
        oneData = infoVect_item[0]
        
        # Create the graph and plot the data
        oneGraph = pg.PlotWidget()
        oneGraph.setSizePolicy(QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding))

        # Create both cursors for the graph
        posStart = oneData._x[0]
        posStop = oneData._x[-1]
        startCursor = pg.InfiniteLine(pos=posStart, bounds=[posStart, posStop], label='start', angle=90, movable=True, pen=pg.mkPen(width=3))
        stopCursor = pg.InfiniteLine(pos=posStop, bounds=[posStart, posStop], label='stop', angle=90, movable=True, pen=pg.mkPen(width=3))
        
        #oneGraph.wheelEvent
        #oneGraph.setMouseEnabled(x=False, y=False)
        oneGraph.plot(oneData._x, oneData._y, pen='blue')        
        # Add the cursors to the new graph
        oneGraph.addItem(startCursor)
        oneGraph.addItem(stopCursor)
        oneGraph.setTitle(oneData._title)
        oneGraph.setLabel('left', 'Magnitude')
        oneGraph.setLabel('bottom', 'Time [s]')
        oneGraph.setMinimumHeight(200)
        
        
        oneGraph.addLegend()
        oneGraph.showGrid(x=True, y=True)
        
        json_rPeaks = []
        if oneData._title == "ECG json" or oneData._title == "ECG mat" :
            if oneData._title == "ECG json":
                try:
                    json_pPeaks, json_qPeaks, json_rPeaks, json_sPeaks, json_tPeaks, json_pOnsets, json_tOffsets = detect_qrs(oneData, "ECG_NeuroKit", "peaks", show = False)
                    oneGraph.plot(oneData._x[json_rPeaks[0]], oneData._y[json_rPeaks[0]], pen=None, symbolSize = 5, symbol='o', symbolBrush='red', name="R peaks")
                except :
                    DialogPopup("Warning", "The quality of "+oneData._title+" is too bad to correctly detect the PQRST peaks.").exec()
            else :
                try:
                    mat_pPeaks, mat_qPeaks, mat_rPeaks, mat_sPeaks, mat_tPeaks, mat_pOnsets, mat_tOffsets = detect_qrs(oneData, "ECG_NeuroKit", "peaks", show = False)
                    oneGraph.plot(oneData._x[mat_rPeaks[0]], oneData._y[mat_rPeaks[0]], pen=None, symbolSize = 5, symbol='o', symbolBrush='red', name="R peaks")
                except :
                    DialogPopup("Warning", "The quality of "+oneData._title+" is too bad to correctly detect the PQRST peaks.").exec()
        if oneData._title == "BPleg": 
            # Tangent method
            try:
                bpOnsets, bpY = getBpOnsets_tang(oneData, json_rPeaks, filt = True, show = False)
                oneGraph.plot(bpOnsets[1], bpY, pen=None, symbolSize = 5, symbol='o', symbolBrush='red', name="Onsets (tangent method)")
            except:
                print("Unable to display the onsets of " + oneData._title + " on the whole graph (tangent method)")
            # 2d deriv method
            try:
                bpOns_2dD = getBpOnsets_2dDeriv(oneData, json_rPeaks, filt = True, show = False)
                oneGraph.plot(oneData._x[bpOns_2dD[0]], oneData._y[bpOns_2dD[0]], pen=None, symbolSize = 5, symbol='o', symbolBrush='green', name="Onsets (2d derivative method)")
            except:
                print("Unable to display the onsets of " + oneData._title + " on the whole graph (2d derivative method)")
                        
        self.layout.addWidget(oneGraph)
        infoVect_item[1] = [startCursor, stopCursor]
        
        self.splitter.addWidget(oneGraph)
        

    def clearGraphLayout(self):
        for i in reversed(range(self.splitter.count())):
            widget = self.splitter.widget(i)
            self.splitter.widget(i).deleteLater()
    
        
    def refreshGraphs(self):
        # Clean the current layout
        self.clearGraphLayout()
        # Add the visible data to the layout
        for i in range(len(self.infosVisible)):
            # If the info is set to visible :
            if self.infosVisible[i]:
                self.addExistingGraph(self.infosVector[i])
        self.synchroCursors()
                

        

#%% Options Menu
    
    def reset(self):
        print(self.name, self.sex, self.age, self.weight, self.height, self.pwv)
        self.name = ""
        self.sex = ""
        self.age = 0
        self.weight = 0
        self.height = 0
        self.pwv = None
        self.clearGraphLayout()
        if not self.infosVector == []:
            for vect in [self.infosVector, self.infosVisible]:
                vect.clear()
        print(self.name, self.sex, self.age, self.weight, self.height, self.pwv)
        

    # Opens the file reading options
    def optionMenuPopUp(self):
        optionMenu = OptionPopup(self.name, self.sex, self.age, self.height, self.weight)
        # In case the metadata are modified
        optionMenu.newName.connect(self.updateName)
        optionMenu.newSex.connect(self.updateSex)
        optionMenu.newAge.connect(self.updateAge)
        optionMenu.newWeight.connect(self.updateWeight)
        optionMenu.newHeight.connect(self.updateHeight)
        optionMenu.exec()

    def updateName(self, newName):
        self.name = newName
    
    def updateSex(self, newSex):
        self.sex = newSex
    
    def updateAge(self, newAge):
        self.age = newAge
    
    def updateHeight(self, newHeight):
        self.height = newHeight
        
    def updateWeight(self, newWeight):
        self.weight = newWeight
        
    # Updates the list of visible graphs based on the DisplDataPopup and refreshed the mainwindow layout
    def updateVisVect(self, newVect):
        #print("In newVisVect : ", newVect)
        self.infosVisible = newVect
        self.refreshGraphs()
        
    # Adds a popup to choos the loaded data to be displayed
    def chooseDataDispl(self):
        if len(self.infosVector) != 0 :
            myDataPopup = DisplDataPopup(self.infosVector, self.infosVisible)
            myDataPopup.newVisVect.connect(self.updateVisVect)
            myDataPopup.exec()
            
#%% Analyze selection

    # Prints a graph with peaks identified and mean graphs based on them
    def analyzeSelection(self):
        if len(self.infosVisible) != 0 :
            # Retrieve the position of the cursors        
            posCurs = [self.infosVector[0][1][0].getXPos(), self.infosVector[0][1][1].getXPos()]
            
            #Search the Peaks in the CURRENTLY VISIBLE ECG JSON then add the results to the list to be sent to the secondary window
            json_pPeaks = []
            json_qPeaks = []
            json_rPeaks = []
            json_sPeaks = []
            json_tPeaks = []
            json_pOnsets = []
            json_tOffsets = []
            
            mat_pPeaks = []
            mat_qPeaks = []
            mat_rPeaks = []
            mat_sPeaks = []
            mat_tPeaks = []
            mat_pOnsets = []
            mat_tOffsets = []
            
            for oneData in self.infosVector:
                if oneData[0]._title == "ECG json":
                    json_pPeaks, json_qPeaks, json_rPeaks, json_sPeaks, json_tPeaks, json_pOnsets, json_tOffsets = detect_qrs(oneData[0], "ECG_NeuroKit", "peaks", show = False)
                elif oneData[0]._title == "ECG mat":
                    mat_pPeaks, mat_qPeaks, mat_rPeaks, mat_sPeaks, mat_tPeaks, mat_pOnsets, mat_tOffsets = detect_qrs(oneData[0], "ECG_NeuroKit", "peaks", show = False)
            
            okMATflag = False
            okJSONflag = False
            meta = []

            try:
                pqrstMAT = [mat_pPeaks, mat_qPeaks, mat_rPeaks, mat_sPeaks, mat_tPeaks, mat_pOnsets, mat_tOffsets]
                okMATflag = True
            except:
                DialogPopup("Error", "No No MatLab ECG found.").exec()
            try:
                meta = [self.name, self.sex, self.age, self.weight, self.height]
                pqrstJSON = [json_pPeaks, json_qPeaks, json_rPeaks, json_sPeaks, json_tPeaks, json_pOnsets, json_tOffsets]
                okJSONflag = True
            except:
                DialogPopup("Error", "No JSON ECG found.").exec()

            if okMATflag == True and okJSONflag == True:
                #print(mat_pPeaks[0][0], mat_qPeaks[0][0], mat_rPeaks[0][0], mat_sPeaks[0][0], mat_tPeaks[0][0])
                self.w = DisplMeanWindow(meta, pqrstJSON, pqrstMAT, self.infosVector, self.infosVisible, timelim=posCurs)
                self.w.show()
            elif okMATflag == True and okJSONflag == False:
                DialogPopup("Error", "Please load a Matlab file AND a json file").exec()
            elif okMATflag == False and okJSONflag == True:
                DialogPopup("Error", "Please load a Matlab file AND a json file").exec()
            else :
                DialogPopup("Error", "No ECG found").exec()


        
#%% Add new graph

    def addNewGraph(self, oneData):
        
        # Create the graph and plot the data
        oneGraph = pg.PlotWidget()
        oneGraph.setSizePolicy(QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding))

        # Create both cursors for the graph
        posStart = oneData._x[0]
        posStop = oneData._x[-1]
        startCursor = pg.InfiniteLine(pos=posStart, bounds=[posStart, posStop], label='start', angle=90, movable=True, pen=pg.mkPen(width=3))
        stopCursor = pg.InfiniteLine(pos=posStop, bounds=[posStart, posStop], label='stop', angle=90, movable=True, pen=pg.mkPen(width=3))
        
        #oneGraph.wheelEvent
        #oneGraph.setMouseEnabled(x=False, y=False)
        oneGraph.plot(oneData._x, oneData._y, pen='blue')        
        # Add the cursors to the new graph
        oneGraph.addItem(startCursor)
        oneGraph.addItem(stopCursor)
        oneGraph.setTitle(oneData._title)
        oneGraph.setLabel('left', 'Magnitude')
        oneGraph.setLabel('bottom', 'Time [s]')
        oneGraph.setMinimumHeight(200)
        
        
        oneGraph.addLegend()
        oneGraph.showGrid(x=True, y=True)
        
        json_rPeaks = []
        if oneData._title == "ECG json" or oneData._title == "ECG mat" :
            if oneData._title == "ECG json":
                try:
                    json_pPeaks, json_qPeaks, json_rPeaks, json_sPeaks, json_tPeaks, json_pOnsets, json_tOffsets = detect_qrs(oneData, "ECG_NeuroKit", "peaks", show = False)
                    oneGraph.plot(oneData._x[json_rPeaks[0]], oneData._y[json_rPeaks[0]], pen=None, symbolSize = 5, symbol='o', symbolBrush='red', name="R peaks")
                except :
                    DialogPopup("Warning", "The quality of "+oneData._title+" is too bad to correctly detect the PQRST peaks.").exec()
            else :
                try:
                    mat_pPeaks, mat_qPeaks, mat_rPeaks, mat_sPeaks, mat_tPeaks, mat_pOnsets, mat_tOffsets = detect_qrs(oneData, "ECG_NeuroKit", "peaks", show = False)
                    oneGraph.plot(oneData._x[mat_rPeaks[0]], oneData._y[mat_rPeaks[0]], pen=None, symbolSize = 5, symbol='o', symbolBrush='red', name="R peaks")
                except :
                    DialogPopup("Warning", "The quality of "+oneData._title+" is too bad to correctly detect the PQRST peaks.").exec()
        if oneData._title == "BPleg": 
            # Tangent method
            try:
                bpOnsets, bpY = getBpOnsets_tang(oneData, json_rPeaks, filt = True, show = False)
                oneGraph.plot(bpOnsets[1], bpY, pen=None, symbolSize = 5, symbol='o', symbolBrush='red', name="Onsets (tangent method)")
            except:
                print("Unable to display the onsets of " + oneData._title + " on the whole graph (tangent method)")
            # 2d deriv method
            try:
                bpOns_2dD = getBpOnsets_2dDeriv(oneData, json_rPeaks, filt = True, show = False)
                oneGraph.plot(oneData._x[bpOns_2dD[0]], oneData._y[bpOns_2dD[0]], pen=None, symbolSize = 5, symbol='o', symbolBrush='green', name="Onsets (2d derivative method)")
            except:
                print("Unable to display the onsets of " + oneData._title + " on the whole graph (2d derivative method)")
        
        
        SCG = ["x_scgLin[m/s^2]", "y_scgLin[m/s^2]", "z_scgLin[m/s^2]", "x_scgRot[deg/s]", "y_scgRot[deg/s]", "z_scgRot[deg/s]"]
        """
        if oneData._title in SCG :
            try :
                b, a = signal.butter(N=3, Wn=0.3, btype='high', fs=oneData._samplerate)
                zi = signal.lfilter_zi(b, a)
                z, _ = signal.lfilter(b, a, oneData._y, zi=zi*oneData._y[0])
                z2, _ = signal.lfilter(b, a, z, zi=zi*z[0])
                scg_clean = signal.filtfilt(b, a, oneData._y)
                
                #print(len(scg_clean))
                oneGraph.plot(oneData._x, scg_clean, pen='green', name="Without respiration and noise")
            except:
                DialogPopup("Warning", "Unable to display " + oneData._title + " cleaned from respiration").exec()
        """
        
        
        self.layout.addWidget(oneGraph)
        
        self.splitter.addWidget(oneGraph)
        
        
        # Note that this information is visible
        self.infosVisible.append(True)
        
        
        
        # Store the displayed information inside the global variable in order to manipulate them later on if necessary
        # Data object, graph, cursors objects associated
        
        self.infosVector.append([oneData, [startCursor, stopCursor]])



#%% Open a file an synchr cursors

    def synchroCursors(self):
        temp = self.infosVector
        tempVis = self.infosVisible
        #print(len(self.infosVisible), len(temp))
        
        def startCursorPos_copy(self):
            for i in range(len(tempVis)):
                if tempVis[i] == True :
                    temp[i][1][0].sigPositionChanged.disconnect(startCursorPos_copy)
                    temp[i][1][0].setPos(self.pos().x())
                    temp[i][1][0].sigPositionChanged.connect(startCursorPos_copy)
            
        def stopCursorPos_copy(self):
            for i in range(len(tempVis)):
                if tempVis[i] == True :
                    temp[i][1][1].sigPositionChanged.disconnect(stopCursorPos_copy)
                    temp[i][1][1].setPos(self.pos().x())
                    temp[i][1][1].sigPositionChanged.connect(stopCursorPos_copy)
       
        
        if len(self.infosVector) > 0 :
            for i in range(len(tempVis)):
                if self.infosVisible[i] == True :
                    self.infosVector[i][1][0].sigPositionChanged.connect(startCursorPos_copy)
                    self.infosVector[i][1][1].sigPositionChanged.connect(stopCursorPos_copy)
    
    
    def open_file(self):
        # open a file explorer and get the selected files names
        file_name, _ = QFileDialog.getOpenFileNames(self, "Open New File", "", " (*.mat *.json *.txt)")
        
        
        typeDatas = []
        if file_name:
            # Si un seul fichier chargé
            if len(file_name) == 1:
                data, typeData = openFile(self, file_name[0])
                allData = getDataToDisplay(data, typeData)
                
                extension = file_name[0].split(".")[-1]
                if extension == "json"  and "meta" in list(data.keys()):
                    self.name = data["meta"]["nameFile"][0:19]
                    self.sex = data["meta"]["Sex[m/f]"]
                    self.age = data["meta"]["Age[y]"]
                    self.weight = data["meta"]["Weight[kg]"]
                    self.height = data["meta"]["Height[cm]"]
                    
                    pwvtxt_meta = QLabel("Identifier : "+self.name+" ; Sex : "+ self.sex+" ; Age : "+str(self.age)+" years ; Weight : "+ str(self.weight)+ " kg ; Height : "+ str(self.height)+" m")
                    self.layout.addWidget(pwvtxt_meta)
                    self.splitter.addWidget(pwvtxt_meta)
                elif extension == "json"  and "metaData" in list(data.keys()):
                    self.name = data["metaData"]["Name"]
                    self.sex = data["metaData"]["Sex"]
                    self.age = data["metaData"]["Age"]
                    self.weight = data["metaData"]["Weight"]
                    self.height = data["metaData"]["Height"]
                    
                    pwvtxt_meta = QLabel("Identifier : "+self.name+" ; Sex : "+ self.sex+" ; Age : "+str(self.age)+" years ; Weight : "+ str(self.weight)+ " kg ; Height : "+ str(self.height)+" m")
                    self.layout.addWidget(pwvtxt_meta)
                    self.splitter.addWidget(pwvtxt_meta)
                    
                # Add a new graph for all the data contained in a single file
                if not "metaData" in list(data.keys()):
                    for oneData in allData:
                        if oneData._title == "ECG":
                            oneData._title = oneData._title + " " + extension
                            
                            corrECG, inverted = nk.ecg_invert(oneData._y, sampling_rate=oneData._samplerate, force=False, show=False)
                            if inverted == True:
                                print(oneData._title , "signal was inverted")
                                oneData._y = corrECG
                            else :
                                print(oneData._title , "signal was not inverted")
                            self.addNewGraph(oneData)
                else :
                    for oneData in allData:
                        if oneData._title == "ECG json" or oneData._title == "ECG mat":
                            
                            corrECG, inverted = nk.ecg_invert(oneData._y, sampling_rate=oneData._samplerate, force=False, show=False)
                            if inverted == True:
                                print(oneData._title , "signal was inverted")
                                oneData._y = corrECG
                            else :
                                print(oneData._title , "signal was not inverted")
                            self.addNewGraph(oneData)
                
                forbidden = ['signal1', 'signal3', 'signal4', 'signal5', 'ECG json', 'ECG mat']
                for oneData in allData:
                    if not oneData._title in forbidden:
                        self.addNewGraph(oneData)


            # Si deux fichiers chargés
            elif len(file_name) == 2 :
                for file in file_name:
                    data, typeData = openFile(self, file)
                    typeDatas.append([typeData, data])
                
                allData = []
                allDataJSON = []
                allDataMLd = []
                
                for typeD in typeDatas:
                    if typeD[0] == 'MatLab':
                        allDataMLd = getDataToDisplay(typeD[1], typeD[0])
                    if typeD[0] == 'JSON':
                        dataJSON = typeD[1]
                        allDataJSON = getDataToDisplay(dataJSON, typeD[0])
                        self.name = dataJSON["meta"]["nameFile"][0:19]
                        self.sex = dataJSON["meta"]["Sex[m/f]"]
                        self.age = dataJSON["meta"]["Age[y]"]
                        self.weight = dataJSON["meta"]["Weight[kg]"]
                        self.height = dataJSON["meta"]["Height[cm]"]
                        
                        pwvtxt_meta = QLabel("Identifier : "+self.name+" ; Sex : "+ self.sex+" ; Age : "+str(self.age)+" years ; Weight : "+ str(self.weight)+ " kg ; Height : "+ str(self.height)+" m")
                        self.layout.addWidget(pwvtxt_meta)
                        self.splitter.addWidget(pwvtxt_meta)

                for graph in allDataJSON:
                    if graph._title == "ECG" :
                        ecg_JSON = graph
                        ecg_JSON._title = "ECG json"
                        ecg_JSON._y = ecg_JSON._y/max(abs(ecg_JSON._y))
                        corrECG, inverted = nk.ecg_invert(ecg_JSON._y, sampling_rate=ecg_JSON._samplerate, force=False, show=False)
                        if inverted == True:
                            print(ecg_JSON._title , "signal was inverted")
                            ecg_JSON._y = corrECG
                        else :
                            print(ecg_JSON._title , "signal was not inverted")
                for graph in allDataMLd:
                    if graph._title == "ECG":
                        ecg_MLd = graph
                        ecg_MLd._title = "ECG mat"
                        corrECG, inverted = nk.ecg_invert(ecg_MLd._y, sampling_rate=ecg_MLd._samplerate, force=False, show=False)
                        if inverted == True:
                            print(ecg_JSON._title , "signal was inverted")
                            ecg_MLd._y = corrECG
                        else :
                            print(ecg_MLd._title , "signal was not inverted")
                            
                    elif graph._title == "BPleg" :
                        # We filter the low frequency noise to get a "flat" signal
                        graph._y = butterHighPass(graph = graph, filtorder = 4, limfreq = 0.75, show=False)
                
                try :
                    which, [start, stop] = getStartStopIndexes(ecg_JSON, ecg_MLd, show=True)
                    # "which" is the longest recording to be trimmed
                except : 
                    DialogPopup("Warning", "Synchronization seems impossible.\n The files might not be linked.").exec()

                
                for graph in allDataJSON:
                    if which._title == "ECG json":
                        # Truncate the unused part of the MatLab data
                        x = graph._x[start:stop]
                        x = x-min(x)
                        y = graph._y[start:stop]
                        truncMLd = SingleGraph(x, y, graph._title, graph._samplerate, graph._step)
                        allData.append(truncMLd)
                    else :
                        allData.append(graph)
                    
                for graph in allDataMLd:
                    if which._title == "ECG mat":
                        # Truncate the unused part of the MatLab data
                        x = graph._x[start:stop]
                        x = x-min(x)
                        y = graph._y[start:stop]
                        truncMLd = SingleGraph(x, y, graph._title, graph._samplerate, graph._step)
                        allData.append(truncMLd)
                    else:
                        allData.append(graph)
                
                
                # Add amplitude vectors of SCG to the list of available data
                try :
                    for oneData in allData :
                        for oneData in allData:
                            if oneData._title == "x_scgLin[m/s^2]":
                                xLin = oneData
                                #allData.append(xLin)
                            if oneData._title == "y_scgLin[m/s^2]":
                                yLin = oneData
                                #allData.append(yLin)
                            if oneData._title == "z_scgLin[m/s^2]":
                                zLin = oneData
                                #allData.append(zLin)
                            if oneData._title == "x_scgRot[deg/s]":
                                xRot = oneData
                                #allData.append(xRot)
                            if oneData._title == "y_scgRot[deg/s]":
                                yRot = oneData
                                #allData.append(yRot)
                            if oneData._title == "z_scgRot[deg/s]":
                                zRot = oneData
                                #allData.append(xLin)

                    amplLin_vect = np.sqrt(xLin._y**2 + yLin._y**2 + zLin._y**2)
                    amplRot_vect = np.sqrt(xRot._y**2 + yRot._y**2 + zRot._y**2)
                    
                    linSCG = SingleGraph(xLin._x, amplLin_vect, "lin_SCGvect[m/s^2]", xLin._samplerate, xLin._step)
                    rotSCG = SingleGraph(xRot._x, amplRot_vect, "rot_SCGvect[deg/s]", xRot._samplerate, xRot._step)
                    
                    for graph in [linSCG, rotSCG]:
                        allData.append(graph)
                        
                except :
                    DialogPopup("Warning", "Amplitude of SCG vector could not be computed").exec()
                
                
                # Add a new graph for all the data contained in both files (first the ECGs)
                for oneData in allData:
                    if oneData._title == "ECG json" or oneData._title == "ECG mat":
                        self.addNewGraph(oneData)
 
    
                forbidden = ['signal1', 'signal3', 'signal4', 'signal5', 'ECG json', 'ECG mat']
                for oneData in allData:
                    if not oneData._title in forbidden:
                        self.addNewGraph(oneData)
            
            else :
                DialogPopup("Error", "You can only load one MatLab/JSON file at a time or one pair MatLab/JSON.").exec()
            
            self.synchroCursors()
                
        
        else : 
            DialogPopup("Warning", "No file selected.").exec()
      
#%% Save the file

    def save_file(self):
        if self.name == "" and self.sex == "" and self.age == 0 and self.height == 0 and self.weight == 0:
            DialogPopup("Warning", "Please fill in the metadata in order to save the file.").exec()
        else :
            meta = [self.name, self.sex, self.age, self.weight, self.height]
            saveRaw(meta, self.infosVector)
            print("Save in a custom file")
        return

#%% Execution

if __name__ == '__main__':
    if not QApplication.instance():
        app = QApplication([])
    else:
        app = QApplication.instance() 
    window = MyWindow()
    window.show()
    app.exec()