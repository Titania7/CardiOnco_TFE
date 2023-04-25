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
    
Most urgent :
-------------
1) Pouvoir ouvrir fichiers matlab depuis les iCare selon leur code TG, SM, UI etc - SUR BASE DE LA DATE ET L'HEURE
2) Cross-correlation betweent the ECGs pour les réaligner et avoir les 2 infos de florine et HK.
   Ce qui est envoyé à HK est APRES ce qu'ils ont enregistré avec les fichiers de florine'

3) Cross corrélation entre les onsets de pression pour trouver le delta T
4) Save all datas selected in a single type of file.


Less urgent :
-------------
6) Add a custom reading option to calibrate single graph
7) Create "save" option to keep analyzed data

Optional :
----------
8) Add toggle button on ECG signal to show peaks on it
9) Make the R peaks movable with mouse control (and deletable !)
10) Register default reading preferences to the main window
11) Create the "complete" option to append new signals to the current file
12) Allow the user to zoom on the x-axis (mouse wheel + horizontal scrollbar controls)
13) Add a mean heart rate on ECG in order to visualize the quality of the signal
    >> ecg_cleaned = nk.ecg_clean(ecg, sampling_rate=300)
    >> quality = nk.ecg_quality(ecg_cleaned, sampling_rate=300)
    >> nk.signal_plot([ecg_cleaned, quality], standardize=True)
    
    
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

"""
#%%

from openFiles import openFile, getDataToDisplay, SingleGraph
from CustomPopups import DialogPopup, OptionPopup, DisplDataPopup, DisplMeanPopup
from simpleDisplay import *

import pyqtgraph as pg
pg.setConfigOption('background', 'w')
pg.setConfigOption('foreground', 'black')


from PyQt6.QtWidgets import *
# QMainWindow, QVBoxLayout, QWidget, QToolBar, QFileDialog, QCheckBox, QScrollArea, QApplication, QHBoxLayout
from PyQt6.QtGui import QAction, QIcon
from PyQt6.QtCore import Qt, QSize

import os
cwd = os.path.dirname(__file__)

#%%

class MyWindow(QMainWindow):
    
    infosVector = []
    infosVisible = []
    infosSelected = []
    
    clean_technique = "ECG_NeuroKit"
    #onset_detect = "inters_tg"
    ecg_delineate = "peaks" # "peak" or "dwt"

    
    pPeaks = []
    qPeaks = []
    rPeaks = []
    sPeaks = []
    tPeaks = []
    pOnsets = []
    tOffsets = []
    
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
        central_widget = QWidget()
        
        
        
        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu("File")
        
        open_action = file_menu.addAction("Open file")
        open_action.triggered.connect(self.open_file)
        save_action = file_menu.addAction("Save file")
        
        # To-do :
        #save_action.triggered.connect(self.save_file)
        
        reading_menu = menu_bar.addMenu("Preferences")
        
        display_action = reading_menu.addAction("Visible data")
        display_action.triggered.connect(self.chooseDataDispl)
        defRead_action = reading_menu.addAction("Reading parameters")
        defRead_action.triggered.connect(self.optionMenuPopUp)

        
        # Add a toolbar
        toolbar = QToolBar()
        self.addToolBar(toolbar)

        button_action = QAction("Analyze selection", self)
        button_action.triggered.connect(self.analyzeSelection)
        toolbar.addAction(button_action)



        # Create a scroll bar
        scroll_area = QScrollArea()
        #scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)

        scroll_area.setWidgetResizable(True)

        # Create a layout to stack the graphs vertically
        self.layout = QVBoxLayout()
        central_widget.setLayout(self.layout)
        
        scroll_area.setWidget(central_widget)
        # Set the central widget of the main window
        self.setCentralWidget(scroll_area)
        
        
        self.setMinimumSize(500, 400)
        # Full screen mode
        #self.showMaximized()
        self.setWindowTitle("CardiOnco")
        self.setWindowIcon(QIcon("icon.png"))

#%% Add an existing graph
    
    def addExistingGraph(self, infoVect_item):
        oneData = infoVect_item[0]
        info_1Graph = QHBoxLayout()
        # Add a checkbox before each graph
        checkbox = QCheckBox()
        checkbox.setChecked(True)
        info_1Graph.addWidget(checkbox)
        # Create both cursors for the graph
        posStart = oneData._x[0]
        posStop = oneData._x[len(oneData._x)-1]
        startCursor = pg.InfiniteLine(pos=posStart, bounds=[posStart, posStop], label='start', angle=90, movable=True, pen=pg.mkPen(width=3))
        stopCursor = pg.InfiniteLine(pos=posStop, bounds=[posStart, posStop], label='stop', angle=90, movable=True, pen=pg.mkPen(width=3))
        # Create the graph and plot the data
        oneGraph = pg.PlotWidget()
        oneGraph.setMouseEnabled(x=False, y=False)
        oneGraph.plot(oneData._x, oneData._y, pen='blue')
        
        if oneData._title == "ECG" :
            oneGraph.plot(oneData._x[self.rPeaks[0]], oneData._y[self.rPeaks[0]], pen=None, symbolSize = 5, symbol='o', symbolBrush='red', name="R peaks")
        if oneData._title == "BPleg": # or oneData._title == "BPleg" or oneData._title == "BParm" :
            #print(oneData._title)
            # Tangent method
            try:
                bpOnsets, bpY = getBpOnsets_tang(oneData, self.rPeaks, show = False)
                oneGraph.plot(bpOnsets[1], bpY, pen=None, symbolSize = 5, symbol='o', symbolBrush='red', name="Onsets (tangent method)")
            except:
                DialogPopup("Warning", "Unable to display the onsets of " + oneData._title + " on the whole graph (tangent method)").exec()
            # 2d deriv method
            try:
                bpOns_2dD = getBpOnsets_2dDeriv(oneData, self.rPeaks, show = False)
                oneGraph.plot(oneData._x[bpOns_2dD[0]], oneData._y[bpOns_2dD[0]], pen=None, symbolSize = 5, symbol='o', symbolBrush='green', name="Onsets (2d derivative method)")
            except:
                DialogPopup("Warning", "Unable to display the onsets of " + oneData._title + " on the whole graph (2d derivative method)").exec()
        
        
        SCG = ["x_scgLin[m/s^2]", "y_scgLin[m/s^2]", "z_scgLin[m/s^2]", "x_scgRot[deg/s]", "y_scgRot[deg/s]", "z_scgRot[deg/s]"]
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


        # Add the cursors to the new graph
        oneGraph.addItem(startCursor)
        oneGraph.addItem(stopCursor)
        oneGraph.setTitle(oneData._title)
        oneGraph.setLabel('left', 'Magnitude')
        oneGraph.setLabel('bottom', 'Time [s]')
        oneGraph.setMinimumHeight(200)
        oneGraph.addLegend()
        oneGraph.showGrid(x=True, y=True)
        # Add the graph to the layout
        info_1Graph.addWidget(oneGraph)
        self.layout.addLayout(info_1Graph)
        infoVect_item[1] = [startCursor, stopCursor]
        

    def clearGraphLayout(self):
        for counter in range(self.layout.count()):
            hbox_layout = self.layout.itemAt(counter).layout()
            if hbox_layout is not None:
                while hbox_layout.count():
                    item = hbox_layout.takeAt(0)
                    widget = item.widget()
                    if widget is not None:
                        widget.deleteLater()
                    else:
                        hbox_layout.removeItem(item)
            else :
                self.layout.removeItem(hbox_layout)
        self.layout.removeItem(self.layout.itemAt(counter))
    
        
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

    # Opens the file reading options
    def optionMenuPopUp(self):
        optionMenu = OptionPopup()
        optionMenu.exec()
        
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
                    
            posCurs = [self.infosVector[0][1][0].getXPos(), self.infosVector[0][1][1].getXPos()]
            #meanPopup = DisplMeanPopup(self.infosVector, self.infosVisible, pPeaks=self.pPeaks, qPeaks=self.qPeaks, rPeaks=self.rPeaks, sPeaks=self.sPeaks, tPeaks=self.tPeaks, pOnsets=self.pOnsets, tOffsets=self.tOffsets)
            
            self.w = DisplMeanPopup(self.height, self.infosVector, self.infosVisible, self.clean_technique, self.ecg_delineate, timelim=posCurs)
            self.w.show()


#%% Add new graph
    def addNewGraph(self, oneData):
        info_1Graph = QHBoxLayout()
        
        # Add a checkbox before each graph
        checkbox = QCheckBox()
        checkbox.setChecked(True)
        info_1Graph.addWidget(checkbox)
        
        # Create both cursors for the graph
        posStart = oneData._x[0]
        posStop = oneData._x[-1]
        startCursor = pg.InfiniteLine(pos=posStart, bounds=[posStart, posStop], label='start', angle=90, movable=True, pen=pg.mkPen(width=3))
        stopCursor = pg.InfiniteLine(pos=posStop, bounds=[posStart, posStop], label='stop', angle=90, movable=True, pen=pg.mkPen(width=3))
        # Create the graph and plot the data
        oneGraph = pg.PlotWidget()
        #oneGraph.wheelEvent
        oneGraph.setMouseEnabled(x=False, y=False)
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
        
        #oneGraph.plotItem.legend.setOffset(0.5)
        #print(oneData._title)
        if oneData._title == "ECG" :
            oneGraph.plot(oneData._x[self.rPeaks[0]], oneData._y[self.rPeaks[0]], pen=None, symbolSize = 5, symbol='o', symbolBrush='red', name="R peaks")
        if oneData._title == "BPleg": # or oneData._title == "BPleg" or oneData._title == "BParm" :
            #print(oneData._title)
            # Tangent method
            try:
                bpOnsets, bpY = getBpOnsets_tang(oneData, self.rPeaks, show = False)
                oneGraph.plot(bpOnsets[1], bpY, pen=None, symbolSize = 5, symbol='o', symbolBrush='red', name="Onsets (tangent method)")
            except:
                DialogPopup("Warning", "Unable to display the onsets of " + oneData._title + " on the whole graph (tangent method)").exec()
            # 2d deriv method
            try:
                bpOns_2dD = getBpOnsets_2dDeriv(oneData, self.rPeaks, show = False)
                oneGraph.plot(oneData._x[bpOns_2dD[0]], oneData._y[bpOns_2dD[0]], pen=None, symbolSize = 5, symbol='o', symbolBrush='green', name="Onsets (2d derivative method)")
            except:
                DialogPopup("Warning", "Unable to display the onsets of " + oneData._title + " on the whole graph (2d derivative method)").exec()
        
        
        SCG = ["x_scgLin[m/s^2]", "y_scgLin[m/s^2]", "z_scgLin[m/s^2]", "x_scgRot[deg/s]", "y_scgRot[deg/s]", "z_scgRot[deg/s]"]
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

        
        
        # Add the graph to the layout
        info_1Graph.addWidget(oneGraph)
        
        
        self.layout.addLayout(info_1Graph)
        # Note that this information is visible
        self.infosVisible.append(True)
        
        # By default, the information in selected for analysis
        self.infosSelected.append(True)
        
        
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
            if len(file_name) == 1:
                data, typeData = openFile(file_name[0])
                allData = getDataToDisplay(data, typeData)
                
                # Add a new graph for all the data contained in a single file
                for oneData in allData:
                    if oneData._title == "ECG":
                        self.pPeaks, self.qPeaks, self.rPeaks, self.sPeaks, self.tPeaks, self.pOnsets, self.tOffsets = detect_qrs(oneData, self.clean_technique, self.ecg_delineate, show = False)
                        self.addNewGraph(oneData)
                
                forbidden = ['signal1', 'signal3', 'signal4', 'signal5', 'ECG']
                for oneData in allData:
                    if not oneData._title in forbidden:
                        self.addNewGraph(oneData)

                self.synchroCursors()
                
            else :
                for file in file_name:
                    data, typeData = openFile(file)
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
                        self.name = dataJSON["Surname"]+" "+str(dataJSON["Date"])
                        self.sex = dataJSON["Sex[m/f]"]
                        self.age = dataJSON["Age[y]"]
                        self.weight = dataJSON["Weight[kg]"]
                        self.height = dataJSON["Height[cm]"]
                        #print(self.name, self.sex, self.age, self.weight, self.height)

                for typeD in typeDatas:
                    for graph in allDataJSON:
                        if graph._title == "ECG":
                            ecg_JSON = graph
                    for graph in allDataMLd:
                        if graph._title == "ECG":
                            ecg_MLd = graph
                    
                
                start, stop = getStartStopIndexes(ecg_JSON, ecg_MLd)
                
            
                for graph in allDataJSON:
                    allData.append(graph)
                    
                for graph in allDataMLd:
                    # Truncate the unused part of the MatLab data
                    x = graph._x[start:stop]
                    x = x-min(x)
                    y = graph._y[start:stop]
                    truncMLd = SingleGraph(x, y, graph._title, graph._samplerate, graph._step)
                    allData.append(truncMLd)
                    
                
            # Add a new graph for all the data contained in a single file
            for oneData in allData:
                if oneData._title == "ECG":
                    self.pPeaks, self.qPeaks, self.rPeaks, self.sPeaks, self.tPeaks, self.pOnsets, self.tOffsets = detect_qrs(oneData, self.clean_technique, self.ecg_delineate, show = False)
                    self.addNewGraph(oneData)
            
            forbidden = ['signal1', 'signal3', 'signal4', 'signal5', 'ECG']
            for oneData in allData:
                if not oneData._title in forbidden:
                    self.addNewGraph(oneData)

            self.synchroCursors()
                    
                
        
        else : 
            DialogPopup("Warning", "No file selected.").exec()
      
#%%

if __name__ == '__main__':
    if not QApplication.instance():
        app = QApplication([])
    else:
        app = QApplication.instance() 
    window = MyWindow()
    window.show()
    app.exec()