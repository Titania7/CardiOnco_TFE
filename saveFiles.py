# -*- coding: utf-8 -*-
"""
Created on Fri Jul 28 16:15:27 2023

@author: Tatiana Dehon
"""

import os
import json


def saveRaw(metadata, allVectList):
    """
    Registers the raw data contained in a file to a normalized json type.
    """
    
    #%% Get the metaData
    name, sex, age, weight, height = metadata
    
    # Get the temporal positions of the cursors
    timeLims = [allVectList[0][1][0].getXPos(), allVectList[0][1][1].getXPos()]
    
    # Register what is contained inside eahc graphs
    singlegraph = []
    for item in allVectList:
        singlegraph.append(item[0])
        print(item[0])
    
    
    newData = {}
    # Add an identifier for the reading processing
    newData["CardiOnco"] = True
    
    # New data
    metadata = {}
    metadata["Name"] = name
    metadata["Age"] = age
    metadata["Sex"] = sex
    metadata["Height"] = height
    metadata["Weight"] = weight
    thisSession = {}
    thisSession["metaData"] = metadata
    thisSession["CursorsTimes"] = timeLims
    
    newTrack = {}
    for graph in singlegraph:
        this_track = {}
        this_track["time"] = graph._x.tolist()
        this_track["amplitude"] = graph._y.tolist()
        this_track["fs"] = graph._samplerate
        this_track["step"] = graph._step
        
        newTrack[graph._title] = this_track
    
    thisSession["Tracks"] = newTrack
    newData[name] = thisSession
    
    data = newData
    
    #%% Check if the registering folder exists

    # Get the Desktop path of the user
    path_desktop = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop')

    # Nom du dossier d'enregistrements
    name_folder = "CardiOnco recordings"
    
    
    
    # Complete path to the folder
    path_folder = os.path.join(path_desktop, name_folder)

    # Check if the folder already exists
    if not os.path.exists(path_folder): # If not, create it
        os.makedirs(path_folder)
        print(f"Folder '{name_folder}' successfully created on Desktop.")
    else: # Else, do nothing
        print(f"The '{name_folder}' folder already exists on Desktop.")
        
    #%% Create a new folder with the patient's code ABC
    name_patientfolder = name[0:3]
    
    # Complete the path to the folder
    path_patientfolder = os.path.join(path_folder, name_patientfolder)
    
    
    # Check if the folder already exists
    if not os.path.exists(path_patientfolder): # If not, create it
        os.makedirs(path_patientfolder)
        print(f"Folder '{name_patientfolder}' successfully created on Desktop.")
    else: # Else, do nothing
        print(f"The '{name_patientfolder}' folder already exists on Desktop.")
    
    #%% Create the path to the new file to be registered

    name_file = name[0:3]+".json"
    # Complete file path
    path_file = os.path.join(path_patientfolder, name_file)


    # Check if the file already exists
    #%% File already exists : modify it
    if os.path.exists(path_file):
        print("The file already exists.")
        # Open file in read mode
        with open(path_file, 'r') as file:
            # Load its data
            data = json.load(file)

        # Modification of the data
        data[name] = thisSession

        # Open file in write mode to register the modifications
        with open(path_file, 'w') as file:
            # Write the new data in the existing file
            json.dump(data, file)
            print("Modifications saved.")
            
    #%% File does not exist yet : save it
    else:
        print("The file does not exist. Impossible to modify it.")
        with open(path_file, 'w') as file:
            json.dump(data, file)
        print("New file registered successfully.")
    
    return path_patientfolder

from CustomPopups import *
def saveMean(qwindowPixmap, metadata):
    
    name, sex, age, weight, height = metadata
    
    # Get the Desktop path of the user
    path_desktop = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop')

    # Nom du dossier d'enregistrements
    name_folder = "CardiOnco recordings"

    # Complete path to the folder
    path_folder = os.path.join(path_desktop, name_folder)
    path_patientfolder = os.path.join(path_folder, name[0:3])

    # Check if the folder already exists
    if not os.path.exists(path_patientfolder): # If not, create it
        print("file does not exist yet")
        #DialogPopup("Warning", "Associated forlder does not exist yet.\nPlease register the raw file before the mean graphs.").exec()
    
    
    # Save the QPixmap to a PNG file
    file_name = os.path.join(path_patientfolder, name+".png")
    qwindowPixmap.save(file_name)
