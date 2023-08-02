# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 10:32:09 2023

@author: Tatiana Dehon
"""
import json
import base64
import os

import numpy as np
import matplotlib.pyplot as plt

#%%

def readJSON(pathJSON : str):
    """
    This function allows to open a .kpy.json file from HeartKinetiks database and convert it to a
    directly readable dictionnary containing the metadata, the ecg and the ecg contained in the original file
    
    -----
    input :
    -----
    - pathJSON : string of the absolute path to the file 
    
    ----
    output :
    ----
    - session_data : dictionary containing metadata, ecg and scg
    """
    
    with open(pathJSON) as f:
        contents = json.load(f)
        
        
    #%% Robustness adaptation
    metadat = ""
    try :
        timestamp = contents["metaData"]["date"]
        metadat = "metaData"
    except :
        timestamp = contents["meta_data"]["date"]
        metadat = "meta_data"
    
    #%% Find the id letters + time
    timestamp = contents[metadat]["date"]
    
    time = timestamp.split("T")[1].split('.')[0]
    hour = time.split(':')[0]
    minute = time.split(':')[1]
    second = time.split(':')[2]

    date = timestamp.split("T")[0]
    day = date.split('-')[2]
    month = date.split('-')[1]
    year = date.split('-')[0]
    date = day+month+year + ' ' + str(hour+minute+second)[0:6]
    idLetters = pathJSON.split("\\")[-2]
    
    nameFile = idLetters + ' ' + date
    
    #%% Verify if ecg and scg are present
    ecg_data = None
    scg_data = None
    for item in contents["signals"]:
        try:
            if item["name"] == "scg":
                scg_data = item
        except:
            print("SCG not found")
        try:
            if item["name"] == "ecg":
                ecg_data = item
        except:
            print("ECG not found")

    # Registering of the metadata    
    age = contents[metadat]["subject"]["age"]["value"]
    height = contents[metadat]["subject"]["height"]["value"]
    weight = contents[metadat]["subject"]["weight"]["value"]
    sex = contents[metadat]["subject"]["sex"]

    # We take the duration at exactly 60s because errors in the files
    duration = 60

    #%% ECG content

    if ecg_data :
        ecg_factor = ecg_data["calibrations"]["I"]["factor"]

        ecg_fs = ecg_data["sample_rate"]["value"]
        ecg_step = 1/ecg_fs
        n_samples_ECG = duration*ecg_fs
        ecg_values = ecg_data["values"]["data"] # String type !

        # We decode the ecg encoding (base64 + int32) :
        decoded_bytes = base64.decodebytes(ecg_values.encode())
        ecg_values = np.frombuffer(decoded_bytes, dtype=np.int32)
        #correction with the given factor
        ecg_values = ecg_values*ecg_factor
        
        """
        # !! Verification (OK)
        plt.title("ECG")
        plt.plot(np.arange(0, duration, ecg_step), ecg_values) # ==> OK
        plt.show()
        """


    #%% SCG content
    if scg_data :
        scg_channels = scg_data["channels"]
        scg_values = scg_data["values"]["data"] # String type !
        # => same decoding as ECG but here base64 + float64
        decoded_bytes = base64.decodebytes(scg_values.encode())
        scg_values = np.frombuffer(decoded_bytes, dtype=np.float64)


        # Values should be organized with 6 columns :
        # number of values per column :
        numSample_perChannel = int(len(scg_values)/len(scg_channels))
        nbr_channels = len(scg_channels)
        scg_values = scg_values.reshape(numSample_perChannel, nbr_channels)
        scg_values = np.array([list(ele) for ele in list(zip(*scg_values))])

        # Error in the definition of the sampling rate in SCG thus we take the mathematical
        # definition of the SCG sampling frequency using the number of samples in the SCG and the
        # duration of the ECG (considered OK)
        scg_fs = numSample_perChannel/duration
        scg_step = 1/scg_fs

        """
        # !! verification (OK)
        
        count = 0
        for item in scg_values:
            plt.title("SCG channel_"+str(count)+ " ("+ scg_channels[count]+")")
            plt.plot(np.arange(0, duration, scg_step), item)
            plt.show()
            count = count+1
        """

    #%% Register the useful data in a dictionary

    session_data = {}

    meta = {}
    meta["age"] = age
    meta["height"] = height
    meta["weight"] = weight
    meta["sex"] = sex
    meta["nameFile"] = nameFile

    session_data["meta"] =  meta

    if ecg_data :
        ecg_final = {}
        ecg_final["duration"] = duration
        ecg_final["fs"] = ecg_fs
        ecg_final["step"] = ecg_step
        ecg_final["nSamples"] = n_samples_ECG
        ecg_final["time"] = list(np.arange(0, duration, ecg_step))
        ecg_final["magnitude"] = list(ecg_values)

        session_data["ecg"] = ecg_final

    if scg_data :
        scg_final = {}
        scg_final["duration"] = duration
        scg_final["fs"] = scg_fs
        scg_final["step"] = scg_step
        scg_final["nSamples"] = numSample_perChannel
        scg_final["time"] = list(np.arange(0, duration, scg_step))
        scg_final["X"] = list(scg_values[0])
        scg_final["Y"] = list(scg_values[1])
        scg_final["Z"] = list(scg_values[2])
        scg_final["RX"] = list(scg_values[3])
        scg_final["RY"] = list(scg_values[4])
        scg_final["RZ"] = list(scg_values[5])

        session_data["scg"] = scg_final


    return session_data


#%% Test the function

folder = r'C:\Users\32493\Documents\Mes_Trucs\Cours\CardiOnco_projectMa2\Scripts_perso\ALL_FILES'
for file_name in os.listdir(folder):
    print(file_name)


count = 1
for folders in os.listdir(folder):
    for file_name in os.listdir(folder + "\\" + folders):
        source = folder + "\\"+ folders + "\\" + file_name
        newData = readJSON(source)
    
        destination = folder + "\\"+ folders + "\\" + newData["meta"]["nameFile"] + ".json"
        try :
            os.rename(source, destination)
        except :
            print("File already exists")
            os.remove(source)
        count += 1





