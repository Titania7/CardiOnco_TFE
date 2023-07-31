# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 13:30:02 2023
@author: Tatiana Dehon

 ========= Toolbox for files opening =========
 
"""

#from simpleDisplay import display_plt, displ_single_graph
from SingleGraph import SingleGraph
from toolbox import *
from CustomPopups import *

import numpy as np
import scipy.io as sc

import os, gc, sys, json, base64, math
from PyQt6.QtWidgets import *




# Where are we in the system ?
cwd = os.path.dirname(__file__)

with open("allNames.json") as f:
    allNames = json.load(f)
#print("Current work directory : ", cwd)

#%% Funcion to call for having the right channels names

def idNames(typeFile : str, nbChannels : int):
    if typeFile in list(allNames.keys()):

        if typeFile == "new_MatLab":
            return allNames[typeFile]['5']
        else:
            return allNames[typeFile][str(nbChannels)]

#%% MatLabdigere function

def openMatLabdigere(pathfile : str):
    
    # Load the file in memory
    mat = sc.loadmat(pathfile)
    
    
    
    if 'n' in list(mat.keys()):
        print("Florine type")
        
        nbrMeasures = mat['n'][0][0] # int type => OK
        #print("Number of measures = ", nbrMeasures)
        

        data={}
        for i in range(nbrMeasures):
            name = 'b'+str((i+1))
            nbrChannels = mat[name].shape[1]
            #print("Number of channels = ", nbrChannels)
            trackNames = idNames('new_MatLab', nbrChannels)
            
            
            temp = []
            for j in range(len(mat[name])):
                temp.append(mat[name][j])
            finalTemp = [list(ele) for ele in list(zip(*temp))]
            count = 0
            temp = {}
            for graph in finalTemp :
                temp[trackNames[count]] = {"signal": np.array(graph), "samplerate": 200}
                count = count+1
            data["Measure "+str(i+1)] = temp

        #%% AJOUTER NOM DES TRACKS SPÉCIAL « MATLAB_FLO_tracks »
        
        
        
    elif 'data' in list(mat.keys()):
        print("Classic type")
        
        nbrMeasures = mat['data'].shape[0]
        #print("Number of measures = ", nbrMeasures)
        trackNames = idNames('old_MatLab', 8)
        
        # Clean the display
        titlesToSqueeze = ['data', 'dataend', 'datastart', 'firstsampleoffset',
                           'rangemax', 'rangemin', 'samplerate', 'unittextmap']
        for cat in titlesToSqueeze :
            mat[cat] = np.squeeze(mat[cat])

        # Make indexes float to int 
        makeInt = ['dataend', 'datastart']
        for indexes in makeInt :
            mat[indexes] = np.array(mat[indexes], dtype='int')
        
        
        # Count the number of channels
        #print("Number of channels = ", mat['dataend'].shape[0])
        
        # Store useful data in new dictionary
        data={}
        for i in range(nbrMeasures):
            for i in range(mat['dataend'].shape[0]): # For each type of data contained
                temp = np.empty((mat['dataend'][i]-mat['datastart'][i]), dtype='float64')
                k=0
                for j in range(mat['datastart'][i], mat['dataend'][i]):
                    temp[k] = mat['data'][j]
                    k = k+1
                    data[trackNames[i]] = {"signal": temp, "samplerate": mat['samplerate'][i]}

        #%% AJOUTER NOM DES TRACKS « MATLAB_tracks »
        
        
        
        
    # Returns dictionary with data, min-max values and sampling frequency for each channel (here 8)
    return data


#%% JSON function

def openJSON(window, pathfile : str):
    
    print(pathfile)
    
    with open(pathfile) as f:
        contents = json.load(f)
    print(list(contents.keys()))
    
    if "CardiOnco" in list(contents.keys()):
        print("CardiOnco file")
        
        # First we need to know which file we have to display
        keys_list = list(contents.keys())
        # Remove the "CardiOnco" tag
        keys_list.remove("CardiOnco")

        # Show a popup dialog with the list of keys
        key, ok = QInputDialog.getItem(window, "Choose a Key", "Select a key:", keys_list, editable=False)

        if ok and key:
            QMessageBox.information(window, "Selected Key", f"Key: {key}")
            
        toreturn = contents[key]
        
        
        
    
    elif list(contents.keys()) == ["data"]:
        print("Hiba's file")
        list_data = contents["data"] # On récupère la liste

        # On prépare le vecteur des millisecondes (supposées) et des valeurs d'ECG
        timestamps = np.array([])
        monECG = np.array([])

        # On remplit les vecteurs
        for item in list_data:
            timestamps = np.append(timestamps, item["ecg"]["Timestamp"])
            monECG = np.append(monECG, item["ecg"]["Samples"])

        # On calcule la fréquence d'échantillonnage supposée et on la stocke dans fs
        steps_16samples = np.round(np.mean(np.diff(timestamps)))
        step_ms = steps_16samples / 16 # Supposé en millisecondes
        # step = 1/fs
        step_s = step_ms/1000
        fs = 1/step_s
        print("Freq d'échant if 1 ECG = ", fs)

        # On crée un vecteur temporel de même longueur que l'ECG en prenant en compte fs pour l'affichage
        time = np.arange(0, len(monECG)*step_s, step_s)
        
        toret = {}
        ecgDict = {}
        ecgDict['sfreq[Hz]'] = fs
        ecgDict['numSamples'] = len(monECG)
        ecgDict['duration[s]'] = len(monECG)*step_s
        ecgDict['ECG[uV]'] = list(monECG)
        ecgDict['ECGfactor'] = 1
        toret['ECG'] = ecgDict
        
        toreturn = toret

    
    elif "metaData" in list(contents.keys()) or "meta_data" in list(contents.keys()):
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
        date = day+month+year + ' ' + hour+minute+second + timestamp[-1]
        idLetters = pathfile.split("/")[-2]
        
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

            #ecg_fs = ecg_data["sample_rate"]["value"]
            # We force the value at 200 because too many errors in encoding make the system bug
            ecg_fs = 200
            ecg_step = 1/ecg_fs
            n_samples_ECG = duration*ecg_fs
            ecg_values = ecg_data["values"]["data"] # String type !

            # We decode the ecg encoding (base64 + int32) :
            decoded_bytes = base64.decodebytes(ecg_values.encode())
            ecg_values = np.frombuffer(decoded_bytes, dtype=np.int32)
            #correction with the given factor
            #ecg_values = ecg_values*ecg_factor
            
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

        toret = {}
        
        meta = {}
        meta["nameFile"] = nameFile
        meta['Sex[m/f]'] = sex
        meta['Weight[kg]'] = weight
        meta['Height[cm]'] = height
        meta['Age[y]'] = age

        toret["meta"] =  meta

        if ecg_data :
            
            ecgDict = {}
            ecgDict['sfreq[Hz]'] = ecg_fs
            ecgDict['numSamples'] = n_samples_ECG
            ecgDict['duration[s]'] = duration
            ecgDict['ECG[uV]'] = list(ecg_values)
            ecgDict['ECGfactor'] = ecg_factor
            toret['ECG'] = ecgDict
            

        if scg_data :
            scgDict = {}
            scgDict['sfreq[Hz]'] = scg_fs
            scgDict['numSamples'] = numSample_perChannel
            scgDict['duration[s]'] = duration
            scgDict['scgLin[m/s^2]'] = {"x": scg_values[0] , "y": scg_values[1], "z": scg_values[2]}
            scgDict['scgRot[deg/s]'] = {"x": scg_values[3] , "y": scg_values[4], "z": scg_values[5]}
            toret['SCG'] = scgDict                
            
        toreturn = toret

    
    elif 'data' in list(contents.keys()):
        contents = contents['data']
    
    
        typeOfData = list(contents[0].keys())
    
        if 'ecg' in typeOfData:
            keys = list(contents[0]['ecg'].keys())
            #print("keys = ", keys)
            ecg = dict.fromkeys(keys, [])

            finalData = []
            timestamps = []
            for k in range(len(contents)):
                timestamps.append(0.001*contents[k]['ecg'][keys[0]]) # in seconds
                for content in contents[k]['ecg'][keys[1]]:
                    finalData.append(content)
        
            del ecg[keys[0]]
        
            # Detection of the frequency :
            periods = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
            mean_dt_burst = round(np.mean(periods), 4)
            #print("mean_dt_burst [s] = ", mean_dt_burst)
            mean_dt_sample = mean_dt_burst/16
            #print("mean_dt_sample [s] = ", mean_dt_sample)
            ecg["Frequency"] = math.ceil(1/mean_dt_sample)
            #print('final freq [Hz] = ', ecg["Frequency"])
            trackNames = idNames('JSON_ecg', 1)
            del ecg[keys[1]]
            ecg[trackNames[0]] = np.array(finalData)
            #print("keys = ", list(ecg.keys()))
            toreturn = ecg
        
        
        
    
        elif 'acc' in typeOfData:
            keys = list(contents[0]['acc'].keys())
            acc = dict.fromkeys(keys, [])
            spatialKeys = list(contents[0]['acc'][keys[1]][0].keys())
            spatialDict = dict.fromkeys(spatialKeys, [])
            acc[keys[1]] = spatialDict
        
        
            finalDataX = []
            finalDataY = []
            finalDataZ = []
            timestamps = []
            for k in range(len(contents)):
                timestamps.append(contents[k]['acc'][keys[0]]) # in milliseconds
                for content in contents[k]['acc'][keys[1]]:
                    finalDataX.append(content[spatialKeys[0]])
                    finalDataY.append(content[spatialKeys[1]])
                    finalDataZ.append(content[spatialKeys[2]])
        
            # Detection of the frequency :
            timestamps = [x/1000 for x in timestamps] # conversion in seconds
            periods = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
            
            del acc[keys[0]]
        
            # Detection of the frequency :
            periods = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
            mean_dt_burst = round(np.mean(periods), 4)
            print("mean_dt_burst = ", mean_dt_burst)
            mean_dt_sample = mean_dt_burst/16
            print("mean_dt_sample = ", mean_dt_sample)
            acc["Frequency"] = math.ceil(1/mean_dt_sample)
            print('final freq = ', acc["Frequency"])
        
            trackNames = idNames('JSON_acc', 3)
            del acc[keys[1]][spatialKeys[0]], acc[keys[1]][spatialKeys[1]], acc[keys[1]][spatialKeys[2]]
            print(list(acc.keys()))
            acc[keys[1]][trackNames[0]] = finalDataX
            acc[keys[1]][trackNames[1]] = finalDataY
            acc[keys[1]][trackNames[2]] = finalDataZ
        
        
            toreturn = acc
    
    
    elif 'patient' in list(contents.keys()):
        
        toret = {}
        
        patientRoot1 = contents['patient']
        
        firstName = patientRoot1['firstName']
        lastName = patientRoot1['lastName'] # HIZ code here        
        patientSex = patientRoot1['data']['sex']
        patientWeight = patientRoot1['data']['weight']
        patientTall = patientRoot1['data']['height']
        patientAge = patientRoot1['data']['age']
        
        dataRoot1 = contents['datasample']
        
        dateData = dataRoot1['valueDate'] # Or 'openingDate' ?
        
        dataRoot2 = dataRoot1['content']['en']['compoundValue']
        
        for item in dataRoot2:
            listItems = item['content']['en']['compoundValue']
            
            if len(listItems) == 4 : # ECG
                dataECG = listItems
                
                for value in dataECG:
                    currentValue = value['content']['en']
                    
                    if 'numberValue' in list(currentValue.keys()):
                        nbrSamples_ECG = currentValue['numberValue']
                        
                    elif 'measureValue' in list(currentValue.keys()) and currentValue['measureValue']['unit'] == 'seconds':
                        duration_ECG = currentValue['measureValue']['value']
                    
                    elif 'stringValue' in list(currentValue.keys()):
                        ecg_Units = currentValue['stringValue']
                        ecg_Units = json.loads(ecg_Units)
                        ecg_factor = ecg_Units['I']['factor']
                        rawECG = currentValue['timeSeries']['samples']
                        

            elif len(listItems) == 5 : # SCG
                dataSCG = listItems
                
                for value in dataSCG:
                    currentValue = value['content']['en']
                    
                    if 'numberValue' in list(currentValue.keys()):
                        nbrSamples_SCG = currentValue['numberValue']
                    
                    elif 'measureValue' in list(currentValue.keys()) and currentValue['measureValue']['unit'] == 'seconds':
                        duration_SCG = currentValue['measureValue']['value']
                    
                    elif 'stringValue' in list(currentValue.keys()): # SCG Lin or Rot
                        SCG_which = currentValue['stringValue']
                        SCG_which = json.loads(SCG_which)
                        
                        if 'X' in list(SCG_which.keys()):
                            rawSCG_lin = currentValue['timeSeries']['samples']
                            
                        elif 'RX' in list(SCG_which.keys()):
                            rawSCG_rot = currentValue['timeSeries']['samples']
                    
        onlyECG = []
        for i in range(len(rawECG)):
            onlyECG.append(rawECG[i][1])
        
        #dataSCG = dataRoot2[2]['content']['en']['compoundValue']
                                    
        #rawSCG_lin = dataSCG[1]['content']['en']['timeSeries']['samples']
        onlySCGx = []
        onlySCGy = []
        onlySCGz = []
        
        for i in range(len(rawSCG_lin)):
            onlySCGx.append(rawSCG_lin[i][1])
            onlySCGy.append(rawSCG_lin[i][2])
            onlySCGz.append(rawSCG_lin[i][3])
        
        #rawSCG_rot = dataSCG[3]['content']['en']['timeSeries']['samples'] 
        onlySCGxR = []
        onlySCGyR = []
        onlySCGzR = []
        
        for i in range(len(rawSCG_rot)):
            onlySCGxR.append(rawSCG_rot[i][1])
            onlySCGyR.append(rawSCG_rot[i][2])
            onlySCGzR.append(rawSCG_rot[i][3])
            
        #nbrSamples_SCG = dataSCG[0]['content']['en']['numberValue'] # (6104samples)
        #duration_SCG = dataSCG[2]['content']['en']['measureValue']['value'] # 60s
        
        toret['Name'] = firstName
        toret['Surname'] = lastName
        toret['Date'] = dateData
        
        toret['Sex[m/f]'] = patientSex
        toret['Weight[kg]'] = patientWeight
        toret['Height[cm]'] = patientTall
        toret['Age[y]'] = patientAge
        
        ecgDict = {}
        ecgDict['sfreq[Hz]'] = nbrSamples_ECG/duration_ECG
        ecgDict['numSamples'] = nbrSamples_ECG
        ecgDict['duration[s]'] = duration_ECG
        ecgDict['ECG[uV]'] = np.array(onlyECG)
        ecgDict['ECGfactor'] = ecg_factor
        toret['ECG'] = ecgDict
        
        scgDict = {}
        scgDict['sfreq[Hz]'] = nbrSamples_SCG/duration_SCG
        scgDict['numSamples'] = nbrSamples_SCG
        scgDict['duration[s]'] = duration_SCG
        scgDict['scgLin[m/s^2]'] = {"x": np.array(onlySCGx) , "y": np.array(onlySCGy), "z": np.array(onlySCGz)}
        scgDict['scgRot[deg/s]'] = {"x": np.array(onlySCGxR) , "y": np.array(onlySCGyR), "z": np.array(onlySCGzR)}
        toret['SCG'] = scgDict
        
        toreturn = toret
    
    return toreturn


#%% 1f2bp PureTXT function
        
def pureTXT_1f2bp(pathfile):
    
    charact = ['PATIENT NAME', 'PATIENT MRN', 'WITTCASE NUMBER', 'SAMPLE NUMBER', 'TIME RESOLUTION', 'mmHg RESOLUTION', 'DATA UNITS', 'ECG UNITS', 'ECG RESOLUTION']
    counts = [0, 1, 2, 3, 4, 5, 9, 10, 11]

    data={}
    countline = 0
    flagchar = 0
    nbrdatapoints = 0
    
    with open(pathfile) as f:
        contents = f.readlines()
    del f
        
        # The file is read LINE BY LINE
    for line in contents: # 
            
        # Extraction of information on the top informations
        if countline in counts :

            tocompare = charact[flagchar]+':'
            line = line.replace(tocompare, '')
                
            # Removes spaces to keep only useful information
            line = line.split(' ')
            while '' in line :
                line.remove('')
                
            if len(line) > 1 :
                data[charact[flagchar]] = line
            else :
                data[charact[flagchar]] = line[0]
            flagchar = flagchar +1
                
            
        # Extraction of the signals information
        if countline == 14:
                
            line = line.split(' ')
            while '' in line :
                line.remove('')
            while '\t' in line :
                line.remove('\t')
                
            # Add the different data keys in the dictionary
            for i in range(len(line)) :
                if '\t' in line[i]:
                    line[i] = line[i].replace('\t', '')
                if '\n' in line[i]:
                    line[i] = line[i].replace('\n', '')
                
            data['data'] = dict.fromkeys(line, [])
        
        
        # Counts the number of data points registered
        if countline>14 :
            nbrdatapoints = nbrdatapoints +1
        
        countline = countline+1
        
    # Remove the useless variables        
    del flagchar, line, countline, counts, charact, tocompare
        

    # Read the actual data
    contents = contents[15:]
    #print("Number of data points : ", nbrdatapoints)
    for i in range(0, len(contents)):
            
        contents[i] = contents[i].split(' ')
            
        while '' in contents[i] :
            contents[i].remove('')
        while '\t' in contents[i] :
            contents[i].remove('\t')
            
        # Clean the display of the data
        for j in range(len(contents[i])) :
            if '\t' in contents[i][j]:
                contents[i][j] = contents[i][j].replace('\t', '')
            if '\n' in contents[i][j]:
                contents[i][j] = contents[i][j].replace('\n', '')
            if '(D)' in contents[i][j] :
                contents[i][j] = contents[i][j].replace('(D)', '')
            if '(S)' in contents[i][j] :
                contents[i][j] = contents[i][j].replace('(S)', '')
            if '(R)' in contents[i][j] :
                contents[i][j] = contents[i][j].replace('(R)', '')

    # Keep only the rows with complete data
    contTokeep = []
    for i in range(len(contents)):
        #print(contents[i])
        if len(contents[i]) == len(data['data']):
            contTokeep.append(contents[i])
    
    
    # Transposition of the matrix :
    finalCont = [list(ele) for ele in list(zip(*contTokeep))]

    # Convert all the values to float and replace the N/A by the mean local value.
    for i in range(len(finalCont)):
        for k in range(len(finalCont[i])) :
            if finalCont[i][k] != 'N/A':
                finalCont[i][k] = float(finalCont[i][k])
            else :
                if k == 0 :
                    finalCont[i][k] = finalCont[i][k+1]
                elif k == len(finalCont[i])-1:
                    finalCont[i][k] = finalCont[i][k-1]
                else :
                    finalCont[i][k] = np.mean([finalCont[i][k+1], finalCont[i][k-1]])

            
    
    
    # Put results inside the dictionary "data" to return
    datakeys = list(data['data'].keys())
    #print("datakeys before = ", datakeys)
    for key in datakeys:
        del data['data'][key]
    #print("datakeys after = ", list(data['data'].keys()))
    
    trackNames = idNames('TXT', len(datakeys))
    for k in range(len(datakeys)) :
        data['data'][trackNames[k]] = np.array(finalCont[k])
    
             
    del i,k,datakeys,contTokeep,contents, nbrdatapoints, j
    
    #%% AJOUTER NOM DES TRACKS « TXT_tracks »
    
    return data


            
#%% 2f2bp PureTXT function

def pureTXT_2f2bp(path1, path2):
    
    data1 = pureTXT_1f2bp(path1)
    data2 = pureTXT_1f2bp(path2)

    data1['data1'] = data1['data']
    data1.pop('data')
    data1['data2'] = data2['data']

    del data2
    
    return data1



#%% Automatic file recognition + opening

def openFile(window, txt : str, txt2 : str = None):

    if txt2 != None :
        txt = txt+'.'+txt2

    data = {}
    txt = txt.split('.')
    #print(txt)
    if txt[-1] == 'mat':
        dataType = "MatLab"
        data = openMatLabdigere(txt[0]+'.'+txt[len(txt)-1])
        
    elif txt[-1] == 'json':
        dataType = "JSON" 
        prefix = txt[0]
        if len(txt) == 3 :
            prefix = txt[0]+"."+txt[1]
        data = openJSON(window, prefix+'.'+txt[-1])
        
    elif txt[-1] == 'txt' and len(txt)==2:
        dataType = "TXT_1f"
        data = pureTXT_1f2bp(txt[0]+'.'+txt[-1]) 
        
    elif txt[-1] == 'txt' and len(txt)>2:
        #dataType = "TXT_2f"
        dataType = "TXT_1f"
        data = pureTXT_1f2bp(txt[0]+'.'+txt[1]+'.'+txt[-1]) 
        #data = pureTXT_2f2bp(txt[0]+'.'+txt[1]+'.'+txt[2], txt[3]+'.'+txt[4]+'.'+txt[5])
        
    else : print("Data type not handled")

    #display_plt(data, dataType)
    
    return data, dataType


#%% get data for display

def getDataToDisplay(dataDict : dict, typeData : str):
    allData = []
    
    def commonPart(y, step):
        stop = len(y)*step
        x = np.arange(0, stop, step)
        
        while len(x) < len(y):
            x.append(x[len(x)-1]+1)
        while len(x) > len(y):
            x = np.delete(x, -1)
        
        return x
    
    # MATLAB/LABCHART :
    if typeData == "MatLab":
        
        if "ECG" in list(dataDict.keys()):
            for title in list(dataDict.keys()):
                y = dataDict[title]['signal']
                step = 1/dataDict[title]['samplerate']
            
                x = commonPart(y, step)
                
                
                """
                if title == "ECG":
                    print("Moyenne = ", np.mean(y))
                    print("Minimum = ", np.min(y))
                    print("Maximum = ", np.max(y))
                    print("Différence max-moy = ", np.max(y)-np.mean(y))
                    print("Différence moy-min = ", np.mean(y)-np.min(y))
                    if abs(np.max(y)-np.mean(y)) > abs(np.mean(y)-np.min(y)):
                        print("ECG non inversé", type(y))
                    else : 
                        print("ECG inversé")
                """
                
                
                graphMLd = SingleGraph(x, y, title, dataDict[title]['samplerate'], step)
                allData.append(graphMLd)
        else:        
            for measure in list(dataDict.keys()):
                for title in list(dataDict[measure].keys()):
                    
                    """
                    if title == "ECG":
                        print("Moyenne = ", np.mean(y))
                        print("Minimum = ", np.min(y))
                        print("Maximum = ", np.max(y))
                        print("Différence max-moy = ", np.max(y)-np.mean(y))
                        print("Différence moy-min = ", np.mean(y)-np.min(y))
                        if abs(np.max(y)-np.mean(y)) > abs(np.mean(y)-np.min(y)):
                            print("ECG non inversé", type(y))
                        else : 
                            print("ECG inversé")
                    """
                    
                    y = dataDict[measure][title]['signal']
                    step = 1/dataDict[measure][title]['samplerate']
            
                    x = commonPart(y, step)
                    graphMLd = SingleGraph(x, y, title, dataDict[measure][title]['samplerate'], step)
                    allData.append(graphMLd)
            
        
    elif typeData == "JSON":            
        
        done = False
        if 'meta' in list(dataDict.keys()):
            # First the ECG signal
            try:
                y = np.array(dataDict['ECG']['ECG[uV]'])*dataDict['ECG']['ECGfactor']
                sfreq = dataDict['ECG']['sfreq[Hz]']
                step = 1/sfreq
                x = commonPart(y, step)
                
                graphJSON = SingleGraph(x, y, "ECG", sfreq, step)
                allData.append(graphJSON)
            except:
                DialogPopup("Error", "No ECG found").exec()

            
            # Then the SCG signal
            sfreq = dataDict['SCG']['sfreq[Hz]']
            step = 1/sfreq
            
            interestKeys = ['scgLin[m/s^2]', 'scgRot[deg/s]']
            
            axes = ['x', 'y', 'z']
            for key in interestKeys:
                for ax in axes :
                    y = np.array(dataDict['SCG'][key][ax])
                    x = commonPart(y, step)
                    
                    graphJSON = SingleGraph(x, y, ax+'_'+key, sfreq, step)
                    allData.append(graphJSON)
            
            
            done = True
        
        elif 'CursorsTimes' and "metaData" and "Tracks" in list(dataDict.keys()):
            print("CardiOnco file")
            allDataKeys = list(dataDict["Tracks"].keys())
            
            for key in allDataKeys:
                y = np.array(dataDict["Tracks"][key]["amplitude"])
                x = np.array(dataDict["Tracks"][key]["time"])
                sfreq = dataDict["Tracks"][key]["fs"]
                step = dataDict["Tracks"][key]["step"]
                graphCardiOnco = SingleGraph(x, y, key, sfreq, step)
                allData.append(graphCardiOnco)
            
        
        
        elif 'Sex[m/f]' in list(dataDict.keys()) :
            
            # First the ECG signal
            y = dataDict['ECG']['ECG[uV]']*dataDict['ECG']['ECGfactor']
            sfreq = dataDict['ECG']['sfreq[Hz]']
            step = 1/sfreq
            x = commonPart(y, step)
            
            
            """
            if title == "ECG":
                print("Moyenne = ", np.mean(y))
                print("Minimum = ", np.min(y))
                print("Maximum = ", np.max(y))
                print("Différence max-moy = ", np.max(y)-np.mean(y))
                print("Différence moy-min = ", np.mean(y)-np.min(y))
                if abs(np.max(y)-np.mean(y)) > abs(np.mean(y)-np.min(y)):
                    print("ECG non inversé", type(y))
                else : 
                    print("ECG inversé")
            """
            
            graphJSON = SingleGraph(x, y, "ECG", sfreq, step)
            allData.append(graphJSON)
            
            # Then the SCG signal
            sfreq = dataDict['SCG']['sfreq[Hz]']
            step = 1/sfreq
            
            interestKeys = ['scgLin[m/s^2]', 'scgRot[deg/s]']
            
            axes = ['x', 'y', 'z']
            for key in interestKeys:
                for ax in axes :
                    y = dataDict['SCG'][key][ax]
                    x = commonPart(y, step)
                    
                    graphJSON = SingleGraph(x, y, ax+'_'+key, sfreq, step)
                    allData.append(graphJSON)
         
        
        elif 'ArrayAcc' in list(dataDict.keys()) :

            spatialKeys = list(dataDict['ArrayAcc'].keys())
            for spatData in spatialKeys:
                y = dataDict['ArrayAcc'][spatData]
                sfreq = dataDict['Frequency']
                step = 1/sfreq
                x = commonPart(y, step)
               
                
                graphJSON = SingleGraph(x, y, "Acc_"+spatData, sfreq, step)
                allData.append(graphJSON)
        
        elif 'ECG' in list(dataDict.keys()) and done == False :
            if len(list(dataDict.keys())) == 1 :
                y = np.array(dataDict['ECG']['ECG[uV]'])*dataDict['ECG']['ECGfactor']
                sfreq = dataDict['ECG']['sfreq[Hz]']
                step = 1/sfreq
                x = commonPart(y, step)
                
                graphJSON = SingleGraph(x, y, "ECG", sfreq, step)
                allData.append(graphJSON)
            else :
                y = dataDict["ECG"]
                sfreq = dataDict["Frequency"]
                step = 1/sfreq
                
                x = commonPart(y, step)
                
                
                """
                if title == "ECG":
                    print("Moyenne = ", np.mean(y))
                    print("Minimum = ", np.min(y))
                    print("Maximum = ", np.max(y))
                    print("Différence max-moy = ", np.max(y)-np.mean(y))
                    print("Différence moy-min = ", np.mean(y)-np.min(y))
                    if abs(np.max(y)-np.mean(y)) > abs(np.mean(y)-np.min(y)):
                        print("ECG non inversé", type(y))
                    else : 
                        print("ECG inversé")
                """
                
                graphJSON = SingleGraph(x, y, "ECG", sfreq, step)
                allData.append(graphJSON)
        
            
    elif typeData == "TXT_1f" :
        
        for title in list(dataDict['data'].keys()) :
            y = dataDict['data'][title]
            step = int(dataDict['TIME RESOLUTION'][0])*0.001
            stop = len(y)*step
            sfreq = 1/step
            x = commonPart(y, step)
            
            graph_1TXT = SingleGraph(x, y, title, sfreq, step)
            allData.append(graph_1TXT)
        
        
    elif typeData == "TXT_2f" :
        
        whichDatas = ['data1', 'data2']
        for data in whichDatas :
            for title in list(dataDict[data].keys()) :
                y = dataDict[data][title]
                step = int(dataDict['TIME RESOLUTION'][0])*0.001
                stop = len(y)*step
                x = np.arange(0, stop, step)
                
                while len(x) < len(y):
                    x.append(x[len(x)-1]+1)
                while len(x) > len(y):
                    x = np.delete(x, -1)
            
                graph_2TXT = SingleGraph(x, y, data+' : '+title, 0, step)
                allData.append(graph_2TXT)
            
    return allData
