# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 13:30:02 2023
@author: Tatiana Dehon

 ========= Toolbox for files opening =========
 
"""

#from simpleDisplay import display_plt, displ_single_graph
from SingleGraph import SingleGraph
from simpleDisplay import display_plt, detect_qrs
import math

import os, gc
import json
import numpy as np
import scipy.io as sc

# Where are we in the system ?
cwd = os.path.dirname(__file__)

with open("allNames.json") as f:
    allNames = json.load(f)
#print("Current work directory : ", cwd)

#%% Funcion to call for having the right channels names

def idNames(typeFile : str, nbChannels : int):
    if typeFile in list(allNames.keys()):
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

def openJSON(pathfile : str):
    with open(pathfile) as f:
        contents = json.load(f)
    #print(list(contents.keys()))
    
    if 'data' in list(contents.keys()):
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
    
    """
    elif 'healthcareElementIds' in list(contents.keys()):
        
        toret = {}
        
        myContents = contents['content']['en']['compoundValue']
        
        for key in range(len(myContents)):
            typeData = myContents[key]['labels'][0]['code']
            print("Type of data : ", typeData)
            
            root = myContents[key]['content']['en']
            
            if typeData == "SUBJECT-SEX":
                patientSex = root['stringValue']
                print("\tData : ", patientSex)
                toret['Sex[m/f]'] = patientSex
                
            if typeData == "SUBJECT-WEIGHT":
                patientWeight = root['measureValue']['value']
                #weightUnit = root['measureValue']['unit']
                print("\tData : ", patientWeight)#, weightUnit)
                toret['Weight[kg]'] = patientWeight
            
            if typeData == "SUBJECT-HEIGHT":
                patientTall = root['measureValue']['value']
                #tallUnit = root['measureValue']['unit']
                print("\tData : ", patientTall)#, tallUnit)
                toret['Height[m]'] = patientTall
                
            if typeData == "SUBJECT-AGE":
                patientAge = root['measureValue']['value']
                #ageUnit = root['measureValue']['unit']
                print("\tData : ", patientAge)#, ageUnit)
                toret['Age[y]'] = patientAge
                
                
            if typeData == "SIGNAL-SCG":
                totalSCG = root['compoundValue']
                scgDict = {}
                for item in totalSCG:
                    
                    code = item['labels'][0]['code']
                    subroot = item['content']['en']
                    
                    if code == "SIGNAL-SAMPLING-RATE":
                        sFreqSCG = subroot['measureValue']['value']
                        #sfreqscgUnit = subroot['measureValue']['unit']
                        print("\tsFreqSCG : ", sFreqSCG)#, sfreqscgUnit)
                        scgDict['sfreq[Hz]'] = sFreqSCG
                        
                    
                    if code == "SIGNAL-LENGTH":
                        lenSCG = subroot['numberValue']
                        print("\tLength of SCG signal : ", lenSCG)
                        scgDict['numSamples'] = lenSCG
                    
                    if code == "SIGNAL-DURATION":
                        dureeSCG = subroot['measureValue']['value']
                        #dureescgUnit = subroot['measureValue']['unit']
                        print("\tTime of SCG signal : ", dureeSCG)#, dureescgUnit)
                        scgDict['duration[s]'] = dureeSCG
                        
                    if code == "SIGNAL-SCG-ROTATIONAL":
                        # Pure useful data for computation
                        txyzR_Sig = subroot['timeSeries']['samples']
                        
                        onlySCGx = []
                        onlySCGy = []
                        onlySCGz = []
                        
                        for i in range(len(txyzR_Sig)):
                            onlySCGx.append(txyzR_Sig[i][1])
                            onlySCGy.append(txyzR_Sig[i][2])
                            onlySCGz.append(txyzR_Sig[i][3])
                        
                        #txyzR_Units = subroot['stringValue']
                        #txyzR_Units = json.loads(txyzR_Units)
                        #txyzR_Units = [txyzR_Units['time']['units'], txyzR_Units['RX']['units'], txyzR_Units['RY']['units'], txyzR_Units['RZ']['units']]
                        #print("\ttxyzR_Units :", txyzR_Units)
                        scgDict['scgRot[deg/s]'] = {"x": np.array(onlySCGx) , "y": np.array(onlySCGy), "z": np.array(onlySCGz)}
                        
                    if code == "SIGNAL-SCG-LINEAR":
                        # Pure useful data for computation
                        txyz_Sig = subroot['timeSeries']['samples']
                        
                        onlySCGx = []
                        onlySCGy = []
                        onlySCGz = []
                        
                        for i in range(len(txyz_Sig)):
                            onlySCGx.append(txyz_Sig[i][1])
                            onlySCGy.append(txyz_Sig[i][2])
                            onlySCGz.append(txyz_Sig[i][3])
                        
                        #txyz_Units = subroot['stringValue']
                        #txyz_Units = json.loads(txyz_Units)
                        #txyz_Units = [txyz_Units['time']['units'], txyz_Units['X']['units'], txyz_Units['Y']['units'], txyz_Units['Z']['units']]
                        #print("\ttxyz_Units :", txyz_Units)
                        scgDict['scgLin[m/s^2]'] = {"x": np.array(onlySCGx) , "y": np.array(onlySCGy), "z": np.array(onlySCGz)}

                    
                toret['SCG'] = scgDict
                
                
            if typeData == "SIGNAL-ECG":
                totalECG = root['compoundValue']
                ecgDict = {}
                sFreqECG = 0
                for item in totalECG:
                    
                    code = item['labels'][0]['code']
                    subroot = item['content']['en']
                    
                    if code == "SIGNAL-SAMPLING-RATE":
                        sFreqECG = subroot['measureValue']['value']
                        #sfreqecgUnit = subroot['measureValue']['unit']
                        print("\tsFreqECG : ", sFreqECG)#, sfreqecgUnit)
                        ecgDict['sfreq[Hz]'] = sFreqECG
                    
                    if code == "SIGNAL-LENGTH":
                        lenECG = subroot['numberValue']
                        print("\tLength of ECG signal : ", lenECG)
                        ecgDict['numSamples'] = lenECG
                    
                    if code == "SIGNAL-DURATION":
                        dureeECG = subroot['measureValue']['value']
                        #dureeecgUnit = subroot['measureValue']['unit']
                        print("\tTime of ECG signal : ", dureeECG)#, dureeecgUnit)
                        ecgDict['duration[s]'] = dureeECG
                    
                    if code == "SIGNAL-ECG-LEADS":
                        # Pure useful data for computation
                        ecg_Sig = subroot['timeSeries']['samples']
                        
                        onlyECG = []
                        for i in range(len(ecg_Sig)):
                            onlyECG.append(ecg_Sig[i][1])
                        
                        ecg_Units = subroot['stringValue']
                        ecg_Units = json.loads(ecg_Units)
                        ecg_factor = ecg_Units['I']['factor']
                        #ecg_Units = [ecg_Units['time']['units'], ecg_Units['I']['units']]
                        print("\tECG factor :", ecg_factor)#, "ecg_Units :", ecg_Units)
                        ecgDict['ECG[uV]'] = np.array(onlyECG)
                        ecgDict['ECGfactor'] = ecg_factor
        
                toret['ECG'] = ecgDict
        
        toreturn = toret
        
        # Correction of the sfreq registered
        realDuration = toreturn["ECG"]["duration[s]"]
        real_sfreq_ecg = toreturn["ECG"]["numSamples"]/realDuration
        toreturn["ECG"]["sfreq[Hz]"] = real_sfreq_ecg
        
        
        real_sfreq_scg = toreturn["SCG"]["numSamples"]*real_sfreq_ecg/toreturn["ECG"]["numSamples"]
        toreturn["SCG"]["sfreq[Hz]"] = real_sfreq_scg
    """  
    
    if 'patient' in list(contents.keys()):
        
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

def openFile(txt : str, txt2 : str = None):

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
        data = openJSON(txt[0]+'.'+txt[-1])
        
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
        
        if 'Sex[m/f]' in list(dataDict.keys()) :
            
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
        
        elif 'ECG' in list(dataDict.keys()) :
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
