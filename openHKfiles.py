# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 10:30:38 2023

@author: Tatiana Dehon
"""
import json
import numpy as np

pathfile = "examplefiles\\json\\icare_florine\\0d227f38-053b-42fb-b7a4-25d16364c329.json"

with open(pathfile) as f:
    contents = json.load(f)

myContents = contents['content']['en']['compoundValue']

for key in range(len(myContents)):
    typeData = myContents[key]['labels'][0]['code']
    print("Type of data : ", typeData)
    
    root = myContents[key]['content']['en']
    
    if typeData == "SUBJECT-SEX":
        patientSex = root['stringValue']
        print("\tData : ", patientSex)
        
    if typeData == "SUBJECT-WEIGHT":
        patientWeight = root['measureValue']['value']
        weightUnit = root['measureValue']['unit']
        print("\tData : ", patientWeight, weightUnit)
    
    if typeData == "SUBJECT-HEIGHT":
        patientTall = root['measureValue']['value']
        tallUnit = root['measureValue']['unit']
        print("\tData : ", patientTall, tallUnit)
        
    if typeData == "SUBJECT-AGE":
        patientAge = root['measureValue']['value']
        ageUnit = root['measureValue']['unit']
        print("\tData : ", patientAge, ageUnit)
        
        
    if typeData == "SIGNAL-SCG":
        totalSCG = root['compoundValue']
        for item in totalSCG:
            
            code = item['labels'][0]['code']
            subroot = item['content']['en']
            
            if code == "SIGNAL-SAMPLING-RATE":
                sFreqSCG = subroot['measureValue']['value']
                sfreqscgUnit = subroot['measureValue']['unit']
                print("\tsFreqSCG : ", sFreqSCG, sfreqscgUnit)
            
            if code == "SIGNAL-LENGTH":
                lenSCG = subroot['numberValue']
                print("\tLength of SCG signal : ", lenSCG)
            
            if code == "SIGNAL-DURATION":
                dureeSCG = subroot['measureValue']['value']
                dureescgUnit = subroot['measureValue']['unit']
                print("\tTime of SCG signal : ", dureeSCG, dureescgUnit)
                
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
                
                
                
                
                
                txyzR_Units = subroot['stringValue']
                txyzR_Units = json.loads(txyzR_Units)
                txyzR_Units = [txyzR_Units['time']['units'], txyzR_Units['RX']['units'], txyzR_Units['RY']['units'], txyzR_Units['RZ']['units']]
                print("\ttxyzR_Units :", txyzR_Units)
                #scgDict['scgRot'] = {"x": onlySCGx , "y": onlySCGy, "z": onlySCGz}

                
            if code == "SIGNAL-SCG-LINEAR":
                # Pure useful data for computation
                txyz_Sig = subroot['timeSeries']['samples']
                txyz_Units = subroot['stringValue']
                txyz_Units = json.loads(txyz_Units)
                txyz_Units = [txyz_Units['time']['units'], txyz_Units['X']['units'], txyz_Units['Y']['units'], txyz_Units['Z']['units']]
                print("\ttxyz_Units :", txyz_Units)
        
        
    if typeData == "SIGNAL-ECG":
        totalECG = root['compoundValue']
        sFreqECG = 0
        for item in totalECG:
            
            code = item['labels'][0]['code']
            subroot = item['content']['en']
            
            if code == "SIGNAL-SAMPLING-RATE":
                sFreqECG = subroot['measureValue']['value']
                sfreqecgUnit = subroot['measureValue']['unit']
                print("\tsFreqECG : ", sFreqECG, sfreqecgUnit)
            
            if code == "SIGNAL-LENGTH":
                lenECG = subroot['numberValue']
                print("\tLength of ECG signal : ", lenECG)
            
            if code == "SIGNAL-DURATION":
                
                dureeECG = subroot['measureValue']['value']
                dureeecgUnit = subroot['measureValue']['unit']
                print("\tTime of ECG signal : ", dureeECG, dureeecgUnit)
        
        
            if code == "SIGNAL-ECG-LEADS":
                # Pure useful data for computation
                ecg_Sig = subroot['timeSeries']['samples']
                
                onlyECG = []
                for i in range(len(ecg_Sig)):
                    onlyECG.append(ecg_Sig[i][1])
                
                ecg_Units = subroot['stringValue']
                ecg_Units = json.loads(ecg_Units)
                ecg_factor = ecg_Units['I']['factor']
                ecg_Units = [ecg_Units['time']['units'], ecg_Units['I']['units']]
                print("\tecg_Units :", ecg_Units, " ; ECG factor :", ecg_factor)
                
