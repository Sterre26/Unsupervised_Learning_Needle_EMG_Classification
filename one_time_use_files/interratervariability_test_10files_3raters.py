# -*- coding: utf-8 -*-
"""
Created on Wed May 19 10:42:49 2021

@author: debor
"""

# Visualisation imports
from tkinter.filedialog import askopenfilename
from os.path import dirname

import pandas as pd
import krippendorff
import statsmodels.stats.inter_rater
import numpy as np

#mypath='L:/basic/divd/knf/Onderzoek_studenten/deborahhubers/ai_needle_emg/Data'
filepath = askopenfilename(
            filetypes=[("Wav Files", "*.wav"), ("All Files", "*.*")])
# TODO bedenk iets slimmers voor deze manier van datafiles kiezen
datafile =['0005_R_tib_ant_1_1', '0010_R_vas_med_1_1', '0012_R_tib_ant_3_1', '0017_R_glu_med_2_1', '0054_L_vas_med_1_1', '0079_R_tib_ant_1_1', '0094_L_delt_1_1', '0107_R_gas_lat_1_1', '0136_R_ext_hal_1_1']

mypath = dirname(filepath)

WVPtotal = []
LWtotal = []
DHtotal = []
CVtotal = []

for n in range(len(datafile)):
    annotationWVP= str(mypath+'/'+datafile[n]+'WVP.csv')  
    annotationCV= mypath+'/'+datafile[n]+'CV.csv'
    annotationDH = mypath+'/'+datafile[n]+'DH.csv'
    annotationLW = mypath+'/'+datafile[n]+'LW.csv'

    #Load annotations
    WVP = np.array(pd.read_csv(annotationWVP)).flatten()
    DH = np.array(pd.read_csv(annotationDH)).flatten()
    LW = np.array(pd.read_csv(annotationLW)).flatten()
    CV = np.array(pd.read_csv(annotationCV)).flatten()

    WVPtotal = np.append(WVPtotal , WVP, axis=0)
    DHtotal= np.append(DHtotal , DH, axis=0)
    LWtotal= np.append(LWtotal , LW, axis=0)
    CVtotal = np.append(CVtotal, CV, axis=0)

# Make annotations based on 2s, sliding window 0.1s
def makewindows(data):
    a=int(len(data)/4410)
    newdata=[]
    for n in range(a):
        window = data[n*4410+1:n*4410+88400]
        number_of_rest_annotations = window[window == 1]
        number_of_contraction_annotations = window[window == 2]
        number_of_needle_annotations = window[window == 3]
        number_of_no_annotations = window[window == 0]

        rest_annotation = len(number_of_rest_annotations) / 88400
        contraction_annotation = len(number_of_contraction_annotations) / 88400
        needle_annotation = len(number_of_needle_annotations) / 88400
        no_annotation = len(number_of_no_annotations) / 88400
        # When 50% of the data is something: annotation is something, otherwise: no annotation
        if no_annotation > 0.5:
            anno = 0
        elif rest_annotation > 0.5:
            anno =1 
        elif contraction_annotation > 0.5:
            anno = 2
        elif needle_annotation > 0.5:
            anno = 3
        else:
            anno = 0
        newdata = np.append(newdata,anno)    
    not_annotated = data[data ==0]
    percentage_annotated = 1-len(not_annotated)/len(data)
    
    return newdata, percentage_annotated

WVPtotal, WVPpercentage = makewindows(WVPtotal)
DHtotal, DHpercentage = makewindows(DHtotal)
LWtotal, LWpercentage = makewindows(LWtotal)
CVtotal, CVpercentage = makewindows(CVtotal)

LW=LWtotal.reshape(-1,1)
WVP=WVPtotal.reshape(-1,1)
DH=DHtotal.reshape(-1,1)
CV = CVtotal.reshape(-1,1)

Data=np.append(DH,WVP, axis=1)
Data=np.append(Data,LW,axis=1)
Data=np.append(Data, CV, axis=1)
Data= np.where(Data>3,3,Data) #take non-analysable and needle-movement as one group

# Create frames to test individual raters with neurologist
DataWVP=np.append(LW,WVP, axis=1)
DataWVP= np.where(DataWVP>3,3,DataWVP) #take non-analysable and needle-movement as one group

DataCV=np.append(CV,LW, axis = 1)
DataCV = np.where(DataCV>3,3,DataCV)

DataDH=np.append(DH,LW, axis=1)
DataDH= np.where(DataDH>3,3,DataDH) #take non-analysable and needle-movement as one group

DataDHWVP=np.append(DH,WVP, axis=1)
DataDHWVP= np.where(DataDHWVP>3,3,DataDHWVP) #take non-analysable and needle-movement as one group
# Remove all dataparts that are not annotated by one of the annotaters  
Data = Data[~np.any(Data == 0, axis=1)]
DataWVP = DataWVP[~np.any(DataWVP == 0, axis=1)]
DataCV = DataCV[~np.any(DataCV==0, axis=1)]
DataDH = DataDH[~np.any(DataDH == 0, axis=1)]
DataDHWVP = DataDHWVP[~np.any(DataDHWVP == 0, axis=1)]

#calculate all fleiss kappa's
fleissdata=statsmodels.stats.inter_rater.aggregate_raters(Data, n_cat=None)
fleiss=statsmodels.stats.inter_rater.fleiss_kappa(fleissdata[0], method='fleiss')
print("Fleiss all raters: " + str(fleiss))

fleissdata=statsmodels.stats.inter_rater.aggregate_raters(DataWVP, n_cat=None)
fleiss=statsmodels.stats.inter_rater.fleiss_kappa(fleissdata[0], method='fleiss')
print("Fleiss WVP, LW: " + str(fleiss))

fleissdata=statsmodels.stats.inter_rater.aggregate_raters(DataCV, n_cat=None)
fleiss=statsmodels.stats.inter_rater.fleiss_kappa(fleissdata[0], method='fleiss')
print("Fleiss CV, LW: " + str(fleiss))

fleissdata=statsmodels.stats.inter_rater.aggregate_raters(DataDH, n_cat=None)
fleiss=statsmodels.stats.inter_rater.fleiss_kappa(fleissdata[0], method='fleiss')
print("Fleiss DH, LW: " + str(fleiss))

fleissdata=statsmodels.stats.inter_rater.aggregate_raters(DataDHWVP, n_cat=None)
fleiss=statsmodels.stats.inter_rater.fleiss_kappa(fleissdata[0], method='fleiss')
print("Fleiss DH, WVP: " + str(fleiss))

#Data needs to be transposed because of the krippendorff format
Data=Data.transpose()
DataWVP=DataWVP.transpose()
DataDH=DataDH.transpose()
DataDHWVP=DataDHWVP.transpose()

#Calculate all krippendorff alpha
krippendorf = krippendorff.alpha(Data, level_of_measurement= 'nominal')
print("Krippendorff all raters: " + str(krippendorf))
krippendorf = krippendorff.alpha(DataWVP, level_of_measurement= 'nominal')
print("Krippendorff WVP, LW: " + str(krippendorf))
krippendorf = krippendorff.alpha(DataDH, level_of_measurement= 'nominal')
print("Krippendorff DH, LW: "  + str(krippendorf))
krippendorf = krippendorff.alpha(DataDHWVP, level_of_measurement= 'nominal')
print("Krippendorff DH, WVP: "  + str(krippendorf))

# Print percentage of filled in annotations:
print("Percentage WVP: " + str(WVPpercentage))
print("Percentage LW: " + str(LWpercentage))
print("Percentage DH: " + str(DHpercentage))
print("Percentage CV: " +str(CVpercentage))