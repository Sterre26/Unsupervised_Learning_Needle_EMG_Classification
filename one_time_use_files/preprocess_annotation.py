# Python (default/external) imports
import numpy as np
import pandas as pd
import os
from os.path import dirname, isfile
from collections import Counter
from tqdm import tqdm

""" 
This file creates a majority vote annotation file (.csv file) from multiple experts. 
"""

filepath_multiple_annotations = '/Users/Sterre/files/WAV/annotation_signaltype2/' # folder with both .wav files and .csv files from multiple experts
filepath_majority_vote = filepath_multiple_annotations + 'majority_vote/' # folder to store the majority vote (.csv) files to
if not os.path.exists(filepath_majority_vote):
    os.makedirs(filepath_majority_vote)
    print("Folder to store majority-vote data is created")

mypath = dirname(filepath_multiple_annotations)
print(mypath)
annotated_file_list = \
    [mypath +'/' + f for f in os.listdir(mypath) if ".wav" in f]

def needleplusnonanalysable(data):
    data= np.where(data>3,3,data)
    return data

# TODO: annotatie proces automatiseren? Dat je aan het begin een dict kan opgeven met de initials van de annoteurs 

for annotated_file in tqdm(annotated_file_list):
    datafile = str(annotated_file).split("/")[-1].split(".wav")[0]
    mypath = str(annotated_file).split(datafile)[0]

    annotationWVP = mypath+'/'+datafile+'WVP.csv'
    annotationCV = mypath+'/'+datafile+'CV.csv'
    annotationDH = mypath+'/'+datafile+'DH.csv'
    annotationLW = mypath+'/'+datafile+'LW.csv'
    if 'WVP' in locals():
        del WVP
    if 'DH' in locals():
        del DH
    if 'CV' in locals():
        del CV
    if 'LW' in locals():
        del LW

    if 'ex1' in locals():
        del ex1
    if 'ex2' in locals():
        del ex2
    if 'ex3' in locals():
        del ex3
    if 'ex4' in locals():
        del ex4
    ##Load annotations if exist
    if isfile(annotationWVP):
        WVP = np.array(pd.read_csv(annotationWVP)).flatten()
    else:
        pass
    if isfile(annotationCV):
        CV = np.array(pd.read_csv(annotationCV)).flatten()
    else: pass

    if isfile(annotationDH):
        DH = np.array(pd.read_csv(annotationDH)).flatten()
    else:
        pass

    if isfile(annotationLW):
        LW = np.array(pd.read_csv(annotationLW)).flatten()
    else:
        pass
    #auto = np.array(pd.read_csv(annotationauto)).flatten()
    ## Looking for existing annotations
    if 'WVP' in locals():
        ex1 = WVP
        if 'CV' in locals():
            ex2 = CV
            if 'DH' in locals():
                ex3 = DH
                if 'LW' in locals():
                    ex4 = LW
                    print('Four annotations for this file:' + datafile)
                else:
                    print('Three annotations for this file:' + datafile)  
            elif 'LW' in locals():
                ex3 = LW
                print('Three annotations for this file:' + datafile)
            else: 
                print('Two annotations for this file:' + datafile)
        elif 'DH' in locals():
            ex2 = DH
            if 'LW' in locals():
                ex3 = LW
                print('Three annotations for this file:' + datafile)
            else: 
                print('Two annotations for this file:' + datafile)
        elif 'LW' in locals():
            ex2 = LW
            print('Two annotations for this file:' + datafile)
        else: 
            print('One annotations for this file:' + datafile)

    elif 'CV' in locals():
        ex1 = CV
        if 'DH' in locals():
            ex2 = DH
            if 'LW' in locals():
                ex3 = LW
                print('Three annotations for this file:' + datafile)
            else:
                print('Two annotations for this file:' + datafile)  
        elif 'LW' in locals():
            ex2 = LW
            print('Two annotations for this file:' + datafile)
        else: 
            print('One annotations for this file:' + datafile)
    elif 'DH' in locals():
        ex1 = DH
        if 'LW' in locals():
            ex2 = LW
            print('Two annotations for this file:' + datafile)
    elif 'LW' in locals():
        ex1 = LW
        print('One annotations for this file:' + datafile)
    else:
        print('No annotations for this file:' + datafile)
    annotation={}

    annotation={}

    if 'ex4' in locals():
        ex4 = needleplusnonanalysable(ex4)

    if 'ex2' in locals():
        ex2 = needleplusnonanalysable(ex2)

    if 'ex4' in locals():
        ex4 = needleplusnonanalysable(ex4)

    if 'ex1' in locals():
        ex1 = needleplusnonanalysable(ex1)

    if 'ex4' in locals():
        for x in range(len(ex1)):
            annotation[x]=ex1[x],ex2[x], ex3[x], ex4[x]
    elif 'ex3' in locals():
        for x in range(len(ex1)):
            annotation[x]=ex1[x], ex2[x], ex3[x]
    elif 'ex2' in locals():
        for x in range(len(ex1)):
            annotation[x]=ex1[x],ex2[x]
    else:
        pass
    
    ## calculating majority vote: Tie is no annotations, otherwise majority wins.
    if ('ex1' in locals()) and ('ex2' in locals()): # only calculate when two or more annotations
        finalannotations=np.zeros([1,len(ex1)])
        for x in range (len(ex1)):
            count=Counter(annotation[x])
            top_two = count.most_common(2)
            if len(top_two)>1 and top_two[0][1] == top_two[1][1]:
                # It is a tie
                finalannotations[0,x] = 0
            else:
                finalannotations[0,x] = top_two[0][0]
            # print(top_two[0][0])

        finalannotations=np.array(finalannotations).flatten()
        finalannotations=pd.DataFrame(finalannotations)

        finalpath = filepath_majority_vote + '/'+ datafile + '.csv'
        finalannotations.to_csv(finalpath, index= False)