"""
A (random) number was given to each patient so that files from patients can be kept separate. 

@author: Sterre de Jonge
"""
import pandas as pd
import os

directory_files = '/Users/Sterre/files/WAV/annotation_rest/'
excel_path = '/Users/Sterre/files/WAV/ptnID_filename.xlsx'

excel_file = pd.read_excel(io=excel_path, sheet_name='Blad1')

files = [f for f in os.listdir(directory_files) if ".csv" in f]

howmany = 0

for file in files:
    for index, name in enumerate(excel_file['Filename']):
        if file[:-6] == name or file[:-7] == name or file[:-8] == name or file[:-9] == name: 
            oldname = file
            newname = str(int(excel_file['PtnID'][index])).zfill(3) + '_' + file
            os.rename(directory_files + oldname, directory_files + newname)
            howmany += 1

print("Files renamed: {}".format(howmany))



