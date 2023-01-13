# Unsupervised_Learning_Needle_EMG_Classification
Implementation of two unsupervised learning techniques for the automatic classification of needle EMG signals. This code was developed for a thesis project using hospital needle EMG data acquired from Amsterdam UMC, location AMC (the Netherlands). 

## Installation
Install requirements from requirements.txt. The project is implemented in Tensorflow 2.X. 

'''
pip install -r requirements.txt
'''

## File structure
The files are stored in three folders:

/dataset_generation:     _files to unlabelled and labelled datasets (from annotated .csv); settings for input data creation can be changed in        'options_datagen.py'_

/unsupervised_learning:  _files for training and evaluating unsupervised learning models; settings for training can be changed in 'unsupervised_models_options.py'_

/one_time_use_files:     _files that were used once_

## Usage
In order to train the unsupervised learning models it is important to generate and store the input data (converted to Mel spectrograms) in folders that can be loaded with Keras datagenerator. Results during training are stored in \results folder. 

