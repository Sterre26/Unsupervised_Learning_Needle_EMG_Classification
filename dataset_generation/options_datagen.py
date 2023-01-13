# Python (default/external) imports
import argparse
import numpy as np
import json

""" 
Options file for generation of data (Mel spectrograms)

@author: Sterre de Jonge (2022)
"""

class GenerateDataOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialised = False

    def initialise(self):
        # filepathes
        self.parser.add_argument('--base_working_path', default=r'/Users/Sterre/_Python_folders/ai_needle_emg_internshipSterre/')
        # filepaths: unlablled
        self.parser.add_argument('--experiment_directory_unlabelled', default='/Users/Sterre/files/PROCESSED/mels_8/Training_100percent/')
        self.parser.add_argument('--path_to_wav_data_unlabelled', default='/Users/Sterre/files/WAV/training_unlabelled/')
        # filepaths: signal type
        self.parser.add_argument('--experiment_directory_signal_type', default='/Users/Sterre/files/PROCESSED/mels_8/SignalType/')
        self.parser.add_argument('--path_to_wav_data_signal_type', default='/Users/Sterre/files/WAV/annotation_signaltype/')
        # filepaths: rest
        self.parser.add_argument('--experiment_directory_rest', default='/Users/Sterre/files/PROCESSED/mels_size_square/Rest/original_rest_labelled/')
        self.parser.add_argument('--path_to_wav_data_rest', default='/Users/Sterre/files/WAV/annotation_rest/')
        
        self.parser.add_argument('--override', default=True)

        # creating mel spectrogramssam
        self.parser.add_argument('--sample_time', default=1.48)
        self.parser.add_argument('--sliding_window_time', default=0.1)
        self.parser.add_argument('--hop_length', default=512)
        self.parser.add_argument('--n_mels', default=128)
        self.parser.add_argument('--fmax', default=10000)

        self.parser.add_argument('--max_db', default=80.0) # default is 80.0
        self.parser.add_argument('--ref', default=np.max)   # default is np.max
        
        self.parser.add_argument('--loudness_normalise_level', default=-26.0)

        # method to create mels by
        self.parser.add_argument('--method', default='method1')

    def parse(self):
        if not self.initialised:
            self.initialise()
        self.opt = self.parser.parse_args()

        return self.opt
