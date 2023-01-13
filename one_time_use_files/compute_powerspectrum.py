"""
Compute powerspectrograms of annotated signals

@author: Sterre de Jonge (2022)
"""

# Python (default/external) imports
import numpy as np
from scipy.io import wavfile
import shutil
import os
import pyloudnorm as pyln
import pandas as pd
from tqdm import tqdm
import skimage.io 
from scipy import signal
import matplotlib.pyplot as plt
# custom-made imports
from dataset_generation.create_mels import create_mels

class ComputePowerspectrum:

    def __init__(self, base_working_path, experiment_directory,
                 path_to_annotated_data, path_to_original_data, dataset_generation_override_flag, hop_length, n_mels, fmax, invert,
                 input_dimension_width, sample_time, sliding_window_step_time, loudness_normalise_level=-26.0, border_flag=False, sample_rate=44100
                 ):
        """
        @param base_working_path: str
            Directory path to the root folder of the project.
        @param experiment_directory: str
            Directory where the experiment files are to be stored.
        @param path_to_annotated_data: str
            Directory in which the annotated files are stored.
        @param path_to_original_data: str
            Directory in which the original data is stored.
        @param dataset_generation_override_flag: Boolean
            Flag to indicate whether an existing experiment_directory should be overwritten or not.
        @param sample_time: float
            Time per extracted segment. Inherits default from model_training_options.
        @param sliding_window_step_time: float
            Sliding window size. Inherits default from model_training_options.
        @param loudness_normalise_level: float
            Level to normalise audio to in db.
        @param border_flag: Boolean
            Flag to indicate whether edge cases (border) should be included or not.
        @param sample_rate: int
            Rate original signal was sampled at. Inherits default from model_training_options.
        """
        # Filepaths
        self.base_working_path = base_working_path
        self.path_to_annotated_data = path_to_annotated_data
        self.path_to_original_data = path_to_original_data
        self.experiment_directory = experiment_directory

        # Input data parameters
        self.sample_time = sample_time
        self.sliding_window_step_time = sliding_window_step_time
        self.loudness_normalise_level = loudness_normalise_level
        self.sample_rate = sample_rate

        #librosa parameters
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.fmax = fmax 
        self.input_dimension_width = input_dimension_width
        self.invert = invert
        # Flags
        self.dataset_generation_override_flag = dataset_generation_override_flag
        self.border_flag = border_flag

    def create_dataset(self, needle_flag=False):
        """
        Create the dataset directory structure and populate it with data. A dataset_generation_override_flag is used
        to prevent accidental deletion of a folder. Might want to update this with symlinks later.
        """
        self.compute_input_data_from_annotated_segments()


    def normalise_db(self, data, fs):
        """
        Normalise the dB of a given volume to self.loudness_normalise_level
        @param data: np.array
            Array of floats representing the .wav file.
        @param fs: int
            Sample rate of the signal.
        @return: loudness_normalized_audio: np.array
            Array of floats representing the .wav file normalised in dB.
        """
        # measure the loudness first
        meter = pyln.Meter(fs)  # create BS.1770 meter
        loudness = meter.integrated_loudness(data)

        # loudness normalize audio to self.loudness_normalise_level
        loudness_normalized_audio = pyln.normalize.loudness(data, loudness, self.loudness_normalise_level)
        return loudness_normalized_audio

    def compute_input_data_from_annotated_segments(self, needle_flag=False):
        """
        Use the annotated segments to create input data with length self.sample_time which is cut out at intervals
        of self.sliding_window_step_time apart.
        :return:
        """
        print("Creating input data from annotated segments.")
        # Grabbing all .csv files (i.e. the annotated segments)
        annotated_file_list = \
            [self.path_to_annotated_data + f for f in os.listdir(self.path_to_annotated_data) if ".csv" in f]

        psd_rest = []
        psd_crd = []
        psd_psw = []
        psd_fib = []
        psd_fibpsw = []
        psd_myo = []


        # Loop through all annotated segments
        for annotated_file in tqdm(annotated_file_list):
            filename_base = os.path.basename(annotated_file).split("CV.csv")[0]
            original_data_path = self.path_to_annotated_data + filename_base + ".wav"

            fs, data = wavfile.read(original_data_path)
            data = data.astype('float')
            # data = self.normalise_db(data, fs)
            annotations = np.array(pd.read_csv(annotated_file)).flatten()

            # Cut out ns segments based on labels (this time is variable based on self.sample_time)
            number_of_samples_per_step = np.int(np.floor(self.sample_rate * self.sample_time))
            sliding_window_size = np.int(np.floor(self.sample_rate * self.sliding_window_step_time))
            data_iterations = np.arange(0, len(data)-number_of_samples_per_step, sliding_window_size)

            # Loop through all cut out segments
            count = 0
            for i in range(len(data_iterations)):
                count += 1
                ann = annotations[data_iterations[i]:data_iterations[i]+number_of_samples_per_step]
                d = data[data_iterations[i]:data_iterations[i]+number_of_samples_per_step]
                ann_set = set(ann) 

                f_copy = None

                # Ensure that a cut out segment is indeed of only one class type (i.e. rest, contraction or needle)
                # (with len(ann_set) == 1) and that it is long enough (to prevent edge cases).
                if len(d) == number_of_samples_per_step:
                    
                    # if 100% of the annotation is the same
                    if len(ann_set) == 1: 
                        if ann[0] == 1: 
                            f, Pxx_den = signal.periodogram(d, fs)
                            psd_rest.append(Pxx_den)
                        elif ann[0] == 6: 
                            f, Pxx_den = signal.periodogram(d, fs)
                            psd_fib.append(Pxx_den)
                        elif ann[0] == 7: 
                            f, Pxx_den = signal.periodogram(d, fs)
                            psd_psw.append(Pxx_den)
                        elif ann[0] == 9: 
                            f, Pxx_den = signal.periodogram(d, fs)
                            psd_myo.append(Pxx_den)
                        elif ann[0] == 12: 
                            f, Pxx_den = signal.periodogram(d, fs)
                            psd_crd.append(Pxx_den)
                        elif ann[0] == 0:
                            pass

                    # if two types of annotations exist
                    elif len(ann_set) == 2:
                        if 1.0 in ann_set: # make sure that the other annotation is rest
                            if len(ann[ann==6]) >= len(ann) * 0.05: 
                                f, Pxx_den = signal.periodogram(d, fs)
                                psd_fib.append(Pxx_den)
                            elif len(ann[ann==7]) >= len(ann) * 0.05: 
                                f, Pxx_den = signal.periodogram(d, fs)
                                psd_psw.append(Pxx_den)
                            elif len(ann[ann==9]) >= len(ann) * 0.05: 
                                f, Pxx_den = signal.periodogram(d, fs)
                                psd_myo.append(Pxx_den)
                            elif len(ann[ann==12]) >= len(ann) * 0.05: 
                                f, Pxx_den = signal.periodogram(d, fs)
                                psd_crd.append(Pxx_den)
                                
                    elif len(ann_set) == 3:
                        if 1.0 in ann_set: 
                            if len(ann[ann==6]) >= len(ann) * 0.05 and len(ann[ann==7]) >= len(ann) * 0.05: 
                                f, Pxx_den = signal.periodogram(d, fs)
                                psd_fibpsw.append(Pxx_den)


        psd_rest = np.asarray(psd_rest)
        psd_rest_average = np.average(psd_rest, axis=0)
        psd_rest_std = np.std(psd_rest, axis=0)
        psd_crd = np.asarray(psd_crd)
        psd_crd_average = np.average(psd_crd, axis=0)
        psd_crd_std = np.std(psd_crd, axis=0)
        psd_psw = np.asarray(psd_psw)
        psd_psw_average = np.average(psd_psw, axis=0)
        psd_psw_std = np.std(psd_psw, axis=0)
        psd_fib = np.asarray(psd_fib)
        psd_fib_average = np.average(psd_fib, axis=0)
        psd_fib_std = np.std(psd_fib, axis=0)
        psd_fibpsw = np.asarray(psd_fibpsw)
        psd_fibpsw_average = np.average(psd_fibpsw, axis=0)
        psd_fibpsw_std = np.std(psd_fibpsw, axis=0)
        psd_myo = np.asarray(psd_myo)
        psd_myo_average = np.average(psd_myo, axis=0)
        psd_myo_std = np.std(psd_myo, axis=0)

        plt.figure(1)
        plt.semilogy(f, psd_rest_average)
        plt.fill_between(f, psd_rest_average-psd_rest_std, psd_rest_average+psd_rest_std, 
                        alpha=0.3)
        plt.xlabel('frequency [Hz]')
        plt.ylabel('PSD [V**2/Hz]')
        plt.title('Rest')
        # plt.xlim(0,500)
        plt.ylim(10**-10, 10**-4)

        plt.figure(2)
        plt.semilogy(f, psd_crd_average)
        plt.fill_between(f, psd_crd_average-psd_crd_std, psd_crd_average+psd_crd_std, 
                        alpha=0.3)
        plt.xlabel('frequency [Hz]')
        plt.ylabel('PSD [V**2/Hz]')
        plt.title('CRD')
        plt.xlim(0,500)
        plt.ylim(10**-10, 10**-4)

        plt.figure(3)
        plt.semilogy(f, psd_psw_average)
        plt.fill_between(f, psd_psw_average-psd_psw_std, psd_psw_average+psd_psw_std, 
                        alpha=0.3)
        plt.xlabel('frequency [Hz]')
        plt.ylabel('PSD [V**2/Hz]')
        plt.title('PSW')
        plt.xlim(0,500)
        plt.ylim(10**-10, 10**-4)

        plt.figure(4)
        plt.semilogy(f, psd_fib_average)
        plt.fill_between(f, psd_fib_average-psd_fib_std, psd_fib_average+psd_fib_std, 
                        alpha=0.3)
        plt.xlabel('frequency [Hz]')
        plt.ylabel('PSD [V**2/Hz]')
        plt.title('Fibrillations')
        plt.xlim(0,500)
        plt.ylim(10**-10, 10**-4)

        plt.figure(5)
        plt.semilogy(f, psd_fibpsw_average)
        plt.fill_between(f, psd_fibpsw_average-psd_fibpsw_std, psd_fibpsw_average+psd_fibpsw_std, 
                        alpha=0.3)
        plt.xlabel('frequency [Hz]')
        plt.ylabel('PSD [V**2/Hz]')
        plt.title('Fibrillations+PSW')
        plt.xlim(0,500)
        plt.ylim(10**-10, 10**-4)

        plt.figure(6)
        plt.semilogy(f, psd_myo_average)
        plt.fill_between(f, psd_myo_average-psd_myo_std, psd_myo_average+psd_myo_std, 
                        alpha=0.3)
        plt.xlabel('frequency [Hz]')
        plt.ylabel('PSD [V**2/Hz]')
        plt.title('Myotonic discharge')
        plt.xlim(0,500)
        plt.ylim(10**-10, 10**-4)
        plt.show()
                      
    @staticmethod
    def annotation_array_contains_needle_movement(target_array, target_sequence):
        """
        Added [0::4410] slicing to significantly speed up evaluation.
        @param target_array:
        @param target_sequence:
        @return:
        """
        target_array = target_array[0::4410]
        target_sequence = target_sequence[0::4410]
        return_bool = False
        for i in range(0, len(target_array) - len(target_sequence) + 1):
            if np.array_equal(target_sequence, target_array[i:i+len(target_sequence)]):
                return_bool = True
        return return_bool
