""" 
The method to create Mel spectrogram from time signal.

@author: Sterre de Jonge (2022)
"""

# Python (default/external) imports
import numpy as np
import librosa
import matplotlib.pyplot as plt
import pyloudnorm as pyln

def normalise_db(data, fs, loudness_normalise_level):
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
    loudness_normalized_audio = pyln.normalize.loudness(data, loudness, loudness_normalise_level)
    return loudness_normalized_audio

def create_mels(method, data, sample_rate, n_mels, fmax, hop_length, max_db, ref, sliding_window_step_time, sample_time):
    """ 
    This file creates mel spectrograms using the librosa library. There are three different methods that can be 
    used to create the Mel spectrograms with. Each method has a different order of steps for the computation of
    the Mel spectrograms. 
    """

    number_of_samples_per_segment = np.int(np.floor(sample_rate * sample_time))
    sliding_window_size = np.int(np.floor(sample_rate * sliding_window_step_time))
    data_iterations = np.arange(0, len(data)-number_of_samples_per_segment, sliding_window_size)

    if not data_iterations.any():
            return [], []

    if method == 'method3':

        # Derive mel from entire data 
        mel = librosa.feature.melspectrogram(y=data, sr=sample_rate, n_mels=n_mels, fmax=fmax, hop_length=hop_length)
        mel = np.asarray(mel).reshape(n_mels, len(mel[0]))
        mel = librosa.power_to_db(mel, ref=ref, top_db=max_db)

        # rescale mel to [0,255] to be able to save it as image
        std_mel = (mel - np.min(mel)) / (np.max(mel) - np.min(mel))
        scaled_mel = (std_mel * (255 - 0) + 0).astype(np.uint8)

        mel_full = scaled_mel

        # retrieve seperate mels
        move_bins = sliding_window_step_time / (hop_length / sample_rate)

        for i in range(len(data_iterations)):
            range_start = round(i * move_bins)
            range_end = range_start + 128

            mel = mel_full[:, range_start:range_end]
            mel = np.expand_dims(mel, axis=0)
            if i == 0: mel_singles = mel
            else: mel_singles = np.append(mel_singles, mel, axis=0)

        return mel_full, mel_singles

    if method == 'method2':

        for i in range(len(data_iterations)):
            d = data[data_iterations[i]:data_iterations[i]+number_of_samples_per_segment]
            mel = librosa.feature.melspectrogram(y=d, sr=sample_rate, n_mels=n_mels, fmax=fmax, hop_length=hop_length)
            mel = np.asarray(mel).reshape(n_mels, len(mel))
            mel = librosa.power_to_db(mel, ref=ref, top_db=max_db)
            if i == 0: mels = mel
            else: mels = np.append(mels, mel, axis=0)

        mels_flatten = mels.flatten()    

        # normalize mels with respect to each other
        normalized_mels = (mels_flatten - np.min(mels_flatten)) / (np.max(mels_flatten) - np.min(mels_flatten))
        scaled_mels = (normalized_mels * 255).astype(np.uint8)

        mel_singles = np.reshape(scaled_mels, (len(mels), 128, 128))
        mel_full = None

        return mel_full, mel_singles

    if method == 'method1':

        for i in range(len(data_iterations)):

            d = data[data_iterations[i]:data_iterations[i]+number_of_samples_per_segment]
            mel = librosa.feature.melspectrogram(y=d, sr=sample_rate, n_mels=n_mels, fmax=fmax, hop_length=hop_length)
            mel = np.asarray(mel).reshape(n_mels, len(mel))
            mel = librosa.power_to_db(mel)

            # rescale mel to [0,255] to be able to save it as image
            std_mel = (mel - np.min(mel)) / (np.max(mel) - np.min(mel))
            scaled_mel = (std_mel * (255 - 0) + 0).astype(np.uint8)
            
            mel = np.expand_dims(scaled_mel, axis=0)
            
            if i == 0: mel_singles = mel
            else: mel_singles = np.append(mel_singles, mel, axis=0)

        mel_full = None
     
        return mel_full, mel_singles
