# Python (default/external) imports
import numpy as np
from scipy.io import wavfile
import shutil
import os
from tqdm import tqdm
import skimage.io 
# custom made imports
from create_mels import create_mels, normalise_db
from options_datagen import GenerateDataOptions

""" 
Creates unlabelled dataset 

@author: Sterre de Jonge (2022)
"""

def create_datafolders(path_to_create, override_flag):
    """This function creates the datafolders that will be populated with data"""

    if os.path.exists(path_to_create) and override_flag:
        print("A directory already exists in {}, removing it and creating a new one.".format(path_to_create))
        shutil.rmtree(path_to_create, ignore_errors=True)
        os.makedirs(path_to_create)
        os.makedirs(path_to_create + "train/")
    elif os.path.exists(path_to_create) and not override_flag:
        print("A directory already exists in {} and the override flag is set to {}. If this was not the"
                "desired result run again with different parameters".format(path_to_create,
                                                                            override_flag))
    else:
        print("Creating new directory in {}.".format(path_to_create))
        os.makedirs(path_to_create)
        os.makedirs(path_to_create + "train/")

def save_mels_from_annotated_segments(path_to_store, path_to_files, sample_time, sliding_window_time, n_mels, fmax, hop_length, max_db, ref, method,loudness_normalise_level):
    print("Creating input data from annotated segments.")

    if ".wav" in path_to_files: file_list = [path_to_files]
    else: file_list = [path_to_files + f for f in os.listdir(path_to_files) if ".wav" in f]

    file_list.sort()
    
    for file in tqdm(file_list):
        filename_base = os.path.basename(file).split(".wav")[0]

        # load data
        fs, data = wavfile.read(file)
        data = data.astype('float')
        
        # normalise data to loudness normalisation level
        data = normalise_db(data, fs, loudness_normalise_level)

        # Cut out segments
        number_of_samples_per_step = np.int32(np.floor(fs * sample_time))
        number_of_samples_per_sliding_window = np.int32(np.floor(fs * sliding_window_time))
        data_iterations = np.arange(0, len(data)-number_of_samples_per_step, number_of_samples_per_sliding_window)

        # create mel spectrograms
        mel_full, mel_singles = create_mels(data=data, sample_rate=fs, n_mels=n_mels, fmax=fmax, hop_length=hop_length, 
                                            max_db=max_db, ref=ref, sliding_window_step_time=sliding_window_time, 
                                            sample_time=sample_time, method=method)

        # save images according to annotations
        for i in range(len(data_iterations)):
                # save as PNG
                out = path_to_store + "train/" + filename_base + "_" + str(i) + ".png"
                skimage.io.imsave(out, mel_singles[i])

if __name__ == "__main__":

    config = GenerateDataOptions().parse()

    # create datafolders
    create_datafolders(path_to_create=config.experiment_directory_unlabelled, override_flag=config.override)

    # create mel spectrograms and fill folders
    save_mels_from_annotated_segments(path_to_store=config.experiment_directory_unlabelled, path_to_files=config.path_to_wav_data_unlabelled, 
                                    sample_time=config.sample_time, sliding_window_time=config.sliding_window_time, 
                                    n_mels=config.n_mels, fmax=config.fmax, hop_length=config.hop_length,max_db=config.max_db, 
                                    ref=config.ref, loudness_normalise_level=config.loudness_normalise_level, method=config.method)
