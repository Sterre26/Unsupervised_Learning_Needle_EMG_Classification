""" 
Creates signal type dataset 

@author: Sterre de Jonge (2022)
"""

# Python (default/external) imports
import numpy as np
from scipy.io import wavfile
import shutil
import os
import pandas as pd
from tqdm import tqdm
import random
import skimage.io 
# custom-made imports
from create_mels import create_mels, normalise_db
from options_datagen import GenerateDataOptions

def create_datafolders(path_to_create, override_flag):

    """This function creates the datafolders that will be populated with data"""

    if os.path.exists(path_to_create) and override_flag:
        print("A directory already exists in {}, removing it and creating a new one.".format(path_to_create))
        
        shutil.rmtree(path_to_create, ignore_errors=True)
        os.makedirs(path_to_create)
        os.makedirs(path_to_create + "Validation/Rest/")
        os.makedirs(path_to_create + "Validation/Contraction/")
        os.makedirs(path_to_create + "Validation/Artefact/")
        os.makedirs(path_to_create + "Test/Rest/")
        os.makedirs(path_to_create + "Test/Contraction/")
        os.makedirs(path_to_create + "Test/Artefact/")
    elif os.path.exists(path_to_create) and not override_flag:
        print("A directory already exists in {} and the override flag is set to {}. If this was not the"
                "desired result run again with different parameters".format(path_to_create,
                                                                            override_flag))
    else:
        print("Creating new directory in {}.".format(path_to_create))
        os.makedirs(path_to_create)
        os.makedirs(path_to_create + "Validation/Rest/")
        os.makedirs(path_to_create + "Validation/Contraction/")
        os.makedirs(path_to_create + "Validation/Artefact/")
        os.makedirs(path_to_create + "Test/Rest/")
        os.makedirs(path_to_create + "Test/Contraction/")
        os.makedirs(path_to_create + "Test/Artefact/")

def save_mels_from_annotated_segments(path_to_store, file_list, sample_time, sliding_window_time, n_mels, fmax, hop_length, max_db, ref, method,loudness_normalise_level):
    print("Creating input data from annotated segments.")
    
    for annotated_file in tqdm(file_list):
        filename_base = os.path.basename(annotated_file).split(".csv")[0]
        
        file_name = annotated_file.split(".csv")[0]
        file_path = file_name + ".wav"

        # load data
        fs, data = wavfile.read(file_path)
        data = data.astype('float')
        annotations = np.array(pd.read_csv(annotated_file)).flatten()
        
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
            ann = annotations[data_iterations[i]:data_iterations[i]+number_of_samples_per_step]
            d = data[data_iterations[i]:data_iterations[i]+number_of_samples_per_step]
            ann_set = set(ann) 

            # Ensure that a cut out segment is indeed of only one class type (i.e. rest, contraction or needle)
            # (with len(ann_set) == 1) and that it is long enough (to prevent edge cases).
            if len(d) == number_of_samples_per_step:
                
                if len(ann_set) == 1:
                    if ann[0] == 1: 
                        skimage.io.imsave(path_to_store + "Rest/" + filename_base + "_" + str(i) + ".png", mel_singles[i])
                    elif ann[0] == 2: 
                        skimage.io.imsave(path_to_store + "Contraction/" + filename_base + "_" + str(i) + ".png", mel_singles[i])
                    elif ann[0] == 3: 
                        skimage.io.imsave(path_to_store + "Artefact/" + filename_base + "_" + str(i) + ".png", mel_singles[i])
                        
                        
                        # if needle_flag: # needle_flag = False but maybe for later
                        #     dif = np.max(d) - np.min(d)
                        #     if dif > 1000: skimage.io.imsave(self.VAL_folder + "Artefact/" + filename_base + "_" + str(i) + ".png", mel_singles[i])

                        
                    elif ann[0] == 0: pass
                
                # When 15% of signal contains needle annotation (does not have to be immediate next to each other)
                # if len(ann[ann==3]) >= len(ann) * 0.15: skimage.io.imsave(path_to_store + "Artefact/" + filename_base + "_" + str(i) + ".png", mel_singles[i])
                
                # And when we think that this 15% should be consequtive, this is the code for that:
                elif len(ann_set) == 2 or len(ann_set) == 3:
                    needle_movement = 3 * np.ones(np.int32(np.floor(0.15 * fs * sample_time)))
                    needle_boolean = annotation_array_contains_needle_movement(ann, needle_movement)
                    if needle_boolean: skimage.io.imsave(path_to_store + "Artefact/" + filename_base + "_" + str(i) + ".png", mel_singles[i])
                    
# @staticmethod
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

if __name__ == "__main__":

    config = GenerateDataOptions().parse()

    # create datafolders
    create_datafolders(path_to_create=config.experiment_directory_signal_type, override_flag=config.override)

    # load annotation data
    path_to_annotated_data = config.path_to_wav_data_signal_type

    annotated_file_list = \
            [path_to_annotated_data + f for f in os.listdir(path_to_annotated_data) if ".csv" in f]
    if not annotated_file_list: 
        print("Annotation files (.csv files) are not stored in the same folder as the original wav files")
        exit()
    annotated_file_list.sort()

    # separate validation from test data set
    VAL_file_list = annotated_file_list[:65]
    VAL_folder = config.experiment_directory_signal_type + 'Validation/'
    TEST_file_list = annotated_file_list[65:]
    TEST_folder = config.experiment_directory_signal_type + 'Test/'

    # create mel spectrograms and fill folders
    save_mels_from_annotated_segments(path_to_store=VAL_folder, file_list=VAL_file_list, 
                                    sample_time=config.sample_time, sliding_window_time=config.sliding_window_time, 
                                    n_mels=config.n_mels, fmax=config.fmax, hop_length=config.hop_length,max_db=config.max_db, 
                                    ref=config.ref, loudness_normalise_level=config.loudness_normalise_level, method=config.method)

    # create balanced data sets
    files_rest = [VAL_folder + 'Rest/' + f for f in os.listdir(VAL_folder + 'Rest/') if '.png' in f]
    files_contraction = [VAL_folder + 'Contraction/' + f for f in os.listdir(VAL_folder + 'Contraction/') if '.png' in f]
    files_needle = [VAL_folder + 'Artefact/' + f for f in os.listdir(VAL_folder + 'Artefact/') if '.png' in f]
    files = [files_rest, files_contraction, files_needle]

    max_files = min(map(len, files))
    for filelist in files:
        no_remove_files = len(filelist) - max_files
        print("no of files that need to be removed:", no_remove_files)
        index_remove = random.sample(range(0, len(filelist)), no_remove_files)
        for index in index_remove:
            os.remove(filelist[index])
    
    # create mel spectrograms and fill folders
    save_mels_from_annotated_segments(path_to_store=TEST_folder, file_list=TEST_file_list, 
                                    sample_time=config.sample_time, sliding_window_time=config.sliding_window_time, 
                                    n_mels=config.n_mels, fmax=config.fmax, hop_length=config.hop_length,max_db=config.max_db, 
                                    ref=config.ref, loudness_normalise_level=config.loudness_normalise_level, method=config.method)
    
    # create balanced data sets
    files_rest = [TEST_folder + 'Rest/' + f for f in os.listdir(TEST_folder + 'Rest/') if '.png' in f]
    files_contraction = [TEST_folder + 'Contraction/' + f for f in os.listdir(TEST_folder + 'Contraction/') if '.png' in f]
    files_needle = [TEST_folder + 'Artefact/' + f for f in os.listdir(TEST_folder + 'Artefact/') if '.png' in f]
    files = [files_rest, files_contraction, files_needle]

    max_files = min(map(len, files))
    for filelist in files:
        no_remove_files = len(filelist) - max_files
        print("no of files that need to be removed:", no_remove_files)
        index_remove = random.sample(range(0, len(filelist)), no_remove_files)
        for index in index_remove:
            os.remove(filelist[index])