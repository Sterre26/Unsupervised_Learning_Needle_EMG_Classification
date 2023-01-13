""" 
Creates rest dataset 

@author: Sterre de Jonge (2022)
"""

# Python (default/external) imports
import numpy as np
from scipy.io import wavfile
import shutil
import os
import pandas as pd
from tqdm import tqdm
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
        os.makedirs(path_to_create + "Rest/")
        os.makedirs(path_to_create + "Fibrillation/")
        os.makedirs(path_to_create + "PSW/")
        os.makedirs(path_to_create + "Fasciculation/")
        os.makedirs(path_to_create + "Myotonic_discharge/")
        os.makedirs(path_to_create + "Neuromyotonia/")
        os.makedirs(path_to_create + "Myokimia/")
        os.makedirs(path_to_create + "CRD/")
        os.makedirs(path_to_create + "Endplate_noise/")
        os.makedirs(path_to_create + "Endplate_spikes/")
        os.makedirs(path_to_create + "MU_contraction/")
        os.makedirs(path_to_create + "Fib_PSW/")
    elif os.path.exists(path_to_create) and not override_flag:
        print("A directory already exists in {} and the override flag is set to {}. If this was not the"
                "desired result run again with different parameters".format(path_to_create,
                                                                            override_flag))
    else:
        print("Creating new directory in {}.".format(path_to_create))
        os.makedirs(path_to_create)
        os.makedirs(path_to_create + "Rest/")
        os.makedirs(path_to_create + "Fibrillation/")
        os.makedirs(path_to_create + "PSW/")
        os.makedirs(path_to_create + "Fasciculation/")
        os.makedirs(path_to_create + "Myotonic_discharge/")
        os.makedirs(path_to_create + "Neuromyotonia/")
        os.makedirs(path_to_create + "Myokimia/")
        os.makedirs(path_to_create + "CRD/")
        os.makedirs(path_to_create + "Endplate_noise/")
        os.makedirs(path_to_create + "Endplate_spikes/")
        os.makedirs(path_to_create + "MU_contraction/")
        os.makedirs(path_to_create + "Fib_PSW/")

def save_mels_from_annotated_segments(path_to_store, file_list, sample_time, sliding_window_time, n_mels, fmax, hop_length, max_db, ref, method, loudness_normalise_level):
    print("Creating input data from annotated segments.")
    
    for annotated_file in tqdm(file_list):
        filename_base = os.path.basename(annotated_file).split("CV.csv")[0]
        
        file_name = annotated_file.split("CV.csv")[0]
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

            if len(d) == number_of_samples_per_step:
    
                # if 100% of the annotation is the same
                if len(ann_set) == 1: 
                    if ann[0] == 1: skimage.io.imsave(path_to_store + "Rest/" + filename_base + "_" + str(i) + ".png", mel_singles[i])
                    elif ann[0] == 6: skimage.io.imsave(path_to_store + "Fibrillation/" + filename_base + "_" + str(i) + ".png", mel_singles[i])
                    elif ann[0] == 7: skimage.io.imsave(path_to_store + "PSW/" + filename_base + "_" + str(i) + ".png", mel_singles[i])
                    elif ann[0] == 8: skimage.io.imsave(path_to_store + "Fasciculation/" + filename_base + "_" + str(i) + ".png", mel_singles[i])
                    elif ann[0] == 9: skimage.io.imsave(path_to_store + "Myotonic_discharge/" + filename_base + "_" + str(i) + ".png", mel_singles[i])
                    elif ann[0] == 10: skimage.io.imsave(path_to_store + "Neuromyotonia/" + filename_base + "_" + str(i) + ".png", mel_singles[i])
                    elif ann[0] == 11: skimage.io.imsave(path_to_store + "Myokymia/" + filename_base + "_" + str(i) + ".png", mel_singles[i])
                    elif ann[0] == 12: 
                        skimage.io.imsave(path_to_store + "CRD/" + filename_base + "_" + str(i) + ".png", mel_singles[i])
                        
                    elif ann[0] == 13: skimage.io.imsave(path_to_store + "Enplate_noise/" + filename_base + "_" + str(i) + ".png", mel_singles[i])
                    elif ann[0] == 14: skimage.io.imsave(path_to_store + "Endplate_spikes/" + filename_base + "_" + str(i) + ".png", mel_singles[i])
                    elif ann[0] == 15: skimage.io.imsave(path_to_store + "MU_contraction/" + filename_base + "_" + str(i) + ".png", mel_singles[i])
                    elif ann[0] == 0:
                        pass

                # if two types of annotations exist
                elif len(ann_set) == 2:
                    if 1.0 in ann_set: # make sure that the other annotation is rest
                        if len(ann[ann==6]) >= len(ann) * 0.05: skimage.io.imsave(path_to_store + "Fibrillation/" + filename_base + "_" + str(i) + ".png", mel_singles[i])
                        elif len(ann[ann==7]) >= len(ann) * 0.05: skimage.io.imsave(path_to_store + "PSW/" + filename_base + "_" + str(i) + ".png", mel_singles[i])
                        elif len(ann[ann==8]) >= len(ann) * 0.05: skimage.io.imsave(path_to_store + "Fasciculation/" + filename_base + "_" + str(i) + ".png", mel_singles[i])
                        elif len(ann[ann==9]) >= len(ann) * 0.05: skimage.io.imsave(path_to_store + "Myotonic_discharge/" + filename_base + "_" + str(i) + ".png", mel_singles[i])
                        elif len(ann[ann==10]) >= len(ann) * 0.05: skimage.io.imsave(path_to_store + "Neuromyotonia/" + filename_base + "_" + str(i) + ".png", mel_singles[i])
                        elif len(ann[ann==11]) >= len(ann) * 0.05: skimage.io.imsave(path_to_store + "Myokymia/" + filename_base + "_" + str(i) + ".png", mel_singles[i])
                        elif len(ann[ann==12]) >= len(ann) * 0.05: skimage.io.imsave(path_to_store + "CRD/" + filename_base + "_" + str(i) + ".png", mel_singles[i])
                        elif len(ann[ann==13]) >= len(ann) * 0.05: skimage.io.imsave(path_to_store + "Enplate_noise/" + filename_base + "_" + str(i) + ".png", mel_singles[i])
                        elif len(ann[ann==14]) >= len(ann) * 0.05: skimage.io.imsave(path_to_store + "Endplate_spikes/" + filename_base + "_" + str(i) + ".png", mel_singles[i])
                        elif len(ann[ann==15]) >= len(ann) * 0.05: skimage.io.imsave(path_to_store + "MU_contraction/" + filename_base + "_" + str(i) + ".png", mel_singles[i])

                elif len(ann_set) == 3:
                    if 1.0 in ann_set: 
                        if len(ann[ann==6]) >= len(ann) * 0.05 and len(ann[ann==7]) >= len(ann) * 0.05: 
                            skimage.io.imsave(path_to_store + "Fib_PSW/" + filename_base + "_" + str(i) + ".png", mel_singles[i])

if __name__ == "__main__":

    config = GenerateDataOptions().parse()

    # create datafolders
    create_datafolders(path_to_create=config.experiment_directory_rest, override_flag=config.override)

    # load annotation data
    path_to_annotated_data = config.path_to_wav_data_rest

    annotated_file_list = \
            [path_to_annotated_data + f for f in os.listdir(path_to_annotated_data) if ".csv" in f]
    if not annotated_file_list: 
        print("Annotation files (.csv files) are not stored in the same folder as the original wav files")
        exit()
    annotated_file_list.sort()

    # create mel spectrograms and fill folders
    save_mels_from_annotated_segments(path_to_store=config.experiment_directory_rest, file_list=annotated_file_list, 
                                    sample_time=config.sample_time, sliding_window_time=config.sliding_window_time, 
                                    n_mels=config.n_mels, fmax=config.fmax, hop_length=config.hop_length,max_db=config.max_db, 
                                    ref=config.ref, loudness_normalise_level=config.loudness_normalise_level, method=config.method)
    