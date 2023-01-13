from tensorflow import keras
import keras
from scipy.io.wavfile import read
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import librosa
from tqdm import tqdm
import pyloudnorm as pyln
# custom imports
from options_datagen import GenerateDataOptions
from create_mels import create_mels

"""
This file converts the output from convolutional model (decoder output) to audio. This file is not up to date.

Documentation:
- https://stackoverflow.com/questions/60365904/reconstructing-audio-from-a-melspectrogram-has-some-clipping-with-librosa
- https://github.com/DemisEom/SpecAugment/issues/10

@author: Sterre de Jonge (2022)
"""

wavfile = '/Users/Sterre/files/WAV/test/002_0006_L_int_1_1.wav'
config = GenerateDataOptions().parse()

fs, data = read(wavfile)
print(fs)
data = data.astype('float')

fmax = 10000
n_mels = 128
hop_length = 512

#normalise data # measure the loudness first
meter = pyln.Meter(fs)  # create BS.1770 meter
loudness = meter.integrated_loudness(data)
loudness_normalise_level = -45.0
data = pyln.normalize.loudness(data, loudness, loudness_normalise_level)

number_of_samples_per_step = np.int(np.floor(config.sample_time * config.sample_rate))
sliding_window_size = number_of_samples_per_step
data_iterations = np.arange(0, len(data)-number_of_samples_per_step, sliding_window_size)

# count = 0
# mels_data = np.empty((0,128,128))
# for i in range(len(data_iterations)):
#     count += 1
#     data_samples = data[data_iterations[i]:data_iterations[i]+number_of_samples_per_step]   

#     # mels = librosa.feature.melspectrogram(y=data, sr=samplerate, n_mels=self.n_mels, fmax=fmax, hop_length=self.hop_length)
#     # mels = np.asarray(mels).reshape(self.n_mels, len(mels[0]))
#     # mels = librosa.power_to_db(mels, ref=np.max)

#     # std_mels = (mels - np.min(mels)) / (np.max(mels) - np.min(mels))
#     # scaled_mels = (std_mels * (255 - 0) + 0).astype(np.uint8)
#     # img = scaled_mels[:, 0:128]


#     mels = librosa.feature.melspectrogram(y=data_samples, sr=config.sample_rate, n_mels=128, fmax=config.fmax, hop_length=config.hop_length)
#     mels = np.array(mels)
#     mels = mels.reshape(-1, 128, 128).astype('float32')

#     if mels_data is not None: mels_data = np.append(mels_data, mels, axis=0)    
#     else: mels_data = mels

# D = np.abs(librosa.stft(data))**2
# mels = librosa.feature.melspectrogram(y=data, sr=fs, S=D)

mels = librosa.feature.melspectrogram(y=data, sr=fs, n_mels=n_mels, fmax=fmax, hop_length=hop_length)
mels = np.asarray(mels).reshape(n_mels, len(mels[0]))

mel = mels[:, 0:1280]

audio_predict = librosa.feature.inverse.mel_to_audio(mel, hop_length=hop_length, sr=fs)
# audio_predict = np.reshape(audio_predict, (audio_predict.shape[0]))
# audio_predict = audio_predict.reshape(-1)
# audio_predict = audio_predict.tolist()

audio_o = audio_predict

max_time = len(audio_o)/config.sample_rate
time_steps = np.linspace(0, max_time, len(audio_o))



ax1 = plt.subplot(2,1,1)
ax2 = plt.subplot(2,1,2, sharex=ax1)
ax1.set_title("Original audio file {} (top) and original file converted to mel to audio file (bottom)".format(wavfile))
ax1.plot(time_steps, data[:len(audio_o)], color='k')
ax2.plot(time_steps, audio_o, color='tab:blue')
ax2.set(ylabel="amplitude", xlabel="time [s]", title="Output decoder audio file {}".format(wavfile))
plt.xlabel("Time [s]")
plt.show()
plt.savefig('signal.png')







# ## set parameters ##
# filename = '0002_0006_R_vas_med_1_1'

# location_model = '/Users/Sterre/_Python_folders/ai_needle_emg_internshipSterre/results/arch_one/12092022_1257/convae_model_final.hdf5'
# wavfolder = '/Users/Sterre/files/WAV/annotation_signaltype_setup2/'
# annotationfolder = '/Users/Sterre/files/WAV/annotation_signaltype_setup2/'
# mel_input_folder = '/Users/Sterre/files/TEST/test_' + filename + '/'

# wavfile_i = wavfolder + filename + '.wav' # input wavfile to cae

# # create mels (non-overlapping!)
# config = ModelTrainingOptions().parse()
# dataset_generation_input_parameters = {
#             'base_working_path': config.working_directory,
#             'experiment_directory': mel_input_folder,
#             'path_to_annotated_data': None,
#             'path_to_original_data': wavfile_i,
#             'dataset_generation_override_flag': True,
#             'sample_time': config.sample_time,
#             'sliding_window_step_time': config.sample_time,
#             'invert': False,
#             'border_flag': True,
#             'hop_length': config.hop_length,
#             'n_mels': config.n_mels,
#             'input_dimension_width': config.input_dimension_width,
#             'fmax': 5000
#             }
# dataset_generation = DatasetGenerationUnlabelled(**dataset_generation_input_parameters)
# dataset_generation.create_dataset(needle_flag=False)
# print("Mel spectrograms are created...")

# # load input data 
# file_list = [mel_input_folder + f for f in os.listdir(mel_input_folder) if ".png" in f]
# data_original = load_mels(file_list)
# print("Data is loaded...")

# fs, data = read(wavfile_i)
# data = data.astype('float')

# # normalise data # measure the loudness first
# meter = pyln.Meter(fs)  # create BS.1770 meter
# loudness = meter.integrated_loudness(data)
# loudness_normalise_level = -26.0
# data = pyln.normalize.loudness(data, loudness, loudness_normalise_level)

# number_of_samples_per_step = np.int(np.floor(config.sample_time * config.sample_rate))
# sliding_window_size = number_of_samples_per_step
# data_iterations = np.arange(0, len(data)-number_of_samples_per_step, sliding_window_size)

# count = 0
# mels_data = np.empty((0,175,128))
# for i in range(len(data_iterations)):
#     count += 1
#     data_samples = data[data_iterations[i]:data_iterations[i]+number_of_samples_per_step]   

#     mels = librosa.feature.melspectrogram(y=data_samples, sr=config.sample_rate, n_mels=175, fmax=config.fmax, hop_length=config.hop_length)
#     mels = np.array(mels)
#     mels = mels.reshape(-1, 175, 128).astype('float32')

#     if mels_data is not None: mels_data = np.append(mels_data, mels, axis=0)    
#     else: mels_data = mels

# # load model
# scale_fn = lambda x: 1/(2.**(x-1))
# custom_objects={"scale_fn":scale_fn} # dit werkt niet?
# autoencoder = keras.models.load_model(location_model, compile=False) 
# print("\nPre-trained model loaded succesfully.\n")

# # do predictions
# data_predict = autoencoder.predict(data_original)
# print("Prediction are succesfully made:", data_predict.shape)

# wavfile_i = read(wavfile_i)
# audio_i = wavfile_i[1]

# plt.figure(figsize=(25,5))

# max_time = len(audio_i)/config.sample_rate
# time_steps = np.linspace(0, max_time, len(audio_i))
# plt.plot(time_steps, audio_i, color='k')
# plt.savefig('signal2.png', transparent=True)

# audio_o = []
# print("Predicted mel spectrograms are converted back to audio...")
# for prediction in tqdm(mels_data):
#     prediction = np.array(prediction)
#     prediction = np.reshape(prediction, (175,128))
#     # print("min value", np.min(prediction), "max value", np.max(prediction))


#     audio_predict = librosa.feature.inverse.mel_to_audio(prediction, hop_length=config.hop_length, sr=config.sample_rate)
#     audio_predict = np.reshape(audio_predict, (audio_predict.shape[0]))
#     audio_predict = audio_predict.reshape(-1)
#     audio_predict = audio_predict.tolist()
    
#     audio_o = audio_o + audio_predict

# # read original wav file
# wavfile_i = read(wavfile_i)
# audio_i = wavfile_i[1]

# # annotation file
# annotation = annotationfolder + filename + '.csv' # file with majority vote annotations
# annotation = pd.read_csv(annotation)
# annotation = annotation.values.tolist()
# annotation_rest, annotation_contraction, annotation_needle, annotation_non_analysable = annotation.copy(), annotation.copy(), annotation.copy(), annotation.copy()
# for i, v in enumerate(annotation): 
#     if v != [1.0]: annotation_rest[i] = np.nan # rest is 1
#     if v == [1.0]: annotation_rest[i] = 0 
#     if v != [2.0]: annotation_contraction[i] = np.nan # contraction is 2
#     if v == [2.0]: annotation_contraction[i] = 0 
#     if v != [3.0]: annotation_needle[i] = np.nan # needle is 3
#     if v == [3.0]: annotation_needle[i] = 0 
#     if v != [4.0]: annotation_non_analysable[i] = np.nan # non-analysable is 4
#     if v == [4.0]: annotation_non_analysable[i] = 0 

# max_time = len(audio_o)/config.sample_rate
# time_steps = np.linspace(0, max_time, len(audio_o))

# ax1 = plt.subplot(2,1,1)
# ax2 = plt.subplot(2,1,2, sharex=ax1)
# ax1.set_title("Original audio file {} (top) and original file converted to mel to audio file (bottom)".format(filename))

# ax1.plot(time_steps, audio_i[:len(audio_o)], color='k')
# # ax1.set(ylabel="amplitude", title="Original audio file {}".format(filename))
# # ax1.plot(time_steps,annotation_rest[:len(audio_o)], color='tab:orange', linestyle='-', linewidth=20, alpha=0.4)
# # ax1.plot(time_steps,annotation_contraction[:len(audio_o)], color='tab:green', linestyle='-', linewidth=20, alpha=0.4)
# # ax1.plot(time_steps,annotation_needle[:len(audio_o)], color='tab:pink', linestyle='-', linewidth=20, alpha=0.4)
# # ax1.plot(time_steps,annotation_non_analysable[:len(audio_o)], color='tab:purple', linestyle='-', linewidth=20, alpha=0.4)

# ax2.plot(time_steps, audio_o, color='tab:blue')
# # ax2.set(ylabel="amplitude", xlabel="time [s]", title="Output decoder audio file {}".format(filename))
# ax2.plot(time_steps,annotation_rest[:len(audio_o)], color='tab:orange', linestyle='-', linewidth=20, alpha=0.4)
# ax2.plot(time_steps,annotation_contraction[:len(audio_o)], color='tab:green', linestyle='-', linewidth=20, alpha=0.4)
# ax2.plot(time_steps,annotation_needle[:len(audio_o)], color='tab:pink', linestyle='-', linewidth=20, alpha=0.4)
# ax2.plot(time_steps,annotation_non_analysable[:len(audio_o)], color='tab:purple', linestyle='-', linewidth=20, alpha=0.4)

# plt.xlabel("Time [s]")
# plt.show()

# plt.savefig('signal.png')

