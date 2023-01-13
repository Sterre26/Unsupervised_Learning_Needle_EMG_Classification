import os
from scipy.io import wavfile
import numpy as np
import librosa, librosa.display
import matplotlib.pyplot as plt
import pyloudnorm as pyln
import seaborn as sns
sns.reset_orig()
from scipy import signal
import math
from tqdm import tqdm

from create_mels import normalise_db, create_mels


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

    delta_loudness = loudness_normalise_level - loudness
    gain = np.power(10.0, delta_loudness/20.0)

    output = gain * data

    # sns.set_style('whitegrid')
    # ax = sns.histplot(output.flatten(), bins=1000, kde=True, stat='probability')
    # ax.set(title='Data without clipping at {}'.format(loudness_normalise_level))
    # plt.show()

    # check for potentially clipped samples
    # if np.max(np.abs(output)) >= 1.0:
    #     print(np.max(np.abs(output)))


    return loudness_normalized_audio



path_to_files = '/Users/Sterre/files/WAV/Training_unlabelled/'

file_list = [path_to_files + f for f in os.listdir(path_to_files) if ".wav" in f]
file_list.sort()

# file_list = file_list[:500]

# create mel
fmax = 10000
n_mels = 128
hop_length = 512

ref_max = []
ref_median = []


for i, file in tqdm(enumerate(file_list)):

    filename_base = os.path.basename(file).split(".wav")[0]
    path_to_data = file

    # step 1: load raw data        
    fs, data = wavfile.read(path_to_data)   
    
    sample_rate = 44100
    data = data.astype('float')

    






    # max_time = 60 # maximal numbers of seconds to show!
    # data_len = round(max_time * sample_rate)
    # time_steps = np.linspace(0, max_time, data_len)
    # fig4 = plt.figure(4)
    # fig4.suptitle("Original audio file {} (top) and original file converted to mel to audio file (bottom)".format(wavfile))
    # plt.plot(time_steps, data[:data_len], color='k')
    # plt.xlabel("Time [s]")
    # plt.show()


    # sns.set_style('whitegrid')
    # ax = sns.histplot(data.flatten(), bins=1000, kde=True, stat='probability')
    # ax.set(title='Raw data')
    # plt.show()

    # # step 2: normalise data to [0,1] so that clipping does not occur > maakt dit uit? Er is geen clipping bij -40 dB zonder normalisatie
    # data = (data - np.min(data)) / (np.max(data) - np.min(data))

    # sns.set_style('whitegrid')
    # ax = sns.histplot(data.flatten(), bins=1000, kde=True, stat='probability')
    # ax.set(title='Data normalised to [0,1]')
    # plt.show()

    # step 3: normalise loudness level, pick loudness dependend on whether clipping occurs yes no
    loudness_normalise_level = -26.0
    data = normalise_db(data, fs, loudness_normalise_level=loudness_normalise_level)

    # sns.set_style('whitegrid')
    # ax = sns.histplot(data.flatten(), bins=1000, kde=True, stat='probability')
    # ax.set(title='Loudness normalised to {}'.format(loudness_normalise_level))
    # plt.show()

    # max_time = 10 # maximal numbers of seconds to show!
    # data_len = round(max_time * sample_rate)            
    # data = data[:data_len]
 
    # METHOD 3 : desired but how to make sure that output power_to_db does not clip it to -80 dB
    mel = librosa.feature.melspectrogram(y=data, sr=sample_rate, n_mels=n_mels, fmax=fmax, hop_length=hop_length)
    
    ref_max = ref_max + [np.max(mel)]
    ref_median = ref_median + [np.median(mel)]

    # mel = librosa.power_to_db(mel, ref=np.max, top_db=130.0)
    # mel = np.asarray(mel).reshape(n_mels, len(mel[0]))

    # # min max normalisatie
    # min_mel = np.min(mel)
    # max_mel = np.max(mel)
    # mel = (mel - min_mel) / (max_mel - min_mel)
    # mel = (mel * (255 - 0) + 0).astype(np.uint8)
 
    # if i == 0: mels = mel
    # else: mels = np.append(mels, mel, axis=1)

    

print(np.median(ref_max))
print(np.mean(ref_max))
sns.set_style('whitegrid')
ax = sns.histplot(ref_max, bins=1000, kde=True, stat='probability')
ax.set(xlabel='Ref max')
plt.show()

sns.set_style('whitegrid')
ax = sns.histplot(ref_median, bins=50, kde=True, stat='probability')
ax.set(xlabel='Ref median')
plt.show()


# img = mel
    

# fig2 = plt.figure(2)
# img2 = librosa.display.specshow(img, x_axis='time',
#                 y_axis='mel', sr=sample_rate,
#                 fmax=fmax, 
#                 cmap='magma')
# fig2.colorbar(img2, format='%+2.0f dB')
# fig2.suptitle('Mel-frequency spectrogram')

# # img = mel[:, :50]



# # fig3 = plt.figure(3)
# # img3 = librosa.display.specshow(img, x_axis='time',
# #                 y_axis='mel', sr=sample_rate,
# #                 fmax=fmax, 
# #                 cmap='magma')
# # fig3.colorbar(img3, format='%+2.0f dB')
# # fig3.suptitle('Mel-frequency spectrogram')



#

