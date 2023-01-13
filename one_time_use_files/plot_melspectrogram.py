import numpy as np
import librosa, librosa.display
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
from utils.data_augmentation import Skew, Distort
from PIL import Image
import pyloudnorm as pyln

def normalise_db(data, fs):
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
    loudness_normalized_audio = pyln.normalize.loudness(data, loudness, -26)
    return loudness_normalized_audio

distort = Distort(0.25, 16, 16, 1) #same as in the models
skew = Skew(0.25, 'TILT', 0.5) #same as in the models


filepath = askopenfilename(
            filetypes=[("Npy Files", "*.npy"), ("All Files", "*.*")])
data = np.load(filepath)
samplerate = 44100

'''
fig, ax= plt.subplots(1,1)
ax.plot(np.arange(0, len(data)) / samplerate, data, color='k')
ax.set_ylim(-200,200)
plt.show()
'''

data = normalise_db(data, samplerate)
input_dimensiont_width =  np.int(np.ceil(2* samplerate / 512))

spect = librosa.stft(y=data)
spect = librosa.amplitude_to_db(spect, ref=np.max)
fig,ax=plt.subplots()
img = librosa.display.specshow(spect, sr=samplerate, hop_length=512, x_axis='time', y_axis='linear', ax=ax)
fig.colorbar(img, ax=ax)
plt.xlabel("Time")
plt.ylabel("Frequency")

mel_spect = librosa.feature.melspectrogram(y=data, sr=samplerate , n_mels=128, fmax=5000, fmin=500, hop_length=512)
#mel_spect = np.asarray(mel_spect).reshape(128, input_dimensiont_width)
mel_spect = librosa.power_to_db(mel_spect, ref=np.max)
fig,ax=plt.subplots()
img = librosa.display.specshow(mel_spect, sr=samplerate, hop_length=512, x_axis='time', y_axis='mel', ax=ax)
fig.colorbar(img, ax=ax)
plt.xlabel("Time")
plt.ylabel("Frequency")

max_mel_spect = np.max(mel_spect)
min_mel_spect = np.min(mel_spect)
mel_spect = (mel_spect - min_mel_spect) / (max_mel_spect - min_mel_spect)

im = Image.fromarray(np.uint8(mel_spect*255))
skew = skew.perform_operation(im)
distort = distort.perform_operation(im)
# Convert back to numpy and normalise.
mel_spect_skew = np.array(skew) / 255.0
mel_spect_distort = np.array(distort) /255.0

fig,ax=plt.subplots()
img = librosa.display.specshow(mel_spect_skew, sr=samplerate, hop_length=512, x_axis='time', y_axis='mel', ax=ax)
fig.colorbar(img, ax=ax)
plt.xlabel("Time")
plt.ylabel("Frequency")


fig,ax=plt.subplots()
img = librosa.display.specshow(mel_spect_distort, sr=samplerate, hop_length=512, x_axis='time', y_axis='mel', ax=ax)
fig.colorbar(img, ax=ax)
plt.xlabel("Time")
plt.ylabel("Frequency")



plt.figure(6)
plt.subplot(1,3,1)
librosa.display.specshow(mel_spect, sr=44100, hop_length=512)
plt.subplot(1,3,2)
librosa.display.specshow(mel_spect_skew, sr=44100, hop_length=512)
plt.subplot(1,3,3)
librosa.display.specshow(mel_spect_distort, sr=44100, hop_length=512)
plt.colorbar()
plt.show()

