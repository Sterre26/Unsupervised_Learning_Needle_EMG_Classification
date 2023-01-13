# standard libraries 
import numpy as np
# sklearn
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
# tensorflow & keras
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
# plots 
from matplotlib import pyplot as plt
"""
This file enables you to make the kmeans elbow plot, which is used to determine the optimal number of clusters. 
Documentation: https://www.geeksforgeeks.org/elbow-method-for-optimal-value-of-k-in-kmeans/, https://stackoverflow.com/questions/19197715/scikit-learn-k-means-elbow-criterion
"""

data_training='/Users/Sterre/files/PROCESSED/mels_size_square/Training_100percent_REST'
dim = (128,128)
batch_size = 256

autoencoder = keras.models.load_model('/Users/Sterre/_Python_folders/ai_needle_emg_internshipSterre/results/arch_one/20102022_1339/convae_model_final.hdf5')

datagen = ImageDataGenerator(rescale=1./255) 
train_data = datagen.flow_from_directory(
        directory = data_training, 
        target_size = dim, 
        batch_size = batch_size,
        class_mode = 'input', 
        color_mode = 'grayscale', 
        shuffle = True
        )

# RETRIEVE EMBEDDING from CONVOLUTIONAL AUTOENCODER
feature_model = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer(name='embedding').output)
features_train = feature_model.predict(train_data)
features_train = np.reshape(features_train, newshape=(features_train.shape[0], -1))

distortions, inertias = [], []
mapping1, mapping2 = {}, {}
K = range(1, 20)

for k in K:
    kmeanModel = KMeans(n_clusters=k, n_init=100).fit(features_train)
    # kmeanModel.fit(features_train)

    distortions.append(sum(np.min(cdist(features_train, kmeanModel.cluster_centers_, 
                                        'euclidean'), axis=1)) / features_train.shape[0])
    inertias.append(kmeanModel.inertia_)
    mapping1[k] = sum(np.min(cdist(features_train, kmeanModel.cluster_centers_,
                           'euclidean'), axis=1)) / features_train.shape[0]
    mapping2[k] = kmeanModel.inertia_

fig, (ax1,ax2) = plt.subplots(2,1)

ax1.plot(K, distortions, 'bx-')
ax1.set(ylabel='Distortion')
ax1.set_title('The Elbow Method using Distortion')

ax2.plot(K, inertias, 'gx-')
ax2.set(xlabel='Values of K', ylabel='Inertia')
ax2.set_title('The Elbow Method using Inertia')
plt.show()