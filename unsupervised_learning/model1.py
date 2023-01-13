# standard libraries 
import numpy as np
import numpy.matlib as matlib
import os
from time import time
import pandas as pd
# sklearn
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support, pairwise_distances_argmin
from sklearn.cluster import KMeans
# tensorflow & keras
from tensorflow import keras
import tensorflow_addons as tfa
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model, Sequential
from keras.utils.vis_utils import plot_model
from tqdm.keras import TqdmCallback
from keras.layers import Conv2D, Conv2DTranspose, Dense, Flatten, Reshape, BatchNormalization, LeakyReLU
from keras.callbacks import CSVLogger
from keras.utils.data_utils import Sequence
# hyperopt
from hyperopt import hp, Trials, tpe, fmin, STATUS_OK, space_eval
import pickle
# custom made imports 
from unsupervised_model_options import UnsupervisedOptions
from displays import display_confusionmatrix, display_clusters, display_example_predictions
from load_data import load_mels

"""
Training, evaluation and testing model 1 

@author: Sterre de Jonge (2022)
"""


class ConvAE_clustering:

    def CAE(input_shape, kernel_size, activation, filters, features, batch_norm):
        """ 
        Convolutional autoencoder is constructed (model is returned), independent of number 
        of layers. The stride is set at two so there is a maximum of number of layers (because
        the image size decreases by a factor two at each layer). 
        """

    
        model = Sequential()

        if input_shape[0] % 8 == 0:
            pad_last = 'same'
        else:
            pad_last = 'valid'

        padding = [None] * len(filters)
        for i in range(len(filters)):
            if i < len(filters) - 1:
                padding[i] = 'same'
            else:
                padding[i] = pad_last
        strides = [2] * len(filters)

        division_factor = input_shape[0] / (input_shape[0] / 2**len(filters))

        for i in range(len(filters)):
            if i < len(filters): 
                # convolution
                model.add(Conv2D(filters[i], kernel_size[i], strides[i], padding[i], activation=activation, name = 'conv' + str(i+1), input_shape=input_shape))
                if i < len(filters) -1 and batch_norm is True: model.add(BatchNormalization())

        input_layer = keras.layers.Input(shape=(128,128,1,))
        layer1 = Conv2D(filters=filters[0], kernel_size=kernel_size[0], strides=strides[0], padding=padding[0], activation=activation, name='conv1')(input_layer)
        batch_norm1 = BatchNormalization()(layer1)
        layer2 = Conv2D(filters=filters[1], kernel_size=kernel_size[1], strides=strides[1], padding=padding[1], activation=activation, name='conv2')(batch_norm1)
        batch_norm2 = BatchNormalization()(layer2)
        layer3 = Conv2D(filters=filters[2], kernel_size=kernel_size[2], strides=strides[2], padding=padding[2], activation=activation, name='conv3')(batch_norm2)

        flatten = Flatten()(layer3)
        dense1 = Dense(units=features, name='embedding')(flatten)
        dense2 = Dense(units=filters[-1] * int(input_shape[0]/division_factor)*int(input_shape[0]/division_factor), activation = activation)(dense1)
        reshape = Reshape((int(input_shape[0]/division_factor), int(input_shape[0]/division_factor), filters[-1]))(dense2)

        model.add(Flatten())
        model.add(Dense(units=features, name='embedding'))
        model.add(Dense(units=filters[-1] * int(input_shape[0]/division_factor)*int(input_shape[0]/division_factor), activation = activation))
        model.add(Reshape((int(input_shape[0]/division_factor), int(input_shape[0]/division_factor), filters[-1])))

        filters = filters[::-1]
        filters = filters[1:] 
        kernel_size = kernel_size[::-1]
        padding = padding[::-1]
        for i in range(len(filters)):
            if i < len(filters): 
                # deconvolution
                model.add(Conv2DTranspose(filters[i], kernel_size[i], strides[i], padding[i], activation=activation, name = 'deconv' + str(len(filters)-i+1), input_shape=input_shape))
                if batch_norm is True: model.add(BatchNormalization())

        deconv1 = Conv2DTranspose(filters=filters[0], kernel_size=kernel_size[0], strides=strides[0], padding=padding[0], activation=activation, name='deconv1')(reshape)
        batch_norm3 = BatchNormalization()(deconv1)
        deconv2 = Conv2DTranspose(filters=filters[1], kernel_size=kernel_size[1], strides=strides[1], padding=padding[1], activation=activation, name='deconv2')(batch_norm3)
        batch_norm4 = BatchNormalization()(deconv2)
        deconv3 = Conv2DTranspose(input_shape[2], 5, strides=2, padding='same', name='deconv3')(batch_norm4)
        
        model.add(Conv2DTranspose(input_shape[2], 5, strides=2, padding='same', name='deconv1'))

        model2 = Model(inputs=input_layer, outputs=deconv3)
        
        return model

class Validation(object):

    def __init__(self, clusters, show_displays, save_directory, datatype=None):
        
        self.clusters = clusters
        self.show_displays = show_displays
        self.save_directory = save_directory
        self.previous_cluster_centers = None
        self.data_type = datatype

    def clustering(self, features_train, features_test):        
        km = KMeans(n_clusters = self.clusters, n_init=200)
        self.km_trained = km.fit(features_train)
               
        y_pred_train = self.km_trained.predict(features_train)
        y_pred_test = self.km_trained.predict(features_test)

        delta_label = None

        if self.previous_cluster_centers is not None:
            # change order y_pred_train based on similarity between previous cluster centers
            order = pairwise_distances_argmin(self.previous_cluster_centers, self.km_trained.cluster_centers_)
            
            cluster_centers_new = self.km_trained.cluster_centers_[order]
            if len(set(order)) == len(order): # only if differences were found for all clusters
                y_pred_train = pairwise_distances_argmin(features_train, cluster_centers_new)
                y_pred_train_previous = pairwise_distances_argmin(self.previous_features_train, self.previous_cluster_centers)
                # compute delta label 
                delta_label = np.sum(y_pred_train != y_pred_train_previous).astype(np.float32) / y_pred_train.shape[0]
            
        self.previous_cluster_centers = np.copy(self.km_trained.cluster_centers_)
        self.previous_features_train = np.copy(features_train)

        return y_pred_train, y_pred_test, delta_label

    def clustering_rest(self, features_rest):
        y_pred_rest = self.km_trained.predict(features_rest)
        return y_pred_rest

    def validate(self, y_true, y_pred, labels, argument=None):
        """
        This function computes testing metrics.  
        """
        self.cm = confusion_matrix(y_true, y_pred) # confusion matrix for all clusters
        self.cm_argmax = self.cm.argmax(axis=0) 
        if not set(self.cm_argmax) == set(y_true): 
            # check condition that all all true labels have been assigned a cluster
            return None, None, None, None
        if not len(self.cm_argmax) == len(set(y_pred)):
            # check condition that all clusters are being used
            return None, None, None, None
        y_pred_argmax = np.array([self.cm_argmax[k] for k in y_pred]) # values 0,1,2 instead of 0,1,2,3,etc.
        cm_small = confusion_matrix(y_true, y_pred_argmax) # confusion matrix for all classes
        acc = accuracy_score(y_true, y_pred_argmax)
        precision, recall, fscore, support = precision_recall_fscore_support(y_true, y_pred_argmax, labels = [0,1,2]) # label 2 = rest

        if argument == 'test': 
            display_confusionmatrix(self.cm, cm_small, labels, self.clusters, self.save_directory, show=self.show_displays, soft=False)
            print("soft=False")
            print(self.cm)
        if argument == 'test_soft': 
            display_confusionmatrix(self.cm, cm_small, labels, self.clusters, self.save_directory, show=self.show_displays, soft=True)
            print("soft=True")
            print(self.cm)

        return acc, precision, recall, fscore, support

    def soft_clustering_weights(data, cluster_centres, **kwargs):
        """
        Function to calculate the weights from soft k-means (source: https://towardsdatascience.com/confidence-in-k-means-d7d3a13ca856)
        data: Array of data. shape = N x F, for N data points and F Features
        cluster_centres: Array of cluster centres. shape = Nc x F, for Nc number of clusters. Input kmeans.cluster_centres_ directly.
        param: m - keyword argument, fuzziness of the clustering. Default 2
        """
        # Fuzziness parameter m>=1. Where m=1 => hard segmentation
        m = 2
        if 'm' in kwargs:
            m = kwargs['m']
        
        Nclusters = cluster_centres.shape[0]
        Ndp = data.shape[0]
        Nfeatures = data.shape[1]

        # Get distances from the cluster centres for each data point and each cluster
        EuclidDist = np.zeros((Ndp, Nclusters))
        for i in range(Nclusters):
            EuclidDist[:,i] = np.sum((data-matlib.repmat(cluster_centres[i], Ndp, 1))**2,axis=1)
        
        # Denominator of the weight from wikipedia:
        invWeight = EuclidDist**(2/(m-1))*matlib.repmat(np.sum((1./EuclidDist)**(2/(m-1)),axis=1).reshape(-1,1),1,Nclusters)
        Weight = 1./invWeight
        
        return Weight

    def get_optimal_confidence(self, features, y_pred, cluster_centres=None):

        if cluster_centres is None: cluster_centres = self.km_trained.cluster_centers_
        if cluster_centres is not None: cluster_centres = cluster_centres

        probabilities_range = []
        softDF= pd.DataFrame()
        softDF['y_pred'] = y_pred
        for j in range(self.clusters):
            softDF['p' + str(j)] = 0
            probabilities_range.append('p%d' % j)
        softDF[probabilities_range] = Validation.soft_clustering_weights(features, cluster_centres)

        confidence_range = range(0, 1000, 1)

        list_p0_combined, list_p1_combined, list_p2_combined = [], [], []

        for index_df, item in enumerate(softDF['y_pred']):
            index_p0 = [i for i, val in enumerate(self.cm_argmax) if val==0]
            list_p0_combined.append(sum([softDF['p' + str(i)][index_df] for i in index_p0]))
            index_p1 = [i for i, val in enumerate(self.cm_argmax) if val==1]
            list_p1_combined.append(sum([softDF['p' + str(i)][index_df] for i in index_p1]))
            index_p2 = [i for i, val in enumerate(self.cm_argmax) if val==2]
            list_p2_combined.append(sum([softDF['p' + str(i)][index_df] for i in index_p2]))

        softDF['p0_combined'] = list_p0_combined
        softDF['p1_combined'] = list_p1_combined
        softDF['p2_combined'] = list_p2_combined

        softDF['confidence'] = np.max(softDF[['p0_combined', 'p1_combined', 'p2_combined']].values, axis = 1)

        perc_removed_list = []

        for confidence in confidence_range:
            confidence = confidence / 1000

            y_pred_soft = softDF.loc[softDF['confidence'] > confidence, 'y_pred']
            perc_removed = 100 - ((len(y_pred_soft) / len(y_pred)) * 100)

            if confidence == 0: df1 = pd.DataFrame({"confidence": [confidence], "percentage_removed": [perc_removed]})
            else: 
                df2 = pd.DataFrame({"confidence": [confidence], "percentage_removed": [perc_removed]})
                df1 = pd.concat([df1, df2])

        # find the closest value where 'input' is removed. 
        input = 25
        df1_closest = df1.iloc[(df1['percentage_removed']-input).abs().argsort()[:1]]
        closest_value = df1_closest.iloc[0]['confidence']
        perc_removed = df1_closest.iloc[0]['percentage_removed']
        return closest_value, perc_removed

    def validate_soft_clustering(self, features, y_pred, confidence, y_true, cluster_centres=None):

        if cluster_centres is None: cluster_centres = self.km_trained.cluster_centers_
        if cluster_centres is not None: cluster_centres = cluster_centres

        probabilities_range = []
        softDF= pd.DataFrame()
        softDF['y_pred'] = y_pred
        if y_true is not None: softDF['y_true'] = y_true
        for j in range(self.clusters):
            softDF['p' + str(j)] = 0
            probabilities_range.append('p%d' % j)
        softDF[probabilities_range] = Validation.soft_clustering_weights(features, cluster_centres)

        list_p0_combined, list_p1_combined, list_p2_combined = [], [], []
        
        for index_df, item in enumerate(softDF['y_pred']):
            index_p0 = [i for i, val in enumerate(self.cm_argmax) if val==0]
            list_p0_combined.append(sum([softDF['p' + str(i)][index_df] for i in index_p0]))
            index_p1 = [i for i, val in enumerate(self.cm_argmax) if val==1]
            list_p1_combined.append(sum([softDF['p' + str(i)][index_df] for i in index_p1]))
            index_p2 = [i for i, val in enumerate(self.cm_argmax) if val==2]
            list_p2_combined.append(sum([softDF['p' + str(i)][index_df] for i in index_p2]))

        softDF['p0_combined'] = list_p0_combined
        softDF['p1_combined'] = list_p1_combined
        softDF['p2_combined'] = list_p2_combined

        softDF['confidence'] = np.max(softDF[['p0_combined', 'p1_combined', 'p2_combined']].values, axis = 1)

        y_pred_soft = softDF.loc[softDF['confidence'] > confidence, 'y_pred']
        if y_true is not None: y_true_soft = softDF.loc[softDF['confidence'] > confidence, 'y_true']

        # remove indices where confidence is not reached from 'features_test'
        features_soft=None
        if y_true is not None:
            for index in y_pred_soft.index:
                features_soft_sample = features[index]
                features_soft_sample = np.reshape(features_soft_sample, newshape=(-1, features_soft_sample.shape[0]))
                if features_soft is not None: features_soft = np.append(features_soft, features_soft_sample, axis=0)
                if features_soft is None: features_soft = features_soft_sample
            features_soft = np.reshape(features_soft, newshape=(features_soft.shape[0], -1))

        # percentage removed soft clustering 
        perc_removed = 100 - ((len(y_pred_soft) / len(y_pred)) * 100)
        print("Percentage removed from test/validation data is %.0f" % (perc_removed))

        if y_true is None: return y_pred_soft.index, y_pred_soft
        if y_true is not None: return features_soft, y_pred_soft, y_true_soft

class Hyperopt(object):

    def __init__(self, save_dir, trials_dir, search_space, save_interval_trials, max_trials):
        
        self.save_dir = save_dir
        self.trials_dir = trials_dir
        
        self.search_space = search_space
        self.save_interval_trials = save_interval_trials
        self.max_trials = max_trials 

        self.trials = None

    def model_optimisation(params):
        print('\nHyperparameter optimalisation is performed with following parameters: ', params)

        # load data
        datagen = ImageDataGenerator(rescale=1./255) 
        train_data = datagen.flow_from_directory(
                    directory = directory_training, 
                    target_size = (128,128),
                    batch_size = params['learning']['batchsize'],
                    class_mode = 'input', 
                    color_mode = 'grayscale', 
                    shuffle = False 
            )
        val_data = datagen.flow_from_directory(
                    directory = config.validation_signaltype_70procent, 
                    target_size = (128,128), 
                    batch_size = params['learning']['batchsize'],
                    class_mode = 'categorical', 
                    color_mode = 'grayscale', 
                    shuffle = False 
                    )
        y_true = val_data.classes 

        autoencoder = ConvAE_clustering.CAE(input_shape=(128,128,1), 
                                            kernel_size=params['architecture']['kernel_size'], 
                                            activation=params['activation'], 
                                            filters=params['architecture']['filters'], 
                                            features=params['features'], 
                                            batch_norm=params['batch_normalisation'])
        
        if params['learning']['schedule'] == 'constant_256':
                opt = keras.optimizers.Adam(params['learning']['rate_1'])
        if params['learning']['schedule'] == 'constant_64':
                opt = keras.optimizers.Adam(params['learning']['rate_5'])
        if params['learning']['schedule'] == 'clr_tria1_256':
            clr = tfa.optimizers.CyclicalLearningRate(initial_learning_rate=params['learning']['rate_7'][0], 
                                                    maximal_learning_rate=params['learning']['rate_7'][1], 
                                                    scale_fn=lambda x: 1.0,
                                                    step_size= (train_data.samples / config.batch_size) * 2) 
            opt = keras.optimizers.Adam(clr)
        if params['learning']['schedule'] == 'clr_tria1_64':
            clr = tfa.optimizers.CyclicalLearningRate(initial_learning_rate=params['learning']['rate_8'][0], 
                                                    maximal_learning_rate=params['learning']['rate_8'][1], 
                                                    scale_fn=lambda x: 1.0,
                                                    step_size= (train_data.samples / config.batch_size) * 2) 
            opt = keras.optimizers.Adam(clr)
        if params['learning']['schedule'] == 'clr_tria2_256':
            clr = tfa.optimizers.CyclicalLearningRate(initial_learning_rate=params['learning']['rate_4'][0], 
                                                    maximal_learning_rate=params['learning']['rate_4'][1], 
                                                    scale_fn=lambda x: 1/(2.**(x-1)), 
                                                    step_size= (train_data.samples / config.batch_size) * 2)
            opt = keras.optimizers.Adam(clr)
        if params['learning']['schedule'] == 'clr_tria2_64':
            clr = tfa.optimizers.CyclicalLearningRate(initial_learning_rate=params['learning']['rate_6'][0], 
                                                    maximal_learning_rate=params['learning']['rate_6'][1], 
                                                    scale_fn=lambda x: 1/(2.**(x-1)), 
                                                    step_size= (train_data.samples / config.batch_size) * 2)
            opt = keras.optimizers.Adam(clr)
        
        autoencoder.compile(optimizer=opt, loss=params['loss']) 

        # begin training 
        autoencoder.fit(train_data, 
                        epochs=params['epochs'], 
                        verbose=0, 
                        callbacks=[TqdmCallback(verbose=1)]
                        )

        feature_model = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer(name='embedding').output)
        features_test = feature_model.predict(val_data) 
        features_test = np.reshape(features_test, newshape=(features_test.shape[0], -1))
        features_train = feature_model.predict(train_data)
        features_train = np.reshape(features_train, newshape=(features_train.shape[0], -1))

        km = KMeans(n_clusters = params['clusters'], 
                    n_init=100)
        km_fitted = km.fit(features_train)
        y_pred = km_fitted.predict(features_test)

        cm = confusion_matrix(y_true, y_pred)
        cm_argmax = cm.argmax(axis=0) 
        y_pred = np.array([cm_argmax[k] for k in y_pred]) 
        acc = accuracy_score(y_true, y_pred)

        precision, recall, fscore, support = precision_recall_fscore_support(y_true, y_pred, labels = [0,1,2]) # label 2 = rest
        fscore_rest = fscore[2]

        print('Training for one trial is finished with accuracy: %.4f and f1 rest %.4f' % (acc, fscore_rest))

        # add time when using the smaller search space :)

        return {'loss': -fscore_rest, 'metrics': [acc, precision, recall, fscore], 'status': STATUS_OK}

    def run_trials(self):
        """ This function is called in a for loop to ensure that trials file is periodically saved. """

        if self.trials_dir is not None and self.trials is not None:
            print("Trials is reused from previous run!") # to make sure that trials file is not loaded multiple times
            temp_max_trials = len(self.trials.trials) + self.save_interval_trials
            if temp_max_trials > self.max_trials: 
                print("Maximum number of trials ({}) has already been reach with lenght trials: {}".format(self.max_trials, len(self.trials.trials)))
                return 
            else: print("Rerunning from {} trials to {}".format(len(self.trials.trials), temp_max_trials))
        if self.trials_dir is not None and self.trials is None:
            print("Found saved trials! Loading...")
            self.trials = pickle.load(open(self.trials_dir, "rb"))
            temp_max_trials = len(self.trials.trials) + self.save_interval_trials
            if temp_max_trials > self.max_trials: 
                print("Maximum number of trials ({}) has already been reach with lenght trials: {}".format(self.max_trials, len(self.trials.trials)))
                return
            else: print("Rerunning from {} trials to {}".format(len(self.trials.trials), temp_max_trials))
        if self.trials_dir is None:
            print("\nHyperoptimization of the model is started...")
            self.trials = Trials()
            temp_max_trials = self.save_interval_trials
    
        best = fmin(Hyperopt.model_optimisation, self.search_space, algo=tpe.suggest, max_evals=temp_max_trials, trials=self.trials)
        print("Best:", best)

        self.trials_dir = self.save_dir + 'trials.p'
        with open(self.trials_dir, "wb") as f:
            pickle.dump(self.trials, f)
            print("Trials is saved to", self.trials_dir)

class Callback_CAE(keras.callbacks.Callback):
    def __init__(self, train_data, val_data, y_true, labels, n_clusters, datatype, confidence, save_directory, soft_metrics):
        super(Callback_CAE, self).__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.y_pred_last = None
        self.y_true = y_true
        self.labels = labels
        self.datatype = datatype
        self.validation = Validation(clusters=n_clusters, datatype=self.datatype, show_displays=False, save_directory=False)
        self.confidence = confidence
        self.save_directory = save_directory
        self.soft_metrics = soft_metrics

        # create models folder
        if not os.path.exists(self.save_directory + 'models/'): os.makedirs(self.save_directory + 'models/')
        
    def on_epoch_end(self, epoch, logs={}):

        print(" ...End epoch {} of training".format(epoch+1))
        logs['epoch'] = epoch + 1
        logs['train_loss'] = logs['loss']

        latent_space_model = Model(inputs=self.model.input, outputs=self.model.get_layer(name='embedding').output)

        val_loss = self.model.evaluate(x=self.val_data, verbose=0) 
        logs['val_loss'] = val_loss
        
        features_test = latent_space_model.predict(self.val_data) 
        features_test = np.reshape(features_test, newshape=(features_test.shape[0], -1))

        features_train = latent_space_model.predict(self.train_data)
        features_train = np.reshape(features_train, newshape=(features_train.shape[0], -1))
        if self.val_data is not None: y_pred_train, y_pred_val, delta_label = self.validation.clustering(features_train, features_test)
        else: delta_label = None

        # Training metrics
        if delta_label is not None:
            logs['deltalabel'] = delta_label 
            delta_label= round(delta_label,3)
        print("Training metrics at epoch %s:        delta_label=%s, training loss=%.4f, validation loss=%.4f" % (logs['epoch'], delta_label, logs['train_loss'], logs['val_loss']))
        if delta_label is None: 
            print("Delta_label could not be computed because clustering centers could not be compared with previous runs.")
            logs['deltalabel'] = np.nan # volgens mij nan maar ik weet het niet zeker (ipv None)

        # Testing metrics
        acc, precision, recall, fscore, support = self.validation.validate(self.y_true, y_pred_val, self.labels, argument='train')
        if acc is None: 
            print("Performance could not be obtained because either: not all distinct labels received a designated cluster, or not all clusters where filled with predicted samples.")
            logs['acc'] = np.nan
            logs['f1_rest'] = np.nan
            logs['f1_contraction'] = np.nan
            logs['f1_artefact'] = np.nan
            logs['acc_soft'] = np.nan
            logs['f1_rest_soft'] = np.nan
            logs['f1_contraction_soft'] = np.nan
            logs['f1_artefact_soft'] = np.nan
    
        if acc is not None:
            fscore_artefact, fscore_contraction, fscore_rest = fscore[0], fscore[1], fscore[2]
            logs['acc'] = acc
            logs['f1_rest'] = fscore_rest
            logs['f1_contraction'] = fscore_contraction
            logs['f1_artefact'] = fscore_artefact
        
            print('Testing metrics at epoch %s:         acc=%.3f, F1 rest=%.3f' % (logs['epoch'], acc, fscore_rest))

            if self.soft_metrics is True: 
                confidence, perc_removed = self.validation.get_optimal_confidence(features_train, y_pred_train)
                print(confidence)

                features_test_soft, y_pred_test_soft, y_true_soft = self.validation.validate_soft_clustering(features_test, y_pred_val, confidence, y_true=self.y_true)
                acc_soft, precision_soft, recall_soft, fscore_soft, support = self.validation.validate(y_true_soft, y_pred_test_soft, self.labels, argument='train')
                fscore_artefact_soft, fscore_contraction_soft, fscore_rest_soft = fscore_soft[0], fscore_soft[1], fscore_soft[2]
                logs['acc_soft'] = acc_soft
                logs['f1_rest_soft'] = fscore_rest_soft
                logs['f1_contraction_soft'] = fscore_contraction_soft
                logs['f1_artefact_soft'] = fscore_artefact_soft
                logs['confidence'] = confidence
                logs['perc_removed_train'] = perc_removed
                logs['percentage_removed_val'] = 100 - ((len(y_pred_test_soft) / len(y_pred_val)) * 100)
                print('Soft metrics with confidence %s:  acc=%.3f, F1 rest=%.3f' % (confidence, acc_soft, fscore_rest_soft))

        print('saving model to:', self.save_directory + 'models/convae_model_' + str(logs['epoch']) + '.hdf5 \n')
        self.model.save(self.save_directory + 'models/convae_model_' + str(logs['epoch']) + '.hdf5')

class BalancedDataGenerator(Sequence):
    """
    ImageDataGenerator + RandomOversampling from directory
    Given a directory structure comtaining subdirectories for images divided into 
    classes, e.g.
    DATA/TRAIN
         ├───ClassA
         |   |---- A1.png
         |   |---- A2.png
         |   ...
         └───ClassB
             |---- B1.png
             |---- B2.png
             ...
    Generates batches with minority classes oversampled.
    Parameters:
    data_path: root of the data directories (e.g., DATA/TRAIN)
    classes: list of classes (e.g., ['COVID', 'normal'])
    datagen: instance of Keras ImageDataGenerator. Can include data
             augmentation options.
    target_size: H/W target dimensions of images. Tuple. (e.g., (299, 299))
    batch_size: batch size
    class_mode: class mode for Keras' flow_from_dataframe method. E.g.,
                'binary', 'category', etc. See Keras documentation.
    Reference:
    https://medium.com/analytics-vidhya/how-to-apply-data-augmentation-to-deal-with-unbalanced-datasets-in-20-lines-of-code-ada8521320c9
    
    """
    def __init__(self, data_path, classes, datagen, target_size, batch_size=32, class_mode='categorical'):
        self.data_path = data_path
        self.classes = classes
        self.datagen = datagen
        self.target_size = target_size
        self.class_mode = class_mode

        # We will generate parallel arrays for filenames and labels
        X, y = self.__inventory__()

        self.batch_size = min(batch_size, X.shape[0])
        self.gen, self.steps_per_epoch = balanced_batch_generator(X.reshape(X.shape[0], -1), y, sampler=RandomOverSampler(), batch_size=self.batch_size, keep_sparse=True)
        self._shape = (self.steps_per_epoch * batch_size, *X.shape[1:])
        
    def __len__(self):
        return self.steps_per_epoch

    def __getitem__(self, idx):
        x_batch, y_batch = self.gen.__next__()
        x_batch = x_batch.reshape(-1)

        # Generate a pandas dataframe with x_batch and y_batch as columns
        df = pd.DataFrame()
        df['image'] = x_batch
        df['label'] = y_batch
       
        # With that, load the batch using the flow_from_dataframe method 
        return self.datagen.flow_from_dataframe(df, 
                                                directory=self.data_path,
                                                x_col='image',
                                                y_col='label',
                                                target_size=self.target_size,
                                                classes=self.classes,
                                                class_mode=self.class_mode,
                                                batch_size=self.batch_size, 
                                                color_mode="grayscale",
                                                shuffle=False).next()

    def __inventory__(self):
        '''
        Help function to create X and y
        '''
        X = []
        y = []

        for label in self.classes:
            class_directory = os.path.join(self.data_path, label)
            class_files = os.listdir(class_directory)
            for image in class_files:
                class_file = os.path.join(label, image)
                X.append(class_file)
            y += [label] * len(class_files)

        return np.array(X), np.array(y)

if __name__ == "__main__":

    config = UnsupervisedOptions().parse()

    if config.architecture != "arch_one": #or config.architecture != 'arch_one_rest': 
        print("Wrong architecture type is given in configuration file.")
        exit()

    print("\nMain file is running...")
    print("Configuration settings are stored in %s\n" % config.save_directory)

    if config.data == '5 percent': directory_training = config.training_5p
    if config.data == '100 percent': directory_training = config.training_100p
    if config.data == 'rest':
        directory_training = config.training_rest
        directory_validation = config.validation_rest
    else: directory_validation = config.validation_signaltype_70procent
    
    # remove files with 'tong' in the name from directory (independent of training/testing/hyperopt settings)
    file_list_tong = [directory_training + '/train/' + f for f in os.listdir(directory_training+ '/train') if "tong" in f]
    if len(file_list_tong) == 0: print("Files recorded from tongue were already deleted.")
    else: 
        print("Deleting {} files recorded from tongue in directory {}".format(len(file_list_tong), directory_training))
        for file in file_list_tong: os.remove(file)
        file_list_tong = [directory_training + f for f in os.listdir(directory_training) if "tong" in f]
        if len(file_list_tong) < 1: print("Files are successfully deleted\n")

    # LOAD DATA
    datagen = ImageDataGenerator(rescale=1./255) 
    print("Data generator is initialised \n")
    print("Train data")
    train_data = datagen.flow_from_directory(
                directory = directory_training, 
                target_size = config.input_dimension, 
                batch_size = config.batch_size,
                class_mode = 'input', 
                color_mode = 'grayscale', 
                shuffle=False 
        )
    print("Validation data")    
    val_data = datagen.flow_from_directory(
                directory = directory_validation, 
                target_size = config.input_dimension, 
                batch_size = config.batch_size,
                class_mode = 'input', 
                color_mode = 'grayscale', 
                shuffle = False 
                ) 
    labels_val = list(val_data.class_indices.keys())
    y_true_val = val_data.classes
    
    print("Test data Signal type")
    test_data_st = datagen.flow_from_directory(
                directory = config.test_signaltype_30procent, 
                target_size = config.input_dimension, 
                batch_size = config.batch_size,
                class_mode = 'categorical', 
                color_mode = 'grayscale', 
                shuffle = False 
                )
    labels_test_st = list(test_data_st.class_indices.keys())
    y_true_test_st = test_data_st.classes
    
    print("")

    ## HYPERPARAMETER OPTIMIZATION
    if config.hyperopt is True or config.evaluate_hyperopt is True:

        search_space = {'learning': hp.choice('learning', [
                                    {
                                        'schedule': 'constant_256',
                                        'rate_1': hp.choice('rate_1', [0.001, 0.005, 0.0005, 0.0001, 0.01]),
                                        'batchsize': 256
                                        }, 
                                    {
                                        'schedule': 'clr_tria2_256', 
                                        'rate_4': [10**-5, 0.001],
                                        'batchsize': 256
                                    }, 
                                    {
                                        'schedule': 'constant_64',
                                        'rate_5': hp.choice('rate_5', [10**-6, 10**-5, 10**-4, 0.0005, 0.001, 0.00005, 0.005]),
                                        'batchsize': 64
                                        },
                                    {
                                        'schedule': 'clr_tria2_64', 
                                        'rate_6': [10**-7, 10**-4],
                                        'batchsize': 64
                                    }, 
                                    {
                                        'schedule': 'clr_tria1_256', 
                                        'rate_7': [10**-5, 0.001],
                                        'batchsize': 256
                                    },
                                    {
                                        'schedule': 'clr_tria1_64', 
                                        'rate_8': [10**-7, 10**-4],
                                        'batchsize': 64
                                    }, 
                                ]),
                            'architecture': hp.choice('architecture', [
                                {
                                    'type': 'threelayers',
                                    'kernel_size': hp.choice('kernel_size1', [[5, 5, 3], [7, 5, 3]]), 
                                    'filters': hp.choice('filters1', [[16, 32, 64], [32, 64, 128]])
                                    }, 
                                {
                                    'type': 'fourlayers',
                                    'kernel_size': hp.choice('kernel_size2', [[7, 5, 5, 3], [5, 5, 3, 3]]), 
                                    'filters': hp.choice('filters2', [[16, 32, 64, 128], [32, 64, 128, 256]])
                                    }, 
                                {
                                    'type': 'fivelayers',
                                    'kernel_size': hp.choice('kernel_size3', [[7, 5, 5, 3, 3], [7, 7, 5, 3, 3]]),
                                    'filters': hp.choice('filters3', [[16, 32, 64, 128, 256], [8, 16, 32, 64, 128]])
                                    }, 
                            ]),
                            'activation': hp.choice('activation', ['relu', LeakyReLU(alpha=0.1), LeakyReLU(alpha=0.2), LeakyReLU(alpha=0.3)]),
                            'batch_normalisation': hp.choice('batch_normalisation', [True, False]),
                            'features': hp.choice('features', [10, 12, 14, 16, 18, 20, 24, 28, 32, 36, 40, 48, 56, 64, 128, 256, 512, 1024, 2048, 4096]),  
                            # in text: the search range for features is relatively large, this is because we began with a smaller range but the result showed that 
                            # the best combination of hyperparameters was with the maximum nubmer of features. We therefore increased the range to be able to 
                            # where the optimum number of features lies. 
                            'epochs': hp.choice('epochs', [8]), 
                            'optimizer': hp.choice('optimizer', ['adam']), 
                            'loss': hp.choice('loss', ['mse']), 
                            'clusters': hp.choice('clusters', [5, 6, 7])
            }        

        if config.hyperopt is True:

            # initialise class 
            hyperopt = Hyperopt(save_dir=config.save_directory, 
                                trials_dir=config.hyperopt_directory, 
                                search_space=search_space, 
                                save_interval_trials=config.save_interval_trials, 
                                max_trials=config.max_trials)
            t0 = time()

            for i in range(0, config.max_trials, config.save_interval_trials):
                hyperopt.run_trials()

        # store hyperopt results in Excel file 
        if config.hyperopt_directory is not None and config.evaluate_hyperopt is True:
            print("Trials is loaded to create Excel file in", config.save_directory)
            trials = pickle.load(open(config.hyperopt_directory, "rb"))
            trials_sorted = sorted(trials.results, key=lambda x: x['loss'], reverse=False)
            df_trials = pd.DataFrame(columns=['f1_rest', 'accuracy', 'batch_size', 'learning_schedule', 'learning_rate','layers', 'filters', 
                                            'kernels','activation', 'batchnorm', 'loss', 'features', 'optimiser', 'clusters'])
            for t, trial in enumerate(trials):
                vals = trial.get('misc').get('vals')
                tmp = {}
                for k,v in list(vals.items()):
                    if v: tmp[k] = v[0]
                
                vals_trial = space_eval(search_space, tmp)
                for key in vals_trial['learning']:
                    if key.startswith('rate'):
                        if key == 'rate_1': learning_rate = '256 constant ' + str(vals_trial['learning'][key])
                        if key == 'rate_4': learning_rate = '256 clr2 ' + str(vals_trial['learning'][key])
                        if key == 'rate_5': learning_rate = '64 constant ' + str(vals_trial['learning'][key])
                        if key == 'rate_6': learning_rate = '64 clr2 ' + str(vals_trial['learning'][key])
                        if key == 'rate_7': learning_rate = '256 clr1 ' + str(vals_trial['learning'][key])
                        if key == 'rate_8': learning_rate = '64 clr1 ' + str(vals_trial['learning'][key])

                if trial['misc']['vals']['activation'] == [0]: activation = 'ReLU'
                if trial['misc']['vals']['activation'] == [1]: activation = 'leakyReLU alpha=0.1'
                if trial['misc']['vals']['activation'] == [2]: activation = 'leakyReLU alpha=0.2'
                if trial['misc']['vals']['activation'] == [3]: activation = 'leakyReLU alpha=0.3'

                df_trials.loc[t] = pd.Series({'f1_rest':abs(trial['result']['loss']), 
                                            'accuracy': trial['result']['metrics'][0], 
                                            'batch_size': vals_trial['learning']['batchsize'],
                                            'learning_schedule': vals_trial['learning']['schedule'],
                                            'learning_rate': learning_rate,
                                            'layers': vals_trial['architecture']['type'],
                                            'filters': vals_trial['architecture']['filters'],
                                            'kernels': vals_trial['architecture']['kernel_size'],
                                            'activation': activation,
                                            'batchnorm': vals_trial['batch_normalisation'],
                                            'loss': vals_trial['loss'],
                                            'features': vals_trial['features'],
                                            'optimiser': vals_trial['optimizer'],
                                            'clusters': vals_trial['clusters']
                                            })
            df_trials.to_excel(config.save_directory + 'Trials_excel.xlsx')

    ## TRAIN MODEL 
    if config.train is True:
        print("Training of the model is started...\n")
        t0 = time()

        # model 
        autoencoder = ConvAE_clustering.CAE(input_shape=(128, 128, 1), 
                                            kernel_size=config.kernel_size, 
                                            activation=config.activation, 
                                            filters=config.filters, 
                                            features=config.features, 
                                            batch_norm=config.batch_norm)
        plot_model(autoencoder, to_file=config.save_directory + 'model-architecture.png', show_shapes=True)
        # autoencoder.summary()

        if config.learning_schedule_convae == 'constant' and config.optimizer == 'adam': opt = keras.optimizers.Adam(config.learning_rate_convae)  
        if config.learning_schedule_convae == 'constant' and config.optimizer == 'sgd': opt = keras.optimizers.SGD(config.learning_rate_convae)  
        if config.learning_schedule_convae == 'clr1':
            clr = tfa.optimizers.CyclicalLearningRate(initial_learning_rate=config.learning_rate_convae[0], 
                                                    maximal_learning_rate=config.learning_rate_convae[1], 
                                                        scale_fn=lambda x: 1.0, 
                                                        step_size= (train_data.samples / config.batch_size) * 2) # step_size defines the duration of a single cycle, a size of 2 means you need a total of 4 iterations to complete one cycle 
            
            if config.optimizer == 'adam': opt = keras.optimizers.Adam(clr)
            if config.optimizer == 'sgd': opt = keras.optimizers.SGD(clr)        
        if config.learning_schedule_convae == 'clr2':
            clr = tfa.optimizers.CyclicalLearningRate(initial_learning_rate=config.learning_rate_convae[0], 
                                                    maximal_learning_rate=config.learning_rate_convae[1], 
                                                        scale_fn=lambda x: 1/(2.**(x-1)), 
                                                        step_size= (train_data.samples / config.batch_size) * 2) 
            
            if config.optimizer == 'adam': opt = keras.optimizers.Adam(clr)
            if config.optimizer == 'sgd': opt = keras.optimizers.SGD(clr)

        autoencoder.compile(optimizer=opt, loss=config.loss_convae)

        loss = autoencoder.fit(train_data, epochs=config.epochs_convae, callbacks=[Callback_CAE(train_data, val_data, y_true_val, labels_val, config.clusters, config.data, config.confidence, config.save_directory, soft_metrics=True), CSVLogger(config.save_directory + 'logs-convae.csv')])

    # TEST MODEL
    if config.test is True and config.location_model1_final is not None:

        tsne_bool = True
        makeprediction_bool = True

        t1 = time()

        print("Testing...\n")

        directory = '/Users/Sterre/_Python_folders/ai_needle_emg_internshipSterre/results/arch_one/14102022_0820_bestmodelhyperopt_final'

        validation_directory = (directory + '/evaluation')
        if not os.path.exists(validation_directory): os.makedirs(validation_directory)
        
        try:
            autoencoder = keras.models.load_model(str(config.location_model1_final))
            print("\nPre-trained model loaded succesfully.\n")
        except:
            print("\nCould not load pre-trained model. Try with clr learning rate?")
        # TODO: how to know and how to import a model with cyclical learning rate? Just don't know!
        
        # retrieve latent space from trained model
        latent_space_model = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer(name='embedding').output)
        features_train, features_test = latent_space_model.predict(train_data), latent_space_model.predict(test_data_st) 
        features_train, features_test = np.reshape(features_train, newshape=(features_train.shape[0], -1)), np.reshape(features_test, newshape=(features_test.shape[0], -1))

        # initalise validation object
        validation = Validation(clusters=config.clusters, datatype=config.data, show_displays=config.displays, save_directory=validation_directory)

        # predict
        y_pred_train, y_pred_test, deltalabel = validation.clustering(features_train, features_test)           

        # validate
        acc, precision, recall, fscore, support = validation.validate(y_true_test_st, y_pred_test, labels_test_st, argument='test')

        confidence, perc_removed_train = validation.get_optimal_confidence(features_train, y_pred_train)
        # predict for soft labels
        features_test_soft, y_pred_test_soft, y_true_soft = validation.validate_soft_clustering(features_test, y_pred_test, confidence, y_true_test_st)
        features_train_soft, y_pred_train_soft = validation.validate_soft_clustering(features_train, y_pred_train, confidence, y_true=None)

        indices_soft_y_pred_test, y_pred_test_soft_notused = validation.validate_soft_clustering(features_test, y_pred_test, confidence, y_true=None)

        y_pred_test_soft_clusters = []
        for index in indices_soft_y_pred_test:
            files_soft_sample = y_pred_test[index]
            y_pred_test_soft_clusters.append(files_soft_sample)
        
        # confusion matrix
        cm = confusion_matrix(y_true_test_st, y_pred_test) 
        cm_argmax = cm.argmax(axis=0) 
        y_pred_test = np.array([cm_argmax[k] for k in y_pred_test]) 
        y_pred_train = np.array([cm_argmax[k] for k in y_pred_train])

        cm_soft = confusion_matrix(y_true_soft, y_pred_test_soft_clusters) 
        print(cm_soft)
        cm_argmax_soft = cm_soft.argmax(axis=0)  
        y_pred_test_soft = np.array([cm_argmax_soft[k] for k in y_pred_test_soft])
        y_pred_train_soft = np.array([cm_argmax_soft[k] for k in y_pred_train_soft])

        # validate with soft clustering
        acc_soft, precision_soft, recall_soft, fscore_soft, support_soft = validation.validate(y_true_soft, y_pred_test_soft_clusters, labels_test_st, argument='test_soft')
        
        if tsne_bool is True:
            # display clusters
            display_clusters(features_test, y_true_test_st, y_pred_test, show=config.displays, save_dir=validation_directory, soft=False, num=8)
            display_clusters(features_test_soft, y_true_soft, y_pred_test_soft, show=config.displays, save_dir=validation_directory, soft=True, num=9)

        if makeprediction_bool is True:
            # MAKE PREDICTIONS with CONVOLUTIONAL AUTOENCODER and display them
            fileList_rest = \
                [config.validation_signaltype_70procent + '/Rest/' + file for file in os.listdir(config.validation_signaltype_70procent + '/Rest/') if '.png' in file]
            indices = np.random.randint(len(fileList_rest), size=10)
            fileList_rest_small = []
            for i in indices:
                fileList_rest_small.append(fileList_rest[i])     
            test_data_rest = load_mels(fileList_rest_small)
            fileList_contraction = \
                [config.validation_signaltype_70procent + '/Contraction/' + file for file in os.listdir(config.validation_signaltype_70procent + '/Contraction/') if '.png' in file]
            indices = np.random.randint(len(fileList_contraction), size=10)
            fileList_contraction_small = []
            for i in indices:
                fileList_contraction_small.append(fileList_contraction[i])     
            test_data_contraction = load_mels(fileList_contraction_small)
            fileList_artefact = \
                [config.validation_signaltype_70procent + '/Artefact/' + file for file in os.listdir(config.validation_signaltype_70procent + '/Artefact/') if '.png' in file]
            indices = np.random.randint(len(fileList_artefact), size=10)
            fileList_artefact_small = []
            for i in indices:
                fileList_artefact_small.append(fileList_artefact[i])     
            test_data_artefact = load_mels(fileList_artefact_small)

            display_example_predictions(test_data_rest, autoencoder.predict(test_data_rest), "rest", show=config.displays, save_dir=validation_directory, fig=1)
            display_example_predictions(test_data_contraction, autoencoder.predict(test_data_contraction), "contraction", show=config.displays, save_dir=validation_directory, fig=2)
            display_example_predictions(test_data_artefact, autoencoder.predict(test_data_artefact), "artefact", show=config.displays, save_dir=validation_directory, fig=3)

            batch = next(test_data_st)[0]
            predictions = autoencoder.predict_on_batch(batch)
            display_example_predictions(batch, predictions, "random", show=config.displays, save_dir=validation_directory, fig=4)

        print('\nTesting time: ', time() - t1)

        with open(validation_directory + "/performandce_metrics.txt", "w") as a:
            a.write("Logfile with results for model with location {} \n".format(config.location_model1_final))
            a.write("Signal Type validation data is used; 70 procent of dataset\n")
            a.write("\n")
            a.write("Performance without soft clustering\n")
            a.write("-----------------------------------\n")
            a.write("Accuracy                   \t {}\n".format(round(acc,3)))
            a.write("\n")
            a.write("                           \t Precision     \t\t Recall     \t\t F1 \t\t support\n")
            a.write("Rest metrics               \t {}            \t {}           \t {}      \t {}\n".format(round(precision[2],3), round(recall[2],3), round(fscore[2],3), round(support[2],3)))
            a.write("Contraction metrics        \t {}            \t {}           \t {}      \t {}\n".format(round(precision[1],3), round(recall[1],3), round(fscore[1],3), round(support[1],3)))
            a.write("Needle metrics             \t {}            \t {}           \t {}      \t {}\n".format(round(precision[0],3), round(recall[0],3), round(fscore[0],3), round(support[0],3)))
            a.write("\n")
            a.write("Performance with soft clustering\n")
            a.write("-----------------------------------\n")
            a.write("Confidence                 \t {}\n".format(confidence))
            a.write("\n")            
            a.write("Accuracy                   \t {}\n".format(round(acc_soft,3)))
            a.write("\n")
            a.write("                           \t Precision     \t\t Recall     \t\t F1 \t\t support\n")
            a.write("Rest metrics               \t {}            \t {}           \t {}      \t {}\n".format(round(precision_soft[2],3), round(recall_soft[2],3), round(fscore_soft[2],3), round(support_soft[2],3)))
            a.write("Contraction metrics        \t {}            \t {}           \t {}      \t {}\n".format(round(precision_soft[1],3), round(recall_soft[1],3), round(fscore_soft[1],3), round(support_soft[1],3)))
            a.write("Needle metrics             \t {}            \t {}           \t {}      \t {}\n".format(round(precision_soft[0],3), round(recall_soft[0],3), round(fscore_soft[0],3), round(support_soft[0],3)))
