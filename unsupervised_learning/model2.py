# standard libraries 
import numpy as np
import os
from time import time
import csv
from csv import writer
import pandas as pd
from tqdm import tqdm, trange
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
# sklearn
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support, normalized_mutual_info_score, adjusted_rand_score
from sklearn.cluster import KMeans
# tensorflow & keras
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras
from tqdm.keras import TqdmCallback
from keras.layers import Layer, InputSpec, LeakyReLU, Input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from tensorflow.keras.utils import plot_model
import keras.backend as K
from keras.callbacks import CSVLogger, Callback
# hyperopt
from hyperopt import hp, Trials, tpe, fmin, STATUS_OK, space_eval, STATUS_FAIL
import pickle
# self-made functions
from displays import display_clusters, display_example_predictions, display_confusionmatrix
from load_data import load_mels
from unsupervised_model_options import UnsupervisedOptions
from model1 import Validation, ConvAE_clustering, Callback_CAE

"""
Training, evaluation and testing model 2

@author: Sterre de Jonge (2022)
"""

class Callback_DCEC(Callback):
    def __init__(self, train_data, val_data, y_true_val, labels, n_clusters, datatype, save_directory, batch_size, epoch, y_pred_last):
        super(Callback_DCEC, self).__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.y_pred_last = y_pred_last
        self.y_true_val = y_true_val
        self.labels = labels
        self.validation = Validation(clusters=n_clusters, datatype=datatype, show_displays=False, save_directory=False)
        self.save_directory = save_directory
        self.batch_size = batch_size
        self.epoch = epoch

        # create models folder
        if not os.path.exists(self.save_directory + 'models/'): os.makedirs(self.save_directory + 'models/')

    def on_epoch_end(self, epoch, logs={}):

        print(" ...End epoch {} of training".format(self.epoch))
        logs['epoch'] = self.epoch

        clustering = DCEC(inputs=self.model.input, outputs=self.model.output[1], gamma=config.gamma, batch_size=config.batch_size)

        q_val = clustering.predict(self.val_data)
        y_pred_val = q_val.argmax(1)
        weight = q_val ** 2 / q_val.sum(0)
        p_val = (weight.T / weight.sum(1)).T

        val_input = JoinedGen(self.val_data, p_val, self.batch_size)

        val_loss = self.model.evaluate(x=val_input, verbose=0)
        logs['val_loss'] = val_loss[1]
        logs['val_reconstruction_loss'] = val_loss[2]
        logs['val_clustering_loss'] = val_loss[0]

        q = clustering.predict(self.train_data)
        y_pred_train = q.argmax(1)

        try: 
            delta_label = np.sum(y_pred_train != self.y_pred_last).astype(np.float32) / y_pred.shape[0]
            print("\ndeltalabel = %.3f" % delta_label)
            logs['deltalabel'] = delta_label
            print("Training metrics at epoch %s:        delta_label=%.4f, total loss=%.4f, clustering loss=%.4f, reconstruction loss=%.4f" % (logs['epoch'], delta_label, logs['loss'], logs['clustering_loss'], logs['reconstruction_loss']))
            print("Training metrics at epoch %s:        validation total loss=%.4f, validation clustering loss=%.4f, validation reconstruction loss=%.4f" % (logs['epoch'], logs['val_loss'], logs['val_clustering_loss'], logs['val_reconstruction_loss']))
            self.y_pred_last = y_pred_train

        except: 
            print("Training metrics at epoch %s:        delta_label=%s, total loss=%.3f, clustering loss=%.3f, reconstruction loss=%.3f" % (logs['epoch'], delta_label, logs['loss'], logs['clustering_loss'], logs['reconstruction_loss']))
            print("\ndeltalabel could not be computed")
            logs['deltalabel'] = np.nan

        acc, precision, recall, fscore, support = self.validation.validate(self.y_true_val, y_pred_val, self.labels, argument='train')
        if acc is None: 
            print("Performance could not be obtained because either: not all distinct labels received a designated cluster, or not all clusters where filled with predicted samples.")
            logs['acc'] = np.nan
            logs['nmi'] = np.nan
            logs['ari'] = np.nan
            logs['f1_rest'] = np.nan
            logs['f1_contraction'] = np.nan
            logs['f1_artefact'] = np.nan
        if acc is not None:
            nmi = normalized_mutual_info_score(self.y_true_val, y_pred_val)
            ari = adjusted_rand_score(self.y_true_val, y_pred_val)
            print("Testing metrics at epoch %s:         acc=%.4f, f1 rest=%.4f" % (logs['epoch'], acc, fscore[2]))
            logs['acc'] = acc
            logs['nmi'] = nmi
            logs['ari'] = ari
            logs['f1_rest'] = fscore[2]
            logs['f1_contraction'] = fscore[1]
            logs['f1_artefact'] = fscore[0]

            # soft metrics
            cluster_centres = self.model.get_layer(name='clustering').get_weights_()
            cluster_centres = np.asarray(cluster_centres)
            cluster_centres = np.reshape(cluster_centres, [cluster_centres.shape[1], cluster_centres.shape[2]])

            latent_space_model = Model(inputs=self.model.input, outputs=self.model.get_layer(name='embedding').output)

            features_val = latent_space_model.predict(self.val_data) 
            features_val = np.reshape(features_val, newshape=(features_val.shape[0], -1))

            features_train = latent_space_model.predict(self.train_data)
            features_train = np.reshape(features_train, newshape=(features_train.shape[0], -1))

            confidence, perc_removed_train = self.validation.get_optimal_confidence(features_train, y_pred_train, cluster_centres)
            print("confidence = ", confidence)

            features_val_soft, y_pred_val_soft, y_true_val_soft = self.validation.validate_soft_clustering(features_val, y_pred_val, confidence, y_true=self.y_true_val, cluster_centres=cluster_centres)
            acc_soft, precision_soft, recall_soft, fscore_soft, support = self.validation.validate(y_true_val_soft, y_pred_val_soft, self.labels, argument='train')
            fscore_artefact_soft, fscore_contraction_soft, fscore_rest_soft = fscore_soft[0], fscore_soft[1], fscore_soft[2]
            logs['acc_soft'] = acc_soft
            logs['f1_rest_soft'] = fscore_rest_soft
            logs['f1_contraction_soft'] = fscore_contraction_soft
            logs['f1_artefact_soft'] = fscore_artefact_soft
            logs['confidence'] = confidence
            print('Soft metrics with confidence %s:  acc=%.3f, F1 rest=%.3f' % (confidence, acc_soft, fscore_rest_soft))

            perc_removed_val = 100 - ((len(y_pred_val_soft) / len(y_pred_val)) * 100)

            logs['percentage_removed_val'] = perc_removed_val
            logs['percentage_removed_train'] = perc_removed_train

        print('saving model to:', self.save_directory + 'models/dcec_model_' + str(logs['epoch']) + '.hdf5 \n')
        self.model.save(self.save_directory + 'models/dcec_model_' + str(logs['epoch']) + '.hdf5')
     
class ClusteringLayer(Layer):
    """
    Clustering layer converts input sample (feature) to soft label, i.e. a vector that represents the probability of the
    sample belonging to each cluster. The probability is calculated with student's t-distribution.

    # Example
    ```
        model.add(ClusteringLayer(n_clusters=10))
    ```
    # Arguments
        n_clusters: number of clusters.
        weights: list of Numpy array with shape `(n_clusters, n_features)` witch represents the initial cluster centers.
        alpha: parameter in Student's t-distribution. Default to 1.0.
    # Input shape
        2D tensor with shape: `(n_samples, n_features)`.
    # Output shape
        2D tensor with shape: `(n_samples, n_clusters)`.
    """

    def __init__(self, n_clusters, weights=None, alpha=3.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight(shape=(self.n_clusters, input_dim), initializer='glorot_uniform', name='clusters')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        """ student t-distribution, as same as used in t-SNE algorithm.
                 q_ij = 1/(1+dist(x_i, u_j)^2), then normalize it.
        Arguments:
            inputs: the variable containing data, shape=(n_samples, n_features)
        Return:
            q: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)
        """
        
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))

        q **= (self.alpha + 1.0) / 2
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
        
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def get_weights_(self):
        return self.get_weights()

class DCEC(Model):
    def __init__(self, **kwargs):
        super(DCEC, self).__init__(**kwargs)
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = tf.metrics.Mean(name="kl_loss")
        self.gamma = 0.05
        self.batch_size = 64
        self.p = None

    def train_step(self, data):        
        # Override method 'train_step' to customize how training takes place.
        with tf.GradientTape() as tape:            
            data, p = data[0], data[1] # input data in batches
            reconstruction, q = self(data) # forward pass

            # compute losses
            mse = tf.keras.losses.MeanSquaredError()
            reconstruction_loss = mse(data,reconstruction)
            kld = tf.keras.losses.KLDivergence()
            kl_loss = kld(p, q)
            # compute total loss
            total_loss = reconstruction_loss + kl_loss * self.gamma
        
        # compute gradients with total loss
        grads = tape.gradient(total_loss, self.trainable_weights) 
        # update weights
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights)) 
        # update metrics
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        # return a dict mapping metric names to current value
        return {
            "loss": self.total_loss_tracker.result(), 
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "clustering_loss": self.kl_loss_tracker.result(),
        }

    def test_step(self, data):
        data, p = data[0], data[1] # input data in batches
        reconstruction, q = self(data, training=False) # forward pass

        mse = tf.keras.losses.MeanSquaredError()
        reconstruction_loss = mse(data,reconstruction)
        kld = tf.keras.losses.KLDivergence()
        kl_loss = kld(p,q)
        total_loss = reconstruction_loss + kl_loss * self.gamma

        # update metrics
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return {
            "val_loss": self.total_loss_tracker.result(), 
            "val_reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "val_clustering_loss": self.kl_loss_tracker.result(),
        }

    @property
    def metrics(self):
        return[
            self.total_loss_tracker,
            self.reconstruction_loss_tracker, 
            self.kl_loss_tracker,
        ]

class Hyperopt(object):

    def __init__(self, save_dir, trials_dir, search_space, save_interval_trials, max_trials):
        
        self.save_dir = save_dir
        self.trials_dir = trials_dir
        
        self.search_space = search_space
        self.save_interval_trials = save_interval_trials
        self.max_trials = max_trials 

        self.trials = None

    # @profile
    def model_optimisation(params):
        print('Parameters for current trial:', params, '\n')

        # load data
        datagen = ImageDataGenerator(rescale=1./255) 
        train_data_cae = datagen.flow_from_directory(
                        directory = directory_training, 
                        target_size = (128,128),
                        batch_size = params['learning']['batchsize'],
                        class_mode = 'input', 
                        color_mode = 'grayscale', 
                        shuffle = False 
            )
        train_data_dcec = datagen.flow_from_directory(
                        directory = directory_training, 
                        target_size = (128,128),
                        batch_size = params['learning']['batchsize'],
                        class_mode = None, 
                        color_mode = 'grayscale', 
                        shuffle = False 
            )
        test_data = datagen.flow_from_directory(
                    directory = config.validation_signaltype_70procent, 
                    target_size = (128,128), 
                    batch_size = params['learning']['batchsize'],
                    class_mode = 'categorical', 
                    color_mode = 'grayscale', 
                    shuffle = False 
                    )
        y_true = test_data.classes 
        
        # pre-train model
        autoencoder = ConvAE_clustering.CAE(input_shape=(128,128,1), 
                                            kernel_size=params['architecture']['kernel_size'], 
                                            activation=params['activation'], 
                                            filters=params['architecture']['filters'], 
                                            features=params['features'], 
                                            batch_norm=params['batch_normalisation'])

        if params['learning']['optimiser'] == 'adam':
            if params['learning']['batchsize'] == 256:
                if params['learning']['schedule_256_convae'] == 'constant': opt = tf.keras.optimizers.Adam(params['learning']['rate_256_constant_convae'])
                if params['learning']['schedule_256_convae'] == 'clr1':
                    clr = tfa.optimizers.CyclicalLearningRate(initial_learning_rate=params['learning']['rate_256_clr_convae'][0], 
                                                        maximal_learning_rate=params['learning']['rate_256_clr_convae'][1], 
                                                        scale_fn=lambda x: 1.0,
                                                        step_size= (train_data_cae.samples / config.batch_size) * 2) 
                    opt = tf.keras.optimizers.Adam(clr)
                if params['learning']['schedule_256_convae'] == 'clr2':
                    clr = tfa.optimizers.CyclicalLearningRate(initial_learning_rate=params['learning']['rate_256_clr_convae'][0], 
                                                        maximal_learning_rate=params['learning']['rate_256_clr_convae'][1], 
                                                        scale_fn=lambda x: 1/(2.**(x-1)), 
                                                        step_size= (train_data_cae.samples / config.batch_size) * 2) 
                    opt = tf.keras.optimizers.Adam(clr)
            if params['learning']['batchsize'] == 64:
                if params['learning']['schedule_64_convae'] == 'constant': opt = tf.keras.optimizers.Adam(params['learning']['rate_64_constant_convae'])
                if params['learning']['schedule_64_convae'] == 'clr1':
                    clr = tfa.optimizers.CyclicalLearningRate(initial_learning_rate=params['learning']['rate_64_clr_convae'][0], 
                                                        maximal_learning_rate=params['learning']['rate_64_clr_convae'][1], 
                                                        scale_fn=lambda x: 1.0,
                                                        step_size= (train_data_cae.samples / config.batch_size) * 2) 
                    opt =tf.keras.optimizers.Adam(clr)
                if params['learning']['schedule_64_convae'] == 'clr2':
                    clr = tfa.optimizers.CyclicalLearningRate(initial_learning_rate=params['learning']['rate_64_clr_convae'][0], 
                                                        maximal_learning_rate=params['learning']['rate_64_clr_convae'][1], 
                                                        scale_fn=lambda x: 1/(2.**(x-1)), 
                                                        step_size= (train_data_cae.samples / config.batch_size) * 2) 
                    opt = tf.keras.optimizers.Adam(clr)
        if params['learning']['optimiser'] == 'sgd':
            if params['learning']['batchsize'] == 256:
                if params['learning']['schedule_256_convae'] == 'constant': opt = tf.keras.optimizers.SGD(params['learning']['rate_256_constant_convae'])
                if params['learning']['schedule_256_convae'] == 'clr1':
                    clr = tfa.optimizers.CyclicalLearningRate(initial_learning_rate=params['learning']['rate_256_clr_convae'][0], 
                                                        maximal_learning_rate=params['learning']['rate_256_clr_convae'][1], 
                                                        scale_fn=lambda x: 1.0,
                                                        step_size= (train_data_cae.samples / config.batch_size) * 2) 
                    opt = tf.keras.optimizers.SGD(clr)
                if params['learning']['schedule_256_convae'] == 'clr2':
                    clr = tfa.optimizers.CyclicalLearningRate(initial_learning_rate=params['learning']['rate_256_clr_convae'][0], 
                                                        maximal_learning_rate=params['learning']['rate_256_clr_convae'][1], 
                                                        scale_fn=lambda x: 1/(2.**(x-1)), 
                                                        step_size= (train_data_cae.samples / config.batch_size) * 2) 
                    opt = tf.keras.optimizers.SGD(clr)
            if params['learning']['batchsize'] == 64:
                if params['learning']['schedule_64_convae'] == 'constant': opt = tf.keras.optimizers.SGD(params['learning']['rate_64_constant_convae'])
                if params['learning']['schedule_64_convae'] == 'clr1':
                    clr = tfa.optimizers.CyclicalLearningRate(initial_learning_rate=params['learning']['rate_64_clr_convae'][0], 
                                                        maximal_learning_rate=params['learning']['rate_64_clr_convae'][1], 
                                                        scale_fn=lambda x: 1.0,
                                                        step_size= (train_data_cae.samples / config.batch_size) * 2) 
                    opt =tf.keras.optimizers.SGD(clr)
                if params['learning']['schedule_64_convae'] == 'clr2':
                    clr = tfa.optimizers.CyclicalLearningRate(initial_learning_rate=params['learning']['rate_64_clr_convae'][0], 
                                                        maximal_learning_rate=params['learning']['rate_64_clr_convae'][1], 
                                                        scale_fn=lambda x: 1/(2.**(x-1)), 
                                                        step_size= (train_data_cae.samples / config.batch_size) * 2) 
                    opt = tf.keras.optimizers.SGD(clr)

        autoencoder.compile(optimizer=opt, loss=params['loss_convae']) 

        # begin pre-training convolutional autoenoder 
        autoencoder.fit(train_data_cae, 
                        epochs=params['epochs_convae'], 
                        verbose=0, 
                        callbacks=[TqdmCallback(verbose=1)]
                        )

        feature_model = Model(inputs=autoencoder.input, outputs=autoencoder.get_layer(name='embedding').output)
        features_test = feature_model.predict(test_data, verbose=0) 
        features_test = np.reshape(features_test, newshape=(features_test.shape[0], -1))
        features_train = feature_model.predict(train_data_cae, verbose=0)
        features_train = np.reshape(features_train, newshape=(features_train.shape[0], -1))

        km = KMeans(n_clusters = params['clusters'], n_init=100)
        km_fitted = km.fit(features_train)
        y_pred = km_fitted.predict(features_test)

        cm = confusion_matrix(y_true, y_pred)
        cm_argmax = cm.argmax(axis=0) 
        y_pred = np.array([cm_argmax[k] for k in y_pred]) 
        acc_pretrain = accuracy_score(y_true, y_pred)

        precision_pretrain, recall_pretrain, fscore_pretrain, support = precision_recall_fscore_support(y_true, y_pred, labels = [0,1,2]) # label 2 = rest
        fscore_rest_pretrain = fscore_pretrain[2]

        print('Pretraining is finished with accuracy: %.3f and f1 rest %.3f\n' % (acc_pretrain, fscore_rest_pretrain))

        # begin training dcec
        hidden = autoencoder.get_layer(name='embedding').output
        clustering_layer = ClusteringLayer(params["clusters"], name='clustering')(hidden)

        deepclustering = DCEC(inputs=autoencoder.input, outputs=[autoencoder.output, clustering_layer], gamma=params['gamma'], batch_size=params['learning']['batchsize'])

        if params['learning']['optimiser'] == 'adam':
            if params['learning']['batchsize'] == 256:
                if params['learning']['schedule_256_dcec'] == 'constant': opt = tf.keras.optimizers.Adam(params['learning']['rate_256_constant_dcec'])
                if params['learning']['schedule_256_dcec'] == 'clr1':
                    clr = tfa.optimizers.CyclicalLearningRate(initial_learning_rate=params['learning']['rate_256_clr_dcec'][0], 
                                                        maximal_learning_rate=params['learning']['rate_256_clr_dcec'][1], 
                                                        scale_fn=lambda x: 1.0,
                                                        step_size= (train_data_dcec.samples / config.batch_size) * 2) 
                    opt = tf.keras.optimizers.Adam(clr)
                if params['learning']['schedule_256_dcec'] == 'clr2':
                    clr = tfa.optimizers.CyclicalLearningRate(initial_learning_rate=params['learning']['rate_256_clr_dcec'][0], 
                                                        maximal_learning_rate=params['learning']['rate_256_clr_dcec'][1], 
                                                        scale_fn=lambda x: 1/(2.**(x-1)), 
                                                        step_size= (train_data_dcec.samples / config.batch_size) * 2) 
                    opt = tf.keras.optimizers.Adam(clr)
            if params['learning']['batchsize'] == 64:
                if params['learning']['schedule_64_dcec'] == 'constant': opt = tf.keras.optimizers.Adam(params['learning']['rate_64_constant_dcec'])
                if params['learning']['schedule_64_dcec'] == 'clr1':
                    clr = tfa.optimizers.CyclicalLearningRate(initial_learning_rate=params['learning']['rate_64_clr_dcec'][0], 
                                                        maximal_learning_rate=params['learning']['rate_64_clr_dcec'][1], 
                                                        scale_fn=lambda x: 1.0,
                                                        step_size= (train_data_dcec.samples / config.batch_size) * 2) 
                    opt = tf.keras.optimizers.Adam(clr)
                if params['learning']['schedule_64_dcec'] == 'clr2':
                    clr = tfa.optimizers.CyclicalLearningRate(initial_learning_rate=params['learning']['rate_64_clr_dcec'][0], 
                                                        maximal_learning_rate=params['learning']['rate_64_clr_dcec'][1], 
                                                        scale_fn=lambda x: 1/(2.**(x-1)), 
                                                        step_size= (train_data_dcec.samples / config.batch_size) * 2) 
                    opt = tf.keras.optimizers.Adam(clr)
        if params['learning']['optimiser'] == 'sgd':
            if params['learning']['batchsize'] == 256:
                if params['learning']['schedule_256_dcec'] == 'constant': opt = tf.keras.optimizers.SGD(params['learning']['rate_256_constant_dcec'])
                if params['learning']['schedule_256_dcec'] == 'clr1':
                    clr = tfa.optimizers.CyclicalLearningRate(initial_learning_rate=params['learning']['rate_256_clr_dcec'][0], 
                                                        maximal_learning_rate=params['learning']['rate_256_clr_dcec'][1], 
                                                        scale_fn=lambda x: 1.0,
                                                        step_size= (train_data_dcec.samples / config.batch_size) * 2) 
                    opt = tf.keras.optimizers.SGD(clr)
                if params['learning']['schedule_256_dcec'] == 'clr2':
                    clr = tfa.optimizers.CyclicalLearningRate(initial_learning_rate=params['learning']['rate_256_clr_dcec'][0], 
                                                        maximal_learning_rate=params['learning']['rate_256_clr_dcec'][1], 
                                                        scale_fn=lambda x: 1/(2.**(x-1)), 
                                                        step_size= (train_data_dcec.samples / config.batch_size) * 2) 
                    opt = tf.keras.optimizers.SGD(clr)
            if params['learning']['batchsize'] == 64:
                if params['learning']['schedule_64_dcec'] == 'constant': opt = tf.keras.optimizers.SGD(params['learning']['rate_64_constant_dcec'])
                if params['learning']['schedule_64_dcec'] == 'clr1':
                    clr = tfa.optimizers.CyclicalLearningRate(initial_learning_rate=params['learning']['rate_64_clr_dcec'][0], 
                                                        maximal_learning_rate=params['learning']['rate_64_clr_dcec'][1], 
                                                        scale_fn=lambda x: 1.0,
                                                        step_size= (train_data_dcec.samples / config.batch_size) * 2) 
                    opt = tf.keras.optimizers.SGD(clr)
                if params['learning']['schedule_64_dcec'] == 'clr2':
                    clr = tfa.optimizers.CyclicalLearningRate(initial_learning_rate=params['learning']['rate_64_clr_dcec'][0], 
                                                        maximal_learning_rate=params['learning']['rate_64_clr_dcec'][1], 
                                                        scale_fn=lambda x: 1/(2.**(x-1)), 
                                                        step_size= (train_data_dcec.samples / config.batch_size) * 2) 
                    opt = tf.keras.optimizers.SGD(clr)

        deepclustering.compile(loss=params['loss_dcec'], loss_weights=[params['gamma'],1], optimizer=opt)

        deepclustering.get_layer(name='clustering').set_weights([km.cluster_centers_])
        
        for epoch in range(1, params['epochs_dcec']+1):

            clustering = DCEC(inputs=deepclustering.input, outputs=deepclustering.output[1], gamma=params['gamma'], batch_size=params['learning']['batchsize'])

            q = clustering.predict(train_data_dcec, verbose=0)
            y_pred = q.argmax(1)
            weight = q ** 2 / q.sum(0)
            p = (weight.T / weight.sum(1)).T

            combined_input = JoinedGen(train_data_dcec, p, params['learning']['batchsize'])

            deepclustering.fit(x=combined_input, verbose=0, callbacks=[TqdmCallback(verbose=1)])
            
        clustering = Model(inputs=deepclustering.input, outputs=deepclustering.output[1])
        q_test = clustering.predict(test_data, verbose=0)
        y_pred = q_test.argmax(1)

        cm = confusion_matrix(y_true, y_pred)
        cm_argmax = cm.argmax(axis=0) 
        if not set(cm_argmax) == set(y_true): 
            print("Fail")
            return {'status': STATUS_FAIL}
        if not len(cm_argmax) == len(set(y_pred)):
            print("Fail") 
            return {'status': STATUS_FAIL}
        y_pred = np.array([cm_argmax[k] for k in y_pred]) 
        acc = accuracy_score(y_true, y_pred)

        precision, recall, fscore, support = precision_recall_fscore_support(y_true, y_pred, labels = [0,1,2]) # label 2 = rest
        fscore_rest = fscore[2]

        print('Training for one trial is finished with accuracy: %.3f and f1 rest %.3f\n' % (acc, fscore_rest))

        return {'loss': -fscore_rest, 'metrics': [acc, precision, recall, fscore, acc_pretrain, precision_pretrain, recall_pretrain, fscore_pretrain], 'status': STATUS_OK}

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
            print("Starting hyperoptimization from scratch...\n")
            self.trials = Trials()
            temp_max_trials = self.save_interval_trials
    
        best = fmin(Hyperopt.model_optimisation, self.search_space, algo=tpe.suggest, max_evals=temp_max_trials, trials=self.trials)
        print("Best:", best)

        self.trials_dir = self.save_dir + 'trials.p'
        with open(self.trials_dir, "wb") as f:
            pickle.dump(self.trials, f)
            print("Trials is saved to", self.trials_dir)

class JoinedGen(tf.keras.utils.Sequence):
    def __init__(self, input_gen1, input_gen2, batch_size):
        self.gen1 = input_gen1
        self.gen2 = input_gen2
        self.batch_size = batch_size

    def __len__(self):
        return len(self.gen1)

    def __getitem__(self, i):
        x1 = self.gen1[i]

        if i == len(self.gen1)-1: x2 = self.gen2[i*self.batch_size::]
        else: x2 = self.gen2[i*self.batch_size:i*self.batch_size+self.batch_size]

        return [x1, x2]

    def on_epoch_end(self):
        self.gen1.on_epoch_end()

if __name__ == "__main__":

    config = UnsupervisedOptions().parse()

    if config.architecture != "arch_two": 
        print("Wrong architecture type is given in configuration file.")
        exit()

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
    train_data_dcec = datagen.flow_from_directory(
                directory = directory_training, 
                target_size = config.input_dimension, 
                batch_size = config.batch_size,
                class_mode = None, # has to be None to use in JoinedGen
                color_mode = 'grayscale', 
                subset = 'training', 
                shuffle=False 
        )
    train_data_cae = datagen.flow_from_directory(
                directory = directory_training, 
                target_size = config.input_dimension, 
                batch_size = config.batch_size,
                class_mode = 'input', # has to be input to use in cae.evaluate()
                color_mode = 'grayscale', 
                subset = 'training', 
                shuffle=False 
        )
    print("Validation data")
    val_data_cae = datagen.flow_from_directory(
            directory = config.validation_signaltype_70procent, 
            target_size = config.input_dimension, 
            batch_size = config.batch_size,
            class_mode = 'input', 
            color_mode = 'grayscale', 
            shuffle=False # Has to be off (otherwise y_true is not in order to compute deltalabel)
        )
    labels_val = list(val_data_cae.class_indices.keys())
    y_true_val = val_data_cae.classes
    val_data_dcec = datagen.flow_from_directory(
            directory = config.validation_signaltype_70procent, 
            target_size = config.input_dimension, 
            batch_size = config.batch_size,
            class_mode = None, 
            color_mode = 'grayscale', 
            shuffle=False # Has to be off (otherwise y_true is not in order to compute deltalabel)
        )
    print("Test data")
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
    test_data_dcec = datagen.flow_from_directory(
                directory = config.test_signaltype_30procent, 
                target_size = config.input_dimension, 
                batch_size = config.batch_size,
                class_mode = 'categorical', 
                color_mode = 'grayscale', 
                shuffle = False 
                )
    labels_test_dcec = list(test_data_dcec.class_indices.keys())
    y_true_test_dcec = test_data_dcec.classes
    print("")

    ## HYPERPARAMETER OPTIMIZATION
    if config.hyperopt is True or config.evaluate_hyperopt is True:

        search_space = {'learning': hp.choice('learning', [
                                    {
                                        # 'optimiser': 'adam',
                                        'batchsize': 256,
                                        'schedule_256_convae': hp.choice('schedule_256_convae', ['constant', 'clr1', 'clr2']),
                                        'schedule_256_dcec': hp.choice('schedule_256_dcec', ['constant', 'clr1', 'clr2']),
                                        'rate_256_constant_convae': hp.choice('rate_256_constant_convae', [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]),
                                        'rate_256_constant_dcec': hp.choice('rate_256_constant_dcec', [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]),
                                        'rate_256_clr_convae': [10**-5, 0.001],
                                        'rate_256_clr_dcec':  [10**-5, 0.001]
                                        }, 
                                    {
                                        # 'optimiser': 'adam',
                                        'batchsize': 64,
                                        'schedule_64_convae': hp.choice('schedule_64_convae', ['constant', 'clr1', 'clr2']),
                                        'schedule_64_dcec': hp.choice('schedule_64_dcec', ['constant', 'clr1', 'clr2']),
                                        'rate_64_constant_convae': hp.choice('rate_64_constant_convae', [10**-6, 0.000005, 10**-5, 0.00005, 10**-4, 0.0005, 0.001, 0.005]),
                                        'rate_64_constant_dcec': hp.choice('rate_64_constant_dcec', [10**-6, 0.000005, 10**-5, 0.00005, 10**-4, 0.0005, 0.001, 0.005]),
                                        'rate_64_clr_convae': [10**-7, 10**-4],
                                        'rate_64_clr_dcec':  [10**-7, 10**-4]
                                    },
                                    # {
                                    #     'optimiser': 'sgd',
                                    #     'batchsize': 256,
                                    #     'schedule_256_convae': hp.choice('schedule_256_convae_sgd', ['constant', 'clr1', 'clr2']),
                                    #     'schedule_256_dcec': hp.choice('schedule_256_dcec_sgd', ['constant', 'clr1', 'clr2']),
                                    #     'rate_256_constant_convae': hp.choice('rate_256_constant_convae_sgd', [0.0001, 0.0005, 0.001, 0.005, 0.01]),
                                    #     'rate_256_constant_dcec': hp.choice('rate_256_constant_dcec_sgd', [0.0001, 0.0005, 0.001, 0.005, 0.01]),
                                    #     'rate_256_clr_convae': [10**-4, 0.01],
                                    #     'rate_256_clr_dcec':  [10**-4, 0.01]
                                    #     }, 
                                    # {
                                    #     'optimiser': 'sgd',
                                    #     'batchsize': 64,
                                    #     'schedule_64_convae': hp.choice('schedule_64_convae_sgd', ['constant', 'clr1', 'clr2']),
                                    #     'schedule_64_dcec': hp.choice('schedule_64_dcec_sgd', ['constant', 'clr1', 'clr2']),
                                    #     'rate_64_constant_convae': hp.choice('rate_64_constant_convae_sgd', [0.0001, 0.0005, 0.001, 0.005, 0.01]),
                                    #     'rate_64_constant_dcec': hp.choice('rate_64_constant_dcec_sgd', [0.0001, 0.0005, 0.001, 0.005, 0.01]),
                                    #     'rate_64_clr_convae': [10**-4, 0.01],
                                    #     'rate_64_clr_dcec':  [10**-4, 0.01]
                                    # }
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
                            # 'activation': hp.choice('activation', ['relu', LeakyReLU(alpha=0.1), LeakyReLU(alpha=0.2), LeakyReLU(alpha=0.3), LeakyReLU(alpha=0.4)]),
                            'activation': hp.choice('activation', ['relu', LeakyReLU(alpha=0.1), LeakyReLU(alpha=0.2), LeakyReLU(alpha=0.3)]),
                            'batch_normalisation': hp.choice('batch_normalisation', [True, False]),
                            'features': hp.choice('features', [10, 12, 14, 16, 18, 20, 24, 28, 32, 36, 40, 48, 56, 64, 128, 256, 512, 1024, 2048, 4096]),  
                            'epochs_convae': 8, 
                            'epochs_dcec': 8, 
                            'optimizer': 'adam',
                            'loss_convae': ['mse'],
                            'loss_dcec': ['kld', 'mse'],
                            # 'clusters': hp.choice('clusters', [4, 5, 6, 7, 8, 9]), 
                            'clusters': hp.choice('clusters', [5, 6, 7, 8, 9]), 
                            'gamma': hp.choice('gamma', [0.05, 0.1, 0.2, 0.3, 0.4, 0.5])
            }        

        if config.hyperopt is True:

            # initialise class 
            hyperopt = Hyperopt(save_dir=config.save_directory, 
                                trials_dir=config.hyperopt_directory, 
                                search_space=search_space, 
                                save_interval_trials=config.save_interval_trials, 
                                max_trials=config.max_trials)

            for i in range(0, config.max_trials, config.save_interval_trials):
                hyperopt.run_trials()

        # store hyperopt results in Excel file 
        if config.hyperopt_directory is not None and config.evaluate_hyperopt is True:
            print("Trials is loaded to create Excel file in", config.save_directory)
            trials = pickle.load(open(config.hyperopt_directory, "rb"))
            # trials_sorted = sorted(trials.results, key=lambda x: x['loss'], reverse=False)
            df_trials = pd.DataFrame(columns=['f1_rest', 'f1_rest_pretrain', 'accuracy', 'accuracy_pretrain', 'batch_size', 'learning_schedule_convae', 
                                            'learning_rate_convae', 'learning_schedule_dcec', 'learning_rate_dcec','layers', 'filters', 'kernels',
                                            'activation', 'batchnorm',  'features', 'optimiser', 'clusters', 'gamma'])
            for t, trial in enumerate(trials):
                if not trial['result']['status'] == 'fail':
                    vals = trial.get('misc').get('vals')
                    tmp = {}
                    for k,v in list(vals.items()):
                        if v: tmp[k] = v[0]
                                        
                    vals_trial = space_eval(search_space, tmp)
                    if vals_trial['learning']['batchsize'] == 256:
                        if vals_trial['learning']['schedule_256_convae'] == 'constant':
                            learningrate_convae = '256 constant ' + str(vals_trial['learning']['rate_256_constant_convae'])
                            schedule_convae = 'constant'
                        if vals_trial['learning']['schedule_256_convae'] == 'clr1':
                            learningrate_convae = '256 clr1 ' + str(vals_trial['learning']['rate_256_clr_convae'])
                            schedule_convae = 'clr1'
                        if vals_trial['learning']['schedule_256_convae'] == 'clr2':
                            learningrate_convae = '256 clr2 ' + str(vals_trial['learning']['rate_256_clr_convae'])
                            schedule_convae = 'clr2'
                        if vals_trial['learning']['schedule_256_dcec'] == 'constant':
                            learningrate_dcec = '256 constant ' + str(vals_trial['learning']['rate_256_constant_dcec'])
                            schedule_dcec = 'constant'
                        if vals_trial['learning']['schedule_256_dcec'] == 'clr1':
                            learningrate_dcec = '256 clr1 ' + str(vals_trial['learning']['rate_256_clr_dcec'])
                            schedule_dcec = 'clr1'
                        if vals_trial['learning']['schedule_256_dcec'] == 'clr2':
                            learningrate_dcec = '256 clr2 ' + str(vals_trial['learning']['rate_256_clr_dcec'])
                            schedule_dcec = 'clr2'
                    if vals_trial['learning']['batchsize'] == 64:
                        if vals_trial['learning']['schedule_64_convae'] == 'constant':
                            learningrate_convae = '64 constant ' + str(vals_trial['learning']['rate_64_constant_convae'])
                            schedule_convae = 'constant'
                        if vals_trial['learning']['schedule_64_convae'] == 'clr1':
                            learningrate_convae = '64 clr1 ' + str(vals_trial['learning']['rate_64_clr_convae'])
                            schedule_convae = 'clr1'
                        if vals_trial['learning']['schedule_64_convae'] == 'clr2':
                            learningrate_convae = '64 clr2 ' + str(vals_trial['learning']['rate_64_clr_convae'])
                            schedule_convae = 'clr2'
                        if vals_trial['learning']['schedule_64_dcec'] == 'constant':
                            learningrate_dcec = '64 constant ' + str(vals_trial['learning']['rate_64_constant_dcec'])
                            schedule_dcec = 'constant'
                        if vals_trial['learning']['schedule_64_dcec'] == 'clr1':
                            learningrate_dcec = '64 clr1 ' + str(vals_trial['learning']['rate_64_clr_dcec'])
                            schedule_dcec = 'clr1'
                        if vals_trial['learning']['schedule_64_dcec'] == 'clr2':
                            learningrate_dcec = '64 clr2 ' + str(vals_trial['learning']['rate_64_clr_dcec'])
                            schedule_dcec = 'clr2'

                    if trial['misc']['vals']['activation'] == [0]: activation = 'ReLU'
                    if trial['misc']['vals']['activation'] == [1]: activation = 'leakyReLU alpha=0.1'
                    if trial['misc']['vals']['activation'] == [2]: activation = 'leakyReLU alpha=0.2'
                    if trial['misc']['vals']['activation'] == [3]: activation = 'leakyReLU alpha=0.3'
                    if trial['misc']['vals']['activation'] == [3]: activation = 'leakyReLU alpha=0.4'

                    df_trials.loc[t] = pd.Series({'f1_rest':abs(trial['result']['loss']), 
                                                'accuracy': trial['result']['metrics'][0],
                                                'f1_rest_pretrain': trial['result']['metrics'][-1][2],
                                                'accuracy_pretrain': trial['result']['metrics'][4],
                                                'batch_size': vals_trial['learning']['batchsize'],
                                                'learning_schedule_convae': schedule_convae,
                                                'learning_rate_convae': learningrate_convae,
                                                'learning_schedule_dcec': schedule_dcec,
                                                'learning_rate_dcec': learningrate_dcec,
                                                # 'optimiser': vals_trial['learning']['optimiser'],
                                                'layers': vals_trial['architecture']['type'],
                                                'filters': vals_trial['architecture']['filters'],
                                                'kernels': vals_trial['architecture']['kernel_size'],
                                                'activation': activation,
                                                'batchnorm': vals_trial['batch_normalisation'],
                                                'features': vals_trial['features'],
                                                'clusters': vals_trial['clusters'], 
                                                'gamma': vals_trial['gamma']
                                                })
                df_trials.to_excel(config.save_directory + 'Trials_excel_model2.xlsx')

    ## TRAIN model 
    if config.train is True:

        # Define CAE model 
        cae = ConvAE_clustering.CAE(config.input_dimension + (1,), config.kernel_size, config.activation, config.filters, config.features, config.batch_norm)

        if config.learning_schedule_convae == 'constant': opt = tf.keras.optimizers.Adam(config.learning_rate_convae)  
        if config.learning_schedule_convae == 'clr1':
            clr = tfa.optimizers.CyclicalLearningRate(initial_learning_rate=config.learning_rate_convae[0], 
                                                    maximal_learning_rate=config.learning_rate_convae[1], 
                                                        scale_fn=lambda x: 1.0, 
                                                        step_size= (train_data_cae.samples / config.batch_size) * 2) # step_size defines the duration of a single cycle, a size of 2 means you need a total of 4 iterations to complete one cycle 
            
            opt = tf.keras.optimizers.Adam(clr)
        if config.learning_schedule_convae == 'clr2':
            clr = tfa.optimizers.CyclicalLearningRate(initial_learning_rate=config.learning_rate_convae[0], 
                                                    maximal_learning_rate=config.learning_rate_convae[1], 
                                                        scale_fn=lambda x: 1/(2.**(x-1)), 
                                                        step_size= (train_data_cae.samples / config.batch_size) * 2) 
            
            opt = tf.keras.optimizers.Adam(clr)

        cae.compile(optimizer=opt, loss=config.loss_convae)
            
        hidden = cae.get_layer(name='embedding').output
        encoder = Model(inputs=cae.input, outputs=hidden)

        # Step 1: pretrain model or load pretrained model
        if config.pretrain is True:

            loss = cae.fit(x=train_data_cae, epochs=config.epochs_convae, callbacks=[Callback_CAE(train_data_cae, val_data_cae, y_true_val, labels_val, config.clusters, config.data, config.confidence, config.save_directory, soft_metrics=True), CSVLogger(config.save_directory + 'logs-convae.csv')])

            cae_weights = config.save_directory + '/pretrain_cae_model.h5'
            cae.save_weights(cae_weights)
            print('Pretrained weights are saved to %s' % cae_weights)

            # cae_weights = dcec2.pretrain(train_data, val_data, test_data, epochs=config.epochs_convae, optimizer=opt, loss=config.loss_convae, save_dir=config.save_directory)
        if config.pretrain is False and config.pretrained_cae is not None:
            cae_weights = config.pretrained_cae
            cae.load_weights(cae_weights)
            print('CAE weights from %s are loaded successfully.' % cae_weights)

        else: 
            "Pretraining needs to be performed (set argument --pretrain to True) OR location to pretrained model needs to be provided with argument --pretrained_cae [insert location]"
       
        # Step 2: initialize cluster centers using k-means
        print('Initializing cluster centers with k-means ({} clusters are created)'.format(config.clusters))
        kmeans = KMeans(n_clusters=config.clusters, n_init=100)

        y_pred = kmeans.fit_predict(encoder.predict(train_data_cae))

        # Step 3: Initialize and train DCEC
        clustering_layer = ClusteringLayer(config.clusters, name='clustering')(hidden)
        
        model = DCEC(inputs=cae.input, outputs=[cae.output, clustering_layer], gamma=config.gamma, batch_size=config.batch_size)
        
        plot_model(model, to_file=config.save_directory + '/dcec_model.png', show_shapes=True)
        model.summary()
        
        if config.learning_schedule_dcec == 'constant': opt = tf.keras.optimizers.Adam(config.learning_rate_dcec)  
        if config.learning_schedule_dcec == 'clr1':
            clr = tfa.optimizers.CyclicalLearningRate(initial_learning_rate=config.learning_rate_dcec[0], 
                                                        maximal_learning_rate=config.learning_rate_dcec[1], 
                                                        scale_fn = lambda x: 1.0,           # clr1
                                                        step_size= (train_data_dcec.samples / config.batch_size) * 2) # step_size defines the duration of a single cycle, a size of 2 means you need a total of 4 iterations to complete one cycle 
                
            opt = tf.keras.optimizers.Adam(clr)
        if config.learning_schedule_dcec == 'clr2':
            clr = tfa.optimizers.CyclicalLearningRate(initial_learning_rate=config.learning_rate_dcec[0], 
                                                        maximal_learning_rate=config.learning_rate_dcec[1], 
                                                        scale_fn=lambda x: 1/(2.**(x-1)), # clr2
                                                        step_size= (train_data_dcec.samples / config.batch_size) * 2) # step_size defines the duration of a single cycle, a size of 2 means you need a total of 4 iterations to complete one cycle 
                
            opt = tf.keras.optimizers.Adam(clr)

        model.compile(optimizer=opt)

        model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])
  
        for epoch in range(1, config.epochs_dcec+1):
            print("epoch {}/{}".format(epoch, config.epochs_dcec))
            
            # define seperate instance of clustering output, so that prediction is memory efficient for predicting q
            clustering = DCEC(inputs=model.input, outputs=model.output[1], gamma=config.gamma, batch_size=config.batch_size)
        
            q = clustering.predict(train_data_dcec)
            y_pred = q.argmax(1)
            weight = q ** 2 / q.sum(0)
            p = (weight.T / weight.sum(1)).T

            combined_input = JoinedGen(train_data_dcec, p, config.batch_size)

            loss = model.fit(x=combined_input, epochs=1, callbacks=[Callback_DCEC(train_data=train_data_dcec, val_data=val_data_dcec, y_true_val=y_true_val, labels=labels_val, 
                                                                                n_clusters=config.clusters, datatype=config.data, save_directory=config.save_directory, 
                                                                                batch_size=config.batch_size, epoch=epoch, y_pred_last=y_pred)])
            
            hist_df = pd.DataFrame(loss.history) 
            hist_csv_file = config.save_directory + 'logs-dcec.csv'
            if epoch == 1: 
                with open(hist_csv_file, mode='w') as f:
                    hist_df.to_csv(f)
            else: 
                with open(hist_csv_file, 'a') as f_object:
                    # write new training updates to csv file as new row 
                    writer_object = writer(f_object)

                    values_df = hist_df.iloc[0]
                    values_lst = values_df.tolist()
                    values_lst = [epoch] + values_lst
                    writer_object.writerow(values_lst)
                    f_object.close()

    ## VALIDATE model
    # TEST MODEL
    if config.test is True and config.location_model2_final is not None:

        tsne_bool = True
        makeprediction_bool = True

        clusters = 7
        gamma = 0.05

        t1 = time()

        print("Testing...\n")

        directory = '/Users/Sterre/_Python_folders/ai_needle_emg_internshipSterre/results/arch_two/30122022_0940_bestmodel'

        validation_directory = (directory + '/evaluation')
        if not os.path.exists(validation_directory): os.makedirs(validation_directory)  
        
        try:
            dcec = keras.models.load_model(str(config.location_model2_final), 
                custom_objects={"DCEC": DCEC, "ClusteringLayer": ClusteringLayer}, compile=False)
            print("\nPre-trained model loaded succesfully.\n")
        except:
            print("\nCould not load pre-trained model. Try with clr learning rate?")
        # TODO: how to know and how to import a model with cyclical learning rate? Just don't know! > with compile=False it works now, adding scale_fn as custom object does not.
        
        # retrieve latent space from trained model
        latent_space_model = Model(inputs=dcec.input, outputs=dcec.get_layer(name='embedding').output)
        features_train, features_test = latent_space_model.predict(train_data_cae), latent_space_model.predict(test_data_st) 
        features_train, features_test = np.reshape(features_train, newshape=(features_train.shape[0], -1)), np.reshape(features_test, newshape=(features_test.shape[0], -1))

        # make predictions
        clustering = DCEC(inputs=dcec.input, outputs=dcec.output[1])
        q_train = clustering.predict(train_data_dcec)
        y_pred_train = q_train.argmax(1)

        q_test = clustering.predict(test_data_dcec)
        y_pred_test = q_test.argmax(1)
        
        # initalise validation object
        validation = Validation(clusters=clusters, datatype=config.data, show_displays=config.displays, save_directory=validation_directory)

        # validate
        acc, precision, recall, fscore, support = validation.validate(y_true_test_dcec, y_pred_test, labels_test_dcec, argument='test')
        print(acc)

        # soft metrics
        cluster_centres = dcec.get_layer(name='clustering').get_weights_()
        cluster_centres = np.asarray(cluster_centres)
        cluster_centres = np.reshape(cluster_centres, [cluster_centres.shape[1], cluster_centres.shape[2]])

        latent_space_model = Model(inputs=dcec.input, outputs=dcec.get_layer(name='embedding').output)

        features_test = latent_space_model.predict(test_data_dcec) 
        features_test = np.reshape(features_test, newshape=(features_test.shape[0], -1))

        features_train = latent_space_model.predict(train_data_cae)
        features_train = np.reshape(features_train, newshape=(features_train.shape[0], -1))

        confidence, perc_removed_train = validation.get_optimal_confidence(features_train, y_pred_train, cluster_centres)
        print("confidence = ", confidence)

        features_test_soft, y_pred_test_soft, y_true_soft = validation.validate_soft_clustering(features_test, y_pred_test, confidence, y_true=y_true_test_st, cluster_centres=cluster_centres)
        features_train_soft, y_pred_train_soft = validation.validate_soft_clustering(features_train, y_pred_train, confidence, y_true=None, cluster_centres=cluster_centres)

        indices_soft_y_pred_test, y_pred_test_soft_notused = validation.validate_soft_clustering(features_test, y_pred_test, confidence, y_true=None, cluster_centres=cluster_centres)

        y_pred_test_soft_clusters = []
        for index in indices_soft_y_pred_test:
            files_soft_sample = y_pred_test[index]
            y_pred_test_soft_clusters.append(files_soft_sample)

        # confusion matrix
        cm = confusion_matrix(y_true_test_st, y_pred_test) 
        cm_argmax = cm.argmax(axis=0) 
        y_pred_test = np.array([cm_argmax[k] for k in y_pred_test]) 
        y_pred_train = np.array([cm_argmax[k] for k in y_pred_train])

        cm_soft = confusion_matrix(y_true_soft, y_pred_test_soft) 
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

            autoencoder = DCEC(inputs=dcec.input, outputs=dcec.output[0])

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
            a.write("Logfile with results for model with location {} \n".format(config.location_model2_final))
            a.write("Signal Type test data is used; 30 procent of dataset\n")
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

      
