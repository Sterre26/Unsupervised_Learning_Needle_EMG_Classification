# standard libraries 
import numpy as np
import os
from time import time
import csv
from csv import writer
import pandas as pd
from tqdm import tqdm, trange
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import numpy.matlib as matlib

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
# from displays import display_clusters, display_example_predictions, display_confusionmatrix, display_performance_dcec
from unsupervised_model_options import UnsupervisedOptions
from rest_model1 import ConvAE_clustering, Callback_CAE

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
        
        acc = accuracy_score(y_true, y_pred)
        precision, recall, fscore, support = precision_recall_fscore_support(y_true, y_pred, labels = [0,1,2,3,4,5]) 
        
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

        softDF['confidence'] = np.max(softDF[probabilities_range].values, axis = 1)

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

        softDF['confidence'] = np.max(softDF[probabilities_range].values, axis = 1)

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

class Callback_DCEC(keras.callbacks.Callback):
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

        _, q_val = self.model.predict(self.val_data)
        y_pred_val = q_val.argmax(1)
        weight = q_val ** 2 / q_val.sum(0)
        p_val = (weight.T / weight.sum(1)).T

        val_input = JoinedGen(self.val_data, p_val, self.batch_size)

        clustering = DCEC(inputs=self.model.input, outputs=self.model.output[1], gamma=config.gamma, batch_size=config.batch_size)

        val_loss = self.model.evaluate(x=val_input, verbose=0)
        logs['val_loss'] = val_loss[1]
        logs['val_reconstruction_loss'] = val_loss[2]
        logs['val_clustering_loss'] = val_loss[0]

        q = clustering.predict(self.train_data)
        y_pred_train = q.argmax(1)

        try: 
            delta_label = np.sum(y_pred_train != self.y_pred_last).astype(np.float32) / y_pred_train.shape[0]
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
            logs['f1_rest'] = np.nan
            logs['f1_fibrillation'] = np.nan
            logs['f1_PSW'] = np.nan
            logs['f1_Fib_PSW'] = np.nan
            logs['f1_CRD'] = np.nan
            logs['f1_Myotonic_discharge'] = np.nan         
        if acc is not None:
            logs['acc'] = acc
            logs['f1_rest'] = fscore[5]
            logs['f1_fibrillation'] = fscore[2]
            logs['f1_PSW'] = fscore[4]
            logs['f1_Fib_PSW'] = fscore[1]
            logs['f1_CRD'] = fscore[0]
            logs['f1_Myotonic_discharge'] = fscore[3]

            print('Testing metrics at epoch %s:         acc=%.3f, F1 rest=%.3f, F1 fibrillation=%.3f, F1 PSW=%.3f, F1 fib+PSW=%.3f, F1 CRD=%.3f, F1 myotonic discharge=%.3f' % (logs['epoch'], acc, fscore[5], fscore[2], fscore[4], fscore[1], fscore[0], fscore[3]))

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
            logs['acc_soft'] = acc
            logs['f1_rest_soft'] = fscore[5]
            logs['f1_fibrillation_soft'] = fscore[2]
            logs['f1_PSW_soft'] = fscore[4]
            logs['f1_Fib_PSW_soft'] = fscore[1]
            logs['f1_CRD_soft'] = fscore[0]
            logs['f1_Myotonic_discharge_soft'] = fscore[3]
            logs['confidence'] = confidence
            logs['perc_removed_train'] = perc_removed_train
            logs['percentage_removed_val'] = 100 - ((len(y_pred_val_soft) / len(y_pred_val)) * 100)
            print('Soft metrics with confidence %s:  acc=%.3f, F1 rest=%.3f, F1 fibrillation=%.3f, F1 PSW=%.3f, F1 fib+PSW=%.3f, F1 CRD=%.3f, F1 myotonic discharge=%.3f' % (logs['epoch'], acc_soft, fscore_soft[5], fscore_soft[2], fscore_soft[4], fscore_soft[1], fscore_soft[0], fscore_soft[3]))

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
    def __init__(self, gamma, batch_size, **kwargs):
        super(DCEC, self).__init__(**kwargs)
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = tf.metrics.Mean(name="kl_loss")
        self.gamma = gamma
        self.batch_size = batch_size

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

    if config.architecture != "arch_two_rest": 
        print("Wrong architecture type is given in configuration file.")
        exit()

    print("\nMain file is running...")
    print("Configuration settings are stored in %s\n" % config.save_directory)

    # LOAD DATA
    datagen = ImageDataGenerator(rescale=1./255) 
    print("Data generator is initialised \n")
    print("Train data")
    train_data_cae = datagen.flow_from_directory(
                directory = config.training_rest, 
                target_size = config.input_dimension, 
                batch_size = config.best_dcec['batch_size'],
                class_mode = 'input', 
                color_mode = 'grayscale', 
                shuffle=False 
        )
    train_data_dcec = datagen.flow_from_directory(
                directory = config.training_rest, 
                target_size = config.input_dimension, 
                batch_size = config.best_dcec['batch_size'],
                class_mode = None, 
                color_mode = 'grayscale', 
                shuffle=False 
        )
    print("Validation data")    
    val_data_cae = datagen.flow_from_directory(
                directory = config.validation_rest, 
                target_size = config.input_dimension, 
                batch_size = config.best_dcec['batch_size'],
                class_mode = 'input', 
                color_mode = 'grayscale', 
                shuffle = False 
                ) 
    val_data_dcec = datagen.flow_from_directory(
                directory = config.validation_rest, 
                target_size = config.input_dimension, 
                batch_size = config.best_dcec['batch_size'],
                class_mode = None, 
                color_mode = 'grayscale', 
                shuffle = False 
                ) 
    labels_val = list(val_data_cae.class_indices.keys())
    y_true_val = val_data_cae.classes


    ## TRAIN model 
    if config.train is True:

        autoencoder = ConvAE_clustering.CAE(input_shape=(128, 128, 1), 
                                            kernel_size=config.best_dcec['kernel_size'], 
                                            activation=config.best_dcec['activation'], 
                                            filters=config.best_dcec['filters'], 
                                            features=config.best_dcec['features'], 
                                            batch_norm=config.best_dcec['batch_norm'])
        plot_model(autoencoder, to_file=config.save_directory + 'model-architecture.png', show_shapes=True)
        autoencoder.summary()

        if config.best_dcec['learning_schedule_convae'] == 'constant' and config.best_dcec['optimizer'] == 'adam': opt = tf.keras.optimizers.Adam(config.best_dcec['learning_rate_convae'])  
        if config.best_dcec['learning_schedule_convae'] == 'constant' and config.best_dcec['optimizer'] == 'sgd': opt = tf.keras.optimizers.SGD(config.best_dcec['learning_rate_convae'])  
        if config.best_dcec['learning_schedule_convae'] == 'clr1':
            clr = tfa.optimizers.CyclicalLearningRate(initial_learning_rate=config.best_dcec['learning_rate_convae'][0], 
                                                    maximal_learning_rate=config.best_dcec['learning_rate_convae'][1], 
                                                        scale_fn=lambda x: 1.0, 
                                                        step_size= (train_data_cae.samples / config.batch_size) * 2) # step_size defines the duration of a single cycle, a size of 2 means you need a total of 4 iterations to complete one cycle 
            
            if config.best_dcec['optimizer'] == 'adam': opt = keras.optimizers.Adam(clr)
            if config.best_dcec['optimizer'] == 'sgd': opt = keras.optimizers.SGD(clr)        
        if config.best_dcec['learning_schedule_convae'] == 'clr2':
            clr = tfa.optimizers.CyclicalLearningRate(initial_learning_rate=config.best_dcec['learning_rate_convae'][0], 
                                                    maximal_learning_rate=config.best_dcec['learning_rate_convae'][1], 
                                                        scale_fn=lambda x: 1/(2.**(x-1)), 
                                                        step_size= (train_data_cae.samples / config.batch_size) * 2) 
            
            if config.best_dcec['optimizer'] == 'adam': opt = keras.optimizers.Adam(clr)
            if config.best_dcec['optimizer'] == 'sgd': opt = keras.optimizers.SGD(clr)

        autoencoder.compile(optimizer=opt, loss=config.best_dcec['loss_convae'])

        hidden = autoencoder.get_layer(name='embedding').output
        encoder = Model(inputs=autoencoder.input, outputs=hidden)

        # Step 1: pretrain model or load pretrained model
        if config.pretrain is True:

            loss = autoencoder.fit(train_data_cae, epochs=config.epochs_convae, callbacks=[Callback_CAE(train_data_cae, val_data_cae, y_true_val, labels_val, config.best_dcec['clusters'], config.data, config.save_directory, soft_metrics=True), CSVLogger(config.save_directory + 'logs-convae.csv')])


            cae_weights = config.save_directory + '/pretrain_cae_model.h5'
            autoencoder.save_weights(cae_weights)
            print('Pretrained weights are saved to %s' % cae_weights)

            # cae_weights = dcec2.pretrain(train_data, val_data, test_data, epochs=config.epochs_convae, optimizer=opt, loss=config.loss_convae, save_dir=config.save_directory)
        if config.pretrain is False and config.pretrained_cae is not None:
            cae_weights = config.pretrained_cae
            autoencoder.load_weights(cae_weights)
            print('CAE weights from %s are loaded successfully.' % cae_weights)

        else: 
            "Pretraining needs to be performed (set argument --pretrain to True) OR location to pretrained model needs to be provided with argument --pretrained_cae [insert location]"

        # Step 2: initialize cluster centers using k-means
        print('Initializing cluster centers with k-means ({} clusters are created)'.format(config.best_dcec['clusters']))
        kmeans = KMeans(n_clusters=config.best_dcec['clusters'], n_init=100)

        y_pred = kmeans.fit_predict(encoder.predict(train_data_cae))

        # Step 3: Initialize and train DCEC
        clustering_layer = ClusteringLayer(config.best_dcec['clusters'], name='clustering')(hidden)
        
        model = DCEC(inputs=autoencoder.input, outputs=[autoencoder.output, clustering_layer], gamma=config.best_dcec['gamma'], batch_size=config.best_dcec['batch_size'])
        
        plot_model(model, to_file=config.save_directory + '/dcec_model.png', show_shapes=True)
        model.summary()
        
        if config.best_dcec['learning_schedule_dcec'] == 'constant': opt = tf.keras.optimizers.Adam(config.best_dcec['learning_rate_dcec'])  
        if config.best_dcec['learning_schedule_dcec'] == 'clr1':
            clr = tfa.optimizers.CyclicalLearningRate(initial_learning_rate=config.best_dcec['learning_rate_dcec'][0], 
                                                        maximal_learning_rate=config.best_dcec['learning_rate_dcec'][1], 
                                                        scale_fn = lambda x: 1.0,           # clr1
                                                        step_size= (train_data_dcec.samples / config.best_dcec['batch_size']) * 2) # step_size defines the duration of a single cycle, a size of 2 means you need a total of 4 iterations to complete one cycle 
                
            opt = tf.keras.optimizers.Adam(clr)
        if config.best_dcec['learning_schedule_dcec'] == 'clr2':
            clr = tfa.optimizers.CyclicalLearningRate(initial_learning_rate=config.best_dcec['learning_rate_dcec'][0], 
                                                        maximal_learning_rate=config.best_dcec['learning_rate_dcec'][1], 
                                                        scale_fn=lambda x: 1/(2.**(x-1)), # clr2
                                                        step_size= (train_data_dcec.samples / config.best_dcec['batch_size']) * 2) # step_size defines the duration of a single cycle, a size of 2 means you need a total of 4 iterations to complete one cycle 
                
            opt = tf.keras.optimizers.Adam(clr)

        model.compile(optimizer=opt)

        model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])
  
        for epoch in range(1, config.epochs_dcec+1):
            print("epoch {}/{}".format(epoch, config.epochs_dcec))
            
            # define seperate instance of clustering output, so that prediction is memory efficient for predicting q
            clustering = DCEC(inputs=model.input, outputs=model.output[1], gamma=config.best_dcec['gamma'], batch_size=config.best_dcec['batch_size'])
        
            q = clustering.predict(train_data_dcec)
            y_pred = q.argmax(1)
            weight = q ** 2 / q.sum(0)
            p = (weight.T / weight.sum(1)).T

            combined_input = JoinedGen(train_data_dcec, p, config.best_dcec['batch_size'])

            loss = model.fit(x=combined_input, epochs=1, callbacks=[Callback_DCEC(train_data=train_data_dcec, val_data=val_data_dcec, y_true_val=y_true_val, labels=labels_val, 
                                                                                n_clusters=config.best_dcec['clusters'], datatype=config.data, save_directory=config.save_directory, 
                                                                                batch_size=config.best_dcec['batch_size'], epoch=epoch, y_pred_last=y_pred)])
            
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