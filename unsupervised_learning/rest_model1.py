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
# custom made imports 
from unsupervised_model_options import UnsupervisedOptions
from model1 import ConvAE_clustering

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

class Callback_CAE(keras.callbacks.Callback):
    def __init__(self, train_data, val_data, y_true, labels, n_clusters, datatype, save_directory, soft_metrics):
        super(Callback_CAE, self).__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.y_pred_last = None
        self.y_true = y_true
        self.labels = labels
        self.datatype = datatype
        self.validation = Validation(clusters=n_clusters, datatype=self.datatype, show_displays=False, save_directory=False)
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

        acc, precision, recall, fscore, support = self.validation.validate(self.y_true, y_pred_val, self.labels, argument='train')

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

            confidence, perc_removed = self.validation.get_optimal_confidence(features_train, y_pred_train)
            print(confidence)

            features_test_soft, y_pred_test_soft, y_true_soft = self.validation.validate_soft_clustering(features_test, y_pred_val, confidence, y_true=self.y_true)
            acc_soft, precision_soft, recall_soft, fscore_soft, support = self.validation.validate(y_true_soft, y_pred_test_soft, self.labels, argument='train')
            logs['acc_soft'] = acc_soft
            logs['f1_rest_soft'] = fscore_soft[5]
            logs['f1_fibrillation_soft'] = fscore_soft[2]
            logs['f1_PSW_soft'] = fscore_soft[4]
            logs['f1_Fib_PSW_soft'] = fscore_soft[1]
            logs['f1_CRD_soft'] = fscore_soft[0]
            logs['f1_Myotonic_discharge_soft'] = fscore_soft[3]
            logs['confidence'] = confidence
            logs['perc_removed_train'] = perc_removed
            logs['percentage_removed_val'] = 100 - ((len(y_pred_test_soft) / len(y_pred_val)) * 100)
            print('Soft metrics with confidence %s:  acc=%.3f, F1 rest=%.3f, F1 fibrillation=%.3f, F1 PSW=%.3f, F1 fib+PSW=%.3f, F1 CRD=%.3f, F1 myotonic discharge=%.3f' % (logs['epoch'], acc_soft, fscore_soft[5], fscore_soft[2], fscore_soft[4], fscore_soft[1], fscore_soft[0], fscore_soft[3]))

        print('saving model to:', self.save_directory + 'models/convae_model_' + str(logs['epoch']) + '.hdf5 \n')
        self.model.save(self.save_directory + 'models/convae_model_' + str(logs['epoch']) + '.hdf5')

if __name__ == "__main__":

    config = UnsupervisedOptions().parse()

    if config.architecture != "arch_one_rest": 
        print("Wrong architecture type is given in configuration file.")
        exit()

    print("\nMain file is running...")
    print("Configuration settings are stored in %s\n" % config.save_directory)

    # LOAD DATA
    datagen = ImageDataGenerator(rescale=1./255) 
    print("Data generator is initialised \n")
    print("Train data")
    train_data = datagen.flow_from_directory(
                directory = config.training_rest, 
                target_size = config.input_dimension, 
                batch_size = config.best_cae['batch_size'],
                class_mode = 'input', 
                color_mode = 'grayscale', 
                shuffle=False 
        )
    print("Validation data")    
    val_data = datagen.flow_from_directory(
                directory = config.validation_rest, 
                target_size = config.input_dimension, 
                batch_size = config.best_cae['batch_size'],
                class_mode = 'input', 
                color_mode = 'grayscale', 
                shuffle = False 
                ) 
    labels_val = list(val_data.class_indices.keys())
    y_true_val = val_data.classes

    if config.train is True:

        print("Training of the model is started...\n")
        t0 = time()

        # model 
        autoencoder = ConvAE_clustering.CAE(input_shape=(128, 128, 1), 
                                            kernel_size=config.best_cae['kernel_size'], 
                                            activation=config.best_cae['activation'], 
                                            filters=config.best_cae['filters'], 
                                            features=config.best_cae['features'], 
                                            batch_norm=config.best_cae['batch_norm'])
        plot_model(autoencoder, to_file=config.save_directory + 'model-architecture.png', show_shapes=True)
        # autoencoder.summary()

        if config.best_cae['learning_schedule_convae'] == 'constant' and config.best_cae['optimizer'] == 'adam': opt = keras.optimizers.Adam(config.best_cae['learning_rate_convae'])  
        if config.best_cae['learning_schedule_convae'] == 'constant' and config.best_cae['optimizer'] == 'sgd': opt = keras.optimizers.SGD(config.best_cae['learning_rate_convae'])  
        if config.best_cae['learning_schedule_convae'] == 'clr1':
            clr = tfa.optimizers.CyclicalLearningRate(initial_learning_rate=config.best_cae['learning_rate_convae'][0], 
                                                    maximal_learning_rate=config.best_cae['learning_rate_convae'][1], 
                                                        scale_fn=lambda x: 1.0, 
                                                        step_size= (train_data.samples / config.batch_size) * 2) # step_size defines the duration of a single cycle, a size of 2 means you need a total of 4 iterations to complete one cycle 
            
            if config.best_cae['optimizer'] == 'adam': opt = keras.optimizers.Adam(clr)
            if config.best_cae['optimizer'] == 'sgd': opt = keras.optimizers.SGD(clr)        
        if config.best_cae['learning_schedule_convae'] == 'clr2':
            clr = tfa.optimizers.CyclicalLearningRate(initial_learning_rate=config.best_cae['learning_rate_convae'][0], 
                                                    maximal_learning_rate=config.best_cae['learning_rate_convae'][1], 
                                                        scale_fn=lambda x: 1/(2.**(x-1)), 
                                                        step_size= (train_data.samples / config.batch_size) * 2) 
            
            if config.best_cae['optimizer'] == 'adam': opt = keras.optimizers.Adam(clr)
            if config.best_cae['optimizer'] == 'sgd': opt = keras.optimizers.SGD(clr)

        autoencoder.compile(optimizer=opt, loss=config.best_cae['loss_convae'])

        loss = autoencoder.fit(train_data, epochs=config.epochs_convae, callbacks=[Callback_CAE(train_data, val_data, y_true_val, labels_val, config.best_cae['clusters'], config.data, config.save_directory, soft_metrics=True), CSVLogger(config.save_directory + 'logs-convae.csv')])
    

    if config.test is True and config.location_model1_rest is not None:

        tsne_bool = True
        makeprediction_bool = True

        t1 = time()

        print("Testing...\n")

        directory = '/Users/Sterre/_Python_folders/ai_needle_emg_internshipSterre/results/arch_one_rest/230103_1923_bestmodel'

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
        features_train, features_test = latent_space_model.predict(train_data), latent_space_model.predict(val_data) 
        features_train, features_test = np.reshape(features_train, newshape=(features_train.shape[0], -1)), np.reshape(features_test, newshape=(features_test.shape[0], -1))

        # initalise validation object
        validation = Validation(clusters=config.clusters, datatype=config.data, show_displays=config.displays, save_directory=validation_directory)

        # predict
        y_pred_train, y_pred_test, deltalabel = validation.clustering(features_train, features_test)           

        # validate
        acc, precision, recall, fscore, support = validation.validate(y_true_val, y_pred_test, labels_val, argument='test')

        confidence, perc_removed_train = validation.get_optimal_confidence(features_train, y_pred_train)
        # predict for soft labels
        features_test_soft, y_pred_test_soft, y_true_soft = validation.validate_soft_clustering(features_test, y_pred_test, confidence, y_true_val)
        features_train_soft, y_pred_train_soft = validation.validate_soft_clustering(features_train, y_pred_train, confidence, y_true=None)

        indices_soft_y_pred_test, y_pred_test_soft_notused = validation.validate_soft_clustering(features_test, y_pred_test, confidence, y_true=None)

        y_pred_test_soft_clusters = []
        for index in indices_soft_y_pred_test:
            files_soft_sample = y_pred_test[index]
            y_pred_test_soft_clusters.append(files_soft_sample)
        
        # confusion matrix
        cm = confusion_matrix(y_true_val, y_pred_test) 
        cm_argmax = cm.argmax(axis=0) 
        y_pred_test = np.array([cm_argmax[k] for k in y_pred_test]) 
        y_pred_train = np.array([cm_argmax[k] for k in y_pred_train])

        cm_soft = confusion_matrix(y_true_soft, y_pred_test_soft_clusters) 
        print(cm_soft)
        cm_argmax_soft = cm_soft.argmax(axis=0)  
        y_pred_test_soft = np.array([cm_argmax_soft[k] for k in y_pred_test_soft])
        y_pred_train_soft = np.array([cm_argmax_soft[k] for k in y_pred_train_soft])

        # validate with soft clustering
        acc_soft, precision_soft, recall_soft, fscore_soft, support_soft = validation.validate(y_true_soft, y_pred_test_soft_clusters, labels_val, argument='test_soft')

        
        




        print('\nTesting time: ', time() - t1)

    