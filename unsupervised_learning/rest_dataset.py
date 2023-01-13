# standard libraries 
import numpy as np
import os
import shutil
from PIL import Image
from tqdm import trange
# tensorflow & keras
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
# sklearn
from sklearn.metrics import confusion_matrix
# custom made imports 
from unsupervised_model_options import UnsupervisedOptions
from model1 import Validation
from data_augmentation import Skew, Distort
from model2 import DCEC, ClusteringLayer

"""
This file creates the rest dataset for Mel spectrograms that are classified as rest 
(both from training set and from signal type dataset) by both models. 

@author: Sterre de Jonge (2022)
"""

class MoveFilesClassifiedAsRest:

    def __init__(self):
        print("") 
    
    def move(self):

        # training and validation directories
        directory_training_all = '/Users/Sterre/files/PROCESSED/mels_size_square/Training_100percent'
        directory_validation_all = '/Users/Sterre/files/PROCESSED/mels_size_square/Validation_70procent'
        # final model directoy
        directory_model_1 = config.location_model1_final
        directory_model_2 = config.location_model2_final
        # directories for moving
        directory_source = '/Users/Sterre/files/PROCESSED/mels_size_square/Training_100percent/train/'
        directory_destination = '/Users/Sterre/files/PROCESSED/mels_size_square/Training_m2_rest/train/'
        directory_destination_soft = '/Users/Sterre/files/PROCESSED/mels_size_square/Training_m2_rest_soft/train/'
        
        if not os.path.exists(directory_destination): 
            os.makedirs(directory_destination)
            print("Directory for rest did not exist and is created.")
        if not os.path.exists(directory_destination_soft): 
            os.makedirs(directory_destination_soft)
            print("Directory for soft rest did not exist and is created.")

        # load data
        datagen = ImageDataGenerator(rescale=1./255) 
        print("Data generator is initialised \n")
        print("Train data")
        train_data = datagen.flow_from_directory(
                    directory = directory_training_all, 
                    target_size = config.input_dimension, 
                    batch_size = config.batch_size,
                    class_mode = 'input', 
                    color_mode = 'grayscale', 
                    shuffle=False 
            )
        train_data_dcec = datagen.flow_from_directory(
                    directory = directory_training_all, 
                    target_size = config.input_dimension, 
                    batch_size = config.batch_size,
                    class_mode = None, 
                    color_mode = 'grayscale', 
                    shuffle=False 
            )
        print("Validation data")
        val_data = datagen.flow_from_directory(
                    directory = directory_validation_all, 
                    target_size = config.input_dimension, 
                    batch_size = config.batch_size,
                    class_mode = 'categorical', 
                    color_mode = 'grayscale', 
                    shuffle = False 
                    )
        val_data_dcec = datagen.flow_from_directory(
                    directory = directory_validation_all, 
                    target_size = config.input_dimension, 
                    batch_size = config.batch_size,
                    class_mode = None, 
                    color_mode = 'grayscale', 
                    shuffle = False 
                    )
        labels = list(val_data.class_indices.keys())
        y_true = val_data.classes
        print("")

        # load model
        autoencoder = keras.models.load_model(str(directory_model_1))
        print("Model 1 loaded succesfully.\n")

        try:
            dcec = keras.models.load_model(str(config.location_model2_final), 
                custom_objects={"DCEC": DCEC, "ClusteringLayer": ClusteringLayer}, compile=False)
            print("\nPre-trained model loaded succesfully.\n")
        except:
            print("\nCould not load pre-trained model. Try with clr learning rate?")
        # TODO: how to know and how to import a model with cyclical learning rate? Just don't know! > with compile=False it works now, adding scale_fn as custom object does not.
        
        # retrieve latent space from trained model
        self.latent_space_model = Model(inputs=dcec.input, outputs=dcec.get_layer(name='embedding').output)
        features_train, features_test = self.latent_space_model.predict(train_data), self.latent_space_model.predict(val_data) 
        features_train, features_test = np.reshape(features_train, newshape=(features_train.shape[0], -1)), np.reshape(features_test, newshape=(features_test.shape[0], -1))

        # make predictions
        self.clustering = DCEC(inputs=dcec.input, outputs=dcec.output[1])
        q_train = self.clustering.predict(train_data_dcec)
        y_pred_train = q_train.argmax(1)

        q_test = self.clustering.predict(val_data_dcec)
        y_pred_test = q_test.argmax(1)

        # initalise validation object
        self.validation = Validation(clusters=7, show_displays=False, save_directory='results')

        # figure out which clusters belong to which label!
        cm = confusion_matrix(y_true, y_pred_test) # confusion matrix for all clusters
        self.cm_argmax = cm.argmax(axis=0)

        acc, precision, recall, fscore, support = self.validation.validate(y_true, y_pred_test, labels)

        # soft metrics
        self.cluster_centres = dcec.get_layer(name='clustering').get_weights_()
        self.cluster_centres = np.asarray(self.cluster_centres)
        self.cluster_centres = np.reshape(self.cluster_centres, [self.cluster_centres.shape[1], self.cluster_centres.shape[2]])

        latent_space_model = Model(inputs=dcec.input, outputs=dcec.get_layer(name='embedding').output)

        features_test = latent_space_model.predict(val_data) 
        features_test = np.reshape(features_test, newshape=(features_test.shape[0], -1))

        features_train = latent_space_model.predict(train_data)
        features_train = np.reshape(features_train, newshape=(features_train.shape[0], -1))

        self.confidence, perc_removed_train = self.validation.get_optimal_confidence(features_train, y_pred_train, self.cluster_centres)
        print("confidence = ", self.confidence)

        features_test_soft, y_pred_test_soft, y_true_soft = self.validation.validate_soft_clustering(features_test, y_pred_test, self.confidence, y_true, cluster_centres=self.cluster_centres)
        cm_soft = confusion_matrix(y_true_soft, y_pred_test_soft) # confusion matrix for all clusters
        self.cm_argmax_soft = cm_soft.argmax(axis=0)
            
        y_pred_test = np.array([self.cm_argmax[k] for k in y_pred_test]) # values 0,1,2 instead of 0,1,2,3,etc.
        y_pred_train = np.array([self.cm_argmax[k] for k in y_pred_train])

        # predict for soft labels
        indices_soft, y_pred_train_soft = self.validation.validate_soft_clustering(features_train, y_pred_train, self.confidence, y_true=None, cluster_centres=self.cluster_centres)

        # check if files were already moved
        files_rest = [f for f in os.listdir(directory_destination) if ".png" in f]
        if len(files_rest) > 0: print("Files were already moved to destination folder. If you wish to redo it, you should remove the folders.")
        else:
            # move files classified as rest
            files = [f for f in os.listdir(directory_source) if ".png" in f]
            howmany = 0
            for index, item in enumerate(y_pred_train):
                if item == 2: # class = rest
                    shutil.copyfile(directory_source + files[index], directory_destination + files[index])
                    howmany += 1
            print("Files classified as rest that are moved: {}".format(howmany))

            # move files that are classified as rest with soft clustering

            files_soft = []
            for index in indices_soft:
                files_soft_sample = files[index]
                files_soft.append(files_soft_sample)
            howmany = 0
            for index, item in enumerate(y_pred_train_soft):
                if item == 2: # class = rest
                    shutil.copyfile(directory_source + files_soft[index], directory_destination_soft + files[index])
                    howmany += 1

            print("Files classified as rest that are moved with soft metrics: {}".format(howmany))

    def move_rest(self):
        directory_val_rest = '/Users/Sterre/files/PROCESSED/mels_size_square/Rest/original_rest_labelled'

        # load validation data rest
        datagen = ImageDataGenerator(rescale=1./255) 
        print("Data generator is initialised \n")
        print("Validation data")
        val_data_rest = datagen.flow_from_directory(
                    directory = directory_val_rest, 
                    target_size = config.input_dimension, 
                    batch_size = config.batch_size,
                    class_mode = 'categorical', 
                    color_mode = 'grayscale', 
                    shuffle = False 
                    )
        labels_rest = list(val_data_rest.class_indices.keys())
        y_true_rest = val_data_rest.classes
        print("")

        # retrieve embedding
        features_rest = self.latent_space_model.predict(val_data_rest)
        features_rest = np.reshape(features_rest, newshape=(features_rest.shape[0], -1))

        # predict
        q_rest = self.clustering.predict(val_data_rest)
        y_pred_rest = q_rest.argmax(1)

        

        # get soft labels
        features_test_soft, y_pred_rest_soft, y_true_rest_soft = self.validation.validate_soft_clustering(features_rest, y_pred_rest, self.confidence, y_true_rest, cluster_centres=self.cluster_centres)
        indices_soft, y_pred_rest_soft = self.validation.validate_soft_clustering(features_rest, y_pred_rest, self.confidence, y_true=None, cluster_centres=self.cluster_centres)

        print("confidence", self.confidence)

        y_pred_rest = np.array([self.cm_argmax[k] for k in y_pred_rest])
        y_pred_rest_soft = np.array([self.cm_argmax_soft[k] for k in y_pred_rest_soft])

        # move files
        directory_val_rest = '/Users/Sterre/files/PROCESSED/mels_size_square/Rest/original_rest_labelled/'
        directory_destination = '/Users/Sterre/files/PROCESSED/mels_size_square/Rest/Validation_rest_m2_rest/'
        directory_destination_soft = '/Users/Sterre/files/PROCESSED/mels_size_square/Rest/Validation_rest_m2_soft/'
        if not os.path.exists(directory_destination): 
            os.makedirs(directory_destination)
            print("Directory for rest did not exist and is created.")
        if not os.path.exists(directory_destination_soft): 
            os.makedirs(directory_destination_soft)
            print("Directory for soft rest did not exist and is created.")

        files_CRD = [f for f in os.listdir(directory_val_rest + '/CRD/') if ".png" in f]
        files_Fib_PSW = [f for f in os.listdir(directory_val_rest + '/Fib_PSW/') if ".png" in f]
        files_Fib = [f for f in os.listdir(directory_val_rest + '/Fibrillation/') if ".png" in f]
        files_Myo = [f for f in os.listdir(directory_val_rest + '/Myotonic_discharge/') if ".png" in f]
        files_PSW = [f for f in os.listdir(directory_val_rest + '/PSW/') if ".png" in f]
        files_Rest = [f for f in os.listdir(directory_val_rest + '/Rest/') if ".png" in f]

        files = files_CRD + files_Fib_PSW + files_Fib + files_Myo + files_PSW + files_Rest
        print(len(files))

        indices = [0, len(files_CRD), len(files_CRD)+len(files_Fib_PSW), len(files_CRD)+len(files_Fib_PSW)+len(files_Fib), len(files_CRD)+len(files_Fib_PSW)+len(files_Fib)+len(files_Myo),
                    len(files_CRD)+len(files_Fib_PSW)+len(files_Fib)+len(files_Myo)+len(files_PSW)]

        for i, item in enumerate(y_pred_rest):
            if item == 2:
                if i >= indices[0] and i < indices[1]: shutil.copyfile(directory_val_rest + 'CRD/' + files[i], directory_destination + 'CRD/' + files[i])
                if i >= indices[1] and i < indices[2]: shutil.copyfile(directory_val_rest + 'Fib_PSW/' + files[i], directory_destination + 'Fib_PSW/' + files[i])
                if i >= indices[2] and i < indices[3]: shutil.copyfile(directory_val_rest + 'Fibrillation/' + files[i], directory_destination + 'Fibrillation/' + files[i])
                if i >= indices[3] and i < indices[4]: shutil.copyfile(directory_val_rest + 'Myotonic_discharge/' + files[i], directory_destination + 'Myotonic_discharge/' + files[i])
                if i >= indices[4] and i < indices[5]: shutil.copyfile(directory_val_rest + 'PSW/' + files[i], directory_destination + 'PSW/' + files[i])
                if i >= indices[5]: shutil.copyfile(directory_val_rest + 'Rest/' + files[i], directory_destination + 'Rest/' + files[i])

        for i, item in zip(indices_soft, y_pred_rest_soft):
            if item == 2:
                if i >= indices[0] and i < indices[1]: shutil.copyfile(directory_val_rest + 'CRD/' + files[i], directory_destination_soft + 'CRD/' + files[i])
                if i >= indices[1] and i < indices[2]: shutil.copyfile(directory_val_rest + 'Fib_PSW/' + files[i], directory_destination_soft + 'Fib_PSW/' + files[i])
                if i >= indices[2] and i < indices[3]: shutil.copyfile(directory_val_rest + 'Fibrillation/' + files[i], directory_destination_soft + 'Fibrillation/' + files[i])
                if i >= indices[3] and i < indices[4]: shutil.copyfile(directory_val_rest + 'Myotonic_discharge/' + files[i], directory_destination_soft + 'Myotonic_discharge/' + files[i])
                if i >= indices[4] and i < indices[5]: shutil.copyfile(directory_val_rest + 'PSW/' + files[i], directory_destination_soft + 'PSW/' + files[i])
                if i >= indices[5]: shutil.copyfile(directory_val_rest + 'Rest/' + files[i], directory_destination_soft + 'Rest/' + files[i])
  
        classified_rest = np.count_nonzero(y_pred_rest == 2)

        CRD_rest = len([a for i, a in enumerate(y_true_rest) if a == 0 and y_pred_rest[i] == 2])
        CRD_contraction = len([a for i, a in enumerate(y_true_rest) if a == 0 and y_pred_rest[i] == 1])
        CRD_needle = len([a for i, a in enumerate(y_true_rest) if a == 0 and y_pred_rest[i] == 0])
        Fib_PSW_rest = len([a for i, a in enumerate(y_true_rest) if a == 1 and y_pred_rest[i] == 2])
        Fib_PSW_contraction = len([a for i, a in enumerate(y_true_rest) if a == 1 and y_pred_rest[i] == 1])
        Fib_PSW_needle = len([a for i, a in enumerate(y_true_rest) if a == 1 and y_pred_rest[i] == 0])
        Fib_rest = len([a for i, a in enumerate(y_true_rest) if a == 2 and y_pred_rest[i] == 2])
        Fib_contraction = len([a for i, a in enumerate(y_true_rest) if a == 2 and y_pred_rest[i] == 1])
        Fib_needle = len([a for i, a in enumerate(y_true_rest) if a == 2 and y_pred_rest[i] == 0])
        Myo_rest = len([a for i, a in enumerate(y_true_rest) if a == 3 and y_pred_rest[i] == 2])
        Myo_contraction = len([a for i, a in enumerate(y_true_rest) if a == 3 and y_pred_rest[i] == 1])
        Myo_needle = len([a for i, a in enumerate(y_true_rest) if a == 3 and y_pred_rest[i] == 0])
        PSW_rest = len([a for i, a in enumerate(y_true_rest) if a == 4 and y_pred_rest[i] == 2])
        PSW_contraction = len([a for i, a in enumerate(y_true_rest) if a == 4 and y_pred_rest[i] == 1])
        PSW_needle = len([a for i, a in enumerate(y_true_rest) if a == 4 and y_pred_rest[i] == 0])
        rest_rest = len([a for i, a in enumerate(y_true_rest) if a == 5 and y_pred_rest[i] == 2])
        rest_contraction = len([a for i, a in enumerate(y_true_rest) if a == 5 and y_pred_rest[i] == 1])
        rest_needle = len([a for i, a in enumerate(y_true_rest) if a == 5 and y_pred_rest[i] == 0])

        classified_rest = np.count_nonzero(y_true_rest_soft == 2)
        CRD_rest_soft = len([a for i, a in enumerate(y_true_rest_soft) if a == 0 and y_pred_rest_soft[i] == 2])
        CRD_contraction_soft = len([a for i, a in enumerate(y_true_rest_soft) if a == 0 and y_pred_rest_soft[i] == 1])
        CRD_needle_soft = len([a for i, a in enumerate(y_true_rest_soft) if a == 0 and y_pred_rest_soft[i] == 0])
        Fib_PSW_rest_soft = len([a for i, a in enumerate(y_true_rest_soft) if a == 1 and y_pred_rest_soft[i] == 2])
        Fib_PSW_contraction_soft = len([a for i, a in enumerate(y_true_rest_soft) if a == 1 and y_pred_rest_soft[i] == 1])
        Fib_PSW_needle_soft = len([a for i, a in enumerate(y_true_rest_soft) if a == 1 and y_pred_rest_soft[i] == 0]) 
        Fib_rest_soft = len([a for i, a in enumerate(y_true_rest_soft) if a == 2 and y_pred_rest_soft[i] == 2])
        Fib_contraction_soft = len([a for i, a in enumerate(y_true_rest_soft) if a == 2 and y_pred_rest_soft[i] == 1])
        Fib_needle_soft = len([a for i, a in enumerate(y_true_rest_soft) if a == 2 and y_pred_rest_soft[i] == 0])
        Myo_rest_soft = len([a for i, a in enumerate(y_true_rest_soft) if a == 3 and y_pred_rest_soft[i] == 2])
        Myo_contraction_soft = len([a for i, a in enumerate(y_true_rest_soft) if a == 3 and y_pred_rest_soft[i] == 1])
        Myo_needle_soft = len([a for i, a in enumerate(y_true_rest_soft) if a == 3 and y_pred_rest_soft[i] == 0])
        PSW_rest_soft = len([a for i, a in enumerate(y_true_rest_soft) if a == 4 and y_pred_rest_soft[i] == 2])
        PSW_contraction_soft = len([a for i, a in enumerate(y_true_rest_soft) if a == 4 and y_pred_rest_soft[i] == 1])
        PSW_needle_soft = len([a for i, a in enumerate(y_true_rest_soft) if a == 4 and y_pred_rest_soft[i] == 0])
        rest_rest_soft = len([a for i, a in enumerate(y_true_rest_soft) if a == 5 and y_pred_rest_soft[i] == 2])
        rest_contraction_soft = len([a for i, a in enumerate(y_true_rest_soft) if a == 5 and y_pred_rest_soft[i] == 1])
        rest_needle_soft = len([a for i, a in enumerate(y_true_rest_soft) if a == 5 and y_pred_rest_soft[i] == 0])

        with open("/Users/Sterre/files/PROCESSED/mels_size_square/Rest/val_classification_rest.txt", "w") as a:
            a.write("Mel spectrograms from rest validation set, classified as: rest / contraction / needle\n")
            a.write("True label rest:               \t {}      \t {}      \t {}\n".format(rest_rest, rest_contraction, rest_needle))
            a.write("True label fibrillation:       \t {}      \t {}      \t {}\n".format(Fib_rest, Fib_contraction, Fib_needle))
            a.write("True label PSW:                \t {}      \t {}      \t {}\n".format(PSW_rest, PSW_contraction, PSW_needle))
            a.write("True label fib/PSW:            \t {}      \t {}      \t {}\n".format(Fib_PSW_rest, Fib_PSW_contraction, Fib_PSW_needle))
            a.write("True label myotonic discharge: \t {}      \t {}      \t {}\n".format(Myo_rest, Myo_contraction, Myo_needle))
            a.write("True label CRD:                \t {}      \t {}      \t {}\n".format(CRD_rest, CRD_contraction, CRD_needle))
            a.write("\n")
            a.write("With soft prediction...\n")
            a.write("True label rest:               \t {}      \t {}      \t {}\n".format(rest_rest_soft, rest_contraction_soft, rest_needle_soft))
            a.write("True label fibrillation:       \t {}      \t {}      \t {}\n".format(Fib_rest_soft, Fib_contraction_soft, Fib_needle_soft))
            a.write("True label PSW:                \t {}      \t {}      \t {}\n".format(PSW_rest_soft, PSW_contraction_soft, PSW_needle_soft))
            a.write("True label fib/PSW:            \t {}      \t {}      \t {}\n".format(Fib_PSW_rest_soft, Fib_PSW_contraction_soft, Fib_PSW_needle_soft))
            a.write("True label myotonic discharge: \t {}      \t {}      \t {}\n".format(Myo_rest_soft, Myo_contraction_soft, Myo_needle_soft))
            a.write("True label CRD:                \t {}      \t {}      \t {}\n".format(CRD_rest_soft, CRD_contraction_soft, CRD_needle_soft))

    def augmentation_val(self, directory_rest, max_augmentation):
        # from https://github.com/mdbloice/Augmentor

        directory_class = ['CRD/', 'Fib_PSW/', 'Fibrillation/', 'Myotonic_discharge/', 'PSW/', 'Rest/']

        distort = Distort(0.25, 16, 16, 1)
        skew = Skew(0.25, 'TILT', 0.5) #last number is magnitude

        save_path = '/Users/Sterre/files/PROCESSED/mels_size_square/Rest/Validation_rest_m2_soft_augmentation' + str(max_augmentation) + '/'
        os.mkdir(save_path)

        for c in directory_class:
            files_class = [directory_rest + c + f for f in os.listdir(directory_rest + c) if ".png" in f]
            file_names = [f for f in os.listdir(directory_rest + c) if ".png" in f]

            print("No. of files in class %s is equal to %s" % (c[:-1], len(files_class)))

            num_augmentation = max_augmentation - len(files_class)

            path_class = save_path + c
            os.mkdir(path_class)

            if len(files_class) >= 1:
                for index, file in enumerate(files_class):
                    im = Image.open(file)
                    im.save(path_class + file_names[index])

                for item in range(num_augmentation):
                    random_index = np.random.randint(len(files_class))
                    random_augmention_type = np.random.randint(2)
                    im = Image.open(files_class[random_index])

                    if random_augmention_type == 0:
                        augmented_im = skew.perform_operation(im)
                        augmented_im.save(path_class + str(item) + '.png')

                    elif random_augmention_type == 1:
                        augmented_im = distort.perform_operation(im)
                        augmented_im.save(path_class + str(item) + '.png')


    def augmentation_train(self, directory_rest, max_augmentation):
        # from https://github.com/mdbloice/Augmentor

        distort = Distort(0.25, 16, 16, 1)
        skew = Skew(0.25, 'TILT', 0.5) #last number is magnitude

        save_path = '/Users/Sterre/files/PROCESSED/mels_size_square/Training_m2_rest_soft_augmentation/'

        if not os.path.exists(save_path): 
            os.makedirs(save_path)
            os.makedirs(save_path + 'train/')
            print("Directory for training rest augmentation did not exist and is created.")
        save_path = save_path + 'train/'

        files = [directory_rest + f for f in os.listdir(directory_rest) if ".png" in f]
        file_names = [f for f in os.listdir(directory_rest) if ".png" in f]

        num_augmentation = max_augmentation - len(files)

        for index, file in enumerate(files):
            im = Image.open(file)
            im.save(save_path + file_names[index])

        for item in trange(num_augmentation):
            random_index = np.random.randint(len(files))
            random_augmention_type = np.random.randint(2)
            im = Image.open(files[random_index])

            if random_augmention_type == 0:
                augmented_im = skew.perform_operation(im)
                augmented_im.save(save_path + str(item) + '.png')

            elif random_augmention_type == 1:
                augmented_im = distort.perform_operation(im)
                augmented_im.save(save_path + str(item) + '.png')
       
if __name__ == "__main__":
    
    config = UnsupervisedOptions().parse()

    # if config.architecture != "arch_one_rest": 
    #     print("Wrong architecture type is given in configuration file.")
    #     exit()
    create_rest_datasets = MoveFilesClassifiedAsRest()

    # # STEP 1 
    # # copy files from training directory that were classified as rest to new rest directory
    # create_rest_datasets.move()
        
    # # STEP 2
    # # make sure that files from rest validation set are classified as rest as well... 
    # # create_rest_datasets.move_rest()

    # # STEP 3
    # # perform data augmentation on samples classified as rest  
    # directory_rest = '/Users/Sterre/files/PROCESSED/mels_size_square/Rest/Validation_rest_m2_soft/'
    # max_augmentation = 2000
    # create_rest_datasets.augmentation_val(directory_rest, max_augmentation)

    # perform data augmentation on training data set
    directory_rest = '/Users/Sterre/files/PROCESSED/mels_size_square/Training_m2_rest_soft/train/'
    max_augmentation = 500000
    create_rest_datasets.augmentation_train(directory_rest, max_augmentation)

