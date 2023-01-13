import argparse
import datetime
import os
import keras

class UnsupervisedOptions:
    """
    A class used to parse options for training unsupervised models 
    """

    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialised = False
    
    def initialise(self):

        # Files  
        self.parser.add_argument('--working_directory', type=str, 
                                default=r'/Users/Sterre/_Python_folders/ai_needle_emg_internshipSterre/',
                                help='The working directory with all python files.')
        self.parser.add_argument('--training_5p', type=str,
                                default='/Users/Sterre/files/PROCESSED/mels_size_square/Training_5percent')
        self.parser.add_argument('--training_100p', type=str,
                                default='/Users/Sterre/files/PROCESSED/mels_size_square/Training_100percent') 
        self.parser.add_argument('--validation_signaltype_70procent', type=str, 
                                default='/Users/Sterre/files/PROCESSED/mels_size_square/Validation_70procent')
        self.parser.add_argument('--test_signaltype_30procent', type=str, 
                                default='/Users/Sterre/files/PROCESSED/mels_size_square/Test_30procent')
        self.parser.add_argument('--validation_rest', type=str, 
                                default='/Users/Sterre/files/PROCESSED/mels_size_square/Rest/Validation_rest_m2_soft_augmentation2000')
        self.parser.add_argument('--training_rest', type=str, 
                                default='/Users/Sterre/files/PROCESSED/mels_size_square/Training_rest_testing')

        # type of experiment
        self.parser.add_argument('--train',                         default=True)
        self.parser.add_argument('--pretrain',                      default=True)
        self.parser.add_argument('--test',                          default=False)
        self.parser.add_argument('--hyperopt',                      default=False)
        self.parser.add_argument('--evaluate_hyperopt',             default=False)

        self.parser.add_argument('--architecture',                  default='arch_two_rest', choices=['arch_one', 'arch_two', 'arch_one_rest', 'arch_two_rest'])
        self.parser.add_argument('--data',                          default='100 percent', choices=['5 percent', '100 percent', 'rest'])
        
        # load locations
        self.parser.add_argument('--hyperopt_directory',            default='/Users/Sterre/_Python_folders/ai_needle_emg_internshipSterre/results/arch_two/05122022_1016_evalhyperopt1/trials-4.p')
        self.parser.add_argument('--location_model1_final',         default='/Users/Sterre/_Python_folders/ai_needle_emg_internshipSterre/results/arch_one/14102022_0820_bestmodelhyperopt_final/convae_model_final.hdf5') # format: .hdf5!
        self.parser.add_argument('--location_model2_final',         default='/Users/Sterre/_Python_folders/ai_needle_emg_internshipSterre/results/arch_two/30122022_0940_bestmodel/dcec_model_85.hdf5') # format: .hdf5!        
        self.parser.add_argument('--pretrained_cae',                default='/Users/Sterre/_Python_folders/ai_needle_emg_internshipSterre/results/arch_two/19122022_2231_5p_6/pretrain_cae_model.h5')

        # convae
        self.parser.add_argument('--filters',                       default=[32, 64, 128])
        self.parser.add_argument('--activation',                    default=keras.layers.LeakyReLU(alpha=0.2)) # or LeakyReLU(alpha=0.2) not in string
        self.parser.add_argument('--kernel_size',                   default=[7, 5, 3])
        self.parser.add_argument('--features',                      default=56)
        # training convae
        self.parser.add_argument('--learning_schedule_convae',      default='constant', choices=['constant', 'clr1', 'clr2']) 
        self.parser.add_argument('--learning_rate_convae',          default=0.0001) 
        self.parser.add_argument('--batch_size',                    default=64)
        self.parser.add_argument('--epochs_convae',                 default=1)
        self.parser.add_argument('--loss_convae',                   default='mse')
        self.parser.add_argument('--optimizer',                     default='adam')
        self.parser.add_argument('--batch_norm',                    default=True)
        # hyperopt
        self.parser.add_argument('--max_trials',                    default=500)
        self.parser.add_argument('--save_interval_trials',          default=1)
        # training DCEC
        self.parser.add_argument('--learning_schedule_dcec',        default='clr2') 
        self.parser.add_argument('--learning_rate_dcec',            default= [10**-7, 0.0001]) # [10**-5, 0.001]
        self.parser.add_argument('--epochs_dcec',                   default=100)
        self.parser.add_argument('--loss_dcec',                     default=['mse','kld'])
        # DCEC
        self.parser.add_argument('--gamma',                         default=0.05) # proberen 0.1; 0.2; 0.4; 0.6
        self.parser.add_argument('--tolerance',                     default=0.000001)


        # best parameters CAE
        self.parser.add_argument('--best_cae', default={'batch_size': 64, 
                                                        'learning_schedule_convae': 'constant', 
                                                        'learning_rate_convae': 0.0005, 
                                                        'filters': [32, 64, 128], 
                                                        'kernel_size': [7, 5, 3],
                                                        'activation': 'relu',
                                                        'batch_norm': True, 
                                                        'loss': 'mse', 
                                                        'features': 64, 
                                                        'optimizer': 'adam', 
                                                        'clusters': 6})
        self.parser.add_argument('--best_dcec', default={'batch_size': 64, 
                                                        'learning_schedule_convae': 'constant', 
                                                        'learning_rate_convae': 0.001, 
                                                        'learning_schedule_dcec': 'clr2', 
                                                        'learning_rate_dcec': [10**-7, 0.0001], 
                                                        'filters': [32, 64, 128], 
                                                        'kernel_size': [7, 5, 3],
                                                        'activation': keras.layers.LeakyReLU(alpha=0.2),
                                                        'batch_norm': True, 
                                                        'loss_dcec': ['mse','kld'], 
                                                        'loss_convae': 'mse', 
                                                        'features': 56, 
                                                        'optimizer': 'adam', 
                                                        'clusters': 7, 
                                                        'gamma': 0.05})                                                

        # testing 
        self.parser.add_argument('--confidence',                    default=0.53) # results in 25 percent of the data removed! (0.53)

        #plots
        self.parser.add_argument('--displays',                      default=False)

        #clustering 
        self.parser.add_argument('--clusters',                      default=6)

        self.parser.add_argument('--input_dimension',               default=(128,128))
    
    def parse(self):
        if not self.initialised:
            self.initialise()
        self.opt = self.parser.parse_args()

        # Create a log file
        time = str(datetime.datetime.now())
        time_stamp = time[2:4] + time[5:7] + time[8:10] + "_" + time[11:13] + time[14:16]
        self.opt.save_name = str(time_stamp) + "_" + str(self.opt.architecture) 

        # Update filepaths to reflect the correct base working directory
        self.opt.experiment_directory = self.opt.working_directory + 'results/' + self.opt.architecture + '/'
        self.opt.save_directory = self.opt.experiment_directory + time_stamp + '/'
        os.makedirs(self.opt.save_directory)
        # # Update the input size for the model based on the given data input parameters
        # self.opt.input_dimension_width = np.int(np.ceil(self.opt.sample_time * self.opt.sample_rate / self.opt.hop_length))

    
        # Write a log file.
        with open(self.opt.save_directory + str(self.opt.save_name) + ".txt", "w") as a:
            a.write("Date: {} \n".format(time_stamp))
            a.write("Logfile for experiment for {} \n".format(self.opt.architecture))
            a.write(" \n")
            a.write("Data\n")
            a.write("-------------------------\n")
            # a.write('Experiment directory           \t:{}\n'.format(self.opt.experiment_directory))
            a.write('Data used for training         \t: {}\n'.format(self.opt.data))
            a.write("Data used for testing          \t: Signal type data set [70%] \n")
            a.write("\n")
            if self.opt.architecture == 'arch_one':
                a.write("Training options\n")
                a.write("-------------------------\n")
                a.write("Number of epochs               \t: {}\n".format(self.opt.epochs_convae))
                a.write("Batch size                     \t: {} \n".format(self.opt.batch_size))
                a.write("Learning rate schedule         \t: {}\n".format(self.opt.learning_schedule_convae))
                a.write("Loss                           \t: {}\n".format(self.opt.loss_convae))
                a.write("Optimizer                      \t: {}\n".format(self.opt.optimizer))
                a.write("\n")
                a.write("Architecture\n")
                a.write("-------------------------\n")
                a.write("Kernel size                    \t: {}\n".format(self.opt.kernel_size))
                a.write("Filters                        \t: {}\n".format(self.opt.filters))
                a.write("Activation                     \t: {}\n".format(self.opt.activation))
                a.write("Latent space                   \t: {}\n".format(self.opt.features))
                a.write("\n")
                a.write("Clustering\n")
                a.write("-------------------------\n")
                a.write("Number of clusters             \t: {}\n".format(self.opt.clusters))
            if self.opt.architecture == 'arch_two':
                a.write("Pre-training \n")
                a.write("-------------------------\n")
                a.write("Number of epochs               \t: {}\n".format(self.opt.epochs_convae))
                a.write("Batch size                     \t: {} \n".format(self.opt.batch_size))
                a.write("Learning rate schedule         \t: {}\n".format(self.opt.learning_schedule_convae))
                a.write("Learning rate                  \t: {}\n".format(self.opt.learning_rate_convae))
                a.write("Loss                           \t: {}\n".format(self.opt.loss_convae))
                a.write("Optimizer                      \t: {}\n".format(self.opt.optimizer))
                a.write("\n")
                a.write("Architecture convae during pre-training\n")
                a.write("-------------------------\n")
                a.write("Kernel size                    \t: {}\n".format(self.opt.kernel_size))
                a.write("Filters                        \t: {}\n".format(self.opt.filters))
                a.write("Activation                     \t: {}\n".format(self.opt.activation))
                a.write("Latent space                   \t: {}\n".format(self.opt.features))
                a.write("\n")
                a.write("Architecture DCEC \n")
                a.write("-------------------------\n")
                a.write("Loss                           \t: {}\n".format(self.opt.loss_dcec)) 
                a.write("Optimizer                      \t: {}\n".format(self.opt.optimizer))
                a.write("Learning schedule              \t: {}\n".format(self.opt.learning_schedule_dcec))
                a.write("Learning rate                  \t: {}\n".format(self.opt.learning_rate_dcec))
                a.write("Number of clusters             \t: {}\n".format(self.opt.clusters))
                a.write("Gamma                          \t: {}\n".format(self.opt.gamma))
                a.write("Early stopping at              \t: tol {}\n".format(self.opt.tolerance))
                a.write("Maximal number of epochs       \t: {}\n".format(self.opt.epochs_dcec))
            a.write("\n")
            a.write("Metrics\n")
            a.write("-------------------------\n")
            a.write("Confidence                     \t: {}\n".format(self.opt.confidence))


        return self.opt

             